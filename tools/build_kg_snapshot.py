"""Reproducible SEC + News Knowledge Graph Build & Parquet Snapshot Sync CLI (ADR-00028).

Ingests local SEC EDGAR filing bundles (`data/sec_filings/*`) and optional staged news
events into Memgraph (or an isolated in-memory graph driver when `--offline` is set),
applies cross-source canonical entity linking, runs `EntityResolver` cleanup, and
exports versioned `nodes.parquet` / `edges.parquet` snapshots discoverable by the Workbench.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Literal, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.sec.harvester import EdgarWorker
from ingestion.sec.parser import extract_sec_relationships
from knowledge_graph.entity_resolver import EntityResolver
from knowledge_graph.graph_store import store_graph_entities
from knowledge_graph.memgraph_driver import get_memgraph_driver
from knowledge_graph.snapshot_manager import SnapshotManager

logger = logging.getLogger("tools.build_kg_snapshot")

_MERGE_NODE_RE = re.compile(
    r"MERGE\s*\(\s*\w+\s*:\s*([A-Za-z0-9_]+)\s*\{\s*(id)\s*:\s*\$(\w+)\s*\}\s*\)(.*)",
    re.DOTALL,
)
_MERGE_EDGE_RE = re.compile(
    r"MERGE\s*\([^)]+\)-\[\s*\w+\s*:\s*([A-Za-z0-9_]+)\s*\]->\([^)]+\)(.*)",
    re.DOTALL,
)


class _InMemoryCypherSession:
    """Lightweight Cypher session executing offline ADR-00028 snapshot builds without Memgraph."""

    def __init__(self, driver: "_InMemoryMemgraphCollector") -> None:
        self._driver = driver

    def __enter__(self) -> "_InMemoryCypherSession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Literal[False]:
        return False

    def run(self, query: str, **params: Any) -> List[Dict[str, Any]]:
        q = query.strip()
        if "RETURN labels(n) AS labels" in q:
            return [
                {"labels": list(entry["labels"]), "props": dict(entry["props"])}
                for entry in self._driver.nodes.values()
            ]
        if "RETURN s.id AS source_id" in q:
            return [
                {
                    "source_id": edge["source_id"],
                    "target_id": edge["target_id"],
                    "rel_type": edge["rel_type"],
                    "props": dict(edge["props"]),
                    "source_hash": edge["props"].get("source_hash", ""),
                }
                for edge in self._driver.edges.values()
            ]

        m_edge = _MERGE_EDGE_RE.search(q)
        if m_edge:
            rel_type = m_edge.group(1)
            set_clause = m_edge.group(2)
            if rel_type == "SUBSIDIARY_OF":
                src_id = str(params.get("sub_id", "")).strip()
                tgt_id = str(params.get("parent_id", "")).strip()
            else:
                src_id = str(params.get("source", "")).strip()
                tgt_id = str(params.get("target", "")).strip()
            if src_id and tgt_id:
                edge_key = (src_id, rel_type, tgt_id)
                props = dict(self._driver.edges.get(edge_key, {}).get("props", {}))
                if isinstance(params.get("props"), dict):
                    props.update(params["props"])
                if "fiscal_year" in params:
                    props["fiscal_year"] = params["fiscal_year"]
                if "valid_from" in params:
                    props["valid_from"] = params["valid_from"]
                if "form_type" in params:
                    props["form_type"] = params["form_type"]
                if "section" in params:
                    props["section"] = params["section"]
                if "acc_num" in params:
                    props["accession_number"] = params["acc_num"]
                if "jurisdiction" in params:
                    props["jurisdiction"] = params["jurisdiction"]
                if "source_hash" in params:
                    props["source_hash"] = params["source_hash"]
                if "source_type" in params:
                    props["source_type"] = params["source_type"]
                elif "source_type = 'sec_filing'" in set_clause:
                    props["source_type"] = "sec_filing"
                if "is_temporal_unanchored" in params:
                    props["is_temporal_unanchored"] = bool(params["is_temporal_unanchored"])
                elif "($valid_from IS NULL)" in set_clause:
                    props["is_temporal_unanchored"] = params.get("valid_from") is None
                if "is_current = true" in set_clause:
                    props["is_current"] = True
                if "form_type = '10-K'" in set_clause:
                    props["form_type"] = "10-K"
                self._driver.edges[edge_key] = {
                    "source_id": src_id,
                    "target_id": tgt_id,
                    "rel_type": rel_type,
                    "props": props,
                }
            return []

        m_node = _MERGE_NODE_RE.search(q)
        if m_node:
            label = m_node.group(1)
            id_param = m_node.group(3)
            nid = str(params.get(id_param, "")).strip()
            if nid:
                existing = self._driver.nodes.setdefault(
                    nid,
                    {"labels": [label], "props": {"id": nid}},
                )
                if label not in existing["labels"]:
                    existing["labels"].append(label)
                if isinstance(params.get("props"), dict):
                    existing["props"].update(params["props"])
                for key in ("ticker", "jurisdiction", "published_at", "source_type"):
                    if key in params and params[key] is not None:
                        existing["props"][key] = params[key]
        return []


class _InMemoryMemgraphCollector:
    """In-memory property-graph collector compatible with `EdgarWorker`, `store_graph_entities`, and `SnapshotManager`."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    def session(self) -> _InMemoryCypherSession:
        return _InMemoryCypherSession(self)


def build_and_export_kg_snapshot(
    *,
    sec_filings_dir: Path | str = PROJECT_ROOT / "data" / "sec_filings",
    snapshots_dir: Path | str = PROJECT_ROOT / "data" / "snapshots",
    snapshot_id: Optional[str] = None,
    news_file: Optional[Path | str] = None,
    tickers: Optional[List[str]] = None,
    extract_llm: bool = False,
    resolve_entities: bool = True,
    offline: bool = False,
    memgraph_driver: Optional[Any] = None,
) -> Dict[str, Any]:
    """Ingest local SEC filing bundles + optional news events and export a Parquet snapshot."""
    sec_root = Path(sec_filings_dir)
    snap_root = Path(snapshots_dir)
    snap_id = snapshot_id or f"G_sec_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    allowed_tickers = {t.strip().upper() for t in tickers if t.strip()} if tickers else None

    if memgraph_driver is not None:
        active_driver = memgraph_driver
    elif offline:
        active_driver = _InMemoryMemgraphCollector()
    else:
        active_driver = get_memgraph_driver()

    worker = EdgarWorker(pg_conn=None, memgraph_driver=active_driver)
    sec_subsidiaries_ingested = 0
    sec_edges_ingested = 0
    bundles_processed = 0

    if sec_root.is_dir():
        for company_dir in sorted(sec_root.iterdir(), key=lambda p: p.name):
            if not company_dir.is_dir():
                continue
            dir_ticker = company_dir.name.upper()
            if allowed_tickers and dir_ticker not in allowed_tickers:
                continue

            for bundle_dir in sorted(company_dir.iterdir(), key=lambda p: p.name):
                if not bundle_dir.is_dir():
                    continue
                meta_path = bundle_dir / "metadata.json"
                if not meta_path.is_file():
                    continue
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    continue

                ticker = str(meta.get("ticker") or dir_ticker).upper()
                company_name = str(meta.get("company_name") or ticker)
                cik = str(meta.get("cik") or "").zfill(10)
                acc = str(meta.get("accession_number") or bundle_dir.name)
                filing_date = str(meta.get("filing_date") or "")
                form_type = str(meta.get("form") or "10-K")

                subs_path = bundle_dir / "subsidiaries.json"
                if subs_path.is_file():
                    try:
                        subs = json.loads(subs_path.read_text(encoding="utf-8"))
                        if isinstance(subs, list) and subs:
                            _, sub_edges = worker._persist_subsidiaries(
                                parent_cik=cik,
                                parent_name=company_name,
                                parent_ticker=ticker,
                                subsidiaries=subs,
                                accession_number=acc,
                                filing_date=filing_date,
                            )
                            sec_subsidiaries_ingested += len(sub_edges)
                    except Exception as exc:
                        logger.warning("Failed persisting subsidiaries in %s: %s", bundle_dir, exc)

                preextracted_path = bundle_dir / "graph_data.json"
                if preextracted_path.is_file():
                    try:
                        graph_payload = json.loads(preextracted_path.read_text(encoding="utf-8"))
                        section_text = ""
                        for txt_name in ("item_1_business.txt", "item_1a_risk_factors.txt"):
                            txt_path = bundle_dir / txt_name
                            if txt_path.is_file():
                                section_text += txt_path.read_text(encoding="utf-8") + "\n"
                        _, g_edges = worker._persist_sec_graph_data(
                            primary_name=company_name,
                            primary_ticker=ticker,
                            graph_data=graph_payload,
                            accession_number=acc,
                            filing_date=filing_date,
                            form_type=form_type,
                            section="preextracted",
                            source_text=section_text or None,
                        )
                        sec_edges_ingested += len(g_edges)
                    except Exception as exc:
                        logger.warning("Failed persisting pre-extracted graph in %s: %s", bundle_dir, exc)
                elif extract_llm:
                    for section_label, txt_name in (
                        ("Item 1 Business", "item_1_business.txt"),
                        ("Item 1A Risk Factors", "item_1a_risk_factors.txt"),
                    ):
                        txt_path = bundle_dir / txt_name
                        if not txt_path.is_file():
                            continue
                        section_text = txt_path.read_text(encoding="utf-8")
                        graph_payload = extract_sec_relationships(section_text, source_id=f"{acc}:{section_label}")
                        if graph_payload.get("nodes") or graph_payload.get("edges"):
                            _, g_edges = worker._persist_sec_graph_data(
                                primary_name=company_name,
                                primary_ticker=ticker,
                                graph_data=graph_payload,
                                accession_number=acc,
                                filing_date=filing_date,
                                form_type=form_type,
                                section=section_label,
                                source_text=section_text,
                            )
                            sec_edges_ingested += len(g_edges)

                bundles_processed += 1

    news_ingested = 0
    if news_file:
        news_path = Path(news_file)
        if news_path.is_file():
            raw_txt = news_path.read_text(encoding="utf-8").strip()
            records: List[Dict[str, Any]] = []
            if news_path.suffix.lower() == ".jsonl":
                records = [json.loads(line) for line in raw_txt.splitlines() if line.strip()]
            elif raw_txt:
                parsed = json.loads(raw_txt)
                records = parsed if isinstance(parsed, list) else parsed.get("records", [])
            for idx, item in enumerate(records):
                src_hash = str(item.get("source_hash") or f"news_snapshot_{idx}")
                text = str(item.get("text") or item.get("raw_payload") or "")
                store_graph_entities(
                    source_hash=src_hash,
                    text=text,
                    graph_data=item.get("graph_data"),
                    driver=active_driver,
                    published_at=item.get("published_at") or item.get("fetched_at"),
                    source_type="news",
                )
                news_ingested += 1

    snapshot_mgr = SnapshotManager(base_dir=snap_root)
    export_meta = snapshot_mgr.export_snapshot(
        snapshot_id=snap_id,
        tag="sec_sync",
        description=f"ADR-00028 SEC+News sync ({bundles_processed} SEC bundles, {news_ingested} news records)",
        metadata={
            "bundles_processed": bundles_processed,
            "sec_subsidiaries_ingested": sec_subsidiaries_ingested,
            "sec_edges_ingested": sec_edges_ingested,
            "news_ingested": news_ingested,
            "offline": offline,
        },
        driver=active_driver,
    )

    resolved_summary: Optional[Dict[str, Any]] = None
    if resolve_entities and (snap_root / snap_id / "edges.parquet").is_file():
        try:
            resolver = EntityResolver(snapshot_manager=snapshot_mgr)
            resolved_summary = resolver.resolve_snapshot(snap_id, target_snapshot_id=snap_id)
        except Exception as exc:
            logger.warning("EntityResolver post-export step skipped: %s", exc)

    return {
        "snapshot_id": snap_id,
        "snapshot_dir": str(snap_root / snap_id),
        "bundles_processed": bundles_processed,
        "sec_subsidiaries_ingested": sec_subsidiaries_ingested,
        "sec_edges_ingested": sec_edges_ingested,
        "news_ingested": news_ingested,
        "export": export_meta,
        "resolved": resolved_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build & export SEC + News Parquet KG snapshots (ADR-00028)")
    parser.add_argument("--sec-dir", default=str(PROJECT_ROOT / "data" / "sec_filings"))
    parser.add_argument("--snapshots-dir", default=str(PROJECT_ROOT / "data" / "snapshots"))
    parser.add_argument("--snapshot-id", default=None)
    parser.add_argument("--news-file", default=None)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--extract-llm", action="store_true")
    parser.add_argument("--no-resolve", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    result = build_and_export_kg_snapshot(
        sec_filings_dir=args.sec_dir,
        snapshots_dir=args.snapshots_dir,
        snapshot_id=args.snapshot_id,
        news_file=args.news_file,
        tickers=args.tickers,
        extract_llm=args.extract_llm,
        resolve_entities=not args.no_resolve,
        offline=args.offline,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
