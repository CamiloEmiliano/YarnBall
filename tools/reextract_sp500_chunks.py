"""Resumable Point-in-Time S&P 500 Chunk Re-Extraction Batch Job (ADR-00035 Section 4.7).

Incrementally re-extracts SEC filing bundles for active Point-in-Time S&P 500 constituents
(`universe/sp500.py`) into `document_chunks` + `chunk_entity_links` + compact Memgraph
`evidence_chunk_ids` pointers, tracking completed `(cik, accession_number)` keys in a
JSONL checkpoint so interrupted runs resume idempotently without duplicate LLM calls.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.sec.canon import SECFormType, normalize_form_type
from ingestion.sec.harvester import EdgarWorker
from ingestion.sec.parser import extract_sec_relationships, rank_salient_chunks
from ingestion.segmentation.chunker import FilingMetadata, FinancialSectionSegmenter
from knowledge_graph.chunk_store import (
    DocumentChunkRecord,
    flag_legacy_unchunked_edges,
    store_document_chunks,
)
from retrieval.temporal import actionable_session
from universe.sp500 import SP500UniverseManager

logger = logging.getLogger("tools.reextract_sp500_chunks")

DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "data" / "checkpoints" / "adr00035_sp500_reextract.jsonl"


def load_completed_checkpoints(checkpoint_path: Path | str) -> Set[Tuple[str, str]]:
    """Load completed `(cik, accession_number)` tuples from a JSONL checkpoint file."""
    path = Path(checkpoint_path)
    completed: Set[Tuple[str, str]] = set()
    if not path.is_file():
        return completed
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if rec.get("status") == "completed":
                cik = str(rec.get("cik") or "").strip().zfill(10)
                acc = str(rec.get("accession_number") or "").strip()
                if cik and acc:
                    completed.add((cik, acc))
        except Exception:
            continue
    return completed


def append_checkpoint_record(checkpoint_path: Path | str, record: Dict[str, Any]) -> None:
    """Append one JSONL checkpoint record durably."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def run_sp500_chunk_reextraction(
    *,
    as_of_date: str,
    filings_dir: Path | str = PROJECT_ROOT / "data" / "sec_filings",
    checkpoint_path: Path | str = DEFAULT_CHECKPOINT_PATH,
    max_companies: Optional[int] = None,
    dry_run: bool = False,
    flag_legacy_edges: bool = False,
    embed_chunks: bool = False,
    batch_gpu: bool = False,
    pg_conn: Optional[Any] = None,
    memgraph_driver: Optional[Any] = None,
    model_client: Optional[Any] = None,
    universe_manager: Optional[SP500UniverseManager] = None,
) -> Dict[str, Any]:
    """Execute resumable chunked re-extraction scoped to active PiT S&P 500 constituents (ADR-00035)."""
    universe = universe_manager or SP500UniverseManager()
    constituents = universe.get_constituents_at_date(as_of_date)
    active_tickers = {c.ticker.strip().upper() for c in constituents if c.ticker}
    if max_companies is not None:
        ordered_tickers = sorted(active_tickers)[: max(0, int(max_companies))]
        active_tickers = set(ordered_tickers)

    completed = load_completed_checkpoints(checkpoint_path)
    filings_root = Path(filings_dir)
    segmenter = FinancialSectionSegmenter(max_tokens=800, overlap_tokens=120)
    worker = EdgarWorker(pg_conn=pg_conn, memgraph_driver=memgraph_driver) if not dry_run else None

    flagged_legacy_count = 0
    if flag_legacy_edges and not dry_run:
        flagged_legacy_count = flag_legacy_unchunked_edges(driver=memgraph_driver)

    processed_bundles = 0
    skipped_checkpoints = 0
    chunks_persisted = 0
    edges_persisted = 0

    if filings_root.is_dir():
        for company_dir in sorted(filings_root.iterdir(), key=lambda p: p.name):
            if not company_dir.is_dir():
                continue
            ticker = company_dir.name.strip().upper()
            if active_tickers and ticker not in active_tickers:
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

                cik = str(meta.get("cik") or "0000000000").strip().zfill(10)
                acc = str(meta.get("accession_number") or bundle_dir.name).strip()
                if (cik, acc) in completed:
                    skipped_checkpoints += 1
                    continue

                company_name = str(meta.get("company_name") or ticker)
                filing_date = str(meta.get("filing_date") or as_of_date)
                acceptance_dt = meta.get("acceptance_datetime")
                form_type = normalize_form_type(
                    meta.get("form") or meta.get("form_type"),
                    default=SECFormType.FORM_10K.value,
                )

                session_date, known_at_utc, _ = actionable_session(
                    acceptance_dt,
                    fallback_doc_date=filing_date,
                )
                valid_from_iso = session_date.isoformat() if session_date else None
                known_at_iso = known_at_utc.isoformat() if known_at_utc else None

                filing_meta = FilingMetadata(
                    cik=cik,
                    ticker=ticker,
                    company_name=company_name,
                    form_type=form_type,
                    filing_date=filing_date,
                    accession_number=acc,
                )

                bundle_chunk_count = 0
                bundle_edge_count = 0
                for section_label, txt_name in (
                    ("Item 1 - Business", "item_1_business.txt"),
                    ("Item 1A - Risk Factors", "item_1a_risk_factors.txt"),
                    ("Item 7 - Management Discussion and Analysis", "item_7_mda.txt"),
                ):
                    txt_path = bundle_dir / txt_name
                    if not txt_path.is_file():
                        continue
                    section_text = txt_path.read_text(encoding="utf-8").strip()
                    if not section_text:
                        continue

                    raw_chunks = segmenter.segment_filing(
                        section_text,
                        filing_meta,
                        default_section_name=section_label,
                    )
                    _ = rank_salient_chunks(raw_chunks)
                    bundle_chunk_count += len(raw_chunks)

                    if dry_run:
                        continue

                    chunk_records = [
                        DocumentChunkRecord(
                            chunk_id=c.chunk_id,
                            doc_id=acc,
                            source_type="sec_filing",
                            cik=cik,
                            section=c.section_name,
                            char_start=c.start_char,
                            char_end=c.end_char,
                            text=c.text,
                            valid_from=valid_from_iso,
                            known_at=known_at_iso,
                        )
                        for c in raw_chunks
                    ]
                    if pg_conn is not None and chunk_records:
                        chunks_persisted += store_document_chunks(
                            chunk_records,
                            conn=pg_conn,
                            embed=embed_chunks,
                            batch_gpu=batch_gpu,
                        )

                    extracted = extract_sec_relationships(
                        section_text,
                        source_id=f"{acc}:{section_label}",
                        doc_id=acc,
                        cik=cik,
                        ticker=ticker,
                        company_name=company_name,
                        form_type=form_type,
                        filing_date=filing_date,
                        section_name=section_label,
                        model_client=model_client,
                    )
                    if worker is not None and (extracted.get("nodes") or extracted.get("edges")):
                        _, persisted = worker._persist_sec_graph_data(
                            primary_name=company_name,
                            primary_ticker=ticker,
                            graph_data=extracted,
                            accession_number=acc,
                            filing_date=filing_date,
                            form_type=form_type,
                            section=section_label,
                            source_text=section_text,
                            acceptance_datetime=acceptance_dt,
                            evidence_chunk_ids=[c.chunk_id for c in raw_chunks],
                        )
                        bundle_edge_count += len(persisted)

                edges_persisted += bundle_edge_count
                processed_bundles += 1
                if not dry_run:
                    append_checkpoint_record(
                        checkpoint_path,
                        {
                            "cik": cik,
                            "ticker": ticker,
                            "accession_number": acc,
                            "status": "completed",
                            "chunks": bundle_chunk_count,
                            "edges": bundle_edge_count,
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                    completed.add((cik, acc))

    return {
        "as_of_date": as_of_date,
        "targeted_sp500_tickers": len(active_tickers),
        "processed_bundles": processed_bundles,
        "skipped_checkpoints": skipped_checkpoints,
        "chunks_persisted": chunks_persisted,
        "edges_persisted": edges_persisted,
        "flagged_legacy_edges": flagged_legacy_count,
        "dry_run": dry_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Incremental resumable S&P 500 chunk re-extraction batch job (ADR-00035)."
    )
    parser.add_argument("--as-of", default=date.today().isoformat(), help="PiT S&P 500 membership date (YYYY-MM-DD)")
    parser.add_argument("--filings-dir", default=str(PROJECT_ROOT / "data" / "sec_filings"))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--max-companies", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--flag-legacy-edges", action="store_true")
    parser.add_argument("--embed-chunks", action="store_true")
    parser.add_argument("--batch-gpu", action="store_true")
    args = parser.parse_args()

    summary = run_sp500_chunk_reextraction(
        as_of_date=args.as_of,
        filings_dir=args.filings_dir,
        checkpoint_path=args.checkpoint,
        max_companies=args.max_companies,
        dry_run=args.dry_run,
        flag_legacy_edges=args.flag_legacy_edges,
        embed_chunks=args.embed_chunks,
        batch_gpu=args.batch_gpu,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
