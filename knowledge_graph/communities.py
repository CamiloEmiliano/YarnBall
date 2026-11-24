"""Point-in-Time Structural Community Layer & Global Map-Reduce Reports (ADR-00037).

Implements:
1. Point-in-time structural community detection (`build_communities`) over active
   structural supply-chain topology (`kg_edges` / `G_resolved`, excluding transient
   `:News` / `MENTIONS` impulses and strictly enforcing `valid_from <= as_of` and
   `valid_to IS NULL OR valid_to > as_of`).
2. Primary algorithm: seeded MIT-licensed `graspologic.partition.hierarchical_leiden`
   (edge weight = `share`, default `1.0`). Deterministic fallback: seeded `networkx`
   Louvain with explicit connected-component splitting so every emitted community is
   guaranteed internally connected even when `graspologic` native wheels are absent.
3. Community summarization via `YarnBallModelClient` (`task="COMMUNITY_REPORT"`),
   resolving passage snippets from PostgreSQL `document_chunks` / `chunks.parquet`
   via `evidence_chunk_ids` (ADR-00035).
4. Durable persistence in PostgreSQL `community_reports` and offline snapshot
   materialization in `communities.parquet`.
5. Global thematic map-reduce retrieval restricted strictly to the latest cached
   checkpoint `as_of <= ctx.as_of` for `universe_id`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import networkx as nx
import pyarrow as pa
import pyarrow.parquet as pq

from knowledge_graph.chunk_store import fetch_chunks_by_ids
from knowledge_graph.db import pg_connection
from knowledge_graph.ledger import load_ledger_edges
from retrieval.temporal import PiTContext

logger = logging.getLogger(__name__)

DEFAULT_UNIVERSE_ID = "sp500_pit"
_TRANSIENT_REL_TYPES = frozenset({"MENTIONS", "PUBLISHED_IN", "CO_MENTIONED"})
_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


@dataclass(frozen=True)
class Community:
    """Point-in-time structural entity cluster & summary report (ADR-00037)."""

    universe_id: str
    as_of: str
    level: int
    community_id: str
    member_ids: List[str] = field(default_factory=list)
    summary: str = ""
    key_edges: List[Dict[str, Any]] = field(default_factory=list)
    evidence_chunk_ids: List[str] = field(default_factory=list)
    parent_community_id: Optional[str] = None
    algorithm: str = "hierarchical_leiden"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "universe_id": self.universe_id,
            "as_of": self.as_of,
            "level": self.level,
            "community_id": self.community_id,
            "member_ids": list(self.member_ids),
            "summary": self.summary,
            "key_edges": list(self.key_edges),
            "evidence_chunk_ids": list(self.evidence_chunk_ids),
            "parent_community_id": self.parent_community_id,
            "algorithm": self.algorithm,
        }


def _coerce_as_of_date(as_of: date | str | PiTContext) -> date:
    if isinstance(as_of, PiTContext):
        return as_of.as_of
    if isinstance(as_of, date):
        return as_of
    return date.fromisoformat(str(as_of).strip()[:10])


def _parse_optional_date(raw: Any) -> Optional[date]:
    if raw is None:
        return None
    if isinstance(raw, date):
        return raw
    txt = str(raw).strip()[:10]
    if not txt or txt.lower() in ("none", "null", "nan"):
        return None
    try:
        return date.fromisoformat(txt)
    except ValueError:
        return None


def _extract_edge_props(edge: Mapping[str, Any]) -> Dict[str, Any]:
    props_raw = edge.get("properties")
    if isinstance(props_raw, Mapping):
        merged = dict(props_raw)
    elif isinstance(edge.get("properties_json"), str):
        try:
            loaded = json.loads(str(edge["properties_json"]))
            merged = dict(loaded) if isinstance(loaded, dict) else {}
        except (json.JSONDecodeError, TypeError, ValueError):
            merged = {}
    else:
        merged = {}
    for k, v in edge.items():
        if k not in ("properties", "properties_json") and v is not None:
            merged.setdefault(k, v)
    return merged


def _extract_evidence_chunk_ids(props: Mapping[str, Any]) -> List[str]:
    raw = props.get("evidence_chunk_ids")
    if isinstance(raw, (list, tuple, set)):
        return sorted({str(c).strip() for c in raw if str(c).strip()})
    if isinstance(raw, str) and raw.strip():
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, list):
                return sorted({str(c).strip() for c in loaded if str(c).strip()})
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return []


def filter_structural_edges_pit(
    edges: Iterable[Mapping[str, Any]],
    as_of: date | str | PiTContext,
    *,
    include_unanchored: bool = False,
) -> List[Dict[str, Any]]:
    """Filter structural supply-chain edges strictly active at `as_of` (ADR-00033 & ADR-00037 Invariant #1).

    Excludes:
    - Transient `:News` / `MENTIONS` edges (ADR-00036).
    - Future edges with `valid_from > as_of`.
    - Terminated edges with `valid_to <= as_of` (half-open `[valid_from, valid_to)`).
    - Temporally unanchored edges unless `include_unanchored=True`.
    """
    cutoff = _coerce_as_of_date(as_of)
    active: List[Dict[str, Any]] = []

    for raw in edges:
        props = _extract_edge_props(raw)
        rel_type = str(raw.get("rel_type") or raw.get("type") or props.get("rel_type") or "SUPPLIES_TO").strip().upper()
        if rel_type in _TRANSIENT_REL_TYPES:
            continue

        src = str(raw.get("source_id") or raw.get("source") or "").strip()
        tgt = str(raw.get("target_id") or raw.get("target") or "").strip()
        if not src or not tgt or src == tgt:
            continue

        src_type = str(props.get("source_type") or raw.get("source_type") or "").strip().lower()
        if src_type == "news_impulse":
            continue

        v_from = _parse_optional_date(raw.get("valid_from") if raw.get("valid_from") is not None else props.get("valid_from"))
        v_to = _parse_optional_date(raw.get("valid_to") if raw.get("valid_to") is not None else props.get("valid_to"))

        if v_from is None:
            if not include_unanchored:
                continue
        elif v_from > cutoff:
            continue

        if v_to is not None and v_to <= cutoff:
            continue

        share_raw = raw.get("share") if raw.get("share") is not None else props.get("share")
        try:
            weight = float(share_raw) if share_raw is not None else 1.0
        except (TypeError, ValueError):
            weight = 1.0
        if weight <= 0.0:
            weight = 1.0

        active.append(
            {
                "source_id": src,
                "target_id": tgt,
                "rel_type": rel_type,
                "share": round(min(1.0, weight), 4),
                "weight": round(weight, 6),
                "valid_from": v_from.isoformat() if v_from else None,
                "valid_to": v_to.isoformat() if v_to else None,
                "input_group": str(raw.get("input_group") or props.get("input_group") or props.get("nature") or "default"),
                "evidence_chunk_ids": _extract_evidence_chunk_ids(props),
            }
        )

    active.sort(key=lambda e: (-e["weight"], e["source_id"], e["target_id"], e["rel_type"]))
    return active


def _load_edges_from_snapshot(snapshot_dir: Path | str) -> List[Dict[str, Any]]:
    edges_path = Path(snapshot_dir) / "edges.parquet"
    if not edges_path.is_file():
        return []
    table = pq.read_table(edges_path)
    return list(table.to_pylist())


def _split_into_connected_clusters(
    graph: nx.Graph,
    raw_clusters: Sequence[Set[str]],
) -> List[List[str]]:
    """Guarantee every emitted community is internally connected (Traag et al., 2019 invariant)."""
    connected: List[List[str]] = []
    for members in raw_clusters:
        if not members:
            continue
        sub = graph.subgraph(members)
        for comp in nx.connected_components(sub):
            connected.append(sorted(str(n) for n in comp))
    connected.sort(key=lambda m: (-len(m), m[0] if m else ""))
    return connected


def _run_hierarchical_partition(
    graph: nx.Graph,
    *,
    max_cluster_size: int = 10,
    max_levels: int = 2,
    random_seed: int = 42,
) -> Tuple[str, Dict[int, List[Tuple[str, List[str], Optional[str]]]]]:
    """Run seeded `graspologic.partition.hierarchical_leiden`, falling back to connected seeded Louvain.

    Returns `(algorithm_name, {level: [(community_id, sorted_member_ids, parent_community_id), ...]})`.
    """
    if graph.number_of_nodes() == 0:
        return "hierarchical_leiden", {}

    max_lvl = max(1, min(int(max_levels), 3))
    max_sz = max(2, int(max_cluster_size))

    # 1. Preferred primary algorithm: MIT-licensed graspologic hierarchical_leiden
    try:
        from graspologic.partition import hierarchical_leiden  # type: ignore[import-not-found]

        hl_results = hierarchical_leiden(
            graph,
            max_cluster_size=max_sz,
            random_seed=random_seed,
        )
        by_level_cluster: Dict[int, Dict[str, Set[str]]] = {}
        parent_map: Dict[Tuple[int, str], Optional[str]] = {}
        for item in hl_results:
            lvl = int(getattr(item, "level", 0))
            if lvl >= max_lvl:
                continue
            cid = str(getattr(item, "cluster", "0"))
            node = str(getattr(item, "node", ""))
            parent_cid = getattr(item, "parent_cluster", None)
            if node:
                by_level_cluster.setdefault(lvl, {}).setdefault(cid, set()).add(node)
                parent_map[(lvl, cid)] = str(parent_cid) if parent_cid is not None else None

        levels_out: Dict[int, List[Tuple[str, List[str], Optional[str]]]] = {}
        for lvl in sorted(by_level_cluster.keys()):
            entries: List[Tuple[str, List[str], Optional[str]]] = []
            ordered = sorted(
                by_level_cluster[lvl].items(),
                key=lambda kv: (-len(kv[1]), sorted(kv[1])[0] if kv[1] else ""),
            )
            for idx, (raw_cid, members) in enumerate(ordered, start=1):
                sub_comps = _split_into_connected_clusters(graph, [members])
                for comp_idx, comp_members in enumerate(sub_comps):
                    suffix = f"_{comp_idx}" if len(sub_comps) > 1 else ""
                    canon_cid = f"L{lvl}_C{idx:03d}{suffix}"
                    p_raw = parent_map.get((lvl, raw_cid))
                    entries.append((canon_cid, comp_members, p_raw))
            levels_out[lvl] = entries
        return "hierarchical_leiden", levels_out
    except ImportError:
        logger.debug("graspologic not installed; using deterministic connected NetworkX Louvain fallback.")

    # 2. Deterministic connected fallback: seeded NetworkX Louvain with recursive sub-clustering
    level0_raw = list(nx.community.louvain_communities(graph, weight="weight", seed=random_seed))
    level0_connected = _split_into_connected_clusters(graph, level0_raw)
    levels_fallback: Dict[int, List[Tuple[str, List[str], Optional[str]]]] = {
        0: [
            (f"L0_C{idx:03d}", members, None)
            for idx, members in enumerate(level0_connected, start=1)
        ]
    }

    prev_level = levels_fallback[0]
    for lvl in range(1, max_lvl):
        next_entries: List[Tuple[str, List[str], Optional[str]]] = []
        sub_counter = 1
        for parent_cid, parent_members, _ in prev_level:
            if len(parent_members) <= max_sz:
                next_entries.append((f"L{lvl}_C{sub_counter:03d}", list(parent_members), parent_cid))
                sub_counter += 1
                continue
            sub_g = graph.subgraph(parent_members).copy()
            sub_raw = list(
                nx.community.louvain_communities(
                    sub_g,
                    weight="weight",
                    resolution=1.5,
                    seed=random_seed + lvl,
                )
            )
            sub_connected = _split_into_connected_clusters(sub_g, sub_raw)
            for sub_members in sub_connected:
                next_entries.append((f"L{lvl}_C{sub_counter:03d}", list(sub_members), parent_cid))
                sub_counter += 1
        if next_entries:
            levels_fallback[lvl] = next_entries
            prev_level = next_entries

    return "networkx_louvain_connected_fallback", levels_fallback


def build_communities(
    as_of: date | str | PiTContext,
    *,
    universe_id: str = DEFAULT_UNIVERSE_ID,
    max_cluster_size: int = 10,
    max_levels: int = 2,
    random_seed: int = 42,
    edges: Optional[Iterable[Mapping[str, Any]]] = None,
    pg_conn: Optional[Any] = None,
    snapshot_dir: Optional[Path | str] = None,
    include_unanchored: bool = False,
) -> List[Community]:
    """Build deterministic Point-in-Time hierarchical structural communities as of `as_of` (ADR-00037)."""
    cutoff = _coerce_as_of_date(as_of)
    as_of_iso = cutoff.isoformat()

    raw_edges: Sequence[Mapping[str, Any]]
    if edges is not None:
        raw_edges = list(edges)
    elif snapshot_dir is not None:
        raw_edges = _load_edges_from_snapshot(snapshot_dir)
    else:
        raw_edges = load_ledger_edges(conn=pg_conn)

    pit_edges = filter_structural_edges_pit(
        raw_edges,
        cutoff,
        include_unanchored=include_unanchored,
    )
    if not pit_edges:
        return []

    graph = nx.Graph()
    for e in pit_edges:
        src = e["source_id"]
        tgt = e["target_id"]
        w = float(e["weight"])
        if graph.has_edge(src, tgt):
            graph[src][tgt]["weight"] = max(float(graph[src][tgt]["weight"]), w)
        else:
            graph.add_edge(src, tgt, weight=w)

    algo, levels = _run_hierarchical_partition(
        graph,
        max_cluster_size=max_cluster_size,
        max_levels=max_levels,
        random_seed=random_seed,
    )

    communities: List[Community] = []
    for lvl in sorted(levels.keys()):
        for cid, members, parent_cid in levels[lvl]:
            member_set = set(members)
            internal_edges = [
                e for e in pit_edges if e["source_id"] in member_set and e["target_id"] in member_set
            ]
            internal_edges.sort(key=lambda e: (-float(e["share"]), e["source_id"], e["target_id"]))
            top_edges = internal_edges[:15]
            cids = sorted(
                {
                    chunk_id
                    for e in top_edges
                    for chunk_id in e.get("evidence_chunk_ids", [])
                    if str(chunk_id).strip()
                }
            )
            communities.append(
                Community(
                    universe_id=universe_id,
                    as_of=as_of_iso,
                    level=lvl,
                    community_id=cid,
                    member_ids=list(members),
                    summary="",
                    key_edges=top_edges,
                    evidence_chunk_ids=cids,
                    parent_community_id=parent_cid,
                    algorithm=algo,
                )
            )
    return communities


def _deterministic_community_summary(comm: Community, snippets: Mapping[str, Dict[str, Any]]) -> str:
    members_preview = ", ".join(comm.member_ids[:8])
    if len(comm.member_ids) > 8:
        members_preview += f" (+{len(comm.member_ids) - 8} more)"
    edge_parts = [
        f"{e['source_id']} -[{e['rel_type']} share={e['share']:.2f}]-> {e['target_id']}"
        for e in comm.key_edges[:5]
    ]
    edge_desc = "; ".join(edge_parts) if edge_parts else "structural cluster"
    cited = [f"[{cid}]" for cid in comm.evidence_chunk_ids[:3] if cid in snippets]
    cite_str = f" Evidence: {' '.join(cited)}." if cited else ""
    return (
        f"Structural supply-chain cluster {comm.community_id} (level {comm.level}, as of {comm.as_of}) "
        f"encompassing {len(comm.member_ids)} entities ({members_preview}). "
        f"Key dependencies: {edge_desc}.{cite_str}"
    )


def summarize_communities(
    communities: Sequence[Community],
    *,
    model_client: Optional[Any] = None,
    pg_conn: Optional[Any] = None,
    chunks_parquet_path: Optional[Path | str] = None,
    use_llm: bool = True,
) -> List[Community]:
    """Generate `COMMUNITY_REPORT` summaries via `YarnBallModelClient` grounded in chunk passages."""
    if not communities:
        return []

    all_cids = sorted({cid for c in communities for cid in c.evidence_chunk_ids if cid})
    resolved_chunks: Dict[str, Dict[str, Any]] = {}
    if all_cids and (pg_conn is not None or chunks_parquet_path is not None):
        try:
            resolved_chunks = fetch_chunks_by_ids(
                all_cids,
                conn=pg_conn,
                chunks_parquet_path=chunks_parquet_path,
            )
        except Exception as exc:
            logger.debug("Community chunk snippet fetch skipped/unavailable: %s", exc)

    active_client = model_client
    if use_llm and active_client is None:
        from sft_pipeline.serving.model_client import YarnBallModelClient

        active_client = YarnBallModelClient()

    summarized: List[Community] = []
    for comm in communities:
        summary_text = ""
        if use_llm and active_client is not None:
            edge_lines = "\n".join(
                f"- ({e['source_id']}) -[:{e['rel_type']} {{share: {e['share']:.2f}, valid_from: '{e.get('valid_from')}'}}]-> ({e['target_id']})"
                for e in comm.key_edges[:10]
            )
            passage_lines = "\n".join(
                f"- [{cid}] {' '.join(str(resolved_chunks[cid].get('text', '')).split())[:240]}"
                for cid in comm.evidence_chunk_ids[:5]
                if cid in resolved_chunks
            )
            prompt = (
                f"Summarize structural supply-chain community {comm.community_id} (level {comm.level}) "
                f"as of {comm.as_of} for universe {comm.universe_id}.\n"
                f"Entities ({len(comm.member_ids)}): {', '.join(comm.member_ids[:20])}\n"
                f"Key structural edges:\n{edge_lines or '- (none)'}\n"
                f"Supporting evidence passages:\n{passage_lines or '- (none)'}\n"
                f"Write a concise factual operational risk summary citing key bottlenecks and concentration dependencies."
            )
            resp = active_client.generate(
                prompt=prompt,
                task="COMMUNITY_REPORT",
                temperature=0.1,
                enforce_think_prefix=False,
            )
            if not getattr(resp, "error", None) and getattr(resp, "final_output", "").strip():
                summary_text = str(resp.final_output).strip()

        if not summary_text:
            summary_text = _deterministic_community_summary(comm, resolved_chunks)

        summarized.append(
            Community(
                universe_id=comm.universe_id,
                as_of=comm.as_of,
                level=comm.level,
                community_id=comm.community_id,
                member_ids=list(comm.member_ids),
                summary=summary_text,
                key_edges=list(comm.key_edges),
                evidence_chunk_ids=list(comm.evidence_chunk_ids),
                parent_community_id=comm.parent_community_id,
                algorithm=comm.algorithm,
            )
        )
    return summarized


def persist_community_reports(
    reports: Sequence[Community],
    *,
    conn: Optional[Any] = None,
    embed: bool = False,
    batch_gpu: bool = False,
) -> int:
    """Persist Point-in-Time community reports into PostgreSQL `community_reports`."""
    if not reports:
        return 0

    vec_literals: List[Optional[str]] = [None] * len(reports)
    if embed:
        from embedding.embedder import embed_documents, format_pgvector_literal

        texts = [
            f"{r.community_id} ({', '.join(r.member_ids[:10])}): {r.summary}" for r in reports
        ]
        vectors = embed_documents(texts, batch_gpu=batch_gpu)
        vec_literals = [format_pgvector_literal(v) for v in vectors]

    sql = """
        INSERT INTO community_reports (
            universe_id, as_of, level, community_id,
            member_ids, summary, key_edges, evidence_chunk_ids, embedding
        )
        VALUES (%s, %s::date, %s, %s, %s, %s, %s::jsonb, %s, %s::vector)
        ON CONFLICT (universe_id, as_of, level, community_id) DO UPDATE SET
            member_ids = EXCLUDED.member_ids,
            summary = EXCLUDED.summary,
            key_edges = EXCLUDED.key_edges,
            evidence_chunk_ids = EXCLUDED.evidence_chunk_ids,
            embedding = COALESCE(EXCLUDED.embedding, community_reports.embedding);
    """

    rows = [
        (
            r.universe_id,
            r.as_of,
            int(r.level),
            r.community_id,
            list(r.member_ids),
            r.summary,
            json.dumps(r.key_edges, sort_keys=True, default=str),
            list(r.evidence_chunk_ids),
            vec_lit,
        )
        for r, vec_lit in zip(reports, vec_literals)
    ]

    def _exec(active_conn: Any) -> int:
        with active_conn.cursor() as cur:
            for row in rows:
                cur.execute(sql, row)
        active_conn.commit()
        return len(rows)

    if conn is not None:
        return _exec(conn)

    with pg_connection() as active_conn:
        return _exec(active_conn)


def export_communities_parquet(
    reports: Sequence[Community],
    output_path: Path | str,
) -> Path:
    """Export Point-in-Time community reports to `communities.parquet` for offline snapshots."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("universe_id", pa.string()),
            ("as_of", pa.string()),
            ("level", pa.int64()),
            ("community_id", pa.string()),
            ("member_ids_json", pa.string()),
            ("summary", pa.string()),
            ("key_edges_json", pa.string()),
            ("evidence_chunk_ids_json", pa.string()),
            ("parent_community_id", pa.string()),
            ("algorithm", pa.string()),
        ]
    )
    rows = [
        {
            "universe_id": r.universe_id,
            "as_of": r.as_of,
            "level": int(r.level),
            "community_id": r.community_id,
            "member_ids_json": json.dumps(r.member_ids),
            "summary": r.summary,
            "key_edges_json": json.dumps(r.key_edges, sort_keys=True, default=str),
            "evidence_chunk_ids_json": json.dumps(r.evidence_chunk_ids),
            "parent_community_id": r.parent_community_id,
            "algorithm": r.algorithm,
        }
        for r in reports
    ]
    table = pa.Table.from_pylist(rows, schema=schema) if rows else pa.Table.from_batches([], schema=schema)
    pq.write_table(table, path, compression="snappy")
    return path


def load_community_reports(
    as_of: date | str | PiTContext,
    *,
    universe_id: str = DEFAULT_UNIVERSE_ID,
    level: Optional[int] = None,
    conn: Optional[Any] = None,
    communities_parquet_path: Optional[Path | str] = None,
) -> List[Community]:
    """Load cached community reports for the latest checkpoint `as_of <= requested` (ADR-00037 Invariant #1)."""
    cutoff = _coerce_as_of_date(as_of)
    cutoff_iso = cutoff.isoformat()

    if communities_parquet_path is not None:
        p = Path(communities_parquet_path)
        if not p.is_file():
            return []
        raw_rows = pq.read_table(p).to_pylist()
        eligible: List[Community] = []
        for row in raw_rows:
            if str(row.get("universe_id") or "") != universe_id:
                continue
            row_as_of = _parse_optional_date(row.get("as_of"))
            if row_as_of is None or row_as_of > cutoff:
                continue
            row_lvl = int(row.get("level") or 0)
            if level is not None and row_lvl != level:
                continue
            eligible.append(
                Community(
                    universe_id=str(row["universe_id"]),
                    as_of=row_as_of.isoformat(),
                    level=row_lvl,
                    community_id=str(row["community_id"]),
                    member_ids=list(json.loads(row.get("member_ids_json") or "[]")),
                    summary=str(row.get("summary") or ""),
                    key_edges=list(json.loads(row.get("key_edges_json") or "[]")),
                    evidence_chunk_ids=list(json.loads(row.get("evidence_chunk_ids_json") or "[]")),
                    parent_community_id=row.get("parent_community_id"),
                    algorithm=str(row.get("algorithm") or "hierarchical_leiden"),
                )
            )
        if not eligible:
            return []
        latest_as_of = max(c.as_of for c in eligible)
        return sorted(
            [c for c in eligible if c.as_of == latest_as_of],
            key=lambda c: (c.level, c.community_id),
        )

    def _query_pg(active_conn: Any) -> List[Community]:
        with active_conn.cursor() as cur:
            cur.execute(
                """
                SELECT MAX(as_of)
                FROM community_reports
                WHERE universe_id = %s AND as_of <= %s::date;
                """,
                (universe_id, cutoff_iso),
            )
            max_row = cur.fetchone()
            if not max_row or max_row[0] is None:
                return []
            latest = str(max_row[0])[:10]

            if level is not None:
                cur.execute(
                    """
                    SELECT universe_id, as_of, level, community_id,
                           member_ids, summary, key_edges, evidence_chunk_ids
                    FROM community_reports
                    WHERE universe_id = %s AND as_of = %s::date AND level = %s
                    ORDER BY level ASC, community_id ASC;
                    """,
                    (universe_id, latest, int(level)),
                )
            else:
                cur.execute(
                    """
                    SELECT universe_id, as_of, level, community_id,
                           member_ids, summary, key_edges, evidence_chunk_ids
                    FROM community_reports
                    WHERE universe_id = %s AND as_of = %s::date
                    ORDER BY level ASC, community_id ASC;
                    """,
                    (universe_id, latest),
                )
            out: List[Community] = []
            for row in cur.fetchall() or []:
                edges_val = row[6]
                if isinstance(edges_val, str):
                    edges_val = json.loads(edges_val)
                out.append(
                    Community(
                        universe_id=str(row[0]),
                        as_of=str(row[1])[:10],
                        level=int(row[2]),
                        community_id=str(row[3]),
                        member_ids=list(row[4] or []),
                        summary=str(row[5] or ""),
                        key_edges=list(edges_val or []),
                        evidence_chunk_ids=list(row[7] or []),
                    )
                )
            return out

    if conn is not None:
        return _query_pg(conn)

    with pg_connection() as active_conn:
        return _query_pg(active_conn)


def _score_community_relevance(query: str, comm: Community) -> float:
    q_tokens = set(_TOKEN_RE.findall(query.lower()))
    if not q_tokens:
        return float(len(comm.member_ids)) * 0.01
    doc_text = " ".join(
        [
            comm.community_id,
            " ".join(comm.member_ids),
            comm.summary,
            " ".join(
                f"{e.get('source_id', '')} {e.get('target_id', '')} {e.get('input_group', '')}"
                for e in comm.key_edges
            ),
        ]
    ).lower()
    doc_tokens = set(_TOKEN_RE.findall(doc_text))
    overlap = len(q_tokens.intersection(doc_tokens))
    return float(overlap) / float(len(q_tokens)) + (0.001 * min(len(comm.member_ids), 100))


def search_community_reports_global(
    query: str,
    ctx: PiTContext,
    *,
    universe_id: str = DEFAULT_UNIVERSE_ID,
    top_k: int = 5,
    communities: Optional[Sequence[Community | Mapping[str, Any]]] = None,
    conn: Optional[Any] = None,
    communities_parquet_path: Optional[Path | str] = None,
    model_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Map-reduce global search over cached PiT community reports strictly at `as_of <= ctx.as_of`."""
    if not isinstance(ctx, PiTContext):
        raise TypeError(f"ctx must be a PiTContext instance, got {type(ctx).__name__}")

    loaded: List[Community] = []
    if communities is not None:
        for item in communities:
            if isinstance(item, Community):
                c_obj = item
            else:
                c_obj = Community(
                    universe_id=str(item.get("universe_id") or universe_id),
                    as_of=str(item.get("as_of") or ctx.as_of_iso)[:10],
                    level=int(item.get("level") or 0),
                    community_id=str(item.get("community_id") or "L0_C001"),
                    member_ids=list(item.get("member_ids") or []),
                    summary=str(item.get("summary") or ""),
                    key_edges=list(item.get("key_edges") or []),
                    evidence_chunk_ids=list(item.get("evidence_chunk_ids") or []),
                )
            c_date = _parse_optional_date(c_obj.as_of)
            if c_date is not None and c_date <= ctx.as_of and c_obj.universe_id == universe_id:
                loaded.append(c_obj)
        if loaded:
            latest = max(c.as_of for c in loaded)
            loaded = [c for c in loaded if c.as_of == latest]
    else:
        loaded = load_community_reports(
            ctx.as_of,
            universe_id=universe_id,
            conn=conn,
            communities_parquet_path=communities_parquet_path,
        )

    scored = sorted(
        ((_score_community_relevance(query, c), c) for c in loaded),
        key=lambda pair: (-pair[0], pair[1].level, pair[1].community_id),
    )[: max(1, int(top_k))]

    selected = [
        {**c.to_dict(), "relevance_score": round(score, 4)}
        for score, c in scored
    ]

    if not selected:
        return {
            "universe_id": universe_id,
            "as_of": ctx.as_of_iso,
            "checkpoint_as_of": None,
            "communities": [],
            "verbalized_context": (
                f"No Point-in-Time community reports cached for universe '{universe_id}' as of {ctx.as_of_iso}."
            ),
            "synthesis": (
                f"No cached structural community coverage found for '{universe_id}' on or before {ctx.as_of_iso}."
            ),
            "think_trace": None,
        }

    checkpoint_as_of = selected[0]["as_of"]
    context_lines = [
        f"### Point-in-Time Structural Communities (universe={universe_id}, checkpoint={checkpoint_as_of}):"
    ]
    for idx, comm in enumerate(selected, start=1):
        members_str = ", ".join(comm["member_ids"][:10])
        context_lines.append(
            f"- [C{idx} | {comm['community_id']} L{comm['level']}] Members: {members_str}\n"
            f"  Summary: {comm['summary']}"
        )
    verbalized_context = "\n".join(context_lines)

    synthesis_text = ""
    think_trace: Optional[str] = None
    if model_client is not None:
        prompt = (
            f"Global Thematic Question (as of {ctx.as_of_iso}): {query}\n\n"
            f"{verbalized_context}\n\n"
            f"Synthesize a grounded cross-cluster operational risk answer citing [C1], [C2], etc."
        )
        resp = model_client.generate(
            prompt=prompt,
            task="COMMUNITY_REPORT",
            temperature=0.1,
            enforce_think_prefix=False,
        )
        if not getattr(resp, "error", None) and getattr(resp, "final_output", "").strip():
            synthesis_text = str(resp.final_output).strip()
            think_trace = getattr(resp, "think_trace", None)

    if not synthesis_text:
        synthesis_text = (
            f"Across {len(selected)} structural community cluster(s) as of {checkpoint_as_of}: "
            + " ".join(f"[C{idx}] {c['summary']}" for idx, c in enumerate(selected[:3], start=1))
        )

    return {
        "universe_id": universe_id,
        "as_of": ctx.as_of_iso,
        "checkpoint_as_of": checkpoint_as_of,
        "communities": selected,
        "verbalized_context": verbalized_context,
        "synthesis": synthesis_text,
        "think_trace": think_trace,
    }


__all__ = [
    "Community",
    "DEFAULT_UNIVERSE_ID",
    "build_communities",
    "export_communities_parquet",
    "filter_structural_edges_pit",
    "load_community_reports",
    "persist_community_reports",
    "search_community_reports_global",
    "summarize_communities",
]
