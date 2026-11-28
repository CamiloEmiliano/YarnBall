"""Text-Unit Document Chunk Persistence, Batched Citation Fetch & Hybrid RRF Search (ADR-00035).

Keeps raw filing and news text strictly in PostgreSQL (`document_chunks` + `chunk_entity_links`)
and offline Parquet snapshots (`chunks.parquet`), while Memgraph edges store only compact
`evidence_chunk_ids` pointers (Invariant #1, #2, #5).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from retrieval.temporal import PiTContext

logger = logging.getLogger("knowledge_graph.chunk_store")

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def slugify_section(section: Optional[str]) -> str:
    """Produce a deterministic lowercase slug for a filing or news section name."""
    raw = str(section or "").strip().lower()
    if not raw:
        return "body"
    slug = _NON_ALNUM_RE.sub("_", raw).strip("_")
    return slug or "body"


def make_chunk_id(doc_id: str, section: Optional[str], ordinal: int) -> str:
    """Format deterministic chunk ID `<accession_or_source_hash>#<section_slug>:c<ordinal>` (ADR-00035)."""
    clean_doc = str(doc_id or "unknown").strip()
    sec_slug = slugify_section(section)
    ord_val = max(1, int(ordinal))
    return f"{clean_doc}#{sec_slug}:c{ord_val:03d}"


@dataclass(frozen=True)
class DocumentChunkRecord:
    """Canonical text-unit record persisted in `document_chunks` and `chunks.parquet` (ADR-00035 / ADR-00047)."""

    chunk_id: str
    doc_id: str
    source_type: str
    cik: Optional[str]
    section: str
    char_start: int
    char_end: int
    text: str
    valid_from: Optional[str] = None
    known_at: Optional[str] = None
    form_type: Optional[str] = None
    sec_graph_layer: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _resolve_chunk_form_and_layer(
    raw_form: Optional[Any],
    raw_layer: Optional[Any],
) -> Tuple[Optional[str], Optional[str]]:
    """Normalize SEC `form_type` and derive canonical `sec_graph_layer` via `ingestion.sec.canon`."""
    from ingestion.sec.canon import get_form_spec, normalize_form_type

    form_str = str(raw_form).strip() if raw_form is not None else ""
    norm_form = normalize_form_type(form_str) if form_str else None
    layer_str = str(raw_layer).strip() if raw_layer is not None else ""
    norm_layer: Optional[str] = layer_str or None
    if norm_form and not norm_layer:
        spec = get_form_spec(norm_form)
        if spec is not None:
            norm_layer = spec.graph_layer.value
    return norm_form, norm_layer


def store_document_chunks(
    chunks: Sequence[DocumentChunkRecord | Mapping[str, Any]],
    *,
    conn: Optional[Any] = None,
    embed: bool = False,
    batch_gpu: bool = False,
) -> int:
    """Upsert text-unit records into PostgreSQL `document_chunks` with inherited PiT & form provenance (ADR-00035 / ADR-00047)."""
    if not chunks:
        return 0

    normalized: List[Dict[str, Any]] = []
    for item in chunks:
        rec = item.to_dict() if isinstance(item, DocumentChunkRecord) else dict(item)
        cid = str(rec.get("chunk_id") or "").strip()
        txt = str(rec.get("text") or "").strip()
        if not cid or not txt:
            continue
        norm_form, norm_layer = _resolve_chunk_form_and_layer(
            rec.get("form_type"),
            rec.get("sec_graph_layer"),
        )
        normalized.append(
            {
                "chunk_id": cid,
                "doc_id": str(rec.get("doc_id") or "unknown").strip(),
                "source_type": str(rec.get("source_type") or "sec_filing").strip().lower(),
                "cik": (str(rec["cik"]).strip().zfill(10) if rec.get("cik") else None),
                "section": str(rec.get("section") or "body").strip(),
                "char_start": int(rec.get("char_start") or 0),
                "char_end": int(rec.get("char_end") or len(txt)),
                "text": txt,
                "valid_from": rec.get("valid_from"),
                "known_at": rec.get("known_at"),
                "form_type": norm_form,
                "sec_graph_layer": norm_layer,
            }
        )

    if not normalized:
        return 0

    vec_literals: List[Optional[str]] = [None] * len(normalized)
    if embed:
        from embedding.embedder import embed_documents, format_pgvector_literal

        vectors = embed_documents([r["text"] for r in normalized], batch_gpu=batch_gpu)
        vec_literals = [format_pgvector_literal(v) for v in vectors]

    upsert_sql = """
        INSERT INTO document_chunks (
            chunk_id, doc_id, source_type, cik, section,
            char_start, char_end, text, valid_from, known_at,
            form_type, sec_graph_layer, embedding
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::date, %s::timestamptz, %s, %s, %s::vector)
        ON CONFLICT (chunk_id) DO UPDATE SET
            doc_id = EXCLUDED.doc_id,
            source_type = EXCLUDED.source_type,
            cik = COALESCE(EXCLUDED.cik, document_chunks.cik),
            section = EXCLUDED.section,
            char_start = EXCLUDED.char_start,
            char_end = EXCLUDED.char_end,
            text = EXCLUDED.text,
            valid_from = COALESCE(document_chunks.valid_from, EXCLUDED.valid_from),
            known_at = COALESCE(document_chunks.known_at, EXCLUDED.known_at),
            form_type = COALESCE(EXCLUDED.form_type, document_chunks.form_type),
            sec_graph_layer = COALESCE(EXCLUDED.sec_graph_layer, document_chunks.sec_graph_layer),
            embedding = COALESCE(EXCLUDED.embedding, document_chunks.embedding);
    """

    rows = [
        (
            r["chunk_id"],
            r["doc_id"],
            r["source_type"],
            r["cik"],
            r["section"],
            r["char_start"],
            r["char_end"],
            r["text"],
            r["valid_from"],
            r["known_at"],
            r["form_type"],
            r["sec_graph_layer"],
            vec_lit,
        )
        for r, vec_lit in zip(normalized, vec_literals)
    ]

    def _exec(active_conn: Any) -> int:
        with active_conn.cursor() as cur:
            for row in rows:
                cur.execute(upsert_sql, row)
        active_conn.commit()
        return len(rows)

    if conn is not None:
        return _exec(conn)

    from knowledge_graph.db import pg_connection

    with pg_connection() as active_conn:
        return _exec(active_conn)


def store_chunk_entity_links(
    links: Sequence[Tuple[str, str, Optional[str]]],
    *,
    conn: Optional[Any] = None,
) -> int:
    """Persist `(chunk_id, node_id, valid_from)` provenance links into PostgreSQL `chunk_entity_links`."""
    cleaned: List[Tuple[str, str, Optional[str]]] = []
    seen: set[Tuple[str, str]] = set()
    for chunk_id, node_id, valid_from in links:
        cid = str(chunk_id or "").strip()
        nid = str(node_id or "").strip()
        if not cid or not nid or (cid, nid) in seen:
            continue
        seen.add((cid, nid))
        cleaned.append((cid, nid, valid_from))

    if not cleaned:
        return 0

    sql = """
        INSERT INTO chunk_entity_links (chunk_id, node_id, valid_from)
        VALUES (%s, %s, %s::date)
        ON CONFLICT (chunk_id, node_id) DO UPDATE SET
            valid_from = COALESCE(chunk_entity_links.valid_from, EXCLUDED.valid_from);
    """

    def _exec(active_conn: Any) -> int:
        with active_conn.cursor() as cur:
            for row in cleaned:
                cur.execute(sql, row)
        active_conn.commit()
        return len(cleaned)

    if conn is not None:
        return _exec(conn)

    from knowledge_graph.db import pg_connection

    with pg_connection() as active_conn:
        return _exec(active_conn)


def fetch_chunks_by_ids(
    chunk_ids: Sequence[str],
    *,
    conn: Optional[Any] = None,
    chunks_parquet_path: Optional[Path | str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Resolve passage text for `chunk_ids` in a single batched Postgres or offline Parquet lookup (ADR-00035)."""
    unique_ids = list(dict.fromkeys(str(cid).strip() for cid in chunk_ids if str(cid).strip()))
    if not unique_ids:
        return {}

    if chunks_parquet_path is not None:
        p = Path(chunks_parquet_path)
        if p.is_file():
            import pyarrow.parquet as pq

            want = set(unique_ids)
            table = pq.read_table(p)
            out: Dict[str, Dict[str, Any]] = {}
            for row in table.to_pylist():
                cid = str(row.get("chunk_id") or "")
                if cid in want:
                    out[cid] = dict(row)
            return out
        return {}

    sql = """
        SELECT chunk_id, doc_id, source_type, cik, section,
               char_start, char_end, text, valid_from, known_at,
               form_type, sec_graph_layer
        FROM document_chunks
        WHERE chunk_id = ANY(%s);
    """

    def _exec(active_conn: Any) -> Dict[str, Dict[str, Any]]:
        out_map: Dict[str, Dict[str, Any]] = {}
        with active_conn.cursor() as cur:
            cur.execute(sql, (unique_ids,))
            for row in cur.fetchall():
                cid = str(row[0])
                out_map[cid] = {
                    "chunk_id": cid,
                    "doc_id": row[1],
                    "source_type": row[2],
                    "cik": row[3],
                    "section": row[4],
                    "char_start": int(row[5] or 0),
                    "char_end": int(row[6] or 0),
                    "text": str(row[7] or ""),
                    "valid_from": str(row[8]) if row[8] is not None else None,
                    "known_at": str(row[9]) if row[9] is not None else None,
                    "form_type": (str(row[10]) if len(row) > 10 and row[10] is not None else None),
                    "sec_graph_layer": (str(row[11]) if len(row) > 11 and row[11] is not None else None),
                }
        return out_map

    try:
        if conn is not None:
            return _exec(conn)
        from knowledge_graph.db import pg_connection

        with pg_connection() as active_conn:
            return _exec(active_conn)
    except Exception as exc:
        logger.warning("Batched document_chunks lookup failed for %d ids: %s", len(unique_ids), exc)
        return {}


def search_chunks_hybrid_rrf(
    query: str,
    ctx: PiTContext,
    *,
    top_k: int = 10,
    rrf_k: int = 60,
    conn: Optional[Any] = None,
    allowed_form_types: Optional[Sequence[str]] = None,
    allowed_graph_layers: Optional[Sequence[str]] = None,
    reranker: Optional[Any] = None,
    rerank_top_m: int = 40,
) -> List[Dict[str, Any]]:
    """Execute Point-in-Time hybrid dense + Postgres `tsvector` lexical chunk retrieval via RRF (ADR-00035 / ADR-00047)."""
    if not isinstance(ctx, PiTContext):
        raise TypeError(f"ctx must be a PiTContext instance, got {type(ctx).__name__}")
    clean_q = str(query or "").strip()
    if not clean_q:
        return []

    from embedding.embedder import embed_query, format_pgvector_literal
    from ingestion.sec.canon import normalize_form_type

    q_vec = format_pgvector_literal(embed_query(clean_q))
    candidate_limit = max(top_k * 3, int(rerank_top_m) if reranker is not None else 20)

    norm_forms: List[str] = []
    if allowed_form_types is not None:
        norm_forms = [normalize_form_type(f) for f in allowed_form_types if str(f or "").strip()]
        if not norm_forms:
            return []

    norm_layers: List[str] = []
    if allowed_graph_layers is not None:
        norm_layers = [str(layer).strip().upper() for layer in allowed_graph_layers if str(layer or "").strip()]
        if not norm_layers:
            return []

    filter_clauses: List[str] = []
    filter_params: List[Any] = []
    if norm_forms:
        filter_clauses.append("AND form_type = ANY(%s)")
        filter_params.append(norm_forms)
    if norm_layers:
        filter_clauses.append("AND sec_graph_layer = ANY(%s)")
        filter_params.append(norm_layers)
    extra_where = ("\n          " + "\n          ".join(filter_clauses)) if filter_clauses else ""

    dense_sql = f"""
        SELECT chunk_id, doc_id, source_type, cik, section, text, valid_from, known_at,
               1 - (embedding <=> %s::vector) AS dense_score,
               form_type, sec_graph_layer
        FROM document_chunks
        WHERE embedding IS NOT NULL
          AND valid_from IS NOT NULL
          AND valid_from <= %s::date{extra_where}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """
    lexical_sql = f"""
        SELECT chunk_id, doc_id, source_type, cik, section, text, valid_from, known_at,
               ts_rank_cd(tsv, plainto_tsquery('english', %s)) AS lexical_score,
               form_type, sec_graph_layer
        FROM document_chunks
        WHERE valid_from IS NOT NULL
          AND valid_from <= %s::date
          AND tsv @@ plainto_tsquery('english', %s){extra_where}
        ORDER BY lexical_score DESC, chunk_id ASC
        LIMIT %s;
    """

    def _run(active_conn: Any) -> Tuple[List[Tuple[Any, ...]], List[Tuple[Any, ...]]]:
        with active_conn.cursor() as cur:
            cur.execute(dense_sql, (q_vec, ctx.as_of_iso, *filter_params, q_vec, candidate_limit))
            d_rows = list(cur.fetchall())
            cur.execute(lexical_sql, (clean_q, ctx.as_of_iso, clean_q, *filter_params, candidate_limit))
            l_rows = list(cur.fetchall())
        return d_rows, l_rows

    if conn is not None:
        dense_rows, lexical_rows = _run(conn)
    else:
        from knowledge_graph.db import pg_connection

        with pg_connection() as active_conn:
            dense_rows, lexical_rows = _run(active_conn)

    fused: Dict[str, Dict[str, Any]] = {}

    for rank, row in enumerate(dense_rows, start=1):
        cid = str(row[0])
        entry = fused.setdefault(
            cid,
            {
                "chunk_id": cid,
                "doc_id": row[1],
                "source_type": row[2],
                "cik": row[3],
                "section": row[4],
                "text": str(row[5] or ""),
                "valid_from": str(row[6]) if row[6] is not None else None,
                "known_at": str(row[7]) if row[7] is not None else None,
                "form_type": (str(row[9]) if len(row) > 9 and row[9] is not None else None),
                "sec_graph_layer": (str(row[10]) if len(row) > 10 and row[10] is not None else None),
                "dense_score": 0.0,
                "lexical_score": 0.0,
                "rrf_score": 0.0,
            },
        )
        entry["dense_score"] = float(row[8] or 0.0)
        entry["rrf_score"] += 1.0 / (rrf_k + rank)

    for rank, row in enumerate(lexical_rows, start=1):
        cid = str(row[0])
        entry = fused.setdefault(
            cid,
            {
                "chunk_id": cid,
                "doc_id": row[1],
                "source_type": row[2],
                "cik": row[3],
                "section": row[4],
                "text": str(row[5] or ""),
                "valid_from": str(row[6]) if row[6] is not None else None,
                "known_at": str(row[7]) if row[7] is not None else None,
                "form_type": (str(row[9]) if len(row) > 9 and row[9] is not None else None),
                "sec_graph_layer": (str(row[10]) if len(row) > 10 and row[10] is not None else None),
                "dense_score": 0.0,
                "lexical_score": 0.0,
                "rrf_score": 0.0,
            },
        )
        entry["lexical_score"] = float(row[8] or 0.0)
        entry["rrf_score"] += 1.0 / (rrf_k + rank)

    ranked = sorted(fused.values(), key=lambda item: (-float(item["rrf_score"]), item["chunk_id"]))
    limit_k = max(1, int(top_k))
    if reranker is not None and ranked:
        top_m = ranked[: max(limit_k, int(rerank_top_m))]
        scores = list(reranker.score_pairs(clean_q, [str(c.get("text") or "") for c in top_m]))
        if len(scores) != len(top_m):
            raise ValueError(
                f"Reranker returned {len(scores)} scores for {len(top_m)} candidates."
            )
        for cand, score in zip(top_m, scores):
            cand["rerank_score"] = float(score)
        top_m.sort(
            key=lambda item: (-float(item["rerank_score"]), -float(item["rrf_score"]), item["chunk_id"])
        )
        return top_m[:limit_k]

    return ranked[:limit_k]


def flag_legacy_unchunked_edges(driver: Optional[Any] = None) -> int:
    """Tag pre-ADR-00035 Memgraph edges lacking `evidence_chunk_ids` as `legacy_unchunked` (ADR-00035 Section 4.7)."""
    from knowledge_graph.memgraph_driver import get_memgraph_driver

    active_driver = driver if driver is not None else get_memgraph_driver()
    cypher = """
        MATCH ()-[r]->()
        WHERE r.evidence_chunk_ids IS NULL OR size(r.evidence_chunk_ids) = 0
        SET r.evidence_chunk_ids = [],
            r.evidence_status = 'legacy_unchunked'
        RETURN count(r) AS flagged_count
    """
    with active_driver.session() as session:
        rows = list(session.run(cypher))
        if rows:
            first = rows[0]
            if isinstance(first, dict):
                return int(first.get("flagged_count", 0) or 0)
            try:
                return int(first["flagged_count"] or 0)
            except Exception:
                return 0
    return 0


__all__ = [
    "DocumentChunkRecord",
    "fetch_chunks_by_ids",
    "flag_legacy_unchunked_edges",
    "make_chunk_id",
    "search_chunks_hybrid_rrf",
    "slugify_section",
    "store_chunk_entity_links",
    "store_document_chunks",
]
