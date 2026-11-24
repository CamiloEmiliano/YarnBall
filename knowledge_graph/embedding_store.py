"""Contextual node embedding generation and PostgreSQL pgvector persistence (ADR-00021)."""

from __future__ import annotations

from datetime import date, datetime, timezone
import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import embedding.embedder as _embedder

logger = logging.getLogger(__name__)

_CONTEXT_PRIORITY_KEYS: Tuple[str, ...] = (
    "context",
    "description",
    "summary",
    "role",
    "title",
    "headline",
)
_METADATA_TAG_KEYS: Tuple[str, ...] = (
    "industry",
    "sector",
    "jurisdiction",
    "country",
    "name",
)


def format_node_embedding_text(
    node: Mapping[str, Any],
    max_context_chars: int = 240,
) -> str:
    """Format a graph node with entity type, ticker/CIK, and bounded context snippet."""
    node_id = str(node.get("id", "")).strip()
    entity_type = str(node.get("type") or node.get("entity_type") or "Entity").strip() or "Entity"
    props_raw = node.get("properties")
    props: Mapping[str, Any] = props_raw if isinstance(props_raw, Mapping) else {}

    ticker = str(node.get("ticker") or props.get("ticker") or "").strip() or "N/A"
    cik = str(node.get("cik") or props.get("cik") or "").strip() or "N/A"

    context_parts: List[str] = []
    for key in _CONTEXT_PRIORITY_KEYS:
        val = node.get(key) or props.get(key)
        if val is not None:
            cleaned = " ".join(str(val).split())
            if cleaned:
                context_parts.append(cleaned)
                break

    if not context_parts:
        for key in _METADATA_TAG_KEYS:
            val = node.get(key) or props.get(key)
            if val is None:
                continue
            cleaned = " ".join(str(val).split())
            if not cleaned:
                continue
            if key == "name" and cleaned.lower() == node_id.lower():
                continue
            context_parts.append(f"{key}={cleaned}")

    context_snippet = "; ".join(context_parts) if context_parts else "N/A"
    if len(context_snippet) > max_context_chars:
        context_snippet = context_snippet[: max_context_chars - 3].rstrip() + "..."

    return (
        f"{node_id} | type={entity_type} | ticker={ticker} | cik={cik} | context={context_snippet}"
    )


def _resolve_embedding_validity(
    valid_from: Optional[Union[date, datetime, str]],
    known_at: Optional[Union[datetime, str]],
) -> Tuple[str, Optional[str], bool]:
    """Resolve `(valid_from_iso, known_at_iso, is_unanchored)` for SCD Type 2 node_embeddings (ADR-00033)."""
    from retrieval.temporal import actionable_session

    known_at_iso: Optional[str] = None
    if known_at is not None:
        if isinstance(known_at, datetime):
            dt_utc = known_at.replace(tzinfo=timezone.utc) if known_at.tzinfo is None else known_at.astimezone(timezone.utc)
            known_at_iso = dt_utc.isoformat()
        else:
            raw_k = str(known_at).strip()
            if raw_k and raw_k.lower() not in ("none", "null", "nan"):
                known_at_iso = raw_k

    if valid_from is not None:
        if isinstance(valid_from, datetime):
            return valid_from.date().isoformat(), known_at_iso, False
        if isinstance(valid_from, date):
            return valid_from.isoformat(), known_at_iso, False
        raw_v = str(valid_from).strip()
        if raw_v and raw_v.lower() not in ("none", "null", "nan"):
            return date.fromisoformat(raw_v[:10]).isoformat(), known_at_iso, False

    if known_at is not None:
        session_date, actionable_dt_utc, _ = actionable_session(known_at)
        if session_date is not None:
            return (
                session_date.isoformat(),
                (actionable_dt_utc.isoformat() if actionable_dt_utc else known_at_iso),
                False,
            )

    # Fail-closed: unanchored writes are marked is_unanchored=True so historical mode never reads them.
    return date.today().isoformat(), known_at_iso, True


def store_node_embeddings(
    nodes: Sequence[Mapping[str, Any]],
    source_hash: str = "",
    *,
    valid_from: Optional[Union[date, datetime, str]] = None,
    known_at: Optional[Union[datetime, str]] = None,
) -> None:
    """Persist SCD Type 2 versioned contextual node embeddings and lineage into pgvector (ADR-00033)."""
    if not nodes:
        return

    valid_nodes: List[Mapping[str, Any]] = []
    texts: List[str] = []
    for node in nodes:
        node_id = str(node.get("id", "")).strip()
        if not node_id:
            continue
        valid_nodes.append(node)
        texts.append(format_node_embedding_text(node))

    if not valid_nodes:
        return

    embeddings = _embedder.validate_embedding_vectors(_embedder.embed_documents(texts))
    valid_from_iso, known_at_iso, is_unanchored = _resolve_embedding_validity(valid_from, known_at)
    source_published_at = known_at_iso or (None if is_unanchored else f"{valid_from_iso}T00:00:00+00:00")

    clean_hash = str(source_hash or "").strip()
    close_records: List[Tuple[str, str, str, str]] = []
    records: List[Tuple[str, str, str, str, str, Optional[str], bool]] = []
    source_records: List[Tuple[str, str, str, Optional[str]]] = []
    for node, emb in zip(valid_nodes, embeddings):
        node_id = str(node.get("id", "")).strip()
        entity_type = str(node.get("type") or node.get("entity_type") or "Entity").strip() or "Entity"
        vec_literal = _embedder.format_pgvector_literal(emb)
        close_records.append((valid_from_iso, node_id, valid_from_iso, valid_from_iso))
        records.append(
            (node_id, valid_from_iso, entity_type, vec_literal, clean_hash, known_at_iso, is_unanchored)
        )
        if clean_hash:
            source_records.append((node_id, clean_hash, entity_type, source_published_at))

    close_after = False
    try:
        from tools.init_db import _connect_db

        conn = _connect_db(os.getenv("FINANCIAL_RAG_DB", "financial_rag"))
        close_after = True
    except Exception:
        from .db import pg_connection

        conn = pg_connection()
        close_after = False

    close_prior_sql = """
        UPDATE node_embeddings
        SET valid_to = %s::date,
            updated_at = now()
        WHERE node_id = %s
          AND valid_from < %s::date
          AND (valid_to IS NULL OR valid_to > %s::date);
    """
    upsert_sql = """
        INSERT INTO node_embeddings (
            node_id, valid_from, valid_to, entity_type, embedding, source_hash, known_at, is_unanchored
        )
        VALUES (%s, %s::date, NULL, %s, %s::vector, %s, %s::timestamptz, %s)
        ON CONFLICT (node_id, valid_from) DO UPDATE
        SET embedding = EXCLUDED.embedding,
            entity_type = EXCLUDED.entity_type,
            source_hash = EXCLUDED.source_hash,
            known_at = COALESCE(EXCLUDED.known_at, node_embeddings.known_at),
            is_unanchored = (node_embeddings.is_unanchored AND EXCLUDED.is_unanchored),
            updated_at = now();
    """
    ensure_sources_sql = """
        CREATE TABLE IF NOT EXISTS node_embedding_sources (
            node_id             TEXT NOT NULL,
            source_hash         VARCHAR(64) NOT NULL,
            entity_type         VARCHAR(64),
            source_published_at TIMESTAMPTZ,
            first_seen_at       TIMESTAMPTZ DEFAULT now(),
            PRIMARY KEY (node_id, source_hash)
        );
    """
    upsert_source_sql = """
        INSERT INTO node_embedding_sources (node_id, source_hash, entity_type, source_published_at)
        VALUES (%s, %s, %s, %s::timestamptz)
        ON CONFLICT (node_id, source_hash) DO UPDATE
        SET source_published_at = COALESCE(node_embedding_sources.source_published_at, EXCLUDED.source_published_at);
    """
    try:
        with conn.cursor() as cur:
            for close_rec, rec in zip(close_records, records):
                cur.execute(close_prior_sql, close_rec)
                cur.execute(upsert_sql, rec)
            if source_records:
                cur.execute(ensure_sources_sql)
                for src_rec in source_records:
                    cur.execute(upsert_source_sql, src_rec)
        conn.commit()
        logger.info(
            "Persisted %d versioned node embeddings to PGVECTOR (valid_from=%s, is_unanchored=%s)",
            len(records),
            valid_from_iso,
            is_unanchored,
        )
    except Exception as exc:
        logger.error("Failed to store embeddings in PGVECTOR: %s", exc)
        raise
    finally:
        if close_after and conn:
            conn.close()


def backfill_node_embedding_validity(conn: Any) -> Dict[str, int]:
    """Backfill `source_published_at`, `node_embeddings.valid_from`, and legacy sentinels (ADR-00033 Section 4.6)."""
    updated_news_sources = 0
    updated_sec_sources = 0
    updated_news_embeddings = 0
    updated_sec_embeddings = 0
    marked_unanchored_embeddings = 0
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE node_embedding_sources nes
            SET source_published_at = fnq.published_at
            FROM financial_news_queue fnq
            WHERE nes.source_hash = fnq.source_hash
              AND nes.source_published_at IS NULL
              AND fnq.published_at IS NOT NULL;
            """
        )
        updated_news_sources = max(0, int(getattr(cur, "rowcount", 0) or 0))
        cur.execute(
            """
            UPDATE node_embedding_sources nes
            SET source_published_at = sfq.filing_date::timestamptz
            FROM sec_filings_queue sfq
            WHERE split_part(nes.source_hash, ':', 1) = sfq.accession_number
              AND nes.source_published_at IS NULL
              AND sfq.filing_date IS NOT NULL;
            """
        )
        updated_sec_sources = max(0, int(getattr(cur, "rowcount", 0) or 0))
        cur.execute(
            """
            UPDATE node_embeddings ne
            SET valid_from = (fnq.published_at AT TIME ZONE 'America/New_York')::date,
                known_at = fnq.published_at,
                is_unanchored = FALSE
            FROM financial_news_queue fnq
            WHERE ne.source_hash = fnq.source_hash
              AND ne.known_at IS NULL
              AND fnq.published_at IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM node_embeddings ne2
                  WHERE ne2.node_id = ne.node_id
                    AND ne2.valid_from = (fnq.published_at AT TIME ZONE 'America/New_York')::date
                    AND ne2.ctid <> ne.ctid
              );
            """
        )
        updated_news_embeddings = max(0, int(getattr(cur, "rowcount", 0) or 0))
        cur.execute(
            """
            UPDATE node_embeddings ne
            SET valid_from = sfq.filing_date::date,
                known_at = sfq.filing_date::timestamptz,
                is_unanchored = FALSE
            FROM sec_filings_queue sfq
            WHERE split_part(ne.source_hash, ':', 1) = sfq.accession_number
              AND ne.known_at IS NULL
              AND sfq.filing_date IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM node_embeddings ne2
                  WHERE ne2.node_id = ne.node_id
                    AND ne2.valid_from = sfq.filing_date::date
                    AND ne2.ctid <> ne.ctid
              );
            """
        )
        updated_sec_embeddings = max(0, int(getattr(cur, "rowcount", 0) or 0))
        cur.execute(
            """
            UPDATE node_embeddings ne
            SET is_unanchored = TRUE
            WHERE ne.known_at IS NULL
              AND COALESCE(ne.is_unanchored, FALSE) = FALSE;
            """
        )
        marked_unanchored_embeddings = max(0, int(getattr(cur, "rowcount", 0) or 0))
    conn.commit()
    return {
        "updated_news_sources": updated_news_sources,
        "updated_sec_sources": updated_sec_sources,
        "updated_news_embeddings": updated_news_embeddings,
        "updated_sec_embeddings": updated_sec_embeddings,
        "marked_unanchored_embeddings": marked_unanchored_embeddings,
    }
