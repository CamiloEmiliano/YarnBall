"""Contextual node embedding generation and PostgreSQL pgvector persistence (ADR-00021)."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Mapping, Sequence, Tuple

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


def store_node_embeddings(
    nodes: Sequence[Mapping[str, Any]],
    source_hash: str = "",
) -> None:
    """Persist contextual node embeddings and non-destructive source lineage into pgvector."""
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

    embeddings = _embedder.validate_embedding_vectors(_embedder.embed_texts(texts))

    clean_hash = str(source_hash or "").strip()
    records: List[Tuple[str, str, str, str]] = []
    source_records: List[Tuple[str, str, str]] = []
    for node, emb in zip(valid_nodes, embeddings):
        node_id = str(node.get("id", "")).strip()
        entity_type = str(node.get("type") or node.get("entity_type") or "Entity").strip() or "Entity"
        vec_literal = _embedder.format_pgvector_literal(emb)
        records.append((node_id, entity_type, vec_literal, clean_hash))
        if clean_hash:
            source_records.append((node_id, clean_hash, entity_type))

    close_after = False
    try:
        from tools.init_db import _connect_db

        conn = _connect_db(os.getenv("FINANCIAL_RAG_DB", "financial_rag"))
        close_after = True
    except Exception:
        from .db import pg_connection

        conn = pg_connection()
        close_after = False

    upsert_sql = """
        INSERT INTO node_embeddings (node_id, entity_type, embedding, source_hash)
        VALUES (%s, %s, %s::vector, %s)
        ON CONFLICT (node_id) DO UPDATE
        SET embedding = EXCLUDED.embedding,
            entity_type = EXCLUDED.entity_type,
            source_hash = EXCLUDED.source_hash;
    """
    ensure_sources_sql = """
        CREATE TABLE IF NOT EXISTS node_embedding_sources (
            node_id       TEXT NOT NULL,
            source_hash   VARCHAR(64) NOT NULL,
            entity_type   VARCHAR(64),
            first_seen_at TIMESTAMPTZ DEFAULT now(),
            PRIMARY KEY (node_id, source_hash)
        );
    """
    upsert_source_sql = """
        INSERT INTO node_embedding_sources (node_id, source_hash, entity_type)
        VALUES (%s, %s, %s)
        ON CONFLICT (node_id, source_hash) DO NOTHING;
    """
    try:
        with conn.cursor() as cur:
            for rec in records:
                cur.execute(upsert_sql, rec)
            if source_records:
                cur.execute(ensure_sources_sql)
                for src_rec in source_records:
                    cur.execute(upsert_source_sql, src_rec)
        conn.commit()
        logger.info("Persisted %d contextual node embeddings to PGVECTOR", len(records))
    except Exception as exc:
        logger.error("Failed to store embeddings in PGVECTOR: %s", exc)
        raise
    finally:
        if close_after and conn:
            conn.close()
