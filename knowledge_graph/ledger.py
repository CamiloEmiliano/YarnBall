"""Durable PostgreSQL KG Assertion Ledger, Ephemeral Memgraph Hydrator & Event TTL (ADR-00036).

Implements the durable-before-ephemeral lifecycle separating:
1. Durable PostgreSQL source-of-truth tables (`kg_nodes`, `kg_edges`) storing canonical
   structural entities, typed relations, bitemporal bounds, cascade parameters, and
   `evidence_chunk_ids`.
2. Purely in-memory Memgraph RAM projection hydrated deterministically from the ledger
   on container boot (`hydrate_memgraph_from_ledger`).
3. Bounded transient `:News` RAM residency governed by `MEMGRAPH_EVENT_TTL_DAYS`
   (default: 90 days), evicting stale `:News` nodes from Memgraph while preserving full
   history in PostgreSQL.
4. Readiness inspection (`check_memgraph_readiness`) verifying Bolt connectivity and
   hydration completeness without exposing raw Memgraph Cypher to `retrieval/service.py`
   or `tools/` (ADR-00034 Invariant #1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

from knowledge_graph.db import pg_connection
from knowledge_graph.errors import DocumentStoreUnavailableError, GraphStoreUnavailableError
from knowledge_graph.memgraph_driver import get_memgraph_driver

logger = logging.getLogger(__name__)

DEFAULT_MEMGRAPH_EVENT_TTL_DAYS = 90
MAX_LABEL_LENGTH = 50
MAX_REL_TYPE_LENGTH = 50

FORBIDDEN_MEMGRAPH_PROSE_KEYS: frozenset[str] = frozenset(
    {
        "raw_text",
        "full_text",
        "chunk_text",
        "evidence_text",
        "body_text",
    }
)


def get_memgraph_event_ttl_days(override: Optional[int] = None) -> int:
    """Return configured `:News` RAM TTL in days (`MEMGRAPH_EVENT_TTL_DAYS`, default 90)."""
    if override is not None:
        return max(1, int(override))
    raw = os.getenv("MEMGRAPH_EVENT_TTL_DAYS", str(DEFAULT_MEMGRAPH_EVENT_TTL_DAYS)).strip()
    try:
        parsed = int(raw)
    except ValueError:
        logger.warning(
            "Invalid MEMGRAPH_EVENT_TTL_DAYS=%r; falling back to default %d days.",
            raw,
            DEFAULT_MEMGRAPH_EVENT_TTL_DAYS,
        )
        return DEFAULT_MEMGRAPH_EVENT_TTL_DAYS
    return max(1, parsed)


def _sanitize_label(raw_label: Any, default: str = "Company") -> str:
    cleaned = "".join(c for c in str(raw_label or "").strip() if c.isalnum())[:MAX_LABEL_LENGTH]
    return cleaned or default


def _sanitize_rel_type(raw_rel: Any, default: str = "RELATED_TO") -> str:
    cleaned = "".join(
        c for c in str(raw_rel or "").strip().upper() if c.isalnum() or c == "_"
    )[:MAX_REL_TYPE_LENGTH]
    return cleaned or default


def strip_forbidden_prose(props: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Strip raw prose keys from node/edge property maps before durable or Memgraph writes."""
    if not props:
        return {}
    return {k: v for k, v in dict(props).items() if k not in FORBIDDEN_MEMGRAPH_PROSE_KEYS}


def _parse_json_obj(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            loaded = json.loads(raw)
            return dict(loaded) if isinstance(loaded, dict) else {}
        except (json.JSONDecodeError, TypeError, ValueError):
            return {}
    return {}


def _parse_json_list(raw: Any) -> List[str]:
    if isinstance(raw, (list, tuple, set)):
        return sorted({str(x).strip() for x in raw if str(x).strip()})
    if isinstance(raw, str) and raw.strip():
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, list):
                return sorted({str(x).strip() for x in loaded if str(x).strip()})
        except (json.JSONDecodeError, TypeError, ValueError):
            return []
    return []


@dataclass(frozen=True)
class KGNodeLedgerRecord:
    """Durable PostgreSQL `kg_nodes` ledger record."""

    node_id: str
    label: str = "Company"
    ticker: Optional[str] = None
    cik: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "label": _sanitize_label(self.label),
            "ticker": self.ticker,
            "cik": self.cik,
            "properties": strip_forbidden_prose(self.properties),
        }


@dataclass(frozen=True)
class KGEdgeLedgerRecord:
    """Durable PostgreSQL `kg_edges` ledger record."""

    source_id: str
    target_id: str
    rel_type: str
    input_group: str = "default"
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    known_at: Optional[str] = None
    known_at_imputed: bool = False
    last_affirmed_at: Optional[str] = None
    is_current: bool = True
    is_temporal_unanchored: bool = False
    status: str = "ACTIVE_CURRENT"
    is_active: bool = True
    share: Optional[float] = None
    buffer_days: Optional[float] = None
    param_provenance: Optional[str] = None
    evidence_chunk_ids: List[str] = field(default_factory=list)
    evidence_status: str = "chunked"
    source_type: Optional[str] = None
    source_hash: Optional[str] = None
    accession_number: Optional[str] = None
    form_type: Optional[str] = None
    section: Optional[str] = None
    fiscal_year: Optional[int] = None
    terminated_on: Optional[str] = None
    termination_accession: Optional[str] = None
    termination_known_at: Optional[str] = None
    termination_known_at_imputed: Optional[bool] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "rel_type": _sanitize_rel_type(self.rel_type),
            "input_group": (self.input_group or "default").strip() or "default",
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "known_at": self.known_at,
            "known_at_imputed": bool(self.known_at_imputed),
            "last_affirmed_at": self.last_affirmed_at,
            "is_current": bool(self.is_current),
            "is_temporal_unanchored": bool(self.is_temporal_unanchored),
            "status": self.status or "ACTIVE_CURRENT",
            "is_active": bool(self.is_active),
            "share": float(self.share) if self.share is not None else None,
            "buffer_days": float(self.buffer_days) if self.buffer_days is not None else None,
            "param_provenance": self.param_provenance,
            "evidence_chunk_ids": _parse_json_list(self.evidence_chunk_ids),
            "evidence_status": self.evidence_status or ("chunked" if self.evidence_chunk_ids else "legacy_unchunked"),
            "source_type": self.source_type,
            "source_hash": self.source_hash,
            "accession_number": self.accession_number,
            "form_type": self.form_type,
            "section": self.section,
            "fiscal_year": self.fiscal_year,
            "terminated_on": self.terminated_on,
            "termination_accession": self.termination_accession,
            "termination_known_at": self.termination_known_at,
            "termination_known_at_imputed": self.termination_known_at_imputed,
            "properties": strip_forbidden_prose(self.properties),
        }


def persist_kg_nodes(
    nodes: Sequence[KGNodeLedgerRecord | Mapping[str, Any]],
    *,
    conn: Optional[Any] = None,
) -> int:
    """Upsert canonical entity nodes into durable PostgreSQL `kg_nodes` prior to Memgraph projection."""
    deduped: Dict[str, Dict[str, Any]] = {}
    for item in nodes:
        raw = item.to_dict() if isinstance(item, KGNodeLedgerRecord) else dict(item)
        nid = str(raw.get("node_id") or raw.get("id") or "").strip()
        if not nid:
            continue
        label = _sanitize_label(raw.get("label") or raw.get("type") or "Company")
        props = strip_forbidden_prose(raw.get("properties"))
        ticker = raw.get("ticker") or props.get("ticker")
        cik = raw.get("cik") or props.get("cik")
        existing = deduped.get(nid)
        if existing is None:
            deduped[nid] = {
                "node_id": nid,
                "label": label,
                "ticker": str(ticker).strip() if ticker else None,
                "cik": str(cik).strip() if cik else None,
                "properties": props,
            }
        else:
            existing["properties"] = {**existing["properties"], **props}
            if ticker and not existing["ticker"]:
                existing["ticker"] = str(ticker).strip()
            if cik and not existing["cik"]:
                existing["cik"] = str(cik).strip()

    if not deduped:
        return 0

    sql = """
        INSERT INTO kg_nodes (node_id, label, ticker, cik, properties, updated_at)
        VALUES (%s, %s, %s, %s, %s::jsonb, now())
        ON CONFLICT (node_id) DO UPDATE SET
            label = COALESCE(NULLIF(EXCLUDED.label, ''), kg_nodes.label),
            ticker = COALESCE(EXCLUDED.ticker, kg_nodes.ticker),
            cik = COALESCE(EXCLUDED.cik, kg_nodes.cik),
            properties = kg_nodes.properties || EXCLUDED.properties,
            updated_at = now();
    """

    rows = [
        (
            r["node_id"],
            r["label"],
            r["ticker"],
            r["cik"],
            json.dumps(r["properties"], sort_keys=True, default=str),
        )
        for _, r in sorted(deduped.items())
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


def persist_kg_edges(
    edges: Sequence[KGEdgeLedgerRecord | Mapping[str, Any]],
    *,
    conn: Optional[Any] = None,
) -> int:
    """Upsert canonical structural edges into durable PostgreSQL `kg_edges` prior to Memgraph projection."""
    deduped: Dict[tuple[str, str, str, str], Dict[str, Any]] = {}
    for item in edges:
        raw = item.to_dict() if isinstance(item, KGEdgeLedgerRecord) else dict(item)
        src = str(raw.get("source_id") or raw.get("source") or "").strip()
        tgt = str(raw.get("target_id") or raw.get("target") or "").strip()
        if not src or not tgt:
            continue
        rel = _sanitize_rel_type(raw.get("rel_type") or raw.get("type") or "RELATED_TO")
        props = strip_forbidden_prose(raw.get("properties"))
        input_group = str(
            raw.get("input_group") or props.get("input_group") or props.get("nature") or "default"
        ).strip() or "default"
        cids = _parse_json_list(raw.get("evidence_chunk_ids") or props.get("evidence_chunk_ids") or [])
        ev_status = str(raw.get("evidence_status") or ("chunked" if cids else "legacy_unchunked"))
        valid_from = raw.get("valid_from")
        share_val = raw.get("share") if raw.get("share") is not None else props.get("share")
        buffer_val = raw.get("buffer_days") if raw.get("buffer_days") is not None else props.get("buffer_days")
        param_prov = raw.get("param_provenance") or props.get("param_provenance")

        key = (src, tgt, rel, input_group)
        rec: Dict[str, Any] = {
            "source_id": src,
            "target_id": tgt,
            "rel_type": rel,
            "input_group": input_group,
            "valid_from": str(valid_from)[:10] if valid_from else None,
            "valid_to": str(raw.get("valid_to"))[:10] if raw.get("valid_to") else None,
            "known_at": str(raw.get("known_at")) if raw.get("known_at") else None,
            "known_at_imputed": bool(raw.get("known_at_imputed", False)),
            "last_affirmed_at": str(raw.get("last_affirmed_at") or valid_from)[:10]
            if (raw.get("last_affirmed_at") or valid_from)
            else None,
            "is_current": bool(raw.get("is_current", True)),
            "is_temporal_unanchored": bool(raw.get("is_temporal_unanchored", valid_from is None)),
            "status": str(raw.get("status") or props.get("status") or "ACTIVE_CURRENT"),
            "is_active": bool(raw.get("is_active", props.get("is_active", True))),
            "share": float(share_val) if share_val is not None else None,
            "buffer_days": float(buffer_val) if buffer_val is not None else None,
            "param_provenance": str(param_prov) if param_prov else None,
            "evidence_chunk_ids": cids,
            "evidence_status": ev_status,
            "source_type": str(raw.get("source_type")) if raw.get("source_type") else None,
            "source_hash": str(raw.get("source_hash")) if raw.get("source_hash") else None,
            "accession_number": str(raw.get("accession_number")) if raw.get("accession_number") else None,
            "form_type": str(raw.get("form_type")) if raw.get("form_type") else None,
            "section": str(raw.get("section")) if raw.get("section") else None,
            "fiscal_year": int(raw["fiscal_year"]) if raw.get("fiscal_year") is not None else None,
            "terminated_on": str(raw.get("terminated_on"))[:10] if raw.get("terminated_on") else None,
            "termination_accession": str(raw.get("termination_accession"))
            if raw.get("termination_accession")
            else None,
            "termination_known_at": str(raw.get("termination_known_at"))
            if raw.get("termination_known_at")
            else None,
            "termination_known_at_imputed": raw.get("termination_known_at_imputed"),
            "properties": props,
        }
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = rec
        else:
            existing_cids: List[str] = list(existing["evidence_chunk_ids"])
            rec_cids: List[str] = list(rec["evidence_chunk_ids"])
            merged_cids = sorted({*existing_cids, *rec_cids})
            existing["evidence_chunk_ids"] = merged_cids
            existing["evidence_status"] = "chunked" if merged_cids else "legacy_unchunked"
            existing_props: Dict[str, Any] = dict(existing["properties"])
            rec_props: Dict[str, Any] = dict(rec["properties"])
            existing["properties"] = {**existing_props, **rec_props}

    if not deduped:
        return 0

    sql = """
        INSERT INTO kg_edges (
            source_id, target_id, rel_type, input_group,
            valid_from, valid_to, known_at, known_at_imputed, last_affirmed_at,
            is_current, is_temporal_unanchored, status, is_active,
            share, buffer_days, param_provenance,
            evidence_chunk_ids, evidence_status,
            source_type, source_hash, accession_number, form_type, section, fiscal_year,
            terminated_on, termination_accession, termination_known_at, termination_known_at_imputed,
            properties, updated_at
        )
        VALUES (
            %s, %s, %s, %s,
            %s::date, %s::date, %s::timestamptz, %s, %s::date,
            %s, %s, %s, %s,
            %s, %s, %s,
            %s::jsonb, %s,
            %s, %s, %s, %s, %s, %s,
            %s::date, %s, %s::timestamptz, %s,
            %s::jsonb, now()
        )
        ON CONFLICT (source_id, target_id, rel_type, input_group) DO UPDATE SET
            valid_from = COALESCE(kg_edges.valid_from, EXCLUDED.valid_from),
            valid_to = COALESCE(EXCLUDED.valid_to, kg_edges.valid_to),
            known_at = COALESCE(kg_edges.known_at, EXCLUDED.known_at),
            known_at_imputed = COALESCE(kg_edges.known_at_imputed, EXCLUDED.known_at_imputed),
            last_affirmed_at = COALESCE(EXCLUDED.last_affirmed_at, kg_edges.last_affirmed_at),
            is_current = EXCLUDED.is_current,
            is_temporal_unanchored = (COALESCE(kg_edges.valid_from, EXCLUDED.valid_from) IS NULL),
            status = EXCLUDED.status,
            is_active = EXCLUDED.is_active,
            share = COALESCE(EXCLUDED.share, kg_edges.share),
            buffer_days = COALESCE(EXCLUDED.buffer_days, kg_edges.buffer_days),
            param_provenance = COALESCE(EXCLUDED.param_provenance, kg_edges.param_provenance),
            evidence_chunk_ids = EXCLUDED.evidence_chunk_ids,
            evidence_status = EXCLUDED.evidence_status,
            source_type = COALESCE(EXCLUDED.source_type, kg_edges.source_type),
            source_hash = COALESCE(EXCLUDED.source_hash, kg_edges.source_hash),
            accession_number = COALESCE(EXCLUDED.accession_number, kg_edges.accession_number),
            form_type = COALESCE(EXCLUDED.form_type, kg_edges.form_type),
            section = COALESCE(EXCLUDED.section, kg_edges.section),
            fiscal_year = COALESCE(EXCLUDED.fiscal_year, kg_edges.fiscal_year),
            terminated_on = COALESCE(EXCLUDED.terminated_on, kg_edges.terminated_on),
            termination_accession = COALESCE(EXCLUDED.termination_accession, kg_edges.termination_accession),
            termination_known_at = COALESCE(EXCLUDED.termination_known_at, kg_edges.termination_known_at),
            termination_known_at_imputed = COALESCE(EXCLUDED.termination_known_at_imputed, kg_edges.termination_known_at_imputed),
            properties = kg_edges.properties || EXCLUDED.properties,
            updated_at = now();
    """

    rows = [
        (
            r["source_id"],
            r["target_id"],
            r["rel_type"],
            r["input_group"],
            r["valid_from"],
            r["valid_to"],
            r["known_at"],
            r["known_at_imputed"],
            r["last_affirmed_at"],
            r["is_current"],
            r["is_temporal_unanchored"],
            r["status"],
            r["is_active"],
            r["share"],
            r["buffer_days"],
            r["param_provenance"],
            json.dumps(r["evidence_chunk_ids"]),
            r["evidence_status"],
            r["source_type"],
            r["source_hash"],
            r["accession_number"],
            r["form_type"],
            r["section"],
            r["fiscal_year"],
            r["terminated_on"],
            r["termination_accession"],
            r["termination_known_at"],
            r["termination_known_at_imputed"],
            json.dumps(r["properties"], sort_keys=True, default=str),
        )
        for _, r in sorted(deduped.items())
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


def terminate_kg_edge_in_ledger(
    *,
    source_id: str,
    target_id: str,
    rel_type: str,
    valid_to: str,
    terminated_on: Optional[str],
    accession_number: str,
    known_at: Optional[str],
    known_at_imputed: bool,
    conn: Optional[Any] = None,
) -> int:
    """Close active structural edge(s) in durable PostgreSQL `kg_edges` prior to Memgraph update."""
    sql = """
        UPDATE kg_edges
        SET valid_to = %s::date,
            terminated_on = %s::date,
            termination_accession = %s,
            termination_known_at = %s::timestamptz,
            termination_known_at_imputed = %s,
            status = 'TERMINATED',
            is_current = FALSE,
            is_active = FALSE,
            updated_at = now()
        WHERE source_id = %s
          AND target_id = %s
          AND rel_type = %s
          AND valid_to IS NULL;
    """
    params = (
        valid_to,
        terminated_on,
        accession_number,
        known_at,
        known_at_imputed,
        source_id,
        target_id,
        _sanitize_rel_type(rel_type),
    )

    def _exec(active_conn: Any) -> int:
        with active_conn.cursor() as cur:
            cur.execute(sql, params)
            affected = int(getattr(cur, "rowcount", 0) or 0)
        active_conn.commit()
        return affected

    if conn is not None:
        return _exec(conn)

    with pg_connection() as active_conn:
        return _exec(active_conn)


def load_ledger_nodes(*, conn: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Load all canonical structural nodes deterministically from PostgreSQL `kg_nodes`."""
    sql = """
        SELECT node_id, label, ticker, cik, properties
        FROM kg_nodes
        ORDER BY node_id ASC;
    """

    def _run(active_conn: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with active_conn.cursor() as cur:
            cur.execute(sql)
            for row in cur.fetchall() or []:
                props = strip_forbidden_prose(_parse_json_obj(row[4]))
                if row[2] and not props.get("ticker"):
                    props["ticker"] = str(row[2])
                if row[3] and not props.get("cik"):
                    props["cik"] = str(row[3])
                out.append(
                    {
                        "node_id": str(row[0]),
                        "label": _sanitize_label(row[1]),
                        "ticker": str(row[2]) if row[2] else None,
                        "cik": str(row[3]) if row[3] else None,
                        "properties": props,
                    }
                )
        return out

    if conn is not None:
        return _run(conn)

    with pg_connection() as active_conn:
        return _run(active_conn)


def load_ledger_edges(*, conn: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Load all canonical structural edges deterministically from PostgreSQL `kg_edges`."""
    sql = """
        SELECT source_id, target_id, rel_type, input_group,
               valid_from, valid_to, known_at, known_at_imputed, last_affirmed_at,
               is_current, is_temporal_unanchored, status, is_active,
               share, buffer_days, param_provenance,
               evidence_chunk_ids, evidence_status,
               source_type, source_hash, accession_number, form_type, section, fiscal_year,
               terminated_on, termination_accession, termination_known_at, termination_known_at_imputed,
               properties
        FROM kg_edges
        ORDER BY source_id ASC, target_id ASC, rel_type ASC, input_group ASC;
    """

    def _run(active_conn: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with active_conn.cursor() as cur:
            cur.execute(sql)
            for row in cur.fetchall() or []:
                extra_props = strip_forbidden_prose(_parse_json_obj(row[28]))
                cids = _parse_json_list(row[16])
                out.append(
                    {
                        "source_id": str(row[0]),
                        "target_id": str(row[1]),
                        "rel_type": _sanitize_rel_type(row[2]),
                        "input_group": str(row[3] or "default"),
                        "valid_from": str(row[4])[:10] if row[4] is not None else None,
                        "valid_to": str(row[5])[:10] if row[5] is not None else None,
                        "known_at": str(row[6]) if row[6] is not None else None,
                        "known_at_imputed": bool(row[7]) if row[7] is not None else False,
                        "last_affirmed_at": str(row[8])[:10] if row[8] is not None else None,
                        "is_current": bool(row[9]) if row[9] is not None else True,
                        "is_temporal_unanchored": bool(row[10]) if row[10] is not None else False,
                        "status": str(row[11] or "ACTIVE_CURRENT"),
                        "is_active": bool(row[12]) if row[12] is not None else True,
                        "share": float(row[13]) if row[13] is not None else None,
                        "buffer_days": float(row[14]) if row[14] is not None else None,
                        "param_provenance": str(row[15]) if row[15] is not None else None,
                        "evidence_chunk_ids": cids,
                        "evidence_status": str(row[17] or ("chunked" if cids else "legacy_unchunked")),
                        "source_type": str(row[18]) if row[18] is not None else None,
                        "source_hash": str(row[19]) if row[19] is not None else None,
                        "accession_number": str(row[20]) if row[20] is not None else None,
                        "form_type": str(row[21]) if row[21] is not None else None,
                        "section": str(row[22]) if row[22] is not None else None,
                        "fiscal_year": int(row[23]) if row[23] is not None else None,
                        "terminated_on": str(row[24])[:10] if row[24] is not None else None,
                        "termination_accession": str(row[25]) if row[25] is not None else None,
                        "termination_known_at": str(row[26]) if row[26] is not None else None,
                        "termination_known_at_imputed": row[27],
                        "properties": extra_props,
                    }
                )
        return out

    if conn is not None:
        return _run(conn)

    with pg_connection() as active_conn:
        return _run(active_conn)


def evict_expired_news_nodes(
    *,
    driver: Optional[Any] = None,
    ttl_days: Optional[int] = None,
    now_utc: Optional[datetime] = None,
) -> int:
    """Evict transient `:News` nodes older than `MEMGRAPH_EVENT_TTL_DAYS` from Memgraph RAM.

    Leaves PostgreSQL `financial_news_queue`, `document_chunks`, and `kg_edges` untouched
    (ADR-00036 Section 4.4 / Invariant #4).
    """
    effective_ttl = get_memgraph_event_ttl_days(ttl_days)
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    cutoff_iso = (now - timedelta(days=effective_ttl)).isoformat()

    active_driver = driver if driver is not None else get_memgraph_driver()
    evicted = 0
    with active_driver.session() as session:
        records = session.run(
            """
            MATCH (n:News)
            WHERE n.published_at IS NOT NULL AND toString(n.published_at) < $cutoff_iso
            DETACH DELETE n
            RETURN count(n) AS evicted
            """,
            cutoff_iso=cutoff_iso,
        )
        for rec in records or []:
            if isinstance(rec, Mapping):
                evicted = int(rec.get("evicted", 0) or 0)
            else:
                evicted = int(rec["evicted"] or 0)
    logger.info(
        "Evicted %d expired :News node(s) from Memgraph RAM (ttl_days=%d, cutoff=%s).",
        evicted,
        effective_ttl,
        cutoff_iso,
    )
    return evicted


def hydrate_memgraph_from_ledger(
    *,
    pg_conn: Optional[Any] = None,
    driver: Optional[Any] = None,
    clear_first: bool = False,
    evict_news: bool = True,
    ttl_days: Optional[int] = None,
    now_utc: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Hydrate ephemeral Memgraph RAM deterministically from durable PostgreSQL `kg_nodes` & `kg_edges`."""
    nodes = load_ledger_nodes(conn=pg_conn)
    edges = load_ledger_edges(conn=pg_conn)

    active_driver = driver if driver is not None else get_memgraph_driver()
    hydrated_nodes = 0
    hydrated_edges = 0

    with active_driver.session() as session:
        if clear_first:
            session.run("MATCH (n) DETACH DELETE n")

        for node in nodes:
            nid = node["node_id"]
            label = _sanitize_label(node.get("label", "Company"))
            props = strip_forbidden_prose(node.get("properties"))
            if node.get("ticker"):
                props["ticker"] = node["ticker"]
            if node.get("cik"):
                props["cik"] = node["cik"]
            session.run(
                f"MERGE (n:{label} {{id: $id}}) SET n += $props",
                id=nid,
                props=props,
            )
            hydrated_nodes += 1

        for edge in edges:
            src = edge["source_id"]
            tgt = edge["target_id"]
            rel = _sanitize_rel_type(edge["rel_type"])
            props = strip_forbidden_prose(edge.get("properties"))
            props.update(
                {
                    "input_group": edge["input_group"],
                    "valid_from": edge["valid_from"],
                    "valid_to": edge["valid_to"],
                    "known_at": edge["known_at"],
                    "known_at_imputed": edge["known_at_imputed"],
                    "last_affirmed_at": edge["last_affirmed_at"],
                    "is_current": edge["is_current"],
                    "is_temporal_unanchored": edge["is_temporal_unanchored"],
                    "status": edge["status"],
                    "is_active": edge["is_active"],
                    "share": edge["share"],
                    "buffer_days": edge["buffer_days"],
                    "param_provenance": edge["param_provenance"],
                    "evidence_chunk_ids": edge["evidence_chunk_ids"],
                    "evidence_status": edge["evidence_status"],
                    "source_type": edge["source_type"],
                    "source_hash": edge["source_hash"],
                    "accession_number": edge["accession_number"],
                    "form_type": edge["form_type"],
                    "section": edge["section"],
                    "fiscal_year": edge["fiscal_year"],
                }
            )
            if edge.get("terminated_on") is not None:
                props["terminated_on"] = edge["terminated_on"]
            if edge.get("termination_accession") is not None:
                props["termination_accession"] = edge["termination_accession"]
            if edge.get("termination_known_at") is not None:
                props["termination_known_at"] = edge["termination_known_at"]
            if edge.get("termination_known_at_imputed") is not None:
                props["termination_known_at_imputed"] = edge["termination_known_at_imputed"]

            # Remove None keys so Cypher doesn't overwrite non-null properties unnecessarily
            clean_props = {k: v for k, v in props.items() if v is not None}
            session.run(
                f"""
                MERGE (s {{id: $source}})
                MERGE (t {{id: $target}})
                MERGE (s)-[r:{rel} {{input_group: $input_group}}]->(t)
                SET r += $props
                """,
                source=src,
                target=tgt,
                input_group=edge["input_group"],
                props=clean_props,
            )
            hydrated_edges += 1

    evicted_news = 0
    effective_ttl = get_memgraph_event_ttl_days(ttl_days)
    if evict_news:
        evicted_news = evict_expired_news_nodes(
            driver=active_driver,
            ttl_days=effective_ttl,
            now_utc=now_utc,
        )

    summary = {
        "nodes_hydrated": hydrated_nodes,
        "edges_hydrated": hydrated_edges,
        "evicted_news_nodes": evicted_news,
        "ttl_days": effective_ttl,
        "cleared": clear_first,
    }
    logger.info("Memgraph hydration complete: %s", summary)
    return summary


def check_memgraph_readiness(
    *,
    pg_conn: Optional[Any] = None,
    driver: Optional[Any] = None,
) -> Dict[str, Any]:
    """Verify Memgraph Bolt reachability and durable-ledger hydration completeness (ADR-00036 /ready)."""
    result: Dict[str, Any] = {
        "ready": False,
        "connected": False,
        "hydrated": False,
        "memgraph_edge_count": 0,
        "ledger_edge_count": None,
        "detail": None,
    }
    try:
        active_driver = driver if driver is not None else get_memgraph_driver()
        active_driver.verify_connectivity()
        result["connected"] = True
        with active_driver.session() as session:
            recs = list(
                session.run(
                    "MATCH ()-[r]->() WHERE type(r) <> 'MENTIONS' RETURN count(r) AS cnt"
                )
                or []
            )
            if recs:
                first = recs[0]
                result["memgraph_edge_count"] = int(
                    (first.get("cnt", 0) if isinstance(first, Mapping) else first["cnt"]) or 0
                )
    except (GraphStoreUnavailableError, Exception) as exc:
        result["detail"] = f"memgraph_unreachable: {exc}"
        return result

    # Compare against durable Postgres ledger if available
    ledger_count: Optional[int] = None
    try:
        if pg_conn is not None:
            with pg_conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM kg_edges;")
                row = cur.fetchone()
                ledger_count = int(row[0]) if row and row[0] is not None else 0
        else:
            with pg_connection() as active_pg:
                with active_pg.cursor() as cur:
                    cur.execute("SELECT count(*) FROM kg_edges;")
                    row = cur.fetchone()
                    ledger_count = int(row[0]) if row and row[0] is not None else 0
    except DocumentStoreUnavailableError:
        ledger_count = None
    except Exception as exc:
        logger.debug("Could not inspect kg_edges count for hydration check: %s", exc)
        ledger_count = None

    result["ledger_edge_count"] = ledger_count
    if ledger_count is not None and ledger_count > 0 and result["memgraph_edge_count"] < ledger_count:
        result["hydrated"] = False
        result["ready"] = False
        result["detail"] = (
            f"memgraph_hydration_incomplete: memgraph has {result['memgraph_edge_count']} structural edges, "
            f"durable Postgres ledger has {ledger_count}"
        )
        return result

    result["hydrated"] = True
    result["ready"] = True
    return result


def check_postgres_and_schema_metadata(*, pg_conn: Optional[Any] = None) -> Dict[str, Any]:
    """Verify PostgreSQL connectivity and `embedding_schema_metadata` alignment (ADR-00022 / ADR-00036)."""
    import embedding.embedder as _embedder

    expected_dim = _embedder.get_embedding_dim()
    expected_model = _embedder._get_model_name()
    pg_check: Dict[str, Any] = {"ready": False, "detail": None}
    meta_check: Dict[str, Any] = {
        "ready": False,
        "expected_dim": expected_dim,
        "expected_model": expected_model,
        "db_dim": None,
        "db_model": None,
        "detail": None,
    }

    def _inspect(active_conn: Any) -> None:
        with active_conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()
            pg_check["ready"] = True

            cur.execute("SELECT embedder_model, embedding_dim FROM embedding_schema_metadata WHERE id = 1;")
            row = cur.fetchone()
            if row is None:
                meta_check["ready"] = False
                meta_check["detail"] = "embedding_schema_metadata row missing (run tools/init_db.py)"
            else:
                db_model = str(row[0]) if row[0] is not None else ""
                db_dim = int(row[1]) if row[1] is not None else None
                meta_check["db_model"] = db_model
                meta_check["db_dim"] = db_dim
                if db_dim != expected_dim:
                    meta_check["ready"] = False
                    meta_check["detail"] = (
                        f"embedding_schema_metadata dimension mismatch: db={db_dim}, expected={expected_dim}"
                    )
                elif db_model and db_model != expected_model:
                    meta_check["ready"] = False
                    meta_check["detail"] = (
                        f"embedding_schema_metadata model mismatch: db='{db_model}', expected='{expected_model}'"
                    )
                else:
                    meta_check["ready"] = True

    try:
        if pg_conn is not None:
            _inspect(pg_conn)
        else:
            with pg_connection() as conn:
                _inspect(conn)
    except DocumentStoreUnavailableError as exc:
        pg_check["detail"] = str(exc)
        meta_check["detail"] = "postgres_unavailable"
    except Exception as exc:
        if not pg_check["ready"]:
            pg_check["detail"] = f"postgres_check_failed: {exc}"
            meta_check["detail"] = "postgres_unavailable"
        else:
            meta_check["detail"] = f"embedding_schema_metadata_query_failed: {exc}"

    return {"postgres": pg_check, "embedding_schema_metadata": meta_check}


__all__ = [
    "DEFAULT_MEMGRAPH_EVENT_TTL_DAYS",
    "FORBIDDEN_MEMGRAPH_PROSE_KEYS",
    "KGEdgeLedgerRecord",
    "KGNodeLedgerRecord",
    "check_memgraph_readiness",
    "check_postgres_and_schema_metadata",
    "evict_expired_news_nodes",
    "get_memgraph_event_ttl_days",
    "hydrate_memgraph_from_ledger",
    "load_ledger_edges",
    "load_ledger_nodes",
    "persist_kg_edges",
    "persist_kg_nodes",
    "strip_forbidden_prose",
    "terminate_kg_edge_in_ledger",
]
