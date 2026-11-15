"""Tier-4 Ontology Gate with Relation Repair & CIK Rollup.

Applies deterministic runtime post-processing to extraction outputs (Tasks A and B):
1. Roll up known Exhibit 21 operating subsidiary CIKs to parent entities.
2. Relation Repair: When the model extracts a terminated lifecycle event using an
   off-ontology predicate (e.g. [:TERMINATED]), normalizes the predicate to the
   canonical relationship (e.g. [:SUPPLIES_TO]) while preserving {status: 'TERMINATED', is_active: false}.
3. Prunes non-schema descriptive noise (e.g. [:COMPARES_TO]) to '(none)'.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from ontology.alignment import normalize_entity_name
from ontology.compiler import (
    is_hypothetical_or_hedged,
    is_triple_hedged,
    split_financial_clauses,
)
from ontology.schema import ONTOLOGY_SCHEMA

logger = logging.getLogger(__name__)

# Offline deterministic seed for Exhibit 21 Operating Subsidiary -> Parent CIK Rollup
# (merged beneath PostgreSQL `sec_subsidiaries` + `sec_companies` dynamic records at runtime).
SEED_SUBSIDIARY_TO_PARENT_CIK_MAP: Dict[str, str] = {
    "0001001811": "0001001082",  # DISH DBS -> DISH Network Corp
    "0001061219": "0001045810",  # Mellanox Technologies -> NVIDIA Corp
    "0001108524": "0001018724",  # Audible Inc -> Amazon.com Inc
    "0001326801": "0001326801",  # Meta Platforms Inc (self)
}
SUBSIDIARY_TO_PARENT_CIK_MAP: Dict[str, str] = dict(SEED_SUBSIDIARY_TO_PARENT_CIK_MAP)


@dataclass(frozen=True)
class SubsidiaryRegistry:
    """Snapshot of subsidiary CIK and normalized-name rollup mappings."""

    cik_map: Dict[str, str] = field(default_factory=lambda: dict(SEED_SUBSIDIARY_TO_PARENT_CIK_MAP))
    name_to_parent: Dict[str, Tuple[str, str, Optional[str]]] = field(default_factory=dict)
    loaded_at: float = 0.0
    from_database: bool = False


_REGISTRY_LOCK = threading.Lock()
_CACHED_REGISTRY: Optional[SubsidiaryRegistry] = None


def invalidate_subsidiary_registry_cache() -> None:
    """Invalidate the cached dynamic subsidiary registry so next lookup reloads from PostgreSQL."""
    global _CACHED_REGISTRY
    with _REGISTRY_LOCK:
        _CACHED_REGISTRY = None


def _query_subsidiary_registry_from_conn(conn: Any) -> Tuple[Dict[str, str], Dict[str, Tuple[str, str, Optional[str]]], bool]:
    """Read `sec_subsidiaries` + `sec_companies` mappings from a DB connection/cursor."""
    cik_map: Dict[str, str] = dict(SEED_SUBSIDIARY_TO_PARENT_CIK_MAP)
    name_to_parent: Dict[str, Tuple[str, str, Optional[str]]] = {}
    if conn is None or not hasattr(conn, "cursor"):
        return cik_map, name_to_parent, False

    cur = conn.cursor()
    saw_rows = False
    try:
        # 1. Registered subsidiary companies whose normalized_name matches a `sec_companies` filer CIK
        cur.execute(
            """
            SELECT sub_co.cik, s.parent_cik
            FROM sec_subsidiaries s
            JOIN sec_companies sub_co ON s.normalized_name = sub_co.normalized_name
            WHERE s.parent_cik IS NOT NULL
              AND sub_co.cik IS NOT NULL
              AND sub_co.cik <> s.parent_cik
            """
        )
        for row in cur.fetchall() or []:
            if isinstance(row, (tuple, list)) and len(row) >= 2 and row[0] and row[1]:
                cik_map[str(row[0]).strip()] = str(row[1]).strip()
                saw_rows = True

        # 2. Exhibit 21 subsidiary normalized names -> parent metadata
        cur.execute(
            """
            SELECT s.normalized_name, c.company_name, c.cik, c.ticker
            FROM sec_subsidiaries s
            JOIN sec_companies c ON s.parent_cik = c.cik
            WHERE s.normalized_name IS NOT NULL AND s.normalized_name <> ''
            """
        )
        for row in cur.fetchall() or []:
            if isinstance(row, (tuple, list)) and len(row) >= 3 and row[0] and row[1] and row[2]:
                norm_sub = str(row[0]).strip().lower()
                parent_name = str(row[1]).strip()
                parent_cik = str(row[2]).strip()
                parent_ticker = str(row[3]).strip().upper() if len(row) >= 4 and row[3] else None
                name_to_parent[norm_sub] = (parent_name, parent_cik, parent_ticker)
                saw_rows = True
    finally:
        if hasattr(cur, "close"):
            cur.close()
    return cik_map, name_to_parent, saw_rows


def load_dynamic_subsidiary_registry(
    pg_conn: Optional[Any] = None,
    *,
    force_refresh: bool = False,
    ttl_seconds: float = 300.0,
) -> SubsidiaryRegistry:
    """Load Exhibit 21 subsidiary CIK & parent mappings with TTL caching and offline seed fallback."""
    global _CACHED_REGISTRY
    now = time.monotonic()

    if pg_conn is None and not force_refresh:
        with _REGISTRY_LOCK:
            if _CACHED_REGISTRY is not None and (now - _CACHED_REGISTRY.loaded_at) < ttl_seconds:
                return _CACHED_REGISTRY

    owned_conn = None
    conn = pg_conn
    cik_map: Dict[str, str] = dict(SEED_SUBSIDIARY_TO_PARENT_CIK_MAP)
    name_to_parent: Dict[str, Tuple[str, str, Optional[str]]] = {}
    from_db = False

    try:
        if conn is None:
            from knowledge_graph.db import pg_connection

            owned_conn = pg_connection()
            conn = owned_conn
        cik_map, name_to_parent, from_db = _query_subsidiary_registry_from_conn(conn)
    except Exception as exc:
        logger.debug("Falling back to offline seed subsidiary CIK map: %s", exc)
    finally:
        if owned_conn is not None and hasattr(owned_conn, "close"):
            try:
                owned_conn.close()
            except Exception:
                pass

    registry = SubsidiaryRegistry(
        cik_map=cik_map,
        name_to_parent=name_to_parent,
        loaded_at=now,
        from_database=from_db,
    )
    if pg_conn is None:
        with _REGISTRY_LOCK:
            _CACHED_REGISTRY = registry
    return registry


def load_subsidiary_cik_map(
    pg_conn: Optional[Any] = None,
    *,
    force_refresh: bool = False,
    ttl_seconds: float = 300.0,
) -> Dict[str, str]:
    """Convenience loader returning `{subsidiary_cik: parent_cik}` with offline seed fallback."""
    return load_dynamic_subsidiary_registry(
        pg_conn=pg_conn,
        force_refresh=force_refresh,
        ttl_seconds=ttl_seconds,
    ).cik_map

# Canonical Axis-2 Relational Ontology Predicates (derived from ontology.schema)
CANONICAL_RELATION_PREDICATES: frozenset[str] = frozenset(ONTOLOGY_SCHEMA.keys())

# Lifecycle Termination Predicates that indicate contract end
LIFECYCLE_TERMINATION_PREDICATES: frozenset[str] = frozenset(
    {
        "TERMINATED",
        "ENDED",
        "CANCELLED",
        "CANCELED",
        "DISSOLVED",
        "EXPIRED",
        "SEVERED",
        "DISCONTINUED",
        "TERMINATION",
    }
)

# Completely off-domain descriptive noise predicates that must be pruned to '(none)'
NON_SCHEMA_NOISE_PREDICATES: frozenset[str] = frozenset(
    {
        "COMPARES_TO",
        "RISES_WITH_BENCHMARK",
        "TRADED_IN",
        "REACTS_TO",
        "TRADES_WITH",
        "PEER_REVIEWED_BY",
        "BENCHMARKED_AGAINST",
        "CORRELATES_WITH",
    }
)


_TERMINATED_LICENSE_RE = re.compile(
    r"\b(licens\w*|royalt\w*|patent\w*|intellectual\s+property)\b", re.IGNORECASE
)
_TERMINATED_PARTNER_RE = re.compile(
    r"\b(joint\s+venture|partnership|partner\w*|alliance|collaborat\w*)\b", re.IGNORECASE
)
_TERMINATED_CUSTOMER_RE = re.compile(
    r"\b(customer\w*|purchas\w*|procur\w*|offtake)\b", re.IGNORECASE
)
_TERMINATED_SUPPLIER_RE = re.compile(
    r"\b(suppli\w*|vendor\w*|foundry|manufactur\w*)\b", re.IGNORECASE
)
_CYPHER_NAME_FIELD_RE = re.compile(r"""(?:name|id)\s*:\s*['"]([^'"]+)['"]""")


def infer_terminated_relation(context_text: str, default_rel: str = "SUPPLIES_TO") -> str:
    """Infer the underlying domain predicate for a terminated lifecycle event."""
    ctx = context_text or ""
    if _TERMINATED_LICENSE_RE.search(ctx):
        return "LICENSES_FROM"
    if _TERMINATED_PARTNER_RE.search(ctx):
        return "PARTNERED_WITH"
    if _TERMINATED_CUSTOMER_RE.search(ctx) and not _TERMINATED_SUPPLIER_RE.search(ctx):
        return "CUSTOMER_OF"
    return default_rel


def _matching_evidence_context(
    prompt: Optional[str],
    entities: List[str],
    extra_text: str = "",
) -> str:
    """Return prompt clauses matching triple entities combined with inline metadata."""
    if not prompt:
        return extra_text
    clauses = split_financial_clauses(prompt)
    clean_ents = [e.lower() for e in entities if e and len(e) >= 2]
    matched = [c for c in clauses if any(ent in c.lower() for ent in clean_ents)]
    base = " ".join(matched) if matched else prompt
    return f"{base} {extra_text}".strip()


def _scan_balanced_delimiter(text: str, start: int, open_ch: str, close_ch: str) -> int:
    """Scan past balanced '()' or '[]' respecting string literals and nested braces."""
    if start >= len(text) or text[start] != open_ch:
        return -1
    depth = 0
    idx = start
    n = len(text)
    while idx < n:
        ch = text[idx]
        if ch in ('"', "'"):
            quote = ch
            idx += 1
            while idx < n:
                if text[idx] == "\\" and idx + 1 < n:
                    idx += 2
                    continue
                if text[idx] == quote:
                    idx += 1
                    break
                idx += 1
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return idx + 1
        idx += 1
    return -1


def _extract_cypher_triples(text: str) -> List[Tuple[str, str, str, str]]:
    """Extract (src_node_str, rel_type, rel_props_str, dst_node_str) triples safely."""
    triples: List[Tuple[str, str, str, str]] = []
    n = len(text)
    idx = 0
    while idx < n:
        pos = text.find("(", idx)
        if pos == -1:
            break
        src_end = _scan_balanced_delimiter(text, pos, "(", ")")
        if src_end == -1:
            idx = pos + 1
            continue
        cursor = src_end
        while cursor < n and text[cursor] in " \t\r\n":
            cursor += 1
        if cursor >= n or text[cursor] != "-":
            idx = pos + 1
            continue
        cursor += 1
        while cursor < n and text[cursor] in " \t\r\n":
            cursor += 1
        if cursor >= n or text[cursor] != "[":
            idx = pos + 1
            continue
        rel_end = _scan_balanced_delimiter(text, cursor, "[", "]")
        if rel_end == -1 or text[rel_end : rel_end + 2] != "->":
            idx = pos + 1
            continue
        dst_start = rel_end + 2
        while dst_start < n and text[dst_start] in " \t\r\n":
            dst_start += 1
        dst_end = _scan_balanced_delimiter(text, dst_start, "(", ")")
        if dst_end == -1:
            idx = pos + 1
            continue

        rel_inner = text[cursor + 1 : rel_end - 1].strip()
        rel_match = re.match(r":([A-Za-z0-9_]+)(.*)", rel_inner, flags=re.DOTALL)
        if not rel_match:
            idx = pos + 1
            continue

        rel_type = rel_match.group(1).upper()
        props_str = rel_match.group(2).strip()
        if props_str:
            props_str = f" {props_str}"
        triples.append((text[pos:src_end], rel_type, props_str, text[dst_start:dst_end]))
        idx = dst_end
    return triples


def _extract_entity_label_hint(node_expr: str) -> str:
    match = _CYPHER_NAME_FIELD_RE.search(node_expr)
    return match.group(1).strip() if match else ""


class Tier4OntologyGate:
    """Enforces ontology boundary constraints and repairs terminated lifecycle relations."""

    def __init__(
        self,
        subsidiary_map: Optional[Dict[str, str]] = None,
        canonical_relations: Optional[Set[str]] = None,
        default_repaired_relation: str = "SUPPLIES_TO",
        *,
        pg_conn: Optional[Any] = None,
        subsidiary_name_map: Optional[Dict[str, Tuple[str, str, Optional[str]]]] = None,
    ):
        self._explicit_subsidiary_map = dict(subsidiary_map) if subsidiary_map is not None else None
        self._explicit_name_map = dict(subsidiary_name_map) if subsidiary_name_map is not None else None
        self._pg_conn = pg_conn
        self.canonical_relations = (
            frozenset(canonical_relations)
            if canonical_relations
            else CANONICAL_RELATION_PREDICATES
        )
        self.default_repaired_relation = default_repaired_relation

    @property
    def subsidiary_map(self) -> Dict[str, str]:
        if self._explicit_subsidiary_map is not None:
            return self._explicit_subsidiary_map
        return load_dynamic_subsidiary_registry(pg_conn=self._pg_conn).cik_map

    @property
    def subsidiary_name_map(self) -> Dict[str, Tuple[str, str, Optional[str]]]:
        if self._explicit_name_map is not None:
            return self._explicit_name_map
        if self._explicit_subsidiary_map is not None:
            return {}
        return load_dynamic_subsidiary_registry(pg_conn=self._pg_conn).name_to_parent

    def _rollup_text_ciks(self, text: str) -> str:
        for sub_cik, parent_cik in self.subsidiary_map.items():
            if sub_cik in text and parent_cik not in text:
                text = text.replace(sub_cik, parent_cik)
        return text

    def _repair_or_filter_edge(
        self,
        rel_type: str,
        entities: List[str],
        evidence_extra: str,
        prompt: Optional[str],
        triple_text: str,
    ) -> Optional[Tuple[str, bool]]:
        """Evaluate a single edge predicate and return (canonical_rel, is_terminated_repair) or None."""
        pred = rel_type.strip().upper()
        if pred in NON_SCHEMA_NOISE_PREDICATES:
            return None

        if is_triple_hedged(prompt, entities, triple_text):
            return None

        is_terminated = pred in LIFECYCLE_TERMINATION_PREDICATES or (
            pred not in self.canonical_relations and "TERMINATED" in evidence_extra.upper()
        )
        if is_terminated:
            ctx = _matching_evidence_context(prompt, entities, evidence_extra)
            repaired = infer_terminated_relation(ctx, self.default_repaired_relation)
            return (repaired, True) if repaired in self.canonical_relations else None

        if pred in self.canonical_relations:
            return (pred, False)

        return None

    def gate_json_payload(
        self,
        payload: Dict[str, Any],
        source_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Apply CIK rollup, clause-scoped hedge filtering, noise pruning, and termination repair to JSON payloads."""
        sub_cik_map = self.subsidiary_map
        sub_name_map = self.subsidiary_name_map
        nodes = []
        for node in payload.get("nodes", []) or []:
            node_copy = dict(node)
            props = dict(node_copy.get("properties", {}) or {})
            cik = str(props.get("cik", "")).strip()
            if cik in sub_cik_map:
                props["cik"] = sub_cik_map[cik]
            elif not cik and sub_name_map:
                norm_name = normalize_entity_name(str(props.get("name") or node_copy.get("id") or ""))
                parent_info = sub_name_map.get(norm_name)
                if parent_info:
                    props["cik"] = parent_info[1]
                    props.setdefault("parent_company", parent_info[0])
            node_copy["properties"] = props
            nodes.append(node_copy)

        clean_edges = []
        for edge in payload.get("edges", []) or []:
            src = str(edge.get("source", "")).strip()
            tgt = str(edge.get("target", "")).strip()
            rel_type = str(edge.get("type", "")).strip().upper()
            props = dict(edge.get("properties", {}) or {})
            extra = f"{json.dumps(props)} {rel_type}"
            # Allow canonical & aliased predicates through repair step; defer alias validation to validation.py
            if rel_type not in NON_SCHEMA_NOISE_PREDICATES and rel_type not in LIFECYCLE_TERMINATION_PREDICATES and "TERMINATED" not in extra.upper():
                if is_triple_hedged(source_text, [src, tgt], extra):
                    continue
                edge_copy = dict(edge)
                edge_copy["properties"] = props
                clean_edges.append(edge_copy)
                continue

            outcome = self._repair_or_filter_edge(rel_type, [src, tgt], extra, source_text, extra)
            if outcome is None:
                continue
            repaired_rel, terminated = outcome
            if terminated:
                props.setdefault("status", "TERMINATED")
                props.setdefault("is_active", False)
            edge_copy = dict(edge)
            edge_copy["type"] = repaired_rel
            edge_copy["properties"] = props
            clean_edges.append(edge_copy)

        return {"nodes": nodes, "edges": clean_edges}

    def apply_guardrails(self, prediction: str, task_type: str, prompt: Optional[str] = None) -> str:
        """Apply deterministic Tier-4 post-processing guardrails per edge on Cypher or JSON predictions."""
        cleaned = prediction.strip()
        if not cleaned or cleaned.lower() == "(none)":
            return "(none)"

        if cleaned.startswith("{") and task_type in ("EXTRACT_SEC_GRAPH", "EXTRACT_NEWS_EVENT"):
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and ("nodes" in parsed or "edges" in parsed):
                    gated = self.gate_json_payload(parsed, source_text=prompt)
                    if not gated["edges"] and not gated["nodes"]:
                        return "(none)"
                    if parsed.get("edges") and not gated["edges"]:
                        return "(none)"
                    return json.dumps(gated)
            except json.JSONDecodeError:
                pass

        # 1. Subsidiary CIK Rollup
        cleaned = self._rollup_text_ciks(cleaned)

        # 2. Per-triple relation validation, evidence-scoped hedging, and repair
        if task_type in ("EXTRACT_SEC_GRAPH", "EXTRACT_NEWS_EVENT"):
            return self.repair_extraction_relations(cleaned, prompt=prompt)

        if prompt and is_hypothetical_or_hedged(prompt):
            return "(none)"
        if is_hypothetical_or_hedged(cleaned):
            return "(none)"
        return cleaned

    def repair_extraction_relations(self, text: str, prompt: Optional[str] = None) -> str:
        """Parse Cypher triples cleanly, prune noise edges per triple, and infer terminated context."""
        triples = _extract_cypher_triples(text)
        if not triples:
            if not re.search(r"\b(MATCH|MERGE|CREATE|RETURN)\b", text, re.IGNORECASE):
                return "(none)"
            if prompt and is_hypothetical_or_hedged(prompt):
                return "(none)"
            return text

        surviving: List[str] = []
        for src_expr, pred, props_str, dst_expr in triples:
            src_name = _extract_entity_label_hint(src_expr)
            dst_name = _extract_entity_label_hint(dst_expr)
            entities = [e for e in (src_name, dst_name) if e]
            triple_str = f"{src_expr}-[:{pred}{props_str}]->{dst_expr}"

            outcome = self._repair_or_filter_edge(pred, entities, props_str, prompt, triple_str)
            if outcome is None:
                continue
            final_pred, is_terminated = outcome
            final_props = self._ensure_terminated_status(props_str) if is_terminated else props_str
            surviving.append(f"{src_expr}-[:{final_pred}{final_props}]->{dst_expr}")

        if not surviving:
            return "(none)"
        return "\n".join(surviving)

    def _ensure_terminated_status(self, props_str: str) -> str:
        """Ensure properties contain status: 'TERMINATED' and is_active: false."""
        if not props_str or not props_str.strip():
            return " {status: 'TERMINATED', is_active: false}"

        inner = props_str.strip()
        if inner.startswith("{") and inner.endswith("}"):
            body = inner[1:-1].strip()
        else:
            body = inner

        # Check if status is already set
        if "status" not in body.lower():
            body = f"{body}, status: 'TERMINATED'" if body else "status: 'TERMINATED'"

        if "is_active" not in body.lower():
            body = f"{body}, is_active: false"

        return f" {{{body}}}"


# Singleton instance for quick module-level access
DEFAULT_ONTOLOGY_GATE = Tier4OntologyGate()


def apply_runtime_guardrails(prompt: str, prediction: str, task_type: str) -> str:
    """Convenience functional wrapper matching the Phase 4 runtime guardrail interface."""
    return DEFAULT_ONTOLOGY_GATE.apply_guardrails(prediction, task_type, prompt=prompt)

