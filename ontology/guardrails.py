"""Tier-4 Ontology Gate with Relation Repair & CIK Rollup.

Applies deterministic runtime post-processing to extraction outputs (Tasks A and B):
1. Roll up known Exhibit 21 operating subsidiary CIKs to parent entities.
2. Relation Repair: When the model extracts a terminated lifecycle event using an
   off-ontology predicate (e.g. [:TERMINATED]), normalizes the predicate to the
   canonical relationship (e.g. [:SUPPLIES_TO]) while preserving {status: 'TERMINATED', is_active: false}.
3. Prunes non-schema descriptive noise (e.g. [:COMPARES_TO]) to '(none)'.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from ontology.compiler import (
    is_hypothetical_or_hedged,
    is_triple_hedged,
    split_financial_clauses,
)
from ontology.schema import ONTOLOGY_SCHEMA

# Known Exhibit 21 Operating Subsidiary -> Parent CIK Rollup Map
# (e.g. DISH DBS Corporation 0001001811 rolls up to parent DISH Network Corp 0001001082)
SUBSIDIARY_TO_PARENT_CIK_MAP: Dict[str, str] = {
    "0001001811": "0001001082",  # DISH DBS -> DISH Network Corp
    "0001061219": "0001045810",  # Mellanox Technologies -> NVIDIA Corp
    "0001108524": "0001018724",  # Audible Inc -> Amazon.com Inc
    "0001326801": "0001326801",  # Meta Platforms Inc (self)
}

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
    ):
        self.subsidiary_map = subsidiary_map or SUBSIDIARY_TO_PARENT_CIK_MAP
        self.canonical_relations = (
            frozenset(canonical_relations)
            if canonical_relations
            else CANONICAL_RELATION_PREDICATES
        )
        self.default_repaired_relation = default_repaired_relation

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
        nodes = []
        for node in payload.get("nodes", []) or []:
            node_copy = dict(node)
            props = dict(node_copy.get("properties", {}) or {})
            cik = str(props.get("cik", "")).strip()
            if cik in self.subsidiary_map:
                props["cik"] = self.subsidiary_map[cik]
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

