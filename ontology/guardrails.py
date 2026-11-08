"""Tier-4 Ontology Gate with Relation Repair & CIK Rollup.

Applies deterministic runtime post-processing to extraction outputs (Tasks A and B):
1. Roll up known Exhibit 21 operating subsidiary CIKs to parent entities.
2. Relation Repair: When the model extracts a terminated lifecycle event using an
   off-ontology predicate (e.g. [:TERMINATED]), normalizes the predicate to the
   canonical relationship (e.g. [:SUPPLIES_TO]) while preserving {status: 'TERMINATED', is_active: false}.
3. Prunes non-schema descriptive noise (e.g. [:COMPARES_TO]) to '(none)'.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

# Known Exhibit 21 Operating Subsidiary -> Parent CIK Rollup Map
# (e.g. DISH DBS Corporation 0001001811 rolls up to parent DISH Network Corp 0001001082)
SUBSIDIARY_TO_PARENT_CIK_MAP: Dict[str, str] = {
    "0001001811": "0001001082",  # DISH DBS -> DISH Network Corp
    "0001061219": "0001045810",  # Mellanox Technologies -> NVIDIA Corp
    "0001108524": "0001018724",  # Audible Inc -> Amazon.com Inc
    "0001326801": "0001326801",  # Meta Platforms Inc (self)
}

# Canonical Axis-2 Relational Ontology Predicates
CANONICAL_RELATION_PREDICATES: frozenset[str] = frozenset(
    {
        "SUPPLIES_TO",
        "CUSTOMER_OF",
        "PARTNERED_WITH",
        "SUBSIDIARY_OF",
        "COMPETES_WITH",
        "LICENSES_FROM",
        "LICENSES_TO",
        "SOLE_SOURCE_DEPENDENT_ON",
        "ACQUIRED_BY",
        "DEFAULTED_ON",
        "SPUN_OFF_FROM",
        "FILED",
        "EXPOSED_TO_RISK",
        "OWNS",
        "HAS_SUBSIDIARY",
        "PEER_OF",
    }
)

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

    def apply_guardrails(self, prediction: str, task_type: str) -> str:
        """Apply deterministic Tier-4 post-processing guardrails to an extraction prediction.

        Args:
            prediction: Raw output from the model.
            task_type: Task identifier ('EXTRACT_SEC_GRAPH', 'EXTRACT_NEWS_EVENT', etc.)

        Returns:
            Sanitized, repaired prediction or '(none)'.
        """
        cleaned = prediction.strip()
        if not cleaned:
            return "(none)"
        if cleaned.lower() == "(none)":
            return "(none)"

        # 1. Subsidiary CIK Rollup
        for sub_cik, parent_cik in self.subsidiary_map.items():
            if sub_cik in cleaned and parent_cik not in cleaned:
                cleaned = cleaned.replace(sub_cik, parent_cik)

        # 2. Relation validation and repair for graph extraction tasks
        if task_type in ("EXTRACT_SEC_GRAPH", "EXTRACT_NEWS_EVENT"):
            repaired = self.repair_extraction_relations(cleaned)
            return repaired

        return cleaned

    def repair_extraction_relations(self, text: str) -> str:
        """Parse triples in text, repair lifecycle termination predicates, and prune noise."""
        # Find all relationship patterns: -[:PRED {props}]-> or -[:PRED]->
        rel_pattern = re.compile(
            r"-\[:([A-Z0-9_]+)(\s*\{[^}]*\})?\]->",
            re.IGNORECASE,
        )

        matches = list(rel_pattern.finditer(text))
        if not matches:
            # If no graph relationship pattern exists and not a valid Cypher clause
            if not re.search(r"\b(MATCH|MERGE|CREATE|RETURN)\b", text, re.IGNORECASE):
                return "(none)"
            return text

        result = text
        has_at_least_one_valid_or_repaired_relation = False

        for m in matches:
            pred = m.group(1).upper()
            props_str = m.group(2) or ""

            if pred in self.canonical_relations:
                has_at_least_one_valid_or_repaired_relation = True
                continue

            if pred in LIFECYCLE_TERMINATION_PREDICATES:
                # REPAIR LOGIC: Map to canonical relation and ensure status: 'TERMINATED'
                repaired_pred = self.default_repaired_relation
                new_props = self._ensure_terminated_status(props_str)
                old_span = m.group(0)
                new_span = f"-[:{repaired_pred}{new_props}]->"
                result = result.replace(old_span, new_span, 1)
                has_at_least_one_valid_or_repaired_relation = True
                continue

            if pred in NON_SCHEMA_NOISE_PREDICATES:
                # Explicit noise predicate -> mark for pruning
                continue

            # Unrecognized predicate: if props contain status: 'TERMINATED', repair as well
            if "TERMINATED" in props_str.upper():
                repaired_pred = self.default_repaired_relation
                new_props = self._ensure_terminated_status(props_str)
                old_span = m.group(0)
                new_span = f"-[:{repaired_pred}{new_props}]->"
                result = result.replace(old_span, new_span, 1)
                has_at_least_one_valid_or_repaired_relation = True
                continue

        if not has_at_least_one_valid_or_repaired_relation:
            return "(none)"

        return result

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
    return DEFAULT_ONTOLOGY_GATE.apply_guardrails(prediction, task_type)

