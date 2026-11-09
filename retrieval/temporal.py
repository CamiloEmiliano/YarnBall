"""Temporal State Coalescence and Query Invariant Transformer.

Enforces temporal graph consistency on openCypher queries:
1. Active Shock Propagation: Auto-injects
   `coalesce(r.status, 'ACTIVE_CURRENT') = 'ACTIVE_CURRENT'`
   and as_of_date horizon cutoffs so terminated or future contracts never leak into active contagion paths.
2. Historical Audit: Allows explicit inspection of historical / terminated edges.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple


class TemporalQueryFilter:
    """Transforms Cypher queries to enforce temporal state invariants."""

    def __init__(self, default_as_of_date_param: str = "$as_of_date"):
        self.default_as_of_date_param = default_as_of_date_param

    def inject_active_coalescence(
        self,
        cypher: str,
        as_of_date: Optional[str] = None,
        default_var: str = "r",
    ) -> str:
        """Inject active status coalescence and as_of_date bounds into a Cypher query.

        Args:
            cypher: The original openCypher query string.
            as_of_date: Optional explicit date string ('YYYY-MM-DD'). If None, uses $as_of_date param.
            default_var: Relationship variable to bind if anonymous.

        Returns:
            Rewritten Cypher query with temporal invariants enforced.
        """
        clean = cypher.strip()
        if not clean or clean.lower() == "(none)":
            return clean

        # 1. Identify relationship variable or bind an anonymous one
        bound_cypher, rel_vars = self._ensure_relationship_vars(clean, default_var)
        if not rel_vars:
            # No relationship patterns found; return unmodified
            return clean

        # Use the first relationship variable found for primary temporal filtering
        primary_var = rel_vars[0]

        # 2. Formulate temporal predicates
        date_expr = f"'{as_of_date}'" if as_of_date else self.default_as_of_date_param
        temporal_pred = (
            f"coalesce({primary_var}.status, 'ACTIVE_CURRENT') = 'ACTIVE_CURRENT' "
            f"AND ({primary_var}.valid_from IS NULL OR {primary_var}.valid_from <= {date_expr}) "
            f"AND ({primary_var}.valid_to IS NULL OR {primary_var}.valid_to >= {date_expr})"
        )

        # 3. Check if temporal predicate is already present to prevent duplicate injection
        if f"coalesce({primary_var}.status" in bound_cypher or f"{primary_var}.status = 'ACTIVE_CURRENT'" in bound_cypher:
            return bound_cypher

        # 4. Inject into WHERE clause or create one
        return self._inject_where_predicate(bound_cypher, temporal_pred)

    def inject_audit_filter(
        self,
        cypher: str,
        target_status: str = "TERMINATED",
        as_of_date: Optional[str] = None,
        default_var: str = "r",
    ) -> str:
        """Inject audit filter targeting specific lifecycle states (e.g. TERMINATED)."""
        clean = cypher.strip()
        if not clean or clean.lower() == "(none)":
            return clean

        bound_cypher, rel_vars = self._ensure_relationship_vars(clean, default_var)
        if not rel_vars:
            return clean

        primary_var = rel_vars[0]
        audit_pred = f"{primary_var}.status = '{target_status}'"
        if as_of_date:
            audit_pred += f" AND ({primary_var}.valid_from IS NULL OR {primary_var}.valid_from <= '{as_of_date}')"

        return self._inject_where_predicate(bound_cypher, audit_pred)

    def _ensure_relationship_vars(
        self, cypher: str, default_var: str
    ) -> Tuple[str, List[str]]:
        """Find existing relationship variables or bind anonymous relationships: -[:TYPE]-> to -[r:TYPE]->."""
        found_vars: List[str] = []

        # Find named relationships: [var:TYPE]
        named_pattern = re.compile(r"\[([a-zA-Z_][a-zA-Z0-9_]*):[a-zA-Z0-9_|*.]+")
        for m in named_pattern.finditer(cypher):
            found_vars.append(m.group(1))

        if found_vars:
            return cypher, found_vars

        # If anonymous relationships exist: -[:TYPE]->
        anon_pattern = re.compile(r"-\[:([a-zA-Z0-9_|*.]+)(\s*\{[^}]*\})?\]->")
        match = anon_pattern.search(cypher)
        if match:
            rel_type = match.group(1)
            props = match.group(2) or ""
            old_span = match.group(0)
            new_span = f"-[{default_var}:{rel_type}{props}]->"
            rewritten = cypher.replace(old_span, new_span, 1)
            return rewritten, [default_var]

        return cypher, []

    def _inject_where_predicate(self, cypher: str, predicate: str) -> str:
        """Inject predicate into an existing WHERE clause or insert WHERE before RETURN/WITH."""
        # Check if query already has a WHERE clause
        where_match = re.search(r"\bWHERE\b", cypher, re.IGNORECASE)
        if where_match:
            # Append to existing WHERE clause: WHERE ... AND <predicate>
            where_pos = where_match.end()
            return f"{cypher[:where_pos]} ({predicate}) AND{cypher[where_pos:]}"

        # No WHERE clause exists. Find RETURN or WITH or end of query
        clause_match = re.search(r"\b(RETURN|WITH)\b", cypher, re.IGNORECASE)
        if clause_match:
            start_pos = clause_match.start()
            return f"{cypher[:start_pos].rstrip()} WHERE {predicate} {cypher[start_pos:].lstrip()}"

        # If no RETURN or WITH, append at the end
        return f"{cypher.rstrip()} WHERE {predicate}"


# Default singleton filter
DEFAULT_TEMPORAL_FILTER = TemporalQueryFilter()


def inject_active_temporal_filter(
    cypher: str,
    as_of_date: Optional[str] = None,
    default_var: str = "r",
) -> str:
    """Convenience functional wrapper for active temporal state injection."""
    return DEFAULT_TEMPORAL_FILTER.inject_active_coalescence(cypher, as_of_date, default_var)
