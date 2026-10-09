"""
Legacy Quality Controls Adapter.

DEPRECATION & ARCHITECTURAL NOTICE:
The knowledge graph quality controls, schema axiomatization, DAG cycle breakers,
and SHACL-style domain/range validators have been formalized and migrated to the
first-class `ontology` package (see `ontology/schema.py` and `ontology/validation.py`).

This module re-exports all canonical definitions to maintain 100% backward compatibility
across the codebase and existing test suites.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import date

try:
    from ontology.schema import (
        ONTOLOGY_SCHEMA,
        SYMMETRIC_RELATIONSHIPS,
        DEFAULT_ALLOWED_LABELS,
    )
    from ontology.validation import (
        GENERIC_NOUN_STOPLIST,
        is_generic_placeholder,
        validate_and_orient_triple,
        PROVENANCE_CONFIDENCE_MAP,
        compute_edge_confidence,
        detect_and_break_ownership_cycles,
        parse_iso_date,
        validate_temporal_consistency,
        resolve_edge_state_transitions,
        BENCHMARK_TOP_COMPANIES,
        check_degree_anomaly_quarantine,
    )
except ImportError:
    from graphrag_finance.ontology.schema import (
        ONTOLOGY_SCHEMA,
        SYMMETRIC_RELATIONSHIPS,
        DEFAULT_ALLOWED_LABELS,
    )
    from graphrag_finance.ontology.validation import (
        GENERIC_NOUN_STOPLIST,
        is_generic_placeholder,
        validate_and_orient_triple,
        PROVENANCE_CONFIDENCE_MAP,
        compute_edge_confidence,
        detect_and_break_ownership_cycles,
        parse_iso_date,
        validate_temporal_consistency,
        resolve_edge_state_transitions,
        BENCHMARK_TOP_COMPANIES,
        check_degree_anomaly_quarantine,
    )

logger = logging.getLogger("graph_quality_controls")

__all__ = [
    "GENERIC_NOUN_STOPLIST",
    "is_generic_placeholder",
    "ONTOLOGY_SCHEMA",
    "validate_and_orient_triple",
    "PROVENANCE_CONFIDENCE_MAP",
    "compute_edge_confidence",
    "detect_and_break_ownership_cycles",
    "parse_iso_date",
    "validate_temporal_consistency",
    "resolve_edge_state_transitions",
    "BENCHMARK_TOP_COMPANIES",
    "check_degree_anomaly_quarantine",
    "SYMMETRIC_RELATIONSHIPS",
    "DEFAULT_ALLOWED_LABELS",
]
