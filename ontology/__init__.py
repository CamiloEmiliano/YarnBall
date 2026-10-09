"""
Formal Knowledge Representation & Ontology Engineering Package for YarnBall.

Provides:
- Schema: Concept typologies, closed relational vocabulary, domain & range axioms
- Reification: 5-Axis AnnotatedTriple model (Entity, Relation, Polarity, Materiality, Temporal)
- Validation: SHACL-style domain/range checking, auto-orientation, DAG cycle breaker, stoplist guards
- Alignment: Entity name normalization, symmetric edge canonicalization, Jaro-Winkler linkage
- Compiler: Lexical trigger mining, negation-boundary handling, OpenCypher DSL compiler
"""

from .schema import (
    DEFAULT_ALLOWED_LABELS,
    EntityClass,
    ONTOLOGY_SCHEMA,
    RelationPredicate,
    SYMMETRIC_RELATIONSHIPS,
)
from .reification import (
    AnnotatedTriple,
    CRITICAL_MATERIALITY_TERMS,
    LifecycleStatus,
    MaterialityTier,
    PolarityAxis,
)
from .validation import (
    BENCHMARK_TOP_COMPANIES,
    GENERIC_NOUN_STOPLIST,
    PROVENANCE_CONFIDENCE_MAP,
    check_degree_anomaly_quarantine,
    compute_edge_confidence,
    detect_and_break_ownership_cycles,
    is_generic_placeholder,
    parse_iso_date,
    resolve_edge_state_transitions,
    validate_and_orient_triple,
    validate_temporal_consistency,
)
from .alignment import (
    CORPORATE_SUFFIXES,
    KNOWN_PUBLISHER_NAMES,
    canonicalize_symmetric_edge,
    jaro_winkler_similarity,
    normalize_entity_name,
)
from .compiler import (
    NEGATION_REGEX,
    POLARITY_TRIGGERS,
    classify_financial_materiality,
    classify_polarity_from_text,
    compile_to_cypher_dsl,
    has_unnegated_pattern,
)

__all__ = [
    # Schema
    "EntityClass",
    "RelationPredicate",
    "ONTOLOGY_SCHEMA",
    "SYMMETRIC_RELATIONSHIPS",
    "DEFAULT_ALLOWED_LABELS",
    # Reification
    "AnnotatedTriple",
    "PolarityAxis",
    "MaterialityTier",
    "LifecycleStatus",
    "CRITICAL_MATERIALITY_TERMS",
    # Validation
    "GENERIC_NOUN_STOPLIST",
    "PROVENANCE_CONFIDENCE_MAP",
    "BENCHMARK_TOP_COMPANIES",
    "is_generic_placeholder",
    "validate_and_orient_triple",
    "compute_edge_confidence",
    "detect_and_break_ownership_cycles",
    "parse_iso_date",
    "validate_temporal_consistency",
    "resolve_edge_state_transitions",
    "check_degree_anomaly_quarantine",
    # Alignment
    "CORPORATE_SUFFIXES",
    "KNOWN_PUBLISHER_NAMES",
    "normalize_entity_name",
    "canonicalize_symmetric_edge",
    "jaro_winkler_similarity",
    # Compiler
    "POLARITY_TRIGGERS",
    "NEGATION_REGEX",
    "has_unnegated_pattern",
    "classify_polarity_from_text",
    "classify_financial_materiality",
    "compile_to_cypher_dsl",
]
