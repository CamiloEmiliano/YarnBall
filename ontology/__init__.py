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
    LABEL_NORMALIZATION_MAP,
    ONTOLOGY_SCHEMA,
    PREDICATE_ALIASES,
    RelationPredicate,
    SYMMETRIC_RELATIONSHIPS,
    format_extraction_prompt_vocab,
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
    validate_graph_payload,
    validate_temporal_consistency,
)
from .alignment import (
    CORPORATE_SUFFIXES,
    KNOWN_PUBLISHER_NAMES,
    canonicalize_symmetric_edge,
    extract_initialism,
    jaro_winkler_similarity,
    normalize_entity_name,
)
from .compiler import (
    MODAL_HEDGE_REGEX,
    NEGATION_REGEX,
    POLARITY_TRIGGERS,
    classify_financial_materiality,
    classify_polarity_from_text,
    compile_to_cypher_dsl,
    has_unnegated_pattern,
    is_hypothetical_or_hedged,
    is_triple_hedged,
    split_financial_clauses,
)
from .guardrails import (
    DEFAULT_ONTOLOGY_GATE,
    SEED_SUBSIDIARY_TO_PARENT_CIK_MAP,
    SUBSIDIARY_TO_PARENT_CIK_MAP,
    SubsidiaryRegistry,
    Tier4OntologyGate,
    apply_runtime_guardrails,
    infer_terminated_relation,
    invalidate_subsidiary_registry_cache,
    load_dynamic_subsidiary_registry,
    load_subsidiary_cik_map,
)

__all__ = [
    # Schema
    "EntityClass",
    "RelationPredicate",
    "ONTOLOGY_SCHEMA",
    "PREDICATE_ALIASES",
    "LABEL_NORMALIZATION_MAP",
    "SYMMETRIC_RELATIONSHIPS",
    "DEFAULT_ALLOWED_LABELS",
    "format_extraction_prompt_vocab",
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
    "validate_graph_payload",
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
    "extract_initialism",
    "jaro_winkler_similarity",
    # Compiler
    "POLARITY_TRIGGERS",
    "NEGATION_REGEX",
    "MODAL_HEDGE_REGEX",
    "split_financial_clauses",
    "has_unnegated_pattern",
    "is_hypothetical_or_hedged",
    "is_triple_hedged",
    "classify_polarity_from_text",
    "classify_financial_materiality",
    "compile_to_cypher_dsl",
    # Guardrails
    "Tier4OntologyGate",
    "DEFAULT_ONTOLOGY_GATE",
    "SEED_SUBSIDIARY_TO_PARENT_CIK_MAP",
    "SUBSIDIARY_TO_PARENT_CIK_MAP",
    "SubsidiaryRegistry",
    "load_dynamic_subsidiary_registry",
    "load_subsidiary_cik_map",
    "invalidate_subsidiary_registry_cache",
    "apply_runtime_guardrails",
    "infer_terminated_relation",
]
