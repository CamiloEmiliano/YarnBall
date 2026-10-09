# tests/ontology/test_ontology.py
# -*- coding: utf-8 -*-
"""
Formal unit test suite for the YarnBall Ontology Engineering Package (ontology/).

Covers:
1. Schema & Typology Axiomatization (schema.py)
2. 5-Axis Relational Reification (reification.py)
3. Integrity Guards, SHACL-Style Domain/Range Validation & DAG Cycle Breaker (validation.py)
4. Ontology Alignment & Identity Linkage (alignment.py)
5. Frame Semantics, Negation-Boundary Detection & DSL Compiler (compiler.py)
"""

import json
from pathlib import Path
import pytest

from ontology.schema import (
    EntityClass,
    RelationPredicate,
    ONTOLOGY_SCHEMA,
    SYMMETRIC_RELATIONSHIPS,
    DEFAULT_ALLOWED_LABELS,
)
from ontology.reification import (
    AnnotatedTriple,
    PolarityAxis,
    MaterialityTier,
    LifecycleStatus,
    CRITICAL_MATERIALITY_TERMS,
)
from ontology.validation import (
    GENERIC_NOUN_STOPLIST,
    PROVENANCE_CONFIDENCE_MAP,
    BENCHMARK_TOP_COMPANIES,
    is_generic_placeholder,
    validate_and_orient_triple,
    compute_edge_confidence,
    detect_and_break_ownership_cycles,
    parse_iso_date,
    validate_temporal_consistency,
    resolve_edge_state_transitions,
    check_degree_anomaly_quarantine,
)
from ontology.alignment import (
    CORPORATE_SUFFIXES,
    KNOWN_PUBLISHER_NAMES,
    normalize_entity_name,
    canonicalize_symmetric_edge,
    jaro_winkler_similarity,
)
from ontology.compiler import (
    POLARITY_TRIGGERS,
    NEGATION_REGEX,
    has_unnegated_pattern,
    classify_polarity_from_text,
    classify_financial_materiality,
    compile_to_cypher_dsl,
)


# ----------------------------------------------------------------------
# 1. Schema & Typology Axiomatization (ontology/schema.py)
# ----------------------------------------------------------------------
def test_ontology_schema_concept_typologies():
    """Verify that core concept classes and predicates are strictly defined enums."""
    assert EntityClass.COMPANY.value == "COMPANY"
    assert EntityClass.SUBSIDIARY.value == "SUBSIDIARY"
    assert EntityClass.PERSON.value == "PERSON"
    assert EntityClass.REGULATORY_BODY.value == "REGULATORYBODY"

    assert RelationPredicate.SUBSIDIARY_OF.value == "SUBSIDIARY_OF"
    assert RelationPredicate.SUPPLIES_TO.value == "SUPPLIES_TO"
    assert RelationPredicate.COMPETES_WITH.value == "COMPETES_WITH"
    assert RelationPredicate.CEO_OF.value == "CEO_OF"


def test_ontology_schema_domain_range_axioms():
    """Verify domain/range sets and symmetry flags in ONTOLOGY_SCHEMA."""
    # SUPPLIES_TO: Source in {COMPANY, SUBSIDIARY, SUPPLIER}, Target in {COMPANY, CUSTOMER}, Asymmetric
    src_set, tgt_set, is_sym = ONTOLOGY_SCHEMA["SUPPLIES_TO"]
    assert "COMPANY" in src_set
    assert "COMPANY" in tgt_set
    assert is_sym is False

    # COMPETES_WITH: Symmetric
    comp_src, comp_tgt, comp_sym = ONTOLOGY_SCHEMA["COMPETES_WITH"]
    assert "COMPANY" in comp_src
    assert comp_sym is True

    # Check symmetric set contains expected relations
    assert "COMPETES_WITH" in SYMMETRIC_RELATIONSHIPS
    assert "PARTNERED_WITH" in SYMMETRIC_RELATIONSHIPS
    assert "CO_INVESTS_WITH" in SYMMETRIC_RELATIONSHIPS
    assert "PEER_OF" in SYMMETRIC_RELATIONSHIPS
    assert "COLLABORATES_WITH" in SYMMETRIC_RELATIONSHIPS

    # Asymmetric relations must NOT be in SYMMETRIC_RELATIONSHIPS
    assert "SUBSIDIARY_OF" not in SYMMETRIC_RELATIONSHIPS
    assert "CEO_OF" not in SYMMETRIC_RELATIONSHIPS


def test_ontology_default_allowed_labels():
    """Verify standard node labels for Cypher AST validation."""
    assert "Company" in DEFAULT_ALLOWED_LABELS
    assert "Person" in DEFAULT_ALLOWED_LABELS
    assert "Product" in DEFAULT_ALLOWED_LABELS
    assert "Technology" in DEFAULT_ALLOWED_LABELS


# ----------------------------------------------------------------------
# 2. 5-Axis Relational Reification (ontology/reification.py)
# ----------------------------------------------------------------------
def test_annotated_triple_instantiation_and_serialization():
    """Verify construction and dictionary export of a 5-Axis AnnotatedTriple."""
    triple = AnnotatedTriple(
        source_name="Taiwan Semiconductor Manufacturing Company",
        source_type="Company",
        source_ticker="TSM",
        target_name="Apple Inc.",
        target_type="Company",
        target_ticker="AAPL",
        target_cik="0000320193",
        rel_type="SUPPLIES_TO",
        polarity=PolarityAxis.EXPANDING_BULLISH.value,
        materiality=MaterialityTier.CRITICAL_TIER_1.value,
        status=LifecycleStatus.ACTIVE_CURRENT.value,
        valid_from="2020-01-01",
        provenance="SEC_10K_ITEM1",
        confidence=0.98,
        nature="3nm advanced packaging silicon foundry",
    )

    data = triple.to_dict()
    assert data["source_name"] == "Taiwan Semiconductor Manufacturing Company"
    assert data["source_ticker"] == "TSM"
    assert data["target_ticker"] == "AAPL"
    assert data["polarity"] == "EXPANDING_BULLISH"
    assert data["materiality"] == "CRITICAL_TIER_1"
    assert data["status"] == "ACTIVE_CURRENT"
    assert data["confidence"] == 0.98


def test_annotated_triple_cypher_dsl_formatting():
    """Verify OpenCypher DSL compilation with full 5-Axis properties."""
    triple = AnnotatedTriple(
        source_name="Tim Cook",
        source_type="Person",
        target_name="Apple Inc.",
        target_type="Company",
        target_ticker="AAPL",
        rel_type="CEO_OF",
        polarity="NEUTRAL_STABLE",
        materiality="CRITICAL_TIER_1",
        provenance="SEC_10K_ITEM1",
        confidence=0.98,
    )

    dsl = triple.to_cypher_dsl()
    assert '(:Person {name: "Tim Cook"})' in dsl
    assert '-[:CEO_OF {' in dsl
    assert 'polarity: "NEUTRAL_STABLE"' in dsl
    assert 'materiality: "CRITICAL_TIER_1"' in dsl
    assert 'confidence: 0.98' in dsl
    assert '(:Company {name: "Apple Inc.", ticker: "AAPL"})' in dsl


# ----------------------------------------------------------------------
# 3. Integrity Guards & Validation (ontology/validation.py)
# ----------------------------------------------------------------------
def test_specificity_guard_generic_noun_stoplist():
    """Verify filtering of non-specific abstract nouns and journalistic pronouns."""
    assert is_generic_placeholder("the company") is True
    assert is_generic_placeholder("Our Customers") is True
    assert is_generic_placeholder("certain vendors") is True
    assert is_generic_placeholder("unnamed counterparty") is True
    assert is_generic_placeholder("Management") is True

    # Real discrete entities
    assert is_generic_placeholder("NVIDIA Corporation") is False
    assert is_generic_placeholder("Jensen Huang") is False
    assert is_generic_placeholder("") is False
    assert is_generic_placeholder(None) is False


def test_shacl_domain_range_validation_and_auto_orientation():
    """Verify domain/range checking and directional auto-orientation."""
    # 1. Forward valid: Person -> CEO_OF -> Company
    res = validate_and_orient_triple("Satya Nadella", "Person", "Microsoft Corp", "Company", "CEO_OF")
    assert res is not None
    assert res == ("Satya Nadella", "PERSON", "Microsoft Corp", "COMPANY", "CEO_OF")

    # 2. Inverted: Company -> CEO_OF -> Person (must auto-orient to Person -> Company)
    inv = validate_and_orient_triple("Microsoft Corp", "Company", "Satya Nadella", "Person", "CEO_OF")
    assert inv is not None
    assert inv == ("Satya Nadella", "PERSON", "Microsoft Corp", "COMPANY", "CEO_OF")

    # 3. Fundamentally invalid: Risk -> CEO_OF -> Person
    bad = validate_and_orient_triple("InterestRateRisk", "Risk", "Satya Nadella", "Person", "CEO_OF")
    assert bad is None


def test_epistemic_confidence_calibration():
    """Verify multi-source confidence weighting and corroboration bonuses."""
    # Statutory disclosures
    assert compute_edge_confidence("SEC_EXHIBIT_21") == 1.00
    assert compute_edge_confidence("SEC_FORM_4") == 1.00
    assert compute_edge_confidence("SEC_10K_ITEM1") == 0.98

    # Unverified news
    base_news = compute_edge_confidence("NEWS_EXTRACTION", mention_count=1)
    assert base_news == 0.60

    # Corroboration bonus: 2 distinct sources gives +0.05
    bonus_news = compute_edge_confidence("NEWS_EXTRACTION", distinct_sources_count=2)
    assert bonus_news == 0.65


def test_corporate_ownership_dag_cycle_breaking():
    """Verify DFS cycle detection enforcing Directed Acyclic Graph structure."""
    edges = [
        {"source_id": "HoldingCo", "target_id": "OpCo", "rel_type": "PARENT_OF", "confidence": 0.95},
        {"source_id": "OpCo", "target_id": "SubA", "rel_type": "SUBSIDIARY_OF", "confidence": 0.90},
        {"source_id": "SubA", "target_id": "SubB", "rel_type": "SUBSIDIARY_OF", "confidence": 0.85},
        {"source_id": "SubB", "target_id": "OpCo", "rel_type": "SUBSIDIARY_OF", "confidence": 0.40},  # Weakest cycle edge
    ]

    pruned = detect_and_break_ownership_cycles(edges)
    # 4 edges input -> weakest cycle edge (SubB -> OpCo with conf 0.40) must be removed
    assert len(pruned) == 3
    remaining_pairs = {(e["source_id"], e["target_id"]) for e in pruned}
    assert ("SubB", "OpCo") not in remaining_pairs
    assert ("HoldingCo", "OpCo") in remaining_pairs


def test_temporal_consistency_and_terminal_closure():
    """Verify valid_from <= valid_to and predecessor edge termination."""
    assert validate_temporal_consistency({"valid_from": "2021-01-01", "valid_to": "2023-01-01"}) is True
    assert validate_temporal_consistency({"valid_from": "2024-01-01", "valid_to": "2020-01-01"}) is False

    existing = [
        {"source_id": "A", "target_id": "B", "rel_type": "SUPPLIES_TO", "valid_from": "2020-01-01", "valid_to": None}
    ]
    terminal_event = [
        {"source_id": "A", "target_id": "B", "rel_type": "CONTRACT_TERMINATION", "valid_from": "2023-05-10"}
    ]

    closed = resolve_edge_state_transitions(existing, terminal_event)
    assert closed[0]["valid_to"] == "2023-05-10"
    assert closed[0]["status"] == "TERMINATED"


def test_degree_burst_circuit_breaker(tmp_path: Path):
    """Verify quarantine queue intercepts high-degree burst hallucinations."""
    q_file = tmp_path / "quarantine.jsonl"
    nodes = [{"id": "AAPL"}, {"id": "FakeSpamCo"}]
    edges = (
        [{"source_id": "AAPL", "target_id": f"P_{i}"} for i in range(30)]
        + [{"source_id": "FakeSpamCo", "target_id": f"P_{i}"} for i in range(30)]
    )

    clean_n, clean_e, quarantined = check_degree_anomaly_quarantine(
        nodes=nodes,
        edges=edges,
        degree_burst_threshold=25,
        quarantine_queue_path=q_file,
    )

    clean_ids = {n["id"] for n in clean_n}
    assert "AAPL" in clean_ids
    assert "FakeSpamCo" not in clean_ids
    assert len(quarantined) == 1
    assert quarantined[0]["entity_id"] == "FakeSpamCo"
    assert q_file.exists()


# ----------------------------------------------------------------------
# 4. Ontology Alignment & Identity Linkage (ontology/alignment.py)
# ----------------------------------------------------------------------
def test_entity_name_normalization_and_corporate_suffixes():
    """Verify stripping of legal suffixes and punctuation."""
    assert normalize_entity_name("Apple Inc.") == "apple"
    assert normalize_entity_name("Microsoft Corporation") == "microsoft"
    assert normalize_entity_name("Cisco Systems Inc.") == "cisco"
    assert normalize_entity_name("Taiwan Semiconductor Manufacturing Co., Ltd.") == "taiwan manufacturing"
    assert normalize_entity_name("Alphabet Inc - Class A") == "alphabet class a"


def test_symmetric_edge_canonicalization():
    """Verify canonical endpoint ordering for symmetric relationships."""
    # Symmetric: order must be lexicographical
    assert canonicalize_symmetric_edge("TSMC", "Intel", "COMPETES_WITH") == ("Intel", "TSMC")
    assert canonicalize_symmetric_edge("Intel", "TSMC", "COMPETES_WITH") == ("Intel", "TSMC")

    # Asymmetric: ordering must remain directional
    assert canonicalize_symmetric_edge("TSMC", "Apple", "SUPPLIES_TO") == ("TSMC", "Apple")
    assert canonicalize_symmetric_edge("Apple", "TSMC", "SUPPLIES_TO") == ("Apple", "TSMC")


def test_jaro_winkler_similarity_metric():
    """Verify string similarity metric and prefix weighting."""
    assert jaro_winkler_similarity("NVIDIA", "NVIDIA") == 1.0
    assert jaro_winkler_similarity("Apple", "Banana") < 0.5
    sim_prefix = jaro_winkler_similarity("Microsoft", "Micro hard")
    assert sim_prefix > 0.6


# ----------------------------------------------------------------------
# 5. Semantic Frame Triggers & Compiler (ontology/compiler.py)
# ----------------------------------------------------------------------
def test_negation_boundary_detection():
    """Verify local clause negation filtering (ISSUE-10)."""
    # Unnegated affirmative statement
    assert has_unnegated_pattern(r"defaulted", "The counterparty defaulted on payments.") is True

    # Negated statement within same clause
    assert has_unnegated_pattern(r"defaulted", "The company has not defaulted on its debt covenants.") is False
    assert has_unnegated_pattern(r"breach", "The firm avoided a breach of contract.") is False

    # Clause boundary reset: sentence 1 is negative, sentence 2 is positive
    mixed_text = "The firm has no issues. It defaulted yesterday."
    assert has_unnegated_pattern(r"defaulted", mixed_text) is True


def test_classify_polarity_and_materiality():
    """Verify discourse classification into 5-axis ontological categories."""
    # Disruptive shock via terminal relation
    assert classify_polarity_from_text(rel_type="DEFAULTED_ON") == "DISRUPTIVE_SHOCK"

    # Expanding bullish via trigger
    bull_text = "TSMC achieved record revenue commitments from primary hyperscalers."
    assert classify_polarity_from_text(context_text=bull_text, rel_type="SUPPLIES_TO") == "EXPANDING_BULLISH"

    # Contracting bearish
    bear_text = "Customer cutbacks led to reduced orders and margin compression."
    assert classify_polarity_from_text(context_text=bear_text, rel_type="SUPPLIES_TO") == "CONTRACTING_BEARISH"

    # Materiality: Sole Source = Critical Tier 1
    assert classify_financial_materiality(context_text="primary sole source supplier", rel_type="SUPPLIES_TO") == "CRITICAL_TIER_1"
    # Materiality: SEC Exhibit 21 = Critical Tier 1
    assert classify_financial_materiality(rel_type="SUBSIDIARY_OF", provenance="SEC_EXHIBIT_21") == "CRITICAL_TIER_1"
    # Standard supply = Material Tier 2
    assert classify_financial_materiality(context_text="ongoing commercial supply", rel_type="SUPPLIES_TO") == "MATERIAL_TIER_2"
    # Commodity off-the-shelf = Commodity Tier 3
    assert classify_financial_materiality(context_text="standard vendor off-the-shelf purchase", rel_type="SUPPLIES_TO") == "COMMODITY_TIER_3"


def test_compile_to_cypher_dsl_wrapper():
    """Verify compile_to_cypher_dsl compiles AnnotatedTriple directly."""
    triple = AnnotatedTriple(
        source_name="NVIDIA",
        source_type="Company",
        target_name="TSMC",
        target_type="Company",
        rel_type="CUSTOMER_OF",
    )
    dsl = compile_to_cypher_dsl(triple)
    assert '(:Company {name: "NVIDIA"})-[:CUSTOMER_OF' in dsl
    assert '(:Company {name: "TSMC"})' in dsl
