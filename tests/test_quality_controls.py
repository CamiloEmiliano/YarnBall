# tests/test_quality_controls.py
# -*- coding: utf-8 -*-
"""Comprehensive unit tests for the 6 Institutional Knowledge Graph Quality Controls."""

import json
from pathlib import Path
import pytest

from graph.quality_controls import (
    is_generic_placeholder,
    validate_and_orient_triple,
    compute_edge_confidence,
    detect_and_break_ownership_cycles,
    validate_temporal_consistency,
    resolve_edge_state_transitions,
    check_degree_anomaly_quarantine,
    BENCHMARK_TOP_COMPANIES,
)
from graph.entity_resolver import EntityResolver, is_blacklisted_publisher


# ----------------------------------------------------------------------
# Control 1: Generic Noun & Pronoun Stoplist
# ----------------------------------------------------------------------
def test_generic_noun_pronoun_filter():
    """Verify that generic nouns and legal pronouns are filtered out."""
    assert is_generic_placeholder("The Company") is True
    assert is_generic_placeholder("the customer") is True
    assert is_generic_placeholder("Management") is True
    assert is_generic_placeholder("certain suppliers") is True
    assert is_generic_placeholder("Various Clients") is True
    assert is_generic_placeholder("unnamed counterparty") is True
    assert is_generic_placeholder("our subsidiaries") is True
    assert is_generic_placeholder(None) is False
    assert is_generic_placeholder("") is False

    # Real entities must NOT be filtered
    assert is_generic_placeholder("Apple Inc.") is False
    assert is_generic_placeholder("Microsoft Corporation") is False
    assert is_generic_placeholder("Tim Cook") is False
    assert is_generic_placeholder("NVIDIA") is False


# ----------------------------------------------------------------------
# Control 2: Ontological Domain & Range Schema Validator
# ----------------------------------------------------------------------
def test_ontological_domain_range_schema_validator():
    """Verify domain/range enforcement and auto-orientation for inverted triples."""
    # 1. Valid forward triple
    res = validate_and_orient_triple("Tim Cook", "Person", "Apple Inc.", "Company", "CEO_OF")
    assert res is not None
    src, src_lbl, tgt, tgt_lbl, rel = res
    assert src == "Tim Cook"
    assert tgt == "Apple Inc."
    assert rel == "CEO_OF"

    # 2. Inverted triple auto-orientation (Company -> CEO_OF -> Person => Person -> CEO_OF -> Company)
    inv_res = validate_and_orient_triple("Apple Inc.", "Company", "Tim Cook", "Person", "CEO_OF")
    assert inv_res is not None
    inv_src, inv_src_lbl, inv_tgt, inv_tgt_lbl, inv_rel = inv_res
    assert inv_src == "Tim Cook"
    assert inv_tgt == "Apple Inc."
    assert inv_rel == "CEO_OF"

    # 3. Invalid schema triple (e.g. GeopoliticalRisk -> CEO_OF -> Person)
    bad_res = validate_and_orient_triple("Taiwan Conflict", "GeopoliticalRisk", "Tim Cook", "Person", "CEO_OF")
    assert bad_res is None

    # 4. Valid supply chain triple
    supp_res = validate_and_orient_triple("TSMC", "Company", "Apple Inc.", "Company", "SUPPLIES_TO")
    assert supp_res is not None
    assert supp_res[0] == "TSMC"
    assert supp_res[2] == "Apple Inc."


# ----------------------------------------------------------------------
# Control 3: Multi-Source Evidence Weighting & Confidence Calibration
# ----------------------------------------------------------------------
def test_multi_source_evidence_weighting():
    """Verify calibrated confidence assignments across provenances."""
    # SEC Exhibit 21 ground truth anchor
    assert compute_edge_confidence("SEC_EXHIBIT_21") == 1.00
    # SEC Form 4 insider transaction
    assert compute_edge_confidence("SEC_FORM_4") == 1.00
    # SEC Form 10-K
    assert compute_edge_confidence("SEC_10K_ITEM1") == 0.98
    # SEC Form 8-K
    assert compute_edge_confidence("SEC_8K") == 0.95
    # Multi-teacher LLM consensus
    assert compute_edge_confidence("MULTI_TEACHER_CONSENSUS") == 0.85
    # Single unverified news extraction
    assert compute_edge_confidence("NEWS_EXTRACTION") == 0.60

    # Multi-mention confidence bonus (+0.05 per extra mention)
    conf_1 = compute_edge_confidence("NEWS_EXTRACTION", mention_count=1)
    conf_2 = compute_edge_confidence("NEWS_EXTRACTION", mention_count=2)
    conf_4 = compute_edge_confidence("NEWS_EXTRACTION", mention_count=4)
    assert conf_2 == round(conf_1 + 0.05, 2)
    assert conf_4 == round(conf_1 + 0.15, 2)
    assert conf_4 <= 1.0


# ----------------------------------------------------------------------
# Control 4: Corporate Ownership DAG Cycle Breaker
# ----------------------------------------------------------------------
def test_corporate_ownership_dag_cycle_breaker():
    """Verify that circular ownership dependencies are detected and pruned."""
    edges = [
        # Normal supply edge
        {"source_id": "TSMC", "target_id": "AAPL", "rel_type": "SUPPLIES_TO", "confidence": 0.9},
        # Cycle: Sub A -> Sub B -> Sub C -> Sub A
        {"source_id": "Sub_A", "target_id": "Sub_B", "rel_type": "SUBSIDIARY_OF", "confidence": 0.95},
        {"source_id": "Sub_B", "target_id": "Sub_C", "rel_type": "SUBSIDIARY_OF", "confidence": 0.90},
        {"source_id": "Sub_C", "target_id": "Sub_A", "rel_type": "SUBSIDIARY_OF", "confidence": 0.60}, # Weakest edge
    ]

    cleaned_edges = detect_and_break_ownership_cycles(edges)

    # 4 edges input -> 3 edges remaining (weakest cycle edge Sub_C -> Sub_A dropped)
    assert len(cleaned_edges) == 3
    remaining_ownership = [e for e in cleaned_edges if e["rel_type"] == "SUBSIDIARY_OF"]
    assert len(remaining_ownership) == 2
    # Verify weakest edge was removed
    dropped = [e for e in remaining_ownership if e["source_id"] == "Sub_C" and e["target_id"] == "Sub_A"]
    assert len(dropped) == 0


def test_ownership_cycle_slicing_with_upstream_tail():
    """Verify ISSUE-03 fix: an upstream non-cyclical parent edge is NEVER pruned even if it has lowest confidence."""
    edges = [
        # Upstream tail: HoldingCo -> Sub_A with low confidence (0.2)
        {"source_id": "HoldingCo", "target_id": "Sub_A", "rel_type": "PARENT_OF", "confidence": 0.20},
        # Cycle: Sub_A -> Sub_B -> Sub_C -> Sub_A with higher confidence
        {"source_id": "Sub_A", "target_id": "Sub_B", "rel_type": "SUBSIDIARY_OF", "confidence": 0.90},
        {"source_id": "Sub_B", "target_id": "Sub_C", "rel_type": "SUBSIDIARY_OF", "confidence": 0.85},
        {"source_id": "Sub_C", "target_id": "Sub_A", "rel_type": "SUBSIDIARY_OF", "confidence": 0.70},
    ]

    cleaned_edges = detect_and_break_ownership_cycles(edges)

    # The cycle edge Sub_C -> Sub_A (0.70) must be dropped, NOT HoldingCo -> Sub_A (0.20)
    assert len(cleaned_edges) == 3
    parent_edges = [e for e in cleaned_edges if e.get("rel_type") == "PARENT_OF"]
    assert len(parent_edges) == 1
    assert parent_edges[0]["source_id"] == "HoldingCo"
    assert parent_edges[0]["target_id"] == "Sub_A"


# ----------------------------------------------------------------------
# Control 5: Temporal Consistency & State Transition Guard
# ----------------------------------------------------------------------
def test_temporal_consistency_and_state_transitions():
    """Verify temporal bounds checking and auto-closure of predecessor relationships."""
    # 1. Temporal bounds check
    valid_edge = {"valid_from": "2020-01-01", "valid_to": "2023-01-01"}
    invalid_edge = {"valid_from": "2024-01-01", "valid_to": "2021-01-01"}
    open_edge = {"valid_from": "2022-01-01", "valid_to": None}

    assert validate_temporal_consistency(valid_edge) is True
    assert validate_temporal_consistency(invalid_edge) is False
    assert validate_temporal_consistency(open_edge) is True

    # 2. State transition auto-closure
    existing = [
        {"source_id": "CorpA", "target_id": "CorpB", "rel_type": "PARTNERED_WITH", "valid_from": "2019-01-01", "valid_to": None}
    ]
    new_event = [
        {"source_id": "CorpA", "target_id": "CorpB", "rel_type": "TERMINATED", "valid_from": "2023-06-15"}
    ]

    updated = resolve_edge_state_transitions(existing, new_event)
    assert len(updated) == 1
    assert updated[0]["valid_to"] == "2023-06-15"
    assert updated[0]["status"] == "TERMINATED"


# ----------------------------------------------------------------------
# Control 6: Degree Anomaly Quarantine Circuit Breaker
# ----------------------------------------------------------------------
def test_degree_anomaly_quarantine(tmp_path: Path):
    """Verify degree anomaly circuit breaker quarantines high-burst hallucinated entities."""
    queue_path = tmp_path / "quarantine_test.jsonl"

    nodes = [
        {"id": "AAPL", "label": "Company"},
        {"id": "Hallucinated_Entity_XYZ", "label": "Company"},
    ]

    # Generate 30 edges for AAPL (benchmark entity, allowed)
    aapl_edges = [
        {"source_id": "AAPL", "target_id": f"Partner_{i}", "rel_type": "PARTNERED_WITH"}
        for i in range(30)
    ]

    # Generate 30 edges for Hallucinated_Entity_XYZ (non-benchmark, should be quarantined)
    burst_edges = [
        {"source_id": "Hallucinated_Entity_XYZ", "target_id": f"Target_{i}", "rel_type": "ACQUIRED"}
        for i in range(30)
    ]

    all_edges = aapl_edges + burst_edges

    clean_nodes, clean_edges, quarantined = check_degree_anomaly_quarantine(
        nodes=nodes,
        edges=all_edges,
        degree_burst_threshold=25,
        quarantine_queue_path=queue_path,
    )

    # AAPL is retained, Hallucinated_Entity_XYZ is quarantined
    clean_node_ids = {n["id"] for n in clean_nodes}
    assert "AAPL" in clean_node_ids
    assert "Hallucinated_Entity_XYZ" not in clean_node_ids

    # Check quarantine queue file was written
    assert queue_path.exists()
    assert len(quarantined) == 1
    assert quarantined[0]["entity_id"] == "Hallucinated_Entity_XYZ"


# ----------------------------------------------------------------------
# End-to-End Integrated Resolver Tests
# ----------------------------------------------------------------------
def test_entity_resolver_integrated_controls():
    """Verify full integration of quality controls inside EntityResolver."""
    resolver = EntityResolver()

    raw_nodes = [
        {"id": "Apple Inc.", "label": "Company", "ticker": "AAPL"},
        {"id": "Apple", "label": "Company", "ticker": "AAPL"},
        {"id": "The Motley Fool", "label": "Company"}, # Publisher Stoplist
        {"id": "The Company", "label": "Company"},      # Generic Noun Stoplist
        {"id": "Management", "label": "Executive"},    # Generic Noun Stoplist
        {"id": "Tim Cook", "label": "Person"},
    ]

    node_mapping, resolved_nodes = resolver.resolve_nodes(raw_nodes)

    resolved_ids = {n["id"] for n in resolved_nodes}
    assert "Apple Inc." in resolved_ids
    assert "Tim Cook" in resolved_ids
    # Blacklisted nodes must not exist
    assert "The Motley Fool" not in resolved_ids
    assert "The Company" not in resolved_ids
    assert "Management" not in resolved_ids

    raw_edges = [
        # Inverted CEO triple (Company -> CEO_OF -> Person)
        {"source_id": "Apple", "target_id": "Tim Cook", "rel_type": "CEO_OF", "provenance": "SEC_10K_ITEM1"},
        # Edge to publisher (should be dropped)
        {"source_id": "Apple", "target_id": "The Motley Fool", "rel_type": "MENTIONED_IN"},
        # Edge to generic placeholder (should be dropped)
        {"source_id": "Apple", "target_id": "The Company", "rel_type": "SUBSIDIARY_OF"},
        # Invalid temporal edge (should be dropped)
        {"source_id": "Apple", "target_id": "TSMC", "rel_type": "CUSTOMER_OF", "valid_from": "2024-01-01", "valid_to": "2020-01-01"},
    ]

    label_map = {n["id"]: n["label"] for n in resolved_nodes}
    label_map["TSMC"] = "Company"

    rewired = resolver.rewire_edges(raw_edges, node_mapping, node_label_map=label_map)

    # Only the re-oriented CEO_OF edge should survive
    assert len(rewired) == 1
    ceo_edge = rewired[0]
    assert ceo_edge["source_id"] == "Tim Cook"
    assert ceo_edge["target_id"] == "Apple Inc."
    assert ceo_edge["rel_type"] == "CEO_OF"
    assert ceo_edge["confidence"] == 0.98
    assert ceo_edge["provenance"] == "SEC_10K_ITEM1"
