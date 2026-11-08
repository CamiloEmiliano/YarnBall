"""Unit tests for Stage 6.3: Tier-4 Ontology Gate with Relation Repair."""

import pytest
from ontology.guardrails import (
    Tier4OntologyGate,
    DEFAULT_ONTOLOGY_GATE,
    apply_runtime_guardrails,
)


@pytest.fixture
def gate() -> Tier4OntologyGate:
    return Tier4OntologyGate()


def test_subsidiary_cik_rollup(gate: Tier4OntologyGate):
    # DISH DBS CIK 0001001811 rolls up to DISH Network Corp 0001001082
    raw = "(:Company {name: 'DISH DBS', cik: '0001001811'})-[:SUPPLIES_TO]->(:Company {name: 'EchoStar', cik: '0001415404'})"
    result = gate.apply_guardrails(raw, "EXTRACT_SEC_GRAPH")
    assert "0001001082" in result
    assert "0001001811" not in result


def test_canonical_relations_preserved(gate: Tier4OntologyGate):
    raw = "(:Company {name: 'NVIDIA', cik: '0001045810'})-[:SUPPLIES_TO {status: 'ACTIVE_CURRENT'}]->(:Company {name: 'Microsoft', cik: '0000789019'})"
    result = gate.apply_guardrails(raw, "EXTRACT_SEC_GRAPH")
    assert "[:SUPPLIES_TO" in result
    assert result != "(none)"


def test_phase5_tt05b_long_terminated_01_repaired(gate: Tier4OntologyGate):
    # Exact raw prediction from TT05B_LONG_TERMINATED_01 that previously regressed to (none)
    raw = (
        "(:Company {name: 'Xilinx, Inc.', ticker: 'XLNX', cik: '0000824801'})-"
        "[:TERMINATED {polarity: 'NEUTRAL_STABLE', materiality: 'CRITICAL_TIER_1', "
        "status: 'TERMINATED', provenance: 'SEC_10K_ITEM1', confidence: 0.98, "
        "nature: 'terminated', valid_from: '2026-10-26'}]->"
        "(:Company {name: 'SVB Financial Group', ticker: 'SIVB', cik: '0000719739'})"
    )
    result = gate.apply_guardrails(raw, "EXTRACT_SEC_GRAPH")
    assert result != "(none)", "Repaired extraction must NOT be pruned to (none)"
    assert "[:SUPPLIES_TO" in result, "Predicate must be mapped to canonical relation"
    assert "status: 'TERMINATED'" in result
    assert "0000824801" in result
    assert "0000719739" in result


def test_all_phase5_tt05b_terminated_extractions_repaired(gate: Tier4OntologyGate):
    samples = [
        # TT05B_LONG_TERMINATED_02
        "(:Company {name: 'Vornado Realty Trust', ticker: 'VNO', cik: '0000899689'})-[:TERMINATED {status: 'TERMINATED'}]->(:Company {name: 'Nielsen', ticker: 'NLSN', cik: '0001492633'})",
        # TT05B_LONG_TERMINATED_03
        "(:Company {name: 'Signature Bank', ticker: 'SBNY', cik: '0001128882'})-[:TERMINATED {status: 'TERMINATED'}]->(:Company {name: 'Agilent', ticker: 'A', cik: '0001090872'})",
        # TT05B_LONG_TERMINATED_04
        "(:Company {name: 'Allstate', ticker: 'ALL', cik: '0000899051'})-[:TERMINATED {status: 'TERMINATED'}]->(:Company {name: 'Tiffany', ticker: 'TIF', cik: '0000098246'})",
    ]
    for raw in samples:
        res = gate.apply_guardrails(raw, "EXTRACT_SEC_GRAPH")
        assert res != "(none)"
        assert "[:SUPPLIES_TO" in res
        assert "status: 'TERMINATED'" in res


def test_descriptive_noise_predicates_pruned(gate: Tier4OntologyGate):
    noise_predictions = [
        "(:Company {name: 'NVIDIA', cik: '0001045810'})-[:COMPARES_TO]->(:Company {name: 'AMD', cik: '0000002488'})",
        "(:Company {name: 'Tesla', cik: '0001318605'})-[:RISES_WITH_BENCHMARK]->(:Company {name: 'SP500', cik: '0000000000'})",
        "(:Company {name: 'Apple', cik: '0000320193'})-[:TRADED_IN]->(:Company {name: 'NASDAQ', cik: '0000000001'})",
    ]
    for noise in noise_predictions:
        res = gate.apply_guardrails(noise, "EXTRACT_SEC_GRAPH")
        assert res == "(none)", f"Descriptive noise '{noise}' must be pruned to (none)"


def test_empty_and_none_handled(gate: Tier4OntologyGate):
    assert gate.apply_guardrails("", "EXTRACT_SEC_GRAPH") == "(none)"
    assert gate.apply_guardrails("   ", "EXTRACT_SEC_GRAPH") == "(none)"
    assert gate.apply_guardrails("(none)", "EXTRACT_SEC_GRAPH") == "(none)"
    assert gate.apply_guardrails("No relationships found.", "EXTRACT_SEC_GRAPH") == "(none)"


def test_convenience_wrapper_matches_class():
    raw = "(:Company {name: 'A', cik: '0001001811'})-[:SUPPLIES_TO]->(:Company {name: 'B', cik: '0000789019'})"
    assert apply_runtime_guardrails("prompt", raw, "EXTRACT_SEC_GRAPH") == DEFAULT_ONTOLOGY_GATE.apply_guardrails(raw, "EXTRACT_SEC_GRAPH")

