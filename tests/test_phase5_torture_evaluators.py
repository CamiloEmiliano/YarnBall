"""
Unit Tests for Phase 5 Diagnostic Adversarial Torture Suite Builder, Evaluators, and Decision Engine.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("EDGAR_LOCAL_DATA_DIR", "/tmp/.edgar")

from sft_pipeline.evaluation.phase5_torture_evaluators import (
    evaluate_causal_hop_trace,
    evaluate_epistemic_provenance,
    evaluate_schema_and_cik_integrity,
    evaluate_torture_sample,
    evaluate_verdict_and_polarity,
)
from sft_pipeline.evaluation.phase5_torture_runner import compute_architectural_decisions
from sft_pipeline.evaluation.phase5_torture_suite_builder import build_phase5_torture_suite


def test_build_phase5_torture_suite_invariants(tmp_path: Path) -> None:
    out_file = tmp_path / "suite.jsonl"
    suite = build_phase5_torture_suite(output_path=out_file, seed=20261107)

    assert len(suite) == 150
    assert out_file.is_file()

    axis_counts = Counter(row["axis_id"] for row in suite)
    assert axis_counts == {
        "TT-01": 25,
        "TT-02": 25,
        "TT-03": 25,
        "TT-04": 25,
        "TT-05": 25,
        "TT-06": 25,
    }

    # Verify unique sample_ids and valid 10-digit CIKs wherever allowed_ciks are populated
    sample_ids = [row["sample_id"] for row in suite]
    assert len(set(sample_ids)) == 150
    for row in suite:
        for cik in row.get("expected_assertions", {}).get("allowed_ciks", []):
            assert len(cik) == 10 and cik.isdigit()


def test_verdict_and_polarity_evaluator() -> None:
    disruptive_cot = (
        "<think>\n1. Origin halt hits Tier-1 supplier and cascades downstream across two quarters.\n"
        "2. Margins compress due to unabsorbed overhead.\n</think>\n"
        "**Contagion Analysis Summary**:\n- **Risk Classification**: `DISRUPTIVE_SHOCK` with severe delays."
    )
    contained_cot = (
        "<think>\n1. Secondary vendor outage is fully buffered by 14-month bonded inventory.\n"
        "2. Critical path operates normally at 100% capacity.\n</think>\n"
        "**Contagion Verdict**: `CONTAINED_MINIMAL_CONTAGION`. Zero material disruption."
    )

    assert evaluate_verdict_and_polarity(disruptive_cot, {"expected_verdict": "DISRUPTIVE_SHOCK"})[0] is True
    assert evaluate_verdict_and_polarity(contained_cot, {"expected_verdict": "DISRUPTIVE_SHOCK"})[0] is False
    assert evaluate_verdict_and_polarity(contained_cot, {"expected_verdict": "CONTAINED_MINIMAL_CONTAGION"})[0] is True
    assert evaluate_verdict_and_polarity(disruptive_cot, {"expected_verdict": "CONTAINED_MINIMAL_CONTAGION"})[0] is False

    # Competitor tailwind vs smeared bearish halt
    tailwind_cot = (
        "<think>\n1. Foundry halt cripples Firm X, while rival NVDA uses independent healthy foundry capacity.\n"
        "2. NVDA is unaffected and positioned for market-share gain.\n</think>\n"
        "**Summary**: NVDA is insulated and captures competitive market share upside as a beneficiary."
    )
    smeared_cot = (
        "<think>\n1. Foundry halt disrupts the sector and halts production everywhere.\n"
        "2. Margins compress broadly across peers.\n</think>\n"
        "**Summary**: `DISRUPTIVE_SHOCK` — NVDA halts production immediately due to shortages."
    )
    assert (
        evaluate_verdict_and_polarity(
            tailwind_cot,
            {"expected_verdict": "COMPETITOR_TAILWIND", "queried_entity_ticker": "NVDA"},
        )[0]
        is True
    )
    assert (
        evaluate_verdict_and_polarity(
            smeared_cot,
            {"expected_verdict": "COMPETITOR_TAILWIND", "queried_entity_ticker": "NVDA"},
        )[0]
        is False
    )


def test_causal_hop_trace_and_epistemic_evaluators() -> None:
    good_trace = (
        "<think>\n1. Shock originates at ASML and propagates through TSM to NVDA and finally MSFT.\n"
        "2. Each Tier-1 hop lacks spare requalified capacity over the next two quarters.\n</think>\n"
        "**Summary**: `DISRUPTIVE_SHOCK` propagates to MSFT via unconfirmed implied link."
    )
    dropped_hop = (
        "<think>\n1. Shock originates at ASML and directly hits MSFT downstream in the cloud sector.\n"
        "2. Operating margins compress across enterprise customers.\n</think>\n"
        "**Summary**: `DISRUPTIVE_SHOCK` hits MSFT."
    )

    assertions = {
        "task_type": "CONTAGION_REASONING",
        "required_bridge_entities": ["TSM|Taiwan Semiconductor", "NVDA|NVIDIA", "MSFT|Microsoft"],
        "epistemic_mode": "HIGH_PROB_IMPLIED_EDGE",
    }
    assert evaluate_causal_hop_trace(good_trace, assertions)[0] is True
    assert evaluate_causal_hop_trace(dropped_hop, assertions)[0] is False
    assert evaluate_epistemic_provenance(good_trace, assertions)[0] is True

    unqualified_trace = good_trace.replace("unconfirmed implied", "hard verified")
    assert evaluate_epistemic_provenance(unqualified_trace, assertions)[0] is False


def test_schema_property_whitelist_and_cik_gate() -> None:
    # Company name with periods ('A. O. Smith') must NOT trigger false dot-property violation
    valid_cypher = (
        "MATCH (s:Company)-[r:SUPPLIES_TO]->(t:Company {name: 'A. O. Smith', ticker: 'AOS'})\n"
        "WHERE r.status = 'ACTIVE_CURRENT' AND r.materiality = 'CRITICAL_TIER_1'\n"
        "RETURN s.name, s.ticker;"
    )
    ok, msg = evaluate_schema_and_cik_integrity(
        valid_cypher,
        {
            "task_type": "TEXT_TO_CYPHER",
            "unanswerable_schema_trap": False,
            "required_cypher_patterns": [r"\bMATCH\b", r"\bWHERE\b", r"\bRETURN\b", r"AOS"],
            "allowed_ciks": [],
        },
    )
    assert ok is True, msg

    # Invented properties (.ceo_age, carbon_emissions) must be rejected
    fake_prop_cypher = (
        "MATCH (s:Company)-[:SUPPLIES_TO]->(t:Company {ticker: 'AAPL', carbon_emissions: 100})\n"
        "WHERE s.ceo_age > 60\nRETURN s.name;"
    )
    ok_fake, msg_fake = evaluate_schema_and_cik_integrity(
        fake_prop_cypher,
        {"task_type": "TEXT_TO_CYPHER", "unanswerable_schema_trap": True, "allowed_ciks": []},
    )
    assert ok_fake is False
    assert "ceo_age" in msg_fake and "carbon_emissions" in msg_fake

    # Hallucinated 10-digit CIK on Task A must be rejected
    bad_cik_triple = (
        "(:Company {name: 'Apple Inc.', ticker: 'AAPL', cik: '9999999999'})"
        "-[:SUPPLIES_TO {status: 'ACTIVE_CURRENT'}]->"
        "(:Company {name: 'Microsoft', ticker: 'MSFT', cik: '0000789019'})"
    )
    sample = {
        "task_type": "EXTRACT_SEC_GRAPH",
        "expected_assertions": {
            "expect_none": False,
            "allowed_ciks": ["0000320193", "0000789019"],
            "required_ciks": ["0000320193", "0000789019"],
        },
    }
    res = evaluate_torture_sample(bad_cik_triple, sample)
    assert res["functional_pass"] is False
    assert res["schema_cik_pass"] is False


def test_compute_architectural_decisions_routing() -> None:
    synthetic_records = [
        {
            "axis_id": "TT-01",
            "sub_cohort": "CYPHER_K4_E80_MIDDLE",
            "metadata": {"graph_format": "RAW_CYPHER", "hops": 4, "edge_count": 80},
            "functional_pass": False,
        },
        {
            "axis_id": "TT-01",
            "sub_cohort": "BFS_VERBALIZED_ABLATION",
            "metadata": {"graph_format": "BFS_VERBALIZED_ARROWS", "hops": 4, "edge_count": 80},
            "functional_pass": True,
        },
        {
            "axis_id": "TT-02",
            "sub_cohort": "DIAMOND_ADDITIVE_SHOCK",
            "metadata": {},
            "functional_pass": False,
        },
        {
            "axis_id": "TT-02",
            "sub_cohort": "STALE_TERMINATED_EDGE_TRAP",
            "metadata": {},
            "functional_pass": False,
        },
        {
            "axis_id": "TT-04",
            "sub_cohort": "RAW_EDGE_DROPOUT_BLINDSPOT",
            "metadata": {},
            "functional_pass": False,
        },
        {
            "axis_id": "TT-04",
            "sub_cohort": "GNN_IMPLIED_HIGH_PROB",
            "metadata": {},
            "functional_pass": True,
        },
    ]
    decisions = compute_architectural_decisions(synthetic_records)
    triggers = decisions["phase7_paradigm_triggers"]
    assert triggers["candidate_a_sparse_threshold_cascade"]["recommended"] is True
    assert triggers["candidate_c_self_supervised_link_prediction_gnn"]["recommended"] is True
    assert len(decisions["phase6_deterministic_guardrail_actions"]) >= 2

