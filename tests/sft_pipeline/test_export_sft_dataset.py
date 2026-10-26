# tests/test_export_sft_dataset.py
# -*- coding: utf-8 -*-
"""Unit tests for tools/export_sft_dataset.py."""

import json
from pathlib import Path
import pytest
from unittest.mock import MagicMock

from sft_pipeline import (
    FullSFTDatasetBuilder,
    SFTDatasetExporter,
    SFTRecord,
    ManifoldSample,
    AnnotatedTriple,
)
from universe import SP500Constituent


@pytest.fixture
def mock_universe_mgr():
    mgr = MagicMock()
    mock_constituents = [
        SP500Constituent(ticker="AAPL", cik="0000320193", company_name="Apple Inc.", gics_sector="Information Technology"),
        SP500Constituent(ticker="TSLA", cik="0001318605", company_name="Tesla Inc.", gics_sector="Consumer Discretionary"),
    ]
    mgr.get_all_records.return_value = mock_constituents
    return mgr


def test_format_task_a_sec_graph(mock_universe_mgr, tmp_path: Path):
    exporter = SFTDatasetExporter(output_dir=tmp_path, universe_mgr=mock_universe_mgr)

    sample = ManifoldSample(
        sample_id="SEC_001",
        text_passage="Apple relies on TSMC for 3nm wafer fabrication.",
        grounded_triples=[],
        entities_present=[],
        hop_count=1,
        is_hard_negative=False,
        gics_sector="Information Technology",
        provenance="SEC_10K",
        confidence=0.98,
        difficulty_score=0.4,
    )

    triple = AnnotatedTriple(
        source_name="TSMC",
        source_type="Company",
        target_name="Apple Inc.",
        target_type="Company",
        target_ticker="AAPL",
        rel_type="SUPPLIES_TO",
        polarity="NEUTRAL_STABLE",
        materiality="CRITICAL_TIER_1",
    )

    record = exporter.format_task_a_sec_graph(sample, [triple])
    assert record.metadata["task_type"] == "EXTRACT_SEC_GRAPH"
    assert record.metadata["assigned_student"] == "QWEN_2.5_7B_UNIFIED"
    assert "<|extract_sec_graph|>" in record.prompt
    assert "(:Company" in record.target_completion
    assert '-[:SUPPLIES_TO {' in record.target_completion


def test_format_task_b_news_event_hard_negative(mock_universe_mgr, tmp_path: Path):
    exporter = SFTDatasetExporter(output_dir=tmp_path, universe_mgr=mock_universe_mgr)

    sample = ManifoldSample(
        sample_id="NEG_001",
        text_passage="Both Tesla and Apple traded lower on macroeconomic inflation concerns.",
        grounded_triples=[],
        entities_present=[],
        hop_count=0,
        is_hard_negative=True,
        gics_sector="Cross-Sector",
        provenance="NEWS",
        confidence=1.0,
        difficulty_score=0.85,
    )

    record = exporter.format_task_b_news_event(sample, [])
    assert record.metadata["task_type"] == "EXTRACT_NEWS_EVENT"
    assert record.metadata["assigned_student"] == "QWEN_2.5_7B_UNIFIED"
    assert record.target_completion == "(none)" # Hard negative target is strictly empty!


def test_format_task_c_text_to_cypher(mock_universe_mgr, tmp_path: Path):
    exporter = SFTDatasetExporter(output_dir=tmp_path, universe_mgr=mock_universe_mgr)

    record = exporter.format_task_c_text_to_cypher("Apple Inc.", "AAPL")
    assert record.metadata["task_type"] == "TEXT_TO_CYPHER"
    assert record.metadata["assigned_student"] == "QWEN_2.5_7B_UNIFIED"
    assert "<|text_to_cypher|>" in record.prompt
    assert "MATCH (s:Company)-[r:SUPPLIES_TO]->(t:Company)" in record.target_completion
    assert "t.ticker = 'AAPL'" in record.target_completion


def test_format_task_d_contagion_reasoning(mock_universe_mgr, tmp_path: Path):
    exporter = SFTDatasetExporter(output_dir=tmp_path, universe_mgr=mock_universe_mgr)

    record = exporter.format_task_d_contagion_reasoning(
        focal_company="Apple Inc.",
        focal_ticker="AAPL",
        supplier="TSMC",
        supplier_ticker="TSM",
        shock_scenario="Severe earthquake damages fabrication facilities in Hsinchu",
        impacted_customers=["Foxconn", "Pegatron"],
    )

    assert record.metadata["task_type"] == "CONTAGION_REASONING"
    assert record.metadata["assigned_student"] == "QWEN_2.5_7B_UNIFIED"
    assert record.metadata["provenance"] == "SYNTHETIC_TEMPLATE_SKELETON"
    assert "<think>" in record.target_completion
    assert "</think>" in record.target_completion
    assert "**Contagion Analysis Summary**" in record.target_completion


def test_format_task_e_portfolio_recommendation(mock_universe_mgr, tmp_path: Path):
    exporter = SFTDatasetExporter(output_dir=tmp_path, universe_mgr=mock_universe_mgr)

    record = exporter.format_task_e_portfolio_recommendation(
        focal_company="Tesla Inc.",
        focal_ticker="TSLA",
        risk_exposure="Lithium battery cell supply chokepoint",
        recommended_hedge="Overweight alternative battery chemistry developers and short high-beta EV peers",
    )

    assert record.metadata["task_type"] == "PORTFOLIO_RECOMMENDATION"
    assert record.metadata["assigned_student"] == "QWEN_2.5_7B_UNIFIED"
    assert record.metadata["provenance"] == "SYNTHETIC_TEMPLATE_SKELETON"
    assert "<think>" in record.target_completion
    assert "Strategic Portfolio Recommendation" in record.target_completion


def test_export_full_sft_splits_80_10_10(mock_universe_mgr, tmp_path: Path):
    exporter = SFTDatasetExporter(output_dir=tmp_path, universe_mgr=mock_universe_mgr)

    # 10 Extractor records, 10 Reasoner records
    extractor_recs = [
        exporter.format_task_c_text_to_cypher(f"Company {i}", f"TICK{i}")
        for i in range(10)
    ]
    reasoner_recs = [
        exporter.format_task_e_portfolio_recommendation(f"Company {i}", f"TICK{i}", "Risk", "Hedge")
        for i in range(10)
    ]

    summary = exporter.export_full_sft_splits(extractor_recs, reasoner_recs, train_ratio=0.80, val_ratio=0.10)

    assert summary["unified_splits"]["train"] == 16
    assert summary["unified_splits"]["val"] == 2
    assert summary["unified_splits"]["test"] == 2
    assert summary["task_breakdown"]["extraction_tasks"]["train"] == 8
    assert summary["task_breakdown"]["reasoning_tasks"]["train"] == 8

    assert (tmp_path / "yarnball_sft_train.jsonl").exists()
    assert (tmp_path / "yarnball_sft_val.jsonl").exists()
    assert (tmp_path / "yarnball_sft_test.jsonl").exists()
    assert (tmp_path / "extraction_tasks_train.jsonl").exists()
    assert (tmp_path / "reasoning_tasks_train.jsonl").exists()
    assert (tmp_path / "dataset_summary.json").exists()

    with open(tmp_path / "dataset_summary.json", "r", encoding="utf-8") as f:
        summary_data = json.load(f)
    assert summary_data["schema_version"] == "1.0.0"
    assert summary_data["target_model"] == "Qwen2.5-7B-Instruct (Unified Extractor & Reasoner)"

    with open(tmp_path / "yarnball_sft_train.jsonl", "r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())
    assert first_line["schema_version"] == "1.0.0"


def test_format_task_a_disambiguation(mock_universe_mgr, tmp_path: Path):
    exporter = SFTDatasetExporter(output_dir=tmp_path, universe_mgr=mock_universe_mgr)

    # Positive spin-off target
    rec_pos = exporter.format_task_a_disambiguation(
        focal_company="GE Vernova",
        focal_ticker="GEV",
        confusable_company="General Electric Company",
        confusable_ticker="GE_HIST",
        relationship_context="GE Vernova executed formal separation agreements with General Electric Company.",
        true_relation_cypher="(:Company {name: 'GE Vernova', ticker: 'GEV'})-[:SPUN_OFF_FROM]->(:Company {name: 'General Electric Company'})",
        gics_sector="Industrials",
    )
    assert rec_pos.metadata["task_type"] == "EXTRACT_SEC_GRAPH"
    assert rec_pos.metadata["is_disambiguation"] is True
    assert rec_pos.metadata["focal_entity"] == "GE Vernova"
    assert "SPUN_OFF_FROM" in rec_pos.target_completion

    # Negative target (none) when querying unrelated peer
    rec_neg = exporter.format_task_a_disambiguation(
        focal_company="GE Aerospace",
        focal_ticker="GE",
        confusable_company="GE Vernova",
        confusable_ticker="GEV",
        relationship_context="GE Vernova entered into a 15-year offshore wind turbine maintenance agreement.",
        true_relation_cypher=None,
        gics_sector="Industrials",
    )
    assert rec_neg.target_completion == "(none)"


def test_format_task_b_borderline_volatility(mock_universe_mgr, tmp_path: Path):
    exporter = SFTDatasetExporter(output_dir=tmp_path, universe_mgr=mock_universe_mgr)

    # Borderline volatility (|z| < 2.0) -> MIXED_UNCERTAIN_SIGNAL
    rec_border = exporter.format_task_b_borderline_volatility(
        company_name="Microsoft Corporation",
        ticker="MSFT",
        event_narrative="Quarterly cloud revenue beat expectations but capex expanded.",
        item_code="1.01",
        car_3d_pct=2.45,
        z_score=1.92,
        gics_sector="Information Technology",
    )
    assert rec_border.metadata["task_type"] == "EXTRACT_NEWS_EVENT"
    assert rec_border.metadata["is_borderline_volatility"] is True
    assert "MIXED_UNCERTAIN_SIGNAL" in rec_border.target_completion
    assert "BORDERLINE_SIGNIFICANT" in rec_border.target_completion
    assert "car_3d: 0.0245" in rec_border.target_completion

    # Significant volatility (|z| >= 2.0) -> EXPANDING_BULLISH / STATISTICALLY_SIGNIFICANT
    rec_sig = exporter.format_task_b_borderline_volatility(
        company_name="Microsoft Corporation",
        ticker="MSFT",
        event_narrative="Quarterly results exceeded all guidance metrics.",
        item_code="2.02",
        car_3d_pct=3.80,
        z_score=2.12,
        gics_sector="Information Technology",
    )
    assert "EXPANDING_BULLISH" in rec_sig.target_completion
    assert "STATISTICALLY_SIGNIFICANT" in rec_sig.target_completion


def test_format_task_d_distractor_contagion(mock_universe_mgr, tmp_path: Path):
    exporter = SFTDatasetExporter(output_dir=tmp_path, universe_mgr=mock_universe_mgr)

    rec = exporter.format_task_d_distractor_contagion(
        focal_company="Apple Inc.",
        focal_ticker="AAPL",
        critical_supplier="TSMC",
        critical_supplier_ticker="TSM",
        distractor_supplier="Avery Dennison Corp",
        distractor_supplier_ticker="AVY",
        shock_scenario="Warehouse fire destroys cardboard packaging inventory at Avery Dennison.",
        downstream_customer="Best Buy Co., Inc.",
        gics_sector="Information Technology",
    )

    assert rec.metadata["task_type"] == "CONTAGION_REASONING"
    assert rec.metadata["has_distractor_path"] is True
    assert "<think>" in rec.target_completion
    assert "</think>" in rec.target_completion
    assert "CONTAINED_MINIMAL_CONTAGION" in rec.target_completion
    assert "Avery Dennison" in rec.prompt
    assert "TSMC" in rec.prompt


def test_format_task_e_boundary_hedging(mock_universe_mgr, tmp_path: Path):
    exporter = SFTDatasetExporter(output_dir=tmp_path, universe_mgr=mock_universe_mgr)

    rec = exporter.format_task_e_boundary_hedging(
        focal_company="Pfizer Inc.",
        focal_ticker="PFE",
        divergent_catalyst="Phase III trial achieved primary endpoint, but regulatory rebates mandate 16% discount (|z| = -1.89)",
        z_score=-1.89,
        recommended_hedging="Out-of-the-money put spreads on XLV",
        gics_sector="Health Care",
    )

    assert rec.metadata["task_type"] == "PORTFOLIO_RECOMMENDATION"
    assert rec.metadata["is_boundary_hedging"] is True
    assert "<think>" in rec.target_completion
    assert "</think>" in rec.target_completion
    assert "Strategic Portfolio Recommendation" in rec.target_completion
    assert "Out-of-the-money put spreads on XLV" in rec.target_completion


def test_builder_phase2_generators(mock_universe_mgr, tmp_path: Path):
    builder = FullSFTDatasetBuilder(output_dir=tmp_path, universe_mgr=mock_universe_mgr, seed=42)

    disambig = builder.generate_task_a_disambiguation_records(count=15)
    assert len(disambig) == 15
    assert all(r.metadata["is_disambiguation"] for r in disambig)

    border = builder.generate_task_b_borderline_records(count=15)
    assert len(border) == 15
    assert all(r.metadata["is_borderline_volatility"] for r in border)

    distract = builder.generate_task_d_distractor_records(count=15)
    assert len(distract) == 15
    assert all(r.metadata["has_distractor_path"] for r in distract)

    hedging = builder.generate_task_e_boundary_hedging_records(count=15)
    assert len(hedging) == 15
    assert all(r.metadata["is_boundary_hedging"] for r in hedging)

