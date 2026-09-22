# tests/test_export_sft_dataset.py
# -*- coding: utf-8 -*-
"""Unit tests for tools/export_sft_dataset.py."""

import json
from pathlib import Path
import pytest
from unittest.mock import MagicMock

from tools.export_sft_dataset import SFTDatasetExporter, SFTRecord
from tools.sft_manifold_sampler import ManifoldSample
from tools.sft_taxonomy_annotator import AnnotatedTriple
from tools.sp500_universe import SP500Constituent


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
    assert record.metadata["assigned_student"] == "QWEN_2.5_3B_EXTRACTOR"
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
    assert record.target_completion == "(none)" # Hard negative target is strictly empty!


def test_format_task_c_text_to_cypher(mock_universe_mgr, tmp_path: Path):
    exporter = SFTDatasetExporter(output_dir=tmp_path, universe_mgr=mock_universe_mgr)

    record = exporter.format_task_c_text_to_cypher("Apple Inc.", "AAPL")
    assert record.metadata["task_type"] == "TEXT_TO_CYPHER"
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
    assert record.metadata["assigned_student"] == "QWEN_3_8B_REASONER"
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
    assert record.metadata["assigned_student"] == "QWEN_3_8B_REASONER"
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

    assert summary["extractor_3b"]["train"] == 8
    assert summary["extractor_3b"]["val"] == 1
    assert summary["extractor_3b"]["test"] == 1

    assert (tmp_path / "extractor_3b_train.jsonl").exists()
    assert (tmp_path / "extractor_3b_val.jsonl").exists()
    assert (tmp_path / "extractor_3b_test.jsonl").exists()

    assert (tmp_path / "reasoner_8b_train.jsonl").exists()
    assert (tmp_path / "dataset_summary.json").exists()
