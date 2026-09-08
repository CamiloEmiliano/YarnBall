# tests/test_eval_harness.py
# -*- coding: utf-8 -*-
"""Unit tests for the EvaluationHarness module."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from rag.eval_harness import EvaluationHarness, BENCHMARK_QUERIES
from graph.snapshot_manager import SnapshotManager


def test_benchmark_queries_count_and_schema():
    """Verify that there are exactly 25 golden benchmark queries with required fields."""
    assert len(BENCHMARK_QUERIES) == 25

    categories = set()
    for q in BENCHMARK_QUERIES:
        assert "id" in q
        assert "category" in q
        assert "question" in q
        assert "cypher" in q
        assert "target_entities" in q
        assert len(q["cypher"].strip()) > 0
        assert len(q["target_entities"]) > 0
        categories.add(q["category"])

    # Ensure all 5 archetypes are covered
    expected_categories = {
        "Supply Chain",
        "Competitive Dynamics",
        "Co-Investment",
        "Executive Leadership",
        "Regulatory Impact",
    }
    assert categories == expected_categories


def test_compute_ab_diff_metrics():
    """Test calculation of Path Recovery Gain, compression, and NER gains."""
    raw_eval = {
        "snapshot_id": "G_raw",
        "node_count": 100,
        "edge_count": 200,
        "total_queries": 25,
        "successful_queries": 25,
        "non_empty_returns": 15,
        "total_paths_found": 30,
        "query_execution_rate_pct": 100.0,
        "non_empty_return_rate_pct": 60.0,
        "query_results": [],
    }

    resolved_eval = {
        "snapshot_id": "G_resolved",
        "node_count": 70,  # 30% compression
        "edge_count": 160, # 20% deduplication
        "total_queries": 25,
        "successful_queries": 25,
        "non_empty_returns": 22,
        "total_paths_found": 60, # 100% path recovery gain
        "query_execution_rate_pct": 100.0,
        "non_empty_return_rate_pct": 88.0,
        "query_results": [],
    }

    ab_diff = EvaluationHarness.compute_ab_diff(raw_eval, resolved_eval)
    m = ab_diff["metrics"]

    assert m["path_recovery_gain_pct"] == 100.0
    assert m["entity_compression_pct"] == 30.0
    assert m["edge_deduplication_pct"] == 20.0
    assert m["ner_absolute_gain_pct"] == 28.0


def test_generate_markdown_report_formatting(tmp_path):
    """Test generating a markdown scorecard."""
    harness = EvaluationHarness()
    raw_eval = {
        "snapshot_id": "G_raw",
        "node_count": 100,
        "edge_count": 200,
        "total_paths_found": 20,
        "query_execution_rate_pct": 100.0,
        "non_empty_return_rate_pct": 50.0,
        "query_results": [{"id": q["id"], "paths_count": 1} for q in BENCHMARK_QUERIES],
    }
    resolved_eval = {
        "snapshot_id": "G_resolved",
        "node_count": 75,
        "edge_count": 150,
        "total_paths_found": 40,
        "query_execution_rate_pct": 100.0,
        "non_empty_return_rate_pct": 80.0,
        "query_results": [{"id": q["id"], "paths_count": 2} for q in BENCHMARK_QUERIES],
    }

    ab_diff = EvaluationHarness.compute_ab_diff(raw_eval, resolved_eval)
    report_file = tmp_path / "test_report.md"
    content = harness.generate_markdown_report(ab_diff, output_path=str(report_file))

    assert report_file.exists()
    assert "# YarnBall GraphRAG A/B Evaluation Report" in content
    assert "Path Recovery" in content
    assert "SC-01" in content
    assert "RA-05" in content


def test_run_ab_benchmark_dry_run(tmp_path):
    """Test end-to-end dry-run benchmark execution."""
    mgr = SnapshotManager(base_dir=tmp_path / "snapshots")
    harness = EvaluationHarness(snapshot_manager=mgr)

    report_path = tmp_path / "eval_report.md"
    summary = harness.run_ab_benchmark(
        raw_snapshot_id="G_raw",
        resolved_snapshot_id="G_resolved",
        output_report_path=str(report_path),
        dry_run=True,
    )

    assert "metrics" in summary
    assert summary["metrics"]["path_recovery_gain_pct"] > 0
    assert report_path.exists()
