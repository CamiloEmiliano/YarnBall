"""
Unit tests for DataMapVisualizer (GridSpec 4-panel static and Plotly interactive HTML).
"""

from pathlib import Path
import pytest

try:
    from curation.contracts import CartographyCoordinate, CartographyRegion
    from curation.active_learning.visualizer import DataMapVisualizer
except ImportError:
    from graphrag_finance.curation.contracts import CartographyCoordinate, CartographyRegion
    from graphrag_finance.curation.active_learning.visualizer import DataMapVisualizer


@pytest.fixture
def sample_coordinates():
    """Generates a small representative coordinate set across all 3 regions."""
    coords = []
    # 5 Easy-to-Learn
    for i in range(5):
        coords.append(
            CartographyCoordinate(
                guid=f"easy_{i}",
                confidence=0.88 + i * 0.02,
                variability=0.03 + i * 0.01,
                correctness=1.0,
                forgetfulness=0,
                region=CartographyRegion.EASY_TO_LEARN,
            )
        )
    # 5 Ambiguous
    for i in range(5):
        coords.append(
            CartographyCoordinate(
                guid=f"ambig_{i}",
                confidence=0.50 + i * 0.05,
                variability=0.18 + i * 0.02,
                correctness=0.6,
                forgetfulness=1,
                region=CartographyRegion.AMBIGUOUS,
            )
        )
    # 5 Hard-to-Learn
    for i in range(5):
        coords.append(
            CartographyCoordinate(
                guid=f"hard_{i}",
                confidence=0.12 + i * 0.02,
                variability=0.04 + i * 0.01,
                correctness=0.0,
                forgetfulness=999,
                region=CartographyRegion.HARD_TO_LEARN,
            )
        )
    return coords


def test_plot_static_summary(sample_coordinates, tmp_path: Path):
    """Verify publication-grade 4-panel GridSpec static plot is generated."""
    visualizer = DataMapVisualizer()
    out_png = tmp_path / "datamap_test.png"

    result_path = visualizer.plot_static_summary(sample_coordinates, out_png)
    assert result_path.is_file()
    assert result_path.stat().st_size > 10000  # Non-trivial PNG binary size


def test_plot_static_summary_empty(tmp_path: Path):
    """Verify empty coordinate list falls back gracefully without crashing."""
    visualizer = DataMapVisualizer()
    out_png = tmp_path / "datamap_empty.png"

    result_path = visualizer.plot_static_summary([], out_png)
    assert result_path.is_file()
    assert result_path.stat().st_size > 0


def test_export_interactive_html(sample_coordinates, tmp_path: Path):
    """Verify Plotly interactive HTML export with metadata hover data."""
    visualizer = DataMapVisualizer()
    out_html = tmp_path / "datamap_interactive.html"

    metadata_lookup = {
        "easy_0": {"task": "Task A: SEC Graph DSL", "cik": "0000320193", "target_completion": "(AAPL)-[:SUPPLIES_TO]->(TSM)"},
        "ambig_0": {"task": "Task D: Contagion", "cik": "0000789019", "target_completion": "<think>Multi-hop contagion</think>"},
        "hard_0": {"task": "Task C: Text-to-Cypher", "cik": "0001045810", "target_completion": "MATCH (n:Company) RETURN n"},
    }

    result_path = visualizer.export_interactive_html(sample_coordinates, out_html, metadata_lookup)
    assert result_path.is_file()
    content = result_path.read_text(encoding="utf-8")
    assert "plotly" in content.lower()
    assert "easy_0" in content
    assert "YarnBall Interactive Data Map" in content
