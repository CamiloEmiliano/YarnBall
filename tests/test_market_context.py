# tests/test_market_context.py
# -*- coding: utf-8 -*-
"""Unit tests for tools/fetch_market_context.py and tools/ingest_transcripts.py."""

from pathlib import Path
import pytest
from unittest.mock import MagicMock

from tools.fetch_market_context import MarketContextIntegrator
from tools.ingest_transcripts import EarningsTranscriptIngestor


@pytest.fixture
def mock_universe_mgr():
    mgr = MagicMock()
    mgr.is_constituent.side_effect = lambda ticker, target_date=None: ticker in {"AAPL", "MSFT", "NVDA"}
    return mgr


def test_transcript_parsing_prepared_and_qa(mock_universe_mgr, tmp_path: Path):
    ingestor = EarningsTranscriptIngestor(output_dir=tmp_path, universe_mgr=mock_universe_mgr)

    raw_text = (
        "Good morning. Today we report quarterly earnings with revenue of $50 billion. "
        "We are expanding our partnership with OpenAI. "
        "\n\nQuestion and Answer Session\n"
        "Brad Sills -- Bank of America: Can you comment on Azure AI revenue growth?\n"
        "Satya Nadella -- Chief Executive Officer: Azure AI grew over 30% driven by Enterprise LLM demand.\n"
        "Amy Hood -- Chief Financial Officer: Operating margins expanded 200 bps."
    )

    parsed = ingestor.parse_transcript_text(raw_text)
    assert "revenue of $50 billion" in parsed["prepared_remarks"]
    assert len(parsed["qa_dialogues"]) == 3
    assert "Satya Nadella" in parsed["executives"]
    assert "Amy Hood" in parsed["executives"]
    assert "Brad Sills" in parsed["analysts"]


def test_compute_event_window_metrics(mock_universe_mgr, tmp_path: Path):
    integrator = MarketContextIntegrator(output_dir=tmp_path, universe_mgr=mock_universe_mgr)

    price_history = [
        {"date": "2024-01-10", "close": 100.0},
        {"date": "2024-01-11", "close": 102.0},
        {"date": "2024-01-12", "close": 110.0}, # Event date: big jump (+10%)
        {"date": "2024-01-15", "close": 112.0},
        {"date": "2024-01-16", "close": 114.0},
    ]

    benchmark_history = [
        {"date": "2024-01-10", "close": 500.0},
        {"date": "2024-01-11", "close": 501.0},
        {"date": "2024-01-12", "close": 502.0},
        {"date": "2024-01-15", "close": 503.0},
        {"date": "2024-01-16", "close": 504.0}, # Benchmark only moved ~0.8%
    ]

    metrics = integrator.compute_event_window_metrics(
        ticker="NVDA",
        event_date_str="2024-01-12",
        price_history=price_history,
        benchmark_history=benchmark_history,
        window_days=2,
    )

    assert metrics["ticker"] == "NVDA"
    assert metrics["window_truncated"] is False
    assert metrics["event_return_pct"] > 0
    assert metrics["car_abnormal_return_pct"] > 5.0
    assert metrics["volatility_zscore"] > 0
    assert metrics["polarity_sentiment"] == "EXPANDING_BULLISH"

    # Test truncated window
    trunc_metrics = integrator.compute_event_window_metrics(
        ticker="NVDA",
        event_date_str="2024-01-10",
        price_history=price_history,
        window_days=2,
    )
    assert trunc_metrics["window_truncated"] is True

    # Test parquet serialization
    p_path = integrator.stage_market_context_to_parquet([metrics, trunc_metrics])
    assert p_path.exists()
    import pyarrow.parquet as pq
    read_table = pq.read_table(p_path)
    assert "window_truncated" in read_table.column_names
    assert "volatility_zscore" in read_table.column_names
    assert len(read_table) == 2
