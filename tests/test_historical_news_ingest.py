# tests/test_historical_news_ingest.py
# -*- coding: utf-8 -*-
"""Unit tests for tools/ingest_historical_news.py."""

from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from tools.ingest_historical_news import HistoricalNewsIngestor


@pytest.fixture
def mock_universe_mgr():
    mgr = MagicMock()
    # Mock TSLA, AAPL as active constituents, FAKE as non-constituent
    mgr.is_constituent.side_effect = lambda ticker, target_date=None: ticker in {"AAPL", "TSLA", "NVDA", "MSFT"}
    return mgr


def test_process_news_record_valid(mock_universe_mgr, tmp_path: Path):
    ingestor = HistoricalNewsIngestor(output_dir=tmp_path, universe_mgr=mock_universe_mgr)

    raw_rec = {
        "title": "Apple Signs Long Term Foundry Agreement with TSMC",
        "text": (
            "Apple announced today that it has expanded its strategic supply chain agreement with TSMC. "
            "Under the multi-year deal, TSMC will supply leading-edge 3nm and 2nm semiconductor wafers. "
            "CEO Tim Cook highlighted that this secures chip supplies for upcoming iPhone and Mac lineups. "
            "Analysts view this as a major positive development for both technology leaders in Cupertino and Hsinchu."
        ),
        "date": "2024-03-15",
        "ticker": "AAPL",
        "publisher": "Reuters",
        "url": "https://reuters.com/technology/apple-tsmc-deal-2024",
    }

    processed = ingestor.process_news_record(raw_rec)
    assert processed is not None
    assert processed["title"] == raw_rec["title"]
    assert "AAPL" in processed["ticker_symbols"]
    assert processed["provider"] == "Reuters"
    assert len(processed["source_hash"]) == 32


def test_process_news_record_filters_blacklisted_publisher(mock_universe_mgr, tmp_path: Path):
    ingestor = HistoricalNewsIngestor(output_dir=tmp_path, universe_mgr=mock_universe_mgr)

    raw_rec = {
        "title": "Why You Should Buy Apple Stock Now",
        "text": "Apple is a great company with fantastic revenue and dividends. " * 10,
        "date": "2024-03-15",
        "ticker": "AAPL",
        "publisher": "The Motley Fool", # Blacklisted
    }

    processed = ingestor.process_news_record(raw_rec)
    assert processed is None


def test_process_news_record_filters_non_sp500_tickers(mock_universe_mgr, tmp_path: Path):
    ingestor = HistoricalNewsIngestor(output_dir=tmp_path, universe_mgr=mock_universe_mgr)

    raw_rec = {
        "title": "SmallCap Biotech Reports Phase 1 Results",
        "text": "SmallCap Biotech reported promising early phase clinical trial data today. " * 10,
        "date": "2024-03-15",
        "ticker": "NON_SP500_TICKER",
        "publisher": "PR Newswire",
    }

    processed = ingestor.process_news_record(raw_rec)
    assert processed is None


def test_stage_records_to_parquet(mock_universe_mgr, tmp_path: Path):
    ingestor = HistoricalNewsIngestor(output_dir=tmp_path, universe_mgr=mock_universe_mgr)

    records = [
        {
            "source_hash": "hash_12345678901234567890123456789012",
            "source_url": "https://example.com/article1",
            "title": "NVIDIA Q3 Earnings",
            "raw_text": "NVIDIA posted record quarterly revenue driven by data center AI demand. " * 10,
            "published_at": "2024-11-20",
            "ticker_symbols": ["NVDA"],
            "provider": "Bloomberg",
            "category": "financial_news",
            "status": "pending",
        }
    ]

    out_file = ingestor.stage_records_to_parquet(records, batch_filename="test_news.parquet")
    assert out_file.exists()
