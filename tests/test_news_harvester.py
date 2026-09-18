# tests/test_news_harvester.py
# -*- coding: utf-8 -*-
"""Unit tests for ingest/news_harvester.py."""

from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

from ingest.news_harvester import MultiSourceNewsHarvester


@pytest.fixture
def mock_universe_mgr():
    mgr = MagicMock()
    mgr.is_constituent.side_effect = lambda ticker, target_date=None: ticker in {"AAPL", "NVDA", "MSFT"}
    return mgr


@patch("yfinance.Ticker")
def test_harvest_yfinance_news(mock_yf_ticker, mock_universe_mgr):
    # Mock yfinance Ticker news response
    mock_ticker_instance = MagicMock()
    mock_ticker_instance.news = [
        {
            "title": "NVIDIA Unveils Next-Gen Blackwell Architecture",
            "publisher": "Reuters",
            "link": "https://reuters.com/technology/nvidia-blackwell-2024",
            "providerPublishTime": 1710500000,
            "relatedTickers": ["NVDA", "TSM"],
        },
        {
            "title": "1 Stock to Buy Right Now",
            "publisher": "The Motley Fool", # Blacklisted
            "link": "https://fool.com/buy-now",
            "providerPublishTime": 1710500000,
        },
    ]
    mock_yf_ticker.return_value = mock_ticker_instance

    harvester = MultiSourceNewsHarvester(universe_mgr=mock_universe_mgr)
    results = harvester.harvest_yfinance_news("NVDA")

    # Only Reuters article should survive; Motley Fool must be dropped
    assert len(results) == 1
    assert results[0]["title"] == "NVIDIA Unveils Next-Gen Blackwell Architecture"
    assert results[0]["provider"] == "Reuters"
    assert "NVDA" in results[0]["ticker_symbols"]


@patch("httpx.Client.get")
def test_harvest_google_news_rss(mock_get, mock_universe_mgr):
    sample_rss_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
        <channel>
            <title>Google News</title>
            <item>
                <title>Apple and Broadcom Announce Multi-Billion Dollar Agreement</title>
                <link>https://news.google.com/articles/CAIiE12345</link>
                <pubDate>Mon, 18 Mar 2024 14:30:00 GMT</pubDate>
                <source url="https://bloomberg.com">Bloomberg</source>
            </item>
            <item>
                <title>Top Stocks to Watch Today</title>
                <link>https://news.google.com/articles/CAIiE99999</link>
                <pubDate>Mon, 18 Mar 2024 14:30:00 GMT</pubDate>
                <source url="https://seekingalpha.com">Seeking Alpha</source>
            </item>
        </channel>
    </rss>
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = sample_rss_xml
    mock_get.return_value = mock_resp

    harvester = MultiSourceNewsHarvester(universe_mgr=mock_universe_mgr)
    results = harvester.harvest_google_news_rss("Apple Broadcom deal", ticker="AAPL")

    # Only Bloomberg article survives; Seeking Alpha must be dropped
    assert len(results) == 1
    assert "Broadcom" in results[0]["title"]
    assert results[0]["provider"] == "Bloomberg"
    assert results[0]["ticker_symbols"] == ["AAPL"]
