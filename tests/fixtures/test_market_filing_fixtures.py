"""Verification test for Flow 2 static micro-fixtures."""

import json
from pathlib import Path
import pyarrow.parquet as pq

from ingestion.market.parser import MarketMetricsCalculator

FIXTURES_DIR = Path(__file__).resolve().parent

def test_aapl_8k_disruption_fixture_integrity():
    filing_path = FIXTURES_DIR / "filings" / "aapl_8k_disruption.json"
    assert filing_path.is_file(), f"Filing fixture missing at {filing_path}"

    with open(filing_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["ticker"] == "AAPL"
    assert data["form_type"] == "8-K"
    assert "FabCo Semiconductor" in data["content"]
    assert data["metadata"]["inventory_buffer_days"] == 45
    assert data["metadata"]["disruption_duration_days"] == 90
    assert data["metadata"]["materiality_tier"] == "CRITICAL_TIER_1"


def test_ohlcv_parquet_fixture_integrity_and_shock_metrics():
    parquet_path = FIXTURES_DIR / "market" / "aapl_nvda_ohlcv.parquet"
    assert parquet_path.is_file(), f"Market fixture missing at {parquet_path}"

    table = pq.read_table(parquet_path)
    records = table.to_pylist()

    tickers = {r["ticker"] for r in records}
    assert {"AAPL", "NVDA", "SPY"}.issubset(tickers)

    aapl_history = [r for r in records if r["ticker"] == "AAPL"]
    spy_history = [r for r in records if r["ticker"] == "SPY"]
    assert len(aapl_history) >= 60

    calc = MarketMetricsCalculator()
    metrics = calc.compute_event_window_metrics(
        ticker="AAPL",
        event_date_str="2023-10-17",
        price_history=aapl_history,
        benchmark_history=spy_history,
        window_days=2,
    )

    # Invariant: Must detect a sharp negative shock with z-score <= -3.0
    assert metrics["car_abnormal_return_pct"] < -5.0
    assert metrics["volatility_zscore"] <= -3.0
    assert metrics["polarity_sentiment"] == "DISRUPTIVE_SHOCK"

