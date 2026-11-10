"""System Integration Test (SIT) Flow 2: Material Event Shock Vector, CAR Volatility & Multi-Hop Contagion.

Academic Reference & Governance:
- ADR-00001: SIT Flow 2 Event Shock Vector & Contagion Harness
- Acemoglu, D., Ozdaglar, A., & Tahbaz-Salehi, A. (2015): "Systemic Risk and Stability in Financial Networks"
- SEC Rule 33-10514 / Post-June 2018 Inline XBRL & 8-K Regulatory Ground Truth

Subsystems Crossed:
- ingestion/market/parser.py (MarketMetricsCalculator: CAR, volatility z-score)
- ingestion/sec/ (Form 8-K item 8.01 material event disclosure)
- retrieval/shock_propagation.py (5-axis ShockVector, PropagatedShock, multi-hop BFS traversal)

Invariants Asserted:
1. Offline Isolation Invariant: Zero live external network / Finnhub API calls.
2. Monotonic Hop Decay Invariant: Shock magnitude decays strictly monotonically across hops:
   magnitude(Tier 3) < magnitude(Tier 2) < magnitude(Tier 1) <= magnitude(Origin).
3. Buffer Delay Invariant: Downstream onset date is delayed by exactly buffer_days:
   effective_onset_date == event_date + buffer_days.
4. Buffer Absorption Invariant: Buffer absorbs duration (unbuffered_duration == max(0, duration - buffer_days)).
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import pyarrow.parquet as pq

from ingestion.market.parser import MarketMetricsCalculator
from retrieval.shock_propagation import (
    PropagatedShock,
    ShockVector,
    build_shock_vector_from_metrics,
    propagate_shock_across_edge,
    traverse_multi_hop_contagion,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"


@pytest.fixture
def aapl_8k_event_fixture() -> dict:
    """Load canonical Form 8-K Item 8.01 supplier disruption disclosure."""
    filing_path = FIXTURES_DIR / "filings" / "aapl_8k_disruption.json"
    assert filing_path.is_file(), f"8-K fixture missing at {filing_path}"
    with open(filing_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def ohlcv_market_fixture() -> list[dict]:
    """Load offline 66-day daily OHLCV price series for AAPL, NVDA, and SPY benchmark."""
    parquet_path = FIXTURES_DIR / "market" / "aapl_nvda_ohlcv.parquet"
    assert parquet_path.is_file(), f"Market OHLCV parquet fixture missing at {parquet_path}"
    table = pq.read_table(parquet_path)
    return table.to_pylist()


@pytest.mark.integration
def test_flow_2_e2e_event_shock_vector_and_multi_hop_contagion(
    monkeypatch,
    aapl_8k_event_fixture: dict,
    ohlcv_market_fixture: list[dict],
):
    """Verify SIT Flow 2: Form 8-K + OHLCV -> CAR z-score -> 5-Axis Shock Vector -> Multi-Hop Attenuation."""
    # Invariant 1: Offline Isolation Guardrail (strictly no live external network/API calls)
    def fail_on_network(*args, **kwargs):
        raise RuntimeError("External network access prohibited during SIT Flow 2 offline execution!")

    monkeypatch.setattr("socket.socket.connect", fail_on_network)

    # 1. Ingest 8-K filing metadata
    filing = aapl_8k_event_fixture
    assert filing["form_type"] == "8-K"
    ticker = filing["ticker"]
    filing_meta = filing["metadata"]
    event_date = filing_meta["event_date"]

    # 2. Extract price series from committed local Parquet fixture
    stock_history = [r for r in ohlcv_market_fixture if r["ticker"] == ticker]
    benchmark_history = [r for r in ohlcv_market_fixture if r["ticker"] == "SPY"]
    assert len(stock_history) >= 60
    assert len(benchmark_history) >= 60

    # 3. Calculate Cumulative Abnormal Return (CAR) and volatility z-score
    calc = MarketMetricsCalculator()
    metrics = calc.compute_event_window_metrics(
        ticker=ticker,
        event_date_str=event_date,
        price_history=stock_history,
        benchmark_history=benchmark_history,
        window_days=2,
    )

    # Verify quantitative market grounding
    assert metrics["ticker"] == "AAPL"
    assert metrics["car_abnormal_return_pct"] < -5.0  # Significant abnormal drop
    assert metrics["volatility_zscore"] <= -3.0       # >3 sigma shock event
    assert metrics["polarity_sentiment"] == "DISRUPTIVE_SHOCK"

    # 4. Construct formal 5-Axis ShockVector
    shock_vector = build_shock_vector_from_metrics(metrics, filing_meta)
    assert shock_vector.origin_ticker == "AAPL"
    assert shock_vector.event_date == "2023-10-17"
    assert shock_vector.direction == "DISRUPTIVE_SHOCK"
    assert shock_vector.magnitude_pct == round(abs(metrics["car_abnormal_return_pct"]), 2)
    assert shock_vector.duration_days == 90
    assert shock_vector.confidence == 0.95
    assert shock_vector.channel == "SUPPLY_CHAIN"

    # 5. Define multi-hop supply network graph topology
    # FabCo -> AAPL -> Foxconn -> Consumer Retail
    # Notice: Origin shock occurs at AAPL in this scenario (or propagates from supplier)
    network_edges = [
        {
            "source": "AAPL",
            "target": "FOXCONN",
            "exposure_pct": 50.0,
            "buffer_days": 45,
        },
        {
            "source": "FOXCONN",
            "target": "RETAIL_PARTNER",
            "exposure_pct": 30.0,
            "buffer_days": 15,
        },
    ]

    # 6. Execute multi-hop propagation
    propagated_shocks = traverse_multi_hop_contagion(shock_vector, network_edges)
    assert len(propagated_shocks) == 2

    hop1 = propagated_shocks[0]
    hop2 = propagated_shocks[1]

    # Invariant 2: Monotonic Hop Decay Invariant
    # magnitude(Tier 2) < magnitude(Tier 1) < magnitude(Origin)
    assert hop2.attenuated_magnitude_pct < hop1.attenuated_magnitude_pct < shock_vector.magnitude_pct

    # Hop 1 Invariants
    assert hop1.target_ticker == "FOXCONN"
    assert hop1.parent_ticker == "AAPL"
    assert hop1.hop_distance == 1
    # Invariant 3: Buffer Delay Invariant (2023-10-17 + 45 days = 2023-12-01)
    assert hop1.effective_onset_date == "2023-12-01"
    # Invariant 4: Buffer Absorption Invariant (90 - 45 = 45 days)
    assert hop1.unbuffered_duration_days == 45
    assert hop1.absorption_ratio == 0.50
    # Expected magnitude: magnitude_pct * 0.50 * 0.50 = 0.25 * magnitude_pct
    expected_hop1_mag = round(shock_vector.magnitude_pct * 0.50 * 0.50, 2)
    assert hop1.attenuated_magnitude_pct == expected_hop1_mag

    # Hop 2 Invariants
    assert hop2.target_ticker == "RETAIL_PARTNER"
    assert hop2.parent_ticker == "FOXCONN"
    assert hop2.hop_distance == 2
    # Invariant 3: Cumulative Buffer Delay (2023-12-01 + 15 days = 2023-12-16)
    assert hop2.effective_onset_date == "2023-12-16"
    # Invariant 4: Buffer Absorption (45 - 15 = 30 days)
    assert hop2.unbuffered_duration_days == 30
    assert hop2.absorption_ratio == pytest.approx(round(30 / 45, 2), abs=0.02)
    assert hop2.attenuated_magnitude_pct < hop1.attenuated_magnitude_pct


@pytest.mark.integration
def test_flow_2_buffer_oversupply_residual_friction():
    """Verify that when inventory buffer exceeds shock duration, magnitude collapses to residual 0.05 friction."""
    shock = ShockVector(
        origin_ticker="SUPPLIER",
        event_date="2023-10-17",
        direction="DISRUPTIVE_SHOCK",
        magnitude_pct=12.0,
        duration_days=30,  # 30-day disruption
    )
    # Downstream customer has 60-day buffer (buffer > duration)
    edge = {
        "source": "SUPPLIER",
        "target": "CUSTOMER",
        "exposure_pct": 50.0,
        "buffer_days": 60,
    }

    prop = propagate_shock_across_edge(shock, edge)

    assert prop.unbuffered_duration_days == 0
    assert prop.absorption_ratio == 0.0
    # Residual friction factor is 0.05: 12.0 * 0.50 * 0.05 = 0.30%
    assert prop.attenuated_magnitude_pct == 0.30
    assert prop.effective_onset_date == "2023-12-16"
