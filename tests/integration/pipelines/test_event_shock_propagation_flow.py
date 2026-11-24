"""System Integration Test (SIT) Flow 2: Material Event Shock Vector, CAR Volatility & Multi-Hop Contagion.

Academic Reference & Governance:
- ADR-00001: SIT Flow 2 Event Shock Vector & Contagion Harness
- ADR-00006 / ADR-00008 / ADR-00034: Consolidated onto `contagion.cascade.SparseNetworkCascadeEngine`
- Acemoglu, D., Ozdaglar, A., & Tahbaz-Salehi, A. (2015): "Systemic Risk and Stability in Financial Networks"
- SEC Rule 33-10514 / Post-June 2018 Inline XBRL & 8-K Regulatory Ground Truth

Subsystems Crossed:
- ingestion/market/parser.py (MarketMetricsCalculator: CAR, volatility z-score)
- ingestion/sec/ (Form 8-K item 8.01 material event disclosure)
- contagion/cascade.py (`SparseNetworkCascadeEngine`, `CascadeGraph`, `CascadeEdge`, `ConvergentShockSummary`)

Invariants Asserted:
1. Offline Isolation Invariant: Zero live external network / Finnhub API calls.
2. Monotonic Hop Decay Invariant: Downstream capacity shock attenuates strictly across hops:
   peak_shock(Tier 2) < peak_shock(Tier 1) <= origin_shock.
3. Buffer Delay Invariant: Downstream onset date is delayed by inventory buffer days.
4. Buffer Absorption & Residual Friction Invariant: Full buffer absorption yields 0 unbuffered
   downtime and activates the 5% residual friction floor (`TOLERATED_BUFFERED`).
"""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import pytest
import pyarrow.parquet as pq

from contagion import (
    CascadeEdge,
    CascadeGraph,
    ConvergentShockSummary,
    SparseNetworkCascadeEngine,
)
from ingestion.market.parser import MarketMetricsCalculator

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
    """Verify SIT Flow 2: Form 8-K + OHLCV -> CAR z-score -> SparseNetworkCascadeEngine multi-hop decay."""
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
    duration_days = float(filing_meta.get("disruption_duration_days", 90.0))

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

    assert metrics["ticker"] == "AAPL"
    assert metrics["car_abnormal_return_pct"] < -5.0
    assert metrics["volatility_zscore"] <= -3.0
    assert metrics["polarity_sentiment"] == "DISRUPTIVE_SHOCK"

    # 4. Build multi-hop supply network graph and propagate via SparseNetworkCascadeEngine (ADR-00034)
    graph = CascadeGraph()
    graph.add_edge(CascadeEdge(source="AAPL", target="FOXCONN", share=0.50, buffer_days=45.0, input_group="assembly"))
    graph.add_edge(CascadeEdge(source="FOXCONN", target="RETAIL_PARTNER", share=0.30, buffer_days=0.0, input_group="distribution"))

    engine = SparseNetworkCascadeEngine(disruption_threshold_pct=25.0)
    origin_shock_pct = 100.0
    results = engine.propagate(
        graph=graph,
        initial_shocks={"AAPL": origin_shock_pct},
        initial_durations={"AAPL": duration_days},
        initial_onsets={"AAPL": 0.0},
    )

    hop1: ConvergentShockSummary = results["FOXCONN"]
    hop2: ConvergentShockSummary = results["RETAIL_PARTNER"]

    # Invariant 2: Monotonic Hop Decay Invariant
    assert hop2.peak_shock_pct < hop1.peak_shock_pct < origin_shock_pct
    assert hop1.peak_shock_pct == pytest.approx(50.0, abs=1e-3)
    assert hop2.peak_shock_pct == pytest.approx(15.0, abs=1e-3)

    # Invariant 3: Buffer Delay Invariant (2023-10-17 + 45 days = 2023-12-01)
    base_date = datetime.strptime(event_date, "%Y-%m-%d").date()
    assert hop1.onset_day == pytest.approx(45.0, abs=1e-3)
    assert (base_date + timedelta(days=int(hop1.onset_day))).isoformat() == "2023-12-01"

    # Invariant 4: Buffer Absorption Invariant (90 - 45 = 45 days unbuffered downtime)
    assert hop1.downtime_days == pytest.approx(45.0, abs=1e-3)
    assert hop1.verdict == "DISRUPTIVE_SHOCK"
    assert hop2.onset_day == pytest.approx(45.0, abs=1e-3)
    assert hop2.downtime_days == pytest.approx(45.0, abs=1e-3)
    assert hop2.verdict == "CONTAINED_MINIMAL_CONTAGION"


@pytest.mark.integration
def test_flow_2_buffer_oversupply_residual_friction():
    """Verify that when inventory buffer exceeds shock duration, SparseNetworkCascadeEngine applies 5% friction floor."""
    graph = CascadeGraph()
    graph.add_edge(CascadeEdge(source="SUPPLIER", target="CUSTOMER", share=0.50, buffer_days=60.0, input_group="components"))

    engine = SparseNetworkCascadeEngine(disruption_threshold_pct=25.0)
    results = engine.propagate(
        graph=graph,
        initial_shocks={"SUPPLIER": 100.0},
        initial_durations={"SUPPLIER": 30.0},
        initial_onsets={"SUPPLIER": 0.0},
    )

    prop = results["CUSTOMER"]
    assert prop.downtime_days == 0.0
    assert prop.is_friction_floor is True
    assert prop.peak_shock_pct == pytest.approx(5.0, abs=1e-3)
    assert prop.verdict == "TOLERATED_BUFFERED"


@pytest.mark.integration
def test_flow_2_sparse_network_cascade_engine_modernization():
    """Verify SIT Flow 2 modernized under ADR-00008 using SparseNetworkCascadeEngine."""
    graph = CascadeGraph()
    graph.add_edge(CascadeEdge(source="AAPL", target="FOXCONN", share=0.50, buffer_days=45.0, input_group="assembly"))
    graph.add_edge(CascadeEdge(source="PEGA", target="FOXCONN", share=0.30, buffer_days=15.0, input_group="assembly"))
    graph.add_edge(CascadeEdge(source="FOXCONN", target="RETAIL_PARTNER", share=0.30, buffer_days=0.0, input_group="default"))

    engine = SparseNetworkCascadeEngine(disruption_threshold_pct=25.0)

    results = engine.propagate(
        graph,
        initial_shocks={"AAPL": 1.0, "PEGA": 0.50},
        initial_durations={"AAPL": 90.0, "PEGA": 30.0},
        initial_onsets={"AAPL": 0.0, "PEGA": 0.0},
    )

    fox = results["FOXCONN"]
    assert fox.peak_shock_pct == pytest.approx(50.0, abs=1e-3)
    assert fox.onset_day == pytest.approx(45.0, abs=1e-3)
    assert fox.downtime_days == pytest.approx(45.0, abs=1e-3)
    assert fox.lost_capacity_days == pytest.approx(22.5, abs=1e-3)
    assert fox.verdict == "DISRUPTIVE_SHOCK"

    ret = results["RETAIL_PARTNER"]
    assert ret.peak_shock_pct == pytest.approx(15.0, abs=1e-3)
    assert ret.onset_day == pytest.approx(45.0, abs=1e-3)
    assert ret.downtime_days == pytest.approx(45.0, abs=1e-3)
    assert ret.verdict == "CONTAINED_MINIMAL_CONTAGION"
