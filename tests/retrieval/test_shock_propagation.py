"""Unit tests for Step 2: Multi-Hop Contagion & Shock Propagation Engine."""

from datetime import date
import pytest

from retrieval.shock_propagation import (
    PropagatedShock,
    ShockVector,
    build_shock_vector_from_metrics,
    propagate_shock_across_edge,
    traverse_multi_hop_contagion,
)


def test_build_shock_vector_from_metrics():
    metrics = {
        "ticker": "AAPL",
        "event_date": "2023-10-17",
        "car_abnormal_return_pct": -8.42,
        "volatility_zscore": -5.3,
        "polarity_sentiment": "DISRUPTIVE_SHOCK",
    }
    meta = {
        "disruption_duration_days": 90,
        "inventory_buffer_days": 45,
        "supplier_name": "FabCo Semiconductor",
    }
    shock = build_shock_vector_from_metrics(metrics, meta)

    assert shock.origin_ticker == "AAPL"
    assert shock.event_date == "2023-10-17"
    assert shock.magnitude_pct == 8.42
    assert shock.direction == "DISRUPTIVE_SHOCK"
    assert shock.duration_days == 90
    assert shock.confidence == 0.95
    assert shock.metadata["volatility_zscore"] == -5.3


def test_propagate_shock_single_hop_decay_and_buffer_delay():
    shock = ShockVector(
        origin_ticker="FABCO",
        event_date="2023-10-17",
        direction="DISRUPTIVE_SHOCK",
        magnitude_pct=10.0,
        duration_days=90,
    )
    edge = {
        "source": "FABCO",
        "target": "AAPL",
        "exposure_pct": 50.0,
        "buffer_days": 45,
    }
    prop = propagate_shock_across_edge(shock, edge)

    assert prop.target_ticker == "AAPL"
    assert prop.parent_ticker == "FABCO"
    assert prop.hop_distance == 1
    # Buffer delay: 2023-10-17 + 45 days = 2023-12-01
    assert prop.effective_onset_date == "2023-12-01"
    # Unbuffered duration: 90 - 45 = 45 days
    assert prop.unbuffered_duration_days == 45
    assert prop.absorption_ratio == 0.50
    # Magnitude: 10.0 * 0.50 * 0.50 = 2.5%
    assert prop.attenuated_magnitude_pct == 2.50
    # Invariant: Monotonic decay
    assert prop.attenuated_magnitude_pct < shock.magnitude_pct


def test_traverse_multi_hop_contagion_monotonic_chain():
    initial_shock = ShockVector(
        origin_ticker="FABCO",
        event_date="2023-10-17",
        direction="DISRUPTIVE_SHOCK",
        magnitude_pct=10.0,
        duration_days=90,
    )
    edges = [
        {"source": "FABCO", "target": "AAPL", "exposure_pct": 50.0, "buffer_days": 45},
        {"source": "AAPL", "target": "FOXCONN", "exposure_pct": 30.0, "buffer_days": 15},
    ]

    chain = traverse_multi_hop_contagion(initial_shock, edges)

    assert len(chain) == 2
    hop1 = chain[0]
    hop2 = chain[1]

    assert hop1.target_ticker == "AAPL"
    assert hop1.hop_distance == 1
    assert hop2.target_ticker == "FOXCONN"
    assert hop2.hop_distance == 2

    # Invariant: Monotonic Hop Decay across full chain
    assert hop2.attenuated_magnitude_pct < hop1.attenuated_magnitude_pct < initial_shock.magnitude_pct
    assert hop1.attenuated_magnitude_pct == 2.50
    # Hop 2 magnitude: 2.50 * 0.30 * ((45 - 15) / 45 = 0.67) = ~0.50%
    assert hop2.attenuated_magnitude_pct == 0.50
