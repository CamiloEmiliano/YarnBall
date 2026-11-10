"""Multi-Hop Contagion & Event Shock Propagation Engine.

Academic Reference:
- ADR-00001: SIT Flow 2 Event Shock Vector & Contagion
- Acemoglu, D., Ozdaglar, A., & Tahbaz-Salehi, A. (2015): "Systemic Risk and Stability in Financial Networks"

Calculates:
1. 5-Axis Quantitative Shock Vectors from CAR z-scores and regulatory disclosures.
2. Geometric hop-by-hop attenuation across supply-chain edges.
3. Inventory buffer absorption and temporal shock onset delays.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
import logging
import math
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

logger = logging.getLogger("retrieval.shock_propagation")


class ShockVector(BaseModel):
    """5-Axis Quantitative Shock Vector representing an initial material event disruption."""

    origin_ticker: str
    event_date: str
    direction: str  # DISRUPTIVE_SHOCK, CONTRACTING_BEARISH, EXPANDING_BULLISH, NEUTRAL_STABLE
    magnitude_pct: float  # Absolute CAR or price shock in percentage (e.g. 8.42)
    duration_days: int  # Estimated disruption duration in calendar days (e.g. 90)
    confidence: float = 0.95
    channel: str = "SUPPLY_CHAIN"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PropagatedShock(BaseModel):
    """Downstream shock state propagated across a network edge."""

    target_ticker: str
    parent_ticker: str
    hop_distance: int
    direction: str
    attenuated_magnitude_pct: float
    effective_onset_date: str
    unbuffered_duration_days: int
    absorption_ratio: float
    exposure_pct: float
    buffer_days: int


def build_shock_vector_from_metrics(
    metrics: Dict[str, Any],
    filing_metadata: Optional[Dict[str, Any]] = None,
) -> ShockVector:
    """Construct a formal 5-Axis ShockVector from MarketMetricsCalculator output and 8-K metadata."""
    meta = filing_metadata or {}
    ticker = metrics.get("ticker") or meta.get("ticker") or "UNKNOWN"
    event_date = metrics.get("event_date") or meta.get("event_date") or date.today().isoformat()
    polarity = metrics.get("polarity_sentiment") or "CONTRACTING_BEARISH"
    raw_car = abs(float(metrics.get("car_abnormal_return_pct", 5.0)))
    duration = int(meta.get("disruption_duration_days", 90))

    return ShockVector(
        origin_ticker=ticker.strip().upper(),
        event_date=event_date,
        direction=polarity,
        magnitude_pct=round(raw_car, 2),
        duration_days=duration,
        confidence=0.95 if metrics.get("volatility_zscore", 0.0) <= -2.0 else 0.80,
        channel="SUPPLY_CHAIN",
        metadata={
            "volatility_zscore": metrics.get("volatility_zscore", 0.0),
            "supplier_name": meta.get("supplier_name"),
            "inventory_buffer_days": meta.get("inventory_buffer_days", 0),
        },
    )


def propagate_shock_across_edge(
    shock: Union[ShockVector, PropagatedShock],
    edge: Dict[str, Any],
) -> PropagatedShock:
    """Propagate a shock across a single directed relationship edge with buffer delay and exposure decay.

    Edge format:
    {
        "source": "AAPL",
        "target": "FOXCONN",
        "exposure_pct": 40.0,       # Downstream firm's capacity / revenue exposure to upstream source
        "buffer_days": 45,          # Inventory buffer protecting downstream firm
    }
    """
    target_ticker = str(edge["target"]).strip().upper()
    exposure_pct = float(edge.get("exposure_pct", 50.0))
    buffer_days = int(edge.get("buffer_days", 0))

    if isinstance(shock, ShockVector):
        parent_ticker = shock.origin_ticker
        incoming_magnitude = shock.magnitude_pct
        incoming_duration = shock.duration_days
        base_onset_date_str = shock.event_date
        hop_distance = 1
        direction = shock.direction
    else:
        parent_ticker = shock.target_ticker
        incoming_magnitude = shock.attenuated_magnitude_pct
        incoming_duration = shock.unbuffered_duration_days
        base_onset_date_str = shock.effective_onset_date
        hop_distance = shock.hop_distance + 1
        direction = shock.direction

    # 1. Temporal Delay Invariant: Buffer delays shock onset
    base_date = datetime.strptime(base_onset_date_str, "%Y-%m-%d").date()
    effective_onset_date = (base_date + timedelta(days=buffer_days)).isoformat()

    # 2. Buffer Absorption & Duration Reduction
    if incoming_duration > 0:
        absorption_ratio = max(0.0, 1.0 - (buffer_days / incoming_duration))
        unbuffered_duration = max(0, incoming_duration - buffer_days)
    else:
        absorption_ratio = 0.0
        unbuffered_duration = 0

    # 3. Monotonic Decay Invariant: Attenuate magnitude by exposure and unbuffered ratio
    # If buffer completely covers duration, residual friction factor is 0.05
    effective_attenuation = absorption_ratio if absorption_ratio > 0.0 else 0.05
    attenuated_magnitude = round(
        incoming_magnitude * (exposure_pct / 100.0) * effective_attenuation,
        2,
    )

    return PropagatedShock(
        target_ticker=target_ticker,
        parent_ticker=parent_ticker,
        hop_distance=hop_distance,
        direction=direction,
        attenuated_magnitude_pct=attenuated_magnitude,
        effective_onset_date=effective_onset_date,
        unbuffered_duration_days=unbuffered_duration,
        absorption_ratio=round(absorption_ratio, 2),
        exposure_pct=exposure_pct,
        buffer_days=buffer_days,
    )


def traverse_multi_hop_contagion(
    initial_shock: ShockVector,
    edges: List[Dict[str, Any]],
) -> List[PropagatedShock]:
    """Execute BFS multi-hop contagion traversal across a directed graph edge list."""
    results: List[PropagatedShock] = []
    current_shocks: Dict[str, Union[ShockVector, PropagatedShock]] = {
        initial_shock.origin_ticker: initial_shock
    }

    # Group edges by source
    edges_by_source: Dict[str, List[Dict[str, Any]]] = {}
    for e in edges:
        src = str(e["source"]).strip().upper()
        edges_by_source.setdefault(src, []).append(e)

    visited: set[str] = {initial_shock.origin_ticker}
    queue: List[str] = [initial_shock.origin_ticker]

    while queue:
        curr_node = queue.pop(0)
        curr_shock = current_shocks[curr_node]

        for edge in edges_by_source.get(curr_node, []):
            tgt = str(edge["target"]).strip().upper()
            if tgt in visited:
                continue

            propagated = propagate_shock_across_edge(curr_shock, edge)
            results.append(propagated)
            current_shocks[tgt] = propagated
            visited.add(tgt)
            queue.append(tgt)

    return results
