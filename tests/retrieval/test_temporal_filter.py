"""Unit tests for Stage 6.3 / ADR-00033: Point-in-Time Temporal Query Filter."""

from datetime import date

import pytest

from retrieval.temporal import (
    DEFAULT_TEMPORAL_FILTER,
    PiTContext,
    TemporalQueryFilter,
    inject_active_temporal_filter,
)


@pytest.fixture
def filter_engine() -> TemporalQueryFilter:
    return TemporalQueryFilter(default_as_of_date_param="$as_of_date")


@pytest.fixture
def historical_ctx() -> PiTContext:
    return PiTContext(as_of=date(2024, 3, 1), mode="historical")


@pytest.fixture
def live_ctx() -> PiTContext:
    return PiTContext(as_of=date(2024, 3, 1), mode="live")


def test_inject_historical_no_where(filter_engine: TemporalQueryFilter, historical_ctx: PiTContext):
    cypher = (
        "MATCH (s:Company)-[r:SUPPLIES_TO]->(c:Company {ticker: 'AAPL'}) "
        "RETURN s.name, s.ticker"
    )
    result = filter_engine.inject_active_coalescence(cypher, ctx=historical_ctx)
    assert "ACTIVE_CURRENT" not in result
    assert "r.valid_from IS NOT NULL AND r.valid_from <= '2024-03-01'" in result
    assert "(r.valid_to IS NULL OR r.valid_to > '2024-03-01')" in result
    assert "WHERE" in result
    assert result.index("WHERE") < result.index("RETURN")


def test_inject_live_preserves_status_coalescence(filter_engine: TemporalQueryFilter, live_ctx: PiTContext):
    cypher = (
        "MATCH (s:Company)-[r:SUPPLIES_TO]->(c:Company) "
        "WHERE c.ticker = 'AAPL' "
        "RETURN s.name"
    )
    result = filter_engine.inject_active_coalescence(cypher, ctx=live_ctx)
    assert "coalesce(r.status, 'ACTIVE_CURRENT') = 'ACTIVE_CURRENT'" in result
    assert "r.valid_from IS NOT NULL AND r.valid_from <= '2024-03-01'" in result
    assert "AND" in result
    assert "c.ticker = 'AAPL'" in result


def test_inject_active_coalescence_anonymous_rel(filter_engine: TemporalQueryFilter, historical_ctx: PiTContext):
    cypher = (
        "MATCH (s:Company)-[:SUPPLIES_TO]->(c:Company {ticker: 'NVDA'}) "
        "RETURN s.name"
    )
    result = filter_engine.inject_active_coalescence(cypher, ctx=historical_ctx, default_var="r")
    assert "[r:SUPPLIES_TO]" in result
    assert "r.valid_from IS NOT NULL AND r.valid_from <= '2024-03-01'" in result


def test_inject_varlen_rel_uses_all_predicate(filter_engine: TemporalQueryFilter, historical_ctx: PiTContext):
    cypher = "MATCH (s:Company)-[r:SUPPLIES_TO*1..3]->(c:Company) RETURN s.name"
    result = filter_engine.inject_active_coalescence(cypher, ctx=historical_ctx)
    assert "ALL(_rel IN r WHERE _rel.valid_from IS NOT NULL AND _rel.valid_from <= '2024-03-01'" in result


def test_inject_audit_filter(filter_engine: TemporalQueryFilter, historical_ctx: PiTContext):
    cypher = (
        "MATCH (s:Company)-[r:SUPPLIES_TO]->(c:Company {ticker: 'AAPL'}) "
        "RETURN s.name, r.status"
    )
    result = filter_engine.inject_audit_filter(cypher, ctx=historical_ctx, target_status="TERMINATED")
    assert "r.status = 'TERMINATED'" in result
    assert "2024-03-01" in result
    assert "ACTIVE_CURRENT" not in result


def test_idempotency_does_not_double_inject(filter_engine: TemporalQueryFilter, historical_ctx: PiTContext):
    cypher = "MATCH (s:Company)-[r:SUPPLIES_TO]->(c:Company) RETURN s.name"
    first = filter_engine.inject_active_coalescence(cypher, ctx=historical_ctx)
    second = filter_engine.inject_active_coalescence(first, ctx=historical_ctx)
    assert second == first


def test_convenience_wrapper_matches_engine(historical_ctx: PiTContext):
    cypher = "MATCH (s:Company)-[r:SUPPLIES_TO]->(c:Company) RETURN s.name"
    assert inject_active_temporal_filter(cypher, ctx=historical_ctx) == (
        DEFAULT_TEMPORAL_FILTER.inject_active_coalescence(cypher, ctx=historical_ctx)
    )
