"""Unit tests for Stage 6.3: Temporal State Coalescence Transformer."""

import pytest
from retrieval.temporal import (
    TemporalQueryFilter,
    DEFAULT_TEMPORAL_FILTER,
    inject_active_temporal_filter,
)


@pytest.fixture
def filter_engine() -> TemporalQueryFilter:
    return TemporalQueryFilter(default_as_of_date_param="$as_of_date")


def test_inject_active_coalescence_no_where(filter_engine: TemporalQueryFilter):
    cypher = (
        "MATCH (s:Company)-[r:SUPPLIES_TO]->(c:Company {ticker: 'AAPL'}) "
        "RETURN s.name, s.ticker"
    )
    result = filter_engine.inject_active_coalescence(cypher)
    assert "coalesce(r.status, 'ACTIVE_CURRENT') = 'ACTIVE_CURRENT'" in result
    assert "r.valid_to IS NULL OR r.valid_to >= $as_of_date" in result
    assert "WHERE" in result
    assert result.index("WHERE") < result.index("RETURN")


def test_inject_active_coalescence_existing_where(filter_engine: TemporalQueryFilter):
    cypher = (
        "MATCH (s:Company)-[r:SUPPLIES_TO]->(c:Company) "
        "WHERE c.ticker = 'AAPL' "
        "RETURN s.name"
    )
    result = filter_engine.inject_active_coalescence(cypher)
    assert "coalesce(r.status, 'ACTIVE_CURRENT') = 'ACTIVE_CURRENT'" in result
    assert "AND" in result
    assert "c.ticker = 'AAPL'" in result


def test_inject_active_coalescence_anonymous_rel(filter_engine: TemporalQueryFilter):
    cypher = (
        "MATCH (s:Company)-[:SUPPLIES_TO]->(c:Company {ticker: 'NVDA'}) "
        "RETURN s.name"
    )
    result = filter_engine.inject_active_coalescence(cypher, default_var="r")
    assert "[r:SUPPLIES_TO]" in result
    assert "coalesce(r.status, 'ACTIVE_CURRENT') = 'ACTIVE_CURRENT'" in result


def test_inject_active_coalescence_explicit_date(filter_engine: TemporalQueryFilter):
    cypher = (
        "MATCH (s:Company)-[r:SUPPLIES_TO]->(c:Company {ticker: 'AAPL'}) "
        "RETURN s.name"
    )
    result = filter_engine.inject_active_coalescence(cypher, as_of_date="2024-03-01")
    assert "'2024-03-01'" in result
    assert "$as_of_date" not in result


def test_inject_audit_filter(filter_engine: TemporalQueryFilter):
    cypher = (
        "MATCH (s:Company)-[r:SUPPLIES_TO]->(c:Company {ticker: 'AAPL'}) "
        "RETURN s.name, r.status"
    )
    result = filter_engine.inject_audit_filter(cypher, target_status="TERMINATED", as_of_date="2024-03-01")
    assert "r.status = 'TERMINATED'" in result
    assert "2024-03-01" in result
    assert "ACTIVE_CURRENT" not in result


def test_idempotency_does_not_double_inject(filter_engine: TemporalQueryFilter):
    cypher = (
        "MATCH (s:Company)-[r:SUPPLIES_TO]->(c:Company) "
        "WHERE coalesce(r.status, 'ACTIVE_CURRENT') = 'ACTIVE_CURRENT' "
        "RETURN s.name"
    )
    result = filter_engine.inject_active_coalescence(cypher)
    assert result == cypher


def test_convenience_wrapper_matches_engine():
    cypher = "MATCH (s:Company)-[r:SUPPLIES_TO]->(c:Company) RETURN s.name"
    assert inject_active_temporal_filter(cypher) == DEFAULT_TEMPORAL_FILTER.inject_active_coalescence(cypher)

