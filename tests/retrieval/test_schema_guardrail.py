"""Unit tests for Stage 6.1: CypherASTWhitelistGuardrail."""

import pytest
from retrieval.guardrails.schema_guardrail import (
    CypherASTWhitelistGuardrail,
    DEFAULT_FINANCE_SCHEMA,
    SchemaViolation,
)


@pytest.fixture
def guardrail() -> CypherASTWhitelistGuardrail:
    return CypherASTWhitelistGuardrail(schema=DEFAULT_FINANCE_SCHEMA, max_limit=50)


def test_valid_cypher_standard_query(guardrail: CypherASTWhitelistGuardrail):
    cypher = (
        "MATCH (c:Company {ticker: 'AAPL'})-[:SUPPLIES_TO]->(m:Company) "
        "WHERE m.market_cap > 1000000000 "
        "RETURN m.name, m.cik, m.ticker LIMIT 20"
    )
    result = guardrail.validate(cypher)
    assert result.is_valid is True
    assert result.sanitized_query is not None
    assert "LIMIT 20" in result.sanitized_query
    assert not result.violations


def test_schema_trap_ceo_age_rejected(guardrail: CypherASTWhitelistGuardrail):
    cypher = (
        "MATCH (s:Company)-[:SUPPLIES_TO]->(c:Company {ticker: 'AAPL'}) "
        "WHERE s.ceo_age > 60 AND s.status = 'ACTIVE_CURRENT' "
        "RETURN s.name, s.ticker"
    )
    result = guardrail.validate(cypher)
    assert result.is_valid is False
    assert result.sanitized_query is None
    assert result.refusal_response.startswith("(none)")
    assert any(v.identifier == "ceo_age" for v in result.violations)
    assert "ceo_age" in result.refusal_response


def test_schema_trap_carbon_emissions_rejected(guardrail: CypherASTWhitelistGuardrail):
    cypher = (
        "MATCH (s:Company)-[:SUPPLIES_TO]->(c:Company {ticker: 'AMZN'}) "
        "WHERE s.sc_3_carbon_emissions < 50000 "
        "RETURN s.name"
    )
    result = guardrail.validate(cypher)
    assert result.is_valid is False
    assert any(v.identifier == "sc_3_carbon_emissions" for v in result.violations)


def test_all_phase5_tt06a_schema_traps_rejected(guardrail: CypherASTWhitelistGuardrail):
    traps = [
        ("MATCH (s:Company) WHERE s.ceo_age > 60 RETURN s", "ceo_age"),
        ("MATCH (s:Company) WHERE s.glassdoor_employee_satisfaction > 4.0 RETURN s", "glassdoor_employee_satisfaction"),
        ("MATCH (s:Company) WHERE s.board_gender_ratio > 0.4 RETURN s", "board_gender_ratio"),
        ("MATCH (c:Company) WHERE c.tiktok_brand_sentiment_index > 0.85 RETURN c", "tiktok_brand_sentiment_index"),
        ("MATCH (p:Company) WHERE p.cmo_tenure_years > 5 RETURN p", "cmo_tenure_years"),
        ("MATCH (v:Company) WHERE v.warehouse_unionization_pct > 30 RETURN v", "warehouse_unionization_pct"),
    ]
    for cypher, expected_bad_prop in traps:
        res = guardrail.validate(cypher)
        assert res.is_valid is False, f"Failed to reject trap: {cypher}"
        assert any(v.identifier == expected_bad_prop for v in res.violations)
        assert res.refusal_response.startswith("(none)")


def test_inline_map_property_trap_rejected(guardrail: CypherASTWhitelistGuardrail):
    cypher = "MATCH (c:Company {ticker: 'NVDA', fake_metric: 123}) RETURN c"
    result = guardrail.validate(cypher)
    assert result.is_valid is False
    assert any(v.identifier == "fake_metric" for v in result.violations)


def test_string_literal_does_not_trigger_false_positive(guardrail: CypherASTWhitelistGuardrail):
    # 'ceo_age' is used as a string value in WHERE clause, not a property key
    cypher = "MATCH (c:Company) WHERE c.name = 'ceo_age' RETURN c.ticker LIMIT 10"
    result = guardrail.validate(cypher)
    assert result.is_valid is True
    assert not result.violations


def test_mutating_cypher_rejected(guardrail: CypherASTWhitelistGuardrail):
    mutating_queries = [
        "MATCH (c:Company) DELETE c",
        "MATCH (c:Company {ticker: 'AAPL'}) SET c.status = 'INACTIVE' RETURN c",
        "CREATE (c:Company {name: 'NewCo', ticker: 'NEW'})",
        "MERGE (c:Company {ticker: 'AAPL'}) RETURN c",
    ]
    for q in mutating_queries:
        res = guardrail.validate(q)
        assert res.is_valid is False
        assert any(v.violation_type == "MUTATING_KEYWORD" for v in res.violations)


def test_admin_procedures_rejected(guardrail: CypherASTWhitelistGuardrail):
    admin_queries = [
        "CALL dbms.components() YIELD name",
        "CALL mg.get_config() YIELD config",
    ]
    for q in admin_queries:
        res = guardrail.validate(q)
        assert res.is_valid is False
        assert any(v.violation_type == "ADMIN_PROCEDURE" for v in res.violations)


def test_limit_enforcement(guardrail: CypherASTWhitelistGuardrail):
    # 1. No limit specified -> clamped to max_limit (50)
    res1 = guardrail.validate("MATCH (c:Company) RETURN c.name")
    assert res1.is_valid is True
    assert res1.sanitized_query.endswith("LIMIT 50")

    # 2. Limit exceeds max_limit -> clamped to 50
    res2 = guardrail.validate("MATCH (c:Company) RETURN c.name LIMIT 500")
    assert res2.is_valid is True
    assert "LIMIT 50" in res2.sanitized_query
    assert "500" not in res2.sanitized_query

    # 3. Limit within bound -> preserved
    res3 = guardrail.validate("MATCH (c:Company) RETURN c.name LIMIT 15")
    assert res3.is_valid is True
    assert "LIMIT 15" in res3.sanitized_query


def test_empty_or_invalid_start_rejected(guardrail: CypherASTWhitelistGuardrail):
    assert guardrail.validate("").is_valid is False
    assert guardrail.validate("   ").is_valid is False
    assert guardrail.validate("SELECT * FROM companies").is_valid is False
