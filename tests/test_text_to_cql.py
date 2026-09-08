# tests/test_text_to_cql.py
# -*- coding: utf-8 -*-
"""Unit tests for the TextToCQL engine and safety guardrails."""

import pytest
from unittest.mock import MagicMock, patch

from rag.text_to_cql import TextToCQL


@pytest.fixture
def text_to_cql_engine():
    """Provides a TextToCQL engine instance for unit tests."""
    return TextToCQL(max_limit=50, query_timeout_sec=3.0)


# ----------------------------------------------------------------------
# 1. Guardrail & Safety Validation Tests
# ----------------------------------------------------------------------
def test_validate_cypher_allows_valid_match_queries(text_to_cql_engine):
    """Test that valid read-only MATCH queries pass guardrails."""
    valid_queries = [
        "MATCH (c:Company {id: 'Apple'}) RETURN c",
        "MATCH (a:Company)-[:PARTNERED_WITH]-(b:Company) WHERE a.ticker = 'AAPL' RETURN b.id",
        "MATCH (p:Person)-[:LEADS]->(c:Company) RETURN p.id, c.id LIMIT 10",
        "WITH 'AAPL' AS ticker MATCH (c:Company {ticker: ticker}) RETURN c.id",
    ]
    for q in valid_queries:
        is_valid, err = text_to_cql_engine.validate_cypher(q)
        assert is_valid is True, f"Query '{q}' should be valid, but failed: {err}"
        assert err is None


def test_validate_cypher_blocks_mutating_keywords(text_to_cql_engine):
    """Test that destructive or mutating operations are strictly blocked."""
    mutating_queries = [
        ("MATCH (c:Company) CREATE (n:Company {id: 'Fake'})", "CREATE"),
        ("MATCH (c:Company) DELETE c", "DELETE"),
        ("MATCH (c:Company) DETACH DELETE c", "DELETE"),
        ("MATCH (c:Company) SET c.name = 'Hacked'", "SET"),
        ("MATCH (c:Company) REMOVE c.ticker", "REMOVE"),
        ("MERGE (c:Company {id: 'Apple'})", "MERGE"),
        ("DROP INDEX ON :Company(id)", "DROP"),
        ("ALTER TABLE foo DROP bar", "ALTER"),
        ("GRANT ALL ON DATABASE TO user", "GRANT"),
    ]
    for q, keyword in mutating_queries:
        is_valid, err = text_to_cql_engine.validate_cypher(q)
        assert is_valid is False, f"Mutating query '{q}' should have been blocked"
        assert "forbidden" in err.lower() or "must begin with match" in err.lower()


def test_validate_cypher_blocks_admin_procedures(text_to_cql_engine):
    """Test that administrative procedure calls are blocked."""
    admin_queries = [
        "MATCH (c) CALL dbms.components() RETURN c",
        "MATCH (c) CALL mg.stop() RETURN c",
    ]
    for q in admin_queries:
        is_valid, err = text_to_cql_engine.validate_cypher(q)
        assert is_valid is False
        assert "administrative procedure" in err.lower()


# ----------------------------------------------------------------------
# 2. Resource Bounding Tests
# ----------------------------------------------------------------------
def test_enforce_limit_appends_and_clamps(text_to_cql_engine):
    """Test that LIMIT 50 is appended if missing, and clamped if too large."""
    # Appends LIMIT when missing
    q1 = "MATCH (c:Company) RETURN c"
    assert text_to_cql_engine.enforce_limit(q1) == "MATCH (c:Company) RETURN c LIMIT 50"

    # Clamps excessive LIMIT
    q2 = "MATCH (c:Company) RETURN c LIMIT 500"
    assert text_to_cql_engine.enforce_limit(q2) == "MATCH (c:Company) RETURN c LIMIT 50"

    # Leaves smaller LIMIT intact
    q3 = "MATCH (c:Company) RETURN c LIMIT 10"
    assert text_to_cql_engine.enforce_limit(q3) == "MATCH (c:Company) RETURN c LIMIT 10"


# ----------------------------------------------------------------------
# 3. Cypher Cleaning Tests
# ----------------------------------------------------------------------
def test_clean_cypher_markdown_fences():
    """Test stripping markdown code blocks from LLM responses."""
    fenced_output = "```cypher\nMATCH (c:Company) RETURN c.id LIMIT 10;\n```"
    cleaned = TextToCQL.clean_cypher(fenced_output)
    assert cleaned == "MATCH (c:Company) RETURN c.id LIMIT 10"

    raw_output = "MATCH (a:Company)-[:COMPETES_WITH]-(b:Company) RETURN a.id, b.id;"
    assert TextToCQL.clean_cypher(raw_output) == "MATCH (a:Company)-[:COMPETES_WITH]-(b:Company) RETURN a.id, b.id"


# ----------------------------------------------------------------------
# 4. Schema Extraction Tests
# ----------------------------------------------------------------------
def test_get_active_schema_returns_expected_structure(text_to_cql_engine):
    """Test that get_active_schema returns node labels, relationship types, and properties."""
    schema = text_to_cql_engine.get_active_schema()
    assert "labels" in schema
    assert "relationship_types" in schema
    assert "properties" in schema
    assert "Company" in schema["labels"]
    assert "PARTNERED_WITH" in schema["relationship_types"]


# ----------------------------------------------------------------------
# 5. Guarded Execution & Auto-Correction Tests
# ----------------------------------------------------------------------
def test_execute_query_successful_first_try(text_to_cql_engine):
    """Test successful translation and execution on first attempt."""
    mock_llm_response = "MATCH (c:Company {id: 'Apple Inc.'}) RETURN c.id AS name"
    mock_records = [{"name": "Apple Inc."}]

    mock_session = MagicMock()
    mock_session.run.return_value = mock_records
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    with patch.object(text_to_cql_engine, "_call_llm", return_value=mock_llm_response), \
         patch("rag.text_to_cql.get_memgraph_driver", return_value=mock_driver):

        res = text_to_cql_engine.execute_query("What is Apple's company node?")

        assert res["status"] == "SUCCESS"
        assert res["retries_used"] == 0
        assert res["record_count"] == 1
        assert res["records"][0]["name"] == "Apple Inc."
        assert "LIMIT 50" in res["cypher"]


def test_execute_query_self_corrects_on_error(text_to_cql_engine):
    """Test that engine self-corrects when initial LLM output violates guardrails."""
    # Attempt 1: Mutating query (rejected by guardrail)
    # Attempt 2: Valid read-only query (accepted and executed)
    mock_responses = [
        "MATCH (c:Company) DELETE c",
        "MATCH (c:Company) RETURN c.id LIMIT 10",
    ]

    mock_session = MagicMock()
    mock_session.run.return_value = [{"c.id": "Apple Inc."}]
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    with patch.object(text_to_cql_engine, "_call_llm", side_effect=mock_responses), \
         patch("rag.text_to_cql.get_memgraph_driver", return_value=mock_driver):

        res = text_to_cql_engine.execute_query("Find all companies")

        assert res["status"] == "SUCCESS"
        assert res["retries_used"] == 1  # Self-corrected on attempt 2
        assert res["record_count"] == 1
        assert "DELETE" not in res["cypher"]
