# tests/test_hybrid_retriever.py
# -*- coding: utf-8 -*-
"""Unit tests for the HybridRetriever and GroundedSynthesizer modules."""

import pytest
from unittest.mock import MagicMock, patch

from rag.hybrid_retriever import (
    HybridRetriever,
    GroundedSynthesizer,
    HybridGraphRAGEngine,
)
from rag.text_to_cql import TextToCQL


@pytest.fixture
def hybrid_retriever():
    """Provides a HybridRetriever with a mock TextToCQL engine."""
    mock_cql = MagicMock(spec=TextToCQL)
    mock_cql.execute_query.return_value = {
        "status": "SUCCESS",
        "cypher": "MATCH (c:Company {id: 'Apple'}) RETURN c.id",
        "records": [{"c.id": "Apple Inc."}],
    }
    return HybridRetriever(text_to_cql_engine=mock_cql, top_k_dense=3, subgraph_hops=1)


# ----------------------------------------------------------------------
# 1. Dense Semantic Entity Search Tests
# ----------------------------------------------------------------------
def test_search_dense_entities_returns_formatted_results(hybrid_retriever):
    """Test dense vector similarity search query execution."""
    mock_rows = [
        ("Apple Inc.", "Company", "hash_aapl", 0.95),
        ("TSMC", "Company", "hash_tsmc", 0.88),
    ]

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = mock_rows
    mock_conn.cursor.return_value = mock_cursor

    with patch("rag.hybrid_retriever.pg_connection", return_value=mock_conn), \
         patch("embedding.embedder.embed_texts", return_value=[[0.1] * 768]):

        results = hybrid_retriever.search_dense_entities("Apple chip supplier", top_k=2)

        assert len(results) == 2
        assert results[0]["node_id"] == "Apple Inc."
        assert results[0]["entity_type"] == "Company"
        assert results[0]["similarity"] == 0.95


def test_search_dense_entities_handles_offline_fallback(hybrid_retriever):
    """Test that dense search handles database connection errors gracefully."""
    with patch("rag.hybrid_retriever.pg_connection", side_effect=Exception("DB down")), \
         patch("embedding.embedder.embed_texts", return_value=[[0.1] * 768]):

        results = hybrid_retriever.search_dense_entities("test query")
        assert results == []


# ----------------------------------------------------------------------
# 2. Subgraph Expansion Tests
# ----------------------------------------------------------------------
def test_expand_subgraph_queries_memgraph_triples(hybrid_retriever):
    """Test expanding seed entities in Memgraph to retrieve connecting triples."""
    mock_records = [
        {
            "source": "Apple Inc.",
            "rel_type": "PARTNERED_WITH",
            "target": "TSMC",
            "props": {"context": "A18 chip production"},
            "source_hash": "hash_a18",
        }
    ]

    mock_session = MagicMock()
    mock_session.run.return_value = mock_records
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    with patch("rag.hybrid_retriever.get_memgraph_driver", return_value=mock_driver):
        triples = hybrid_retriever.expand_subgraph(["Apple Inc."])

        assert len(triples) == 1
        assert triples[0]["source"] == "Apple Inc."
        assert triples[0]["relation"] == "PARTNERED_WITH"
        assert triples[0]["target"] == "TSMC"
        assert triples[0]["source_hash"] == "hash_a18"


def test_expand_subgraph_empty_seeds(hybrid_retriever):
    """Test that empty seed list returns empty list immediately."""
    assert hybrid_retriever.expand_subgraph([]) == []


# ----------------------------------------------------------------------
# 3. Article Context Fetching Tests
# ----------------------------------------------------------------------
def test_fetch_article_context_from_postgres(hybrid_retriever):
    """Test querying financial_news_queue for article snippets by source_hash."""
    mock_rows = [
        (
            "hash123",
            "Apple and TSMC Deepen 2nm Partnership",
            "https://reuters.com/article/apple-tsmc",
            "2026-05-01 10:00:00",
            "Apple has committed billions to secure TSMC's 2nm production lines for next-gen silicon.",
            "Reuters Tech",
        )
    ]

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = mock_rows
    mock_conn.cursor.return_value = mock_cursor

    with patch("rag.hybrid_retriever.pg_connection", return_value=mock_conn):
        articles = hybrid_retriever.fetch_article_context(["hash123"])

        assert len(articles) == 1
        assert articles[0]["title"] == "Apple and TSMC Deepen 2nm Partnership"
        assert "https://reuters.com/article/apple-tsmc" in articles[0]["url"]
        assert "TSMC's 2nm" in articles[0]["snippet"]


# ----------------------------------------------------------------------
# 4. Grounded Synthesis & Negative Grounding Tests
# ----------------------------------------------------------------------
def test_synthesizer_negative_grounding_on_empty_context():
    """Test that synthesizer refuses to answer when no graph or article context exists."""
    synthesizer = GroundedSynthesizer()
    empty_context = {
        "subgraph_triples": [],
        "cql_result": {"records": []},
        "articles": [],
    }

    result = synthesizer.synthesize("Who acquired StartupXYZ?", empty_context)

    assert result["is_grounded"] is False
    assert "no verified relationships" in result["answer"].lower()
    assert result["citations"] == []


def test_synthesizer_extracts_citations():
    """Test that citations are accurately extracted from generated answers."""
    synthesizer = GroundedSynthesizer()
    articles = [
        {
            "title": "NVIDIA Blackwell Ramp Accelerates",
            "url": "https://bloomberg.com/news/nvda-blackwell",
            "published_at": "2026-04-30",
            "snippet": "NVIDIA is ramping Blackwell chips...",
        }
    ]

    answer = "NVIDIA is ramping up Blackwell chip production as reported in [NVIDIA Blackwell Ramp Accelerates](https://bloomberg.com/news/nvda-blackwell)."
    citations = synthesizer.extract_citations(answer, articles)

    assert len(citations) == 1
    assert citations[0]["title"] == "NVIDIA Blackwell Ramp Accelerates"
    assert citations[0]["url"] == "https://bloomberg.com/news/nvda-blackwell"


# ----------------------------------------------------------------------
# 5. Full End-to-End Pipeline Tests
# ----------------------------------------------------------------------
def test_hybrid_graphrag_engine_end_to_end():
    """Test full HybridGraphRAGEngine question-to-answer execution."""
    mock_retriever = MagicMock(spec=HybridRetriever)
    mock_retriever.retrieve.return_value = {
        "query": "Which foundry manufactures Apple chips?",
        "cql_result": {"cypher": "MATCH (a:Company)-[:PRODUCES]-(t:Company)", "records": [{"t.id": "TSMC"}]},
        "subgraph_triples": [{"source": "Apple Inc.", "relation": "PARTNERED_WITH", "target": "TSMC", "source_hash": "h1"}],
        "articles": [{"title": "Apple Contract", "url": "https://ft.com/apple", "published_at": "2026-05-01", "snippet": "..."}],
        "latency_ms": 12.5,
    }

    mock_synthesizer = MagicMock(spec=GroundedSynthesizer)
    mock_synthesizer.synthesize.return_value = {
        "query": "Which foundry manufactures Apple chips?",
        "answer": "TSMC manufactures custom silicon for Apple under an exclusive partnership [Apple Contract](https://ft.com/apple).",
        "is_grounded": True,
        "citations": [{"title": "Apple Contract", "url": "https://ft.com/apple"}],
        "subgraph_triples": [{"source": "Apple Inc.", "relation": "PARTNERED_WITH", "target": "TSMC"}],
        "cql_records": [{"t.id": "TSMC"}],
    }

    engine = HybridGraphRAGEngine(retriever=mock_retriever, synthesizer=mock_synthesizer)
    response = engine.answer_query("Which foundry manufactures Apple chips?")

    assert response["is_grounded"] is True
    assert "TSMC" in response["answer"]
    assert len(response["citations"]) == 1
    assert response["cypher_query"] == "MATCH (a:Company)-[:PRODUCES]-(t:Company)"
