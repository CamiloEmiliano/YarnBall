"""Unit tests for `HybridSeeder`, `query_bi`, `GroundedSynthesizer`, and deprecated `HybridGraphRAGEngine` shim (ADR-00034)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch
import pytest

from retrieval import (
    HybridSeeder,
    InMemorySubgraphStore,
    PiTContext,
    SeedHit,
    TextToCQL,
    query_bi,
)
from sft_pipeline.rejection_queue import RuntimeRejectionQueue
from sft_pipeline.serving.model_client import ModelResponse, YarnBallModelClient
from synthesis import GroundedSynthesizer, HybridGraphRAGEngine

DEFAULT_CTX = PiTContext(as_of=date(2026, 5, 1), mode="historical")


@pytest.fixture
def hybrid_seeder() -> HybridSeeder:
    """Provides a HybridSeeder with a mock TextToCQL engine."""
    mock_cql = MagicMock(spec=TextToCQL)
    mock_cql.execute_query.return_value = {
        "status": "SUCCESS",
        "cypher": "MATCH (c:Company {id: 'Apple'}) RETURN c.id",
        "records": [{"c.id": "Apple Inc."}],
    }
    return HybridSeeder(text_to_cql_engine=mock_cql, top_k_dense=3)


# ----------------------------------------------------------------------
# 1. Dense Semantic Entity Search Tests
# ----------------------------------------------------------------------
def test_search_dense_entities_returns_formatted_results(hybrid_seeder: HybridSeeder) -> None:
    """Test dense vector similarity search query execution over SCD Type 2 node_embeddings."""
    mock_rows = [
        ("Apple Inc.", "Company", "hash_aapl", 0.95),
        ("TSMC", "Company", "hash_tsmc", 0.88),
    ]

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = mock_rows
    mock_conn.cursor.return_value = mock_cursor

    with patch("retrieval.hybrid.pg_connection", return_value=mock_conn), \
         patch("embedding.embedder.embed_texts", return_value=[[0.1] * 768]):
        results = hybrid_seeder.search_dense_entities("Apple chip supplier", ctx=DEFAULT_CTX, top_k=2)

        assert len(results) == 2
        assert results[0]["node_id"] == "Apple Inc."
        assert results[0]["entity_type"] == "Company"
        assert results[0]["similarity"] == 0.95


def test_search_dense_entities_handles_offline_fallback(hybrid_seeder: HybridSeeder) -> None:
    """ADR-00036: Dense search re-raises DocumentStoreUnavailableError instead of returning []."""
    from knowledge_graph.errors import DocumentStoreUnavailableError

    with patch("retrieval.hybrid.pg_connection", side_effect=DocumentStoreUnavailableError("DB down")), \
         patch("embedding.embedder.embed_texts", return_value=[[0.1] * 1024]):
        with pytest.raises(DocumentStoreUnavailableError, match="DB down"):
            hybrid_seeder.search_dense_entities("test query", ctx=DEFAULT_CTX)


# ----------------------------------------------------------------------
# 2. Durable Postgres Passage-to-Entity Lineage & SeedHit Ranking
# ----------------------------------------------------------------------
def test_expand_passage_entities_queries_durable_postgres_lineage(hybrid_seeder: HybridSeeder) -> None:
    """ADR-00034: expand_passage_entities resolves via PostgreSQL node_embedding_sources (no Memgraph)."""
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [("AAPL",), ("TSM",)]
    mock_conn.cursor.return_value = mock_cursor

    assert not hasattr(hybrid_seeder, "expand_subgraph")

    with patch("retrieval.hybrid.pg_connection", return_value=mock_conn):
        entities = hybrid_seeder.expand_passage_entities(["hash_news_1"], ctx=DEFAULT_CTX)

    assert entities == ["AAPL", "TSM"]
    sql_executed, params = mock_cursor.execute.call_args[0]
    assert "FROM node_embedding_sources" in sql_executed
    assert "COALESCE(nes.source_published_at, fnq.published_at) <= %s::timestamptz" in sql_executed
    assert params == (["hash_news_1"], DEFAULT_CTX.cutoff_utc_iso, 20)


def test_hybrid_seeder_returns_ranked_seed_hits(hybrid_seeder: HybridSeeder) -> None:
    """HybridSeeder.seed returns ranked SeedHit(node_id, score, origin) records."""
    with patch.object(
        hybrid_seeder,
        "search_dense_entities",
        return_value=[{"node_id": "TSM", "entity_type": "Company", "source_hash": "h1", "similarity": 0.94}],
    ), patch.object(
        hybrid_seeder,
        "search_dense_passages",
        return_value=[{"source_hash": "h2", "title": "TSMC news", "url": "https://ex.com", "similarity": 0.90}],
    ), patch.object(
        hybrid_seeder,
        "expand_passage_entities",
        return_value=["AAPL"],
    ), patch.object(
        hybrid_seeder,
        "fetch_article_context",
        return_value=[],
    ):
        hits = hybrid_seeder.seed("TSMC wafer supply to Apple", ctx=DEFAULT_CTX)

    assert len(hits) == 2
    assert isinstance(hits[0], SeedHit)
    assert hits[0].node_id == "TSM" and hits[0].origin == "dense_entity"
    assert hits[1].node_id == "AAPL" and hits[1].origin == "dense_passage"
    assert hits[0].score > hits[1].score


# ----------------------------------------------------------------------
# 3. Article Context Fetching Tests
# ----------------------------------------------------------------------
def test_fetch_article_context_from_postgres(hybrid_seeder: HybridSeeder) -> None:
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

    with patch("retrieval.hybrid.pg_connection", return_value=mock_conn):
        articles = hybrid_seeder.fetch_article_context(["hash123"], ctx=DEFAULT_CTX)

        assert len(articles) == 1
        assert articles[0]["title"] == "Apple and TSMC Deepen 2nm Partnership"
        assert "https://reuters.com/article/apple-tsmc" in articles[0]["url"]
        assert "TSMC's 2nm" in articles[0]["snippet"]


# ----------------------------------------------------------------------
# 4. Grounded Synthesis & YarnBallModelClient Tests
# ----------------------------------------------------------------------
def test_synthesizer_negative_grounding_on_empty_context(tmp_path: Path) -> None:
    """Test that synthesizer refuses to answer when no graph or article context exists and logs refusal."""
    rej_queue = RuntimeRejectionQueue(base_dir=tmp_path / "rej")
    synthesizer = GroundedSynthesizer(rejection_queue=rej_queue)
    empty_context = {
        "subgraph_triples": [],
        "cql_result": {"records": []},
        "articles": [],
    }

    result = synthesizer.synthesize("Who acquired StartupXYZ?", empty_context)

    assert result["is_grounded"] is False
    assert "no verified relationships" in result["answer"].lower()
    assert result["citations"] == []
    assert len(rej_queue.load_records()) == 1


def test_synthesizer_routes_through_yarnball_model_client(tmp_path: Path) -> None:
    """ADR-00034: GroundedSynthesizer routes through YarnBallModelClient (task=GROUNDED_SYNTHESIS)."""
    seen_payloads: List[Dict[str, Any]] = []

    def _fake_transport(payload: Dict[str, Any]) -> Dict[str, Any]:
        seen_payloads.append(payload)
        return {
            "response": "TSMC manufactures silicon for Apple [Apple Contract](https://ft.com/apple)."
        }

    client = YarnBallModelClient(transport_fn=_fake_transport)
    synthesizer = GroundedSynthesizer(
        model_client=client,
        rejection_queue=RuntimeRejectionQueue(base_dir=tmp_path / "rej"),
    )
    context = {
        "subgraph_triples": [{"source": "TSM", "rel_type": "SUPPLIES_TO", "target": "AAPL"}],
        "cql_result": {"records": []},
        "articles": [
            {
                "title": "Apple Contract",
                "url": "https://ft.com/apple",
                "published_at": "2026-05-01",
                "snippet": "TSMC fabricates custom silicon for Apple.",
            }
        ],
    }

    result = synthesizer.synthesize("Which foundry manufactures Apple chips?", context)
    assert result["is_grounded"] is True
    assert "TSMC" in result["answer"]
    assert len(result["citations"]) == 1
    assert len(seen_payloads) == 1
    assert seen_payloads[0]["model"] == "yarnball-qwen:7b"


def test_synthesizer_extracts_citations() -> None:
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
# 5. Consolidated `query_bi` Facade & Deprecated Shim End-to-End Tests
# ----------------------------------------------------------------------
def test_query_bi_and_deprecated_engine_shim_end_to_end() -> None:
    """Verify `query_bi` coordinates HybridSeeder + SubgraphStorePort, and deprecated shim delegates to it."""
    store = InMemorySubgraphStore(
        [
            {
                "source": "TSM",
                "target": "AAPL",
                "rel_type": "SUPPLIES_TO",
                "nature": "wafers",
                "share": 0.40,
                "buffer_days": 30.0,
                "valid_from": "2020-01-02",
            }
        ]
    )

    class _MockClient(YarnBallModelClient):
        def generate(self, prompt: str, task: str = "CONTAGION_REASONING", **kwargs: Any) -> ModelResponse:
            if task == "TEXT_TO_CYPHER":
                out = "MATCH (s:Company {ticker: 'TSM'})-[r:SUPPLIES_TO]->(t:Company) RETURN s, r, t"
            else:
                out = "TSM supplies wafers to AAPL with 30-day buffer mitigation."
            return ModelResponse(
                raw_text=out,
                think_trace=None,
                final_output=out,
                task=task,
                latency_sec=0.01,
                prefill_ms=1.0,
                tokens_per_sec=100.0,
                prompt_tokens=10,
                generated_tokens=10,
            )

    seeder = MagicMock(spec=HybridSeeder)
    seeder.seed_with_context.return_value = {
        "hits": [SeedHit(node_id="TSM", score=0.95, origin="dense_entity", source_hash="h1")],
        "articles": [
            {
                "title": "Apple Contract",
                "url": "https://ft.com/apple",
                "published_at": "2026-05-01",
                "snippet": "TSMC supplies Apple.",
            }
        ],
    }

    res = query_bi(
        "Which foundry supplies Apple chips?",
        ctx=DEFAULT_CTX,
        store=store,
        seeder=seeder,
        model_client=_MockClient(),
    )
    assert res.is_valid_query is True
    assert len(res.retrieved_subgraph) == 1
    assert res.retrieved_subgraph[0]["source"] == "TSM"
    assert res.retrieved_subgraph[0]["target"] == "AAPL"
    assert len(res.seed_hits) == 1
    assert res.seed_hits[0]["node_id"] == "TSM"

    with pytest.warns(DeprecationWarning):
        shim = HybridGraphRAGEngine(store=store, seeder=seeder, model_client=_MockClient())

    response = shim.answer_query("Which foundry supplies Apple chips?", ctx=DEFAULT_CTX)
    assert response["is_grounded"] is True
    assert "TSM" in response["answer"]
    assert len(response["citations"]) == 1
    assert len(response["subgraph_triples"]) == 1
