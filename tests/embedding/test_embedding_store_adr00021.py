"""Unit tests for contextual node embeddings, pgvector serialization, and source provenance (ADR-00021)."""

from __future__ import annotations

from datetime import date
from typing import Any, List, Tuple
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pytest

import embedding.embedder as embedder
from knowledge_graph.embedding_store import format_node_embedding_text, store_node_embeddings
from retrieval.hybrid import HybridSeeder
from retrieval.service import app as retrieval_app
from retrieval.temporal import PiTContext
from tools.utils import store_embedding


class _RecordingCursor:
    def __init__(self, executed: List[Tuple[str, Any]], rows: List[Tuple[Any, ...]] | None = None) -> None:
        self.executed = executed
        self._rows = rows or []

    def __enter__(self) -> "_RecordingCursor":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))

    def fetchall(self) -> List[Tuple[Any, ...]]:
        return list(self._rows)

    def close(self) -> None:
        return None


class _RecordingConnection:
    def __init__(self, rows: List[Tuple[Any, ...]] | None = None) -> None:
        self.executed: List[Tuple[str, Any]] = []
        self.committed = False
        self.closed = False
        self._rows = rows or []

    def __enter__(self) -> "_RecordingConnection":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def cursor(self) -> _RecordingCursor:
        return _RecordingCursor(self.executed, self._rows)

    def commit(self) -> None:
        self.committed = True

    def close(self) -> None:
        self.closed = True


def test_format_node_embedding_text_includes_type_identifiers_and_context() -> None:
    formatted = format_node_embedding_text(
        {
            "id": "Apple Inc.",
            "type": "Company",
            "properties": {
                "ticker": "AAPL",
                "cik": "0000320193",
                "context": "Designs smartphones and custom silicon chips.",
            },
        }
    )
    assert formatted == (
        "Apple Inc. | type=Company | ticker=AAPL | cik=0000320193 | "
        "context=Designs smartphones and custom silicon chips."
    )

    fallback = format_node_embedding_text({"id": "Delta", "type": "Company", "sector": "Industrials"})
    assert fallback == "Delta | type=Company | ticker=N/A | cik=N/A | context=sector=Industrials"

    long_text = "word " * 100
    bounded = format_node_embedding_text({"id": "TSM", "type": "Company", "context": long_text}, max_context_chars=60)
    ctx_part = bounded.split("context=", 1)[1]
    assert len(ctx_part) <= 60
    assert ctx_part.endswith("...")


def test_format_pgvector_literal_canonical_serialization_and_validation() -> None:
    literal = embedder.format_pgvector_literal([0.5, 0.25, -0.5])
    assert literal == "[0.5,0.25,-0.5]"

    with pytest.raises(embedder.InvalidEmbeddingError):
        embedder.format_pgvector_literal([0.0, 0.0, 0.0])

    with pytest.raises(embedder.InvalidEmbeddingError):
        embedder.format_pgvector_literal([float("nan"), 1.0])


def test_store_node_embeddings_embeds_contextual_text_and_preserves_provenance() -> None:
    conn = _RecordingConnection()
    captured_texts: List[str] = []

    def _fake_embed(texts: List[str]) -> List[List[float]]:
        captured_texts.extend(texts)
        return [[0.5, 0.5] for _ in texts]

    nodes = [
        {"id": "Apple Inc.", "type": "Company", "ticker": "AAPL", "cik": "0000320193"},
    ]

    with patch("tools.init_db._connect_db", return_value=conn), \
         patch("embedding.embedder.embed_texts", side_effect=_fake_embed):
        store_node_embeddings(nodes, source_hash="hash_first", valid_from="2024-01-02")
        store_node_embeddings(nodes, source_hash="hash_second", valid_from="2024-06-03")

    assert len(captured_texts) == 2
    assert captured_texts[0].startswith("Apple Inc. | type=Company | ticker=AAPL | cik=0000320193")

    node_upserts = [item for item in conn.executed if "INSERT INTO node_embeddings " in item[0]]
    source_inserts = [item for item in conn.executed if "INSERT INTO node_embedding_sources " in item[0]]

    assert len(node_upserts) == 2
    assert node_upserts[0][1][3] == "[0.5,0.5]"
    assert node_upserts[0][1][4] == "hash_first"
    assert node_upserts[1][1][4] == "hash_second"

    assert [item[1][:3] for item in source_inserts] == [
        ("Apple Inc.", "hash_first", "Company"),
        ("Apple Inc.", "hash_second", "Company"),
    ]
    assert conn.committed is True
    assert conn.closed is True


def test_all_pgvector_sql_paths_pass_serialized_vector_literal() -> None:
    # 1. tools.utils.store_embedding
    utils_conn = _RecordingConnection()
    with patch("tools.utils._db_connection", return_value=utils_conn):
        store_embedding("hash_news", [0.25, 0.75])
    assert utils_conn.executed[0][1] == ("[0.25,0.75]", "hash_news")

    # 2. HybridSeeder.search_dense_entities
    hybrid_conn = _RecordingConnection(rows=[("Apple Inc.", "Company", "hash1", 0.91)])
    seeder = HybridSeeder(text_to_cql_engine=MagicMock())
    ctx = PiTContext(as_of=date(2024, 7, 1), mode="historical")
    with patch("retrieval.hybrid.pg_connection", return_value=hybrid_conn), \
         patch("embedding.embedder.embed_texts", return_value=[[0.6, 0.8]]):
        res = seeder.search_dense_entities("foundry", ctx=ctx, top_k=3)
    assert len(res) == 1
    assert hybrid_conn.executed[0][1] == ("[0.6,0.8]", "2024-07-01", "2024-07-01", "[0.6,0.8]", 3)

    # 3. retrieval.service POST /search (delegating via query_bi -> HybridSeeder)
    svc_conn = _RecordingConnection(rows=[("TSM", "Company", "hash_tsm", 0.92)])
    client = TestClient(retrieval_app)
    with patch("retrieval.hybrid.pg_connection", return_value=svc_conn), \
         patch("embedding.embedder.embed_texts", return_value=[[0.6, 0.8]]), \
         patch("retrieval.facade.MemgraphSubgraphStore.traverse", return_value=[]), \
         patch("retrieval.facade.MemgraphSubgraphStore.execute_guarded", return_value=[]), \
         patch("sft_pipeline.serving.model_client.YarnBallModelClient.generate", return_value=MagicMock(final_output="MATCH (c:Company) RETURN c LIMIT 1", think_trace=None, error=None)):
        resp = client.post("/search", json={"query": "TSM", "as_of": "2024-07-01", "k": 1})
    assert resp.status_code == 200
    assert svc_conn.executed[0][1] == ("[0.6,0.8]", "2024-07-01", "2024-07-01", "[0.6,0.8]", 1)
