"""Unit tests for fail-fast embedder initialization and zero-vector prohibition (ADR-00020)."""

from __future__ import annotations

from datetime import date
import builtins
import json
import math
from typing import Any
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pytest

import embedding.embedder as embedder
from knowledge_graph.embedding_store import store_node_embeddings
from retrieval.hybrid import HybridSeeder
from retrieval.service import app as retrieval_app
from retrieval.temporal import PiTContext
from streaming.consumer import process_streaming_event, run_consumer
from tools.utils import store_embedding


@pytest.fixture(autouse=True)
def _clean_embedder_state(monkeypatch: pytest.MonkeyPatch) -> Any:
    embedder.reset_embedder_cache()
    monkeypatch.delenv("YARNBALL_EMBEDDER_MOCK", raising=False)
    yield
    embedder.reset_embedder_cache()


def _simulate_missing_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def _blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError("No module named 'sentence_transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)


def test_get_embedder_model_fails_fast_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _simulate_missing_sentence_transformers(monkeypatch)
    monkeypatch.setenv("EMBEDDER_MODEL", "sentence-transformers/all-mpnet-base-v2")

    with pytest.raises(embedder.EmbedderUnavailableError, match="all-mpnet-base-v2"):
        embedder.get_embedder_model()

    with pytest.raises(embedder.EmbedderUnavailableError):
        embedder.embed_texts(["Apple supply chain risk"])


def test_explicit_mock_mode_emits_deterministic_nonzero_unit_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _simulate_missing_sentence_transformers(monkeypatch)

    vecs_direct = embedder.embed_texts(["Taiwan Semiconductor", "Apple Inc."], allow_mock=True)
    assert len(vecs_direct) == 2
    assert len(vecs_direct[0]) == 768
    assert math.isclose(sum(v * v for v in vecs_direct[0]), 1.0, rel_tol=1e-6)
    assert vecs_direct[0] != vecs_direct[1]

    embedder.reset_embedder_cache()
    monkeypatch.setenv("YARNBALL_EMBEDDER_MOCK", "1")
    vecs_env = embedder.embed_texts(["Taiwan Semiconductor"])
    assert vecs_env[0] == vecs_direct[0]


@pytest.mark.parametrize(
    "bad_vectors",
    [
        [[]],
        [[0.0] * 768],
        [[1e-6] + [0.0] * 767],  # ||v||^2 = 1e-12 <= 1e-8
        [[float("nan"), 1.0]],
        [[float("inf"), 1.0]],
        [[float("-inf"), 1.0]],
    ],
)
def test_validate_embedding_vectors_rejects_degenerate_inputs(bad_vectors: list[list[float]]) -> None:
    with pytest.raises(embedder.InvalidEmbeddingError):
        embedder.validate_embedding_vectors(bad_vectors)


def test_verify_embedder_ready_reports_status_without_crashing(monkeypatch: pytest.MonkeyPatch) -> None:
    _simulate_missing_sentence_transformers(monkeypatch)

    offline_status = embedder.verify_embedder_ready(raise_on_error=False)
    assert offline_status["healthy"] is False
    assert offline_status["ready"] is False
    assert offline_status["dim"] == 0
    assert offline_status["mock"] is False

    mock_status = embedder.verify_embedder_ready(allow_mock=True, raise_on_error=False)
    assert mock_status["healthy"] is True
    assert mock_status["ready"] is True
    assert mock_status["dim"] == 768
    assert mock_status["mock"] is True


def test_store_node_embeddings_blocks_zero_vectors_before_db() -> None:
    with patch("embedding.embedder.embed_texts", return_value=[[0.0] * 768]):
        with patch("knowledge_graph.db.pg_connection") as mock_pg:
            with pytest.raises(embedder.InvalidEmbeddingError):
                store_node_embeddings([{"id": "TSM", "type": "Company"}], source_hash="h1")
            mock_pg.assert_not_called()


def test_store_embedding_blocks_zero_and_nan_vectors_before_db() -> None:
    with patch("tools.utils._db_connection") as mock_db:
        with pytest.raises(embedder.InvalidEmbeddingError):
            store_embedding("hash1", [0.0] * 768)
        with pytest.raises(embedder.InvalidEmbeddingError):
            store_embedding("hash2", [float("nan"), 1.0])
        mock_db.assert_not_called()


def test_hybrid_retriever_raises_on_zero_or_unavailable_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _simulate_missing_sentence_transformers(monkeypatch)
    seeder = HybridSeeder(text_to_cql_engine=MagicMock())
    ctx = PiTContext(as_of=date(2024, 5, 1), mode="historical")

    with pytest.raises(embedder.EmbedderUnavailableError):
        seeder.search_dense_entities("lithography disruption", ctx=ctx)

    with patch("embedding.embedder.embed_texts", return_value=[[0.0] * 768]):
        with patch("retrieval.hybrid.pg_connection") as mock_pg:
            with pytest.raises(embedder.InvalidEmbeddingError):
                seeder.search_dense_entities("lithography disruption", ctx=ctx)
            mock_pg.assert_not_called()


def test_retrieval_service_health_and_search_error_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    _simulate_missing_sentence_transformers(monkeypatch)
    client = TestClient(retrieval_app)

    health_down = client.get("/health")
    assert health_down.status_code == 503
    assert health_down.json()["ready"] is False

    search_down = client.post("/search", json={"query": "TSMC fab", "as_of": "2024-05-01", "k": 3})
    assert search_down.status_code == 503

    monkeypatch.setenv("YARNBALL_EMBEDDER_MOCK", "1")
    health_ok = client.get("/health")
    assert health_ok.status_code == 200
    assert health_ok.json()["ready"] is True
    assert health_ok.json()["mock"] is True


def test_streaming_consumer_routes_embedder_failure_to_dlt_and_checks_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _simulate_missing_sentence_transformers(monkeypatch)

    with pytest.raises(embedder.EmbedderUnavailableError):
        run_consumer()

    dlt_calls: list[dict[str, Any]] = []
    monkeypatch.setattr("streaming.consumer.store_raw", lambda **kwargs: None)
    monkeypatch.setattr(
        "streaming.consumer.send_to_dlt",
        lambda producer, message, payload, reason, error=None: dlt_calls.append(
            {"reason": reason, "error": error}
        ),
    )

    payload = {
        "source": "sec_8k",
        "raw_payload": json.dumps({"title": "TSMC Fab Notice", "content": "Minor power dip"}),
        "fetched_at": "2026-01-01T00:00:00+00:00",
        "source_hash": "hash_embed_fail",
    }
    ok = process_streaming_event(payload, dlt_producer=MagicMock(), skip_embedding=False)
    assert ok is False
    assert len(dlt_calls) == 1
    assert dlt_calls[0]["reason"] == "consumer_message_failed"
