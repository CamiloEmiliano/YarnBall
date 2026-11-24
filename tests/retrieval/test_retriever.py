from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest

from knowledge_graph.errors import DocumentStoreUnavailableError
from knowledge_graph.memgraph_driver import _DummyDriver
from retrieval.service import app

client = TestClient(app)


@pytest.fixture(scope="module")
def insert_test_embeddings():
    # Insert two dummy node embeddings into PGVECTOR only when a live Postgres instance is reachable
    from embedding.embedder import get_embedding_dim
    from knowledge_graph.db import pg_connection

    try:
        conn_ctx = pg_connection()
    except DocumentStoreUnavailableError:
        pytest.skip("PostgreSQL connection not available for test_retriever")
    if conn_ctx is None:
        pytest.skip("PostgreSQL connection not available for test_retriever")

    dim = get_embedding_dim()
    create_sql = f"""
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS node_embeddings (
            node_id TEXT NOT NULL,
            valid_from DATE NOT NULL,
            valid_to DATE NULL,
            entity_type TEXT,
            embedding VECTOR({dim}) NOT NULL,
            source_hash TEXT,
            known_at TIMESTAMPTZ,
            is_unanchored BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT now(),
            PRIMARY KEY (node_id, valid_from)
        );
    """
    sql = """
        INSERT INTO node_embeddings (node_id, valid_from, valid_to, entity_type, embedding, source_hash, is_unanchored)
        VALUES (%s, %s::date, NULL, %s, %s, %s, FALSE)
        ON CONFLICT (node_id, valid_from) DO UPDATE
        SET embedding = EXCLUDED.embedding,
            is_unanchored = FALSE;
    """
    dummy_vec = [1.0] + [0.0] * (dim - 1)
    with conn_ctx as conn:
        with conn.cursor() as cur:
            cur.execute(create_sql)
            cur.execute("DELETE FROM node_embeddings;")
            cur.execute(sql, ("TestNode1", "2024-01-02", "Entity", dummy_vec, "test"))
            cur.execute(sql, ("TestNode2", "2024-01-02", "Entity", dummy_vec, "test"))
        conn.commit()
    yield
    try:
        with pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM node_embeddings WHERE node_id IN ('TestNode1','TestNode2');")
            conn.commit()
    except DocumentStoreUnavailableError:
        pass


def test_search_endpoint(insert_test_embeddings):
    from embedding.embedder import get_embedding_dim

    dim = get_embedding_dim()
    with patch("knowledge_graph.memgraph_driver.get_memgraph_driver", return_value=_DummyDriver()), \
         patch("embedding.embedder.embed_texts", return_value=[[1.0] + [0.0] * (dim - 1)]):
        response = client.post("/search", json={"query": "any text", "as_of": "2024-06-01", "k": 2})
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, dict)
    assert payload["as_of_date"] == "2024-06-01"
    returned_ids = {entry["node_id"] for entry in payload["seed_hits"]}
    assert returned_ids == {"TestNode1", "TestNode2"}
    assert "retrieved_subgraph" in payload
    assert "verbalized_subgraph" in payload


def test_search_rejects_zero_vector_adr00020():
    from embedding.embedder import get_embedding_dim

    dim = get_embedding_dim()
    with patch("knowledge_graph.memgraph_driver.get_memgraph_driver", return_value=_DummyDriver()), \
         patch("embedding.embedder.embed_texts", return_value=[[0.0] * dim]):
        response = client.post("/search", json={"query": "any text", "as_of": "2024-06-01", "k": 2})
    assert response.status_code == 422
    assert "Zero-norm" in response.json()["detail"]
