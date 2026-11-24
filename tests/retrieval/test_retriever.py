import os
import pytest
from unittest.mock import patch
from knowledge_graph.memgraph_driver import _DummyDriver

# Mock the memgraph driver globally for these tests to avoid connecting to live Memgraph container
_driver_patcher = patch('knowledge_graph.memgraph_driver.get_memgraph_driver', return_value=_DummyDriver())
_driver_patcher.start()

from fastapi.testclient import TestClient
from retrieval.service import app

client = TestClient(app)

@pytest.fixture(scope="module")
def insert_test_embeddings():
    # Insert two dummy node embeddings into PGVECTOR if connection is available
    from knowledge_graph.db import pg_connection
    conn_ctx = pg_connection()
    if conn_ctx is None:
        pytest.skip("PostgreSQL connection not available for test_retriever")
    create_sql = """
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS node_embeddings (
            node_id TEXT NOT NULL,
            valid_from DATE NOT NULL,
            valid_to DATE NULL,
            entity_type TEXT,
            embedding VECTOR(768) NOT NULL,
            source_hash TEXT,
            known_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ DEFAULT now(),
            PRIMARY KEY (node_id, valid_from)
        );
    """
    sql = """
        INSERT INTO node_embeddings (node_id, valid_from, valid_to, entity_type, embedding, source_hash)
        VALUES (%s, %s::date, NULL, %s, %s, %s)
        ON CONFLICT (node_id, valid_from) DO UPDATE
        SET embedding = EXCLUDED.embedding;
    """
    dummy_vec = [1.0] + [0.0] * 767
    with conn_ctx as conn:
        if conn is None:
            pytest.skip("PostgreSQL connection not available for test_retriever")
        with conn.cursor() as cur:
            cur.execute(create_sql)
            cur.execute("DELETE FROM node_embeddings;")
            cur.execute(sql, ("TestNode1", "2024-01-02", "Entity", dummy_vec, "test"))
            cur.execute(sql, ("TestNode2", "2024-01-02", "Entity", dummy_vec, "test"))
        conn.commit()
    yield
    # Cleanup after tests
    cleanup_ctx = pg_connection()
    if cleanup_ctx is not None:
        with cleanup_ctx as conn:
            if conn is not None:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM node_embeddings WHERE node_id IN ('TestNode1','TestNode2');")
                conn.commit()


def test_search_endpoint(insert_test_embeddings):
    # Mock embed_texts to return a non-zero unit vector (ADR-00020)
    with patch('embedding.embedder.embed_texts', return_value=[[1.0] + [0.0] * 767]):
        response = client.post("/search", json={"query": "any text", "as_of": "2024-06-01", "k": 2})
    assert response.status_code == 200
    payload = response.json()
    # Should return a list with enriched nodes for each of the two test IDs
    assert isinstance(payload, list)
    returned_ids = {entry["root_id"] for entry in payload}
    assert returned_ids == {"TestNode1", "TestNode2"}
    for entry in payload:
        # The enrichment returns at least the node itself in "nodes"
        assert "nodes" in entry
        assert any(node.get("id") == entry["root_id"] for node in entry["nodes"])


def test_search_rejects_zero_vector_adr00020():
    with patch('embedding.embedder.embed_texts', return_value=[[0.0] * 768]):
        response = client.post("/search", json={"query": "any text", "as_of": "2024-06-01", "k": 2})
    assert response.status_code == 422
    assert "Zero-norm" in response.json()["detail"]


def teardown_module(module):
    _driver_patcher.stop()
