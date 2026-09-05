import os
import pytest
from unittest.mock import patch
from graph.memgraph_driver import _DummyDriver

# Mock the memgraph driver globally for these tests to avoid connecting to live Memgraph container
_driver_patcher = patch('graph.memgraph_driver.get_memgraph_driver', return_value=_DummyDriver())
_driver_patcher.start()

from fastapi.testclient import TestClient
from rag.retriever.main import app

client = TestClient(app)

@pytest.fixture(scope="module")
def insert_test_embeddings():
    # Insert two dummy node embeddings into PGVECTOR
    from graph.db import pg_connection
    sql = """
        INSERT INTO node_embeddings (node_id, entity_type, embedding, source_hash)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (node_id) DO UPDATE
        SET embedding = EXCLUDED.embedding;
    """
    dummy_vec = [0.0] * 768
    with pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM node_embeddings;")
            cur.execute(sql, ("TestNode1", "Entity", dummy_vec, "test"))
            cur.execute(sql, ("TestNode2", "Entity", dummy_vec, "test"))
        conn.commit()
    yield
    # Cleanup after tests
    with pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM node_embeddings WHERE node_id IN ('TestNode1','TestNode2');")
        conn.commit()


def test_search_endpoint(insert_test_embeddings):
    # Mock embed_texts to return a zero vector (so the k‑NN will match our test rows)
    with patch('embedding.embedder.embed_texts', return_value=[[0.0] * 768]):
        response = client.post("/search", json={"query": "any text", "k": 2})
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


def teardown_module(module):
    _driver_patcher.stop()
