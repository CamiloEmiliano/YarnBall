import os
import pytest
import psycopg2
from psycopg2.extras import RealDictCursor

# Module under test
from graph.embedding_store import store_node_embeddings

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------
@pytest.fixture(scope="session")
def pg_conn():
    dsn = os.getenv("POSTGRES_URL")
    conn = psycopg2.connect(dsn, cursor_factory=RealDictCursor)
    # Ensure the pgvector extension and table exist
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE EXTENSION IF NOT EXISTS vector;
            CREATE TABLE IF NOT EXISTS node_embeddings (
                node_id TEXT PRIMARY KEY,
                entity_type TEXT,
                embedding VECTOR(768) NOT NULL,
                source_hash TEXT,
                created_at TIMESTAMPTZ DEFAULT now()
            );
            """
        )
        conn.commit()
    yield conn
    # Cleanup after the entire test session
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS node_embeddings;")
        conn.commit()
    conn.close()


def fetch_embedding(conn, node_id):
    """Helper to pull a single row from the embedding table for assertions."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT node_id, entity_type, embedding, source_hash FROM node_embeddings WHERE node_id = %s;",
            (node_id, ),
        )
        row = cur.fetchone()
        if row and "embedding" in row and isinstance(row["embedding"], str):
            import json
            row["embedding"] = json.loads(row["embedding"])
        return row

# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_store_node_embeddings_basic(pg_conn):
    # Arrange – minimal node payloads
    nodes = [
        {"id": "CompanyA", "type": "Company"},
        {"id": "PersonB", "type": "Person"},
    ]

    # Act
    store_node_embeddings(nodes, source_hash="testhash123")

    # Assert – each node persisted with a 768‑dim vector
    for node in nodes:
        row = fetch_embedding(pg_conn, node["id"])
        assert row is not None, f"{node['id']} not found in PG"
        assert row["entity_type"] == node["type"]
        # psycopg2 returns the vector as a list‑like object
        assert len(row["embedding"]) == 768
        assert row["source_hash"] == "testhash123"


def test_upsert_updates_existing_row(pg_conn):
    node = {"id": "CompanyX", "type": "Company"}

    # First insert
    store_node_embeddings([node], source_hash="first")
    first_row = fetch_embedding(pg_conn, node["id"])
    assert first_row["source_hash"] == "first"

    # Second insert with different hash – should UPDATE, not duplicate
    store_node_embeddings([node], source_hash="second")
    second_row = fetch_embedding(pg_conn, node["id"])
    assert second_row["source_hash"] == "second"

    # Verify only a single row exists for that node_id
    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM node_embeddings WHERE node_id = %s;",
            (node["id"], ),
        )
        count = cur.fetchone()["count"]
    assert count == 1
