"""Unit tests for unified SEC + news dense vector coverage and EMBEDDING_DIM safety (ADR-00022)."""

from __future__ import annotations

from datetime import date
from typing import Any, List, Tuple
from unittest.mock import MagicMock, patch

import pytest

import embedding.embedder as embedder
from ingestion.sec.harvester import EdgarWorker
from retrieval.hybrid import HybridRetriever
from retrieval.temporal import PiTContext
from streaming.consumer import run_consumer
from tools import init_db


class _SchemaCursor:
    def __init__(self, executed: List[Tuple[str, Any]], meta_row: Any = None) -> None:
        self.executed = executed
        self._meta_row = meta_row
        self._last_sql = ""

    def execute(self, sql: str, params: Any = None) -> None:
        self._last_sql = sql
        self.executed.append((sql, params))

    def fetchone(self) -> Any:
        if "embedding_schema_metadata" in self._last_sql:
            return self._meta_row
        return None

    def close(self) -> None:
        return None


class _SchemaConn:
    def __init__(self, meta_row: Any = None) -> None:
        self.executed: List[Tuple[str, Any]] = []
        self._meta_row = meta_row

    def cursor(self) -> _SchemaCursor:
        return _SchemaCursor(self.executed, meta_row=self._meta_row)

    def commit(self) -> None:
        return None

    def close(self) -> None:
        return None


def test_edgar_worker_emits_sec_entity_and_subsidiary_embeddings() -> None:
    mock_driver = MagicMock()
    worker = EdgarWorker(memgraph_driver=mock_driver, pg_conn=None)
    worker.linker.link_entity = MagicMock(return_value=None)

    emitted_calls: List[Tuple[List[dict[str, Any]], str]] = []

    def _capture_emit(nodes: List[dict[str, Any]], source_hash: str = "") -> None:
        emitted_calls.append((list(nodes), source_hash))

    with patch("knowledge_graph.embedding_store.store_node_embeddings", side_effect=_capture_emit):
        worker._persist_sec_graph_data(
            primary_name="Apple Inc.",
            primary_ticker="AAPL",
            graph_data={
                "nodes": [
                    {"id": "TSMC", "type": "Company", "properties": {"ticker": "TSM"}},
                ],
                "edges": [
                    {"source": "TSMC", "target": "Apple Inc.", "type": "SUPPLIES_TO", "properties": {}},
                ],
            },
            accession_number="0000320193-26-000010",
            filing_date="2026-10-30",
            form_type="10-K",
            section="Item 1 Business",
        )
        worker._persist_subsidiaries(
            parent_cik="0000320193",
            parent_name="Apple Inc.",
            parent_ticker="AAPL",
            subsidiaries=[{"name": "Apple Operations Europe", "jurisdiction": "Ireland"}],
            accession_number="0000320193-26-000010",
            filing_date="2026-10-30",
        )

    assert len(emitted_calls) == 2
    graph_nodes, graph_hash = emitted_calls[0]
    assert graph_hash == "0000320193-26-000010:Item 1 Business"
    assert [n["id"] for n in graph_nodes] == ["Apple Inc.", "TSMC"]

    sub_nodes, sub_hash = emitted_calls[1]
    assert sub_hash == "0000320193-26-000010:Exhibit21"
    assert [n["id"] for n in sub_nodes] == ["Apple Inc.", "Apple Operations Europe"]


def test_search_dense_passages_enforces_pit_cutoff_and_links_news_entities() -> None:
    executed_queries: List[Tuple[str, Any]] = []

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()

    def _record_execute(sql: str, params: Any = None) -> None:
        executed_queries.append((sql, params))

    mock_cursor.execute.side_effect = _record_execute
    mock_cursor.fetchall.return_value = [
        (
            "hash_euv_1",
            "EUV Lithography Export Curb",
            "https://example.com/euv",
            "2024-06-15 12:00:00+00:00",
            "Export controls restrict extreme ultraviolet lithography systems.",
            "Reuters",
            0.93,
        )
    ]
    mock_conn.cursor.return_value = mock_cursor

    mock_cql = MagicMock()
    mock_cql.execute_query.return_value = {"records": []}
    retriever = HybridRetriever(text_to_cql_engine=mock_cql, top_k_dense=4)

    mock_memgraph_session = MagicMock()
    mock_memgraph_session.run.side_effect = [
        [{"entity_id": "ASML"}],
        [
            {
                "source": "ASML",
                "rel_type": "SUPPLIES_TO",
                "target": "TSMC",
                "props": {},
                "source_hash": "hash_euv_1",
            }
        ],
    ]
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_memgraph_session

    ctx = PiTContext(as_of=date(2024, 7, 1), mode="historical")
    with patch("retrieval.hybrid.pg_connection", return_value=mock_conn), \
         patch("retrieval.hybrid.get_memgraph_driver", return_value=mock_driver), \
         patch("embedding.embedder.embed_texts", return_value=[[0.6, 0.8]]):
        res = retriever.retrieve("EUV lithography export restriction", ctx=ctx)

    assert len(res["dense_passages"]) == 1
    assert res["dense_passages"][0]["source_hash"] == "hash_euv_1"
    assert "ASML" in [t["source"] for t in res["subgraph_triples"]]
    assert "hash_euv_1" in res["source_hashes"]

    passage_sql, passage_params = next(
        (sql, params) for sql, params in executed_queries if "FROM financial_news_queue" in sql and "embedding IS NOT NULL" in sql
    )
    assert "published_at IS NOT NULL" in passage_sql
    assert "published_at <= %s::timestamptz" in passage_sql
    assert passage_params == ("[0.6,0.8]", ctx.cutoff_utc_iso, "[0.6,0.8]", 4)


def test_configurable_embedding_dim_ddl_and_startup_mismatch_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedder.reset_embedder_cache()
    monkeypatch.setenv("EMBEDDING_DIM", "384")
    monkeypatch.setenv("YARNBALL_EMBEDDER_MOCK", "1")

    conn = _SchemaConn(meta_row=None)
    with patch("tools.init_db._connect_db", return_value=conn):
        init_db.create_queue_table("financial_rag")

    sql_blob = "\n".join(sql for sql, _ in conn.executed)
    assert "vector(384)" in sql_blob
    assert "vector(768)" not in sql_blob

    # Existing schema recorded dim=768 must fail loudly when EMBEDDING_DIM=384
    mismatched_conn = _SchemaConn(meta_row=("sentence-transformers/all-mpnet-base-v2", 768))
    with patch("tools.init_db._connect_db", return_value=mismatched_conn):
        with pytest.raises(ValueError, match="dimension mismatch"):
            init_db.create_queue_table("financial_rag")

    # Worker startup fails if active embedder output dim disagrees with EMBEDDING_DIM
    with patch("embedding.embedder.embed_texts", return_value=[[0.5] * 768]):
        with pytest.raises(embedder.InvalidEmbeddingError, match="dimension mismatch"):
            run_consumer()
