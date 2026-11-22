"""Unit & integration tests for cross-source entity linking and live-to-snapshot KG sync (ADR-00028)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from explorer.workbench_api import create_workbench_app
from ingestion.sec.harvester import EdgarWorker
from ingestion.sec.parser import EdgarEntityLinker
from knowledge_graph.graph_store import store_graph_entities
from knowledge_graph.memgraph_consumer import process_graph_message
from streaming.consumer import process_streaming_event
from tools.build_kg_snapshot import _InMemoryMemgraphCollector, build_and_export_kg_snapshot


def test_store_graph_entities_canonicalizes_entities_and_stamps_news_provenance() -> None:
    driver = _InMemoryMemgraphCollector()

    store_graph_entities(
        source_hash="news_hash_001",
        text="Apple partnered with Beats on spatial audio hardware.",
        graph_data={
            "nodes": [
                {"id": "Apple", "type": "Company", "properties": {}},
                {"id": "Beats", "type": "Company", "properties": {}},
            ],
            "edges": [
                {"source": "Apple", "target": "Beats", "type": "PARTNERED_WITH", "properties": {}},
            ],
        },
        driver=driver,
        published_at="2026-10-15T14:30:00Z",
    )

    assert "Apple Inc." in driver.nodes
    assert "Apple" not in driver.nodes
    assert driver.nodes["Apple Inc."]["props"]["ticker"] == "AAPL"
    assert driver.nodes["Apple Inc."]["props"]["cik"] == "0000320193"

    news_node = driver.nodes["news_hash_001"]
    assert news_node["props"]["published_at"] == "2026-10-15T14:30:00Z"
    assert news_node["props"]["source_type"] == "news"

    edge = driver.edges[("Apple Inc.", "PARTNERED_WITH", "Beats")]
    assert edge["props"]["source_type"] == "news"
    assert edge["props"]["valid_from"] == "2026-10-15"
    assert edge["props"]["is_current"] is True
    assert edge["props"]["is_temporal_unanchored"] is False


def test_store_graph_entities_flags_unanchored_edges_when_published_at_absent() -> None:
    driver = _InMemoryMemgraphCollector()
    store_graph_entities(
        source_hash="news_unanchored",
        text="Apple partnered with Beats.",
        graph_data={
            "nodes": [
                {"id": "Apple", "type": "Company", "properties": {}},
                {"id": "Beats", "type": "Company", "properties": {}},
            ],
            "edges": [
                {"source": "Apple", "target": "Beats", "type": "PARTNERED_WITH", "properties": {}},
            ],
        },
        driver=driver,
        published_at=None,
    )
    edge = driver.edges[("Apple Inc.", "PARTNERED_WITH", "Beats")]
    assert edge["props"]["source_type"] == "news"
    assert edge["props"]["valid_from"] is None
    assert edge["props"]["is_temporal_unanchored"] is True


def test_edgar_worker_stamps_sec_filing_source_type() -> None:
    driver = _InMemoryMemgraphCollector()
    worker = EdgarWorker(pg_conn=None, memgraph_driver=driver)
    worker._persist_subsidiaries(
        parent_cik="0000320193",
        parent_name="Apple Inc.",
        parent_ticker="AAPL",
        subsidiaries=[{"name": "Apple Operations Europe", "jurisdiction": "Ireland"}],
        accession_number="0000320193-25-000079",
        filing_date="2025-10-31",
    )
    sub_edge = driver.edges[("Apple Operations Europe", "SUBSIDIARY_OF", "Apple Inc.")]
    assert sub_edge["props"]["source_type"] == "sec_filing"
    assert sub_edge["props"]["valid_from"] == "2025-10-31"
    assert sub_edge["props"]["is_temporal_unanchored"] is False


def test_consumers_forward_published_at_and_news_source_type() -> None:
    with patch("knowledge_graph.memgraph_consumer.store_graph_entities") as mock_store:
        process_graph_message(
            {
                "source_hash": "h_memgraph",
                "published_at": "2026-09-01T09:00:00Z",
                "raw_payload": json.dumps({"title": "Apple launches update", "summary": "New chipsets"}),
            }
        )
    assert mock_store.call_args.kwargs["published_at"] == "2026-09-01T09:00:00Z"
    assert mock_store.call_args.kwargs["source_type"] == "news"

    driver = _InMemoryMemgraphCollector()
    with patch("streaming.consumer.store_raw", return_value=None):
        ok = process_streaming_event(
            {
                "source": "finnhub",
                "raw_payload": json.dumps({"title": "Apple partners with Sony", "published_at": "2026-09-05T12:00:00Z"}),
                "fetched_at": "2026-09-05T12:05:00Z",
                "source_hash": "h_stream",
            },
            dlt_producer=MagicMock(),
            graph_driver=driver,
            graph_data={
                "nodes": [
                    {"id": "Apple", "type": "Company", "properties": {}},
                    {"id": "Sony", "type": "Company", "properties": {}},
                ],
                "edges": [{"source": "Apple", "target": "Sony", "type": "PARTNERED_WITH", "properties": {}}],
            },
            skip_embedding=True,
        )
    assert ok is True
    assert driver.edges[("Apple Inc.", "PARTNERED_WITH", "Sony")]["props"]["source_type"] == "news"
    assert driver.edges[("Apple Inc.", "PARTNERED_WITH", "Sony")]["props"]["valid_from"] == "2026-09-05"


def test_build_and_export_kg_snapshot_connects_sec_filings_and_news_for_workbench(tmp_path: Path) -> None:
    sec_bundle = tmp_path / "data" / "sec_filings" / "AAPL" / "10K_0000320193-25-000079"
    sec_bundle.mkdir(parents=True, exist_ok=True)
    (sec_bundle / "metadata.json").write_text(
        json.dumps(
            {
                "ticker": "AAPL",
                "accession_number": "0000320193-25-000079",
                "form": "10-K",
                "filing_date": "2025-10-31",
                "company_name": "Apple Inc.",
                "cik": "0000320193",
            }
        ),
        encoding="utf-8",
    )
    (sec_bundle / "subsidiaries.json").write_text(
        json.dumps([{"name": "Apple Operations Europe", "jurisdiction": "Ireland"}]),
        encoding="utf-8",
    )
    (sec_bundle / "graph_data.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "TSMC", "type": "Company", "properties": {"ticker": "TSM"}},
                ],
                "edges": [
                    {"source": "TSMC", "target": "Apple Inc.", "type": "SUPPLIES_TO", "properties": {"share": 0.4}},
                ],
            }
        ),
        encoding="utf-8",
    )

    news_file = tmp_path / "staged_news.jsonl"
    news_file.write_text(
        json.dumps(
            {
                "source_hash": "news_apple_tsmc",
                "published_at": "2026-01-10T08:00:00Z",
                "text": "Apple expanded packaging partnership with TSMC.",
                "graph_data": {
                    "nodes": [
                        {"id": "Apple", "type": "Company", "properties": {}},
                        {"id": "TSMC", "type": "Company", "properties": {"ticker": "TSM"}},
                    ],
                    "edges": [
                        {"source": "Apple", "target": "TSMC", "type": "PARTNERED_WITH", "properties": {}},
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    snapshots_dir = tmp_path / "data" / "snapshots"
    summary = build_and_export_kg_snapshot(
        sec_filings_dir=tmp_path / "data" / "sec_filings",
        snapshots_dir=snapshots_dir,
        snapshot_id="G_sec_test",
        news_file=news_file,
        resolve_entities=False,
        offline=True,
    )
    assert summary["bundles_processed"] == 1
    assert summary["sec_subsidiaries_ingested"] == 1
    assert summary["sec_edges_ingested"] == 1
    assert summary["news_ingested"] == 1
    assert (snapshots_dir / "G_sec_test" / "nodes.parquet").is_file()
    assert (snapshots_dir / "G_sec_test" / "edges.parquet").is_file()

    # Workbench discovers exported snapshot and serves linked SEC + News provenance cleanly
    app = create_workbench_app(project_root=tmp_path)
    client = TestClient(app)
    cat = client.get("/api/catalog").json()
    assert [s["snapshot_id"] for s in cat["snapshots"]] == ["G_sec_test"]

    sub = client.get("/api/subgraph", params={"snapshot_id": "G_sec_test", "ticker": "AAPL", "max_hops": 2}).json()
    source_types = {(e["rel_type"], e.get("source_type")) for e in sub["edges"]}
    assert ("SUPPLIES_TO", "sec_filing") in source_types
    assert ("SUBSIDIARY_OF", "sec_filing") in source_types
    assert ("PARTNERED_WITH", "news") in source_types
