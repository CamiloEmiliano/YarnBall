# tests/test_entity_resolver.py
# -*- coding: utf-8 -*-
"""Unit tests for the EntityResolver module."""

import json
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from unittest.mock import MagicMock, patch

from graph.entity_resolver import EntityResolver
from graph.snapshot_manager import SnapshotManager


@pytest.fixture
def temp_resolver(tmp_path):
    """Provides an EntityResolver with an isolated temporary snapshot directory."""
    snapshot_dir = tmp_path / "snapshots"
    mgr = SnapshotManager(base_dir=snapshot_dir)
    return EntityResolver(snapshot_manager=mgr)


def test_resolve_nodes_exact_ticker(temp_resolver):
    """Test clustering nodes with matching ticker symbols."""
    raw_nodes = [
        {"id": "Apple", "label": "Company", "ticker": "AAPL", "source_hash": "hash1", "properties_json": "{}"},
        {"id": "Apple Inc.", "label": "Company", "ticker": "AAPL", "source_hash": "hash2", "properties_json": '{"sector": "Tech"}'},
        {"id": "Microsoft", "label": "Company", "ticker": "MSFT", "source_hash": "hash3", "properties_json": "{}"},
    ]

    mapping, resolved = temp_resolver.resolve_nodes(raw_nodes)

    assert mapping["Apple"] == "Apple Inc."
    assert mapping["Apple Inc."] == "Apple Inc."
    assert mapping["Microsoft"] == "Microsoft"

    assert len(resolved) == 2
    apple_node = next(n for n in resolved if n["id"] == "Apple Inc.")
    assert apple_node["ticker"] == "AAPL"
    props = json.loads(apple_node["properties_json"])
    assert "Apple" in props["aliases"]
    assert props["mention_count"] == 2
    assert "hash1" in props["source_hashes"]
    assert "hash2" in props["source_hashes"]


def test_resolve_nodes_fuzzy_name_matching(temp_resolver):
    """Test clustering nodes based on Jaro-Winkler / suffix normalization."""
    raw_nodes = [
        {"id": "NVIDIA Corporation", "label": "Company", "ticker": "NVDA", "source_hash": "h1"},
        {"id": "NVIDIA Corp", "label": "Company", "ticker": None, "source_hash": "h2"},
    ]

    mapping, resolved = temp_resolver.resolve_nodes(raw_nodes)

    assert mapping["NVIDIA Corp"] == "NVIDIA Corporation"
    assert len(resolved) == 1
    assert resolved[0]["id"] == "NVIDIA Corporation"
    assert resolved[0]["ticker"] == "NVDA"


def test_rewire_edges_remaps_and_drops_self_loops(temp_resolver):
    """Test that edge endpoints are remapped and internal self-loops are dropped."""
    node_mapping = {
        "Apple": "Apple Inc.",
        "Apple Inc.": "Apple Inc.",
        "TSMC": "Taiwan Semiconductor",
    }

    raw_edges = [
        # Normal edge
        {"source_id": "Apple", "target_id": "TSMC", "rel_type": "PARTNERED_WITH", "source_hash": "h1"},
        # Self-loop after resolution
        {"source_id": "Apple", "target_id": "Apple Inc.", "rel_type": "PARTNERED_WITH", "source_hash": "h2"},
    ]

    rewired = temp_resolver.rewire_edges(raw_edges, node_mapping)

    assert len(rewired) == 1
    edge = rewired[0]
    assert edge["source_id"] == "Apple Inc." or edge["target_id"] == "Apple Inc."
    assert edge["source_id"] == "Taiwan Semiconductor" or edge["target_id"] == "Taiwan Semiconductor"


def test_rewire_edges_canonicalizes_symmetric_relations(temp_resolver):
    """Test that symmetric edges (e.g. COMPETES_WITH) are ordered canonically and deduplicated."""
    node_mapping = {
        "MSFT": "Microsoft",
        "AAPL": "Apple Inc.",
    }

    raw_edges = [
        {"source_id": "AAPL", "target_id": "MSFT", "rel_type": "COMPETES_WITH", "source_hash": "h1"},
        {"source_id": "MSFT", "target_id": "AAPL", "rel_type": "COMPETES_WITH", "source_hash": "h2"},
    ]

    rewired = temp_resolver.rewire_edges(raw_edges, node_mapping)

    # Should be deduplicated into a single edge with evidence_count = 2
    assert len(rewired) == 1
    edge = rewired[0]
    assert edge["source_id"] == "Apple Inc."
    assert edge["target_id"] == "Microsoft"
    assert edge["rel_type"] == "COMPETES_WITH"

    props = json.loads(edge["properties_json"])
    assert props["evidence_count"] == 2
    assert "h1" in props["source_hashes"]
    assert "h2" in props["source_hashes"]


def test_resolve_snapshot_full_pipeline(temp_resolver, tmp_path):
    """Test full end-to-end resolution of a snapshot directory."""
    mgr = temp_resolver.snapshot_mgr
    raw_dir = mgr.base_dir / "G_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Write mock G_raw Parquet files
    raw_nodes = [
        {"id": "Tesla Motors", "label": "Company", "ticker": "TSLA", "source_hash": "h1", "properties_json": "{}"},
        {"id": "Tesla Inc.", "label": "Company", "ticker": "TSLA", "source_hash": "h2", "properties_json": "{}"},
        {"id": "Panasonic", "label": "Company", "ticker": "PCRFY", "source_hash": "h3", "properties_json": "{}"},
    ]
    raw_edges = [
        {"source_id": "Tesla Motors", "target_id": "Panasonic", "rel_type": "PARTNERED_WITH", "source_hash": "h1", "properties_json": "{}"},
        {"source_id": "Panasonic", "target_id": "Tesla Inc.", "rel_type": "PARTNERED_WITH", "source_hash": "h2", "properties_json": "{}"},
    ]

    node_schema = pa.schema([
        ("id", pa.string()), ("label", pa.string()), ("ticker", pa.string()),
        ("source_hash", pa.string()), ("properties_json", pa.string()),
    ])
    edge_schema = pa.schema([
        ("source_id", pa.string()), ("target_id", pa.string()), ("rel_type", pa.string()),
        ("source_hash", pa.string()), ("properties_json", pa.string()),
    ])

    pq.write_table(pa.Table.from_pylist(raw_nodes, schema=node_schema), raw_dir / "nodes.parquet")
    pq.write_table(pa.Table.from_pylist(raw_edges, schema=edge_schema), raw_dir / "edges.parquet")

    with patch("graph.entity_resolver.pg_connection") as mock_pg:
        mock_pg.return_value.__enter__.return_value = MagicMock()
        res = temp_resolver.resolve_snapshot("G_raw", "G_resolved")

        assert res["status"] == "resolved"
        assert res["raw_nodes"] == 3
        assert res["resolved_nodes"] == 2  # Tesla merged into 1
        assert res["raw_edges"] == 2
        assert res["resolved_edges"] == 1  # 2 partner edges deduplicated into 1

        resolved_dir = mgr.base_dir / "G_resolved"
        assert (resolved_dir / "nodes.parquet").exists()
        assert (resolved_dir / "edges.parquet").exists()
