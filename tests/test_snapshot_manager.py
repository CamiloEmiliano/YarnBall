# tests/test_snapshot_manager.py
# -*- coding: utf-8 -*-
"""Unit tests for the Parquet-native SnapshotManager."""

import os
import json
import pytest
import pyarrow.parquet as pq
from pathlib import Path
from unittest.mock import MagicMock, patch

from graph.snapshot_manager import SnapshotManager


@pytest.fixture
def temp_snapshot_dir(tmp_path):
    """Provides a temporary directory for snapshot tests."""
    return tmp_path / "snapshots"


def test_snapshot_manager_init(temp_snapshot_dir):
    """Test SnapshotManager initialization and directory creation."""
    mgr = SnapshotManager(base_dir=temp_snapshot_dir)
    assert temp_snapshot_dir.exists()
    assert mgr.base_dir == temp_snapshot_dir


def test_export_snapshot_creates_parquet_files(temp_snapshot_dir):
    """Test that export_snapshot queries Memgraph and outputs valid Parquet files."""
    mgr = SnapshotManager(base_dir=temp_snapshot_dir)

    mock_nodes = [
        {"labels": ["Company"], "props": {"id": "Apple Inc.", "ticker": "AAPL", "sector": "Tech"}},
        {"labels": ["Company"], "props": {"id": "NVIDIA", "ticker": "NVDA", "sector": "Semis"}},
    ]
    mock_edges = [
        {
            "source_id": "Apple Inc.",
            "target_id": "NVIDIA",
            "rel_type": "PARTNERED_WITH",
            "props": {"context": "AI chips", "source_hash": "hash123"},
            "source_hash": "hash123",
        }
    ]

    mock_session = MagicMock()
    # First call returns nodes, second call returns edges
    mock_session.run.side_effect = [mock_nodes, mock_edges]
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    with patch("graph.snapshot_manager.get_memgraph_driver", return_value=mock_driver), \
         patch("graph.snapshot_manager.pg_connection") as mock_pg_conn:
        
        mock_conn = MagicMock()
        mock_pg_conn.return_value.__enter__.return_value = mock_conn

        res = mgr.export_snapshot(
            snapshot_id="test_snapshot_v1",
            tag="raw",
            description="Test raw graph",
            metadata={"source": "pytest"}
        )

        assert res["snapshot_id"] == "test_snapshot_v1"
        assert res["tag"] == "raw"
        assert res["node_count"] == 2
        assert res["edge_count"] == 1

        # Verify Parquet files on disk
        snapshot_path = temp_snapshot_dir / "test_snapshot_v1"
        nodes_file = snapshot_path / "nodes.parquet"
        edges_file = snapshot_path / "edges.parquet"

        assert nodes_file.exists()
        assert edges_file.exists()

        node_tbl = pq.read_table(nodes_file)
        edge_tbl = pq.read_table(edges_file)

        assert len(node_tbl) == 2
        assert len(edge_tbl) == 1

        node_rows = node_tbl.to_pylist()
        assert node_rows[0]["id"] == "Apple Inc."
        assert node_rows[0]["ticker"] == "AAPL"
        assert json.loads(node_rows[0]["properties_json"]) == {"sector": "Tech"}

        edge_rows = edge_tbl.to_pylist()
        assert edge_rows[0]["source_id"] == "Apple Inc."
        assert edge_rows[0]["target_id"] == "NVIDIA"
        assert edge_rows[0]["rel_type"] == "PARTNERED_WITH"


def test_restore_snapshot_executes_cypher_unwind(temp_snapshot_dir):
    """Test that restore_snapshot reads Parquet files and calls UNWIND queries."""
    mgr = SnapshotManager(base_dir=temp_snapshot_dir)

    # First export a snapshot
    mock_nodes = [
        {"labels": ["Company"], "props": {"id": "Microsoft", "ticker": "MSFT"}},
    ]
    mock_edges = [
        {
            "source_id": "Microsoft",
            "target_id": "OpenAI",
            "rel_type": "INVESTS_IN",
            "props": {"amount": "10B"},
            "source_hash": "hash_msft",
        }
    ]

    mock_session = MagicMock()
    mock_session.run.side_effect = [mock_nodes, mock_edges, None, None, None]
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    with patch("graph.snapshot_manager.get_memgraph_driver", return_value=mock_driver), \
         patch("graph.snapshot_manager.pg_connection") as mock_pg_conn:
        mock_pg_conn.return_value.__enter__.return_value = MagicMock()

        mgr.export_snapshot("G_test", "raw", "Test")

        # Now restore
        restore_res = mgr.restore_snapshot("G_test")
        assert restore_res["snapshot_id"] == "G_test"
        assert restore_res["status"] == "restored"
        assert restore_res["node_count"] == 1
        assert restore_res["edge_count"] == 1

        # Check that DETACH DELETE was called
        calls = [call_args[0][0] for call_args in mock_session.run.call_args_list]
        assert any("DETACH DELETE" in str(c) for c in calls)
        assert any("UNWIND $batch AS row" in str(c) for c in calls)


def test_restore_missing_snapshot_raises_error(temp_snapshot_dir):
    """Test that restoring a non-existent snapshot raises FileNotFoundError."""
    mgr = SnapshotManager(base_dir=temp_snapshot_dir)
    with pytest.raises(FileNotFoundError):
        mgr.restore_snapshot("non_existent_snapshot")


def test_list_snapshots_disk_fallback(temp_snapshot_dir):
    """Test list_snapshots discovers snapshots on disk when DB is offline."""
    mgr = SnapshotManager(base_dir=temp_snapshot_dir)

    # Export a snapshot
    mock_nodes = [{"labels": ["Company"], "props": {"id": "Tesla", "ticker": "TSLA"}}]
    mock_session = MagicMock()
    mock_session.run.side_effect = [mock_nodes, []]
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    with patch("graph.snapshot_manager.get_memgraph_driver", return_value=mock_driver), \
         patch("graph.snapshot_manager.pg_connection", side_effect=Exception("DB down")):
        mgr.export_snapshot("G_raw_tsla", "raw", "Tesla snapshot")

        snapshots = mgr.list_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0]["snapshot_id"] == "G_raw_tsla"
        assert snapshots[0]["node_count"] == 1

        single = mgr.get_snapshot("G_raw_tsla")
        assert single is not None
        assert single["snapshot_id"] == "G_raw_tsla"
