"""Unit tests for Dynamic In-Memory Offline Subgraph Store (ADR-00015)."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from retrieval.offline_store import (
    CypherSubgraphQuery,
    InMemorySubgraphStore,
    parse_cypher_subgraph_query,
)
from retrieval.temporal import TemporalQueryFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _write_snapshot_parquet(
    snapshot_dir: Path,
    nodes: list[dict],
    edges: list[dict],
) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    node_schema = pa.schema([
        ("id", pa.string()),
        ("label", pa.string()),
        ("ticker", pa.string()),
        ("source_hash", pa.string()),
        ("properties_json", pa.string()),
    ])
    edge_schema = pa.schema([
        ("source_id", pa.string()),
        ("target_id", pa.string()),
        ("rel_type", pa.string()),
        ("source_hash", pa.string()),
        ("properties_json", pa.string()),
    ])
    pq.write_table(pa.Table.from_pylist(nodes, schema=node_schema), snapshot_dir / "nodes.parquet")
    pq.write_table(pa.Table.from_pylist(edges, schema=edge_schema), snapshot_dir / "edges.parquet")
    return snapshot_dir


def test_parse_cypher_subgraph_query_extracts_scope_hops_and_pit():
    parsed = parse_cypher_subgraph_query(
        "MATCH (s:Company {ticker: 'NVDA'})-[r:SUPPLIES_TO*1..3]->(t:Company) "
        "WHERE coalesce(r.status, 'ACTIVE_CURRENT') = 'ACTIVE_CURRENT' "
        "AND (r.valid_from IS NULL OR r.valid_from <= '2024-06-15') RETURN s, r, t;"
    )
    assert isinstance(parsed, CypherSubgraphQuery)
    assert parsed.is_read_only is True
    assert parsed.has_match is True
    assert parsed.root_ticker == "NVDA"
    assert parsed.rel_types == ("SUPPLIES_TO",)
    assert parsed.max_hops == 3
    assert parsed.as_of_date == "2024-06-15"
    assert parsed.required_status == "ACTIVE_CURRENT"

    # Parameterized ticker and $as_of_date resolution
    param_parsed = parse_cypher_subgraph_query(
        "MATCH (s:Company {ticker: $ticker})-[r:SUPPLIES_TO|PARTNERED_WITH*2]->(t:Company) RETURN s, r, t",
        params={"ticker": "qcom", "as_of_date": "2025-01-10T12:00:00Z"},
    )
    assert param_parsed.root_ticker == "QCOM"
    assert param_parsed.rel_types == ("SUPPLIES_TO", "PARTNERED_WITH")
    assert param_parsed.max_hops == 2
    assert param_parsed.as_of_date == "2025-01-10"

    # Mutating Cypher statements must be refused
    for unsafe in (
        "CREATE (n:Company {ticker: 'EVIL'})",
        "MATCH (s:Company) MERGE (s)-[:SUPPLIES_TO]->(t:Company)",
        "MATCH (s:Company)-[r]->(t) DELETE r",
    ):
        assert parse_cypher_subgraph_query(unsafe).is_read_only is False


def test_bfs_root_hop_isolation_and_cycle_safety():
    store = InMemorySubgraphStore([
        {"source": "NVDA", "target": "OEM_A", "rel_type": "SUPPLIES_TO", "share": 0.40, "buffer_days": 15.0},
        {"source": "OEM_A", "target": "CLOUD_B", "rel_type": "SUPPLIES_TO", "share": 0.30, "buffer_days": 30.0},
        {"source": "CLOUD_B", "target": "NVDA", "rel_type": "SUPPLIES_TO", "share": 0.10, "buffer_days": 60.0},
        {"source": "INTC", "target": "DELL", "rel_type": "SUPPLIES_TO", "share": 0.50, "buffer_days": 20.0},
        {"source": "NVDA", "target": "AMD", "rel_type": "COMPETES_WITH", "share": 0.25},
    ])

    with store.session() as session:
        hop1 = session.run("MATCH (s:Company {ticker: 'NVDA'})-[r:SUPPLIES_TO*1..1]->(t:Company) RETURN s, r, t")
        assert len(hop1) == 1
        assert hop1[0]["target"] == "OEM_A"
        assert hop1[0]["hop"] == 1

        hop3 = session.run("MATCH (s:Company {ticker: 'NVDA'})-[r:SUPPLIES_TO*1..3]->(t:Company) RETURN s, r, t")
        targets = {(r["source"], r["target"], r["hop"]) for r in hop3}
        assert ("NVDA", "OEM_A", 1) in targets
        assert ("OEM_A", "CLOUD_B", 2) in targets
        assert ("CLOUD_B", "NVDA", 3) in targets
        # Must exclude unrelated INTC->DELL and non-SUPPLIES_TO NVDA->AMD
        assert all(r["source"] != "INTC" and r["rel_type"] == "SUPPLIES_TO" for r in hop3)

        # Write query no-ops safely
        assert session.run("MATCH (s)-[r]->(t) DELETE r") == []


def test_pit_and_lifecycle_filtering_with_temporal_query_filter():
    store = InMemorySubgraphStore([
        {
            "source": "TSM",
            "target": "AAPL",
            "rel_type": "SUPPLIES_TO",
            "nature": "legacy_soc",
            "status": "ACTIVE_CURRENT",
            "valid_from": "2019-01-01",
            "valid_to": "2022-12-31",
            "share": 0.30,
        },
        {
            "source": "TSM",
            "target": "NVDA",
            "rel_type": "SUPPLIES_TO",
            "nature": "ai_accelerators",
            "status": "ACTIVE_CURRENT",
            "valid_from": "2023-06-01",
            "valid_to": None,
            "share": 0.45,
        },
        {
            "source": "TSM",
            "target": "QCOM",
            "rel_type": "SUPPLIES_TO",
            "nature": "terminated_contract",
            "status": "TERMINATED",
            "valid_from": "2020-01-01",
            "valid_to": "2026-12-31",
            "share": 0.20,
        },
    ])

    tf = TemporalQueryFilter()
    base_cypher = "MATCH (s:Company {ticker: 'TSM'})-[r:SUPPLIES_TO*1..2]->(t:Company) RETURN s, r, t"

    # As of 2021-05-01: only legacy AAPL contract is active & in-window (NVDA is future, QCOM is terminated)
    c_2021 = tf.inject_active_coalescence(base_cypher, as_of_date="2021-05-01")
    with store.session() as session:
        res_2021 = session.run(c_2021, as_of_date="2021-05-01")
        assert [r["target"] for r in res_2021] == ["AAPL"]

        # As of 2024-03-01: AAPL expired, NVDA active
        c_2024 = tf.inject_active_coalescence(base_cypher, as_of_date="2024-03-01")
        res_2024 = session.run(c_2024, as_of_date="2024-03-01")
        assert [r["target"] for r in res_2024] == ["NVDA"]

        # Audit filter inspects TERMINATED edges cleanly
        c_audit = tf.inject_audit_filter(base_cypher, target_status="TERMINATED", as_of_date="2024-03-01")
        res_audit = session.run(c_audit, as_of_date="2024-03-01")
        assert [r["target"] for r in res_audit] == ["QCOM"]


def test_from_parquet_snapshot_synthetic_and_repo_g_resolved(tmp_path):
    snap_dir = _write_snapshot_parquet(
        tmp_path / "snap_2024",
        nodes=[
            {"id": "fab_1", "label": "Company", "ticker": "INTC", "source_hash": None, "properties_json": "{}"},
            {"id": "oem_1", "label": "Company", "ticker": "DELL", "source_hash": None, "properties_json": "{}"},
            {"id": "oem_2", "label": "Company", "ticker": "HPQ", "source_hash": None, "properties_json": "{}"},
        ],
        edges=[
            {
                "source_id": "fab_1",
                "target_id": "oem_1",
                "rel_type": "SUPPLIES_TO",
                "source_hash": "hash1",
                "properties_json": json.dumps({
                    "nature": "server_cpus",
                    "share": 0.35,
                    "buffer_days": 30.0,
                    "valid_from": "2023-01-01",
                }),
            },
            {
                "source_id": "oem_1",
                "target_id": "oem_2",
                "rel_type": "SUPPLIES_TO",
                "source_hash": "hash2",
                "properties_json": json.dumps({
                    "nature": "racks",
                    "exposure_pct": 25.0,
                    "buffer_days": 60.0,
                }),
            },
        ],
    )
    store = InMemorySubgraphStore.from_parquet_snapshot(snap_dir)
    with store.session() as session:
        rows = session.run(
            "MATCH (s:Company {ticker: 'INTC'})-[r:SUPPLIES_TO*1..3]->(t:Company) RETURN s, r, t",
            as_of_date="2024-08-01",
        )
        assert [(r["source"], r["target"], r["hop"], r["exposure_pct"]) for r in rows] == [
            ("INTC", "DELL", 1, 35.0),
            ("DELL", "HPQ", 2, 25.0),
        ]

    # Verify compatibility with actual repository `data/snapshots/G_resolved` artifact if present
    repo_resolved = PROJECT_ROOT / "data" / "snapshots" / "G_resolved"
    if (repo_resolved / "edges.parquet").is_file():
        resolved_store = InMemorySubgraphStore.from_parquet_snapshot(repo_resolved)
        with resolved_store.session() as session:
            asml_chain = session.run(
                "MATCH (s:Company {ticker: 'ASML'})-[r:SUPPLIES_TO*1..3]->(t:Company) RETURN s, r, t"
            )
            assert [(r["source"], r["target"], r["hop"]) for r in asml_chain] == [
                ("ASML", "TSM", 1),
                ("TSM", "AAPL", 2),
            ]


def test_from_edges_file_json_and_jsonl(tmp_path):
    edges = [
        {"source": "MU", "target": "NVDA", "rel_type": "SUPPLIES_TO", "nature": "hbm3e", "share": 0.30, "buffer_days": 20.0},
        {"source": "NVDA", "target": "MSFT", "rel_type": "SUPPLIES_TO", "nature": "gpus", "share": 0.40, "buffer_days": 45.0},
    ]

    json_file = tmp_path / "edges.json"
    json_file.write_text(json.dumps({"edges": edges}), encoding="utf-8")
    store_json = InMemorySubgraphStore.from_edges_file(json_file)
    with store_json.session() as session:
        res = session.run("MATCH (s:Company {ticker: 'MU'})-[r:SUPPLIES_TO*1..2]->(t:Company) RETURN s, r, t")
        assert len(res) == 2

    jsonl_file = tmp_path / "edges.jsonl"
    jsonl_file.write_text("\n".join(json.dumps(e) for e in edges) + "\n", encoding="utf-8")
    store_jsonl = InMemorySubgraphStore.from_edges_file(jsonl_file)
    with store_jsonl.session() as session:
        res = session.run("MATCH (s:Company {ticker: 'MU'})-[r:SUPPLIES_TO*1..1]->(t:Company) RETURN s, r, t")
        assert len(res) == 1
        assert res[0]["target"] == "NVDA"

    with pytest.raises(FileNotFoundError):
        InMemorySubgraphStore.from_edges_file(tmp_path / "missing.json")
