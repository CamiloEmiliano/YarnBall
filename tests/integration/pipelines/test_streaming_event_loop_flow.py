"""System Integration Test (SIT) Flow 4: Streaming Event Loop & Incremental Graph Update.

Academic Reference & Governance:
- ADR-00003: SIT Flow 4 Streaming Event Graph Update
- Distributed Messaging At-Least-Once Semantics & Graph Idempotency

Subsystems Crossed:
- streaming/ (process_streaming_event, ingest_message, DLT routing)
- curation/ (structured event parsing)
- knowledge_graph/ (store_graph_entities, Cypher MERGE statement execution)

Invariants Asserted:
1. Streaming Idempotency Invariant: Replaying the identical 8-K event sequence N times
   yields exactly zero duplicate nodes and zero duplicate edges (delta_nodes == 0, delta_edges == 0).
2. Zero Primary Graph Poisoning Invariant: Malformed payloads failing validation write
   exactly zero nodes and zero edges to the primary graph store.
3. Dead-Letter Quarantine (DLT) Invariant: Every rejected payload routes deterministically
   to the Dead-Letter Queue with structured failure metadata.
4. Offline Isolation Invariant: Execution runs 100% offline without live Kafka brokers or external APIs.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
from typing import Any, Dict, List, Optional, Tuple
import pytest

from streaming.consumer import process_streaming_event
from streaming.producer import KafkaProducerFallback


class InMemoryIdempotentGraphStore:
    """In-memory property graph store tracking nodes and directed edges with Cypher MERGE semantics."""

    def __init__(self):
        # Key: (label, id) -> properties
        self.nodes: Dict[Tuple[str, str], Dict[str, Any]] = {}
        # Key: (source, rel_type, target) -> properties
        self.edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.query_log: List[Dict[str, Any]] = []

    def session(self):
        store = self

        class _Session:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def run(self, query: str, **kwargs):
                store.query_log.append({"query": query, "kwargs": kwargs})

                # 1. MERGE (n:News {id: $source_hash})
                if "MERGE (n:News" in query:
                    source_hash = str(kwargs.get("source_hash", "")).strip()
                    if source_hash:
                        key = ("News", source_hash)
                        if key not in store.nodes:
                            store.nodes[key] = {"id": source_hash}
                        store.nodes[key]["updated_at"] = kwargs.get("updated_at")

                # 2. MERGE (n:Label {id: $id}) SET n += $props
                elif "MERGE (n:" in query:
                    match = re.search(r"MERGE \(n:([A-Za-z0-9_]+)", query)
                    label = match.group(1) if match else "Entity"
                    node_id = str(kwargs.get("id", "")).strip()
                    props = kwargs.get("props", {}) or {}
                    if node_id:
                        key = (label, node_id)
                        if key not in store.nodes:
                            store.nodes[key] = {"id": node_id}
                        store.nodes[key].update(props)

                # 3. MERGE (news)-[:MENTIONS]->(e)
                elif "MERGE (news)-[:MENTIONS]->(e)" in query:
                    source_hash = str(kwargs.get("source_hash", "")).strip()
                    node_id = str(kwargs.get("id", "")).strip()
                    if source_hash and node_id:
                        edge_key = (source_hash, "MENTIONS", node_id)
                        if edge_key not in store.edges:
                            store.edges[edge_key] = {"source": source_hash, "target": node_id}

                # 4. MERGE (s)-[r:REL_TYPE]->(t)
                elif "MERGE (s)-[r:" in query:
                    match = re.search(r"MERGE \(s\)-\[r:([A-Za-z0-9_]+)\]->\(t\)", query)
                    rel_type = match.group(1) if match else "RELATED_TO"
                    source = str(kwargs.get("source", "")).strip()
                    target = str(kwargs.get("target", "")).strip()
                    props = kwargs.get("props", {}) or {}
                    if source and target:
                        edge_key = (source, rel_type, target)
                        if edge_key not in store.edges:
                            store.edges[edge_key] = {
                                "source": source,
                                "target": target,
                                "rel_type": rel_type,
                            }
                        store.edges[edge_key].update(props)
                        if "source_hash" in kwargs:
                            store.edges[edge_key]["source_hash"] = kwargs["source_hash"]

        return _Session()


class InMemoryRawMessageStore:
    """In-memory raw message table with ON CONFLICT (payload_hash) DO NOTHING semantics."""

    def __init__(self):
        self.records: Dict[str, Dict[str, Any]] = {}

    def store_raw(self, raw_payload: str, fetched_at: datetime, payload_hash: str) -> None:
        if payload_hash in self.records:
            return  # Idempotent skip
        self.records[payload_hash] = {
            "raw_payload": raw_payload,
            "fetched_at": fetched_at,
            "payload_hash": payload_hash,
        }


class MockDLTProducer:
    """Mock Kafka producer capturing dead-letter messages."""

    def __init__(self):
        self.sent_messages: List[Dict[str, Any]] = []

    def send(self, topic: str, value: Any) -> Any:
        val = value if isinstance(value, dict) else json.loads(value.decode("utf-8") if isinstance(value, bytes) else str(value))
        self.sent_messages.append({"topic": topic, "value": val})

        class _MockFuture:
            def get(self, timeout: Optional[float] = None) -> None:
                return None

        return _MockFuture()

    def flush(self) -> None:
        pass


@pytest.fixture
def mock_raw_store() -> InMemoryRawMessageStore:
    return InMemoryRawMessageStore()


@pytest.fixture
def mock_graph_store() -> InMemoryIdempotentGraphStore:
    return InMemoryIdempotentGraphStore()


@pytest.fixture
def mock_dlt_producer() -> MockDLTProducer:
    return MockDLTProducer()


@pytest.fixture
def sample_8k_event_payload() -> Dict[str, Any]:
    raw_data = {
        "cik": "0000320193",
        "ticker": "AAPL",
        "form_type": "8-K",
        "period_of_report": "2023-10-17",
        "title": "Item 8.01: FabCo Optical Transceiver Disruption",
        "summary": "Sole-source optical transceiver supplier FabCo Semiconductor experienced emergency shutdown.",
        "content": "On October 17, 2023, FabCo Semiconductor declared a 90-day facility disruption.",
    }
    return {
        "raw_payload": json.dumps(raw_data),
        "fetched_at": "2023-10-18T10:00:00+00:00",
        "source_hash": "sha256_8k_aapl_20231017_disruption",
    }


@pytest.fixture
def sample_structured_graph_data() -> Dict[str, Any]:
    return {
        "nodes": [
            {
                "id": "Apple Inc.",
                "type": "Company",
                "properties": {
                    "ticker": "AAPL",
                    "cik": "0000320193",
                    "exchange": "NASDAQ",
                    "period_of_report": "2023-10-17",
                },
            },
            {
                "id": "FabCo Semiconductor",
                "type": "Company",
                "properties": {
                    "supplier_type": "SOLE_SOURCE_OPTICAL",
                    "disruption_duration_days": 90,
                },
            },
        ],
        "edges": [
            {
                "source": "FabCo Semiconductor",
                "target": "Apple Inc.",
                "type": "SUPPLIES_TO",
                "properties": {
                    "materiality": "CRITICAL_TIER_1",
                    "loss_ratio": 0.50,
                    "buffer_days": 45,
                    "status": "ACTIVE_CURRENT",
                },
            }
        ],
    }


@pytest.mark.integration
def test_streaming_event_loop_replay_idempotency(
    monkeypatch,
    mock_raw_store: InMemoryRawMessageStore,
    mock_graph_store: InMemoryIdempotentGraphStore,
    mock_dlt_producer: MockDLTProducer,
    sample_8k_event_payload: Dict[str, Any],
    sample_structured_graph_data: Dict[str, Any],
):
    """Verify Invariant 1: Replaying the identical 8-K event sequence yields zero duplicate nodes and edges."""
    # Invariant 4: Offline Isolation Guardrail (strictly no live external socket connections)
    def fail_on_network(*args, **kwargs):
        raise RuntimeError("External network access prohibited during SIT Flow 4 execution!")

    monkeypatch.setattr("socket.socket.connect", fail_on_network)
    monkeypatch.setattr("streaming.consumer.store_raw", mock_raw_store.store_raw)

    # --- PASS 1: Initial Ingestion ---
    success_pass_1 = process_streaming_event(
        sample_8k_event_payload,
        dlt_producer=mock_dlt_producer,
        graph_driver=mock_graph_store,
        graph_data=sample_structured_graph_data,
        skip_embedding=True,
    )
    assert success_pass_1 is True

    # Validate Pass 1 state:
    # Raw store has 1 message
    assert len(mock_raw_store.records) == 1
    # Nodes: News (1) + Apple Inc (1) + FabCo (1) = 3
    # Edges: News->Apple (1) + News->FabCo (1) + FabCo->Apple (1) = 3
    nodes_pass_1 = len(mock_graph_store.nodes)
    edges_pass_1 = len(mock_graph_store.edges)
    assert nodes_pass_1 == 3
    assert edges_pass_1 == 3
    assert len(mock_dlt_producer.sent_messages) == 0

    # Invariant: Verify node properties
    apple_node = mock_graph_store.nodes.get(("Company", "Apple Inc."))
    assert apple_node is not None
    assert apple_node["ticker"] == "AAPL"
    assert apple_node["cik"] == "0000320193"
    assert apple_node["period_of_report"] == "2023-10-17"

    # Invariant: Verify edge properties
    supply_edge = mock_graph_store.edges.get(("FabCo Semiconductor", "SUPPLIES_TO", "Apple Inc."))
    assert supply_edge is not None
    assert supply_edge["buffer_days"] == 45
    assert supply_edge["materiality"] == "CRITICAL_TIER_1"

    # --- PASS 2: Exact Message Replay (simulating Kafka consumer restart / offset replay) ---
    success_pass_2 = process_streaming_event(
        sample_8k_event_payload,
        dlt_producer=mock_dlt_producer,
        graph_driver=mock_graph_store,
        graph_data=sample_structured_graph_data,
        skip_embedding=True,
    )
    assert success_pass_2 is True

    # Invariant 1: Zero Duplicate Topology Invariant
    # Raw store remains at 1 record (ON CONFLICT DO NOTHING)
    assert len(mock_raw_store.records) == 1
    # Nodes and edges count remains identical (MERGE idempotency)
    nodes_pass_2 = len(mock_graph_store.nodes)
    edges_pass_2 = len(mock_graph_store.edges)
    assert nodes_pass_2 == nodes_pass_1, f"Duplicate nodes detected: {nodes_pass_2} != {nodes_pass_1}"
    assert edges_pass_2 == edges_pass_1, f"Duplicate edges detected: {edges_pass_2} != {edges_pass_1}"

    # Properties and timestamps remain intact (no corruption or timestamp shifting)
    assert apple_node["period_of_report"] == "2023-10-17"
    assert supply_edge["buffer_days"] == 45


@pytest.mark.integration
def test_streaming_event_loop_dead_letter_quarantine_on_malformed_payload(
    monkeypatch,
    mock_raw_store: InMemoryRawMessageStore,
    mock_graph_store: InMemoryIdempotentGraphStore,
    mock_dlt_producer: MockDLTProducer,
):
    """Verify Invariant 2 & 3: Malformed payloads write zero graph entities and quarantine to DLT."""
    monkeypatch.setattr("streaming.consumer.store_raw", mock_raw_store.store_raw)
    malformed_cases = [
        # Case A: Missing fetched_at
        {"raw_payload": json.dumps({"title": "Test"}), "source_hash": "hash_1"},
        # Case B: Missing source_hash
        {"raw_payload": json.dumps({"title": "Test"}), "fetched_at": "2023-10-18T10:00:00+00:00"},
        # Case C: raw_payload is not a string
        {"raw_payload": None, "fetched_at": "2023-10-18T10:00:00+00:00", "source_hash": "hash_3"},
        # Case D: Empty dict
        {},
    ]

    for idx, bad_payload in enumerate(malformed_cases):
        success = process_streaming_event(
            bad_payload,
            dlt_producer=mock_dlt_producer,
            graph_driver=mock_graph_store,
            skip_embedding=True,
        )
        assert success is False, f"Malformed case {idx} should have failed validation"

    # Invariant 2: Zero Primary Graph Poisoning Invariant
    assert len(mock_graph_store.nodes) == 0, "Graph store was poisoned with nodes from malformed payloads!"
    assert len(mock_graph_store.edges) == 0, "Graph store was poisoned with edges from malformed payloads!"
    assert len(mock_graph_store.query_log) == 0, "Database session executed queries on malformed payloads!"

    # Invariant 3: Dead-Letter Queue Quarantine
    assert len(mock_dlt_producer.sent_messages) == len(malformed_cases)
    for msg in mock_dlt_producer.sent_messages:
        assert msg["value"]["reason"] == "invalid_kafka_payload"

