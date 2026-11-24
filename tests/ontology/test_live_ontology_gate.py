"""Verification tests for ADR-00023: Unified Live Ontology Gate & Schema Convergence."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from ingestion.sec.harvester import EdgarWorker
from ingestion.sec.parser import SEC_EXTRACTION_PROMPT
from knowledge_graph.graph_store import SYSTEM_EXTRACTION_PROMPT, store_graph_entities
from ontology.guardrails import CANONICAL_RELATION_PREDICATES
from ontology.schema import ONTOLOGY_SCHEMA
from ontology.validation import validate_and_orient_triple, validate_graph_payload
from retrieval.guardrails.schema_guardrail import DEFAULT_FINANCE_SCHEMA
from sft_pipeline.rejection_queue import ErrorStage, RuntimeRejectionQueue


def test_schema_prompts_and_guardrails_stay_converged():
    """Prompts, Tier-4 guardrails, and Cypher AST whitelist derive from ONTOLOGY_SCHEMA."""
    assert CANONICAL_RELATION_PREDICATES == frozenset(ONTOLOGY_SCHEMA.keys())
    assert "SOLE_SOURCE_DEPENDENT_ON" in ONTOLOGY_SCHEMA
    assert set(ONTOLOGY_SCHEMA.keys()).issubset(DEFAULT_FINANCE_SCHEMA.allowed_rel_types)
    for rel in ("SUPPLIES_TO", "SOLE_SOURCE_DEPENDENT_ON", "ACQUIRED_BY", "DEFAULTED_ON"):
        assert rel in SYSTEM_EXTRACTION_PROMPT
        assert rel in SEC_EXTRACTION_PROMPT


def test_strict_validation_closes_open_world_bypass_and_applies_aliases():
    """Unrecognized predicates fail strict checks; valid aliases canonicalize deterministically."""
    assert validate_and_orient_triple("Apple", "Company", "TSMC", "Company", "CORRELATES_WITH") is None
    # LEADS aliases to EXECUTIVE_OF (Person -> Company); Company -> Company must fail
    assert validate_and_orient_triple("Apple", "Company", "Beats", "Company", "LEADS") is None
    assert validate_and_orient_triple("Tim Cook", "Person", "Apple", "Company", "LEADS") == (
        "Tim Cook",
        "PERSON",
        "Apple",
        "COMPANY",
        "EXECUTIVE_OF",
    )
    # ACQUIRED flips direction to ACQUIRED_BY
    assert validate_and_orient_triple("Apple", "Company", "Beats", "Company", "ACQUIRED") == (
        "Beats",
        "COMPANY",
        "Apple",
        "COMPANY",
        "ACQUIRED_BY",
    )
    assert validate_and_orient_triple(
        "Apple", "Company", "FabCo", "Company", "SOLE_SOURCE_DEPENDENT_ON"
    ) == ("Apple", "COMPANY", "FabCo", "COMPANY", "SOLE_SOURCE_DEPENDENT_ON")


def test_validate_graph_payload_filters_placeholders_repairs_and_quarantines(tmp_path: Path):
    """validate_graph_payload drops placeholders/off-schema triples and logs to rejection queue."""
    queue = RuntimeRejectionQueue(base_dir=tmp_path)
    payload = {
        "nodes": [
            {"id": "Management", "type": "Person", "properties": {}},
            {"id": "TSMC", "type": "Company", "properties": {"ticker": "TSM"}},
            {"id": "Intel", "type": "Company", "properties": {"ticker": "INTC"}},
            {"id": "Tim Cook", "type": "Person", "properties": {}},
            {"id": "Apple Inc.", "type": "Company", "properties": {"ticker": "AAPL"}},
        ],
        "edges": [
            {"source": "Apple Inc.", "target": "Management", "type": "CEO_OF", "properties": {}},
            {"source": "TSMC", "target": "Intel", "type": "COMPETES_WITH", "properties": {}},
            {"source": "Apple Inc.", "target": "Tim Cook", "type": "CEO_OF", "properties": {}},
            {"source": "TSMC", "target": "Intel", "type": "HALLUCINATED_LINK", "properties": {}},
        ],
    }

    gated = validate_graph_payload(payload, source_id="hash-gate-1", rejection_queue=queue)
    node_ids = [n["id"] for n in gated["nodes"]]
    assert "Management" not in node_ids
    assert set(node_ids) == {"TSMC", "Intel", "Tim Cook", "Apple Inc."}

    assert gated["edges"] == [
        # Canonicalized alphabetically for symmetric COMPETES_WITH
        {"source": "Intel", "target": "TSMC", "type": "COMPETES_WITH", "properties": {}},
        # Auto-oriented Person -> CEO_OF -> Company
        {"source": "Tim Cook", "target": "Apple Inc.", "type": "CEO_OF", "properties": {}},
    ]

    records = queue.load_records()
    assert len(records) == 3  # Management node, Management edge, HALLUCINATED_LINK
    assert all(r.error_stage == ErrorStage.ONTOLOGY_PRUNING for r in records)


def test_store_graph_entities_gates_writes_before_memgraph(tmp_path: Path):
    """store_graph_entities applies inline gating so bad predicates/placeholders never reach Memgraph."""
    queue = RuntimeRejectionQueue(base_dir=tmp_path)
    mock_driver = MagicMock()
    session = mock_driver.session.return_value.__enter__.return_value

    # Purely invalid payload -> zero Memgraph session activity
    store_graph_entities(
        source_hash="hash-all-bad",
        text="placeholder",
        graph_data={
            "nodes": [{"id": "the company", "type": "Company", "properties": {}}],
            "edges": [{"source": "the company", "target": "certain suppliers", "type": "CORRELATES_WITH"}],
        },
        driver=mock_driver,
        rejection_queue=queue,
    )
    mock_driver.session.assert_not_called()

    # Mixed valid + invalid payload -> only normalized canonical triples written
    store_graph_entities(
        source_hash="hash-mixed",
        text="Apple acquired Beats.",
        graph_data={
            "nodes": [
                {"id": "Apple Inc.", "type": "Company", "properties": {"ticker": "AAPL"}},
                {"id": "Beats", "type": "Company", "properties": {}},
            ],
            "edges": [
                {"source": "Apple Inc.", "target": "Beats", "type": "ACQUIRED", "properties": {}},
                {"source": "Apple Inc.", "target": "Beats", "type": "UNRELATED_VERB", "properties": {}},
            ],
        },
        driver=mock_driver,
        rejection_queue=queue,
    )
    executed_queries = [call.args[0] for call in session.run.call_args_list]
    assert any("MERGE (s)-[r:ACQUIRED_BY]->(t)" in q for q in executed_queries)
    assert not any("UNRELATED_VERB" in q for q in executed_queries)


def test_edgar_worker_gates_sec_persisted_edges():
    """EdgarWorker._persist_sec_graph_data filters placeholders and off-ontology triples before MERGE."""
    mock_driver = MagicMock()
    worker = EdgarWorker(memgraph_driver=mock_driver, pg_conn=None)
    worker.linker.link_entity = MagicMock(return_value=None)

    nodes, edges = worker._persist_sec_graph_data(
        primary_name="Apple Inc.",
        primary_ticker="AAPL",
        graph_data={
            "nodes": [
                {"id": "Certain Suppliers", "type": "Company", "properties": {}},
                {"id": "TSMC", "type": "Company", "properties": {"ticker": "TSM"}},
            ],
            "edges": [
                {"source": "Certain Suppliers", "target": "Apple Inc.", "type": "SUPPLIES_TO", "properties": {}},
                {"source": "TSMC", "target": "Apple Inc.", "type": "HALLUCINATED_EDGE", "properties": {}},
                {"source": "Apple Inc.", "target": "TSMC", "type": "SOLE_SOURCE_DEPENDENT_ON", "properties": {}},
            ],
        },
        accession_number="0000320193-26-000010",
        filing_date="2026-10-30",
        form_type="10-K",
        section="Item 1 Business",
    )
    assert "Certain Suppliers" not in nodes
    assert "TSMC" in nodes
    assert edges == [
        {
            "source": "Apple Inc.",
            "target": "TSMC",
            "type": "SOLE_SOURCE_DEPENDENT_ON",
            "evidence_chunk_ids": ["0000320193-26-000010#item_1_business:c001"],
        }
    ]
