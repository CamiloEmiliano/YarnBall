"""Verification tests for ADR-00029: Structured LLM extraction & failure quarantine."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion.sec.harvester import EdgarWorker
from ingestion.sec.parser import extract_sec_relationships
from knowledge_graph.graph_store import (
    GraphExtractionError,
    _clean_json_response,
    extract_entities,
    parse_extraction_payload,
)
from knowledge_graph.memgraph_consumer import dispatch_memgraph_message
from sft_pipeline.rejection_queue import ErrorStage, RuntimeRejectionQueue
from sft_pipeline.serving.model_client import YarnBallModelClient


def test_parse_strips_think_block_with_curly_braces():
    """Reasoning <think> blocks with stray curly braces must not break balanced JSON extraction."""
    raw = (
        "<think>Considering schema {foo: bar} and entities {Apple}...</think>\n"
        'Here is the extracted graph:\n```json\n'
        '{"nodes": [{"id": "Apple Inc.", "type": "Company", "properties": {"ticker": "AAPL"}}], '
        '"edges": []}\n```'
    )
    parsed = parse_extraction_payload(raw)
    assert parsed["nodes"] == [
        {"id": "Apple Inc.", "type": "Company", "properties": {"ticker": "AAPL"}}
    ]
    assert parsed["edges"] == []
    assert _clean_json_response(raw) == parsed


def test_parse_opencypher_triples_with_parentheses_in_quoted_name():
    """Fine-tuned yarnball-qwen OpenCypher triples normalize cleanly into ExtractionGraphPayload."""
    cypher = (
        '(:Company {name: "Amcor", ticker: "AMCR", cik: "0001748790"})'
        '-[:ACQUIRED_BY {polarity: "NEUTRAL_STABLE", materiality: "CRITICAL_TIER_1", '
        'confidence: 0.98}]->'
        '(:Company {name: "Alphabet Inc. (Class A)", ticker: "GOOGL", cik: "0001652044"})'
    )
    parsed = parse_extraction_payload(cypher)
    assert len(parsed["nodes"]) == 2
    by_id = {n["id"]: n for n in parsed["nodes"]}
    assert by_id["Amcor"]["type"] == "Company"
    assert by_id["Amcor"]["properties"]["ticker"] == "AMCR"
    assert by_id["Alphabet Inc. (Class A)"]["properties"]["cik"] == "0001652044"
    assert parsed["edges"] == [
        {
            "source": "Amcor",
            "target": "Alphabet Inc. (Class A)",
            "type": "ACQUIRED_BY",
            "properties": {
                "polarity": "NEUTRAL_STABLE",
                "materiality": "CRITICAL_TIER_1",
                "confidence": 0.98,
            },
        }
    ]


def test_none_control_returns_empty_without_quarantine(tmp_path: Path):
    """Explicit '(none)' completion is valid empty output and writes zero rejection records."""
    queue = RuntimeRejectionQueue(base_dir=tmp_path)
    client = YarnBallModelClient(transport_fn=lambda _payload: {"response": "(none)"})

    res = extract_entities(
        "Treasury yields pressured equity valuations broadly.",
        model_client=client,
        rejection_queue=queue,
        source_id="hash-neg-1",
    )
    assert res == {"nodes": [], "edges": []}
    assert queue.load_records() == []


@pytest.mark.parametrize(
    "bad_completion",
    [
        "",
        "<think>unfinished thought without payload",
        '{"status": "ok"}',
        '{"nodes": [{"id": "", "type": "Company"}], "edges": []}',
        "(:Company {name: \"Broken\"})-[:SUPPLIES_TO]->",
    ],
)
def test_malformed_output_quarantines_and_raises(tmp_path: Path, bad_completion: str):
    """Malformed, empty, or schema-invalid completions log to RuntimeRejectionQueue and raise."""
    queue = RuntimeRejectionQueue(base_dir=tmp_path)
    client = YarnBallModelClient(transport_fn=lambda _payload: {"response": bad_completion})

    with pytest.raises(GraphExtractionError) as exc_info:
        extract_entities(
            "Apple Inc. announced Q4 supply chain updates.",
            model_client=client,
            rejection_queue=queue,
            source_id="hash-bad-1",
        )

    assert exc_info.value.error_stage == ErrorStage.EXTRACTION_PARSE_ERROR
    records = queue.load_records()
    assert len(records) == 1
    assert records[0].task_id == "EXTRACT_NEWS_EVENT"
    assert records[0].error_stage == ErrorStage.EXTRACTION_PARSE_ERROR
    assert records[0].sample_id == "hash-bad-1"
    assert records[0].metadata["source_id"] == "hash-bad-1"


def test_ollama_transport_error_quarantines_and_raises(tmp_path: Path):
    """Transport/Ollama errors log LLM_UNAVAILABLE into RuntimeRejectionQueue and raise."""
    queue = RuntimeRejectionQueue(base_dir=tmp_path)

    def failing_transport(_payload):
        raise ConnectionError("Connection refused: Ollama unreachable")

    client = YarnBallModelClient(transport_fn=failing_transport)
    with pytest.raises(GraphExtractionError) as exc_info:
        extract_entities(
            "TSMC expands Arizona fab construction.",
            model_client=client,
            rejection_queue=queue,
            source_id="hash-conn-1",
        )

    assert exc_info.value.error_stage == ErrorStage.LLM_UNAVAILABLE
    records = queue.load_records()
    assert len(records) == 1
    assert records[0].error_stage == ErrorStage.LLM_UNAVAILABLE
    assert "Connection refused" in records[0].error_message


def test_sec_extraction_quarantine_and_harvester_section_isolation(tmp_path: Path):
    """SEC parser quarantines broken completions, while EdgarWorker keeps valid filing sections."""
    queue = RuntimeRejectionQueue(base_dir=tmp_path)
    bad_client = YarnBallModelClient(transport_fn=lambda _p: {"response": "not valid json or cypher"})

    with pytest.raises(GraphExtractionError):
        extract_sec_relationships(
            "Supplier disclosures for Item 1 Business.",
            model_client=bad_client,
            rejection_queue=queue,
            source_id="0001-26-0001:item_1_business",
        )
    assert len(queue.load_records()) == 1
    assert queue.load_records()[0].task_id == "EXTRACT_SEC_GRAPH"

    mock_driver = MagicMock()
    worker = EdgarWorker(memgraph_driver=mock_driver, pg_conn=None)
    fake_filing = MagicMock()
    fake_filing.company = "Apple Inc."
    fake_filing.cik = "0000320193"
    worker.client.fetch_latest_10k = MagicMock(return_value=fake_filing)
    worker.client.extract_10k_sections = MagicMock(
        return_value={
            "accession_number": "0000320193-26-000010",
            "filing_date": "2026-10-30",
            "report_date": "2026-09-30",
            "subsidiaries": [{"name": "Apple Operations International", "jurisdiction": "Ireland"}],
            "item_1_business": "We rely on partner TSMC.",
            "item_1a_risk_factors": "Tariff and geopolitical risk factors.",
        }
    )
    worker.linker.link_entity = MagicMock(
        side_effect=lambda name=None, ticker=None: (
            {"cik": "0000320193", "company_name": "Apple Inc.", "ticker": "AAPL"}
            if (ticker == "AAPL" or name == "Apple Inc.")
            else None
        )
    )

    def _side_effect(text, source_id=None):
        if "TSMC" in text:
            raise GraphExtractionError("Broken Item 1 completion")
        return {
            "nodes": [{"id": "Tariff Risk", "type": "Metric", "properties": {}}],
            "edges": [{"source": "Apple Inc.", "target": "Tariff Risk", "type": "EXPOSED_TO", "properties": {}}],
        }

    with patch("ingestion.sec.harvester.extract_sec_relationships", side_effect=_side_effect):
        result = worker.process_company_10k("AAPL")

    assert result["status"] == "completed"
    # Exhibit 21 (parent + subsidiary = 2 nodes, 1 edge) + Item 1A (parent + metric = 2 nodes, 1 edge) preserved
    assert result["nodes"] == 4
    assert result["edges"] == 2


def test_memgraph_consumer_routes_graph_extraction_error_to_dlt():
    """dispatch_memgraph_message routes GraphExtractionError to send_to_dlt with reason='llm_extraction_failed'."""
    payload = {
        "source_hash": "hash-dlt-42",
        "raw_payload": json.dumps({"title": "NVDA Earnings", "summary": "Broken completion test"}),
    }
    producer = MagicMock()

    with patch(
        "knowledge_graph.memgraph_consumer.process_graph_message",
        side_effect=GraphExtractionError("Malformed JSON from yarnball-qwen"),
    ), patch("knowledge_graph.memgraph_consumer.send_to_dlt") as mock_dlt:
        status = dispatch_memgraph_message(payload, producer=producer, message="msg-obj")

    assert status == "dlt_extraction_failed"
    mock_dlt.assert_called_once_with(
        producer=producer,
        message="msg-obj",
        payload=payload,
        reason="llm_extraction_failed",
        error="Malformed JSON from yarnball-qwen",
    )
