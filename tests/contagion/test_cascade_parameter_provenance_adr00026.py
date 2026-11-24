"""Unit tests for supply-chain cascade parameter extraction and provenance tagging (ADR-00026)."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from contagion.parameters import (
    ParameterProvenance,
    extract_cascade_parameters_from_text,
    resolve_edge_cascade_parameters,
)
from explorer.workbench_api import create_workbench_app
from ingestion.sec.harvester import EdgarWorker
from knowledge_graph.graph_store import store_graph_entities
from retrieval.offline_store import InMemorySubgraphStore
from retrieval.pipeline import PipelineResult
from retrieval.reporting import build_operational_contagion_report, format_operational_contagion_report


def test_extract_cascade_parameters_from_text_quantitative_and_sole_source() -> None:
    res = extract_cascade_parameters_from_text(
        "TSMC supplied approximately 65% of our wafer requirements with roughly 45 calendar days of inventory."
    )
    assert res["share"] == 0.65
    assert res["buffer_days"] == 45.0
    assert res["dependency_domain"] == "PHYSICAL_COMPONENT"

    weeks_res = extract_cascade_parameters_from_text(
        "Sole-source cloud hosting contract with 3 weeks of safety stock."
    )
    assert weeks_res["sole_source"] is True
    assert weeks_res["buffer_days"] == 21.0
    assert weeks_res["dependency_domain"] == "DIGITAL_INFRASTRUCTURE"


def test_resolve_edge_cascade_parameters_distinguishes_extracted_sole_source_and_defaults() -> None:
    quant = resolve_edge_cascade_parameters(
        {"source": "TSM", "target": "AAPL", "share": 0.35, "buffer_days": 30.0, "nature": "wafer"}
    )
    assert quant["param_provenance"] == ParameterProvenance.EXTRACTED_QUANTITATIVE.value
    assert quant["share"] == 0.35
    assert quant["buffer_days"] == 30.0
    assert quant["share_is_default"] is False
    assert quant["buffer_days_is_default"] is False

    sole = resolve_edge_cascade_parameters(
        {"source": "ASML", "target": "TSM", "rel_type": "SOLE_SOURCE_DEPENDENT_ON", "nature": "EUV"}
    )
    assert sole["param_provenance"] == ParameterProvenance.EXTRACTED_QUALITATIVE_SOLE_SOURCE.value
    assert sole["share"] == 1.0
    assert sole["buffer_days"] == 0.0
    assert sole["share_is_default"] is True
    assert sole["buffer_days_is_default"] is True

    defaulted = resolve_edge_cascade_parameters(
        {"source": "SUP", "target": "OEM", "rel_type": "SUPPLIES_TO"}
    )
    assert defaulted["param_provenance"] == ParameterProvenance.DEFAULT_CONSERVATIVE_BOUND.value
    assert defaulted["share"] == 1.0
    assert defaulted["buffer_days"] == 0.0
    assert defaulted["share_is_default"] is True
    assert defaulted["buffer_days_is_default"] is True


def test_inmemory_store_and_operational_report_surface_provenance_counts() -> None:
    store = InMemorySubgraphStore(
        edges=[
            {
                "source": "CRUS",
                "target": "AAPL",
                "rel_type": "SUPPLIES_TO",
                "nature": "audio_codecs",
                "share": 0.25,
                "buffer_days": 45.0,
            },
            {
                "source": "CRUS",
                "target": "SONOS",
                "rel_type": "SUPPLIES_TO",
                "nature": "amplifiers",
            },
        ]
    )
    edges = store.query_edges("MATCH (a)-[r:SUPPLIES_TO]->(b) RETURN a, r, b")
    by_target = {e["target"]: e for e in edges}
    assert by_target["AAPL"]["param_provenance"] == "EXTRACTED_QUANTITATIVE"
    assert by_target["AAPL"]["share_is_default"] is False
    assert by_target["SONOS"]["param_provenance"] == "DEFAULT_CONSERVATIVE_BOUND"
    assert by_target["SONOS"]["share"] == 1.0
    assert by_target["SONOS"]["buffer_days"] == 0.0
    assert by_target["SONOS"]["share_is_default"] is True
    assert by_target["SONOS"]["buffer_days_is_default"] is True

    pipeline_res = PipelineResult(
        query="Contagion from CRUS",
        as_of_date="2024-05-01",
        is_valid_query=True,
        retrieved_subgraph=edges,
        verbalized_subgraph="CRUS -> AAPL / SONOS",
        contagion_verdict="AAPL buffered; SONOS exposed under conservative default bound.",
        total_latency_ms=10.0,
    )
    report = build_operational_contagion_report(
        result=pipeline_res,
        event_id="EV-1",
        sec_item="Item 8.01",
        disrupted_entity_ticker="CRUS",
        filing_date="2026-10-01",
        disruption_category="EQUIPMENT_BREAKDOWN_FAB_HALT",
        event_summary="Fab halt",
        shock_duration_days=60.0,
        shock_magnitude_pct=100.0,
    )
    assert report.telemetry.extracted_parameter_edge_count == 1
    assert report.telemetry.defaulted_parameter_edge_count == 1
    desc_by_tgt = {e.target: e for e in report.active_topology_edges}
    assert desc_by_tgt["AAPL"].param_provenance == ParameterProvenance.EXTRACTED_QUANTITATIVE
    assert desc_by_tgt["SONOS"].param_provenance == ParameterProvenance.DEFAULT_CONSERVATIVE_BOUND
    assert desc_by_tgt["SONOS"].buffer_days_is_default is True

    rendered = format_operational_contagion_report(report)
    assert "Parameter Provenance:         1 extracted, 1 default bounds" in rendered


def test_memgraph_writers_stamp_param_provenance_properties() -> None:
    session = MagicMock()
    driver = MagicMock()
    driver.session.return_value.__enter__.return_value = session

    store_graph_entities(
        source_hash="news_1",
        text="TSMC supplies chips to Apple Inc.",
        graph_data={
            "nodes": [
                {"id": "TSMC", "type": "Company", "properties": {}},
                {"id": "Apple Inc.", "type": "Company", "properties": {}},
            ],
            "edges": [
                {"source": "TSMC", "target": "Apple Inc.", "type": "SUPPLIES_TO", "properties": {"share": 0.4}},
            ],
        },
        driver=driver,
    )
    edge_calls = [c for c in session.run.call_args_list if "SET r += $props" in c.args[0]]
    assert len(edge_calls) == 1
    assert edge_calls[0].kwargs["props"]["param_provenance"] == "EXTRACTED_QUANTITATIVE"
    assert edge_calls[0].kwargs["props"]["share"] == 0.4
    assert edge_calls[0].kwargs["props"]["buffer_days_is_default"] is True

    sec_session = MagicMock()
    sec_driver = MagicMock()
    sec_driver.session.return_value.__enter__.return_value = sec_session
    worker = EdgarWorker(memgraph_driver=sec_driver, pg_conn=None)
    worker.linker.link_entity = MagicMock(return_value=None)
    worker._persist_sec_graph_data(
        primary_name="Apple Inc.",
        primary_ticker="AAPL",
        graph_data={
            "nodes": [{"id": "TSMC", "type": "Company", "properties": {}}],
            "edges": [{"source": "TSMC", "target": "Apple Inc.", "type": "SUPPLIES_TO", "properties": {}}],
        },
        accession_number="0000320193-26-000010",
        filing_date="2026-10-30",
        form_type="10-K",
        section="Item 1 Business",
    )
    sec_edge_calls = [c for c in sec_session.run.call_args_list if "SET r += $props" in c.args[0]]
    assert len(sec_edge_calls) == 1
    assert sec_edge_calls[0].kwargs["props"]["param_provenance"] == "DEFAULT_CONSERVATIVE_BOUND"


def test_workbench_subgraph_api_exposes_edge_param_provenance(tmp_path) -> None:
    edges_file = tmp_path / "data" / "snapshots" / "edges.json"
    edges_file.parent.mkdir(parents=True, exist_ok=True)
    edges_file.write_text(
        '{"edges": [{"source": "TSM", "target": "AAPL", "rel_type": "SUPPLIES_TO", "valid_from": "2020-01-02"}]}',
        encoding="utf-8",
    )
    app = create_workbench_app(project_root=tmp_path)
    client = TestClient(app)
    resp = client.get(
        "/api/subgraph",
        params={"edges_file": "data/snapshots/edges.json", "ticker": "TSM", "as_of_date": "2024-06-01"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload["edges"]) == 1
    assert payload["edges"][0]["param_provenance"] == "DEFAULT_CONSERVATIVE_BOUND"
    assert payload["edges"][0]["share_is_default"] is True
    assert payload["edges"][0]["buffer_days_is_default"] is True
