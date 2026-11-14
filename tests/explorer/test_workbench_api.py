"""API & security guardrail tests for `explorer.workbench_api` (ADR-00017)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from fastapi.testclient import TestClient

from explorer.workbench_api import create_workbench_app


def _write_snapshot(snap_dir: Path, nodes: list[dict], edges: list[dict]) -> None:
    snap_dir.mkdir(parents=True, exist_ok=True)
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
    pq.write_table(pa.Table.from_pylist(nodes, schema=node_schema), snap_dir / "nodes.parquet")
    pq.write_table(pa.Table.from_pylist(edges, schema=edge_schema), snap_dir / "edges.parquet")


def _build_fixture_workspace(tmp_path: Path):
    project_root = tmp_path / "repo"
    snapshots_dir = project_root / "data" / "snapshots"
    filings_dir = project_root / "tests" / "fixtures" / "filings"
    cohorts_dir = project_root / "data" / "cohorts"
    manifests_dir = project_root / "data" / "manifests"
    for folder in (snapshots_dir, filings_dir, cohorts_dir, manifests_dir):
        folder.mkdir(parents=True, exist_ok=True)

    nodes = [
        {"id": "n_asml", "label": "Company", "ticker": "ASML", "source_hash": "h1", "properties_json": "{}"},
        {"id": "n_tsm", "label": "Company", "ticker": "TSM", "source_hash": "h2", "properties_json": "{}"},
        {"id": "n_aapl", "label": "Company", "ticker": "AAPL", "source_hash": "h3", "properties_json": "{}"},
        {"id": "n_msft", "label": "Company", "ticker": "MSFT", "source_hash": "h4", "properties_json": "{}"},
    ]
    edges = [
        {
            "source_id": "n_asml",
            "target_id": "n_tsm",
            "rel_type": "SUPPLIES_TO",
            "source_hash": "e1",
            "properties_json": json.dumps({
                "nature": "euv_tools",
                "share": 0.90,
                "buffer_days": 30.0,
                "status": "ACTIVE_CURRENT",
                "valid_from": "2020-01-01",
            }),
        },
        {
            "source_id": "n_tsm",
            "target_id": "n_aapl",
            "rel_type": "SUPPLIES_TO",
            "source_hash": "e2",
            "properties_json": json.dumps({
                "nature": "advanced_wafers",
                "share": 0.85,
                "buffer_days": 15.0,
                "status": "ACTIVE_CURRENT",
                "valid_from": "2020-01-01",
            }),
        },
        {
            "source_id": "n_aapl",
            "target_id": "n_msft",
            "rel_type": "COMPETES_WITH",
            "source_hash": "e3",
            "properties_json": json.dumps({
                "nature": "consumer_ai",
                "status": "ACTIVE_CURRENT",
                "valid_from": "2024-06-01",
            }),
        },
    ]
    _write_snapshot(snapshots_dir / "G_resolved", nodes, edges)

    manifest_edges = {
        "edges": [
            {
                "source": "CRUS",
                "target": "AAPL",
                "rel_type": "SUPPLIES_TO",
                "nature": "audio_codecs",
                "share": 0.25,
                "buffer_days": 45.0,
                "valid_from": "2020-01-01",
            }
        ]
    }
    (manifests_dir / "custom_edges.json").write_text(json.dumps(manifest_edges), encoding="utf-8")

    crus_filing = {
        "cik": "0000772406",
        "ticker": "CRUS",
        "company_name": "Cirrus Logic Inc.",
        "form_type": "8-K",
        "filing_date": "2024-10-28",
        "items": ["Item 8.01 Other Events"],
        "headline": "Emergency facility shutdown at wafer fabrication plant",
        "content": (
            "On October 27, 2024, Cirrus Logic Inc. experienced an emergency operational halt "
            "at its primary fabrication facility affecting audio codec supply for 60 calendar days."
        ),
        "metadata": {
            "affected_product": "audio_codecs",
            "disruption_duration_days": 60,
            "shock_magnitude_pct": 100.0,
        },
    }
    (filings_dir / "crus_8k.json").write_text(json.dumps(crus_filing), encoding="utf-8")

    cohort_rows = [
        {
            "report_id": "REP-1",
            "event_id": "EV-1",
            "filing_date": "2024-10-28",
            "sec_item": "Item 8.01 Other Events",
            "disruption_category": "FACILITY_FIRE_OR_SHUTDOWN",
            "disrupted_entity_ticker": "CRUS",
            "target_entity_ticker": "AAPL",
            "hop_distance": 1,
            "verdict": "UNBUFFERED_DISRUPTION",
            "model_mode": "MOCK_DETERMINISTIC",
        },
        {
            "report_id": "REP-2",
            "event_id": "EV-2",
            "filing_date": "2024-11-01",
            "sec_item": "Item 5.02 Departure/Appointment of Officers",
            "disruption_category": "OTHER_UNCLASSIFIED_MATERIAL_HALT",
            "disrupted_entity_ticker": "MSFT",
            "target_entity_ticker": "MSFT",
            "hop_distance": 0,
            "verdict": "SAFE",
            "model_mode": "MOCK_DETERMINISTIC",
        },
    ]
    (cohorts_dir / "sample_cohort.jsonl").write_text(
        "\n".join(json.dumps(row) for row in cohort_rows) + "\n",
        encoding="utf-8",
    )

    outside_file = tmp_path / "outside_secret.json"
    outside_file.write_text(json.dumps(crus_filing), encoding="utf-8")

    app = create_workbench_app(project_root=project_root)
    return project_root, outside_file, ClientBundle(TestClient(app))


class ClientBundle:
    def __init__(self, client: TestClient) -> None:
        self.client = client


def test_readiness_and_catalog_discovery(tmp_path, monkeypatch):
    _, _, bundle = _build_fixture_workspace(tmp_path)
    monkeypatch.setattr(
        "explorer.workbench_api.YarnBallModelClient.check_health",
        lambda self, timeout_sec=1.0: (False, "offline test mock"),
    )

    root_resp = bundle.client.get("/")
    assert root_resp.status_code == 200
    assert "YarnBall Local Workbench API" in root_resp.text
    assert "/docs" in root_resp.text

    cat_resp = bundle.client.get("/api/catalog")
    assert cat_resp.status_code == 200
    cat = cat_resp.json()
    assert [s["snapshot_id"] for s in cat["snapshots"]] == ["G_resolved"]
    assert cat["snapshots"][0]["path"] == "data/snapshots/G_resolved"
    assert cat["snapshots"][0]["tickers"] == ["AAPL", "ASML", "MSFT", "TSM"]
    assert [f["name"] for f in cat["filings"]] == ["crus_8k.json"]
    assert [m["name"] for m in cat["edge_manifests"]] == ["custom_edges.json"]
    assert [c["name"] for c in cat["cohorts"]] == ["sample_cohort.jsonl"]
    assert cat["ollama"] == {"healthy": False, "detail": "offline test mock"}


def test_subgraph_bidirectional_and_pit_queries(tmp_path):
    _, _, bundle = _build_fixture_workspace(tmp_path)

    resp_both = bundle.client.get(
        "/api/subgraph",
        params={"snapshot_id": "G_resolved", "ticker": "TSM", "max_hops": 1, "direction": "both", "as_of_date": "2024-10-01"},
    )
    assert resp_both.status_code == 200
    payload = resp_both.json()
    assert payload["root_ticker"] == "TSM"
    assert payload["available_tickers"] == ["AAPL", "ASML", "MSFT", "TSM"]
    assert [(e["source"], e["target"], e["traversal_direction"]) for e in payload["edges"]] == [
        ("ASML", "TSM", "upstream"),
        ("TSM", "AAPL", "downstream"),
    ]

    # Pre-2024-06-01 excludes AAPL -> MSFT edge
    resp_pit = bundle.client.get(
        "/api/subgraph",
        params={"snapshot_id": "G_resolved", "ticker": "AAPL", "max_hops": 1, "direction": "downstream", "as_of_date": "2024-01-01"},
    )
    assert resp_pit.status_code == 200
    assert resp_pit.json()["edges"] == []

    # Edge manifest query works as well
    resp_manifest = bundle.client.get(
        "/api/subgraph",
        params={"edges_file": "data/manifests/custom_edges.json", "ticker": "CRUS", "direction": "downstream"},
    )
    assert resp_manifest.status_code == 200
    assert [(e["source"], e["target"]) for e in resp_manifest.json()["edges"]] == [("CRUS", "AAPL")]


def test_evaluate_8k_fixture_and_inline_negative_control(tmp_path):
    _, _, bundle = _build_fixture_workspace(tmp_path)

    # 1. File fixture evaluation in deterministic mock mode
    eval_resp = bundle.client.post(
        "/api/evaluate-8k",
        json={
            "filing_path": "tests/fixtures/filings/crus_8k.json",
            "use_mock": True,
        },
    )
    assert eval_resp.status_code == 200
    body = eval_resp.json()
    assert body["assessment"]["is_material_operational_disruption"] is True
    assert body["report"]["disrupted_entity_ticker"] == "CRUS"
    assert body["report"]["model_mode"] == "MOCK_DETERMINISTIC"
    assert body["citation_violations"] == []
    assert len(body["timeseries_rows"]) >= 1

    # 2. Inline Safe Harbor boilerplate filing must gate down to zero shock & clean up temp files
    temp_dirs_created: list[Path] = []
    orig_temp_dir = tempfile.TemporaryDirectory

    class _TrackingTempDir:
        def __init__(self, *args, **kwargs):
            self._inner = orig_temp_dir(*args, **kwargs)
            temp_dirs_created.append(Path(self._inner.name))

        def __enter__(self):
            return self._inner.__enter__()

        def __exit__(self, exc_type, exc_val, exc_tb):
            return self._inner.__exit__(exc_type, exc_val, exc_tb)

    try:
        tempfile.TemporaryDirectory = _TrackingTempDir  # type: ignore[assignment]
        neg_resp = bundle.client.post(
            "/api/evaluate-8k",
            json={
                "inline_filing": {
                    "ticker": "ACME",
                    "company_name": "Acme Corp",
                    "filing_date": "2024-11-05",
                    "items": ["Item 5.02 Departure/Appointment of Officers"],
                    "headline": "Appointment of Chief Accounting Officer",
                    "content": (
                        "Acme Corp appointed Jane Doe as Chief Accounting Officer.\n\n"
                        "Cautionary Note Regarding Forward-Looking Statements: "
                        "Statements regarding facility fires or disruptions are forward-looking "
                        "and actual results may differ materially due to risks and uncertainties."
                    ),
                },
                "use_mock": True,
            },
        )
    finally:
        tempfile.TemporaryDirectory = orig_temp_dir  # type: ignore[assignment]

    assert neg_resp.status_code == 200
    neg_body = neg_resp.json()
    assert neg_body["assessment"]["is_material_operational_disruption"] is False
    assert neg_body["assessment"]["shock_magnitude_pct"] == 0.0
    assert neg_body["assessment"]["stripped_paragraph_count"] >= 1
    assert len(temp_dirs_created) == 1
    assert not temp_dirs_created[0].exists()


def test_cohort_filtering_and_pagination(tmp_path):
    _, _, bundle = _build_fixture_workspace(tmp_path)

    resp_all = bundle.client.post(
        "/api/cohort",
        json={"cohort_path": "data/cohorts/sample_cohort.jsonl"},
    )
    assert resp_all.status_code == 200
    data_all = resp_all.json()
    assert data_all["total_records"] == 2
    assert data_all["filtered_records"] == 2
    assert data_all["verdict_counts"] == {"SAFE": 1, "UNBUFFERED_DISRUPTION": 1}

    resp_filtered = bundle.client.post(
        "/api/cohort",
        json={
            "cohort_path": "data/cohorts/sample_cohort.jsonl",
            "verdict": "UNBUFFERED_DISRUPTION",
            "ticker": "AAPL",
        },
    )
    assert resp_filtered.status_code == 200
    data_filtered = resp_filtered.json()
    assert data_filtered["filtered_records"] == 1
    assert data_filtered["records"][0]["report_id"] == "REP-1"


def test_path_traversal_outside_allowlist_rejected_with_403(tmp_path):
    _, outside_file, bundle = _build_fixture_workspace(tmp_path)

    # 1. Traversal via `edges_file`
    resp_edges = bundle.client.get("/api/subgraph", params={"edges_file": "../outside_secret.json"})
    assert resp_edges.status_code == 403

    # 2. Traversal via `snapshot_id` escaping `data/snapshots` into sibling workspace folder
    resp_snap = bundle.client.get("/api/subgraph", params={"snapshot_id": "../cohorts"})
    assert resp_snap.status_code == 403

    # 3. Traversal via `filing_path` (both relative traversal and absolute outside path)
    resp_rel = bundle.client.post("/api/evaluate-8k", json={"filing_path": "../outside_secret.json"})
    assert resp_rel.status_code == 403
    resp_abs = bundle.client.post("/api/evaluate-8k", json={"filing_path": str(outside_file)})
    assert resp_abs.status_code == 403

    # 4. Traversal via `cohort_path`
    resp_cohort = bundle.client.post("/api/cohort", json={"cohort_path": "../outside_secret.json"})
    assert resp_cohort.status_code == 403


def test_static_workbench_shell_and_zero_cdn_offline_invariant(tmp_path):
    """Verifies ADR-00018 static assets, zero external CDNs, and XSS-safe DOM rendering."""
    _, _, bundle = _build_fixture_workspace(tmp_path)

    root_resp = bundle.client.get("/")
    assert root_resp.status_code == 200
    assert "Event Audit (8-K" in root_resp.text
    assert "PiT KG Explorer" in root_resp.text
    assert "Cohort Browser" in root_resp.text
    assert "/static/styles.css" in root_resp.text
    assert "/static/app.js" in root_resp.text

    css_resp = bundle.client.get("/static/styles.css")
    assert css_resp.status_code == 200
    assert "--bg:" in css_resp.text

    js_resp = bundle.client.get("/static/app.js")
    assert js_resp.status_code == 200
    assert "/api/evaluate-8k" in js_resp.text
    assert ".innerHTML" not in js_resp.text

    for asset_text in (root_resp.text, css_resp.text, js_resp.text):
        scrubbed = asset_text.replace("http://www.w3.org/2000/svg", "")
        assert "http://" not in scrubbed.lower()
        assert "https://" not in scrubbed.lower()


def test_ui_catalog_to_event_audit_flow(tmp_path, monkeypatch):
    """Simulates initial Event Audit UI flow: /api/catalog -> POST /api/evaluate-8k."""
    _, _, bundle = _build_fixture_workspace(tmp_path)
    monkeypatch.setattr(
        "explorer.workbench_api.YarnBallModelClient.check_health",
        lambda self, timeout_sec=1.0: (False, "offline test mock"),
    )

    cat_resp = bundle.client.get("/api/catalog")
    assert cat_resp.status_code == 200
    filings = cat_resp.json()["filings"]
    assert len(filings) == 1

    eval_resp = bundle.client.post(
        "/api/evaluate-8k",
        json={"filing_path": filings[0]["path"], "use_mock": True},
    )
    assert eval_resp.status_code == 200
    payload = eval_resp.json()
    assert payload["assessment"]["gating_reason"] == "SANITIZED_LEXICAL_DISRUPTION_MATCH"
    assert payload["citation_violations"] == []
    assert any(n["entity_ticker"] == "AAPL" and n["verdict"] == "UNBUFFERED_DISRUPTION" for n in payload["report"]["network_impacts"])
    assert len(payload["timeseries_rows"]) == len(payload["report"]["network_impacts"])


def test_adr00019_kg_explorer_and_cohort_browser_assets_and_flow(tmp_path, monkeypatch):
    """Verifies ADR-00019 interactive PiT KG Explorer and Cohort Browser assets & API flow."""
    _, _, bundle = _build_fixture_workspace(tmp_path)
    monkeypatch.setattr(
        "explorer.workbench_api.YarnBallModelClient.check_health",
        lambda self, timeout_sec=1.0: (False, "offline test mock"),
    )

    root_resp = bundle.client.get("/")
    assert root_resp.status_code == 200
    assert "Placeholder until ADR-00019" not in root_resp.text
    for dom_id in ("kgSvgMount", "kgProvenancePanel", "cohortTbody", "cohortSummaryRow"):
        assert dom_id in root_resp.text

    js_resp = bundle.client.get("/static/app.js")
    assert js_resp.status_code == 200
    assert "/api/subgraph?" in js_resp.text
    assert "/api/cohort" in js_resp.text
    assert "Inspect Network" in js_resp.text
    assert ".innerHTML" not in js_resp.text

    # Simulate catalog -> PiT KG Explorer -> Cohort Browser -> Inspect Network jump flow
    cat = bundle.client.get("/api/catalog").json()
    snap_id = cat["snapshots"][0]["snapshot_id"]
    cohort_path = cat["cohorts"][0]["path"]

    cohort_resp = bundle.client.post(
        "/api/cohort",
        json={"cohort_path": cohort_path, "verdict": "UNBUFFERED_DISRUPTION"},
    )
    assert cohort_resp.status_code == 200
    rec = cohort_resp.json()["records"][0]
    assert rec["disrupted_entity_ticker"] == "CRUS"

    # Cross-navigate using disrupted/target ticker & filing_date on G_resolved
    sub_resp = bundle.client.get(
        "/api/subgraph",
        params={
            "snapshot_id": snap_id,
            "ticker": rec["target_entity_ticker"],
            "as_of_date": rec["filing_date"],
            "direction": "both",
            "max_hops": 2,
        },
    )
    assert sub_resp.status_code == 200
    edges = sub_resp.json()["edges"]
    assert any(e["source"] == "TSM" and e["target"] == "AAPL" for e in edges)



