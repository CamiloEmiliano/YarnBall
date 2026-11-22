"""Unit & integration tests for canonical staged Form 8-K envelopes and workbench discovery (ADR-00027)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
import pytest

from explorer.workbench_api import create_workbench_app
from ingestion.sec.disruption_gating import normalize_staged_8k_payload
from ingestion.sec.harvester import HistoricalSECHarvester
from tools.dry_run_sec_8k_contagion import run_sec_8k_dry_run
from universe.sp500 import SP500Constituent


def test_normalize_staged_8k_payload_infers_identity_items_and_headline(tmp_path: Path) -> None:
    staged_file = tmp_path / "data" / "dry_run_staging" / "sec" / "AAPL" / "2020" / "8K_0001193125-20-225672" / "form8k_events.json"
    staged_file.parent.mkdir(parents=True, exist_ok=True)
    legacy_payload = {
        "accession_number": "0001193125-20-225672",
        "filing_date": "2020-08-20",
        "form": "8-K",
        "items": "8.01,9.01",
        "text": (
            "UNITED STATES SECURITIES AND EXCHANGE COMMISSION\n\n"
            "FORM 8-K\n\n"
            "Item 8.01 Other Events.\n\n"
            "On August 20, 2020, Apple Inc. completed its senior notes issuance cleanly."
        ),
    }
    staged_file.write_text(json.dumps(legacy_payload), encoding="utf-8")

    norm = normalize_staged_8k_payload(legacy_payload, filing_path=staged_file, require_identity=True)
    assert norm["ticker"] == "AAPL"
    assert norm["cik"] == "0000320193"
    assert "Apple" in norm["company_name"]
    assert norm["form_type"] == "8-K"
    assert norm["items"] == ["Item 8.01", "Item 9.01"]
    assert "Item 8.01 Other Events" in norm["headline"]
    assert "Apple Inc. completed" in norm["content"]
    assert norm["text"] == norm["content"]


def test_normalize_staged_8k_payload_rejects_unresolvable_identity() -> None:
    with pytest.raises(ValueError, match="entity identity"):
        normalize_staged_8k_payload(
            {
                "accession_number": "9999999999-24-000001",
                "filing_date": "2024-05-01",
                "items": "8.01",
                "text": "Generic disclosure with no company identifier.",
            },
            require_identity=True,
        )


def test_historical_harvester_writes_canonical_8k_envelope(tmp_path: Path) -> None:
    mock_client = MagicMock()
    mock_client.fetch_recent_8k.return_value = [
        {
            "accession_number": "0000320193-23-000080",
            "filing_date": "2023-08-03",
            "form": "8-K",
            "items": "2.02,9.01",
            "text": "Item 2.02 Results of Operations.\n\nApple Inc. announced quarterly revenue results.",
        }
    ]
    harvester = HistoricalSECHarvester(client=mock_client, output_dir=tmp_path, rate_limit_delay_sec=0.0)
    constituent = SP500Constituent(
        ticker="AAPL",
        cik="0000320193",
        company_name="Apple Inc.",
        gics_sector="Information Technology",
    )
    harvester.harvest_company_year(constituent, year=2023, forms=["8-K"])

    out_file = tmp_path / "AAPL" / "2023" / "8K_0000320193-23-000080" / "form8k_events.json"
    written = json.loads(out_file.read_text(encoding="utf-8"))
    assert written["cik"] == "0000320193"
    assert written["ticker"] == "AAPL"
    assert written["company_name"] == "Apple Inc."
    assert written["form_type"] == "8-K"
    assert written["items"] == ["Item 2.02", "Item 9.01"]
    assert "content" in written and "headline" in written


def test_dry_run_and_workbench_discover_and_evaluate_legacy_staged_8k(tmp_path: Path) -> None:
    staged = tmp_path / "data" / "dry_run_staging" / "sec" / "AAPL" / "2020" / "8K_0001193125-20-225672" / "form8k_events.json"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_text(
        json.dumps(
            {
                "accession_number": "0001193125-20-225672",
                "filing_date": "2020-08-20",
                "form": "8-K",
                "items": "8.01,9.01",
                "text": "Item 8.01 Other Events.\n\nApple Inc. completed routine debt offering with zero operational downtime.",
            }
        ),
        encoding="utf-8",
    )

    report, _ = run_sec_8k_dry_run(filing_path=staged, offline_isolation=True, use_mock=True)
    assert report.disrupted_entity_ticker == "AAPL"
    assert report.disrupted_entity_cik == "0000320193"
    assert report.disrupted_entity_ticker != "UNKNOWN"

    app = create_workbench_app(project_root=tmp_path)
    client = TestClient(app)
    cat = client.get("/api/catalog").json()
    assert len(cat["filings"]) == 1
    assert cat["filings"][0]["ticker"] == "AAPL"
    assert cat["filings"][0]["items"] == ["Item 8.01", "Item 9.01"]
    assert "AAPL" in cat["filings"][0]["name"]

    eval_resp = client.post(
        "/api/evaluate-8k",
        json={"filing_path": cat["filings"][0]["path"], "use_mock": True},
    )
    assert eval_resp.status_code == 200
    eval_data = eval_resp.json()
    assert eval_data["report"]["disrupted_entity_ticker"] == "AAPL"
    assert eval_data["report"]["disrupted_entity_cik"] == "0000320193"
