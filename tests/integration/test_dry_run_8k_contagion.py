"""Integration tests for ADR-00009 SEC Form 8-K Computational Reading Dry Run."""

from __future__ import annotations

import json
from pathlib import Path
import time
import pytest

from retrieval.reporting import (
    ContagionSeverityVerdict,
    ModelExecutionMode,
    OperationalContagionReport,
    OperationalDisruptionCategory,
    SECForm8KItem,
    format_operational_contagion_report,
    validate_verbatim_citations,
)
from sft_pipeline.serving.model_client import ModelResponse, YarnBallModelClient
from tools.dry_run_sec_8k_contagion import (
    export_report_to_jsonl,
    main as dry_run_cli_main,
    run_sec_8k_dry_run,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "filings"


class MockDryRunModelClient(YarnBallModelClient):
    """Deterministic model client simulating aligned yarnball-qwen:7b inference for dry-run testing."""

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        task = kwargs.get("task")
        if task == "TEXT_TO_CYPHER" or "Text-to-Cypher" in prompt:
            return ModelResponse(
                raw_text="MATCH (s:Company {ticker: 'CRUS'})-[r:SUPPLIES_TO]->(t:Company) RETURN s, r, t;",
                final_output="MATCH (s:Company {ticker: 'CRUS'})-[r:SUPPLIES_TO]->(t:Company) RETURN s, r, t;",
                task="TEXT_TO_CYPHER",
                latency_sec=0.01,
                prefill_ms=1.0,
                tokens_per_sec=100.0,
                prompt_tokens=15,
                generated_tokens=15,
            )
        else:
            think_trace = (
                "Analyzing Cirrus Logic (CRUS) audio codec fab halt of 60 days. Apple maintains a 45-day "
                "inventory buffer. The buffer delays onset to Day 45 and absorbs 45 days of shock, leaving "
                "15 unbuffered downtime days with a 20.0% capacity contraction. Foxconn maintains a 90-day "
                "assembly buffer, fully absorbing downstream impact."
            )
            final_output = (
                "Apple Inc. (AAPL) experiences UNBUFFERED_DISRUPTION resulting from a 20.0% peak capacity "
                "contraction across its audio component lines with 15.0 days of unbuffered downtime post-buffer "
                "absorption. Foxconn (FOXCONN) and Hewlett-Packard (HPQ) are classified as BUFFER_ABSORBED, as "
                "their 90-day inventory safety stocks fully absorb operational transmission."
            )
            return ModelResponse(
                raw_text=f"<think>\n{think_trace}\n</think>\n{final_output}",
                think_trace=think_trace,
                final_output=final_output,
                task="CONTAGION_REASONING",
                latency_sec=0.08,
                prefill_ms=8.0,
                tokens_per_sec=90.0,
                prompt_tokens=80,
                generated_tokens=70,
            )


@pytest.fixture
def offline_guardrail(monkeypatch):
    """Offline Isolation Guardrail: Strictly prevents remote socket connections (ADR-00009 Invariant 4)."""
    def fail_on_network(*args, **kwargs):
        raise RuntimeError("External network access prohibited during dry-run offline execution!")

    monkeypatch.setattr("socket.socket.connect", fail_on_network)


@pytest.mark.integration
def test_sec_8k_dry_run_crus_fab_fire_e2e(offline_guardrail):
    """Verify end-to-end dry run on Cirrus Logic (CRUS) 8-K Item 8.01 fab fire disclosure."""
    filing_path = FIXTURES_DIR / "crus_8k_disruption.json"
    assert filing_path.is_file(), f"Fixture missing at: {filing_path}"

    start_time = time.perf_counter()
    report, pipeline_res = run_sec_8k_dry_run(
        filing_path=filing_path,
        model_client=MockDryRunModelClient(),
        target_ticker="AAPL",
    )
    elapsed = time.perf_counter() - start_time

    # 1. Processing Throughput (< 30 Seconds)
    assert elapsed < 30.0, f"Dry-run execution took {elapsed:.2f}s, exceeding 30s threshold"

    # 2. Authentic Filing Identity & Disruption Classification
    assert report.disrupted_entity_ticker == "CRUS"
    assert report.disrupted_entity_name == "Cirrus Logic Inc."
    assert report.sec_item == SECForm8KItem.ITEM_8_01.value
    assert report.filing_date == "2024-10-28"
    assert report.disruption_category == OperationalDisruptionCategory.FACILITY_FIRE_OR_EXPLOSION.value
    assert report.shock_duration_days == 60.0
    assert report.shock_magnitude_pct == 100.0

    # 3. Deterministic Network Impacts (Exact Numerical Preservation)
    impact_map = {i.entity_ticker: i for i in report.network_impacts}
    assert "AAPL" in impact_map
    assert "HPQ" in impact_map
    assert "FOXCONN" in impact_map

    # AAPL: 45d buffer absorbed 45d of 60d shock -> 15d unbuffered downtime
    aapl_impact = impact_map["AAPL"]
    assert aapl_impact.hop_distance == 1
    assert aapl_impact.input_group == "audio_codecs"
    assert aapl_impact.procurement_share_pct == 20.0
    assert aapl_impact.inventory_buffer_days == 45.0
    assert aapl_impact.shock_onset_day == 45.0
    assert aapl_impact.unbuffered_downtime_days == 15.0
    assert aapl_impact.capacity_contraction_pct == 20.0
    assert aapl_impact.lost_capacity_days == 3.0
    assert aapl_impact.verdict == ContagionSeverityVerdict.UNBUFFERED_DISRUPTION

    # HPQ: 90d buffer absorbed 100% of 60d shock -> 0d downtime
    hpq_impact = impact_map["HPQ"]
    assert hpq_impact.hop_distance == 1
    assert hpq_impact.input_group == "audio_codecs"
    assert hpq_impact.inventory_buffer_days == 90.0
    assert hpq_impact.shock_onset_day is None
    assert hpq_impact.unbuffered_downtime_days == 0.0
    assert hpq_impact.verdict == ContagionSeverityVerdict.BUFFER_ABSORBED

    # FOXCONN: 90d buffer absorbed upstream shock -> 0d downtime
    fox_impact = impact_map["FOXCONN"]
    assert fox_impact.hop_distance == 2
    assert fox_impact.unbuffered_downtime_days == 0.0
    assert fox_impact.verdict == ContagionSeverityVerdict.BUFFER_ABSORBED

    # 4. Topology and Stability Properties
    assert report.is_acyclic_dag is True
    assert report.spectral_radius == 0.0
    assert len(report.active_topology_edges) == 3

    # 5. Model Chain-of-Thought Demuxing & Verbatim Citations
    assert report.contagion_think_trace is not None
    assert "Analyzing Cirrus Logic (CRUS)" in report.contagion_think_trace
    violations = validate_verbatim_citations(report)
    assert len(violations) == 0, f"Verbatim citation violations: {violations}"

    # 6. Formatted ASCII/Markdown Visual Report Verification
    rendered = format_operational_contagion_report(report)
    assert "YARNBALL OPERATIONAL CONTAGION REPORT" in rendered
    assert "1. DETERMINISTIC CASCADE SHOCK METRICS" in rendered
    assert "2. ACTIVE KNOWLEDGE GRAPH TOPOLOGY" in rendered
    assert "3. COMPUTATIONAL READING & REASONING" in rendered
    assert "<think>" in rendered
    assert "4. PIPELINE TELEMETRY" in rendered


@pytest.mark.integration
def test_sec_8k_dry_run_aapl_optical_disruption(offline_guardrail):
    """Verify dry run on Apple Inc. (AAPL) 8-K Item 8.01 optical supplier disruption disclosure."""
    filing_path = FIXTURES_DIR / "aapl_8k_disruption.json"
    assert filing_path.is_file(), f"Fixture missing at: {filing_path}"

    start_time = time.perf_counter()
    report, _ = run_sec_8k_dry_run(
        filing_path=filing_path,
        model_client=MockDryRunModelClient(),
        target_ticker="AAPL",
    )
    elapsed = time.perf_counter() - start_time

    assert elapsed < 30.0
    assert report.disrupted_entity_ticker == "FABCO"
    assert report.disrupted_entity_name == "FabCo Semiconductor"
    assert report.shock_duration_days == 90.0

    impact_map = {i.entity_ticker: i for i in report.network_impacts}
    assert "AAPL" in impact_map

    # AAPL has 45d buffer against 90d disruption -> 45d unbuffered downtime
    aapl_impact = impact_map["AAPL"]
    assert aapl_impact.inventory_buffer_days == 45.0
    assert aapl_impact.unbuffered_downtime_days == 45.0
    assert aapl_impact.verdict == ContagionSeverityVerdict.UNBUFFERED_DISRUPTION


@pytest.mark.integration
def test_domain_boundary_isolation_invariant(offline_guardrail):
    """Assert dry-run report contains zero speculative trading or portfolio allocation fields."""
    filing_path = FIXTURES_DIR / "crus_8k_disruption.json"
    report, _ = run_sec_8k_dry_run(
        filing_path=filing_path,
        model_client=MockDryRunModelClient(),
    )

    report_dict = report.model_dump()
    forbidden_terms = ["trade", "order", "option", "strike", "collar", "put", "call", "basket"]

    for key, val in report_dict.items():
        key_tokens = set(key.lower().split("_"))
        assert not key_tokens.intersection(forbidden_terms), f"Forbidden term in report key: {key}"
        if isinstance(val, str):
            for term in ["short leg", "long leg", "directional basket", "option collar"]:
                assert term not in val.lower(), f"Forbidden trading phrase '{term}' in report field '{key}'"


@pytest.mark.integration
def test_sec_8k_dry_run_cli_mock_and_jsonl_export(offline_guardrail, tmp_path, monkeypatch, capsys):
    """Verify ADR-00013 `--mock` execution and `--export-jsonl` 1NF time-series join export."""
    filing_path = FIXTURES_DIR / "crus_8k_disruption.json"
    jsonl_out = tmp_path / "exports" / "contagion_events.jsonl"

    monkeypatch.setattr(
        "sys.argv",
        [
            "dry_run_sec_8k_contagion.py",
            "--filing",
            str(filing_path),
            "--mock",
            "--export-jsonl",
            str(jsonl_out),
            "--validate-citations",
        ],
    )

    dry_run_cli_main()
    captured = capsys.readouterr()
    assert "YARNBALL OPERATIONAL CONTAGION REPORT" in captured.out
    assert "UNBUFFERED_DISRUPTION" in captured.out

    assert jsonl_out.is_file()
    lines = [line for line in jsonl_out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 3  # AAPL, HPQ, FOXCONN

    rows = [json.loads(line) for line in lines]
    by_target = {r["target_entity_ticker"]: r for r in rows}

    assert by_target["AAPL"]["model_mode"] == ModelExecutionMode.MOCK_DETERMINISTIC.value
    assert by_target["AAPL"]["calendar_onset_date"] == "2024-12-12"
    assert by_target["AAPL"]["calendar_recovery_date"] == "2024-12-27"
    assert by_target["AAPL"]["verdict"] == "UNBUFFERED_DISRUPTION"

    assert by_target["HPQ"]["calendar_onset_date"] is None
    assert by_target["HPQ"]["calendar_recovery_date"] is None
    assert by_target["HPQ"]["verdict"] == "BUFFER_ABSORBED"
    assert by_target["FOXCONN"]["verdict"] == "BUFFER_ABSORBED"

