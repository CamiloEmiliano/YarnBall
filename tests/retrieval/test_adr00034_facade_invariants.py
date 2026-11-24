"""ADR-00034 Single BI Read Facade, Consolidation & Store Parity Invariant Tests.

Covers all 4 Invariants in ADR-00034 Section 5 plus `/search` & `/report` endpoints:
1. Invariant #1 (Single Entry AST Check): outside `retrieval/facade.py` and `retrieval/pipeline.py`,
   no module in `retrieval/`, `synthesis/`, `explorer/`, or `tools/` executes Memgraph read Cypher.
2. Invariant #2 (Single LLM Boundary): `httpx`, `qwen3:8b`, `OLLAMA_MODEL_NAME`, and `OLLAMA_API_BASE`
   never appear under `retrieval/` or `synthesis/`.
3. Invariant #3 (Domain Isolation & Legacy Retirement): `PORTFOLIO_RECOMMENDATION`, `portfolio_context`,
   `market_telemetry`, `hedging_recommendation`, and `hedging_think_trace` never appear under `retrieval/`;
   `retrieval/shock_propagation.py` and `HybridRetriever.expand_subgraph` are deleted.
4. Invariant #4 (Store Parity): `MemgraphSubgraphStore` and `InMemorySubgraphStore` return identical
   edge sets and cascade metrics (tolerance 1e-9) for the same fixture universe and `PiTContext`.
"""

from __future__ import annotations

import ast
from datetime import date
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence
from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest

import retrieval
from retrieval.facade import MemgraphSubgraphStore, SubgraphStorePort, query_bi
from retrieval.hybrid import HybridSeeder, SeedHit
from retrieval.offline_store import InMemorySubgraphStore
from retrieval.reporting import OperationalContagionReport
from retrieval.service import app as retrieval_app
from retrieval.temporal import PiTContext
from sft_pipeline.serving.model_client import ModelResponse, YarnBallModelClient

REPO_ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------
# Invariant #1: Single Entry (No Direct Memgraph Read Cypher Outside Facade/Pipeline)
# ----------------------------------------------------------------------
def test_adr00034_invariant1_single_memgraph_read_entry_ast() -> None:
    """Invariant #1: Outside retrieval/facade.py and retrieval/pipeline.py, no read Cypher runs directly."""
    allowed_files = {
        "retrieval/facade.py",
        "retrieval/pipeline.py",
    }
    offenders: List[str] = []

    for pkg in ("retrieval", "synthesis", "explorer", "tools"):
        pkg_dir = REPO_ROOT / pkg
        for py_path in sorted(pkg_dir.rglob("*.py")):
            rel = str(py_path.relative_to(REPO_ROOT))
            if rel in allowed_files:
                continue
            tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=rel)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fn = node.func
                if isinstance(fn, ast.Attribute) and fn.attr == "run":
                    if isinstance(fn.value, ast.Name) and fn.value.id == "session":
                        offenders.append(f"{rel}:{node.lineno} direct session.run(...) outside facade/pipeline")
                if isinstance(fn, ast.Attribute) and fn.attr == "execute_query":
                    if isinstance(fn.value, ast.Name) and "driver" in fn.value.id:
                        offenders.append(f"{rel}:{node.lineno} direct driver.execute_query(...) outside facade/pipeline")

    assert offenders == [], f"ADR-00034 Invariant #1 violations: {offenders}"


# ----------------------------------------------------------------------
# Invariant #2: Single LLM Boundary
# ----------------------------------------------------------------------
def test_adr00034_invariant2_single_llm_boundary_no_httpx_or_legacy_ollama() -> None:
    """Invariant #2: httpx, qwen3:8b, OLLAMA_MODEL_NAME, and OLLAMA_API_BASE never appear in retrieval/ or synthesis/."""
    forbidden_re = re.compile(r"\b(?:httpx|OLLAMA_MODEL_NAME|OLLAMA_API_BASE)\b|qwen3:8b")
    violations: List[str] = []

    for pkg in ("retrieval", "synthesis"):
        pkg_dir = REPO_ROOT / pkg
        for py_path in sorted(pkg_dir.rglob("*.py")):
            rel = str(py_path.relative_to(REPO_ROOT))
            for idx, line in enumerate(py_path.read_text(encoding="utf-8").splitlines(), start=1):
                match = forbidden_re.search(line)
                if match:
                    violations.append(f"{rel}:{idx}: matched '{match.group(0)}'")

    assert violations == [], f"ADR-00034 Invariant #2 violations: {violations}"


# ----------------------------------------------------------------------
# Invariant #3: Domain Isolation & Legacy Module Retirement
# ----------------------------------------------------------------------
def test_adr00034_invariant3_domain_isolation_and_legacy_retirement() -> None:
    """Invariant #3: Task E identifiers absent under retrieval/; shock_propagation & expand_subgraph removed."""
    assert not (REPO_ROOT / "retrieval" / "shock_propagation.py").exists()
    assert not hasattr(retrieval, "ShockVector")
    assert not hasattr(retrieval, "PropagatedShock")
    assert not hasattr(retrieval, "HybridRetriever")
    assert not hasattr(HybridSeeder, "expand_subgraph")

    forbidden = (
        "PORTFOLIO_RECOMMENDATION",
        "portfolio_context",
        "market_telemetry",
        "hedging_recommendation",
        "hedging_think_trace",
    )
    violations: List[str] = []
    for py_path in sorted((REPO_ROOT / "retrieval").rglob("*.py")):
        rel = str(py_path.relative_to(REPO_ROOT))
        text = py_path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                violations.append(f"{rel}: '{token}'")

    assert violations == [], f"ADR-00034 Invariant #3 violations: {violations}"


# ----------------------------------------------------------------------
# Invariant #4: SubgraphStorePort Parity (MemgraphSubgraphStore vs InMemorySubgraphStore)
# ----------------------------------------------------------------------
class _MemgraphFixtureDriver:
    """Simulates Memgraph record formatting over the same fixture universe as InMemorySubgraphStore."""

    def __init__(self, backing_store: InMemorySubgraphStore) -> None:
        self._backing = backing_store
        self.last_unanchored_excluded: int = 0

    def session(self) -> Any:
        driver = self

        class _Session:
            def __enter__(self) -> "_Session":
                return self

            def __exit__(self, *args: Any) -> None:
                return None

            def run(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
                as_of = str(kwargs.get("as_of_date") or "")
                mode = "live" if "ACTIVE_CURRENT" in query else "historical"
                include_unanchored = "valid_from IS NULL" in query and "valid_from IS NOT NULL" not in query
                ctx = PiTContext.from_date(as_of, mode=mode, include_unanchored=include_unanchored)  # type: ignore[arg-type]

                if "seeds" in kwargs:
                    hop_m = re.search(r"\[\s*r\*1\.\.(\d+)\s*\]", query)
                    hops = int(hop_m.group(1)) if hop_m else 2
                    limit = int(kwargs.get("limit", 50))
                    edges = driver._backing.traverse(kwargs["seeds"], hops=hops, ctx=ctx, limit=limit)
                else:
                    edges = driver._backing.query_edges(query, params=kwargs)

                driver.last_unanchored_excluded = driver._backing.last_unanchored_excluded
                # Return Memgraph projection records matching MemgraphSubgraphStore.traverse/execute_guarded
                memgraph_rows: List[Dict[str, Any]] = []
                for e in edges:
                    props = {
                        k: v
                        for k, v in e.items()
                        if k not in ("source", "target", "rel_type", "hop", "source_id", "target_id", "source_hash")
                    }
                    memgraph_rows.append(
                        {
                            "source": e["source"],
                            "source_id": e.get("source_id", e["source"]),
                            "rel_type": e["rel_type"],
                            "target": e["target"],
                            "target_id": e.get("target_id", e["target"]),
                            "props": props,
                            "source_hash": e.get("source_hash"),
                            "hop": e.get("hop", 1),
                        }
                    )
                return memgraph_rows

        return _Session()


class _DeterministicParityModelClient(YarnBallModelClient):
    def generate(
        self,
        prompt: str,
        task: str = "CONTAGION_REASONING",
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_new_tokens: Optional[int] = None,
        enforce_think_prefix: bool = True,
        num_ctx: int = 8192,
    ) -> ModelResponse:
        if task == "TEXT_TO_CYPHER":
            out = "MATCH (s:Company {ticker: 'ASML'})-[r:SUPPLIES_TO*1..2]->(t:Company) RETURN s, r, t"
        else:
            out = "Deterministic parity synthesis over validated cascade metrics."
        return ModelResponse(
            raw_text=out,
            think_trace="Parity reasoning trace.",
            final_output=out,
            task=task,
            latency_sec=0.01,
            prefill_ms=1.0,
            tokens_per_sec=100.0,
            prompt_tokens=20,
            generated_tokens=20,
        )


def test_adr00034_invariant4_live_and_offline_store_parity_within_1e9() -> None:
    """Invariant #4: MemgraphSubgraphStore and InMemorySubgraphStore yield identical edges & cascade metrics."""
    fixture_edges = [
        {
            "source": "ASML",
            "target": "TSM",
            "rel_type": "SUPPLIES_TO",
            "nature": "euv_lithography",
            "share": 0.90,
            "buffer_days": 15.0,
            "status": "ACTIVE_CURRENT",
            "valid_from": "2020-01-02",
            "valid_to": None,
        },
        {
            "source": "TSM",
            "target": "AAPL",
            "rel_type": "SUPPLIES_TO",
            "nature": "advanced_wafers",
            "share": 0.80,
            "buffer_days": 20.0,
            "status": "TERMINATED",  # Terminated later in 2024-06-03; active on 2024-05-01!
            "valid_from": "2020-01-02",
            "valid_to": "2024-06-03",
        },
        {
            "source": "TSM",
            "target": "NVDA",
            "rel_type": "SUPPLIES_TO",
            "nature": "future_node_edge",
            "share": 0.70,
            "buffer_days": 10.0,
            "status": "ACTIVE_CURRENT",
            "valid_from": "2025-01-02",  # Future edge excluded on 2024-05-01
            "valid_to": None,
        },
        {
            "source": "TSM",
            "target": "UNANCHORED",
            "rel_type": "SUPPLIES_TO",
            "nature": "legacy",
            "share": 0.50,
            "buffer_days": 10.0,
            "status": "ACTIVE_CURRENT",
            "valid_from": None,  # Unanchored edge excluded by default
            "valid_to": None,
        },
    ]

    offline_store = InMemorySubgraphStore(fixture_edges)
    live_store = MemgraphSubgraphStore(driver=_MemgraphFixtureDriver(InMemorySubgraphStore(fixture_edges)))

    assert isinstance(offline_store, SubgraphStorePort)
    assert isinstance(live_store, SubgraphStorePort)

    ctx = PiTContext(as_of=date(2024, 5, 1), mode="historical")

    # 1. Direct SubgraphStorePort.traverse parity
    offline_edges = offline_store.traverse(["ASML"], hops=2, ctx=ctx)
    live_edges = live_store.traverse(["ASML"], hops=2, ctx=ctx)

    def _edge_signature(edges: Sequence[Mapping[str, Any]]) -> List[tuple[Any, ...]]:
        return [
            (
                e["hop"],
                e["source"],
                e["rel_type"],
                e["target"],
                e["input_group"],
                round(float(e["share"]), 9),
                round(float(e["buffer_days"]), 9),
                e["valid_from"],
                e["valid_to"],
            )
            for e in edges
        ]

    assert _edge_signature(live_edges) == _edge_signature(offline_edges)
    assert [(e["source"], e["target"]) for e in live_edges] == [("ASML", "TSM"), ("TSM", "AAPL")]

    # 2. Full query_bi + cascade parity within 1e-9
    res_offline = query_bi(
        "Assess ASML lithography outage contagion to AAPL",
        ctx=ctx,
        store=offline_store,
        model_client=_DeterministicParityModelClient(),
        target_ticker="AAPL",
        shock_origin="ASML",
        initial_shocks={"ASML": 100.0},
        initial_durations={"ASML": 60.0},
        initial_onsets={"ASML": 0.0},
    )
    res_live = query_bi(
        "Assess ASML lithography outage contagion to AAPL",
        ctx=ctx,
        store=live_store,
        model_client=_DeterministicParityModelClient(),
        target_ticker="AAPL",
        shock_origin="ASML",
        initial_shocks={"ASML": 100.0},
        initial_durations={"ASML": 60.0},
        initial_onsets={"ASML": 0.0},
    )

    assert _edge_signature(res_live.retrieved_subgraph) == _edge_signature(res_offline.retrieved_subgraph)
    assert set(res_live.cascade_summaries.keys()) == set(res_offline.cascade_summaries.keys())
    for entity in res_offline.cascade_summaries:
        s_off = res_offline.cascade_summaries[entity]
        s_live = res_live.cascade_summaries[entity]
        assert s_live.verdict == s_off.verdict
        assert s_live.peak_shock_pct == pytest.approx(s_off.peak_shock_pct, abs=1e-9)
        assert s_live.impairment_score == pytest.approx(s_off.impairment_score, abs=1e-9)
        assert s_live.downtime_days == pytest.approx(s_off.downtime_days, abs=1e-9)
        assert s_live.lost_capacity_days == pytest.approx(s_off.lost_capacity_days, abs=1e-9)
        assert s_live.spectral_radius == pytest.approx(s_off.spectral_radius, abs=1e-9)
        if s_off.onset_day is None:
            assert s_live.onset_day is None
        else:
            assert s_live.onset_day == pytest.approx(s_off.onset_day, abs=1e-9)


# ----------------------------------------------------------------------
# Service Endpoints: POST /search and POST /report Delegation to query_bi
# ----------------------------------------------------------------------
def test_service_search_and_report_delegate_directly_to_query_bi() -> None:
    """ADR-00034 Section 4.5: /search and /report delegate to query_bi and return Domain 1 contracts."""
    fixture_store = InMemorySubgraphStore(
        [
            {
                "source": "CRUS",
                "target": "AAPL",
                "rel_type": "SUPPLIES_TO",
                "nature": "audio_codecs",
                "share": 0.25,
                "buffer_days": 30.0,
                "valid_from": "2020-01-02",
            }
        ]
    )

    def _fake_query_bi(question: str, ctx: PiTContext, **kwargs: Any) -> Any:
        kwargs["store"] = fixture_store
        kwargs["seeder"] = None
        kwargs["model_client"] = _DeterministicParityModelClient()
        return query_bi(question, ctx=ctx, **kwargs)

    client = TestClient(retrieval_app)

    # 1. /search requires as_of and returns PipelineResult payload
    missing_date = client.post("/search", json={"query": "CRUS outage"})
    assert missing_date.status_code == 422

    with patch("retrieval.service.query_bi", side_effect=_fake_query_bi):
        search_resp = client.post(
            "/search",
            json={
                "query": "CRUS fab halt impact on AAPL",
                "as_of": "2024-05-01",
                "mode": "historical",
                "hops": 2,
                "shock_origin": "CRUS",
                "target_ticker": "AAPL",
            },
        )
        assert search_resp.status_code == 200
        search_data = search_resp.json()
        assert search_data["as_of_date"] == "2024-05-01"
        assert search_data["is_valid_query"] is True
        assert "hedging_recommendation" not in search_data

        # 2. /report returns validated OperationalContagionReport JSON
        report_resp = client.post(
            "/report",
            json={
                "query": "CRUS fab halt impact on AAPL",
                "as_of": "2024-05-01",
                "mode": "historical",
                "event_id": "SEC_8K_20240501_CRUS",
                "sec_item": "Item 8.01",
                "disrupted_entity_ticker": "CRUS",
                "disruption_category": "FACILITY_FIRE_OR_EXPLOSION",
                "event_summary": "CRUS fab outage affecting audio codecs.",
                "shock_duration_days": 60.0,
                "shock_magnitude_pct": 100.0,
                "target_ticker": "AAPL",
                "model_mode": "MOCK_DETERMINISTIC",
            },
        )
        assert report_resp.status_code == 200
        parsed_report = OperationalContagionReport.model_validate(report_resp.json())
        assert parsed_report.disrupted_entity_ticker == "CRUS"
        assert parsed_report.event_id == "SEC_8K_20240501_CRUS"
        assert parsed_report.filing_date == "2024-05-01"
