"""Integration test for Flow 3: Natural Language Query -> Text-to-Cypher -> Schema Guardrail -> Temporal Filter -> Graph Execution -> Contagion & Hedging Synthesis.

Crosses: retrieval/pipeline.py, retrieval/guardrails/, retrieval/temporal.py, sft_pipeline/serving/, sft_pipeline/rejection_queue.py.
Verifies:
1. End-to-end multi-hop retrieval and causal synthesis flow.
2. Invariant: Whitelisted Cypher executes cleanly against seeded graph topology.
3. Invariant: Temporal query filter injects active contract status.
4. Invariant: Mode-bleed prevention and <think> CoT formatting in contagion verdict.
5. Invariant: Schema property traps are intercepted and recorded in the rejection queue.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest

from retrieval.guardrails.schema_guardrail import CypherASTWhitelistGuardrail
from retrieval.pipeline import (
    AnalystQuery,
    GraphRAGPipelineOrchestrator,
    HopOrderedSubgraphVerbalizer,
    PipelineResult,
)
from retrieval.temporal import TemporalQueryFilter
from sft_pipeline.rejection_queue import ErrorStage, RuntimeRejectionQueue
from sft_pipeline.serving.model_client import ModelResponse, YarnBallModelClient


class SeededGraphStore:
    """Deterministic in-memory graph driver simulating Memgraph supply chain state."""

    def __init__(self):
        # Topology: TSM -> NVDA -> MSFT
        self.edges = [
            {
                "source": "TSM",
                "target": "NVDA",
                "rel_type": "SUPPLIES_TO",
                "status": "ACTIVE_CURRENT",
                "loss_ratio": 0.85,
                "revenue_share_pct": 12.0,
                "buffer_days": 60,
                "hop": 1,
            },
            {
                "source": "NVDA",
                "target": "MSFT",
                "rel_type": "SUPPLIES_TO",
                "status": "ACTIVE_CURRENT",
                "loss_ratio": 0.40,
                "revenue_share_pct": 8.5,
                "buffer_days": 45,
                "hop": 2,
            },
        ]
        self.query_log: List[str] = []

    def session(self):
        store = self

        class _Session:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def run(self, query: str, **kwargs):
                store.query_log.append(query)
                # Return matching edges if query looks for suppliers or active paths
                if "SUPPLIES_TO" in query:
                    return store.edges
                return []

        return _Session()


class DeterministicModelTransport:
    """Mock transport providing realistic LLM responses across operational tasks."""

    def __init__(self, fail_schema_check: bool = False):
        self.fail_schema_check = fail_schema_check
        self.prompts_received: List[Dict[str, str]] = []

    def __call__(
        self,
        prompt: str,
        task: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> ModelResponse:
        self.prompts_received.append({"prompt": prompt, "task": task})

        if task in ("C", "TEXT_TO_CYPHER"):
            if self.fail_schema_check:
                text = "MATCH (c:Company) WHERE c.ceo_age < 50 RETURN c.name, c.ceo_age"
            else:
                text = "MATCH (s:Company {ticker: 'TSM'})-[r:SUPPLIES_TO*1..2]->(t:Company) RETURN s, r, t"
            return ModelResponse(
                raw_text=text,
                think_trace=None,
                final_output=text,
                task="TEXT_TO_CYPHER",
                latency_sec=0.012,
                prefill_ms=5.0,
                tokens_per_sec=85.0,
                prompt_tokens=25,
                generated_tokens=22,
            )

        elif task in ("D", "CONTAGION_REASONING"):
            think = (
                "Step 1: TSM is Tier-1 supplier to NVDA (loss_ratio: 0.85).\n"
                "Step 2: NVDA has 60-day inventory buffer mitigating near-term wafer shortfall.\n"
                "Step 3: Downstream MSFT experiences lagged delivery delay after day 45."
            )
            synthesis = (
                "TSM factory halt triggers severe shock at NVDA (85% component vulnerability), "
                "partially buffered by 60-day safety stock. Contagion propagates downstream to MSFT "
                "with an estimated 34% operational impairment starting in 6 to 8 weeks."
            )
            raw = f"<think>\n{think}\n</think>\n{synthesis}"
            return ModelResponse(
                raw_text=raw,
                think_trace=think,
                final_output=synthesis,
                task="CONTAGION_REASONING",
                latency_sec=0.045,
                prefill_ms=12.0,
                tokens_per_sec=75.0,
                prompt_tokens=150,
                generated_tokens=90,
            )

        elif task in ("E", "PORTFOLIO_RECOMMENDATION"):
            think = (
                "Downstream impact concentrates on MSFT and NVDA equity.\n"
                "Target delta hedge: 30-day put spread on NVDA to cover buffer burn."
            )
            synthesis = (
                "Hedging Directive: Implement a bear put collar on NVDA (buy 95% put, sell 105% call, 45-day expiry) "
                "and reallocate 15% MSFT exposure to non-semiconductor cloud alternatives."
            )
            raw = f"<think>\n{think}\n</think>\n{synthesis}"
            return ModelResponse(
                raw_text=raw,
                think_trace=think,
                final_output=synthesis,
                task="PORTFOLIO_RECOMMENDATION",
                latency_sec=0.038,
                prefill_ms=10.0,
                tokens_per_sec=80.0,
                prompt_tokens=120,
                generated_tokens=70,
            )

        return ModelResponse(
            raw_text="mock",
            think_trace=None,
            final_output="mock",
            task="UNKNOWN",
            latency_sec=0.005,
            prefill_ms=2.0,
            tokens_per_sec=100.0,
            prompt_tokens=10,
            generated_tokens=10,
        )


@pytest.mark.integration
def test_e2e_nl_to_cypher_rag_pipeline_flow(tmp_path: Path):
    """Verify full Flow 3: NL Query -> Cypher -> Guardrail -> Temporal Filter -> Subgraph -> Contagion -> Hedging."""
    graph_store = SeededGraphStore()
    transport = DeterministicModelTransport(fail_schema_check=False)

    class TestClient(YarnBallModelClient):
        def generate(self, *args, **kwargs):
            return transport(*args, **kwargs)

    rejection_queue = RuntimeRejectionQueue(base_dir=tmp_path / "rejections")
    orchestrator = GraphRAGPipelineOrchestrator(
        model_client=TestClient(),
        graph_driver=graph_store,
        rejection_queue=rejection_queue,
    )

    query = AnalystQuery(
        query="If TSM halts wafer production, what is the contagion impact on NVDA and downstream MSFT?",
        as_of_date="2024-06-01",
        portfolio_context={"NVDA": 250000, "MSFT": 400000},
        market_telemetry={"volatility_z_score": 2.1, "car_30d": -0.045},
    )

    result = orchestrator.run(query)

    # 1. Pipeline execution status
    assert result.is_valid_query is True
    assert result.refusal_reason is None
    assert result.total_latency_ms > 0

    # 2. Cypher & Temporal Invariants
    assert result.cypher_query is not None
    assert "coalesce(r.status, 'ACTIVE_CURRENT')" in result.cypher_query
    assert "2024-06-01" in result.cypher_query
    assert len(graph_store.query_log) == 1

    # 3. Hop-ordered verbalization
    assert len(result.retrieved_subgraph) == 2
    assert "[Hop Distance 1]" in result.verbalized_subgraph
    assert "[Hop Distance 2]" in result.verbalized_subgraph
    assert "(TSM) -[:SUPPLIES_TO" in result.verbalized_subgraph
    assert "(NVDA) -[:SUPPLIES_TO" in result.verbalized_subgraph

    # 4. Contagion Reasoning Synthesis (Task D)
    assert result.contagion_think_trace is not None
    assert "60-day inventory buffer" in result.contagion_think_trace
    assert "34% operational impairment" in result.contagion_verdict

    # 5. Portfolio Hedging Recommendation (Task E)
    assert result.hedging_recommendation is not None
    assert "bear put collar on NVDA" in result.hedging_recommendation
    assert result.hedging_think_trace is not None

    # 6. Rejection Queue Invariant
    assert len(rejection_queue.load_records()) == 0


@pytest.mark.integration
def test_e2e_schema_trap_refusal_short_circuits_and_logs(tmp_path: Path):
    """Verify Flow 3 short-circuits on schema violations and stages DPO failure sample."""
    graph_store = SeededGraphStore()
    transport = DeterministicModelTransport(fail_schema_check=True)

    class TestClient(YarnBallModelClient):
        def generate(self, *args, **kwargs):
            return transport(*args, **kwargs)

    rejection_queue = RuntimeRejectionQueue(base_dir=tmp_path / "rejections")
    orchestrator = GraphRAGPipelineOrchestrator(
        model_client=TestClient(),
        graph_driver=graph_store,
        rejection_queue=rejection_queue,
    )

    query = AnalystQuery(
        query="List suppliers where CEO age is under 50",
        as_of_date="2024-06-01",
    )

    result = orchestrator.run(query)

    # 1. Must refuse immediately
    assert result.is_valid_query is False
    assert result.refusal_reason is not None
    assert "ceo_age" in result.refusal_reason
    assert result.contagion_verdict == "(none)"

    # 2. Database was never queried
    assert len(graph_store.query_log) == 0
    assert len(result.retrieved_subgraph) == 0

    # 3. Tripped query recorded in Runtime Rejection Queue
    rejections = rejection_queue.load_records()
    assert len(rejections) == 1
    assert rejections[0].error_stage == ErrorStage.SCHEMA_AST_VIOLATION
    assert "ceo_age" in rejections[0].error_message

