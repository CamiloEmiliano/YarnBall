"""Unit and integration tests for End-to-End GraphRAG Pipeline Orchestrator (Stage 6.5).

Tests:
1. Complete happy path: AnalystQuery -> Cypher -> Whitelist -> Temporal -> Graph Driver -> Verbalizer -> Task D -> Task E.
2. Schema trap refusal short-circuit: Guardrail catches unwhitelisted attributes (e.g. 'ceo_age'),
   logs to Rejection Queue, and returns refusal without querying graph.
3. Hop-ordered verbalizer: Ascending hop distance sorting, directional arrows, and clutter ceiling (<=25 edges).
4. Empty graph response handling: Graceful handling of zero matched relationships.
5. In-memory dummy graph driver and mock model engine execution.
"""

from __future__ import annotations

import os
import shutil
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


class MockModelClient(YarnBallModelClient):
    """Deterministic mock model client for testing pipeline orchestration."""

    def __init__(self, cypher_output: str = "", contagion_output: str = "", hedging_output: str = ""):
        self.cypher_output = cypher_output
        self.contagion_output = contagion_output
        self.hedging_output = hedging_output
        self.call_history: List[Dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        task: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> ModelResponse:
        self.call_history.append({"prompt": prompt, "task": task, "temperature": temperature})

        if task in ("C", "TEXT_TO_CYPHER"):
            raw = self.cypher_output or "MATCH (s:Company {ticker: 'NVDA'})-[r:SUPPLIES_TO]->(t:Company {ticker: 'AAPL'}) RETURN s, r, t"
            return ModelResponse(
                raw_text=raw,
                think_trace=None,
                final_output=raw,
                task="TEXT_TO_CYPHER",
                latency_sec=0.01,
                prefill_ms=5.0,
                tokens_per_sec=100.0,
                prompt_tokens=20,
                generated_tokens=20,
            )

        elif task in ("D", "CONTAGION_REASONING"):
            raw = (
                "<think>\nEvaluating 45-day buffer vs shock magnitude.\n</think>\n"
                + (self.contagion_output or "AAPL faces moderate disruption mitigated by 45-day buffer stock.")
            )
            return ModelResponse(
                raw_text=raw,
                think_trace="Evaluating 45-day buffer vs shock magnitude.",
                final_output=self.contagion_output or "AAPL faces moderate disruption mitigated by 45-day buffer stock.",
                task="CONTAGION_REASONING",
                latency_sec=0.025,
                prefill_ms=10.0,
                tokens_per_sec=80.0,
                prompt_tokens=40,
                generated_tokens=50,
            )

        elif task in ("E", "PORTFOLIO_RECOMMENDATION"):
            raw = (
                "<think>\nAnalyzing IV and delta hedge ratios.\n</think>\n"
                + (self.hedging_output or "Recommend purchase of AAPL 30-day 95% put collars.")
            )
            return ModelResponse(
                raw_text=raw,
                think_trace="Analyzing IV and delta hedge ratios.",
                final_output=self.hedging_output or "Recommend purchase of AAPL 30-day 95% put collars.",
                task="PORTFOLIO_RECOMMENDATION",
                latency_sec=0.02,
                prefill_ms=10.0,
                tokens_per_sec=90.0,
                prompt_tokens=50,
                generated_tokens=40,
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


class MockGraphDriver:
    """Mock graph database driver providing deterministic graph records."""

    def __init__(self, records: Optional[List[Dict[str, Any]]] = None):
        self.records = records or []
        self.executed_queries: List[str] = []

    def session(self):
        driver = self

        class _Session:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def run(self, query, **kwargs):
                driver.executed_queries.append(query)
                return driver.records

        return _Session()


@pytest.fixture
def temp_rejection_dir():
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_hop_ordered_subgraph_verbalizer():
    """Test HopOrderedSubgraphVerbalizer sorts ascending by hop and clamps at 25 edges."""
    verbalizer = HopOrderedSubgraphVerbalizer(max_edges=3)

    edges = [
        {"source": "INTC", "target": "MSFT", "hop": 2, "rel_type": "SUPPLIES_TO", "loss_ratio": 0.3},
        {"source": "TSM", "target": "NVDA", "hop": 1, "rel_type": "SUPPLIES_TO", "loss_ratio": 0.8},
        {"source": "ASML", "target": "TSM", "hop": 1, "rel_type": "SUPPLIES_TO", "loss_ratio": 0.95},
        {"source": "MSFT", "target": "AMZN", "hop": 3, "rel_type": "COMPETES_WITH"},
    ]

    formatted = verbalizer.verbalize(edges)
    assert "[Hop Distance 1]" in formatted
    assert "[Hop Distance 2]" in formatted
    assert "(ASML) -[:SUPPLIES_TO" in formatted
    assert "(TSM) -[:SUPPLIES_TO" in formatted
    assert "(INTC) -[:SUPPLIES_TO" in formatted
    # 4th edge should be pruned due to max_edges=3
    assert "MSFT) -[:COMPETES_WITH" not in formatted
    assert "1 additional distant edges pruned" in formatted


def test_pipeline_happy_path(temp_rejection_dir):
    """Test full pipeline traversal with valid query and mock graph return."""
    model = MockModelClient()
    driver = MockGraphDriver([
        {
            "source": "NVDA",
            "target": "AAPL",
            "rel_type": "SUPPLIES_TO",
            "status": "ACTIVE_CURRENT",
            "loss_ratio": 0.45,
            "hop": 1,
        }
    ])
    rej_queue = RuntimeRejectionQueue(base_dir=temp_rejection_dir)

    orchestrator = GraphRAGPipelineOrchestrator(
        model_client=model,
        graph_driver=driver,
        rejection_queue=rej_queue,
    )

    query = AnalystQuery(
        query="If NVDA suffers a factory disruption, how does AAPL mitigate the shock?",
        as_of_date="2024-05-01",
        portfolio_context={"AAPL": 150000},
        market_telemetry={"volatility_z_score": 1.8},
    )

    result = orchestrator.run(query)

    assert result.is_valid_query is True
    assert result.refusal_reason is None
    assert "coalesce(r.status, 'ACTIVE_CURRENT')" in result.cypher_query
    assert len(result.retrieved_subgraph) == 1
    assert "NVDA" in result.verbalized_subgraph
    assert "AAPL faces moderate disruption" in result.contagion_verdict
    assert result.contagion_think_trace == "Evaluating 45-day buffer vs shock magnitude."
    assert "AAPL 30-day 95% put collars" in result.hedging_recommendation
    assert result.total_latency_ms > 0
    assert "task_c_cypher_ms" in result.latency_breakdown_ms
    assert "task_d_contagion_ms" in result.latency_breakdown_ms
    assert "task_e_hedging_ms" in result.latency_breakdown_ms

    # Verify rejection queue has 0 rejections
    assert len(rej_queue.load_records()) == 0


def test_pipeline_schema_trap_refusal(temp_rejection_dir):
    """Test pipeline short-circuits and refuses queries asking for hallucinated schema fields."""
    model = MockModelClient(
        cypher_output="MATCH (c:Company) WHERE c.ceo_age < 50 RETURN c.name, c.ceo_age"
    )
    driver = MockGraphDriver()
    rej_queue = RuntimeRejectionQueue(base_dir=temp_rejection_dir)

    orchestrator = GraphRAGPipelineOrchestrator(
        model_client=model,
        graph_driver=driver,
        rejection_queue=rej_queue,
    )

    query = AnalystQuery(
        query="Find all suppliers whose CEO is under 50 years old.",
        as_of_date="2024-05-01",
    )

    result = orchestrator.run(query)

    # Must short-circuit immediately
    assert result.is_valid_query is False
    assert result.refusal_reason is not None
    assert "ceo_age" in result.refusal_reason
    assert result.contagion_verdict == "(none)"
    assert len(result.retrieved_subgraph) == 0
    assert len(driver.executed_queries) == 0  # Graph driver was NEVER contacted

    # Verify failure was logged to Rejection Queue
    rejections = rej_queue.load_records()
    assert len(rejections) == 1
    assert rejections[0].error_stage == ErrorStage.SCHEMA_AST_VIOLATION
    assert "ceo_age" in rejections[0].error_message


def test_pipeline_empty_subgraph_handling(temp_rejection_dir):
    """Test pipeline gracefully handles zero graph results without crashing."""
    model = MockModelClient()
    driver = MockGraphDriver([])  # Zero records returned
    rej_queue = RuntimeRejectionQueue(base_dir=temp_rejection_dir)

    orchestrator = GraphRAGPipelineOrchestrator(
        model_client=model,
        graph_driver=driver,
        rejection_queue=rej_queue,
    )

    query = AnalystQuery(query="Are there any direct links between UnknownCorp and AcmeCorp?")
    result = orchestrator.run(query)

    assert result.is_valid_query is True
    assert len(result.retrieved_subgraph) == 0
    assert "No matching supply-chain relationships discovered" in result.verbalized_subgraph
    assert result.contagion_verdict != "(none)"
