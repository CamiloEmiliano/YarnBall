"""Unit and integration tests for Step 7.3: Pipeline & Prompt Verbalizer Integration.

Verifies:
1. HopOrderedSubgraphVerbalizer formatting with ConvergentShockSummary.
2. End-to-end GraphRAG pipeline cascade execution on diamond network (TT-02A).
3. Injection of pre-computed cascade shock metrics into Task D and Task E prompts.
4. Inventory buffer absorption and friction floor integration in pipeline.
5. Telemetry latency breakdown tracking for cascade_engine_ms.
"""

from typing import Any, Dict, List, Optional
import pytest

from contagion import ConvergentShockSummary
from retrieval.pipeline import (
    AnalystQuery,
    GraphRAGPipelineOrchestrator,
    HopOrderedSubgraphVerbalizer,
    PipelineResult,
)
from sft_pipeline.serving.model_client import ModelResponse, YarnBallModelClient


class MockModelClient(YarnBallModelClient):
    """Deterministic mock model client recording prompt history for verification."""

    def __init__(self) -> None:
        self.call_history: List[Dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        task: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> ModelResponse:
        self.call_history.append({"prompt": prompt, "task": task})

        if task in ("C", "TEXT_TO_CYPHER"):
            raw = "MATCH (s:Company)-[r:SUPPLIES_TO]->(t:Company) RETURN s, r, t"
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
            output = "CONFIRMED_DISRUPTION: Customer C faces severe 36.25% operational impairment."
            return ModelResponse(
                raw_text=f"<think>\nVerified 36.25% shock.\n</think>\n{output}",
                think_trace="Verified 36.25% shock.",
                final_output=output,
                task="CONTAGION_REASONING",
                latency_sec=0.02,
                prefill_ms=10.0,
                tokens_per_sec=80.0,
                prompt_tokens=40,
                generated_tokens=30,
            )
        elif task in ("E", "PORTFOLIO_RECOMMENDATION"):
            output = "Recommend 60-day put collar overlay on Customer C."
            return ModelResponse(
                raw_text=f"<think>\nAnalyzing risk.\n</think>\n{output}",
                think_trace="Analyzing risk.",
                final_output=output,
                task="PORTFOLIO_RECOMMENDATION",
                latency_sec=0.02,
                prefill_ms=10.0,
                tokens_per_sec=90.0,
                prompt_tokens=50,
                generated_tokens=30,
            )
        return ModelResponse(
            raw_text="none",
            think_trace=None,
            final_output="none",
            task=task,
            latency_sec=0.01,
            prefill_ms=1.0,
            tokens_per_sec=100.0,
            prompt_tokens=10,
            generated_tokens=10,
        )


class TestHopOrderedSubgraphVerbalizerCascade:
    """Tests for verbalizer integration with ConvergentShockSummary."""

    def test_verbalize_with_cascade_summary(self) -> None:
        verbalizer = HopOrderedSubgraphVerbalizer()
        edges = [
            {"source": "m1", "target": "C", "share": 0.40, "hop": 1, "status": "ACTIVE_CURRENT"},
            {"source": "m2", "target": "C", "share": 0.45, "hop": 1, "status": "ACTIVE_CURRENT"},
        ]
        summary = ConvergentShockSummary(
            target_entity="C",
            peak_shock_pct=36.25,
            impairment_score=0.3625,
            onset_day=10.0,
            downtime_days=50.0,
            lost_capacity_days=18.125,
            verdict="DISRUPTIVE_SHOCK",
            contributing_edges=[
                {"source": "m1", "input_group": "substrates", "share_pct": 40.0, "source_shock_pct": 40.0},
                {"source": "m2", "input_group": "substrates", "share_pct": 45.0, "source_shock_pct": 45.0},
            ],
            spectral_radius=0.0,
            iteration_count=2,
        )

        verbalized = verbalizer.verbalize(edges, cascade_summary=summary)

        assert "### Deterministic Cascade Shock Metrics" in verbalized
        assert "Target Entity: C" in verbalized
        assert "36.25% capacity loss" in verbalized
        assert "Day 10.0 (Inventory Buffer Delay)" in verbalized
        assert "50.0 calendar days" in verbalized
        assert "DISRUPTIVE_SHOCK" in verbalized
        assert "Active Contributing Upstream Branches:" in verbalized
        assert "(m1) -> Group: 'substrates'" in verbalized
        assert "### Active Subgraph Topology" in verbalized


class TestPipelineCascadeIntegration:
    """Integration tests for GraphRAGPipelineOrchestrator with SparseNetworkCascadeEngine."""

    def test_diamond_pipeline_cascade_handshake(self) -> None:
        """Verifies that diamond graph execution triggers cascade engine and feeds Task D / E."""
        diamond_edges = [
            {"source": "s1", "target": "m1", "nature": "wafers", "share": 0.40, "hop": 1},
            {"source": "s2", "target": "m2", "nature": "wafers", "share": 0.45, "hop": 1},
            {"source": "m1", "target": "C", "nature": "substrates", "share": 0.40, "hop": 2},
            {"source": "m2", "target": "C", "nature": "substrates", "share": 0.45, "hop": 2},
        ]

        def dummy_driver(cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
            return diamond_edges

        mock_client = MockModelClient()
        orchestrator = GraphRAGPipelineOrchestrator(
            model_client=mock_client,
            graph_driver=dummy_driver,
        )

        query = AnalystQuery(
            query="Assess supply disruption from port blockade at s1 and s2 on Customer C.",
            shock_origin="s1, s2",
            shock_magnitude_pct=100.0,
            target_ticker="C",
            portfolio_context={"holdings": ["C"], "nav_usd": 1000000},
        )

        result: PipelineResult = orchestrator.run(query)

        # 1. Pipeline result checks
        assert result.is_valid_query
        assert result.convergent_shock_summary is not None
        summary = result.convergent_shock_summary
        assert summary.target_entity == "C"
        assert summary.peak_shock_pct == pytest.approx(36.25, abs=1e-3)
        assert summary.verdict == "DISRUPTIVE_SHOCK"

        # 2. Telemetry latency check
        assert "cascade_engine_ms" in result.latency_breakdown_ms
        assert result.latency_breakdown_ms["cascade_engine_ms"] >= 0.0

        # 3. Model prompt checks (handshake verification)
        # Task D prompt must contain the exact computed metrics
        task_d_call = next(c for c in mock_client.call_history if c["task"] == "CONTAGION_REASONING")
        assert "36.25% capacity loss" in task_d_call["prompt"]
        assert "DISRUPTIVE_SHOCK" in task_d_call["prompt"]

        # Task E prompt must contain the deterministic shock verdict
        task_e_call = next(c for c in mock_client.call_history if c["task"] == "PORTFOLIO_RECOMMENDATION")
        assert "Deterministic Cascade Verdict: DISRUPTIVE_SHOCK" in task_e_call["prompt"]
        assert "36.25% capacity loss" in task_e_call["prompt"]

    def test_pipeline_inventory_buffer_absorption_handshake(self) -> None:
        """Verifies buffer absorption in the pipeline yields TOLERATED_BUFFERED and 5% friction."""
        buffered_edges = [
            {"source": "s1", "target": "C", "share": 0.80, "buffer_days": 60.0, "hop": 1},
        ]

        def dummy_driver(cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
            return buffered_edges

        mock_client = MockModelClient()
        orchestrator = GraphRAGPipelineOrchestrator(
            model_client=mock_client,
            graph_driver=dummy_driver,
        )

        # 50% shock for 60 days: 0.5 * 60 = 30 days <= 60 days buffer -> fully absorbed
        query = AnalystQuery(
            query="Assess disruption at s1 on C.",
            initial_shocks={"s1": 50.0},
            initial_durations={"s1": 60.0},
            target_ticker="C",
        )

        result: PipelineResult = orchestrator.run(query)

        assert result.convergent_shock_summary is not None
        summary = result.convergent_shock_summary
        assert summary.target_entity == "C"
        assert summary.downtime_days == 0.0
        assert summary.is_friction_floor
        assert summary.peak_shock_pct == pytest.approx(5.0, abs=1e-3)
        assert summary.verdict == "TOLERATED_BUFFERED"
