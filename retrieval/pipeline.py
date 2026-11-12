"""End-to-End GraphRAG Pipeline Orchestrator.

Implements the multi-stage operational GraphRAG execution pipeline:
1. Query Ingestion & Context Preparation (AnalystQuery)
2. Task C: Text-to-Cypher generation via YarnBallModelClient
3. Schema & AST Guardrail Verification via CypherASTWhitelistGuardrail
   - Trips route directly to RuntimeRejectionQueue and short-circuit to '(none)'
4. Temporal State Coalescence Injection via TemporalQueryFilter
5. Graph Execution against Memgraph / Neo4j driver
6. Hop-Ordered Subgraph Verbalizer:
   - Formats edges sorted ascending by hop distance (k=1 -> k=2 -> k=3)
   - Directional path arrows: Source -> TARGET
   - Clutter Ceiling: Caps at Top-E <= 25 edges (prevents Phase 5 TT-01 clutter drop)
7. Task D: Contagion Synthesis with prefix forcing (<think>\n)
8. Task E: Portfolio Hedging Recommendation with market telemetry
9. Telemetry & Latency Breakdown logging
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from contagion import (
    CascadeEdge,
    CascadeGraph,
    CascadeNode,
    ConvergentShockSummary,
    SparseNetworkCascadeEngine,
)
from retrieval.guardrails.schema_guardrail import CypherASTWhitelistGuardrail, ValidationResult
from retrieval.temporal import TemporalQueryFilter
from sft_pipeline.rejection_queue import ErrorStage, RuntimeRejectionQueue
from sft_pipeline.serving.model_client import ModelResponse, YarnBallModelClient

logger = logging.getLogger("retrieval.pipeline")

MAX_SUBGRAPH_EDGES = 25


class AnalystQuery(BaseModel):
    """Input query specification for the GraphRAG pipeline."""

    query: str
    as_of_date: Optional[str] = None
    portfolio_context: Optional[Dict[str, Any]] = None
    target_ticker: Optional[str] = None
    shock_origin: Optional[str] = None
    shock_magnitude_pct: Optional[float] = None
    market_telemetry: Optional[Dict[str, Any]] = None
    initial_shocks: Optional[Dict[str, float]] = None
    initial_durations: Optional[Dict[str, float]] = None
    initial_onsets: Optional[Dict[str, float]] = None


class PipelineResult(BaseModel):
    """Unified response object from the end-to-end GraphRAG pipeline."""

    query: str
    as_of_date: Optional[str] = None
    cypher_query: Optional[str] = None
    is_valid_query: bool
    refusal_reason: Optional[str] = None
    retrieved_subgraph: List[Dict[str, Any]] = Field(default_factory=list)
    verbalized_subgraph: str
    contagion_verdict: str
    contagion_think_trace: Optional[str] = None
    hedging_recommendation: Optional[str] = None
    hedging_think_trace: Optional[str] = None
    latency_breakdown_ms: Dict[str, float] = Field(default_factory=dict)
    total_latency_ms: float
    convergent_shock_summary: Optional[ConvergentShockSummary] = None
    cascade_summaries: Dict[str, ConvergentShockSummary] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline result to JSON-serializable dictionary."""
        d = self.model_dump()
        d["edge_count"] = len(self.retrieved_subgraph)
        return d

    def to_operational_report(
        self,
        event_id: str,
        sec_item: str,
        disrupted_entity_ticker: str,
        filing_date: str,
        disruption_category: str,
        event_summary: str,
        shock_duration_days: float,
        shock_magnitude_pct: float,
        disrupted_entity_name: Optional[str] = None,
        disrupted_entity_cik: Optional[str] = None,
        report_id: Optional[str] = None,
    ) -> Any:
        """Convert pipeline result and event metadata into a validated OperationalContagionReport (ADR-00011)."""
        from .reporting import build_operational_contagion_report

        return build_operational_contagion_report(
            result=self,
            event_id=event_id,
            sec_item=sec_item,
            disrupted_entity_ticker=disrupted_entity_ticker,
            filing_date=filing_date,
            disruption_category=disruption_category,
            event_summary=event_summary,
            shock_duration_days=shock_duration_days,
            shock_magnitude_pct=shock_magnitude_pct,
            disrupted_entity_name=disrupted_entity_name,
            disrupted_entity_cik=disrupted_entity_cik,
            report_id=report_id,
        )


class HopOrderedSubgraphVerbalizer:
    """Verbalizes raw graph query results into clean, hop-ordered textual context.

    Enforces:
    - Sorting by topological hop distance ascending (k=1 -> k=2 -> k=3).
    - Top-E <= 25 edge ceiling to prevent TT-01 context clutter degradation.
    - Explicit directional path formatting: Source -> Rel [loss_ratio, status] -> Target.
    - Deterministic pre-computed cascade shock metric formatting (ADR-00006).
    """

    def __init__(self, max_edges: int = MAX_SUBGRAPH_EDGES):
        self.max_edges = max_edges

    def verbalize(
        self,
        edges: List[Dict[str, Any]],
        cascade_summary: Optional[ConvergentShockSummary] = None,
        all_summaries: Optional[Dict[str, ConvergentShockSummary]] = None,
    ) -> str:
        """Format a list of edge dicts and optional cascade shock summary into structured markdown text."""
        sections: List[str] = []

        if cascade_summary:
            onset_str = f"Day {cascade_summary.onset_day:.1f}" if cascade_summary.onset_day is not None else "Immediate"
            casc_lines = [
                "### Deterministic Cascade Shock Metrics (Sparse Network Cascade Engine):",
                f"- Target Entity: {cascade_summary.target_entity}",
                f"- Operational Impairment: {cascade_summary.impairment_score:.4f} ({cascade_summary.peak_shock_pct:.2f}% capacity loss)",
                f"- Disruption Onset: {onset_str} (Inventory Buffer Delay)",
                f"- Unbuffered Downtime: {cascade_summary.downtime_days:.1f} calendar days",
                f"- Lost Output Capacity-Days: {cascade_summary.lost_capacity_days:.2f} days",
                f"- Spectral Radius (Cycle Stability): {cascade_summary.spectral_radius}",
                f"- Institutional Shock Verdict: {cascade_summary.verdict}",
                f"- Residual Friction Floor Active: {cascade_summary.is_friction_floor}",
                f"- Out of Horizon (>5 hops): {cascade_summary.is_out_of_horizon}",
            ]
            if cascade_summary.contributing_edges:
                casc_lines.append("- Active Contributing Upstream Branches:")
                for c in cascade_summary.contributing_edges:
                    casc_lines.append(
                        f"  * ({c['source']}) -> Group: '{c['input_group']}', "
                        f"Share: {c['share_pct']}%, Source Shock: {c['source_shock_pct']}%"
                    )
            sections.append("\n".join(casc_lines))

        if not edges:
            sections.append("No matching supply-chain relationships discovered in active knowledge graph.")
            return "\n\n".join(sections)

        # 1. Sort edges: hop_distance ascending, then source ticker, target ticker
        sorted_edges = sorted(
            edges,
            key=lambda e: (
                e.get("hop", e.get("hops", e.get("distance", 1))),
                e.get("source", ""),
                e.get("target", ""),
            ),
        )

        # 2. Cap at max_edges (clutter cliff defense)
        clamped_edges = sorted_edges[: self.max_edges]
        lines = [
            f"### Active Subgraph Topology ({len(clamped_edges)} edges retrieved, sorted by hop distance):"
        ]

        current_hop = None
        for edge in clamped_edges:
            hop = edge.get("hop", edge.get("hops", edge.get("distance", 1)))
            if hop != current_hop:
                current_hop = hop
                lines.append(f"\n[Hop Distance {current_hop}]")

            source = edge.get("source", edge.get("source_ticker", "UNKNOWN"))
            target = edge.get("target", edge.get("target_ticker", "UNKNOWN"))
            rel_type = edge.get("rel_type", edge.get("type", "SUPPLIES_TO"))
            status = edge.get("status", "ACTIVE_CURRENT")
            loss_ratio = edge.get("loss_ratio", edge.get("loss_pct"))
            revenue_share = edge.get("revenue_share_pct", edge.get("revenue_share"))

            props = [f"status: '{status}'"]
            if loss_ratio is not None:
                props.append(f"loss_ratio: {loss_ratio}")
            if revenue_share is not None:
                props.append(f"revenue_share: {revenue_share}%")
            if "buffer_days" in edge:
                props.append(f"buffer_days: {edge['buffer_days']}")

            prop_str = f" {{{', '.join(props)}}}" if props else ""
            lines.append(f"- ({source}) -[:{rel_type}{prop_str}]-> ({target})")

        if len(sorted_edges) > self.max_edges:
            lines.append(
                f"\n(Note: {len(sorted_edges) - self.max_edges} additional distant edges pruned to maintain context fidelity)"
            )

        sections.append("\n".join(lines))
        return "\n\n".join(sections)


class GraphRAGPipelineOrchestrator:
    """Coordinated end-to-end GraphRAG pipeline engine.

    Wires Stage 6.1 (Guardrail), Stage 6.3 (Temporal), Stage 6.4 (ModelClient),
    and Stage 6.6 (RejectionQueue) into a coherent, verifiable operational agent.
    """

    def __init__(
        self,
        model_client: Optional[YarnBallModelClient] = None,
        schema_guardrail: Optional[CypherASTWhitelistGuardrail] = None,
        temporal_filter: Optional[TemporalQueryFilter] = None,
        rejection_queue: Optional[RuntimeRejectionQueue] = None,
        graph_driver: Optional[Any] = None,
        verbalizer: Optional[HopOrderedSubgraphVerbalizer] = None,
        cascade_engine: Optional[SparseNetworkCascadeEngine] = None,
    ):
        self.model_client = model_client or YarnBallModelClient()
        self.schema_guardrail = schema_guardrail or CypherASTWhitelistGuardrail()
        self.temporal_filter = temporal_filter or TemporalQueryFilter()
        self.rejection_queue = rejection_queue or RuntimeRejectionQueue()
        self.verbalizer = verbalizer or HopOrderedSubgraphVerbalizer()
        self.graph_driver = graph_driver
        self.cascade_engine = cascade_engine or SparseNetworkCascadeEngine()

    def execute_graph_query(
        self, cypher: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute validated and temporalized Cypher against the graph driver."""
        if self.graph_driver is None:
            logger.debug("No graph driver provided; returning empty subgraph.")
            return []

        params = params or {}
        results: List[Dict[str, Any]] = []

        try:
            # Check if driver is callable or has session() method
            if hasattr(self.graph_driver, "session"):
                with self.graph_driver.session() as session:
                    records = session.run(cypher, **params)
                    for record in records:
                        if hasattr(record, "data"):
                            results.append(record.data())
                        elif isinstance(record, dict):
                            results.append(record)
                        else:
                            # Attempt dict conversion
                            try:
                                results.append(dict(record))
                            except Exception:
                                results.append({"raw": str(record)})
            elif callable(self.graph_driver):
                raw = self.graph_driver(cypher, params)
                if isinstance(raw, list):
                    results = raw
                else:
                    results = [raw]
            else:
                logger.warning("Unrecognized graph_driver interface: %s", type(self.graph_driver))
        except Exception as exc:
            logger.error("Graph driver execution error: %s", exc)
            self.rejection_queue.record_failure(
                task_id="retrieval_driver_error",
                prompt=cypher,
                completion=str(exc),
                error_stage=ErrorStage.SYNTAX_MALFORMED,
                error_message=f"Graph execution failed: {exc}",
            )

        return results

    def _normalize_graph_records_to_edges(
        self, records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Normalize raw database records into standardized edge dictionaries."""
        edges: List[Dict[str, Any]] = []

        for r in records:
            # Check standard expected edge structure
            if "source" in r and "target" in r:
                edges.append(r)
                continue

            # Check if record has path or node/rel structure
            if "r" in r and isinstance(r["r"], dict):
                rel = r["r"]
                source = r.get("s", {}).get("ticker") or r.get("source") or "UNKNOWN"
                target = r.get("t", {}).get("ticker") or r.get("target") or "UNKNOWN"
                edge_dict = {
                    "source": source,
                    "target": target,
                    "rel_type": rel.get("type", "SUPPLIES_TO"),
                    "status": rel.get("status", "ACTIVE_CURRENT"),
                    "loss_ratio": rel.get("loss_ratio"),
                    "hop": r.get("hop", 1),
                }
                edges.append(edge_dict)
            elif "edges" in r and isinstance(r["edges"], list):
                edges.extend(r["edges"])
            else:
                # Flat fallback
                edges.append(r)

        return edges

    def _build_cascade_graph(self, normalized_edges: List[Dict[str, Any]]) -> CascadeGraph:
        """Constructs a CascadeGraph from normalized database records."""
        graph = CascadeGraph()
        for r in normalized_edges:
            source = str(r.get("source") or r.get("source_ticker") or "UNKNOWN")
            target = str(r.get("target") or r.get("target_ticker") or "UNKNOWN")
            input_group = str(r.get("input_group") or r.get("nature") or r.get("group") or "default")

            # Extract procurement share a_uv
            if "share" in r and r["share"] is not None:
                share_val = float(r["share"])
            elif "exposure_pct" in r and r["exposure_pct"] is not None:
                share_val = float(r["exposure_pct"]) / 100.0
            elif "revenue_share_pct" in r and r["revenue_share_pct"] is not None:
                share_val = float(r["revenue_share_pct"]) / 100.0
            elif "revenue_share" in r and r["revenue_share"] is not None:
                val = float(r["revenue_share"])
                share_val = val / 100.0 if val > 1.0 else val
            elif "loss_ratio" in r and r["loss_ratio"] is not None:
                val = float(r["loss_ratio"])
                share_val = val / 100.0 if val > 1.0 else val
            else:
                share_val = 1.0

            share_val = max(0.0, min(1.0, share_val))
            buffer_days = float(r.get("buffer_days", 0.0) or 0.0)
            materiality = str(r.get("materiality", r.get("materiality_tier", "NON_CRITICAL")))

            graph.add_edge(
                CascadeEdge(
                    source=source,
                    target=target,
                    input_group=input_group,
                    share=share_val,
                    buffer_days=buffer_days,
                    materiality_tier=materiality,
                )
            )
        return graph

    def run(self, analyst_query: AnalystQuery) -> PipelineResult:
        """Run the full end-to-end GraphRAG operational pipeline."""
        start_time = time.perf_counter()
        latencies: Dict[str, float] = {}

        # -------------------------------------------------------------
        # STEP 1: Text-to-Cypher Generation (Task C)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        cypher_prompt = (
            f"Generate an openCypher query to retrieve supply-chain relationships for:\n"
            f"Query: {analyst_query.query}\n"
            f"As-of Date: {analyst_query.as_of_date or 'CURRENT'}"
        )
        cypher_resp: ModelResponse = self.model_client.generate(
            prompt=cypher_prompt,
            task="TEXT_TO_CYPHER",
            temperature=0.0,
        )
        latencies["task_c_cypher_ms"] = (time.perf_counter() - t0) * 1000.0

        generated_cypher = cypher_resp.final_output.strip()

        # -------------------------------------------------------------
        # STEP 2: Deterministic Schema & AST Guardrail (Stage 6.1)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        validation: ValidationResult = self.schema_guardrail.validate(generated_cypher)
        latencies["guardrail_validation_ms"] = (time.perf_counter() - t0) * 1000.0

        if not validation.is_valid:
            # Query violates AST schema whitelist or requests unanswerable field
            logger.info(
                "Cypher AST validation tripped: %s. Short-circuiting with '(none)'.",
                validation.error_summary,
            )

            # Log tripped sample to Stage 6.6 Rejection Queue
            self.rejection_queue.record_failure(
                task_id=f"pipeline_guardrail_{int(time.time())}",
                prompt=analyst_query.query,
                completion=generated_cypher,
                error_stage=ErrorStage.SCHEMA_AST_VIOLATION,
                error_message=validation.error_summary or "Schema AST violation",
                metadata={
                    "violations": [v.message for v in validation.violations],
                    "as_of_date": analyst_query.as_of_date,
                },
            )

            total_ms = (time.perf_counter() - start_time) * 1000.0
            return PipelineResult(
                query=analyst_query.query,
                as_of_date=analyst_query.as_of_date,
                cypher_query=generated_cypher,
                is_valid_query=False,
                refusal_reason=validation.error_summary,
                retrieved_subgraph=[],
                verbalized_subgraph="No subgraph retrieved. Query refused due to schema violation.",
                contagion_verdict="(none)",
                contagion_think_trace=None,
                hedging_recommendation=None,
                hedging_think_trace=None,
                latency_breakdown_ms=latencies,
                total_latency_ms=total_ms,
            )

        sanitized_cypher = validation.sanitized_query or generated_cypher

        # -------------------------------------------------------------
        # STEP 3: Temporal Coalescence Injection (Stage 6.3)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        temporalized_cypher = self.temporal_filter.inject_active_coalescence(
            cypher=sanitized_cypher,
            as_of_date=analyst_query.as_of_date,
        )
        latencies["temporal_transform_ms"] = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # STEP 4: Graph Database Execution
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        raw_records = self.execute_graph_query(
            temporalized_cypher,
            params={"as_of_date": analyst_query.as_of_date} if analyst_query.as_of_date else {},
        )
        normalized_edges = self._normalize_graph_records_to_edges(raw_records)
        latencies["graph_execution_ms"] = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # STEP 5: Hop-Ordered Subgraph Verbalization & Cascade Diffusion
        # -------------------------------------------------------------
        t0 = time.perf_counter()

        # Build initial shock parameters
        initial_shocks: Dict[str, float] = {}
        if analyst_query.initial_shocks:
            initial_shocks.update(analyst_query.initial_shocks)
        elif analyst_query.shock_origin:
            mag = analyst_query.shock_magnitude_pct if analyst_query.shock_magnitude_pct is not None else 100.0
            for orig in analyst_query.shock_origin.split(","):
                orig_clean = orig.strip()
                if orig_clean:
                    initial_shocks[orig_clean] = mag

        cascade_summaries: Dict[str, ConvergentShockSummary] = {}
        target_summary: Optional[ConvergentShockSummary] = None

        if normalized_edges and initial_shocks:
            t_casc = time.perf_counter()
            cascade_graph = self._build_cascade_graph(normalized_edges)
            try:
                cascade_summaries = self.cascade_engine.propagate(
                    graph=cascade_graph,
                    initial_shocks=initial_shocks,
                    initial_durations=analyst_query.initial_durations,
                    initial_onsets=analyst_query.initial_onsets,
                )
                if analyst_query.target_ticker and analyst_query.target_ticker in cascade_summaries:
                    target_summary = cascade_summaries[analyst_query.target_ticker]
                elif cascade_summaries:
                    # Select primary downstream target (max shock not in initial shock seeds)
                    non_seeds = {
                        nid: s for nid, s in cascade_summaries.items() if nid not in initial_shocks
                    }
                    if non_seeds:
                        target_summary = max(non_seeds.values(), key=lambda s: s.peak_shock_pct)
                    else:
                        target_summary = max(cascade_summaries.values(), key=lambda s: s.peak_shock_pct)
            except Exception as exc:
                logger.warning("Cascade engine propagation warning: %s", exc)
            latencies["cascade_engine_ms"] = (time.perf_counter() - t_casc) * 1000.0

        verbalized_context = self.verbalizer.verbalize(
            normalized_edges,
            cascade_summary=target_summary,
            all_summaries=cascade_summaries,
        )
        latencies["verbalization_ms"] = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # STEP 6: Contagion Reasoning Synthesis (Task D)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        contagion_prompt = (
            f"Query: {analyst_query.query}\n\n"
            f"{verbalized_context}\n\n"
            f"Analyze potential supply-chain disruption propagation, inventory buffer mitigation, "
            f"and operational risk transmission. Base your reasoning on the deterministic cascade "
            f"shock metrics pre-computed above. Provide a rigorous assessment."
        )
        contagion_resp: ModelResponse = self.model_client.generate(
            prompt=contagion_prompt,
            task="CONTAGION_REASONING",
            temperature=0.1,
        )
        latencies["task_d_contagion_ms"] = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # STEP 7: Portfolio Recommendation & Hedging (Task E)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        hedging_resp: Optional[ModelResponse] = None
        if analyst_query.portfolio_context or analyst_query.market_telemetry:
            shock_summary_line = ""
            if target_summary:
                shock_summary_line = (
                    f"Deterministic Cascade Verdict: {target_summary.verdict} "
                    f"({target_summary.peak_shock_pct:.2f}% capacity loss, "
                    f"{target_summary.downtime_days:.1f} days downtime).\n"
                )
            hedging_prompt = (
                f"Contagion Assessment:\n{contagion_resp.final_output}\n\n"
                f"{shock_summary_line}"
                f"Portfolio Context: {analyst_query.portfolio_context}\n"
                f"Market Telemetry: {analyst_query.market_telemetry}\n\n"
                f"Recommend specific portfolio rebalancing, option collars, or hedge ratios."
            )
            hedging_resp = self.model_client.generate(
                prompt=hedging_prompt,
                task="PORTFOLIO_RECOMMENDATION",
                temperature=0.1,
            )
        latencies["task_e_hedging_ms"] = (time.perf_counter() - t0) * 1000.0

        total_ms = (time.perf_counter() - start_time) * 1000.0

        return PipelineResult(
            query=analyst_query.query,
            as_of_date=analyst_query.as_of_date,
            cypher_query=temporalized_cypher,
            is_valid_query=True,
            refusal_reason=None,
            retrieved_subgraph=normalized_edges,
            verbalized_subgraph=verbalized_context,
            contagion_verdict=contagion_resp.final_output,
            contagion_think_trace=contagion_resp.think_trace,
            hedging_recommendation=hedging_resp.final_output if hedging_resp else None,
            hedging_think_trace=hedging_resp.think_trace if hedging_resp else None,
            latency_breakdown_ms=latencies,
            total_latency_ms=total_ms,
            convergent_shock_summary=target_summary,
            cascade_summaries=cascade_summaries,
        )
