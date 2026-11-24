"""End-to-End GraphRAG Pipeline Orchestrator (ADR-00033 / ADR-00034).

Implements the consolidated operational Domain 1 GraphRAG execution pipeline:
1. Query Ingestion & Context Preparation (`AnalystQuery` with mandatory `PiTContext`)
2. Optional Hybrid Point-in-Time Seeding (`HybridSeeder`: SCD2 dense entities + news passages)
3. Task C: Text-to-Cypher generation via `YarnBallModelClient`
4. Schema & AST Guardrail Verification via `CypherASTWhitelistGuardrail`
   - Trips route directly to `RuntimeRejectionQueue` and short-circuit to '(none)'
5. Temporal Interval Enforcement via `TemporalQueryFilter` (ADR-00033)
6. Unified Graph Execution via `SubgraphStorePort` (`MemgraphSubgraphStore` / `InMemorySubgraphStore`)
7. Deterministic Cascade Diffusion (`SparseNetworkCascadeEngine`) & Hop-Ordered Subgraph Verbalizer
   - Sorts edges ascending by hop distance (`k=1 -> k=2 -> k=3`)
   - Enforces Top-E <= 25 clutter ceiling
8. Task D: Contagion Reasoning Synthesis via `YarnBallModelClient` with prefix forcing (`<think>\n`)
9. Telemetry & Latency Breakdown logging
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, Field, model_validator

from contagion import (
    CascadeEdge,
    CascadeGraph,
    ConvergentShockSummary,
    SparseNetworkCascadeEngine,
)
from knowledge_graph.errors import StoreUnavailableError
from retrieval.guardrails.schema_guardrail import CypherASTWhitelistGuardrail, ValidationResult
from retrieval.temporal import PiTContext, TemporalQueryFilter, validate_hops
from sft_pipeline.rejection_queue import ErrorStage, RuntimeRejectionQueue
from sft_pipeline.serving.model_client import ModelResponse, YarnBallModelClient

if TYPE_CHECKING:
    from retrieval.facade import SubgraphStorePort
    from retrieval.hybrid import HybridSeeder

logger = logging.getLogger("retrieval.pipeline")

MAX_SUBGRAPH_EDGES = 25


class AnalystQuery(BaseModel):
    """Domain 1 operational BI query specification requiring explicit `PiTContext` (ADR-00033 / ADR-00034)."""

    query: str
    ctx: PiTContext
    target_ticker: Optional[str] = None
    shock_origin: Optional[str] = None
    shock_magnitude_pct: Optional[float] = None
    initial_shocks: Optional[Dict[str, float]] = None
    initial_durations: Optional[Dict[str, float]] = None
    initial_onsets: Optional[Dict[str, float]] = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_pit_context(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        if payload.get("ctx") is None:
            raw_date = payload.pop("as_of_date", None)
            if raw_date is None or not str(raw_date).strip():
                raise ValueError(
                    "AnalystQuery requires explicit `ctx` (PiTContext) or non-empty `as_of_date` (ADR-00033)."
                )
            pit_mode = payload.pop("pit_mode", "historical")
            include_unanchored = bool(payload.pop("include_unanchored", False))
            payload["ctx"] = PiTContext.from_date(
                raw_date,
                mode=pit_mode,
                include_unanchored=include_unanchored,
            )
        else:
            payload.pop("as_of_date", None)
            payload.pop("pit_mode", None)
            payload.pop("include_unanchored", None)
        return payload

    @property
    def as_of_date(self) -> str:
        """Canonical ISO-8601 `as_of` date derived from `self.ctx`."""
        return self.ctx.as_of_iso


class PipelineResult(BaseModel):
    """Unified Domain 1 response object from the consolidated BI GraphRAG pipeline (ADR-00034)."""

    query: str
    as_of_date: str
    cypher_query: Optional[str] = None
    is_valid_query: bool
    degraded: bool = False
    degraded_reasons: List[str] = Field(default_factory=list)
    refusal_reason: Optional[str] = None
    retrieved_subgraph: List[Dict[str, Any]] = Field(default_factory=list)
    seed_hits: List[Dict[str, Any]] = Field(default_factory=list)
    articles: List[Dict[str, Any]] = Field(default_factory=list)
    unanchored_excluded: int = 0
    verbalized_subgraph: str
    contagion_verdict: str
    contagion_think_trace: Optional[str] = None
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
        filing_timestamp: Optional[str] = None,
        model_mode: str = "LIVE_LLM",
    ) -> Any:
        """Convert pipeline result and event metadata into a validated OperationalContagionReport (ADR-00011)."""
        if self.degraded:
            reasons = "; ".join(self.degraded_reasons) or self.refusal_reason or "backing store unavailable"
            raise StoreUnavailableError(
                f"Cannot build OperationalContagionReport from degraded retrieval result: {reasons}"
            )
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
            filing_timestamp=filing_timestamp,
            model_mode=model_mode,
        )


class HopOrderedSubgraphVerbalizer:
    """Verbalizes raw graph query results into clean, hop-ordered textual context.

    Enforces:
    - Sorting by topological hop distance ascending (k=1 -> k=2 -> k=3).
    - Top-E <= 25 edge ceiling to prevent TT-01 context clutter degradation.
    - Explicit directional path formatting: Source -> Rel [loss_ratio, status] -> Target.
    - Deterministic pre-computed cascade shock metric formatting (ADR-00006).
    """

    def __init__(
        self,
        max_edges: int = MAX_SUBGRAPH_EDGES,
        chunks_parquet_path: Optional[Path | str] = None,
        chunk_fetcher: Optional[Callable[[Sequence[str]], Dict[str, Dict[str, Any]]]] = None,
    ):
        self.max_edges = max_edges
        self.chunks_parquet_path = chunks_parquet_path
        self.chunk_fetcher = chunk_fetcher

    def _fetch_evidence_chunks(self, chunk_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        unique_ids = list(dict.fromkeys(str(cid).strip() for cid in chunk_ids if str(cid).strip()))
        if not unique_ids:
            return {}
        if self.chunk_fetcher is not None:
            return self.chunk_fetcher(unique_ids)
        from knowledge_graph.chunk_store import fetch_chunks_by_ids

        return fetch_chunks_by_ids(unique_ids, chunks_parquet_path=self.chunks_parquet_path)

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

        # 2. Cap at max_edges (clutter cliff defense) & collect evidence chunk IDs ONLY for clamped edges (ADR-00035)
        clamped_edges = sorted_edges[: self.max_edges]
        clamped_chunk_ids: List[str] = []
        edge_chunk_lists: List[List[str]] = []
        for edge in clamped_edges:
            props_raw = edge.get("properties")
            eprops: Mapping[str, Any] = props_raw if isinstance(props_raw, Mapping) else {}
            raw_cids = edge.get("evidence_chunk_ids") or eprops.get("evidence_chunk_ids") or []
            cids = [str(c).strip() for c in raw_cids if str(c).strip()] if isinstance(raw_cids, (list, tuple, set)) else []
            edge_chunk_lists.append(cids)
            clamped_chunk_ids.extend(cids)

        resolved_chunks = self._fetch_evidence_chunks(clamped_chunk_ids) if clamped_chunk_ids else {}

        lines = [
            f"### Active Subgraph Topology ({len(clamped_edges)} edges retrieved, sorted by hop distance):"
        ]
        evidence_passage_lines: List[str] = []

        current_hop = None
        for edge_idx, (edge, cids) in enumerate(zip(clamped_edges, edge_chunk_lists), start=1):
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
            citation_suffix = ""
            if cids:
                primary_cid = cids[0]
                citation_suffix = f" [E{edge_idx} ← {primary_cid}]"
                chunk_info = resolved_chunks.get(primary_cid)
                if chunk_info and chunk_info.get("text"):
                    snippet = " ".join(str(chunk_info["text"]).split())[:240]
                    evidence_passage_lines.append(f'- [E{edge_idx} ← {primary_cid}] "{snippet}"')

            lines.append(f"- ({source}) -[:{rel_type}{prop_str}]-> ({target}){citation_suffix}")

        if len(sorted_edges) > self.max_edges:
            lines.append(
                f"\n(Note: {len(sorted_edges) - self.max_edges} additional distant edges pruned to maintain context fidelity)"
            )

        sections.append("\n".join(lines))
        if evidence_passage_lines:
            sections.append(
                "### Supporting Evidence Passages (ADR-00035):\n" + "\n".join(evidence_passage_lines)
            )
        return "\n\n".join(sections)


class GraphRAGPipelineOrchestrator:
    """Coordinated end-to-end Domain 1 BI GraphRAG pipeline engine (ADR-00034).

    Wires `HybridSeeder` (optional), `CypherASTWhitelistGuardrail`, `TemporalQueryFilter`,
    `SubgraphStorePort`, `SparseNetworkCascadeEngine`, `HopOrderedSubgraphVerbalizer`,
    `YarnBallModelClient`, and `RuntimeRejectionQueue` into one auditable execution path.
    """

    def __init__(
        self,
        model_client: Optional[YarnBallModelClient] = None,
        schema_guardrail: Optional[CypherASTWhitelistGuardrail] = None,
        temporal_filter: Optional[TemporalQueryFilter] = None,
        rejection_queue: Optional[RuntimeRejectionQueue] = None,
        graph_driver: Optional[Any] = None,
        store: Optional["SubgraphStorePort"] = None,
        seeder: Optional["HybridSeeder"] = None,
        verbalizer: Optional[HopOrderedSubgraphVerbalizer] = None,
        cascade_engine: Optional[SparseNetworkCascadeEngine] = None,
        seed_hops: int = 2,
    ):
        from retrieval.facade import MemgraphSubgraphStore, SubgraphStorePort

        self.model_client = model_client or YarnBallModelClient()
        self.schema_guardrail = schema_guardrail or CypherASTWhitelistGuardrail()
        self.temporal_filter = temporal_filter or TemporalQueryFilter()
        self.rejection_queue = rejection_queue or RuntimeRejectionQueue()
        self.verbalizer = verbalizer or HopOrderedSubgraphVerbalizer()
        self.cascade_engine = cascade_engine or SparseNetworkCascadeEngine()
        self.seeder = seeder
        self.seed_hops = validate_hops(seed_hops)

        self.graph_driver = graph_driver
        self.store: Optional[SubgraphStorePort] = None
        if store is not None:
            self.store = store
        elif graph_driver is not None:
            if isinstance(graph_driver, SubgraphStorePort):
                self.store = graph_driver
            elif hasattr(graph_driver, "session"):
                self.store = MemgraphSubgraphStore(
                    driver=graph_driver,
                    temporal_filter=self.temporal_filter,
                    schema_guardrail=self.schema_guardrail,
                )

    def execute_graph_query(
        self,
        cypher: str,
        ctx: PiTContext,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute validated and ADR-00033 temporalized Cypher via `SubgraphStorePort`."""
        if not isinstance(ctx, PiTContext):
            raise TypeError(f"ctx must be a PiTContext instance, got {type(ctx).__name__}")
        if self.store is None and self.graph_driver is None:
            logger.debug("No subgraph store or graph driver provided; returning empty subgraph.")
            return []

        query_params = dict(params or {})
        query_params.setdefault("as_of_date", ctx.as_of_iso)

        try:
            if self.store is not None:
                return self.store.execute_guarded(cypher, query_params, ctx)
            if callable(self.graph_driver):
                scoped_cypher = self.temporal_filter.inject_active_coalescence(cypher, ctx)
                raw = self.graph_driver(scoped_cypher, query_params)
                return raw if isinstance(raw, list) else [raw]
            logger.warning("Unrecognized graph_driver interface: %s", type(self.graph_driver))
            return []
        except StoreUnavailableError:
            logger.error("Graph store unavailable during query execution.")
            raise
        except Exception as exc:
            logger.error("Graph store execution error: %s", exc)
            self.rejection_queue.record_failure(
                task_id="retrieval_driver_error",
                prompt=cypher,
                completion=str(exc),
                error_stage=ErrorStage.SYNTAX_MALFORMED,
                error_message=f"Graph execution failed: {exc}",
            )
            raise

    def _normalize_graph_records_to_edges(
        self, records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Normalize raw database records into standardized edge dictionaries."""
        edges: List[Dict[str, Any]] = []

        for r in records:
            if "source" in r and "target" in r:
                edges.append(r)
                continue

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
                edges.append(r)

        return edges

    @staticmethod
    def _merge_edges(
        primary_edges: List[Dict[str, Any]],
        seeded_edges: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Deduplicate edges across guarded Cypher and hybrid seed traversal, keeping min hop."""
        if not seeded_edges:
            return primary_edges
        if not primary_edges:
            return seeded_edges

        merged: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        passthrough: List[Dict[str, Any]] = []

        for edge in [*primary_edges, *seeded_edges]:
            if "source" not in edge or "target" not in edge:
                passthrough.append(edge)
                continue
            src = str(edge.get("source", ""))
            rel = str(edge.get("rel_type", edge.get("type", "SUPPLIES_TO")))
            tgt = str(edge.get("target", ""))
            grp = str(edge.get("input_group") or edge.get("nature") or "default")
            key = (src, rel, tgt, grp)
            if key not in merged or int(edge.get("hop", 1) or 1) < int(merged[key].get("hop", 1) or 1):
                merged[key] = edge

        ordered = sorted(
            merged.values(),
            key=lambda e: (
                int(e.get("hop", 1) or 1),
                str(e.get("source", "")),
                str(e.get("target", "")),
                str(e.get("rel_type", "")),
                str(e.get("input_group") or e.get("nature") or ""),
            ),
        )
        return [*ordered, *passthrough]

    def _build_cascade_graph(self, normalized_edges: List[Dict[str, Any]]) -> CascadeGraph:
        """Constructs a CascadeGraph from normalized database records (ADR-00026)."""
        from contagion.parameters import resolve_edge_cascade_parameters

        graph = CascadeGraph()
        for r in normalized_edges:
            source = str(r.get("source") or r.get("source_ticker") or "UNKNOWN")
            target = str(r.get("target") or r.get("target_ticker") or "UNKNOWN")
            resolved = resolve_edge_cascade_parameters(r)
            materiality = str(r.get("materiality", r.get("materiality_tier", "NON_CRITICAL")))

            graph.add_edge(
                CascadeEdge(
                    source=source,
                    target=target,
                    input_group=str(resolved["input_group"]),
                    share=float(resolved["share"]),
                    buffer_days=float(resolved["buffer_days"]),
                    materiality_tier=materiality,
                    metadata={
                        "param_provenance": resolved["param_provenance"],
                        "share_is_default": resolved["share_is_default"],
                        "buffer_days_is_default": resolved["buffer_days_is_default"],
                        "dependency_domain": resolved["dependency_domain"],
                    },
                )
            )
        return graph

    def run(self, analyst_query: AnalystQuery) -> PipelineResult:
        """Run the consolidated Domain 1 GraphRAG operational pipeline."""
        start_time = time.perf_counter()
        latencies: Dict[str, float] = {}
        degraded_reasons: List[str] = []

        def _build_degraded(reason: str, *, cypher_text: Optional[str] = None) -> PipelineResult:
            reasons = [*degraded_reasons, reason]
            total_ms = (time.perf_counter() - start_time) * 1000.0
            return PipelineResult(
                query=analyst_query.query,
                as_of_date=analyst_query.as_of_date,
                cypher_query=cypher_text,
                is_valid_query=False,
                degraded=True,
                degraded_reasons=reasons,
                refusal_reason=reason,
                retrieved_subgraph=[],
                seed_hits=seed_hits_payload,
                articles=articles,
                verbalized_subgraph=f"DEGRADED: Backing store unavailable ({reason}). Exposure cannot be evaluated.",
                contagion_verdict=f"DEGRADED: Backing store unavailable ({reason}). Exposure cannot be determined.",
                contagion_think_trace=None,
                latency_breakdown_ms=latencies,
                total_latency_ms=total_ms,
            )

        # -------------------------------------------------------------
        # STEP 0 (Optional): Hybrid Point-in-Time Seeding (ADR-00034)
        # -------------------------------------------------------------
        seed_hits_payload: List[Dict[str, Any]] = []
        seed_ids: List[str] = []
        articles: List[Dict[str, Any]] = []
        if self.seeder is not None:
            t_seed = time.perf_counter()
            try:
                seed_bundle = self.seeder.seed_with_context(
                    analyst_query.query,
                    ctx=analyst_query.ctx,
                    store=self.store,
                    include_cql=False,
                )
            except StoreUnavailableError as exc:
                latencies["hybrid_seeding_ms"] = (time.perf_counter() - t_seed) * 1000.0
                return _build_degraded(f"seeder_store_unavailable: {exc}")
            hits = seed_bundle.get("hits", [])
            seed_hits_payload = [h.to_dict() if hasattr(h, "to_dict") else dict(h) for h in hits]
            seed_ids = [str(h["node_id"]) for h in seed_hits_payload if h.get("node_id")]
            articles = list(seed_bundle.get("articles", []))
            latencies["hybrid_seeding_ms"] = (time.perf_counter() - t_seed) * 1000.0

        # -------------------------------------------------------------
        # STEP 1: Text-to-Cypher Generation (Task C)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        cypher_prompt = (
            f"Generate an openCypher query to retrieve supply-chain relationships for:\n"
            f"Query: {analyst_query.query}\n"
            f"As-of Date: {analyst_query.as_of_date}"
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
        skip_cypher_execution = False
        sanitized_cypher = ""
        temporalized_cypher: Optional[str] = None

        if cypher_resp.error and seed_ids:
            # Model server unavailable, but hybrid dense seeding succeeded: fall through to guarded traversal.
            logger.warning(
                "Task C model unavailable (%s); continuing with %d hybrid seed(s).",
                cypher_resp.error,
                len(seed_ids),
            )
            skip_cypher_execution = True
            latencies["guardrail_validation_ms"] = (time.perf_counter() - t0) * 1000.0
        else:
            validation: ValidationResult = self.schema_guardrail.validate(generated_cypher)
            latencies["guardrail_validation_ms"] = (time.perf_counter() - t0) * 1000.0

            if not validation.is_valid:
                logger.info(
                    "Cypher AST validation tripped: %s. Short-circuiting with '(none)'.",
                    validation.error_summary,
                )
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
                    seed_hits=seed_hits_payload,
                    articles=articles,
                    verbalized_subgraph="No subgraph retrieved. Query refused due to schema violation.",
                    contagion_verdict="(none)",
                    contagion_think_trace=None,
                    latency_breakdown_ms=latencies,
                    total_latency_ms=total_ms,
                )

            sanitized_cypher = validation.sanitized_query or generated_cypher

        # -------------------------------------------------------------
        # STEP 3: Temporal Interval Injection (Stage 6.3 / ADR-00033)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        if not skip_cypher_execution:
            temporalized_cypher = self.temporal_filter.inject_active_coalescence(
                cypher=sanitized_cypher,
                ctx=analyst_query.ctx,
            )
        latencies["temporal_transform_ms"] = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # STEP 4: Unified SubgraphStorePort Execution & Seed Traversal
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        raw_records: List[Dict[str, Any]] = []
        if not skip_cypher_execution:
            try:
                raw_records = self.execute_graph_query(
                    sanitized_cypher,
                    ctx=analyst_query.ctx,
                    params={"as_of_date": analyst_query.ctx.as_of_iso},
                )
            except StoreUnavailableError as exc:
                latencies["graph_execution_ms"] = (time.perf_counter() - t0) * 1000.0
                return _build_degraded(
                    f"graph_store_unavailable: {exc}",
                    cypher_text=temporalized_cypher,
                )
            except Exception as exc:
                degraded_reasons.append(f"graph_execution_failed: {exc}")
                logger.warning("Graph query execution degraded: %s", exc)
        unanchored_excluded = int(
            getattr(self.store, "last_unanchored_excluded", 0)
            or getattr(self.graph_driver, "last_unanchored_excluded", 0)
            or 0
        )
        cypher_edges = self._normalize_graph_records_to_edges(raw_records)

        seeded_edges: List[Dict[str, Any]] = []
        if seed_ids and self.store is not None:
            try:
                seeded_edges = self.store.traverse(
                    seed_ids,
                    hops=self.seed_hops,
                    ctx=analyst_query.ctx,
                )
                unanchored_excluded = max(
                    unanchored_excluded,
                    int(getattr(self.store, "last_unanchored_excluded", 0) or 0),
                )
            except StoreUnavailableError as exc:
                latencies["graph_execution_ms"] = (time.perf_counter() - t0) * 1000.0
                return _build_degraded(
                    f"graph_store_unavailable: {exc}",
                    cypher_text=temporalized_cypher,
                )
            except Exception as exc:
                degraded_reasons.append(f"seed_traversal_failed: {exc}")
                logger.warning("SubgraphStorePort seed traversal degraded: %s", exc)

        normalized_edges = self._merge_edges(cypher_edges, seeded_edges)
        latencies["graph_execution_ms"] = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # STEP 5: Hop-Ordered Subgraph Verbalization & Cascade Diffusion
        # -------------------------------------------------------------
        t0 = time.perf_counter()

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
                    non_seeds = {
                        nid: s for nid, s in cascade_summaries.items() if nid not in initial_shocks
                    }
                    if non_seeds:
                        target_summary = max(non_seeds.values(), key=lambda s: s.peak_shock_pct)
                    else:
                        target_summary = max(cascade_summaries.values(), key=lambda s: s.peak_shock_pct)
            except Exception as exc:
                degraded_reasons.append(f"cascade_propagation_failed: {exc}")
                logger.warning("Cascade engine propagation degraded: %s", exc)
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

        total_ms = (time.perf_counter() - start_time) * 1000.0

        return PipelineResult(
            query=analyst_query.query,
            as_of_date=analyst_query.as_of_date,
            cypher_query=temporalized_cypher,
            is_valid_query=not bool(degraded_reasons),
            degraded=bool(degraded_reasons),
            degraded_reasons=degraded_reasons,
            refusal_reason="; ".join(degraded_reasons) if degraded_reasons else None,
            retrieved_subgraph=normalized_edges,
            seed_hits=seed_hits_payload,
            articles=articles,
            unanchored_excluded=unanchored_excluded,
            verbalized_subgraph=verbalized_context,
            contagion_verdict=contagion_resp.final_output,
            contagion_think_trace=contagion_resp.think_trace,
            latency_breakdown_ms=latencies,
            total_latency_ms=total_ms,
            convergent_shock_summary=target_summary,
            cascade_summaries=cascade_summaries,
        )
