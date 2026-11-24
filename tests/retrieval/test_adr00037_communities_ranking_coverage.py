"""ADR-00037 Point-in-Time Community Layer, Impact-Ranked Context & Report Coverage Invariant Tests.

Covers all 4 Invariants in ADR-00037 Section 5 plus rules-first routing and Workbench badges:
1. Invariant #1 (Point-in-time communities): no `community_reports` row for `as_of = T`
   references an edge with `valid_from > T`, `valid_to <= T`, or transient `:News`/`MENTIONS`
   edges; seeded hierarchical partitioning is deterministic and produces internally connected
   clusters; `communities.parquet` export & PiT load never leak future checkpoints.
2. Deterministic rules-first routing in `query_bi` (zero LLM call): entity-only -> `local`,
   macro/cluster-only -> `global` (map-reduce over cached PiT communities), entity + cluster ->
   `hybrid`, explicit override always honored.
3. Impact-ranked context clamping (`HopOrderedSubgraphVerbalizer`): ranks by
   `(cascade_impact desc, hop asc, provenance_rank desc)` within the 25-edge ceiling so
   high-impact 2-hop/3-hop edges are kept over low-impact 1-hop edges.
4. Invariants #2, #3 & #4 (Report coverage always present, no silent truncation, 3-segment
   provenance grading, 1NF propagation, Domain 1 isolation, Workbench UI badges).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import networkx as nx
import pytest

from contagion import ConvergentShockSummary
from contagion.parameters import ParameterProvenance
from knowledge_graph.communities import (
    Community,
    build_communities,
    export_communities_parquet,
    load_community_reports,
    summarize_communities,
)
from retrieval.facade import query_bi
from retrieval.offline_store import InMemorySubgraphStore
from retrieval.pipeline import HopOrderedSubgraphVerbalizer, MAX_SUBGRAPH_EDGES, PipelineResult
from retrieval.reporting import (
    ContagionTimeSeriesEventRecord,
    ModelExecutionMode,
    OperationalContagionReport,
    ReportCoverage,
    build_operational_contagion_report,
    classify_provenance_tier,
    flatten_report_to_timeseries_records,
    format_operational_contagion_report,
)
from retrieval.routing import classify_query_route
from retrieval.temporal import PiTContext
from sft_pipeline.serving.model_client import ModelResponse, YarnBallModelClient

REPO_ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------
# Invariant #1: Point-in-Time Structural Communities & Parquet Round-Trip
# ----------------------------------------------------------------------
def test_adr00037_invariant1_pit_communities_exclude_future_terminated_and_news_edges(
    tmp_path: Path,
) -> None:
    """No community at `as_of=T` references edges with `valid_from > T`, `valid_to <= T`, or `:News`."""
    edges = [
        # Cluster A (active as of 2022-09-30): Semiconductor foundry chain
        {
            "source_id": "ASML",
            "target_id": "TSM",
            "rel_type": "SUPPLIES_TO",
            "share": 0.90,
            "valid_from": "2020-01-02",
            "evidence_chunk_ids": ["acc1#item_1:c001"],
        },
        {
            "source_id": "TSM",
            "target_id": "AAPL",
            "rel_type": "SUPPLIES_TO",
            "share": 0.80,
            "valid_from": "2020-01-02",
            "evidence_chunk_ids": ["acc2#item_1:c001"],
        },
        {
            "source_id": "TSM",
            "target_id": "NVDA",
            "rel_type": "SUPPLIES_TO",
            "share": 0.75,
            "valid_from": "2021-03-01",
            "evidence_chunk_ids": ["acc3#item_1:c001"],
        },
        # Cluster B (active as of 2022-09-30): Cloud & enterprise pair
        {
            "source_id": "MSFT",
            "target_id": "ORCL",
            "rel_type": "PARTNERED_WITH",
            "share": 0.50,
            "valid_from": "2021-01-04",
            "evidence_chunk_ids": ["acc4#item_1:c001"],
        },
        # FUTURE edge (valid_from > 2022-09-30) -> MUST BE EXCLUDED!
        {
            "source_id": "FUTURE_VENDOR",
            "target_id": "AAPL",
            "rel_type": "SUPPLIES_TO",
            "share": 0.99,
            "valid_from": "2023-02-15",
            "evidence_chunk_ids": ["acc_future#item_1:c001"],
        },
        # TERMINATED edge (valid_to <= 2022-09-30) -> MUST BE EXCLUDED!
        {
            "source_id": "OLD_SUPPLIER",
            "target_id": "AAPL",
            "rel_type": "SUPPLIES_TO",
            "share": 0.70,
            "valid_from": "2019-01-02",
            "valid_to": "2022-06-01",
            "evidence_chunk_ids": ["acc_old#item_1:c001"],
        },
        # Transient :News MENTIONS edge -> MUST BE EXCLUDED!
        {
            "source_id": "NEWS_HASH_1",
            "target_id": "AAPL",
            "rel_type": "MENTIONS",
            "share": 1.0,
            "valid_from": "2022-08-01",
        },
    ]

    as_of = date(2022, 9, 30)
    run1 = build_communities(as_of, edges=edges, universe_id="sp500_pit", max_cluster_size=4, random_seed=42)
    run2 = build_communities(as_of, edges=edges, universe_id="sp500_pit", max_cluster_size=4, random_seed=42)

    assert len(run1) >= 2
    # Deterministic reproducibility under fixed random_seed
    assert [(c.level, c.community_id, c.member_ids) for c in run1] == [
        (c.level, c.community_id, c.member_ids) for c in run2
    ]

    # Invariant #1: Zero future, terminated, or transient news edges/entities
    all_members = {m for c in run1 for m in c.member_ids}
    assert "FUTURE_VENDOR" not in all_members
    assert "OLD_SUPPLIER" not in all_members
    assert "NEWS_HASH_1" not in all_members
    assert {"ASML", "TSM", "AAPL", "NVDA", "MSFT", "ORCL"}.issubset(all_members)

    # Verify every key_edge has valid_from <= as_of and every emitted cluster is internally connected
    full_g = nx.Graph()
    for c in run1:
        for ke in c.key_edges:
            assert ke["valid_from"] <= "2022-09-30"
            assert ke["rel_type"] != "MENTIONS"
            full_g.add_edge(ke["source_id"], ke["target_id"])
    for c in run1:
        assert nx.is_connected(full_g.subgraph(c.member_ids)), f"Disconnected cluster: {c.community_id}"

    # Summarize via YarnBallModelClient (task=COMMUNITY_REPORT) and verify Parquet PiT checkpoint isolation
    seen_tasks: List[str] = []

    class _SummaryClient(YarnBallModelClient):
        def generate(
            self,
            prompt: str,
            task: str = "COMMUNITY_REPORT",
            system_prompt: Optional[str] = None,
            temperature: float = 0.0,
            max_new_tokens: Optional[int] = None,
            enforce_think_prefix: bool = True,
            num_ctx: int = 8192,
        ) -> ModelResponse:
            seen_tasks.append(task)
            out = "Taiwanese foundry cluster centered on TSM supplying AAPL and NVDA wafers."
            return ModelResponse(
                raw_text=out,
                think_trace=None,
                final_output=out,
                task=task,
                latency_sec=0.01,
                prefill_ms=1.0,
                tokens_per_sec=100.0,
                prompt_tokens=20,
                generated_tokens=15,
            )

    summarized_2022 = summarize_communities(run1, model_client=_SummaryClient(), use_llm=True)
    assert seen_tasks and all(t == "COMMUNITY_REPORT" for t in seen_tasks)

    # Write BOTH a 2022-09-30 checkpoint and a 2023-03-31 checkpoint into communities.parquet
    future_comm = Community(
        universe_id="sp500_pit",
        as_of="2023-03-31",
        level=0,
        community_id="L0_C999",
        member_ids=["FUTURE_VENDOR", "AAPL"],
        summary="Future 2023 cluster that must not leak into 2022-Q3 queries.",
        key_edges=[{"source_id": "FUTURE_VENDOR", "target_id": "AAPL", "rel_type": "SUPPLIES_TO", "valid_from": "2023-02-15"}],
    )
    parquet_path = export_communities_parquet([*summarized_2022, future_comm], tmp_path / "communities.parquet")

    loaded_2022 = load_community_reports("2022-09-30", universe_id="sp500_pit", communities_parquet_path=parquet_path)
    assert loaded_2022
    assert all(c.as_of == "2022-09-30" for c in loaded_2022)
    assert all(c.community_id != "L0_C999" for c in loaded_2022)


# ----------------------------------------------------------------------
# Deterministic Rules-First Routing (`local`, `global`, `hybrid`, override)
# ----------------------------------------------------------------------
def test_adr00037_deterministic_rules_first_routing_and_query_bi_global_hybrid(
    tmp_path: Path,
) -> None:
    """Rules-first router classifies local/global/hybrid with zero LLM call and honors explicit overrides."""
    mock_linker = MagicMock()
    mock_linker.link_entity.side_effect = lambda name=None, ticker=None: (
        {"ticker": "AAPL", "company_name": "Apple Inc.", "cik": "0000320193"}
        if (ticker == "AAPL" or (name and "apple" in name.lower()))
        else (
            {"ticker": "TSM", "company_name": "Taiwan Semiconductor Manufacturing", "cik": "0001046179"}
            if (ticker == "TSM" or (name and "tsmc" in name.lower()))
            else None
        )
    )

    # 1. Entity only -> local
    r_local, meta_local = classify_query_route("What is Apple's exposure to TSMC?", linker=mock_linker)
    assert r_local == "local"
    assert meta_local["override"] is False

    # 2. Thematic / sector / cluster phrasing with no anchor entity -> global
    r_global, meta_global = classify_query_route(
        "Which S&P 500 clusters depend on Taiwanese foundry capacity as of 2022-Q3?",
        linker=mock_linker,
    )
    assert r_global == "global"
    assert meta_global["resolved_entities"] == []
    assert len(meta_global["macro_cues"]) >= 1

    # 3. Both anchor entity and macro/cluster cues -> hybrid
    r_hybrid, meta_hybrid = classify_query_route(
        "How does Apple compare across S&P 500 semiconductor supply-chain clusters?",
        linker=mock_linker,
    )
    assert r_hybrid == "hybrid"
    assert len(meta_hybrid["resolved_entities"]) >= 1
    assert len(meta_hybrid["macro_cues"]) >= 1

    # 4. Explicit override always wins
    r_ov, meta_ov = classify_query_route("What is Apple's exposure to TSMC?", route="global", linker=mock_linker)
    assert r_ov == "global"
    assert meta_ov["override"] is True

    # 5. End-to-end `query_bi` global route over PiT `communities.parquet` (zero future leakage)
    comm_2022 = Community(
        universe_id="sp500_pit",
        as_of="2022-09-30",
        level=0,
        community_id="L0_C001",
        member_ids=["TSM", "AAPL", "NVDA"],
        summary="Taiwanese foundry cluster supplying leading-edge wafers to Apple and NVIDIA.",
        key_edges=[{"source_id": "TSM", "target_id": "AAPL", "rel_type": "SUPPLIES_TO", "share": 0.80, "valid_from": "2020-01-02"}],
    )
    comm_2024 = Community(
        universe_id="sp500_pit",
        as_of="2024-03-31",
        level=0,
        community_id="L0_C002",
        member_ids=["FUTURE_CORP"],
        summary="Future 2024 cluster.",
    )
    comm_parquet = export_communities_parquet([comm_2022, comm_2024], tmp_path / "communities.parquet")

    ctx_2022 = PiTContext(as_of=date(2022, 9, 30), mode="historical")
    global_res = query_bi(
        "Which S&P 500 clusters depend on Taiwanese foundry capacity?",
        ctx=ctx_2022,
        communities_parquet_path=comm_parquet,
        linker=mock_linker,
    )
    assert global_res.route_used == "global"
    assert global_res.degraded is False
    assert len(global_res.community_reports) == 1
    assert global_res.community_reports[0]["community_id"] == "L0_C001"
    assert "Taiwanese foundry cluster" in global_res.contagion_verdict

    # 6. End-to-end `query_bi` hybrid route combines local cascade + PiT community reports
    store = InMemorySubgraphStore(
        [
            {
                "source": "TSM",
                "target": "AAPL",
                "rel_type": "SUPPLIES_TO",
                "nature": "wafers",
                "share": 0.80,
                "buffer_days": 15.0,
                "valid_from": "2020-01-02",
            }
        ]
    )

    class _HybridClient(YarnBallModelClient):
        def generate(
            self,
            prompt: str,
            task: str = "CONTAGION_REASONING",
            system_prompt: Optional[str] = None,
            temperature: float = 0.0,
            max_new_tokens: Optional[int] = None,
            enforce_think_prefix: bool = True,
            num_ctx: int = 8192,
        ) -> ModelResponse:
            out = (
                "MATCH (s:Company {ticker: 'TSM'})-[r:SUPPLIES_TO]->(t:Company) RETURN s, r, t LIMIT 25"
                if task == "TEXT_TO_CYPHER"
                else "Apple carries elevated wafer concentration relative to peer clusters."
            )
            return ModelResponse(
                raw_text=out,
                think_trace=None,
                final_output=out,
                task=task,
                latency_sec=0.01,
                prefill_ms=1.0,
                tokens_per_sec=100.0,
                prompt_tokens=10,
                generated_tokens=10,
            )

    hybrid_res = query_bi(
        "How does Apple compare across S&P 500 semiconductor clusters?",
        ctx=ctx_2022,
        store=store,
        model_client=_HybridClient(),
        communities_parquet_path=comm_parquet,
        linker=mock_linker,
    )
    assert hybrid_res.route_used == "hybrid"
    assert len(hybrid_res.retrieved_subgraph) == 1
    assert len(hybrid_res.community_reports) == 1
    assert "Point-in-Time Structural Communities" in hybrid_res.verbalized_subgraph


# ----------------------------------------------------------------------
# Impact-Ranked Context Clamping (MAX_SUBGRAPH_EDGES = 25)
# ----------------------------------------------------------------------
def test_adr00037_impact_ranked_verbalizer_keeps_high_impact_deep_edges_within_25_cap() -> None:
    """Impact-ranked clamping keeps high-impact 2-hop/3-hop edges over low-impact 1-hop edges."""
    edges: List[Dict[str, Any]] = []
    # 26 low-impact / zero-impact 1-hop edges (exceeding MAX_SUBGRAPH_EDGES=25)
    for i in range(26):
        edges.append(
            {
                "source": "TSM",
                "target": f"MINOR_{i:02d}",
                "rel_type": "SUPPLIES_TO",
                "hop": 1,
                "share": 0.05,
                "param_provenance": ParameterProvenance.DEFAULT_CONSERVATIVE_BOUND.value,
                "evidence_chunk_ids": [f"acc#c{i:03d}"],
            }
        )
    # 1 high-impact 3-hop critical edge that MUST survive the 25-edge clamp!
    edges.append(
        {
            "source": "PACKAGER_TW",
            "target": "AAPL",
            "rel_type": "SUPPLIES_TO",
            "hop": 3,
            "input_group": "cowos_packaging",
            "share": 0.90,
            "param_provenance": ParameterProvenance.EXTRACTED_QUANTITATIVE.value,
            "evidence_chunk_ids": ["0000320193-24-000106#item_1:c001"],
        }
    )

    aapl_summary = ConvergentShockSummary(
        target_entity="AAPL",
        impairment_score=0.72,
        peak_shock_pct=72.0,
        onset_day=15.0,
        downtime_days=45.0,
        lost_capacity_days=32.4,
        spectral_radius=0.0,
        verdict="CRITICAL_UNBUFFERED_DISRUPTION",
        is_friction_floor=False,
        is_out_of_horizon=False,
        contributing_edges=[
            {
                "source": "PACKAGER_TW",
                "input_group": "cowos_packaging",
                "share_pct": 90.0,
                "source_shock_pct": 80.0,
                "buffer_days": 15.0,
            }
        ],
    )

    verbalizer = HopOrderedSubgraphVerbalizer(
        max_edges=MAX_SUBGRAPH_EDGES,
        chunk_fetcher=lambda cids: {cid: {"chunk_id": cid, "text": f"Passage for {cid}"} for cid in cids},
    )
    rendered = verbalizer.verbalize(edges, cascade_summary=aapl_summary, all_summaries={"AAPL": aapl_summary})

    assert verbalizer.last_edges_total == 27
    assert verbalizer.last_edges_shown == 25
    assert verbalizer.last_truncated is True
    assert verbalizer.last_dropped_impact_share == 0.0  # All dropped edges had 0 cascade impact
    assert "(PACKAGER_TW) -[:SUPPLIES_TO" in rendered
    assert "cascade_impact: 32.40" in rendered


# ----------------------------------------------------------------------
# Invariants #2, #3 & #4: ReportCoverage, 3-Segment Provenance, 1NF & Workbench UI
# ----------------------------------------------------------------------
def test_adr00037_invariants_2_3_4_coverage_provenance_grading_1nf_and_domain_isolation() -> None:
    """Verify 3-segment provenance grading, no silent truncation, degraded guards, 1NF join, and Domain 1 isolation."""
    # 1. Exact 3-segment piecewise provenance thresholds (ASC 280 10% & 50% majority)
    assert classify_provenance_tier(0.0) == "FILING_GROUNDED"
    assert classify_provenance_tier(0.10) == "FILING_GROUNDED"
    assert classify_provenance_tier(0.101) == "PARTIALLY_ASSUMED"
    assert classify_provenance_tier(0.499) == "PARTIALLY_ASSUMED"
    assert classify_provenance_tier(0.50) == "STRESS_UPPER_BOUND"
    assert classify_provenance_tier(1.0) == "STRESS_UPPER_BOUND"

    # 2. Invariant #3: ReportCoverage never allows silent truncation when edges_total > edges_shown
    cov_trunc = ReportCoverage(edges_total=40, edges_shown=25, truncated=False)
    assert cov_trunc.truncated is True

    # 3. Invariant #2: data_status="ok" is impossible while report is degraded
    with pytest.raises(ValueError, match="Invariant #2"):
        OperationalContagionReport(
            report_id="REP-1",
            event_id="EVT-1",
            sec_item="Item 8.01",
            disrupted_entity_ticker="TSM",
            filing_date="2024-11-01",
            disruption_category="EQUIPMENT_BREAKDOWN_FAB_HALT",
            event_summary="Fab halt",
            shock_duration_days=30.0,
            shock_magnitude_pct=100.0,
            operational_synthesis="Degraded.",
            model_mode=ModelExecutionMode.DEGRADED_UNREACHABLE,
            coverage=ReportCoverage(data_status="ok"),
        )

    # 4. Build report with majority DEFAULT_CONSERVATIVE_BOUND impact -> STRESS_UPPER_BOUND
    aapl_summary = ConvergentShockSummary(
        target_entity="AAPL",
        impairment_score=1.0,
        peak_shock_pct=100.0,
        onset_day=0.0,
        downtime_days=30.0,
        lost_capacity_days=30.0,
        spectral_radius=0.0,
        verdict="CRITICAL",
        is_friction_floor=False,
        is_out_of_horizon=False,
    )
    pipe_res = PipelineResult(
        query="TSMC halt",
        as_of_date="2024-11-01",
        is_valid_query=True,
        retrieved_subgraph=[
            {
                "source": "TSM",
                "target": "AAPL",
                "rel_type": "SUPPLIES_TO",
                "hop": 1,
                "evidence_status": "legacy_unchunked",
            }
        ],
        unanchored_excluded=2,
        edges_total=30,
        edges_shown=25,
        truncated=True,
        dropped_impact_share=0.12,
        legacy_unchunked_evidence=1,
        verbalized_subgraph="Subgraph",
        contagion_verdict="AAPL faces 100.0% capacity contraction for 30.0 days.",
        total_latency_ms=10.0,
        cascade_summaries={"AAPL": aapl_summary},
    )
    report = build_operational_contagion_report(
        result=pipe_res,
        event_id="EVT-STRESS",
        sec_item="Item 8.01",
        disrupted_entity_ticker="TSM",
        filing_date="2024-11-01",
        disruption_category="EQUIPMENT_BREAKDOWN_FAB_HALT",
        event_summary="Fab halt",
        shock_duration_days=30.0,
        shock_magnitude_pct=100.0,
    )
    assert report.coverage.data_status == "ok"
    assert report.coverage.provenance_tier == "STRESS_UPPER_BOUND"
    assert report.coverage.default_param_impact_share == 1.0
    assert report.coverage.truncated is True
    assert report.coverage.edges_total == 30
    assert report.coverage.edges_shown == 25
    assert report.coverage.dropped_impact_share == 0.12
    assert report.coverage.unanchored_excluded == 2
    assert report.coverage.legacy_unchunked_evidence == 1

    # 5. Build report with zero retrieved edges -> data_status == "no_coverage"
    empty_pipe = PipelineResult(
        query="Unknown",
        as_of_date="2024-11-01",
        is_valid_query=True,
        retrieved_subgraph=[],
        verbalized_subgraph="None",
        contagion_verdict="No exposure found.",
        total_latency_ms=5.0,
    )
    no_cov_report = build_operational_contagion_report(
        result=empty_pipe,
        event_id="EVT-EMPTY",
        sec_item="Item 8.01",
        disrupted_entity_ticker="XYZ",
        filing_date="2024-11-01",
        disruption_category="OTHER_UNCLASSIFIED_MATERIAL_HALT",
        event_summary="No edges",
        shock_duration_days=10.0,
        shock_magnitude_pct=50.0,
    )
    assert no_cov_report.coverage.data_status == "no_coverage"

    # 6. 1NF time-series join propagation (ADR-00013 / ADR-00037)
    ts_rows = flatten_report_to_timeseries_records(report)
    assert len(ts_rows) == 1
    assert ts_rows[0].data_status == "ok"
    assert ts_rows[0].provenance_tier == "STRESS_UPPER_BOUND"
    assert ts_rows[0].default_param_impact_share == 1.0
    assert ts_rows[0].truncated is True
    assert ts_rows[0].edges_total == 30
    assert ts_rows[0].edges_shown == 25

    rendered_txt = format_operational_contagion_report(report)
    assert "STRESS_UPPER_BOUND" in rendered_txt
    assert "TRUNCATED" in rendered_txt

    # 7. Invariant #4: Domain isolation preserved across ReportCoverage & ContagionTimeSeriesEventRecord
    forbidden = {
        "trade", "trading", "order", "basket", "short", "long",
        "option", "options", "strike", "put", "call", "collar",
        "portfolio", "recommendation", "price_target", "hedge_ratio",
    }
    for field_name in list(ReportCoverage.model_fields.keys()) + list(ContagionTimeSeriesEventRecord.model_fields.keys()):
        assert not set(field_name.lower().split("_")).intersection(forbidden), f"Forbidden term in {field_name}"

    # 8. Workbench frontend renders provenance & status badges (ADR-00037 Section 4.5)
    app_js = (REPO_ROOT / "explorer" / "static" / "app.js").read_text(encoding="utf-8")
    for token in ("provenance_tier", "FILING_GROUNDED", "PARTIALLY_ASSUMED", "Truncated", "Degraded", "No coverage"):
        assert token in app_js, f"Missing Workbench badge rendering token '{token}' in explorer/static/app.js"
