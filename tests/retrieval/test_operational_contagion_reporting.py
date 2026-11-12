"""Unit and Contract Tests for ADR-00011 Operational Contagion Reporting Schema."""

import json
import pytest
from pydantic import ValidationError

from contagion.cascade import ConvergentShockSummary
from retrieval.pipeline import PipelineResult
from retrieval.reporting import (
    SECForm8KItem,
    SEC_ITEM_DESCRIPTIONS,
    OperationalDisruptionCategory,
    DependencyDomain,
    OperationalScope,
    ContagionSeverityVerdict,
    NetworkNodeImpact,
    TopologicalEdgeDescriptor,
    ExecutionTelemetry,
    OperationalContagionReport,
    classify_contagion_verdict,
    format_operational_contagion_report,
    build_operational_contagion_report,
    validate_verbatim_citations,
)


class TestControlledTaxonomies:
    """Verifies ADR-00011 Section 4.1 controlled categorical taxonomies."""

    def test_sec_form_8k_item_canon_coverage(self):
        """Verify all 9 statutory sections of Form 8-K are covered exhaustively."""
        items = list(SECForm8KItem)
        assert len(items) >= 28

        # Section 1
        assert SECForm8KItem.ITEM_1_01.value == "Item 1.01"
        assert SECForm8KItem.ITEM_1_05.value == "Item 1.05"
        # Section 2
        assert SECForm8KItem.ITEM_2_01.value == "Item 2.01"
        assert SECForm8KItem.ITEM_2_06.value == "Item 2.06"
        # Section 3
        assert SECForm8KItem.ITEM_3_01.value == "Item 3.01"
        assert SECForm8KItem.ITEM_3_03.value == "Item 3.03"
        # Section 4
        assert SECForm8KItem.ITEM_4_01.value == "Item 4.01"
        assert SECForm8KItem.ITEM_4_02.value == "Item 4.02"
        # Section 5
        assert SECForm8KItem.ITEM_5_01.value == "Item 5.01"
        assert SECForm8KItem.ITEM_5_08.value == "Item 5.08"
        # Section 6
        assert SECForm8KItem.ITEM_6_01.value == "Item 6.01"
        assert SECForm8KItem.ITEM_6_05.value == "Item 6.05"
        # Section 7
        assert SECForm8KItem.ITEM_7_01.value == "Item 7.01"
        # Section 8
        assert SECForm8KItem.ITEM_8_01.value == "Item 8.01"
        # Section 9
        assert SECForm8KItem.ITEM_9_01.value == "Item 9.01"

        # Every item must have a statutory title
        for item in SECForm8KItem:
            assert item.value in SEC_ITEM_DESCRIPTIONS
            assert len(SEC_ITEM_DESCRIPTIONS[item.value]) > 0

    def test_operational_disruption_categories(self):
        """Verify operational disruption categories across all mechanisms."""
        cats = [c.value for c in OperationalDisruptionCategory]
        # Physical
        assert "FACILITY_FIRE_OR_EXPLOSION" in cats
        assert "NATURAL_DISASTER_FORCE_MAJEURE" in cats
        assert "EQUIPMENT_BREAKDOWN_FAB_HALT" in cats
        # Commercial
        assert "PRIMARY_SUPPLY_CONTRACT_TERMINATED" in cats
        assert "COMMERCIAL_INSOLVENCY_BANKRUPTCY" in cats
        # Human capital
        assert "LABOR_UNION_STRIKE_WALKOUT" in cats
        assert "RANSOMWARE_CYBER_INFRASTRUCTURE_HALT" in cats
        # Logistics & regulatory
        assert "MARITIME_CANAL_PORT_LOGISTICS_CHOKEPOINT" in cats
        assert "REGULATORY_SAFETY_RECALL_OR_FDA_HOLD" in cats
        assert "OTHER_UNCLASSIFIED_MATERIAL_HALT" in cats

    def test_dependency_domain_and_scope(self):
        """Verify non-widget domain and scope categories."""
        assert DependencyDomain.PHYSICAL_COMPONENT.value == "PHYSICAL_COMPONENT"
        assert DependencyDomain.DIGITAL_INFRASTRUCTURE.value == "DIGITAL_INFRASTRUCTURE"
        assert DependencyDomain.COMMERCIAL_ENTERPRISE.value == "COMMERCIAL_ENTERPRISE"

        assert OperationalScope.CORE_OPERATIONAL_DELIVERY.value == "CORE_OPERATIONAL_DELIVERY"
        assert OperationalScope.GROWTH_CAPEX_AND_RD.value == "GROWTH_CAPEX_AND_RD"


class TestContagionSeverityVerdicts:
    """Verifies ADR-00011 Section 4.3 zero-magic-number physical boundary conditions."""

    def test_unbuffered_disruption_boundary(self):
        """Any unbuffered downtime > 0 is UNBUFFERED_DISRUPTION regardless of percentage."""
        v1 = classify_contagion_verdict(unbuffered_downtime_days=15.0, peak_shock_pct=20.0)
        assert v1 == ContagionSeverityVerdict.UNBUFFERED_DISRUPTION

        # Micro-downtime (e.g. 0.1 day) with small shock (e.g. 5%) is still UNBUFFERED_DISRUPTION
        v2 = classify_contagion_verdict(unbuffered_downtime_days=0.1, peak_shock_pct=5.0)
        assert v2 == ContagionSeverityVerdict.UNBUFFERED_DISRUPTION

    def test_buffer_absorbed_boundary(self):
        """Zero unbuffered downtime with upstream shock is BUFFER_ABSORBED, not a magic threshold."""
        # High upstream shock (e.g. 80%) fully absorbed by 90-day buffer -> BUFFER_ABSORBED
        v1 = classify_contagion_verdict(unbuffered_downtime_days=0.0, upstream_shock_pct=80.0, peak_shock_pct=0.0)
        assert v1 == ContagionSeverityVerdict.BUFFER_ABSORBED

        # Non-zero peak shock but zero downtime
        v2 = classify_contagion_verdict(unbuffered_downtime_days=0.0, upstream_shock_pct=0.0, peak_shock_pct=12.5)
        assert v2 == ContagionSeverityVerdict.BUFFER_ABSORBED

    def test_unaffected_boundary(self):
        """Zero shock and zero downtime is UNAFFECTED."""
        v = classify_contagion_verdict(unbuffered_downtime_days=0.0, upstream_shock_pct=0.0, peak_shock_pct=0.0)
        assert v == ContagionSeverityVerdict.UNAFFECTED


class TestPydanticModelContracts:
    """Verifies Pydantic model validation and normalization."""

    def test_network_node_impact_normalization(self):
        impact = NetworkNodeImpact(
            entity_ticker="AAPL",
            hop_distance=1,
            input_group="Audio Codecs & Amplifiers",
            procurement_share_pct=20.0,
            inventory_buffer_days=45.0,
            shock_onset_day=45.0,
            unbuffered_downtime_days=15.0,
            capacity_contraction_pct=20.0,
            lost_capacity_days=3.0,
            verdict=ContagionSeverityVerdict.UNBUFFERED_DISRUPTION,
        )
        assert impact.input_group == "audio_codecs_&_amplifiers"
        assert impact.entity_ticker == "AAPL"
        assert impact.procurement_share_pct == 20.0

    def test_network_node_impact_bounds_rejection(self):
        with pytest.raises(ValidationError):
            NetworkNodeImpact(
                entity_ticker="AAPL",
                hop_distance=0,  # ge=1 required
                verdict=ContagionSeverityVerdict.UNBUFFERED_DISRUPTION,
            )

        with pytest.raises(ValidationError):
            NetworkNodeImpact(
                entity_ticker="AAPL",
                procurement_share_pct=150.0,  # le=100.0 required
                verdict=ContagionSeverityVerdict.UNBUFFERED_DISRUPTION,
            )

    def test_topological_edge_descriptor_normalization(self):
        edge = TopologicalEdgeDescriptor(
            source="CRUS",
            target="AAPL",
            hop_distance=1,
            nature="Power Management IC",
            share=0.25,
            buffer_days=30.0,
        )
        assert edge.nature == "power_management_ic"
        assert edge.share == 0.25

    def test_report_json_roundtrip(self):
        report = OperationalContagionReport(
            report_id="REP-20241028-CRUS-001",
            event_id="SEC_8K_20241028_CRUS_FIRE",
            filing_type="SEC Form 8-K",
            sec_item="Item 8.01",
            disrupted_entity_ticker="CRUS",
            disrupted_entity_name="Cirrus Logic Inc.",
            disrupted_entity_cik="0000772406",
            filing_date="2024-10-28",
            disruption_category="FACILITY_FIRE_OR_EXPLOSION",
            event_summary="Fab fire at primary wafer production facility.",
            shock_duration_days=60.0,
            shock_magnitude_pct=100.0,
            network_impacts=[
                NetworkNodeImpact(
                    entity_ticker="AAPL",
                    hop_distance=1,
                    input_group="audio_codecs",
                    procurement_share_pct=20.0,
                    inventory_buffer_days=45.0,
                    shock_onset_day=45.0,
                    unbuffered_downtime_days=15.0,
                    capacity_contraction_pct=20.0,
                    lost_capacity_days=3.0,
                    verdict=ContagionSeverityVerdict.UNBUFFERED_DISRUPTION,
                )
            ],
            spectral_radius=0.0,
            is_acyclic_dag=True,
            active_topology_edges=[
                TopologicalEdgeDescriptor(
                    source="CRUS",
                    target="AAPL",
                    hop_distance=1,
                    nature="audio_codecs",
                    share=0.20,
                    buffer_days=45.0,
                )
            ],
            contagion_think_trace="Analyzing 60-day fab halt...",
            operational_synthesis="AAPL experiences UNBUFFERED_DISRUPTION of 20.0% contraction and 15.0 days downtime.",
            telemetry=ExecutionTelemetry(task_c_cypher_ms=12.0, total_latency_ms=160.0),
        )

        dumped = report.model_dump_json()
        loaded = OperationalContagionReport.model_validate_json(dumped)

        assert loaded.report_id == report.report_id
        assert len(loaded.network_impacts) == 1
        assert loaded.network_impacts[0].entity_ticker == "AAPL"
        assert loaded.network_impacts[0].verdict == ContagionSeverityVerdict.UNBUFFERED_DISRUPTION


class TestVisualInspectionFormatter:
    """Verifies ADR-00011 Section 4.5B Visual Inspection Report formatting."""

    @pytest.fixture
    def canonical_report(self) -> OperationalContagionReport:
        return OperationalContagionReport(
            report_id="REP-20241028-CRUS-001",
            event_id="SEC_8K_20241028_CRUS_FIRE",
            filing_type="SEC Form 8-K",
            sec_item="Item 8.01",
            disrupted_entity_ticker="CRUS",
            disrupted_entity_name="Cirrus Logic Inc.",
            disrupted_entity_cik="0000772406",
            filing_date="2024-10-28",
            disruption_category="FACILITY_FIRE_OR_EXPLOSION",
            event_summary="Fab fire at primary wafer production facility.",
            shock_duration_days=60.0,
            shock_magnitude_pct=100.0,
            network_impacts=[
                NetworkNodeImpact(
                    entity_ticker="AAPL",
                    hop_distance=1,
                    input_group="audio_codecs",
                    procurement_share_pct=20.0,
                    inventory_buffer_days=45.0,
                    shock_onset_day=45.0,
                    unbuffered_downtime_days=15.0,
                    capacity_contraction_pct=20.0,
                    lost_capacity_days=3.0,
                    verdict=ContagionSeverityVerdict.UNBUFFERED_DISRUPTION,
                ),
                NetworkNodeImpact(
                    entity_ticker="HPQ",
                    hop_distance=1,
                    input_group="audio_codecs",
                    procurement_share_pct=10.0,
                    inventory_buffer_days=90.0,
                    shock_onset_day=None,
                    unbuffered_downtime_days=0.0,
                    capacity_contraction_pct=0.0,
                    lost_capacity_days=0.0,
                    verdict=ContagionSeverityVerdict.BUFFER_ABSORBED,
                ),
                NetworkNodeImpact(
                    entity_ticker="FOXCONN",
                    hop_distance=2,
                    input_group="assembly",
                    procurement_share_pct=50.0,
                    inventory_buffer_days=90.0,
                    shock_onset_day=None,
                    unbuffered_downtime_days=0.0,
                    capacity_contraction_pct=0.0,
                    lost_capacity_days=0.0,
                    verdict=ContagionSeverityVerdict.BUFFER_ABSORBED,
                ),
            ],
            spectral_radius=0.0,
            is_acyclic_dag=True,
            active_topology_edges=[
                TopologicalEdgeDescriptor(
                    source="CRUS",
                    target="AAPL",
                    rel_type="SUPPLIES_TO",
                    hop_distance=1,
                    nature="audio_codecs",
                    share=0.20,
                    buffer_days=45.0,
                ),
                TopologicalEdgeDescriptor(
                    source="CRUS",
                    target="HPQ",
                    rel_type="SUPPLIES_TO",
                    hop_distance=1,
                    nature="audio_codecs",
                    share=0.10,
                    buffer_days=90.0,
                ),
                TopologicalEdgeDescriptor(
                    source="AAPL",
                    target="FOXCONN",
                    rel_type="SUPPLIES_TO",
                    hop_distance=2,
                    nature="assembly",
                    share=0.50,
                    buffer_days=90.0,
                ),
            ],
            contagion_think_trace="Analyzing Cirrus Logic (CRUS) audio codec fab halt of 60 days...",
            operational_synthesis=(
                "Apple Inc. (AAPL) experiences UNBUFFERED_DISRUPTION resulting from a 20.0% "
                "peak capacity contraction across its audio component lines with 15.0 days of "
                "unbuffered downtime post-buffer absorption. Foxconn (FOXCONN) and Hewlett-Packard "
                "(HPQ) are classified as BUFFER_ABSORBED, as their 90-day inventory safety stocks "
                "fully absorb operational transmission with zero downstream downtime."
            ),
            telemetry=ExecutionTelemetry(
                task_c_cypher_ms=12.4,
                ast_guardrail_ms=1.1,
                graph_traversal_ms=4.2,
                cascade_propagation_ms=2.8,
                task_d_contagion_ms=145.0,
                total_latency_ms=165.5,
            ),
        )

    def test_rendered_report_exact_structure(self, canonical_report: OperationalContagionReport):
        rendered = format_operational_contagion_report(canonical_report)

        # Header check
        assert "YARNBALL OPERATIONAL CONTAGION REPORT" in rendered
        assert "REPORT ID:          REP-20241028-CRUS-001" in rendered
        assert "SEC STATUTORY ITEM: Item 8.01 (Other Events)" in rendered
        assert "DISRUPTED ENTITY:   Cirrus Logic Inc. (CRUS | CIK: 0000772406)" in rendered
        assert "SHOCK SPECIFICATION: 100.0% plant halt for 60.0 calendar days" in rendered

        # Section 1
        assert "1. DETERMINISTIC CASCADE SHOCK METRICS (SparseNetworkCascadeEngine)" in rendered
        lines = rendered.splitlines()
        hdr_idx = lines.index("Entity | Hop | Input Group   | Share  | Buffer | Onset  | Downtime | Contraction | Verdict")
        assert lines[hdr_idx + 2] == "AAPL   | 1   | audio_codecs  |  20.0% |  45.0d | Day 45 |    15.0d |      -20.0% | UNBUFFERED_DISRUPTION"
        assert lines[hdr_idx + 3] == "HPQ    | 1   | audio_codecs  |  10.0% |  90.0d | None   |     0.0d |        0.0% | BUFFER_ABSORBED"
        assert lines[hdr_idx + 4] == "FOXCONN| 2   | assembly      |  50.0% |  90.0d | None   |     0.0d |        0.0% | BUFFER_ABSORBED"
        assert "Spectral Radius: 0.0000 (Topology is provably acyclic DAG)" in rendered

        # Section 2
        assert "2. ACTIVE KNOWLEDGE GRAPH TOPOLOGY" in rendered
        assert "[Hop Distance 1]" in rendered
        assert "- (CRUS) -[:SUPPLIES_TO {nature: 'audio_codecs', share: 0.20, buffer_days: 45.0}]-> (AAPL)" in rendered
        assert "[Hop Distance 2]" in rendered
        assert "- (AAPL) -[:SUPPLIES_TO {nature: 'assembly', share: 0.50, buffer_days: 90.0}]-> (FOXCONN)" in rendered

        # Section 3
        assert "3. COMPUTATIONAL READING & REASONING (yarnball-qwen:7b)" in rendered
        assert "<think>" in rendered
        assert "Operational Synthesis:" in rendered

        # Section 4
        assert "4. PIPELINE TELEMETRY" in rendered
        assert "Cypher Retrieval (Task C):     12.4 ms" in rendered
        assert "Cascade Propagation:            2.8 ms" in rendered
        assert "Total Pipeline Latency:       165.5 ms" in rendered

    def test_spectral_radius_feedback_notes(self, canonical_report: OperationalContagionReport):
        # Damped feedback
        canonical_report.spectral_radius = 0.45
        canonical_report.is_acyclic_dag = False
        out = format_operational_contagion_report(canonical_report)
        assert "Spectral Radius: 0.4500 (Dampened circular feedback, rho < 1.0)" in out

        # Runaway / critical feedback
        canonical_report.spectral_radius = 1.35
        out = format_operational_contagion_report(canonical_report)
        assert "Spectral Radius: 1.3500 (CRITICAL: Amplifying cyclic feedback, rho >= 1.0)" in out


class TestPipelineResultIntegration:
    """Verifies build_operational_contagion_report and PipelineResult.to_operational_report()."""

    def test_build_report_from_pipeline_result(self):
        # Construct realistic PipelineResult with cascade summaries
        summary_aapl = ConvergentShockSummary(
            target_entity="AAPL",
            peak_shock_pct=20.0,
            impairment_score=0.20,
            onset_day=45.0,
            downtime_days=15.0,
            lost_capacity_days=3.0,
            verdict="DISRUPTIVE_SHOCK",
            spectral_radius=0.0,
            contributing_edges=[
                {
                    "source": "CRUS",
                    "input_group": "audio_codecs",
                    "share_pct": 20.0,
                    "buffer_days": 45.0,
                    "source_shock_pct": 100.0,
                }
            ],
        )

        pipeline_res = PipelineResult(
            query="Analyze supply chain disruption from CRUS fab halt to AAPL",
            is_valid_query=True,
            retrieved_subgraph=[
                {
                    "source": "CRUS",
                    "target": "AAPL",
                    "nature": "audio_codecs",
                    "share": 0.20,
                    "buffer_days": 45.0,
                    "hop": 1,
                }
            ],
            verbalized_subgraph="CRUS supplies to AAPL",
            contagion_verdict="AAPL experiences 20.0% contraction and 15.0 days downtime.",
            contagion_think_trace="Analyzing fab halt propagation...",
            total_latency_ms=155.0,
            latency_breakdown_ms={
                "task_c_cypher_ms": 10.5,
                "guardrail_validation_ms": 1.2,
                "graph_execution_ms": 3.8,
                "cascade_engine_ms": 2.1,
                "task_d_contagion_ms": 137.4,
            },
            convergent_shock_summary=summary_aapl,
            cascade_summaries={"AAPL": summary_aapl},
        )

        report = pipeline_res.to_operational_report(
            event_id="SEC_8K_20241028_CRUS_FIRE",
            sec_item=SECForm8KItem.ITEM_8_01,
            disrupted_entity_ticker="CRUS",
            disrupted_entity_name="Cirrus Logic Inc.",
            disrupted_entity_cik="0000772406",
            filing_date="2024-10-28",
            disruption_category=OperationalDisruptionCategory.FACILITY_FIRE_OR_EXPLOSION,
            event_summary="Cirrus Logic manufacturing plant halt.",
            shock_duration_days=60.0,
            shock_magnitude_pct=100.0,
        )

        assert isinstance(report, OperationalContagionReport)
        assert report.report_id == "REP-20241028-CRUS-001"
        assert report.disrupted_entity_ticker == "CRUS"
        assert len(report.network_impacts) == 1
        assert report.network_impacts[0].entity_ticker == "AAPL"
        assert report.network_impacts[0].unbuffered_downtime_days == 15.0
        assert report.network_impacts[0].capacity_contraction_pct == 20.0
        assert report.network_impacts[0].verdict == ContagionSeverityVerdict.UNBUFFERED_DISRUPTION
        assert report.is_acyclic_dag is True
        assert report.telemetry.task_c_cypher_ms == 10.5
        assert report.telemetry.cascade_propagation_ms == 2.1


class TestVerbatimCitationValidator:
    """Verifies ADR-00011 Section 4.4 Invariant 2 verbatim metric citation enforcement."""

    def test_valid_verbatim_citation(self):
        report = OperationalContagionReport(
            report_id="REP-001",
            event_id="EVT-001",
            sec_item="Item 8.01",
            disrupted_entity_ticker="CRUS",
            filing_date="2024-10-28",
            disruption_category="FACILITY_FIRE_OR_EXPLOSION",
            event_summary="Summary",
            shock_duration_days=60.0,
            shock_magnitude_pct=100.0,
            network_impacts=[
                NetworkNodeImpact(
                    entity_ticker="AAPL",
                    hop_distance=1,
                    input_group="audio_codecs",
                    unbuffered_downtime_days=15.0,
                    capacity_contraction_pct=20.0,
                    verdict=ContagionSeverityVerdict.UNBUFFERED_DISRUPTION,
                )
            ],
            operational_synthesis="AAPL suffers a 20.0% capacity contraction with 15.0 days of unbuffered downtime.",
        )
        violations = validate_verbatim_citations(report)
        assert len(violations) == 0

    def test_missing_verbatim_citation_triggers_violations(self):
        report = OperationalContagionReport(
            report_id="REP-001",
            event_id="EVT-001",
            sec_item="Item 8.01",
            disrupted_entity_ticker="CRUS",
            filing_date="2024-10-28",
            disruption_category="FACILITY_FIRE_OR_EXPLOSION",
            event_summary="Summary",
            shock_duration_days=60.0,
            shock_magnitude_pct=100.0,
            network_impacts=[
                NetworkNodeImpact(
                    entity_ticker="AAPL",
                    hop_distance=1,
                    input_group="audio_codecs",
                    unbuffered_downtime_days=15.0,
                    capacity_contraction_pct=20.0,
                    verdict=ContagionSeverityVerdict.UNBUFFERED_DISRUPTION,
                )
            ],
            operational_synthesis="AAPL faces significant headwinds and disruption across its supply chain lines.",
        )
        violations = validate_verbatim_citations(report)
        assert len(violations) >= 1
        assert "unbuffered capacity contraction" in violations[0]


class TestDomainIsolationInvariant:
    """Verifies ADR-00011 Section 5 Domain Isolation Invariant (Domain 1 vs Domain 2)."""

    def test_report_schema_contains_zero_trading_fields(self):
        """OperationalContagionReport must contain zero trading, basket, or order execution fields."""
        disallowed_tokens = {
            "trade",
            "trading",
            "order",
            "basket",
            "short",
            "long",
            "option",
            "options",
            "strike",
            "put",
            "call",
            "collar",
            "portfolio",
            "recommendation",
            "price_target",
            "hedge_ratio",
        }
        report_field_names = list(OperationalContagionReport.model_fields.keys())
        impact_field_names = list(NetworkNodeImpact.model_fields.keys())

        for field in report_field_names + impact_field_names:
            field_tokens = set(field.lower().split("_"))
            overlap = field_tokens.intersection(disallowed_tokens)
            assert not overlap, f"Forbidden trading term(s) {overlap} in field '{field}'"
