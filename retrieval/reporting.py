"""Operational Contagion Report Schema & Language Specification (ADR-00011).

Defines the formal Pydantic contract, controlled categorical taxonomies,
deterministic mathematical metrics, and visual ASCII/Markdown formatting
engine for downstream consumption of YarnBall's computational reading pipeline.
"""

from collections import deque
from datetime import date, timedelta
from enum import Enum
import math
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field, field_validator, model_validator

from contagion.parameters import ParameterProvenance, resolve_edge_cascade_parameters

# ADR-00037 Section 4.5: 3-Segment Provenance Grading Thresholds
DEFAULT_FILING_GROUNDED_MAX_SHARE = 0.10   # ASC 280 statutory 10% materiality threshold
DEFAULT_STRESS_UPPER_BOUND_MIN_SHARE = 0.50  # Majority dominance threshold

CoverageDataStatus = Literal["ok", "degraded", "no_coverage"]
ProvenanceTier = Literal["FILING_GROUNDED", "PARTIALLY_ASSUMED", "STRESS_UPPER_BOUND"]


def classify_provenance_tier(
    default_param_impact_share: float,
    *,
    filing_grounded_max: float = DEFAULT_FILING_GROUNDED_MAX_SHARE,
    stress_upper_min: float = DEFAULT_STRESS_UPPER_BOUND_MIN_SHARE,
) -> ProvenanceTier:
    """Apply 3-segment piecewise provenance grading over `default_param_impact_share` (ADR-00037).

    - `<= 0.10`: `FILING_GROUNDED` (>=90% of propagated loss backed by SEC/extracted params)
    - `0.10 < x < 0.50`: `PARTIALLY_ASSUMED` (material assumption share; edge inspection advised)
    - `>= 0.50`: `STRESS_UPPER_BOUND` (majority rests on `share=1.0, buffer=0` worst-case priors)
    """
    share = max(0.0, min(1.0, float(default_param_impact_share)))
    if share <= filing_grounded_max:
        return "FILING_GROUNDED"
    if share >= stress_upper_min:
        return "STRESS_UPPER_BOUND"
    return "PARTIALLY_ASSUMED"


# ==============================================================================
# 1. CONTROLLED CATEGORICAL TAXONOMIES (CLOSED ENUMS)
# ==============================================================================


class OperationalDisruptionCategory(str, Enum):
    """Physical or commercial mechanism transmitting operational risk through the supply network."""

    # Physical Manufacturing & Fab Outages
    FACILITY_FIRE_OR_EXPLOSION = "FACILITY_FIRE_OR_EXPLOSION"
    NATURAL_DISASTER_FORCE_MAJEURE = "NATURAL_DISASTER_FORCE_MAJEURE"
    UTILITY_POWER_GRID_FAILURE = "UTILITY_POWER_GRID_FAILURE"
    EQUIPMENT_BREAKDOWN_FAB_HALT = "EQUIPMENT_BREAKDOWN_FAB_HALT"

    # Commercial & Contractual Severance
    PRIMARY_SUPPLY_CONTRACT_TERMINATED = "PRIMARY_SUPPLY_CONTRACT_TERMINATED"
    COMMERCIAL_INSOLVENCY_BANKRUPTCY = "COMMERCIAL_INSOLVENCY_BANKRUPTCY"
    DEBT_COVENANT_DEFAULT_CREDIT_FREEZE = "DEBT_COVENANT_DEFAULT_CREDIT_FREEZE"
    PERMANENT_FACILITY_CLOSURE_EXIT = "PERMANENT_FACILITY_CLOSURE_EXIT"

    # Human Capital & Security
    LABOR_UNION_STRIKE_WALKOUT = "LABOR_UNION_STRIKE_WALKOUT"
    RANSOMWARE_CYBER_INFRASTRUCTURE_HALT = "RANSOMWARE_CYBER_INFRASTRUCTURE_HALT"
    KEY_EXECUTIVE_CRISIS_REPLACEMENT = "KEY_EXECUTIVE_CRISIS_REPLACEMENT"

    # Logistics, Legal & Regulatory Interventions
    MARITIME_CANAL_PORT_LOGISTICS_CHOKEPOINT = "MARITIME_CANAL_PORT_LOGISTICS_CHOKEPOINT"
    GOVERNMENT_EXPORT_BAN_OR_SANCTION = "GOVERNMENT_EXPORT_BAN_OR_SANCTION"
    PATENT_IP_INJUNCTION_HALT = "PATENT_IP_INJUNCTION_HALT"
    REGULATORY_SAFETY_RECALL_OR_FDA_HOLD = "REGULATORY_SAFETY_RECALL_OR_FDA_HOLD"
    ACCOUNTING_FRAUD_FINANCIAL_RESTATEMENT = "ACCOUNTING_FRAUD_FINANCIAL_RESTATEMENT"

    # Allocation & Unclassified
    VENDOR_COMPONENT_ALLOCATION_CURTAILMENT = "VENDOR_COMPONENT_ALLOCATION_CURTAILMENT"
    OTHER_UNCLASSIFIED_MATERIAL_HALT = "OTHER_UNCLASSIFIED_MATERIAL_HALT"


# Re-exported from the canonical SEC EDGAR registry (ADR-00045) after OperationalDisruptionCategory
# is bound so ingestion.sec.disruption_gating never encounters circular initialization.
from ingestion.sec.canon import SECForm8KItem, SEC_ITEM_DESCRIPTIONS  # noqa: E402,F401


class DependencyDomain(str, Enum):
    """Domain of supplier dependency, preventing conflation of physical parts and digital infra."""

    PHYSICAL_COMPONENT = "PHYSICAL_COMPONENT"
    DIGITAL_INFRASTRUCTURE = "DIGITAL_INFRASTRUCTURE"
    COMMERCIAL_ENTERPRISE = "COMMERCIAL_ENTERPRISE"


class OperationalScope(str, Enum):
    """Scope of operations impacted by the input dependency."""

    CORE_OPERATIONAL_DELIVERY = "CORE_OPERATIONAL_DELIVERY"
    GROWTH_CAPEX_AND_RD = "GROWTH_CAPEX_AND_RD"


class ContagionSeverityVerdict(str, Enum):
    """Standardized physical boundary condition verdicts (zero magic numbers)."""

    UNBUFFERED_DISRUPTION = "UNBUFFERED_DISRUPTION"
    BUFFER_ABSORBED = "BUFFER_ABSORBED"
    UNAFFECTED = "UNAFFECTED"


class ModelExecutionMode(str, Enum):
    """Execution provenance of neural inference during report generation (ADR-00013)."""

    LIVE_LLM = "LIVE_LLM"
    MOCK_DETERMINISTIC = "MOCK_DETERMINISTIC"
    DEGRADED_UNREACHABLE = "DEGRADED_UNREACHABLE"


def classify_contagion_verdict(
    unbuffered_downtime_days: float,
    upstream_shock_pct: float = 0.0,
    peak_shock_pct: float = 0.0,
) -> ContagionSeverityVerdict:
    """Classifies operational state strictly by the physical boundary condition of unbuffered downtime."""
    if unbuffered_downtime_days > 0.0:
        return ContagionSeverityVerdict.UNBUFFERED_DISRUPTION
    elif upstream_shock_pct > 0.0 or peak_shock_pct > 0.0:
        return ContagionSeverityVerdict.BUFFER_ABSORBED
    else:
        return ContagionSeverityVerdict.UNAFFECTED


# ==============================================================================
# 2. STRUCTURED PYDANTIC CONTRACTS
# ==============================================================================

class NetworkNodeImpact(BaseModel):
    """Deterministic cascade impact on a specific node in the supply network."""

    entity_ticker: str
    hop_distance: int = Field(default=1, ge=1)
    input_group: str = Field(default="default")
    procurement_share_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    inventory_buffer_days: float = Field(default=0.0, ge=0.0)
    shock_onset_day: Optional[float] = None
    unbuffered_downtime_days: float = Field(default=0.0, ge=0.0)
    capacity_contraction_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    lost_capacity_days: float = Field(default=0.0, ge=0.0)
    verdict: ContagionSeverityVerdict
    dependency_domain: DependencyDomain = DependencyDomain.PHYSICAL_COMPONENT
    operational_scope: OperationalScope = OperationalScope.CORE_OPERATIONAL_DELIVERY
    param_provenance: ParameterProvenance = ParameterProvenance.DEFAULT_CONSERVATIVE_BOUND
    share_is_default: bool = False
    buffer_days_is_default: bool = False

    @field_validator("input_group")
    @classmethod
    def normalize_input_group(cls, v: str) -> str:
        """Enforce lowercase snake_case open-world descriptor invariant."""
        cleaned = v.strip().lower().replace(" ", "_").replace("-", "_")
        return cleaned or "default"


class TopologicalEdgeDescriptor(BaseModel):
    """Verified active supply-chain relationship traversed during contagion analysis."""

    source: str
    target: str
    rel_type: str = "SUPPLIES_TO"
    hop_distance: int = Field(default=1, ge=1)
    nature: str = Field(default="default")
    share: float = Field(default=1.0, ge=0.0, le=1.0)
    buffer_days: float = Field(default=0.0, ge=0.0)
    param_provenance: ParameterProvenance = ParameterProvenance.DEFAULT_CONSERVATIVE_BOUND
    share_is_default: bool = False
    buffer_days_is_default: bool = False
    dependency_domain: DependencyDomain = DependencyDomain.PHYSICAL_COMPONENT
    operational_scope: OperationalScope = OperationalScope.CORE_OPERATIONAL_DELIVERY

    @field_validator("nature")
    @classmethod
    def normalize_nature(cls, v: str) -> str:
        cleaned = v.strip().lower().replace(" ", "_").replace("-", "_")
        return cleaned or "default"


class ExecutionTelemetry(BaseModel):
    """Per-stage latency breakdown and performance telemetry."""

    task_c_cypher_ms: float = 0.0
    ast_guardrail_ms: float = 0.0
    graph_traversal_ms: float = 0.0
    cascade_propagation_ms: float = 0.0
    task_d_contagion_ms: float = 0.0
    total_latency_ms: float = 0.0
    extracted_parameter_edge_count: int = 0
    defaulted_parameter_edge_count: int = 0


class ReportCoverage(BaseModel):
    """Structured coverage, truncation, and 3-segment provenance grading metadata (ADR-00037)."""

    data_status: CoverageDataStatus = "ok"
    edges_total: int = Field(default=0, ge=0)
    edges_shown: int = Field(default=0, ge=0)
    truncated: bool = False
    dropped_impact_share: float = Field(default=0.0, ge=0.0, le=1.0)
    unanchored_excluded: int = Field(default=0, ge=0)
    legacy_unchunked_evidence: int = Field(default=0, ge=0)
    default_param_impact_share: float = Field(default=0.0, ge=0.0, le=1.0)
    provenance_tier: ProvenanceTier = "FILING_GROUNDED"

    @model_validator(mode="after")
    def _enforce_truncation_invariant(self) -> "ReportCoverage":
        """ADR-00037 Invariant #3: Truncation is never silent when `edges_total > edges_shown`."""
        if self.edges_total > self.edges_shown:
            self.truncated = True
        return self


class OperationalContagionReport(BaseModel):
    """Unified machine-readable schema for YarnBall Operational Contagion Intelligence (ADR-00011 / ADR-00037)."""

    report_id: str
    event_id: str
    filing_type: str = "SEC Form 8-K"
    sec_item: str
    disrupted_entity_ticker: str
    disrupted_entity_cik: Optional[str] = None
    disrupted_entity_name: Optional[str] = None
    filing_date: str
    filing_timestamp: Optional[str] = None
    disruption_category: str
    event_summary: str
    shock_duration_days: float = Field(ge=0.0)
    shock_magnitude_pct: float = Field(ge=0.0, le=100.0)
    network_impacts: List[NetworkNodeImpact] = Field(default_factory=list)
    spectral_radius: float = Field(default=0.0, ge=0.0)
    is_acyclic_dag: bool = True
    active_topology_edges: List[TopologicalEdgeDescriptor] = Field(default_factory=list)
    contagion_think_trace: Optional[str] = None
    operational_synthesis: str
    telemetry: ExecutionTelemetry = Field(default_factory=ExecutionTelemetry)
    model_mode: ModelExecutionMode = ModelExecutionMode.LIVE_LLM
    coverage: ReportCoverage = Field(default_factory=ReportCoverage)

    @model_validator(mode="after")
    def _enforce_coverage_degraded_invariant(self) -> "OperationalContagionReport":
        """ADR-00037 Invariant #2: `data_status='ok'` is impossible while any store or model is degraded."""
        if self.model_mode == ModelExecutionMode.DEGRADED_UNREACHABLE:
            if "coverage" in self.model_fields_set and self.coverage.data_status == "ok":
                raise ValueError(
                    "ADR-00037 Invariant #2 violation: coverage.data_status='ok' is forbidden when report is degraded."
                )
            if self.coverage.data_status == "ok":
                self.coverage = self.coverage.model_copy(update={"data_status": "degraded"})
        return self


class ContagionTimeSeriesEventRecord(BaseModel):
    """Flattened 1NF operational contagion event row for time-series relational joins (ADR-00013 / ADR-00037)."""

    report_id: str
    event_id: str
    filing_date: str
    filing_timestamp: Optional[str] = None
    sec_item: str
    disruption_category: str
    disrupted_entity_ticker: str
    disrupted_entity_cik: Optional[str] = None
    target_entity_ticker: str
    target_entity_cik: Optional[str] = None
    hop_distance: int = Field(default=0, ge=0)
    input_group: str = Field(default="default")
    dependency_domain: str = DependencyDomain.PHYSICAL_COMPONENT.value
    operational_scope: str = OperationalScope.CORE_OPERATIONAL_DELIVERY.value
    procurement_share_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    inventory_buffer_days: float = Field(default=0.0, ge=0.0)
    shock_onset_day: Optional[float] = None
    unbuffered_downtime_days: float = Field(default=0.0, ge=0.0)
    capacity_contraction_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    lost_capacity_days: float = Field(default=0.0, ge=0.0)
    calendar_onset_date: Optional[str] = None
    calendar_recovery_date: Optional[str] = None
    verdict: str
    is_acyclic_dag: bool = True
    spectral_radius: float = Field(default=0.0, ge=0.0)
    model_mode: str = ModelExecutionMode.LIVE_LLM.value
    data_status: str = "ok"
    provenance_tier: str = "FILING_GROUNDED"
    default_param_impact_share: float = Field(default=0.0, ge=0.0, le=1.0)
    truncated: bool = False
    edges_total: int = Field(default=0, ge=0)
    edges_shown: int = Field(default=0, ge=0)
    unanchored_excluded: int = Field(default=0, ge=0)
    legacy_unchunked_evidence: int = Field(default=0, ge=0)


# ==============================================================================
# 3. DETERMINISTIC VISUAL INSPECTION FORMATTER (SECTION 4.5B)
# ==============================================================================

def format_operational_contagion_report(report: OperationalContagionReport) -> str:
    """Renders the standardized ASCII/Markdown visual inspection report matching ADR-00011 Section 4.5B."""
    lines: List[str] = []

    # 1. Report Header Banner
    lines.append("=" * 80)
    lines.append("YARNBALL OPERATIONAL CONTAGION REPORT")
    lines.append("=" * 80)
    lines.append(f"REPORT ID:          {report.report_id}")
    lines.append(f"EVENT ID:           {report.event_id}")
    lines.append(f"FILING FORM:        {report.filing_type}")

    # SEC Item with canonical title if known
    sec_item_str = report.sec_item
    if "(" not in sec_item_str and sec_item_str in SEC_ITEM_DESCRIPTIONS:
        sec_item_str = f"{sec_item_str} ({SEC_ITEM_DESCRIPTIONS[sec_item_str]})"
    lines.append(f"SEC STATUTORY ITEM: {sec_item_str}")

    lines.append(f"FILING DATE:        {report.filing_date}")

    # Disrupted Entity presentation
    if report.disrupted_entity_name and report.disrupted_entity_cik:
        entity_display = f"{report.disrupted_entity_name} ({report.disrupted_entity_ticker} | CIK: {report.disrupted_entity_cik})"
    elif report.disrupted_entity_name:
        entity_display = f"{report.disrupted_entity_name} ({report.disrupted_entity_ticker})"
    elif report.disrupted_entity_cik:
        entity_display = f"{report.disrupted_entity_ticker} (CIK: {report.disrupted_entity_cik})"
    else:
        entity_display = report.disrupted_entity_ticker
    lines.append(f"DISRUPTED ENTITY:   {entity_display}")

    lines.append(f"DISRUPTION TYPE:    {report.disruption_category}")
    lines.append(
        f"SHOCK SPECIFICATION: {report.shock_magnitude_pct:.1f}% plant halt for {report.shock_duration_days:.1f} calendar days"
    )
    lines.append("")

    # 2. Section 1: Deterministic Cascade Shock Metrics
    lines.append("-" * 80)
    lines.append("1. DETERMINISTIC CASCADE SHOCK METRICS (SparseNetworkCascadeEngine)")
    lines.append("-" * 80)
    lines.append("Entity | Hop | Input Group   | Share  | Buffer | Onset  | Downtime | Contraction | Verdict")
    lines.append("-------|-----|---------------|--------|--------|--------|----------|-------------|----------------------------")

    for impact in report.network_impacts:
        ent_col = f"{impact.entity_ticker:<7}"
        hop_col = f" {impact.hop_distance:<4}"
        grp_col = f" {impact.input_group:<14}"
        share_col = f" {impact.procurement_share_pct:5.1f}% "
        buf_col = f" {impact.inventory_buffer_days:5.1f}d "

        if impact.shock_onset_day is not None:
            onset_txt = f"Day {int(impact.shock_onset_day)}"
            onset_col = f" {onset_txt:<6} "
        else:
            onset_col = " None   "

        down_col = f" {impact.unbuffered_downtime_days:7.1f}d "

        if impact.capacity_contraction_pct > 0.0:
            contraction_val = -impact.capacity_contraction_pct
            cont_col = f" {contraction_val:10.1f}% "
        else:
            cont_col = f" {0.0:10.1f}% "

        verdict_str = impact.verdict.value if hasattr(impact.verdict, "value") else str(impact.verdict)
        verdict_col = f" {verdict_str}"

        row = (
            f"{ent_col}|{hop_col}|{grp_col}|{share_col}|{buf_col}|"
            f"{onset_col}|{down_col}|{cont_col}|{verdict_col}"
        )
        lines.append(row)

    lines.append("")

    # Spectral radius summary
    if report.is_acyclic_dag or report.spectral_radius == 0.0:
        dag_note = "Topology is provably acyclic DAG"
    elif report.spectral_radius < 1.0:
        dag_note = "Dampened circular feedback, rho < 1.0"
    else:
        dag_note = "CRITICAL: Amplifying cyclic feedback, rho >= 1.0"

    lines.append(f"Spectral Radius: {report.spectral_radius:.4f} ({dag_note})")
    lines.append("")

    # 3. Section 2: Active Knowledge Graph Topology
    lines.append("-" * 80)
    lines.append("2. ACTIVE KNOWLEDGE GRAPH TOPOLOGY")
    lines.append("-" * 80)

    # Group edges by hop distance ascending
    hop_groups: Dict[int, List[TopologicalEdgeDescriptor]] = {}
    for edge in report.active_topology_edges:
        hop_groups.setdefault(edge.hop_distance, []).append(edge)

    for hop in sorted(hop_groups.keys()):
        lines.append(f"[Hop Distance {hop}]")
        for edge in hop_groups[hop]:
            edge_line = (
                f"- ({edge.source}) -[:{edge.rel_type} "
                f"{{nature: '{edge.nature}', share: {edge.share:.2f}, buffer_days: {edge.buffer_days:.1f}}}]-> "
                f"({edge.target})"
            )
            lines.append(edge_line)
        lines.append("")

    # 4. Section 3: Computational Reading & Reasoning
    lines.append("-" * 80)
    lines.append("3. COMPUTATIONAL READING & REASONING (yarnball-qwen:7b)")
    lines.append("-" * 80)

    if report.contagion_think_trace and report.contagion_think_trace.strip():
        lines.append("<think>")
        lines.append(report.contagion_think_trace.strip())
        lines.append("</think>")
        lines.append("")

    lines.append("Operational Synthesis:")
    lines.append(report.operational_synthesis.strip())
    lines.append("")

    # 5. Section 4: Pipeline Telemetry
    lines.append("-" * 80)
    lines.append("4. PIPELINE TELEMETRY")
    lines.append("-" * 80)
    t = report.telemetry
    lines.append(f"{'Cypher Retrieval (Task C):':<30}{t.task_c_cypher_ms:5.1f} ms")
    lines.append(f"{'AST Guardrail Check:':<30}{t.ast_guardrail_ms:5.1f} ms")
    lines.append(f"{'Graph Traversal:':<30}{t.graph_traversal_ms:5.1f} ms")
    lines.append(f"{'Cascade Propagation:':<30}{t.cascade_propagation_ms:5.1f} ms")
    lines.append(f"{'Model Contagion Reasoning:':<30}{t.task_d_contagion_ms:5.1f} ms")
    lines.append(f"{'Total Pipeline Latency:':<30}{t.total_latency_ms:5.1f} ms")
    if t.extracted_parameter_edge_count > 0 or t.defaulted_parameter_edge_count > 0:
        lines.append(
            f"{'Parameter Provenance:':<30}"
            f"{t.extracted_parameter_edge_count} extracted, "
            f"{t.defaulted_parameter_edge_count} default bounds"
        )
    cov = report.coverage
    lines.append(
        f"{'Coverage & Provenance:':<30}"
        f"status={cov.data_status}, tier={cov.provenance_tier}, "
        f"edges={cov.edges_shown}/{cov.edges_total}"
        f"{' (TRUNCATED)' if cov.truncated else ''}, "
        f"default_impact_share={cov.default_param_impact_share:.1%}"
    )
    lines.append("=" * 80)

    return "\n".join(lines)


# ==============================================================================
# 4. PIPELINE REPORT BUILDER
# ==============================================================================

def build_operational_contagion_report(
    result: Any,  # PipelineResult
    event_id: str,
    sec_item: Union[SECForm8KItem, str],
    disrupted_entity_ticker: str,
    filing_date: str,
    disruption_category: Union[OperationalDisruptionCategory, str],
    event_summary: str,
    shock_duration_days: float,
    shock_magnitude_pct: float,
    disrupted_entity_name: Optional[str] = None,
    disrupted_entity_cik: Optional[str] = None,
    report_id: Optional[str] = None,
    filing_timestamp: Optional[str] = None,
    model_mode: Union[ModelExecutionMode, str] = ModelExecutionMode.LIVE_LLM,
) -> OperationalContagionReport:
    """Builds a validated OperationalContagionReport from a PipelineResult and event metadata."""
    ticker_clean = disrupted_entity_ticker.strip().upper()

    # Generate canonical report_id if missing
    if not report_id:
        date_stamp = filing_date.replace("-", "").strip()
        report_id = f"REP-{date_stamp}-{ticker_clean}-001"

    # Extract SEC item and Disruption category string representations
    sec_item_val = sec_item.value if isinstance(sec_item, SECForm8KItem) else str(sec_item)
    disrupt_cat_val = (
        disruption_category.value
        if isinstance(disruption_category, OperationalDisruptionCategory)
        else str(disruption_category)
    )

    # 1. Compute hop distances from shock origin across retrieved edges
    raw_edges: List[Dict[str, Any]] = getattr(result, "retrieved_subgraph", []) or []
    adj: Dict[str, List[str]] = {}
    for r in raw_edges:
        s = str(r.get("source") or r.get("source_ticker") or "").upper()
        t = str(r.get("target") or r.get("target_ticker") or "").upper()
        if s and t:
            adj.setdefault(s, []).append(t)

    distances: Dict[str, int] = {ticker_clean: 0}
    queue: deque = deque([ticker_clean])
    while queue:
        curr = queue.popleft()
        d_curr = distances[curr]
        for neighbor in adj.get(curr, []):
            if neighbor not in distances:
                distances[neighbor] = d_curr + 1
                queue.append(neighbor)

    # 2. Build Active Topology Edge Descriptors
    active_edges: List[TopologicalEdgeDescriptor] = []
    for r in raw_edges:
        s = str(r.get("source") or r.get("source_ticker") or "UNKNOWN").upper()
        t = str(r.get("target") or r.get("target_ticker") or "UNKNOWN").upper()
        rel_type = str(r.get("rel_type") or r.get("type") or "SUPPLIES_TO")
        resolved = resolve_edge_cascade_parameters({**r, "rel_type": rel_type})

        # Hop distance of the target node
        hop_dist = distances.get(t, distances.get(s, 0) + 1)
        if hop_dist < 1:
            hop_dist = 1

        domain_str = str(resolved["dependency_domain"])
        domain_enum = (
            DependencyDomain(domain_str)
            if domain_str in DependencyDomain._value2member_map_
            else DependencyDomain.PHYSICAL_COMPONENT
        )

        active_edges.append(
            TopologicalEdgeDescriptor(
                source=s,
                target=t,
                rel_type=rel_type,
                hop_distance=hop_dist,
                nature=str(resolved["input_group"]),
                share=round(float(resolved["share"]), 4),
                buffer_days=float(resolved["buffer_days"]),
                param_provenance=ParameterProvenance(str(resolved["param_provenance"])),
                share_is_default=bool(resolved["share_is_default"]),
                buffer_days_is_default=bool(resolved["buffer_days_is_default"]),
                dependency_domain=domain_enum,
            )
        )

    # Sort active edges by hop distance then source then target
    active_edges.sort(key=lambda e: (e.hop_distance, e.source, e.target))

    # 3. Build Network Node Impacts from Cascade Summaries
    cascade_summaries: Dict[str, Any] = getattr(result, "cascade_summaries", {}) or {}
    if not cascade_summaries and getattr(result, "convergent_shock_summary", None):
        tgt_sum = result.convergent_shock_summary
        cascade_summaries = {tgt_sum.target_entity: tgt_sum}

    impacts: List[NetworkNodeImpact] = []
    for nid, summary in cascade_summaries.items():
        nid_clean = str(nid).upper()
        # Exclude shock origin from downstream impacts
        if nid_clean == ticker_clean:
            continue

        hop_dist = distances.get(nid_clean, 1)

        # Look up incoming edge properties for this node
        input_grp = "default"
        share_pct = 0.0
        buf_days = 0.0
        imp_prov = ParameterProvenance.DEFAULT_CONSERVATIVE_BOUND
        imp_share_def = False
        imp_buf_def = False
        imp_domain = DependencyDomain.PHYSICAL_COMPONENT

        incoming = [e for e in active_edges if e.target == nid_clean]
        if incoming:
            imp_prov = incoming[0].param_provenance
            imp_share_def = incoming[0].share_is_default
            imp_buf_def = incoming[0].buffer_days_is_default
            imp_domain = incoming[0].dependency_domain

        if hasattr(summary, "contributing_edges") and summary.contributing_edges:
            first_c = summary.contributing_edges[0]
            input_grp = first_c.get("input_group", "default")
            share_pct = float(first_c.get("share_pct", 0.0))
            buf_days = float(first_c.get("buffer_days", 0.0))
        elif incoming:
            input_grp = incoming[0].nature
            share_pct = round(incoming[0].share * 100.0, 2)
            buf_days = incoming[0].buffer_days

        downtime = float(getattr(summary, "downtime_days", 0.0) or 0.0)
        peak_shock = float(getattr(summary, "peak_shock_pct", 0.0) or 0.0)
        lost_cap = float(getattr(summary, "lost_capacity_days", 0.0) or 0.0)
        onset_day = getattr(summary, "onset_day", None)

        verdict = classify_contagion_verdict(
            unbuffered_downtime_days=downtime,
            upstream_shock_pct=shock_magnitude_pct,
            peak_shock_pct=peak_shock,
        )

        impacts.append(
            NetworkNodeImpact(
                entity_ticker=nid_clean,
                hop_distance=hop_dist,
                input_group=input_grp,
                procurement_share_pct=share_pct,
                inventory_buffer_days=buf_days,
                shock_onset_day=onset_day,
                unbuffered_downtime_days=downtime,
                capacity_contraction_pct=peak_shock,
                lost_capacity_days=lost_cap,
                verdict=verdict,
                dependency_domain=imp_domain,
                param_provenance=imp_prov,
                share_is_default=imp_share_def,
                buffer_days_is_default=imp_buf_def,
            )
        )

    # Sort impacts by hop distance ascending then ticker alphabetical
    impacts.sort(key=lambda i: (i.hop_distance, i.entity_ticker))

    # 4. Spectral Radius & DAG Status
    spectral_rad = 0.0
    for summ in cascade_summaries.values():
        if hasattr(summ, "spectral_radius") and summ.spectral_radius is not None:
            spectral_rad = float(summ.spectral_radius)
            break

    is_dag = spectral_rad == 0.0

    # 5. Extract Telemetry Breakdown
    defaulted_edge_count = sum(
        1 for edge in active_edges if edge.param_provenance == ParameterProvenance.DEFAULT_CONSERVATIVE_BOUND
    )
    extracted_edge_count = len(active_edges) - defaulted_edge_count
    lat_breakdown = getattr(result, "latency_breakdown_ms", {}) or {}
    telemetry = ExecutionTelemetry(
        task_c_cypher_ms=round(float(lat_breakdown.get("task_c_cypher_ms", 0.0)), 1),
        ast_guardrail_ms=round(float(lat_breakdown.get("guardrail_validation_ms", 0.0)), 1),
        graph_traversal_ms=round(float(lat_breakdown.get("graph_execution_ms", 0.0)), 1),
        cascade_propagation_ms=round(float(lat_breakdown.get("cascade_engine_ms", 0.0)), 1),
        task_d_contagion_ms=round(float(lat_breakdown.get("task_d_contagion_ms", 0.0)), 1),
        total_latency_ms=round(float(getattr(result, "total_latency_ms", 0.0) or 0.0), 1),
        extracted_parameter_edge_count=extracted_edge_count,
        defaulted_parameter_edge_count=defaulted_edge_count,
    )

    # 6. Model synthesis, think trace & structured ReportCoverage (ADR-00037)
    think_trace = getattr(result, "contagion_think_trace", None)
    operational_synthesis = getattr(result, "contagion_verdict", "") or "No operational synthesis generated."
    mode_enum = (
        model_mode
        if isinstance(model_mode, ModelExecutionMode)
        else ModelExecutionMode(str(model_mode))
    )

    edges_total_val = int(getattr(result, "edges_total", 0) or len(raw_edges))
    if edges_total_val < len(raw_edges):
        edges_total_val = len(raw_edges)
    raw_edges_shown = getattr(result, "edges_shown", None)
    if isinstance(raw_edges_shown, int) and raw_edges_shown > 0:
        edges_shown_val = min(raw_edges_shown, edges_total_val)
    else:
        edges_shown_val = min(len(raw_edges), 25)
    truncated_val = bool(getattr(result, "truncated", False)) or (edges_total_val > edges_shown_val)
    dropped_impact_share_val = max(
        0.0, min(1.0, float(getattr(result, "dropped_impact_share", 0.0) or 0.0))
    )
    unanchored_excluded_val = max(0, int(getattr(result, "unanchored_excluded", 0) or 0))

    raw_legacy = getattr(result, "legacy_unchunked_evidence", None)
    if isinstance(raw_legacy, int) and raw_legacy > 0:
        legacy_unchunked_val = raw_legacy
    else:
        legacy_unchunked_val = 0
        for r in raw_edges:
            raw_p = r.get("properties")
            r_props: Dict[str, Any] = raw_p if isinstance(raw_p, dict) else {}
            ev_st = str(r.get("evidence_status") or r_props.get("evidence_status") or "").strip().lower()
            cids = r.get("evidence_chunk_ids") or r_props.get("evidence_chunk_ids")
            if ev_st == "legacy_unchunked" or not cids:
                legacy_unchunked_val += 1

    # Compute share of propagated impact resting on DEFAULT_CONSERVATIVE_BOUND (share=1.0, buffer=0)
    total_lost_cap = sum(max(0.0, float(i.lost_capacity_days)) for i in impacts)
    total_contraction = sum(max(0.0, float(i.capacity_contraction_pct)) for i in impacts)
    if total_lost_cap > 0.0:
        default_impact_num = sum(
            max(0.0, float(i.lost_capacity_days))
            for i in impacts
            if i.param_provenance == ParameterProvenance.DEFAULT_CONSERVATIVE_BOUND
        )
        default_param_impact_share = round(min(1.0, default_impact_num / total_lost_cap), 6)
    elif total_contraction > 0.0:
        default_impact_num = sum(
            max(0.0, float(i.capacity_contraction_pct))
            for i in impacts
            if i.param_provenance == ParameterProvenance.DEFAULT_CONSERVATIVE_BOUND
        )
        default_param_impact_share = round(min(1.0, default_impact_num / total_contraction), 6)
    elif active_edges:
        default_param_impact_share = round(defaulted_edge_count / float(len(active_edges)), 6)
    else:
        default_param_impact_share = 0.0

    provenance_tier = classify_provenance_tier(default_param_impact_share)
    is_degraded = bool(getattr(result, "degraded", False)) or mode_enum == ModelExecutionMode.DEGRADED_UNREACHABLE
    data_status: CoverageDataStatus
    if is_degraded:
        data_status = "degraded"
    elif edges_total_val == 0:
        data_status = "no_coverage"
    else:
        data_status = "ok"

    coverage = ReportCoverage(
        data_status=data_status,
        edges_total=edges_total_val,
        edges_shown=edges_shown_val,
        truncated=truncated_val,
        dropped_impact_share=dropped_impact_share_val,
        unanchored_excluded=unanchored_excluded_val,
        legacy_unchunked_evidence=legacy_unchunked_val,
        default_param_impact_share=default_param_impact_share,
        provenance_tier=provenance_tier,
    )

    return OperationalContagionReport(
        report_id=report_id,
        event_id=event_id,
        filing_type="SEC Form 8-K",
        sec_item=sec_item_val,
        disrupted_entity_ticker=ticker_clean,
        disrupted_entity_cik=disrupted_entity_cik,
        disrupted_entity_name=disrupted_entity_name,
        filing_date=filing_date,
        filing_timestamp=filing_timestamp,
        disruption_category=disrupt_cat_val,
        event_summary=event_summary,
        shock_duration_days=shock_duration_days,
        shock_magnitude_pct=shock_magnitude_pct,
        network_impacts=impacts,
        spectral_radius=spectral_rad,
        is_acyclic_dag=is_dag,
        active_topology_edges=active_edges,
        contagion_think_trace=think_trace,
        operational_synthesis=operational_synthesis,
        telemetry=telemetry,
        model_mode=mode_enum,
        coverage=coverage,
    )


# ==============================================================================
# 5. VERBATIM CITATION VALIDATOR
# ==============================================================================

def validate_verbatim_citations(report: OperationalContagionReport) -> List[str]:
    """Validates that prose synthesis cites continuous deterministic metrics verbatim (Section 4.4 Invariant 2)."""
    violations: List[str] = []
    prose = report.operational_synthesis.lower()

    for impact in report.network_impacts:
        if impact.verdict == ContagionSeverityVerdict.UNBUFFERED_DISRUPTION:
            # Check percentage citation
            pct_str_1 = f"{impact.capacity_contraction_pct:.1f}%".lower()
            pct_str_2 = f"{impact.capacity_contraction_pct:.2f}%".lower()
            pct_str_3 = f"{impact.capacity_contraction_pct:.0f}%".lower()
            if not any(p in prose for p in (pct_str_1, pct_str_2, pct_str_3)):
                violations.append(
                    f"Entity {impact.entity_ticker} unbuffered capacity contraction ({pct_str_1}) not cited verbatim in prose."
                )

            # Check downtime citation
            days_str_1 = f"{impact.unbuffered_downtime_days:.1f}".lower()
            days_str_2 = f"{int(impact.unbuffered_downtime_days)}".lower()
            if not any(d in prose for d in (days_str_1, days_str_2)):
                violations.append(
                    f"Entity {impact.entity_ticker} unbuffered downtime ({days_str_1} days) not cited verbatim in prose."
                )

    return violations


# ==============================================================================
# 6. 1NF TIME-SERIES JOIN FLATTENER (ADR-00013)
# ==============================================================================

def _project_calendar_dates(
    filing_date: str,
    shock_onset_day: Optional[float],
    unbuffered_downtime_days: float,
) -> Tuple[Optional[str], Optional[str]]:
    """Projects ISO-8601 calendar onset and recovery dates from filing_date and cascade offsets."""
    if shock_onset_day is None:
        return None, None
    base_date = date.fromisoformat(filing_date.strip()[:10])
    onset_dt = base_date + timedelta(days=int(math.floor(shock_onset_day)))
    onset_str = onset_dt.isoformat()
    if unbuffered_downtime_days > 0.0:
        recovery_dt = onset_dt + timedelta(days=int(math.ceil(unbuffered_downtime_days)))
        return onset_str, recovery_dt.isoformat()
    return onset_str, None


def flatten_report_to_timeseries_records(
    report: OperationalContagionReport,
    model_mode: Optional[Union[ModelExecutionMode, str]] = None,
    target_cik_map: Optional[Dict[str, Optional[str]]] = None,
) -> List[ContagionTimeSeriesEventRecord]:
    """Flattens OperationalContagionReport into 1NF rows for time-series joins (ADR-00013)."""
    if model_mode is None:
        mode_val = report.model_mode.value
    elif isinstance(model_mode, ModelExecutionMode):
        mode_val = model_mode.value
    else:
        mode_val = ModelExecutionMode(str(model_mode)).value

    cik_lookup = {k.strip().upper(): v for k, v in (target_cik_map or {}).items()}
    base_kwargs: Dict[str, Any] = {
        "report_id": report.report_id,
        "event_id": report.event_id,
        "filing_date": report.filing_date,
        "filing_timestamp": report.filing_timestamp,
        "sec_item": report.sec_item,
        "disruption_category": report.disruption_category,
        "disrupted_entity_ticker": report.disrupted_entity_ticker,
        "disrupted_entity_cik": report.disrupted_entity_cik,
        "is_acyclic_dag": report.is_acyclic_dag,
        "spectral_radius": report.spectral_radius,
        "model_mode": mode_val,
        "data_status": report.coverage.data_status,
        "provenance_tier": report.coverage.provenance_tier,
        "default_param_impact_share": report.coverage.default_param_impact_share,
        "truncated": report.coverage.truncated,
        "edges_total": report.coverage.edges_total,
        "edges_shown": report.coverage.edges_shown,
        "unanchored_excluded": report.coverage.unanchored_excluded,
        "legacy_unchunked_evidence": report.coverage.legacy_unchunked_evidence,
    }

    if not report.network_impacts:
        return [
            ContagionTimeSeriesEventRecord(
                **base_kwargs,
                target_entity_ticker=report.disrupted_entity_ticker,
                target_entity_cik=report.disrupted_entity_cik,
                hop_distance=0,
                input_group="root",
                verdict=ContagionSeverityVerdict.UNAFFECTED.value,
            )
        ]

    records: List[ContagionTimeSeriesEventRecord] = []
    for impact in report.network_impacts:
        onset_str, recovery_str = _project_calendar_dates(
            report.filing_date,
            impact.shock_onset_day,
            impact.unbuffered_downtime_days,
        )
        tgt_ticker = impact.entity_ticker.strip().upper()
        records.append(
            ContagionTimeSeriesEventRecord(
                **base_kwargs,
                target_entity_ticker=tgt_ticker,
                target_entity_cik=cik_lookup.get(tgt_ticker),
                hop_distance=impact.hop_distance,
                input_group=impact.input_group,
                dependency_domain=impact.dependency_domain.value,
                operational_scope=impact.operational_scope.value,
                procurement_share_pct=impact.procurement_share_pct,
                inventory_buffer_days=impact.inventory_buffer_days,
                shock_onset_day=impact.shock_onset_day,
                unbuffered_downtime_days=impact.unbuffered_downtime_days,
                capacity_contraction_pct=impact.capacity_contraction_pct,
                lost_capacity_days=impact.lost_capacity_days,
                calendar_onset_date=onset_str,
                calendar_recovery_date=recovery_str,
                verdict=impact.verdict.value,
            )
        )
    return records
