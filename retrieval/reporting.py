"""Operational Contagion Report Schema & Language Specification (ADR-00011).

Defines the formal Pydantic contract, controlled categorical taxonomies,
deterministic mathematical metrics, and visual ASCII/Markdown formatting
engine for downstream consumption of YarnBall's computational reading pipeline.
"""

from collections import deque
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from pydantic import BaseModel, Field, field_validator


# ==============================================================================
# 1. CONTROLLED CATEGORICAL TAXONOMIES (CLOSED ENUMS)
# ==============================================================================

class SECForm8KItem(str, Enum):
    """Exhaustive statutory SEC Form 8-K Item canon across all 9 statutory sections."""
    # Section 1: Registrant's Business and Operations
    ITEM_1_01 = "Item 1.01"  # Entry into a Material Definitive Agreement
    ITEM_1_02 = "Item 1.02"  # Termination of a Material Definitive Agreement
    ITEM_1_03 = "Item 1.03"  # Bankruptcy or Receivership
    ITEM_1_04 = "Item 1.04"  # Mine Safety - Reporting of Shutdowns & Violations
    ITEM_1_05 = "Item 1.05"  # Material Cybersecurity Incidents
    # Section 2: Financial Information
    ITEM_2_01 = "Item 2.01"  # Completion of Acquisition or Disposition of Assets
    ITEM_2_02 = "Item 2.02"  # Results of Operations and Financial Condition
    ITEM_2_03 = "Item 2.03"  # Creation of a Direct Financial Obligation
    ITEM_2_04 = "Item 2.04"  # Acceleration of Obligations (Debt/Credit Default)
    ITEM_2_05 = "Item 2.05"  # Costs Associated with Exit/Disposal (Plant Closures)
    ITEM_2_06 = "Item 2.06"  # Material Impairments (Asset/Plant Write-Offs)
    # Section 3: Securities and Trading Markets
    ITEM_3_01 = "Item 3.01"  # Notice of Delisting / Failure to Satisfy Listing Rule
    ITEM_3_02 = "Item 3.02"  # Unregistered Sales of Equity Securities
    ITEM_3_03 = "Item 3.03"  # Material Modification to Rights of Security Holders
    # Section 4: Accountants and Financial Statements
    ITEM_4_01 = "Item 4.01"  # Changes in Registrant's Certifying Accountant
    ITEM_4_02 = "Item 4.02"  # Non-Reliance on Previously Issued Financial Statements
    # Section 5: Corporate Governance and Management
    ITEM_5_01 = "Item 5.01"  # Changes in Control of Registrant
    ITEM_5_02 = "Item 5.02"  # Departure/Election of Directors or Principal Officers
    ITEM_5_03 = "Item 5.03"  # Amendments to Articles of Incorporation or Bylaws
    ITEM_5_04 = "Item 5.04"  # Temporary Suspension of Trading Under Benefit Plans
    ITEM_5_05 = "Item 5.05"  # Amendments/Waivers to Code of Ethics
    ITEM_5_06 = "Item 5.06"  # Change in Shell Company Status
    ITEM_5_07 = "Item 5.07"  # Submission of Matters to a Vote of Security Holders
    ITEM_5_08 = "Item 5.08"  # Shareholder Director Nominations
    # Section 6: Asset-Backed Securities
    ITEM_6_01 = "Item 6.01"  # ABS Informational and Computational Material
    ITEM_6_02 = "Item 6.02"  # Change of Servicer or Trustee
    ITEM_6_03 = "Item 6.03"  # Change in Credit Enhancement or External Support
    ITEM_6_04 = "Item 6.04"  # Failure to Make a Required Distribution
    ITEM_6_05 = "Item 6.05"  # Securities Act Updating Disclosure
    # Section 7: Regulation FD
    ITEM_7_01 = "Item 7.01"  # Regulation FD Disclosure
    # Section 8: Other Material Events
    ITEM_8_01 = "Item 8.01"  # Other Events (Statutory General Catch-All)
    # Section 9: Financial Statements and Exhibits
    ITEM_9_01 = "Item 9.01"  # Financial Statements and Exhibits


SEC_ITEM_DESCRIPTIONS: Dict[str, str] = {
    "Item 1.01": "Entry into a Material Definitive Agreement",
    "Item 1.02": "Termination of a Material Definitive Agreement",
    "Item 1.03": "Bankruptcy or Receivership",
    "Item 1.04": "Mine Safety - Reporting of Shutdowns and Violations",
    "Item 1.05": "Material Cybersecurity Incidents",
    "Item 2.01": "Completion of Acquisition or Disposition of Assets",
    "Item 2.02": "Results of Operations and Financial Condition",
    "Item 2.03": "Creation of a Direct Financial Obligation",
    "Item 2.04": "Acceleration of Obligations (Debt/Credit Default)",
    "Item 2.05": "Costs Associated with Exit/Disposal Activities (Plant Closures)",
    "Item 2.06": "Material Impairments (Asset/Plant Write-Offs)",
    "Item 3.01": "Notice of Delisting or Failure to Satisfy a Continued Listing Rule",
    "Item 3.02": "Unregistered Sales of Equity Securities",
    "Item 3.03": "Material Modification to Rights of Security Holders",
    "Item 4.01": "Changes in Registrant's Certifying Accountant",
    "Item 4.02": "Non-Reliance on Previously Issued Financial Statements",
    "Item 5.01": "Changes in Control of Registrant",
    "Item 5.02": "Departure of Directors or Certain Officers; Election of Directors",
    "Item 5.03": "Amendments to Articles of Incorporation or Bylaws",
    "Item 5.04": "Temporary Suspension of Trading Under Employee Benefit Plans",
    "Item 5.05": "Amendments to the Registrant's Code of Ethics",
    "Item 5.06": "Change in Shell Company Status",
    "Item 5.07": "Submission of Matters to a Vote of Security Holders",
    "Item 5.08": "Shareholder Director Nominations",
    "Item 6.01": "ABS Informational and Computational Material",
    "Item 6.02": "Change of Servicer or Trustee",
    "Item 6.03": "Change in Credit Enhancement or External Support",
    "Item 6.04": "Failure to Make a Required Distribution",
    "Item 6.05": "Securities Act Updating Disclosure",
    "Item 7.01": "Regulation FD Disclosure",
    "Item 8.01": "Other Events",
    "Item 9.01": "Financial Statements and Exhibits",
}


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


class OperationalContagionReport(BaseModel):
    """Unified machine-readable schema for YarnBall Operational Contagion Intelligence (ADR-00011)."""

    report_id: str
    event_id: str
    filing_type: str = "SEC Form 8-K"
    sec_item: str
    disrupted_entity_ticker: str
    disrupted_entity_cik: Optional[str] = None
    disrupted_entity_name: Optional[str] = None
    filing_date: str
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
        nature = str(r.get("nature") or r.get("input_group") or r.get("group") or "default")

        # Share conversion
        share_val = 1.0
        if "share" in r and r["share"] is not None:
            share_val = float(r["share"])
        elif "exposure_pct" in r and r["exposure_pct"] is not None:
            share_val = float(r["exposure_pct"]) / 100.0
        elif "revenue_share_pct" in r and r["revenue_share_pct"] is not None:
            share_val = float(r["revenue_share_pct"]) / 100.0
        elif "loss_ratio" in r and r["loss_ratio"] is not None:
            val = float(r["loss_ratio"])
            share_val = val / 100.0 if val > 1.0 else val
        share_val = max(0.0, min(1.0, share_val))

        buffer_days = float(r.get("buffer_days", 0.0) or 0.0)

        # Hop distance of the target node
        hop_dist = distances.get(t, distances.get(s, 0) + 1)
        if hop_dist < 1:
            hop_dist = 1

        active_edges.append(
            TopologicalEdgeDescriptor(
                source=s,
                target=t,
                rel_type=rel_type,
                hop_distance=hop_dist,
                nature=nature,
                share=round(share_val, 4),
                buffer_days=buffer_days,
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

        if hasattr(summary, "contributing_edges") and summary.contributing_edges:
            first_c = summary.contributing_edges[0]
            input_grp = first_c.get("input_group", "default")
            share_pct = float(first_c.get("share_pct", 0.0))
            buf_days = float(first_c.get("buffer_days", 0.0))
        else:
            # Find incoming edge in active edges
            incoming = [e for e in active_edges if e.target == nid_clean]
            if incoming:
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
    lat_breakdown = getattr(result, "latency_breakdown_ms", {}) or {}
    telemetry = ExecutionTelemetry(
        task_c_cypher_ms=round(float(lat_breakdown.get("task_c_cypher_ms", 0.0)), 1),
        ast_guardrail_ms=round(float(lat_breakdown.get("guardrail_validation_ms", 0.0)), 1),
        graph_traversal_ms=round(float(lat_breakdown.get("graph_execution_ms", 0.0)), 1),
        cascade_propagation_ms=round(float(lat_breakdown.get("cascade_engine_ms", 0.0)), 1),
        task_d_contagion_ms=round(float(lat_breakdown.get("task_d_contagion_ms", 0.0)), 1),
        total_latency_ms=round(float(getattr(result, "total_latency_ms", 0.0) or 0.0), 1),
    )

    # 6. Model synthesis and think trace
    think_trace = getattr(result, "contagion_think_trace", None)
    operational_synthesis = getattr(result, "contagion_verdict", "") or "No operational synthesis generated."

    return OperationalContagionReport(
        report_id=report_id,
        event_id=event_id,
        filing_type="SEC Form 8-K",
        sec_item=sec_item_val,
        disrupted_entity_ticker=ticker_clean,
        disrupted_entity_cik=disrupted_entity_cik,
        disrupted_entity_name=disrupted_entity_name,
        filing_date=filing_date,
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
