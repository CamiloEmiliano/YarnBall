# ADR-00011: Operational Contagion Report Schema & Language Specification

<!--
STRICT FILENAME CONVENTION:
- Format: ADR-[5_DIGIT_ID]-[40_CHARACTER_SLUG].md
- Total Filename Length: Exactly 53 characters (including '.md')
- Slug Specification: Exactly 40 characters using lowercase alphanumeric and hyphens.
- Padding Rule: If the natural slug is under 40 characters, pad with trailing hyphens ('-').
Filename: ADR-00011-operational-contagion-report-schema-----.md (53 chars)
-->

- **ADR ID**: ADR-00011
- **Filename**: `ADR-00011-operational-contagion-report-schema-----.md`
- **Date**: 2026-11-11 23:30 UTC
- **Status**: ACCEPTED
- **Target Subsystem**: `retrieval` / `contagion` / `evaluation`
- **Linked Commits**: [`7e6aa39`](../../retrieval/reporting.py)

---

## 1. Context & Problem Statement

YarnBall's core mission is the computational reading of unstructured corporate regulatory disclosures (SEC Form 8-K, Form 10-K, Form 10-Q) to trace supply-chain contagion and operational disruption across networks of public companies.

Between ADR-00006 and ADR-00008, the platform developed:
1. A deterministic network diffusion algorithm ([`SparseNetworkCascadeEngine`](../../contagion/cascade.py#L228)).
2. An end-to-end GraphRAG pipeline ([`GraphRAGPipelineOrchestrator`](../../retrieval/pipeline.py#L187)).
3. A unified SIT verification suite ([`tests/integration/test_unified_sit_contract.py`](../../tests/integration/test_unified_sit_contract.py)).

However, the platform currently lacks a **formalized, standardized language and output specification** for the final artifact emitted by the computational reading pipeline: the **Operational Contagion Report**.

Currently:
- The orchestrator emits a raw Pydantic model (`PipelineResult`) containing disparate fields.
- There is no standardized vocabulary for regulatory disruption events, input dependency classifications, or operational downtime metrics.
- As discovered during the review of ADR-00009, attempting to jump directly to downstream trading strategies (e.g., "directional baskets", "short legs", "options hedges") conflated **Operational Contagion Intelligence** (Domain 1) with **Trading Strategy & Portfolio Execution** (Domain 2).
- Downstream algorithmic trading systems and quantitative analysts cannot reliably consume the output of YarnBall without a strict, machine-readable schema governed by a closed vocabulary.

This ADR formalizes the exact vocabulary, deterministic mathematical metrics, severity states, verbalization invariants, and output templates governing the **Operational Contagion Report**.

---

## 2. Decision Drivers & Constraints

1. **Strict Domain Boundary Isolation**:
   - The report language must exist entirely within **Domain 1: Operational & Physical Contagion**.
   - It must report objective, verifiable physical and operational facts: plant halts, component dependencies, safety stock depletion, unbuffered downtime, and capacity loss.
   - It must strictly exclude speculative trading recommendations (no stock picks, option strikes, price targets, or execution orders). Downstream strategy engines consume this report as an informational feature stream.
2. **Zero-Hallucination Quantitative Grounding**:
   - Language models cannot be trusted to perform financial arithmetic or network algebra.
   - All numbers appearing in the report (percentages, days, loss ratios) must originate deterministically from `SparseNetworkCascadeEngine`. The model is strictly restricted to qualitative extraction and explanatory verbalization.
3. **Controlled Categorical Vocabulary (Closed Enums)**:
   - External algorithms require exact string matches, not conversational adjectives.
   - Every categorical field in the report (event types, component nature, severity verdicts) must adhere to closed, validated enum definitions.
4. **Dual Presentation Contract**:
   - The specification must define both:
     - A machine-readable **JSON / Pydantic contract** (`OperationalContagionReport`).
     - A standardized, human-readable **ASCII/Markdown inspection report** for terminal logging and human audit.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Relying on Raw `PipelineResult` Output**:
  - Expose the raw internal dictionary from `GraphRAGPipelineOrchestrator.run()`.
  - *Rejection Rationale*: `PipelineResult` contains raw pipeline internals (e.g. `is_valid_query`, internal Cypher strings, raw unparsed edge dicts) without standardized formatting, section demarcations, or domain-validated event schemas.
- **Alternative B: Merging Trading Baskets into the Report**:
  - Including `SHORT_BASKET` and `LONG_BASKET` sections directly in the contagion report.
  - *Rejection Rationale*: Premature and violates domain boundaries. Operational disruption is a structural reality; trading baskets are subjective portfolio decisions dependent on external factors (liquidity, market sentiment, risk limits) that YarnBall does not track.

---

## 4. Proposed Architecture: Language & Schema Formalization

### 4.1 Controlled Entity & Event Taxonomy

The SEC created Form 8-K under Section 13 or 15(d) of the Securities Exchange Act of 1934 to mandate public disclosure of *all* material corporate events ensuring fair and reliable markets. We cannot afford to ignore or cherry-pick statutory items.

Therefore, the event taxonomy is strictly partitioned into:
1. **The Complete Statutory SEC Form 8-K Item Canon (`SECForm8KItem`)**: An exhaustive, 1-to-1 mapping across all 9 statutory sections of Form 8-K.
2. **The Operational Disruption Mechanism (`OperationalDisruptionCategory`)**: The physical or commercial mechanism through which the event transmits operational risk through the network.
3. **Dependency Domain & Operational Scope**: Distinguishing widget manufacturing from digital infrastructure (e.g. Salesforce SaaS) and current operations from forward growth capex.

#### A. Exhaustive Statutory SEC Form 8-K Canon

- **Section 1: Registrant's Business and Operations**
  - `Item 1.01`: Entry into a Material Definitive Agreement
  - `Item 1.02`: Termination of a Material Definitive Agreement
  - `Item 1.03`: Bankruptcy or Receivership
  - `Item 1.04`: Mine Safety - Reporting of Shutdowns & Violations
  - `Item 1.05`: Material Cybersecurity Incidents
- **Section 2: Financial Information**
  - `Item 2.01`: Completion of Acquisition or Disposition of Assets
  - `Item 2.02`: Results of Operations and Financial Condition
  - `Item 2.03`: Creation of a Direct Financial Obligation
  - `Item 2.04`: Acceleration of Obligations (Debt/Credit Default)
  - `Item 2.05`: Costs Associated with Exit/Disposal (Plant Closures)
  - `Item 2.06`: Material Impairments (Asset/Plant Write-Offs)
- **Section 3: Securities and Trading Markets**
  - `Item 3.01`: Delisting Notice / Failure to Satisfy Listing Rule
  - `Item 3.02`: Unregistered Sales of Equity Securities
  - `Item 3.03`: Material Modification to Rights of Security Holders
- **Section 4: Accountants and Financial Statements**
  - `Item 4.01`: Changes in Registrant's Certifying Accountant
  - `Item 4.02`: Non-Reliance on Previously Issued Financial Statements (Restatements)
- **Section 5: Corporate Governance and Management**
  - `Item 5.01`: Changes in Control of Registrant
  - `Item 5.02`: Departure/Election of Directors or Officers (CEO/CFO Firing/Replacement)
  - `Item 5.03`: Amendments to Articles of Incorporation/Bylaws
  - `Item 5.04`: Temporary Suspension of Trading Under Benefit Plans
  - `Item 5.05`: Amendments/Waiver to Code of Ethics
  - `Item 5.06`: Change in Shell Company Status
  - `Item 5.07`: Submission of Matters to a Vote of Security Holders
  - `Item 5.08`: Shareholder Director Nominations
- **Section 6: Asset-Backed Securities**
  - `Items 6.01 – 6.05`: ABS Informational, Servicer, Credit Support, & Distribution Disclosures
- **Section 7: Regulation FD**
  - `Item 7.01`: Regulation FD Disclosures
- **Section 8: Other Material Events**
  - `Item 8.01`: Statutory General Catch-All for Material Events
- **Section 9: Financial Statements and Exhibits**
  - `Item 9.01`: Financial Statements and Exhibits

#### B. Operational Disruption Mechanisms

- **Physical Manufacturing & Fab Outages (primarily Item 8.01)**:
  - `FACILITY_FIRE_OR_EXPLOSION`, `NATURAL_DISASTER_FORCE_MAJEURE`, `UTILITY_POWER_GRID_FAILURE`, `EQUIPMENT_BREAKDOWN_FAB_HALT`
- **Commercial & Contractual Severance (primarily Items 1.01, 1.02, 1.03, 2.04, 2.05)**:
  - `PRIMARY_SUPPLY_CONTRACT_TERMINATED`, `COMMERCIAL_INSOLVENCY_BANKRUPTCY`, `DEBT_COVENANT_DEFAULT_CREDIT_FREEZE`, `PERMANENT_FACILITY_CLOSURE_EXIT`
- **Human Capital & Security (primarily Items 1.05, 5.02, 8.01)**:
  - `LABOR_UNION_STRIKE_WALKOUT`, `RANSOMWARE_CYBER_INFRASTRUCTURE_HALT`, `KEY_EXECUTIVE_CRISIS_REPLACEMENT`
- **Logistics, Legal & Regulatory Interventions (primarily Items 4.02, 8.01)**:
  - `MARITIME_CANAL_PORT_LOGISTICS_CHOKEPOINT`, `GOVERNMENT_EXPORT_BAN_OR_SANCTION`, `PATENT_IP_INJUNCTION_HALT`, `REGULATORY_SAFETY_RECALL_OR_FDA_HOLD`, `ACCOUNTING_FRAUD_FINANCIAL_RESTATEMENT`
- **Allocation & Unclassified Catch-All**:
  - `VENDOR_COMPONENT_ALLOCATION_CURTAILMENT`, `OTHER_UNCLASSIFIED_MATERIAL_HALT`

#### C. Dependency Domains & Operational Scope (Handling Non-Widget / SaaS Firms)

To prevent misapplying physical manufacturing concepts onto pure software or service firms (e.g. Salesforce), graph edges specify:

1. **Dependency Domain**:
   - `PHYSICAL_COMPONENT`: Hardware manufacturing (buffer = warehouse safety stock days).
   - `DIGITAL_INFRASTRUCTURE`: Cloud hosting, compute reservations, API services (buffer = contracted reservation term / multi-region failover RTO).
   - `COMMERCIAL_ENTERPRISE`: Customer/vendor contracts (buffer = contracted subscription term / deferred revenue days).
2. **Operational Scope**:
   - `CORE_OPERATIONAL_DELIVERY`: Required for current revenue-generating service operations.
   - `GROWTH_CAPEX_AND_RD`: Required for future product R&D or expansion (e.g. GPU scarcity affecting future AI models, but zero impact on current CRM delivery). Shocks to this scope do not produce operational downtime ($D_v = 0$).

#### D. Component Input Groups: Open-World Normalized Descriptors

Input groups are formalized as **open-world normalized string identifiers** (`input_group: str`) extracted dynamically from SEC filings:
- **Normalization Invariant**: Lowercase snake_case, stripped of part numbers and vendor brand names (`Cirrus CS42L42 Audio Processing IC` $\to$ `audio_codecs`).
- **Leontief Bottleneck Property**: The cascade engine aggregates procurement shares $a_{uv}$ strictly within matching `input_group` tags:
  $$\sum_{u \in \text{Inputs}_k(v)} a_{uv} \le 1.0 \quad \forall k \in \mathcal{K}_v$$
- **Cross-Industry Examples**:
  - *Electronics*: `foundry`, `audio_codecs`, `memory`, `power_management`, `assembly`, `packaging`, `substrates`
  - *Aerospace*: `avionics`, `propulsion_turbines`, `titanium_forgings`, `fuselage_structures`, `hydraulic_actuators`
  - *Automotive*: `battery_cells`, `chassis_stampings`, `wiring_harnesses`, `lidar_sensors`, `transmission_gears`
  - *Biopharma*: `active_pharmaceutical_ingredients`, `sterile_vials`, `cold_chain_logistics`, `bioreactor_media`
  - *Fallback Group*: Defaults to `"default"` when specific sub-components are undisclosed.

---

### 4.2 Deterministic Mathematical Metrics

The report presents exact metrics computed by the cascade engine under the following formal mathematical definitions:

1. **Procurement Share ($a_{uv}$)**:
   $$a_{uv} \in [0.0, 1.0]$$
   The proportion of node $v$'s component input group supplied by node $u$, where $\sum_{u \in \text{group}} a_{uv} \le 1.0$.

2. **Inventory Safety Stock / Buffer ($B_v$)**:
   $$B_v \ge 0.0 \quad \text{(Calendar Days)}$$
   The Time-to-Survive (TTS) duration that node $v$ can maintain normal operations during a complete supply halt without incurring downstream production curtailment.

3. **Peak Operational Contagion Shock ($h_v$)**:
   $$h_v = \max_{k \in \mathcal{K}_v} \left( \sum_{u \in \text{Inputs}_k(v)} a_{uv} \cdot h_u \right) \in [0.0, 1.0]$$
   The Leontief bottleneck capacity contraction across all essential input groups $\mathcal{K}_v$.

4. **Shock Onset Delay ($t_v^{\text{onset}}$)**:
   $$t_v^{\text{onset}} = \begin{cases} t_u^{\text{onset}} + B_v, & \text{if } h_u \cdot D_u > B_v \\ \infty \text{ (None)}, & \text{if } h_u \cdot D_u \le B_v \end{cases}$$
   The calendar day on which node $v$'s inventory buffer is completely exhausted and unbuffered production downtime begins.

5. **Unbuffered Operational Downtime ($D_v^{\text{unbuffered}}$)**:
   $$D_v^{\text{unbuffered}} = \max\left(0.0, \, D_{\text{transmitted}} - B_v\right) \quad \text{(Calendar Days)}$$
   The net duration of active operational shutdown experienced by node $v$ after buffer absorption.

6. **Lost Output Capacity-Days ($L_v$)**:
   $$L_v = h_v \cdot D_v^{\text{unbuffered}} \quad \text{(Capacity-Days)}$$
   The integral volume of lost production capacity over the unbuffered downtime window.

7. **Network Spectral Radius ($\rho(A)$)**:
   $$\rho(A) = \max_i |\lambda_i(A)|$$
   The spectral radius of the weighted procurement matrix. When $\rho(A) < 1.0$, the supply network is provably stable under feedback cycle dampening.

---

### 4.3 Standardized Contagion Severity Verdicts (Zero Magic Numbers)

Rather than enforcing arbitrary percentage thresholds (e.g., a magic 25% cutoff), the operational state of each company in the network is governed strictly by the physical boundary condition of whether unbuffered downtime was transmitted:

| Verdict | Boundary Condition | Physical / Operational State |
| :--- | :--- | :--- |
| **`UNBUFFERED_DISRUPTION`** | $D_v^{\text{unbuffered}} > 0.0$ | Inventory safety stock or compute redundancy was exhausted; active operational downtime was transmitted. Continuous metrics ($h_v$, $D_v$, $L_v$) report exact severity. |
| **`BUFFER_ABSORBED`** | $D_v^{\text{unbuffered}} = 0.0$ **and** $h_u > 0.0$ | Upstream shock occurred, but downstream safety stock or high-availability failover absorbed 100% of the deficit. Zero downtime transmitted. |
| **`UNAFFECTED`** | $h_v = 0.0$ **and** $D_v^{\text{unbuffered}} = 0.0$ | No topological supply path from the shock origin, or component dependency is inactive/unrelated. |

All quantitative severity nuance is conveyed through the continuous metrics ($h_v$, $D_v$, $L_v$) without subjective categorization.

---

### 4.4 Model Reasoning Verbalization Invariants

When synthesizing the textual narrative in the report, `yarnball-qwen:7b` must strictly adhere to the following behavioral guardrails:

1. **Mandatory Chain-of-Thought `<think>` Demuxing**:
   - The model must output an explicit `<think>...</think>` block preceding its synthesis.
   - The trace must articulate: (a) origin shock parameters, (b) intermediate node buffer durations, (c) net unbuffered downtime calculations, and (d) downstream propagation dampening.
2. **Deterministic Metric Verbatim Citation**:
   - The model is strictly prohibited from estimating, rounding, or hallucinating numerical values.
   - All percentages and day counts in the prose must match the pre-computed metrics in `ConvergentShockSummary` verbatim.
3. **Prohibition of Epistemic Hedging**:
   - Phrases indicating conversational vagueness (e.g., *"might face headwinds"*, *"could potentially suffer down the line"*, *"investors should be cautious"*) are rejected by the output validator.
   - Language must be direct, clinical, and grounded in network mechanics.

---

### 4.5 Standardized Report Schema & Template

#### A. Structured Report Contract Specification

The report contract is implemented as a validated Pydantic model (`OperationalContagionReport` in `retrieval/reporting.py`) with the following required fields:

| Section | Field Name | Data Type | Description |
| :--- | :--- | :--- | :--- |
| **Event Identity** | `report_id` | `str` | Unique report identifier (`REP-YYYYMMDD-<TICKER>-<SEQ>`) |
| | `event_id` | `str` | Upstream filing event identifier (e.g. `SEC_8K_20241028_CRUS_FIRE`) |
| | `filing_type` | `str` | Regulatory disclosure form (`Form 8-K`) |
| | `sec_item` | `str` | Statutory SEC Item (e.g., `Item 8.01`) |
| | `disrupted_entity_ticker` | `str` | Exchange ticker of shock origin (`CRUS`) |
| | `disrupted_entity_cik` | `str` | SEC Central Index Key (`0000772406`) |
| | `filing_date` | `str` | Publication date in `YYYY-MM-DD` UTC |
| | `disruption_category` | `str` | Operational mechanism (e.g. `FACILITY_FIRE_OR_EXPLOSION`) |
| | `event_summary` | `str` | Factual regulatory legal text summary |
| | `shock_duration_days` | `float` | Event duration $D_0$ in calendar days |
| | `shock_magnitude_pct` | `float` | Facility/supply contraction $h_0$ in percent |
| **Network Impacts** | `network_impacts` | `List[NetworkNodeImpact]` | List of evaluated downstream nodes (hop, input group, share, buffer, onset, downtime, capacity loss, verdict) |
| | `spectral_radius` | `float` | Network matrix spectral radius $\rho(A)$ |
| | `is_acyclic_dag` | `bool` | True if network topology is provably acyclic DAG |
| **Graph Topology** | `active_topology_edges`| `List[TopologicalEdgeDescriptor]`| Verified active supply-chain edges traversed |
| **Model Synthesis** | `contagion_think_trace`| `str` | Demuxed Chain-of-Thought `<think>` reasoning block |
| | `operational_synthesis`| `str` | Direct, clinical operational impact summary |
| **Telemetry** | `telemetry` | `ExecutionTelemetry` | Per-stage latency breakdown (Task C, AST, traversal, cascade, Task D) and total elapsed ms |

#### B. Standardized Visual Inspection Template

```markdown
================================================================================
YARNBALL OPERATIONAL CONTAGION REPORT
================================================================================
REPORT ID:          REP-20241028-CRUS-001
EVENT ID:           SEC_8K_20241028_CRUS_FIRE
FILING FORM:        SEC Form 8-K
SEC STATUTORY ITEM: Item 8.01 (Other Events)
FILING DATE:        2024-10-28
DISRUPTED ENTITY:   Cirrus Logic Inc. (CRUS | CIK: 0000772406)
DISRUPTION TYPE:    FACILITY_FIRE_OR_EXPLOSION
SHOCK SPECIFICATION: 100.0% plant halt for 60.0 calendar days

--------------------------------------------------------------------------------
1. DETERMINISTIC CASCADE SHOCK METRICS (SparseNetworkCascadeEngine)
--------------------------------------------------------------------------------
Entity | Hop | Input Group   | Share  | Buffer | Onset  | Downtime | Contraction | Verdict
-------|-----|---------------|--------|--------|--------|----------|-------------|----------------------------
AAPL   | 1   | audio_codecs  |  20.0% |  45.0d | Day 45 |    15.0d |      -20.0% | UNBUFFERED_DISRUPTION
HPQ    | 1   | audio_codecs  |  10.0% |  90.0d | None   |     0.0d |        0.0% | BUFFER_ABSORBED
FOXCONN| 2   | assembly      |  50.0% |  90.0d | None   |     0.0d |        0.0% | BUFFER_ABSORBED

Spectral Radius: 0.0000 (Topology is provably acyclic DAG)

--------------------------------------------------------------------------------
2. ACTIVE KNOWLEDGE GRAPH TOPOLOGY
--------------------------------------------------------------------------------
[Hop Distance 1]
- (CRUS) -[:SUPPLIES_TO {nature: 'audio_codecs', share: 0.20, buffer_days: 45.0}]-> (AAPL)
- (CRUS) -[:SUPPLIES_TO {nature: 'audio_codecs', share: 0.10, buffer_days: 90.0}]-> (HPQ)

[Hop Distance 2]
- (AAPL) -[:SUPPLIES_TO {nature: 'assembly', share: 0.50, buffer_days: 90.0}]-> (FOXCONN)

--------------------------------------------------------------------------------
3. COMPUTATIONAL READING & REASONING (yarnball-qwen:7b)
--------------------------------------------------------------------------------
<think>
Analyzing Cirrus Logic (CRUS) audio codec fab halt of 60 days. Apple maintains a 45-day
buffer stock. The buffer delays onset to Day 45 and absorbs 45 days of shock,
leaving 15 unbuffered downtime days with a 20.0% capacity contraction. Foxconn
maintains a 90-day assembly buffer, which fully absorbs Apple's 15-day deficit.
HP maintains a 90-day audio component buffer, completely absorbing the disruption.
</think>

Operational Synthesis:
Apple Inc. (AAPL) experiences UNBUFFERED_DISRUPTION resulting from a 20.0%
peak capacity contraction across its audio component lines with 15.0 days of
unbuffered downtime post-buffer absorption. Foxconn (FOXCONN) and Hewlett-Packard
(HPQ) are classified as BUFFER_ABSORBED, as their 90-day inventory safety stocks
fully absorb operational transmission with zero downstream downtime.

--------------------------------------------------------------------------------
4. PIPELINE TELEMETRY
--------------------------------------------------------------------------------
Cypher Retrieval (Task C):     12.4 ms
AST Guardrail Check:            1.1 ms
Graph Traversal:                4.2 ms
Cascade Propagation:            2.8 ms
Model Contagion Reasoning:    145.0 ms
Total Pipeline Latency:       165.5 ms
================================================================================
```

---

### 4.6 Concrete Deliverables

1. **Schema Module**: Implement `retrieval/reporting.py` defining:
   - `DisruptionEventType`, `NetworkNodeImpact`, and `OperationalContagionReport` Pydantic models.
   - `format_operational_contagion_report(report: OperationalContagionReport) -> str` formatting engine.
2. **Pipeline Integration**:
   - Update `GraphRAGPipelineOrchestrator` to generate `OperationalContagionReport` directly on request or as part of `PipelineResult`.
3. **Automated Unit & Contract Tests**:
   - Author `tests/retrieval/test_operational_contagion_reporting.py` verifying schema validation, metric citation fidelity, and template rendering.

---

## 5. Invariants & Guardrails

1. **Domain Isolation Invariant**:
   - `OperationalContagionReport` must contain zero fields referencing trading actions, order recommendations, options, or market prices.
2. **Exact Numerical Invariance**:
   - `report.network_impacts[i].unbuffered_downtime_days` must equal `convergent_shock_summary.downtime_days` within float precision $\pm 10^{-6}$.
3. **Buffer Absorption Invariant**:
   - If a node has $B_v \ge h_u D_u$, its verdict must strictly be `TOLERATED_BUFFERED` or `NEUTRAL_UNAFFECTED`, with $D_v^{\text{unbuffered}} = 0.0$.
4. **Verbatim Textual Citation**:
   - The prose in `operational_synthesis` must contain the exact string representations of the numbers in `network_impacts` (e.g., `"20.0%"` and `"15.0 days"`).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Establishes a permanent, immutable language and contract for all operational reports generated by YarnBall.
  - Enables clean decoupling: YarnBall produces `OperationalContagionReport`, and any downstream trading algorithm can consume it via a standard Pydantic parser without brittle text scraping.
  - Unblocks ADR-00009: The SEC 8-K dry run now has an explicit, agreed-upon target artifact to produce and verify.
- **Negative / Trade-offs**:
  - Requires maintaining the reporting formatter and updating test assertions if report section headers evolve.

