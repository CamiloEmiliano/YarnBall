# ADR-00014: SEC 8-K Boilerplate Sanitization & Disruption Gating

- **ADR ID**: ADR-00014
- **Filename**: `ADR-00014-sec-8k-boilerplate-sanitization-gating--.md`
- **Date**: 2026-11-13 06:55 UTC
- **Status**: ACCEPTED
- **Target Subsystem**: `ingestion` / `retrieval` / `tools`
- **Linked Commits**: [`ingestion/sec/disruption_gating.py`](../../ingestion/sec/disruption_gating.py), [`tools/dry_run_sec_8k_contagion.py`](../../tools/dry_run_sec_8k_contagion.py)

---

## 1. Context & Problem Statement

In [`tools/dry_run_sec_8k_contagion.py`](../../tools/dry_run_sec_8k_contagion.py), SEC Form 8-K disruption classification currently relies on flat substring matching (`_infer_disruption_category`) directly over raw `(headline + content)` strings and naïve list-order item extraction (`_extract_sec_item`). While functional on synthetic test fixtures, ingesting authentic 2018–2026 Form 8-K disclosures via [`EdgarClient`](../../ingestion/sec/client.py) exposes three immediate failure modes:

1. **PSLRA Safe Harbor Boilerplate False Positives**:
   Real Form 8-K disclosures systematically contain boilerplate *Forward-Looking Statements* / Safe Harbor disclaimers enumerating generic catastrophe risks (*"...including fires, earthquakes, cyberattacks, ransomware, power outages, strikes, and supply chain disruptions"*). Flat substring checks (`"fire" in text`, `"cyber" in text`) misclassify routine filings as acute physical or cyber shocks.
2. **Cover Page, Signature, and Exhibit Contamination**:
   Raw EDGAR filings embed standardized checkbox cover pages, Exchange Act signature blocks, and Item 9.01 exhibit titles. Unsanitized extraction mistakes exhibit titles or generic legal boilerplate (`"terminate"`, `"power"`) for material operational events.
3. **Naïve Multi-Item Selection & Synthetic Default Escalation**:
   When a Form 8-K discloses multiple items (e.g., `["Item 9.01", "Item 1.05"]`), iterating linearly selects the first list match rather than the highest-precedence operational item. Furthermore, when a routine filing (`Item 5.02` director change or `Item 2.02` earnings release) lacks explicit shock metadata, default fallbacks (`100.0%` capacity halt for `60.0` days) risk fabricating synthetic cascades instead of passing a clean zero-shock negative control.

---

## 2. Decision Drivers & Constraints

1. **Zero False Positives from Safe Harbor Boilerplate**:
   - Risk factor enumerations inside PSLRA Forward-Looking Statement disclaimers must be stripped prior to keyword or statutory disruption classification.
2. **Deterministic Statutory Item Precedence**:
   - Primary item resolution must rank operational/structural disruption items above general catch-alls (`Item 8.01`) and procedural sections (`Item 9.01`).
3. **Explicit Non-Disruption Negative Control Path**:
   - Routine statutory disclosures with no substantive operational disruption evidence must gate cleanly to `is_material_operational_disruption = False` with $h_0 = 0.0, D_0 = 0.0$, preserving valid negative controls for downstream time-series joins ([`ADR-00013`](ADR-00013-contagion-event-time-series-join-schema-.md)).
4. **Zero-I/O, Linear-Time Pure Python Determinism**:
   - Sanitization and gating must execute statelessly in-memory with zero external network calls or extra LLM round-trips, using module-level pre-compiled, $O(N)$ single-pass string/regex rules (no catastrophic backtracking) over bounded filing text rather than brittle wall-clock SLAs.
5. **Strict Backward Compatibility**:
   - Canonical test fixtures (`tests/fixtures/filings/aapl_8k_disruption.json`, `crus_8k_disruption.json`) must retain identical operational classifications.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Delegate Boilerplate Filtering Solely to the LLM Prompt**:
  - *Rationale for Rejection*: Consumes valuable context budget, adds inference latency before cascade calculation even starts, and fails when operating deterministically in offline/mock batch evaluation.
- **Alternative B: Restrict Ingestion Exclusively to `Item 1.05` and `Item 8.01`**:
  - *Rationale for Rejection*: Material supply disruptions are legally reported across multiple statutory codes, including `Item 1.02` (Supply Agreement Termination), `Item 1.03` (Bankruptcy), `Item 1.04` (Mine Shutdowns), `Item 2.05` (Plant Closures), and `Item 2.06` (Facility Impairments).
- **Alternative C: Retain Unsanitized Substring Matching**:
  - *Rationale for Rejection*: Contaminates downstream empirical cohorts with high false-positive rates whenever routine corporate disclosures carry standard Safe Harbor legal text.

---

## 4. Decision Outcome

Introduce a dedicated pure-Python sanitization and gating module [`ingestion/sec/disruption_gating.py`](../../ingestion/sec/disruption_gating.py) invoked by [`tools/dry_run_sec_8k_contagion.py`](../../tools/dry_run_sec_8k_contagion.py) and downstream SEC harvesters.

### 4.1 Paragraph-Scoped Boilerplate Sanitizer (`sanitize_sec_8k_text`)
Splits raw filing text on paragraph boundaries and applies deterministic structural stripping:
1. **Cover Page & Metadata Header Pruning**: Removes standard EDGAR cover-page boilerplate (`"CHECK THE APPROPRIATE BOX"`, `"COMMISSION FILE NUMBER"`, `"EMERGING GROWTH COMPANY"`).
2. **PSLRA Safe Harbor Removal**: Identifies and drops standalone disclaimer paragraphs/sections matching statutory Safe Harbor signatures (`"private securities litigation reform act"`, `"forward-looking statements"`, `"cautionary statement regarding forward-looking"`, `"actual results may differ materially"`).
3. **Trailing Signature & Exhibit Pruning**: Truncates boilerplate after standard closing signatures (`"pursuant to the requirements of the securities exchange act of 1934"`, `"exhibits and financial statement schedules"`).

### 4.2 Statutory Item Precedence Hierarchy (`select_primary_sec_8k_item`)
Given a list of Form 8-K items (e.g., from `filing.items`), assigns deterministic priority tiers rather than trusting raw list order:

| Priority Tier | Statutory Items | Rationale |
| :--- | :--- | :--- |
| **Tier 1 (Acute Disruption)** | `Item 1.05`, `Item 1.04`, `Item 1.03`, `Item 1.02`, `Item 2.05`, `Item 2.06`, `Item 2.04` | Direct statutory disclosures of cyber incidents, shutdowns, insolvencies, contract severances, or plant exits |
| **Tier 2 (Event / Agreement)** | `Item 1.01`, `Item 8.01` | Material supply agreements or general operational event catch-all (fires, weather, force majeure) |
| **Tier 3 (Financial / FD)** | `Item 2.01`, `Item 2.02`, `Item 2.03`, `Item 7.01` | Financial reporting or M&A disclosures (non-disruptive unless sanitized prose confirms an operational halt) |
| **Tier 4 (Governance / Procedural)** | `Item 3.01`–`Item 6.05`, `Item 9.01` | Governance changes, executive departures, and exhibit indices |

### 4.3 Anchored Disruption Gating (`assess_sec_8k_disruption`)
Produces a structured Pydantic verdict (`SanitizedDisruptionAssessment`):
- **Statutory Anchoring**: Tier 1 items anchor directly when appropriate (e.g. `Item 1.05` $\to$ `RANSOMWARE_CYBER_INFRASTRUCTURE_HALT`; `Item 1.02` + supply/vendor context $\to$ `PRIMARY_SUPPLY_CONTRACT_TERMINATED`; `Item 1.03` $\to$ `COMMERCIAL_INSOLVENCY_BANKRUPTCY`).
- **Sanitized Lexical Classification**: Keyword patterns for physical hazards (fire, explosion, flood, power grid, fab halt, maritime port blockage, regulatory recall) execute strictly against `sanitized_headline + sanitized_content`.
- **Negative-Control Shock Gating**:
  - When `primary_sec_item` is Tier 3 or Tier 4 and sanitized narrative contains zero explicit physical/supply disruption triggers (and metadata provides no explicit shock parameters), `is_material_operational_disruption` resolves to `False`.
  - Gated non-disruptions override any default dry-run magnitude/duration fallbacks to $h_0 = 0.0\%$ and $D_0 = 0.0\text{ days}$, ensuring downstream cascade propagation yields `UNAFFECTED` cleanly.

---

## 5. Invariants & Guardrails

1. **Safe Harbor Immunity Invariant**:
   - Disruption keywords appearing exclusively inside stripped Safe Harbor or Exchange Act signature sections MUST never influence `disruption_category` or trigger `is_material_operational_disruption = True`.
2. **Zero Synthetic Fallback Fabrication**:
   - A routine non-disruptive Form 8-K (`is_material_operational_disruption == False`) MUST emit `shock_magnitude_pct = 0.0` and `shock_duration_days = 0.0` rather than inheriting mock default assumptions (`100%` / `60d`).
3. **Pure Deterministic Execution**:
   - Sanitization and gating operate deterministically on in-memory strings with zero external network calls or stochastic LLM sampling.
4. **Zero Absolute Paths Invariant**:
   - All module and ADR links remain strictly repository-relative (`../../<path>`).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Eliminates false-positive supply shocks caused by standard corporate legal disclaimers in real 2018–2026 EDGAR filings.
  - Preserves clean zero-shock negative control rows in `--export-jsonl` datasets ([`ADR-00013`](ADR-00013-contagion-event-time-series-join-schema-.md)) for unbiased empirical event studies.
- **Negative / Trade-offs**:
  - Paragraph-level PSLRA disclaimer removal relies on clear paragraph segmentation in EDGAR text; if an unformatted plain-text filing merges an operational disclosure directly into a single monolithic paragraph with a Safe Harbor disclaimer, keyword extraction must be protected by conservative inline boundary checks rather than discarding the entire filing.

