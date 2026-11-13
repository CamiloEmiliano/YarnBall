# ADR-00018: Workbench UI Shell & Form 8-K Event Audit View

- **ADR ID**: ADR-00018
- **Filename**: `ADR-00018-workbench-ui-shell-and-8k-audit-view----.md`
- **Date**: 2026-11-13 19:51 UTC
- **Status**: ACCEPTED
- **Target Subsystem**: `explorer` / `tests`
- **Linked Commits**: [`explorer/workbench_api.py`](../../explorer/workbench_api.py), [`explorer/static/index.html`](../../explorer/static/index.html), [`explorer/static/styles.css`](../../explorer/static/styles.css), [`explorer/static/app.js`](../../explorer/static/app.js)

---

## 1. Context & Problem Statement

[`ADR-00017`](ADR-00017-local-fastapi-kg-contagion-workbench-app.md) established the local FastAPI JSON backend (`GET /api/catalog`, `GET /api/subgraph`, `POST /api/evaluate-8k`, `POST /api/cohort`) served via [`tools/serve_explorer.py`](../../tools/serve_explorer.py) at `127.0.0.1:8050`. However, `GET /` currently serves only an unstyled placeholder HTML readiness page linking to `/docs`.

To make YarnBall's deterministic preprocessing, cascade mathematics, and verbatim citation audits human-inspectable in the browser without coupling every future screen into one monolithic PR, the frontend is introduced incrementally:
- **ADR-00018 (this ADR)**: Defines the overall single-page workbench chrome (header, mode/health status, navigation tabs), visual styling conventions, XSS-safe DOM rendering rules, and the complete **Event Audit (8-K -> Contagion)** view backed by `GET /api/catalog` and `POST /api/evaluate-8k`.
- **Follow-up ADR-00019 (deferred)**: Will implement the **PiT KG Explorer** (`/api/subgraph`) and **Cohort Browser** (`/api/cohort`) tab contents inside the shell established here.

To enable human visual review *before* ADR acceptance, this ADR adopts an interactive HTML prototype reviewed in-IDE as its reference layout target.

---

## 2. Decision Drivers & Constraints

1. **Pre-Acceptance Visual Verification**:
   - UI aesthetics, panel hierarchy, Gantt buffer-vs-downtime legibility, and density are reviewed via a self-contained IDE preview artifact before writing production static files or marking this ADR `ACCEPTED`.
2. **Zero Runtime CDN & Zero Node Build Toolchain**:
   - Production frontend assets (`explorer/static/`) MUST execute strictly offline inside `127.0.0.1:8050` with **zero external CDN scripts, fonts, or stylesheets**, zero Node/`npm` bundling, and zero runtime external network calls.
3. **Strict XSS-Safe Read-Only Rendering**:
   - SEC Form 8-K narrative text, entity names, reasoning traces, and synthesis strings MUST be inserted exclusively via DOM text APIs (`textContent`)—never raw HTML interpolation (`innerHTML`)—preventing script injection from arbitrary filing inputs.
4. **Evidence-Grounded Inspection (No Hypothetical Stress Sliders)**:
   - The Event Audit view evaluates either repository Form 8-K fixtures or pasted inline Form 8-K payloads (respecting [`assess_sec_8k_disruption`](../../ingestion/sec/disruption_gating.py) negative-control gating), alongside optional allowlisted snapshot/manifest selection—without ungrounded synthetic shock-slider widgets.
5. **Scope Isolation**:
   - API schemas in [`explorer/workbench_api.py`](../../explorer/workbench_api.py) remain unchanged; only static asset serving (`GET /` and `/static/*`) is added on the server side.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: CDN-Loaded Tailwind / React / Chart Libraries at Runtime**:
  - *Rationale for Rejection*: Violates strict air-gapped / offline reproducibility (`127.0.0.1:8050` must render identically with no internet access) and introduces dependency drift removed in [`ADR-00016`](ADR-00016-decommission-legacy-chainlit-ui-frontend.md).
- **Alternative B: Server-Rendered Jinja2 Templates + Page Reloads**:
  - *Rationale for Rejection*: Full-page reloads reset inline 8-K drafts, snapshot selections, and filter state while requiring extra template dependency wiring instead of consuming ADR-00017's clean JSON contracts directly.
- **Alternative C: Build All Three Workbench Views in ADR-00018**:
  - *Rationale for Rejection*: Couples SVG subgraph force/layer geometry (`PiT KG Explorer`) and 1NF pagination tables (`Cohort Browser`) with Event Audit & shell styling in an oversized diff.

---

## 4. Decision Outcome

Introduce static, zero-build frontend assets under `explorer/static/` and mount them cleanly in [`explorer/workbench_api.py`](../../explorer/workbench_api.py):

### 4.1 Static Serving Wiring (`explorer/workbench_api.py`)
- Serve `explorer/static/index.html` cleanly at `GET /` and mount `explorer/static/` at `/static` (`styles.css`, `app.js`).
- Preserve `/docs`, `/api/catalog`, `/api/subgraph`, `/api/evaluate-8k`, and `/api/cohort` contracts unchanged.

### 4.2 Workbench Visual Architecture & Navigation Shell
- **Design Language**: High-clarity analytical workspace using native CSS custom properties with automatic light/dark adaptation (`prefers-color-scheme`) plus an explicit UI theme toggle (`Auto / Dark / Light`) and compact tabular numerals (`font-variant-numeric: tabular-nums`) for all financial days, procurement shares, and spectral radius values.
- **Top Application Bar**:
  - Brand mark (`YarnBall`) + subtitle (`Audit-First Financial GraphRAG Workbench`).
  - Live **Ollama Health Pill** (`GET /api/catalog` `ollama.healthy`) and explicit execution mode selector (`MOCK_DETERMINISTIC` default; `LIVE_LLM` enabled cleanly when Ollama is reachable).
  - Read-only / local confinement indicator badge (`127.0.0.1` local mode).
- **Tab Bar**:
  - `Event Audit (8-K -> Contagion)` (active in ADR-00018).
  - `PiT KG Explorer` and `Cohort Browser` rendered as structured preview tabs with explicit `Coming in ADR-00019` notices so shell ergonomics are testable end-to-end without stubbing half-finished visualizations.

### 4.3 Event Audit View (`Event Audit (8-K -> Contagion)`)
Two-column responsive layout (configuration rail on left; three stacked audit cards on right):

1. **Input & Context Panel (Left Rail)**:
   - **Source Mode Switch**: `Repository Fixture` (populated from `GET /api/catalog` `filings`) vs `Inline 8-K Payload` (`ticker`, `company_name`, `filing_date`, `items`, `headline`, `content`, optional metadata JSON).
   - **Graph Topology Selector**: `Built-in Canonical Supply Net` (CLI parity default) or any discovered `snapshot_id` / `edges_file` from `/api/catalog` + optional `target_ticker` focus.
   - **Evaluate Action**: Submits `POST /api/evaluate-8k` with loading state and explicit error surface.
2. **Card 1 — Boilerplate Sanitization & Disruption Gating (`SanitizedDisruptionAssessment`)**:
   - Displays `primary_sec_item`, `item_priority_tier` badge, `gating_reason`, `stripped_paragraph_count`, `shock_magnitude_pct`, and `shock_duration_days`.
   - When `is_material_operational_disruption == false` (negative control), prominently renders a zero-shock guardrail banner explaining why cascade propagation remained quiescent.
   - Interactive `Sanitized vs. Raw Input` toggle showing exact boilerplate paragraphs stripped (PSLRA Safe Harbor, signatures, exhibit indices) via safe `textContent` rendering.
3. **Card 2 — Downstream Cascade Impact & Buffer-vs-Outage Gantt Timeline**:
   - Topology health summary strip: `is_acyclic_dag`, `spectral_radius`, `disrupted_entity_ticker`, `disruption_category`.
   - **Interactive SVG Gantt Timeline** per impacted `NetworkNodeImpact`:
     - Translucent/neutral bar from Day `0` to `min(shock_duration_days, inventory_buffer_days)` representing absorbed inventory buffer grace window.
     - Accent/alert bar from `shock_onset_day` to `shock_onset_day + unbuffered_downtime_days` representing `UNBUFFERED_DISRUPTION` downtime, annotated with `capacity_contraction_pct` and `lost_capacity_days`.
   - Sortable/compact breakdown table sorted by `(hop_distance, verdict, entity_ticker)`.
4. **Card 3 — Verbatim Citation Audit, Synthesis & 1NF Export Preview**:
   - Prominent **Citation Guardrail Badge**: green `0 Violations` pill when `citation_violations` is empty; amber/red list of exact mismatched percentages/days if non-empty.
   - `operational_synthesis` prose block + collapsible `contagion_think_trace` + execution latency telemetry pills (`ast_guardrail_ms`, `graph_traversal_ms`, `cascade_propagation_ms`, `total_latency_ms`).
   - Collapsible **1NF Time-Series Rows (`ContagionTimeSeriesEventRecord`)** table preview with client-side JSONL copy/download helper for inspection.

---

## 5. Invariants & Guardrails

1. **Zero External Runtime Assets**:
   - Automated tests MUST verify served HTML/CSS/JS contain no `http://` or `https://` `<script>`/`<link>` CDN references.
2. **XSS-Safe DOM Insertion**:
   - All server-returned strings (filing headlines, sanitized text, synthesis, think traces) MUST be rendered using `element.textContent` / SVG text nodes rather than HTML string concatenation.
3. **Read-Only Server Parity**:
   - Adding static UI routes MUST NOT weaken ADR-00017 path-confinement (`HTTP 403`) or inline temporary-file cleanup guarantees.
4. **Zero Absolute Paths Invariant**:
   - All repository and peer-ADR references remain strictly repository-relative (`../../<path>`).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Delivers a completely offline, audit-focused browser workbench for SEC 8-K gating, supply-chain downtime visualization, and verbatim citation verification grounded directly in [`OperationalContagionReport`](../../retrieval/reporting.py).
  - Pre-acceptance prototype review lets visual spacing, color contrast, and timeline design stabilize before implementation.
- **Negative / Trade-offs**:
  - `PiT KG Explorer` and `Cohort Browser` tabs display placeholder notices until **ADR-00019** is authored, reviewed, and accepted.

