# ADR-00019: PiT KG Explorer & Cohort Browser Views

- **ADR ID**: ADR-00019
- **Filename**: `ADR-00019-pit-kg-explorer-and-cohort-browser-views.md`
- **Date**: 2026-11-13 22:26 UTC
- **Status**: ACCEPTED
- **Target Subsystem**: `explorer` / `tests`
- **Linked Commits**: Pending acceptance ([`explorer/workbench_api.py`](../../explorer/workbench_api.py), [`explorer/static/index.html`](../../explorer/static/index.html))

---

## 1. Context & Problem Statement

[`ADR-00017`](ADR-00017-local-fastapi-kg-contagion-workbench-app.md) delivered the local read-only FastAPI endpoints (`GET /api/subgraph`, `POST /api/cohort`), and [`ADR-00018`](ADR-00018-workbench-ui-shell-and-8k-audit-view----.md) implemented the offline zero-CDN workbench shell (`explorer/static/`) and the **Event Audit (8-K -> Contagion)** view. Currently, the remaining two navigation tabs (**PiT KG Explorer** and **Cohort Browser**) display static placeholder notices.

To complete the diagnostic workbench for empirical iteration against market behavior:
1. **PiT KG Explorer**: Analysts need a visual Point-in-Time microscope over local Parquet snapshots (`data/snapshots/*`) and edge manifests—showing upstream suppliers, downstream customers/competitors, hop depth, and exact temporal validity (`[valid_from, valid_to]`, `status`) at a chosen `as_of_date`, with click-to-inspect edge provenance (`share`, `buffer_days`, `confidence`, `evidence_count`, `nature`, `source_hash`) and click-to-recenter node navigation.
2. **Cohort Browser**: Analysts need a faceted browser over exported `ADR-00013` `.jsonl` batches (`ContagionTimeSeriesEventRecord`) supporting filtering (`verdict`, `disruption_category`, `sec_item`, `ticker`), aggregate breakdown bars, pagination, and fast cross-navigation into the **Event Audit** or **PiT KG Explorer** views for root-cause inspection.

---

## 2. Decision Drivers & Constraints

1. **Zero-CDN Deterministic SVG Graph Layout**:
   - Follow the strict offline runtime invariant from [`ADR-00018`](ADR-00018-workbench-ui-shell-and-8k-audit-view----.md): implement the interactive subgraph diagram in pure native SVG + vanilla ES2020 inside [`explorer/static/`](../../explorer/static/app.js) without D3, Cytoscape CDN bundles, or Node/`npm`.
2. **Layered Upstream -> Root -> Downstream Readability**:
   - Position nodes deterministically by signed hop tier (`Upstream -k .. -1` on the left, `Root 0` centered, `Downstream +1 .. +k` on the right) rather than non-deterministic force simulation jitter, preserving legible directional flow and edge labels (`rel_type`, `share`, `buffer_days`).
3. **Zero-Lookahead Temporal Clarity**:
   - Emphasize `as_of_date` Point-in-Time scrubbing and lifecycle `status` controls so users can confirm no future edges (`valid_from > as_of_date` or `valid_to < as_of_date`) enter historical analyses.
4. **XSS-Safe DOM & Strict Path Confinement Reuse**:
   - Render all node IDs, company names, provenance JSON, and cohort cell strings exclusively via DOM `textContent` / SVG text nodes and existing allowlisted API contracts (`GET /api/subgraph`, `POST /api/cohort`) with zero backend weakening.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Embed External Graph JS Libraries (Cytoscape / D3 / Sigma via CDN)**:
  - *Rationale for Rejection*: Violates strict zero-CDN offline operation inside `127.0.0.1:8050` and increases bundle complexity unnecessarily for bounded 1–5 hop ego subgraphs.
- **Alternative B: Random/Non-Deterministic Spring-Force Layout**:
  - *Rationale for Rejection*: Nondeterministic coordinates shift node positions between `as_of_date` scrub steps, making visual comparison across historical dates confusing. Layered hop columns hold anchors stable as edges appear or disappear.
- **Alternative C: Separate Standalone CLI Dashboards per View**:
  - *Rationale for Rejection*: Fragments navigation; cross-linking a flagged `UNBUFFERED_DISRUPTION` row in Cohort Browser directly into `PiT KG Explorer` requires unified workbench state inside [`explorer/static/`](../../explorer/static/index.html).

---

## 4. Decision Outcome

Replace the two `ADR-00019` placeholder panels inside [`explorer/static/index.html`](../../explorer/static/index.html), [`explorer/static/styles.css`](../../explorer/static/styles.css), and [`explorer/static/app.js`](../../explorer/static/app.js) with full interactive views backed by existing [`explorer/workbench_api.py`](../../explorer/workbench_api.py) endpoints:

### 4.1 Tab 2 — PiT KG Explorer (`GET /api/subgraph`)
Two-column layout pairing query controls + provenance inspector with a deterministic interactive SVG canvas:
- **Control Rail**:
  - Snapshot / Edge Manifest dropdown populated from `GET /api/catalog` (`snapshots` + `edge_manifests`).
  - Focal `ticker` dropdown (auto-populated from `available_tickers`, including `ALL` full-snapshot mode).
  - `direction` segmented control (`both`, `upstream`, `downstream`), `max_hops` selector (`1–5`), optional `rel_types` filter (`ALL`, `SUPPLIES_TO`, `COMPETES_WITH`), and `required_status` selector (`ACTIVE_CURRENT` vs. `ALL`).
  - `as_of_date` date input + quick PiT preset chips (or date-range step buttons) to reveal temporal edge transitions cleanly.
- **Interactive SVG Ego Graph Canvas**:
  - Groups nodes into ordered vertical columns: `UPSTREAM` (`-max_hops..-1`) -> `ROOT` (`0`) -> `DOWNSTREAM` (`+1..+max_hops`), falling back cleanly to topological source-to-target rank when viewing an unscoped snapshot (`ALL`).
  - Visually distinguishes `SUPPLIES_TO` (solid directional curve) vs. `COMPETES_WITH` / other relations (dashed curve) and highlights edges carrying tight buffers (`buffer_days`) or high `share`.
  - **Node Click**: Re-centers the `PiT KG Explorer` around that ticker (`ticker = node.ticker`).
  - **Edge Click**: Highlights the relationship path and populates the **Edge Provenance Inspector** card with `source -> target`, `rel_type`, `nature`/`input_group`, `share`, `buffer_days`, `confidence`, `evidence_count`, `valid_from`, `valid_to`, `status`, and `source_hash` (if present).
  - Keyboard/table accessibility fallback: Compact sorted edge table underneath the SVG with click-to-select parity.

### 4.2 Tab 3 — Cohort Browser (`POST /api/cohort`)
- **Filter Bar & Export Selector**:
  - Cohort `.jsonl` dropdown populated from `GET /api/catalog` (`cohorts`), plus manual allowlisted workspace relative path entry for newly exported run batches.
  - Facet filters: `verdict`, `disruption_category`, `sec_item`, and `ticker` (matching either `disrupted_entity_ticker` or `target_entity_ticker`), plus page size (`limit`) and Prev/Next pagination (`offset`).
- **Aggregate Distribution Summary**:
  - Summary KPIs (`total_records`, `filtered_records`) and pure-CSS/HTML horizontal proportion bars for `verdict_counts`, `category_counts`, and `sec_item_counts`.
- **1NF Event Table & Cross-View Inspect Action**:
  - Tabular view of matching `ContagionTimeSeriesEventRecord` rows (`filing_date`, `disrupted_entity_ticker`, `target_entity_ticker`, `hop_distance`, `sec_item`, `disruption_category`, `calendar_onset_date`, `calendar_recovery_date`, `capacity_contraction_pct`, `unbuffered_downtime_days`, `verdict`, `model_mode`).
  - Row action button **Inspect Network** switches to `PiT KG Explorer` pre-filled with `ticker = disrupted_entity_ticker` and `as_of_date = filing_date` to audit the exact Point-in-Time topology that produced that verdict.

---

## 5. Invariants & Guardrails

1. **Pre-Acceptance Prototype Review**:
   - Visual hierarchy, SVG hop spacing, provenance card layout, and Cohort filter ergonomics are reviewed via the interactive prototype artifact prior to accepting and implementing this ADR.
2. **Offline Zero-CDN & XSS-Safe Rendering**:
   - All newly added HTML/CSS/JS/SVG MUST preserve [`ADR-00018`](ADR-00018-workbench-ui-shell-and-8k-audit-view----.md) invariants: zero `http://`/`https://` network assets (excluding the `http://www.w3.org/2000/svg` DOM namespace string) and zero `.innerHTML` usage.
3. **Strict Domain 1 Purity**:
   - `Cohort Browser` and `PiT KG Explorer` display strictly operational supply-chain structure, calendar onset/recovery dates, and contagion verdicts—never equity pricing or trade signals.
4. **Zero Absolute Paths Invariant**:
   - All repository and peer-ADR references remain strictly repository-relative (`../../<path>`).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Completes all three workbench tabs defined across [`ADR-00017`](ADR-00017-local-fastapi-kg-contagion-workbench-app.md)–`ADR-00019` over a zero-DB, zero-Node stack.
  - Enables seamless drill-down from aggregate backtest cohort anomalies straight into exact historical Point-in-Time graph edges (`Inspect Network`).
- **Negative / Trade-offs**:
  - Very large dense snapshots viewed in unscoped `ALL` mode require sensible edge rendering caps in the UI (defaulting to root-focused ego queries of `1–5` hops for clarity).
