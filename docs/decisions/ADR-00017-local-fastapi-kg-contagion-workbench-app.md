# ADR-00017: Local FastAPI Workbench Backend & API Contracts

- **ADR ID**: ADR-00017
- **Filename**: `ADR-00017-local-fastapi-kg-contagion-workbench-app.md`
- **Date**: 2026-11-13 09:46 UTC
- **Status**: ACCEPTED
- **Target Subsystem**: `explorer` / `retrieval` / `tools`
- **Linked Commits**: [`explorer/workbench_api.py`](../../explorer/workbench_api.py), [`tools/serve_explorer.py`](../../tools/serve_explorer.py)

---

## 1. Context & Problem Statement

Following the removal of the obsolete Chainlit frontend ([`ADR-00016`](ADR-00016-decommission-legacy-chainlit-ui-frontend.md)), YarnBall outputs are accessible only via CLI scripts and direct Python calls ([`tools/dry_run_sec_8k_contagion.py`](../../tools/dry_run_sec_8k_contagion.py)). To power a local interactive browser workbench cleanly in narrow, reviewable steps, the system first needs a standalone, well-tested **FastAPI JSON backend** exposing:

1. **Snapshot & Filing Discovery (`/api/catalog`)**: Enumerating local Parquet snapshots (`data/snapshots/*`), JSON/JSONL edge manifests, Form 8-K fixtures (`tests/fixtures/filings/*`), exported `ADR-00013` JSONL cohorts, and local Ollama availability.
2. **Bidirectional Point-in-Time Subgraph Queries (`/api/subgraph`)**: Querying [`InMemorySubgraphStore`](../../retrieval/offline_store.py) around an issuer ticker (`downstream`, `upstream`, or `both`) at a chosen `as_of_date` with full edge provenance.
3. **Observed SEC 8-K Evaluation (`/api/evaluate-8k`)**: Running [`assess_sec_8k_disruption`](../../ingestion/sec/disruption_gating.py) and `run_sec_8k_dry_run` on a selected fixture or submitted 8-K text/JSON to emit sanitized text, cascade verdicts, verbatim citation checks, and 1NF rows.
4. **Cohort Export Filtering (`/api/cohort`)**: Reading and filtering `ContagionTimeSeriesEventRecord` JSONL exports without loading raw files outside workspace boundaries.

Browser UI shell styling, interactive SVG contagion graphs, and Gantt views are intentionally deferred to follow-up UI ADRs so frontend look-and-feel can be prototyped and reviewed independently.

---

## 2. Decision Drivers & Constraints

1. **Backend/Frontend Decoupling**:
   - Establish clean Pydantic HTTP contracts first so UI templates and interactive prototypes can evolve independently without re-architecting Python services.
2. **Evidence-Driven Auditability (No Synthetic Shock Simulators)**:
   - Evaluation endpoints accept observed SEC Form 8-K disclosures (or local fixtures) and Point-in-Time snapshot parameters—not ungrounded manual stress-shock sliders.
3. **Zero Extra Dependencies & Zero External Databases**:
   - Build strictly on `fastapi`, `uvicorn`, `pydantic`, `pyarrow`, and [`InMemorySubgraphStore`](../../retrieval/offline_store.py) already installed in the runtime—no running Memgraph, PostgreSQL, or Node/`npm` required.
4. **Strict Read-Only & Workspace Path Confinement**:
   - Bind strictly to `127.0.0.1` by default, never mutate snapshot or filing files on disk, and reject path traversal outside allowlisted workspace directories (`HTTP 400`/`403`).
5. **Deterministic Offline Default**:
   - `/api/evaluate-8k` defaults to `use_mock = True` (`MOCK_DETERMINISTIC`), with explicit `use_mock = False` opt-in when local Ollama (`yarnball-qwen:7b`) is reachable.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Combine FastAPI Backend and Full HTML/SVG/Gantt Frontend in One ADR**:
  - *Rationale for Rejection*: Couples backend API security/traversal logic with frontend layout styling in a single oversized diff, hindering step-by-step human review.
- **Alternative B: Extend Legacy `retrieval/service.py` Directly**:
  - *Rationale for Rejection*: [`retrieval/service.py`](../../retrieval/service.py) is a containerized Docker microservice tied to live PostgreSQL (`pgvector`) and Memgraph `enrich_nodes`, which conflicts with zero-DB offline exploration over `InMemorySubgraphStore`.

---

## 4. Decision Outcome

Introduce the pure-Python backend package [`explorer/`](../../explorer/__init__.py) and CLI launcher [`tools/serve_explorer.py`](../../tools/serve_explorer.py):

### 4.1 Configurable App Factory (`explorer/workbench_api.py`)
Expose `create_workbench_app(project_root=..., allowlisted_roots=...) -> FastAPI` returning typed Pydantic responses:
- `GET /`: Minimal unstyled readiness page / health status summary linking to `/docs` (OpenAPI) until the UI shell ADR lands.
- `GET /api/catalog`: Lists allowlisted Parquet snapshot directories, `.json`/`.jsonl` edge manifests, Form 8-K fixtures, `.jsonl` cohort exports, and live Ollama health (`YarnBallModelClient.check_health(timeout_sec=1.0)`).
- `GET /api/subgraph`: Loads an allowlisted snapshot or edge manifest via `InMemorySubgraphStore`, filters by `ticker` (returning bidirectional/upstream/downstream edges within `max_hops` and active `[valid_from, valid_to]` at `as_of_date`), and includes full snapshot ticker inventory and edge provenance (`share`, `buffer_days`, `nature`, `status`, `valid_from`, `valid_to`, `source_hash`).
- `POST /api/evaluate-8k`: Accepts either an allowlisted `filing_path` or an inline 8-K payload (`ticker`, `company_name`, `filing_date`, `items`, `headline`, `content`, `metadata`) alongside optional `snapshot_id`/`edges_file`, `target_ticker`, and `use_mock`. Executes `assess_sec_8k_disruption` + `run_sec_8k_dry_run` cleanly (using an ephemeral temporary payload file only when inline JSON is posted, leaving workspace files untouched) and returns:
  - `assessment` (`SanitizedDisruptionAssessment`)
  - `report` (`OperationalContagionReport`)
  - `citation_violations` (`validate_verbatim_citations(report)`)
  - `timeseries_rows` (`flatten_report_to_timeseries_records(report)`)
- `POST /api/cohort`: Parses an allowlisted `ADR-00013` `.jsonl` cohort file with optional filters (`verdict`, `disruption_category`, `sec_item`, `ticker`, `limit`, `offset`) and returns aggregate counts plus matching `ContagionTimeSeriesEventRecord` rows.

### 4.2 CLI Launcher (`tools/serve_explorer.py`)
Starts `uvicorn` bound cleanly to `127.0.0.1` (`--host`, `--port` defaulting to `8050`, `--open` optional browser launch, `--reload`).

---

## 5. Invariants & Guardrails

1. **Workspace Path Confinement**:
   - Every file/directory input (`snapshot_id`, `snapshot_dir`, `edges_file`, `filing_path`, `cohort_path`) MUST be resolved via `Path.resolve()` and verified inside configured allowlisted workspace roots (`Path.is_relative_to`), returning `HTTP 400`/`403` on traversal attempts.
2. **Strict Read-Only Guarantee**:
   - Snapshot Parquet files, edge manifests, fixtures, and JSONL exports MUST never be modified or overwritten by any endpoint.
3. **Inline 8-K Evaluation Ephemerality**:
   - When evaluating pasted inline Form 8-K payloads, temporary files MUST be scoped inside an isolated `tempfile` context cleaned up immediately after execution.
4. **Zero Absolute Paths Invariant**:
   - All repository and peer-ADR references remain strictly repository-relative (`../../<path>`).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Delivers clean, OpenAPI-documented, `TestClient`-verified Python endpoints ready to back an interactive local browser workbench in follow-up **ADR-00018**.
- **Negative / Trade-offs**:
  - Only a minimal readiness response is served at `GET /` upon completion of ADR-00017; visual styling, SVG hop graphs, and Gantt timelines require follow-up UI ADRs.
