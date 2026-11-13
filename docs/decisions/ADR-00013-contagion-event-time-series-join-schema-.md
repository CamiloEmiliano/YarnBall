# ADR-00013: Contagion Event Telemetry & Time-Series Join Contract

- **ADR ID**: ADR-00013
- **Filename**: `ADR-00013-contagion-event-time-series-join-schema-.md`
- **Date**: 2026-11-13 06:30 UTC
- **Status**: ACCEPTED
- **Target Subsystem**: `retrieval` / `tools` / `sft_pipeline`
- **Linked Commits**: Implemented in [`retrieval/reporting.py`](../../retrieval/reporting.py), [`sft_pipeline/serving/model_client.py`](../../sft_pipeline/serving/model_client.py), [`tools/dry_run_sec_8k_contagion.py`](../../tools/dry_run_sec_8k_contagion.py)

---

## 1. Context & Problem Statement

With the core mathematical cascade engine ([`ADR-00006`](ADR-00006-phase-7-sparse-network-cascade-engine---.md)), standardized contagion report schema ([`ADR-00011`](ADR-00011-operational-contagion-report-schema-----.md)), and SEC Form 8-K dry-run harness ([`ADR-00009`](ADR-00009-sec-8k-computational-reading-dry-run----.md)) in place, YarnBall is transitioning toward empirical validation. Rigorous data science evaluation—conducted in a dedicated external Cookiecutter research repository—requires joining YarnBall's operational shock predictions against 2018–2026 frequency-agnostic OHLCV candlestick time series (minutely, hourly, daily, or weekly bars) to perform econometric event studies (e.g., testing whether predicted onset delays correspond to delayed abnormal equity returns, and whether `BUFFER_ABSORBED` verdicts correlate with zero downstream price variance).

Two concrete software gaps currently obstruct reliable programmatic consumption:

1. **Nested Reporting vs. Frequency-Agnostic Relational Joins**:
   `OperationalContagionReport` in [`retrieval/reporting.py`](../../retrieval/reporting.py) emits deeply nested JSON containing lists of `NetworkNodeImpact` records with relative day offsets (`shock_onset_day: float`). Downstream data science pipelines cannot trivially execute relational joins (`df.merge`) against arbitrary-frequency time series without flattening nested arrays, projecting calendar onset/recovery dates, and preserving second-level filing acceptance timestamps (`filing_timestamp`) when available.
2. **Silent Client Failures & Lack of Offline CLI Execution Flag**:
   In [`sft_pipeline/serving/model_client.py`](../../sft_pipeline/serving/model_client.py), an unreachable local Ollama server triggers an exception handler that silently returns `final_output="(none)"`. When invoked via [`tools/dry_run_sec_8k_contagion.py`](../../tools/dry_run_sec_8k_contagion.py), `(none)` passes into Task C (Text-to-Cypher), trips `CypherASTWhitelistGuardrail`, aborts graph retrieval, and outputs an empty cascade table without warning the caller of the serving outage or offering a deterministic offline execution mode.

---

## 2. Decision Drivers & Constraints

1. **First Normal Form (1NF) Time-Series Join Readiness**:
   - Exported event records must represent one flat observation per impacted node/input group, keyed by `(event_id, disrupted_entity_ticker, target_entity_ticker, filing_date)` with deterministic ISO-8601 calendar projections.
2. **Strict Domain 1 Boundary Preservation**:
   - Consistent with the formal rejection of [`ADR-00010`](ADR-00010-directional-basket-peer-cluster-mapping-.md), exported schemas must strictly carry physical/operational network metrics and dates—never trading signals, long/short weights, or options overlays.
3. **Provenance & Execution Isolation**:
   - Exported rows must explicitly record whether neural inference was performed by a live model (`LIVE_LLM`) or an offline deterministic mock (`MOCK_DETERMINISTIC`) so simulated runs never pollute empirical backtest tables.
4. **Zero Silent Failures**:
   - Automated batch pipelines must immediately distinguish an unreachable LLM endpoint from a legitimately unaffected supply chain.
5. **Backward Compatibility**:
   - Existing `OperationalContagionReport` schemas, ASCII/Markdown visualizers, and integration tests remain strictly intact.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Parse ASCII Console Tables in the Research Repository**:
  - *Rationale for Rejection*: Regex scraping human-oriented ASCII tables is brittle and breaks whenever column widths or formatting descriptors evolve.
- **Alternative B: Embed Market Prices (OHLCV / CAR) Inside `OperationalContagionReport`**:
  - *Rationale for Rejection*: Violates architectural separation of concerns. YarnBall is a Domain 1 operational contagion intelligence engine; equity return modeling and benchmark adjustments belong in the external research repository.
- **Alternative C: Maintain Silent `(none)` Fallback on HTTP Connection Refused**:
  - *Rationale for Rejection*: Causes confusing false negatives (`0` impacted nodes) when an analyst forgets to start the local inference server prior to running a batch script.

---

## 4. Decision Outcome

### 4.1 Flattened Time-Series Join Schema (`ContagionTimeSeriesEventRecord`)
Define a flattened Pydantic record model and conversion helper (`flatten_report_to_timeseries_records`) inside [`retrieval/reporting.py`](../../retrieval/reporting.py) mapping each `NetworkNodeImpact` in an `OperationalContagionReport` into a relational row:

| Field | Type | Description / Derivation Formula |
| :--- | :--- | :--- |
| `report_id` | `str` | Parent report identifier |
| `event_id` | `str` | Deterministic Form 8-K event identifier |
| `filing_date` | `str` | ISO-8601 disclosure date $T_0$ (`YYYY-MM-DD`) |
| `filing_timestamp` | `Optional[str]` | Optional ISO-8601 UTC EDGAR acceptance timestamp (`YYYY-MM-DDTHH:MM:SSZ`) for intraday bar alignment |
| `sec_item` | `str` | Statutory Form 8-K Item code (`Item 1.01`–`Item 9.01`) |
| `disruption_category` | `str` | Controlled `OperationalDisruptionCategory` enum value |
| `disrupted_entity_ticker` | `str` | Upstream shock origin ticker |
| `disrupted_entity_cik` | `Optional[str]` | Upstream shock origin 10-digit SEC CIK |
| `target_entity_ticker` | `str` | Downstream impacted entity ticker (market series join key) |
| `target_entity_cik` | `Optional[str]` | Downstream impacted entity 10-digit SEC CIK (regulatory/fundamental join key) |
| `hop_distance` | `int` | Topological hop distance from shock origin |
| `input_group` | `str` | Normalized snake_case input dependency group |
| `dependency_domain` | `str` | `PHYSICAL_COMPONENT`, `DIGITAL_INFRASTRUCTURE`, or `COMMERCIAL_ENTERPRISE` |
| `operational_scope` | `str` | `CORE_OPERATIONAL_DELIVERY` or `GROWTH_CAPEX_AND_RD` |
| `procurement_share_pct` | `float` | Procurement exposure share ($a_{uv} \times 100$) |
| `inventory_buffer_days` | `float` | Inventory safety stock buffer $B_{uv}$ in calendar days |
| `shock_onset_day` | `Optional[float]` | Relative onset delay in days ($B_{uv} / h_u$), or `None` |
| `unbuffered_downtime_days` | `float` | Post-buffer disruption duration $D_{\text{down}}$ |
| `capacity_contraction_pct` | `float` | Peak capacity shortfall percentage $h_v \times 100$ |
| `lost_capacity_days` | `float` | Cumulative capacity contraction ($h_v \times D_{\text{down}}$) |
| `calendar_onset_date` | `Optional[str]` | ISO-8601 date $T_0 + \lfloor \text{shock\_onset\_day} \rfloor$ days (`None` if unbreached) |
| `calendar_recovery_date` | `Optional[str]` | ISO-8601 date $\text{calendar\_onset\_date} + \lceil \text{unbuffered\_downtime\_days} \rceil$ days (`None` if downtime $= 0$) |
| `verdict` | `str` | `UNBUFFERED_DISRUPTION`, `BUFFER_ABSORBED`, or `UNAFFECTED` |
| `is_acyclic_dag` | `bool` | Whether retrieved topology is a provable DAG |
| `spectral_radius` | `float` | Effective graph spectral radius $\rho(W)$ |
| `model_mode` | `str` | Execution provenance: `LIVE_LLM`, `MOCK_DETERMINISTIC`, or `DEGRADED_UNREACHABLE` |

### 4.2 CLI Batch Export & Deterministic Mock Mode
Enhance [`tools/dry_run_sec_8k_contagion.py`](../../tools/dry_run_sec_8k_contagion.py) with two explicit batch flags:
- `--export-jsonl <PATH>`: Flattens the generated report via `flatten_report_to_timeseries_records` and appends one JSON line per impacted node to `<PATH>` (creating parent directories automatically). If a report contains zero downstream `network_impacts`, emits a single root-level sentinel row with `target_entity_ticker = disrupted_entity_ticker`, `hop_distance = 0`, and `verdict = "UNAFFECTED"` to preserve negative-control observations.
- `--mock`: Executes the dry-run pipeline using a deterministic offline `MockDryRunModelClient` (with `model_mode="MOCK_DETERMINISTIC"`) for zero-GPU pipeline mechanics testing without mutating live serving configs.

### 4.3 Explicit Serving Pre-Flight Diagnostics
Enhance [`YarnBallModelClient`](../../sft_pipeline/serving/model_client.py):
- Introduce `check_health(timeout_sec: float = 2.0) -> Tuple[bool, str]` querying the base `/api/tags` endpoint (automatically returning `(True, "mock_transport")` when an in-memory `transport_fn` is attached).
- Emit an explicit structured `WARNING` log when HTTP inference fails due to connection refusal or timeout, naming the unreachable URL and advising `ollama serve` or `--mock`, rather than silently masking the outage behind downstream AST guardrail rejections.

---

## 5. Invariants & Guardrails

1. **Calendar Projection Determinism**:
   - When `shock_onset_day is None`, both `calendar_onset_date` and `calendar_recovery_date` MUST evaluate to `None`. Calendar arithmetic uses exact `datetime.date` increments from `filing_date` (`YYYY-MM-DD`).
2. **Provenance Integrity (`model_mode`)**:
   - Every exported JSONL row must carry a non-empty `model_mode` tag (`LIVE_LLM`, `MOCK_DETERMINISTIC`, or `DEGRADED_UNREACHABLE`) to guarantee empirical datasets never silently mix live neural outputs with deterministic test mocks.
3. **Domain 1 Metric Exclusivity**:
   - No price targets, equity tickers outside the supply topology, trading directions, or options metrics may appear in `ContagionTimeSeriesEventRecord`.
4. **Zero Absolute Paths Invariant**:
   - All repository references and file links remain strictly relative (`../../<path>`).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Enables one-line ingestion (`pandas.read_json(..., lines=True)`) in the external Cookiecutter data science repository for joining operational cascades against frequency-agnostic OHLCV candlestick time series (minutely, hourly, daily, or weekly bars) across the 2018–2026 horizon.
  - Eliminates silent Ollama outages during CLI dry runs via explicit health diagnostics and `--mock` execution support.
- **Negative / Trade-offs**:
  - Calendar date projections (`calendar_onset_date`, `calendar_recovery_date`) advance in raw calendar days rather than exchange trading days; downstream econometric event studies remain responsible for snapping calendar dates or `filing_timestamp` values to their chosen bar interval and active exchange calendar.

