# ADR-00038: Deferred First-Instance Event-Study Backtest & Signal-Decay Calibration

- **ADR ID**: ADR-00038
- **Filename**: `ADR-00038-event-study-backtest-and-signal-decay---.md`
- **Date**: 2026-11-23 08:20 UTC
- **Status**: PROPOSED
- **Target Subsystem**: `evaluation` / `retrieval` / `contagion` / `tools`
- **Linked Commits**: Pending acceptance

---

## 1. Context & Problem Statement

Several architectural parameters across [`ADR-00033`](ADR-00033-bitemporal-pit-contract-for-read-paths--.md)–[`ADR-00037`](ADR-00037-pit-community-layer-and-impact-ranking--.md) depend on the empirical physics linking a news item or Form 8-K disclosure to forward stock returns:

1. **Direct news vs. indirect supply-chain contagion**: Under regular-session-only execution (no after-hours trading; 16:00 ET / 13:00 ET cutoff per [`ADR-00033`](ADR-00033-bitemporal-pit-contract-for-read-paths--.md)), first-order price reactions on the *directly named* ticker are largely absorbed before or at the next regular open. YarnBall's economic thesis rests on the *slower* indirect channel—post-event supply-chain drift over days to weeks on downstream customers connected via the Knowledge Graph.
2. **Unmeasured physical parameters**: Without empirical calibration, key architecture knobs rely on theoretic priors:
   - `MEMGRAPH_EVENT_TTL_DAYS = 90` ([`ADR-00036`](ADR-00036-fail-fast-infra-and-deployable-topology-.md)) assumes transient shock diffusion resolves within one quarterly earnings cycle.
   - Whether general news streams add tradable lead time over statutory SEC Form 8-Ks alone—or mostly inject lagging echoes—remains unmeasured.
   - Whether extraction recall ([`ADR-00035`](ADR-00035-text-unit-chunk-layer-and-evidence-links.md)) or disruption/novelty gating ([`ADR-00014`](ADR-00014-sec-8k-boilerplate-sanitization-gating--.md)) is the tighter real-world bottleneck is unverified.
3. **Why this study is formally deferred (not executed upfront)**: Running a statistically honest multi-year event study requires survivorship-bias-free historical Open/Close OHLCV (including delisted names), scheduled quarterly earnings dates, and a curated, deduplicated multi-year news archive with reliable UTC timestamps. Building that curation pipeline today would stall verified P0/P1 fixes (`ADR-00033`–`ADR-00036`). Instead, the system adopts a clean, parameterized theoretic baseline now and records the exact experimental specification here so it can be executed and used to recalibrate parameters once data prerequisites are met.

---

## 2. Decision Drivers & Constraints

1. **Tradable causality, not overnight-gap artifacts**: Metrics must isolate what is tradable in regular NYSE hours after `actionable_session(known_at)` ([`ADR-00033`](ADR-00033-bitemporal-pit-contract-for-read-paths--.md)), excluding uncapturable overnight gaps.
2. **Reverse-causality detection**: Free/syndicated news archives often lag real market moves (journalists writing stories *after* a stock drops). The study must detect lagging echoes automatically rather than misattributing them as causes.
3. **Domain isolation** ([`ADR-00010`](ADR-00010-directional-basket-peer-cluster-mapping-.md), [`ADR-00011`](ADR-00011-operational-contagion-report-schema-----.md)): This harness lives strictly in offline evaluation/tooling (`evaluation/` / `tools/`). It never introduces return targets, trading rules, or portfolio fields into Domain 1 runtime contracts (`OperationalContagionReport`).
4. **Data-source policy** (`.agents/rules/data_sources.md`): Historical filings come from SEC EDGAR, historical news from Form 8-K + local FNSPID/curated archives, and historical OHLCV from Yahoo Finance (`yfinance`) or a future survivorship-bias-free daily equity feed—never Finnhub historical queries.
5. **Reproducible parameterization**: Every calibrated threshold maps directly to a configuration knob in `ADR-00033`–`ADR-00037` without requiring schema rewrites.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Block `ADR-00033`–`ADR-00037` until a full multi-year backtest is curated and executed.** Rejected. Curating survivorship-free equity data, earnings calendars, and deduplicated historical news is a major standalone effort and would leave confirmed P0 Point-in-Time leakage and silent DB fallback bugs unfixed.
- **Alternative B: Rely permanently on theoretic defaults without recording an evaluation specification.** Rejected. Leaves `MEMGRAPH_EVENT_TTL_DAYS`, news ingestion value, and cascade buffer priors permanently unvalidated against market reality.
- **Alternative C: Measure `Close(t) -> Close(t + h)` returns on raw un-deduplicated news articles.** Rejected. Credits untradable after-hours gaps, double-counts multi-day news echo storms, and confounds reverse causality with genuine information lead.

---

## 4. Decision Outcome

Establish a **Deferred First-Instance Event-Study & Signal-Decay Calibration Protocol**, to be implemented as an offline CLI harness (`tools/eval_event_signal_decay.py` + `evaluation/event_study.py`) once the data prerequisites below are satisfied.

### 4.1 Data Prerequisites Gate (Activation Criteria)
1. **Survivorship-bias-free daily OHLCV (Open & Close) & indexed split registry**: Point-in-Time S&P 500 constituents ([`universe/sp500.py`](../../universe/sp500.py)), offline forward/reverse stock-split registry (`sp500_stock_splits` / `data/sp500_stock_splits_historical.json` per [`ADR-00039`](ADR-00039-sp500-pit-stock-split-corporate-actions-.md)), plus SPY and GICS sector ETF benchmarks.
2. **Quarterly earnings-announcement calendar** covering the study window, to isolate or exclude events within $\pm 1$ trading day of scheduled earnings.
3. **Timestamped historical event corpus**: SEC Form 8-K filings (with EDGAR `acceptance_datetime`) and curated historical news with UTC `published_at`.

### 4.2 First-Instance Novelty Gate
An article or filing on source entity $A$ qualifies as a **first-instance impulse** at timestamp `known_at` if and only if no prior event on $A$ with embedding cosine similarity $\ge \tau_{\text{sim}}$ (default `0.75` using `all-mpnet-base-v2`) occurred within the preceding $W$ trading days (default `10`). Subsequent similar articles within the cooldown window are tagged as echoes and excluded from impulse triggers.

### 4.3 Experimental Matrix ($2 \times 2$ Stratification)
Evaluate every qualified first-instance impulse across four disjoint cells:
- **Source Axis**: Statutory **Form 8-K disruption** vs. **First-instance general news**.
- **Transmission Axis**:
  - **Direct Leg ($A$)**: Forward returns on source ticker $A$ explicitly named in the text.
  - **Pure KG-Inferred Indirect Leg ($B$)**: Forward returns on downstream customer/counterparty $B$ reachable via Point-in-Time structural edges (`valid_from <= t_0` and `valid_to IS NULL OR valid_to > t_0` per [`ADR-00033`](ADR-00033-bitemporal-pit-contract-for-read-paths--.md)) where $B$ is **explicitly absent** from the event text.

### 4.4 Return Decomposition & Confounder Controls
Anchor each impulse to $t_0 = \text{actionable\_session}(\text{known\_at})$ ([`ADR-00033`](ADR-00033-bitemporal-pit-contract-for-read-paths--.md)) and compute market- and sector-residualized Cumulative Abnormal Returns (extending [`ingestion/market/parser.py`](../../ingestion/market/parser.py) `MarketMetricsCalculator` and [`ADR-00013`](ADR-00013-contagion-event-time-series-join-schema-.md) event joins):
1. **Pre-event echo diagnostic (`CAR[-3, 0]`)**: Measures abnormal return strictly *prior* to $t_0$. Flat $\text{CAR}[-3, 0] \approx 0$ confirms genuine pre-move novelty; large pre-event moves expose lagging-echo / reverse-causality bias.
2. **Untradable reaction (`Close(t_0 - 1) -> Open(t_0)`)**: Logged separately and **excluded** from tradable signal scoring.
3. **Tradable forward drift curve (`Open(t_0) -> Close(t_0 + h)`)** across horizons $h \in \{1, 3, 5, 10, 20, 40, 60\}$ trading days (plus `Close(t_0) -> Close(t_0 + h)` sensitivity check).
4. **Cascade physics & confounder stratification**:
   - Compare cascade-unabsorbed shocks (`downtime > buffer_days`) vs. absorbed shocks (`downtime <= buffer_days`) computed by [`contagion/cascade.py`](../../contagion/cascade.py) `SparseNetworkCascadeEngine`.
   - Separate events within $\pm 1$ trading day of scheduled earnings—and windows intersecting forward or reverse stock-split declaration/ex-dates ([`ADR-00039`](ADR-00039-sp500-pit-stock-split-corporate-actions-.md))—from unconfounded operational events.

### 4.5 Parameter Calibration Feedback Map
Once executed with adequate sample size, empirical results recalibrate the theoretic baseline without architectural rework:

| Empirical Finding | Architectural / Config Calibration |
|---|---|
| Empirical drift half-life / plateau horizon $h^*$ | Sets `MEMGRAPH_EVENT_TTL_DAYS` ([`ADR-00036`](ADR-00036-fail-fast-infra-and-deployable-topology-.md)) to match measured signal decay (e.g., $\lceil 2 h^* \rceil$). |
| News vs. Form 8-K incremental tradable CAR | Determines whether live news streams trigger cascade alerts or are restricted to corroboration / offline storage. |
| Insufficient indirect KG pairs vs. high false-positive rate | Prioritizes full-section extraction recall ([`ADR-00035`](ADR-00035-text-unit-chunk-layer-and-evidence-links.md)) vs. stricter disruption/novelty gating ([`ADR-00014`](ADR-00014-sec-8k-boilerplate-sanitization-gating--.md)). |
| Absorbed vs. unabsorbed CAR separation | Calibrates default `buffer_days` and `share` bounds in [`contagion/parameters.py`](../../contagion/parameters.py) ([`ADR-00026`](ADR-00026-cascade-parameter-extraction-provenance-.md)). |

---

## 5. Invariants & Guardrails

1. **Zero overnight-gap credit**: Tradable performance metrics MUST begin at `Open(t_0)` or later; `Close(t_0 - 1) -> Open(t_0)` is reported solely as untradable reaction.
2. **Mandatory pre-event echo diagnostic**: Every study output MUST report `CAR[-3, 0]` alongside post-event horizons so lagging news archives cannot masquerade as causal predictors.
3. **Strict Point-in-Time graph & universe**: Indirect links $A \to B$ MUST be retrieved with `PiTContext(as_of=t_0, mode="historical")` ([`ADR-00033`](ADR-00033-bitemporal-pit-contract-for-read-paths--.md)) and evaluated against historical S&P 500 constituents from [`universe/sp500.py`](../../universe/sp500.py).
4. **Offline evaluation isolation**: The backtest harness lives strictly outside Domain 1 runtime serving and never adds return labels or trading recommendations to `OperationalContagionReport` ([`ADR-00010`](ADR-00010-directional-basket-peer-cluster-mapping-.md), [`ADR-00011`](ADR-00011-operational-contagion-report-schema-----.md)).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Unblocks immediate engineering execution of `ADR-00033`–`ADR-00037` on a clean theoretic baseline without losing sight of empirical market physics.
  - Provides an unambiguous, leak-free experimental protocol ready to validate and tune `MEMGRAPH_EVENT_TTL_DAYS`, news ingestion value, and cascade priors once curated data is acquired.
- **Negative / Trade-offs**:
  - Until the data prerequisites in Section 4.1 are acquired and this ADR is implemented, `MEMGRAPH_EVENT_TTL_DAYS = 90` and indirect contagion utility remain theoretic priors rather than empirically verified measurements.

---

## 7. Open Questions for Review (Upon Activation)

1. **Historical market data source**: When activating this study, which survivorship-bias-free daily Open/Close + delisted equity + earnings-date dataset will supplement `yfinance`?
2. **Calibration threshold**: What minimum sample size $N$ and statistical significance gate (e.g., $t$-stat / bootstrap CI adjusted for clustered event dates) should be required before changing production defaults such as `MEMGRAPH_EVENT_TTL_DAYS`?
