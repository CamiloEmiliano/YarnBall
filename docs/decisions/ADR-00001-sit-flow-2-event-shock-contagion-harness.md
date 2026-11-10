# ADR-00001: SIT Flow 2 Event Shock Vector & Contagion

<!--
STRICT FILENAME CONVENTION:
- Format: ADR-[5_DIGIT_ID]-[40_CHARACTER_SLUG].md
- Total Filename Length: Exactly 53 characters (including '.md')
- Slug Specification: Exactly 40 characters using lowercase alphanumeric and hyphens.
- Padding Rule: If the natural slug is under 40 characters, pad with trailing hyphens ('-').
Filename: ADR-00001-sit-flow-2-event-shock-contagion-harness.md (53 chars)
-->

- **ADR ID**: ADR-00001
- **Filename**: `ADR-00001-sit-flow-2-event-shock-contagion-harness.md`
- **Date**: 2026-11-09 22:24
- **Status**: PROPOSED
- **Target Subsystem**: `retrieval` / `ingestion` / `knowledge_graph`
- **Linked Commits**: *Pending implementation*

---

## 1. Context & Problem Statement

In financial knowledge graphs, natural language disclosures (such as SEC Form 8-K Item 8.01 Other Events or news alerts) describe operational disruptions (foundry halts, supplier bankruptcies, export bans). However, relying purely on unstructured text extraction to evaluate shock propagation creates two severe failure modes:
1. **Uncalibrated Severity**: An LLM cannot reliably distinguish between a minor transient delay and a catastrophic contract breach without quantitative market grounding.
2. **Topological Attenuation Deficiencies**: As demonstrated in Phase 5 torture testing (`TT-02A`), language models evaluate supply paths independently and fail to compute additive parallel loss (11% pass rate on diamond compounding).

System Integration Test (SIT) **Flow 2** exists to integrate and verify the complete cross-package pipeline:
`SEC 8-K Event + yfinance Historical OHLCV -> Cumulative Abnormal Return (CAR) z-score -> 5-Axis Shock Vector -> Multi-Hop Knowledge Graph Traversal & Attenuation`.

---

## 2. Decision Drivers & Constraints

1. **Strict Data Source Governance (`.agents/rules/data_sources.md`)**:
   - Zero live external network calls during test execution.
   - Strictly NO queries to the Finnhub API (reserved exclusively for production forward streaming).
   - Strictly NO live web scrapers.
2. **Offline Determinism & Performance Budget**:
   - Tier 2 SIT tests must execute 100% offline within `< 2.0 seconds` using small committed micro-fixtures (< 50 KB).
3. **Point-in-Time Temporal Consistency**:
   - Event pricing windows (`[t_event - 1, t_event + 1]`) and historical volatility baselines must strictly precede or match the 8-K disclosure timestamp with zero future lookahead.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Live API Fetching at Test Time (`yfinance.download()`)**:
  - *Rejected*: External network dependencies cause non-deterministic test failures, rate-limiting stalls, and CI/CD fragility. Violates the offline hermetic test principle.
- **Alternative B: Qualitative Text-Only Shock Prompting (LLM Guessing Severity)**:
  - *Rejected*: LLMs exhibit hallucination and severe mode-bleed under numerical reasoning. Empirical benchmark results show qualitative reasoning fails to compound convergent shocks.
- **Alternative C: Unattenuated All-Pairs BFS Graph Diffusion**:
  - *Rejected*: Real supply chains possess physical inventory buffers (e.g. 90-day substrate stockpiles) and capacity substitution. Shocks must attenuate geometrically across hops.

---

## 4. Decision Outcome

We establish **Flow 2** as a deterministic, offline pipeline integration test in `tests/integration/pipelines/test_event_shock_propagation_flow.py` with the following architectural components:

1. **Static Micro-Fixtures (`tests/fixtures/`)**:
   - `tests/fixtures/filings/aapl_8k_disruption.json` (~10 KB): A canonical Item 8.01 disclosure detailing a critical component supplier disruption.
   - `tests/fixtures/market/aapl_nvda_ohlcv.parquet` (~15 KB): 60 trading days of offline daily OHLCV prices spanning the event window.
2. **Quantitative Shock Vector Calculation**:
   - Ingest 8-K event timestamp and compute Cumulative Abnormal Return (CAR) over event window `[-1, +1]`:
     `CAR = sum(R_stock - R_market)`
   - Compute historical 30-day volatility standard deviation `sigma` and calculate normalized event z-score:
     `z_score = CAR / (sigma * sqrt(3))`
   - Construct formal 5-Axis Shock Vector:
     `{direction: 'CONTRACTING_BEARISH', magnitude_pct: abs(CAR), duration_days: 90, confidence: 0.95, channel: 'SUPPLY_CHAIN'}`.
3. **Multi-Hop Propagation Invariant**:
   - 1-hop downstream partner (`(Supplier)-[:SUPPLIES_TO]->(Tier1_Customer)`): Receives shock modulated by supplier revenue exposure ratio and inventory buffer days.
   - 2-hop downstream customer (`(Tier1_Customer)-[:SUPPLIES_TO]->(Tier2_Customer)`): Receives attenuated shock decayed by buffer absorption factor.

---

## 5. Invariants & Guardrails

1. **Offline Isolation Invariant**: `test_event_shock_propagation_flow.py` must never invoke `urllib`, `requests`, or socket connections to external market APIs.
2. **Monotonic Hop Decay Invariant**: For a single linear chain `A -> B -> C`, the propagated shock intensity must strictly decrease:
   `magnitude(C) < magnitude(B) <= magnitude(A)`.
3. **Buffer Delay Invariant**: An active inventory buffer of `B` days must mathematically delay shock onset by `B` calendar days in the propagated downstream payload.

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Proves that SEC regulatory events, quantitative market volatility, and property graph traversals interoperate seamlessly across package seams.
  - Provides the empirical mathematical benchmark required before implementing Phase 7 (Sparse Iterative Network Cascade).
- **Negative / Trade-offs**:
  - Requires maintaining static Parquet micro-fixtures in `tests/fixtures/market/`.

