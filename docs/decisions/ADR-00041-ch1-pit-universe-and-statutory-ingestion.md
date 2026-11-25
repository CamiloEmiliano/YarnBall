# ADR-00041: Chapter 1 Technical Report — PiT Universe, Market Time & Statutory Ingestion

- **ADR ID**: ADR-00041
- **Filename**: `ADR-00041-ch1-pit-universe-and-statutory-ingestion.md`
- **Date**: 2026-11-24 19:08 UTC
- **Status**: ACCEPTED
- **Target Subsystem**: `docs` / `universe` / `ingestion` / `streaming`
- **Linked Commits**: [`480157f`](../../docs/technical-report/01-pit-universe-and-ingestion.md), [`480157f`](../../mkdocs.yml), [`480157f`](../../tests/docs/test_docs_integrity.py)

---

## 1. Context & Problem Statement

1. **Pre-ADR foundational ingestion modules lack formal quantitative & architectural documentation.** While [`ADR-00040`](ADR-00040-ch0-system-atlas-and-docs-integrity-----.md) establishes the 5-chapter Dual-Lens Technical Report architecture and Chapter 0 System Atlas, the upstream data-grounding boundary of YarnBall spans a deep mix of **pre-ADR foundational code** and **late-stage ADR refinements**:
   - Foundational modules predating [`ADR-00001`](ADR-00001-sit-flow-2-event-shock-contagion-harness.md)—including the S&P 500 historical constituent builder and universe manager ([`universe/builder.py`](../../universe/builder.py), [`universe/sp500.py`](../../universe/sp500.py)), market OHLCV / Cumulative Abnormal Return (CAR) / volatility $z$-score calculator ([`ingestion/market/parser.py`](../../ingestion/market/parser.py)), multi-source news harvesters and `TokenBucket` rate limiter ([`ingestion/news/client.py`](../../ingestion/news/client.py), [`ingestion/news/harvester.py`](../../ingestion/news/harvester.py), [`ingestion/news/parser.py`](../../ingestion/news/parser.py)), earnings transcript ingestors ([`ingestion/transcripts/harvester.py`](../../ingestion/transcripts/harvester.py)), APScheduler runner ([`tools/scheduler.py`](../../tools/scheduler.py)), and Kafka streaming + Dead-Letter Topic (DLT) pipeline ([`streaming/consumer.py`](../../streaming/consumer.py), [`streaming/producer.py`](../../streaming/producer.py))—have never been unified into a formal domain and systems specification.
   - Later ADRs layered bitemporal U.S. equity session cutoffs ([`ADR-00033`](ADR-00033-bitemporal-pit-contract-for-read-paths--.md)), Form 8-K boilerplate sanitization and statutory disruption gating ([`ADR-00014`](ADR-00014-sec-8k-boilerplate-sanitization-gating--.md), [`ADR-00027`](ADR-00027-canonical-staged-8k-envelope-and-catalog.md)), section-aware 800-token chunking with targeted 1-pass gleaning ([`ADR-00035`](ADR-00035-text-unit-chunk-layer-and-evidence-links.md)), and the Point-in-Time S&P 500 stock-split & reverse-split registry ([`ADR-00039`](ADR-00039-sp500-pit-stock-split-corporate-actions-.md)) directly onto those pre-ADR pipelines.
2. **Financial validity begins at the temporal, corporate-action, and statutory boundary.** In quantitative finance, downstream graph algebra and language-model reasoning are invalid if the input universe suffers from survivorship bias, if unindexed forward or reverse splits distort or confound directional return windows, if after-hours `16:05 ET` SEC filings leak into same-day close backtests, if PSLRA Safe Harbor boilerplate triggers false supply-chain shocks, or if live streaming APIs (`Finnhub`) are misused for multi-year historical backfills (`2018–2025`).
3. **Need for Blog Part 1 + AST Ingestion Reference.** Chapter 1 (`docs/technical-report/01-pit-universe-and-ingestion.md`) must explain these financial/statistical mechanisms as a standalone technical report (*Blog Part 1*) while providing AST-verified call graphs, topic-level UML intuition, and schema contracts for every module in `universe/`, `ingestion/`, `streaming/`, and [`retrieval/temporal.py`](../../retrieval/temporal.py).

---

## 2. Decision Drivers & Constraints

1. **Survivorship-free, split-clean & look-ahead-free rigor**: formalize S&P 500 point-in-time index membership (`[date_added, date_removed)`), exact rational forward/reverse split factors and drift confounders ([`ADR-00039`](ADR-00039-sp500-pit-stock-split-corporate-actions-.md)), and U.S. cash-equity regular-session close alignment (`16:00 ET` / `13:00 ET` early closes, NYSE Rule 7.2 holidays, unscheduled mourning closures).
2. **Strict historical vs. live data-source discipline**: explicitly document why historical backtesting and SFT construction (`2018–2025`) rely exclusively on SEC EDGAR (`10-K`, `10-Q`, `8-K`, `Exhibit 21`), `yfinance` OHLCV + offline split registry, local FNSPID Parquet archives, and cached transcripts, while `Finnhub` + Kafka streaming are strictly reserved for forward live operation.
3. **Dual-level visual intuition (`ADR-00040` §4.1)**: pair a clean **Part A Topic-Level UML Diagram** (`classDiagram`) showing conceptual transformations from raw market/regulatory disclosures into bitemporal chunks and shock assessments with **Part B AST Call Graphs** (`flowchart`) tracing exact Python execution paths.
4. **Honest Pre-ADR vs. ADR provenance & zero-drift verification**: distinguish pre-ADR implementations from ADR-governed refinements in the Theory-to-Code Crosswalk Table, and enforce all file, `module:symbol`, and constant citations via `tests/docs/test_docs_integrity.py`.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Document `retrieval/temporal.py` exclusively in Chapter 3 (Retrieval) and omit it from Chapter 1.** Rejected. Although [`retrieval/temporal.py`](../../retrieval/temporal.py) resides physically inside `retrieval/` (one of the shared-kernel couplings noted in [`ADR-00040`](ADR-00040-ch0-system-atlas-and-docs-integrity-----.md)), `actionable_session` and NYSE holiday/close rules are executed *at ingestion time* by [`ingestion/sec/harvester.py`](../../ingestion/sec/harvester.py) and [`knowledge_graph/chunk_store.py`](../../knowledge_graph/chunk_store.py) to stamp `valid_from` and `known_at`. Chapter 1 owns the market-calendar clock (`actionable_session`), while Chapter 3 references Chapter 1 and covers Cypher read-path filter injection (`TemporalQueryFilter`).
- **Alternative B: Split `streaming/` into a separate standalone chapter.** Rejected. [`streaming/consumer.py`](../../streaming/consumer.py) and [`streaming/producer.py`](../../streaming/producer.py) are the live transport envelope for `ingestion/news/` (`send_record` $\rightarrow$ Kafka $\rightarrow$ `process_streaming_event` $\rightarrow$ DLT), making them part of the same ingestion bounded context.
- **Alternative C: Document only SEC EDGAR ingestion and skip `market/`, `news/`, and `transcripts/`.** Rejected. Event-window CAR / volatility $z$-score polarity in [`ingestion/market/parser.py`](../../ingestion/market/parser.py), split adjustments/confounders ([`ADR-00039`](ADR-00039-sp500-pit-stock-split-corporate-actions-.md)), and historical-vs-live news/transcript boundaries are essential to understanding how YarnBall grounds events and constructs SFT datasets in Chapter 4.

---

## 4. Decision Outcome

Author **`docs/technical-report/01-pit-universe-and-ingestion.md`** (*Part 1: Point-in-Time Equity Universes, Market Time & Statutory Ingestion*) and register it in `mkdocs.yml` and `tests/docs/test_docs_integrity.py` with the following exact structure:

### 4.1 Part A — Domain & Theoretical Technical Report (*What & Why* / Blog Part 1)

1. **Survivorship Bias, Corporate Split Actions & Point-in-Time Equity Universe Construction**:
   - Financial failure mode of static constituent lists (omitting delisted, bankrupt, or acquired issuers such as SVB/First Republic/M&A exits and retroactively backfilling modern mega-caps into 2018 cohorts).
   - Formal half-open membership indicator over [`SP500Constituent`](../../universe/sp500.py):
     $$\mathbb{I}_{i \in \mathcal{U}(t)} = \mathbb{I}\!\left(d_i^{\text{add}} \le t\right) \cdot \mathbb{I}\!\left(d_i^{\text{rem}} = \varnothing \,\lor\, t < d_i^{\text{rem}}\right)$$
   - Point-in-Time Stock-Split & Reverse-Split Algebra ([`ADR-00039`](ADR-00039-sp500-pit-stock-split-corporate-actions-.md)): why on-the-fly EDGAR split parsing fails in backtests, why composite `(cik, ticker, ex_date)` keying is required for dual-class S&P 500 issuers (`GOOGL`/`GOOG`), exact rational cumulative split factor $F(i, t_1, t_2) = \prod_{t_1 < d_{\text{ex}}(s) \le t_2} (N_s / D_s)$, and separating mechanical ex-date share rescaling from forward-split bullish / reverse-split bearish drift confounders.
   - Identifier hygiene: 10-digit zero-padded SEC CIKs, dot-to-hyphen dual-class share normalization (`BRK.B` $\rightarrow$ `BRK-B`), primary exchange canonicalization, 11-sector GICS stratification, and lexical collision defense via `DICTIONARY_TICKERS` (`ALL`, `NOW`, `ON`, `IT`, `KEY`, `FAST`, `LOW`, single-char tickers).
2. **Market Microstructure Time, Exchange Calendars & Bitemporal Grounding (`ADR-00033`)**:
   - Why calendar-date equality (`filing_date <= query_date`) leaks after-hours disclosures (`16:01–21:30 ET` EDGAR acceptances) into regular-session close snapshots.
   - Formal actionable-session operator $\tau_{\text{act}}(t_{\text{known}}, d_{\text{doc}})$ in [`retrieval/temporal.py`](../../retrieval/temporal.py): NYSE/NASDAQ/Cboe holiday rules (NYSE Rule 7.2 Saturday/Sunday observation asymmetry for New Year's Day, Meeus-Jones-Butcher Good Friday, Juneteenth $\ge 2022$, and unscheduled mourning closures `2018-12-05` & `2025-01-09`), `16:00 ET` regular close vs. `13:00 ET` scheduled half-days, and bitemporal coordinates $(t_{\text{valid\_from}}, t_{\text{valid\_to}}, t_{\text{known\_at}})$.
3. **Statutory Disclosure Structure, Form 8-K Sanitization & Disruption Gating (`ADR-00014`, `ADR-00027`)**:
   - Anatomy of SEC Forms `10-K` (`Item 1`, `Item 1A`, `Item 7`, `Exhibit 21`), `10-Q`, and `8-K`, and why raw filings poison LLMs via PSLRA Safe Harbor risk enumerations (*"fire, flood, earthquake, cyberattack..."*) and signature/exhibit boilerplate.
   - Deterministic $O(N)$ paragraph sanitization, the 4-tier statutory precedence lattice (`Tier 1` acute operational halts $\succ$ `Tier 2` material agreements/other events $\succ$ `Tier 3` financial/Reg FD $\succ$ `Tier 4` governance/exhibits), lexical + statutory category resolution, and zero-shock negative-control gating (`shock_magnitude_pct = 0.0` when `is_material_operational_disruption = False`).
4. **Section-Aware Segmentation, Salience Ordering & Targeted 1-Pass Gleaning (`ADR-00035`)**:
   - Why fixed character truncation drops tail suppliers/customers in long `Item 1` / `Item 1A` disclosures, and why naive chunking loses issuer/period identity across chunk boundaries.
   - Conservative BPE token upper bound $\hat{T}(x) = \lfloor \max(1.35\,W(x),\, C(x)/3.7) + 1 \rfloor$, `800`-token window / `120`-token sliding overlap with immutable metadata headers, deterministic SHA-based `chunk_id` generation, non-truncating salience sorting (`rank_salient_chunks`), and the bounded Pass-2 gleaning trigger (`should_glean_chunk`: high-signal section $\land \neg\text{runaway} \land (|E_1| \ge 1 \lor T_{\text{gen}} \ge \lfloor 0.85\,T_{\max} \rfloor)$).
5. **Event-Window Market Econometrics & Historical-vs-Live Source Partitioning**:
   - Event-study abnormal return and volatility scaling in [`ingestion/market/parser.py`](../../ingestion/market/parser.py) (with split-confounder tagging per [`ADR-00039`](ADR-00039-sp500-pit-stock-split-corporate-actions-.md)):
     $$\text{CAR}_{i,[T-k,T+k]} = R_{i,[T-k,T+k]} - R_{m,[T-k,T+k]}, \qquad z_i = \frac{\text{CAR}_{i,[T-k,T+k]}}{\hat{\sigma}_i \sqrt{\Delta t}}$$
     and discrete volatility-scaled polarity regimes (`EXPANDING_BULLISH` $\ge +2.0$, `CONTRACTING_BEARISH` $\le -1.5$, `DISRUPTIVE_SHOCK` $\le -3.0$, `NEUTRAL_STABLE` otherwise).
   - Operational boundary between historical multi-year sources (`edgartools`, `yfinance` + offline `sp500_stock_splits_historical.json`, FNSPID Parquet, cached transcripts) and live streaming (`Finnhub` continuous `TokenBucket` $b(t) = \min(B, b(t_0) + r(t-t_0))$ at $60\text{ req/min}$, domain failure circuit breaker, Kafka + `<topic>.dlt` quarantine).
6. **Topic-Level UML Diagram & Theory-to-Code Crosswalk Table**:
   - Mermaid `classDiagram` modeling the Chapter-1 conceptual topics (`EquitiesUniverseAndSplitsPiT`, `MarketSessionClock`, `StatutoryFilingIngestor`, `DisruptionSanitizerGate`, `SectionChunkAndGleaner`, `MarketAndLiveEventStream`) and their typed outputs (`SP500Constituent`, `SP500StockSplit`, `PiTContext`, `SanitizedDisruptionAssessment`, `DocumentChunkRecord`, `MarketEventMetrics`).
   - Full Crosswalk Table mapping every mathematical symbol, statutory rule, and constant to `module:symbol`, provenance (`Pre-ADR Foundation` vs. `ADR-00003`/`00005`/`00014`/`00027`/`00033`/`00035`/`00039`), and unit/SIT test files.

### 4.2 Part B — Software Architecture, Call Graphs & Entry Points (*Where & How*)

1. **Subsystem Module Inventory**: complete file-by-file breakdown across `universe/` (`sp500.py`, `builder.py`, and planned `splits_builder.py` per [`ADR-00039`](ADR-00039-sp500-pit-stock-split-corporate-actions-.md)), `ingestion/sec/` (4 modules), `ingestion/segmentation/` (1 module), `ingestion/market/` (3 modules), `ingestion/news/` (3 modules), `ingestion/transcripts/` (3 modules), `ingestion/task_runner.py`, `streaming/` (3 modules), [`retrieval/temporal.py`](../../retrieval/temporal.py), [`tools/scheduler.py`](../../tools/scheduler.py), and [`tools/reextract_sp500_chunks.py`](../../tools/reextract_sp500_chunks.py).
2. **AST-Verified Execution Call Graphs (3 Mermaid `flowchart` pipelines)**:
   - *Pipeline 1 — SEC Form 10-K / Exhibit 21 Chunked Ingestion & Gleaning*: `EdgarWorker.process_company_10k` $\rightarrow$ `EdgarClient` $\rightarrow$ `EdgarEntityLinker.link_entity` $\rightarrow$ `actionable_session` $\rightarrow$ `FinancialSectionSegmenter.segment_filing` $\rightarrow$ `store_document_chunks` $\rightarrow$ `extract_sec_relationships` (Pass 1 + conditional Pass 2 gleaning) $\rightarrow$ `merge_chunk_extraction_payloads` $\rightarrow$ `_persist_sec_graph_data`.
   - *Pipeline 2 — SEC Form 8-K Sanitization & Disruption Gating*: `normalize_staged_8k_payload` $\rightarrow$ `sanitize_sec_8k_text` $\rightarrow$ `select_primary_sec_8k_item` $\rightarrow$ `assess_sec_8k_disruption` $\rightarrow$ `SanitizedDisruptionAssessment`.
   - *Pipeline 3 — Live News Scheduling, Rate-Limited Fetch & Kafka DLT Consumer*: `schedule_jobs` $\rightarrow$ `fetch_finnhub` (`TokenBucket.wait`) $\rightarrow$ `send_record` $\rightarrow$ `run_consumer` / `process_streaming_event` $\rightarrow$ `store_raw` + `embed_texts` + `store_graph_entities` (with `ConsumerError` / runtime failures routed to `send_to_dlt`).
3. **CLI Entry Points, Environment Variables & Persistence Contracts**:
   - CLI invocations (`universe/builder.py`, `universe/sp500.py`, `universe/splits_builder.py`, `ingestion/sec/client.py`, `ingestion/sec/harvester.py`, `ingestion/market/harvester.py`, `ingestion/transcripts/harvester.py`, `tools/scheduler.py`, `tools/reextract_sp500_chunks.py`).
   - Exact schema contracts for `sp500_historical_constituents`, `sp500_stock_splits` ([`ADR-00039`](ADR-00039-sp500-pit-stock-split-corporate-actions-.md)), `sec_filings_queue`, `sec_subsidiaries`, `document_chunks`, `financial_news_queue`, `domain_status`, staged 8-K JSON envelopes, and resumable chunk re-extraction JSONL checkpoints.

---

## 5. Invariants & Guardrails

1. **All `ADR-00040` Series Invariants Apply**: zero host-specific paths, pure GFM + `$...$`/`$$...$$` + `mermaid` in `01-pit-universe-and-ingestion.md` (zero `mkdocstrings` `:::` directives in narrative prose), AST-verified call-graph edges, and honest `Pre-ADR Foundation` vs. `ADR-XXXXX` attribution.
2. **Chapter-1 Constant & Symbol Verification**: `tests/docs/test_docs_integrity.py` must verify all `module:symbol` references in Chapter 1's Crosswalk Table alongside Chapter 1 constants (`REGULAR_CLOSE_ET = 16:00`, `EARLY_CLOSE_ET = 13:00`, `max_tokens = 800`, `overlap_tokens = 120`, gleaning `0.85` threshold, `TokenBucket` `60/60`, CAR $z$-score thresholds `+2.0`/`-1.5`/`-3.0`, and `DEFAULT_MAX_FILING_CHARS = 50_000`).
3. **Data-Source Governance Fidelity**: Chapter 1 must explicitly reinforce the project-wide prohibition against querying `Finnhub` or `yfinance.Ticker.news` for multi-year historical backfills or SFT generation.

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Formalizes the complete temporal, corporate-action, statutory, and econometric foundation of YarnBall—bridging pre-ADR `universe/`, `ingestion/`, and `streaming/` code with `ADR-00014`, `ADR-00027`, `ADR-00033`, `ADR-00035`, and `ADR-00039` in a single blog-ready chapter (`Part 1`).
  - Establishes clean handoff boundaries to Chapter 2 (`ADR-00042`: how extracted triples and entities pass through the 5-axis ontology, entity resolver, and Postgres/Memgraph ledger), Chapter 3 (`ADR-00043`: how `PiTContext` and `SanitizedDisruptionAssessment` drive hybrid retrieval and cascade propagation), and Chapter 4 (`ADR-00044`: how CAR/$z$-scores and quarantined extractions feed SFT/GRPO curation).
- **Negative / Trade-offs**:
  - Changes to market-close rules, split-table schema, chunking defaults, Form 8-K priority tiers, or CAR polarity thresholds will require synchronized updates to `01-pit-universe-and-ingestion.md` to pass `test_docs_integrity.py`.
