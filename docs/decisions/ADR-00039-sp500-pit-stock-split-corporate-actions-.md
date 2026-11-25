# ADR-00039: Point-in-Time S&P 500 Stock-Split & Reverse-Split Registry

- **ADR ID**: ADR-00039
- **Filename**: `ADR-00039-sp500-pit-stock-split-corporate-actions-.md`
- **Date**: 2026-11-24 21:50 UTC
- **Status**: PROPOSED
- **Target Subsystem**: `universe` / `ingestion` / `evaluation` / `tools`
- **Linked Commits**: Pending acceptance

---

## 1. Context & Problem Statement

1. **On-the-fly SEC EDGAR split inference is prohibitive during backtesting.** Inferring forward or reverse stock splits dynamically from raw SEC Form 8-K (`Item 5.03` / `Item 8.01`) narratives or `10-K`/`10-Q` XBRL tags (`us-gaap:StockholdersEquityNoteStockSplitConversionRatio1`) inside event-study loops ([`ADR-00038`](ADR-00038-event-study-backtest-and-signal-decay---.md)) or historical market-context builders ([`ingestion/market/harvester.py`](../../ingestion/market/harvester.py)) incurs $O(\text{events} \times \text{filings})$ I/O and parsing overhead while exposing backtests to noisy per-issuer XBRL formatting.
2. **Implicit `yfinance` price adjustment is opaque and survivorship-biased.** [`YahooFinanceClient.fetch_historical_ohlcv`](../../ingestion/market/client.py) currently calls `yf.download(...)` without pinning `auto_adjust`, relying implicitly on library-default backward adjustment. That implicit behavior:
   - Silently omits delisted, merged, or receivership S&P 500 constituents tracked in [`universe/sp500.py`](../../universe/sp500.py) (`SIVB`, `FRC`, `CELG`, `RTN`, `TWTR`, `XLNX`, etc.).
   - Conflates cash-dividend adjustments with share-split rescaling and overwrites historical nominal share prices needed to reconcile contemporaneous dollar figures cited in SEC filings and news text.
   - Provides zero structured metadata indicating *when* a split was declared or executed.
3. **Stock splits and reverse splits directly confound directional equity movement.** Beyond the mechanical $N:D$ share-count discontinuity at the `ex_date` open:
   - **Forward splits** ($N > D$, e.g., `NVDA` `10:1`, `AAPL` `4:1`, `GOOGL`/`GOOG` `20:1`, `AMZN` `20:1`) exhibit positive liquidity and signaling drift around both board declaration (`declared_date`) and execution (`ex_date`).
   - **Reverse splits** ($N < D$, e.g., `GE` `1:8` on `2021-08-02`, `AIG` `1:20`, `C` `1:10`) reflect corporate restructuring or distress and exhibit negative abnormal drift.
   Without an indexed local split table, [`MarketMetricsCalculator`](../../ingestion/market/parser.py) (`EXPANDING_BULLISH` / `CONTRACTING_BEARISH` / `DISRUPTIVE_SHOCK` polarity) and [`ADR-00038`](ADR-00038-event-study-backtest-and-signal-decay---.md) event-study regressions risk misattributing split-driven directional moves—or unadjusted mechanical jumps—to supply-chain contagion.

---

## 2. Decision Drivers & Constraints

1. **Zero network / zero EDGAR latency at backtest & query time**: all split lookups and cumulative factor calculations must execute in memory in $O(1)$ / $O(\log K)$ from a committed/cached local artifact (`data/sp500_stock_splits_historical.json`) mirrored in PostgreSQL (`sp500_stock_splits`).
2. **Dual-class & ticker-turnover identity safety**: the primary key must disambiguate dual-class S&P 500 shares sharing a single 10-digit SEC CIK (e.g., Alphabet `GOOGL` and `GOOG` under `0001652044`) while preserving corporate continuity across ticker renames (`FB` $\rightarrow$ `META`, `UTX`/`RTN` $\rightarrow$ `RTX`).
3. **Exact rational split arithmetic**: ratios must be stored as exact positive integers `(split_numerator, split_denominator)` rather than lossy floats alone (`1:3`, `3:2`, `1:8`, stock-dividend splits) so multi-split historical horizons compose without floating-point drift.
4. **Bitemporal (`ADR-00033`) discipline**: separate board announcement discovery (`declared_date`, `known_at`) from regular-session split-adjusted trading start (`ex_date`) per [`retrieval/temporal.py`](../../retrieval/temporal.py).
5. **Data-source governance**: offline table construction uses `yfinance` (`Ticker.splits`) for active symbols plus **one-time offline batch** SEC EDGAR Form 8-K / XBRL harvesting (`edgartools`) and curated turnover verification for delisted historical constituents—never querying `Finnhub` for historical data.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Parse SEC EDGAR Form 8-K / XBRL split disclosures on-the-fly during backtesting.** Rejected. Orders of magnitude too slow for multi-year cross-sectional backtests and brittle across issuer-specific XBRL extensions.
- **Alternative B: Rely exclusively on `yfinance` default `auto_adjust=True` without a dedicated split registry.** Rejected. Drops delisted S&P 500 constituents, mixes dividend reinvestment into price series, loses nominal point-in-time prices, and leaves split-announcement / ex-date directional drift completely unflagged.
- **Alternative C: Key `sp500_stock_splits` on `(cik, ex_date)` alone or `(ticker, ex_date)` alone.** Rejected. `(cik, ex_date)` collides whenever two S&P 500 share classes under one CIK split on the same date (`GOOGL` and `GOOG` on `2022-07-18`), whereas `(ticker, ex_date)` breaks under ticker reuse and corporate symbol changes.

---

## 4. Decision Outcome

### 4.1 Durable PostgreSQL Table (`sp500_stock_splits`) & Local Cache (`data/sp500_stock_splits_historical.json`)

Extend [`tools/init_db.py`](../../tools/init_db.py) and [`universe/sp500.py`](../../universe/sp500.py) (`SP500StockSplit` dataclass + `SP500UniverseManager`) with dual offline JSON + PostgreSQL persistence:

- **Composite Primary Key**: `PRIMARY KEY (cik, ticker, ex_date)` with `cik` normalized to 10 digits and `ticker` normalized to canonical uppercase hyphenated notation (`BRK-B`).
- **Columns & Constraints**:
  - `split_numerator INTEGER NOT NULL CHECK (split_numerator > 0)` ($N$: post-split shares)
  - `split_denominator INTEGER NOT NULL CHECK (split_denominator > 0)` ($D$: pre-split shares, `CHECK (split_numerator <> split_denominator)`)
  - `split_factor DOUBLE PRECISION NOT NULL` ($N / D > 1.0$ for forward splits; $N / D < 1.0$ for reverse splits)
  - `split_type VARCHAR(16) NOT NULL` (`FORWARD_SPLIT` when $N > D$; `REVERSE_SPLIT` when $N < D$)
  - `declared_date DATE NULL`, `known_at TIMESTAMPTZ NULL`, `known_at_imputed BOOLEAN NOT NULL DEFAULT FALSE` (announcement timestamp for pre-ex-date signaling drift under [`ADR-00033`](ADR-00033-bitemporal-pit-contract-for-read-paths--.md))
  - `source_provider VARCHAR(32) NOT NULL` (`yfinance_actions` | `sec_edgar_offline` | `curated_delisted`)
  - `accession_number VARCHAR(32) NULL` (statutory SEC Form 8-K / 10-K provenance when available)
  - Indexes on `(cik, ex_date)`, `(ticker, ex_date)`, and `(declared_date)`.

### 4.2 Offline Batch Split Builder & Exact Rational Factor Algebra

1. **Offline Builder (`universe/splits_builder.py`)**:
   - Iterates all current and historical turnover constituents in [`SP500UniverseManager`](../../universe/sp500.py).
   - Harvests split series for active tickers via `yfinance` (`Ticker.splits`), converting float ratios to exact `(split_numerator, split_denominator)` integers via bounded rational approximation (`Fraction(val).limit_denominator(10_000)`).
   - Backfills delisted historical S&P 500 constituents and enriches `declared_date`, `known_at`, and `accession_number` via an **offline batch** SEC EDGAR pass (`edgartools` Form 8-K `Item 5.03`/`8.01` + XBRL `StockholdersEquityNoteStockSplitConversionRatio1`), writing `data/sp500_stock_splits_historical.json` and syncing to PostgreSQL via `SP500UniverseManager.sync_splits_to_postgres()`.
2. **In-Memory Cumulative Split Factor & Confounder Lookup**:
   - For security $i$ over trading sessions $t_1 < t_2$, exact rational cumulative adjustment factor:
     $$F(i, t_1, t_2) = \prod_{s \in \mathcal{S}(i) :\, t_1 < d_{\text{ex}}(s) \le t_2} \frac{N_s}{D_s} \in \mathbb{Q}_{>0}$$
   - Given raw unadjusted prices $P_{\text{raw}}$, pure split-adjusted return (free of dividend-reinvestment distortion) is:
     $$R_{i,[t_1,t_2]}^{\text{split-adj}} = \frac{P_{\text{raw}}(i, t_2) \cdot F(i, t_1, t_2)}{P_{\text{raw}}(i, t_1)} - 1$$
   - `has_split_confounder(identifier, start_date, end_date, include_declared=True)` flags any event window intersecting either an `ex_date` or `actionable_session(known_at, declared_date)`.

### 4.3 Integration into Market Context & Event-Study Backtesting

1. **Explicit Adjustment Mode in [`YahooFinanceClient`](../../ingestion/market/client.py)**:
   - Pin `auto_adjust` explicitly (`auto_adjust=False` when pairing raw OHLCV with `F(i, t_1, t_2)`, or explicit `auto_adjust=True` when total-return series are requested) so library-version upgrades never silently flip price semantics.
2. **Confounder Metadata in [`MarketMetricsCalculator`](../../ingestion/market/parser.py) & [`MarketContextIntegrator`](../../ingestion/market/harvester.py)**:
   - Add `split_adjustment_factor`, `has_split_in_window`, and `split_confounder_type` (`FORWARD_SPLIT` | `REVERSE_SPLIT` | `None`) to `compute_event_window_metrics` and `sp500_market_context.parquet`.
3. **Event-Study Stratification in [`ADR-00038`](ADR-00038-event-study-backtest-and-signal-decay---.md)**:
   - Stratify or exclude event windows overlapping S&P 500 forward/reverse split announcements or ex-dates alongside quarterly earnings windows.

---

## 5. Invariants & Guardrails

1. **Zero On-the-Fly EDGAR Split Queries**: neither `MarketMetricsCalculator`, `YahooFinanceClient`, nor any backtest runner may query SEC EDGAR at runtime to infer stock splits; all runtime consumers read strictly from `SP500UniverseManager` (`data/sp500_stock_splits_historical.json` / `sp500_stock_splits`).
2. **Exact Integer Ratio Invariant**: every row must satisfy `split_numerator > 0`, `split_denominator > 0`, `split_numerator != split_denominator`, and `abs(split_factor - split_numerator / split_denominator) < 1e-12`.
3. **Dual-Class Non-Collision**: composite key `(cik, ticker, ex_date)` must preserve distinct split records for multi-class S&P 500 issuers (`GOOGL`/`GOOG`, `FOXA`/`FOX`, `NWSA`/`NWS`).
4. **Domain 1 Isolation**: split factors and return adjustments remain strictly inside `universe/`, `ingestion/market/`, `sft_pipeline/`, and `evaluation/`—never introducing equity price targets or trading signals into `OperationalContagionReport` ([`ADR-00010`](ADR-00010-directional-basket-peer-cluster-mapping-.md), [`ADR-00011`](ADR-00011-operational-contagion-report-schema-----.md)).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Eliminates mechanical split jump errors, delisted constituent split blind spots, and unmeasured split-announcement/ex-date drift confounders with $O(1)$ in-memory backtest lookups.
  - Preserves exact nominal historical share prices alongside exact rational split-adjusted returns.
- **Negative / Trade-offs**:
  - Requires periodic offline refreshes of `data/sp500_stock_splits_historical.json` via `universe/splits_builder.py` when new S&P 500 forward or reverse splits take effect.
