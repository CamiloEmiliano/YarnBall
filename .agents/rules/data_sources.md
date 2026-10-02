---
trigger: always_on
description: Core data source and historical SFT generation constraints for YarnBall.
---

## YarnBall Data Source and SFT Generation Rules

### 1. Finnhub Data Constraints
- **Do NOT query or assume Finnhub for historical SFT dataset construction or multi-year backtesting (2018–2025)**: Finnhub's free tier history is limited to ~1 year and will fail on multi-year historical queries.
- **Finnhub Role**: Strictly reserved for live, real-time forward streaming in production runs.

### 2. Historical Data & SFT Dataset Grounding
- **Regulatory Disclosures (Primary Ground Truth)**: Use SEC EDGAR public filings (Form 10-K, Form 10-Q, Exhibit 21 subsidiaries, Form 4) via `edgartools` / `EdgarClient` (free public depth back to 2018+).
- **Historical Corporate News & Events**: Sourced exclusively from SEC Form 8-K material current reports (mandatory legal disclosures of material events, earnings, contracts, and litigation) and open archives (FNSPID / local parquet archives).
- **Market Pricing & Event Volatility**: Historical daily OHLCV prices, Cumulative Abnormal Returns (CAR), and volatility z-scores must be pulled via Yahoo Finance (`yfinance`), which provides full 2018–2025 depth.
- **Transcripts**: Corporate quarterly earnings call transcripts via local/cached archives.

### 3. Dry Run Data Sources Policy
- **Permitted / Standard Data Sources for Dry Runs**:
  - SEC EDGAR public filings (Form 10-K, 10-Q, 8-K) via `edgartools`.
  - Historical daily OHLCV prices and CAR metrics via `yfinance`.
  - Historical corporate news and market commentary via SEC Form 8-K releases and local FNSPID parquet archives.
  - S&P 500 point-in-time constituent registry (`sp500_constituents_historical.json`).
- **Excluded from Dry Runs**:
  - Do NOT query Finnhub API (reserved for production forward streaming).
  - Do NOT use `yfinance.Ticker.news` or live web scrapers (FNSPID is collinear with online news feeds; live scraping is ephemeral, paywalled, and unneeded).
