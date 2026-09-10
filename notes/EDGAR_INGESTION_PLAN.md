# SEC EDGAR Regulatory Ingestion & Knowledge Graph Extraction Plan

This document provides the complete, exhaustive technical design, database schemas, algorithm specifications, and integration workflows for ingesting **SEC EDGAR regulatory filings** (10-K, 10-Q, 8-K, Form 4, 13F) into YarnBall using **`edgartools`** ([`dgunning/edgartools`](https://github.com/dgunning/edgartools)), backed by a **PostgreSQL Master CIK Registry (`sec_companies`)** with `pg_trgm` GIN trigram indexing and **S&P 500 / High-Liquidity Ingestion Scoping**.

---

## 1. Executive Summary & Value Proposition

1. **Audited Ground-Truth Relationships**: Financial news provides real-time sentiment and rumors; SEC filings provide legally binding, audited disclosures signed under penalty of perjury.
2. **Deterministic Entity Anchoring with CIK**: Every U.S. public company and foreign private issuer is assigned an immutable 10-digit Central Index Key (CIK), permanently eliminating corporate name and ticker ambiguity.
3. **Dual Ingestion Universe**:
   - **Master CIK Registry (`sec_companies`)**: Stores all ~10,500 active public companies (< 2.5 MB disk/RAM) so that any supplier, customer, or partner mentioned in an annual report resolves immediately.
   - **Scoped Ingestion (`is_sp500 = TRUE` / `FINNHUB_TICKERS`)**: Heavy filing downloads and LLM extraction are strictly focused on **S&P 500 large caps (~85% of total market cap & volume)** and your configured community tickers, eliminating micro-cap noise.
4. **Automated Temporal Tracking**: Extracted relationships automatically inherit audited temporal validity: `fiscal_year`, `filing_date`, `period_end`, `valid_from`, `valid_to`, `is_current`, `accession_number`, and `source_url`.

---

## 2. High-Impact Filing Taxonomy & Extraction Targets

| Filing Type | Section / Exhibit | Extracted Intelligence & Graph Relationships |
| :--- | :--- | :--- |
| **10-K (Annual)** | **Item 1 (Business)** | - **Customer Concentration**: Under ASC 280, companies must disclose any customer representing $\ge 10\%$ of revenue $\to$ $100\%$ verified `CUSTOMER_OF` and `SUPPLIES_TO` edges.<br>- **Supplier & Foundry Dependencies**: Disclosures of sole-source suppliers (e.g. TSMC, ASML).<br>- **Product Lines**: Hardware, software, and cloud revenue categories $\to$ `PRODUCES` edges. |
| **10-K (Annual)** | **Item 1A (Risk Factors)** | - Discrete corporate and macroeconomic vulnerabilities (e.g., geopolitical supply chain chokepoints, raw material pricing, regulatory scrutiny) $\to$ `EXPOSED_TO` edges and risk nodes. |
| **10-K (Annual)** | **Item 7 (MD&A)** | - Capital expenditures (CapEx), supply commitments, research and development (R&D) investments. |
| **10-K (Annual)** | **Exhibit 21** | - Complete legal corporate subsidiary tree $\to$ `SUBSIDIARY_OF` edges with ownership percentage and jurisdiction. |
| **8-K (Material Events)** | **Item 1.01** | - Entry into a Material Definitive Agreement (Joint Ventures, Multi-Year Supply Contracts, Strategic Alliances) $\to$ `PARTNERED_WITH` edges with effective and expiration dates. |
| **8-K (Material Events)** | **Item 2.01** | - Completion of Acquisition or Disposition of Assets (M&A) $\to$ `ACQUIRED` edges with transaction value and closing date. |
| **8-K (Material Events)** | **Item 5.02** | - Departure or Appointment of Principal Officers or Directors $\to$ `LEADS`, `SERVES_ON_BOARD_OF`, `RESIGNED_FROM` edges. |
| **Form 4** | **Table I & II** | - Insider transactions (CEOs, CFOs, Board Members buying/selling equity) $\to$ `TRANSACTED` edges with share count and price. |
| **13F / 13D** | **Holdings Table** | - Institutional fund holdings (Vanguard, BlackRock, Berkshire Hathaway) $\to$ `INVESTS_IN` and `HOLDS_STAKE` edges ($>5\%$ activist stakes). |

---

## 3. The 4-Tier Hierarchical Entity Linking Cascade

When an entity is mentioned in a news headline, user query, or unstructured text (*"Taiwan Semi"*, *"$AAPL"*, *"Waymo"*, *"Alphabet Inc"*), it is resolved to official SEC descriptors (`CIK`, `ticker`, `legal_title`, `sic_code`) through a 4-tier cascade:

```
                     News Article Entity Reference / Query 
                   (e.g., "$AAPL", "Taiwan Semi", "Waymo")
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │ TIER 1: Exact SQL Ticker Match (`ticker = 'AAPL'`)      │
        │ - 100% precision instant lookup                        │
        └────────────────────────────┬───────────────────────────┘
                                     │ (If no exact ticker)
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │ TIER 2: Exact Suffix-Normalized SQL Match               │
        │ - Strip legal suffixes (`Inc`, `Corp`)                 │
        │ - `normalized_name = 'apple'`                          │
        └────────────────────────────┬───────────────────────────┘
                                     │ (If no exact name match)
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │ TIER 3: SQL Trigram Similarity (`pg_trgm` GIN Index)   │
        │ - `WHERE normalized_name % 'taiwan semi'`              │
        │ - Orders by `similarity()` desc (threshold > 0.88)     │
        └────────────────────────────┬───────────────────────────┘
                                     │ (If subsidiary)
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │ TIER 4: Exhibit 21 Subsidiary Tree Mapping             │
        │ - Maps subsidiary mentions (Waymo, DeepMind, AWS) back │
        │   to parent CIK (Alphabet Inc, Amazon.com Inc)         │
        └────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                 [ SEC Anchor: CIK 0000320193 | AAPL ]
```

### Tier 2 Legal Corporate Suffixes Stripped:
`inc`, `inc.`, `incorporated`, `corp`, `corp.`, `corporation`, `llc`, `l.l.c.`, `ltd`, `ltd.`, `limited`, `plc`, `p.l.c.`, `sa`, `s.a.`, `nv`, `n.v.`, `co`, `co.`, `company`, `holdings`, `holding`, `group`, `gmbh`, `ag`, `se`, `lp`, `l.p.`

---

## 4. Complete PostgreSQL Schema Specifications

The following tables and extensions will be added to `tools/init_db.py`:

```sql
-- 1. Enable Trigram Extension for fast fuzzy string matching
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 2. Master SEC Public Company Dictionary (~10,500 rows, < 2.5 MB)
CREATE TABLE IF NOT EXISTS sec_companies (
    cik VARCHAR(10) PRIMARY KEY,              -- 10-digit zero-padded CIK (e.g. '0000320193')
    ticker VARCHAR(10),                        -- Primary exchange ticker (e.g. 'AAPL')
    company_name TEXT NOT NULL,                -- Official SEC legal name (e.g. 'Apple Inc.')
    normalized_name TEXT NOT NULL,             -- Lowercase suffix-stripped name (e.g. 'apple')
    exchange VARCHAR(20),                      -- 'Nasdaq', 'NYSE', 'CBOE', etc.
    is_sp500 BOOLEAN DEFAULT FALSE,         -- S&P 500 index membership flag
    market_cap_tier VARCHAR(20),             -- 'Mega', 'Large', 'Mid', 'Small'
    sic_code VARCHAR(10),                      -- Standard Industrial Classification (e.g. '3571')
    sic_description TEXT,                      -- Industry category (e.g. 'Electronic Computers')
    fiscal_year_end VARCHAR(4),                -- e.g. '0930' for September 30
    state_of_incorporation VARCHAR(10),
    last_synced_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Fast B-Tree & GIN Trigram Indexes
CREATE INDEX IF NOT EXISTS idx_sec_ticker ON sec_companies(ticker);
CREATE INDEX IF NOT EXISTS idx_sec_normalized_name ON sec_companies(normalized_name);
CREATE INDEX IF NOT EXISTS idx_sec_sp500 ON sec_companies(is_sp500) WHERE is_sp500 = TRUE;
CREATE INDEX IF NOT EXISTS idx_sec_name_trgm ON sec_companies USING gin (normalized_name gin_trgm_ops);

-- 3. Corporate Subsidiaries Registry (Populated from 10-K Exhibit 21)
CREATE TABLE IF NOT EXISTS sec_subsidiaries (
    id SERIAL PRIMARY KEY,
    parent_cik VARCHAR(10) REFERENCES sec_companies(cik) ON DELETE CASCADE,
    subsidiary_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    jurisdiction TEXT,
    ownership_percentage NUMERIC DEFAULT 100.0,
    fiscal_year INT NOT NULL,
    accession_number VARCHAR(30) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT uq_sec_subsidiary UNIQUE (parent_cik, normalized_name, fiscal_year)
);
CREATE INDEX IF NOT EXISTS idx_sec_sub_name ON sec_subsidiaries(normalized_name);
CREATE INDEX IF NOT EXISTS idx_sec_sub_trgm ON sec_subsidiaries USING gin (normalized_name gin_trgm_ops);

-- 4. Parsed Regulatory Filing Sections & Provenance Queue
CREATE TABLE IF NOT EXISTS sec_filings_queue (
    id SERIAL PRIMARY KEY,
    cik VARCHAR(10) REFERENCES sec_companies(cik) ON DELETE CASCADE,
    ticker VARCHAR(10),
    company_name TEXT,
    filing_type VARCHAR(10) NOT NULL,          -- '10-K', '10-Q', '8-K', 'Form 4'
    accession_number VARCHAR(30) NOT NULL,     -- SEC unique accession (e.g. '0000320193-25-000106')
    filing_date DATE NOT NULL,
    report_period DATE NOT NULL,
    fiscal_year INT,
    fiscal_period VARCHAR(10),                 -- 'FY', 'Q1', 'Q2', 'Q3'
    section_name VARCHAR(50) NOT NULL,         -- 'Item 1', 'Item 1A', 'Item 2.01', 'Exhibit 21'
    clean_text TEXT NOT NULL,                  -- Clean extracted narrative
    sec_url TEXT NOT NULL,                     -- Official iXBRL/EDGAR document link
    status VARCHAR(20) DEFAULT 'pending',      -- 'pending', 'extracted', 'failed'
    extracted_edges_count INT DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT uq_sec_filing_section UNIQUE (accession_number, section_name)
);
CREATE INDEX IF NOT EXISTS idx_sec_filings_cik ON sec_filings_queue(cik);
CREATE INDEX IF NOT EXISTS idx_sec_filings_ticker ON sec_filings_queue(ticker);
CREATE INDEX IF NOT EXISTS idx_sec_filings_status ON sec_filings_queue(status);
```

---

## 5. End-to-End Pipeline Architecture

```
  +-----------------------------------------------------------------+
  |              SEC EDGAR Master Publication Files                 |
  |  https://www.sec.gov/files/company_tickers_exchange.json        |
  +--------------------------------+--------------------------------+
                                   | Bulk Sync (< 1 sec)
                                   v
  +-----------------------------------------------------------------+
  |          PostgreSQL Master Registry: 'sec_companies'            |
  |  - CIK (PK), Ticker, Company Name, Normalized Name, Exchange    |
  |  - is_sp500 BOOLEAN, market_cap_tier VARCHAR                    |
  |  - GIN Index on Trigrams: (normalized_name gin_trgm_ops)        |
  +--------------------------------+--------------------------------+
                                   |
         ┌─────────────────────────┴─────────────────────────┐
         | Fast SQL-Level Entity Linking (4-Tier Cascade)    |
         v                                                   v
  [ Ingested News Tickers / Mentions ]           [ User Query in UI ]
         │                                                   │
         └─────────────────────────┬─────────────────────────┘
                                   v Filtered by S&P 500 / Target Community
  +-----------------------------------------------------------------+
  |             edgartools Pipeline (ingest/edgar_client.py)        |
  |  - set_identity("YarnBallResearch admin@yarnball.org")          |
  |  - Company("AAPL").get_filings(form=["10-K", "8-K"])            |
  |  - Extracts tenk['Item 1'], tenk['Item 1A'], eightk.items       |
  +--------------------------------+--------------------------------+
                                   |
                                   v Emits to Kafka: 'sec_filings_raw'
  +-----------------------------------------------------------------+
  |             PostgreSQL Storage ('sec_filings_queue')            |
  |  - Stores section text with UNIQUE(accession_number, section)   |
  |  - Carries official SEC EDGAR URLs for provenance citations     |
  +--------------------------------+--------------------------------+
                                   |
                                   v
  +-----------------------------------------------------------------+
  |             Knowledge Graph Extraction Worker                   |
  |  - Deterministic Rule Extraction (Exhibit 21, Customer tables)  |
  |  - Guarded LLM Extraction on Business/Risk Text (qwen3:8b)      |
  +--------------------------------+--------------------------------+
                                   |
                                   v Merges into Memgraph
  +-----------------------------------------------------------------+
  |                Memgraph Knowledge Graph State                   |
  |  - (:Company {cik: '0000320193', ticker: 'AAPL', name: '...'})  |
  |  - -[:SUPPLIES_TO {source: '10-K', fiscal_year: 2025}]->        |
  |  - -[:SUBSIDIARY_OF {source: 'Ex-21', ownership: 1.0}]->        |
  |  - -[:ACQUIRED {source: '8-K Item 2.01', date: '2024-06-12'}]-> |
  +-----------------------------------------------------------------+
```

---

## 6. Implementation Modules & Code Specifications

### 6.1. `ingest/edgar_linker.py` (Entity Linking & S&P 500 Sync Engine)
- **`normalize_name(name: str) -> str`**: Lowercases, strips punctuation, and removes corporate suffixes.
- **`sync_sec_companies()`**: Downloads `company_tickers_exchange.json` from the SEC and bulk-upserts ~10,500 records into `sec_companies`.
- **`sync_sp500_list()`**: Fetches the active S&P 500 constituents (e.g. from Wikipedia / SEC index) and updates `is_sp500 = TRUE` in `sec_companies`.
- **`resolve_entity(query_text: str, ticker_hint: Optional[str] = None) -> Optional[Dict[str, Any]]`**:
  - **Tier 1**: `SELECT * FROM sec_companies WHERE ticker = UPPER(%s) LIMIT 1`
  - **Tier 2**: `SELECT * FROM sec_companies WHERE normalized_name = %s LIMIT 1`
  - **Tier 3**: `SELECT *, similarity(normalized_name, %s) AS score FROM sec_companies WHERE normalized_name % %s AND similarity(normalized_name, %s) >= 0.88 ORDER BY score DESC LIMIT 1`
  - **Tier 4**: `SELECT p.* FROM sec_subsidiaries s JOIN sec_companies p ON s.parent_cik = p.cik WHERE s.normalized_name = %s OR s.normalized_name % %s ORDER BY similarity(s.normalized_name, %s) DESC LIMIT 1`

### 6.2. `ingest/edgar_client.py` (Filing Downloader with `edgartools`)
- Integrates `edgartools` with `set_identity(os.getenv("SEC_EDGAR_USER_AGENT", "YarnBallResearch admin@yarnball.org"))`.
- **`fetch_company_10k_sections(ticker_or_cik: str, fiscal_year: Optional[int] = None) -> List[Dict[str, Any]]`**:
  - Accesses `tenk['Item 1']` (Business / Customer concentration), `tenk['Item 1A']` (Risk Factors), and `tenk['Item 7']` (MD&A).
  - Accesses `tenk.exhibits` for Exhibit 21 (Subsidiaries).
- **`fetch_recent_8k_events(ticker_or_cik: str, days_back: int = 90) -> List[Dict[str, Any]]`**:
  - Iterates over recent 8-K filings and extracts Item 1.01 (Material Agreements), Item 2.01 (M&A), Item 5.02 (Executive changes).
- **`fetch_sp500_universe(form_types=["10-K", "8-K"])`**:
  - Batch helper for ingesting all active S&P 500 members.

### 6.3. `ingest/edgar_worker.py` (Knowledge Graph Builder & Memgraph Integrator)
- Consumes structured filing objects from `sec_filings_raw` or direct batch execution.
- **Deterministic Rule Extraction**:
  - Parses Exhibit 21 table into `sec_subsidiaries` and creates `(:Company)-[:SUBSIDIARY_OF {ownership: 1.0, source: 'Exhibit 21'}]->(:Company)` in Memgraph.
- **Guarded LLM Extraction (Ollama `qwen3:8b`)**:
  - Runs structured JSON extraction on Item 1 Business narrative to extract verified `SUPPLIES_TO`, `CUSTOMER_OF`, `PARTNERED_WITH`, `COMPETES_WITH` edges.
- **Automated Temporal Stamping**:
  - Stamps all edges with `fiscal_year`, `filing_date`, `period_end`, `valid_from`, `valid_to`, `is_current`, `accession_number`, and `source_url`.
- **Memgraph Parameterized Batch UNWIND**:
  - Idempotently merges nodes and edges using Cypher `MERGE`.

---

## 7. Knowledge Graph Schema in Memgraph

### Entity Vertices:
```cypher
(:Company {
    id: "Apple Inc.",
    cik: "0000320193",
    ticker: "AAPL",
    exchange: "Nasdaq",
    is_sp500: true,
    market_cap_tier: "Mega",
    sic_code: "3571",
    sic_description: "Electronic Computers"
})
```

### Typed Temporal Edges:
```cypher
(asml:Company {ticker: 'ASML'})-[:SUPPLIES_TO {
    component: "Extreme Ultraviolet (EUV) Lithography",
    fiscal_year: 2025,
    filing_date: "2025-10-31",
    period_end: "2025-09-30",
    valid_from: "2024-10-01",
    valid_to: "2025-09-30",
    is_current: true,
    confidence: 1.0,
    source_filing: "10-K (0000320193-25-000106)",
    source_url: "https://www.sec.gov/ix?doc=/Archives/edgar/data/320193/000032019325000106/aapl-20250927.htm"
}]->(tsm:Company {ticker: 'TSM'})
```

---

## 8. Verification & Testing Plan

### Automated Unit Tests
1. **`tests/test_edgar_linker.py`**:
   - `test_sync_sec_companies_populates_postgres`: Verifies bulk sync into `sec_companies`.
   - `test_tier_1_ticker_exact_match`: Verifies `AAPL` $\to$ `0000320193`.
   - `test_tier_2_suffix_normalization`: Verifies `"Alphabet Inc."`, `"Microsoft Corporation"`, `"Tesla, LLC"` resolve to their official CIKs.
   - `test_tier_3_trigram_fuzzy_match`: Verifies `"Taiwan Semi"` $\to$ `0001046179` and `"JPMorgan"` $\to$ `0000019617`.
   - `test_tier_4_subsidiary_resolution`: Verifies `"Waymo"` resolves to Alphabet CIK `0001652044`.
2. **`tests/test_edgar_client.py`**:
   - `test_edgartools_initialization`: Verifies User-Agent and identity compliance.
   - `test_fetch_10k_item_extraction`: Verifies clean extraction of Item 1 and Item 1A from a sample 10-K fixture.
   - `test_fetch_8k_item_extraction`: Verifies extraction of Item 2.01 (M&A) from an 8-K fixture.
3. **`tests/test_edgar_worker.py`**:
   - `test_deterministic_exhibit_21_ingestion`: Verifies `SUBSIDIARY_OF` edges in Memgraph.
   - `test_temporal_edge_property_stamping`: Verifies `fiscal_year`, `valid_from`, `valid_to`, `is_current` properties.

### Live End-to-End Verification
1. Run `python tools/init_db.py` to create `sec_companies`, `sec_subsidiaries`, and `sec_filings_queue`.
2. Run `python -m ingest.edgar_linker --sync` to populate the master registry.
3. Ingest latest 10-K filings for Apple (`AAPL`) and Nvidia (`NVDA`).
4. Launch Chainlit (`chainlit run rag/app.py -w`) and query:
   - *"What suppliers and customer concentrations are disclosed in Apple's latest 10-K filing?"*
   - *"Who are Nvidia's verified subsidiaries from Exhibit 21?"*
5. Verify that responses include verified multi-hop paths in PyVis and clickable official SEC EDGAR citation links.
