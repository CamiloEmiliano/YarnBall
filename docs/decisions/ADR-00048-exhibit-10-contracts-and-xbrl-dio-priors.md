# ADR-00048: Phase 2b Operational Exhibit 10 Material Contracts & Offline XBRL DIO / ASC 280 Priors

- **ADR ID**: ADR-00048
- **Filename**: `ADR-00048-exhibit-10-contracts-and-xbrl-dio-priors.md`
- **Date**: 2026-11-28 06:29 UTC
- **Status**: ACCEPTED
- **Target Subsystem**: `ingestion` / `contagion` / `ontology` / `retrieval` / `tools` / `tests`
- **Linked Commits**: [`865f4db`](../../ingestion/sec/xbrl.py), [`865f4db`](../../ingestion/sec/client.py), [`865f4db`](../../ingestion/sec/harvester.py), [`865f4db`](../../ingestion/sec/canon.py), [`865f4db`](../../contagion/parameters.py), [`865f4db`](../../ontology/validation.py), [`865f4db`](../../tools/init_db.py), [`865f4db`](../../tests/ingestion/sec/test_sec_phase2b_ex10_xbrl.py)

---

## 1. Context & Problem Statement

[`ADR-00045`](ADR-00045-canonical-sec-edgar-filing-registry-----.md) established YarnBall's canonical SEC registry, [`ADR-00046`](ADR-00046-sec-core-sections-amendments-exhibit-991.md) expanded core narrative extraction (`10-K`/`10-K/A` 6 sections, part-qualified `10-Q`/`10-Q/A`, `8-K`/`8-K/A` + `Exhibit 99.1`), and [`ADR-00047`](ADR-00047-kg-form-ablation-embedder-reranker-study.md) built the controlled 4-pillar KG form-ablation and embedder scorecard.

Yet one major quantitative gap still limits the real-world fidelity of YarnBall's [`SparseNetworkCascadeEngine`](../../contagion/cascade.py): **epistemic fallback dominance (`STRESS_UPPER_BOUND`)**. Under [`ADR-00026`](ADR-00026-cascade-parameter-extraction-provenance-.md), whenever an extracted supply-chain edge omits an explicit procurement `share` or inventory `buffer_days`, [`resolve_edge_cascade_parameters`](../../contagion/parameters.py) fails closed to worst-case bounds (`share = 1.0`, `buffer_days = 0.0`, `DEFAULT_CONSERVATIVE_BOUND`). In real S&P 500 `10-K Item 1` / `Item 1A` prose, companies routinely name suppliers and customers without stating exact contract exclusivity, volume floors, or days of inventory—causing a high fraction of downstream cascade reports to classify as `STRESS_UPPER_BOUND` ($\ge 0.50$ assumed impact share in [`retrieval/reporting.py`](../../retrieval/reporting.py)) rather than `FILING_GROUNDED` ($\le 0.10$).

Two statutory SEC EDGAR data sources already registered as **Phase 2b (`PLANNED`)** in [`ingestion/sec/canon.py`](../../ingestion/sec/canon.py) directly close this quantitative gap across all 11 S&P 500 GICS sectors:

1. **`Exhibit 10` (Material Contracts — `EX-10.x`)**:
   - Contains binding operational agreements—wafer foundry and component supply agreements, pharma CDMO/licensing deals, LNG/pipeline off-take contracts, utility Power Purchase Agreements (PPAs), PBM contracts, co-branded card agreements, and spin-off Transition Services Agreements (TSAs)—that state explicit counterparties, exclusivity/sole-source clauses, minimum volume/take-or-pay commitments, lead times, and termination triggers.
   - **Noise hazard**: Roughly 70–85% of raw `EX-10.x` attachments on EDGAR are executive employment/severance agreements, RSU/stock-option award templates, director compensation plans, or boilerplate revolving credit amendments. Ingesting unfiltered `Exhibit 10` filings would flood `document_chunks` and degrade extraction precision.
2. **`Exhibit 101 / XBRL` (`companyfacts` JSON)**:
   - Every domestic S&P 500 issuer files machine-readable US-GAAP XBRL balance-sheet and footnote facts containing two high-signal quantitative priors:
     - **Days Inventory Outstanding ($\text{DIO}$)** computed deterministically from GAAP `InventoryNet` and annualized `CostOfGoodsAndServicesSold` for **physical-inventory sectors only**—providing a filing-derived empirical prior for downstream customer safety-stock buffers $B_{uv}$ when an edge omits a contract-specific buffer.
     - **ASC 280 Major-Customer Concentration (`ConcentrationRiskPercentage1`)**—capturing statutory $\ge 10\%$ single-customer revenue shares directly from structured XBRL disclosures without LLM parsing variance.
   - **Domain & PiT hazards**: Computing $\text{DIO}$ for `Financials`, `Real Estate`, `Utilities`, or pure software/cloud firms is economically meaningless (`Inventory = 0` or non-physical assets), and querying live EDGAR `companyfacts` during retrieval or backtesting would violate offline latency and Point-in-Time (`ADR-00033`) guarantees.

---

## 2. Decision Drivers & Constraints

1. **Measurable Lift in `FILING_GROUNDED` Provenance Without Hub/Noise Regression**: Both `Exhibit 10` and XBRL `companyfacts` must lift grounded cascade parameter coverage while remaining strictly verifiable under [`ADR-00047`](ADR-00047-kg-form-ablation-embedder-reranker-study.md)'s LOFO promotion gates.
2. **Honest Provenance Separation (`ADR-00026`)**: An issuer-level XBRL balance-sheet $\text{DIO}$ is a **company-level proxy**, not a per-supplier contract disclosure. It must carry its own explicit `ParameterProvenance` tier (`XBRL_ISSUER_DIO_PROXY`)—strictly subordinate to edge-specific `EXTRACTED_QUANTITATIVE` and strictly distinct from uncalibrated `DEFAULT_CONSERVATIVE_BOUND`.
3. **Aggressive Pre-Ingestion Filtering on `Exhibit 10`**: Executive compensation, equity-plan, governance, and routine debt-facility `EX-10` attachments must be rejected *before* full text download/chunking via a deterministic title/description filter, and kept contracts must undergo operational clause windowing (`<= 50_000` chars) so legal boilerplate never inflates extraction tokens.
4. **Physical-Sector Guard on XBRL $\text{DIO}$**: XBRL $\text{DIO}$ buffer priors apply **exclusively** to physical-component supply edges (`dependency_domain == "PHYSICAL_COMPONENT"`) on physical-inventory GICS sectors—never to `Financials`, `Real Estate`, `Utilities`, `Communication Services`, or `DIGITAL_INFRASTRUCTURE` / `COMMERCIAL_ENTERPRISE` edges.
5. **Zero Runtime Network Calls & Strict PiT Discipline (`ADR-00033`)**: XBRL `companyfacts` are harvested and normalized **offline** into a versioned local Parquet/JSON cache (mirrored in PostgreSQL `sec_xbrl_priors`) indexed by `(cik, period_end, filed_date)` with `actionable_session(filed_date)` anchoring so historical `PiTContext` lookups never read future financial statements.
6. **Zero New Cross-Package Import Cycles (`ADR-00045` Invariants #1 and #5)**: `contagion/` remains a zero-dependency leaf, `ingestion/sec/canon.py` remains stdlib-only, and `ontology/` remains free of `ingestion` imports (confidence parity enforced via unit tests).

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Ingest all `EX-10.x` attachments verbatim as full-length text.** Rejected. Overwhelms `document_chunks` and LLM extraction budgets with executive compensation plans, stock-option templates, and 150-page boilerplate legal schedules that carry zero operational supply-chain signal.
- **Alternative B: Label firm-wide XBRL $\text{DIO}$ as `EXTRACTED_QUANTITATIVE` identically to edge-specific filing disclosures.** Rejected. Violates [`ADR-00026`](ADR-00026-cascade-parameter-extraction-provenance-.md) transparency: firm-wide inventory days aggregate across all product lines and may overstate or understate safety stock for a specific sole-source component. An explicit `XBRL_ISSUER_DIO_PROXY` tier preserves complete auditability in `OperationalContagionReport` and the Workbench UI.
- **Alternative C: Apply XBRL $\text{DIO}$ uniformly across all 11 GICS sectors or query SEC EDGAR `companyfacts` live during `query_bi`.** Rejected. Banks, insurers, REITs, and SaaS vendors do not buffer supply disruptions with physical GAAP inventory, and live EDGAR calls break offline reproducibility and sub-second `query_bi` latency.

---

## 4. Decision Outcome

### 4.1 Operational `Exhibit 10` Filter & Clause-Windowed Extractor (`ingestion/sec/client.py` & `ingestion/sec/harvester.py`)

1. **Pre-Fetch Classification Gate (`classify_exhibit_10_attachment`)**:
   - Inspect each attachment with `document_type` matching `^EX[- ]?10(?:\.\d+)?\b` (skipping binaries/XBRL `EX-101`/`EX-104`).
   - **Deterministic Exclusion Stoplist** (evaluated first on attachment `description` / `document` name, rejecting non-operational agreements):
     `employment`, `severance`, `change in control`, `incentive plan`, `equity plan`, `stock option`, `restricted stock`, `rsu`, `psu`, `bonus plan`, `deferred compensation`, `director compensation`, `indemnification`, `retirement`, `pension`, `credit agreement`, `loan agreement`, `promissory note`, `indenture`, `underwriting`, `registration rights`, `voting agreement`, `stockholders agreement`, `share repurchase`.
   - **Deterministic Operational Allowlist** (must match at least one operational contract family):
     `supply`, `manufacturing`, `foundry`, `wafer`, `off-take` / `offtake`, `license`, `collaboration`, `co-development`, `cdmo`, `contract manufacturing`, `power purchase` / `ppa`, `pipeline`, `transportation`, `gathering`, `terminal`, `procurement`, `distribution`, `pharmacy benefit` / `pbm`, `co-brand`, `transition services` / `tsa`, `joint venture`, `strategic alliance`, `capacity reservation`, `cloud services`, `data center`, `hosting`, `logistics`, `freight`, `charter`, `purchase agreement` (when not excluded by share/note/equity stoplist).
2. **Targeted Operational Clause Windowing (`extract_operational_contract_clauses`)**:
   - For accepted `EX-10.x` contracts, preserve the **preamble / parties & recitals** (first ~`4_000` chars identifying exact legal counterparties and effective date) plus paragraphs/sections matching operational clause headings or high-signal cues (`exclusiv`, `sole source`, `minimum`, `take-or-pay`, `capacity`, `inventory`, `safety stock`, `buffer`, `lead time`, `delivery`, `force majeure`, `termination`, `royalty`, `facility`, `plant`, `fab`, `supply`, `volume`), stripping standard boilerplate tail exhibits/signature blocks and bounding each contract independently to `DEFAULT_MAX_FILING_CHARS = 50_000`.
3. **Client & Harvester Integration**:
   - Add `EdgarClient.extract_exhibit_10_contracts(filing, *, max_contracts=5, max_chars=DEFAULT_MAX_FILING_CHARS) -> List[Dict[str, Any]]` (each entry containing `exhibit_key`, `description`, `matched_category`, `text`, `raw_char_count`, `filtered_char_count`).
   - Add opt-in staging in `HistoricalSECHarvester.harvest_company_year(..., include_exhibit_10: bool = False)` (and explicit `"Exhibit 10"` support in `extract_10k_sections` / `fetch_recent_8k` when requested), persisting `exhibit_10_contracts.json` alongside `metadata.json` so [`ADR-00047`](ADR-00047-kg-form-ablation-embedder-reranker-study.md) can run clean LOFO ablation comparing `F4` vs. `F4 + Exhibit 10`.
   - Stamp extracted `Exhibit 10` chunks with `section="Exhibit 10 - Material Contract"`, `form_type`, and `sec_graph_layer="DOMAIN1_OPERATIONAL"`.

### 4.2 Offline XBRL `companyfacts` Parser & PiT Priors Store ([`ingestion/sec/xbrl.py`](../../ingestion/sec/xbrl.py))

Create a clean, offline-first XBRL module ([`ingestion/sec/xbrl.py`](../../ingestion/sec/xbrl.py)) consuming either local cached EDGAR `companyfacts` JSON files (`CIK{10-digit}.json`) or `edgartools` `Company.get_facts()` during offline batch harvesting:

1. **Physical-Inventory GICS Sector Allowlist**:
   - `PHYSICAL_INVENTORY_GICS_SECTORS`: `{"Information Technology", "Industrials", "Consumer Discretionary", "Consumer Staples", "Health Care", "Energy", "Materials"}`.
   - Explicitly excluded from $\text{DIO}$ buffer imputation: `Financials`, `Real Estate`, `Utilities`, `Communication Services` (and any edge whose `dependency_domain != "PHYSICAL_COMPONENT"`).
2. **Point-in-Time Days Inventory Outstanding ($\text{DIO}$)**:
   - Tag fallback ladders (strictly `units == "USD"`):
     - **Inventory numerator**: `InventoryNet` $\succ$ `InventoryFinishedGoods` + `InventoryWorkInProcess` + `InventoryRawMaterialsAndSupplies` $\succ$ `InventoryGross`.
     - **COGS denominator**: `CostOfGoodsAndServicesSold` $\succ$ `CostOfRevenue` $\succ$ `CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization` $\succ$ `CostOfGoodsSold`.
   - Prefer annual `FY` (`10-K` / `10-K/A`) flow duration ($\ge 330$ days) matched to period-end `Inventory`, or annualized quarterly `COGS` ($4 \times \text{quarterly COGS}$ from `10-Q` / `10-Q/A` duration `75–105` days) when a fresher quarterly filing precedes the next `10-K`:
     $$\text{DIO}_{i,p} = 365 \times \frac{\text{Inventory}_{i,p}}{\text{COGS}_{i,p}^{\text{ann}}}$$
   - Reject negative, zero, non-finite, or implausible outlier observations ($\text{DIO} \le 0$ or $\text{DIO} > 365.0$ days, logged as skipped rather than clamped silently).
   - Stamp every record with `(cik, ticker, gics_sector, period_end, filed_date, valid_from, known_at, accession_number, form_type, dio_days, inventory_usd, cogs_annualized_usd, inventory_tag, cogs_tag)` where `valid_from = actionable_session(None, fallback_doc_date=filed_date)` ([`ADR-00033`](ADR-00033-bitemporal-pit-contract-for-read-paths--.md)).
3. **ASC 280 Major-Customer Concentration (`ConcentrationRiskPercentage1`)**:
   - Extract `us-gaap:ConcentrationRiskPercentage1` facts with `units == "pure"` and `0.10 <= val <= 1.0` (retaining `filed`, `end`, `accn`, `form`, and any XBRL frame/segment metadata), persisted alongside $\text{DIO}$ records as `max_customer_concentration_share` and structured concentration rows.
4. **Durable Storage & Local Offline Cache**:
   - Persist locally to `data/sec_xbrl_priors.parquet` (or per-ticker JSON cache under `data/sec_historical/{TICKER}/xbrl_priors.json`) and mirror into PostgreSQL `sec_xbrl_priors` in [`tools/init_db.py`](../../tools/init_db.py) (`PRIMARY KEY (cik, metric_name, period_end, accession_number)` + index on `(cik, metric_name, valid_from DESC)`).
   - Provide an $O(\log K)$ in-memory PiT lookup helper `lookup_pit_xbrl_dio(cik_or_ticker, as_of, *, gics_sector=None) -> Optional[XBRLPriorRecord]` returning strictly the latest record with `valid_from <= as_of`.

### 4.3 Cascade Parameter Provenance Integration (`contagion/parameters.py`, `retrieval/reporting.py`, `ingestion/sec/canon.py`, `ontology/validation.py`)

1. **New Honest Provenance Tier in [`contagion/parameters.py`](../../contagion/parameters.py)** (preserving `contagion/` as a zero-dependency leaf):
   - Add `ParameterProvenance.XBRL_ISSUER_DIO_PROXY = "XBRL_ISSUER_DIO_PROXY"` with `PROVENANCE_CONFIDENCE_RANK = 2` (below `EXTRACTED_QUANTITATIVE = 3`, peer to `EXTRACTED_QUALITATIVE_SOLE_SOURCE = 2`, strictly above `DEFAULT_CONSERVATIVE_BOUND = 1`).
   - Extend `resolve_edge_cascade_parameters(edge, source_text=None, *, target_xbrl_dio_days: Optional[float] = None, target_gics_sector: Optional[str] = None)`:
     - **Precedence rule**:
       1. Edge-specific explicit `buffer_days` (from filing/Exhibit 10 text) $\rightarrow$ `EXTRACTED_QUANTITATIVE` (always wins!).
       2. Else if `target_xbrl_dio_days` is positive, finite, `<= 365.0`, `dependency_domain == "PHYSICAL_COMPONENT"`, and (`target_gics_sector is None` or `target_gics_sector in PHYSICAL_INVENTORY_GICS_SECTORS`) $\rightarrow$ set `buffer_days = target_xbrl_dio_days`, `buffer_days_is_default = False`, and promote `param_provenance` from `DEFAULT_CONSERVATIVE_BOUND` to `XBRL_ISSUER_DIO_PROXY` (while leaving `share_is_default = True` transparently recorded when `share` itself was not extracted!).
       3. Else $\rightarrow$ retain fail-closed `buffer_days = 0.0`, `DEFAULT_CONSERVATIVE_BOUND`.
   - Zero breaking changes when callers omit `target_xbrl_dio_days`.
2. **Canon & Ontology Confidence Sync**:
   - Promote `SEC_EXHIBIT_CANON["Exhibit 10"]` (`provenance_key = "SEC_EXHIBIT_10"`, `base_confidence = 0.98`) and `SEC_EXHIBIT_CANON["Exhibit 101 / XBRL"]` (`provenance_key = "SEC_XBRL_COMPANYFACTS"`, `base_confidence = 1.00`) from `PLANNED` to `ACTIVE` in [`ingestion/sec/canon.py`](../../ingestion/sec/canon.py), and add matching entries in [`ontology/validation.py`](../../ontology/validation.py) `PROVENANCE_CONFIDENCE_MAP` (maintaining zero `ontology -> ingestion` imports per `ADR-00045` Invariant #5).
   - Register `SECSectionSpec` for `Exhibit 10 - Material Contract` in [`ingestion/sec/canon.py`](../../ingestion/sec/canon.py) (`high_signal_gleaning=True`).

---

## 5. Invariants & Guardrails

1. **Zero Executive-Compensation / Debt-Boilerplate Leakage**: `classify_exhibit_10_attachment` must deterministically reject employment agreements, RSU/option plans, indemnification agreements, and routine credit facilities before text extraction or chunking.
2. **Strict Edge-Specific Precedence Over XBRL Priors**: Edge-specific `buffer_days` extracted from narrative filings or `Exhibit 10` contracts (`EXTRACTED_QUANTITATIVE`) must **always** override issuer-level `target_xbrl_dio_days` (`XBRL_ISSUER_DIO_PROXY`).
3. **Physical-Sector & Physical-Domain Guard**: `XBRL_ISSUER_DIO_PROXY` must **never** be applied to `Financials`, `Real Estate`, `Utilities`, `Communication Services`, or `DIGITAL_INFRASTRUCTURE` / `COMMERCIAL_ENTERPRISE` edges.
4. **Strict Point-in-Time XBRL Anchoring (`ADR-00033`)**: Every XBRL $\text{DIO}$ and ASC 280 record must be indexed by `actionable_session(None, fallback_doc_date=filed_date)`, and `lookup_pit_xbrl_dio(cik, as_of)` must never return any fact with `valid_from > as_of`.
5. **Zero Runtime Network Calls & Leaf Discipline**: `contagion/parameters.py` stays pure stdlib (`0` internal imports), `ingestion/sec/canon.py` stays stdlib-only, `ontology/` stays free of `ingestion` imports, and `query_bi` / backtesting never call live SEC EDGAR APIs.
6. **Empirical Promotion via `ADR-00047`**: `Exhibit 10` harvesting stays opt-in (`include_exhibit_10=False` by default) until evaluated through [`ADR-00047`](ADR-00047-kg-form-ablation-embedder-reranker-study.md)'s LOFO promotion gates on a multi-sector S&P 500 snapshot.

---

## 6. Verification Plan

1. **New Phase 2b Unit & Integration Suite** ([`tests/ingestion/sec/test_sec_phase2b_ex10_xbrl.py`](../../tests/ingestion/sec/test_sec_phase2b_ex10_xbrl.py)):
   - **Exhibit 10 Classifier**: Verify operational contracts (foundry supply, pharma CDMO, LNG off-take, utility PPA, PBM, TSA) pass while executive employment, RSU award, director compensation, indemnification, and credit agreements are deterministically rejected.
   - **Exhibit 10 Clause Windowing**: Verify recitals + operational clauses (exclusivity, volume floors, buffer days, termination) are preserved under `50_000` chars while boilerplate legal definitions/notices are trimmed.
   - **XBRL `companyfacts` Parser**: Feed deterministic multi-quarter/multi-year `companyfacts` JSON fixtures and verify exact $\text{DIO}$ calculation (annual `FY` and annualized `10-Q`), outlier rejection (`<= 0` or `> 365`), ASC 280 `ConcentrationRiskPercentage1` parsing, strict `valid_from <= as_of` PiT lookups (no future balance-sheet leakage), and exclusion of `Financials` / `Real Estate` / `Utilities` / `Communication Services`.
   - **Cascade Provenance Precedence**: Verify `resolve_edge_cascade_parameters`:
     - Edge with explicit `buffer_days=45` ignores `target_xbrl_dio_days=62` and stays `EXTRACTED_QUANTITATIVE`;
     - Physical edge without `buffer_days` adopts `target_xbrl_dio_days=62` with `param_provenance == "XBRL_ISSUER_DIO_PROXY"` and `buffer_days_is_default is False`;
     - `Financials` or `DIGITAL_INFRASTRUCTURE` edge rejects `target_xbrl_dio_days` and stays `DEFAULT_CONSERVATIVE_BOUND`.
   - **Canon & Ontology Parity**: Update `tests/ingestion/sec/test_sec_canon_integrity.py` to assert `Exhibit 10` and `Exhibit 101 / XBRL` are `ACTIVE`, `SEC_EXHIBIT_10` / `SEC_XBRL_COMPANYFACTS` match `PROVENANCE_CONFIDENCE_MAP`, and zero `ontology -> ingestion` imports exist.
2. **Full-Corpus Static & Regression Gate**:
   - `.venv/bin/mypy` (0 errors across full corpus + new tests).
   - `.venv/bin/pytest --ignore=tests/knowledge_graph/test_embedding_store.py -v` and `tests/docs/test_adr_integrity.py` + `tests/docs/test_docs_integrity.py` (updating `Table 0.4` `ingestion` `.py` count `22` $\rightarrow$ `23` for `ingestion/sec/xbrl.py` and `SEC_SECTION_CANON` count if extended).
   - `.venv/bin/graphify update .`.

---

## 7. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Directly attacks YarnBall's largest cascade-provenance bottleneck (`DEFAULT_CONSERVATIVE_BOUND` / `STRESS_UPPER_BOUND` dominance) using two deterministic, statutory SEC data sources.
  - Captures binding sole-source, off-take, CDMO, PPA, and PBM agreements across all 11 S&P 500 GICS sectors while blocking ~70–85% of compensation/debt `EX-10` noise upfront.
  - Provides honest, PiT-safe `XBRL_ISSUER_DIO_PROXY` inventory buffer bounds for physical supply-chain sectors without contaminating financial/software edges or fabricating contract-level precision.
- **Negative / Trade-offs**:
  - Adds one new module (`ingestion/sec/xbrl.py`, incrementing `ingestion` `.py` count `22` $\rightarrow$ `23` in `00-system-atlas.md`), one new `ParameterProvenance` enum value (`XBRL_ISSUER_DIO_PROXY`), and one new PostgreSQL table (`sec_xbrl_priors`).
  - Firm-wide XBRL $\text{DIO}$ remains a company-level approximation (blending finished goods, WIP, and raw materials unless raw-material tags alone are reported); downstream reports and the Workbench UI surface `XBRL_ISSUER_DIO_PROXY` explicitly so analysts distinguish it from contract-specific disclosures.
