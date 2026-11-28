# ADR-00049: Phase 2c-i Foreign Private Issuer & Canadian Upstream Filings (20-F, 20-F/A, 40-F, 6-K & Exhibit 8)

- **ADR ID**: ADR-00049
- **Filename**: `ADR-00049-foreign-sec-filings-20f-40f-6k-exhibit-8.md`
- **Date**: 2026-11-28 07:45 UTC
- **Status**: ACCEPTED
- **Target Subsystem**: `ingestion` / `ontology` / `retrieval` / `tests` / `docs`
- **Linked Commits**: [`ec39c7a`](../../ingestion/sec/foreign.py), [`ec39c7a`](../../ingestion/sec/client.py), [`ec39c7a`](../../ingestion/sec/harvester.py), [`ec39c7a`](../../ingestion/sec/parser.py), [`ec39c7a`](../../ingestion/sec/canon.py), [`ec39c7a`](../../ingestion/sec/xbrl.py), [`ec39c7a`](../../ontology/validation.py), [`ec39c7a`](../../tests/ingestion/sec/test_sec_phase2c_foreign_filings.py)

---

## 1. Context & Problem Statement

YarnBall's domestic S&P 500 ingestion pipeline ([`ADR-00045`](ADR-00045-canonical-sec-edgar-filing-registry-----.md), [`ADR-00046`](ADR-00046-sec-core-sections-amendments-exhibit-991.md), [`ADR-00048`](ADR-00048-exhibit-10-contracts-and-xbrl-dio-priors.md)) extracts high-precision supply-chain topology, material contracts, and quantitative inventory priors from domestic U.S. filings (`10-K`/`10-K/A`, `10-Q`/`10-Q/A`, `8-K`/`8-K/A`, `Exhibit 10`, `Exhibit 21`, and US-GAAP XBRL).

However, critical Tier-1 upstream bottlenecks across the S&P 500 are **Foreign Private Issuers (FPIs)** or **Canadian MJDS issuers** that do not file domestic `10-K` or `8-K` reports on SEC EDGAR:
- **Semiconductors & Electronics**: TSMC (`TSM`), ASML (`ASML`), STMicroelectronics (`STM`), Arm (`ARM`), UMC (`UMC`), ASE Technology (`ASX`), Sony (`SONY`).
- **Pharmaceuticals & Biologics**: Novo Nordisk (`NVO`), AstraZeneca (`AZN`), Novartis (`NVS`), Sanofi (`SNY`), GSK (`GSK`), Takeda (`TAK`).
- **North American Pipelines, Class-I Rail & Raw Materials (Canadian MJDS)**: Enbridge (`ENB`), TC Energy (`TRP`), Canadian National Railway (`CNI`), CPKC (`CP`), Nutrien (`NTR`), Canadian Natural Resources (`CNQ`), Suncor (`SU`), plus global miners BHP (`BHP`), Rio Tinto (`RIO`), and Vale (`VALE`).

Because YarnBall currently ignores these issuers' primary EDGAR disclosures (`20-F`, `20-F/A`, `40-F`, `6-K`, and `Exhibit 8`), two structural blind spots remain:
1. **Asymmetric Upstream Visibility**: U.S. S&P 500 `10-K` filings frequently refer to overseas foundries, API manufacturers, or pipeline operators without naming specific plant sites, capacity limits, or customer concentration; the foreign supplier's own `20-F`/`40-F` and `Exhibit 8` subsidiaries schedule expose those exact physical dependencies.
2. **Delayed Overseas Shock Detection**: When an earthquake hits a Taiwanese fab, an export-control notice hits a European lithography vendor, or an outage hits a Canadian pipeline/rail corridor, the earliest statutory SEC disclosure is the supplier's **Form `6-K` (+ `Exhibit 99.1`)**—often days before affected S&P 500 customers file a domestic `8-K`.

Additionally, foreign SEC filings introduce three practical ingestion hazards not present in domestic U.S. filings:
- **SEC Rule 306 Bilingual / Summary-Only `6-K` Attachments**: While primary SEC filings are legally required to be in English (`17 CFR § 232.306`), `6-K` attachments may be brief English summaries, side-by-side bilingual documents (English + French/Chinese/Spanish), or non-English appendices.
- **Unnumbered `6-K` Structure**: Unlike Form `8-K` (which carries 32 statutory item codes), Form `6-K` has no standardized operational item numbering—making boilerplate stripping and zero-shock negative-control gating essential to reject routine monthly revenue notices, dividend declarations, and AGM proxies.
- **International Legal Suffixes, Diacritics & IFRS Currencies**: `Exhibit 8` subsidiary schedules contain accents and non-U.S. corporate forms (`B.V.`, `K.K.`, `S.r.l.`, `S.p.A.`, `S.A.S.`, `Pty Ltd`, `Oyj`, `KGaA`), while `20-F` XBRL facts use the `ifrs-full` taxonomy and local reporting currencies (`TWD`, `EUR`, `CAD`, `JPY`, `GBP`, `CHF`).

---

## 2. Decision Drivers & Constraints

1. **Pure S&P 500 Universe Isolation**: Foreign issuers must **never** be mixed into [`SP500UniverseManager`](../../universe/sp500.py) or `get_default_harvest_forms()` (which remains strictly the 6 domestic S&P 500 core forms `10-K`, `10-K/A`, `10-Q`, `10-Q/A`, `8-K`, `8-K/A`). Foreign ingestion operates over an explicit, opt-in counterparty registry (`ForeignCounterpartyRegistry`) seeded by Tier-1 S&P 500 upstream suppliers and optionally enriched by graph-discovered counterparties.
2. **Deterministic English / Bilingual Filtering (Rule 306 Safety)**: Before LLM extraction or `6-K` disruption gating, a stateless zero-LLM language/script filter must isolate English operational prose from bilingual side-by-side blocks and drop non-English / garbled attachments without wasting LLM tokens.
3. **Zero-Shock Negative-Control Gating on Form `6-K`**: Routine `6-K` submissions (monthly revenue tables, quarterly earnings, dividend announcements, AGM notices, share repurchases) must deterministically yield `is_material_operational_disruption = False` and strictly zero shock parameters (`0.0`).
4. **Strict `ADR-00033` Invariant #8 Preservation**: `Form 6-K` retains `allow_contract_termination = False` in [`ingestion/sec/canon.py`](../../ingestion/sec/canon.py); only domestic Form `8-K` / `8-K/A` may unilaterally close `valid_to` on active graph edges. Every `20-F`, `20-F/A`, `40-F`, and `6-K` filing is anchored to NYSE trading sessions via [`actionable_session`](../../retrieval/temporal.py).
5. **International Subsidiary Rollup (`Exhibit 8`)**: `normalize_company_name` in [`ingestion/sec/parser.py`](../../ingestion/sec/parser.py) must support Unicode NFKD diacritic folding and common non-U.S. legal suffixes without introducing false-positive stripping on normal English words, and `Exhibit 8` subsidiaries must persist into `sec_subsidiaries` and Memgraph `SUBSIDIARY_OF` edges identically to `Exhibit 21`.
6. **Currency-Invariant IFRS XBRL DIO Extension**: When parsing `20-F` `companyfacts` JSON, [`ingestion/sec/xbrl.py`](../../ingestion/sec/xbrl.py) may extract `ifrs-full` (`Inventories` / `CostOfSales`) $\text{DIO}$ strictly when numerator and denominator share the **exact same currency unit $c$** ($\text{DIO} = 365 \times \text{Inv}_c / \text{COGS}_c$), preserving physical-sector guards and zero currency-mixing.
7. **Zero New Cross-Package Import Cycles (`ADR-00045` Invariants #1 and #5)**: `ingestion/sec/canon.py` remains stdlib-only, `ontology/` remains free of `ingestion` imports, and `contagion/` stays a pure leaf.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Append foreign ADRs directly into `sp500_constituents_historical.json` and harvest them inside default domestic batch runs.** Rejected. Corrupts survivorship-free S&P 500 index membership (`universe/sp500.py`), breaks point-in-time index denominator invariants, and couples domestic batch runs to non-index issuers.
- **Alternative B: Call an external translation API or LLM language-detection pass on every `6-K` attachment.** Rejected. Violates YarnBall's offline reproducibility, zero runtime external API dependence, and 8 GB GPU budget; deterministic Unicode script & Latin-letter ratio filtering cleanly handles SEC Rule 306 English/bilingual filings in $O(N)$ microseconds.
- **Alternative C: Allow Form `6-K` press releases to terminate active graph edges (`valid_to`).** Rejected. Violates [`ADR-00033`](ADR-00033-bitemporal-pit-contract-for-read-paths--.md) Invariant #8; Form `6-K` lacks standardized statutory `Item 1.02` termination semantics and often carries brief home-country summaries.

---

## 4. Decision Outcome

### 4.1 Canon & Ontology Confidence Activation ([`ingestion/sec/canon.py`](../../ingestion/sec/canon.py) & [`ontology/validation.py`](../../ontology/validation.py))

1. **Promote Phase 2c-i Forms & Exhibit to `ACTIVE`**:
   - Promote `FORM_20F` (`provenance_key="SEC_20F"`, `base_confidence=0.98`), `FORM_20FA` (`provenance_key="SEC_20F"`, `base_confidence=0.98`), `FORM_40F` (`provenance_key="SEC_40F"`, `base_confidence=0.98`), `FORM_6K` (`provenance_key="SEC_6K"`, `base_confidence=0.95`), and `SEC_EXHIBIT_CANON["Exhibit 8"]` (`provenance_key="SEC_EXHIBIT_8"`, `base_confidence=1.00`) from `PLANNED` to `ACTIVE`.
   - Keep `default_batch_harvest = False` on all four foreign forms so `get_default_harvest_forms()` remains strictly the 6 domestic S&P 500 core forms, and add `get_foreign_harvest_forms() -> Tuple[str, ...]` returning `("20-F", "20-F/A", "40-F", "6-K")`.
   - Add matching entries `"SEC_20F": 0.98`, `"SEC_40F": 0.98`, `"SEC_6K": 0.95`, and `"SEC_EXHIBIT_8": 1.00` to `PROVENANCE_CONFIDENCE_MAP` in [`ontology/validation.py`](../../ontology/validation.py) (enforced via parity tests without adding any `ontology -> ingestion` import).
2. **Extend `SEC_SECTION_CANON` (`26` $\rightarrow$ `29` canonical sections)**:
   - Retain existing `Item 3.D - Risk Factors (20-F)` and `Item 4 - Information on the Company (20-F)` (`high_signal_gleaning=True`).
   - Add:
     - `Item 5 - Operating and Financial Review (20-F)` (`applicable_forms=(FORM_20F, FORM_20FA)`, `high_signal_gleaning=False`, `sector_gleaning_only=("Financials",)`).
     - `Exhibit 8 - Significant Subsidiaries (20-F)` (`applicable_forms=(FORM_20F, FORM_20FA, FORM_F1)`, `high_signal_gleaning=False`).
     - `Annual Information Form (40-F)` (`applicable_forms=(FORM_40F,)`, `high_signal_gleaning=True`).

### 4.2 Dedicated Foreign Counterparty Module ([`ingestion/sec/foreign.py`](../../ingestion/sec/foreign.py))

Create [`ingestion/sec/foreign.py`](../../ingestion/sec/foreign.py) encapsulating foreign-issuer registries, SEC Rule 306 language guards, and Form `6-K` sanitization/disruption gating:

1. **Curated Foreign Counterparty Registry**:
   - Frozen dataclass `ForeignCounterparty(ticker, cik, company_name, home_country, annual_form, gics_sector, sp500_supply_role)`.
   - `DEFAULT_TIER1_FOREIGN_COUNTERPARTIES`: deterministic seed list of Tier-1 non-U.S. S&P 500 upstream suppliers across Semiconductors (`TSM`, `ASML`, `STM`, `ARM`, `UMC`, `ASX`), Pharma/Biotech (`NVO`, `AZN`, `NVS`, `SNY`, `GSK`, `TAK`), Canadian MJDS Energy/Rail/Materials (`ENB`, `TRP`, `CNI`, `CP`, `NTR`, `CNQ`, `SU`), and Global Materials/Industrial Tech (`BHP`, `RIO`, `VALE`, `SONY`, `SAP`, `INFY`, `TM`).
   - Helper `lookup_foreign_counterparty(ticker_or_cik)` and `filter_foreign_counterparties(tickers=None, gics_sectors=None, annual_form=None)`.
2. **SEC Rule 306 Bilingual & English Operational Guard (`filter_english_operational_text`)**:
   - Stateless $O(N)$ paragraph/line filter evaluating Latin alphabetic ratio against total alphabetic characters (`min_latin_letter_ratio = 0.70`), plus a lightweight check rejecting pure numeric tables or untranslated CJK/Cyrillic/Arabic appendices while preserving English paragraphs from side-by-side bilingual documents.
   - Returns `(filtered_text, LanguageFilterReport(kept_paragraphs, dropped_paragraphs, latin_letter_ratio, is_usable_english))`.
3. **Form `6-K` Boilerplate Sanitization & Zero-Shock Disruption Gating (`sanitize_sec_6k_text`, `assess_sec_6k_disruption`)**:
   - Strips Form `6-K` statutory cover-page markers (`report of foreign private issuer pursuant to rule 13a-16 or 15d-16`, `annual reports under cover of form 20-f or form 40-f`, Exchange Act signature tails), preserves attached `Exhibit 99.x` operational press releases independently bounded at `DEFAULT_MAX_FILING_CHARS = 50_000`, and applies `filter_english_operational_text`.
   - Applies strict operational-disruption gating: routine `6-K` disclosures (monthly revenue reports, earnings/dividend releases, AGM/proxy notices, share buybacks, bond issuances) without corroborated operational-halt / disruption lexical evidence (e.g., earthquake/fab halt, pipeline outage, export-control restriction, plant closure, cyberattack, strike/rail stoppage) deterministically emit `is_material_operational_disruption = False`, `recommended_shock_magnitude_pct = 0.0`, `recommended_shock_duration_days = 0.0`.

### 4.3 Client, Parser & Harvester Integration ([`ingestion/sec/client.py`](../../ingestion/sec/client.py), [`ingestion/sec/parser.py`](../../ingestion/sec/parser.py), [`ingestion/sec/harvester.py`](../../ingestion/sec/harvester.py))

1. **International Name Normalization ([`ingestion/sec/parser.py`](../../ingestion/sec/parser.py))**:
   - Add NFKD Unicode diacritic folding (`unicodedata.normalize("NFKD", text)` stripping combining marks) inside `normalize_company_name` so accented European legal names match ASCII mentions cleanly.
   - Extend `CORP_SUFFIXES` carefully with unambiguous international corporate forms: `b.v.`/`bv`, `k.k.`/`kk`, `s.r.l.`/`srl`, `s.p.a.`/`spa`, `s.a.s.`/`sas`, `pty`, `oyj`, `kgaa`, `a/s`, `kabushiki kaisha`, `aktiengesellschaft`, `societe anonyme` (explicitly excluding ambiguous 2-letter English words like bare `as` or `se`).
2. **`EdgarClient` Foreign Extractors ([`ingestion/sec/client.py`](../../ingestion/sec/client.py))**:
   - `fetch_20f_filings(ticker_or_cik, year=None, *, form_type="20-F", limit=2)` and `extract_20f_sections(filing, *, include_exhibit_10=False)` extracting `item_3d_risk_factors`, `item_4_company_info`, `item_5_operating_review`, `subsidiaries` (via `extract_exhibit_8_subsidiaries`), and optional `exhibit_10_contracts`.
   - `extract_exhibit_8_subsidiaries(filing)` detecting `EX-8` / `EX-8.1` / `"significant subsidiaries"` attachments and parsing them with the shared subsidiary table parser.
   - `fetch_40f_filings(ticker_or_cik, year=None, limit=2)` and `extract_40f_sections(filing)` extracting Canadian MJDS Annual Information Form (`AIF`) and MD&A narrative sections (including `Exhibit 99.x` AIF attachments bounded at `DEFAULT_MAX_FILING_CHARS`).
   - `fetch_recent_6k(ticker_or_cik, limit=10, year=None, *, include_exhibits=True, max_chars=DEFAULT_MAX_FILING_CHARS)` extracting Form `6-K` body + `Exhibit 99.x` attachments and running `filter_english_operational_text`.
3. **Opt-In Foreign Batch Harvesting & Worker Persistence ([`ingestion/sec/harvester.py`](../../ingestion/sec/harvester.py))**:
   - Add `HistoricalSECHarvester.harvest_foreign_counterparty_year(counterparty, year, *, forms=None, include_exhibit_10=False)` staging under `{output_dir}/FOREIGN/{TICKER}/{YEAR}/` (`20F_...`, `20FA_...`, `40F_...`, `6K_...` with `metadata.json`, section `.txt` files, `exhibit_8_subsidiaries.json`, and `form6k_events.json`).
   - Add `EdgarWorker.process_foreign_annual(ticker_or_cik, year=None, *, form_type="20-F", gics_sector=None)` ingesting `20-F`/`20-F/A`/`40-F` narrative chunks + `Exhibit 8` subsidiaries into `sec_subsidiaries`, `document_chunks`, `kg_nodes`/`kg_edges`, and Memgraph with `form_type` preserved and `actionable_session` PiT timestamps.

### 4.4 Currency-Invariant IFRS XBRL DIO Extension ([`ingestion/sec/xbrl.py`](../../ingestion/sec/xbrl.py))

- Extend `extract_xbrl_dio_priors_from_companyfacts` to inspect `facts["ifrs-full"]` when `us-gaap` yields no DIO observations:
  - **Inventory numerator**: `ifrs-full:Inventories` (and `ifrs-full:CurrentInventories`).
  - **COGS denominator**: `ifrs-full:CostOfSales`.
  - **Strict same-currency pairing**: Pair inventory and COGS facts strictly within the **same ISO currency code** (`(currency_code, period_end, accn)`, e.g., `TWD`/`TWD`, `EUR`/`EUR`, `CAD`/`CAD`, `JPY`/`JPY`) so $\text{DIO}_{i,p} = 365 \times \text{Inv}_{i,p,c} / \text{COGS}_{i,p,c}^{\text{ann}}$ is strictly dimensionless and currency-invariant. Never divide across mismatched currencies.
  - Retain all existing physical-sector guards (`PHYSICAL_INVENTORY_GICS_SECTORS`), `(0, 365]` outlier rejection, and `actionable_session` PiT anchoring.

---

## 5. Invariants & Guardrails

1. **Domestic S&P 500 Isolation**: `get_default_harvest_forms()` stays strictly `("10-K", "10-K/A", "10-Q", "10-Q/A", "8-K", "8-K/A")`, and [`SP500UniverseManager`](../../universe/sp500.py) is untouched by foreign counterparties.
2. **Rule 306 Language Safety**: `filter_english_operational_text` must strip non-English paragraphs from bilingual `6-K` attachments and reject purely non-English exhibits deterministically in $O(N)$ time without calling an LLM.
3. **Zero-Shock Negative-Control `6-K` Gating**: Routine `6-K` filings (monthly revenue updates, dividends, AGM notices, earnings without operational disruption) must produce `is_material_operational_disruption == False` and `0.0` shock magnitude/duration.
4. **Strict Termination Eligibility (`ADR-00033` Invariant #8)**: `is_termination_eligible_form("6-K")` and `is_termination_eligible_form("20-F")` and `is_termination_eligible_form("40-F")` must remain strictly `False`.
5. **Same-Currency IFRS DIO Invariant**: `extract_xbrl_dio_priors_from_companyfacts` must never pair an inventory value in currency $c_1$ with a COGS value in currency $c_2 \neq c_1$.
6. **Zero New Cross-Package Import Cycles & Confidence Parity (`ADR-00045` Invariants #1 and #5)**: `SEC_20F`, `SEC_40F`, `SEC_6K`, and `SEC_EXHIBIT_8` match `ontology/validation.py:PROVENANCE_CONFIDENCE_MAP` exactly with zero `ontology -> ingestion` imports; `contagion/` stays a zero-dependency leaf.

---

## 6. Verification Plan

1. **New Phase 2c-i Unit & Integration Suite** ([`tests/ingestion/sec/test_sec_phase2c_foreign_filings.py`](../../tests/ingestion/sec/test_sec_phase2c_foreign_filings.py)):
   - **Rule 306 Language & Bilingual Filter**: Verify English operational prose passes, bilingual English+Chinese / English+French documents keep English paragraphs and drop foreign-script blocks, and pure non-English appendices are flagged `is_usable_english == False`.
   - **Form `6-K` Sanitization & Disruption Gating**: Verify a TSMC-style monthly revenue `6-K` and a dividend/AGM `6-K` produce zero shock (`is_material_operational_disruption is False`, `0.0`), while a Taiwan fab earthquake `6-K` (`Exhibit 99.1`) or Canadian pipeline outage `6-K` fires the expected disruption category with non-zero shock parameters.
   - **Form `20-F` / `20-F/A` / `40-F` & `Exhibit 8` Extraction**: Verify `Item 3.D`, `Item 4`, `Item 5`, `40-F` AIF/MD&A, and `Exhibit 8` subsidiary parsing, international suffix & diacritic normalization (`"Société Industrielle S.A."`, `"ASML Netherlands B.V."`, `"Murata Manufacturing K.K."`), opt-in foreign harvesting under `FOREIGN/{TICKER}/{YEAR}/`, and `EdgarWorker.process_foreign_annual` chunk/subsidiary persistence.
   - **Currency-Invariant IFRS XBRL DIO**: Verify `ifrs-full` `TWD`/`TWD` and `EUR`/`EUR` DIO calculation succeeds with strict PiT lookup, while mismatched `USD`/`TWD` facts and non-physical foreign banks (`Financials`) are deterministically rejected.
   - **Canon & Ontology Parity**: Update [`tests/ingestion/sec/test_sec_canon_integrity.py`](../../tests/ingestion/sec/test_sec_canon_integrity.py) to verify `20-F`/`20-F/A`/`40-F`/`6-K`/`Exhibit 8` are `ACTIVE`, `get_default_harvest_forms()` stays domestic-only, `is_termination_eligible_form("6-K")` stays `False`, and `SEC_20F`/`SEC_40F`/`SEC_6K`/`SEC_EXHIBIT_8` match `PROVENANCE_CONFIDENCE_MAP`.
2. **Full-Corpus Static & Regression Gate**:
   - `.venv/bin/mypy` (0 errors across full corpus + new tests).
   - `.venv/bin/pytest --ignore=tests/knowledge_graph/test_embedding_store.py -q`, including `tests/docs/test_adr_integrity.py` and `tests/docs/test_docs_integrity.py` (updating `Table 0.4` `ingestion` `.py` count `23` $\rightarrow$ `24` for `ingestion/sec/foreign.py` and `docs/reference/ingestion-and-universe.md`).
   - `.venv/bin/graphify update .`.

---

## 7. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Closes YarnBall's largest non-U.S. upstream blind spot across semiconductors, European pharma, Canadian pipelines/rail, and global raw materials without contaminating the survivorship-free S&P 500 universe.
  - Enables early shock detection from foreign supplier Form `6-K` disclosures while suppressing routine foreign press releases and non-English boilerplate deterministically before LLM calls.
  - Extends subsidiary CIK rollup (`Exhibit 8`) and currency-invariant physical-sector XBRL $\text{DIO}$ priors (`ifrs-full`) seamlessly to non-U.S. counterparties.
- **Negative / Trade-offs**:
  - Adds one new module (`ingestion/sec/foreign.py`, incrementing `ingestion` `.py` count `23` $\rightarrow$ `24` in `00-system-atlas.md`) and 3 new canonical sections (`26` $\rightarrow$ `29`) in `ingestion/sec/canon.py`.
  - Curated foreign counterparty coverage requires periodic maintenance (mitigated by allowing graph-discovered foreign tickers/CIKs to be passed directly to `harvest_foreign_counterparty_year`).
