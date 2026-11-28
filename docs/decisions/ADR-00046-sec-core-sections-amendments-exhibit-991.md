# ADR-00046: Phase 2a Core SEC Expansion — Sections, Amendments, Exhibit 99.1 & Sector Gleaning

- **ADR ID**: ADR-00046
- **Filename**: `ADR-00046-sec-core-sections-amendments-exhibit-991.md`
- **Date**: 2026-11-28 04:19 UTC
- **Status**: PROPOSED
- **Target Subsystem**: `ingestion` / `tests` / `docs`
- **Linked Commits**: Pending acceptance

---

## 1. Context & Problem Statement

[`ADR-00045`](ADR-00045-canonical-sec-edgar-filing-registry-----.md) consolidated all 33 canonical SEC EDGAR form types, 9 exhibits, 32 Form 8-K items, and 25 statutory sections into [`ingestion/sec/canon.py`](../../ingestion/sec/canon.py). Commit `5ff2243` subsequently resolved the Form 8-K sanitization truncation bug in [`ingestion/sec/disruption_gating.py`](../../ingestion/sec/disruption_gating.py) so attached `Exhibit 99.1` press releases survive Exchange Act signature stripping and receive independent `50_000`-character budgets.

However, an audit of [`ingestion/sec/client.py`](../../ingestion/sec/client.py), [`ingestion/sec/harvester.py`](../../ingestion/sec/harvester.py), and [`ingestion/sec/parser.py`](../../ingestion/sec/parser.py) exposes five concrete extraction blind spots inside YarnBall's *existing* core statutory forms (`10-K`, `10-Q`, `8-K` and their `/A` amendments) across the 11 S&P 500 GICS sectors:

1. **Form 8-K omits `Exhibit 99.1` attachments, omits `8-K/A`, and truncates at 10K characters**:
   - [`EdgarClient.fetch_recent_8k`](../../ingestion/sec/client.py) only queries `form="8-K"` (missing `8-K/A` amended current reports despite `8-K/A` being `ACTIVE` and termination-eligible under [`ADR-00033`](ADR-00033-bitemporal-pit-contract-for-read-paths--.md) Invariant #8).
   - It hard-slices primary text to `filing.text()[:10000]` (below `DEFAULT_MAX_FILING_CHARS = 50_000`) and never inspects `filing.attachments` for `EX-99.1` / `EX-99.2` press releases—leaving real-world filings that state *"A copy of the press release is attached hereto as Exhibit 99.1"* as two-sentence wrappers with no operational facts.
2. **Form 10-K extraction drops `Item 1C`, `Item 2`, `Item 3`, and silently drops `Item 7`**:
   - [`EdgarClient.extract_10k_sections`](../../ingestion/sec/client.py) extracts only `item_1_business`, `item_1a_risk_factors`, and `subsidiaries` (`Exhibit 21`).
   - [`EdgarWorker.process_company_10k`](../../ingestion/sec/harvester.py) already loops over `("item_7_mda", "Item 7 - Management Discussion and Analysis")`, but `sections.get("item_7_mda", "")` silently evaluates to `""` on every filing because `extract_10k_sections` never populates it.
   - `Item 1C` (Cybersecurity), `Item 2` (Properties — physical factories, refineries, mines, pipelines, power stations, REIT portfolios), and `Item 3` (Legal Proceedings — FDA/EPA enforcement, recalls, patent/antitrust injunctions) are never extracted or staged.
3. **Form 10-Q naive `[:15000]` truncation cuts off narrative sections**:
   - [`EdgarClient.fetch_10q_filings`](../../ingestion/sec/client.py) pulls only `filing.text()[:15000]`. On real S&P 500 `10-Q` filings, the first 15,000 characters consist almost entirely of the cover page, Table of Contents, and condensed GAAP balance sheet/income statement tables—chopping off **Part I Item 2 (MD&A)**, **Part II Item 1 (Legal Proceedings)**, and **Part II Item 1A (Risk Factor Updates)** before chunking.
   - Furthermore, `edgartools` `TenQ` requires part-qualified keys (`"Part I, Item 2"`, `"Part II, Item 1"`, `"Part II, Item 1A"`) because bare `"Item 1"` resolves to Part I Financial Statements rather than Part II Legal Proceedings.
4. **Periodic Amendments (`10-K/A`, `10-Q/A`) are never fetched or staged**:
   - Restated segment/customer concentration numbers, delayed `Exhibit 21` subsidiary schedules, and corrected quarterly risk disclosures filed on `10-K/A` or `10-Q/A` are invisible to [`HistoricalSECHarvester`](../../ingestion/sec/harvester.py) and [`EdgarWorker`](../../ingestion/sec/harvester.py).
5. **Sector-aware Pass-2 gleaning metadata in `canon.py` is unwired**:
   - [`SEC_SECTION_CANON`](../../ingestion/sec/canon.py) already defines `sector_gleaning_only` (`Item 2` $\rightarrow$ `Real Estate`, `Item 4` $\rightarrow$ `Materials` / `Energy`, `Item 7` $\rightarrow$ `Financials`), yet `is_high_signal_section`, `is_high_signal_gleaning_section`, `should_glean_chunk`, and `extract_sec_relationships` do not accept an optional `gics_sector` parameter—forcing an all-or-nothing gleaning policy across heterogeneous S&P 500 sectors.

---

## 2. Decision Drivers & Constraints

1. **Narrow, self-contained Phase 2a scope**: Limit this ADR strictly to expanding YarnBall's *existing* core SEC forms (`10-K`, `10-K/A`, `10-Q`, `10-Q/A`, `8-K`, `8-K/A`) and `Exhibit 99.1`. Defer `Exhibit 10` material contracts and offline XBRL `companyfacts` (`DIO` / ASC 280) to Phase 2b, and foreign/M&A/governance forms (`20-F`, `40-F`, `6-K`, `Form 10`, `S-4`, `DEF 14A`, `SD`, `25`) to Phase 2c/2d.
2. **Strict Point-in-Time (PiT) bitemporal integrity (`ADR-00033`)**: Every fact, chunk, subsidiary, or termination extracted from an amended filing (`10-K/A`, `10-Q/A`, `8-K/A`) must be stamped with the **amendment's own `acceptance_datetime` $\rightarrow$ `known_at` and `valid_from`**, never retroactively backdated to the original filing's timestamp.
3. **Deterministic form-ablation compatibility (`ADR-00045`)**: Form selection must be driven explicitly by canonical `SECFormType` values in `forms=[...]`. Passing `forms=["10-K", "10-Q", "8-K"]` for an ablation run must **never** silently fetch `10-K/A`, `10-Q/A`, or `8-K/A` behind the caller's back.
4. **Zero-internal-dependency leaf preservation**: [`ingestion/sec/canon.py`](../../ingestion/sec/canon.py) must remain Python-stdlib-only and `ontology/` must remain free of `ingestion` imports (`ADR-00045` Invariants #1 and #5).
5. **Backward compatibility for existing callers & fixtures**: Existing method signatures (`extract_10k_sections`, `fetch_recent_8k`, `fetch_10q_filings`, `is_high_signal_section`, `should_glean_chunk`, `extract_sec_relationships`) must remain callable with all existing positional/keyword arguments.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Bundle Phase 2b (`Exhibit 10` material contracts + offline XBRL `companyfacts`) into ADR-00046.** Rejected. `Exhibit 10` legal agreements require dedicated contract-clause filtering and XBRL `companyfacts` requires new numeric parser modules and `contagion/parameters.py` integration. Keeping ADR-00046 scoped purely to Phase 2a keeps diffs reviewable, preserves human-in-the-loop review velocity, and immediately unblocks clean form-ablation experiments (`10-K` sections, `10-Q` narrative, `8-K Ex. 99.1`, and `/A` amendments).
- **Alternative B: Fetch `/A` amendments implicitly via a hidden `include_amendments=True` flag inside `fetch_latest_10k` / `fetch_10q_filings` / `fetch_recent_8k` while keeping `forms=["10-K", "10-Q", "8-K"]`.** Rejected. Implicit amendment injection breaks Leave-One-Form-Out (LOFO) ablation determinism: an experiment requesting `forms=["8-K"]` to ablate `8-K/A` would still silently ingest `8-K/A`. Promoting `10-K/A` and `10-Q/A` to `ACTIVE` and controlling all six core forms explicitly via `forms` (`default_batch_harvest=True` for `10-K`, `10-K/A`, `10-Q`, `10-Q/A`, `8-K`, `8-K/A`) keeps production harvesting complete and ablation runs 100% explicit.
- **Alternative C: Increase `10-Q` raw text truncation from `15_000` to `200_000` characters instead of extracting part-qualified sections.** Rejected. Raw `10-Q` dumps front-load 30,000–80,000 characters of GAAP financial tables and boilerplate notes, inflating `document_chunks` storage and LLM token spend while diluting relational salience. Extracting part-qualified `TenQ` sections (`Part I Item 2`, `Part I Item 3`, `Part II Item 1`, `Part II Item 1A`) with deterministic regex/text fallback captures the operational narrative cleanly.

---

## 4. Decision Outcome

### 4.1 Canon Promotion & Sector-Aware Gleaning (`ingestion/sec/canon.py`)

1. **Promote Phase 2a forms & exhibit to `ACTIVE`**:
   - Promote `SECFormType.FORM_10KA` (`10-K/A`) and `SECFormType.FORM_10QA` (`10-Q/A`) from `PLANNED` to `ACTIVE`, and set `default_batch_harvest=True` on `10-K/A`, `10-Q/A`, and `8-K/A` so `get_default_harvest_forms()` returns `("10-K", "10-K/A", "10-Q", "10-Q/A", "8-K", "8-K/A")` (all `ACTIVE` + `DOMAIN1_OPERATIONAL`).
   - Promote `SEC_EXHIBIT_CANON["Exhibit 99.1"]` from `PLANNED` to `ACTIVE` (keeping `provenance_key=None` or aligning with `SEC_8K` confidence `0.95` without altering `ontology/validation.py`'s existing keys), and set `tracked_exhibits=("Exhibit 99.1",)` on `FORM_8K` and `FORM_8KA`.
2. **Sector-Aware `is_high_signal_section` Helper**:
   - Extend `is_high_signal_section(section_name: Optional[str], *, gics_sector: Optional[str] = None) -> bool`:
     - Preserves existing behavior identically when `gics_sector is None` (returns `True` iff `spec.high_signal_gleaning` matches and `Item 1B` exclusion does not trigger).
     - When `gics_sector` is supplied (case-insensitive normalized match), additionally returns `True` if `spec.sector_gleaning_only` contains that sector (`Item 2 - Properties` for `Real Estate`, `Item 4 - Mine Safety Disclosures` for `Materials` / `Energy`, `Item 7 - Management Discussion and Analysis` for `Financials`), matching either `spec.canonical_name` or `spec.gleaning_match_pattern`.
   - Ensure `SECSectionSpec` entries with `sector_gleaning_only` (`Item 2`, `Item 4`, `Item 7`) carry explicit `gleaning_match_pattern` regexes so shorthand labels (`"Item 2"`, `"Item 7"`, `"MD&A"`) resolve reliably when `gics_sector` matches.
   - Add canonical `10-Q` part-qualified section aliases/patterns in `SEC_SECTION_CANON` (or match existing `Item 2`/`Item 3`/`Item 1A` headers cleanly under `FORM_10Q` and `FORM_10QA`) without breaking `10-K` `Item 2 - Properties` vs. `10-Q` `Part I, Item 2 - MD&A` disambiguation.

### 4.2 Core EDGAR Client Extraction Upgrades (`ingestion/sec/client.py`)

1. **`extract_10k_sections(self, filing: Any) -> Dict[str, Any]`** (works identically for both `10-K` and `10-K/A`):
   - Preserve exact normalized `form` from `getattr(filing, "form", SECFormType.FORM_10K.value)` via `normalize_form_type` (distinguishing `"10-K"` vs. `"10-K/A"`).
   - Extract all six narrative sections using a unified resilient accessor (`_extract_report_item` supporting `edgartools` properties like `tenk.business`, `tenk.risk_factors`, `tenk.management_discussion`, plus `tenk["Item X"]` / `tenk["Item X."]` dict fallbacks):
     - `item_1_business` (`Item 1`)
     - `item_1a_risk_factors` (`Item 1A`)
     - `item_1c_cybersecurity` (`Item 1C`)
     - `item_2_properties` (`Item 2`)
     - `item_3_legal_proceedings` (`Item 3`)
     - `item_7_mda` (`Item 7`)
   - Retain `subsidiaries` via `extract_exhibit_21_subsidiaries(filing)` (capturing delayed `Exhibit 21` schedules filed via `10-K/A`).
2. **`fetch_10k_filings(self, ticker_or_cik: str, year: Optional[int] = None, *, form_type: str = "10-K", limit: int = 2) -> List[Any]`**:
   - Clean parameterized fetcher supporting both `"10-K"` and `"10-K/A"` (with `fetch_latest_10k` retained for 100% backward compatibility).
3. **`extract_10q_sections(self, filing: Any) -> Dict[str, Any]` & `fetch_10q_filings(..., form_type: str = "10-Q")`**:
   - Extract part-qualified `10-Q` / `10-Q/A` narrative sections via `filing.obj()` (`TenQ`):
     - `part1_item2_mda` (`"Part I, Item 2"`, `"part_i_item_2"`)
     - `part1_item3_market_risk` (`"Part I, Item 3"`, `"part_i_item_3"`)
     - `part2_item1_legal_proceedings` (`"Part II, Item 1"`, `"part_ii_item_1"`)
     - `part2_item1a_risk_factors` (`"Part II, Item 1A"`, `"part_ii_item_1a"`)
   - Compose `text` by concatenating extracted narrative sections with explicit statutory headers (`"PART I, ITEM 2 - MANAGEMENT'S DISCUSSION AND ANALYSIS"`, `"PART II, ITEM 1 - LEGAL PROCEEDINGS"`, `"PART II, ITEM 1A - RISK FACTORS"`) so downstream `FinancialSectionSegmenter` labels every chunk unambiguously.
   - If `filing.obj()` yields no structured sections (e.g., legacy HTML or mock objects providing only `.text()`), fall back cleanly to `filing.text()[:DEFAULT_MAX_FILING_CHARS]` (`50_000`, replacing the blind `[:15000]` cap).
4. **`extract_exhibit_99_attachments(self, filing: Any, *, max_chars: int = DEFAULT_MAX_FILING_CHARS) -> Dict[str, str]` & `fetch_recent_8k(..., form_type: str = "8-K", include_exhibits: bool = True)`**:
   - Parameterize `form_type` so `"8-K"` and `"8-K/A"` are fetched cleanly without cross-bleeding during form ablation, preserving each filing's exact canonical `form`.
   - Raise primary `8-K` / `8-K/A` text bound from `[:10000]` to `[:DEFAULT_MAX_FILING_CHARS]` (`50_000`).
   - Inspect `filing.attachments` for `EX-99.1`, `EX-99.2`, ... (`document_type` matching `EX-99` or description mentioning `exhibit 99` / `press release`, explicitly skipping binary/XBRL `EX-101` / `EX-104` / images / `.zip` / `.pdf` attachments).
   - Bound each extracted exhibit independently to `max_chars` (`50_000`) and attach structured fields on each returned 8-K dict:
     - `"exhibits"`: `{doc_type: bounded_text, ...}` (e.g., `{"EX-99.1": "..."}`)
     - `"exhibit_99_1"`: primary `EX-99.1` text (when present)
   - Because commit `5ff2243` wired `normalize_staged_8k_payload`, `assess_sec_8k_disruption`, `dry_run_sec_8k_contagion.py`, and `workbench_api.py` to read `exhibit_99_1` / `exhibits` / `exhibit_texts` with independent `50_000`-char sanitization, fetched `Exhibit 99.1` press releases flow end-to-end into disruption gating automatically.

### 4.3 Harvester Staging, Extraction & Amendment PIT Provenance (`ingestion/sec/harvester.py` & `ingestion/sec/parser.py`)

1. **`HistoricalSECHarvester.harvest_company_year(...)`**:
   - Support all six Phase 2a core forms explicitly when present in `forms_to_fetch`:
     - `10-K` (staged under `10K_{acc_num}/`) and `10-K/A` (staged under `10KA_{acc_num}/`): persist `metadata.json` (including `acceptance_datetime` and exact `form`), `item_1_business.txt`, `item_1a_risk_factors.txt`, `item_1c_cybersecurity.txt`, `item_2_properties.txt`, `item_3_legal_proceedings.txt`, `item_7_mda.txt`, and `subsidiaries.json`.
     - `10-Q` (staged under `10Q_{acc_num}/`) and `10-Q/A` (staged under `10QA_{acc_num}/`): persist `metadata.json` (with `acceptance_datetime`, exact `form`, and extracted section map) plus structured `quarterly_text.txt` and per-section `.txt` files when non-empty.
     - `8-K` (staged under `8K_{acc_num}/`) and `8-K/A` (staged under `8KA_{acc_num}/`): pass `exhibits` and `exhibit_99_1` through `normalize_staged_8k_payload` into `form8k_events.json` (preserving exact `form_type`, `filing_timestamp` from `acceptance_datetime`, and `exhibit_texts`).
   - Record separate stats counters (`10ka_count`, `10qa_count`, `8ka_count`, `exhibit_99_1_count`) in `summary.json` alongside existing keys (`10k_downloaded`, `10q_count`, `8k_count`, `form4_count`, `subsidiaries_count`).
2. **`EdgarWorker.process_company_10k(...)` (extended for `10-K` and `10-K/A`)**:
   - Accept optional `form_type: str = SECFormType.FORM_10K.value` and `gics_sector: Optional[str] = None` (auto-resolved from the offline S&P 500 registry when omitted).
   - Expand `section_specs` from 3 entries to all 6 extracted `10-K` / `10-K/A` sections (`Item 1`, `Item 1A`, `Item 1C`, `Item 2`, `Item 3`, `Item 7`), dynamically recording actually-populated `items_present` in `sec_filings_queue` instead of hardcoding `["Item 1", "Item 1A", "Exhibit 21"]`.
   - Pass exact normalized `form_type` (`"10-K"` or `"10-K/A"`), `gics_sector`, and the filing's own `acceptance_datetime` into `extract_sec_relationships` and `_persist_sec_graph_data`, ensuring every amended edge and chunk carries the amendment's own `known_at` and `valid_from` (`ADR-00033`).
3. **Sector-Aware Gleaning Wiring (`ingestion/sec/parser.py`)**:
   - Add optional keyword parameter `gics_sector: Optional[str] = None` to:
     - `is_high_signal_gleaning_section(section_name, *, gics_sector=None)`
     - `should_glean_chunk(..., gics_sector=None)`
     - `extract_sec_relationships(..., gics_sector=None)`
   - Zero behavior change for callers that do not pass `gics_sector`; enables sector-targeted Pass-2 gleaning (`Item 2` for `Real Estate`, `Item 4` for `Materials` / `Energy`, `Item 7` for `Financials`) whenever sector metadata is provided.

---

## 5. Invariants & Guardrails

1. **Strict Amendment PiT Hygiene (`ADR-00033`)**: `10-K/A`, `10-Q/A`, and `8-K/A` records must always preserve their exact amended `form_type` (`"10-K/A"`, `"10-Q/A"`, `"8-K/A"`) and derive `valid_from` / `known_at` strictly from the amendment's own `acceptance_datetime` (falling back to the amendment's `filing_date`), never from the original filing date or `report_date`.
2. **Independent `50_000`-Char Exhibit Budgeting**: Primary `8-K`/`8-K/A` text and each extracted `EX-99.x` attachment are each bounded independently to `DEFAULT_MAX_FILING_CHARS = 50_000`; binary and XBRL attachments (`EX-101`, `EX-104`, images, PDFs) are never ingested as narrative text.
3. **10-Q Part-Qualified Disambiguation**: `extract_10q_sections` must look up `"Part I, Item 2"` (MD&A), `"Part II, Item 1"` (Legal Proceedings), and `"Part II, Item 1A"` (Risk Factors) explicitly and must never confuse `Part I, Item 1` (Financial Statements) with `Part II, Item 1` (Legal Proceedings).
4. **Explicit Form-Ablation Determinism**: Passing an explicit `forms=[...]` list to `HistoricalSECHarvester.harvest_company_year` must fetch **only** the canonical forms listed (e.g., `forms=["10-K", "8-K"]` must not fetch `10-K/A`, `8-K/A`, `10-Q`, or `4`).
5. **Sector-Gleaning Monotonicity & Safety**: `is_high_signal_section(section, gics_sector=None)` remains invariant for all globally high-signal and `Item 1B` excluded sections; sector-scoped sections (`Item 2`, `Item 4`, `Item 7`) qualify for Pass-2 gleaning **only** when `gics_sector` matches their declared `sector_gleaning_only` tuple.
6. **Zero New Import Cycles & Ontology Parity (`ADR-00045` Invariants #1 and #5)**: `ingestion/sec/canon.py` remains stdlib-only and `ontology/` remains free of `ingestion` imports.

---

## 6. Verification Plan

1. **New Phase 2a Unit & Integration Suite** (`tests/ingestion/sec/test_sec_phase2a_expansion.py` — created upon approval):
   - Mocked `edgartools` `TenK` / `TenQ` / `8-K` / `8-K/A` fixtures verifying:
     - `extract_10k_sections` populates all 6 sections (`Item 1`, `1A`, `1C`, `2`, `3`, `7`) + `Exhibit 21` and preserves `"10-K"` vs. `"10-K/A"`.
     - `extract_10q_sections` extracts part-qualified `Part I Item 2`, `Part I Item 3`, `Part II Item 1`, `Part II Item 1A` without pulling `Part I Item 1` GAAP tables, and falls back to `50_000`-char text cleanly when `.obj()` lacks sections.
     - `fetch_recent_8k` captures primary text up to `50_000` chars, extracts `EX-99.1` / `EX-99.2` text attachments while skipping `EX-101` / `EX-104` / binary files, supports `form_type="8-K/A"`, and feeds into `assess_sec_8k_disruption` to detect disruptions located exclusively inside `Exhibit 99.1`.
     - `HistoricalSECHarvester.harvest_company_year` stages all six core forms (`10-K`, `10-K/A`, `10-Q`, `10-Q/A`, `8-K`, `8-K/A`) cleanly under `forms=None`, strictly respects single-form ablation lists (`forms=["10-K"]`), and preserves amendment `acceptance_datetime` + `form_type`.
     - `EdgarWorker.process_company_10k` processes all 6 extracted sections, stamps `10-K/A` edges/chunks with the amendment's own `known_at`, and passes `gics_sector` into gleaning.
     - Sector-aware gleaning tests confirming `Item 2` gleans for `Real Estate` (not `Information Technology`), `Item 4` gleans for `Materials` / `Energy`, `Item 7` gleans for `Financials`, and `Item 1B` stays excluded for all sectors.
2. **Existing Canon & Regression Suites**:
   - Update `tests/ingestion/sec/test_sec_canon_integrity.py` for the promoted `ACTIVE` forms (`10-K/A`, `10-Q/A`), `Exhibit 99.1`, and `get_default_harvest_forms()` (`6` Domain-1 active forms).
   - Run full-corpus `.venv/bin/mypy`, `.venv/bin/pytest --ignore=tests/knowledge_graph/test_embedding_store.py -v`, `.venv/bin/pytest tests/docs/test_adr_integrity.py`, and `.venv/bin/graphify update .`.
3. **Documentation Sync**:
   - Update [`docs/reference/ingestion-and-universe.md`](../reference/ingestion-and-universe.md) and [`docs/technical-report/01-pit-universe-and-ingestion.md`](../technical-report/01-pit-universe-and-ingestion.md) to reflect Phase 2a `ACTIVE` statuses and expanded `10-K` / `10-Q` / `8-K` extraction coverage.

---

## 7. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Closes the silent `Item 7` drop and unlocks `Item 1C`, `Item 2`, and `Item 3` across all 11 S&P 500 GICS sectors.
  - Replaces blind `10-Q[:15000]` and `8-K[:10000]` truncation with part-qualified `10-Q` narrative extraction and full `50_000`-char `8-K` + `Exhibit 99.1` press-release capture.
  - Brings `10-K/A`, `10-Q/A`, and `8-K/A` amendments into PiT-safe harvesting while preserving strict explicit form-ablation control for **ADR-00047**.
  - Activates sector-aware Pass-2 gleaning without inflating extraction token costs across unrelated sectors.
- **Negative / Trade-offs**:
  - Default batch harvesting (`forms=None`) increases EDGAR requests per company-year (checking `/A` amendments and `EX-99.x` attachments), mitigated by existing SEC rate-limiting (`rate_limit_delay_sec`) and explicit `forms=[...]` overrides.
  - Extracting `Item 1C`, `Item 2`, `Item 3`, and `Item 7` increases `document_chunks` volume per `10-K` (empirically quantified in **ADR-00047**'s form/section ablation scorecard before promoting any further Phase 2b/2c forms).
