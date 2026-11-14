# ADR-00027: Canonical Staged Form 8-K Envelope & Workbench Discovery

- **ADR ID**: ADR-00027
- **Filename**: `ADR-00027-canonical-staged-8k-envelope-and-catalog.md`
- **Date**: 2026-11-14 19:25 UTC
- **Status**: PROPOSED
- **Target Subsystem**: `ingestion` / `tools` / `explorer`
- **Linked Commits**: Pending acceptance

---

## 1. Context & Problem Statement

Testing [`tools/dry_run_sec_8k_contagion.py`](../../tools/dry_run_sec_8k_contagion.py) and [`explorer/workbench_api.py`](../../explorer/workbench_api.py) against actual staged SEC Form 8-K files on disk (`data/dry_run_staging/sec/AAPL/2020/8K_0001193125-20-225672/form8k_events.json`) exposed a silent contract failure:

1. **Staged 8-K Schema Mismatch**: The staged `form8k_events.json` emits keys `accession_number`, `filing_date`, `form`, `items`, and `text`, omitting `ticker`, `company_name`, `cik`, and `headline`. Running `dry_run_sec_8k_contagion.py --filing ... --mock` exited `0` while parsing `DISRUPTED ENTITY: Unknown Corporate Entity (UNKNOWN | CIK: 0000000000)` and `0.0% plant halt for 0.0 calendar days`.
2. **Incomplete Catalog Discovery**: `GET /api/catalog` in `explorer/workbench_api.py` discovers only the two synthetic fixture files under `tests/fixtures/filings/`, returning zero real staged 8-K filings from `data/dry_run_staging/sec/` or `data/sec_filings/`.

---

## 2. Decision Drivers & Constraints

1. **Zero Silent `UNKNOWN` Entity Misclassification**: Real staged 8-Ks must resolve accurate `ticker`, `cik`, `company_name`, normalized `items` (`List[str]`), `headline`, and `content`/`text`.
2. **Backward Compatibility With Already-Staged Archives**: Existing 2018–2024 `form8k_events.json` files already harvested on disk must load cleanly without forcing an immediate re-download from SEC EDGAR.
3. **Preserved Allowlist Security**: Expanding workbench `/api/catalog` discovery to `data/dry_run_staging/sec/` and `data/sec_filings/` must remain strictly confined to allowlisted workspace roots (`ADR-00017`).

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Require Re-Harvesting All Staged SEC 8-K Files Before Testing**:
  - *Rationale for Rejection*: Unnecessarily blocks E2E testing on external EDGAR downloads when `ticker`, `cik`, `filing_date`, `items`, and `text` are already present in the directory path (`.../{TICKER}/{YEAR}/8K_{ACCESSION}/form8k_events.json`) and EDGAR cover block.
- **Alternative B: Duplicate `InlineForm8KPayload` Logic Separately in CLI vs. FastAPI**:
  - *Rationale for Rejection*: Reintroduces schema drift between `dry_run_sec_8k_contagion.py` and `explorer/workbench_api.py`.

---

## 4. Decision Outcome

Define a **Canonical Staged Form 8-K Envelope & Loader** shared across `ingestion/sec/`, [`tools/dry_run_sec_8k_contagion.py`](../../tools/dry_run_sec_8k_contagion.py), and [`explorer/workbench_api.py`](../../explorer/workbench_api.py):

1. **Canonical Harvest Output Envelope**:
   - Update SEC 8-K staging (`HistoricalSECHarvester`) to always write `cik`, `ticker`, `company_name`, `form_type`, `accession_number`, `filing_date`, `items` (`List[str]`), `headline`, and `content` (`text` preserved as optional alias).
2. **Shared Normalizer (`normalize_staged_8k_payload`)**:
   - Accept both fixture (`InlineForm8KPayload`) and legacy staged (`form8k_events.json`) shapes:
     - Map `text` $\rightarrow$ `content` and comma-delimited `"8.01,9.01"` $\rightarrow$ `["Item 8.01", "Item 9.01"]`,
     - Infer missing `ticker` from enclosing staging path (`sec/{TICKER}/...`) and cross-reference `cik`/`company_name` via `SP500UniverseManager` or `sec_companies` lookup, and
     - Synthesize concise `headline` from the first non-boilerplate item heading when `headline` is absent.
   - Fail cleanly with a clear validation warning/error if `ticker` and `company_name` remain unresolvable rather than silently reporting `UNKNOWN`.
3. **Workbench Catalog Discovery Extension**:
   - Extend `_discover_filings` in `explorer/workbench_api.py` to include allowlisted `form8k_events.json` staged files under `data/dry_run_staging/sec/` (alongside `tests/fixtures/filings/`) with bounded metadata loading so `/api/catalog` stays fast.

---

## 5. Invariants & Guardrails

1. **No Silent `UNKNOWN` Entity Fallbacks**: Valid staged SEC 8-K files containing a determinable ticker/accession MUST resolve entity identity prior to disruption gating.
2. **Shared Single Normalizer**: Both CLI dry-run tools and FastAPI `POST /api/evaluate-8k` MUST call the same normalizer helper.
3. **Zero Absolute Paths Invariant**: All references remain strictly repository-relative (`../../<path>`).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Unlocks immediate E2E testing in CLI and **Event Audit** UI against the real 2018–2024 `AAPL` Form 8-K filings already staged on disk.
  - Eliminates silent `0.0%` / `UNKNOWN` false-negative audits caused solely by field-name mismatch.
- **Negative / Trade-offs**:
  - Scanning large `data/dry_run_staging/sec/` trees during `/api/catalog` requires lightweight caching or bounded depth scanning so catalog requests stay sub-second.
