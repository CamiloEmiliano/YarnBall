# ADR-00045: Canonical SEC EDGAR Filing Registry, Section Canon & Form-Ablation Contract

- **ADR ID**: ADR-00045
- **Filename**: `ADR-00045-canonical-sec-edgar-filing-registry-----.md`
- **Date**: 2026-11-27 08:28 UTC
- **Status**: PROPOSED
- **Target Subsystem**: `ingestion` / `retrieval` / `ontology` / `tools` / `tests`
- **Linked Commits**: Pending acceptance

---

## 1. Context & Problem Statement

Statutory SEC EDGAR filings are YarnBall's primary ground-truth data substrate across all **11 GICS sectors of the S&P 500** (`Information Technology`, `Health Care`, `Financials`, `Industrials`, `Consumer Discretionary`, `Consumer Staples`, `Energy`, `Materials`, `Utilities`, `Real Estate`, and `Communication Services`) as well as their non-U.S. upstream Tier-1 counterparties. Yet an audit of all 13 runtime packages reveals that SEC form lists, statutory item enums, section-splitting regexes, Pass-2 gleaning eligibility rules, termination form guards, and epistemic confidence weights are **fragmented across seven modules in four different packages**, causing silent drift, an inverted dependency, and zero ability to empirically measure how individual SEC form types help or hurt Knowledge Graph (KG) quality:

1. **Conflicting hardcoded form lists**:
   - [`ingestion/sec/harvester.py`](../../ingestion/sec/harvester.py) (`HistoricalSECHarvester.harvest_company_year`) hardcodes default batch forms as `["10-K", "8-K", "10-Q", "4"]` (silently pulling `Form 4` insider trades by default) and hardcodes `items_present=["Item 1", "Item 1A", "Exhibit 21"]` for `10-K` queue rows.
   - [`ingestion/sec/client.py`](../../ingestion/sec/client.py) claims `13F` support in its module docstring despite having no `13F` fetcher, omits `10-Q` in `download_filings_for_ticker`, omits `10-K/A` and `10-Q/A` amendments, and truncates `fetch_recent_8k` to `filing.text()[:10000]` without pulling attached `Exhibit 99.1` press releases.
   - [`ingestion/sec/harvester.py`](../../ingestion/sec/harvester.py) hardcodes the [`ADR-00033`](ADR-00033-bitemporal-pit-contract-for-read-paths--.md) Invariant #8 termination allowlist `("8-K", "8-K/A")` inline inside `terminate_relationship_from_8k`.
   - [`tools/build_kg_snapshot.py`](../../tools/build_kg_snapshot.py) and [`tools/reextract_sp500_chunks.py`](../../tools/reextract_sp500_chunks.py) repeat ad-hoc `"10-K"` string fallbacks.
2. **Inverted downstream dependency for statutory Form 8-K items**:
   - `SECForm8KItem` (all 32 statutory items across Sections 1–9) and `SEC_ITEM_DESCRIPTIONS` currently live inside downstream [`retrieval/reporting.py`](../../retrieval/reporting.py), forcing upstream [`ingestion/sec/disruption_gating.py`](../../ingestion/sec/disruption_gating.py) to import `retrieval` and reinforcing the bidirectional `ingestion <--> retrieval` coupling documented in [`00-system-atlas.md`](../technical-report/00-system-atlas.md).
3. **Disconnected section segmentation, disruption gating, and gleaning rules**:
   - [`ingestion/sec/disruption_gating.py`](../../ingestion/sec/disruption_gating.py) defines the 4-tier `_TIER_ITEMS` severity precedence separately from `SECForm8KItem`.
   - [`ingestion/segmentation/chunker.py`](../../ingestion/segmentation/chunker.py) defines 13 `DEFAULT_SECTION_PATTERNS` covering only four of 32 Form 8-K items (`Item 1.01`, `1.02`, `2.01`, `8.01`)—omitting Tier-1 disruption items (`Item 1.03`, `1.04`, `1.05`, `2.04`, `2.05`, `2.06`), `Item 7.01`, and `10-K Item 1C` (Cybersecurity).
   - [`ingestion/sec/parser.py`](../../ingestion/sec/parser.py) maintains a separate `_HIGH_SIGNAL_GLEANING_RE` regex that can silently drift from `DEFAULT_SECTION_PATTERNS` and omits cross-sector S&P 500 sections (`Item 2 - Properties` for REITs, `Item 7` credit/counterparty disclosures for Financials).
4. **Disconnected source-reliability priors**:
   - [`ontology/validation.py`](../../ontology/validation.py) defines `PROVENANCE_CONFIDENCE_MAP` (`SEC_EXHIBIT_21 = 1.00`, `SEC_10K_ITEM1 = 0.98`, `SEC_10Q = 0.95`, `SEC_8K = 0.95`, `SEC_FORM_4 = 1.00`) with no parity link back to the SEC ingestion registry.
5. **No empirical form-ablation substrate**:
   - Because forms are not uniformly normalized, categorized by graph layer, or tracked via a single canon, YarnBall currently cannot run controlled **leave-one-form-out (LOFO)** or **additive form-inclusion** experiments to quantitatively prove whether adding or removing forms (such as `Form 4`, `13F-HR`, `SC 13D/13G`, `424B2`, `Exhibit 10`, or `20-F`/`6-K`) improves or degrades KG modularity, extraction precision, retrieval Recall@K, or cascade provenance grounding.

---

## 2. Decision Drivers & Constraints

1. **Single exhaustive, typed source of truth**: Every relevant SEC EDGAR form type, exhibit, structured XBRL feed, Form 8-K item, priority tier, section regex, gleaning rule, termination policy, graph-layer tag, and base confidence prior must live in one inspectable Python module (`ingestion/sec/canon.py`).
2. **Zero-internal-dependency leaf discipline**: `ingestion/sec/canon.py` must import strictly from the Python standard library (`dataclasses`, `enum`, `re`, `typing`) so every subsystem can consume it cleanly.
3. **Zero new inter-package import cycles**: Consolidating the SEC canon must **not** add an `ontology <--> ingestion` import cycle (`ontology` remains free of `ingestion` imports, with numeric confidence parity enforced by automated unit tests).
4. **Empirical over dogmatic classification**: Forms hypothesized to introduce hub collapse or boilerplate noise (`13F-HR`, `SC 13G`, `424B2`, `S-3`, `Form 4`) are explicitly registered with typed `SECFormCategory`, `SECGraphLayer`, and `SECFormStatus.OPTIONAL_ABLATION` metadata so their KG quality impact can be measured empirically via form-removal / form-addition studies rather than assumed without data.
5. **Backward compatibility & phased delivery**: Existing public imports (`retrieval.reporting.SECForm8KItem`, `ingestion.segmentation.chunker.DEFAULT_SECTION_PATTERNS`, `ingestion.sec.parser.is_high_signal_gleaning_section`) remain unbroken via thin re-exports. Phase 1 performs pure structural consolidation before Phase 2 expands EDGAR fetchers and runs controlled KG ablation studies.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: External YAML or JSON manifest (`config/sec_forms.yaml`).** Rejected. Breaks static `mypy` enum verification (`SECFormType`, `SECForm8KItem`), splits compiled section regexes away from typed code, and introduces runtime file-I/O path fragility in offline unit tests.
- **Alternative B: House the SEC filing canon inside `ontology/sec_canon.py`.** Rejected. `ontology/` owns the 20-class / 26-predicate financial TBox and semantic validation—not SEC EDGAR transport, exhibits, or section grammar. Housing the canon in `ingestion/sec/canon.py` maximizes domain cohesion where engineers inspect EDGAR ingestion, while `ingestion`, `retrieval`, `knowledge_graph`, `sft_pipeline`, `explorer`, and `tools` already depend on `ingestion`.
- **Alternative C: Permanently hard-exclude non-core forms (`13F-HR`, `SC 13D/13G`, `Form 4`, `424B2`) from the canon.** Rejected. Excluding them from the schema prevents standardized tagging and controlled Leave-One-Form-Out / additive ablation benchmarks measuring how each form (and embedder capacity) empirically impacts KG quality.

---

## 4. Decision Outcome

### 4.1 Zero-Dependency Exhaustive SEC EDGAR Canon (`ingestion/sec/canon.py`)

Create `ingestion/sec/canon.py` (Python standard library only) defining six tightly typed primitives:

1. **Taxonomy Enums (`SECFormCategory`, `SECGraphLayer`, `SECIngestionMode`, `SECFormStatus`)**:
   - `SECFormCategory(str, Enum)`:
     - `CORE_PERIODIC` (`10-K`, `10-K/A`, `10-Q`, `10-Q/A`)
     - `CURRENT_EVENT` (`8-K`, `8-K/A`)
     - `CORPORATE_ACTION_MA_SPINOFF` (`10`, `S-1`, `F-1`, `S-4`, `F-4`, `DEFM14A`, `SC TO-T`, `SC 14D-9`)
     - `FOREIGN_COUNTERPARTY` (`20-F`, `20-F/A`, `40-F`, `6-K`)
     - `GOVERNANCE_PROXY` (`DEF 14A`)
     - `SECTOR_RAW_MATERIALS` (`SD`, `ARS`)
     - `DISTRESS_LATE_FILING` (`NT 10-K`, `NT 10-Q`, `25`)
     - `OWNERSHIP_INSIDER` (`4`, `13F-HR`, `SC 13D`, `SC 13G`)
     - `CAPITAL_MARKETS_OFFERING` (`S-3`, `424B2`, `424B5`, `FWP`, `D`)
   - `SECGraphLayer(str, Enum)`:
     - `DOMAIN1_OPERATIONAL`: physical supply-chain, customer, subsidiary, licensing, facility, M&A, spin-off, and operational disruption topology (eligible for `SparseNetworkCascadeEngine`).
     - `GOVERNANCE_INTERLOCK`: executive, director, and Reg S-K Item 404 related-party edges.
     - `CAPITAL_OWNERSHIP`: institutional holdings, activist stakes, and insider beneficial ownership (strictly isolated from physical cascade propagation unless explicitly ablated in flat-graph experiments).
     - `FINANCING_OFFERING`: shelf registrations, structured notes, and debt/equity capital-markets prospectuses.
   - `SECIngestionMode(str, Enum)`: `TEXT_LLM_EXTRACTION`, `STRUCTURED_XML_OR_XBRL`, `HYBRID`.
   - `SECFormStatus(str, Enum)`: `ACTIVE`, `PLANNED`, `OPTIONAL_ABLATION`.

2. **Exhaustive Form Registry (`SECFormType`, `SECFormSpec`, `SEC_FORM_CANON` — 33 Canonical Forms)**:
   Each `SECFormSpec` is a frozen dataclass with fields `(form_type, category, graph_layer, ingestion_mode, status, default_batch_harvest, allow_contract_termination, tracked_exhibits, provenance_key, base_confidence, roadmap_phase, sector_scope, description)`:

   | `SECFormType` | Category | Graph Layer | Ingestion Mode | Phase 1 Status | Default Batch | `valid_to` Close | Base Conf. | Roadmap Phase & Full S&P 500 Role |
   | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
   | `10-K` | `CORE_PERIODIC` | `DOMAIN1_OPERATIONAL` | `HYBRID` | `ACTIVE` | `True` | `False` | `0.98` | **Phase 1** — Annual business, risks, properties, MD&A, `Ex. 21` (all 11 sectors). |
   | `10-Q` | `CORE_PERIODIC` | `DOMAIN1_OPERATIONAL` | `HYBRID` | `ACTIVE` | `True` | `False` | `0.95` | **Phase 1** — Quarterly segment, supply, legal & operational updates (all 11 sectors). |
   | `8-K` | `CURRENT_EVENT` | `DOMAIN1_OPERATIONAL` | `TEXT_LLM_EXTRACTION` | `ACTIVE` | `True` | `True` | `0.95` | **Phase 1** — Material current events, disruptions & terminations (all 11 sectors). |
   | `8-K/A` | `CURRENT_EVENT` | `DOMAIN1_OPERATIONAL` | `TEXT_LLM_EXTRACTION` | `ACTIVE` | `False` | `True` | `0.95` | **Phase 1** — Amended 8-K corrections & termination updates (stamped at amendment `known_at`). |
   | `10-K/A` | `CORE_PERIODIC` | `DOMAIN1_OPERATIONAL` | `HYBRID` | `PLANNED` | `False` | `False` | `0.98` | **Phase 2a** — Restatements, delayed `Ex. 21` & Part III updates (amendment `known_at`). |
   | `10-Q/A` | `CORE_PERIODIC` | `DOMAIN1_OPERATIONAL` | `HYBRID` | `PLANNED` | `False` | `False` | `0.95` | **Phase 2a** — Amended quarterly operational/segment disclosures (amendment `known_at`). |
   | `10` | `CORPORATE_ACTION_MA_SPINOFF` | `DOMAIN1_OPERATIONAL` | `TEXT_LLM_EXTRACTION` | `PLANNED` | `False` | `False` | `0.98` | **Phase 2c** — Spin-off registration & `Ex. 99.1` TSAs / carved-out supply contracts (`GEV`, `GEHC`, `SOLV`, `KVUE`, `CARR`, `OTIS`, `VLTO`). |
   | `S-1` | `CORPORATE_ACTION_MA_SPINOFF` | `DOMAIN1_OPERATIONAL` | `TEXT_LLM_EXTRACTION` | `PLANNED` | `False` | `False` | `0.98` | **Phase 2c** — Domestic IPO prospectus filling the pre-first-`10-K` blind spot. |
   | `F-1` | `CORPORATE_ACTION_MA_SPINOFF` | `DOMAIN1_OPERATIONAL` | `TEXT_LLM_EXTRACTION` | `PLANNED` | `False` | `False` | `0.98` | **Phase 2c** — Foreign Private Issuer IPO prospectus (e.g., `ARM`). |
   | `S-4` | `CORPORATE_ACTION_MA_SPINOFF` | `DOMAIN1_OPERATIONAL` | `TEXT_LLM_EXTRACTION` | `PLANNED` | `False` | `False` | `0.95` | **Phase 2c** — Domestic stock-merger registration: combined facilities & antitrust divestitures (`XOM`+`PXD`, `CVX`+`HES`, `COF`+`DFS`). |
   | `F-4` | `CORPORATE_ACTION_MA_SPINOFF` | `DOMAIN1_OPERATIONAL` | `TEXT_LLM_EXTRACTION` | `PLANNED` | `False` | `False` | `0.95` | **Phase 2c** — Cross-border merger registration statements. |
   | `DEFM14A` | `CORPORATE_ACTION_MA_SPINOFF` | `DOMAIN1_OPERATIONAL` | `TEXT_LLM_EXTRACTION` | `PLANNED` | `False` | `False` | `0.95` | **Phase 2c** — Definitive merger proxy: FTC/DOJ divestiture remedies, closing & breakup terms. |
   | `SC TO-T` | `CORPORATE_ACTION_MA_SPINOFF` | `DOMAIN1_OPERATIONAL` | `TEXT_LLM_EXTRACTION` | `PLANNED` | `False` | `False` | `0.95` | **Phase 2c** — Third-party cash tender offers (common in Health Care/Biotech & Industrial acquisitions). |
   | `SC 14D-9` | `CORPORATE_ACTION_MA_SPINOFF` | `DOMAIN1_OPERATIONAL` | `TEXT_LLM_EXTRACTION` | `PLANNED` | `False` | `False` | `0.95` | **Phase 2c** — Target board solicitation/recommendation detailing tender-offer terms & counterparties. |
   | `20-F` | `FOREIGN_COUNTERPARTY` | `DOMAIN1_OPERATIONAL` | `HYBRID` | `PLANNED` | `False` | `False` | `0.98` | **Phase 2c** — Foreign Private Issuer annual report + `Ex. 8` subsidiaries (`NVO`, `AZN`, `NVS`, `RIO`, `BHP`, `SHEL`, `TSM`, `ASML`, `ARM`). |
   | `20-F/A` | `FOREIGN_COUNTERPARTY` | `DOMAIN1_OPERATIONAL` | `HYBRID` | `PLANNED` | `False` | `False` | `0.98` | **Phase 2c** — Amended Foreign Private Issuer annual report (amendment `known_at`). |
   | `40-F` | `FOREIGN_COUNTERPARTY` | `DOMAIN1_OPERATIONAL` | `HYBRID` | `PLANNED` | `False` | `False` | `0.98` | **Phase 2c** — Canadian MJDS annual report for upstream North American pipelines, energy, fertilizer & rail (`ENB`, `TRP`, `CNQ`, `NTR`, `CNI`, `CP`). |
   | `6-K` | `FOREIGN_COUNTERPARTY` | `DOMAIN1_OPERATIONAL` | `TEXT_LLM_EXTRACTION` | `PLANNED` | `False` | `False` | `0.95` | **Phase 2c** — Foreign Private Issuer current reports (plant halts, earthquakes, export controls, monthly output). |
   | `DEF 14A` | `GOVERNANCE_PROXY` | `GOVERNANCE_INTERLOCK` | `TEXT_LLM_EXTRACTION` | `PLANNED` | `False` | `False` | `0.95` | **Phase 2d** — Definitive proxy: board/executive interlocks (`BOARD_MEMBER_OF`) & Reg S-K Item 404 Related-Party contracts. |
   | `SD` | `SECTOR_RAW_MATERIALS` | `DOMAIN1_OPERATIONAL` | `TEXT_LLM_EXTRACTION` | `PLANNED` | `False` | `False` | `0.95` | **Phase 2d** — Conflict Minerals (Dodd-Frank §1502): Tier-2/3 `3TG` smelters & geographic origin (Materials, Industrials, Aerospace, Electronics). |
   | `NT 10-K` | `DISTRESS_LATE_FILING` | `DOMAIN1_OPERATIONAL` | `TEXT_LLM_EXTRACTION` | `PLANNED` | `False` | `False` | `0.95` | **Phase 2d** — Late annual filing notice: fast audit, accounting, or cyber distress indicator. |
   | `NT 10-Q` | `DISTRESS_LATE_FILING` | `DOMAIN1_OPERATIONAL` | `TEXT_LLM_EXTRACTION` | `PLANNED` | `False` | `False` | `0.95` | **Phase 2d** — Late quarterly filing notice. |
   | `25` | `DISTRESS_LATE_FILING` | `DOMAIN1_OPERATIONAL` | `STRUCTURED_XML_OR_XBRL` | `PLANNED` | `False` | `False` | `1.00` | **Phase 2d** — Exchange delisting notification: statutory ground truth for historical S&P 500 removals. |
   | `4` | `OWNERSHIP_INSIDER` | `CAPITAL_OWNERSHIP` | `STRUCTURED_XML_OR_XBRL` | `OPTIONAL_ABLATION` | `False` | `False` | `1.00` | **Ablation** — Insider trades XML (existing fetcher in `client.py`; opt-in via `forms=["4"]` for KG ablation). |
   | `13F-HR` | `OWNERSHIP_INSIDER` | `CAPITAL_OWNERSHIP` | `STRUCTURED_XML_OR_XBRL` | `OPTIONAL_ABLATION` | `False` | `False` | `1.00` | **Ablation** — Quarterly institutional holdings XML (prime target for flat-vs-layered hub-collapse study). |
   | `SC 13D` | `OWNERSHIP_INSIDER` | `CAPITAL_OWNERSHIP` | `TEXT_LLM_EXTRACTION` | `OPTIONAL_ABLATION` | `False` | `False` | `0.95` | **Ablation** — Activist >5% stake disclosures (spin-off/restructuring campaigns vs. ownership clutter). |
   | `SC 13G` | `OWNERSHIP_INSIDER` | `CAPITAL_OWNERSHIP` | `TEXT_LLM_EXTRACTION` | `OPTIONAL_ABLATION` | `False` | `False` | `0.95` | **Ablation** — Passive >5% institutional stakes (Vanguard/BlackRock/State Street hub-ablation control). |
   | `ARS` | `SECTOR_RAW_MATERIALS` | `DOMAIN1_OPERATIONAL` | `TEXT_LLM_EXTRACTION` | `OPTIONAL_ABLATION` | `False` | `False` | `0.90` | **Ablation** — Glossy Annual Report to Security Holders (redundant vs. `10-K` ablation test). |
   | `S-3` | `CAPITAL_MARKETS_OFFERING` | `FINANCING_OFFERING` | `TEXT_LLM_EXTRACTION` | `OPTIONAL_ABLATION` | `False` | `False` | `0.90` | **Ablation** — Shelf registration statements (volume-vs-precision ablation test). |
   | `424B2` | `CAPITAL_MARKETS_OFFERING` | `FINANCING_OFFERING` | `TEXT_LLM_EXTRACTION` | `OPTIONAL_ABLATION` | `False` | `False` | `0.85` | **Ablation** — Structured-note pricing supplements (high-volume boilerplate & embedder-scaling ablation test). |
   | `424B5` | `CAPITAL_MARKETS_OFFERING` | `FINANCING_OFFERING` | `TEXT_LLM_EXTRACTION` | `OPTIONAL_ABLATION` | `False` | `False` | `0.85` | **Ablation** — Debt/equity shelf takedown prospectuses (financing-noise ablation test). |
   | `FWP` | `CAPITAL_MARKETS_OFFERING` | `FINANCING_OFFERING` | `TEXT_LLM_EXTRACTION` | `OPTIONAL_ABLATION` | `False` | `False` | `0.85` | **Ablation** — Free Writing Prospectuses (marketing term sheets ablation test). |
   | `D` | `CAPITAL_MARKETS_OFFERING` | `FINANCING_OFFERING` | `STRUCTURED_XML_OR_XBRL` | `OPTIONAL_ABLATION` | `False` | `False` | `0.90` | **Ablation** — Exempt private-placement notices (low-signal ablation control). |

3. **Exhaustive Exhibits & Structured Feeds Registry (`SECExhibitSpec`, `SEC_EXHIBIT_CANON` — 9 Entries)**:
   - `Exhibit 21` (`ACTIVE`, `10-K`/`10-K/A`/`S-1`, `provenance_key = "SEC_EXHIBIT_21"`, `base_confidence = 1.00`): Domestic operating subsidiaries.
   - `Exhibit 99.1` (`PLANNED` Phase 2a/2c, `8-K`/`8-K/A`/`6-K`/`10`): Attached press releases (FDA recalls, refinery outages, plant strikes, cyber halts) and spin-off Information Statements.
   - `Exhibit 10` (`PLANNED` Phase 2b, `10-K`/`10-Q`/`8-K`/`10`/`S-1`): Material contracts (pharma licensing/CDMO, LNG/pipeline off-take, DoD/airline, utility PPAs, co-branded cards, PBM contracts).
   - `Exhibit 101 / XBRL` (`PLANNED` Phase 2b, `10-K`/`10-Q`/`20-F`): Inline XBRL & EDGAR `companyfacts` (`DIO` inventory bounds for physical sectors, ASC 280 major-customer shares, [`ADR-00039`](ADR-00039-sp500-pit-stock-split-corporate-actions-.md) stock splits).
   - `Exhibit 8` (`PLANNED` Phase 2c, `20-F`/`20-F/A`/`F-1`): Foreign Private Issuer significant subsidiaries.
   - `Exhibit 2` (`PLANNED` Phase 2c, `8-K`/`S-4`/`F-4`/`10-K`): Plan of Acquisition, Reorganization, Merger, or Liquidation.
   - `Exhibit 95` (`PLANNED` Phase 2d, `10-K`/`10-Q`): Mine Safety Disclosures (Dodd-Frank §1503, linked to `8-K Item 1.04`).
   - `Exhibit 96` (`PLANNED` Phase 2d, `10-K`/`20-F`): Mining Technical Report Summaries (Reg S-K Subpart 1300).
   - `Exhibit 4` (`OPTIONAL_ABLATION`, `10-K`/`8-K`): Debt indentures and covenant instruments.

4. **Relocated Statutory Form 8-K Canon (`SECForm8KItem`, `SEC_ITEM_DESCRIPTIONS`, `SEC_8K_PRIORITY_TIERS`)**:
   - Moves `SECForm8KItem` (all 32 statutory items `Item 1.01`–`Item 9.01` across Sections 1–9) and `SEC_ITEM_DESCRIPTIONS` out of [`retrieval/reporting.py`](../../retrieval/reporting.py) into `ingestion/sec/canon.py`.
   - Moves `_TIER_ITEMS` (Tiers 1–4 severity lattice from [`ADR-00014`](ADR-00014-sec-8k-boilerplate-sanitization-gating--.md)) out of [`ingestion/sec/disruption_gating.py`](../../ingestion/sec/disruption_gating.py) into `SEC_8K_PRIORITY_TIERS` in `ingestion/sec/canon.py`.
   - [`retrieval/reporting.py`](../../retrieval/reporting.py) re-exports `SECForm8KItem` and `SEC_ITEM_DESCRIPTIONS` so all existing imports (`from retrieval.reporting import SECForm8KItem`, `from retrieval import SECForm8KItem`) remain 100% backward-compatible.

5. **Unified Statutory Section & Gleaning Registry (`SECSectionSpec`, `SEC_SECTION_CANON` — 25 Sections)**:
   - Each `SECSectionSpec` pairs `(canonical_name, header_pattern, applicable_forms, high_signal_gleaning, sector_gleaning_only, gleaning_match_pattern)`.
   - Unifies all 13 existing `10-K`/`10-Q`/`8-K`/`Exhibit 21` patterns from [`ingestion/segmentation/chunker.py`](../../ingestion/segmentation/chunker.py) **plus**:
     - `10-K`: `Item 1C - Cybersecurity`, `Item 4 - Mine Safety Disclosures`;
     - `8-K` Tier-1 & Reg-FD headers previously unsegmented: `Item 1.03` (Bankruptcy/Receivership), `Item 1.04` (Mine Safety Shutdowns), `Item 1.05` (Material Cybersecurity Incidents), `Item 2.02` (Results of Operations), `Item 2.04` (Acceleration of Financial Obligations), `Item 2.05` (Exit or Disposal Activities / Plant Closures), `Item 2.06` (Material Impairments), `Item 7.01` (Regulation FD);
     - `20-F` (reserved for Phase 2c): `Item 3.D - Risk Factors`, `Item 4 - Information on the Company`.
   - Drives **both** `DEFAULT_SECTION_PATTERNS` in [`ingestion/segmentation/chunker.py`](../../ingestion/segmentation/chunker.py) **and** `is_high_signal_gleaning_section` in [`ingestion/sec/parser.py`](../../ingestion/sec/parser.py) from one table (preserving `Item 1B` exclusion and keeping Phase-1 gleaning eligibility strictly consistent while making Tier-1 8-K operational items high-signal).

6. **Canonical Helper Functions**:
   - `normalize_form_type(raw: Optional[str]) -> str`
   - `get_default_harvest_forms() -> Tuple[str, ...]` (returns active `default_batch_harvest=True` forms: `("10-K", "10-Q", "8-K")`)
   - `get_forms_by_status(status: SECFormStatus) -> Tuple[SECFormType, ...]`
   - `get_forms_by_layer(layer: SECGraphLayer) -> Tuple[SECFormType, ...]`
   - `is_termination_eligible_form(form_type: Optional[str]) -> bool` (`True` iff `allow_contract_termination=True`)
   - `get_section_patterns(form_type: Optional[str] = None) -> List[Tuple[str, Pattern[str]]]`
   - `is_high_signal_section(section_name: Optional[str]) -> bool`
   - `get_sec_provenance_confidence_map() -> Dict[str, float]`

### 4.2 Consumer Rewiring Map (Phase 1)

| Consumer Module | Refactor Action |
| :--- | :--- |
| `ingestion/sec/canon.py` | New zero-internal-dependency exhaustive SEC registry module. |
| [`ingestion/sec/__init__.py`](../../ingestion/sec/__init__.py) | Export `SECFormType`, `SECFormStatus`, `SECFormCategory`, `SECGraphLayer`, `SECForm8KItem`, `SEC_FORM_CANON`, `SEC_EXHIBIT_CANON`, `SEC_SECTION_CANON`. |
| [`ingestion/sec/client.py`](../../ingestion/sec/client.py) | Remove stale `13F` docstring claim; reference `SECFormType` constants in `get_filings(form=...)` and staged metadata dicts. |
| [`ingestion/sec/harvester.py`](../../ingestion/sec/harvester.py) | Replace hardcoded `["10-K", "8-K", "10-Q", "4"]` with `list(get_default_harvest_forms())`; replace inline `("8-K", "8-K/A")` check with `is_termination_eligible_form(form_type)`; normalize `form_type` via `normalize_form_type`. |
| [`ingestion/sec/disruption_gating.py`](../../ingestion/sec/disruption_gating.py) | Import `SECForm8KItem` and `SEC_8K_PRIORITY_TIERS` from `ingestion.sec.canon` instead of redefining `_TIER_ITEMS` or importing `SECForm8KItem` from `retrieval.reporting`. |
| [`ingestion/segmentation/chunker.py`](../../ingestion/segmentation/chunker.py) | Construct `DEFAULT_SECTION_PATTERNS` directly from `ingestion.sec.canon.get_section_patterns()`. |
| [`ingestion/sec/parser.py`](../../ingestion/sec/parser.py) | Delegate `is_high_signal_gleaning_section` to `ingestion.sec.canon.is_high_signal_section`, removing duplicate `_HIGH_SIGNAL_GLEANING_RE`. |
| [`retrieval/reporting.py`](../../retrieval/reporting.py) | Import and re-export `SECForm8KItem` and `SEC_ITEM_DESCRIPTIONS` from `ingestion.sec.canon` (removing duplicate definitions). |
| [`tools/build_kg_snapshot.py`](../../tools/build_kg_snapshot.py) & [`tools/reextract_sp500_chunks.py`](../../tools/reextract_sp500_chunks.py) | Normalize form-type defaults via `normalize_form_type` / `SECFormType.FORM_10K.value`. |

### 4.3 Multi-Phase Execution Roadmap

- **Phase 1 (Scope of This ADR — Exhaustive Canon Consolidation & Drift Gates)**:
  - Implement `ingestion/sec/canon.py`, rewire all consumers in Section 4.2, add `tests/ingestion/sec/test_sec_canon_integrity.py`, and update [`00-system-atlas.md`](../technical-report/00-system-atlas.md) (`Table 0.4` file count), [`01-pit-universe-and-ingestion.md`](../technical-report/01-pit-universe-and-ingestion.md) (consolidated SEC EDGAR canon table), and [`docs/reference/ingestion-and-universe.md`](../reference/ingestion-and-universe.md).
- **Phase 2 (Follow-On ADRs — Full S&P 500 Expansion & Empirical Ablation)**:
  - **Phase 2a (`8-K Exhibit 99.1`, `10-K/A`, `10-Q/A`)**: Fetch attached `Exhibit 99.1` press releases inside `EdgarClient.fetch_recent_8k` and ingest amended periodic filings stamped with amendment `known_at`.
  - **Phase 2b (`Exhibit 10`, Sector-Aware Gleaning & Offline XBRL `companyfacts`)**: Extract material contracts (`Exhibit 10`), evaluate sector-aware Pass-2 gleaning (`Item 2` REITs, `Item 7` Financials), and ingest offline XBRL Days Inventory Outstanding (physical sectors only) + ASC 280 major-customer shares to reduce `DEFAULT_CONSERVATIVE_BOUND` fallback share.
  - **Phase 2c (`Form 10`, `S-1`/`F-1`, `S-4`/`F-4`/`DEFM14A`, `SC TO-T`/`SC 14D-9`, `20-F`/`20-F/A`/`40-F`/`6-K`)**: Add spin-off TSAs, IPO prospectuses, merger/tender-offer antitrust divestitures, and non-U.S. upstream Tier-1 counterparty disclosures.
  - **Phase 2d (`DEF 14A`, `Exhibit 95`/`96`, `Form SD`, `NT 10-K`/`NT 10-Q`, `Form 25`)**: Add board interlocks / Item 404 related-party contracts, mine-safety & reserve technical reports, Tier-2/3 conflict-mineral smelters, late-filing distress alerts, and statutory exchange delisting notices.

### 4.4 Empirical KG Quality & SEC Form-Removal Ablation Contract

Because every persisted node, edge, and text chunk carries normalized `form_type` (and `section`/`exhibit`) stamped from `SECFormType`, `ingestion/sec/canon.py` provides the deterministic slice key for a **Leave-One-Form-Out (LOFO) and Additive Form-Inclusion KG Quality Study** (to be built as a dedicated offline evaluation harness in Phase 2):

1. **Controlled Experimental Axes**:
   - **Form-Removal / Inclusion Axis**: Evaluate a fixed S&P 500 multi-sector cohort (covering all 11 GICS sectors) across incremental subsets: Baseline (`10-K` + `Ex. 21`) $\rightarrow$ `+ 8-K` $\rightarrow$ `+ 8-K Ex. 99.1` $\rightarrow$ `+ 10-Q` $\rightarrow$ `+ Ex. 10 / XBRL` $\rightarrow$ `+ Form 10 / S-4 / 20-F / 6-K` $\rightarrow$ `+ OPTIONAL_ABLATION` forms (`4`, `13F-HR`, `SC 13D/13G`, `424B2`, `S-3`).
   - **Graph-Isolation Axis**: Compare ingesting `CAPITAL_OWNERSHIP` and `FINANCING_OFFERING` forms into the flat operational graph versus strictly segregated `SECGraphLayer` partitions.
   - **Retrieval / Embedder Axis**: Compare `qwen3-embedding:0.6b` (`1024`-d) vs. larger embedders and compact cross-encoder rerankers under controlled form-volume dilution to measure empirically whether embedder scale compensates for low-signal filing volume.
2. **Quantitative KG Quality Scorecard (Numbers & Plots Required Before Any Architectural Claim)**:
   - **Topology & Community Health**: Leiden modularity $Q$, max node degree & top-1% degree Gini, 2-hop ego edge expansion relative to `MAX_SUBGRAPH_EDGES = 25`, and `dropped_impact_share`.
   - **Epistemic Grounding**: Share of `OperationalContagionReport` downstream impact classified as `FILING_GROUNDED` ($\le 0.10$) vs. `MIXED_PROVENANCE` vs. `STRESS_UPPER_BOUND` ($\ge 0.50$).
   - **Extraction & Reasoning Fidelity**: `RuntimeRejectionQueue` failure rate stratified by `form_type` and `ErrorStage`, holdout ontology rejection rate ([`ADR-00031`](ADR-00031-iterative-predicate-convergence-protocol.md)), and Phase-5 4-axis torture accuracy (`Verdict`, `HopTrace`, `Provenance`, `CIK`).
   - **Retrieval Precision & Systems Cost**: Hybrid RRF Recall@K / MRR on chunk-cited queries, total `document_chunks`, `pgvector` HNSW footprint, and extraction token budget per form type.

---

## 5. Invariants & Guardrails

1. **Zero-Internal-Dependency Canon**: `ingestion/sec/canon.py` must import strictly from the Python standard library; AST tests forbid importing any of YarnBall's 13 runtime packages inside `canon.py`.
2. **Complete Form Canon Bijection**: Every `SECFormType` enum member must have an entry in `SEC_FORM_CANON` with valid `SECFormCategory`, `SECGraphLayer`, `SECIngestionMode`, `SECFormStatus`, and `0.0 < base_confidence <= 1.0`.
3. **Exhaustive Form 8-K Coverage**: Every member of `SECForm8KItem` (all 32 items) must appear in **both** `SEC_ITEM_DESCRIPTIONS` and **exactly once** across `SEC_8K_PRIORITY_TIERS` (zero missing items, zero duplicates across tiers).
4. **Strict Termination Eligibility (`ADR-00033` Invariant #8)**: `is_termination_eligible_form(f)` must return `True` **if and only if** `f` normalizes to `8-K` or `8-K/A`.
5. **Zero Drift with Ontology Confidence Priors**: Every `ACTIVE` form and exhibit in `SEC_FORM_CANON` / `SEC_EXHIBIT_CANON` (plus `Form 4`) with a `provenance_key` must match `ontology.validation.PROVENANCE_CONFIDENCE_MAP[provenance_key]` identically—enforced in unit tests without introducing an `ontology -> ingestion` import cycle.
6. **Single Section & Gleaning Source**: `ingestion.segmentation.chunker.DEFAULT_SECTION_PATTERNS` must equal `get_section_patterns()`, and `ingestion.sec.parser.is_high_signal_gleaning_section` must delegate directly to `ingestion.sec.canon.is_high_signal_section` (excluding `Item 1B` while covering all `high_signal_gleaning=True` sections).
7. **Domain-1 Default Batch Hygiene**: `get_default_harvest_forms()` must return exclusively `ACTIVE` + `DOMAIN1_OPERATIONAL` forms (`"10-K"`, `"10-Q"`, `"8-K"`); `OPTIONAL_ABLATION` and `PLANNED` forms enter harvesting only when explicitly requested by caller or ablation harness.

---

## 6. Verification Plan

1. **New Canon Integrity Suite (`tests/ingestion/sec/test_sec_canon_integrity.py`)**:
   - AST check verifying `ingestion/sec/canon.py` has zero internal YarnBall imports.
   - Bijection checks across all `SECFormType` members $\leftrightarrow$ `SEC_FORM_CANON`, and all 32 `SECForm8KItem` entries $\leftrightarrow$ `SEC_ITEM_DESCRIPTIONS` and `SEC_8K_PRIORITY_TIERS`.
   - Termination guard assertions (`8-K`, `8-K/A` allowed; `10-K`, `10-Q`, `4`, `20-F`, `6-K`, `13F-HR`, `424B2`, empty string rejected).
   - Exact numeric parity between `get_sec_provenance_confidence_map()` and `ontology.validation.PROVENANCE_CONFIDENCE_MAP`.
   - Backward-compatibility identity check: `retrieval.reporting.SECForm8KItem is ingestion.sec.canon.SECForm8KItem`.
   - Segmentation + gleaning alignment check confirming Tier-1 8-K headers (`Item 1.03`, `1.05`, `2.05`, `2.06`) and `10-K Item 1C` segment cleanly and `Item 1B` remains excluded from gleaning.
2. **Full-Corpus Static & Regression Gate**:
   - `.venv/bin/mypy` (0 errors across full corpus).
   - `.venv/bin/pytest --ignore=tests/knowledge_graph/test_embedding_store.py -v` (all unit, integration, architecture, and docs integrity suites passing, including `test_chapter0_mermaid_package_dependency_graph_matches_ast`).
   - `.venv/bin/graphify update .` to synchronize the codebase knowledge graph.

---

## 7. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Eliminates seven-file scatter of SEC filing definitions in favor of one exhaustive, statically typed registry (`ingestion/sec/canon.py`) covering 33 SEC form types, 9 exhibits/structured feeds, 32 Form 8-K items, and 25 statutory sections.
  - Fixes the inverted `SECForm8KItem` dependency from `ingestion` into `retrieval/reporting.py` and removes the phantom `13F` docstring claim.
  - Creates the clean experimental substrate (`form_type`, `SECGraphLayer`, `OPTIONAL_ABLATION`) needed to run controlled Leave-One-Form-Out KG quality and embedder-scaling ablation studies with real numbers and plots.
- **Negative / Trade-offs**:
  - Callers expecting `HistoricalSECHarvester.harvest_company_year(...)` to harvest `Form 4` implicitly without passing `forms=[...]` must pass `forms=["10-K", "8-K", "10-Q", "4"]` explicitly.
  - Adding `ingestion/sec/canon.py` increments the `ingestion` `.py` file count by `+1` (`21` $\rightarrow$ `22`), requiring the corresponding update in [`00-system-atlas.md`](../technical-report/00-system-atlas.md) (`Table 0.4`, mechanically enforced by [`tests/docs/test_docs_integrity.py`](../../tests/docs/test_docs_integrity.py)).
