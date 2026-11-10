# ADR-00005: SIT Flow 1 PiT SEC Filing to KG Ingest

<!--
STRICT FILENAME CONVENTION:
- Format: ADR-[5_DIGIT_ID]-[40_CHARACTER_SLUG].md
- Total Filename Length: Exactly 53 characters (including '.md')
- Slug Specification: Exactly 40 characters using lowercase alphanumeric and hyphens.
- Padding Rule: If the natural slug is under 40 characters, pad with trailing hyphens ('-').
Filename: ADR-00005-sit-flow-1-pit-sec-filing-to-kg-ingest--.md (53 chars)
-->

- **ADR ID**: ADR-00005
- **Filename**: `ADR-00005-sit-flow-1-pit-sec-filing-to-kg-ingest--.md`
- **Date**: 2026-11-10 03:12
- **Status**: PROPOSED
- **Target Subsystem**: `universe` / `ingestion` / `curation` / `ontology` / `knowledge_graph`
- **Linked Commits**: `4ae944b` (Flow 1 pipeline integration test harness), `041e541` (universe constituent resolution and dictionary-ticker guardrails)

---

## 1. Context & Problem Statement

Transforming raw SEC regulatory filings (Form 10-K, Form 10-Q, Exhibit 21) into an institutional Financial Knowledge Graph exposes the pipeline to four critical failure modes:

1. **Survivorship & Ticker Collision Bias**: Tickers change over time, are recycled across exchanges, or collide with common English dictionary words (e.g. `A`, `ALL`, `NOW`, `ON`, `IT`). Ingesting historical filings using ticker symbols as primary node identifiers causes cross-company contamination and lookahead bias.
2. **Long-Context Semantic Drift & Negation Drop**: Monolithic 10-K filings regularly exceed 50,000 words. Torture testing revealed that feeding unsegmented chunks (> 2,500 tokens) to extraction models results in a catastrophic drop in negation sensitivity (0/5 pass rate) and causes openCypher syntax errors.
3. **Contract Termination Dilemma**: When an SEC filing discloses that a supplier agreement was cancelled, LLMs frequently extract off-ontology predicates like `[:TERMINATED]`. If the schema gate discards `[:TERMINATED]` to `(none)`, valuable regulatory intelligence is lost. Conversely, if the relationship is forced into `[:SUPPLIES_TO]` without lifecycle flags, downstream risk engines traverse an inactive contract, producing false-positive contagion shock warnings.
4. **Subsidiary Graph Fragmentation**: Contracts signed by wholly-owned operating subsidiaries (e.g. DISH DBS Corporation) fail to connect to the parent publicly-traded entity (DISH Network Corp) without Exhibit 21 corporate tree resolution.

System Integration Test (SIT) **Flow 1** formally verifies the end-to-end ingestion journey:
`Historical Date & Ticker -> Point-in-Time CIK Resolution -> Section Segmentation (<= 1,500 tokens) -> Tier-4 Ontology Gate Repair & Rollup -> Cypher Schema AST Whitelist Verification`.

---

## 2. Decision Drivers & Constraints

1. **Immutable Entity Grounding**:
   - Nodes must be uniquely identified by their permanent 10-digit zero-padded SEC Central Index Key (CIK) rather than ephemeral stock tickers.
2. **Point-in-Time (PiT) Temporal Consistency**:
   - Entity constituent membership and supply relationships must be evaluated strictly as of the filing date (`as_of_date`), preventing future corporate actions from leaking into historical snapshots.
3. **Strict 1,500-Token Chunking Boundary**:
   - Input segments passed to entity/relation extraction models must not exceed 1,500 tokens to preserve negation sensitivity and syntax precision.
4. **Lifecycle Relation Preservation**:
   - Terminated relationships must not be discarded; they must be normalized into canonical relation predicates with `{status: 'TERMINATED', is_active: false}` properties.
5. **Exhibit 21 Corporate Rollup**:
   - Operating subsidiaries must roll up to their parent issuer CIK prior to graph insertion.
6. **Offline Determinism**:
   - Tier 2 pipeline integration tests must execute 100% offline using local fixtures in `< 3.0 seconds` without live SEC EDGAR network queries.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Ticker-Based Primary Graph Keys**:
  - *Rejected*: Tickers change during corporate reorganizations (e.g. FB -> META) and collide with English words (`CAN`, `ARE`, `FAST`). CIKs are immutable legal identifiers assigned by the SEC.
- **Alternative B: Monolithic Section Ingestion (> 3,000 tokens)**:
  - *Rejected*: Empirical torture testing demonstrated that large chunks cause attention decay, dropping negation modifiers and failing openCypher bracket balance checks.
- **Alternative C: Discarding Terminated Contracts to `(none)`**:
  - *Rejected*: Severed contracts provide vital competitive and supply-shock signals. Pruning them blinds the knowledge graph to historical contract events.

---

## 4. Decision Outcome

We establish **Flow 1** as an automated integration pipeline and architectural contract in `tests/integration/pipelines/test_filing_to_graph_flow.py` orchestrating:

1. **Point-in-Time Universe Manager (`universe/sp500.py`)**:
   - Resolves historical S&P 500 constituents and 10-digit CIKs based on chronological entry/exit registry records.
2. **Financial Section Segmenter (`ingestion/segmentation/chunker.py`)**:
   - Chunks SEC filings along Item boundaries (Item 1 Business, Item 1A Risk Factors) enforcing a strict `max_tokens=1500` limit with structured filing header metadata.
3. **Tier-4 Ontology Gate (`ontology/guardrails.py`)**:
   - Applies deterministic runtime repair:
     - Maps `LIFECYCLE_TERMINATION_PREDICATES` (`TERMINATED`, `CANCELLED`, `EXPIRED`) to canonical `SUPPLIES_TO` while setting `{status: 'TERMINATED', is_active: false}`.
     - Rolls up operating subsidiary CIKs to parent issuer CIKs via `SUBSIDIARY_TO_PARENT_CIK_MAP`.
     - Prunes speculative text via `is_hypothetical_or_hedged()` and discards non-schema noise predicates (`[:COMPARES_TO]`).
4. **Cypher AST Whitelist Guardrail (`retrieval/guardrails/schema_guardrail.py`)**:
   - Verifies that compiled Cypher mutation statements conform to whitelisted labels, relationships, and property keys.

---

## 5. Invariants & Guardrails

1. **Point-in-Time CIK Invariant**:
   - Every company node generated from filings must carry a valid 10-digit zero-padded CIK matching the constituent registry as of the filing date.
2. **Token Ceiling Invariant**:
   - Chunks produced by `FinancialSectionSegmenter` must not exceed 1,500 tokens (`chunk.token_count <= 1500`) and must retain section headers and CIK metadata.
3. **Lifecycle Relation Repair Invariant**:
   - When an extraction output contains a termination verb, the gate must emit a canonical predicate containing `{status: 'TERMINATED', is_active: false}`.
4. **Subsidiary Rollup Invariant**:
   - A recognized Exhibit 21 subsidiary CIK must never appear as an isolated root node; it must roll up to its parent entity CIK.
5. **Schema AST Whitelist Invariant**:
   - All generated graph mutation statements must pass AST token verification before execution against the graph store.

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Guarantees point-in-time historical fidelity with zero lookahead or survivorship bias.
  - Prevents supply-chain shock algorithms from falsely traversing terminated relationships while preserving historical contract records.
  - Connects subsidiary disclosures directly to parent issuer equities.
- **Negative / Trade-offs**:
  - Requires maintaining the Exhibit 21 subsidiary-to-parent CIK mapping registry in `ontology/guardrails.py`.
