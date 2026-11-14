# ADR-00024: Clause-Scoped Negation, Epistemic Hedging & Structural Relation Filtering

- **ADR ID**: ADR-00024
- **Filename**: `ADR-00024-clause-scoped-negation-and-hedge-filter-.md`
- **Date**: 2026-11-14 19:25 UTC
- **Status**: PROPOSED
- **Target Subsystem**: `ontology`
- **Linked Commits**: Pending acceptance

---

## 1. Context & Problem Statement

Boundary testing [`ontology/guardrails.py`](../../ontology/guardrails.py) and [`ontology/compiler.py`](../../ontology/compiler.py) against real SEC Form 10-K Risk Factors (`data/sec_filings/`) and multi-clause financial sentences exposed four concrete linguistic fragilities:

1. **Whole-Prompt Modal Hedge Wipeout**: `Tier4OntologyGate.apply_guardrails(..., prompt=...)` rejects the **entire** extracted prediction to `"(none)"` if `is_hypothetical_or_hedged(prompt)` matches anywhere inside `prompt`. Probing real Item 1A Risk Factors showed `AMZN` (`'may from time to time consider'`) and `GOOGL` (`'might'`) trigger `HEDGED: True`—meaning any legal hedge anywhere in a multi-sentence chunk nukes valid factual supplier relationships extracted from surrounding sentences.
2. **Fixed 60-Character Comma-Crossing Negation Window**: `has_unnegated_pattern` looks back 60 characters stopping only at `.`, `;`, `!`. Probing `"Apple avoided supply chain delays, and expanded production capacity at TSMC."` and `"Apple terminated its agreement with LegacyChip and expanded orders with TSMC."` caused `classify_polarity_from_text` to return `NEUTRAL_STABLE` because `"avoided"` and `"terminated"` (listed inside `NEGATION_REGEX`) sit within 60 chars to the left of `"expanded"`, suppressing valid `EXPANDING_BULLISH` signals across comma/conjunction boundaries.
3. **Unpruned Noise Predicates in Multi-Edge Outputs**: In `repair_extraction_relations`, if an output contains both a canonical relation and a `NON_SCHEMA_NOISE_PREDICATES` edge (e.g., `(A)-[:SUPPLIES_TO]->(B) (B)-[:CORRELATES_WITH]->(C)`), `has_at_least_one_valid_or_repaired_relation` becomes `True` and returns `result` *including* the unpruned `[:CORRELATES_WITH]` edge.
4. **Blind Termination Repair & Cypher-Only Regex Parsing**: `repair_extraction_relations` parses only raw Cypher `-[:PRED {props}]->` strings (failing on nested braces `{...}` and inapplicable to JSON extraction) and repairs every `TERMINATED`/`CANCELLED` edge blindly to `SUPPLIES_TO`, corrupting terminated licensing, partnership, or customer relations.

---

## 2. Decision Drivers & Constraints

1. **Sentence/Clause Locality**: Hedging and negation cues must evaluate only the specific clause or sentence supporting a candidate triple—not an entire 10-K risk passage or adjacent company's clause.
2. **Per-Edge Structural Filtering**: Noise pruning and predicate repair must operate per parsed triple/edge (JSON dict or AST-tokenized clause), never via coarse string-level flags that leak mixed noise edges.
3. **Semantic Preservation on Lifecycle Events**: Repairing a `TERMINATED` predicate must respect the underlying domain context (`LICENSES_FROM`, `PARTNERED_WITH`, `SUPPLIES_TO`) rather than mislabeling every terminated relationship as a supplier edge.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Shrink `window_chars` From 60 to 25**:
  - *Rationale for Rejection*: Still breaks across commas/conjunctions on long clauses while failing on normal passive constructs (`"did not, during fiscal 2023, expand..."`).
- **Alternative B: Remove `MODAL_HEDGE_REGEX` Completely**:
  - *Rationale for Rejection*: Allows purely speculative M&A rumors (`"evaluating the possibility of acquiring..."`) to enter Memgraph as factual `ACQUIRED_BY` edges. Scoping hedging to the target sentence/clause preserves specificity without collateral damage.

---

## 4. Decision Outcome

Refactor linguistic gating and relation repair inside [`ontology/compiler.py`](../../ontology/compiler.py) and [`ontology/guardrails.py`](../../ontology/guardrails.py):

1. **Clause-Level Splitting (`split_financial_clauses`)**:
   - Segment text on sentence boundaries plus coordinating/subordinate boundaries (`,\s*(?:and|but|while|however|although)\b`, `;`, `.`, `!`, `\band\b` between distinct verb phrases) before running `NEGATION_REGEX` and `MODAL_HEDGE_REGEX`.
   - Remove non-negation terminal lifecycle verbs (`terminated`, `dissolved`, `ended`, `cancelled`, `canceled`) from `NEGATION_REGEX` where they represent negative economic events rather than grammatical negation, or evaluate polarity per entity-bearing clause.
2. **Evidence-Span Hedge Checking**:
   - In `Tier4OntologyGate.apply_guardrails`, apply `is_hypothetical_or_hedged` to the **prediction or evidence sentence matching the extracted entities**, not an entire unsplit filing section `prompt`.
3. **Triple-Level Pruning & Context-Aware Termination Repair**:
   - Parse predictions into discrete triple records (supporting both JSON `{nodes, edges}` from `graph_store.py` and Cypher clause lists).
   - Actively drop any individual edge whose predicate belongs to `NON_SCHEMA_NOISE_PREDICATES` (returning `"(none)"` only when zero valid edges remain).
   - When repairing a `LIFECYCLE_TERMINATION_PREDICATES` edge, infer the base predicate from co-occurring/context cues (`license`/`royalty` $\rightarrow$ `LICENSES_FROM`, `joint venture`/`partnership` $\rightarrow$ `PARTNERED_WITH`, `customer` $\rightarrow$ `CUSTOMER_OF`, falling back to `SUPPLIES_TO` only when supply cues or default apply).

---

## 5. Invariants & Guardrails

1. **No Cross-Clause Polarity Contamination**: Negation or termination on Entity A in Clause 1 MUST NOT invert or suppress an explicit action on Entity B in Clause 2.
2. **Zero Mixed-Noise Leakage**: No individual edge carrying a `NON_SCHEMA_NOISE_PREDICATES` predicate may survive `Tier4OntologyGate` even alongside valid edges.
3. **Sentence-Scoped Hedge Suppression**: A boilerplate modal hedge in an unrelated Risk Factors sentence MUST NOT discard factual disclosures elsewhere in the same filing chunk.
4. **Zero Absolute Paths Invariant**: All references remain strictly repository-relative (`../../<path>`).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Prevents wholesale extraction loss on real SEC Item 1A Risk Factors (`AMZN`, `GOOGL`) while maintaining strict suppression of speculative M&A/contract rumors.
  - Eliminates `CORRELATES_WITH` noise leakage and extends guardrails to JSON extraction payloads.
- **Negative / Trade-offs**:
  - Clause splitting introduces modest regex/parser complexity and requires curated regression fixtures covering multi-clause SEC sentences.
