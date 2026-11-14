# ADR-00023: Unified Live Ontology Gate & Schema Convergence Across Writers

- **ADR ID**: ADR-00023
- **Filename**: `ADR-00023-unified-live-ontology-gate-and-schema---.md`
- **Date**: 2026-11-14 19:25 UTC
- **Status**: PROPOSED
- **Target Subsystem**: `ontology` / `knowledge_graph` / `ingestion`
- **Linked Commits**: Pending acceptance

---

## 1. Context & Problem Statement

Audit of [`ontology/`](../../ontology/README.md) against the live Memgraph writers revealed two fundamental structural breaks between the formal ontology specification and actual graph generation:

1. **Live KG Writers Bypass Ontology Validation**: Neither `store_graph_entities` in [`knowledge_graph/graph_store.py`](../../knowledge_graph/graph_store.py) (news stream) nor `EdgarWorker` in [`ingestion/sec/harvester.py`](../../ingestion/sec/harvester.py) (SEC filings) calls `validate_and_orient_triple`, `is_generic_placeholder`, `detect_and_break_ownership_cycles`, or `Tier4OntologyGate`. Those checks only execute inside [`knowledge_graph/entity_resolver.py`](../../knowledge_graph/entity_resolver.py) as an offline/post-hoc pass.
2. **Predicate & Schema Drift Across Modules + Silent Bypass**:
   - `graph_store.py` (`SYSTEM_EXTRACTION_PROMPT`) prompts the news LLM for `ACQUIRED`, `INVESTS_IN`, `LEADS`, `PRODUCES`, `IMPACTS`, `REPORTS`, `PARTNERED_WITH`, `COMPETES_WITH`.
   - [`ontology/schema.py`](../../ontology/schema.py) (`ONTOLOGY_SCHEMA`) defines `ACQUIRED_BY`, `CEO_OF`/`EXECUTIVE_OF`, and omits `INVESTS_IN`, `PRODUCES`, `IMPACTS`, `REPORTS`, and `SOLE_SOURCE_DEPENDENT_ON`.
   - [`ontology/guardrails.py`](../../ontology/guardrails.py) (`CANONICAL_RELATION_PREDICATES`) includes `SOLE_SOURCE_DEPENDENT_ON`, `SPUN_OFF_FROM`, `FILED`, `OWNS`, `HAS_SUBSIDIARY`, which are absent from `ONTOLOGY_SCHEMA`.
   - **Verified probe**: [`ontology/validation.py`](../../ontology/validation.py) returns any predicate not listed in `ONTOLOGY_SCHEMA` unmodified (`if clean_rel not in ONTOLOGY_SCHEMA: return (src_id, src_lbl, tgt_id, tgt_lbl, clean_rel)`), allowing invalid triples such as `(Apple:COMPANY)-[:LEADS]->(Beats:COMPANY)` to pass domain/range validation silently.

---

## 2. Decision Drivers & Constraints

1. **Single Source of Ontological Truth**: Prompt vocabularies, domain/range axioms (`ONTOLOGY_SCHEMA`), and Tier-4 guardrails (`CANONICAL_RELATION_PREDICATES`) must derive from one unified specification.
2. **Inline Write-Time Enforcement**: Generic placeholders, invalid domain/range pairs, and off-ontology predicates must be blocked or canonicalized *before* `MERGE` into Memgraph, not only in optional offline batch runs.
3. **Closed-World Strictness With Explicit Quarantine**: Unknown predicates should never silently bypass SHACL-style domain/range axioms; they must either map via an explicit alias table or enter a quarantine log for ontology review.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Retain Open Custom Predicate Passthrough in `validate_and_orient_triple`**:
  - *Rationale for Rejection*: Renders `ONTOLOGY_SCHEMA` advisory only; any hallucinated verb (`CORRELATES_WITH`, `MENTIONED_NEAR`) or vocabulary-drifted predicate (`LEADS`, `ACQUIRED`) bypasses domain/range checks completely.
- **Alternative B: Leave Validation Exclusively in Post-Hoc `EntityResolver`**:
  - *Rationale for Rejection*: Allows corrupted entities (`"Management"`, inverted `Company -> CEO_OF -> Person` edges, cyclic ownership) to pollute live Memgraph and `node_embeddings` before batch cleanup runs.

---

## 4. Decision Outcome

Converge the ontology registry and wire live validation into all Memgraph ingestion paths:

1. **Unified Registry in [`ontology/schema.py`](../../ontology/schema.py)**:
   - Reconcile `RelationPredicate`, `ONTOLOGY_SCHEMA`, `graph_store.SYSTEM_EXTRACTION_PROMPT`, and `guardrails.CANONICAL_RELATION_PREDICATES` into a single canonical predicate & alias mapping (e.g., mapping `ACQUIRED` $\rightarrow$ inverse `ACQUIRED_BY`, `LEADS` $\rightarrow$ `EXECUTIVE_OF`, explicitly modeling or deprecating `INVESTS_IN`, `PRODUCES`, `IMPACTS`, `REPORTS`, `SOLE_SOURCE_DEPENDENT_ON`, `OWNS`, `SPUN_OFF_FROM`, `FILED`).
   - Generate `SYSTEM_EXTRACTION_PROMPT` allowed node/edge lists dynamically from `ontology/schema.py` so prompts cannot drift from validators.
2. **Strict Mode in `validate_and_orient_triple`**:
   - Remove silent passthrough for predicates outside `ONTOLOGY_SCHEMA`: map known aliases (`PREDICATE_ALIASES`) first, and return `None` (rejected/quarantined) for unrecognized predicates in strict mode (`strict=True` default for production writers).
3. **Shared `validate_graph_payload` Write-Time Gate**:
   - Insert a single deterministic pre-write gate called by both `store_graph_entities` (`graph_store.py`) and `EdgarWorker` (`ingestion/sec/harvester.py`) that enforces:
     - `is_generic_placeholder` node filtering,
     - `validate_and_orient_triple(..., strict=True)` domain/range + inversion repair,
     - symmetric edge canonicalization (`canonicalize_symmetric_edge`), and
     - quarantine logging for dropped triples.

---

## 5. Invariants & Guardrails

1. **Zero Prompt-Schema Drift**: Allowed node labels and edge types in LLM extraction prompts MUST derive programmatically from [`ontology/schema.py`](../../ontology/schema.py).
2. **No Ungated Memgraph Writes**: Every automated ingestion writer (`graph_store.py`, `ingestion/sec/harvester.py`) MUST pass candidate nodes/edges through the canonical ontology gate prior to `MERGE`.
3. **Explicit Rejection Over Silent Passthrough**: Unknown predicates MUST NOT pass `validate_and_orient_triple(..., strict=True)` unchecked.
4. **Zero Absolute Paths Invariant**: All references remain strictly repository-relative (`../../<path>`).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Guarantees Memgraph news edges and SEC edges obey one unified type system and domain/range algebra from the moment of ingestion.
  - Exposes new recurring relation candidates cleanly in quarantine logs instead of fragmenting edge types.
- **Negative / Trade-offs**:
  - Strict validation may initially drop recall on unmapped news predicates until `PREDICATE_ALIASES` and `ONTOLOGY_SCHEMA` cover desired financial relations.
