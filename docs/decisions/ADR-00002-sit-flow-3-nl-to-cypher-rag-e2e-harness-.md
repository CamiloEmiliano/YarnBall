# ADR-00002: SIT Flow 3 NL-to-Cypher RAG E2E Harness

<!--
STRICT FILENAME CONVENTION:
- Format: ADR-[5_DIGIT_ID]-[40_CHARACTER_SLUG].md
- Total Filename Length: Exactly 53 characters (including '.md')
- Slug Specification: Exactly 40 characters using lowercase alphanumeric and hyphens.
- Padding Rule: If the natural slug is under 40 characters, pad with trailing hyphens ('-').
Filename: ADR-00002-sit-flow-3-nl-to-cypher-rag-e2e-harness-.md (53 chars)
-->

- **ADR ID**: ADR-00002
- **Filename**: `ADR-00002-sit-flow-3-nl-to-cypher-rag-e2e-harness-.md`
- **Date**: 2026-11-10 03:58 UTC
- **Status**: ACCEPTED
- **Target Subsystem**: `retrieval` / `sft_pipeline` / `interactive` / `synthesis`
- **Linked Commits**: `4ae944b` (Flow 3 test harness implementation), `e9a3ef9` (typing and pydantic standardization)

---

## 1. Context & Problem Statement

In a financial GraphRAG system, converting natural language analyst queries into property graph database queries (Text-to-Cypher) presents severe operational and reliability risks:
1. **Schema Property Traps & Cypher Injections**: Language models often hallucinate non-existent node properties (such as `c.ceo_age` or `c.market_cap_rank`) or generate unconstrained Cartesian product traversals that exhaust database memory.
2. **Point-in-Time Temporal Leakage**: Queries that omit temporal filtering evaluate historical events against modern, modified supply-chain relationships, introducing severe lookahead bias into systemic risk analysis.
3. **Mode-Bleed and Missing CoT Traces**: In multi-task language models, text generation can cross-contaminate reasoning modes (e.g. producing unstructured conversation instead of structured chain-of-thought `<think>...</think>` traces grounded in retrieved subgraph nodes).
4. **Lack of Automated Failure Harvesters for Alignment**: When an LLM generates invalid Cypher or hallucinates schema properties, discarding the query silently wastes valuable hard-negative training signals needed for Direct Preference Optimization (DPO).

System Integration Test (SIT) **Flow 3** integrates and validates the entire cross-package analytical loop:
`Natural Language Query -> yarnball-qwen:7b Text-to-Cypher -> Schema AST Whitelist Guardrail -> Temporal Query Encasement -> Graph Execution -> Hop-Ordered Subgraph Verbalization -> Contagion & Hedging Synthesis -> Runtime Rejection Queue Logging`.

---

## 2. Decision Drivers & Constraints

1. **Zero Unvalidated Cypher Execution**:
   - No generated openCypher statement may touch the property graph database engine without passing Abstract Syntax Tree (AST) validation against a strict schema whitelist.
2. **Deterministic Negative Control Short-Circuiting**:
   - Queries referencing non-existent properties, prohibited traversal depths, or out-of-scope concepts must fail fast, return `(none)`, and completely bypass database execution (zero driver calls).
3. **Point-in-Time Temporal Encasement**:
   - Every read query must automatically inject active contract constraints (`coalesce(r.status, 'ACTIVE_CURRENT')`) and enforce the specified historical cutoff date (`as_of_date`).
4. **Automated DPO Hard-Negative Mining**:
   - Every guardrail failure must automatically capture the input query, failed Cypher output, and error classification into the Runtime Rejection Queue (`sft_pipeline/rejection_queue.py`) for offline DPO preference pair synthesis.
5. **Deterministic Testing Budget**:
   - The end-to-end integration harness must run deterministically in `< 3.0 seconds` without requiring live external network services or external model hosting.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Direct Text-to-Cypher Execution without AST Guardrails**:
  - *Rejected*: Financial graph databases are vulnerable to runaway unbounded paths (`[*1..15]`) and hallucinated schema attributes. Direct execution exposes the graph engine to denial-of-service and runtime syntax crashes.
- **Alternative B: Regular Expression (Regex) Validation of Generated Cypher**:
  - *Rejected*: Regular expressions cannot parse nested clauses, function calls, or graph pattern matching accurately. They fail to catch subtle schema traps embedded in `WHERE` predicates.
- **Alternative C: Discarding Invalid LLM Queries Silently**:
  - *Rejected*: Discarding failures throws away the most informative training data for iterative model alignment. Capturing syntax and schema violations into a rejection queue directly powers DPO inflection-lag and mode-bleed training.

---

## 4. Decision Outcome

We establish **Flow 3** as an end-to-end integration and architectural contract in `tests/integration/e2e/test_nl_to_cypher_rag_e2e.py` with the following architectural components:

1. **Pipeline Orchestrator (`retrieval/pipeline.py`)**:
   - Coordinates query ingestion, model serving, guardrail validation, temporal injection, graph query execution, subgraph verbalization, and dual synthesis tasks (Task D: Contagion Reasoning, Task E: Portfolio Hedging).
2. **Schema AST Whitelist Guardrail (`retrieval/guardrails/schema_guardrail.py`)**:
   - Parses the model-generated openCypher into an AST, validating all labels, relationship types, and property keys against the financial ontology before execution.
3. **Temporal Query Filter (`retrieval/temporal.py`)**:
   - Automatically injects active contract predicates and temporal boundary clauses matching the query's `as_of_date`.
4. **Hop-Ordered Subgraph Verbalizer (`retrieval/pipeline.py`)**:
   - Formats the retrieved graph edges into ordered hierarchical layers (`[Hop Distance 1]`, `[Hop Distance 2]`) preserving directed topology and transmission weights.
5. **Runtime Rejection Queue (`sft_pipeline/rejection_queue.py`)**:
   - Serializes rejected queries into an immutable JSONL queue with structured error tags (`SCHEMA_AST_VIOLATION`, `UNBOUNDED_TRAVERSAL`, `MODE_BLEED`).

---

## 5. Invariants & Guardrails

1. **AST Whitelist Invariant**:
   - Generated openCypher must strictly match whitelisted labels (`Company`), relationship types (`SUPPLIES_TO`, `HAS_SUBSIDIARY`), and valid property keys.
2. **Zero-Database-Load on Refusal Invariant**:
   - When a query violates schema guardrails, the graph database session must never be invoked (`len(query_log) == 0`), returning refusal reason and `(none)` verdict.
3. **Temporal Encasement Invariant**:
   - The executed Cypher query must explicitly contain active status filtering (`coalesce(r.status, 'ACTIVE_CURRENT')`) and the `as_of_date` string.
4. **Rejection Provenance Invariant**:
   - Every guardrail failure must log a structured record containing the offending prompt, raw response, error stage, and error message to the rejection directory.
5. **Reasoning Trace Invariant**:
   - Downstream contagion and hedging responses must contain `<think>...</think>` traces grounded in retrieved subgraph nodes.

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Proves the complete closed-loop GraphRAG stack from natural language input to actionable portfolio hedging output.
  - Ensures absolute database safety and zero temporal leakage across all analytical queries.
  - Automatically mines production hard-negatives to train subsequent generations of `yarnball-qwen`.
- **Negative / Trade-offs**:
  - Introduces slight AST parsing latency overhead (~5 to 10 ms per query).
  - Requires maintaining the Cypher AST whitelist in sync with ontology schema evolutions.

