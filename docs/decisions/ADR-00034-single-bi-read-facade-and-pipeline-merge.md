# ADR-00034: Single BI Read Facade & Retrieval Pipeline Consolidation

- **ADR ID**: ADR-00034
- **Filename**: `ADR-00034-single-bi-read-facade-and-pipeline-merge.md`
- **Date**: 2026-11-22 08:48 UTC
- **Status**: PROPOSED
- **Target Subsystem**: `retrieval` / `synthesis` / `explorer` / `tests`
- **Linked Commits**: Pending acceptance

---

## 1. Context & Problem Statement

Two read pipelines with different guarantees answer the same kind of question:

| | Path A | Path B |
|---|---|---|
| Entry | `GraphRAGPipelineOrchestrator.run(AnalystQuery)` ([`retrieval/pipeline.py`](../../retrieval/pipeline.py)) | `FinancialAnalysisEngine` / alias `HybridGraphRAGEngine` ([`synthesis/engine.py`](../../synthesis/engine.py)) and `/search` ([`retrieval/service.py`](../../retrieval/service.py)) |
| Query guard | `CypherASTWhitelistGuardrail` | none on `expand_subgraph` / `enrich_nodes` |
| Temporal | `TemporalQueryFilter` | none ([`ADR-00033`](ADR-00033-bitemporal-pit-contract-for-read-paths--.md)) |
| Cascade metrics | `SparseNetworkCascadeEngine` | none |
| Context budget | `HopOrderedSubgraphVerbalizer` (25 edges) | `LIMIT 50`, unordered |
| LLM client | `YarnBallModelClient` | raw `httpx` in `GroundedSynthesizer` |

Confirmed in source:

1. **[`ADR-00030`](ADR-00030-purge-non-ft-qwen-runtime-dependencies--.md) violation**: [`synthesis/synthesizer.py`](../../synthesis/synthesizer.py) defines `OLLAMA_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "qwen3:8b")` and calls Ollama through `httpx.Client`. If `httpx` is missing, it logs `httpx not installed; returning fallback response`. Synthesis output skips the fine-tuned `yarnball-qwen:7b`, its ChatML templates, and `RuntimeRejectionQueue`.
2. **Domain boundary breach**: `AnalystQuery` carries `portfolio_context` and `market_telemetry`, and `GraphRAGPipelineOrchestrator.run` issues `task="PORTFOLIO_RECOMMENDATION"` when either is present. This contradicts [`ADR-00010`](ADR-00010-directional-basket-peer-cluster-mapping-.md) (`REJECTED`: out of domain scope) and the Domain 1 contract of [`ADR-00011`](ADR-00011-operational-contagion-report-schema-----.md).
3. **Duplicate propagation module**: [`retrieval/shock_propagation.py`](../../retrieval/shock_propagation.py) ([`ADR-00001`](ADR-00001-sit-flow-2-event-shock-contagion-harness.md)) is still exported from `retrieval/__init__.py` and tested, alongside [`contagion/cascade.py`](../../contagion/cascade.py) ([`ADR-00006`](ADR-00006-phase-7-sparse-network-cascade-engine---.md)).
4. **Too many entry-point names**: `GraphRAGPipelineOrchestrator`, `HybridRetriever`, `FinancialAnalysisEngine`, `HybridGraphRAGEngine`, `GroundedSynthesizer`, and `/search` all look like valid ways to query the graph.

A Domain 2 consumer calling the most obvious HTTP endpoint (`/search`) gets the weakest guarantees in the codebase.

---

## 2. Decision Drivers & Constraints

1. **One contract to Domain 2**: every BI answer is an `OperationalContagionReport` produced under the same guard, temporal, cascade, and serialization rules.
2. **Keep hybrid recall**: dense entity and passage seeding (ADR-00021/00022) finds entities Text-to-Cypher misses. It has to feed the guarded pipeline, not replace it.
3. **Live and offline parity**: Memgraph and `InMemorySubgraphStore` must produce identical edge sets for the same query and `PiTContext`.
4. **Single LLM boundary** (ADR-00030).
5. **Domain isolation** (ADR-00010, ADR-00011).

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Patch Path B to parity and keep both paths.** Rejected. That doubles the surface where guarantees must be maintained, and drift will come back.
- **Alternative B: Delete Path B entirely.** Rejected. It throws away dense seeding recall and the passage-to-entity bridge (`expand_passage_entities`).
- **Alternative C: Move the `PORTFOLIO_RECOMMENDATION` hook to a downstream adapter inside this repository.** Open (see open question 1). Deleting it is the minimal option that complies with ADR-00010.

---

## 4. Decision Outcome

1. **Single facade, new `retrieval/facade.py`**:
   ```python
   def query_bi(question: str, ctx: PiTContext, *, store: SubgraphStorePort,
                seeder: HybridSeeder | None = None) -> PipelineResult
   ```
   This is the only public read entry point. `PipelineResult.to_operational_report()` stays the serialization boundary.
2. **`SubgraphStorePort` Protocol**: `traverse(seed_ids, hops, ctx) -> list[dict]` and `execute_guarded(cypher, params, ctx) -> list[dict]`, with two implementations:
   - `MemgraphSubgraphStore`, wrapping the current `execute_graph_query` plus `TemporalQueryFilter` over the ephemeral in-memory Memgraph projection hydrated from the durable Postgres KG ledger ([`ADR-00036`](ADR-00036-fail-fast-infra-and-deployable-topology-.md)).
   - [`retrieval/offline_store.py`](../../retrieval/offline_store.py) `InMemorySubgraphStore`, adapted to the same Protocol over resolved Parquet/ledger exports. Because both projections hydrate from the same canonical resolved ledger after `EntityResolver` runs, live and offline topologies converge without split-brain drift.
3. **`HybridRetriever` becomes `HybridSeeder`** ([`retrieval/hybrid.py`](../../retrieval/hybrid.py)): it keeps `search_dense_entities` (over SCD Type 2 versioned `node_embeddings`, [`ADR-00033`](ADR-00033-bitemporal-pit-contract-for-read-paths--.md)), `search_dense_passages`, `expand_passage_entities`, and the `TextToCQL` seed query, and returns ranked `SeedHit(node_id, score, origin)` values to the orchestrator's seed step. `expand_passage_entities` resolves passage-to-entity links through durable Postgres lineage (`node_embedding_sources` / `kg_edges`) rather than assuming historical `:News` nodes remain permanently in Memgraph RAM beyond `MEMGRAPH_EVENT_TTL_DAYS` ([`ADR-00036`](ADR-00036-fail-fast-infra-and-deployable-topology-.md)). `expand_subgraph` is **removed**; all traversal goes through `SubgraphStorePort.traverse`.
4. **Synthesis behind the model client**: `GroundedSynthesizer` calls `YarnBallModelClient` (new task `GROUNDED_SYNTHESIS` if no existing task fits). `OLLAMA_MODEL_NAME`, `OLLAMA_API_BASE`, and the `httpx` fallback are deleted, and refusals go to `RuntimeRejectionQueue`. `FinancialAnalysisEngine` and `HybridGraphRAGEngine` become a deprecated shim over `query_bi` for one release, then are removed.
5. **Service**: `/search` delegates to `query_bi` and requires `as_of`. New `/report` returns `OperationalContagionReport` JSON. `enrich_nodes` stays only as a node-detail helper and is no longer used to build LLM context.
6. **Domain isolation**: remove `portfolio_context`, `market_telemetry`, and the `PORTFOLIO_RECOMMENDATION` step from [`retrieval/pipeline.py`](../../retrieval/pipeline.py). Extend `TestDomainIsolationInvariant` ([`tests/retrieval/test_operational_contagion_reporting.py`](../../tests/retrieval/test_operational_contagion_reporting.py)) with an AST check that forbids those identifiers anywhere under `retrieval/`.
7. **Callers to migrate**: [`explorer/workbench_api.py`](../../explorer/workbench_api.py), [`tools/dry_run_sec_8k_contagion.py`](../../tools/dry_run_sec_8k_contagion.py), and [`tests/retrieval/test_hybrid_retriever.py`](../../tests/retrieval/test_hybrid_retriever.py) move to `query_bi`.

---

## 5. Invariants & Guardrails

1. **Single entry**: outside `retrieval/facade.py` and `retrieval/pipeline.py`, no module in `retrieval/`, `synthesis/`, `explorer/`, or `tools/` executes Memgraph read Cypher directly. Enforced by an AST test.
2. **Single LLM boundary**: `rg "httpx|qwen3:8b|OLLAMA_MODEL_NAME" retrieval/ synthesis/` returns nothing.
3. **Domain isolation**: `PORTFOLIO_RECOMMENDATION`, `portfolio_context`, and `market_telemetry` never appear under `retrieval/`.
4. **Store parity**: for a fixture universe and `PiTContext`, both `SubgraphStorePort` implementations return identical edge sets and cascade metrics (tolerance `1e-9`).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Domain 2 gets one contract with one set of guarantees.
  - Fine-tuned model outputs are used consistently, with rejection telemetry for synthesis.
  - Contributors can find the entry point immediately.
- **Negative / Trade-offs**:
  - Breaking change for `/search` consumers (adds required `as_of`, changes the response shape).
  - Synthesis inherits the latency of the full pipeline (cascade plus verbalizer) even for simple lookups.
  - Tests that build `HybridGraphRAGEngine` directly must be rewritten.

---

## 7. Open Questions for Review

1. **Task E disposition**: delete the `PORTFOLIO_RECOMMENDATION` hook outright, or move it to a separate Domain 2 adapter package that consumes `OperationalContagionReport`?
2. **`retrieval/shock_propagation.py`**: retire it in favor of `contagion/cascade.py` (and re-point the ADR-00001 SIT harness), or keep it as a separate first-order shock model with a documented role?
3. **`/search` compatibility**: keep a deprecated `/search` with the old shape for one release, or switch it immediately?
