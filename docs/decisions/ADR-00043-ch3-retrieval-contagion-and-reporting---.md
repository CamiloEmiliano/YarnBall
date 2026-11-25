# ADR-00043: Chapter 3 Technical Report — Retrieval, Contagion Algebra & Reporting

- **ADR ID**: ADR-00043
- **Filename**: `ADR-00043-ch3-retrieval-contagion-and-reporting---.md`
- **Date**: 2026-11-25 04:29 UTC
- **Status**: PROPOSED
- **Target Subsystem**: `docs` / `contagion` / `retrieval` / `synthesis` / `explorer`
- **Linked Commits**: Pending acceptance

---

## 1. Context & Problem Statement

1. **Core analytical engine spans four tightly coupled read-path subsystems and 16+ ADRs.** Once statutory filings and events are ingested (Chapter 1, [`ADR-00041`](ADR-00041-ch1-pit-universe-and-statutory-ingestion.md)) and materialized into the bitemporal PostgreSQL ledger, ephemeral Memgraph projection, and Leiden community tables (Chapter 2, [`ADR-00042`](ADR-00042-ch2-ontology-entity-linkage-and-ledger--.md)), YarnBall's analytical read plane executes across `contagion/`, `retrieval/`, `synthesis/`, and `explorer/`:
   - **Pre-ADR foundational components**: Cypher AST read-only whitelist & schema validation ([`retrieval/guardrails/schema_guardrail.py`](../../retrieval/guardrails/schema_guardrail.py)), natural-language-to-OpenCypher translation ([`retrieval/text_to_cql.py`](../../retrieval/text_to_cql.py)), entity metadata enrichment ([`retrieval/enricher.py`](../../retrieval/enricher.py)), and citation-grounded synthesis ([`synthesis/synthesizer.py`](../../synthesis/synthesizer.py), [`synthesis/engine.py`](../../synthesis/engine.py)).
   - **ADR-governed evolution**: Grouped-Leontief sparse network cascade propagation and 18-topology adversarial verification ([`ADR-00006`](ADR-00006-phase-7-sparse-network-cascade-engine---.md), [`ADR-00007`](ADR-00007-cascade-engine-adversarial-verification-.md), [`ADR-00026`](ADR-00026-cascade-parameter-extraction-provenance-.md)), Domain-1 operational contagion report contracts and 1NF time-series flattening ([`ADR-00009`](ADR-00009-sec-8k-computational-reading-dry-run----.md)–[`ADR-00011`](ADR-00011-operational-contagion-report-schema-----.md), [`ADR-00013`](ADR-00013-contagion-event-time-series-join-schema-.md)), offline subgraph parity ([`ADR-00015`](ADR-00015-dynamic-in-memory-offline-subgraph-store.md)), the local FastAPI Workbench replacing Chainlit ([`ADR-00016`](ADR-00016-decommission-legacy-chainlit-ui-frontend.md)–[`ADR-00019`](ADR-00019-pit-kg-explorer-and-cohort-browser-views.md)), mandatory `PiTContext` Cypher injection ([`ADR-00033`](ADR-00033-bitemporal-pit-contract-for-read-paths--.md)), the consolidated `query_bi` read facade ([`ADR-00034`](ADR-00034-single-bi-read-facade-and-pipeline-merge.md)), chunk-backed verbalization ([`ADR-00035`](ADR-00035-text-unit-chunk-layer-and-evidence-links.md)), fail-fast readiness ([`ADR-00036`](ADR-00036-fail-fast-infra-and-deployable-topology-.md)), and rules-first local/global/hybrid routing with provenance-graded `ReportCoverage` ([`ADR-00037`](ADR-00037-pit-community-layer-and-impact-ranking--.md)).
2. **Production economics and GraphRAG retrieval require unified mathematical treatment.** Supply-chain shocks do not diffuse linearly like PageRank: substitutable dual-source vendors within an input group combine additively, whereas complementary component groups act as a strict Leontief bottleneck buffered by physical safety stock (`buffer_days`) and damped across cyclic Strongly Connected Components (SCCs).
3. **Need for Blog Part 3 + AST Read-Path Reference.** Chapter 3 (`docs/technical-report/03-retrieval-contagion-and-reporting.md`) must present this production-network economics, hybrid retrieval, and epistemic reporting stack as a cohesive technical report (*Blog Part 3*) paired with AST-verified call graphs, FastAPI contracts, and Pydantic report schemas.

---

## 2. Decision Drivers & Constraints

1. **Derive the complete cascade propagation & timing algebra**: state the exact Grouped-Leontief fixed-point recurrence, Tarjan SCC feedback identification, spectral radius diagnostic $\rho(W_{\text{eff}})$, Simchi-Levi Time-to-Survive (TTS) onset/downtime propagation, active-buffer friction floor, and 3-tier parameter provenance hierarchy.
2. **Document the single-entry `query_bi` invariant & live/offline isomorphism (`ADR-00033`–`00037`)**: explain how `SubgraphStorePort` guarantees identical Point-in-Time edge traversal across live `MemgraphSubgraphStore` and offline `InMemorySubgraphStore`, and why query routing (`local` / `global` / `hybrid`) is deterministic and zero-LLM.
3. **Preserve the Domain 1 vs. Domain 2 boundary & fail-honest reporting**: explain why `OperationalContagionReport` emits physical capacity contraction, onset/downtime days, and `FILING_GROUNDED` / `MIXED_PROVENANCE` / `STRESS_UPPER_BOUND` provenance grades while strictly prohibiting equity price targets or trading recommendations.
4. **Dual-level visual intuition (`ADR-00040` §4.1)**: pair a **Part A Topic-Level UML Diagram** (`classDiagram`) of the retrieval, cascade, synthesis, and reporting concepts with **Part B AST Call Graphs** (`flowchart`) and FastAPI route catalogs.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Split `contagion/` (cascade math) and `retrieval/` + `explorer/` into two separate chapters.** Rejected. `SparseNetworkCascadeEngine` is invoked directly inside `GraphRAGPipelineOrchestrator` ([`retrieval/pipeline.py`](../../retrieval/pipeline.py)) to rank subgraph edges (`MAX_SUBGRAPH_EDGES = 25`), compute `dropped_impact_share`, and populate `OperationalContagionReport`. Separating the cascade math from the retrieval/reporting pipeline that parameterizes and renders it would split one continuous analytical flow in half.
- **Alternative B: Omit `explorer/` (`workbench_api.py`) as "frontend-only plumbing."** Rejected. [`explorer/workbench_api.py`](../../explorer/workbench_api.py) is a Domain-1 analytical consumer enforcing `PiTContext`, staged Form 8-K audit evaluation, Point-in-Time KG ego exploration, and GICS cohort browsing ([`ADR-00017`](ADR-00017-local-fastapi-kg-contagion-workbench-app.md)–[`ADR-00019`](ADR-00019-pit-kg-explorer-and-cohort-browser-views.md)).
- **Alternative C: Present cascade propagation without its economic literature & pathological topology proofs.** Rejected. Grounding `SparseNetworkCascadeEngine` in Inoue & Todo (2019), Simchi-Levi et al. (2015), Acemoglu et al. (2012), and Bardoscia et al. (2017)—alongside the 10 canonical and 8 pathological topologies of [`ADR-00007`](ADR-00007-cascade-engine-adversarial-verification-.md)—is essential to explaining *why* every equation in [`contagion/cascade.py`](../../contagion/cascade.py) has its exact functional form.

---

## 4. Decision Outcome

Author **`docs/technical-report/03-retrieval-contagion-and-reporting.md`** (*Part 3: Hybrid GraphRAG Retrieval, Network Contagion Algebra & Provenance-Graded Reporting*) and register it in `mkdocs.yml` and `tests/docs/test_docs_integrity.py` with the following exact structure:

### 4.1 Part A — Domain & Theoretical Technical Report (*What & Why* / Blog Part 3)

1. **Production-Network Economics & Grouped-Leontief Cascade Algebra (`ADR-00006`, `ADR-00007`, `ADR-00026`)**:
   - Why standard linear diffusion (PageRank / Katz centrality) mismodels physical supply chains: interchangeable dual-source suppliers within input group $k$ aggregate additively ($\sum_{u \in \text{In}_k(v)} a_{uv} \le 1$), while distinct component groups act as complementary Leontief bottlenecks ($\max_k$).
   - Exact iterative fixed-point shock recurrence in [`SparseNetworkCascadeEngine`](../../contagion/cascade.py):
     $$h_v^{(r)} = \max\!\left(h_v^{(0)},\; \max_{k \in \mathcal{K}(v)} \min\!\left(1,\; \sum_{u \in \text{In}_k(v)} \mathbb{I}\!\big(B_{uv} = 0 \,\lor\, h_u^{(r-1)} d_u > B_{uv}\big)\, a_{uv}^{\text{eff}}\, h_u^{(r-1)}\right)\right)$$
   - Cyclic feedback damping via Tarjan's $O(V + E)$ SCC decomposition ($a_{uv}^{\text{eff}} = \gamma a_{uv}$ with $\gamma = 0.50$ iff $\text{SCC}(u) = \text{SCC}(v) \land \text{dist}(v) \le \text{dist}(u)$), spectral radius $\rho(W_{\text{eff}})$, Simchi-Levi Time-to-Survive (TTS) onset/downtime mechanics ($t_{uv} = t_u + B_{uv}/h_u$, $d_{uv} = \max(0, d_u - B_{uv}/h_u)$, $t_v^{\text{onset}} = \min t_{uv}$, $d_v^{\text{net}} = \max(t_{uv}+d_{uv}) - t_v^{\text{onset}}$), active-buffer friction floor (`0.05` $\rightarrow$ `TOLERATED_BUFFERED`), `disruption_threshold_pct = 25.0`, and the 10 canonical + 8 pathological topology invariants (`ADR-00007`).
   - Empirical cascade parameter extraction & 3-tier provenance in [`contagion/parameters.py`](../../contagion/parameters.py): `EXTRACTED_QUANTITATIVE` $\succ$ `EXTRACTED_QUALITATIVE_SOLE_SOURCE` $\succ$ `DEFAULT_CONSERVATIVE_BOUND`.
2. **Point-in-Time Read Contract, Store Isomorphism & Cypher AST Guardrails (`ADR-00015`, `ADR-00033`, `ADR-00034`)**:
   - Single-entry `query_bi` architecture ([`retrieval/facade.py`](../../retrieval/facade.py)) and `SubgraphStorePort` isomorphism between live `MemgraphSubgraphStore` and offline Parquet-backed `InMemorySubgraphStore` ([`retrieval/offline_store.py`](../../retrieval/offline_store.py)).
   - Read-time `TemporalQueryFilter` ([`retrieval/temporal.py`](../../retrieval/temporal.py)): half-open `[valid_from, valid_to)` rewriting across single-hop `[r:TYPE]` and variable-length `[r*1..H]` paths (`ALL(_rel IN r WHERE ...)`), excluding mutable `r.status` and unanchored edges in `historical` mode.
   - Read-only Cypher AST & schema whitelisting ([`CypherASTWhitelistGuardrail`](../../retrieval/guardrails/schema_guardrail.py)) blocking mutating clauses (`CREATE`, `MERGE`, `DELETE`, `SET`, `DROP`), unbounded hops (`MIN_TRAVERSAL_HOPS = 1`, `MAX_TRAVERSAL_HOPS = 10`), and off-schema labels/predicates.
3. **Hybrid Dense/Lexical Seeding, Rules-First Routing & Impact-Ranked Context Verbalization (`ADR-00034`, `ADR-00035`, `ADR-00037`)**:
   - Multi-channel seeding via [`HybridSeeder`](../../retrieval/hybrid.py) (SCD Type 2 `node_embeddings` cosine similarity + PostgreSQL `node_embedding_sources` durable lineage surviving Memgraph `:News` TTL eviction + guarded `TextToCQL`) paired with Reciprocal Rank Fusion (`search_chunks_hybrid_rrf` in [`knowledge_graph/chunk_store.py`](../../knowledge_graph/chunk_store.py)) combining `pgvector` HNSW and PostgreSQL `tsvector` GIN full-text ranks.
   - Deterministic rules-first query router ([`classify_query_route`](../../retrieval/routing.py)): why entity-anchor + macro-cue classification routes queries to `local` (ego cascade), `global` (map-reduce over PiT `community_reports` with `as_of <= ctx.as_of`), or `hybrid` in $O(1)$ without LLM latency or mode-bleed.
   - Cascade-impact-ordered context pruning in [`HopOrderedSubgraphVerbalizer`](../../retrieval/pipeline.py) (`MAX_SUBGRAPH_EDGES = 25`), tracking `dropped_impact_share` so tail truncation is explicitly quantified rather than silent.
4. **Provenance-Graded Operational Reporting, Grounded Synthesis & Workbench Audit (`ADR-00010`, `ADR-00011`, `ADR-00013`, `ADR-00017`–`00019`, `ADR-00037`)**:
   - Epistemic honesty in [`OperationalContagionReport`](../../retrieval/reporting.py): computing `default_param_impact_share` to classify reports into `FILING_GROUNDED` ($\le 0.10$), `MIXED_PROVENANCE` ($0.10$–$0.50$), or `STRESS_UPPER_BOUND` ($\ge 0.50$), enforcing `ReportCoverage` invariants (`truncated = True` whenever `edges_total > edges_shown`; `data_status != "ok"` whenever `model_mode == DEGRADED_UNREACHABLE`), verbatim chunk citation verification (`validate_verbatim_citations`), and 1NF `ContagionTimeSeriesEventRecord` flattening for downstream econometric joins (`ADR-00013`, `ADR-00038`).
   - Citation-strict narrative synthesis ([`GroundedSynthesizer`](../../synthesis/synthesizer.py), `NO_VERIFIED_EVIDENCE_MESSAGE` abstention) and interactive audit workflows in [`explorer/workbench_api.py`](../../explorer/workbench_api.py).
5. **Topic-Level UML Diagram & Theory-to-Code Crosswalk Table**:
   - Mermaid `classDiagram` modeling `RulesFirstQueryRouter`, `HybridPiTSeeder`, `SubgraphStorePort`, `SparseNetworkCascadeEngine`, `ImpactRankedVerbalizer`, `GroundedSynthesizer`, and `OperationalContagionReport`.
   - Full Crosswalk Table mapping every cascade equation, Cypher guardrail, routing rule, and coverage threshold to `module:symbol`, provenance (`Pre-ADR Foundation` vs. `ADR-00001`/`00002`/`00006`/`00007`/`00009`–`00011`/`00013`/`00015`–`00019`/`00026`/`00033`–`00037`), and verification test files.

### 4.2 Part B — Software Architecture, Call Graphs & Entry Points (*Where & How*)

1. **Subsystem Module Inventory**: complete file-by-file breakdown across `contagion/` (2 modules), `retrieval/` (10 modules including `guardrails/schema_guardrail.py`), `synthesis/` (3 modules), `explorer/` (`workbench_api.py` + static SPA assets), [`tools/dry_run_sec_8k_contagion.py`](../../tools/dry_run_sec_8k_contagion.py), and [`tools/serve_explorer.py`](../../tools/serve_explorer.py).
2. **AST-Verified Execution Call Graphs (3 Mermaid `flowchart` pipelines)**:
   - *Pipeline 1 — Unified `query_bi` Analytical Execution (`local` / `global` / `hybrid`)*: `query_bi` $\rightarrow$ `classify_query_route` $\rightarrow$ `GraphRAGPipelineOrchestrator.run` (`HybridSeeder` + `SubgraphStorePort.traverse` + `SparseNetworkCascadeEngine.propagate` + `fetch_chunks_by_ids` + `HopOrderedSubgraphVerbalizer`) / `search_community_reports_global` $\rightarrow$ `GroundedSynthesizer.synthesize` $\rightarrow$ `PipelineResult`.
   - *Pipeline 2 — SEC Form 8-K Operational Contagion Report & 1NF Export*: `run_sec_8k_dry_run` / `POST /report` / `POST /api/workbench/8k/evaluate` $\rightarrow$ `assess_sec_8k_disruption` $\rightarrow$ `query_bi` / `SparseNetworkCascadeEngine.propagate` $\rightarrow$ `build_operational_contagion_report` $\rightarrow$ `validate_verbatim_citations` & `flatten_report_to_timeseries_records`.
   - *Pipeline 3 — Guarded NL-to-Cypher Translation & Execution*: `TextToCQL.translate` $\rightarrow$ `SubgraphStorePort.execute_guarded` (`CypherASTWhitelistGuardrail.validate` $\rightarrow$ `TemporalQueryFilter.inject_active_coalescence` $\rightarrow$ Memgraph / `InMemorySubgraphStore`).
3. **FastAPI Service Catalog, CLI Tools & Pydantic Schema Reference**:
   - Every HTTP route and request/response contract in [`retrieval/service.py`](../../retrieval/service.py) (`/health`, `/ready`, `/search`, `/report`) and [`explorer/workbench_api.py`](../../explorer/workbench_api.py) (`/api/workbench/catalog`, `/api/workbench/8k/evaluate`, `/api/workbench/kg/subgraph`, `/api/workbench/cohort/filter`, etc.), plus CLI invocations (`tools/dry_run_sec_8k_contagion.py`, `tools/serve_explorer.py`).

---

## 5. Invariants & Guardrails

1. **All `ADR-00040` Series Invariants Apply**: zero host-specific paths, pure GFM + `$...$`/`$$...$$` + `mermaid` in `03-retrieval-contagion-and-reporting.md` (zero `mkdocstrings` `:::` directives in narrative prose), AST-verified call-graph edges, and honest `Pre-ADR Foundation` vs. `ADR-XXXXX` attribution.
2. **Chapter-3 Constant & Symbol Verification**: `tests/docs/test_docs_integrity.py` must verify all `module:symbol` references in Chapter 3's Crosswalk Table alongside live constants (`gamma = 0.50`, `friction_floor = 0.05`, `max_iterations = 5`, `tolerance = 1e-9`, `disruption_threshold_pct = 25.0`, `MIN_TRAVERSAL_HOPS = 1`, `MAX_TRAVERSAL_HOPS = 10`, `MAX_SUBGRAPH_EDGES = 25`, `DEFAULT_FILING_GROUNDED_MAX_SHARE = 0.10`, `DEFAULT_STRESS_UPPER_BOUND_MIN_SHARE = 0.50`).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Consolidates YarnBall's flagship network-contagion math, bitemporal GraphRAG retrieval, and fail-honest Domain-1 reporting contracts into a standalone *Blog Part 3* and complete API/pipeline manual.
  - Sets up Chapter 4 (`ADR-00044`) by showing every runtime LLM task (`EXTRACT_SEC_GRAPH`, `TEXT_TO_CYPHER`, `CONTAGION_REASONING`, `COMMUNITY_REPORT`) and quarantine feedback hook (`RuntimeRejectionQueue`) that the Model Alignment Foundry trains and evaluates.
- **Negative / Trade-offs**:
  - Modifying cascade hyperparameters, `MAX_SUBGRAPH_EDGES`, `ReportCoverage` provenance thresholds, or FastAPI route signatures will require synchronized updates to `03-retrieval-contagion-and-reporting.md` to pass `test_docs_integrity.py`.
