# ADR-00042: Chapter 2 Technical Report — Ontology, Entity Linkage & Graph Ledger

- **ADR ID**: ADR-00042
- **Filename**: `ADR-00042-ch2-ontology-entity-linkage-and-ledger--.md`
- **Date**: 2026-11-25 04:22 UTC
- **Status**: PROPOSED
- **Target Subsystem**: `docs` / `ontology` / `embedding` / `knowledge_graph`
- **Linked Commits**: Pending acceptance

---

## 1. Context & Problem Statement

1. **Pre-ADR ontological and record-linkage foundations blended with 14+ post-`ADR-00001` refinements.** Between raw chunked disclosures (Chapter 1, [`ADR-00041`](ADR-00041-ch1-pit-universe-and-statutory-ingestion.md)) and read-time contagion analysis (Chapter 3) sits YarnBall's core knowledge-representation, entity-resolution, vector-indexing, and graph-persistence layer (`ontology/`, `embedding/`, `knowledge_graph/`). Its design combines:
   - **Pre-ADR foundational modules**: the OWL 2 / Description Logic TBox schema ([`ontology/schema.py`](../../ontology/schema.py)), 5-axis GAAP ASC 280 reification model ([`ontology/reification.py`](../../ontology/reification.py)), frame-semantic OpenCypher DSL compiler ([`ontology/compiler.py`](../../ontology/compiler.py)), Jaro-Winkler / initialism / symmetric-edge alignment ([`ontology/alignment.py`](../../ontology/alignment.py)), multi-source epistemic confidence & ownership-DAG cycle breaking ([`ontology/validation.py`](../../ontology/validation.py)), and Union-Find entity resolution ([`knowledge_graph/entity_resolver.py`](../../knowledge_graph/entity_resolver.py)).
   - **Post-`ADR-00001` architectural hardening**: fail-fast asymmetric `1024`-dim `qwen3-embedding:0.6b` embeddings and SCD Type 2 `pgvector` node versioning ([`ADR-00020`](ADR-00020-fail-fast-embedder-and-zero-vector-guard.md)–[`ADR-00022`](ADR-00022-unified-sec-news-dense-vector-coverage--.md), [`ADR-00033`](ADR-00033-bitemporal-pit-contract-for-read-paths--.md), [`ADR-00035`](ADR-00035-text-unit-chunk-layer-and-evidence-links.md)), unified write-time ontology gating with clause-scoped negation/hedging, placeholder stoplists, dynamic Exhibit 21 CIK rollup, and predicate convergence ([`ADR-00023`](ADR-00023-unified-live-ontology-gate-and-schema---.md)–[`ADR-00025`](ADR-00025-entity-stoplist-and-dynamic-cik-rollup--.md), [`ADR-00029`](ADR-00029-structured-llm-extraction-and-quarantine.md), [`ADR-00031`](ADR-00031-iterative-predicate-convergence-protocol.md)), cross-source live-to-snapshot linking ([`ADR-00028`](ADR-00028-cross-source-entity-linking-and-kg-sync-.md)), the durable PostgreSQL ledger vs. ephemeral RAM Memgraph architecture ([`ADR-00036`](ADR-00036-fail-fast-infra-and-deployable-topology-.md)), and Point-in-Time hierarchical Leiden communities ([`ADR-00037`](ADR-00037-pit-community-layer-and-impact-ranking--.md)).
2. **Unchecked LLM graph extraction fails catastrophically on financial prose.** Without formal domain/range axioms, clause-scoped modality gates, subsidiary-to-parent CIK rollup, and conflict-guarded record linkage, language models fabricate edges from counterfactual risk factors (*"may consider..."*), collapse competitors sharing generic words, create cyclic corporate ownership graphs, or exhaust in-memory graph RAM with raw filing prose and unbounded news nodes.
3. **Need for Blog Part 2 + AST Knowledge-Ledger Reference.** Chapter 2 (`docs/technical-report/02-ontology-linkage-and-ledger.md`) must synthesize this representation, linkage, vector, community, and storage stack into a standalone publication-grade technical report (*Blog Part 2*) paired with AST-verified call graphs and relational/graph/Parquet schemas.

---

## 2. Decision Drivers & Constraints

1. **Formalize the algebraic & statistical foundations**: document the TBox domain/range axioms, 5-axis reification, clause-scoped modality/negation logic, Jaro-Winkler / Levenshtein / CIK-guarded Union-Find record linkage, multi-source confidence calibration, ownership-DAG cycle breaking, Leiden modularity / internal connectivity invariants, and SCD Type 2 vector versioning.
2. **Explain the durable-before-ephemeral storage split (`ADR-00036`)**: articulate *why* PostgreSQL (`kg_nodes`, `kg_edges`, `document_chunks`, `node_embeddings`, `community_reports`) is the durable bitemporal source of truth while Memgraph operates purely as a prose-stripped, TTL-bounded (`MEMGRAPH_EVENT_TTL_DAYS = 90`) in-memory traversal projection alongside offline Parquet snapshots.
3. **Dual-level visual intuition (`ADR-00040` §4.1)**: pair a **Part A Topic-Level UML Diagram** (`classDiagram`) of the ontology, linkage, vector, ledger, and community concepts with **Part B AST Call Graphs** (`flowchart`) tracing write-time validation, snapshot materialization, and Memgraph boot hydration.
4. **Honest Pre-ADR vs. ADR provenance & mechanized drift checks**: attribute pre-ADR implementations vs. ADR evolutions accurately in the Theory-to-Code Crosswalk Table, validated by `tests/docs/test_docs_integrity.py`.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Split `embedding/` into Chapter 3 (Retrieval) instead of Chapter 2.** Rejected. Dense vector generation (`qwen3-embedding:0.6b`), zero-norm validation (`\|v\|_2^2 \ge 10^{-8}`), VRAM-preserving CPU live serving (`num_gpu: 0`), `embedding_schema_metadata` locking, and SCD Type 2 `node_embeddings` / `document_chunks` persistence happen **at write/index time** alongside `knowledge_graph/`. Chapter 2 covers write-time vector indexing and storage; Chapter 3 (`ADR-00043`) covers read-time hybrid RRF & `pgvector` seeding.
- **Alternative B: Place `knowledge_graph/communities.py` (`ADR-00037`) entirely in Chapter 3.** Rejected. `build_communities`, `summarize_communities`, `persist_community_reports`, and `export_communities_parquet` are structural graph-partitioning and snapshot-materialization operations executed over `kg_edges` (`tools/build_kg_snapshot.py`). Chapter 2 documents PiT Leiden partitioning and persistence, while Chapter 3 documents read-time `GLOBAL_THEMATIC` query routing and map-reduce synthesis over those precomputed reports.
- **Alternative C: Document `ontology/` as a static enum list without linguistic & graph-theoretic invariants.** Rejected. The ontology's real quantitative value lies in its active write-time enforcement: clause-scoped hedge/negation suppression, dynamic Exhibit 21 CIK rollup, lifecycle termination repair, ownership DAG acyclicity, degree-burst quarantine, and `ADR-00031` holdout rejection-rate convergence ($\le 2\%$).

---

## 4. Decision Outcome

Author **`docs/technical-report/02-ontology-linkage-and-ledger.md`** (*Part 2: Formal Financial Ontology, Entity Linkage & Durable-vs-Ephemeral Graph Storage*) and register it in `mkdocs.yml` and `tests/docs/test_docs_integrity.py` with the following exact structure:

### 4.1 Part A — Domain & Theoretical Technical Report (*What & Why* / Blog Part 2)

1. **5-Axis Reified Financial Ontology & TBox Axiomatization**:
   - Why flat `(subject, predicate, object)` triples fail institutional supply-chain analysis, and how [`ontology/schema.py`](../../ontology/schema.py) + [`ontology/reification.py`](../../ontology/reification.py) formalize 20 `EntityClass` concepts and 26 `RelationPredicate` types under domain/range tuples $(\mathcal{D}_R, \mathcal{R}_R, \text{sym}_R)$, deterministic alias inversion (`PREDICATE_ALIASES`), and symmetric lexicographic ordering $R(a, b) \equiv R(\min(a,b), \max(a,b))$.
   - The 5-Axis reification tuple (`AnnotatedTriple`): Entity & Identifier Grounding (Axis 1), Relational Predicate (Axis 2), Directional Economic Polarity (Axis 3), US GAAP ASC 280 Financial Materiality Tier (`CRITICAL_TIER_1` $\ge 10\%$ revenue / sole-source vs. `MATERIAL_TIER_2` vs. `COMMODITY_TIER_3`, Axis 4), and Bitemporal Regulatory Lifecycle (`ACTIVE_CURRENT` vs. `TERMINATED`, Axis 5).
2. **Linguistic Modality Gating, Structural Graph Axioms & Predicate Convergence (`ADR-00023`–`00025`, `00029`, `00031`)**:
   - Syntactic clause boundary segmentation (`split_financial_clauses`) in [`ontology/compiler.py`](../../ontology/compiler.py): why epistemic hedges (`MODAL_HEDGE_REGEX`) and polarity negations (`NEGATION_REGEX` within a 60-character pre-trigger window) must be evaluated strictly within entity-bearing clauses of `chunk.text` ([`ADR-00035`](ADR-00035-text-unit-chunk-layer-and-evidence-links.md)).
   - `Tier4OntologyGate` ([`ontology/guardrails.py`](../../ontology/guardrails.py)) & inline validation ([`ontology/validation.py`](../../ontology/validation.py)): anonymized-counterparty placeholder rejection (`"Customer A"`, `"a major supplier"`), context-aware termination repair (`[:TERMINATED]` $\rightarrow$ canonical relation + `{status: "TERMINATED", is_active: false}`), predecessor edge interval closure (`resolve_edge_state_transitions`), multi-source confidence calibration:
     $$c(p, k) = \min\!\Big(1.0,\; c_0(p) + 0.05 \cdot \min(4,\, k - 1)\Big)$$
     iterative DFS ownership-DAG cycle breaking (`detect_and_break_ownership_cycles` pruning $\arg\min_{e \in C} c(e)$), degree-burst hallucination quarantine (`degree_burst_threshold = 25`), and `ADR-00031` holdout predicate convergence (`target_max_rate = 0.02`, `min_support = 3` in [`ontology/convergence.py`](../../ontology/convergence.py)).
3. **Cross-Source Entity Resolution & Subsidiary Rollup (`ADR-00025`, `ADR-00028`)**:
   - Two-stage linkage across [`EdgarEntityLinker`](../../ingestion/sec/parser.py) (4-tier Postgres `pg_trgm` lookup: exact ticker $\rightarrow$ normalized name $\rightarrow$ trigram similarity $\rightarrow$ `sec_subsidiaries`) and [`EntityResolver`](../../knowledge_graph/entity_resolver.py) (Union-Find equivalence classes guarded by `safe_union` CIK/ticker non-conflict constraints).
   - Dynamic Exhibit 21 subsidiary-to-parent CIK rollup (`load_dynamic_subsidiary_registry`), publisher/clickbait stoplists, `DICTIONARY_TICKERS`-guarded initialism matching, bounded Levenshtein ($\le 2$), Jaro-Winkler similarity:
     $$s_{\text{JW}}(s_1, s_2) = s_J(s_1, s_2) + \ell\, p\,\big(1 - s_J(s_1, s_2)\big) \ge 0.88, \qquad s_J = \frac{1}{3}\!\left(\frac{m}{|s_1|} + \frac{m}{|s_2|} + \frac{m - t}{m}\right)$$
     and lexicographic canonical representative scoring `(-is_sub, -is_descriptive, has_suffix, length_score)`.
4. **Asymmetric Dense Embeddings & SCD Type 2 Bitemporal Vector Indexing (`ADR-00020`–`00022`, `00033`, `00035`)**:
   - Hardware-aware serving geometry in [`embedding/embedder.py`](../../embedding/embedder.py): `qwen3-embedding:0.6b` (`EMBEDDING_DIM = 1024`), instruction-prefixed `embed_query` vs. raw `embed_documents`, CPU-pinned live inference (`num_gpu: 0`) preserving 8 GB GPU residency for `yarnball-qwen:7b` vs. `unload_ollama_model` GPU batching, finite/non-zero norm guard $\|v\|_2^2 \ge 10^{-8}$, and `embedding_schema_metadata` startup enforcement.
   - Slowly Changing Dimension (SCD) Type 2 `node_embeddings` (`PRIMARY KEY (node_id, valid_from)`, partial HNSW index on `valid_to IS NULL`) in [`knowledge_graph/embedding_store.py`](../../knowledge_graph/embedding_store.py) and dual HNSW + `tsvector` GIN indexing on `document_chunks` ([`knowledge_graph/chunk_store.py`](../../knowledge_graph/chunk_store.py)).
5. **Durable PostgreSQL Ledger, Ephemeral RAM Memgraph & PiT Leiden Communities (`ADR-00036`, `ADR-00037`)**:
   - Architectural separation in [`knowledge_graph/ledger.py`](../../knowledge_graph/ledger.py): durable PostgreSQL tables (`kg_nodes`, `kg_edges` keyed on `(source_id, target_id, rel_type, input_group)`) written *before* ephemeral in-memory Memgraph `MERGE`, prose-key stripping (`FORBIDDEN_MEMGRAPH_PROSE_KEYS`), deterministic boot hydration (`hydrate_memgraph_from_ledger`), and 90-day transient `:News` RAM eviction (`MEMGRAPH_EVENT_TTL_DAYS = 90`).
   - Point-in-Time structural community detection ([`knowledge_graph/communities.py`](../../knowledge_graph/communities.py)): filtering active structural supply-chain edges `[valid_from, valid_to)` (excluding transient `MENTIONS`), seeded `hierarchical_leiden` (`graspologic`) with connected-component-guaranteed `networkx` Louvain fallback (Traag et al., 2019 internal connectivity guarantee), and chunk-grounded `COMMUNITY_REPORT` persistence (`community_reports` + `communities.parquet`).
6. **Topic-Level UML Diagram & Theory-to-Code Crosswalk Table**:
   - Mermaid `classDiagram` modeling the Chapter-2 bounded concepts (`TBoxAnd5AxisOntology`, `ClauseAndStructuralGate`, `UnionFindEntityResolver`, `BitemporalVectorIndex`, `DurableKGLedger`, `EphemeralMemgraphProjection`, `PiTLeidenCommunityPartition`) and their invariants.
   - Full Crosswalk Table mapping every axiom, similarity formula, threshold, and ledger invariant to `module:symbol`, provenance (`Pre-ADR Foundation` vs. `ADR-00020`–`00025`, `00028`–`00029`, `00031`, `00033`, `00035`–`00037`), and unit/architecture test suites.

### 4.2 Part B — Software Architecture, Call Graphs & Entry Points (*Where & How*)

1. **Subsystem Module Inventory**: complete file-by-file breakdown across `ontology/` (6 modules), `embedding/` (1 module), `knowledge_graph/` (11 modules), [`ingestion/sec/parser.py`](../../ingestion/sec/parser.py) (`EdgarEntityLinker`), [`contagion/parameters.py`](../../contagion/parameters.py) (edge-level `share` / `buffer_days` / `param_provenance` write-time stamping), [`tools/init_db.py`](../../tools/init_db.py), [`tools/hydrate_memgraph.py`](../../tools/hydrate_memgraph.py), and [`tools/build_kg_snapshot.py`](../../tools/build_kg_snapshot.py).
2. **AST-Verified Execution Call Graphs (3 Mermaid `flowchart` pipelines)**:
   - *Pipeline 1 — Structured Extraction, Inline Ontology Gate & Durable-Before-Ephemeral Write*: `store_graph_entities` $\rightarrow$ `run_structured_extraction_with_telemetry` $\rightarrow$ `validate_graph_payload` (`DEFAULT_ONTOLOGY_GATE.gate_json_payload` $\rightarrow$ `validate_and_orient_triple` $\rightarrow$ `detect_and_break_ownership_cycles`) $\rightarrow$ `persist_kg_nodes` / `persist_kg_edges` $\rightarrow$ `store_node_embeddings` $\rightarrow$ Memgraph Bolt `MERGE` + `evict_expired_news_nodes` (with failures routed to `RuntimeRejectionQueue`).
   - *Pipeline 2 — Cross-Source Entity Resolution, PiT Community Partitioning & Offline Parquet Snapshot Build*: `build_and_export_kg_snapshot` $\rightarrow$ `EntityResolver.resolve_nodes` & `rewire_edges` $\rightarrow$ `SnapshotManager` (`nodes.parquet`, `edges.parquet`, `chunks.parquet`) $\rightarrow$ `build_communities` (`filter_structural_edges_pit` $\rightarrow$ `hierarchical_leiden` / connected Louvain) $\rightarrow$ `summarize_communities` $\rightarrow$ `persist_community_reports` & `export_communities_parquet`.
   - *Pipeline 3 — Container Boot Readiness & Ephemeral Memgraph Hydration*: `tools/hydrate_memgraph.py:main` $\rightarrow$ `hydrate_memgraph_from_ledger` (`load_ledger_nodes`, `load_ledger_edges`, `evict_expired_news_nodes`) $\rightarrow$ `check_memgraph_readiness` & `check_postgres_and_schema_metadata`.
3. **CLI / Consumer Entry Points & Storage Schema Reference**:
   - Entry points (`tools/init_db.py`, `tools/hydrate_memgraph.py`, `tools/build_kg_snapshot.py`, `knowledge_graph/memgraph_consumer.py`).
   - Complete column/index contracts for `kg_nodes`, `kg_edges`, `node_embeddings`, `node_embedding_sources`, `embedding_schema_metadata`, `document_chunks`, `chunk_entity_links`, `community_reports`, `graph_snapshots`, `sec_companies`, `sec_subsidiaries`, and the 4 offline snapshot Parquet files.

---

## 5. Invariants & Guardrails

1. **All `ADR-00040` Series Invariants Apply**: zero host-specific paths, pure GFM + `$...$`/`$$...$$` + `mermaid` in `02-ontology-linkage-and-ledger.md` (zero `mkdocstrings` `:::` directives in narrative prose), AST-verified call-graph edges, and honest `Pre-ADR Foundation` vs. `ADR-XXXXX` attribution.
2. **Chapter-2 Constant & Symbol Verification**: `tests/docs/test_docs_integrity.py` must verify all `module:symbol` references in Chapter 2's Crosswalk Table alongside live constants (`20` `EntityClass` values, `26` `RelationPredicate` values, `EMBEDDING_DIM = 1024`, `_MIN_NORM_SQ = 1e-8`, `DEFAULT_MEMGRAPH_EVENT_TTL_DAYS = 90`, Jaro-Winkler `0.88` threshold, `degree_burst_threshold = 25`, and `ADR-00031` `target_max_rate = 0.02`).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Unifies 18 modules and over a dozen post-`ADR-00001` ontology, embedding, and storage decisions alongside their pre-ADR mathematical foundations into a single cohesive *Blog Part 2* and engineering schema manual.
  - Creates a clean boundary for Chapter 3 (`ADR-00043`), which can treat the bitemporal ledger, Memgraph projection, `pgvector`/`tsvector` indexes, and `community_reports` as already-defined storage primitives and focus purely on `query_bi` retrieval, Cypher guardrails, cascade propagation algebra, and provenance-graded reporting.
- **Negative / Trade-offs**:
  - Adding/removing predicates in `ontology/schema.py`, altering `EMBEDDING_DIM` or `MEMGRAPH_EVENT_TTL_DAYS`, or modifying `kg_nodes` / `kg_edges` DDL will require synchronized updates to `02-ontology-linkage-and-ledger.md` to pass `test_docs_integrity.py`.
