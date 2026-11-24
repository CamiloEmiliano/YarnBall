# ADR-00036: Fail-Fast Storage, Ephemeral Memgraph Lifecycle & Deployable Topology

- **ADR ID**: ADR-00036
- **Filename**: `ADR-00036-fail-fast-infra-and-deployable-topology-.md`
- **Date**: 2026-11-22 08:48 UTC
- **Status**: PROPOSED
- **Target Subsystem**: `knowledge_graph` / `retrieval` / `docker` / `tests`
- **Linked Commits**: Pending acceptance

---

## 1. Context & Problem Statement

1. **Silent graph fallback**: [`knowledge_graph/memgraph_driver.py`](../../knowledge_graph/memgraph_driver.py) returns `_DummyDriver()` when the driver can't be created or connected (on `except Exception`). The dummy returns empty results.
2. **Silent relational fallback**: [`knowledge_graph/db.py`](../../knowledge_graph/db.py) returns an in-function `_MockConnection` when Postgres is unreachable (docstring: *"Returns _MockConnection for offline test environments if the DB is unreachable"*).
3. **Swallowed read errors**: `HybridRetriever.expand_subgraph` ([`retrieval/hybrid.py`](../../retrieval/hybrid.py)) catches `Exception` and logs at `DEBUG`, returning `[]`.
4. **The result is a dangerous false negative.** An outage reads downstream as "no supply-chain exposure", the worst possible error for a risk system feeding trading decisions. It also contradicts the fail-fast stance of [`ADR-00020`](ADR-00020-fail-fast-embedder-and-zero-vector-guard.md) (explicit `allow_mock` / `YARNBALL_EMBEDDER_MOCK`).
5. **Ephemeral Memgraph without a durable graph source of truth**: [`docker-compose.yml`](../../docker-compose.yml) intentionally mounts **no persistent volume** on `memgraph` because Memgraph is meant to operate purely in-memory. However, [`tools/init_db.py`](../../tools/init_db.py) contains no Postgres tables for extracted KG nodes or edges—meaning all LLM-extracted graph assertions currently live *only* in volatile Memgraph RAM until someone manually runs `SnapshotManager.export_snapshot`. A container restart wipes out hours of local `yarnball-qwen:7b` extraction compute. Meanwhile, writing unbounded `:News` nodes into Memgraph RAM mixes transient event impulses into permanent topology, risking RAM bloat and stale-event pollution.
6. **The retrieval container can't start.** [`retrieval/Dockerfile`](../../retrieval/Dockerfile) copies only `retrieval`, `knowledge_graph`, `embedding`, and `tools`, while `retrieval/` imports `ontology.schema`, `contagion`, `contagion.parameters`, `sft_pipeline.rejection_queue`, and `sft_pipeline.serving.model_client`.
7. **Dead compose service**: [`docker-compose.yml`](../../docker-compose.yml) `scraper` runs `python -m ingest.scraping_worker`, and no `ingest/` package exists. No compose service runs the streaming consumers (`streaming/consumer.py`, `knowledge_graph/memgraph_consumer.py`).
8. **Tests depend on implicit fallbacks**: [`tests/retrieval/test_retriever.py`](../../tests/retrieval/test_retriever.py) references the dummy/mock classes.

---

## 2. Decision Drivers & Constraints

1. **An outage must be distinguishable from an empty answer.**
2. **Pure in-memory Memgraph intent**: Memgraph stays stateless/volatile in RAM (no disk volumes required), while expensive LLM extraction results must never be lost on container restart.
3. **Slow topology vs. fast impulses**: structural supply-chain edges (10-K/10-Q/Exhibit 21/8-K Item 1.01–1.02) persist across years and fit easily in `< 1 GB` RAM at S&P 500 scale; transient news shocks decay over a finite horizon and must not accumulate unboundedly in RAM.
4. **Horizontal scaling without graph sharding**: Memgraph Community/MAGE does not horizontally shard a single graph, and cross-shard multi-hop traversal destroys latency. Scale is achieved by keeping the graph thin and scaling reads statelessly across workers.
5. **Offline development stays easy**: tests and the zero-DB Workbench ([`ADR-00015`](ADR-00015-dynamic-in-memory-offline-subgraph-store.md), [`ADR-00017`](ADR-00017-local-fastapi-kg-contagion-workbench-app.md)) must keep working, but only through explicit opt-in.
6. **Containers that build actually run.**

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Keep the fallbacks but log at `WARNING`.** Rejected. Logs aren't part of the data contract, and Domain 2 consumes reports, not logs.
- **Alternative B: Remove mocks entirely and require live databases for all tests.** Rejected. It makes CI heavy and breaks the offline Workbench guarantees.
- **Alternative C: Rely on `COPY . .` (like the root `Dockerfile`) or an un-packaged `PYTHONPATH` hack alone.** Rejected. `COPY . .` sweeps `data/`, `tests/`, and `docs/` into serving images and invalidates layer caching, while omitting `pyproject.toml` prevents clean dependency groups and external/editable installs; both selective `COPY` and `pyproject.toml` belong together.
- **Alternative D: Make Memgraph persistent on disk as the sole system of record.** Rejected. Contradicts the pure in-memory intent, ties durability to Memgraph's internal snapshot format, and still allows unbounded `:News` growth in RAM.
- **Alternative E: Horizontally shard Memgraph across cluster nodes.** Rejected. Unsupported in Memgraph Community/MAGE and degrades 3–5 hop supply-chain traversals; a thin structural graph fits comfortably on a single node, while read workloads scale statelessly.

---

## 4. Decision Outcome

1. **Explicit opt-in mocks**: `get_memgraph_driver()` and `get_connection()` raise `GraphStoreUnavailableError` / `DocumentStoreUnavailableError` (new, in `knowledge_graph/errors.py`) unless `YARNBALL_GRAPH_MOCK=1` / `YARNBALL_PG_MOCK=1` are set, mirroring `YARNBALL_EMBEDDER_MOCK`. `tests/conftest.py` sets these flags only for tests marked `offline`.
2. **No swallowed read errors**: read paths in `retrieval/` re-raise store errors. The facade ([`ADR-00034`](ADR-00034-single-bi-read-facade-and-pipeline-merge.md)) turns them into an explicit degraded outcome and never into an empty report.
3. **Durable Postgres KG assertion ledger (`kg_nodes`, `kg_edges`)**:
   - Create durable Postgres tables `kg_nodes` and `kg_edges` in [`tools/init_db.py`](../../tools/init_db.py) storing canonical entity metadata, typed structural relations, temporal fields (`valid_from`, `valid_to`, `known_at`, `known_at_imputed`, `last_affirmed_at`), cascade parameters (`share`, `buffer_days`, `input_group`, `param_provenance`), and `evidence_chunk_ids`.
   - Both sanctioned writers (`store_graph_entities` in [`knowledge_graph/graph_store.py`](../../knowledge_graph/graph_store.py) and `EdgarWorker._persist_sec_graph_data` in [`ingestion/sec/harvester.py`](../../ingestion/sec/harvester.py)) commit validated assertions to Postgres **immediately** before projecting into Memgraph.
4. **Ephemeral Thin Memgraph lifecycle & configurable event TTL**:
   - Memgraph stays **purely in-memory** (no volume in [`docker-compose.yml`](../../docker-compose.yml)).
   - A deterministic startup hydrator (`tools/hydrate_memgraph.py`, run on boot and optionally after batch `EntityResolver` merges) populates Memgraph from the durable Postgres ledger in seconds, guaranteeing live-vs-offline topology parity ([`ADR-00034`](ADR-00034-single-bi-read-facade-and-pipeline-merge.md)).
   - **Slow topology (full history in RAM)**: all structural supply-chain nodes and edges stay hydrated in Memgraph RAM (`< 1 GB` at S&P 500 scale) and are filtered per query by `PiTContext` ([`ADR-00033`](ADR-00033-bitemporal-pit-contract-for-read-paths--.md)).
   - **Fast event impulses (bounded RAM TTL)**: raw prose and chunks stay strictly in Postgres ([`ADR-00035`](ADR-00035-text-unit-chunk-layer-and-evidence-links.md)). Transient `:News` nodes projected into Memgraph are governed by a configurable environment variable **`MEMGRAPH_EVENT_TTL_DAYS`** (theoretic default **`90`**, covering one full quarterly earnings cycle before post-event supply-chain drift resolves). A periodic sweep evicts `:News` nodes older than `MEMGRAPH_EVENT_TTL_DAYS` from Memgraph RAM while leaving Postgres history intact. This default is designed to be empirically recalibrated later under [`ADR-00038`](ADR-00038-event-study-backtest-and-signal-decay---.md).
   - **Horizontal read scaling**: parallel backtests and multi-strategy reads scale horizontally across stateless read-only `InMemorySubgraphStore` worker processes (or read-only Memgraph instances hydrated from the same ledger), never by sharding Memgraph.
5. **Readiness**: `/health` stays a liveness check. New `/ready` checks Memgraph (including hydration completeness), Postgres, `embedding_schema_metadata` alignment ([`ADR-00022`](ADR-00022-unified-sec-news-dense-vector-coverage--.md)), `verify_embedder_ready`, and Ollama model presence (`yarnball-qwen:7b` and `qwen3-embedding:0.6b` per [`ADR-00035`](ADR-00035-text-unit-chunk-layer-and-evidence-links.md)), returning `503` with a reason list on failure.
6. **Complementary Python & Docker packaging**: author a root `pyproject.toml` defining optional dependency extras (`[retrieval]`, `[ingestion]`, `[dev]`) for clean local/external installs without `PYTHONPATH` hacks, **and** use two-stage layer caching in [`retrieval/Dockerfile`](../../retrieval/Dockerfile): copy `pyproject.toml` first to cache third-party dependencies, then selectively `COPY` only runtime source packages (`retrieval`, `knowledge_graph`, `embedding`, `ontology`, `contagion`, `synthesis`, `curation`, `sft_pipeline`, `universe`, `tools`) followed by `pip install --no-deps .`. Moving default embeddings onto Ollama `/api/embed` ([`ADR-00035`](ADR-00035-text-unit-chunk-layer-and-evidence-links.md)) also lets `retrieval/Dockerfile` drop heavy in-container PyTorch/`sentence-transformers` dependencies. A CI smoke step builds the image and runs `python -c "import retrieval.service, retrieval.pipeline, synthesis.synthesizer"` inside it.
7. **Compose topology**: delete the dead `scraper` service (`ingest.scraping_worker`) outright from [`docker-compose.yml`](../../docker-compose.yml) (live web scraping is excluded per `.agents/rules/data_sources.md`, and `finnhub_scheduler` + `ingestion/task_runner.py` already cover sanctioned ingestion). Add `memgraph_hydrator` (one-shot init after `postgres` + `memgraph`), `stream_consumer` (`python -m streaming.consumer`), and `memgraph_consumer` services with `depends_on: condition: service_healthy`.

---

## 5. Invariants & Guardrails

1. **No implicit dummies**: when no mock environment flags are set, an unreachable store raises. It never returns an object that answers queries.
2. **No empty-on-error reads**: no `except Exception` in `retrieval/` returns an empty collection without re-raising or marking the result degraded. Enforced by an AST test.
3. **Durable-before-ephemeral write order**: every validated structural KG edge is persisted in Postgres `kg_edges` prior to Memgraph projection; wiping and restarting Memgraph followed by `tools/hydrate_memgraph.py` reproduces 100% of structural topology deterministically.
4. **Thin graph & bounded event RAM**: raw filing/chunk prose never enters Memgraph; transient `:News` nodes older than `MEMGRAPH_EVENT_TTL_DAYS` are evicted from RAM without deleting durable Postgres rows.
5. **Container import smoke**: every Dockerfile image must import its entrypoint module in CI.
6. **Compose references exist**: every `command` module in `docker-compose.yml` resolves to an importable module (static test).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Outages surface as errors or degraded flags instead of false "no exposure" answers.
  - Memgraph stays genuinely ephemeral and lightweight in RAM while zero LLM extraction work is ever lost on restart.
  - Nightly or post-resolution Memgraph re-hydration eliminates live-vs-snapshot entity drift ([`ADR-00034`](ADR-00034-single-bi-read-facade-and-pipeline-merge.md)).
  - The retrieval service image actually starts, and live ingestion becomes deployable.
- **Negative / Trade-offs**:
  - Developers must set mock flags (or use the `offline` marker) to run without databases.
  - Adds a lightweight Postgres write to `kg_nodes` / `kg_edges` in the two sanctioned writers and a boot hydration step on Memgraph container startup.
  - Some existing tests that rely on implicit fallback need fixture updates.

---

## 7. Open Questions for Review

1. ~~**Packaging**~~ **Resolved (2026-11-23)**: keep both as complementary layers—add `pyproject.toml` (with optional extras and `pip install --no-deps .`) **and** retain selective cached `COPY` of runtime packages in `retrieval/Dockerfile` (see Section 4.6).
2. ~~**Scraper service**~~ **Resolved (2026-11-23)**: delete `scraper` (`ingest.scraping_worker`) outright from `docker-compose.yml` per `.agents/rules/data_sources.md` (see Section 4.7).
