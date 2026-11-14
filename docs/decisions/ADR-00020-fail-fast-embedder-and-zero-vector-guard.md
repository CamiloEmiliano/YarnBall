# ADR-00020: Fail-Fast Embedder Initialization & Zero-Vector Prohibition

- **ADR ID**: ADR-00020
- **Filename**: `ADR-00020-fail-fast-embedder-and-zero-vector-guard.md`
- **Date**: 2026-11-14 19:25 UTC
- **Status**: PROPOSED
- **Target Subsystem**: `embedding` / `retrieval` / `streaming`
- **Linked Commits**: Pending acceptance

---

## 1. Context & Problem Statement

During end-to-end readiness probing of the current architecture, [`embedding/embedder.py`](../../embedding/embedder.py) silently caught the absence of `sentence_transformers` (`No module named 'sentence_transformers'`) inside `get_embedder_model()` and substituted `_FallbackModel`, returning `768`-dimensional all-zero vectors (`[0.0] * 768`) for every input text (`embed_texts(...)`).

This silent degradation creates severe downstream corruption risks:
1. [`streaming/consumer.py`](../../streaming/consumer.py) (`process_streaming_event`) and [`knowledge_graph/embedding_store.py`](../../knowledge_graph/embedding_store.py) (`store_node_embeddings`) persist all-zero vectors into `financial_news_queue.embedding` and `node_embeddings` as if they were valid dense representations.
2. [`retrieval/hybrid.py`](../../retrieval/hybrid.py) (`HybridRetriever.search_dense_entities`) and [`retrieval/service.py`](../../retrieval/service.py) (`/search`) execute pgvector cosine distance (`embedding <=> %s::vector`) against zero vectors, collapsing ranking quality without raising an alert or HTTP failure.

---

## 2. Decision Drivers & Constraints

1. **No Silent Semantic Corruption**: A missing ML dependency, missing local weights, or misconfigured `EMBEDDER_MODEL` must never silently write degenerate zero vectors into production PostgreSQL `pgvector` tables.
2. **Deterministic Offline Unit Tests**: Unit tests that intentionally avoid loading HuggingFace models must still run cleanly via explicit, opt-in mock injection rather than implicit production fallback.
3. **Zero-Norm Cosine Invariant**: Cosine similarity over normalized vectors ($||v||_2 = 1$) is undefined for zero-norm vectors ($||v||_2 = 0$); degenerate vectors must be rejected before persistence or query execution.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Keep Silent `_FallbackModel` Zero Vectors With Warning Logs**:
  - *Rationale for Rejection*: Warnings inside worker containers (`consumer`, `memgraph_consumer`, `retriever`) are easily missed during batch ingestion, leaving `node_embeddings` polluted with unrecoverable `[0.0] * 768` rows indistinguishable by SQL schema from real embeddings.
- **Alternative B: Random Deterministic Hash Vectors on Fallback**:
  - *Rationale for Rejection*: Masks environment misconfiguration (`sentence-transformers` absent in `.venv` vs. container) and injects pseudo-random vectors that have no semantic correlation with real model checkpoints in the same vector column.

---

## 4. Decision Outcome

Adopt **Fail-Fast Embedder Initialization & Zero-Vector Gating** in [`embedding/embedder.py`](../../embedding/embedder.py), [`knowledge_graph/embedding_store.py`](../../knowledge_graph/embedding_store.py), [`retrieval/hybrid.py`](../../retrieval/hybrid.py), and [`retrieval/service.py`](../../retrieval/service.py):

1. **Fail-Fast Live Default (`allow_mock=False`)**:
   - Remove implicit silent fallback to `_FallbackModel` in `get_embedder_model()`. If `sentence_transformers` cannot be imported or `SentenceTransformer(model_name)` fails to load, raise a clear `EmbedderUnavailableError` (subclass of `RuntimeError`) naming `EMBEDDER_MODEL`.
   - Support explicit test/dry-run mocking only when explicitly opted in (`allow_mock=True` or `YARNBALL_EMBEDDER_MOCK=1` for local mock testing), tagging telemetry accordingly.
2. **Startup / Catalog Readiness Probe (`verify_embedder_ready`)**:
   - Expose a lightweight readiness check used by worker startup (`streaming/consumer.py`, `retrieval/service.py`) and optionally surfaced in workbench readiness telemetry ([`explorer/workbench_api.py`](../../explorer/workbench_api.py)).
3. **Zero-Norm Vector Guard (`validate_embedding_vectors`)**:
   - Enforce finite values and non-zero L2 norm ($\sum v_i^2 > 10^{-8}$) before executing `INSERT INTO node_embeddings`, `UPDATE financial_news_queue`, or `ORDER BY embedding <=> %s::vector` queries.

---

## 5. Invariants & Guardrails

1. **Zero-Vector Prohibition**: Production and non-mock pipelines MUST NEVER persist or query all-zero or NaN/Inf embedding vectors in `node_embeddings` or `financial_news_queue`.
2. **Explicit Mock Opt-In**: Any mock embedder path MUST be explicitly requested (`allow_mock=True` or test fixture patching), never silently triggered by an `ImportError`.
3. **Zero Absolute Paths Invariant**: All references remain strictly repository-relative (`../../<path>`).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Eliminates silent `pgvector` pollution when `sentence-transformers` or model weights are missing in an environment.
  - Makes container and local `.venv` dependency drift immediately visible at startup.
- **Negative / Trade-offs**:
  - Environments running without `sentence-transformers` (such as lightweight local `.venv` setups) will fail immediately on live embedding calls unless `--mock` / test doubles are explicitly selected.
