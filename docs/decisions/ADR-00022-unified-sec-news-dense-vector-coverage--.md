# ADR-00022: Unified SEC & News Dense Vector Coverage & Configurable Dimensions

- **ADR ID**: ADR-00022
- **Filename**: `ADR-00022-unified-sec-news-dense-vector-coverage--.md`
- **Date**: 2026-11-14 19:25 UTC
- **Status**: PROPOSED
- **Target Subsystem**: `retrieval` / `ingestion` / `knowledge_graph` / `tools`
- **Linked Commits**: Pending acceptance

---

## 1. Context & Problem Statement

Audit of the current dense retrieval and ingestion paths uncovered an asymmetric split between what is embedded and what is actually queried:

1. **SEC Entities Are Invisible to Dense Vector Search**: [`knowledge_graph/graph_store.py`](../../knowledge_graph/graph_store.py) calls `store_node_embeddings(...)` during news extraction, whereas `EdgarWorker` in [`ingestion/sec/harvester.py`](../../ingestion/sec/harvester.py) extracts statutory 10-K/10-Q/8-K/Exhibit 21 entities and relationships into Memgraph without calling `store_node_embeddings`.
2. **Article Vectors Are Written Never Queried**: [`streaming/consumer.py`](../../streaming/consumer.py) embeds article text and stores it in `financial_news_queue.embedding` via [`tools/utils.py`](../../tools/utils.py) (`store_embedding`), **but** neither [`retrieval/hybrid.py`](../../retrieval/hybrid.py) (`HybridRetriever`) nor [`retrieval/service.py`](../../retrieval/service.py) (`/search`) ever runs vector similarity against `financial_news_queue.embedding`—both query only `node_embeddings`.
3. **Hardcoded `vector(768)` Schema**: [`tools/init_db.py`](../../tools/init_db.py) hardcodes `vector(768)` on both `financial_news_queue` and `node_embeddings`, preventing dimensionally distinct financial/domain embedders (`384`, `1024`, `1536`) without schema drift crashes.

---

## 2. Decision Drivers & Constraints

1. **Retrieval Parity Across SEC & News**: Statutory 10-K/8-K entities (`EdgarWorker`) and unstructured news entities (`memgraph_consumer` / `graph_store`) must both be discoverable via dense entity search.
2. **Hybrid Entity + Passage Retrieval**: Dense search must leverage both `node_embeddings` (for graph anchor selection) and `financial_news_queue.embedding` / SEC passage vectors (for textual evidence lookup), bounded by Point-in-Time timestamps (`published_at <= as_of_date`).
3. **Model-Schema Dimension Alignment**: Database DDL dimension and active `EMBEDDER_MODEL` output dimension must match explicitly via configuration and startup validation.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Keep `node_embeddings` News-Only**:
  - *Rationale for Rejection*: Causes `HybridRetriever` to miss core SEC supply-chain companies and subsidiaries that appear in statutory filings before or without recent Finnhub news.
- **Alternative B: Drop `financial_news_queue.embedding` Entirely**:
  - *Rationale for Rejection*: Entity names alone cannot answer thematic queries (e.g., `"EUV lithography export restriction"` or `"optical transceiver fabrication halt"`) where narrative passage embeddings provide crucial recall.

---

## 4. Decision Outcome

Establish **Unified SEC + News Vector Coverage & Parameterized Schema Dimensions** across [`ingestion/sec/harvester.py`](../../ingestion/sec/harvester.py), [`retrieval/hybrid.py`](../../retrieval/hybrid.py), [`retrieval/service.py`](../../retrieval/service.py), and [`tools/init_db.py`](../../tools/init_db.py):

1. **SEC Entity Embedding Emission**:
   - Have `EdgarWorker` invoke contextual `store_node_embeddings(...)` (per [`ADR-00021`](ADR-00021-contextual-node-embeddings-and-pgvector-.md)) for merged canonical company and subsidiary nodes, tagging `source_hash` / `accession_number` provenance.
2. **Dual-Index Dense Retrieval in `HybridRetriever`**:
   - Retain `search_dense_entities` over `node_embeddings` AND add `search_dense_passages` over `financial_news_queue.embedding` (with optional `published_at <= as_of_date` PiT filter and `WHERE embedding IS NOT NULL`), linking matched articles back to Memgraph `:News {id: source_hash}` and entity neighborhoods.
3. **Configurable Vector Dimension (`EMBEDDING_DIM`)**:
   - Parameterize `tools/init_db.py` table creation via `EMBEDDING_DIM` (default `768`) and record the active model name + dimension so mismatches between `EMBEDDER_MODEL` and PostgreSQL columns fail loudly during initialization.

---

## 5. Invariants & Guardrails

1. **SEC + News Parity**: Both SEC and news Memgraph ingestion pipelines MUST emit `node_embeddings` entries for persisted entities.
2. **PiT Safe Passage Search**: Any dense query against `financial_news_queue.embedding` used in historical evaluation MUST enforce `published_at <= as_of_date` to avoid lookahead leakage.
3. **Dimension Match Check**: Worker startup MUST assert `len(embed_texts(["probe"])[0]) == EMBEDDING_DIM`.
4. **Zero Absolute Paths Invariant**: All references remain strictly repository-relative (`../../<path>`).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Eliminates dead-write waste on `financial_news_queue.embedding` and closes the SEC coverage blind spot in `HybridRetriever`.
  - Makes future embedding model upgrades explicit and schema-safe.
- **Negative / Trade-offs**:
  - Adds embedding compute overhead during SEC bulk ingestion (`EdgarWorker`) and requires vector index maintenance (`HNSW` / `IVFFlat`) as corpus size grows.
