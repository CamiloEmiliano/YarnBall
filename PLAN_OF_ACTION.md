# End-to-End Execution Plan: Finnhub to Memgraph Knowledge Graph

## 1. Executive Summary & Objective

The objective of this plan is to resolve all blockers and bridge the architectural gaps identified during the codebase audit, enabling a seamless end-to-end run:
1. **Ingestion:** Fetch live financial news from the Finnhub API.
2. **Buffering & Queueing:** Publish raw events to Kafka (`financial_news_raw`).
3. **Scraping & Cleaning:** Consume from Kafka, check the self-learning blacklist (`domain_status`), scrape full body text with Trafilatura, and insert into PostgreSQL (`financial_news_queue`).
4. **Knowledge Graph Extraction:** Pass non-paywalled extracted full text to local Ollama (`qwen3:8b`) to extract financial entities and semantic relationships.
5. **Graph & Vector Persistence:**
   - Persist entities (`Company`, `Person`, `Product`, etc.) and relations (`INVESTS_IN`, `COMPETES_WITH`, `ACQUIRED`, etc.) into **Memgraph** (`bolt://memgraph:7687`).
   - Generate dense vector embeddings with `SentenceTransformer` and upsert into PostgreSQL `node_embeddings` (pgvector).
6. **Visualization:** Open **Memgraph Lab** at `http://localhost:3000` to interactively view the resulting graph.

---

## 2. Issues to Address (Audit Findings)

| Priority | Component | Issue Description | Target File(s) |
| :--- | :--- | :--- | :--- |
| **P0 - Blocker** | PostgreSQL DDL | `node_embeddings` table does not exist in `financial_rag` database | `tools/init_db.py` |
| **P0 - Blocker** | Ollama LLM Client | Hardcoded `http://localhost:11434/api/generate` in container fails | `graph/graph_store.py` |
| **P0 - Blocker** | Pipeline Wiring | Scraper does not invoke entity extraction or embedding generation | `ingest/scraping_worker.py`, `graph/graph_store.py` |
| **P1 - Ingestion** | Date Filtering | Hardcoded 30-min window (`HIST_START`/`HIST_END`) filters out all current news | `.env`, `ingest/finnhub_client.py` |
| **P1 - Reliability**| Database Driver | `_build_dsn()` mutates URL into invalid format; `_MockConnection` hides errors | `graph/db.py` |

---

## 3. Step-by-Step Implementation Roadmap

### Phase 1: Database Schema & Infrastructure Preparation

- [x] **1.1. Create `node_embeddings` table in `tools/init_db.py`:**
  - Added DDL to ensure `node_embeddings` (node_id, entity_type, embedding vector(768), source_hash) is created during initialization.
  - Executed DDL on live `graphrag_pg` container (`financial_rag` database).
- [x] **1.2. Verify `domain_status` and `financial_news_queue` constraints:**
  - Verified pgvector extension, unique index on `source_hash`, and tables are present and healthy.

### Phase 2: Ollama & LLM Connectivity Fix

- [x] **2.1. Dynamic Ollama Endpoint in `graph/graph_store.py`:**
  - Removed hardcoded `http://localhost:11434/api/generate`.
  - Configured dynamic detection: uses `OLLAMA_API_BASE` or `http://host.docker.internal:11434/api/generate` inside Docker.
  - Added `QWEN_API_BASE` and `QWEN_MODEL_NAME=qwen3:8b` to `.env`.
- [x] **2.2. Verify LLM extraction response handling:**
  - Verified JSON extraction and timeout extension (120s) for Qwen 8B.

### Phase 3: Pipeline Integration (Scraper -> Graph -> Embeddings)

- [x] **3.1. Link Scraper Worker to Graph & Embedding Storage:**
  - Wired `process_scraping_message` in `ingest/scraping_worker.py` to trigger `store_graph_entities()` upon successful article scraping.
  - Wired `store_graph_entities()` to trigger `store_node_embeddings()` into PostgreSQL pgvector (`node_embeddings`).
  - Refactored `graph/embedding_store.py` to support `psycopg 3` natively without `psycopg2.extras`.
  - Verified live: Ollama extraction, Memgraph graph generation, and PostgreSQL pgvector insertion executed and validated.

### Phase 4: Ingestion Calibration & Configuration Cleanup

- [x] **4.1. Respect 30-minute Test Window in `.env` & `ingest/finnhub_client.py`:**
  - Preserved the bounded 30-minute historical window (`HIST_START` / `HIST_END`) in `.env` specifically for testing to prevent excessive fetching.
- [x] **4.2. Refactor `graph/db.py` connection management:**
  - Fixed DSN construction using `urllib.parse` to avoid duplicate database paths (`financial_rag/financial_rag`).
  - Added real `psycopg 3` connectivity fallback with proper logging.
  - Synced to `graphrag_scraper` and `graphrag_retriever` containers.

### Phase 5: Container Synchronization & Live Verification

- [x] **5.1. Synchronize Code to Containers:**
  - Copied all updated modules (`init_db.py`, `graph_store.py`, `embedding_store.py`, `db.py`, `scraping_worker.py`, `.env`, and `perf_test_scraper.py`) into running containers.
- [x] **5.2. Run End-to-End Test (`tests/perf_test_scraper.py`):**
  - Fetched Finnhub news items within the 30-minute window (`HIST_START` / `HIST_END`).
  - Successfully extracted articles using Trafilatura with explicit URL logging.
  - Successfully ran Qwen 8B entity and relationship extraction.
  - Successfully populated 141 nodes and 271 relationship edges in Memgraph.
  - Successfully populated 136 vector embeddings in PostgreSQL pgvector (`node_embeddings`).

---

## 4. Verification & Visualization Guide

### 4.1. Automated Verification Checks

1. **PostgreSQL Verification:**
   ```bash
   docker exec graphrag_pg psql -U postgres -d financial_rag -c "
   SELECT count(*) AS queued_articles FROM financial_news_queue;
   SELECT count(*) AS embedded_nodes FROM node_embeddings;
   SELECT * FROM domain_status;
   "
   ```

2. **Memgraph Verification (Bolt):**
   ```bash
   docker exec memgraph sh -c "echo 'MATCH (n) RETURN count(n) AS node_count;' | mgconsole --username memgraph --password memgraph"
   docker exec memgraph sh -c "echo 'MATCH ()-[r]->() RETURN count(r) AS rel_count;' | mgconsole --username memgraph --password memgraph"
   ```

### 4.2. Interactive Visualization (Memgraph Lab)

1. Open your browser and navigate to:
   ```
   http://localhost:3000
   ```
2. Connect using the credentials:
   - **Host:** `memgraph` (or `localhost` if connecting from external client)
   - **Port:** `7687`
   - **Username:** `memgraph`
   - **Password:** `memgraph`
3. Run the following Cypher query in the Query Execution tab to view the visual graph:
   ```cypher
   MATCH (n)-[r]->(m) 
   RETURN n, r, m 
   LIMIT 100;
   ```
4. Click on nodes (e.g., Apple, Tim Cook, AI Sector) to inspect their extracted properties and connecting relationships.
