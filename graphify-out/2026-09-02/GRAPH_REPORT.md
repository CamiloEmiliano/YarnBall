# Graph Report - graphrag_finance  (2026-09-02)

## Corpus Check
- 38 files · ~12,972 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 369 nodes · 589 edges · 19 communities (17 shown, 2 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 14 edges (avg confidence: 0.69)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- kafka_consumer.py
- store_graph_entities
- finnhub_client.py
- InitDbTests
- psycopg2/__init__.py
- TestKafkaProducer
- utils.py
- RuntimeError
- pg_connection
- scraping_worker.py
- Financial‑RAG Ingestion Module
- FinnhubClientContractTests
- task_runner.py
- 3. Step-by-Step Implementation Roadmap

## God Nodes (most connected - your core abstractions)
1. `process_scraping_message()` - 18 edges
2. `_connect_db()` - 12 edges
3. `pg_connection()` - 11 edges
4. `store_graph_entities()` - 11 edges
5. `send_to_dlt()` - 11 edges
6. `ScrapingWorkerTests` - 11 edges
7. `insert_raw_message()` - 11 edges
8. `store_node_embeddings()` - 10 edges
9. `publish_message()` - 10 edges
10. `process_graph_message()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `run_consumer()` --calls--> `embed_texts()`  [EXTRACTED]
  kafka_pipeline/kafka_consumer.py → embedding/embedder.py
- `pg_connection()` --calls--> `connect()`  [EXTRACTED]
  graph/db.py → tools/psycopg2/__init__.py
- `pg_connection()` --calls--> `_MockConnection`  [EXTRACTED]
  graph/db.py → tools/psycopg2/__init__.py
- `insert_test_embeddings()` --calls--> `pg_connection()`  [EXTRACTED]
  tests/test_retriever.py → graph/db.py
- `store_node_embeddings()` --calls--> `_connect_db()`  [EXTRACTED]
  graph/embedding_store.py → tools/init_db.py

## Import Cycles
- None detected.

## Communities (19 total, 2 thin omitted)

### Community 0 - "kafka_consumer.py"
Cohesion: 0.10
Nodes (31): _call(), ConsumerError, _deserialize_message(), ingest_message(), _is_valid_payload(), process_message(), Any, datetime (+23 more)

### Community 1 - "store_graph_entities"
Cohesion: 0.06
Nodes (30): _clean_json_response(), _default_ollama_url(), extract_entities(), _extract_entities_via_ollama(), Extract entities and relationships from text using local LLM., Extract entities and relationships and persist them to Memgraph., Detect whether running inside Docker container or host., Extract and parse JSON from LLM output, handling markdown fences or reasoning… (+22 more)

### Community 2 - "finnhub_client.py"
Cohesion: 0.08
Nodes (35): check_api(), _date_range(), fetch_finnhub(), _fetch_json_if_available(), _finnhub_params(), _get_json(), httpx, _is_api_available() (+27 more)

### Community 4 - "psycopg2/__init__.py"
Cohesion: 0.07
Nodes (12): Exception, str, connect(), _MockConnection, _MockCursor, OperationalError, Extended mock psycopg2 package to satisfy test imports. Provides: - ``connect``…, Return a mock connection mimicking ``psycopg2.connect``. Additional arguments… (+4 more)

### Community 5 - "TestKafkaProducer"
Cohesion: 0.14
Nodes (8): KafkaProducerFallback, patch, If producer.send().get() raises, the function logs failure and returns False., _producer should instantiate KafkaProducer with the bootstrap server from the…, When the kafka library cannot be imported, the fallback class is used.…, When producer is None, send_to_dlt logs a skip event and returns False., A healthy producer should send the event, log success and return True., TestKafkaProducer

### Community 6 - "utils.py"
Cohesion: 0.13
Nodes (12): TestCase, InsertRawMessageTests, insert_raw_message(), _kafka_producer(), _payload_hash(), datetime, KafkaProducer, Publish a raw ingestion event to Kafka for downstream database persistence. (+4 more)

### Community 7 - "RuntimeError"
Cohesion: 0.07
Nodes (9): KafkaConsumerFallback, _Client, _Response, DLTQueueError, KafkaConsumerFallback, Raised when a message cannot be routed to the dead‑letter topic., RuntimeError, _PsycopgStub (+1 more)

### Community 8 - "pg_connection"
Cohesion: 0.12
Nodes (19): BaseModel, _DummyModel, embed_texts(), Return a list of embedding vectors for the supplied texts. The vectors are…, _build_dsn(), pg_connection(), Return a psycopg2 connection. If *database* is provided, the DSN is adjusted to…, Construct DSN for PostgreSQL. Uses POSTGRES_URL if set; otherwise builds from… (+11 more)

### Community 9 - "scraping_worker.py"
Cohesion: 0.07
Nodes (38): compute_source_hash(), extract_text_from_html(), fetch_and_extract(), get_domain_from_url(), insert_extracted_article(), is_domain_blacklisted(), is_paywall_or_stub(), process_scraping_message() (+30 more)

### Community 10 - "Financial‑RAG Ingestion Module"
Cohesion: 0.20
Nodes (9): API Contract Notes, Database setup (`python init_db.py`), Environment Variables, Financial‑RAG Ingestion Module, Kafka consumer (`python kafka_consumer.py`), Program Flow, Quick start, Requirements (+1 more)

### Community 12 - "task_runner.py"
Cohesion: 0.16
Nodes (11): Run a synchronous fetch function in a thread executor., Execute all fetch functions registered via the @ingest_task decorator. Each…, run_all(), _run_sync(), Execute all registered ingestion tasks concurrently., run_all(), BackgroundScheduler, CronTrigger (+3 more)

### Community 17 - "3. Step-by-Step Implementation Roadmap"
Cohesion: 0.15
Nodes (12): 1. Executive Summary & Objective, 2. Issues to Address (Audit Findings), 3. Step-by-Step Implementation Roadmap, 4.1. Automated Verification Checks, 4.2. Interactive Visualization (Memgraph Lab), 4. Verification & Visualization Guide, End-to-End Execution Plan: Finnhub to Memgraph Knowledge Graph, Phase 1: Database Schema & Infrastructure Preparation (+4 more)

## Knowledge Gaps
- **18 isolated node(s):** `httpx`, `_SQLStub`, `1. Executive Summary & Objective`, `2. Issues to Address (Audit Findings)`, `Phase 1: Database Schema & Infrastructure Preparation` (+13 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `store_graph_entities()` connect `store_graph_entities` to `pg_connection`, `scraping_worker.py`, `kafka_consumer.py`?**
  _High betweenness centrality (0.127) - this node is a cross-community bridge._
- **Why does `process_scraping_message()` connect `scraping_worker.py` to `store_graph_entities`, `finnhub_client.py`?**
  _High betweenness centrality (0.078) - this node is a cross-community bridge._
- **Why does `store_node_embeddings()` connect `pg_connection` to `scraping_worker.py`, `store_graph_entities`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **What connects `httpx`, `_SQLStub`, `1. Executive Summary & Objective` to the rest of the system?**
  _18 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `kafka_consumer.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09634146341463415 - nodes in this community are weakly interconnected._
- **Should `store_graph_entities` be split into smaller, more focused modules?**
  _Cohesion score 0.05952380952380952 - nodes in this community are weakly interconnected._
- **Should `finnhub_client.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07777777777777778 - nodes in this community are weakly interconnected._