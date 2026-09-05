# Graph Report - graphrag_finance  (2026-08-27)

## Corpus Check
- 37 files · ~9,738 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 322 nodes · 504 edges · 20 communities (17 shown, 3 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 15 edges (avg confidence: 0.68)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- kafka_producer.py
- graph_store.py
- finnhub_client.py
- api_ingest.py
- psycopg2/__init__.py
- utils.py
- kafka_consumer.py
- RuntimeError
- pg_connection
- InitDbTests
- Financial‑RAG Ingestion Module
- run_all
- dispatcher.py
- _Client
- ApiIngestContractTests

## God Nodes (most connected - your core abstractions)
1. `publish_message()` - 12 edges
2. `pg_connection()` - 11 edges
3. `send_to_dlt()` - 11 edges
4. `insert_raw_message()` - 11 edges
5. `fetch_finnhub()` - 9 edges
6. `run_consumer()` - 9 edges
7. `TestKafkaProducer` - 9 edges
8. `_MockConnection` - 9 edges
9. `embed_texts()` - 8 edges
10. `store_node_embeddings()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `store_node_embeddings()` --calls--> `execute_values()`  [INFERRED]
  graph/embedding_store.py → tools/psycopg2/__init__.py
- `run_consumer()` --calls--> `embed_texts()`  [EXTRACTED]
  kafka/kafka_consumer.py → embedding/embedder.py
- `pg_connection()` --calls--> `connect()`  [EXTRACTED]
  graph/db.py → tools/psycopg2/__init__.py
- `pg_connection()` --calls--> `_MockConnection`  [EXTRACTED]
  graph/db.py → tools/psycopg2/__init__.py
- `send_record()` --calls--> `publish_message()`  [EXTRACTED]
  ingest/kafka_driver.py → kafka/kafka_producer.py

## Import Cycles
- None detected.

## Communities (20 total, 3 thin omitted)

### Community 0 - "kafka_producer.py"
Cohesion: 0.08
Nodes (28): _make_envelope(), Any, Build the internal message format that kafka_consumer._is_valid_payload…, Called by the ingestion modules. Returns True on successful publish, False…, send_record(), _ensure_topics(), get_producer(), _init_topic() (+20 more)

### Community 1 - "graph_store.py"
Cohesion: 0.10
Nodes (14): _DummyChat, _DummyChoice, _DummyMessage, _extract_entities_with_qwen(), OpenAI, Extract entities using a locally‑hosted Qwen model. The model is loaded once…, _DummyDriver, _DummySession (+6 more)

### Community 2 - "finnhub_client.py"
Cohesion: 0.13
Nodes (19): check_api(), _date_range(), fetch_finnhub(), _fetch_json_if_available(), _finnhub_params(), _get_json(), httpx, _is_api_available() (+11 more)

### Community 3 - "api_ingest.py"
Cohesion: 0.15
Nodes (17): _date_range(), fetch_free_historical_news(), _fetch_json_if_available(), _finnhub_params(), _get_json(), httpx, _is_api_available(), _process_finnhub() (+9 more)

### Community 4 - "psycopg2/__init__.py"
Cohesion: 0.08
Nodes (10): Exception, str, execute_values(), _MockConnection, _MockCursor, OperationalError, Extended mock psycopg2 package to satisfy test imports. Provides: - ``connect``…, Placeholder for ``psycopg2.OperationalError``. (+2 more)

### Community 5 - "utils.py"
Cohesion: 0.10
Nodes (21): create_queue_table(), ensure_database(), get_dsn(), Database initialisation for the Financial - RAG ingestion pipeline. - Creates…, Create `financial_news_queue` with the required columns., Return a DSN that connects to the given database. Constructs the DSN from…, Create the target DB if it does not already exist., _SQLStub (+13 more)

### Community 6 - "kafka_consumer.py"
Cohesion: 0.10
Nodes (23): Extract entities using Qwen and persist them to Memgraph., store_graph_entities(), _deserialize_message(), Continuously consume messages from the main Kafka topic and persist extracted…, run_memgraph_consumer(), _call(), ConsumerError, _deserialize_message() (+15 more)

### Community 7 - "RuntimeError"
Cohesion: 0.09
Nodes (8): _Client, _Response, DLTQueueError, KafkaConsumerFallback, Raised when a message cannot be routed to the dead‑letter topic., RuntimeError, _Psycopg2Stub, KafkaProducerFallback

### Community 8 - "pg_connection"
Cohesion: 0.10
Nodes (22): BaseModel, _DummyModel, embed_texts(), Return a list of embedding vectors for the supplied texts. The vectors are…, _build_dsn(), pg_connection(), Return a psycopg2 connection. If *database* is provided, the DSN is adjusted to…, Construct DSN for PostgreSQL. Uses POSTGRES_URL if set; otherwise builds from… (+14 more)

### Community 10 - "Financial‑RAG Ingestion Module"
Cohesion: 0.20
Nodes (9): API Contract Notes, Database setup (`python init_db.py`), Environment Variables, Financial‑RAG Ingestion Module, Kafka consumer (`python kafka_consumer.py`), Program Flow, Quick start, Requirements (+1 more)

### Community 11 - "run_all"
Cohesion: 0.50
Nodes (4): Utility to run a coroutine and log unexpected errors., Discover every module in `ingest/` that defines a `fetch_*` function and…, run_all(), _run_coroutine()

### Community 12 - "dispatcher.py"
Cohesion: 0.17
Nodes (10): Run a synchronous fetch function in a thread executor., Execute all fetch functions registered via the @ingest_task decorator. Each…, run_all(), _run_sync(), Ingestion package - entry point for the collection of micro-scripts., BackgroundScheduler, CronTrigger, main() (+2 more)

## Knowledge Gaps
- **10 isolated node(s):** `httpx`, `httpx`, `_SQLStub`, `Quick start`, `Requirements` (+5 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `embed_texts()` connect `pg_connection` to `kafka_consumer.py`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **Why does `publish_message()` connect `kafka_producer.py` to `api_ingest.py`?**
  _High betweenness centrality (0.061) - this node is a cross-community bridge._
- **Why does `send_to_dlt()` connect `kafka_producer.py` to `kafka_consumer.py`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **What connects `httpx`, `httpx`, `_SQLStub` to the rest of the system?**
  _10 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `kafka_producer.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07822410147991543 - nodes in this community are weakly interconnected._
- **Should `graph_store.py` be split into smaller, more focused modules?**
  _Cohesion score 0.09686609686609686 - nodes in this community are weakly interconnected._
- **Should `finnhub_client.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1339031339031339 - nodes in this community are weakly interconnected._