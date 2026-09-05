# Graph Report - graphrag_finance  (2026-08-26)

## Corpus Check
- 41 files · ~10,875 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 312 nodes · 488 edges · 25 communities (21 shown, 4 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 12 edges (avg confidence: 0.67)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- kafka_producer.py
- graph_store.py
- kafka_consumer.py
- api_ingest.py
- utils.py
- rss_ingest.py
- _MockConnection
- RuntimeError
- pg_connection
- ApiIngestContractTests
- Financial‑RAG Ingestion Module
- run_all
- ingest/__init__.py
- schedule_jobs
- check_alphavantage_api.py
- check_finnhub_api.py
- finnhub_client.py
- InitDbTests
- run_all

## God Nodes (most connected - your core abstractions)
1. `publish_message()` - 12 edges
2. `send_to_dlt()` - 11 edges
3. `insert_raw_message()` - 11 edges
4. `pg_connection()` - 10 edges
5. `fetch_free_historical_news()` - 10 edges
6. `send_record()` - 10 edges
7. `run_consumer()` - 9 edges
8. `_MockConnection` - 9 edges
9. `TestKafkaProducer` - 9 edges
10. `embed_texts()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `pg_connection()` --calls--> `_MockConnection`  [INFERRED]
  graph/db.py → psycopg2/__init__.py
- `store_node_embeddings()` --calls--> `execute_values()`  [INFERRED]
  graph/embedding_store.py → psycopg2/__init__.py
- `run_consumer()` --calls--> `embed_texts()`  [EXTRACTED]
  kafka/kafka_consumer.py → embedding/embedder.py
- `run_consumer()` --calls--> `store_graph_entities()`  [EXTRACTED]
  kafka/kafka_consumer.py → graph/graph_store.py
- `send_record()` --calls--> `publish_message()`  [EXTRACTED]
  ingest/kafka_driver.py → kafka/kafka_producer.py

## Import Cycles
- None detected.

## Communities (25 total, 4 thin omitted)

### Community 0 - "kafka_producer.py"
Cohesion: 0.08
Nodes (29): Exception, _make_envelope(), Any, Build the internal message format that kafka_consumer._is_valid_payload…, Called by the ingestion modules. Returns True on successful publish, False…, send_record(), _ensure_topics(), get_producer() (+21 more)

### Community 1 - "graph_store.py"
Cohesion: 0.09
Nodes (19): _DummyChat, _DummyChoice, _DummyMessage, _extract_entities_with_qwen(), OpenAI, Extract entities using a locally‑hosted Qwen model. The model is loaded once…, Extract entities using Qwen and persist them to Memgraph., store_graph_entities() (+11 more)

### Community 2 - "kafka_consumer.py"
Cohesion: 0.20
Nodes (14): _call(), ConsumerError, _deserialize_message(), ingest_message(), _is_valid_payload(), process_message(), Any, datetime (+6 more)

### Community 3 - "api_ingest.py"
Cohesion: 0.23
Nodes (19): _alphavantage_params(), _alphavantage_response_ok(), _date_range(), fetch_free_historical_news(), _fetch_guardian_all_pages(), _fetch_json_if_available(), _finnhub_params(), _get_json() (+11 more)

### Community 4 - "utils.py"
Cohesion: 0.10
Nodes (22): TestCase, InsertRawMessageTests, create_queue_table(), ensure_database(), get_dsn(), Database initialisation for the Financial - RAG ingestion pipeline. - Creates…, Return a DSN that connects to the given database. Constructs the DSN from…, Create the target DB if it does not already exist. (+14 more)

### Community 5 - "rss_ingest.py"
Cohesion: 0.14
Nodes (16): FeedParserDict, Publish a record to Kafka. The payload is wrapped in a simple dict containing…, send_record(), fetch_rss(), fetch_rss_historical_news(), _is_feed_available(), _parse_feed_if_available(), _process_investing_rss() (+8 more)

### Community 6 - "_MockConnection"
Cohesion: 0.08
Nodes (12): connect(), execute_values(), _MockConnection, _MockCursor, OperationalError, Extended mock psycopg2 package to satisfy test imports. Provides: - ``connect``…, Return a mock connection mimicking ``psycopg2.connect``. Additional arguments…, Very small shim for ``psycopg2.extras.execute_values``. The real function… (+4 more)

### Community 7 - "RuntimeError"
Cohesion: 0.18
Nodes (5): DLTQueueError, KafkaConsumerFallback, Raised when a message cannot be routed to the dead‑letter topic., RuntimeError, KafkaProducerFallback

### Community 8 - "pg_connection"
Cohesion: 0.10
Nodes (22): BaseModel, _DummyModel, embed_texts(), Return a list of embedding vectors for the supplied texts. The vectors are…, _build_dsn(), pg_connection(), Return a psycopg2 connection. If *database* is provided, the DSN is adjusted to…, Construct DSN for PostgreSQL. Uses POSTGRES_URL if set; otherwise builds from… (+14 more)

### Community 10 - "Financial‑RAG Ingestion Module"
Cohesion: 0.20
Nodes (9): API Contract Notes, Database setup (`python init_db.py`), Environment Variables, Financial‑RAG Ingestion Module, Kafka consumer (`python kafka_consumer.py`), Program Flow, Quick start, Requirements (+1 more)

### Community 11 - "run_all"
Cohesion: 0.50
Nodes (4): Utility to run a coroutine and log unexpected errors., Discover every module in `ingest/` that defines a `fetch_*` function and…, run_all(), _run_coroutine()

### Community 17 - "schedule_jobs"
Cohesion: 0.70
Nodes (4): BackgroundScheduler, main(), run_ingestion(), schedule_jobs()

### Community 19 - "check_alphavantage_api.py"
Cohesion: 0.83
Nodes (3): _date_range(), fetch_for_ticker(), main()

### Community 22 - "finnhub_client.py"
Cohesion: 0.19
Nodes (11): _date_range(), fetch_finnhub(), _fetch_json_if_available(), _finnhub_params(), _process_finnhub(), Any, Reuse the date-range helper from the original `api_ingest` module. Importing…, Transform Finnhub raw items into the payload format expected downstream.… (+3 more)

### Community 24 - "run_all"
Cohesion: 0.50
Nodes (4): Run a synchronous fetch function in a thread executor., Execute all `fetch_*` functions defined in the `ingest` package. Each fetch…, run_all(), _run_sync()

## Knowledge Gaps
- **7 isolated node(s):** `Quick start`, `Requirements`, `Environment Variables`, `API Contract Notes`, `Runtime ingestion (`python main.py`)` (+2 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `embed_texts()` connect `pg_connection` to `kafka_consumer.py`?**
  _High betweenness centrality (0.154) - this node is a cross-community bridge._
- **Why does `store_node_embeddings()` connect `pg_connection` to `_MockConnection`?**
  _High betweenness centrality (0.088) - this node is a cross-community bridge._
- **Why does `publish_message()` connect `kafka_producer.py` to `rss_ingest.py`?**
  _High betweenness centrality (0.083) - this node is a cross-community bridge._
- **What connects `Quick start`, `Requirements`, `Environment Variables` to the rest of the system?**
  _7 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `kafka_producer.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07575757575757576 - nodes in this community are weakly interconnected._
- **Should `graph_store.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08522727272727272 - nodes in this community are weakly interconnected._
- **Should `utils.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10098522167487685 - nodes in this community are weakly interconnected._