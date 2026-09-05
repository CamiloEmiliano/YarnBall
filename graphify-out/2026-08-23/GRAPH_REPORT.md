# Graph Report - graphrag_finance  (2026-08-21)

## Corpus Check
- 34 files · ~9,207 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 249 nodes · 409 edges · 19 communities (16 shown, 3 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 9 edges (avg confidence: 0.63)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- kafka_producer.py
- graph_store.py
- kafka_consumer.py
- api_ingest.py
- utils.py
- rss_ingest.py
- InitDbTests
- TestKafkaProducer
- retriever/main.py
- ApiIngestContractTests
- Financial‑RAG Ingestion Module
- run_all
- ingest/__init__.py
- schedule_jobs

## God Nodes (most connected - your core abstractions)
1. `publish_message()` - 12 edges
2. `fetch_free_historical_news()` - 11 edges
3. `send_to_dlt()` - 11 edges
4. `insert_raw_message()` - 11 edges
5. `ApiIngestContractTests` - 10 edges
6. `run_consumer()` - 9 edges
7. `TestKafkaProducer` - 9 edges
8. `embed_texts()` - 8 edges
9. `store_graph_entities()` - 8 edges
10. `send_record()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `run_consumer()` --calls--> `embed_texts()`  [EXTRACTED]
  kafka/kafka_consumer.py → embedding/embedder.py
- `run_consumer()` --calls--> `store_graph_entities()`  [EXTRACTED]
  kafka/kafka_consumer.py → graph/graph_store.py
- `fetch_rss_historical_news()` --calls--> `store_raw()`  [EXTRACTED]
  ingest/rss_ingest.py → tools/utils.py
- `KafkaConsumerTests` --uses--> `ConsumerError`  [INFERRED]
  tests/test_kafka_consumer.py → kafka/kafka_consumer.py
- `store_raw()` --calls--> `insert_raw_message()`  [EXTRACTED]
  kafka/kafka_consumer.py → tools/utils.py

## Import Cycles
- None detected.

## Communities (19 total, 3 thin omitted)

### Community 0 - "kafka_producer.py"
Cohesion: 0.12
Nodes (22): Publish a record to Kafka. The payload is wrapped in a simple dict containing…, send_record(), _make_envelope(), Any, Build the internal message format that kafka_consumer._is_valid_payload…, Called by the ingestion modules. Returns True on successful publish, False…, send_record(), _ensure_topics() (+14 more)

### Community 1 - "graph_store.py"
Cohesion: 0.12
Nodes (16): _DummyChat, _DummyChoice, _DummyMessage, _extract_entities_with_qwen(), OpenAI, Extract entities using a locally‑hosted Qwen model. The model is loaded once…, Extract entities using Qwen and persist them to Memgraph., store_graph_entities() (+8 more)

### Community 2 - "kafka_consumer.py"
Cohesion: 0.20
Nodes (14): _call(), ConsumerError, _deserialize_message(), ingest_message(), _is_valid_payload(), process_message(), Any, datetime (+6 more)

### Community 3 - "api_ingest.py"
Cohesion: 0.21
Nodes (23): _alphavantage_params(), _alphavantage_response_ok(), _date_range(), fetch_free_historical_news(), _fetch_guardian_all_pages(), _fetch_json_if_available(), _fetch_newsapi_all_pages(), _finnhub_params() (+15 more)

### Community 4 - "utils.py"
Cohesion: 0.10
Nodes (22): TestCase, InsertRawMessageTests, create_queue_table(), ensure_database(), get_dsn(), Database initialisation for the Financial - RAG ingestion pipeline. - Creates…, Return a DSN that connects to the given database. Prefer a full…, Create the target DB if it does not already exist. (+14 more)

### Community 5 - "rss_ingest.py"
Cohesion: 0.16
Nodes (14): FeedParserDict, fetch_rss(), fetch_rss_historical_news(), _is_feed_available(), _parse_feed_if_available(), _process_investing_rss(), _process_yahoo_rss(), _published_date() (+6 more)

### Community 7 - "TestKafkaProducer"
Cohesion: 0.08
Nodes (13): DLTQueueError, KafkaConsumerFallback, Raised when a message cannot be routed to the dead‑letter topic., KafkaProducerFallback, RuntimeError, patch, If producer.send().get() raises, the function logs failure and returns False., _producer should instantiate KafkaProducer with the bootstrap server from the… (+5 more)

### Community 8 - "retriever/main.py"
Cohesion: 0.11
Nodes (20): BaseModel, _DummyModel, embed_texts(), Return a list of embedding vectors for the supplied texts. The vectors are…, pg_connection(), Return a new psycopg2 connection using the DSN defined above., Persist embeddings for a list of graph nodes into the PGVECTOR table. Each…, store_node_embeddings() (+12 more)

### Community 10 - "Financial‑RAG Ingestion Module"
Cohesion: 0.20
Nodes (9): API Contract Notes, Database setup (`python init_db.py`), Environment Variables, Financial‑RAG Ingestion Module, Kafka consumer (`python kafka_consumer.py`), Program Flow, Quick start, Requirements (+1 more)

### Community 11 - "run_all"
Cohesion: 0.50
Nodes (4): Utility to run a coroutine and log unexpected errors., Discover every module in `ingest/` that defines a `fetch_*` function and…, run_all(), _run_coroutine()

### Community 17 - "schedule_jobs"
Cohesion: 0.70
Nodes (4): BackgroundScheduler, main(), run_ingestion(), schedule_jobs()

## Knowledge Gaps
- **7 isolated node(s):** `Quick start`, `Requirements`, `Environment Variables`, `API Contract Notes`, `Runtime ingestion (`python main.py`)` (+2 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `embed_texts()` connect `retriever/main.py` to `kafka_consumer.py`?**
  _High betweenness centrality (0.117) - this node is a cross-community bridge._
- **Why does `send_to_dlt()` connect `kafka_producer.py` to `kafka_consumer.py`, `TestKafkaProducer`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **What connects `Quick start`, `Requirements`, `Environment Variables` to the rest of the system?**
  _7 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `kafka_producer.py` be split into smaller, more focused modules?**
  _Cohesion score 0.12315270935960591 - nodes in this community are weakly interconnected._
- **Should `graph_store.py` be split into smaller, more focused modules?**
  _Cohesion score 0.12 - nodes in this community are weakly interconnected._
- **Should `utils.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10098522167487685 - nodes in this community are weakly interconnected._
- **Should `TestKafkaProducer` be split into smaller, more focused modules?**
  _Cohesion score 0.08275862068965517 - nodes in this community are weakly interconnected._