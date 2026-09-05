# Graph Report - graphrag_finance  (2026-08-20)

## Corpus Check
- 26 files · ~8,002 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 218 nodes · 354 edges · 17 communities (15 shown, 2 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 8 edges (avg confidence: 0.65)
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
- RuntimeError
- ApiIngestContractTests
- Financial‑RAG Ingestion Module
- run_all
- ingest/__init__.py

## God Nodes (most connected - your core abstractions)
1. `publish_message()` - 12 edges
2. `fetch_free_historical_news()` - 11 edges
3. `send_to_dlt()` - 11 edges
4. `insert_raw_message()` - 11 edges
5. `ApiIngestContractTests` - 10 edges
6. `run_consumer()` - 9 edges
7. `TestKafkaProducer` - 9 edges
8. `store_graph_entities()` - 8 edges
9. `send_record()` - 8 edges
10. `fetch_rss_historical_news()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `run_consumer()` --calls--> `store_graph_entities()`  [EXTRACTED]
  kafka/kafka_consumer.py → graph/graph_store.py
- `send_record()` --calls--> `publish_message()`  [EXTRACTED]
  ingest/kafka_driver.py → kafka/kafka_producer.py
- `fetch_rss_historical_news()` --calls--> `store_raw()`  [EXTRACTED]
  ingest/rss_ingest.py → tools/utils.py
- `KafkaConsumerTests` --uses--> `ConsumerError`  [INFERRED]
  tests/test_kafka_consumer.py → kafka/kafka_consumer.py
- `store_raw()` --calls--> `insert_raw_message()`  [EXTRACTED]
  kafka/kafka_consumer.py → tools/utils.py

## Import Cycles
- None detected.

## Communities (17 total, 2 thin omitted)

### Community 0 - "kafka_producer.py"
Cohesion: 0.14
Nodes (20): _make_envelope(), Any, Build the internal message format that kafka_consumer._is_valid_payload…, Called by the ingestion modules. Returns True on successful publish, False…, send_record(), _ensure_topics(), get_producer(), _init_topic() (+12 more)

### Community 1 - "graph_store.py"
Cohesion: 0.13
Nodes (15): _DummyChat, _DummyChoice, _DummyMessage, _extract_entities_with_qwen(), OpenAI, Extract entities using Qwen and persist them to Memgraph., store_graph_entities(), get_memgraph_driver() (+7 more)

### Community 2 - "kafka_consumer.py"
Cohesion: 0.14
Nodes (17): _DummyModel, embed_texts(), Return a list of embedding vectors for the supplied texts. The vectors are…, _call(), ConsumerError, _deserialize_message(), ingest_message(), _is_valid_payload() (+9 more)

### Community 3 - "api_ingest.py"
Cohesion: 0.21
Nodes (23): _alphavantage_params(), _alphavantage_response_ok(), _date_range(), fetch_free_historical_news(), _fetch_guardian_all_pages(), _fetch_json_if_available(), _fetch_newsapi_all_pages(), _finnhub_params() (+15 more)

### Community 4 - "utils.py"
Cohesion: 0.14
Nodes (15): TestCase, InsertRawMessageTests, _db_connection(), insert_raw_message(), _kafka_producer(), _payload_hash(), datetime, KafkaProducer (+7 more)

### Community 5 - "rss_ingest.py"
Cohesion: 0.14
Nodes (16): FeedParserDict, Publish a record to Kafka. The payload is wrapped in a simple dict containing…, send_record(), fetch_rss(), fetch_rss_historical_news(), _is_feed_available(), _parse_feed_if_available(), _process_investing_rss() (+8 more)

### Community 6 - "InitDbTests"
Cohesion: 0.16
Nodes (9): InitDbTests, patch, create_queue_table(), ensure_database(), get_dsn(), Database initialisation for the Financial - RAG ingestion pipeline. - Creates…, Return a DSN that connects to the given database. Prefer a full…, Create the target DB if it does not already exist. (+1 more)

### Community 7 - "TestKafkaProducer"
Cohesion: 0.14
Nodes (8): KafkaProducerFallback, patch, If producer.send().get() raises, the function logs failure and returns False., _producer should instantiate KafkaProducer with the bootstrap server from the…, When the kafka library cannot be imported, the fallback class is used.…, When producer is None, send_to_dlt logs a skip event and returns False., A healthy producer should send the event, log success and return True., TestKafkaProducer

### Community 8 - "RuntimeError"
Cohesion: 0.18
Nodes (5): DLTQueueError, KafkaConsumerFallback, Raised when a message cannot be routed to the dead‑letter topic., RuntimeError, KafkaProducerFallback

### Community 10 - "Financial‑RAG Ingestion Module"
Cohesion: 0.20
Nodes (9): API Contract Notes, Database setup (`python init_db.py`), Environment Variables, Financial‑RAG Ingestion Module, Kafka consumer (`python kafka_consumer.py`), Program Flow, Quick start, Requirements (+1 more)

### Community 11 - "run_all"
Cohesion: 0.50
Nodes (4): Utility to run a coroutine and log unexpected errors., Discover every module in `ingest/` that defines a `fetch_*` function and…, run_all(), _run_coroutine()

## Knowledge Gaps
- **7 isolated node(s):** `Quick start`, `Requirements`, `Environment Variables`, `API Contract Notes`, `Runtime ingestion (`python main.py`)` (+2 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `publish_message()` connect `kafka_producer.py` to `rss_ingest.py`?**
  _High betweenness centrality (0.080) - this node is a cross-community bridge._
- **Why does `send_to_dlt()` connect `kafka_producer.py` to `kafka_consumer.py`, `TestKafkaProducer`?**
  _High betweenness centrality (0.072) - this node is a cross-community bridge._
- **What connects `Quick start`, `Requirements`, `Environment Variables` to the rest of the system?**
  _7 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `kafka_producer.py` be split into smaller, more focused modules?**
  _Cohesion score 0.13846153846153847 - nodes in this community are weakly interconnected._
- **Should `graph_store.py` be split into smaller, more focused modules?**
  _Cohesion score 0.13043478260869565 - nodes in this community are weakly interconnected._
- **Should `kafka_consumer.py` be split into smaller, more focused modules?**
  _Cohesion score 0.14333333333333334 - nodes in this community are weakly interconnected._
- **Should `utils.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1380952380952381 - nodes in this community are weakly interconnected._