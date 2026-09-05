# Graph Report - graphrag_finance  (2026-08-18)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 212 nodes · 350 edges · 17 communities (14 shown, 3 thin omitted)
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
- RssIngestTests
- run_all
- ingest/__init__.py

## God Nodes (most connected - your core abstractions)
1. `publish_message()` - 12 edges
2. `send_to_dlt()` - 11 edges
3. `fetch_free_historical_news()` - 11 edges
4. `insert_raw_message()` - 11 edges
5. `ApiIngestContractTests` - 10 edges
6. `TestKafkaProducer` - 9 edges
7. `run_consumer()` - 9 edges
8. `get_producer()` - 8 edges
9. `store_graph_entities()` - 8 edges
10. `send_record()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `TestKafkaProducer` --uses--> `KafkaProducerFallback`  [INFERRED]
  tests/test_kafka_producer.py → kafka/kafka_producer.py
- `KafkaConsumerTests` --uses--> `ConsumerError`  [INFERRED]
  tests/test_kafka_consumer.py → kafka/kafka_consumer.py
- `send_record()` --calls--> `publish_message()`  [EXTRACTED]
  ingest/kafka_driver.py → kafka/kafka_producer.py
- `run_consumer()` --calls--> `store_graph_entities()`  [EXTRACTED]
  kafka/kafka_consumer.py → graph/graph_store.py
- `run_consumer()` --calls--> `insert_raw_message()`  [EXTRACTED]
  kafka/kafka_consumer.py → tools/utils.py

## Import Cycles
- None detected.

## Communities (17 total, 3 thin omitted)

### Community 0 - "kafka_producer.py"
Cohesion: 0.11
Nodes (21): _make_envelope(), Any, Build the internal message format that kafka_consumer._is_valid_payload…, Called by the ingestion modules. Returns True on successful publish, False…, send_record(), _ensure_topics(), get_producer(), _init_topic() (+13 more)

### Community 1 - "graph_store.py"
Cohesion: 0.11
Nodes (14): _DummyChat, _DummyChoice, _DummyDriver, _DummyMessage, _DummySession, _extract_entities_with_qwen(), get_memgraph_driver(), GraphDatabase (+6 more)

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
Cohesion: 0.23
Nodes (14): FeedParserDict, Publish a record to Kafka. The payload is wrapped in a simple dict containing…, send_record(), fetch_rss(), fetch_rss_historical_news(), _is_feed_available(), _parse_feed_if_available(), _process_investing_rss() (+6 more)

### Community 6 - "InitDbTests"
Cohesion: 0.16
Nodes (9): InitDbTests, patch, create_queue_table(), ensure_database(), get_dsn(), Database initialisation for the Financial - RAG ingestion pipeline. - Creates…, Return a DSN that connects to the given database. Prefer a full…, Create the target DB if it does not already exist. (+1 more)

### Community 7 - "TestKafkaProducer"
Cohesion: 0.19
Nodes (7): patch, If producer.send().get() raises, the function logs failure and returns False., _producer should instantiate KafkaProducer with the bootstrap server from the…, When the kafka library cannot be imported, the fallback class is used.…, When producer is None, send_to_dlt logs a skip event and returns False., A healthy producer should send the event, log success and return True., TestKafkaProducer

### Community 8 - "RuntimeError"
Cohesion: 0.18
Nodes (5): DLTQueueError, KafkaConsumerFallback, Raised when a message cannot be routed to the dead‑letter topic., RuntimeError, KafkaProducerFallback

### Community 11 - "run_all"
Cohesion: 0.50
Nodes (4): Utility to run a coroutine and log unexpected errors., Discover every module in `ingest/` that defines a `fetch_*` function and…, run_all(), _run_coroutine()

## Knowledge Gaps
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `publish_message()` connect `kafka_producer.py` to `rss_ingest.py`?**
  _High betweenness centrality (0.086) - this node is a cross-community bridge._
- **Why does `send_to_dlt()` connect `kafka_producer.py` to `kafka_consumer.py`, `TestKafkaProducer`?**
  _High betweenness centrality (0.079) - this node is a cross-community bridge._
- **Should `kafka_producer.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11494252873563218 - nodes in this community are weakly interconnected._
- **Should `graph_store.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10826210826210826 - nodes in this community are weakly interconnected._
- **Should `kafka_consumer.py` be split into smaller, more focused modules?**
  _Cohesion score 0.14333333333333334 - nodes in this community are weakly interconnected._
- **Should `utils.py` be split into smaller, more focused modules?**
  _Cohesion score 0.1380952380952381 - nodes in this community are weakly interconnected._