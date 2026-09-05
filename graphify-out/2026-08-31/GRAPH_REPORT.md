# Graph Report - graphrag_finance  (2026-08-31)

## Corpus Check
- 34 files · ~9,654 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 308 nodes · 473 edges · 18 communities (17 shown, 1 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 14 edges (avg confidence: 0.69)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- kafka_producer.py
- process_graph_message
- finnhub_client.py
- init_db.py
- psycopg2/__init__.py
- kafka_consumer.py
- utils.py
- RuntimeError
- pg_connection
- embed_texts
- Financial‑RAG Ingestion Module
- task_runner.py
- _Client

## God Nodes (most connected - your core abstractions)
1. `pg_connection()` - 11 edges
2. `send_to_dlt()` - 11 edges
3. `insert_raw_message()` - 11 edges
4. `publish_message()` - 10 edges
5. `process_graph_message()` - 9 edges
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
- `search()` --calls--> `embed_texts()`  [EXTRACTED]
  rag/retriever/main.py → embedding/embedder.py
- `pg_connection()` --calls--> `connect()`  [EXTRACTED]
  graph/db.py → tools/psycopg2/__init__.py
- `pg_connection()` --calls--> `_MockConnection`  [EXTRACTED]
  graph/db.py → tools/psycopg2/__init__.py

## Import Cycles
- None detected.

## Communities (18 total, 1 thin omitted)

### Community 0 - "kafka_producer.py"
Cohesion: 0.08
Nodes (28): _make_envelope(), Any, Build the internal message format that kafka_consumer._is_valid_payload…, Called by the ingestion modules. Returns True on successful publish, False…, send_record(), _ensure_topics(), get_producer(), _init_topic() (+20 more)

### Community 1 - "process_graph_message"
Cohesion: 0.15
Nodes (16): _clean_json_response(), extract_entities(), _extract_entities_via_ollama(), Extract entities and relationships from text using local LLM., Extract entities and relationships and persist them to Memgraph., Extract and parse JSON from LLM output, handling markdown fences or reasoning…, Query local Ollama server for entity extraction., store_graph_entities() (+8 more)

### Community 2 - "finnhub_client.py"
Cohesion: 0.08
Nodes (24): check_api(), _date_range(), fetch_finnhub(), _fetch_json_if_available(), _finnhub_params(), _get_json(), httpx, _is_api_available() (+16 more)

### Community 3 - "init_db.py"
Cohesion: 0.11
Nodes (13): InitDbTests, patch, _connect_db(), create_queue_table(), ensure_database(), _load_real_psycopg2(), _Psycopg2Stub, Database initialisation for the Financial - RAG ingestion pipeline. - Creates… (+5 more)

### Community 4 - "psycopg2/__init__.py"
Cohesion: 0.08
Nodes (12): Exception, str, connect(), execute_values(), _MockConnection, _MockCursor, OperationalError, Extended mock psycopg2 package to satisfy test imports. Provides: - ``connect``… (+4 more)

### Community 5 - "kafka_consumer.py"
Cohesion: 0.20
Nodes (14): _call(), ConsumerError, _deserialize_message(), ingest_message(), _is_valid_payload(), process_message(), Any, datetime (+6 more)

### Community 6 - "utils.py"
Cohesion: 0.11
Nodes (17): TestCase, InsertRawMessageTests, get_dsn(), Return a DSN that connects to the given database. Constructs the DSN from…, _db_connection(), insert_raw_message(), _kafka_producer(), _payload_hash() (+9 more)

### Community 7 - "RuntimeError"
Cohesion: 0.12
Nodes (6): KafkaConsumerFallback, DLTQueueError, KafkaConsumerFallback, Raised when a message cannot be routed to the dead‑letter topic., RuntimeError, KafkaProducerFallback

### Community 8 - "pg_connection"
Cohesion: 0.08
Nodes (20): BaseModel, _build_dsn(), pg_connection(), Return a psycopg2 connection. If *database* is provided, the DSN is adjusted to…, Construct DSN for PostgreSQL. Uses POSTGRES_URL if set; otherwise builds from…, close_memgraph_driver(), _DummyDriver, _DummySession (+12 more)

### Community 9 - "embed_texts"
Cohesion: 0.18
Nodes (11): _DummyModel, embed_texts(), Return a list of embedding vectors for the supplied texts. The vectors are…, Persist embeddings for a list of graph nodes into the PGVECTOR table. Each…, store_node_embeddings(), fetch_embedding(), pg_conn(), fixture (+3 more)

### Community 10 - "Financial‑RAG Ingestion Module"
Cohesion: 0.20
Nodes (9): API Contract Notes, Database setup (`python init_db.py`), Environment Variables, Financial‑RAG Ingestion Module, Kafka consumer (`python kafka_consumer.py`), Program Flow, Quick start, Requirements (+1 more)

### Community 12 - "task_runner.py"
Cohesion: 0.16
Nodes (11): Run a synchronous fetch function in a thread executor., Execute all fetch functions registered via the @ingest_task decorator. Each…, run_all(), _run_sync(), Execute all registered ingestion tasks concurrently., run_all(), BackgroundScheduler, CronTrigger (+3 more)

## Knowledge Gaps
- **9 isolated node(s):** `httpx`, `_SQLStub`, `Quick start`, `Requirements`, `Environment Variables` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `embed_texts()` connect `embed_texts` to `pg_connection`, `kafka_consumer.py`?**
  _High betweenness centrality (0.066) - this node is a cross-community bridge._
- **Why does `send_to_dlt()` connect `kafka_producer.py` to `kafka_consumer.py`?**
  _High betweenness centrality (0.063) - this node is a cross-community bridge._
- **Why does `store_graph_entities()` connect `process_graph_message` to `pg_connection`, `kafka_consumer.py`?**
  _High betweenness centrality (0.054) - this node is a cross-community bridge._
- **What connects `httpx`, `_SQLStub`, `Quick start` to the rest of the system?**
  _9 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `kafka_producer.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07822410147991543 - nodes in this community are weakly interconnected._
- **Should `process_graph_message` be split into smaller, more focused modules?**
  _Cohesion score 0.14624505928853754 - nodes in this community are weakly interconnected._
- **Should `finnhub_client.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08408408408408409 - nodes in this community are weakly interconnected._