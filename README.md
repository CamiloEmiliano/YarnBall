# GRAPH‑RAG Ingestion Module

A collection of tiny, single‑purpose Python scripts that fetch raw financial‑news
payloads, publish them to Kafka, and persist them into the PostgreSQL staging queue
(`financial_news_queue`) through a dedicated consumer.

## Quick start

```powershell
# 1️ Create a virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\activate

# 2 Install dependencies
pip install -r requirements.txt

# 3 Configure secrets (optional API keys for services that need them)
#    Copy `.env.example` → `.env` and fill in any keys:
#    FINNHUB_API_KEY.

# 4 Initialise the DB (run once)
python init_db.py

# 5 Start the Kafka consumer
python kafka_consumer.py

# 6 Run the orchestrator – it will call every `fetch_*` function automatically
python main.py
```

## Requirements

- Python 3.13 or greater

## Environment Variables

Start from `.env.example` and create your local `.env`.

Required

- `POSTGRES_URL`: PostgreSQL DSN used by ingestion writers.
- `KAFKA_BOOTSTRAP_SERVERS`: Kafka broker list for producers and consumer.
- `KAFKA_TOPIC`: Topic used for raw ingestion events.

Optional (shared)

- `HIST_START`: Historical window start timestamp in UTC (`%Y-%m-%d %H:%M:%S`).
- `HIST_END`: Historical window end timestamp in UTC (`%Y-%m-%d %H:%M:%S`).
- `FINNHUB_TICKERS`: Comma-separated tickers used by ticker-based jobs.
- `KAFKA_GROUP_ID`: Consumer group for `kafka_consumer.py`.
- `KAFKA_DLT_TOPIC`: Dead-letter topic for invalid/failed consumer messages.

Optional (API-based ingestion in `ingest/finnhub_client.py`)

- `FINNHUB_API_KEY`
- `FINNHUB_TICKERS`: Comma-separated tickers (e.g. `AAPL,MSFT,GOOGL`).

## API Contract Notes

`ingest/finnhub_client.py` is aligned to Finnhub docs for parameter names, date formats,
rate limiting (TokenBucket: 60 calls/min), and response shapes:

- Finnhub (`/company-news`)
	- Uses `symbol`, `from`, `to` with `YYYY-MM-DD` format.
	- Parses the documented top-level array response.

## Program Flow

### Runtime ingestion (`python main.py`)

```mermaid
flowchart TD
	A[Start main.py] --> B{Python >= 3.13?}
	B -- No --> Z[Raise RuntimeError and stop]
	B -- Yes --> C[task_runner.run_all]

	C --> D[Discover registered tasks via @ingest_task]
	D --> E[Wrap each fetch function in asyncio.to_thread]
	E --> F[asyncio.gather executes fetchers concurrently]

	F --> I[fetch_finnhub in finnhub_client.py]

	I --> I1{FINNHUB_API_KEY set?}
	I1 -- Yes --> I2[TokenBucket rate limiter: 60 req/min]
	I2 --> I3[Loop tickers and fetch Finnhub]
	I1 -- No --> I4[Skip Finnhub]

	S --> S1[Compute payload hash]
	S1 --> S2[Publish message to Kafka topic]
	S2 --> S3[kafka_consumer.py reads topic]
	S3 --> S4[INSERT into financial_news_queue]
	S4 --> S5[ON CONFLICT source_hash DO NOTHING]
	S5 --> T[Log queued_raw and stored_raw events]
```

### Kafka consumer (`python kafka_consumer.py`)

```mermaid
flowchart TD
	A[Start kafka_consumer.py] --> B[Connect to Kafka topic]
	B --> C[Consume raw event]
	C --> D[Deserialize JSON payload]
	D --> E{Payload valid?}
	E -- No --> F[Publish event to dead-letter topic]
	F --> G[Commit and continue]
	E -- Yes --> H[Insert into financial_news_queue]
	H --> I{Insert succeeds?}
	I -- No --> J[Publish event to dead-letter topic]
	J --> G
	I -- Yes --> K[Commit and continue]
	H --> L{source_hash already exists?}
	L -- Yes --> M[Postgres ignores duplicate]
	L -- No --> N[Persist new row]
	M --> K
	N --> K
```

### Database setup (`python init_db.py`)

```mermaid
flowchart TD
	A[Start init_db.py] --> B[Optional load .env]
	B --> C[Read FINANCIAL_RAG_DB default financial_rag]
	C --> D[ensure_database]
	D --> E[Connect to postgres admin DB]
	E --> F{Target DB exists?}
	F -- No --> G[CREATE DATABASE]
	F -- Yes --> H[Continue]
	G --> I[create_queue_table]
	H --> I
	I --> J[Connect to target DB]
	J --> K[CREATE TABLE IF NOT EXISTS financial_news_queue]
	K --> L[Ensure UNIQUE constraint uq_source_hash]
	L --> M[Commit and exit]
```