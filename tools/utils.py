import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
try:
    import psycopg2
except ImportError:
    from tools import psycopg2
try:
    from dotenv import load_dotenv
except ImportError:
    # dotenv not available; define no-op
    def load_dotenv(*args, **kwargs):
        pass
try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except ImportError:
    # tenacity not installed; provide no-op decorators and helpers
    def retry(*dargs, **dkwargs):
        def decorator(fn):
            return fn
        return decorator
    def stop_after_attempt(n):
        return None
    def wait_exponential(*args, **kwargs):
        return None

class KafkaProducerFallback:
    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("kafka-python is required for queue publishing")

    def send(self, *args, **kwargs):
        raise RuntimeError("kafka-python is required for queue publishing")

    def close(self) -> None:
        return None

try:
    from kafka import KafkaProducer as KafkaProducerType
except ImportError:  # pragma: no cover - exercised in minimal environments
    KafkaProducerType = KafkaProducerFallback

KafkaProducer = KafkaProducerType

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
from .init_db import get_dsn

# ----------------------------------------------------------------------
# Logging – JSON lines (easy to ship to ELK, Cloud Logging, etc.)
# ----------------------------------------------------------------------
logger = logging.getLogger("ingest")
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(
        '{\n    "time":"%(asctime)s",\n    "level":"%(levelname)s",\n    "msg":%(message)s}\n'
    )
)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC")
_producer: KafkaProducer | None = None

# ----------------------------------------------------------------------
# Database helper – inserts raw payloads with a dedup hash.
# ----------------------------------------------------------------------
def _db_connection():
    # Expect a POSTGRES_URL env var (Postgres). Adjust as needed.
    dsn = get_dsn()
    return psycopg2.connect(dsn)

def _payload_hash(payload: str) -> str:
    """Short deterministic hash used for dupe‑checking."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

def _kafka_producer() -> KafkaProducer:
    global _producer

    if _producer is None:
        _producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        )

    return _producer

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
def insert_raw_message(
    raw_payload: str, fetched_at: datetime, payload_hash: str
) -> None:
    """Insert raw ingestion payload and extracted metadata into the financial_news_queue table.

    The `raw_payload` is a JSON string from the source processors. It may contain keys such as
    ``title``, ``url`` (source_url), ``published`` (ISO timestamp), ``author``, ``category``,
    ``tags`` (list), ``ticker_symbols`` or ``ticker``, ``sentiment`` and ``provider``.
    Missing keys are stored as NULL.
    """
    try:
        data = json.loads(raw_payload)
    except Exception:
        logger.warning(
            "Failed to parse raw_payload JSON; inserting only raw_text."
        )
        data = {}

    title = data.get("title")
    source_url = data.get("url") or data.get("source_url")
    # Parse published timestamp if present
    published_at = None
    if "published" in data:
        try:
            published_at = datetime.fromisoformat(data["published"]).replace(
                tzinfo=timezone.utc
            )
        except Exception:
            try:
                published_at = datetime.fromtimestamp(int(data["published"]))
            except Exception:
                published_at = None
    author = data.get("author")
    category = data.get("category")
    tags = (
        json.dumps(data.get("tags"))
        if isinstance(data.get("tags"), (list, dict))
        else None
    )
    ticker_symbols = (
        json.dumps(data.get("ticker_symbols"))
        if isinstance(data.get("ticker_symbols"), (list, dict))
        else json.dumps([data.get("ticker")]) if data.get("ticker") else None
    )
    sentiment_score = data.get("sentiment") or data.get("sentiment_score")
    provider = data.get("source") or data.get("provider")
    ingestion_start = None
    ingestion_end = None

    sql = """
        INSERT INTO financial_news_queue (
            raw_text, fetched_at, status, source_hash,
            source_url, title, published_at, author, category,
            tags, ticker_symbols, sentiment_score, provider,
            ingestion_window_start, ingestion_window_end
        ) VALUES (
            %s, %s, %s, %s,
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s
        ) ON CONFLICT (source_hash) DO NOTHING;
    """
    with _db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    raw_payload,
                    fetched_at,
                    "pending",  # status placeholder for compatibility with tests
                    payload_hash,
                    source_url,
                    title,
                    published_at,
                    author,
                    category,
                    tags,
                    ticker_symbols,
                    sentiment_score,
                    provider,
                    ingestion_start,
                    ingestion_end,
                ),
            )
        conn.commit()

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
)
def store_raw(source: str, raw_payload: str) -> None:
    """
    Publish a raw ingestion event to Kafka for downstream database persistence.
    """
    payload_hash = _payload_hash(raw_payload)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    message = {
        "source": source,
        "raw_payload": raw_payload,
        "fetched_at": now.isoformat(),
        "source_hash": payload_hash,
    }
    future = _kafka_producer().send(
        KAFKA_TOPIC, key=payload_hash.encode("utf-8"), value=message
    )
    future.get(timeout=30)
    logger.info(
        json.dumps(
            {
                "event": "queued_raw",
                "source": source,
                "hash": payload_hash,
                "size": len(raw_payload),
            }
        )
    )

def store_embedding(payload_hash: str, vector: list[float]) -> None:
    """Persist a pgvector embedding for an existing row.
    The vector is stored as a PostgreSQL `vector` type.
    """
    sql = """
        UPDATE financial_news_queue
        SET embedding = %s::vector
        WHERE source_hash = %s;
    """
    vector_str = "[" + ",".join(map(str, vector)) + "]"
    with _db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (vector_str, payload_hash))
        conn.commit()