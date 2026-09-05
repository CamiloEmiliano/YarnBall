# ingest/scraping_worker.py
# -*- coding: utf-8 -*-
"""Full-text Web Scraping & Self-Learning Blacklist Worker.

Consumes raw news events, extracts article content with Trafilatura, maintains a
dynamic self-learning domain blacklist in `domain_status`, and idempotently
inserts extracted articles into `financial_news_queue`.
"""

import hashlib
import json
import logging
import os
import re
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# PostgreSQL connector
from tools.init_db import _connect_db, get_dsn

# Optional Kafka consumer
try:
    from kafka import KafkaConsumer
except ImportError:
    KafkaConsumer = None

# Optional Trafilatura
try:
    import trafilatura
except ImportError:
    trafilatura = None

# Optional httpx
try:
    import httpx
except ImportError:
    httpx = None

# ----------------------------------------------------------------------
# Logging – JSON lines
# ----------------------------------------------------------------------
logger = logging.getLogger("scraping_worker")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            '{\n    "time":"%(asctime)s",\n    "level":"%(levelname)s",\n    "msg":%(message)s}\n'
        )
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Paywall indicator phrases
PAYWALL_PHRASES = [
    "subscribe to continue reading",
    "this article is exclusive to subscribers",
    "sign in for full access",
    "subscribe for unlimited access",
    "already a subscriber",
    "you have reached your limit of free articles",
    "to read the full story",
    "subscriber-only content",
    "join now to read",
]

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def get_domain_from_url(url: str) -> str:
    """Extract clean lower-case domain from URL."""
    if not url:
        return ""
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    if ":" in netloc:
        netloc = netloc.split(":")[0]
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def compute_source_hash(url: str, raw_payload: Optional[str] = None) -> str:
    """Compute deterministic SHA-256 hash for deduplication."""
    identifier = url.strip() if url else (raw_payload or "")
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:32]


def is_paywall_or_stub(text: Optional[str], min_word_count: int = 40) -> bool:
    """Check if extracted text is a paywall barrier or truncated stub."""
    if not text:
        return True
    words = text.split()
    if len(words) < min_word_count:
        return True
    lower_text = text.lower()
    for phrase in PAYWALL_PHRASES:
        if phrase in lower_text:
            return True
    return False


# ----------------------------------------------------------------------
# Dynamic Domain Blacklist (PostgreSQL)
# ----------------------------------------------------------------------
def is_domain_blacklisted(domain: str, conn=None) -> bool:
    """Check whether a domain is currently blacklisted in PostgreSQL domain_status."""
    if not domain:
        return False
    close_after = False
    if conn is None:
        try:
            conn = _connect_db(os.getenv("FINANCIAL_RAG_DB", "financial_rag"))
            close_after = True
        except Exception as exc:
            logger.warning(f"DB connection failed while checking domain {domain}: {exc}")
            return False

    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status FROM domain_status WHERE domain = %s;",
                (domain,),
            )
            row = cur.fetchone()
            if row:
                status = row[0] if isinstance(row, (tuple, list)) else row.get("status")
                return status == "blacklisted"
            return False
    except Exception as exc:
        logger.warning(f"Error checking domain status for {domain}: {exc}")
        return False
    finally:
        if close_after and conn:
            conn.close()


def record_domain_success(domain: str, conn=None) -> None:
    """Reset consecutive failures to 0 on successful scrape."""
    if not domain:
        return
    close_after = False
    if conn is None:
        try:
            conn = _connect_db(os.getenv("FINANCIAL_RAG_DB", "financial_rag"))
            close_after = True
        except Exception:
            return

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO domain_status (domain, consecutive_failures, status, updated_at)
                VALUES (%s, 0, 'allowed', now())
                ON CONFLICT (domain) DO UPDATE
                SET consecutive_failures = 0, status = 'allowed', updated_at = now();
                """,
                (domain,),
            )
        conn.commit()
    except Exception as exc:
        logger.warning(f"Failed to record domain success for {domain}: {exc}")
    finally:
        if close_after and conn:
            conn.close()


def record_domain_failure(domain: str, failure_threshold: int = 3, conn=None) -> None:
    """Increment failure count and auto-blacklist if threshold reached."""
    if not domain:
        return
    close_after = False
    if conn is None:
        try:
            conn = _connect_db(os.getenv("FINANCIAL_RAG_DB", "financial_rag"))
            close_after = True
        except Exception:
            return

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO domain_status (domain, consecutive_failures, status, updated_at)
                VALUES (%s, 1, 'allowed', now())
                ON CONFLICT (domain) DO UPDATE
                SET consecutive_failures = domain_status.consecutive_failures + 1,
                    status = CASE 
                        WHEN domain_status.consecutive_failures + 1 >= %s THEN 'blacklisted' 
                        ELSE 'allowed' 
                    END,
                    updated_at = now();
                """,
                (domain, failure_threshold),
            )
        conn.commit()
    except Exception as exc:
        logger.warning(f"Failed to record domain failure for {domain}: {exc}")
    finally:
        if close_after and conn:
            conn.close()


# ----------------------------------------------------------------------
# Text Extraction
# ----------------------------------------------------------------------
def extract_text_from_html(html: str) -> Optional[str]:
    """Extract clean body text from HTML using Trafilatura (or regex fallback)."""
    if not html:
        return None
    if trafilatura is not None:
        extracted = trafilatura.extract(
            html,
            output_format="txt",
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
        if extracted:
            return extracted.strip()

    # Fallback: simple tag stripper
    clean = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r"<[^>]+>", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean if clean else None


def fetch_and_extract(url: str, raw_html: Optional[str] = None) -> Tuple[Optional[str], Dict[str, Any]]:
    """Fetch URL and extract clean text and metadata."""
    metadata: Dict[str, Any] = {}
    html = raw_html

    if not html:
        if trafilatura is not None:
            try:
                html = trafilatura.fetch_url(url)
            except Exception as exc:
                logger.warning(f"Trafilatura fetch failed for {url}: {exc}")
                html = None

        if not html and httpx is not None:
            try:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                resp = httpx.get(url, headers=headers, timeout=10.0, follow_redirects=True)
                if resp.status_code == 200:
                    html = resp.text
            except Exception as exc:
                logger.warning(f"HTTPX fetch failed for {url}: {exc}")
                html = None

    if not html:
        return None, metadata

    text = extract_text_from_html(html)

    # Check for paywall or stub
    if is_paywall_or_stub(text):
        return None, metadata

    return text, metadata


# ----------------------------------------------------------------------
# Pipeline Message Processing
# ----------------------------------------------------------------------
def insert_extracted_article(
    extracted_text: str,
    source_url: str,
    source_hash: str,
    title: Optional[str] = None,
    published_at: Optional[datetime] = None,
    author: Optional[str] = None,
    category: Optional[str] = None,
    tags: Optional[Any] = None,
    ticker_symbols: Optional[Any] = None,
    sentiment_score: Optional[float] = None,
    provider: Optional[str] = None,
    conn=None,
) -> bool:
    """Idempotently insert extracted article into financial_news_queue."""
    close_after = False
    if conn is None:
        conn = _connect_db(os.getenv("FINANCIAL_RAG_DB", "financial_rag"))
        close_after = True

    try:
        tags_json = json.dumps(tags) if isinstance(tags, (list, dict)) else None
        tickers_json = json.dumps(ticker_symbols) if isinstance(ticker_symbols, (list, dict)) else (
            json.dumps([ticker_symbols]) if ticker_symbols else None
        )
        now = datetime.now(timezone.utc)

        sql = """
            INSERT INTO financial_news_queue (
                raw_text, fetched_at, status, source_hash,
                source_url, title, published_at, author, category,
                tags, ticker_symbols, sentiment_score, provider
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s
            ) ON CONFLICT (source_hash) DO NOTHING;
        """
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    extracted_text,
                    now,
                    "extracted",
                    source_hash,
                    source_url,
                    title,
                    published_at,
                    author,
                    category,
                    tags_json,
                    tickers_json,
                    sentiment_score,
                    provider,
                ),
            )
        conn.commit()
        return True
    finally:
        if close_after and conn:
            conn.close()


def process_scraping_message(message_data: Dict[str, Any], conn=None) -> bool:
    """Process a single raw news message through the scraping pipeline."""
    # Parse payload if nested
    raw = message_data.get("raw_payload") if "raw_payload" in message_data else message_data.get("payload")
    if raw is not None and isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except Exception:
            payload = {}
    elif raw is not None and isinstance(raw, dict):
        payload = raw
    else:
        payload = message_data

    url = payload.get("url") or payload.get("source_url") or message_data.get("url")
    if not url:
        logger.warning("Message missing URL; cannot scrape.")
        return False

    domain = get_domain_from_url(url)

    # 1. Check blacklist
    if is_domain_blacklisted(domain, conn=conn):
        logger.info(f"Skipping blacklisted domain: {domain} ({url})")
        return False

    # 2. Extract full text
    raw_html = payload.get("raw_html") or message_data.get("raw_html")
    extracted_text, _ = fetch_and_extract(url, raw_html=raw_html)

    if not extracted_text:
        # Failed or paywall detected -> record failure
        logger.warning(f"Extraction failed / paywall detected for: {url} (domain: {domain})")
        record_domain_failure(domain, failure_threshold=3, conn=conn)
        return False

    # Success -> record domain success
    record_domain_success(domain, conn=conn)

    # 3. Compute hash and insert
    source_hash = payload.get("source_hash") or compute_source_hash(url)
    title = payload.get("title") or payload.get("headline")
    author = payload.get("author")
    category = payload.get("category")
    tags = payload.get("tags")
    ticker_symbols = payload.get("ticker_symbols") or payload.get("ticker")
    sentiment_score = payload.get("sentiment") or payload.get("sentiment_score")
    provider = payload.get("source") or payload.get("provider")

    published_at = None
    pub_val = payload.get("published") or payload.get("datetime")
    if pub_val:
        try:
            if isinstance(pub_val, (int, float)):
                published_at = datetime.fromtimestamp(pub_val, tz=timezone.utc)
            else:
                published_at = datetime.fromisoformat(str(pub_val)).replace(tzinfo=timezone.utc)
        except Exception:
            published_at = None

    inserted = insert_extracted_article(
        extracted_text=extracted_text,
        source_url=url,
        source_hash=source_hash,
        title=title,
        published_at=published_at,
        author=author,
        category=category,
        tags=tags,
        ticker_symbols=ticker_symbols,
        sentiment_score=sentiment_score,
        provider=provider,
        conn=conn,
    )

    # 4. Extract knowledge graph & persist to Memgraph and PGVECTOR
    enable_graph = os.getenv("ENABLE_GRAPH_EXTRACTION", "true").lower() in ("true", "1", "yes")
    if enable_graph and extracted_text:
        try:
            from graph.graph_store import store_graph_entities
            store_graph_entities(source_hash=source_hash, text=extracted_text)
        except Exception as exc:
            logger.warning(f"Graph extraction failed for {source_hash}: {exc}")

    logger.info(f"Successfully processed and queued article: {title or 'No Title'} (URL: {url})")
    return inserted


# ----------------------------------------------------------------------
# Worker Loop
# ----------------------------------------------------------------------
def run_worker(topic: Optional[str] = None, max_messages: Optional[int] = None) -> None:
    """Run continuous scraping worker consuming from Kafka."""
    kafka_topic = topic or os.getenv("KAFKA_TOPIC", "financial_news_raw")
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    group_id = os.getenv("KAFKA_GROUP_ID", "scraping-worker-group")

    if KafkaConsumer is None:
        raise RuntimeError("kafka-python is required to run the Kafka worker.")

    consumer = KafkaConsumer(
        kafka_topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    logger.info(f"Starting scraping worker on topic '{kafka_topic}'...")
    processed_count = 0

    running = True
    def shutdown_handler(signum, frame):
        nonlocal running
        logger.info("Shutdown signal received; closing consumer.")
        running = False

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        for message in consumer:
            if not running:
                break
            try:
                process_scraping_message(message.value)
                processed_count += 1
                if max_messages and processed_count >= max_messages:
                    break
            except Exception as exc:
                logger.error(f"Error processing message: {exc}")
    finally:
        consumer.close()
        logger.info("Scraping worker stopped.")


if __name__ == "__main__":
    run_worker()
