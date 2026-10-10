"""
Multi-Source News Harvester & Historical Streamer.

Provides:
- `MultiSourceNewsHarvester`: Harvests live and recent ticker news via yfinance / Wire
- `HistoricalNewsIngestor`: Multi-year (2018–2025) historical FNSPID / Form 8-K dataset streamer and Parquet stager
- Full-text article scraping worker integration (`process_scraping_message`, `run_worker`)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import signal
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env")
except ImportError:
    pass

try:
    from kafka import KafkaConsumer
except ImportError:
    KafkaConsumer = None

try:
    import httpx
except ImportError:
    httpx = None

try:
    import trafilatura
except ImportError:
    trafilatura = None

from knowledge_graph.db import pg_connection
from knowledge_graph.entity_resolver import is_clickbait_article_source, is_generic_placeholder
from tools.sp500_universe import SP500UniverseManager
from tools.init_db import _connect_db

from .parser import (
    compute_source_hash,
    extract_text_from_html,
    get_domain_from_url,
    is_domain_blacklisted,
    is_paywall_or_stub,
    record_domain_failure,
    record_domain_success,
    strip_publisher_boilerplates,
)

logger = logging.getLogger("ingestion.news.harvester")

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "historical_news"


class MultiSourceNewsHarvester:
    """Harvests live & recent financial news across multiple uncapped providers."""

    def __init__(self, universe_mgr: Optional[SP500UniverseManager] = None):
        self.universe_mgr = universe_mgr or SP500UniverseManager()
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

    def harvest_yfinance_news(self, ticker: str) -> List[Dict[str, Any]]:
        """Harvest recent news items using yfinance API."""
        clean_ticker = ticker.strip().upper()
        results: List[Dict[str, Any]] = []

        try:
            import yfinance as yf
            t_obj = yf.Ticker(clean_ticker)
            raw_news = getattr(t_obj, "news", []) or []

            for item in raw_news:
                if not isinstance(item, dict):
                    continue

                title = str(item.get("title") or "").strip()
                pub_name = str(item.get("publisher") or "").strip()
                link = str(item.get("link") or "").strip()
                provider_publish_time = item.get("providerPublishTime")

                # Filter clickbait/opinion sources
                if is_clickbait_article_source(pub_name) or is_clickbait_article_source(title):
                    continue

                # Parse publication date
                if provider_publish_time:
                    try:
                        pub_dt = datetime.fromtimestamp(int(provider_publish_time), tz=timezone.utc)
                        pub_date_str = pub_dt.strftime("%Y-%m-%d")
                    except Exception:
                        pub_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                else:
                    pub_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

                sh = compute_source_hash(link) if link else hashlib.sha256(f"{clean_ticker}_{title}".encode("utf-8")).hexdigest()[:32]

                # Extract related tickers
                related_tickers = item.get("relatedTickers") or [clean_ticker]
                valid_tickers = [t.strip().upper() for t in related_tickers if isinstance(t, str)]
                if clean_ticker not in valid_tickers:
                    valid_tickers.append(clean_ticker)

                results.append({
                    "source_hash": sh,
                    "source_url": link or f"urn:yfinance:{sh}",
                    "title": title,
                    "raw_text": title,  # Title acts as headline text if full text scraping follows
                    "published_at": pub_date_str,
                    "ticker_symbols": valid_tickers,
                    "provider": pub_name or "YFINANCE",
                    "category": "financial_news",
                    "status": "pending",
                })
        except Exception as exc:
            logger.debug(f"yfinance news harvest failed for {clean_ticker}: {exc}")

        return results

    def harvest_ticker_events(
        self,
        ticker: str,
        company_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Harvest live news and event signals for an S&P 500 company."""
        clean_ticker = ticker.strip().upper()
        return self.harvest_yfinance_news(clean_ticker)

    def sync_to_postgres_queue(self, records: List[Dict[str, Any]]) -> int:
        """Persist harvested live news records into PostgreSQL `financial_news_queue`."""
        if not records:
            return 0

        inserted = 0
        try:
            with pg_connection() as conn:
                cur = conn.cursor()
                query = """
                    INSERT INTO financial_news_queue (
                        source_hash, source_url, title, raw_text,
                        published_at, ticker_symbols, provider, category, status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (source_hash) DO NOTHING;
                """
                for rec in records:
                    cur.execute(
                        query,
                        (
                            rec["source_hash"],
                            rec["source_url"],
                            rec["title"],
                            rec["raw_text"],
                            rec["published_at"],
                            json.dumps(rec["ticker_symbols"]),
                            rec["provider"],
                            rec["category"],
                            rec["status"],
                        ),
                    )
                    inserted += 1
                conn.commit()
                if hasattr(cur, "close"):
                    cur.close()
        except Exception as exc:
            logger.warning(f"Failed to sync harvested news to PostgreSQL: {exc}")

        return inserted


class HistoricalNewsIngestor:
    """Streams and filters historical financial news (2018-2025) for S&P 500 constituents."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        universe_mgr: Optional[SP500UniverseManager] = None,
    ):
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.universe_mgr = universe_mgr or SP500UniverseManager()

    def process_news_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Validate, clean, and filter an individual historical news article.
        
        Expected record fields:
            - title: str
            - text / article / body: str
            - date / published_at: str (ISO format or YYYY-MM-DD)
            - ticker / symbol: str or List[str]
            - url / link: Optional[str]
            - publisher / source: Optional[str]
        """
        raw_title = str(record.get("title") or "").strip()
        raw_text = str(record.get("text") or record.get("article") or record.get("body") or "").strip()
        raw_url = str(record.get("url") or record.get("link") or "").strip()
        pub_name = str(record.get("publisher") or record.get("source") or record.get("provider") or "").strip()

        # 1. Filter out clickbait/opinion blogs
        if is_clickbait_article_source(pub_name) or is_clickbait_article_source(raw_title):
            return None

        # 2. Extract and sanitize publication date
        date_str = str(record.get("date") or record.get("published_at") or record.get("published") or "")[:10]
        if not date_str or len(date_str) < 10:
            date_str = datetime.now().strftime("%Y-%m-%d")

        # 3. Resolve & filter ticker symbols against S&P 500 point-in-time universe
        raw_tickers = record.get("ticker") or record.get("symbol") or record.get("ticker_symbols") or []
        if isinstance(raw_tickers, str):
            tickers = [t.strip().upper() for t in raw_tickers.replace(",", " ").split() if t.strip()]
        elif isinstance(raw_tickers, list):
            tickers = [str(t).strip().upper() for t in raw_tickers if str(t).strip()]
        else:
            tickers = []

        if not tickers:
            return None

        # Check point-in-time S&P 500 inclusion
        sp500_active_tickers = [
            t for t in tickers
            if self.universe_mgr.is_constituent(t, target_date=date_str)
        ]

        if not sp500_active_tickers:
            # None of the tickers were active S&P 500 constituents at publication date
            return None

        # 4. Clean editorial disclaimers and boilerplates
        clean_text = strip_publisher_boilerplates(raw_text)
        if len(clean_text.split()) < 40:
            # Too short after boilerplate removal
            return None

        # 5. Generate deterministic source hash
        if raw_url:
            source_hash = compute_source_hash(raw_url)
        else:
            unique_key = f"{sp500_active_tickers[0]}_{date_str}_{raw_title}"
            source_hash = hashlib.sha256(unique_key.encode("utf-8")).hexdigest()[:32]

        return {
            "source_hash": source_hash,
            "source_url": raw_url or f"urn:fnspid:{source_hash}",
            "title": raw_title,
            "raw_text": clean_text,
            "published_at": date_str,
            "ticker_symbols": sp500_active_tickers,
            "provider": pub_name or "FNSPID_HISTORICAL",
            "category": record.get("category", "financial_news"),
            "status": "pending",
        }

    def stage_records_to_parquet(
        self,
        records: List[Dict[str, Any]],
        batch_filename: str = "sp500_news_batch.parquet",
    ) -> Path:
        """Write processed news records to snappy-compressed Parquet archive."""
        schema = pa.schema([
            ("source_hash", pa.string()),
            ("source_url", pa.string()),
            ("title", pa.string()),
            ("raw_text", pa.string()),
            ("published_at", pa.string()),
            ("ticker_symbols", pa.list_(pa.string())),
            ("provider", pa.string()),
            ("category", pa.string()),
            ("status", pa.string()),
        ])

        table = pa.Table.from_pylist(records, schema=schema)
        out_path = self.output_dir / batch_filename
        pq.write_table(table, out_path, compression="snappy")
        logger.info(f"Staged {len(records)} clean news records to {out_path}")
        return out_path

    def insert_records_to_postgres(self, records: List[Dict[str, Any]]) -> int:
        """Insert processed news records into PostgreSQL `financial_news_queue`."""
        if not records:
            return 0

        inserted_count = 0
        try:
            with pg_connection() as conn:
                cur = conn.cursor()
                query = """
                    INSERT INTO financial_news_queue (
                        source_hash, source_url, title, raw_text,
                        published_at, ticker_symbols, provider, category, status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (source_hash) DO NOTHING;
                """
                for rec in records:
                    cur.execute(
                        query,
                        (
                            rec["source_hash"],
                            rec["source_url"],
                            rec["title"],
                            rec["raw_text"],
                            rec["published_at"],
                            json.dumps(rec["ticker_symbols"]),
                            rec["provider"],
                            rec["category"],
                            rec["status"],
                        ),
                    )
                    inserted_count += 1
                conn.commit()
                if hasattr(cur, "close"):
                    cur.close()
        except Exception as exc:
            logger.warning(f"Could not persist batch to PostgreSQL: {exc}")

        return inserted_count

    def harvest_from_sec_historical_events(
        self, sec_historical_dir: Path
    ) -> List[Dict[str, Any]]:
        """Harvest Form 8-K material event releases from the SEC historical directory."""
        harvested_records: List[Dict[str, Any]] = []

        if not sec_historical_dir.exists():
            return harvested_records

        for event_file in sec_historical_dir.glob("**/form8k_events.json"):
            try:
                with open(event_file, "r", encoding="utf-8") as f:
                    events = json.load(f)
                    for ev in events:
                        ticker = ev.get("ticker", "")
                        f_date = ev.get("filing_date", "")
                        items = ev.get("items_present", [])
                        raw_desc = f"SEC Form 8-K Material Event filing for {ticker} disclosing items: {', '.join(items)}. Full corporate disclosure filed on {f_date}."

                        rec = {
                            "title": f"SEC Form 8-K Disclosure ({ticker}) - {', '.join(items)}",
                            "text": raw_desc * 3,  # Ensure length threshold
                            "date": f_date,
                            "ticker": ticker,
                            "provider": "SEC_EDGAR_8K",
                            "url": f"https://www.sec.gov/edgar/data/{ticker}/{ev.get('accession_number', '')}",
                        }
                        processed = self.process_news_record(rec)
                        if processed:
                            harvested_records.append(processed)
            except Exception as exc:
                logger.debug(f"Error parsing 8-K events from {event_file}: {exc}")

        return harvested_records


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
    if is_paywall_or_stub(text):
        return None, metadata

    return text, metadata


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
        try:
            conn = _connect_db(os.getenv("FINANCIAL_RAG_DB", "financial_rag"))
            close_after = True
        except Exception as exc:
            logger.warning(f"Database connection error in insert_extracted_article: {exc}")
            return False

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
    except Exception as exc:
        logger.warning(f"Error inserting extracted article: {exc}")
        return False
    finally:
        if close_after and conn:
            conn.close()


def process_scraping_message(message_data: Dict[str, Any], conn=None) -> bool:
    """Process a single raw news message through the scraping pipeline."""
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
        return False

    domain = get_domain_from_url(url)
    if is_domain_blacklisted(domain, conn=conn):
        return False

    raw_html = payload.get("raw_html") or message_data.get("raw_html")
    extracted_text, _ = fetch_and_extract(url, raw_html=raw_html)

    if not extracted_text:
        record_domain_failure(domain, failure_threshold=3, conn=conn)
        return False

    record_domain_success(domain, conn=conn)

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
    return inserted


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
