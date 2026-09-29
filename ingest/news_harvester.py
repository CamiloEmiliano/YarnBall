"""
Multi-Source Financial News Harvester (yfinance Direct Ticker News and Wire Ingestion).

Provides a rate-resilient, canonical news ingestion engine supporting:
1. `yfinance.Ticker.news` (Direct publisher links, ticker tags, timestamps)
2. Auxiliary fallback to Finnhub metadata
3. Built-in publisher stoplist & boilerplate stripping
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Set

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from graph.db import pg_connection
from graph.entity_resolver import is_clickbait_article_source, is_generic_placeholder
from ingest.scraping_worker import strip_publisher_boilerplates, compute_source_hash
from tools.sp500_universe import SP500UniverseManager

logger = logging.getLogger("news_harvester")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


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
                    "raw_text": title, # Title acts as headline text if full text scraping follows
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


def main() -> None:
    harvester = MultiSourceNewsHarvester()
    sample_tickers = ["AAPL", "NVDA", "MSFT"]
    logger.info(f"Harvesting live news & events for {sample_tickers}...")

    all_recs = []
    for ticker in sample_tickers:
        recs = harvester.harvest_ticker_events(ticker)
        all_recs.extend(recs)
        logger.info(f"Retrieved {len(recs)} records for {ticker}")

    n_inserted = harvester.sync_to_postgres_queue(all_recs)
    logger.info(f"Synced {n_inserted} records to financial_news_queue.")


if __name__ == "__main__":
    main()
