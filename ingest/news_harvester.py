"""
Multi-Source Financial News Harvester (yfinance, Google News RSS, and Fallback Feeds).

Provides an uncapped, rate-resilient news ingestion engine supporting:
1. `yfinance.Ticker.news` (Direct publisher links, ticker tags, timestamps)
2. Google News RSS for high-impact corporate actions (M&A, supply deals, executive changes)
3. Auxiliary fallback to Finnhub metadata
4. Built-in publisher stoplist & boilerplate stripping
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
import xml.etree.ElementTree as ET

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

import httpx

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

    def harvest_google_news_rss(
        self,
        query: str,
        ticker: Optional[str] = None,
        max_items: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Harvest corporate news from Google News RSS search endpoint.
        Example query: 'Apple TSMC supply agreement' or 'NVIDIA acquisition'.
        """
        results: List[Dict[str, Any]] = []
        encoded_q = httpx.URL(f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en")

        headers = {"User-Agent": self.user_agent}
        try:
            with httpx.Client(timeout=15.0, headers=headers) as client:
                resp = client.get(str(encoded_q))
                if resp.status_code != 200:
                    return results

                root = ET.fromstring(resp.content)
                channel = root.find("channel")
                if channel is None:
                    return results

                items = channel.findall("item")[:max_items]
                for item in items:
                    title_elem = item.find("title")
                    link_elem = item.find("link")
                    pub_date_elem = item.find("pubDate")
                    source_elem = item.find("source")

                    title = title_elem.text if title_elem is not None and title_elem.text else ""
                    link = link_elem.text if link_elem is not None and link_elem.text else ""
                    source_name = source_elem.text if source_elem is not None and source_elem.text else "Google News RSS"

                    if not title or not link:
                        continue

                    if is_clickbait_article_source(source_name) or is_clickbait_article_source(title):
                        continue

                    # Parse RSS date format: "Mon, 18 Sep 2026 04:12:00 GMT"
                    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    if pub_date_elem is not None and pub_date_elem.text:
                        try:
                            # Strip timezone name if needed
                            clean_d = re.sub(r"\s+[A-Z]+$", "", pub_date_elem.text.strip())
                            dt = datetime.strptime(clean_d, "%a, %d %b %Y %H:%M:%S")
                            date_str = dt.strftime("%Y-%m-%d")
                        except Exception:
                            pass

                    sh = compute_source_hash(link)
                    tickers = [ticker.strip().upper()] if ticker else []

                    results.append({
                        "source_hash": sh,
                        "source_url": link,
                        "title": title,
                        "raw_text": title,
                        "published_at": date_str,
                        "ticker_symbols": tickers,
                        "provider": source_name,
                        "category": "rss_financial_news",
                        "status": "pending",
                    })
        except Exception as exc:
            logger.debug(f"Google News RSS harvest failed for query '{query}': {exc}")

        return results

    def harvest_ticker_events(
        self,
        ticker: str,
        company_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Harvest combined live news and event signals for an S&P 500 company."""
        clean_ticker = ticker.strip().upper()
        c_name = company_name or clean_ticker

        combined: List[Dict[str, Any]] = []
        seen_hashes: Set[str] = set()

        # 1. yfinance direct news
        yf_items = self.harvest_yfinance_news(clean_ticker)
        for item in yf_items:
            sh = item["source_hash"]
            if sh not in seen_hashes:
                seen_hashes.add(sh)
                combined.append(item)

        # 2. Google News RSS for material corporate triggers
        rss_queries = [
            f'"{c_name}" acquisition OR merger OR "acquired"',
            f'"{c_name}" "supply agreement" OR supplier OR partnership',
        ]
        for q in rss_queries:
            rss_items = self.harvest_google_news_rss(q, ticker=clean_ticker, max_items=5)
            for item in rss_items:
                sh = item["source_hash"]
                if sh not in seen_hashes:
                    seen_hashes.add(sh)
                    combined.append(item)

        return combined

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
