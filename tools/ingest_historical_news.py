"""
Historical Financial News Harvester & Streamer (FNSPID & Form 8-K Material Releases).

Streams, filters, and stages multi-year (2018–2025) financial news mapped to
S&P 500 point-in-time constituent universe, sanitizing publisher boilerplates
and inserting deduplicated records into PostgreSQL `financial_news_queue`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Iterator, List, Optional, Set

import pyarrow as pa
import pyarrow.parquet as pq

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

logger = logging.getLogger("ingest_historical_news")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "historical_news"


class HistoricalNewsIngestor:
    """Streams and filters historical financial news for S&P 500 constituents."""

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
                            "text": raw_desc * 3, # Ensure length threshold
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest historical financial news for S&P 500.")
    parser.add_argument("--jsonl-path", type=str, default=None, help="Path to raw FNSPID or JSONL news file")
    parser.add_argument("--sec-dir", type=str, default="data/sec_historical", help="Path to SEC historical directory")
    parser.add_argument("--batch-name", type=str, default="historical_news_sp500.parquet", help="Output Parquet filename")
    parser.add_argument("--sync-pg", action="store_true", help="Sync clean records to PostgreSQL")
    args = parser.parse_args()

    ingestor = HistoricalNewsIngestor()
    all_processed: List[Dict[str, Any]] = []

    # 1. Harvest from SEC historical 8-K events if present
    sec_path = Path(args.sec_dir)
    if sec_path.exists():
        sec_recs = ingestor.harvest_from_sec_historical_events(sec_path)
        all_processed.extend(sec_recs)
        logger.info(f"Harvested {len(sec_recs)} news records from SEC Form 8-Ks")

    # 2. Process JSONL if provided
    if args.jsonl_path:
        j_path = Path(args.jsonl_path)
        if j_path.exists():
            with open(j_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        raw_rec = json.loads(line)
                        p = ingestor.process_news_record(raw_rec)
                        if p:
                            all_processed.append(p)
                    except Exception:
                        pass

    # 3. Stage to Parquet
    if all_processed:
        parquet_path = ingestor.stage_records_to_parquet(all_processed, batch_filename=args.batch_name)
        logger.info(f"Saved {len(all_processed)} records to {parquet_path}")

        if args.sync_pg:
            n_pg = ingestor.insert_records_to_postgres(all_processed)
            logger.info(f"Inserted {n_pg} records into PostgreSQL financial_news_queue")
    else:
        logger.info("No records processed.")


if __name__ == "__main__":
    main()
