"""
SEC EDGAR Knowledge Graph Ingestion Pipeline.

Orchestrates:
1. Syncing official SEC company tickers into PostgreSQL `sec_companies`
2. Ingesting staged/live Form 10-K filings into Memgraph & PostgreSQL
3. Extracting supply-chain, competitor, risk, and subsidiary edges with temporal metadata
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

from graph.memgraph_driver import get_memgraph_driver
from tools.init_db import get_dsn, ensure_database, create_queue_table, _connect_db, _load_real_psycopg
from ingest.edgar_linker import sync_sec_companies_from_sec, EdgarEntityLinker
from ingest.edgar_worker import EdgarWorker
from ingest.edgar_client import DEFAULT_SP500_BENCHMARK


logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s", "level":"%(levelname)s", "msg":"%(message)s"}'
)
logger = logging.getLogger("sec_ingestion_runner")


def main():
    parser = argparse.ArgumentParser(description="Ingest SEC EDGAR filings into Knowledge Graph.")
    parser.add_argument(
        "--sync-cik",
        action="store_true",
        help="Sync master CIK company registry from SEC.gov into PostgreSQL",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="AAPL,MSFT,NVDA,AMZN,GOOGL",
        help="Comma-separated tickers to process into Memgraph",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize/verify PostgreSQL schema and extensions",
    )

    args = parser.parse_args()

    db_name = os.getenv("FINANCIAL_RAG_DB", "financial_rag")

    # 1. Initialize PostgreSQL schema if requested
    if args.init_db:
        logger.info("Initializing PostgreSQL schema and extensions...")
        try:
            ensure_database(db_name)
            create_queue_table(db_name)
            logger.info("PostgreSQL schema initialization complete.")
        except Exception as e:
            logger.warning(f"Database init warning (check if PostgreSQL is running): {e}")

    # 2. Connect to PostgreSQL
    pg_conn = None
    try:
        pg_conn = _connect_db(db_name)
        logger.info(f"Connected to PostgreSQL ({db_name})")
    except Exception as e:
        logger.warning(f"Could not connect to PostgreSQL: {e}. Running in memory-only / mock mode.")

    # 3. Sync Master CIK Registry if requested
    if args.sync_cik and pg_conn:
        logger.info("Syncing master SEC company registry from SEC.gov...")
        count = sync_sec_companies_from_sec(pg_conn, sp500_tickers=DEFAULT_SP500_BENCHMARK)
        logger.info(f"Synced {count} companies into sec_companies.")

    # 4. Process Target Filings into Memgraph
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    logger.info(f"Starting Knowledge Graph ingestion for tickers: {tickers}")

    try:
        driver = get_memgraph_driver()
        worker = EdgarWorker(pg_conn=pg_conn, memgraph_driver=driver)

        results = []
        for ticker in tickers:
            logger.info(f"Processing 10-K knowledge graph extraction for {ticker}...")
            res = worker.process_company_10k(ticker)
            results.append({"ticker": ticker, **res})

        logger.info("=== SEC EDGAR GRAPH INGESTION SUMMARY ===")
        for r in results:
            logger.info(
                f"Ticker: {r['ticker']:<6} | Status: {r.get('status')} | Acc: {r.get('accession_number')} | "
                f"Nodes: {r.get('nodes', 0)} | Edges: {r.get('edges', 0)}"
            )

    except Exception as e:
        logger.error(f"Error during graph ingestion: {e}")
    finally:
        if pg_conn:
            pg_conn.close()


if __name__ == "__main__":
    main()
