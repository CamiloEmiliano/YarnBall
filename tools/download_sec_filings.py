"""
SEC EDGAR Filings Downloader and Local Staging Utility.

Fetches 10-K (Item 1 Business, Item 1A Risk Factors, Exhibit 21 Subsidiaries),
8-K, and Form 4 filings for target companies, saves raw text and structured metadata
into `data/sec_filings/<TICKER>/<ACCESSION>/`, and registers data with DVC.
"""

import argparse
import json
import logging
import os
import subprocess
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

from ingest.edgar_client import EdgarClient, DEFAULT_SP500_BENCHMARK

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s", "level":"%(levelname)s", "msg":"%(message)s"}'
)
logger = logging.getLogger("sec_downloader")


def download_filings_for_ticker(
    client: EdgarClient,
    ticker: str,
    output_base_dir: Path,
    year: Optional[int] = None,
    include_8k: bool = False,
    include_form4: bool = False,
) -> dict:
    """Download and stage 10-K, 8-K, and Form 4 filings for a single ticker."""
    clean_ticker = ticker.strip().upper()
    logger.info(f"Downloading filings for {clean_ticker}...")

    stats = {
        "ticker": clean_ticker,
        "10k_downloaded": False,
        "8k_count": 0,
        "form4_count": 0,
        "accession_number": None,
        "subsidiaries_count": 0,
    }

    ticker_dir = output_base_dir / clean_ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch Form 10-K
    filing_10k = client.fetch_latest_10k(clean_ticker, year=year)
    if filing_10k:
        sections = client.extract_10k_sections(filing_10k)
        acc_num = sections.get("accession_number", "UNKNOWN")
        stats["accession_number"] = acc_num
        stats["10k_downloaded"] = True

        acc_dir = ticker_dir / f"10K_{acc_num}"
        acc_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        meta = {
            "ticker": clean_ticker,
            "accession_number": acc_num,
            "form": "10-K",
            "filing_date": sections.get("filing_date"),
            "report_date": sections.get("report_date"),
            "company_name": getattr(filing_10k, "company", clean_ticker),
            "cik": str(getattr(filing_10k, "cik", "")).zfill(10),
            "has_item_1": bool(sections.get("item_1_business")),
            "has_item_1a": bool(sections.get("item_1a_risk_factors")),
            "subsidiaries_count": len(sections.get("subsidiaries", [])),
        }
        with open(acc_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Save Item 1 (Business)
        item1 = sections.get("item_1_business", "")
        if item1:
            with open(acc_dir / "item_1_business.txt", "w", encoding="utf-8") as f:
                f.write(item1)

        # Save Item 1A (Risk Factors)
        item1a = sections.get("item_1a_risk_factors", "")
        if item1a:
            with open(acc_dir / "item_1a_risk_factors.txt", "w", encoding="utf-8") as f:
                f.write(item1a)

        # Save Exhibit 21 (Subsidiaries)
        subs = sections.get("subsidiaries", [])
        stats["subsidiaries_count"] = len(subs)
        with open(acc_dir / "subsidiaries.json", "w", encoding="utf-8") as f:
            json.dump(subs, f, indent=2)

        logger.info(
            f"Successfully staged 10-K for {clean_ticker} ({acc_num}): "
            f"Item 1 ({len(item1)} chars), Item 1A ({len(item1a)} chars), {len(subs)} subsidiaries"
        )
    else:
        logger.warning(f"No 10-K filing available for {clean_ticker}")

    # 2. Optionally fetch 8-K filings
    if include_8k:
        events = client.fetch_recent_8k(clean_ticker, limit=3)
        stats["8k_count"] = len(events)
        for event in events:
            ev_acc = event.get("accession_number", "UNKNOWN")
            ev_dir = ticker_dir / f"8K_{ev_acc}"
            ev_dir.mkdir(parents=True, exist_ok=True)
            with open(ev_dir / "event.json", "w", encoding="utf-8") as f:
                json.dump(event, f, indent=2)

    return stats


def track_with_dvc(data_dir: Path) -> bool:
    """Register data directory with DVC."""
    try:
        dvc_bin = sys.executable.replace("python", "dvc")
        if not os.path.exists(dvc_bin):
            dvc_bin = "dvc"

        logger.info(f"Adding {data_dir} to DVC tracking...")
        res = subprocess.run(
            [dvc_bin, "add", str(data_dir)],
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            logger.info(f"DVC successfully tracked {data_dir}. Output:\n{res.stdout}")
            return True
        else:
            logger.warning(f"DVC add failed: {res.stderr}")
            return False
    except Exception as e:
        logger.warning(f"Could not execute DVC tracking: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download and stage SEC EDGAR filings.")
    parser.add_argument(
        "--tickers",
        type=str,
        default="AAPL,MSFT,NVDA,AMZN,GOOGL",
        help="Comma-separated ticker list (or 'SP500_TOP20', 'SP500_ALL')",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Fiscal year to download (default: latest)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sec_filings",
        help="Local staging directory for raw filings",
    )
    parser.add_argument(
        "--include-8k",
        action="store_true",
        help="Include recent Form 8-K material events",
    )
    parser.add_argument(
        "--no-dvc",
        action="store_true",
        help="Skip automatic DVC tracking",
    )

    args = parser.parse_args()

    # Parse ticker selection
    if args.tickers == "SP500_TOP20":
        tickers = DEFAULT_SP500_BENCHMARK[:20]
    elif args.tickers == "SP500_ALL":
        tickers = DEFAULT_SP500_BENCHMARK
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    output_path = ROOT_DIR / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    client = EdgarClient()

    results = []
    for ticker in tickers:
        try:
            res = download_filings_for_ticker(
                client=client,
                ticker=ticker,
                output_base_dir=output_path,
                year=args.year,
                include_8k=args.include_8k,
            )
            results.append(res)
        except Exception as e:
            logger.error(f"Failed to process ticker {ticker}: {e}")

    # Summary table
    logger.info("=== SEC EDGAR DOWNLOAD SUMMARY ===")
    for r in results:
        status = "OK" if r.get("10k_downloaded") else "FAILED"
        logger.info(
            f"Ticker: {r['ticker']:<6} | Status: {status:<6} | Acc: {r.get('accession_number')} | "
            f"Subsidiaries: {r.get('subsidiaries_count', 0)}"
        )

    # Track with DVC
    if not args.no_dvc and results:
        track_with_dvc(output_path)


if __name__ == "__main__":
    main()
