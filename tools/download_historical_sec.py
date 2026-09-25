"""
Multi-Year SEC Historical Batch Harvester for S&P 500 Constituents (2018-2025+).

Harvests Form 10-K (Item 1 Business, Item 1A Risks, Exhibit 21 Subsidiaries),
Form 10-Q, Form 8-K (Material Events), and Form 4 (Insider Transactions)
for point-in-time S&P 500 constituents, stages clean text into data/sec_historical/,
and tracks datasets with DVC.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    env_file = PROJECT_ROOT / ".env"
    if env_file.is_file():
        load_dotenv(dotenv_path=env_file)
except ImportError:
    pass

from ingest.edgar_client import EdgarClient
from tools.sp500_universe import SP500Constituent, SP500UniverseManager

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s", "level":"%(levelname)s", "msg":"%(message)s"}'
)
logger = logging.getLogger("sec_historical_harvester")

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "sec_historical"


class HistoricalSECHarvester:
    """Batch harvester for historical S&P 500 SEC filings with rate limiting and section extraction."""

    def __init__(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        client: Optional[EdgarClient] = None,
        rate_limit_delay_sec: float = 0.1,  # 10 req/s compliance
    ):
        self.output_dir = Path(output_dir)
        self.client = client or EdgarClient()
        self.universe_manager = SP500UniverseManager()
        self.rate_limit_delay = rate_limit_delay_sec

    def harvest_company_year(
        self,
        constituent: SP500Constituent,
        year: int,
        forms: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Harvest all requested filings for a specific company and fiscal year."""
        forms_to_fetch = [f.upper() for f in (forms or ["10-K", "8-K", "10-Q", "4"])]
        ticker = constituent.ticker.upper()
        company_year_dir = self.output_dir / ticker / str(year)
        company_year_dir.mkdir(parents=True, exist_ok=True)

        stats: Dict[str, Any] = {
            "ticker": ticker,
            "cik": constituent.cik,
            "company_name": constituent.company_name,
            "gics_sector": constituent.gics_sector,
            "year": year,
            "10k_downloaded": False,
            "10q_count": 0,
            "8k_count": 0,
            "form4_count": 0,
            "subsidiaries_count": 0,
            "filings_saved": [],
        }

        # 1. Form 10-K
        if "10-K" in forms_to_fetch:
            time.sleep(self.rate_limit_delay)
            filing_10k = self.client.fetch_latest_10k(ticker, year=year)
            if filing_10k:
                sections = self.client.extract_10k_sections(filing_10k)
                acc_num = sections.get("accession_number", "UNKNOWN")
                acc_dir = company_year_dir / f"10K_{acc_num}"
                acc_dir.mkdir(parents=True, exist_ok=True)

                # Save 10-K metadata
                meta = {
                    "ticker": ticker,
                    "cik": constituent.cik,
                    "company_name": constituent.company_name,
                    "gics_sector": constituent.gics_sector,
                    "form": "10-K",
                    "year": year,
                    "accession_number": acc_num,
                    "filing_date": sections.get("filing_date"),
                    "report_date": sections.get("report_date"),
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
                if subs:
                    with open(acc_dir / "subsidiaries.json", "w", encoding="utf-8") as f:
                        json.dump(subs, f, indent=2)

                stats["10k_downloaded"] = True
                stats["subsidiaries_count"] = len(subs)
                stats["filings_saved"].append(f"10K_{acc_num}")

        # 2. Form 10-Q
        if "10-Q" in forms_to_fetch:
            time.sleep(self.rate_limit_delay)
            filings_10q = self.client.fetch_10q_filings(ticker, year=year, limit=4)
            for q_filing in filings_10q:
                acc_num = q_filing.get("accession_number", "UNKNOWN")
                acc_dir = company_year_dir / f"10Q_{acc_num}"
                acc_dir.mkdir(parents=True, exist_ok=True)

                with open(acc_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(q_filing, f, indent=2)

                q_text = q_filing.get("text", "")
                if q_text:
                    with open(acc_dir / "quarterly_text.txt", "w", encoding="utf-8") as f:
                        f.write(q_text)

                stats["10q_count"] += 1
                stats["filings_saved"].append(f"10Q_{acc_num}")

        # 3. Form 8-K
        if "8-K" in forms_to_fetch:
            time.sleep(self.rate_limit_delay)
            filings_8k = self.client.fetch_recent_8k(ticker, limit=5, year=year)
            for k_filing in filings_8k:
                acc_num = k_filing.get("accession_number", "UNKNOWN")
                acc_dir = company_year_dir / f"8K_{acc_num}"
                acc_dir.mkdir(parents=True, exist_ok=True)

                with open(acc_dir / "form8k_events.json", "w", encoding="utf-8") as f:
                    json.dump(k_filing, f, indent=2)

                stats["8k_count"] += 1
                stats["filings_saved"].append(f"8K_{acc_num}")

        # 4. Form 4
        if "4" in forms_to_fetch or "FORM4" in forms_to_fetch:
            time.sleep(self.rate_limit_delay)
            filings_4 = self.client.fetch_form4_transactions(ticker, limit=10, year=year)
            if filings_4:
                acc_dir = company_year_dir / "Form4_Transactions"
                acc_dir.mkdir(parents=True, exist_ok=True)
                with open(acc_dir / "form4_transactions.json", "w", encoding="utf-8") as f:
                    json.dump(filings_4, f, indent=2)
                stats["form4_count"] = len(filings_4)
                stats["filings_saved"].append("Form4_Transactions")

        # Write top-level company year summary
        with open(company_year_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        return stats

    def harvest_year(
        self,
        year: int,
        tickers: Optional[List[str]] = None,
        forms: Optional[List[str]] = None,
        max_companies: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Harvest filings for all point-in-time S&P 500 constituents in a given year."""
        logger.info(f"=== Starting S&P 500 SEC Historical Harvest for Year {year} ===")
        # Union constituents active across start, mid, or end of year to prevent intra-year turnover bias
        constituent_map: Dict[str, SP500Constituent] = {}
        for anchor_date in [f"{year}-01-01", f"{year}-06-30", f"{year}-12-31"]:
            for c in self.universe_manager.get_constituents_at_date(anchor_date):
                constituent_map[c.ticker.upper()] = c
        constituents = list(constituent_map.values())

        if tickers:
            ticker_set = {t.upper() for t in tickers}
            constituents = [c for c in constituents if c.ticker.upper() in ticker_set]

        if max_companies:
            constituents = constituents[:max_companies]

        logger.info(f"Targeting {len(constituents)} unique constituents active at any point in year {year}")

        results = []
        for i, c in enumerate(constituents, 1):
            logger.info(f"[{i}/{len(constituents)}] Harvesting {c.ticker} ({c.company_name}) [{c.gics_sector}] for {year}...")
            res = self.harvest_company_year(c, year, forms=forms)
            results.append(res)

        return results

    def harvest_multi_year(
        self,
        years: List[int],
        tickers: Optional[List[str]] = None,
        forms: Optional[List[str]] = None,
        max_companies: Optional[int] = None,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Sweep multiple fiscal years across historical S&P 500 constituents."""
        all_year_results = {}
        for yr in sorted(years):
            res = self.harvest_year(yr, tickers=tickers, forms=forms, max_companies=max_companies)
            all_year_results[yr] = res
        return all_year_results

    def register_dvc(self) -> None:
        """Track harvested historical SEC filings with DVC."""
        try:
            logger.info("Registering data/sec_historical with DVC...")
            subprocess.run(
                [sys.executable, "-m", "dvc", "add", str(self.output_dir)],
                cwd=str(PROJECT_ROOT),
                check=True,
                capture_output=True,
            )
            logger.info("Successfully tracked data/sec_historical with DVC")
        except Exception as e:
            logger.warning(f"DVC registration skipped or failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Year SEC Historical Batch Harvester (S&P 500)")
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024], help="Fiscal years to harvest (e.g. 2020 2021 2022 2023 2024)")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to harvest (default: all S&P 500)")
    parser.add_argument("--forms", nargs="+", default=["10-K", "8-K", "10-Q", "4"], help="Form types to harvest")
    parser.add_argument("--max-companies", type=int, help="Limit number of companies per year (for pilot testing)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for harvested filings")
    parser.add_argument("--dvc", action="store_true", help="Auto-register output with DVC after harvesting")
    args = parser.parse_args()

    harvester = HistoricalSECHarvester(output_dir=Path(args.output_dir))
    results = harvester.harvest_multi_year(
        years=args.years,
        tickers=args.tickers,
        forms=args.forms,
        max_companies=args.max_companies,
    )

    total_10k = sum(sum(1 for c in yr_res if c["10k_downloaded"]) for yr_res in results.values())
    total_8k = sum(sum(c["8k_count"] for c in yr_res) for yr_res in results.values())
    total_10q = sum(sum(c["10q_count"] for c in yr_res) for yr_res in results.values())
    total_subs = sum(sum(c["subsidiaries_count"] for c in yr_res) for yr_res in results.values())

    print("\n============================================================")
    print("S&P 500 SEC Historical Harvest Summary:")
    print(f"  Years Processed   : {args.years}")
    print(f"  Form 10-Ks Staged : {total_10k}")
    print(f"  Form 10-Qs Staged : {total_10q}")
    print(f"  Form 8-Ks Staged  : {total_8k}")
    print(f"  Subsidiaries Found: {total_subs}")
    print(f"  Output Directory  : {args.output_dir}")
    print("============================================================\n")

    if args.dvc:
        harvester.register_dvc()


if __name__ == "__main__":
    main()
