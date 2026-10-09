"""
SEC EDGAR Client Wrapper using `edgartools`.

Provides standardized methods to:
- Authenticate / set identity with SEC EDGAR
- Fetch 10-K, 10-Q, 8-K, Form 4, and 13F filings
- Extract structured sections (Item 1 Business, Item 1A Risk Factors, Exhibit 21 Subsidiaries)
- Parse Exhibit 21 list of subsidiaries into structured records
- CLI utility for targeted filing downloads
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

try:
    from edgar import Company, Filing, get_filings, set_identity
    EDGAR_AVAILABLE = True
except ImportError:
    Company = None
    Filing = None
    get_filings = None
    set_identity = None
    EDGAR_AVAILABLE = False

logger = logging.getLogger("ingestion.sec.client")

# Default S&P 500 benchmark seed tickers for testing and focused graph construction
DEFAULT_SP500_BENCHMARK = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK.B", "UNH", "JNJ",
    "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP", "KO",
    "BAC", "COST", "AVGO", "TMO", "CSCO", "MCD", "WMT", "ABT", "ACN", "DIS",
    "LIN", "ADBE", "NFLX", "TXN", "PM", "AMD", "QCOM", "VZ", "CRM", "NKE",
    "INTC", "BMY", "WFC", "COP", "RTX", "HON", "IBM", "AMGN", "GE", "CAT"
]


class EdgarClient:
    """Wrapper around `edgartools` with error handling, rate limiting, and section extraction."""

    def __init__(self, user_agent: Optional[str] = None):
        self.user_agent = user_agent or os.getenv(
            "SEC_EDGAR_USER_AGENT", "YarnBallResearch admin@yarnball.org"
        )
        if EDGAR_AVAILABLE:
            try:
                set_identity(self.user_agent)
                logger.info(f"Initialized SEC Edgar identity: {self.user_agent}")
            except Exception as e:
                logger.warning(f"Could not set edgartools identity: {e}")
        else:
            logger.warning("edgartools is not installed; EdgarClient will operate in stub mode.")

    def get_company(self, ticker_or_cik: str) -> Optional[Any]:
        """Fetch Company object via edgartools."""
        if not EDGAR_AVAILABLE:
            return None
        try:
            return Company(ticker_or_cik)
        except Exception as e:
            logger.error(f"Failed to load SEC Company for '{ticker_or_cik}': {e}")
            return None

    def fetch_latest_10k(self, ticker_or_cik: str, year: Optional[int] = None) -> Optional[Any]:
        """Retrieve the latest or year-specific 10-K filing."""
        company = self.get_company(ticker_or_cik)
        if not company:
            return None

        try:
            filings = company.get_filings(form="10-K")
            if not filings:
                return None
            if year:
                # Filter filings by filing date / fiscal year
                for filing in filings:
                    if str(year) in str(filing.filing_date) or str(year) in str(filing.report_date):
                        return filing
            return filings[0]
        except Exception as e:
            logger.error(f"Error fetching 10-K for '{ticker_or_cik}': {e}")
            return None

    def extract_10k_sections(self, filing: Any) -> Dict[str, Any]:
        """Extract structured sections from a 10-K filing using edgartools TenK parser."""
        result = {
            "accession_number": getattr(filing, "accession_number", getattr(filing, "accession_no", "UNKNOWN")),
            "filing_date": str(getattr(filing, "filing_date", "")),
            "report_date": str(getattr(filing, "report_date", "")),
            "form": "10-K",
            "item_1_business": "",
            "item_1a_risk_factors": "",
            "subsidiaries": [],
        }

        if not filing:
            return result

        try:
            tenk = filing.obj()
            if tenk:
                # Extract Item 1 (Business)
                try:
                    if hasattr(tenk, "item_1") and tenk.item_1:
                        result["item_1_business"] = str(tenk.item_1)
                    elif hasattr(tenk, "__getitem__"):
                        item1 = tenk["Item 1"] or tenk["Item 1."]
                        if item1:
                            result["item_1_business"] = str(item1)
                except Exception as e:
                    logger.debug(f"Item 1 extraction fallback: {e}")

                # Extract Item 1A (Risk Factors)
                try:
                    if hasattr(tenk, "item_1a") and tenk.item_1a:
                        result["item_1a_risk_factors"] = str(tenk.item_1a)
                    elif hasattr(tenk, "__getitem__"):
                        item1a = tenk["Item 1A"] or tenk["Item 1A."]
                        if item1a:
                            result["item_1a_risk_factors"] = str(item1a)
                except Exception as e:
                    logger.debug(f"Item 1A extraction fallback: {e}")

            # Extract Exhibit 21 (Subsidiaries)
            result["subsidiaries"] = self.extract_exhibit_21_subsidiaries(filing)

        except Exception as e:
            logger.warning(f"Error extracting sections from 10-K filing: {e}")

        return result

    def extract_exhibit_21_subsidiaries(self, filing: Any) -> List[Dict[str, str]]:
        """Find and parse Exhibit 21 attachments from filing."""
        subsidiaries = []
        if not filing:
            return subsidiaries

        try:
            attachments = getattr(filing, "attachments", None)
            if attachments:
                for att in attachments:
                    desc = str(getattr(att, "description", "")).lower()
                    doc_type = str(getattr(att, "document_type", "")).upper()
                    if "EX-21" in doc_type or "exhibit 21" in desc or "subsidiaries" in desc:
                        content = att.text() if hasattr(att, "text") else ""
                        if content:
                            subsidiaries.extend(self._parse_subsidiaries_text(content))
        except Exception as e:
            logger.debug(f"Exhibit 21 extraction error: {e}")

        return subsidiaries

    def _parse_subsidiaries_text(self, text: str) -> List[Dict[str, str]]:
        """Parse raw text/HTML table of subsidiaries into names and jurisdictions."""
        results = []
        boilerplate_pattern = re.compile(
            r"(exhibit\s*21|subsidiaries\s+of|following\s+is\s+a\s+list|omitting\s+subsidiaries|"
            r"considered\s+in\s+the\s+aggregate|significant\s+subsidiary|jurisdiction\s+of|"
            r"state\s+or\s+other|percent\s+owned|all\s+100%|item\s+\d+|table\s+of\s+contents|"
            r"^\s*owned\)?\s*$)",
            re.IGNORECASE,
        )

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines:
            # Skip boilerplate and header rows
            if boilerplate_pattern.search(line):
                continue
            if re.search(r"\b(name|state|jurisdiction|country|incorporation|subsidiary)\b", line, re.I) and len(line) < 60:
                continue

            # Split by tabs or 2+ spaces
            parts = re.split(r"\t+|\s{2,}", line)
            if len(parts) >= 2:
                name = parts[0].strip(" -:;,")
                jurisdiction = parts[1].strip(" -:;,")
                if len(name) >= 3 and len(name) < 120 and not boilerplate_pattern.search(name):
                    results.append({"name": name, "jurisdiction": jurisdiction or "Unknown"})
            elif len(parts) == 1:
                name = parts[0].strip(" -:;,")
                if len(name) >= 4 and len(name) < 100 and " " in name and not boilerplate_pattern.search(name):
                    results.append({"name": name, "jurisdiction": "Unknown"})
        return results

    def fetch_recent_8k(self, ticker_or_cik: str, limit: int = 5, year: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch recent or year-specific 8-K material event filings."""
        company = self.get_company(ticker_or_cik)
        if not company:
            return []

        results = []
        try:
            filings = company.get_filings(form="8-K")
            if not filings:
                return []
            count = 0
            for filing in filings:
                filing_date_str = str(getattr(filing, "filing_date", ""))
                if year and str(year) not in filing_date_str:
                    continue
                results.append({
                    "accession_number": getattr(filing, "accession_number", getattr(filing, "accession_no", "UNKNOWN")),
                    "filing_date": filing_date_str,
                    "form": "8-K",
                    "items": getattr(filing, "items", []),
                    "text": filing.text()[:10000] if hasattr(filing, "text") else "",
                })
                count += 1
                if count >= limit:
                    break
        except Exception as e:
            logger.error(f"Error fetching 8-K for '{ticker_or_cik}': {e}")
        return results

    def fetch_10q_filings(self, ticker_or_cik: str, year: Optional[int] = None, limit: int = 4) -> List[Dict[str, Any]]:
        """Fetch Form 10-Q quarterly reports for a company."""
        company = self.get_company(ticker_or_cik)
        if not company:
            return []

        results = []
        try:
            filings = company.get_filings(form="10-Q")
            if not filings:
                return []
            count = 0
            for filing in filings:
                filing_date_str = str(getattr(filing, "filing_date", ""))
                if year and str(year) not in filing_date_str:
                    continue
                results.append({
                    "accession_number": getattr(filing, "accession_number", getattr(filing, "accession_no", "UNKNOWN")),
                    "filing_date": filing_date_str,
                    "report_date": str(getattr(filing, "report_date", "")),
                    "form": "10-Q",
                    "text": filing.text()[:15000] if hasattr(filing, "text") else "",
                })
                count += 1
                if count >= limit:
                    break
        except Exception as e:
            logger.error(f"Error fetching 10-Q for '{ticker_or_cik}': {e}")
        return results

    def fetch_form4_transactions(self, ticker_or_cik: str, limit: int = 20, year: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch Form 4 insider transactions."""
        company = self.get_company(ticker_or_cik)
        if not company:
            return []

        results = []
        try:
            filings = company.get_filings(form="4")
            if not filings:
                return []
            count = 0
            for filing in filings:
                filing_date_str = str(getattr(filing, "filing_date", ""))
                if year and str(year) not in filing_date_str:
                    continue
                results.append({
                    "accession_number": getattr(filing, "accession_number", getattr(filing, "accession_no", "UNKNOWN")),
                    "filing_date": filing_date_str,
                    "form": "4",
                })
                count += 1
                if count >= limit:
                    break
        except Exception as e:
            logger.error(f"Error fetching Form 4 for '{ticker_or_cik}': {e}")
        return results


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

    # 1. Fetch 10-K
    tenk_filing = client.fetch_latest_10k(clean_ticker, year=year)
    if tenk_filing:
        sections = client.extract_10k_sections(tenk_filing)
        accession = sections.get("accession_number", "UNKNOWN").replace("-", "")
        stats["accession_number"] = accession
        stats["10k_downloaded"] = True
        stats["subsidiaries_count"] = len(sections.get("subsidiaries", []))

        filing_dir = ticker_dir / accession
        filing_dir.mkdir(parents=True, exist_ok=True)

        meta_path = filing_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "ticker": clean_ticker,
                "accession_number": sections.get("accession_number"),
                "filing_date": sections.get("filing_date"),
                "report_date": sections.get("report_date"),
                "form": "10-K",
                "subsidiaries_count": stats["subsidiaries_count"],
            }, f, indent=2)

        if sections.get("item_1_business"):
            with open(filing_dir / "item_1_business.txt", "w", encoding="utf-8") as f:
                f.write(sections["item_1_business"])

        if sections.get("item_1a_risk_factors"):
            with open(filing_dir / "item_1a_risk_factors.txt", "w", encoding="utf-8") as f:
                f.write(sections["item_1a_risk_factors"])

        if sections.get("subsidiaries"):
            with open(filing_dir / "exhibit_21_subsidiaries.json", "w", encoding="utf-8") as f:
                json.dump(sections["subsidiaries"], f, indent=2)

        logger.info(f"[{clean_ticker}] 10-K saved to {filing_dir} ({stats['subsidiaries_count']} subsidiaries)")
    else:
        logger.warning(f"[{clean_ticker}] No 10-K filing found for year={year}")

    # 2. Optionally fetch Form 8-K
    if include_8k:
        eight_ks = client.fetch_recent_8k(clean_ticker, limit=5, year=year)
        stats["8k_count"] = len(eight_ks)
        if eight_ks:
            eightk_dir = ticker_dir / "8K"
            eightk_dir.mkdir(parents=True, exist_ok=True)
            for e in eight_ks:
                acc = e.get("accession_number", "unknown").replace("-", "")
                with open(eightk_dir / f"{acc}.json", "w", encoding="utf-8") as f:
                    json.dump(e, f, indent=2)

    # 3. Optionally fetch Form 4
    if include_form4:
        form4s = client.fetch_form4_transactions(clean_ticker, limit=10, year=year)
        stats["form4_count"] = len(form4s)
        if form4s:
            form4_dir = ticker_dir / "FORM4"
            form4_dir.mkdir(parents=True, exist_ok=True)
            with open(form4_dir / "transactions.json", "w", encoding="utf-8") as f:
                json.dump(form4s, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(description="SEC EDGAR Filings Downloader and Staging CLI.")
    parser.add_argument(
        "--tickers",
        type=str,
        default="AAPL,MSFT,NVDA",
        help="Comma-separated list of stock tickers or 'SP500' for benchmark seed",
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
        help="Local staging directory for downloaded filings",
    )
    parser.add_argument(
        "--include-8k",
        action="store_true",
        help="Include Form 8-K material event filings",
    )
    parser.add_argument(
        "--include-form4",
        action="store_true",
        help="Include Form 4 insider transaction filings",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.tickers.upper() == "SP500":
        tickers = DEFAULT_SP500_BENCHMARK
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    client = EdgarClient()
    for t in tickers:
        download_filings_for_ticker(
            client=client,
            ticker=t,
            output_base_dir=output_dir,
            year=args.year,
            include_8k=args.include_8k,
            include_form4=args.include_form4,
        )


if __name__ == "__main__":
    main()
