"""
SEC EDGAR Client Wrapper using `edgartools`.

Provides standardized methods to:
- Authenticate / set identity with SEC EDGAR
- Fetch 10-K, 8-K, Form 4, and 13F filings
- Extract structured sections (Item 1 Business, Item 1A Risk Factors, Exhibit 21 Subsidiaries)
- Parse Exhibit 21 list of subsidiaries into structured records
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional

try:
    from edgar import Company, Filing, get_filings, set_identity
    EDGAR_AVAILABLE = True
except ImportError:
    EDGAR_AVAILABLE = False

logger = logging.getLogger("edgar_client")

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
        """Extract structured sections from a 10-K filing using edgartools TenK parser.
        
        Extracts:
        - item_1 (Business description, key suppliers/customers/partnerships)
        - item_1a (Risk Factors)
        - exhibit_21_subsidiaries (Structured list of subsidiary names & jurisdictions)
        """
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
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines:
            # Skip obvious header rows
            if re.search(r"\b(name|state|jurisdiction|country|incorporation|subsidiary)\b", line, re.I) and len(line) < 60:
                continue
            # Simple heuristic splitting by multiple spaces or tabs
            parts = re.split(r"\t+|\s{3,}", line)
            if len(parts) >= 2:
                name = parts[0].strip()
                jurisdiction = parts[1].strip()
                if len(name) > 2 and len(name) < 150:
                    results.append({"name": name, "jurisdiction": jurisdiction})
            elif len(parts) == 1 and len(parts[0]) > 3 and len(parts[0]) < 120:
                results.append({"name": parts[0].strip(), "jurisdiction": "Unknown"})
        return results

    def fetch_recent_8k(self, ticker_or_cik: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch recent 8-K material event filings."""
        company = self.get_company(ticker_or_cik)
        if not company:
            return []

        results = []
        try:
            filings = company.get_filings(form="8-K")
            if not filings:
                return []
            for filing in filings[:limit]:
                results.append({
                    "accession_number": getattr(filing, "accession_number", getattr(filing, "accession_no", "UNKNOWN")),
                    "filing_date": str(getattr(filing, "filing_date", "")),
                    "form": "8-K",
                    "items": getattr(filing, "items", []),
                    "text": filing.text()[:10000] if hasattr(filing, "text") else "",
                })
        except Exception as e:
            logger.error(f"Error fetching 8-K for '{ticker_or_cik}': {e}")
        return results

    def fetch_form4_transactions(self, ticker_or_cik: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch Form 4 insider transactions."""
        company = self.get_company(ticker_or_cik)
        if not company:
            return []

        results = []
        try:
            filings = company.get_filings(form="4")
            if not filings:
                return []
            for filing in filings[:limit]:
                results.append({
                    "accession_number": getattr(filing, "accession_number", getattr(filing, "accession_no", "UNKNOWN")),
                    "filing_date": str(getattr(filing, "filing_date", "")),
                    "form": "4",
                })
        except Exception as e:
            logger.error(f"Error fetching Form 4 for '{ticker_or_cik}': {e}")
        return results
