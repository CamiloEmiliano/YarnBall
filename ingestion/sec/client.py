"""
SEC EDGAR Client Wrapper using `edgartools`.

Provides standardized methods to:
- Authenticate / set identity with SEC EDGAR
- Fetch 10-K, 10-K/A, 10-Q, 10-Q/A, 8-K, 8-K/A, and opt-in Form 4 filings per `ingestion/sec/canon.py` (ADR-00045 / ADR-00046)
- Extract structured 10-K/10-K/A sections (Items 1, 1A, 1C, 2, 3, 7, Exhibit 21)
- Extract part-qualified 10-Q/10-Q/A narrative sections (Part I Items 2 & 3, Part II Items 1 & 1A)
- Extract attached Form 8-K/8-K/A Exhibit 99.x press releases independently bounded at `DEFAULT_MAX_FILING_CHARS`
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
from typing import Any, Dict, List, Optional, Tuple

from .canon import SECFormType, normalize_form_type
from .disruption_gating import DEFAULT_MAX_FILING_CHARS

try:
    os.environ.setdefault("EDGAR_LOCAL_DATA_DIR", "/tmp/edgar")
    Path("/tmp/edgar").mkdir(parents=True, exist_ok=True)
    from edgar import Company, Filing, get_filings, set_identity
    EDGAR_AVAILABLE = True
except (ImportError, OSError):
    Company = None  # type: ignore[assignment,misc,no-redef]
    Filing = None  # type: ignore[assignment,misc,no-redef]
    get_filings = None  # type: ignore[assignment,no-redef]
    set_identity = None  # type: ignore[assignment,no-redef]
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
        if EDGAR_AVAILABLE and self.user_agent:
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

    def fetch_10k_filings(
        self,
        ticker_or_cik: str,
        year: Optional[int] = None,
        *,
        form_type: str = SECFormType.FORM_10K.value,
        limit: int = 2,
    ) -> List[Any]:
        """Retrieve 10-K or 10-K/A filings for a company, optionally filtered by year (ADR-00046)."""
        company = self.get_company(ticker_or_cik)
        if not company:
            return []

        canonical_form = normalize_form_type(form_type, default=SECFormType.FORM_10K.value)
        matched: List[Any] = []
        try:
            filings = company.get_filings(form=canonical_form)
            if not filings:
                return []
            for filing in filings:
                if year is not None:
                    filing_date = str(getattr(filing, "filing_date", ""))
                    report_date = str(getattr(filing, "report_date", ""))
                    if str(year) not in filing_date and str(year) not in report_date:
                        continue
                matched.append(filing)
                if len(matched) >= limit:
                    break
        except Exception as e:
            logger.error(f"Error fetching {canonical_form} for '{ticker_or_cik}': {e}")
        return matched

    def fetch_latest_10k(self, ticker_or_cik: str, year: Optional[int] = None) -> Optional[Any]:
        """Retrieve the latest or year-specific 10-K filing."""
        company = self.get_company(ticker_or_cik)
        if not company:
            return None

        try:
            filings = company.get_filings(form=SECFormType.FORM_10K.value)
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

    @staticmethod
    def _coerce_section_text(value: Any) -> str:
        """Safely coerce an edgartools section value into clean text without stringifying mock objects."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        text_fn = getattr(value, "text", None)
        if callable(text_fn):
            try:
                rendered = text_fn()
                if isinstance(rendered, str):
                    return rendered.strip()
            except Exception:
                return ""
        return ""

    @classmethod
    def _extract_report_item(
        cls,
        report_obj: Any,
        attr_names: Tuple[str, ...],
        item_keys: Tuple[str, ...],
    ) -> str:
        """Extract a single narrative item from an edgartools TenK/TenQ object using property + dict lookup."""
        if report_obj is None:
            return ""
        for attr in attr_names:
            try:
                text = cls._coerce_section_text(getattr(report_obj, attr, None))
                if text:
                    return text
            except Exception:
                continue
        if hasattr(report_obj, "__getitem__"):
            for key in item_keys:
                try:
                    text = cls._coerce_section_text(report_obj[key])
                    if text:
                        return text
                except Exception:
                    continue
        return ""

    @staticmethod
    def _extract_acceptance_datetime(filing: Any) -> Optional[str]:
        """Extract EDGAR acceptance_datetime timestamp from an edgartools filing if present."""
        raw = getattr(filing, "acceptance_datetime", None)
        if raw is None:
            header = getattr(filing, "header", None)
            if header is not None:
                raw = getattr(header, "acceptance_datetime", None)
        if raw is None:
            return None
        if hasattr(raw, "isoformat"):
            iso_val = raw.isoformat()
            if isinstance(iso_val, str) and iso_val.strip():
                return iso_val.strip()
        if isinstance(raw, str):
            text = raw.strip()
            return text or None
        return None

    def extract_10k_sections(self, filing: Any) -> Dict[str, Any]:
        """Extract structured sections from a 10-K or 10-K/A filing using edgartools TenK parser (ADR-00046)."""
        raw_form = getattr(filing, "form", None)
        canonical_form = (
            normalize_form_type(raw_form, default=SECFormType.FORM_10K.value)
            if isinstance(raw_form, str)
            else SECFormType.FORM_10K.value
        )
        result: Dict[str, Any] = {
            "accession_number": getattr(filing, "accession_number", getattr(filing, "accession_no", "UNKNOWN")),
            "filing_date": str(getattr(filing, "filing_date", "")),
            "acceptance_datetime": self._extract_acceptance_datetime(filing),
            "report_date": str(getattr(filing, "report_date", "")),
            "form": canonical_form,
            "item_1_business": "",
            "item_1a_risk_factors": "",
            "item_1c_cybersecurity": "",
            "item_2_properties": "",
            "item_3_legal_proceedings": "",
            "item_7_mda": "",
            "subsidiaries": [],
        }

        if not filing:
            return result

        try:
            tenk = filing.obj() if hasattr(filing, "obj") else None
            if tenk is not None:
                section_map: Tuple[Tuple[str, Tuple[str, ...], Tuple[str, ...]], ...] = (
                    ("item_1_business", ("item_1", "business"), ("Item 1", "Item 1.")),
                    ("item_1a_risk_factors", ("item_1a", "risk_factors"), ("Item 1A", "Item 1A.")),
                    ("item_1c_cybersecurity", ("item_1c", "cybersecurity"), ("Item 1C", "Item 1C.")),
                    ("item_2_properties", ("item_2", "properties"), ("Item 2", "Item 2.")),
                    ("item_3_legal_proceedings", ("item_3", "legal_proceedings"), ("Item 3", "Item 3.")),
                    ("item_7_mda", ("item_7", "management_discussion"), ("Item 7", "Item 7.")),
                )
                for result_key, attr_names, item_keys in section_map:
                    result[result_key] = self._extract_report_item(tenk, attr_names, item_keys)

            # Extract Exhibit 21 (Subsidiaries) — captures delayed Exhibit 21 schedules on 10-K/A as well
            result["subsidiaries"] = self.extract_exhibit_21_subsidiaries(filing)

        except Exception as e:
            logger.warning(f"Error extracting sections from {canonical_form} filing: {e}")

        return result

    def extract_exhibit_21_subsidiaries(self, filing: Any) -> List[Dict[str, str]]:
        """Find and parse Exhibit 21 attachments from filing."""
        subsidiaries: List[Dict[str, str]] = []
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

    _EXCLUDED_EXHIBIT_TYPES_RE = re.compile(
        r"^(?:EX[- \t]*)?(?:101|104)(?:\b|[._-])|^(?:GRAPHIC|ZIP|EXCEL|XML|JSON|PDF)\b",
        re.IGNORECASE,
    )
    _EXCLUDED_EXHIBIT_EXTS: Tuple[str, ...] = (
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".svg",
        ".pdf",
        ".zip",
        ".xls",
        ".xlsx",
        ".xsd",
        ".xml",
    )
    _EXHIBIT_99_TYPE_RE = re.compile(r"^EX[- \t]?99(?:\.(\d+))?\b", re.IGNORECASE)
    _EXHIBIT_99_DESC_RE = re.compile(r"\bexhibit\s*99(?:\.(\d+))?\b", re.IGNORECASE)

    def extract_exhibit_99_attachments(
        self,
        filing: Any,
        *,
        max_chars: int = DEFAULT_MAX_FILING_CHARS,
    ) -> Dict[str, str]:
        """Extract Exhibit 99.x narrative press-release attachments, each bounded independently (ADR-00046)."""
        exhibits: Dict[str, str] = {}
        if not filing:
            return exhibits

        try:
            attachments = getattr(filing, "attachments", None)
            if not attachments:
                return exhibits
            for att in attachments:
                raw_type = getattr(att, "document_type", "")
                doc_type = raw_type.strip().upper() if isinstance(raw_type, str) else ""
                raw_desc = getattr(att, "description", "")
                desc = raw_desc.strip() if isinstance(raw_desc, str) else ""
                raw_doc = getattr(att, "document", "")
                doc_name = raw_doc.strip().lower() if isinstance(raw_doc, str) else ""

                if doc_type and self._EXCLUDED_EXHIBIT_TYPES_RE.match(doc_type):
                    continue
                if doc_name and doc_name.endswith(self._EXCLUDED_EXHIBIT_EXTS):
                    continue

                type_match = self._EXHIBIT_99_TYPE_RE.match(doc_type) if doc_type else None
                desc_match = self._EXHIBIT_99_DESC_RE.search(desc) if desc else None
                is_press_release = not doc_type and "press release" in desc.lower()
                if not type_match and not desc_match and not is_press_release:
                    continue

                if type_match:
                    suffix = type_match.group(1)
                    exhibit_key = f"EX-99.{suffix}" if suffix else "EX-99"
                elif desc_match:
                    suffix = desc_match.group(1)
                    exhibit_key = f"EX-99.{suffix}" if suffix else "EX-99.1"
                else:
                    exhibit_key = "EX-99.1"

                text = self._coerce_section_text(att)
                if not text:
                    continue
                bounded = text[:max_chars] if max_chars > 0 else text
                if exhibit_key in exhibits:
                    suffix_idx = 2
                    while f"{exhibit_key}_{suffix_idx}" in exhibits:
                        suffix_idx += 1
                    exhibit_key = f"{exhibit_key}_{suffix_idx}"
                exhibits[exhibit_key] = bounded
        except Exception as e:
            logger.debug(f"Exhibit 99.x extraction error: {e}")

        return exhibits

    def fetch_recent_8k(
        self,
        ticker_or_cik: str,
        limit: int = 5,
        year: Optional[int] = None,
        *,
        form_type: str = SECFormType.FORM_8K.value,
        include_exhibits: bool = True,
        max_chars: int = DEFAULT_MAX_FILING_CHARS,
    ) -> List[Dict[str, Any]]:
        """Fetch recent or year-specific Form 8-K / 8-K/A filings with attached Exhibit 99.x press releases."""
        company = self.get_company(ticker_or_cik)
        if not company:
            return []

        canonical_form = normalize_form_type(form_type, default=SECFormType.FORM_8K.value)
        results: List[Dict[str, Any]] = []
        try:
            filings = company.get_filings(form=canonical_form)
            if not filings:
                return []
            count = 0
            for filing in filings:
                filing_date_str = str(getattr(filing, "filing_date", ""))
                if year and str(year) not in filing_date_str:
                    continue
                raw_text = self._coerce_section_text(filing)
                bounded_text = raw_text[:max_chars] if max_chars > 0 else raw_text
                exhibits = (
                    self.extract_exhibit_99_attachments(filing, max_chars=max_chars)
                    if include_exhibits
                    else {}
                )
                raw_filing_form = getattr(filing, "form", None)
                resolved_form = (
                    normalize_form_type(raw_filing_form, default=canonical_form)
                    if isinstance(raw_filing_form, str)
                    else canonical_form
                )
                record: Dict[str, Any] = {
                    "accession_number": getattr(filing, "accession_number", getattr(filing, "accession_no", "UNKNOWN")),
                    "filing_date": filing_date_str,
                    "acceptance_datetime": self._extract_acceptance_datetime(filing),
                    "form": resolved_form,
                    "items": getattr(filing, "items", []),
                    "text": bounded_text,
                    "exhibits": exhibits,
                }
                if "EX-99.1" in exhibits:
                    record["exhibit_99_1"] = exhibits["EX-99.1"]
                elif exhibits:
                    record["exhibit_99_1"] = next(iter(exhibits.values()))
                results.append(record)
                count += 1
                if count >= limit:
                    break
        except Exception as e:
            logger.error(f"Error fetching {canonical_form} for '{ticker_or_cik}': {e}")
        return results

    def extract_10q_sections(
        self,
        filing: Any,
        *,
        max_fallback_chars: int = DEFAULT_MAX_FILING_CHARS,
    ) -> Dict[str, Any]:
        """Extract part-qualified narrative sections from a Form 10-Q or 10-Q/A filing (ADR-00046)."""
        raw_form = getattr(filing, "form", None)
        canonical_form = (
            normalize_form_type(raw_form, default=SECFormType.FORM_10Q.value)
            if isinstance(raw_form, str)
            else SECFormType.FORM_10Q.value
        )
        result: Dict[str, Any] = {
            "accession_number": getattr(filing, "accession_number", getattr(filing, "accession_no", "UNKNOWN")),
            "filing_date": str(getattr(filing, "filing_date", "")),
            "acceptance_datetime": self._extract_acceptance_datetime(filing),
            "report_date": str(getattr(filing, "report_date", "")),
            "form": canonical_form,
            "part1_item2_mda": "",
            "part1_item3_market_risk": "",
            "part2_item1_legal_proceedings": "",
            "part2_item1a_risk_factors": "",
            "text": "",
        }
        if not filing:
            return result

        try:
            tenq = filing.obj() if hasattr(filing, "obj") else None
            if tenq is not None:
                # Strictly part-qualified keys so Part I Item 1 (GAAP tables) never collides with Part II Item 1 (Legal)
                section_map: Tuple[Tuple[str, Tuple[str, ...], Tuple[str, ...]], ...] = (
                    ("part1_item2_mda", ("part1_item2", "management_discussion"), ("Part I, Item 2", "part_i_item_2")),
                    ("part1_item3_market_risk", ("part1_item3",), ("Part I, Item 3", "part_i_item_3")),
                    ("part2_item1_legal_proceedings", ("part2_item1",), ("Part II, Item 1", "part_ii_item_1")),
                    ("part2_item1a_risk_factors", ("part2_item1a", "risk_factors"), ("Part II, Item 1A", "part_ii_item_1a")),
                )
                for result_key, attr_names, item_keys in section_map:
                    result[result_key] = self._extract_report_item(tenq, attr_names, item_keys)
        except Exception as e:
            logger.warning(f"Error extracting sections from {canonical_form} filing: {e}")

        composed_parts: List[str] = []
        labeled_sections: Tuple[Tuple[str, str], ...] = (
            ("part1_item2_mda", "PART I, ITEM 2 - MANAGEMENT'S DISCUSSION AND ANALYSIS"),
            ("part1_item3_market_risk", "PART I, ITEM 3 - QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK"),
            ("part2_item1_legal_proceedings", "PART II, ITEM 1 - LEGAL PROCEEDINGS"),
            ("part2_item1a_risk_factors", "PART II, ITEM 1A - RISK FACTORS"),
        )
        for key, heading in labeled_sections:
            sec_text = result[key]
            if sec_text:
                composed_parts.append(f"{heading}\n\n{sec_text}")

        if composed_parts:
            result["text"] = "\n\n".join(composed_parts)
        else:
            fallback_raw = self._coerce_section_text(filing)
            result["text"] = fallback_raw[:max_fallback_chars] if max_fallback_chars > 0 else fallback_raw

        return result

    def fetch_10q_filings(
        self,
        ticker_or_cik: str,
        year: Optional[int] = None,
        limit: int = 4,
        *,
        form_type: str = SECFormType.FORM_10Q.value,
        max_fallback_chars: int = DEFAULT_MAX_FILING_CHARS,
    ) -> List[Dict[str, Any]]:
        """Fetch Form 10-Q or 10-Q/A quarterly reports with part-qualified narrative section extraction."""
        company = self.get_company(ticker_or_cik)
        if not company:
            return []

        canonical_form = normalize_form_type(form_type, default=SECFormType.FORM_10Q.value)
        results: List[Dict[str, Any]] = []
        try:
            filings = company.get_filings(form=canonical_form)
            if not filings:
                return []
            count = 0
            for filing in filings:
                filing_date_str = str(getattr(filing, "filing_date", ""))
                if year and str(year) not in filing_date_str:
                    continue
                extracted = self.extract_10q_sections(filing, max_fallback_chars=max_fallback_chars)
                if not extracted.get("form"):
                    extracted["form"] = canonical_form
                elif extracted["form"] == SECFormType.FORM_10Q.value and canonical_form == SECFormType.FORM_10QA.value:
                    extracted["form"] = canonical_form
                results.append(extracted)
                count += 1
                if count >= limit:
                    break
        except Exception as e:
            logger.error(f"Error fetching {canonical_form} for '{ticker_or_cik}': {e}")
        return results

    def fetch_form4_transactions(self, ticker_or_cik: str, limit: int = 20, year: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch Form 4 insider transactions."""
        company = self.get_company(ticker_or_cik)
        if not company:
            return []

        results = []
        try:
            filings = company.get_filings(form=SECFormType.FORM_4.value)
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
                    "form": SECFormType.FORM_4.value,
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
                "acceptance_datetime": sections.get("acceptance_datetime"),
                "report_date": sections.get("report_date"),
                "form": sections.get("form") or SECFormType.FORM_10K.value,
                "subsidiaries_count": stats["subsidiaries_count"],
            }, f, indent=2)

        for section_key, filename in (
            ("item_1_business", "item_1_business.txt"),
            ("item_1a_risk_factors", "item_1a_risk_factors.txt"),
            ("item_1c_cybersecurity", "item_1c_cybersecurity.txt"),
            ("item_2_properties", "item_2_properties.txt"),
            ("item_3_legal_proceedings", "item_3_legal_proceedings.txt"),
            ("item_7_mda", "item_7_mda.txt"),
        ):
            if sections.get(section_key):
                with open(filing_dir / filename, "w", encoding="utf-8") as f:
                    f.write(sections[section_key])

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
