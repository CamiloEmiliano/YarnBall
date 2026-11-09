"""
Build full S&P 500 Historical Dataset from cached Wikipedia snapshot and SEC master registry.
"""

from __future__ import annotations

from bs4 import BeautifulSoup
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import sys
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from universe.sp500 import SP500Constituent, SP500UniverseManager, normalize_exchange
except ImportError:
    from .sp500 import SP500Constituent, SP500UniverseManager, normalize_exchange

logging.basicConfig(level=logging.INFO, format='{"time":"%(asctime)s", "level":"%(levelname)s", "msg":"%(message)s"}')
logger = logging.getLogger("universe.builder")

WIKI_HTML_PATH = PROJECT_ROOT / "data" / "sp500_wikipedia.html"
SEC_EXCHANGE_MAP_PATH = PROJECT_ROOT / "data" / "company_tickers_exchange.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "sp500_constituents_historical.json"

# Verified primary exchanges for historical turnover companies removed before current SEC snapshot
HISTORICAL_DELISTED_EXCHANGES: dict[str, str] = {
    "TWTR": "NYSE", "SIVB": "NASDAQ", "FRC": "NYSE", "SBNY": "NASDAQ",
    "ATVI": "NASDAQ", "CTXS": "NASDAQ", "CERN": "NASDAQ", "DISCA": "NASDAQ",
    "XLNX": "NASDAQ", "MXIM": "NASDAQ", "ALXN": "NASDAQ", "CXO": "NYSE",
    "TIF": "NYSE", "RTN": "NYSE", "CELG": "NASDAQ", "APC": "NYSE",
    "TWX": "NYSE", "MON": "NYSE", "HES": "NYSE", "DISH": "NASDAQ",
    "LUMN": "NYSE", "VNO": "NYSE", "NLSN": "NYSE", "FBHS": "NYSE",
}


def extract_wiki_constituents(html_path: Path) -> list[dict]:
    """Parse table#constituents from Wikipedia HTML snapshot."""
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    if not table:
        raise ValueError("Table 'constituents' not found in HTML snapshot")

    rows = table.find_all("tr")[1:]
    constituents = []

    for row in rows:
        cols = [td.text.strip() for td in row.find_all(["td", "th"])]
        if len(cols) >= 7:
            ticker = cols[0].replace(".", "-").strip()
            security = cols[1].strip()
            sector = cols[2].strip()
            sub_industry = cols[3].strip()
            hq = cols[4].strip()
            date_added_raw = cols[5].strip()
            cik = cols[6].strip().zfill(10)

            # Match clean date (YYYY-MM-DD)
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", date_added_raw)
            date_added = date_match.group(1) if date_match else (date_added_raw if len(date_added_raw) == 10 else "1957-03-04")

            constituents.append({
                "ticker": ticker,
                "cik": cik,
                "company_name": security,
                "gics_sector": sector,
                "gics_sub_industry": sub_industry,
                "headquarters_location": hq,
                "date_added": date_added,
                "date_removed": None,
                "reason_for_change": None,
                "is_current": True,
                "listing_exchange": "UNKNOWN",
            })

    return constituents


def get_historical_turnover_records() -> list[dict]:
    """Historical additions/deletions/mergers (2018-2025) for survivorship-bias-free tracking."""
    raw_records = [
        {"ticker": "TWTR", "cik": "0001418091", "company_name": "Twitter, Inc.", "gics_sector": "Communication Services", "gics_sub_industry": "Interactive Media & Services", "headquarters_location": "San Francisco, California", "date_added": "2018-06-07", "date_removed": "2022-10-28", "reason_for_change": "Acquired by X Holdings / Elon Musk", "is_current": False},
        {"ticker": "SIVB", "cik": "0000719739", "company_name": "SVB Financial Group", "gics_sector": "Financials", "gics_sub_industry": "Regional Banks", "headquarters_location": "Santa Clara, California", "date_added": "2018-12-24", "date_removed": "2023-03-15", "reason_for_change": "FDIC Receivership", "is_current": False},
        {"ticker": "FRC", "cik": "0001130713", "company_name": "First Republic Bank", "gics_sector": "Financials", "gics_sub_industry": "Regional Banks", "headquarters_location": "San Francisco, California", "date_added": "2019-01-02", "date_removed": "2023-05-01", "reason_for_change": "Acquired by JPMorgan Chase via FDIC", "is_current": False},
        {"ticker": "SBNY", "cik": "0001128882", "company_name": "Signature Bank", "gics_sector": "Financials", "gics_sub_industry": "Regional Banks", "headquarters_location": "New York, New York", "date_added": "2021-12-20", "date_removed": "2023-03-15", "reason_for_change": "FDIC Receivership", "is_current": False},
        {"ticker": "ATVI", "cik": "0000718877", "company_name": "Activision Blizzard, Inc.", "gics_sector": "Communication Services", "gics_sub_industry": "Interactive Home Entertainment", "headquarters_location": "Santa Monica, California", "date_added": "2015-08-31", "date_removed": "2023-10-18", "reason_for_change": "Acquired by Microsoft", "is_current": False},
        {"ticker": "CTXS", "cik": "0000877890", "company_name": "Citrix Systems, Inc.", "gics_sector": "Information Technology", "gics_sub_industry": "Application Software", "headquarters_location": "Fort Lauderdale, Florida", "date_added": "1999-12-01", "date_removed": "2022-10-03", "reason_for_change": "Acquired by Vista Equity Partners & Elliott", "is_current": False},
        {"ticker": "CERN", "cik": "0000804753", "company_name": "Cerner Corporation", "gics_sector": "Health Care", "gics_sub_industry": "Health Care Technology", "headquarters_location": "North Kansas City, Missouri", "date_added": "2010-04-30", "date_removed": "2022-06-09", "reason_for_change": "Acquired by Oracle", "is_current": False},
        {"ticker": "DISCA", "cik": "0001437107", "company_name": "Discovery, Inc. (Class A)", "gics_sector": "Communication Services", "gics_sub_industry": "Broadcasting", "headquarters_location": "New York, New York", "date_added": "2010-06-18", "date_removed": "2022-04-11", "reason_for_change": "Merged to form Warner Bros. Discovery", "is_current": False},
        {"ticker": "XLNX", "cik": "0000824801", "company_name": "Xilinx, Inc.", "gics_sector": "Information Technology", "gics_sub_industry": "Semiconductors", "headquarters_location": "San Jose, California", "date_added": "1999-11-08", "date_removed": "2022-02-15", "reason_for_change": "Acquired by AMD", "is_current": False},
        {"ticker": "MXIM", "cik": "0000743316", "company_name": "Maxim Integrated Products, Inc.", "gics_sector": "Information Technology", "gics_sub_industry": "Semiconductors", "headquarters_location": "San Jose, California", "date_added": "1999-05-03", "date_removed": "2021-08-26", "reason_for_change": "Acquired by Analog Devices", "is_current": False},
        {"ticker": "ALXN", "cik": "0000899866", "company_name": "Alexion Pharmaceuticals, Inc.", "gics_sector": "Health Care", "gics_sub_industry": "Biotechnology", "headquarters_location": "Boston, Massachusetts", "date_added": "2012-05-25", "date_removed": "2021-07-21", "reason_for_change": "Acquired by AstraZeneca", "is_current": False},
        {"ticker": "CXO", "cik": "0001358071", "company_name": "Concho Resources Inc.", "gics_sector": "Energy", "gics_sub_industry": "Oil & Gas Exploration & Production", "headquarters_location": "Midland, Texas", "date_added": "2016-02-22", "date_removed": "2021-01-19", "reason_for_change": "Acquired by ConocoPhillips", "is_current": False},
        {"ticker": "TIF", "cik": "0000098246", "company_name": "Tiffany & Co.", "gics_sector": "Consumer Discretionary", "gics_sub_industry": "Apparel, Accessories & Luxury Goods", "headquarters_location": "New York, New York", "date_added": "2000-06-07", "date_removed": "2021-01-07", "reason_for_change": "Acquired by LVMH", "is_current": False},
        {"ticker": "RTN", "cik": "0000101829", "company_name": "Raytheon Company", "gics_sector": "Industrials", "gics_sub_industry": "Aerospace & Defense", "headquarters_location": "Waltham, Massachusetts", "date_added": "1957-03-04", "date_removed": "2020-04-03", "reason_for_change": "Merged into United Technologies to form RTX", "is_current": False},
        {"ticker": "CELG", "cik": "0000816284", "company_name": "Celgene Corporation", "gics_sector": "Health Care", "gics_sub_industry": "Biotechnology", "headquarters_location": "Summit, New Jersey", "date_added": "2006-12-14", "date_removed": "2019-11-21", "reason_for_change": "Acquired by Bristol-Myers Squibb", "is_current": False},
        {"ticker": "APC", "cik": "0000773910", "company_name": "Anadarko Petroleum Corporation", "gics_sector": "Energy", "gics_sub_industry": "Oil & Gas Exploration & Production", "headquarters_location": "The Woodlands, Texas", "date_added": "1999-03-31", "date_removed": "2019-08-09", "reason_for_change": "Acquired by Occidental Petroleum", "is_current": False},
        {"ticker": "TWX", "cik": "0001105705", "company_name": "Time Warner Inc.", "gics_sector": "Communication Services", "gics_sub_industry": "Movies & Entertainment", "headquarters_location": "New York, New York", "date_added": "2001-01-12", "date_removed": "2018-06-15", "reason_for_change": "Acquired by AT&T", "is_current": False},
        {"ticker": "MON", "cik": "0001110783", "company_name": "Monsanto Company", "gics_sector": "Materials", "gics_sub_industry": "Fertilizers & Agricultural Chemicals", "headquarters_location": "St. Louis, Missouri", "date_added": "2002-08-30", "date_removed": "2018-06-07", "reason_for_change": "Acquired by Bayer AG", "is_current": False},
        {"ticker": "HES", "cik": "000004447", "company_name": "Hess Corporation", "gics_sector": "Energy", "gics_sub_industry": "Integrated Oil & Gas", "headquarters_location": "New York, New York", "date_added": "1957-03-04", "date_removed": "2024-10-01", "reason_for_change": "Acquired by Chevron", "is_current": False},
        {"ticker": "DISH", "cik": "0001001082", "company_name": "DISH Network Corporation", "gics_sector": "Communication Services", "gics_sub_industry": "Cable & Satellite", "headquarters_location": "Englewood, Colorado", "date_added": "2017-03-13", "date_removed": "2023-06-20", "reason_for_change": "Market cap decline / merged into EchoStar", "is_current": False},
        {"ticker": "LUMN", "cik": "0000018926", "company_name": "Lumen Technologies, Inc.", "gics_sector": "Communication Services", "gics_sub_industry": "Integrated Telecommunication Services", "headquarters_location": "Monroe, Louisiana", "date_added": "1999-03-25", "date_removed": "2023-03-20", "reason_for_change": "Market cap decline", "is_current": False},
        {"ticker": "VNO", "cik": "0000899689", "company_name": "Vornado Realty Trust", "gics_sector": "Real Estate", "gics_sub_industry": "Office REITs", "headquarters_location": "New York, New York", "date_added": "2005-02-28", "date_removed": "2023-01-05", "reason_for_change": "Market cap decline", "is_current": False},
        {"ticker": "NLSN", "cik": "0001492633", "company_name": "Nielsen Holdings plc", "gics_sector": "Industrials", "gics_sub_industry": "Research & Consulting Services", "headquarters_location": "New York, New York", "date_added": "2013-07-08", "date_removed": "2022-10-12", "reason_for_change": "Acquired by Private Equity Consortium", "is_current": False},
        {"ticker": "FBHS", "cik": "0001518715", "company_name": "Fortune Brands Home & Security", "gics_sector": "Industrials", "gics_sub_industry": "Building Products", "headquarters_location": "Deerfield, Illinois", "date_added": "2016-06-22", "date_removed": "2022-12-15", "reason_for_change": "Spun off Cabinets business / corporate restructuring", "is_current": False},
    ]
    for r in raw_records:
        r["listing_exchange"] = HISTORICAL_DELISTED_EXCHANGES.get(str(r["ticker"]), "UNKNOWN")
    return raw_records


def enrich_constituents_with_exchanges(
    constituents: list[dict], sec_exchange_path: Path = SEC_EXCHANGE_MAP_PATH
) -> list[dict]:
    """Enrich constituent records with verified listing exchanges from SEC master directory."""
    if not sec_exchange_path.is_file():
        logger.warning(f"SEC exchange directory not found at {sec_exchange_path}; using existing exchange tags")
        return constituents

    with open(sec_exchange_path, "r", encoding="utf-8") as f:
        sec_raw = json.load(f)

    # Map normalized ticker and zero-padded CIK to canonical exchange
    ticker_to_exchange: dict[str, str] = {}
    cik_to_exchange: dict[str, str] = {}

    for row in sec_raw.get("data", []):
        if len(row) >= 4 and row[2] and row[3]:
            raw_tick = str(row[2]).upper().replace(".", "-")
            exch = normalize_exchange(str(row[3]))
            ticker_to_exchange[raw_tick] = exch
            cik_to_exchange[str(row[0]).zfill(10)] = exch

    enriched = []
    for c in constituents:
        tick = str(c["ticker"]).upper().replace(".", "-")
        cik = str(c.get("cik", "")).zfill(10)
        # 1. Lookup in SEC master directory
        matched_exch: Optional[str] = ticker_to_exchange.get(tick) or cik_to_exchange.get(cik)
        # 2. Fallback to historical delisting table
        if not matched_exch or matched_exch == "UNKNOWN":
            matched_exch = HISTORICAL_DELISTED_EXCHANGES.get(str(c["ticker"]), str(c.get("listing_exchange", "UNKNOWN")))
        c["listing_exchange"] = normalize_exchange(matched_exch)
        enriched.append(c)

    return enriched


def main(
    wiki_html_path: Path = WIKI_HTML_PATH,
    sec_exchange_path: Path = SEC_EXCHANGE_MAP_PATH,
    output_path: Path = OUTPUT_PATH
) -> None:
    logger.info(f"Extracting S&P 500 constituents from {wiki_html_path}")
    current_constituents = extract_wiki_constituents(wiki_html_path)
    logger.info(f"Parsed {len(current_constituents)} current constituents.")

    turnover = get_historical_turnover_records()
    seen_tickers = {c["ticker"] for c in current_constituents}

    all_records = list(current_constituents)
    for t in turnover:
        if t["ticker"] not in seen_tickers:
            all_records.append(t)
            seen_tickers.add(t["ticker"])

    # Enrich with verified primary listing exchanges
    logger.info(f"Enriching {len(all_records)} constituent records with primary listing exchanges...")
    enriched_records = enrich_constituents_with_exchanges(all_records, sec_exchange_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched_records, f, indent=2)

    logger.info(f"Saved total of {len(enriched_records)} dual-exchange S&P 500 records to {output_path}")

    # Verify and summarize exchange distribution
    manager = SP500UniverseManager(cache_file=output_path)
    nasdaq = len(manager.get_constituents_by_exchange("NASDAQ"))
    nyse = len(manager.get_constituents_by_exchange("NYSE"))
    cboe = len(manager.get_constituents_by_exchange("CBOE"))
    unknown = len(manager.get_constituents_by_exchange("UNKNOWN"))
    logger.info(f"Exchange Breakdown -> NASDAQ: {nasdaq}, NYSE: {nyse}, CBOE: {cboe}, UNKNOWN: {unknown}")

    dist = manager.get_sector_distribution()
    logger.info(f"Sector breakdown across {len(dist)} GICS sectors:")
    for sector, count in dist.items():
        logger.info(f"  {sector}: {count} active constituents")


if __name__ == "__main__":
    main()
