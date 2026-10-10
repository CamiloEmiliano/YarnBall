"""
S&P 500 Historical Constituent Universe Manager.

Manages point-in-time S&P 500 membership (2018-2025+) with GICS sectors,
CIK mappings, index turnover (additions/deletions), and PostgreSQL synchronization.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date, datetime
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    env_path = PROJECT_ROOT / ".env"
    if env_path.is_file():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

from tools.init_db import _connect_db, get_dsn

logger = logging.getLogger("sp500_universe")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('{"time":"%(asctime)s", "level":"%(levelname)s", "msg":"%(message)s"}'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_CACHE_PATH = DATA_DIR / "sp500_constituents_historical.json"


@dataclass
class SP500Constituent:
    ticker: str
    cik: str
    company_name: str
    gics_sector: str
    gics_sub_industry: str = ""
    headquarters_location: str = ""
    date_added: Optional[str] = None  # YYYY-MM-DD
    date_removed: Optional[str] = None  # YYYY-MM-DD
    reason_for_change: Optional[str] = None
    is_current: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SP500UniverseManager:
    """Manages S&P 500 constituents, point-in-time lookups, and DB synchronization."""

    def __init__(self, cache_file: Path = DEFAULT_CACHE_PATH):
        self.cache_file = Path(cache_file)
        self._constituents: List[SP500Constituent] = []
        self._load_cache()

    def _load_cache(self) -> None:
        """Load constituent records from local cache if present, else initialize built-in dataset."""
        if self.cache_file.is_file():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._constituents = [SP500Constituent(**item) for item in data]
                logger.info(f"Loaded {len(self._constituents)} S&P 500 constituent records from {self.cache_file}")
                return
            except Exception as exc:
                logger.warning(f"Error reading cache {self.cache_file}: {exc}; rebuilding from seed data")

        self._constituents = self._get_seed_constituents()
        self.save_cache()

    def save_cache(self) -> None:
        """Persist current in-memory constituent list to JSON cache."""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self._constituents], f, indent=2)
        logger.info(f"Saved {len(self._constituents)} constituents to {self.cache_file}")

    def get_all_records(self) -> List[SP500Constituent]:
        """Return all constituent records (current and historical)."""
        return list(self._constituents)

    def get_constituents_at_date(self, target_date: str | date | datetime) -> List[SP500Constituent]:
        """Return point-in-time S&P 500 constituents active on target_date."""
        if isinstance(target_date, str):
            t_date = datetime.strptime(target_date[:10], "%Y-%m-%d").date()
        elif isinstance(target_date, datetime):
            t_date = target_date.date()
        else:
            t_date = target_date

        active: List[SP500Constituent] = []
        for c in self._constituents:
            added = datetime.strptime(c.date_added, "%Y-%m-%d").date() if c.date_added else date(1957, 3, 4)
            removed = datetime.strptime(c.date_removed, "%Y-%m-%d").date() if c.date_removed else None

            if added <= t_date:
                if removed is None or removed > t_date:
                    active.append(c)

        return active

    def get_current_constituents(self) -> List[SP500Constituent]:
        """Return all currently active S&P 500 constituents."""
        return [c for c in self._constituents if c.is_current and (c.date_removed is None)]

    def get_sector_distribution(self, target_date: Optional[str | date | datetime] = None) -> Dict[str, int]:
        """Calculate distribution across the 11 GICS economic sectors."""
        constituents = self.get_constituents_at_date(target_date) if target_date else self.get_current_constituents()
        dist: Dict[str, int] = {}
        for c in constituents:
            dist[c.gics_sector] = dist.get(c.gics_sector, 0) + 1
        return dict(sorted(dist.items(), key=lambda x: x[1], reverse=True))

    def is_sp500(self, identifier: str, target_date: Optional[str | date | datetime] = None) -> bool:
        """Check if a ticker or CIK was in the S&P 500 at target_date (or current if None)."""
        ident_clean = identifier.strip().upper()
        ident_cik = str(int(ident_clean)).zfill(10) if ident_clean.isdigit() else ""

        constituents = self.get_constituents_at_date(target_date) if target_date else self.get_current_constituents()
        for c in constituents:
            if c.ticker.upper() == ident_clean or (ident_cik and c.cik.zfill(10) == ident_cik):
                return True
        return False

    def is_constituent(self, identifier: str, target_date: Optional[str | date | datetime] = None) -> bool:
        """Alias for is_sp500."""
        return self.is_sp500(identifier, target_date=target_date)

    def sync_to_postgres(self, db_name: str = "financial_rag") -> int:
        """
        Upsert historical S&P 500 constituents into `sp500_historical_constituents`
        and update `is_sp500 = TRUE` in `sec_companies`.
        """
        conn = _connect_db(db_name)
        cur = conn.cursor()
        synced_count = 0

        try:
            for c in self._constituents:
                cur.execute("""
                    INSERT INTO sp500_historical_constituents (
                        ticker, cik, company_name, gics_sector, gics_sub_industry,
                        headquarters_location, date_added, date_removed, reason_for_change, is_current
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING;
                """, (
                    c.ticker,
                    c.cik.zfill(10) if c.cik else None,
                    c.company_name,
                    c.gics_sector,
                    c.gics_sub_industry,
                    c.headquarters_location,
                    c.date_added,
                    c.date_removed,
                    c.reason_for_change,
                    c.is_current
                ))

                # Also mark is_sp500 in sec_companies if ticker or CIK matches
                if c.cik:
                    cur.execute("""
                        UPDATE sec_companies
                        SET is_sp500 = TRUE
                        WHERE cik = %s OR ticker = %s;
                    """, (c.cik.zfill(10), c.ticker))
                else:
                    cur.execute("""
                        UPDATE sec_companies
                        SET is_sp500 = TRUE
                        WHERE ticker = %s;
                    """, (c.ticker,))

                synced_count += 1

            conn.commit()
            logger.info(f"Successfully synced {synced_count} S&P 500 constituent records into PostgreSQL ({db_name})")
        finally:
            if hasattr(cur, "close"):
                cur.close()
            conn.close()

        return synced_count

    def _get_seed_constituents(self) -> List[SP500Constituent]:
        """Comprehensive seed registry covering core S&P 500 constituents and 2018-2025 turnover."""
        seed_data = [
            # Technology
            SP500Constituent("AAPL", "0000320193", "Apple Inc.", "Information Technology", "Technology Hardware, Storage & Peripherals", "Cupertino, California", "1982-11-30"),
            SP500Constituent("MSFT", "0000789019", "Microsoft Corporation", "Information Technology", "Systems Software", "Redmond, Washington", "1994-06-01"),
            SP500Constituent("NVDA", "0001045810", "NVIDIA Corporation", "Information Technology", "Semiconductors", "Santa Clara, California", "2001-11-30"),
            SP500Constituent("AVGO", "0001730168", "Broadcom Inc.", "Information Technology", "Semiconductors", "Palo Alto, California", "2014-05-08"),
            SP500Constituent("ORCL", "0001341439", "Oracle Corporation", "Information Technology", "Application Software", "Austin, Texas", "1989-08-31"),
            SP500Constituent("CRM", "0001108524", "Salesforce, Inc.", "Information Technology", "Application Software", "San Francisco, California", "2008-09-15"),
            SP500Constituent("AMD", "0000002488", "Advanced Micro Devices, Inc.", "Information Technology", "Semiconductors", "Santa Clara, California", "2017-03-20"),
            SP500Constituent("INTC", "0000050863", "Intel Corporation", "Information Technology", "Semiconductors", "Santa Clara, California", "1976-12-31"),
            SP500Constituent("QCOM", "0000804328", "QUALCOMM Incorporated", "Information Technology", "Semiconductors", "San Diego, California", "1999-07-22"),
            SP500Constituent("TXN", "0000097476", "Texas Instruments Incorporated", "Information Technology", "Semiconductors", "Dallas, Texas", "1957-03-04"),
            SP500Constituent("IBM", "0000051143", "International Business Machines Corp", "Information Technology", "IT Consulting & Other Services", "Armonk, New York", "1957-03-04"),
            SP500Constituent("NOW", "0001373715", "ServiceNow, Inc.", "Information Technology", "Systems Software", "Santa Clara, California", "2019-11-21"),
            SP500Constituent("INTU", "0000896878", "Intuit Inc.", "Information Technology", "Application Software", "Mountain View, California", "2000-12-05"),
            SP500Constituent("AMAT", "0000006951", "Applied Materials, Inc.", "Information Technology", "Semiconductor Equipment", "Santa Clara, California", "1995-03-16"),
            SP500Constituent("LRCX", "0000707549", "Lam Research Corporation", "Information Technology", "Semiconductor Equipment", "Fremont, California", "2012-06-29"),
            SP500Constituent("MU", "0000723125", "Micron Technology, Inc.", "Information Technology", "Semiconductors", "Boise, Idaho", "1994-09-27"),
            SP500Constituent("ADI", "0000006281", "Analog Devices, Inc.", "Information Technology", "Semiconductors", "Wilmington, Massachusetts", "1999-10-12"),
            SP500Constituent("KLAC", "0000314606", "KLA Corporation", "Information Technology", "Semiconductor Equipment", "Milpitas, California", "1997-09-30"),
            SP500Constituent("SNPS", "0000883241", "Synopsys, Inc.", "Information Technology", "Application Software", "Sunnyvale, California", "2017-03-16"),
            SP500Constituent("CDNS", "0000813672", "Cadence Design Systems, Inc.", "Information Technology", "Application Software", "San Jose, California", "2017-09-18"),
            SP500Constituent("CRWD", "0001535527", "CrowdStrike Holdings, Inc.", "Information Technology", "Systems Software", "Austin, Texas", "2024-06-24"),
            SP500Constituent("PLTR", "0001321655", "Palantir Technologies Inc.", "Information Technology", "Application Software", "Denver, Colorado", "2024-09-23"),
            SP500Constituent("DELL", "0001571996", "Dell Technologies Inc.", "Information Technology", "Technology Hardware, Storage & Peripherals", "Round Rock, Texas", "2024-09-23"),
            SP500Constituent("SMCI", "0001375365", "Super Micro Computer, Inc.", "Information Technology", "Technology Hardware, Storage & Peripherals", "San Jose, California", "2024-03-18"),
            
            # Communication Services
            SP500Constituent("GOOGL", "0001652044", "Alphabet Inc. (Class A)", "Communication Services", "Interactive Media & Services", "Mountain View, California", "2014-04-03"),
            SP500Constituent("GOOG", "0001652044", "Alphabet Inc. (Class C)", "Communication Services", "Interactive Media & Services", "Mountain View, California", "2006-04-03"),
            SP500Constituent("META", "0001326801", "Meta Platforms, Inc.", "Communication Services", "Interactive Media & Services", "Menlo Park, California", "2013-12-23"),
            SP500Constituent("NFLX", "0001065280", "Netflix, Inc.", "Communication Services", "Movies & Entertainment", "Los Gatos, California", "2010-12-20"),
            SP500Constituent("DIS", "0001744489", "The Walt Disney Company", "Communication Services", "Movies & Entertainment", "Burbank, California", "1976-06-30"),
            SP500Constituent("CMCSA", "0001166691", "Comcast Corporation", "Communication Services", "Cable & Satellite", "Philadelphia, Pennsylvania", "2002-11-19"),
            SP500Constituent("VZ", "0000732712", "Verizon Communications Inc.", "Communication Services", "Integrated Telecommunication Services", "New York, New York", "1983-11-30"),
            SP500Constituent("T", "0000720672", "AT&T Inc.", "Communication Services", "Integrated Telecommunication Services", "Dallas, Texas", "1983-11-30"),
            SP500Constituent("TMUS", "0001283699", "T-Mobile US, Inc.", "Communication Services", "Wireless Telecommunication Services", "Bellevue, Washington", "2019-07-15"),

            # Consumer Discretionary
            SP500Constituent("AMZN", "0001018724", "Amazon.com, Inc.", "Consumer Discretionary", "Broadline Retail", "Seattle, Washington", "2005-11-18"),
            SP500Constituent("TSLA", "0001318605", "Tesla, Inc.", "Consumer Discretionary", "Automobile Manufacturers", "Austin, Texas", "2020-12-21"),
            SP500Constituent("HD", "0000354950", "The Home Depot, Inc.", "Consumer Discretionary", "Home Improvement Retail", "Atlanta, Georgia", "1988-03-31"),
            SP500Constituent("MCD", "0000063908", "McDonald's Corporation", "Consumer Discretionary", "Restaurants", "Chicago, Illinois", "1970-06-30"),
            SP500Constituent("NKE", "0000320187", "NIKE, Inc.", "Consumer Discretionary", "Footwear", "Beaverton, Oregon", "1988-11-30"),
            SP500Constituent("SBUX", "0000829224", "Starbucks Corporation", "Consumer Discretionary", "Restaurants", "Seattle, Washington", "2000-06-07"),
            SP500Constituent("TJX", "0000109198", "The TJX Companies, Inc.", "Consumer Discretionary", "Apparel Retail", "Framingham, Massachusetts", "1989-10-31"),
            SP500Constituent("LOW", "0000060667", "Lowe's Companies, Inc.", "Consumer Discretionary", "Home Improvement Retail", "Mooresville, North Carolina", "1984-02-29"),
            SP500Constituent("BKNG", "0001075531", "Booking Holdings Inc.", "Consumer Discretionary", "Hotels, Resorts & Cruise Lines", "Norwalk, Connecticut", "2009-11-06"),

            # Financials
            SP500Constituent("BRK.B", "0001067983", "Berkshire Hathaway Inc.", "Financials", "Multi-Sector Holdings", "Omaha, Nebraska", "2010-02-16"),
            SP500Constituent("JPM", "0000019617", "JPMorgan Chase & Co.", "Financials", "Diversified Banks", "New York, New York", "1975-06-30"),
            SP500Constituent("V", "0001403161", "Visa Inc.", "Financials", "Transaction & Payment Processing Services", "San Francisco, California", "2009-12-21"),
            SP500Constituent("MA", "0001141391", "Mastercard Incorporated", "Financials", "Transaction & Payment Processing Services", "Purchase, New York", "2008-07-18"),
            SP500Constituent("BAC", "0000070858", "Bank of America Corporation", "Financials", "Diversified Banks", "Charlotte, North Carolina", "1976-06-30"),
            SP500Constituent("WFC", "0000072971", "Wells Fargo & Company", "Financials", "Diversified Banks", "San Francisco, California", "1976-06-30"),
            SP500Constituent("GS", "0000886982", "The Goldman Sachs Group, Inc.", "Financials", "Investment Banking & Brokerage", "New York, New York", "2002-07-22"),
            SP500Constituent("MS", "0000895421", "Morgan Stanley", "Financials", "Investment Banking & Brokerage", "New York, New York", "1993-05-03"),
            SP500Constituent("BLK", "0001364742", "BlackRock, Inc.", "Financials", "Asset Management & Custody Banks", "New York, New York", "2011-04-04"),
            SP500Constituent("KKR", "0001404912", "KKR & Co. Inc.", "Financials", "Asset Management & Custody Banks", "New York, New York", "2024-06-24"),
            SP500Constituent("BX", "0001393818", "Blackstone Inc.", "Financials", "Asset Management & Custody Banks", "New York, New York", "2023-09-18"),

            # Health Care
            SP500Constituent("LLY", "0000059478", "Eli Lilly and Company", "Health Care", "Pharmaceuticals", "Indianapolis, Indiana", "1957-03-04"),
            SP500Constituent("UNH", "0000731766", "UnitedHealth Group Incorporated", "Health Care", "Managed Healthcare", "Minnetonka, Minnesota", "1994-07-01"),
            SP500Constituent("JNJ", "0000200406", "Johnson & Johnson", "Health Care", "Pharmaceuticals", "New Brunswick, New Jersey", "1973-06-30"),
            SP500Constituent("ABBV", "0001551152", "AbbVie Inc.", "Health Care", "Biotechnology", "North Chicago, Illinois", "2013-01-02"),
            SP500Constituent("MRK", "0000310158", "Merck & Co., Inc.", "Health Care", "Pharmaceuticals", "Rahway, New Jersey", "1957-03-04"),
            SP500Constituent("TMO", "0000097745", "Thermo Fisher Scientific Inc.", "Health Care", "Life Sciences Tools & Services", "Waltham, Massachusetts", "2004-08-04"),
            SP500Constituent("ABT", "0000001800", "Abbott Laboratories", "Health Care", "Health Care Equipment", "North Chicago, Illinois", "1957-03-04"),
            SP500Constituent("PFE", "0000078003", "Pfizer Inc.", "Health Care", "Pharmaceuticals", "New York, New York", "1957-03-04"),
            SP500Constituent("DHR", "0000313616", "Danaher Corporation", "Health Care", "Life Sciences Tools & Services", "Washington, D.C.", "1998-11-18"),
            SP500Constituent("BMY", "0000014272", "Bristol-Myers Squibb Company", "Health Care", "Pharmaceuticals", "Princeton, New Jersey", "1957-03-04"),

            # Consumer Staples
            SP500Constituent("WMT", "0000104169", "Walmart Inc.", "Consumer Staples", "Consumer Staples Merchandise Retail", "Bentonville, Arkansas", "1982-08-31"),
            SP500Constituent("PG", "0000080424", "The Procter & Gamble Company", "Consumer Staples", "Household Products", "Cincinnati, Ohio", "1957-03-04"),
            SP500Constituent("COST", "0000909832", "Costco Wholesale Corporation", "Consumer Staples", "Consumer Staples Merchandise Retail", "Issaquah, Washington", "1993-10-01"),
            SP500Constituent("KO", "0000021344", "The Coca-Cola Company", "Consumer Staples", "Soft Drinks & Non-alcoholic Beverages", "Atlanta, Georgia", "1957-03-04"),
            SP500Constituent("PEP", "0000077476", "PepsiCo, Inc.", "Consumer Staples", "Soft Drinks & Non-alcoholic Beverages", "Purchase, New York", "1957-03-04"),
            SP500Constituent("PM", "0001413329", "Philip Morris International Inc.", "Consumer Staples", "Tobacco", "Stamford, Connecticut", "2008-03-31"),
            SP500Constituent("MDLZ", "0001103982", "Mondelez International, Inc.", "Consumer Staples", "Packaged Foods & Meats", "Chicago, Illinois", "2012-10-02"),

            # Energy
            SP500Constituent("XOM", "0000034088", "Exxon Mobil Corporation", "Energy", "Integrated Oil & Gas", "Spring, Texas", "1957-03-04"),
            SP500Constituent("CVX", "0000093410", "Chevron Corporation", "Energy", "Integrated Oil & Gas", "San Ramon, California", "1957-03-04"),
            SP500Constituent("COP", "0001163165", "ConocoPhillips", "Energy", "Oil & Gas Exploration & Production", "Houston, Texas", "1957-03-04"),
            SP500Constituent("EOG", "0000821189", "EOG Resources, Inc.", "Energy", "Oil & Gas Exploration & Production", "Houston, Texas", "2000-11-02"),
            SP500Constituent("SLB", "0000087347", "Schlumberger Limited", "Energy", "Oil & Gas Equipment & Services", "Houston, Texas", "1965-03-31"),

            # Industrials
            SP500Constituent("CAT", "0000018230", "Caterpillar Inc.", "Industrials", "Construction Machinery & Heavy Transportation Equipment", "Irving, Texas", "1957-03-04"),
            SP500Constituent("GE", "0000040545", "General Electric Company", "Industrials", "Industrial Conglomerates", "Evendale, Ohio", "1957-03-04"),
            SP500Constituent("UNP", "0000100885", "Union Pacific Corporation", "Industrials", "Rail Transportation", "Omaha, Nebraska", "1957-03-04"),
            SP500Constituent("HON", "0000773840", "Honeywell International Inc.", "Industrials", "Industrial Conglomerates", "Charlotte, North Carolina", "1964-03-31"),
            SP500Constituent("RTX", "0000101829", "RTX Corporation", "Industrials", "Aerospace & Defense", "Arlington, Virginia", "1957-03-04"),
            SP500Constituent("BA", "0000012927", "The Boeing Company", "Industrials", "Aerospace & Defense", "Arlington, Virginia", "1957-03-04"),
            SP500Constituent("LMT", "0000936468", "Lockheed Martin Corporation", "Industrials", "Aerospace & Defense", "Bethesda, Maryland", "1995-03-16"),
            SP500Constituent("DE", "0000315189", "Deere & Company", "Industrials", "Agricultural & Farm Machinery", "Moline, Illinois", "1957-03-04"),

            # Utilities
            SP500Constituent("NEE", "0000753308", "NextEra Energy, Inc.", "Utilities", "Electric Utilities", "Juno Beach, Florida", "1976-06-30"),
            SP500Constituent("SO", "0000092122", "The Southern Company", "Utilities", "Electric Utilities", "Atlanta, Georgia", "1957-03-04"),
            SP500Constituent("DUK", "0001326160", "Duke Energy Corporation", "Utilities", "Electric Utilities", "Charlotte, North Carolina", "1976-06-30"),
            SP500Constituent("CEG", "0001868275", "Constellation Energy Corporation", "Utilities", "Electric Utilities", "Baltimore, Maryland", "2022-02-02"),
            SP500Constituent("VST", "0001692819", "Vistra Corp.", "Utilities", "Independent Power Producers & Energy Traders", "Irving, Texas", "2024-05-08"),

            # Real Estate
            SP500Constituent("PLD", "0001045609", "Prologis, Inc.", "Real Estate", "Industrial REITs", "San Francisco, California", "2003-07-17"),
            SP500Constituent("AMT", "0001053507", "American Tower Corporation", "Real Estate", "Telecom Tower REITs", "Boston, Massachusetts", "2007-11-19"),
            SP500Constituent("EQIX", "0001101239", "Equinix, Inc.", "Real Estate", "Data Center REITs", "Redwood City, California", "2015-03-20"),
            SP500Constituent("PSA", "0001393311", "Public Storage", "Real Estate", "Self-Storage REITs", "Glendale, California", "2005-08-19"),

            # Materials
            SP500Constituent("LIN", "0001707925", "Linde plc", "Materials", "Industrial Gases", "Woking, United Kingdom", "1992-07-01"),
            SP500Constituent("SHW", "0000089800", "The Sherwin-Williams Company", "Materials", "Specialty Chemicals", "Cleveland, Ohio", "1964-06-30"),
            SP500Constituent("FCX", "0000831259", "Freeport-McMoRan Inc.", "Materials", "Copper", "Phoenix, Arizona", "2011-07-01"),
            SP500Constituent("ECL", "0000031462", "Ecolab Inc.", "Materials", "Specialty Chemicals", "Saint Paul, Minnesota", "1989-01-31"),

            # Historical Delisted / Merged / Removed Constituents (2018-2024 for Survivorship Bias Prevention)
            SP500Constituent("TWTR", "0001418091", "Twitter, Inc.", "Communication Services", "Interactive Media & Services", "San Francisco, California", "2018-06-07", "2022-10-28", "Acquired by X Holdings / Elon Musk", is_current=False),
            SP500Constituent("SIVB", "0000719739", "SVB Financial Group", "Financials", "Regional Banks", "Santa Clara, California", "2018-12-24", "2023-03-15", "FDIC Receivership", is_current=False),
            SP500Constituent("FRC", "0001130713", "First Republic Bank", "Financials", "Regional Banks", "San Francisco, California", "2019-01-02", "2023-05-01", "Acquired by JPMorgan Chase via FDIC", is_current=False),
            SP500Constituent("SBNY", "0001128882", "Signature Bank", "Financials", "Regional Banks", "New York, New York", "2021-12-20", "2023-03-15", "FDIC Receivership", is_current=False),
            SP500Constituent("ATVI", "0000718877", "Activision Blizzard, Inc.", "Communication Services", "Interactive Home Entertainment", "Santa Monica, California", "2015-08-31", "2023-10-18", "Acquired by Microsoft", is_current=False),
            SP500Constituent("CTXS", "0000877890", "Citrix Systems, Inc.", "Information Technology", "Application Software", "Fort Lauderdale, Florida", "1999-12-01", "2022-10-03", "Acquired by Vista Equity Partners & Elliott", is_current=False),
            SP500Constituent("CERN", "0000804753", "Cerner Corporation", "Health Care", "Health Care Technology", "North Kansas City, Missouri", "2010-04-30", "2022-06-09", "Acquired by Oracle", is_current=False),
            SP500Constituent("DISCA", "0001437107", "Discovery, Inc. (Class A)", "Communication Services", "Broadcasting", "New York, New York", "2010-06-18", "2022-04-11", "Merged to form Warner Bros. Discovery", is_current=False),
            SP500Constituent("XLNX", "0000824801", "Xilinx, Inc.", "Information Technology", "Semiconductors", "San Jose, California", "1999-11-08", "2022-02-15", "Acquired by AMD", is_current=False),
            SP500Constituent("MXIM", "0000743316", "Maxim Integrated Products, Inc.", "Information Technology", "Semiconductors", "San Jose, California", "1999-05-03", "2021-08-26", "Acquired by Analog Devices", is_current=False),
            SP500Constituent("ALXN", "0000899866", "Alexion Pharmaceuticals, Inc.", "Health Care", "Biotechnology", "Boston, Massachusetts", "2012-05-25", "2021-07-21", "Acquired by AstraZeneca", is_current=False),
            SP500Constituent("CXO", "0001358071", "Concho Resources Inc.", "Energy", "Oil & Gas Exploration & Production", "Midland, Texas", "2016-02-22", "2021-01-19", "Acquired by ConocoPhillips", is_current=False),
            SP500Constituent("TIF", "0000098246", "Tiffany & Co.", "Consumer Discretionary", "Apparel, Accessories & Luxury Goods", "New York, New York", "2000-06-07", "2021-01-07", "Acquired by LVMH", is_current=False),
            SP500Constituent("RTN", "0000101829", "Raytheon Company", "Industrials", "Aerospace & Defense", "Waltham, Massachusetts", "1957-03-04", "2020-04-03", "Merged into United Technologies to form RTX", is_current=False),
            SP500Constituent("CELG", "0000816284", "Celgene Corporation", "Health Care", "Biotechnology", "Summit, New Jersey", "2006-12-14", "2019-11-21", "Acquired by Bristol-Myers Squibb", is_current=False),
            SP500Constituent("APC", "0000773910", "Anadarko Petroleum Corporation", "Energy", "Oil & Gas Exploration & Production", "The Woodlands, Texas", "1999-03-31", "2019-08-09", "Acquired by Occidental Petroleum", is_current=False),
            SP500Constituent("TWX", "0001105705", "Time Warner Inc.", "Communication Services", "Movies & Entertainment", "New York, New York", "2001-01-12", "2018-06-15", "Acquired by AT&T", is_current=False),
            SP500Constituent("MON", "0001110783", "Monsanto Company", "Materials", "Fertilizers & Agricultural Chemicals", "St. Louis, Missouri", "2002-08-30", "2018-06-07", "Acquired by Bayer AG", is_current=False),
        ]
        return seed_data


def main() -> None:
    parser = argparse.ArgumentParser(description="S&P 500 Historical Constituent Universe Manager")
    parser.add_argument("--sync", action="store_true", help="Sync S&P 500 constituents to PostgreSQL")
    parser.add_argument("--db", default=os.getenv("FINANCIAL_RAG_DB", "financial_rag"), help="PostgreSQL database name")
    parser.add_argument("--date", help="Query constituents active at point-in-time date (YYYY-MM-DD)")
    parser.add_argument("--sector", help="Filter constituents by GICS Sector")
    parser.add_argument("--check", help="Check if ticker or CIK is in S&P 500")
    parser.add_argument("--stats", action="store_true", help="Display GICS sector distribution stats")
    parser.add_argument("--export", help="Export queried constituents to JSON file")
    args = parser.parse_args()

    manager = SP500UniverseManager()

    if args.sync:
        count = manager.sync_to_postgres(args.db)
        print(f"Synced {count} S&P 500 constituents to PostgreSQL ({args.db})")

    if args.stats:
        target_date = args.date if args.date else None
        dist = manager.get_sector_distribution(target_date)
        print("\nS&P 500 GICS Sector Distribution" + (f" (as of {target_date})" if target_date else " (Current)") + ":")
        print("------------------------------------------------------------")
        for sector, count in dist.items():
            print(f"  {sector:<30}: {count:>3} companies")
        print("------------------------------------------------------------")
        print(f"  Total Active Constituents: {sum(dist.values()):>3}")

    if args.check:
        is_in = manager.is_sp500(args.check, args.date)
        date_str = f" as of {args.date}" if args.date else " currently"
        print(f"Ticker/CIK '{args.check}' {'IS' if is_in else 'IS NOT'} in S&P 500{date_str}.")

    if args.sector:
        constituents = manager.get_constituents_at_date(args.date) if args.date else manager.get_current_constituents()
        matched = [c for c in constituents if args.sector.lower() in c.gics_sector.lower()]
        print(f"\nConstituents in Sector matching '{args.sector}':")
        for c in matched:
            print(f"  - [{c.ticker}] {c.company_name} (CIK: {c.cik}) - {c.gics_sub_industry}")

    if args.export:
        constituents = manager.get_constituents_at_date(args.date) if args.date else manager.get_current_constituents()
        out_path = Path(args.export)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in constituents], f, indent=2)
        print(f"Exported {len(constituents)} constituents to {out_path}")


if __name__ == "__main__":
    main()
