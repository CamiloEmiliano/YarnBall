"""
SEC Entity Linker & Master CIK Registry Module.

Provides:
- Company name normalization with legal suffix stripping
- Fast 4-tier hierarchical entity resolution against PostgreSQL sec_companies and sec_subsidiaries
- In-memory caching for resolved entities
- Synchronization utility to populate/refresh the SEC master CIK registry
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("edgar_linker")

# Common corporate suffixes to strip during normalization
CORP_SUFFIXES = [
    r"\binc(?:\.|\b)",
    r"\bincorporated\b",
    r"\bcorp(?:\.|\b)",
    r"\bcorporation\b",
    r"\bllc(?:\.|\b)",
    r"\bl\.l\.c\.(?:\.|\b)",
    r"\bltd(?:\.|\b)",
    r"\blimited\b",
    r"\bco(?:\.|\b)",
    r"\bcompany\b",
    r"\bplc(?:\.|\b)",
    r"\bp\.l\.c\.(?:\.|\b)",
    r"\bholdings\b",
    r"\bholding\b",
    r"\bgroup\b",
    r"\bsa\b",
    r"\bs\.a\.(?:\.|\b)",
    r"\bag\b",
    r"\bgmbh\b",
    r"\bnv\b",
    r"\bn\.v\.(?:\.|\b)",
    r"\bthe\b",
    r"\bclass\b",
    r"\bcl(?:\.|\b)",
    r"\bseries\b",
    r"\bcommon stock\b",
    r"\bordinary shares\b",
]

SUFFIX_REGEX = re.compile(
    r"(" + "|".join(CORP_SUFFIXES) + r")",
    flags=re.IGNORECASE,
)


def normalize_company_name(name: Optional[str]) -> str:
    """Normalize a company name by removing punctuation, extra spaces, and common legal suffixes.
    
    Examples:
        'Apple Inc.' -> 'apple'
        'Microsoft Corporation' -> 'microsoft'
        'Taiwan Semiconductor Manufacturing Co., Ltd.' -> 'taiwan semiconductor manufacturing'
    """
    if not name:
        return ""
    
    # 1. Lowercase and replace punctuation with spaces
    text = name.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    
    # 2. Strip legal entity suffixes
    text = SUFFIX_REGEX.sub(" ", text)
    
    # 3. Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


class EdgarEntityLinker:
    """4-Tier Hierarchical Entity Linker for SEC EDGAR and Financial Graph Entities.
    
    Cascade Tiers:
    - Tier 1: Exact Ticker match in `sec_companies`
    - Tier 2: Exact normalized name match in `sec_companies`
    - Tier 3: Trigram similarity match in `sec_companies` (similarity >= threshold)
    - Tier 4: Subsidiary lookup in `sec_subsidiaries` (exact or trigram)
    """

    def __init__(self, pg_conn=None, trigram_threshold: float = 0.88):
        self.pg_conn = pg_conn
        self.trigram_threshold = trigram_threshold
        self._cache: Dict[str, Optional[Dict[str, Any]]] = {}

    def link_entity(
        self,
        name: Optional[str] = None,
        ticker: Optional[str] = None,
        cursor=None,
    ) -> Optional[Dict[str, Any]]:
        """Resolve an entity mention to its canonical SEC master registry record.
        
        Returns dict with keys:
            cik, ticker, company_name, normalized_name, sic, is_sp500, match_tier, match_score, resolved_via
        """
        cache_key = f"{ticker or ''}::{name or ''}".strip(":")
        if not cache_key:
            return None
        
        if cache_key in self._cache:
            return self._cache[cache_key]

        res = self._resolve_db(name=name, ticker=ticker, cursor=cursor)
        self._cache[cache_key] = res
        return res

    def _resolve_db(
        self,
        name: Optional[str] = None,
        ticker: Optional[str] = None,
        cursor=None,
    ) -> Optional[Dict[str, Any]]:
        if not self.pg_conn and cursor is None:
            logger.debug("No database connection provided for EdgarEntityLinker")
            return None

        should_close_cur = False
        if cursor is None:
            cursor = self.pg_conn.cursor()
            should_close_cur = True

        try:
            # --- Tier 1: Exact Ticker Lookup ---
            if ticker:
                clean_ticker = ticker.strip().upper()
                cursor.execute(
                    """
                    SELECT cik, ticker, company_name, normalized_name, sic, sic_description, is_sp500
                    FROM sec_companies
                    WHERE ticker = %s
                    LIMIT 1;
                    """,
                    (clean_ticker,),
                )
                row = cursor.fetchone()
                if row:
                    return {
                        "cik": row[0],
                        "ticker": row[1],
                        "company_name": row[2],
                        "normalized_name": row[3],
                        "sic": row[4],
                        "sic_description": row[5],
                        "is_sp500": bool(row[6]),
                        "match_tier": "tier_1_ticker",
                        "match_score": 1.0,
                        "resolved_via": "exact_ticker",
                    }

            # --- Tier 2: Exact Normalized Name Lookup ---
            if name:
                norm_name = normalize_company_name(name)
                if norm_name:
                    cursor.execute(
                        """
                        SELECT cik, ticker, company_name, normalized_name, sic, sic_description, is_sp500
                        FROM sec_companies
                        WHERE normalized_name = %s
                        LIMIT 1;
                        """,
                        (norm_name,),
                    )
                    row = cursor.fetchone()
                    if row:
                        return {
                            "cik": row[0],
                            "ticker": row[1],
                            "company_name": row[2],
                            "normalized_name": row[3],
                            "sic": row[4],
                            "sic_description": row[5],
                            "is_sp500": bool(row[6]),
                            "match_tier": "tier_2_exact_name",
                            "match_score": 1.0,
                            "resolved_via": "normalized_name",
                        }

                    # --- Tier 3: Trigram Fuzzy Matching ---
                    try:
                        cursor.execute(
                            """
                            SELECT cik, ticker, company_name, normalized_name, sic, sic_description, is_sp500,
                                   similarity(normalized_name, %s) AS score
                            FROM sec_companies
                            WHERE similarity(normalized_name, %s) >= %s
                            ORDER BY score DESC
                            LIMIT 1;
                            """,
                            (norm_name, norm_name, self.trigram_threshold),
                        )
                        row = cursor.fetchone()
                        if row:
                            return {
                                "cik": row[0],
                                "ticker": row[1],
                                "company_name": row[2],
                                "normalized_name": row[3],
                                "sic": row[4],
                                "sic_description": row[5],
                                "is_sp500": bool(row[6]),
                                "match_tier": "tier_3_trigram_fuzzy",
                                "match_score": float(row[7]),
                                "resolved_via": "trigram_similarity",
                            }
                    except Exception as e:
                        logger.debug(f"Trigram query fallback/skipped: {e}")

                    # --- Tier 4: Exhibit 21 Subsidiary Matching ---
                    cursor.execute(
                        """
                        SELECT c.cik, c.ticker, c.company_name, c.normalized_name, c.sic, c.sic_description, c.is_sp500,
                               s.subsidiary_name
                        FROM sec_subsidiaries s
                        JOIN sec_companies c ON s.parent_cik = c.cik
                        WHERE s.normalized_name = %s
                        LIMIT 1;
                        """,
                        (norm_name,),
                    )
                    row = cursor.fetchone()
                    if row:
                        return {
                            "cik": row[0],
                            "ticker": row[1],
                            "company_name": row[2],
                            "normalized_name": row[3],
                            "sic": row[4],
                            "sic_description": row[5],
                            "is_sp500": bool(row[6]),
                            "match_tier": "tier_4_subsidiary",
                            "match_score": 0.95,
                            "resolved_via": f"subsidiary_match:{row[7]}",
                        }

            return None
        finally:
            if should_close_cur and cursor:
                cursor.close()

    def clear_cache(self):
        """Clear resolution cache."""
        self._cache.clear()


def sync_sec_companies_from_sec(pg_conn, sp500_tickers: Optional[List[str]] = None) -> int:
    """Fetch official SEC company_tickers.json and sync into `sec_companies` table.
    
    Endpoint: https://www.sec.gov/files/company_tickers.json
    Header required: User-Agent: YarnBallResearch admin@yarnball.org
    """
    import httpx

    user_agent = os.getenv("SEC_EDGAR_USER_AGENT", "YarnBallResearch admin@yarnball.org")
    headers = {"User-Agent": user_agent}

    url = "https://www.sec.gov/files/company_tickers.json"
    logger.info(f"Downloading master SEC company registry from {url}")

    sp500_set = set(t.strip().upper() for t in (sp500_tickers or []))

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch SEC company tickers: {e}")
        return 0

    records = []
    for item in data.values():
        cik_str = str(item.get("cik_str", "")).zfill(10)
        ticker = str(item.get("ticker", "")).strip().upper()
        title = str(item.get("title", "")).strip()
        norm_title = normalize_company_name(title)
        is_sp500 = ticker in sp500_set

        records.append((cik_str, ticker, title, norm_title, is_sp500))

    if not records:
        return 0

    cur = pg_conn.cursor()
    try:
        # Upsert records
        query = """
        INSERT INTO sec_companies (cik, ticker, company_name, normalized_name, is_sp500, updated_at)
        VALUES (%s, %s, %s, %s, %s, now())
        ON CONFLICT (cik) DO UPDATE SET
            ticker = EXCLUDED.ticker,
            company_name = EXCLUDED.company_name,
            normalized_name = EXCLUDED.normalized_name,
            is_sp500 = EXCLUDED.is_sp500,
            updated_at = now();
        """
        cur.executemany(query, records)
        pg_conn.commit()
        logger.info(f"Successfully synced {len(records)} SEC master company records into sec_companies")
        return len(records)
    finally:
        cur.close()
