"""
SEC Entity Linker & Semantic Filing Parser Module.

Provides:
- Company name normalization with legal corporate suffix stripping
- 4-tier hierarchical entity resolution against PostgreSQL sec_companies & sec_subsidiaries
- SEC master CIK registry synchronization from SEC.gov
- Salient context paragraph selector for relationally dense text
- Local LLM relationship extraction schema and prompt for SEC Form 10-K/8-K sections
"""

from __future__ import annotations

from collections import OrderedDict
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("ingestion.sec.parser")

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
    - Tier 3: Trigram similarity match in `sec_companies` (similarity >= threshold, calibrated to 0.40)
    - Tier 4: Subsidiary lookup in `sec_subsidiaries` (exact or trigram)
    """

    def __init__(
        self,
        pg_conn=None,
        trigram_threshold: float = 0.40,
        max_cache_size: int = 10000,
    ):
        self.pg_conn = pg_conn
        self.trigram_threshold = trigram_threshold
        self.max_cache_size = max_cache_size
        self._cache: OrderedDict[str, Optional[Dict[str, Any]]] = OrderedDict()
        self.tier3_failures_total: int = 0

    def link_entity(
        self,
        name: Optional[str] = None,
        ticker: Optional[str] = None,
        event_date: Optional[str] = None,
        cursor=None,
    ) -> Optional[Dict[str, Any]]:
        """Resolve an entity mention to its canonical SEC master registry record.
        
        Returns dict with keys:
            cik, ticker, company_name, normalized_name, sic, is_sp500, match_tier, match_score, resolved_via
        """
        norm_ticker = ticker.strip().upper() if ticker else ""
        norm_name = normalize_company_name(name) if name else ""
        date_part = f"::{event_date.strip()}" if event_date else ""
        cache_key = f"{norm_ticker}::{norm_name}{date_part}".strip(":")
        if not cache_key:
            return None
        
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        res = self._resolve_db(name=name, ticker=ticker, event_date=event_date, cursor=cursor)
        
        if len(self._cache) >= self.max_cache_size:
            self._cache.popitem(last=False)
        self._cache[cache_key] = res
        return res

    def _resolve_db(
        self,
        name: Optional[str] = None,
        ticker: Optional[str] = None,
        event_date: Optional[str] = None,
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
                        self.tier3_failures_total += 1
                        logger.warning(f"[EDGAR_LINKER_TRGM_ERR] Tier 3 trigram query failed (total_failures={self.tier3_failures_total}): {e}")

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
    """Fetch official SEC company_tickers.json and sync into `sec_companies` table."""
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
        if hasattr(cur, "close"):
            cur.close()


SEC_EXTRACTION_PROMPT = """You are an expert financial SEC filing knowledge graph extractor.
Analyze the following section from an SEC filing (e.g. Form 10-K Item 1 Business / Item 1A Risks, or Form 8-K).
Extract key corporate entities, suppliers, customers, strategic partners, competitors, key products, and material risks.

Output ONLY a valid JSON object matching this schema:
{
    "nodes": [
        {"id": "<Full Company or Entity Name>", "type": "Company|Product|Technology|RiskFactor|Person|RegulatoryBody", "properties": {"ticker": "<TICKER if known>"}}
    ],
    "edges": [
        {"source": "<Source Entity>", "target": "<Target Entity>", "type": "<RELATIONSHIP_TYPE>", "properties": {"nature": "<brief context>"}}
    ]
}

Allowed relationship types:
- SUPPLIES_TO (source sells/supplies goods/services to target)
- CUSTOMER_OF (source buys goods/services from target)
- COMPETES_WITH (source competes directly with target)
- PARTNERED_WITH (strategic alliance, joint venture, distribution agreement)
- LICENSES_TO / LICENSES_FROM (IP or patent licensing)
- OWNS / CONTROLS (equity stake, joint venture)
- EXPOSED_TO_RISK (company exposed to specific operational/geopolitical/supply risk)
- ACQUIRED / MERGED_WITH (M&A events)

Do not include any explanation or markdown formatting. Output raw JSON only."""


def select_salient_sec_context(text: str, max_chars: int = 2500) -> str:
    """Select the most relationally dense paragraphs from SEC filing text (e.g. Item 1 or Item 1A)."""
    if not text or len(text) <= max_chars:
        return text

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\r\n\s*\r\n", text) if len(p.strip()) > 40]
    if not paragraphs:
        return text[:max_chars]

    keywords = [
        r"\bsuppl(y|ier|iers|ies)\b",
        r"\bcustomer(s)?\b",
        r"\bvendor(s)?\b",
        r"\bmanufactur(er|ers|ing)\b",
        r"\bclient(s)?\b",
        r"\bdistribut(or|ors|ion)\b",
        r"\bcompet(e|es|ing|itor|itors|ition)\b",
        r"\bpartner(s|ship|ships)?\b",
        r"\balliance(s)?\b",
        r"\bjoint venture(s)?\b",
        r"\blicens(e|es|ing|or|ee)\b",
        r"\bacqui(re|red|sition|sitions)\b",
        r"\bmerg(e|ed|er|ers)\b",
        r"\bsubsidiar(y|ies)\b",
        r"\bdependen(t|ce)\b",
        r"\bsole source\b|\bsingle source\b",
        r"\breliance\b",
    ]
    pattern = re.compile("|".join(keywords), re.IGNORECASE)

    scored = []
    for idx, p in enumerate(paragraphs):
        matches = len(pattern.findall(p))
        if matches > 0:
            scored.append((matches, idx, p))

    if not scored:
        return text[:max_chars]

    scored.sort(key=lambda x: (-x[0], x[1]))

    selected = []
    total_len = 0
    for _, idx, p in scored:
        if total_len + len(p) + 2 <= max_chars:
            selected.append((idx, p))
            total_len += len(p) + 2
        elif not selected:
            selected.append((idx, p[:max_chars]))
            break

    selected.sort(key=lambda x: x[0])
    return "\n\n".join(p for _, p in selected)


def extract_sec_relationships(text: str, max_chars: int = 2500) -> Dict[str, Any]:
    """Extract entities and relationships from SEC text chunk using local LLM."""
    if not text or not text.strip():
        return {"nodes": [], "edges": []}

    try:
        import httpx
    except ImportError:
        logger.warning("httpx not available; skipping LLM extraction")
        return {"nodes": [], "edges": []}

    from knowledge_graph.graph_store import _clean_json_response, OLLAMA_API_BASE, OLLAMA_MODEL_NAME

    chunk = select_salient_sec_context(text, max_chars=max_chars)
    prompt = f"{SEC_EXTRACTION_PROMPT}\n\nSEC Text Section:\n{chunk}\n\nJSON:"

    gen_url = OLLAMA_API_BASE
    if "/v1" in gen_url:
        gen_url = gen_url.split("/v1")[0].rstrip("/") + "/api/generate"

    try:
        with httpx.Client(timeout=90.0) as client:
            resp = client.post(
                gen_url,
                json={
                    "model": OLLAMA_MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return _clean_json_response(data.get("response", ""))
    except Exception as e:
        logger.error(f"Failed SEC LLM extraction call: {e}")
        return {"nodes": [], "edges": []}
