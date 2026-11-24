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

    def _resolve_offline(
        self,
        name: Optional[str] = None,
        ticker: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Offline fallback against static S&P 500 JSON registry when Postgres is unavailable."""
        from ingestion.sec.disruption_gating import _load_sp500_identity_index

        by_ticker, _ = _load_sp500_identity_index()
        if not by_ticker:
            return None

        if ticker:
            clean_ticker = ticker.strip().upper()
            hit = by_ticker.get(clean_ticker)
            if hit:
                return {
                    "cik": hit["cik"],
                    "ticker": hit["ticker"],
                    "company_name": hit["company_name"],
                    "normalized_name": normalize_company_name(hit["company_name"]),
                    "sic": None,
                    "sic_description": None,
                    "is_sp500": True,
                    "match_tier": "offline_tier_1_ticker",
                    "match_score": 1.0,
                    "resolved_via": "offline_sp500_registry",
                }

        if name:
            norm_input = normalize_company_name(name)
            if norm_input:
                for entry in by_ticker.values():
                    if normalize_company_name(entry["company_name"]) == norm_input:
                        return {
                            "cik": entry["cik"],
                            "ticker": entry["ticker"],
                            "company_name": entry["company_name"],
                            "normalized_name": norm_input,
                            "sic": None,
                            "sic_description": None,
                            "is_sp500": True,
                            "match_tier": "offline_tier_2_name",
                            "match_score": 1.0,
                            "resolved_via": "offline_sp500_registry",
                        }
        return None

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
            return self._resolve_offline(name=name, ticker=ticker)

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


from ontology.schema import format_extraction_prompt_vocab

_SEC_ALLOWED_LABELS, _SEC_ALLOWED_RELS = format_extraction_prompt_vocab()

SEC_EXTRACTION_PROMPT = f"""You are an expert financial SEC filing knowledge graph extractor.
Analyze the following section from an SEC filing (e.g. Form 10-K Item 1 Business / Item 1A Risks, or Form 8-K).
Extract key corporate entities, suppliers, customers, strategic partners, competitors, key products, and material risks.

Output ONLY a valid JSON object matching this schema:
{{
    "nodes": [
        {{"id": "<Full Company or Entity Name>", "type": "<{_SEC_ALLOWED_LABELS}>", "properties": {{"ticker": "<TICKER if known>"}}}}
    ],
    "edges": [
        {{"source": "<Source Entity>", "target": "<Target Entity>", "type": "<RELATIONSHIP_TYPE>", "properties": {{"nature": "<brief context>", "input_group": "<component or service group>", "share": "<optional 0.0-1.0 procurement/revenue share ONLY if explicitly stated>", "buffer_days": "<optional inventory buffer days ONLY if explicitly stated>", "sole_source": "<true ONLY if explicitly sole/single source>"}}}}
    ]
}}

Allowed relationship types: {_SEC_ALLOWED_RELS}.
Do not include any explanation or markdown formatting. Output raw JSON only."""


_SALIENT_KEYWORDS: Tuple[str, ...] = (
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
)
_SALIENT_PATTERN = re.compile("|".join(_SALIENT_KEYWORDS), re.IGNORECASE)

_HIGH_SIGNAL_GLEANING_RE = re.compile(
    r"\bitem\s*(?:1(?:\s*[-–—.\s]+business)?\b|1a\b|1\.01\b|1\.02\b|2\.01\b|8\.01\b)",
    re.IGNORECASE,
)


def score_chunk_salience(text: str) -> int:
    """Count relational supply-chain/corporate cue matches for chunk priority ranking (ADR-00035)."""
    if not text:
        return 0
    return len(_SALIENT_PATTERN.findall(text))


def rank_salient_chunks(chunks: List[Any]) -> List[Any]:
    """Order chunks descending by relational salience without dropping or truncating any chunk (Invariant #4)."""
    indexed = list(enumerate(chunks))
    indexed.sort(key=lambda pair: (-score_chunk_salience(getattr(pair[1], "text", "")), pair[0]))
    return [chunk for _, chunk in indexed]


def is_high_signal_gleaning_section(section_name: Optional[str]) -> bool:
    """Return True for high-signal SEC sections eligible for targeted 1-pass gleaning (ADR-00035 Section 4.4)."""
    if not section_name:
        return False
    raw = str(section_name).strip()
    if re.search(r"\bitem\s*1b\b", raw, re.IGNORECASE):
        return False
    return bool(_HIGH_SIGNAL_GLEANING_RE.search(raw))


def should_glean_chunk(
    section_name: Optional[str],
    pass1_valid_edge_count: int,
    generated_tokens: int,
    runaway_detected: bool,
    *,
    max_new_tokens: int = 512,
    enabled: Optional[bool] = None,
) -> bool:
    """Determine whether a chunk qualifies for targeted 1-pass gleaning (ADR-00035 Section 4.4)."""
    if enabled is None:
        env_val = os.getenv("ENABLE_TARGETED_GLEANING", "1").strip().lower()
        is_enabled = env_val not in ("0", "false", "no", "off")
    else:
        is_enabled = bool(enabled)

    if not is_enabled or runaway_detected:
        return False
    if not is_high_signal_gleaning_section(section_name):
        return False
    token_threshold = int(0.85 * max_new_tokens)
    return pass1_valid_edge_count >= 1 or int(generated_tokens) >= token_threshold


def select_salient_sec_context(text: str, max_chars: int = 2500) -> str:
    """Legacy salience helper retained for backward-compatible tests; runtime extractors use `rank_salient_chunks`."""
    if not text or len(text) <= max_chars:
        return text

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\r\n\s*\r\n", text) if len(p.strip()) > 40]
    if not paragraphs:
        return text[:max_chars]

    scored = []
    for idx, p in enumerate(paragraphs):
        matches = score_chunk_salience(p)
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


def _format_gleaning_prompt(chunk_prompt_text: str, pass1_edges: List[Dict[str, Any]]) -> str:
    """Build compact Pass-2 targeted gleaning prompt with explicit empty-JSON stop condition (ADR-00035)."""
    triple_lines = [
        f"- ({e.get('source', '')}, {e.get('type', 'RELATED_TO')}, {e.get('target', '')})"
        for e in pass1_edges
    ]
    summary = "\n".join(triple_lines) if triple_lines else "(none extracted in Pass 1)"
    return (
        f"SEC Text Chunk:\n{chunk_prompt_text}\n\n"
        f"Pass-1 Already Extracted Triples:\n{summary}\n\n"
        "Review the SEC text chunk above for any ADDITIONAL explicitly stated corporate entities or "
        "relationships that were missed in Pass 1 (such as secondary suppliers, customers, or risk dependencies). "
        'Do NOT repeat triples already listed above. If no additional verified relationships are stated, output exactly: {"nodes": [], "edges": []}\n\n'
        "JSON:"
    )


def extract_sec_relationships(
    text: str,
    max_chars: Optional[int] = None,
    model_client: Optional[Any] = None,
    rejection_queue: Optional[Any] = None,
    source_id: Optional[str] = None,
    *,
    doc_id: Optional[str] = None,
    cik: Optional[str] = None,
    ticker: Optional[str] = None,
    company_name: Optional[str] = None,
    form_type: str = "10-K",
    filing_date: Optional[str] = None,
    section_name: Optional[str] = None,
    max_tokens: int = 800,
    overlap_tokens: int = 120,
    enable_gleaning: Optional[bool] = None,
) -> Dict[str, Any]:
    """Segment full SEC section text without truncation, extract per chunk, glean, and merge (ADR-00035)."""
    del max_chars  # Intentionally unused: ADR-00035 Invariant #4 forbids character-budget truncation.
    if not text or not text.strip():
        return {"nodes": [], "edges": []}

    from ingestion.segmentation.chunker import FilingMetadata, FinancialSectionSegmenter
    from knowledge_graph.graph_store import (
        GraphExtractionError,
        _quarantine_extraction_failure,
        merge_chunk_extraction_payloads,
        run_structured_extraction_with_telemetry,
    )
    from ontology.validation import validate_graph_payload
    from sft_pipeline.rejection_queue import ErrorStage

    resolved_doc_id = doc_id
    resolved_section = section_name
    if source_id and ":" in source_id:
        src_prefix, src_sec = source_id.split(":", 1)
        if not resolved_doc_id and src_prefix.strip():
            resolved_doc_id = src_prefix.strip()
        if not resolved_section and src_sec.strip():
            resolved_section = src_sec.strip()
    elif source_id and not resolved_doc_id:
        resolved_doc_id = source_id.strip()

    meta = FilingMetadata(
        cik=cik or "0000000000",
        ticker=ticker,
        company_name=company_name,
        form_type=form_type,
        filing_date=filing_date,
        doc_id=resolved_doc_id,
    )
    segmenter = FinancialSectionSegmenter(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    raw_chunks = segmenter.segment_filing(text, meta, default_section_name=resolved_section)
    ranked_chunks = rank_salient_chunks(raw_chunks)

    chunk_payloads: List[Dict[str, Any]] = []
    gleaning_new_edges_count = 0
    first_error: Optional[GraphExtractionError] = None

    for chunk in ranked_chunks:
        try:
            pass1_raw, pass1_resp = run_structured_extraction_with_telemetry(
                prompt=f"SEC Text Section:\n{chunk.full_prompt_text}\n\nJSON:",
                task="EXTRACT_SEC_GRAPH",
                system_prompt=SEC_EXTRACTION_PROMPT,
                model_client=model_client,
                rejection_queue=rejection_queue,
                source_id=source_id or chunk.chunk_id,
                chunk_id=chunk.chunk_id,
                timeout_sec=90.0,
            )
        except GraphExtractionError as exc:
            if first_error is None:
                first_error = exc
            continue

        # Invariant #3: Ontology gate negation/hedge checks run strictly against `chunk.text`
        pass1_valid = validate_graph_payload(
            pass1_raw,
            source_id=chunk.chunk_id,
            rejection_queue=rejection_queue,
            strict=True,
            default_node_type="Company",
            source_text=chunk.text,
        )
        for edge in pass1_valid.get("edges", []):
            props = dict(edge.get("properties") or {})
            props["evidence_chunk_ids"] = sorted(
                {*(props.get("evidence_chunk_ids") or []), chunk.chunk_id}
            )
            props.setdefault("extraction_pass", 1)
            edge["properties"] = props

        combined_chunk_payloads = [
            {
                "chunk_id": chunk.chunk_id,
                "nodes": pass1_valid.get("nodes", []),
                "edges": pass1_valid.get("edges", []),
            }
        ]

        if should_glean_chunk(
            chunk.section_name,
            len(pass1_valid.get("edges", [])),
            getattr(pass1_resp, "generated_tokens", 0),
            bool(getattr(pass1_resp, "runaway_detected", False)),
            enabled=enable_gleaning,
        ):
            glean_prompt = _format_gleaning_prompt(chunk.full_prompt_text, pass1_valid.get("edges", []))
            try:
                pass2_raw, _ = run_structured_extraction_with_telemetry(
                    prompt=glean_prompt,
                    task="EXTRACT_SEC_GRAPH",
                    system_prompt=SEC_EXTRACTION_PROMPT,
                    model_client=model_client,
                    rejection_queue=None,  # Handled non-destructively with GLEANING_PARSE_FAILED below
                    source_id=chunk.chunk_id,
                    chunk_id=chunk.chunk_id,
                    timeout_sec=90.0,
                    quarantine_on_error=False,
                )
                pass2_valid = validate_graph_payload(
                    pass2_raw,
                    source_id=f"{chunk.chunk_id}:glean",
                    rejection_queue=rejection_queue,
                    strict=True,
                    default_node_type="Company",
                    source_text=chunk.text,
                )
                pass1_keys = {
                    (
                        str(e.get("source", "")).strip().lower(),
                        str(e.get("type", "")).strip().upper(),
                        str(e.get("target", "")).strip().lower(),
                    )
                    for e in pass1_valid.get("edges", [])
                }
                rescued_edges: List[Dict[str, Any]] = []
                for edge in pass2_valid.get("edges", []):
                    ekey = (
                        str(edge.get("source", "")).strip().lower(),
                        str(edge.get("type", "")).strip().upper(),
                        str(edge.get("target", "")).strip().lower(),
                    )
                    if ekey in pass1_keys:
                        continue
                    props = dict(edge.get("properties") or {})
                    props["evidence_chunk_ids"] = sorted(
                        {*(props.get("evidence_chunk_ids") or []), chunk.chunk_id}
                    )
                    props["extraction_pass"] = 2
                    edge["properties"] = props
                    rescued_edges.append(edge)
                if rescued_edges or pass2_valid.get("nodes"):
                    gleaning_new_edges_count += len(rescued_edges)
                    combined_chunk_payloads.append(
                        {
                            "chunk_id": chunk.chunk_id,
                            "nodes": pass2_valid.get("nodes", []),
                            "edges": rescued_edges,
                        }
                    )
            except GraphExtractionError as glean_exc:
                logger.warning(
                    "Targeted gleaning Pass 2 failed non-destructively for %s (Pass 1 retained): %s",
                    chunk.chunk_id,
                    glean_exc,
                )
                _quarantine_extraction_failure(
                    task_id="EXTRACT_SEC_GRAPH",
                    prompt=glean_prompt,
                    raw_completion=str(glean_exc),
                    error_stage=ErrorStage.GLEANING_PARSE_FAILED,
                    error_message=f"Pass-2 gleaning failed (Pass 1 preserved): {glean_exc}",
                    source_id=chunk.chunk_id,
                    rejection_queue=rejection_queue,
                    metadata={"chunk_id": chunk.chunk_id, "extraction_pass": 2},
                )

        chunk_payloads.extend(combined_chunk_payloads)

    if not chunk_payloads and first_error is not None:
        raise first_error

    merged = merge_chunk_extraction_payloads(chunk_payloads)
    if gleaning_new_edges_count > 0:
        logger.info(
            "SEC chunk extraction completed (%d chunks, gleaning_new_edges_count=%d)",
            len(ranked_chunks),
            gleaning_new_edges_count,
        )
    merged["gleaning_new_edges_count"] = gleaning_new_edges_count
    merged["chunk_ids"] = [c.chunk_id for c in raw_chunks]
    return merged
