"""SEC Form 8-K Boilerplate Sanitization & Disruption Gating (ADR-00014).

Provides stateless, linear-time, deterministic preprocessing for raw SEC Form 8-K
disclosures prior to supply-chain cascade propagation and LLM reasoning:
1. Paragraph-scoped removal of EDGAR cover pages, PSLRA Safe Harbor risk
   enumerations, Exchange Act signatures, and Item 9.01 exhibit indices.
2. Deterministic priority-tier resolution across multi-item Form 8-K filings.
3. Statutory anchoring and zero-shock negative-control gating to prevent
   false-positive supply shocks or synthetic fallback escalation.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from pydantic import BaseModel, Field

from ingestion.sec.canon import SEC_8K_PRIORITY_TIERS, SECForm8KItem
from retrieval.reporting import OperationalDisruptionCategory

DEFAULT_MAX_FILING_CHARS: int = 50_000
DEFAULT_SHOCK_MAGNITUDE_PCT: float = 100.0
DEFAULT_SHOCK_DURATION_DAYS: float = 60.0

# Pre-compiled linear-time structural splitters (no nested quantifiers).
_SIGNATURE_CUTOFF_RE = re.compile(
    r"\bpursuant\s+to\s+the\s+requirements\s+of\s+the\s+securities\s+exchange\s+act\s+of\s+1934\b",
    re.IGNORECASE,
)
_FENCED_EXHIBIT_HEADER_RE = re.compile(
    r"(?:^|\r?\n)[ \t]*(?:"
    r"[-=_*]{2,}[ \t]*EX(?:HIBIT)?[- \t]+99(?:\.\d+)?\b[^\r\n]*|"
    r"<TYPE>[ \t]*EX-99(?:\.\d+)?\b[^\r\n]*"
    r")(?=\r?\n|$)",
    re.IGNORECASE,
)
_ATTACHED_EXHIBIT_HEADER_RE = re.compile(
    r"(?:^|\r?\n)[ \t]*(?:"
    r"[-=_*]{2,}[ \t]*EX(?:HIBIT)?[- \t]+99(?:\.\d+)?\b[^\r\n]*|"
    r"EX(?:HIBIT)?[- \t]+99(?:\.\d+)?(?:[ \t]*[-=_*]{2,})?[ \t]*|"
    r"<TYPE>[ \t]*EX-99(?:\.\d+)?\b[^\r\n]*"
    r")(?=\r?\n|$)",
    re.IGNORECASE,
)
_EXHIBIT_INDEX_PREDECESSOR_RE = re.compile(
    r"(?:exhibit\s+index\b|exhibit\s+no\.?\b|exhibit\s+number\b|item\s+9\.01\b)",
    re.IGNORECASE,
)
_XBRL_EXHIBIT_KEY_RE = re.compile(r"^(?:ex[- \t]*)?(?:101|104)(?:\b|[._-])", re.IGNORECASE)
_INLINE_SECTION_SPLIT_RE = re.compile(
    r"\r?\n(?=\s*(?:item\s+\d+\.\d{2}\b|forward-looking\s+statements|cautionary\s+(?:note|statement)\b|"
    r"safe\s+harbor\b|signatures?\b))",
    re.IGNORECASE,
)
_INLINE_SAFE_HARBOR_RE = re.compile(
    r"(?=\b(?:cautionary\s+(?:note|statement)\s+(?:regarding|concerning)\s+forward-looking\s+statements|"
    r"forward-looking\s+statements\s*:|"
    r"safe\s+harbor\s+statement\s*:))",
    re.IGNORECASE,
)
_PARAGRAPH_SPLIT_RE = re.compile(r"\r?\n\s*\r?\n")
_ITEM_CODE_RE = re.compile(r"\b(?:item\s+)?(\d+\.\d{2})\b", re.IGNORECASE)

_COVER_PAGE_MARKERS: Tuple[str, ...] = (
    "check the appropriate box",
    "commission file number",
    "emerging growth company",
    "indicate by check mark whether",
    "securities registered pursuant to section 12",
    "exact name of registrant as specified in its charter",
    "state or other jurisdiction of incorporation",
    "irs employer identification no",
    "registrant's telephone number",
)

_SAFE_HARBOR_STRONG_MARKERS: Tuple[str, ...] = (
    "private securities litigation reform act",
    "cautionary statement regarding forward-looking",
    "cautionary note regarding forward-looking",
    "cautionary statement concerning forward-looking",
    "forward-looking statements within the meaning",
    "actual results may differ materially",
    "safe harbor statement",
)
_SAFE_HARBOR_HEADER_RE = re.compile(
    r"^\s*(?:cautionary\s+(?:note|statement)\s+(?:regarding|concerning)\s+)?"
    r"(?:forward-looking\s+statements|safe\s+harbor)\b",
    re.IGNORECASE,
)
_SAFE_HARBOR_QUALIFIERS: Tuple[str, ...] = (
    "differ materially",
    "risks and uncertainties",
    "no obligation",
    "undertake no obligation",
    "as of the date",
)
_EXHIBIT_INDEX_RE = re.compile(
    r"^\s*(?:item\s+9\.01\b|exhibits\s+and\s+financial\s+statement\s+schedules\b|exhibit\s+no\.?\b)",
    re.IGNORECASE,
)
_SIGNATURES_HEADER_RE = re.compile(r"^\s*signatures?\s*$", re.IGNORECASE)

# Canonical Priority Tiers (ADR-00014 Section 4.2, consolidated in ingestion/sec/canon.py per ADR-00045)
_TIER_ITEMS: Tuple[Tuple[int, Tuple[str, ...]], ...] = SEC_8K_PRIORITY_TIERS

_ITEM_PRIORITY_INDEX: Dict[str, Tuple[int, int]] = {}
_CODE_TO_CANONICAL_ITEM: Dict[str, str] = {}
for _tier, _items in _TIER_ITEMS:
    for _rank, _item_val in enumerate(_items):
        _ITEM_PRIORITY_INDEX[_item_val] = (_tier, _rank)
        _code = _item_val.split()[-1]
        _CODE_TO_CANONICAL_ITEM[_code] = _item_val

# Ordered lexical triggers over sanitized text (word-boundary guarded, linear time).
_LEXICAL_DISRUPTION_PATTERNS: Tuple[Tuple[re.Pattern[str], OperationalDisruptionCategory], ...] = (
    (
        re.compile(r"\b(?:fire|fires|explosion|explosions|blast)\b", re.IGNORECASE),
        OperationalDisruptionCategory.FACILITY_FIRE_OR_EXPLOSION,
    ),
    (
        re.compile(
            r"\b(?:flood|flooding|earthquake|hurricane|typhoon|tornado|force\s+majeure)\b",
            re.IGNORECASE,
        ),
        OperationalDisruptionCategory.NATURAL_DISASTER_FORCE_MAJEURE,
    ),
    (
        re.compile(
            r"\b(?:power\s+(?:outage|failure|loss|grid|shortage)|grid\s+failure|blackout|utility\s+outage)\b",
            re.IGNORECASE,
        ),
        OperationalDisruptionCategory.UTILITY_POWER_GRID_FAILURE,
    ),
    (
        re.compile(
            r"\b(?:ransomware|cyberattack|cyber\s+incident|cybersecurity\s+incident|"
            r"network\s+intrusion|unauthorized\s+access)\b",
            re.IGNORECASE,
        ),
        OperationalDisruptionCategory.RANSOMWARE_CYBER_INFRASTRUCTURE_HALT,
    ),
    (
        re.compile(
            r"\b(?:equipment\s+breakdown|machinery\s+failure|fab(?:rication)?\s+(?:halt|shutdown)|"
            r"emergency\s+(?:halt|shutdown)|operational\s+halt|production\s+(?:halt|stoppage|curtailment)|"
            r"plant\s+shutdown|facility\s+(?:shutdown|disruption)|cleanroom\s+(?:halt|contamination|outage))\b",
            re.IGNORECASE,
        ),
        OperationalDisruptionCategory.EQUIPMENT_BREAKDOWN_FAB_HALT,
    ),
    (
        re.compile(
            r"\b(?:supply\s+(?:agreement|contract)\s+(?:terminated|termination)|"
            r"terminat(?:ed|ion)\s+of\s+(?:the\s+)?(?:supply|manufacturing|foundry)\s+(?:agreement|contract)|"
            r"contract\s+severance)\b",
            re.IGNORECASE,
        ),
        OperationalDisruptionCategory.PRIMARY_SUPPLY_CONTRACT_TERMINATED,
    ),
    (
        re.compile(
            r"\b(?:port\s+(?:closure|congestion|blockage|strike|halt)|canal\s+blockage|"
            r"maritime\s+(?:disruption|blockade|chokepoint)|shipping\s+suspension)\b",
            re.IGNORECASE,
        ),
        OperationalDisruptionCategory.MARITIME_CANAL_PORT_LOGISTICS_CHOKEPOINT,
    ),
    (
        re.compile(
            r"\b(?:product\s+recall|safety\s+recall|fda\s+(?:hold|warning|suspension)|"
            r"clinical\s+hold|regulatory\s+halt)\b",
            re.IGNORECASE,
        ),
        OperationalDisruptionCategory.REGULATORY_SAFETY_RECALL_OR_FDA_HOLD,
    ),
    (
        re.compile(r"\b(?:labor\s+strike|union\s+strike|work\s+stoppage|walkout)\b", re.IGNORECASE),
        OperationalDisruptionCategory.LABOR_UNION_STRIKE_WALKOUT,
    ),
    (
        re.compile(
            r"\b(?:plant\s+closure|facility\s+closure|cease\s+operations\s+at\s+(?:its|the)\s+(?:plant|facility|fab))\b",
            re.IGNORECASE,
        ),
        OperationalDisruptionCategory.PERMANENT_FACILITY_CLOSURE_EXIT,
    ),
)

_SUPPLY_CONTEXT_RE = re.compile(
    r"\b(?:supplier|supply|vendor|manufacturing|foundry|wafer|procurement|component|fab)\b",
    re.IGNORECASE,
)

# Statutory anchoring for Tier 1 acute disruption items when text/metadata implies disruption.
_TIER_1_STATUTORY_CATEGORIES: Dict[str, OperationalDisruptionCategory] = {
    SECForm8KItem.ITEM_1_05.value: OperationalDisruptionCategory.RANSOMWARE_CYBER_INFRASTRUCTURE_HALT,
    SECForm8KItem.ITEM_1_03.value: OperationalDisruptionCategory.COMMERCIAL_INSOLVENCY_BANKRUPTCY,
    SECForm8KItem.ITEM_1_04.value: OperationalDisruptionCategory.EQUIPMENT_BREAKDOWN_FAB_HALT,
    SECForm8KItem.ITEM_2_04.value: OperationalDisruptionCategory.DEBT_COVENANT_DEFAULT_CREDIT_FREEZE,
    SECForm8KItem.ITEM_2_05.value: OperationalDisruptionCategory.PERMANENT_FACILITY_CLOSURE_EXIT,
    SECForm8KItem.ITEM_2_06.value: OperationalDisruptionCategory.PERMANENT_FACILITY_CLOSURE_EXIT,
}


class SanitizedDisruptionAssessment(BaseModel):
    """Structured gating and classification verdict produced prior to cascade propagation."""

    primary_sec_item: str = Field(..., description="Canonical SEC Form 8-K item selected via tier precedence")
    item_priority_tier: int = Field(..., ge=1, le=4, description="Statutory priority tier (1=Acute .. 4=Procedural)")
    is_material_operational_disruption: bool = Field(
        ...,
        description="True when statutory item or sanitized narrative evidences a material operational shock",
    )
    disruption_category: OperationalDisruptionCategory = Field(
        ...,
        description="Operational disruption mechanism classified strictly over sanitized text and item code",
    )
    shock_magnitude_pct: float = Field(..., ge=0.0, le=100.0)
    shock_duration_days: float = Field(..., ge=0.0)
    sanitized_headline: str = Field(...)
    sanitized_content: str = Field(...)
    stripped_paragraph_count: int = Field(..., ge=0)
    gating_reason: str = Field(...)


def _is_cover_or_exhibit_boilerplate(paragraph: str, lower: str) -> bool:
    """Identifies EDGAR cover-page metadata, signature headers, and Item 9.01 exhibit indices."""
    if any(marker in lower for marker in _COVER_PAGE_MARKERS):
        return True
    if _SIGNATURES_HEADER_RE.match(paragraph):
        return True
    if _EXHIBIT_INDEX_RE.match(paragraph):
        return True
    if "exhibit no" in lower and "description" in lower:
        return True
    return False


def _is_safe_harbor_paragraph(paragraph: str, lower: str) -> bool:
    """Identifies PSLRA Forward-Looking Statement disclaimer paragraphs."""
    if any(marker in lower for marker in _SAFE_HARBOR_STRONG_MARKERS):
        return True
    if _SAFE_HARBOR_HEADER_RE.match(paragraph):
        return True
    if "forward-looking statements" in lower and any(q in lower for q in _SAFE_HARBOR_QUALIFIERS):
        return True
    return False


def _is_exhibit_index_predecessor(prefix: str) -> bool:
    """Returns True when the immediately preceding non-empty line is an Exhibit Index header."""
    for line in reversed(prefix.splitlines()):
        stripped = line.strip()
        if stripped:
            return bool(_EXHIBIT_INDEX_PREDECESSOR_RE.search(stripped))
    return False


def _split_document_segments(raw_text: str) -> Tuple[List[str], int]:
    """Splits combined primary 8-K + attached Exhibit 99.x text without discarding exhibits at signatures.

    Returns:
        Tuple of (ordered document segments ready for independent per-document bounding & sanitization,
        number of Exchange Act signature sections stripped during splitting).
    """
    # 1. Split first on explicitly fenced or EDGAR-tagged Exhibit 99.x document boundaries.
    fenced_matches = list(_FENCED_EXHIBIT_HEADER_RE.finditer(raw_text))
    if fenced_matches:
        coarse_docs: List[str] = []
        first_chunk = raw_text[: fenced_matches[0].start()]
        if first_chunk.strip():
            coarse_docs.append(first_chunk)
        for idx, match in enumerate(fenced_matches):
            end_pos = fenced_matches[idx + 1].start() if idx + 1 < len(fenced_matches) else len(raw_text)
            body = raw_text[match.end() : end_pos]
            if body.strip():
                coarse_docs.append(body)
    else:
        coarse_docs = [raw_text]

    # 2. Within each document segment, truncate its Exchange Act signature tail while preserving any
    #    standalone attached Exhibit 99.x document that follows the signature block.
    segments: List[str] = []
    signatures_stripped = 0
    for doc in coarse_docs:
        sig_match = _SIGNATURE_CUTOFF_RE.search(doc)
        if not sig_match:
            segments.append(doc)
            continue

        signatures_stripped += 1
        head = doc[: sig_match.start()]
        if head.strip():
            segments.append(head)

        tail = doc[sig_match.end() :]
        valid_exhibits: List[re.Match[str]] = []
        for ex_match in _ATTACHED_EXHIBIT_HEADER_RE.finditer(tail):
            header_text = ex_match.group(0).strip()
            is_fenced_or_tagged = header_text.startswith(("-", "=", "_", "*", "<"))
            if not is_fenced_or_tagged and _is_exhibit_index_predecessor(tail[: ex_match.start()]):
                continue
            valid_exhibits.append(ex_match)

        for idx, ex_match in enumerate(valid_exhibits):
            end_pos = valid_exhibits[idx + 1].start() if idx + 1 < len(valid_exhibits) else len(tail)
            ex_body = tail[ex_match.end() : end_pos]
            if ex_body.strip():
                segments.append(ex_body)

    return segments, signatures_stripped


def _sanitize_single_document(raw_doc: str, max_chars: int = DEFAULT_MAX_FILING_CHARS) -> Tuple[str, int]:
    """Bounds and sanitizes a single document segment (primary 8-K or one attached exhibit)."""
    if not raw_doc:
        return "", 0

    bounded = raw_doc[:max_chars] if max_chars > 0 else raw_doc
    stripped_count = 0

    # Strip any remaining Exchange Act signature tail inside this individual document segment.
    sig_match = _SIGNATURE_CUTOFF_RE.search(bounded)
    if sig_match:
        bounded = bounded[: sig_match.start()]
        stripped_count += 1

    # Split monolithic or single-newline blocks on explicit disclaimer/exhibit headers.
    expanded = _INLINE_SECTION_SPLIT_RE.sub("\n\n", bounded)
    paragraphs = _PARAGRAPH_SPLIT_RE.split(expanded)

    kept_segments: List[str] = []
    for raw_para in paragraphs:
        para = raw_para.strip()
        if not para:
            continue

        # Handle unformatted plain-text blocks gluing operational prose to a Safe Harbor tail.
        sub_blocks = _INLINE_SAFE_HARBOR_RE.split(para)
        for sub in sub_blocks:
            segment = sub.strip()
            if not segment:
                continue
            lower = segment.lower()
            if _is_cover_or_exhibit_boilerplate(segment, lower) or _is_safe_harbor_paragraph(segment, lower):
                stripped_count += 1
                continue
            kept_segments.append(segment)

    return "\n\n".join(kept_segments), stripped_count


def _sanitize_with_stats(raw_text: str, max_chars: int = DEFAULT_MAX_FILING_CHARS) -> Tuple[str, int]:
    """Single-pass linear sanitization returning cleaned text and stripped block count.

    Attached Exhibit 99.x documents appearing after the primary Exchange Act signature block
    are split into independent segments before per-document `max_chars` bounding and
    boilerplate/PSLRA Safe Harbor stripping.
    """
    if not raw_text:
        return "", 0

    segments, stripped_count = _split_document_segments(raw_text)
    cleaned_docs: List[str] = []
    for segment in segments:
        cleaned, seg_stripped = _sanitize_single_document(segment, max_chars=max_chars)
        stripped_count += seg_stripped
        if cleaned:
            cleaned_docs.append(cleaned)

    return "\n\n".join(cleaned_docs), stripped_count


def extract_8k_exhibit_texts(payload: Mapping[str, Any]) -> List[str]:
    """Extracts attached narrative exhibit texts (e.g., Exhibit 99.1) from a Form 8-K payload."""
    collected: List[str] = []
    seen: set[str] = set()

    def _add(raw_val: Any) -> None:
        if not isinstance(raw_val, str):
            return
        stripped = raw_val.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            collected.append(raw_val)

    def _collect_from_mapping(source: Mapping[str, Any]) -> None:
        raw_list = source.get("exhibit_texts")
        if isinstance(raw_list, (list, tuple)):
            for item in raw_list:
                _add(item)

        for key in ("exhibit_99_1_text", "exhibit_99_1"):
            _add(source.get(key))

        raw_exhibits = source.get("exhibits")
        if isinstance(raw_exhibits, Mapping):
            for ex_key, ex_val in raw_exhibits.items():
                if _XBRL_EXHIBIT_KEY_RE.match(str(ex_key).strip()):
                    continue
                if isinstance(ex_val, str):
                    _add(ex_val)
                elif isinstance(ex_val, Mapping):
                    text_val = ex_val.get("text") if ex_val.get("text") is not None else ex_val.get("content")
                    _add(text_val)
        elif isinstance(raw_exhibits, (list, tuple)):
            for entry in raw_exhibits:
                if isinstance(entry, str):
                    _add(entry)
                elif isinstance(entry, Mapping):
                    ex_id = str(
                        entry.get("exhibit_type")
                        or entry.get("type")
                        or entry.get("exhibit_no")
                        or entry.get("key")
                        or ""
                    ).strip()
                    if ex_id and _XBRL_EXHIBIT_KEY_RE.match(ex_id):
                        continue
                    text_val = entry.get("text") if entry.get("text") is not None else entry.get("content")
                    _add(text_val)

    _collect_from_mapping(payload)
    meta = payload.get("metadata")
    if isinstance(meta, Mapping):
        _collect_from_mapping(meta)

    return collected


def sanitize_sec_8k_text(raw_text: str, max_chars: int = DEFAULT_MAX_FILING_CHARS) -> str:
    """Strips EDGAR cover-page boilerplate, PSLRA Safe Harbor blocks, signatures, and exhibit tables."""
    sanitized, _ = _sanitize_with_stats(raw_text, max_chars=max_chars)
    return sanitized


def get_sec_8k_item_tier(item: str) -> int:
    """Returns statutory priority tier (1=Acute Disruption .. 4=Governance/Procedural)."""
    match = _ITEM_CODE_RE.search(item or "")
    if match:
        canonical = _CODE_TO_CANONICAL_ITEM.get(match.group(1))
        if canonical and canonical in _ITEM_PRIORITY_INDEX:
            return _ITEM_PRIORITY_INDEX[canonical][0]
    return 4 if "9.01" in (item or "") else 2


def select_primary_sec_8k_item(items: Sequence[str]) -> str:
    """Selects the highest-precedence canonical Form 8-K item across priority tiers."""
    best_item: Optional[str] = None
    best_key: Optional[Tuple[int, int]] = None

    for raw_item in items:
        match = _ITEM_CODE_RE.search(str(raw_item))
        if not match:
            continue
        canonical = _CODE_TO_CANONICAL_ITEM.get(match.group(1))
        if not canonical:
            continue
        rank_key = _ITEM_PRIORITY_INDEX[canonical]
        if best_key is None or rank_key < best_key:
            best_key = rank_key
            best_item = canonical

    return best_item if best_item is not None else SECForm8KItem.ITEM_8_01.value


def _classify_sanitized_text(text: str) -> Optional[OperationalDisruptionCategory]:
    """Matches first physical/commercial disruption trigger over already-sanitized narrative."""
    if not text:
        return None
    for pattern, category in _LEXICAL_DISRUPTION_PATTERNS:
        if pattern.search(text):
            return category
    return None


def _parse_optional_positive_float(metadata: Mapping[str, Any], key: str) -> Optional[float]:
    val = metadata.get(key)
    if val is None or isinstance(val, bool):
        return None
    try:
        parsed = float(val)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0.0 else None


def assess_sec_8k_disruption(
    items: Sequence[str],
    headline: str,
    content: str,
    metadata: Optional[Mapping[str, Any]] = None,
    max_chars: int = DEFAULT_MAX_FILING_CHARS,
    exhibit_texts: Optional[Sequence[str]] = None,
) -> SanitizedDisruptionAssessment:
    """Executes boilerplate sanitization, tier resolution, and negative-control gating."""
    meta = metadata or {}
    primary_item = select_primary_sec_8k_item(items)
    tier = get_sec_8k_item_tier(primary_item)

    sanitized_headline, h_stripped = _sanitize_with_stats(headline or "", max_chars=max_chars)
    sanitized_primary, c_stripped = _sanitize_with_stats(content or "", max_chars=max_chars)
    total_stripped = h_stripped + c_stripped

    resolved_exhibits: List[str] = (
        [ex for ex in exhibit_texts if isinstance(ex, str) and ex.strip()]
        if exhibit_texts is not None
        else (extract_8k_exhibit_texts(meta) if meta else [])
    )
    sanitized_parts: List[str] = [sanitized_primary] if sanitized_primary else []
    for exhibit_raw in resolved_exhibits:
        sanitized_exhibit, ex_stripped = _sanitize_with_stats(exhibit_raw, max_chars=max_chars)
        total_stripped += ex_stripped
        if sanitized_exhibit:
            sanitized_parts.append(sanitized_exhibit)

    sanitized_content = "\n\n".join(sanitized_parts)
    combined_sanitized = "\n".join(part for part in (sanitized_headline, sanitized_content) if part).strip()
    lexical_category = _classify_sanitized_text(combined_sanitized)

    explicit_mag = _parse_optional_positive_float(meta, "shock_magnitude_pct")
    explicit_dur = _parse_optional_positive_float(meta, "disruption_duration_days")
    has_explicit_shock_meta = explicit_mag is not None or explicit_dur is not None

    is_disruption = False
    category = OperationalDisruptionCategory.OTHER_UNCLASSIFIED_MATERIAL_HALT
    gating_reason = "NON_DISRUPTIVE_NEGATIVE_CONTROL"

    if lexical_category is not None:
        is_disruption = True
        category = lexical_category
        gating_reason = "SANITIZED_LEXICAL_DISRUPTION_MATCH"
    elif tier == 1:
        if primary_item == SECForm8KItem.ITEM_1_02.value:
            if _SUPPLY_CONTEXT_RE.search(combined_sanitized) or has_explicit_shock_meta:
                is_disruption = True
                category = OperationalDisruptionCategory.PRIMARY_SUPPLY_CONTRACT_TERMINATED
                gating_reason = "TIER_1_SUPPLY_CONTRACT_TERMINATION"
            else:
                gating_reason = "TIER_1_ITEM_1_02_NON_SUPPLY_NEGATIVE_CONTROL"
        else:
            is_disruption = True
            category = _TIER_1_STATUTORY_CATEGORIES.get(
                primary_item,
                OperationalDisruptionCategory.OTHER_UNCLASSIFIED_MATERIAL_HALT,
            )
            gating_reason = f"TIER_1_STATUTORY_ANCHOR_{primary_item.replace(' ', '_').replace('.', '_')}"
    elif has_explicit_shock_meta:
        is_disruption = True
        gating_reason = "EXPLICIT_METADATA_SHOCK_OVERRIDE"

    if is_disruption:
        shock_magnitude_pct = explicit_mag if explicit_mag is not None else DEFAULT_SHOCK_MAGNITUDE_PCT
        shock_duration_days = explicit_dur if explicit_dur is not None else DEFAULT_SHOCK_DURATION_DAYS
    else:
        shock_magnitude_pct = 0.0
        shock_duration_days = 0.0

    return SanitizedDisruptionAssessment(
        primary_sec_item=primary_item,
        item_priority_tier=tier,
        is_material_operational_disruption=is_disruption,
        disruption_category=category,
        shock_magnitude_pct=shock_magnitude_pct,
        shock_duration_days=shock_duration_days,
        sanitized_headline=sanitized_headline,
        sanitized_content=sanitized_content,
        stripped_paragraph_count=total_stripped,
        gating_reason=gating_reason,
    )


_SP500_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "data" / "sp500_constituents_historical.json"
_TICKER_TOKEN_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
_ACCESSION_DIR_RE = re.compile(r"^8K_([0-9A-Za-z\-]+)$", re.IGNORECASE)
_CIK_PREFIX_RE = re.compile(r"^(\d{10})-")
_CHARTER_NAME_RE = re.compile(
    r"^\s*([^\n]+?)\s*\n+\s*\(Exact name of Registrant",
    re.IGNORECASE | re.MULTILINE,
)


@lru_cache(maxsize=1)
def _load_sp500_identity_index() -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    by_ticker: Dict[str, Dict[str, str]] = {}
    by_cik: Dict[str, Dict[str, str]] = {}
    if not _SP500_REGISTRY_PATH.is_file():
        return by_ticker, by_cik
    try:
        rows = json.loads(_SP500_REGISTRY_PATH.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            return by_ticker, by_cik
        for row in rows:
            if not isinstance(row, dict):
                continue
            ticker = str(row.get("ticker") or "").strip().upper()
            cik_raw = re.sub(r"\D", "", str(row.get("cik") or ""))
            cik = cik_raw.zfill(10) if cik_raw else ""
            name = str(row.get("company_name") or "").strip()
            entry = {"ticker": ticker, "cik": cik, "company_name": name}
            if ticker and ticker not in by_ticker:
                by_ticker[ticker] = entry
            if cik and cik not in by_cik:
                by_cik[cik] = entry
    except Exception:
        return by_ticker, by_cik
    return by_ticker, by_cik


def _normalize_8k_items(raw_items: Any) -> List[str]:
    if isinstance(raw_items, str):
        tokens = [part.strip() for part in raw_items.split(",") if part.strip()]
    elif isinstance(raw_items, (list, tuple)):
        tokens = [str(part).strip() for part in raw_items if str(part).strip()]
    else:
        tokens = []

    normalized: List[str] = []
    for token in tokens:
        m = _ITEM_CODE_RE.search(token)
        if m and token == m.group(1):
            normalized.append(f"Item {m.group(1)}")
        elif m and token.lower().startswith("item "):
            normalized.append(token)
        elif m:
            normalized.append(f"Item {m.group(1)}")
        else:
            normalized.append(token)
    return normalized or ["Item 8.01 Other Events"]


def _infer_path_metadata(filing_path: Optional[Union[str, Path]]) -> Tuple[Optional[str], Optional[str]]:
    if not filing_path:
        return None, None
    p = Path(filing_path)
    parts = list(p.parts)
    inferred_acc: Optional[str] = None
    for part in reversed(parts):
        m_acc = _ACCESSION_DIR_RE.match(part)
        if m_acc:
            inferred_acc = m_acc.group(1)
            break

    inferred_ticker: Optional[str] = None
    # Patterns: .../sec/{TICKER}/{YEAR}/8K_{ACC}/form8k_events.json or .../{TICKER}/8K_{ACC}/...
    for idx, part in enumerate(parts):
        if part.lower() == "sec" and idx + 1 < len(parts):
            cand = parts[idx + 1].upper()
            if _TICKER_TOKEN_RE.match(cand):
                inferred_ticker = cand
                break
    if inferred_ticker is None and len(parts) >= 3:
        for idx in range(len(parts) - 2, -1, -1):
            seg = parts[idx].upper()
            if _ACCESSION_DIR_RE.match(seg) or re.match(r"^(?:19|20)\d{2}$", seg):
                continue
            if seg not in ("SEC", "FILINGS", "FIXTURES", "TESTS", "DATA", "DRY_RUN_STAGING"):
                if _TICKER_TOKEN_RE.match(seg):
                    inferred_ticker = seg
                break
    return inferred_ticker, inferred_acc


def _synthesize_8k_headline(
    content: str,
    items: Sequence[str],
    exhibit_texts: Optional[Sequence[str]] = None,
) -> str:
    cleaned_parts = [sanitize_sec_8k_text(content)]
    for ex in exhibit_texts or ():
        if isinstance(ex, str) and ex.strip():
            cleaned_ex = sanitize_sec_8k_text(ex)
            if cleaned_ex:
                cleaned_parts.append(cleaned_ex)
    cleaned_body = "\n".join(part for part in cleaned_parts if part)
    lines = [line.strip() for line in cleaned_body.splitlines() if line.strip()]
    cover_skip = re.compile(
        r"^(?:united\s+states|securities\s+and\s+exchange\s+commission|"
        r"united\s+states\s+securities\s+and\s+exchange\s+commission|"
        r"washington,\s*d\.?c\.?|form\s+8-k|current\s+report)\b",
        re.IGNORECASE,
    )
    item_heading = ""
    body_snippet = ""
    for line in lines:
        normalized_line = re.sub(r"\s+", " ", line)
        if re.match(r"^item\s+\d+\.\d{2}\b", normalized_line, re.IGNORECASE):
            if not item_heading:
                item_heading = normalized_line.rstrip(".")
            continue
        if cover_skip.match(normalized_line):
            continue
        if len(normalized_line) >= 20:
            body_snippet = normalized_line
            break

    if item_heading and body_snippet:
        combined = f"{item_heading}: {body_snippet}"
    elif item_heading:
        combined = item_heading
    elif body_snippet:
        combined = body_snippet
    else:
        combined = f"Form 8-K ({', '.join(items[:2])})"

    if len(combined) > 140:
        return combined[:137].rstrip() + "..."
    return combined


def normalize_staged_8k_payload(
    payload: Mapping[str, Any],
    *,
    filing_path: Optional[Union[str, Path]] = None,
    require_identity: bool = True,
) -> Dict[str, Any]:
    """Normalize both canonical inline fixtures and legacy staged `form8k_events.json` payloads (ADR-00027)."""
    if not isinstance(payload, Mapping):
        raise ValueError("SEC Form 8-K payload must be a JSON object/mapping.")

    by_ticker, by_cik = _load_sp500_identity_index()
    path_ticker, path_acc = _infer_path_metadata(filing_path)

    content = str(
        payload.get("content")
        if payload.get("content") is not None
        else (payload.get("text") or "")
    )
    items = _normalize_8k_items(payload.get("items"))
    form_type = str(payload.get("form_type") or payload.get("form") or "8-K").strip() or "8-K"
    accession_number = str(payload.get("accession_number") or path_acc or "").strip()
    filing_date = str(payload.get("filing_date") or "2024-10-28").strip() or "2024-10-28"
    metadata = dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), Mapping) else {}
    filing_timestamp = payload.get("filing_timestamp") or metadata.get("filing_timestamp")

    raw_ticker = str(payload.get("ticker") or "").strip().upper()
    ticker = "" if raw_ticker in ("", "UNKNOWN") else raw_ticker
    if not ticker and path_ticker:
        ticker = path_ticker

    raw_cik_digits = re.sub(r"\D", "", str(payload.get("cik") or ""))
    cik = "" if (not raw_cik_digits or raw_cik_digits == "0000000000") else raw_cik_digits.zfill(10)
    if not cik and accession_number:
        m_cik = _CIK_PREFIX_RE.match(accession_number)
        if m_cik and m_cik.group(1) != "0001193125":
            # Skip SEC filing agent prefix 0001193125 (DFIN/filers), otherwise issuer prefix
            cik = m_cik.group(1)

    raw_company = str(payload.get("company_name") or "").strip()
    company_name = "" if raw_company.lower() in ("", "unknown", "unknown corporate entity") else raw_company

    # Cross-reference ticker <-> CIK/company_name via static S&P 500 ground-truth index
    if ticker and ticker in by_ticker:
        entry = by_ticker[ticker]
        if not cik and entry["cik"]:
            cik = entry["cik"]
        if not company_name and entry["company_name"]:
            company_name = entry["company_name"]
    elif cik and cik in by_cik:
        entry = by_cik[cik]
        if not ticker and entry["ticker"]:
            ticker = entry["ticker"]
        if not company_name and entry["company_name"]:
            company_name = entry["company_name"]

    if not company_name and content:
        m_charter = _CHARTER_NAME_RE.search(content[:2000])
        if m_charter:
            candidate = " ".join(m_charter.group(1).split()).strip()
            if candidate and candidate.lower() != "unknown corporate entity":
                company_name = candidate

    if require_identity and not ticker and not company_name:
        raise ValueError(
            "Unable to resolve SEC Form 8-K entity identity (`ticker` and `company_name` are both missing)."
        )

    if not ticker and company_name:
        # Derive clean identifier from resolved company_name if ticker wasn't present
        ticker = company_name
    if not company_name and ticker:
        company_name = ticker
    if not cik:
        cik = "0000000000"

    exhibit_texts = extract_8k_exhibit_texts(payload)
    raw_headline = str(payload.get("headline") or "").strip()
    headline = raw_headline or _synthesize_8k_headline(content, items, exhibit_texts=exhibit_texts)

    normalized: Dict[str, Any] = dict(payload)
    normalized.update(
        {
            "cik": cik,
            "ticker": ticker,
            "company_name": company_name,
            "form_type": form_type,
            "accession_number": accession_number or None,
            "filing_date": filing_date,
            "filing_timestamp": filing_timestamp,
            "items": items,
            "headline": headline,
            "content": content,
            "text": content,
            "exhibit_texts": exhibit_texts,
            "metadata": metadata,
        }
    )
    return normalized
