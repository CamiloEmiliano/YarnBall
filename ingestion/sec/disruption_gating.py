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

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from pydantic import BaseModel, Field

from retrieval.reporting import OperationalDisruptionCategory, SECForm8KItem

DEFAULT_MAX_FILING_CHARS: int = 50_000
DEFAULT_SHOCK_MAGNITUDE_PCT: float = 100.0
DEFAULT_SHOCK_DURATION_DAYS: float = 60.0

# Pre-compiled linear-time structural splitters (no nested quantifiers).
_SIGNATURE_CUTOFF_RE = re.compile(
    r"\bpursuant\s+to\s+the\s+requirements\s+of\s+the\s+securities\s+exchange\s+act\s+of\s+1934\b",
    re.IGNORECASE,
)
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

# Canonical Priority Tiers (ADR-00014 Section 4.2)
_TIER_ITEMS: Tuple[Tuple[int, Tuple[str, ...]], ...] = (
    (
        1,
        (
            SECForm8KItem.ITEM_1_05.value,
            SECForm8KItem.ITEM_1_04.value,
            SECForm8KItem.ITEM_1_03.value,
            SECForm8KItem.ITEM_1_02.value,
            SECForm8KItem.ITEM_2_05.value,
            SECForm8KItem.ITEM_2_06.value,
            SECForm8KItem.ITEM_2_04.value,
        ),
    ),
    (
        2,
        (
            SECForm8KItem.ITEM_1_01.value,
            SECForm8KItem.ITEM_8_01.value,
        ),
    ),
    (
        3,
        (
            SECForm8KItem.ITEM_2_01.value,
            SECForm8KItem.ITEM_2_02.value,
            SECForm8KItem.ITEM_2_03.value,
            SECForm8KItem.ITEM_7_01.value,
        ),
    ),
    (
        4,
        (
            SECForm8KItem.ITEM_3_01.value,
            SECForm8KItem.ITEM_3_02.value,
            SECForm8KItem.ITEM_3_03.value,
            SECForm8KItem.ITEM_4_01.value,
            SECForm8KItem.ITEM_4_02.value,
            SECForm8KItem.ITEM_5_01.value,
            SECForm8KItem.ITEM_5_02.value,
            SECForm8KItem.ITEM_5_03.value,
            SECForm8KItem.ITEM_5_04.value,
            SECForm8KItem.ITEM_5_05.value,
            SECForm8KItem.ITEM_5_06.value,
            SECForm8KItem.ITEM_5_07.value,
            SECForm8KItem.ITEM_5_08.value,
            SECForm8KItem.ITEM_6_01.value,
            SECForm8KItem.ITEM_6_02.value,
            SECForm8KItem.ITEM_6_03.value,
            SECForm8KItem.ITEM_6_04.value,
            SECForm8KItem.ITEM_6_05.value,
            SECForm8KItem.ITEM_9_01.value,
        ),
    ),
)

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


def _sanitize_with_stats(raw_text: str, max_chars: int = DEFAULT_MAX_FILING_CHARS) -> Tuple[str, int]:
    """Single-pass linear sanitization returning cleaned text and stripped block count."""
    if not raw_text:
        return "", 0

    bounded = raw_text[:max_chars] if max_chars > 0 else raw_text
    stripped_count = 0

    # 1. Truncate trailing Exchange Act signature section onward.
    sig_match = _SIGNATURE_CUTOFF_RE.search(bounded)
    if sig_match:
        bounded = bounded[: sig_match.start()]
        stripped_count += 1

    # 2. Split monolithic or single-newline blocks on explicit disclaimer/exhibit headers.
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
) -> SanitizedDisruptionAssessment:
    """Executes boilerplate sanitization, tier resolution, and negative-control gating."""
    meta = metadata or {}
    primary_item = select_primary_sec_8k_item(items)
    tier = get_sec_8k_item_tier(primary_item)

    sanitized_headline, h_stripped = _sanitize_with_stats(headline or "", max_chars=max_chars)
    sanitized_content, c_stripped = _sanitize_with_stats(content or "", max_chars=max_chars)
    total_stripped = h_stripped + c_stripped

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
