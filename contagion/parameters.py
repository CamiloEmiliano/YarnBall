"""Supply-chain cascade parameter extraction and provenance tracking (ADR-00026 / ADR-00048)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple


MAX_PLAUSIBLE_XBRL_DIO_DAYS: float = 365.0

PHYSICAL_INVENTORY_GICS_SECTORS: FrozenSet[str] = frozenset(
    {
        "Information Technology",
        "Industrials",
        "Consumer Discretionary",
        "Consumer Staples",
        "Health Care",
        "Energy",
        "Materials",
    }
)
_PHYSICAL_INVENTORY_GICS_SECTORS_LOWER: FrozenSet[str] = frozenset(
    sector.lower() for sector in PHYSICAL_INVENTORY_GICS_SECTORS
)


def is_physical_inventory_gics_sector(gics_sector: Optional[str]) -> bool:
    """Return True iff `gics_sector` carries physical GAAP inventory eligible for XBRL DIO priors (ADR-00048)."""
    if not gics_sector:
        return False
    return str(gics_sector).strip().lower() in _PHYSICAL_INVENTORY_GICS_SECTORS_LOWER


class ParameterProvenance(str, Enum):
    """Explicit empirical origin of supply-chain cascade edge parameters (ADR-00026 / ADR-00048)."""

    EXTRACTED_QUANTITATIVE = "EXTRACTED_QUANTITATIVE"
    EXTRACTED_QUALITATIVE_SOLE_SOURCE = "EXTRACTED_QUALITATIVE_SOLE_SOURCE"
    XBRL_ISSUER_DIO_PROXY = "XBRL_ISSUER_DIO_PROXY"
    DEFAULT_CONSERVATIVE_BOUND = "DEFAULT_CONSERVATIVE_BOUND"


PROVENANCE_CONFIDENCE_RANK: Dict[str, int] = {
    ParameterProvenance.EXTRACTED_QUANTITATIVE.value: 3,
    ParameterProvenance.EXTRACTED_QUALITATIVE_SOLE_SOURCE.value: 2,
    ParameterProvenance.XBRL_ISSUER_DIO_PROXY.value: 2,
    ParameterProvenance.DEFAULT_CONSERVATIVE_BOUND.value: 1,
}


@dataclass(frozen=True)
class ParameterProvenanceRecord:
    """Parameter provenance bound to its supporting `document_chunks` IDs (ADR-00035)."""

    provenance: ParameterProvenance
    evidence_chunk_ids: Tuple[str, ...] = ()


_SHARE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(\d+(?:\.\d+)?)\s*%\s*(?:of\s+)?(?:our\s+|total\s+)?(?:revenue|purchases?|procurement|supply|wafer\s+supply|capacity|components?|requirements?|output)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:account(?:ed|s)?\s+for|represent(?:ed|ing|s)?|compris(?:ed|ing|es)?|suppl(?:ied|ies))\s+(?:approximately\s+|roughly\s+|about\s+|nearly\s+)?(\d+(?:\.\d+)?)\s*%",
        re.IGNORECASE,
    ),
)

_BUFFER_DAY_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(\d+(?:\.\d+)?)\s*(?:calendar\s+|business\s+)?days?\s*(?:of\s+)?(?:inventory|safety\s+stock|buffer|supply|stockpile)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:inventory|safety\s+stock|buffer|stockpile)\s+(?:of|buffer\s+of|lasts?|covering|sufficient\s+for)\s+(?:approximately\s+|roughly\s+)?(\d+(?:\.\d+)?)\s*(?:calendar\s+|business\s+)?days?",
        re.IGNORECASE,
    ),
)

_BUFFER_WEEK_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*weeks?\s*(?:of\s+)?(?:inventory|safety\s+stock|buffer|supply|stockpile)",
    re.IGNORECASE,
)

_BUFFER_MONTH_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*months?\s*(?:of\s+)?(?:inventory|safety\s+stock|buffer|supply|stockpile)",
    re.IGNORECASE,
)

_SOLE_SOURCE_PATTERN = re.compile(
    r"\b(?:sole[-\s]+source|single[-\s]+source|only\s+qualified\s+supplier)\b",
    re.IGNORECASE,
)

_DIGITAL_INFRA_PATTERN = re.compile(
    r"\b(?:cloud|saas|software|erp|datacenter|data\s+center|network|telecom|cyber)\b",
    re.IGNORECASE,
)

_COMMERCIAL_PATTERN = re.compile(
    r"\b(?:retail|distributor|reseller|channel|broker|sales\s+partner)\b",
    re.IGNORECASE,
)


def _to_finite_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _infer_dependency_domain(text: str, explicit_domain: Optional[str] = None) -> str:
    if explicit_domain:
        clean = str(explicit_domain).strip().upper()
        if clean in ("PHYSICAL_COMPONENT", "DIGITAL_INFRASTRUCTURE", "COMMERCIAL_ENTERPRISE"):
            return clean
    if text and _DIGITAL_INFRA_PATTERN.search(text):
        return "DIGITAL_INFRASTRUCTURE"
    if text and _COMMERCIAL_PATTERN.search(text):
        return "COMMERCIAL_ENTERPRISE"
    return "PHYSICAL_COMPONENT"


def extract_cascade_parameters_from_text(
    text: Optional[str],
    nature: str = "",
    rel_type: str = "",
) -> Dict[str, Any]:
    """Deterministically extract stated share, buffer_days, and sole-source cues from clause text."""
    combined = " ".join(part for part in (str(text or "").strip(), str(nature or "").strip()) if part)
    extracted_share: Optional[float] = None
    for pat in _SHARE_PATTERNS:
        m = pat.search(combined)
        if m:
            pct = _to_finite_float(m.group(1))
            if pct is not None:
                extracted_share = max(0.0, min(1.0, pct / 100.0))
                break

    extracted_buffer: Optional[float] = None
    for pat in _BUFFER_DAY_PATTERNS:
        m = pat.search(combined)
        if m:
            days = _to_finite_float(m.group(1))
            if days is not None:
                extracted_buffer = max(0.0, days)
                break
    if extracted_buffer is None:
        m_wk = _BUFFER_WEEK_PATTERN.search(combined)
        if m_wk:
            weeks = _to_finite_float(m_wk.group(1))
            if weeks is not None:
                extracted_buffer = max(0.0, weeks * 7.0)
    if extracted_buffer is None:
        m_mo = _BUFFER_MONTH_PATTERN.search(combined)
        if m_mo:
            months = _to_finite_float(m_mo.group(1))
            if months is not None:
                extracted_buffer = max(0.0, months * 30.0)

    is_sole_source = (
        str(rel_type or "").strip().upper() == "SOLE_SOURCE_DEPENDENT_ON"
        or bool(_SOLE_SOURCE_PATTERN.search(combined))
    )
    return {
        "share": extracted_share,
        "buffer_days": extracted_buffer,
        "sole_source": is_sole_source,
        "dependency_domain": _infer_dependency_domain(combined),
    }


def resolve_edge_cascade_parameters(
    edge: Mapping[str, Any],
    source_text: Optional[str] = None,
    *,
    target_xbrl_dio_days: Optional[float] = None,
    target_gics_sector: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize cascade edge parameters and emit explicit ParameterProvenance metadata.

    Precedence for inventory `buffer_days` (ADR-00026 / ADR-00048):
    1. Explicit edge or filing-text `buffer_days` (`EXTRACTED_QUANTITATIVE`) always wins.
    2. Else if `target_xbrl_dio_days` is finite in `(0, 365]`, `dependency_domain == "PHYSICAL_COMPONENT"`,
       and `target_gics_sector` is either omitted or in `PHYSICAL_INVENTORY_GICS_SECTORS`, apply the
       issuer-level XBRL DIO proxy (`XBRL_ISSUER_DIO_PROXY`) while keeping `share_is_default` honest.
    3. Else fail closed to `buffer_days = 0.0` (`DEFAULT_CONSERVATIVE_BOUND`).
    """
    props_raw = edge.get("properties")
    props: Mapping[str, Any] = props_raw if isinstance(props_raw, Mapping) else {}

    def _lookup(key: str) -> Any:
        if key in edge and edge[key] is not None:
            return edge[key]
        if key in props and props[key] is not None:
            return props[key]
        return None

    rel_type = str(_lookup("rel_type") or _lookup("type") or "SUPPLIES_TO").strip().upper()
    raw_group = (
        _lookup("input_group")
        or _lookup("nature")
        or _lookup("group")
        or _lookup("component")
        or "default"
    )
    input_group = str(raw_group).strip().lower().replace(" ", "_").replace("-", "_") or "default"

    explicit_share: Optional[float] = None
    for key, div in (
        ("share", 1.0),
        ("exposure_pct", 100.0),
        ("revenue_share_pct", 100.0),
        ("share_pct", 100.0),
    ):
        val = _to_finite_float(_lookup(key))
        if val is not None:
            explicit_share = max(0.0, min(1.0, val / div))
            break
    if explicit_share is None:
        for key in ("revenue_share", "loss_ratio"):
            val = _to_finite_float(_lookup(key))
            if val is not None:
                scaled = val / 100.0 if val > 1.0 else val
                explicit_share = max(0.0, min(1.0, scaled))
                break

    explicit_buffer: Optional[float] = None
    for key in ("buffer_days", "inventory_buffer_days"):
        val = _to_finite_float(_lookup(key))
        if val is not None:
            explicit_buffer = max(0.0, val)
            break
    if explicit_buffer is None:
        wks = _to_finite_float(_lookup("weeks_of_supply") or _lookup("inventory_weeks"))
        if wks is not None:
            explicit_buffer = max(0.0, wks * 7.0)

    clause_text = str(
        source_text
        or _lookup("evidence_text")
        or _lookup("context")
        or _lookup("description")
        or ""
    ).strip()
    text_cues = extract_cascade_parameters_from_text(
        clause_text,
        nature=str(_lookup("nature") or ""),
        rel_type=rel_type,
    )

    if explicit_share is None and text_cues["share"] is not None:
        explicit_share = float(text_cues["share"])
    if explicit_buffer is None and text_cues["buffer_days"] is not None:
        explicit_buffer = float(text_cues["buffer_days"])

    sole_source_flag = bool(_lookup("sole_source") or _lookup("single_source") or text_cues["sole_source"])

    raw_prov = str(_lookup("param_provenance") or "").strip().upper()
    valid_provs = {p.value for p in ParameterProvenance}
    if raw_prov in valid_provs:
        provenance = ParameterProvenance(raw_prov)
    elif explicit_share is not None or explicit_buffer is not None:
        provenance = ParameterProvenance.EXTRACTED_QUANTITATIVE
    elif sole_source_flag:
        provenance = ParameterProvenance.EXTRACTED_QUALITATIVE_SOLE_SOURCE
    else:
        provenance = ParameterProvenance.DEFAULT_CONSERVATIVE_BOUND

    share_is_default = (explicit_share is None) or bool(_lookup("share_is_default"))
    buffer_was_defaulted = bool(_lookup("buffer_days_is_default"))
    buffer_is_default = (explicit_buffer is None) or buffer_was_defaulted

    final_share = 1.0 if explicit_share is None else explicit_share
    final_buffer = 0.0 if explicit_buffer is None else explicit_buffer
    dependency_domain = _infer_dependency_domain(
        f"{input_group} {clause_text}",
        explicit_domain=str(_lookup("dependency_domain") or "") or None,
    )

    dio_raw = target_xbrl_dio_days if target_xbrl_dio_days is not None else _lookup("target_xbrl_dio_days")
    sector_raw = target_gics_sector if target_gics_sector is not None else _lookup("target_gics_sector")
    xbrl_dio_val = _to_finite_float(dio_raw)
    sector_eligible = (
        True
        if sector_raw is None or not str(sector_raw).strip()
        else is_physical_inventory_gics_sector(str(sector_raw))
    )
    can_apply_xbrl_buffer = (
        explicit_buffer is None
        or (
            buffer_was_defaulted
            and explicit_buffer == 0.0
            and provenance == ParameterProvenance.DEFAULT_CONSERVATIVE_BOUND
        )
    )
    if (
        can_apply_xbrl_buffer
        and xbrl_dio_val is not None
        and 0.0 < xbrl_dio_val <= MAX_PLAUSIBLE_XBRL_DIO_DAYS
        and dependency_domain == "PHYSICAL_COMPONENT"
        and sector_eligible
    ):
        final_buffer = round(float(xbrl_dio_val), 6)
        buffer_is_default = False
        if provenance == ParameterProvenance.DEFAULT_CONSERVATIVE_BOUND:
            provenance = ParameterProvenance.XBRL_ISSUER_DIO_PROXY

    raw_chunk_ids = _lookup("evidence_chunk_ids")
    evidence_chunk_ids: List[str] = []
    if isinstance(raw_chunk_ids, (list, tuple, set)):
        evidence_chunk_ids = sorted({str(c).strip() for c in raw_chunk_ids if str(c).strip()})
    elif isinstance(raw_chunk_ids, str) and raw_chunk_ids.strip():
        evidence_chunk_ids = [raw_chunk_ids.strip()]
    single_cid = _lookup("chunk_id")
    if isinstance(single_cid, str) and single_cid.strip() and single_cid.strip() not in evidence_chunk_ids:
        evidence_chunk_ids = sorted([*evidence_chunk_ids, single_cid.strip()])

    return {
        "share": round(final_share, 6),
        "buffer_days": float(final_buffer),
        "input_group": input_group,
        "dependency_domain": dependency_domain,
        "param_provenance": provenance.value,
        "share_is_default": share_is_default,
        "buffer_days_is_default": buffer_is_default,
        "evidence_chunk_ids": evidence_chunk_ids,
    }
