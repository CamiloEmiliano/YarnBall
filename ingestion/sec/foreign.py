"""Foreign Private Issuer & Canadian Upstream Counterparty Ingestion Helpers (ADR-00049).

Provides:
1. Isolated `ForeignCounterparty` registry (`DEFAULT_TIER1_FOREIGN_COUNTERPARTIES`)
   keeping non-U.S. Tier-1 suppliers cleanly separated from `SP500UniverseManager`.
2. Stateless O(N) SEC Rule 306 English / bilingual script filter (`filter_english_operational_text`)
   preventing non-English appendices or duplicated bilingual blocks from polluting extraction.
3. Linear-time Form 6-K boilerplate sanitization and zero-shock negative-control disruption gating
   (`sanitize_sec_6k_text`, `assess_sec_6k_disruption`).
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import unicodedata

from ingestion.sec.canon import SECFormType
from ingestion.sec.disruption_gating import (
    DEFAULT_MAX_FILING_CHARS,
    DEFAULT_SHOCK_DURATION_DAYS,
    DEFAULT_SHOCK_MAGNITUDE_PCT,
    SanitizedDisruptionAssessment,
    _LEXICAL_DISRUPTION_PATTERNS,
    _sanitize_with_stats,
    extract_8k_exhibit_texts,
)
from retrieval.reporting import OperationalDisruptionCategory


@dataclass(frozen=True)
class ForeignCounterparty:
    """Metadata describing a non-U.S. Tier-1 upstream counterparty filing 20-F, 40-F, or 6-K."""

    ticker: str
    cik: str
    company_name: str
    home_country: str
    annual_form: str
    gics_sector: str
    sp500_supply_role: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "cik": self.cik,
            "company_name": self.company_name,
            "home_country": self.home_country,
            "annual_form": self.annual_form,
            "gics_sector": self.gics_sector,
            "sp500_supply_role": self.sp500_supply_role,
        }


DEFAULT_TIER1_FOREIGN_COUNTERPARTIES: Tuple[ForeignCounterparty, ...] = (
    # Semiconductors & Electronics (Form 20-F)
    ForeignCounterparty(
        ticker="TSM",
        cik="0001046179",
        company_name="Taiwan Semiconductor Manufacturing Company Limited",
        home_country="Taiwan",
        annual_form=SECFormType.FORM_20F.value,
        gics_sector="Information Technology",
        sp500_supply_role="Advanced logic wafer foundry (NVDA, AAPL, AMD, QCOM, AVGO)",
    ),
    ForeignCounterparty(
        ticker="ASML",
        cik="0000937966",
        company_name="ASML Holding N.V.",
        home_country="Netherlands",
        annual_form=SECFormType.FORM_20F.value,
        gics_sector="Information Technology",
        sp500_supply_role="Sole-source EUV lithography equipment (INTC, TSM, MU)",
    ),
    ForeignCounterparty(
        ticker="STM",
        cik="0000932787",
        company_name="STMicroelectronics N.V.",
        home_country="Switzerland",
        annual_form=SECFormType.FORM_20F.value,
        gics_sector="Information Technology",
        sp500_supply_role="Automotive SiC power & microcontroller supplier (TSLA, AAPL)",
    ),
    ForeignCounterparty(
        ticker="ARM",
        cik="0001973239",
        company_name="Arm Holdings plc",
        home_country="United Kingdom",
        annual_form=SECFormType.FORM_20F.value,
        gics_sector="Information Technology",
        sp500_supply_role="CPU instruction-set architecture licensor (AAPL, QCOM, NVDA, AMZN)",
    ),
    ForeignCounterparty(
        ticker="UMC",
        cik="0001033767",
        company_name="United Microelectronics Corporation",
        home_country="Taiwan",
        annual_form=SECFormType.FORM_20F.value,
        gics_sector="Information Technology",
        sp500_supply_role="Specialty-node wafer foundry (QCOM, TXN, AMD)",
    ),
    ForeignCounterparty(
        ticker="ASX",
        cik="0001122411",
        company_name="ASE Technology Holding Co., Ltd.",
        home_country="Taiwan",
        annual_form=SECFormType.FORM_20F.value,
        gics_sector="Information Technology",
        sp500_supply_role="OSAT advanced semiconductor packaging & testing",
    ),
    ForeignCounterparty(
        ticker="SONY",
        cik="0000313838",
        company_name="Sony Group Corporation",
        home_country="Japan",
        annual_form=SECFormType.FORM_20F.value,
        gics_sector="Consumer Discretionary",
        sp500_supply_role="CMOS image sensor supplier (AAPL, automotive ADAS)",
    ),
    ForeignCounterparty(
        ticker="SAP",
        cik="0001000184",
        company_name="SAP SE",
        home_country="Germany",
        annual_form=SECFormType.FORM_20F.value,
        gics_sector="Information Technology",
        sp500_supply_role="Enterprise ERP & supply-chain software partner",
    ),
    ForeignCounterparty(
        ticker="INFY",
        cik="0001067491",
        company_name="Infosys Limited",
        home_country="India",
        annual_form=SECFormType.FORM_20F.value,
        gics_sector="Information Technology",
        sp500_supply_role="Enterprise IT & cloud systems integrator",
    ),
    ForeignCounterparty(
        ticker="TM",
        cik="0001094517",
        company_name="Toyota Motor Corporation",
        home_country="Japan",
        annual_form=SECFormType.FORM_20F.value,
        gics_sector="Consumer Discretionary",
        sp500_supply_role="Global automotive OEM & Tier-1 parts network",
    ),
    # Pharmaceuticals & Biologics (Form 20-F)
    ForeignCounterparty(
        ticker="NVO",
        cik="0000353278",
        company_name="Novo Nordisk A/S",
        home_country="Denmark",
        annual_form=SECFormType.FORM_20F.value,
        gics_sector="Health Care",
        sp500_supply_role="GLP-1 & insulin biologics supplier (CVS, UNH, MCK, CAH)",
    ),
    ForeignCounterparty(
        ticker="AZN",
        cik="0000901832",
        company_name="AstraZeneca PLC",
        home_country="United Kingdom",
        annual_form=SECFormType.FORM_20F.value,
        gics_sector="Health Care",
        sp500_supply_role="Oncology & biopharma co-development partner (MRK, BMY)",
    ),
    ForeignCounterparty(
        ticker="NVS",
        cik="0001114448",
        company_name="Novartis AG",
        home_country="Switzerland",
        annual_form=SECFormType.FORM_20F.value,
        gics_sector="Health Care",
        sp500_supply_role="Global pharmaceuticals & radiopharmaceuticals supplier",
    ),
    ForeignCounterparty(
        ticker="SNY",
        cik="0001121404",
        company_name="Sanofi",
        home_country="France",
        annual_form=SECFormType.FORM_20F.value,
        gics_sector="Health Care",
        sp500_supply_role="Immunology collaboration & vaccines partner (REGN)",
    ),
    ForeignCounterparty(
        ticker="GSK",
        cik="0001131399",
        company_name="GSK plc",
        home_country="United Kingdom",
        annual_form=SECFormType.FORM_20F.value,
        gics_sector="Health Care",
        sp500_supply_role="Vaccines & specialty pharmaceuticals supplier",
    ),
    ForeignCounterparty(
        ticker="TAK",
        cik="0001395064",
        company_name="Takeda Pharmaceutical Company Limited",
        home_country="Japan",
        annual_form=SECFormType.FORM_20F.value,
        gics_sector="Health Care",
        sp500_supply_role="Plasma-derived therapies & specialty pharma supplier",
    ),
    # Canadian MJDS Energy, Pipelines, Class-I Rail & Fertilizers (Form 40-F)
    ForeignCounterparty(
        ticker="ENB",
        cik="0000895728",
        company_name="Enbridge Inc.",
        home_country="Canada",
        annual_form=SECFormType.FORM_40F.value,
        gics_sector="Energy",
        sp500_supply_role="Mainline crude & natural gas pipeline carrier to U.S. refineries",
    ),
    ForeignCounterparty(
        ticker="TRP",
        cik="0001232384",
        company_name="TC Energy Corporation",
        home_country="Canada",
        annual_form=SECFormType.FORM_40F.value,
        gics_sector="Energy",
        sp500_supply_role="North American natural gas & power pipeline operator",
    ),
    ForeignCounterparty(
        ticker="CNI",
        cik="0000016868",
        company_name="Canadian National Railway Company",
        home_country="Canada",
        annual_form=SECFormType.FORM_40F.value,
        gics_sector="Industrials",
        sp500_supply_role="Cross-border Class-I freight rail corridor",
    ),
    ForeignCounterparty(
        ticker="CP",
        cik="0000016875",
        company_name="Canadian Pacific Kansas City Limited",
        home_country="Canada",
        annual_form=SECFormType.FORM_40F.value,
        gics_sector="Industrials",
        sp500_supply_role="Canada-U.S.-Mexico Class-I freight rail network",
    ),
    ForeignCounterparty(
        ticker="NTR",
        cik="0001725964",
        company_name="Nutrien Ltd.",
        home_country="Canada",
        annual_form=SECFormType.FORM_40F.value,
        gics_sector="Materials",
        sp500_supply_role="Potash & nitrogen fertilizer supplier to U.S. agriculture",
    ),
    ForeignCounterparty(
        ticker="CNQ",
        cik="0001017413",
        company_name="Canadian Natural Resources Limited",
        home_country="Canada",
        annual_form=SECFormType.FORM_40F.value,
        gics_sector="Energy",
        sp500_supply_role="Heavy crude oil feedstock producer for U.S. Gulf/Midwest refiners",
    ),
    ForeignCounterparty(
        ticker="SU",
        cik="0000311337",
        company_name="Suncor Energy Inc.",
        home_country="Canada",
        annual_form=SECFormType.FORM_40F.value,
        gics_sector="Energy",
        sp500_supply_role="Integrated oil sands & refining counterparty",
    ),
    # Global Upstream Mining & Raw Materials (Form 20-F)
    ForeignCounterparty(
        ticker="BHP",
        cik="0000811809",
        company_name="BHP Group Limited",
        home_country="Australia",
        annual_form=SECFormType.FORM_20F.value,
        gics_sector="Materials",
        sp500_supply_role="Copper, iron ore, and metallurgical coal upstream supplier",
    ),
    ForeignCounterparty(
        ticker="RIO",
        cik="0000863064",
        company_name="Rio Tinto plc",
        home_country="United Kingdom",
        annual_form=SECFormType.FORM_20F.value,
        gics_sector="Materials",
        sp500_supply_role="Aluminum, copper, titanium, and lithium upstream supplier",
    ),
    ForeignCounterparty(
        ticker="VALE",
        cik="0000917851",
        company_name="Vale S.A.",
        home_country="Brazil",
        annual_form=SECFormType.FORM_20F.value,
        gics_sector="Materials",
        sp500_supply_role="Iron ore & Class-1 battery nickel supplier",
    ),
)

_BY_TICKER: Dict[str, ForeignCounterparty] = {
    c.ticker.upper(): c for c in DEFAULT_TIER1_FOREIGN_COUNTERPARTIES
}
_BY_CIK: Dict[str, ForeignCounterparty] = {
    c.cik.zfill(10): c for c in DEFAULT_TIER1_FOREIGN_COUNTERPARTIES
}


def lookup_foreign_counterparty(ticker_or_cik: Optional[str]) -> Optional[ForeignCounterparty]:
    """Look up a curated Tier-1 foreign counterparty by ticker or CIK (zero network calls)."""
    if not ticker_or_cik:
        return None
    raw = str(ticker_or_cik).strip()
    if not raw:
        return None
    upper = raw.upper()
    if upper in _BY_TICKER:
        return _BY_TICKER[upper]
    digits = re.sub(r"\D", "", raw)
    if digits:
        return _BY_CIK.get(digits.zfill(10))
    return None


def is_foreign_counterparty(ticker_or_cik: Optional[str]) -> bool:
    """Return True iff `ticker_or_cik` is in the curated Tier-1 foreign counterparty catalog."""
    return lookup_foreign_counterparty(ticker_or_cik) is not None


def filter_foreign_counterparties(
    *,
    tickers: Optional[Iterable[str]] = None,
    gics_sectors: Optional[Iterable[str]] = None,
    annual_form: Optional[str] = None,
) -> List[ForeignCounterparty]:
    """Filter curated Tier-1 foreign counterparties by ticker, GICS sector, or annual form (`20-F`/`40-F`)."""
    ticker_set = {str(t).strip().upper() for t in tickers if str(t).strip()} if tickers is not None else None
    sector_set = {str(s).strip().lower() for s in gics_sectors if str(s).strip()} if gics_sectors is not None else None
    target_form = str(annual_form).strip().upper() if annual_form else None

    results: List[ForeignCounterparty] = []
    for entry in DEFAULT_TIER1_FOREIGN_COUNTERPARTIES:
        if ticker_set is not None and entry.ticker.upper() not in ticker_set:
            continue
        if sector_set is not None and entry.gics_sector.lower() not in sector_set:
            continue
        if target_form is not None and entry.annual_form.upper() != target_form:
            continue
        results.append(entry)
    return results


# ==============================================================================
# SEC Rule 306 English / Bilingual Operational Text Filter (O(N), Zero-LLM)
# ==============================================================================

_ENGLISH_STOPWORD_RE = re.compile(
    r"\b(?:the|and|of|to|in|for|with|that|this|from|are|was|were|has|have|will|our|its|on|at|by|as|is|be|we|not)\b",
    re.IGNORECASE,
)
_NON_ENGLISH_LATIN_CUE_RE = re.compile(
    r"\b(?:les|des|une|dans|pour|avec|sur|par|qui|est|sont|nous|notre|soci[eé]t[eé]|communiqu[eé]|"
    r"rapport|trimestriel|assembl[eé]e|los|las|del|para|con|por|una|nuestro|informe|hecho|relevante|"
    r"der|die|das|und|mit|f[uü]r|von|ein|eine|wir|unser|gesch[aä]ftsbericht)\b",
    re.IGNORECASE,
)
_PARAGRAPH_SPLIT_RE = re.compile(r"\r?\n\s*\r?\n+")


@dataclass(frozen=True)
class LanguageFilterReport:
    """Telemetry from deterministic SEC Rule 306 English / bilingual filtering."""

    kept_paragraphs: int
    dropped_paragraphs: int
    latin_letter_ratio: float
    english_stopword_hits: int
    is_usable_english: bool


def _latin_alpha_counts(text: str) -> Tuple[int, int]:
    """Return `(latin_letters, total_letters)` for `text` using Unicode script names."""
    latin = 0
    total = 0
    for ch in text:
        if ch.isalpha():
            total += 1
            if "A" <= ch <= "Z" or "a" <= ch <= "z":
                latin += 1
            else:
                name = unicodedata.name(ch, "")
                if "LATIN" in name:
                    latin += 1
    return latin, total


def _is_non_english_latin_prose(block: str) -> bool:
    """Detect French/Spanish/German paragraphs inside bilingual side-by-side 6-K/40-F filings."""
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", block)
    if len(words) < 6:
        return False
    en_hits = len(_ENGLISH_STOPWORD_RE.findall(block))
    non_en_hits = len(_NON_ENGLISH_LATIN_CUE_RE.findall(block))
    return non_en_hits >= 3 and non_en_hits > en_hits


def filter_english_operational_text(
    raw_text: str,
    *,
    min_latin_letter_ratio: float = 0.70,
    min_chars: int = 40,
) -> Tuple[str, LanguageFilterReport]:
    """Isolate English operational prose from bilingual or non-English SEC Rule 306 attachments."""
    if not raw_text or not raw_text.strip():
        return "", LanguageFilterReport(0, 0, 0.0, 0, False)

    cleaned = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(cleaned) if p.strip()]

    kept: List[str] = []
    dropped = 0

    for para in paragraphs:
        latin, total = _latin_alpha_counts(para)
        if total == 0:
            dropped += 1
            continue
        if latin < total:
            # Inspect line-by-line whenever a paragraph contains any non-Latin script (e.g. CJK header + English body)
            kept_lines: List[str] = []
            dropped_any_line = False
            for line in para.splitlines():
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                l_lat, l_tot = _latin_alpha_counts(stripped_line)
                if l_tot > 0 and (l_lat / l_tot) >= min_latin_letter_ratio and not _is_non_english_latin_prose(stripped_line):
                    kept_lines.append(stripped_line)
                else:
                    dropped_any_line = True
            if kept_lines:
                kept.append("\n".join(kept_lines))
                if dropped_any_line:
                    dropped += 1
            else:
                dropped += 1
            continue

        if _is_non_english_latin_prose(para):
            dropped += 1
            continue

        kept.append(para)

    filtered = "\n\n".join(kept).strip()
    out_latin, out_total = _latin_alpha_counts(filtered)
    out_ratio = round(out_latin / out_total, 4) if out_total > 0 else 0.0
    en_hits = len(_ENGLISH_STOPWORD_RE.findall(filtered))
    word_count = len(re.findall(r"[A-Za-z]+", filtered))
    is_usable = bool(
        len(filtered) >= min_chars
        and out_ratio >= min_latin_letter_ratio
        and (en_hits >= 2 or word_count <= 6)
    )

    return filtered, LanguageFilterReport(
        kept_paragraphs=len(kept),
        dropped_paragraphs=dropped,
        latin_letter_ratio=out_ratio,
        english_stopword_hits=en_hits,
        is_usable_english=is_usable,
    )


# ==============================================================================
# Form 6-K Boilerplate Sanitization & Zero-Shock Disruption Gating
# ==============================================================================

_FORM_6K_COVER_MARKERS: Tuple[str, ...] = (
    "report of foreign private issuer",
    "pursuant to rule 13a-16 or 15d-16",
    "under cover of form 20-f",
    "under cover of form 40-f",
    "indicate by check mark whether the registrant files",
    "translation of registrant's name into english",
    "address of principal executive office",
    "furnishing the information contained in this form",
)

_FOREIGN_EXTRA_DISRUPTION_PATTERNS: Tuple[Tuple[re.Pattern[str], OperationalDisruptionCategory], ...] = (
    (
        re.compile(
            r"\b(?:export\s+(?:control|ban|restriction|embargo|license\s+(?:revocation|denial|suspension))|"
            r"trade\s+embargo|sanctions?\s+(?:imposed|prohibit|ban|restriction))\b",
            re.IGNORECASE,
        ),
        OperationalDisruptionCategory.GOVERNMENT_EXPORT_BAN_OR_SANCTION,
    ),
    (
        re.compile(
            r"\b(?:rail(?:way|road)?\s+(?:blockade|stoppage|strike|shutdown|halt)|"
            r"border\s+crossing\s+(?:closure|blockade))\b",
            re.IGNORECASE,
        ),
        OperationalDisruptionCategory.MARITIME_CANAL_PORT_LOGISTICS_CHOKEPOINT,
    ),
    (
        re.compile(
            r"\b(?:pipeline\s+(?:rupture|leak|outage|shutdown|halt|curtailment)|"
            r"refinery\s+(?:outage|shutdown|halt)|"
            r"mine\s+(?:collapse|flooding|shutdown|halt|suspension)|"
            r"smelter\s+(?:outage|shutdown|halt))\b",
            re.IGNORECASE,
        ),
        OperationalDisruptionCategory.EQUIPMENT_BREAKDOWN_FAB_HALT,
    ),
)


def _classify_sanitized_6k_text(sanitized_text: str) -> Optional[OperationalDisruptionCategory]:
    if not sanitized_text:
        return None
    for pattern, category in (*_LEXICAL_DISRUPTION_PATTERNS, *_FOREIGN_EXTRA_DISRUPTION_PATTERNS):
        if pattern.search(sanitized_text):
            return category
    return None


def _parse_positive_float(meta: Mapping[str, Any], key: str) -> Optional[float]:
    raw = meta.get(key)
    if raw is None or isinstance(raw, bool):
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if val > 0.0 else None


def sanitize_sec_6k_text(
    raw_text: str,
    *,
    max_chars: int = DEFAULT_MAX_FILING_CHARS,
) -> Tuple[str, int, LanguageFilterReport]:
    """Sanitize Form 6-K cover boilerplate/PSLRA tails, preserve Exhibit 99.x, and enforce Rule 306 English guard."""
    if not raw_text:
        return "", 0, LanguageFilterReport(0, 0, 0.0, 0, False)

    base_clean, stripped_count = _sanitize_with_stats(raw_text, max_chars=max_chars)
    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(base_clean) if p.strip()]
    kept_after_cover: List[str] = []
    for para in paragraphs:
        lower = para.lower()
        if any(marker in lower for marker in _FORM_6K_COVER_MARKERS):
            stripped_count += 1
            continue
        kept_after_cover.append(para)

    filtered_text, lang_report = filter_english_operational_text("\n\n".join(kept_after_cover))
    stripped_count += lang_report.dropped_paragraphs
    return filtered_text, stripped_count, lang_report


def assess_sec_6k_disruption(
    headline: str,
    content: str,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    max_chars: int = DEFAULT_MAX_FILING_CHARS,
    exhibit_texts: Optional[Sequence[str]] = None,
) -> SanitizedDisruptionAssessment:
    """Gate Form 6-K filings with Rule 306 English filtering and strict zero-shock negative controls (ADR-00049)."""
    meta = metadata or {}
    sanitized_headline, h_stripped, h_lang = sanitize_sec_6k_text(headline or "", max_chars=max_chars)
    sanitized_primary, c_stripped, c_lang = sanitize_sec_6k_text(content or "", max_chars=max_chars)
    total_stripped = h_stripped + c_stripped

    resolved_exhibits: List[str] = (
        [ex for ex in exhibit_texts if isinstance(ex, str) and ex.strip()]
        if exhibit_texts is not None
        else (extract_8k_exhibit_texts(meta) if meta else [])
    )
    sanitized_parts: List[str] = [sanitized_primary] if sanitized_primary else []
    exhibit_usable = False
    for ex_raw in resolved_exhibits:
        ex_clean, ex_stripped, ex_lang = sanitize_sec_6k_text(ex_raw, max_chars=max_chars)
        total_stripped += ex_stripped
        if ex_clean:
            sanitized_parts.append(ex_clean)
            if ex_lang.is_usable_english:
                exhibit_usable = True

    sanitized_content = "\n\n".join(sanitized_parts)
    combined_sanitized = "\n".join(part for part in (sanitized_headline, sanitized_content) if part).strip()
    is_usable_english = h_lang.is_usable_english or c_lang.is_usable_english or exhibit_usable

    if not is_usable_english:
        return SanitizedDisruptionAssessment(
            primary_sec_item="Form 6-K",
            item_priority_tier=4,
            is_material_operational_disruption=False,
            disruption_category=OperationalDisruptionCategory.OTHER_UNCLASSIFIED_MATERIAL_HALT,
            shock_magnitude_pct=0.0,
            shock_duration_days=0.0,
            sanitized_headline=sanitized_headline,
            sanitized_content=sanitized_content,
            stripped_paragraph_count=total_stripped,
            gating_reason="NON_ENGLISH_OR_EMPTY_6K_SKIPPED",
        )

    lexical_category = _classify_sanitized_6k_text(combined_sanitized)
    explicit_mag = _parse_positive_float(meta, "shock_magnitude_pct")
    explicit_dur = _parse_positive_float(meta, "disruption_duration_days")
    has_explicit_shock_meta = explicit_mag is not None or explicit_dur is not None

    if lexical_category is not None:
        return SanitizedDisruptionAssessment(
            primary_sec_item="Form 6-K",
            item_priority_tier=1,
            is_material_operational_disruption=True,
            disruption_category=lexical_category,
            shock_magnitude_pct=explicit_mag if explicit_mag is not None else DEFAULT_SHOCK_MAGNITUDE_PCT,
            shock_duration_days=explicit_dur if explicit_dur is not None else DEFAULT_SHOCK_DURATION_DAYS,
            sanitized_headline=sanitized_headline,
            sanitized_content=sanitized_content,
            stripped_paragraph_count=total_stripped,
            gating_reason="SANITIZED_6K_LEXICAL_DISRUPTION_MATCH",
        )

    if has_explicit_shock_meta:
        return SanitizedDisruptionAssessment(
            primary_sec_item="Form 6-K",
            item_priority_tier=2,
            is_material_operational_disruption=True,
            disruption_category=OperationalDisruptionCategory.OTHER_UNCLASSIFIED_MATERIAL_HALT,
            shock_magnitude_pct=explicit_mag if explicit_mag is not None else DEFAULT_SHOCK_MAGNITUDE_PCT,
            shock_duration_days=explicit_dur if explicit_dur is not None else DEFAULT_SHOCK_DURATION_DAYS,
            sanitized_headline=sanitized_headline,
            sanitized_content=sanitized_content,
            stripped_paragraph_count=total_stripped,
            gating_reason="EXPLICIT_6K_METADATA_SHOCK_OVERRIDE",
        )

    return SanitizedDisruptionAssessment(
        primary_sec_item="Form 6-K",
        item_priority_tier=4,
        is_material_operational_disruption=False,
        disruption_category=OperationalDisruptionCategory.OTHER_UNCLASSIFIED_MATERIAL_HALT,
        shock_magnitude_pct=0.0,
        shock_duration_days=0.0,
        sanitized_headline=sanitized_headline,
        sanitized_content=sanitized_content,
        stripped_paragraph_count=total_stripped,
        gating_reason="NON_DISRUPTIVE_6K_NEGATIVE_CONTROL",
    )


__all__ = [
    "DEFAULT_TIER1_FOREIGN_COUNTERPARTIES",
    "ForeignCounterparty",
    "LanguageFilterReport",
    "assess_sec_6k_disruption",
    "filter_english_operational_text",
    "filter_foreign_counterparties",
    "is_foreign_counterparty",
    "lookup_foreign_counterparty",
    "sanitize_sec_6k_text",
]
