"""Offline SEC EDGAR XBRL `companyfacts` Parser & Point-in-Time Priors Store (ADR-00048).

Deterministically extracts two statutory quantitative priors from offline US-GAAP XBRL
`companyfacts` JSON payloads without runtime network calls:
1. Days Inventory Outstanding (`DIO`) for physical-inventory GICS sectors only,
   used as an explicit `XBRL_ISSUER_DIO_PROXY` fallback buffer in `contagion/parameters.py`.
2. ASC 280 Major-Customer Revenue Concentration (`us-gaap:ConcentrationRiskPercentage1`)
   for statutory >= 10% customer dependency disclosures.

All observations are strictly Point-in-Time anchored via `retrieval.temporal.actionable_session`
(`ADR-00033`) and queried in O(log K) via prefix-max binary search over `valid_from`.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import date, datetime
import json
import logging
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from contagion.parameters import (
    MAX_PLAUSIBLE_XBRL_DIO_DAYS,
    PHYSICAL_INVENTORY_GICS_SECTORS,
    is_physical_inventory_gics_sector,
)
from ingestion.sec.canon import normalize_form_type
from retrieval.temporal import actionable_session

logger = logging.getLogger("ingestion.sec.xbrl")

METRIC_XBRL_DIO_DAYS = "XBRL_DIO_DAYS"
METRIC_ASC280_CONCENTRATION_SHARE = "ASC280_CONCENTRATION_SHARE"

INVENTORY_PRIMARY_TAG = "InventoryNet"
INVENTORY_COMPONENT_TAGS: Tuple[str, ...] = (
    "InventoryFinishedGoods",
    "InventoryWorkInProcess",
    "InventoryRawMaterialsAndSupplies",
)
INVENTORY_FALLBACK_TAG = "InventoryGross"

COGS_TAG_LADDER: Tuple[str, ...] = (
    "CostOfGoodsAndServicesSold",
    "CostOfRevenue",
    "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
    "CostOfGoodsSold",
)

ASC280_CONCENTRATION_TAG = "ConcentrationRiskPercentage1"

ALLOWED_XBRL_PERIODIC_FORMS = frozenset(
    {"10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "20-F/A"}
)

ANNUAL_DURATION_MIN_DAYS = 330
ANNUAL_DURATION_MAX_DAYS = 385
QUARTERLY_DURATION_MIN_DAYS = 75
QUARTERLY_DURATION_MAX_DAYS = 105

DEFAULT_XBRL_PRIORS_PARQUET = (
    Path(__file__).resolve().parents[2] / "data" / "sec_xbrl_priors.parquet"
)


def normalize_cik(raw_cik: Optional[Union[str, int]]) -> str:
    """Normalize a CIK string or integer into a 10-digit zero-padded canonical string."""
    if raw_cik is None:
        return ""
    digits = re.sub(r"\D", "", str(raw_cik).strip())
    if not digits:
        return ""
    return digits.zfill(10)


def _normalize_date_str(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.date().isoformat()
    if isinstance(raw, date):
        return raw.isoformat()
    text = str(raw).strip()
    if len(text) < 10:
        return None
    candidate = text[:10]
    try:
        date.fromisoformat(candidate)
        return candidate
    except ValueError:
        return None


def _to_finite_float(raw: Any) -> Optional[float]:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


def is_physical_inventory_sector(gics_sector: Optional[str]) -> bool:
    """Return True iff `gics_sector` is eligible for XBRL physical-inventory DIO imputation."""
    return is_physical_inventory_gics_sector(gics_sector)


@dataclass(frozen=True)
class XBRLDIOPriorRecord:
    """Point-in-Time anchored issuer-level Days Inventory Outstanding prior (ADR-00048)."""

    cik: str
    ticker: Optional[str]
    gics_sector: Optional[str]
    period_end: str
    filed_date: str
    valid_from: str
    known_at: Optional[str]
    accession_number: str
    form_type: str
    dio_days: float
    inventory_usd: float
    cogs_annualized_usd: float
    inventory_tag: str
    cogs_tag: str
    period_basis: str = "ANNUAL"
    metric_name: str = METRIC_XBRL_DIO_DAYS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cik": self.cik,
            "ticker": self.ticker,
            "gics_sector": self.gics_sector,
            "metric_name": self.metric_name,
            "period_end": self.period_end,
            "filed_date": self.filed_date,
            "valid_from": self.valid_from,
            "known_at": self.known_at,
            "accession_number": self.accession_number,
            "form_type": self.form_type,
            "dio_days": self.dio_days,
            "inventory_usd": self.inventory_usd,
            "cogs_annualized_usd": self.cogs_annualized_usd,
            "inventory_tag": self.inventory_tag,
            "cogs_tag": self.cogs_tag,
            "period_basis": self.period_basis,
        }


# Alias matching ADR-00048 signature shorthand
XBRLPriorRecord = XBRLDIOPriorRecord


@dataclass(frozen=True)
class XBRLConcentrationRecord:
    """Point-in-Time anchored ASC 280 major-customer concentration share (ADR-00048)."""

    cik: str
    ticker: Optional[str]
    gics_sector: Optional[str]
    period_end: str
    filed_date: str
    valid_from: str
    known_at: Optional[str]
    accession_number: str
    form_type: str
    concentration_share: float
    frame: Optional[str] = None
    segment: Optional[str] = None
    metric_name: str = METRIC_ASC280_CONCENTRATION_SHARE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cik": self.cik,
            "ticker": self.ticker,
            "gics_sector": self.gics_sector,
            "metric_name": self.metric_name,
            "period_end": self.period_end,
            "filed_date": self.filed_date,
            "valid_from": self.valid_from,
            "known_at": self.known_at,
            "accession_number": self.accession_number,
            "form_type": self.form_type,
            "concentration_share": self.concentration_share,
            "frame": self.frame,
            "segment": self.segment,
        }


def _extract_us_gaap_facts(companyfacts: Mapping[str, Any]) -> Mapping[str, Any]:
    facts = companyfacts.get("facts")
    if isinstance(facts, Mapping):
        us_gaap = facts.get("us-gaap")
        if isinstance(us_gaap, Mapping):
            return us_gaap
    return {}


def _iter_tag_unit_items(
    us_gaap: Mapping[str, Any],
    tag: str,
    unit_key: str,
) -> List[Mapping[str, Any]]:
    concept = us_gaap.get(tag)
    if not isinstance(concept, Mapping):
        return []
    units = concept.get("units")
    if not isinstance(units, Mapping):
        return []
    items = units.get(unit_key)
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, Mapping)]


def _extract_filing_key(item: Mapping[str, Any]) -> Optional[Tuple[str, str, str, str]]:
    """Return `(period_end, accession_number, filed_date, form_type)` for a valid periodic XBRL fact."""
    period_end = _normalize_date_str(item.get("end"))
    filed_date = _normalize_date_str(item.get("filed"))
    if not period_end or not filed_date:
        return None
    form_type = normalize_form_type(str(item.get("form") or ""))
    if form_type not in ALLOWED_XBRL_PERIODIC_FORMS:
        return None
    accn = str(item.get("accn") or "").strip() or f"FILED_{filed_date}"
    return (period_end, accn, filed_date, form_type)


def extract_xbrl_dio_priors_from_companyfacts(
    companyfacts: Mapping[str, Any],
    *,
    cik: Optional[Union[str, int]] = None,
    ticker: Optional[str] = None,
    gics_sector: Optional[str] = None,
) -> List[XBRLDIOPriorRecord]:
    """Extract Point-in-Time Days Inventory Outstanding records from an EDGAR `companyfacts` JSON dict.

    Returns an empty list immediately when `gics_sector` is supplied and is outside
    `PHYSICAL_INVENTORY_GICS_SECTORS` (Financials, Real Estate, Utilities, Communication Services).
    """
    if gics_sector is not None and str(gics_sector).strip() and not is_physical_inventory_sector(gics_sector):
        logger.debug(
            "Skipping XBRL DIO extraction for non-physical sector '%s' (ticker=%s, cik=%s)",
            gics_sector,
            ticker,
            cik,
        )
        return []

    resolved_cik = normalize_cik(cik if cik is not None else companyfacts.get("cik"))
    if not resolved_cik:
        return []
    clean_ticker = str(ticker).strip().upper() if ticker and str(ticker).strip() else None
    clean_sector = str(gics_sector).strip() if gics_sector and str(gics_sector).strip() else None

    us_gaap = _extract_us_gaap_facts(companyfacts)
    if not us_gaap:
        return []

    # Step 1: Collect point-in-time USD inventory observations keyed by (period_end, accn)
    # Value tuple: (priority_rank, inventory_usd, inventory_tag, filed_date, form_type)
    # priority_rank: 0 = InventoryNet, 1 = InventoryComponentsSum, 2 = InventoryGross
    inventory_by_key: Dict[Tuple[str, str], Tuple[int, float, str, str, str]] = {}

    # 1a. Primary tag: InventoryNet
    for item in _iter_tag_unit_items(us_gaap, INVENTORY_PRIMARY_TAG, "USD"):
        meta = _extract_filing_key(item)
        val = _to_finite_float(item.get("val"))
        if meta is None or val is None or val <= 0.0:
            continue
        period_end, accn, filed_date, form_type = meta
        inventory_by_key[(period_end, accn)] = (0, val, INVENTORY_PRIMARY_TAG, filed_date, form_type)

    # 1b. Component tags sum (Finished Goods + WIP + Raw Materials) where InventoryNet is absent
    component_buckets: Dict[Tuple[str, str], Dict[str, Tuple[float, str, str]]] = {}
    for comp_tag in INVENTORY_COMPONENT_TAGS:
        for item in _iter_tag_unit_items(us_gaap, comp_tag, "USD"):
            meta = _extract_filing_key(item)
            val = _to_finite_float(item.get("val"))
            if meta is None or val is None or val < 0.0:
                continue
            period_end, accn, filed_date, form_type = meta
            key = (period_end, accn)
            if key in inventory_by_key:
                continue
            component_buckets.setdefault(key, {})[comp_tag] = (val, filed_date, form_type)

    for key, comp_map in component_buckets.items():
        if key in inventory_by_key:
            continue
        total_comp = sum(entry[0] for entry in comp_map.values())
        if total_comp > 0.0:
            sample = next(iter(comp_map.values()))
            inventory_by_key[key] = (
                1,
                total_comp,
                "+".join(sorted(comp_map.keys())),
                sample[1],
                sample[2],
            )

    # 1c. Fallback tag: InventoryGross
    for item in _iter_tag_unit_items(us_gaap, INVENTORY_FALLBACK_TAG, "USD"):
        meta = _extract_filing_key(item)
        val = _to_finite_float(item.get("val"))
        if meta is None or val is None or val <= 0.0:
            continue
        period_end, accn, filed_date, form_type = meta
        key = (period_end, accn)
        if key not in inventory_by_key:
            inventory_by_key[key] = (2, val, INVENTORY_FALLBACK_TAG, filed_date, form_type)

    if not inventory_by_key:
        return []

    # Step 2: Collect USD COGS flow observations keyed by (period_end, accn)
    # Prefer ANNUAL over QUARTERLY_ANNUALIZED, and earlier ladder tags over later ones.
    # Value tuple: (basis_rank, tag_rank, cogs_annualized, cogs_tag, period_basis, filed_date, form_type)
    cogs_by_key: Dict[Tuple[str, str], Tuple[int, int, float, str, str, str, str]] = {}

    for tag_rank, cogs_tag in enumerate(COGS_TAG_LADDER):
        for item in _iter_tag_unit_items(us_gaap, cogs_tag, "USD"):
            meta = _extract_filing_key(item)
            val = _to_finite_float(item.get("val"))
            start_str = _normalize_date_str(item.get("start"))
            if meta is None or val is None or val <= 0.0 or not start_str:
                continue
            period_end, accn, filed_date, form_type = meta
            key = (period_end, accn)
            if key not in inventory_by_key:
                continue

            start_dt = date.fromisoformat(start_str)
            end_dt = date.fromisoformat(period_end)
            duration_days = (end_dt - start_dt).days + 1
            if ANNUAL_DURATION_MIN_DAYS <= duration_days <= ANNUAL_DURATION_MAX_DAYS:
                basis_rank = 0
                period_basis = "ANNUAL"
                cogs_ann = val
            elif QUARTERLY_DURATION_MIN_DAYS <= duration_days <= QUARTERLY_DURATION_MAX_DAYS:
                basis_rank = 1
                period_basis = "QUARTERLY_ANNUALIZED"
                cogs_ann = val * 4.0
            else:
                continue

            candidate = (basis_rank, tag_rank, cogs_ann, cogs_tag, period_basis, filed_date, form_type)
            existing = cogs_by_key.get(key)
            if existing is None or (basis_rank, tag_rank) < (existing[0], existing[1]):
                cogs_by_key[key] = candidate

    # Step 3: Pair inventory & COGS, validate bounds, and anchor strictly via actionable_session
    records: List[XBRLDIOPriorRecord] = []
    for key, inv_entry in inventory_by_key.items():
        cogs_entry = cogs_by_key.get(key)
        if cogs_entry is None:
            continue

        period_end, accn = key
        _, inv_usd, inv_tag, inv_filed, inv_form = inv_entry
        _, _, cogs_ann, cogs_tag, period_basis, cogs_filed, cogs_form = cogs_entry
        filed_date = max(inv_filed, cogs_filed)
        form_type = inv_form or cogs_form

        if inv_usd <= 0.0 or cogs_ann <= 0.0:
            continue
        dio_days = 365.0 * inv_usd / cogs_ann
        if not math.isfinite(dio_days) or dio_days <= 0.0 or dio_days > MAX_PLAUSIBLE_XBRL_DIO_DAYS:
            logger.debug(
                "Skipping implausible XBRL DIO %.2f for cik=%s period=%s accn=%s (inv=%.2f, cogs_ann=%.2f)",
                dio_days,
                resolved_cik,
                period_end,
                accn,
                inv_usd,
                cogs_ann,
            )
            continue

        session_date, known_at_utc, _ = actionable_session(None, fallback_doc_date=filed_date)
        if session_date is None:
            continue

        records.append(
            XBRLDIOPriorRecord(
                cik=resolved_cik,
                ticker=clean_ticker,
                gics_sector=clean_sector,
                period_end=period_end,
                filed_date=filed_date,
                valid_from=session_date.isoformat(),
                known_at=known_at_utc.isoformat() if known_at_utc is not None else None,
                accession_number=accn,
                form_type=form_type,
                dio_days=round(dio_days, 4),
                inventory_usd=round(inv_usd, 2),
                cogs_annualized_usd=round(cogs_ann, 2),
                inventory_tag=inv_tag,
                cogs_tag=cogs_tag,
                period_basis=period_basis,
            )
        )

    records.sort(key=lambda r: (r.valid_from, r.period_end, r.filed_date, r.accession_number))
    return records


def extract_asc280_concentration_from_companyfacts(
    companyfacts: Mapping[str, Any],
    *,
    cik: Optional[Union[str, int]] = None,
    ticker: Optional[str] = None,
    gics_sector: Optional[str] = None,
) -> List[XBRLConcentrationRecord]:
    """Extract ASC 280 >=10% major-customer concentration shares (`ConcentrationRiskPercentage1`)."""
    resolved_cik = normalize_cik(cik if cik is not None else companyfacts.get("cik"))
    if not resolved_cik:
        return []
    clean_ticker = str(ticker).strip().upper() if ticker and str(ticker).strip() else None
    clean_sector = str(gics_sector).strip() if gics_sector and str(gics_sector).strip() else None

    us_gaap = _extract_us_gaap_facts(companyfacts)
    if not us_gaap:
        return []

    # Deduplicate by (period_end, accn), keeping maximum statutory customer share disclosed in that filing/period
    best_by_key: Dict[Tuple[str, str], XBRLConcentrationRecord] = {}
    for item in _iter_tag_unit_items(us_gaap, ASC280_CONCENTRATION_TAG, "pure"):
        meta = _extract_filing_key(item)
        val = _to_finite_float(item.get("val"))
        if meta is None or val is None:
            continue
        if not (0.10 <= val <= 1.0):
            continue
        period_end, accn, filed_date, form_type = meta
        session_date, known_at_utc, _ = actionable_session(None, fallback_doc_date=filed_date)
        if session_date is None:
            continue

        raw_frame = item.get("frame")
        frame = str(raw_frame).strip() if isinstance(raw_frame, str) and raw_frame.strip() else None
        raw_seg = item.get("segment")
        segment: Optional[str]
        if isinstance(raw_seg, str) and raw_seg.strip():
            segment = raw_seg.strip()
        elif isinstance(raw_seg, Mapping):
            segment = json.dumps(raw_seg, sort_keys=True)
        else:
            segment = None

        rec = XBRLConcentrationRecord(
            cik=resolved_cik,
            ticker=clean_ticker,
            gics_sector=clean_sector,
            period_end=period_end,
            filed_date=filed_date,
            valid_from=session_date.isoformat(),
            known_at=known_at_utc.isoformat() if known_at_utc is not None else None,
            accession_number=accn,
            form_type=form_type,
            concentration_share=round(val, 6),
            frame=frame,
            segment=segment,
        )
        key = (period_end, accn)
        existing = best_by_key.get(key)
        if existing is None or rec.concentration_share > existing.concentration_share:
            best_by_key[key] = rec

    records = list(best_by_key.values())
    records.sort(key=lambda r: (r.valid_from, r.period_end, r.filed_date, r.accession_number))
    return records


def _dio_rank_key(rec: XBRLDIOPriorRecord) -> Tuple[str, str, int, str, str]:
    return (
        rec.period_end,
        rec.valid_from,
        1 if rec.period_basis == "ANNUAL" else 0,
        rec.filed_date,
        rec.accession_number,
    )


def _conc_rank_key(rec: XBRLConcentrationRecord) -> Tuple[str, str, float, str, str]:
    return (
        rec.period_end,
        rec.valid_from,
        rec.concentration_share,
        rec.filed_date,
        rec.accession_number,
    )


class XBRLPriorStore:
    """Offline O(log K) Point-in-Time lookup store for XBRL DIO & ASC 280 concentration priors."""

    def __init__(
        self,
        dio_records: Iterable[XBRLDIOPriorRecord] = (),
        concentration_records: Iterable[XBRLConcentrationRecord] = (),
    ) -> None:
        self._dio_by_key: Dict[Tuple[str, str, str], XBRLDIOPriorRecord] = {}
        self._conc_by_key: Dict[Tuple[str, str, str], XBRLConcentrationRecord] = {}
        self._dio_index: Dict[str, Tuple[List[str], List[XBRLDIOPriorRecord]]] = {}
        self._conc_index: Dict[str, Tuple[List[str], List[XBRLConcentrationRecord]]] = {}
        self._dirty = True
        self.add_dio_records(dio_records)
        self.add_concentration_records(concentration_records)

    def add_dio_records(self, records: Iterable[XBRLDIOPriorRecord]) -> int:
        added = 0
        for rec in records:
            self._dio_by_key[(rec.cik, rec.period_end, rec.accession_number)] = rec
            added += 1
        if added:
            self._dirty = True
        return added

    def add_concentration_records(self, records: Iterable[XBRLConcentrationRecord]) -> int:
        added = 0
        for rec in records:
            self._conc_by_key[(rec.cik, rec.period_end, rec.accession_number)] = rec
            added += 1
        if added:
            self._dirty = True
        return added

    def add_companyfacts(
        self,
        companyfacts: Mapping[str, Any],
        *,
        cik: Optional[Union[str, int]] = None,
        ticker: Optional[str] = None,
        gics_sector: Optional[str] = None,
    ) -> Tuple[int, int]:
        dio_recs = extract_xbrl_dio_priors_from_companyfacts(
            companyfacts,
            cik=cik,
            ticker=ticker,
            gics_sector=gics_sector,
        )
        conc_recs = extract_asc280_concentration_from_companyfacts(
            companyfacts,
            cik=cik,
            ticker=ticker,
            gics_sector=gics_sector,
        )
        return (
            self.add_dio_records(dio_recs),
            self.add_concentration_records(conc_recs),
        )

    @property
    def dio_records(self) -> List[XBRLDIOPriorRecord]:
        return sorted(
            self._dio_by_key.values(),
            key=lambda r: (r.cik, r.valid_from, r.period_end, r.accession_number),
        )

    @property
    def concentration_records(self) -> List[XBRLConcentrationRecord]:
        return sorted(
            self._conc_by_key.values(),
            key=lambda r: (r.cik, r.valid_from, r.period_end, r.accession_number),
        )

    def _rebuild_indices(self) -> None:
        if not self._dirty:
            return
        dio_grouped: Dict[str, List[XBRLDIOPriorRecord]] = {}
        for rec in self._dio_by_key.values():
            dio_grouped.setdefault(rec.cik, []).append(rec)
            if rec.ticker:
                dio_grouped.setdefault(rec.ticker.upper(), []).append(rec)

        self._dio_index = {}
        for entity_key, recs in dio_grouped.items():
            recs.sort(key=lambda r: (r.valid_from, r.period_end, r.filed_date, r.accession_number))
            dates: List[str] = []
            prefix_best: List[XBRLDIOPriorRecord] = []
            best: Optional[XBRLDIOPriorRecord] = None
            for rec in recs:
                dates.append(rec.valid_from)
                if best is None or _dio_rank_key(rec) > _dio_rank_key(best):
                    best = rec
                prefix_best.append(best)
            self._dio_index[entity_key] = (dates, prefix_best)

        conc_grouped: Dict[str, List[XBRLConcentrationRecord]] = {}
        for conc_rec in self._conc_by_key.values():
            conc_grouped.setdefault(conc_rec.cik, []).append(conc_rec)
            if conc_rec.ticker:
                conc_grouped.setdefault(conc_rec.ticker.upper(), []).append(conc_rec)

        self._conc_index = {}
        for entity_key, conc_recs in conc_grouped.items():
            conc_recs.sort(key=lambda r: (r.valid_from, r.period_end, r.filed_date, r.accession_number))
            conc_dates: List[str] = []
            conc_prefix_best: List[XBRLConcentrationRecord] = []
            conc_best: Optional[XBRLConcentrationRecord] = None
            for conc_rec in conc_recs:
                conc_dates.append(conc_rec.valid_from)
                if conc_best is None or _conc_rank_key(conc_rec) > _conc_rank_key(conc_best):
                    conc_best = conc_rec
                conc_prefix_best.append(conc_best)
            self._conc_index[entity_key] = (conc_dates, conc_prefix_best)

        self._dirty = False

    def _lookup_key_candidates(self, cik_or_ticker: Union[str, int]) -> Tuple[str, ...]:
        raw = str(cik_or_ticker).strip()
        if not raw:
            return ()
        candidates: List[str] = []
        if raw.isdigit():
            candidates.append(normalize_cik(raw))
        else:
            candidates.append(raw.upper())
            padded = normalize_cik(raw)
            if padded and padded not in candidates:
                candidates.append(padded)
        return tuple(candidates)

    def lookup_pit_dio(
        self,
        cik_or_ticker: Union[str, int],
        as_of: Union[str, date, datetime],
        *,
        gics_sector: Optional[str] = None,
    ) -> Optional[XBRLDIOPriorRecord]:
        """O(log K) Point-in-Time lookup returning the latest DIO record with `valid_from <= as_of`."""
        if gics_sector is not None and str(gics_sector).strip() and not is_physical_inventory_sector(gics_sector):
            return None
        as_of_iso = _normalize_date_str(as_of)
        if not as_of_iso:
            return None

        self._rebuild_indices()
        for key in self._lookup_key_candidates(cik_or_ticker):
            bucket = self._dio_index.get(key)
            if bucket is None:
                continue
            dates, prefix_best = bucket
            idx = bisect.bisect_right(dates, as_of_iso) - 1
            if idx < 0:
                return None
            rec = prefix_best[idx]
            if rec.gics_sector and not is_physical_inventory_sector(rec.gics_sector):
                return None
            return rec
        return None

    def lookup_pit_max_concentration(
        self,
        cik_or_ticker: Union[str, int],
        as_of: Union[str, date, datetime],
    ) -> Optional[XBRLConcentrationRecord]:
        """O(log K) Point-in-Time lookup returning the latest ASC 280 concentration share with `valid_from <= as_of`."""
        as_of_iso = _normalize_date_str(as_of)
        if not as_of_iso:
            return None

        self._rebuild_indices()
        for key in self._lookup_key_candidates(cik_or_ticker):
            bucket = self._conc_index.get(key)
            if bucket is None:
                continue
            dates, prefix_best = bucket
            idx = bisect.bisect_right(dates, as_of_iso) - 1
            if idx < 0:
                return None
            return prefix_best[idx]
        return None

    def to_unified_rows(self) -> List[Dict[str, Any]]:
        """Serialize both DIO and ASC 280 records into the unified relational/Parquet row schema."""
        rows: List[Dict[str, Any]] = []
        for rec in self.dio_records:
            rows.append(
                {
                    "cik": rec.cik,
                    "ticker": rec.ticker,
                    "gics_sector": rec.gics_sector,
                    "metric_name": rec.metric_name,
                    "period_end": rec.period_end,
                    "filed_date": rec.filed_date,
                    "valid_from": rec.valid_from,
                    "known_at": rec.known_at,
                    "accession_number": rec.accession_number,
                    "form_type": rec.form_type,
                    "metric_value": rec.dio_days,
                    "inventory_usd": rec.inventory_usd,
                    "cogs_annualized_usd": rec.cogs_annualized_usd,
                    "inventory_tag": rec.inventory_tag,
                    "cogs_tag": rec.cogs_tag,
                    "period_basis": rec.period_basis,
                    "metadata_json": "{}",
                }
            )
        for conc in self.concentration_records:
            meta: Dict[str, Any] = {}
            if conc.frame:
                meta["frame"] = conc.frame
            if conc.segment:
                meta["segment"] = conc.segment
            rows.append(
                {
                    "cik": conc.cik,
                    "ticker": conc.ticker,
                    "gics_sector": conc.gics_sector,
                    "metric_name": conc.metric_name,
                    "period_end": conc.period_end,
                    "filed_date": conc.filed_date,
                    "valid_from": conc.valid_from,
                    "known_at": conc.known_at,
                    "accession_number": conc.accession_number,
                    "form_type": conc.form_type,
                    "metric_value": conc.concentration_share,
                    "inventory_usd": None,
                    "cogs_annualized_usd": None,
                    "inventory_tag": None,
                    "cogs_tag": None,
                    "period_basis": None,
                    "metadata_json": json.dumps(meta, sort_keys=True),
                }
            )
        return rows

    @classmethod
    def from_unified_rows(cls, rows: Sequence[Mapping[str, Any]]) -> "XBRLPriorStore":
        dio_recs: List[XBRLDIOPriorRecord] = []
        conc_recs: List[XBRLConcentrationRecord] = []
        for row in rows:
            metric = str(row.get("metric_name") or "").strip()
            cik = normalize_cik(row.get("cik"))
            if not cik:
                continue
            ticker_raw = row.get("ticker")
            ticker = str(ticker_raw).strip().upper() if ticker_raw else None
            sector_raw = row.get("gics_sector")
            gics_sector = str(sector_raw).strip() if sector_raw else None
            period_end = str(row.get("period_end") or "")[:10]
            filed_date = str(row.get("filed_date") or "")[:10]
            valid_from = str(row.get("valid_from") or "")[:10]
            known_at_raw = row.get("known_at")
            known_at = str(known_at_raw) if known_at_raw else None
            accn = str(row.get("accession_number") or "")
            form_type = str(row.get("form_type") or "")
            metric_value = _to_finite_float(row.get("metric_value"))
            if metric_value is None or not period_end or not valid_from:
                continue

            if metric == METRIC_XBRL_DIO_DAYS:
                dio_recs.append(
                    XBRLDIOPriorRecord(
                        cik=cik,
                        ticker=ticker,
                        gics_sector=gics_sector,
                        period_end=period_end,
                        filed_date=filed_date,
                        valid_from=valid_from,
                        known_at=known_at,
                        accession_number=accn,
                        form_type=form_type,
                        dio_days=metric_value,
                        inventory_usd=_to_finite_float(row.get("inventory_usd")) or 0.0,
                        cogs_annualized_usd=_to_finite_float(row.get("cogs_annualized_usd")) or 0.0,
                        inventory_tag=str(row.get("inventory_tag") or ""),
                        cogs_tag=str(row.get("cogs_tag") or ""),
                        period_basis=str(row.get("period_basis") or "ANNUAL"),
                    )
                )
            elif metric == METRIC_ASC280_CONCENTRATION_SHARE:
                meta_raw = row.get("metadata_json") or "{}"
                meta_obj: Dict[str, Any] = {}
                if isinstance(meta_raw, str):
                    try:
                        parsed = json.loads(meta_raw)
                        if isinstance(parsed, dict):
                            meta_obj = parsed
                    except json.JSONDecodeError:
                        pass
                elif isinstance(meta_raw, Mapping):
                    meta_obj = dict(meta_raw)
                conc_recs.append(
                    XBRLConcentrationRecord(
                        cik=cik,
                        ticker=ticker,
                        gics_sector=gics_sector,
                        period_end=period_end,
                        filed_date=filed_date,
                        valid_from=valid_from,
                        known_at=known_at,
                        accession_number=accn,
                        form_type=form_type,
                        concentration_share=metric_value,
                        frame=str(meta_obj["frame"]) if meta_obj.get("frame") else None,
                        segment=str(meta_obj["segment"]) if meta_obj.get("segment") else None,
                    )
                )
        return cls(dio_records=dio_recs, concentration_records=conc_recs)

    def save_parquet(self, path: Union[str, Path] = DEFAULT_XBRL_PRIORS_PARQUET) -> Path:
        """Persist all XBRL DIO and ASC 280 records to a local Snappy Parquet file."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.to_unified_rows()
        schema = pa.schema(
            [
                ("cik", pa.string()),
                ("ticker", pa.string()),
                ("gics_sector", pa.string()),
                ("metric_name", pa.string()),
                ("period_end", pa.string()),
                ("filed_date", pa.string()),
                ("valid_from", pa.string()),
                ("known_at", pa.string()),
                ("accession_number", pa.string()),
                ("form_type", pa.string()),
                ("metric_value", pa.float64()),
                ("inventory_usd", pa.float64()),
                ("cogs_annualized_usd", pa.float64()),
                ("inventory_tag", pa.string()),
                ("cogs_tag", pa.string()),
                ("period_basis", pa.string()),
                ("metadata_json", pa.string()),
            ]
        )
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, out_path, compression="snappy")
        return out_path

    @classmethod
    def load_parquet(cls, path: Union[str, Path] = DEFAULT_XBRL_PRIORS_PARQUET) -> "XBRLPriorStore":
        """Load a persisted XBRL priors Parquet archive into an O(log K) PiT store."""
        import pyarrow.parquet as pq

        in_path = Path(path)
        if not in_path.is_file():
            return cls()
        table = pq.read_table(in_path)
        return cls.from_unified_rows(table.to_pylist())

    def save_json(self, path: Union[str, Path]) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dio_records": [r.to_dict() for r in self.dio_records],
            "concentration_records": [r.to_dict() for r in self.concentration_records],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out_path

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "XBRLPriorStore":
        in_path = Path(path)
        if not in_path.is_file():
            return cls()
        raw = json.loads(in_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            return cls()
        dio_recs = [
            XBRLDIOPriorRecord(**item)
            for item in (raw.get("dio_records") or [])
            if isinstance(item, Mapping)
        ]
        conc_recs = [
            XBRLConcentrationRecord(**item)
            for item in (raw.get("concentration_records") or [])
            if isinstance(item, Mapping)
        ]
        return cls(dio_records=dio_recs, concentration_records=conc_recs)

    def sync_to_postgres(self, conn: Any) -> int:
        """Idempotently upsert all XBRL priors into PostgreSQL `sec_xbrl_priors`."""
        rows = self.to_unified_rows()
        if not rows or conn is None:
            return 0
        sql = """
            INSERT INTO sec_xbrl_priors (
                cik, ticker, gics_sector, metric_name, period_end, filed_date,
                valid_from, known_at, accession_number, form_type, metric_value,
                inventory_usd, cogs_annualized_usd, inventory_tag, cogs_tag,
                period_basis, metadata
            )
            VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s::jsonb
            )
            ON CONFLICT (cik, metric_name, period_end, accession_number) DO UPDATE SET
                ticker = EXCLUDED.ticker,
                gics_sector = EXCLUDED.gics_sector,
                filed_date = EXCLUDED.filed_date,
                valid_from = EXCLUDED.valid_from,
                known_at = EXCLUDED.known_at,
                form_type = EXCLUDED.form_type,
                metric_value = EXCLUDED.metric_value,
                inventory_usd = EXCLUDED.inventory_usd,
                cogs_annualized_usd = EXCLUDED.cogs_annualized_usd,
                inventory_tag = EXCLUDED.inventory_tag,
                cogs_tag = EXCLUDED.cogs_tag,
                period_basis = EXCLUDED.period_basis,
                metadata = EXCLUDED.metadata;
        """
        params = [
            (
                r["cik"],
                r["ticker"],
                r["gics_sector"],
                r["metric_name"],
                r["period_end"],
                r["filed_date"],
                r["valid_from"],
                r["known_at"],
                r["accession_number"],
                r["form_type"],
                r["metric_value"],
                r["inventory_usd"],
                r["cogs_annualized_usd"],
                r["inventory_tag"],
                r["cogs_tag"],
                r["period_basis"],
                r["metadata_json"],
            )
            for r in rows
        ]
        cur = conn.cursor()
        try:
            cur.executemany(sql, params)
            conn.commit()
        finally:
            if hasattr(cur, "close"):
                cur.close()
        return len(params)


def lookup_pit_xbrl_dio(
    cik_or_ticker: Union[str, int],
    as_of: Union[str, date, datetime],
    *,
    gics_sector: Optional[str] = None,
    store: Optional[XBRLPriorStore] = None,
    parquet_path: Optional[Union[str, Path]] = None,
) -> Optional[XBRLDIOPriorRecord]:
    """Offline O(log K) Point-in-Time XBRL DIO lookup (ADR-00048; zero network calls)."""
    active_store = store
    if active_store is None:
        resolved_path = Path(parquet_path) if parquet_path is not None else DEFAULT_XBRL_PRIORS_PARQUET
        if not resolved_path.is_file():
            return None
        active_store = XBRLPriorStore.load_parquet(resolved_path)
    return active_store.lookup_pit_dio(cik_or_ticker, as_of, gics_sector=gics_sector)


def fetch_companyfacts_json(
    ticker_or_cik: Union[str, int],
    *,
    client: Optional[Any] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """Load an issuer's EDGAR `companyfacts` JSON from local cache first, or offline batch client."""
    raw = str(ticker_or_cik).strip()
    if not raw:
        return None
    padded = normalize_cik(raw) if raw.isdigit() else ""

    if cache_dir is not None:
        cdir = Path(cache_dir)
        candidates: List[Path] = []
        if padded:
            candidates.append(cdir / f"CIK{padded}.json")
            candidates.append(cdir / f"{padded}.json")
        candidates.append(cdir / f"{raw.upper()}.json")
        for candidate in candidates:
            if candidate.is_file():
                parsed = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    return parsed

    if client is None:
        return None

    try:
        company = client.get_company(raw)
        if company is None:
            return None
        facts_obj = company.get_facts() if hasattr(company, "get_facts") else getattr(company, "facts", None)
        if facts_obj is None:
            return None
        if isinstance(facts_obj, dict):
            return facts_obj
        for attr in ("raw", "_raw", "data"):
            raw_attr = getattr(facts_obj, attr, None)
            if isinstance(raw_attr, dict) and "facts" in raw_attr:
                return raw_attr
        to_dict = getattr(facts_obj, "to_dict", None)
        if callable(to_dict):
            d = to_dict()
            if isinstance(d, dict):
                return d
    except Exception as exc:
        logger.warning("Failed fetching XBRL companyfacts for '%s': %s", raw, exc)
    return None


__all__ = [
    "ASC280_CONCENTRATION_TAG",
    "COGS_TAG_LADDER",
    "DEFAULT_XBRL_PRIORS_PARQUET",
    "INVENTORY_COMPONENT_TAGS",
    "INVENTORY_FALLBACK_TAG",
    "INVENTORY_PRIMARY_TAG",
    "MAX_PLAUSIBLE_XBRL_DIO_DAYS",
    "METRIC_ASC280_CONCENTRATION_SHARE",
    "METRIC_XBRL_DIO_DAYS",
    "PHYSICAL_INVENTORY_GICS_SECTORS",
    "XBRLConcentrationRecord",
    "XBRLDIOPriorRecord",
    "XBRLPriorRecord",
    "XBRLPriorStore",
    "extract_asc280_concentration_from_companyfacts",
    "extract_xbrl_dio_priors_from_companyfacts",
    "fetch_companyfacts_json",
    "is_physical_inventory_sector",
    "lookup_pit_xbrl_dio",
    "normalize_cik",
]
