"""Point-in-Time Read Contract, U.S. Equity Session Calendar & Temporal Query Filter (ADR-00033).

Enforces strict Point-in-Time (PiT) temporal semantics across YarnBall read and write paths:
1. `PiTContext`: Mandatory non-optional Point-in-Time context (`as_of`, `mode`, `include_unanchored`).
2. U.S. Equity Regular-Session Calendar (NYSE / NASDAQ / Cboe):
   - Regular close: 16:00 ET (13:00 ET on scheduled early-close half-days).
   - `actionable_session(known_at, fallback_doc_date)` derives the first regular trading session
     in which an SEC filing or news disclosure is actionable without after-hours trading.
3. `TemporalQueryFilter`:
   - `historical` mode: half-open validity interval `[valid_from, valid_to)` without filtering on
     mutable current-state `r.status` / `r.is_current`; excludes unanchored edges unless opted in.
   - `live` mode: preserves `coalesce(r.status, 'ACTIVE_CURRENT') = 'ACTIVE_CURRENT'` alongside
     interval bounds.
   - Supports both single-hop `[r:TYPE]` and variable-length `[r*1..H]` Cypher patterns via
     `ALL(_rel IN r WHERE ...)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
import re
from typing import Any, List, Literal, Optional, Tuple, Union
from zoneinfo import ZoneInfo

US_EASTERN = ZoneInfo("America/New_York")
REGULAR_CLOSE_ET = time(16, 0, 0)
EARLY_CLOSE_ET = time(13, 0, 0)
MIN_TRAVERSAL_HOPS = 1
MAX_TRAVERSAL_HOPS = 10

# Historical unscheduled U.S. cash-equity closures (NYSE / NASDAQ / Cboe)
_UNSCHEDULED_CLOSURES: frozenset[date] = frozenset(
    {
        date(2018, 12, 5),  # National Day of Mourning — George H.W. Bush
        date(2025, 1, 9),   # National Day of Mourning — Jimmy Carter
    }
)

_EDGAR_COMPACT_TS_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})$")
_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _easter_sunday(year: int) -> date:
    """Compute Gregorian Easter Sunday (Anonymous Gregorian / Meeus-Jones-Butcher algorithm)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """Return the n-th (1-indexed) occurrence of `weekday` (0=Mon..6=Sun) in `(year, month)`."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + (n - 1) * 7)


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """Return the last occurrence of `weekday` (0=Mon..6=Sun) in `(year, month)`."""
    if month == 12:
        next_first = date(year + 1, 1, 1)
    else:
        next_first = date(year, month + 1, 1)
    last_day = next_first - timedelta(days=1)
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


def _observed_fixed_holiday(
    year: int, month: int, day: int, *, observe_saturday: bool = True
) -> Optional[date]:
    """Apply U.S. equity exchange weekend observation rules (NYSE Rule 7.2 / NASDAQ / Cboe)."""
    d = date(year, month, day)
    if d.weekday() == 5:  # Saturday
        return d - timedelta(days=1) if observe_saturday else None
    if d.weekday() == 6:  # Sunday
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=64)
def us_equity_holidays(year: int) -> frozenset[date]:
    """Return closed U.S. equity exchange holidays for `year` (NYSE / NASDAQ / Cboe)."""
    holidays: set[date] = set()

    # New Year's Day (if Jan 1 falls on Saturday, prior Dec 31 is NOT closed under NYSE Rule 7.2)
    nyd = _observed_fixed_holiday(year, 1, 1, observe_saturday=False)
    if nyd is not None:
        holidays.add(nyd)

    # MLK Jr. Day (3rd Monday of January)
    holidays.add(_nth_weekday_of_month(year, 1, 0, 3))
    # Washington's Birthday / Presidents' Day (3rd Monday of February)
    holidays.add(_nth_weekday_of_month(year, 2, 0, 3))
    # Good Friday (2 days before Easter Sunday)
    holidays.add(_easter_sunday(year) - timedelta(days=2))
    # Memorial Day (last Monday of May)
    holidays.add(_last_weekday_of_month(year, 5, 0))

    # Juneteenth National Independence Day (observed by U.S. equity exchanges starting 2022)
    if year >= 2022:
        juneteenth = _observed_fixed_holiday(year, 6, 19, observe_saturday=True)
        if juneteenth is not None:
            holidays.add(juneteenth)

    # Independence Day
    july4 = _observed_fixed_holiday(year, 7, 4, observe_saturday=True)
    if july4 is not None:
        holidays.add(july4)

    # Labor Day (1st Monday of September)
    holidays.add(_nth_weekday_of_month(year, 9, 0, 1))
    # Thanksgiving Day (4th Thursday of November)
    holidays.add(_nth_weekday_of_month(year, 11, 3, 4))

    # Christmas Day
    xmas = _observed_fixed_holiday(year, 12, 25, observe_saturday=True)
    if xmas is not None:
        holidays.add(xmas)

    for closed_day in _UNSCHEDULED_CLOSURES:
        if closed_day.year == year:
            holidays.add(closed_day)

    return frozenset(holidays)


@lru_cache(maxsize=64)
def us_equity_early_closes(year: int) -> frozenset[date]:
    """Return scheduled 13:00 ET early-close trading sessions for `year` (NYSE / NASDAQ / Cboe)."""
    holidays = us_equity_holidays(year)
    early: set[date] = set()

    # Day after Thanksgiving (Friday after 4th Thursday of November)
    thanksgiving = _nth_weekday_of_month(year, 11, 3, 4)
    day_after_thanksgiving = thanksgiving + timedelta(days=1)
    if day_after_thanksgiving not in holidays:
        early.add(day_after_thanksgiving)

    # July 3rd when it is an open weekday (i.e., July 4 falls on Tue–Fri)
    july3 = date(year, 7, 3)
    if july3.weekday() < 5 and july3 not in holidays:
        early.add(july3)

    # Christmas Eve (Dec 24) when it is an open weekday (Mon–Thu)
    xmas_eve = date(year, 12, 24)
    if xmas_eve.weekday() < 5 and xmas_eve not in holidays:
        early.add(xmas_eve)

    return frozenset(early)


def is_us_equity_trading_day(d: date) -> bool:
    """Return True if `d` is an open U.S. equity trading session (NYSE / NASDAQ / Cboe)."""
    return d.weekday() < 5 and d not in us_equity_holidays(d.year)


def us_equity_regular_close_et(d: date) -> Optional[time]:
    """Return the ET regular-session close time (`16:00` or `13:00`) for `d`, or None if closed."""
    if not is_us_equity_trading_day(d):
        return None
    if d in us_equity_early_closes(d.year):
        return EARLY_CLOSE_ET
    return REGULAR_CLOSE_ET


def next_us_equity_trading_day(d: date) -> date:
    """Return the first open U.S. equity trading session strictly after `d`."""
    cur = d + timedelta(days=1)
    while not is_us_equity_trading_day(cur):
        cur += timedelta(days=1)
    return cur


def session_close_utc(d: date) -> datetime:
    """Return the UTC timestamp of the regular-session close for calendar date `d`."""
    close_et = us_equity_regular_close_et(d) or REGULAR_CLOSE_ET
    return datetime.combine(d, close_et, tzinfo=US_EASTERN).astimezone(timezone.utc)


def _parse_timestamp_or_date(
    raw: Optional[Union[datetime, date, str]],
) -> Tuple[Optional[datetime], Optional[date]]:
    """Parse an input into `(known_at_utc, doc_date)`.

    Returns `(dt_utc, dt_utc.date())` when a time-of-day component is present,
    `(None, parsed_date)` when only a calendar date is present, or `(None, None)` when empty.
    """
    if raw is None:
        return None, None
    if isinstance(raw, datetime):
        dt_utc = (
            raw.replace(tzinfo=timezone.utc)
            if raw.tzinfo is None
            else raw.astimezone(timezone.utc)
        )
        return dt_utc, dt_utc.date()
    if isinstance(raw, date):
        return None, raw

    text = str(raw).strip()
    if not text or text.lower() in ("none", "null", "nan"):
        return None, None

    if _DATE_ONLY_RE.match(text):
        try:
            return None, date.fromisoformat(text)
        except ValueError:
            return None, None

    compact_match = _EDGAR_COMPACT_TS_RE.match(text)
    if compact_match:
        yr, mo, dy, hr, mn, sc = (int(g) for g in compact_match.groups())
        # EDGAR acceptance_datetime headers are US/Eastern when naive 14-digit strings
        dt_et = datetime(yr, mo, dy, hr, mn, sc, tzinfo=US_EASTERN)
        dt_utc = dt_et.astimezone(timezone.utc)
        return dt_utc, dt_utc.date()

    normalized = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed_dt = datetime.fromisoformat(normalized)
        dt_utc = (
            parsed_dt.replace(tzinfo=timezone.utc)
            if parsed_dt.tzinfo is None
            else parsed_dt.astimezone(timezone.utc)
        )
        return dt_utc, dt_utc.date()
    except ValueError:
        # Fallback: try leading YYYY-MM-DD prefix as date-only
        if len(text) >= 10 and _DATE_ONLY_RE.match(text[:10]):
            try:
                return None, date.fromisoformat(text[:10])
            except ValueError:
                return None, None
        return None, None


def actionable_session(
    known_at: Optional[Union[datetime, date, str]],
    fallback_doc_date: Optional[Union[date, str]] = None,
) -> Tuple[Optional[date], Optional[datetime], bool]:
    """Derive `(valid_from_session, known_at_utc, known_at_imputed)` per ADR-00033 Section 4.2.

    - If `known_at` carries a timestamp: convert to U.S. Eastern; if on an open U.S. equity
      trading day at or before the regular-session close (16:00 ET, or 13:00 ET on early-close
      days), the disclosure is actionable in that session; otherwise on the next open session.
    - If `known_at` is missing or date-only (imputation required): conservatively assign
      `next_us_equity_trading_day(doc_date)` with `known_at_imputed=True`.
    - If neither timestamp nor document date is available: return `(None, None, True)`.
    """
    dt_utc, parsed_date = _parse_timestamp_or_date(known_at)
    if dt_utc is not None:
        dt_et = dt_utc.astimezone(US_EASTERN)
        close_et = us_equity_regular_close_et(dt_et.date())
        if close_et is not None and dt_et.time() <= close_et:
            return dt_et.date(), dt_utc, False
        return next_us_equity_trading_day(dt_et.date()), dt_utc, False

    doc_date = parsed_date
    if doc_date is None and fallback_doc_date is not None:
        _, doc_date = _parse_timestamp_or_date(fallback_doc_date)

    if doc_date is not None:
        return next_us_equity_trading_day(doc_date), None, True

    return None, None, True


def validate_hops(
    hops: Any,
    *,
    min_hops: int = MIN_TRAVERSAL_HOPS,
    max_hops: int = MAX_TRAVERSAL_HOPS,
) -> int:
    """Validate a graph traversal hop bound as an integer in `[min_hops, max_hops]` (ADR-00033)."""
    if isinstance(hops, bool) or not isinstance(hops, int):
        raise ValueError(
            f"Traversal hops must be an int in [{min_hops}, {max_hops}], got {hops!r}"
        )
    if hops < min_hops or hops > max_hops:
        raise ValueError(
            f"Traversal hops={hops} out of allowed range [{min_hops}, {max_hops}]"
        )
    return hops


@dataclass(frozen=True)
class PiTContext:
    """Mandatory Point-in-Time execution context for all read paths (ADR-00033)."""

    as_of: date
    mode: Literal["historical", "live"] = "historical"
    include_unanchored: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.as_of, datetime):
            object.__setattr__(self, "as_of", self.as_of.date())
        elif isinstance(self.as_of, str):
            clean = self.as_of.strip()
            if not clean:
                raise ValueError("PiTContext.as_of must be a valid ISO date, got empty string.")
            object.__setattr__(self, "as_of", date.fromisoformat(clean[:10]))
        elif not isinstance(self.as_of, date):
            raise TypeError(
                f"PiTContext.as_of must be a date or ISO date string, got {type(self.as_of).__name__}"
            )

        if self.mode not in ("historical", "live"):
            raise ValueError(f"PiTContext.mode must be 'historical' or 'live', got {self.mode!r}")

    @property
    def as_of_iso(self) -> str:
        """ISO-8601 calendar date string (`YYYY-MM-DD`)."""
        return self.as_of.isoformat()

    @property
    def cutoff_utc(self) -> datetime:
        """UTC timestamp of the regular-session close for `as_of`."""
        return session_close_utc(self.as_of)

    @property
    def cutoff_utc_iso(self) -> str:
        """ISO-8601 UTC timestamp string of the regular-session close for `as_of`."""
        return self.cutoff_utc.isoformat()

    @classmethod
    def from_date(
        cls,
        as_of: Union[date, datetime, str],
        mode: Literal["historical", "live"] = "historical",
        include_unanchored: bool = False,
    ) -> "PiTContext":
        """Construct a validated `PiTContext` from a `date`, `datetime`, or ISO string."""
        if isinstance(as_of, datetime):
            parsed = as_of.date()
        elif isinstance(as_of, date):
            parsed = as_of
        elif isinstance(as_of, str) and as_of.strip():
            parsed = date.fromisoformat(as_of.strip()[:10])
        else:
            raise ValueError(f"Invalid as_of date for PiTContext: {as_of!r}")
        return cls(as_of=parsed, mode=mode, include_unanchored=include_unanchored)


# Regex matching any Cypher relationship bracket `[...]` between dashes
_REL_BRACKET_RE = re.compile(
    r"(-)\s*\[([^\]]*)\]\s*(-[><]?|<-?)",
    re.IGNORECASE,
)
_REL_SPEC_RE = re.compile(
    r"^\s*(?P<var>[A-Za-z_][A-Za-z0-9_]*)?"
    r"(?:\s*:\s*(?P<types>[A-Za-z0-9_|]+))?"
    r"(?P<varlen>\s*\*\s*(?:\d*\s*\.\.\s*\d*|\d+)?)?"
    r"(?P<props>\s*\{[^}]*\})?\s*$",
    re.IGNORECASE,
)


class TemporalQueryFilter:
    """Transforms openCypher queries to enforce ADR-00033 Point-in-Time invariants."""

    def __init__(self, default_as_of_date_param: str = "$as_of_date"):
        self.default_as_of_date_param = default_as_of_date_param

    @staticmethod
    def _build_edge_predicate(var_expr: str, date_iso: str, ctx: PiTContext) -> str:
        """Build the half-open `[valid_from, valid_to)` predicate for a single relationship var."""
        parts: List[str] = []
        if ctx.mode == "live":
            parts.append(f"coalesce({var_expr}.status, 'ACTIVE_CURRENT') = 'ACTIVE_CURRENT'")

        if ctx.include_unanchored:
            parts.append(f"({var_expr}.valid_from IS NULL OR {var_expr}.valid_from <= '{date_iso}')")
        else:
            parts.append(f"{var_expr}.valid_from IS NOT NULL AND {var_expr}.valid_from <= '{date_iso}'")

        parts.append(f"({var_expr}.valid_to IS NULL OR {var_expr}.valid_to > '{date_iso}')")
        return " AND ".join(parts)

    def inject_active_coalescence(
        self,
        cypher: str,
        ctx: PiTContext,
        default_var: str = "r",
    ) -> str:
        """Inject Point-in-Time temporal bounds into an openCypher query per ADR-00033.

        Args:
            cypher: The original openCypher query string.
            ctx: Mandatory `PiTContext` controlling `as_of`, `historical` vs `live` mode,
                 and `include_unanchored`.
            default_var: Relationship variable prefix to bind if anonymous.

        Returns:
            Rewritten Cypher query with temporal invariants enforced on every matched relationship.
        """
        if not isinstance(ctx, PiTContext):
            raise TypeError(f"ctx must be a PiTContext instance, got {type(ctx).__name__}")

        clean = cypher.strip()
        if not clean or clean.lower() == "(none)":
            return clean

        bound_cypher, rel_specs = self._ensure_relationship_vars(clean, default_var)
        if not rel_specs:
            return clean

        date_iso = ctx.as_of_iso
        predicates: List[str] = []
        for rel_var, is_varlen in rel_specs:
            if is_varlen:
                if f"ALL(_rel IN {rel_var} WHERE" in bound_cypher:
                    continue
                inner = self._build_edge_predicate("_rel", date_iso, ctx)
                predicates.append(f"ALL(_rel IN {rel_var} WHERE {inner})")
            else:
                if f"{rel_var}.valid_from" in bound_cypher and f"{rel_var}.valid_to" in bound_cypher:
                    continue
                predicates.append(self._build_edge_predicate(rel_var, date_iso, ctx))

        if not predicates:
            return bound_cypher

        combined_pred = " AND ".join(predicates)
        return self._inject_where_predicate(bound_cypher, combined_pred)

    def inject_audit_filter(
        self,
        cypher: str,
        ctx: PiTContext,
        target_status: str = "TERMINATED",
        default_var: str = "r",
    ) -> str:
        """Inject audit filter targeting specific lifecycle states (e.g. TERMINATED)."""
        if not isinstance(ctx, PiTContext):
            raise TypeError(f"ctx must be a PiTContext instance, got {type(ctx).__name__}")

        clean = cypher.strip()
        if not clean or clean.lower() == "(none)":
            return clean

        bound_cypher, rel_specs = self._ensure_relationship_vars(clean, default_var)
        if not rel_specs:
            return clean

        primary_var, is_varlen = rel_specs[0]
        date_iso = ctx.as_of_iso
        if ctx.include_unanchored:
            from_clause = f"({primary_var}.valid_from IS NULL OR {primary_var}.valid_from <= '{date_iso}')"
        else:
            from_clause = f"{primary_var}.valid_from IS NOT NULL AND {primary_var}.valid_from <= '{date_iso}'"

        if is_varlen:
            inner_from = (
                f"(_rel.valid_from IS NULL OR _rel.valid_from <= '{date_iso}')"
                if ctx.include_unanchored
                else f"_rel.valid_from IS NOT NULL AND _rel.valid_from <= '{date_iso}'"
            )
            audit_pred = f"ALL(_rel IN {primary_var} WHERE _rel.status = '{target_status}' AND {inner_from})"
        else:
            audit_pred = f"{primary_var}.status = '{target_status}' AND {from_clause}"

        return self._inject_where_predicate(bound_cypher, audit_pred)

    def inject_news_cutoff(
        self,
        cypher: str,
        ctx: PiTContext,
        news_var: str = "n",
    ) -> str:
        """Inject a Point-in-Time `published_at` ceiling on `:News` node lookups."""
        if not isinstance(ctx, PiTContext):
            raise TypeError(f"ctx must be a PiTContext instance, got {type(ctx).__name__}")

        clean = cypher.strip()
        if not clean:
            return clean
        pred = f"{news_var}.published_at IS NOT NULL AND {news_var}.published_at <= '{ctx.cutoff_utc_iso}'"
        if f"{news_var}.published_at <=" in clean:
            return clean
        return self._inject_where_predicate(clean, pred)

    def _ensure_relationship_vars(
        self, cypher: str, default_var: str
    ) -> Tuple[str, List[Tuple[str, bool]]]:
        """Extract all relationship variables and bind anonymous relationships deterministically."""
        rel_specs: List[Tuple[str, bool]] = []
        used_vars: set[str] = set()
        counter = 0

        # Pre-collect existing named variables so generated bindings never collide
        for match in _REL_BRACKET_RE.finditer(cypher):
            spec = _REL_SPEC_RE.match(match.group(2))
            if spec and spec.group("var"):
                used_vars.add(spec.group("var"))

        def _replace(match: re.Match[str]) -> str:
            nonlocal counter
            left_dash, inner, right_dash = match.group(1), match.group(2), match.group(3)
            spec = _REL_SPEC_RE.match(inner)
            if spec is None:
                return match.group(0)

            var_name = spec.group("var")
            types_part = spec.group("types")
            varlen_part = spec.group("varlen") or ""
            props_part = spec.group("props") or ""
            is_varlen = bool(varlen_part.strip())

            if not var_name:
                candidate = default_var if counter == 0 and default_var not in used_vars else f"{default_var}_{counter}"
                while candidate in used_vars:
                    counter += 1
                    candidate = f"{default_var}_{counter}"
                var_name = candidate
                used_vars.add(var_name)
                counter += 1
                rel_specs.append((var_name, is_varlen))
                type_str = f":{types_part}" if types_part else ""
                return f"{left_dash}[{var_name}{type_str}{varlen_part}{props_part}]{right_dash}"

            rel_specs.append((var_name, is_varlen))
            return match.group(0)

        rewritten = _REL_BRACKET_RE.sub(_replace, cypher)
        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: List[Tuple[str, bool]] = []
        for var_name, is_varlen in rel_specs:
            if var_name not in seen:
                seen.add(var_name)
                deduped.append((var_name, is_varlen))
        return rewritten, deduped

    def _inject_where_predicate(self, cypher: str, predicate: str) -> str:
        """Inject predicate into the query's active MATCH/OPTIONAL MATCH WHERE clause."""
        # If query has OPTIONAL MATCH, scope WHERE to the OPTIONAL MATCH block rather than earlier MATCH
        opt_match = re.search(r"\bOPTIONAL\s+MATCH\b", cypher, re.IGNORECASE)
        search_start = opt_match.end() if opt_match else 0

        tail = cypher[search_start:]
        clause_match = re.search(r"\b(RETURN|WITH|ORDER\s+BY|LIMIT)\b", tail, re.IGNORECASE)
        clause_end = search_start + clause_match.start() if clause_match else len(cypher)
        segment = cypher[search_start:clause_end]

        where_match = re.search(r"\bWHERE\b", segment, re.IGNORECASE)
        if where_match:
            where_pos = search_start + where_match.end()
            return f"{cypher[:where_pos]} ({predicate}) AND{cypher[where_pos:]}"

        if clause_match:
            return (
                f"{cypher[:clause_end].rstrip()} WHERE {predicate} "
                f"{cypher[clause_end:].lstrip()}"
            )

        return f"{cypher.rstrip()} WHERE {predicate}"


# Default singleton filter
DEFAULT_TEMPORAL_FILTER = TemporalQueryFilter()


def inject_active_temporal_filter(
    cypher: str,
    ctx: PiTContext,
    default_var: str = "r",
) -> str:
    """Convenience functional wrapper for ADR-00033 temporal state injection."""
    return DEFAULT_TEMPORAL_FILTER.inject_active_coalescence(cypher, ctx, default_var)
