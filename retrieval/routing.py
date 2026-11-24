"""Deterministic Rules-First BI Query Router (ADR-00037 Section 4.3).

Routes analytical BI queries across `local`, `global`, and `hybrid` execution paths
without invoking an LLM (zero added latency, zero mode-bleed risk, preserving 7B/9B
SFT/RLAIF curriculum capacity under ADR-00012):
- Explicit `route` override (`"local"`, `"global"`, `"hybrid"`) is always honored.
- Under `"auto"`:
  1. Resolved anchor entity mention(s) only -> `"local"` (ego traversal + cascade math).
  2. Macro / sector / cluster / thematic cues with no anchor entity -> `"global"`
     (map-reduce over cached PiT `community_reports` where `as_of <= ctx.as_of`).
  3. Both resolved anchor entity and macro / cluster cues -> `"hybrid"`
     (`local` cascade plus top-matching PiT community reports).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

RouteMode = Literal["auto", "local", "global", "hybrid"]
ResolvedRoute = Literal["local", "global", "hybrid"]

_MACRO_CLUSTER_CUE_RE = re.compile(
    r"\b("
    r"clusters?|sectors?|industries|industry-wide|ecosystems?|systemic|thematic|"
    r"s&p\s*500|sp500|across\s+(?:the\s+)?(?:s&p|sp500|index|market|all\s+companies|sectors?)|"
    r"which\s+(?:s&p\s*500\s+)?(?:clusters?|companies|firms|sectors?|groups?)|"
    r"concentration\s+risk|supply[- ]chain\s+clusters?|peer\s+groups?|"
    r"macro(?:economic)?|geopolitical|taiwanese\s+foundry|foundry\s+capacity|"
    r"broad\s+exposure|regional\s+exposure|index-wide|market-wide"
    r")\b",
    re.IGNORECASE,
)

_TICKER_TOKEN_RE = re.compile(r"\b[A-Z]{2,5}\b")
_CAPITALIZED_SPAN_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&.-]+(?:\s+(?:Inc\.?|Corp\.?|Corporation|Ltd\.?|Co\.?|Semiconductor|Manufacturing|Holdings|Technologies|Systems|Devices|Electronics|Group|Platforms|Micro|Motors))?)\b"
)

_NON_ENTITY_ACRONYMS: frozenset[str] = frozenset(
    {
        "AI", "AM", "API", "AS", "AT", "BI", "BY", "CEO", "CFO", "CIK", "CN", "CO",
        "DAG", "DB", "DO", "EDGAR", "EDT", "EST", "EU", "EV", "FAB", "FAQ", "FDA",
        "FOR", "FX", "GAAP", "GDP", "GPU", "HOW", "ID", "IF", "IN", "IP", "IS",
        "IT", "JP", "KG", "LLM", "MD", "MDA", "NA", "NO", "NYSE", "OF", "ON", "OR",
        "PC", "PIT", "PM", "Q1", "Q2", "Q3", "Q4", "RAG", "RF", "SEC", "SFT", "SO",
        "SP", "SP500", "THE", "TO", "TW", "UK", "US", "USA", "USD", "VS", "WHAT",
        "WHEN", "WHICH", "WHO", "WHY", "WITH",
    }
)

_NON_ENTITY_PROPER_NOUNS: frozenset[str] = frozenset(
    {
        "across", "after", "all", "america", "american", "analyze", "asia", "asian",
        "before", "china", "chinese", "compare", "describe", "during", "europe",
        "european", "explain", "foundry", "global", "how", "identify", "impact",
        "in", "index", "industry", "item", "japan", "japanese", "korea", "korean",
        "list", "macro", "market", "of", "on", "point", "quarter", "report",
        "risk", "sector", "semiconductor", "show", "supply", "taiwan", "taiwanese",
        "the", "what", "when", "where", "which", "who", "why",
    }
)


def _extract_candidate_mentions(query: str) -> Tuple[List[str], List[str]]:
    """Extract candidate uppercase ticker tokens and capitalized company spans."""
    tickers: List[str] = []
    for tok in _TICKER_TOKEN_RE.findall(query):
        if tok not in _NON_ENTITY_ACRONYMS and tok not in tickers:
            tickers.append(tok)

    names: List[str] = []
    for m in _CAPITALIZED_SPAN_RE.finditer(query):
        span = m.group(1).strip(".,;:?!()[]\"'")
        if not span:
            continue
        lower_first = span.split()[0].lower()
        if lower_first in _NON_ENTITY_PROPER_NOUNS:
            continue
        if span.upper() in _NON_ENTITY_ACRONYMS:
            continue
        if span not in names:
            names.append(span)

    return tickers, names


def _resolve_anchor_entities(query: str, linker: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Resolve concrete corporate entity mentions using offline-safe `EdgarEntityLinker`."""
    active_linker = linker
    if active_linker is None:
        from ingestion.sec.parser import EdgarEntityLinker

        active_linker = EdgarEntityLinker(pg_conn=None)

    tickers, names = _extract_candidate_mentions(query)
    resolved: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    for tk in tickers:
        hit = active_linker.link_entity(ticker=tk)
        if hit and (hit.get("ticker") or hit.get("company_name")):
            key = str(hit.get("ticker") or hit.get("company_name")).upper()
            if key not in seen_ids:
                seen_ids.add(key)
                resolved.append(
                    {
                        "mention": tk,
                        "ticker": hit.get("ticker"),
                        "company_name": hit.get("company_name"),
                        "cik": hit.get("cik"),
                    }
                )

    for nm in names:
        hit = active_linker.link_entity(name=nm)
        if hit and (hit.get("ticker") or hit.get("company_name")):
            key = str(hit.get("ticker") or hit.get("company_name")).upper()
            if key not in seen_ids:
                seen_ids.add(key)
                resolved.append(
                    {
                        "mention": nm,
                        "ticker": hit.get("ticker"),
                        "company_name": hit.get("company_name"),
                        "cik": hit.get("cik"),
                    }
                )

    return resolved


def classify_query_route(
    query: str,
    *,
    route: RouteMode = "auto",
    linker: Optional[Any] = None,
) -> Tuple[ResolvedRoute, Dict[str, Any]]:
    """Classify a BI query deterministically into `'local'`, `'global'`, or `'hybrid'` (ADR-00037)."""
    if route in ("local", "global", "hybrid"):
        return route, {
            "requested_route": route,
            "resolved_route": route,
            "override": True,
            "resolved_entities": [],
            "macro_cues": [],
        }

    if route != "auto":
        raise ValueError(f"Unsupported route={route!r}; expected one of 'auto', 'local', 'global', 'hybrid'.")

    macro_cues = sorted({m.group(0).strip().lower() for m in _MACRO_CLUSTER_CUE_RE.finditer(query or "")})
    resolved_entities = _resolve_anchor_entities(query or "", linker=linker)

    has_entities = len(resolved_entities) > 0
    has_macro = len(macro_cues) > 0

    resolved: ResolvedRoute
    if has_entities and has_macro:
        resolved = "hybrid"
    elif has_macro and not has_entities:
        resolved = "global"
    else:
        resolved = "local"

    return resolved, {
        "requested_route": "auto",
        "resolved_route": resolved,
        "override": False,
        "resolved_entities": resolved_entities,
        "macro_cues": macro_cues,
    }


__all__ = [
    "ResolvedRoute",
    "RouteMode",
    "classify_query_route",
]
