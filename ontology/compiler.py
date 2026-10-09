"""
Ontological DSL Compiler & Lexical Trigger Harvesting.

Academic Foundations:
- Fillmore, C. J. (1982): "Frame Semantics" (Semantic Role Labeling & Trigger Mining)
- OpenCypher Specification (ISO/IEC 39075 GQL Graph Query Language)

Translates natural language financial clauses into formal 5-Axis OpenCypher DSL targets.
Extracts discourse polarity cues with local clause negation-boundary checking.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

try:
    from .reification import (
        AnnotatedTriple,
        CRITICAL_MATERIALITY_TERMS,
        MaterialityTier,
        PolarityAxis,
    )
except ImportError:
    try:
        from ontology.reification import (
            AnnotatedTriple,
            CRITICAL_MATERIALITY_TERMS,
            MaterialityTier,
            PolarityAxis,
        )
    except ImportError:
        from graphrag_finance.ontology.reification import (
            AnnotatedTriple,
            CRITICAL_MATERIALITY_TERMS,
            MaterialityTier,
            PolarityAxis,
        )

# ----------------------------------------------------------------------
# Polarity Lexical Triggers (Semantic Frame Anchors)
# ----------------------------------------------------------------------
POLARITY_TRIGGERS: Dict[str, List[str]] = {
    "DISRUPTIVE_SHOCK": [
        r"defaulted", r"bankruptcy", r"chapter 11", r"emergency\s+shutdown",
        r"export\s+ban", r"trade\s+embargo", r"catastrophic", r"force\s+majeure",
        r"terminated\s+for\s+cause", r"breach\s+of\s+contract",
    ],
    "CONTRACTING_BEARISH": [
        r"cutbacks?", r"reduced\s+orders?", r"margin\s+compression",
        r"downgraded", r"canceled", r"lawsuit", r"contract\s+termination",
        r"investigation", r"delayed", r"dropped", r"loss\s+of\s+customer",
    ],
    "EXPANDING_BULLISH": [
        r"expand(?:s|ed|ing)?", r"multi-year\s+(?:deal|agreement|contract|partnership)",
        r"surged?", r"record\s+(?:revenue|volume|orders?|commitment)",
        r"partnership\s+expansion", r"awarded", r"joint\s+venture",
        r"strategic\s+(?:collaboration|agreement|partnership)",
        r"capacity\s+increase", r"sole\s+source\s+win", r"accelerat(?:es?|ed|ing)",
    ],
}

NEGATION_REGEX = re.compile(
    r"\b(not|no|never|avoided|avoiding|without|prevented|preventing|neither|nor|denied|unlikely\s+to)\b",
    re.IGNORECASE,
)


def has_unnegated_pattern(pattern: str, text: str, window_chars: int = 45) -> bool:
    """
    Check if regex pattern matches in text without being preceded by a negation cue
    in the local syntactic clause.
    """
    if not text:
        return False
    for m in re.finditer(pattern, text, flags=re.IGNORECASE):
        start = m.start()
        # Look back within the local clause (stopping at clause/sentence boundaries ., ;, !)
        preceding = text[max(0, start - window_chars) : start]
        clause_boundary = max(preceding.rfind("."), preceding.rfind(";"), preceding.rfind("!"))
        if clause_boundary != -1:
            preceding = preceding[clause_boundary + 1 :]

        if not NEGATION_REGEX.search(preceding):
            return True
    return False


def classify_polarity_from_text(
    context_text: str = "",
    rel_type: Optional[str] = None,
    market_sentiment: Optional[str] = None,
) -> str:
    """
    Axis 3: Classify Directional Sentiment & Economic Polarity with Negation-Scope Protection.
    Applies priority hierarchy: Explicit Market Sentiment -> Relational Terminal Shock -> Lexical Triggers -> Neutral Stable.
    """
    if market_sentiment in ["EXPANDING_BULLISH", "NEUTRAL_STABLE", "CONTRACTING_BEARISH", "DISRUPTIVE_SHOCK"]:
        return market_sentiment

    clean_rel = (rel_type or "").strip().upper()
    if clean_rel in ["DEFAULTED_ON", "TERMINATED", "CONTRACT_TERMINATION"]:
        return PolarityAxis.DISRUPTIVE_SHOCK.value

    text_val = context_text or ""

    # Check triggers with negation filtering: most severe to positive
    for shock_pat in POLARITY_TRIGGERS["DISRUPTIVE_SHOCK"]:
        if has_unnegated_pattern(shock_pat, text_val):
            return PolarityAxis.DISRUPTIVE_SHOCK.value

    for bear_pat in POLARITY_TRIGGERS["CONTRACTING_BEARISH"]:
        if has_unnegated_pattern(bear_pat, text_val):
            return PolarityAxis.CONTRACTING_BEARISH.value

    for bull_pat in POLARITY_TRIGGERS["EXPANDING_BULLISH"]:
        if has_unnegated_pattern(bull_pat, text_val):
            return PolarityAxis.EXPANDING_BULLISH.value

    return PolarityAxis.NEUTRAL_STABLE.value


def classify_financial_materiality(
    context_text: str = "",
    rel_type: str = "",
    provenance: Optional[str] = None,
) -> str:
    """
    Axis 4: Determine Financial Materiality & Criticality Tier.
    """
    clean_rel = rel_type.strip().upper()
    prov = (provenance or "").upper()
    text_lower = (context_text or "").lower()

    # Tier 1 Critical: Exhibit 21 subsidiaries, Sole-Source, Form 8-K M&A, Defaults
    if clean_rel in ["SOLE_SOURCE_DEPENDENT_ON", "ACQUIRED_BY", "DEFAULTED_ON"]:
        return MaterialityTier.CRITICAL_TIER_1.value
    if prov in ["SEC_EXHIBIT_21", "SEC_8K"]:
        return MaterialityTier.CRITICAL_TIER_1.value

    for term in CRITICAL_MATERIALITY_TERMS:
        if term in text_lower:
            return MaterialityTier.CRITICAL_TIER_1.value

    # Tier 3 Commodity: Generic off-the-shelf suppliers
    if any(w in text_lower for w in ["routine supplier", "standard vendor", "off-the-shelf", "minor supply"]):
        return MaterialityTier.COMMODITY_TIER_3.value

    return MaterialityTier.MATERIAL_TIER_2.value


def compile_to_cypher_dsl(triple: AnnotatedTriple) -> str:
    """
    Compiles an AnnotatedTriple into a valid, strict OpenCypher DSL statement.
    """
    return triple.to_cypher_dsl()
