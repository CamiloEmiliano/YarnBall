"""
5-Axis Financial Annotation Engine & Taxonomy Formatter.

Annotates financial knowledge graph triples across 5 orthogonal dimensions:
- Axis 1: Entity Typology (Company, Subsidiary, Person, RegulatoryBody, Product, CommodityRisk)
- Axis 2: Relational Ontology (SUPPLIES_TO, SOLE_SOURCE_DEPENDENT_ON, LICENSES_FROM, etc.)
- Axis 3: Directional Polarity & Sentiment (EXPANDING_BULLISH, NEUTRAL_STABLE, CONTRACTING_BEARISH, DISRUPTIVE_SHOCK)
- Axis 4: Financial Materiality & Criticality (CRITICAL_TIER_1, MATERIAL_TIER_2, COMMODITY_TIER_3)
- Axis 5: Temporal Provenance & Lifecycle (status, valid_from, valid_to, provenance, confidence)
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from graph.quality_controls import (
    validate_and_orient_triple,
    compute_edge_confidence,
    is_generic_placeholder,
)
from graph.entity_resolver import _jaro_winkler_similarity
from tools.sp500_universe import SP500UniverseManager

logger = logging.getLogger("taxonomy_annotator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Polarity Lexical Triggers
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
    """Check if pattern matches in text without being preceded by a negation cue in the local clause."""
    if not text:
        return False
    for m in re.finditer(pattern, text, flags=re.IGNORECASE):
        start = m.start()
        # Look back within the local clause (stopping at clause/sentence boundaries ., ;, !)
        preceding = text[max(0, start - window_chars):start]
        clause_boundary = max(preceding.rfind("."), preceding.rfind(";"), preceding.rfind("!"))
        if clause_boundary != -1:
            preceding = preceding[clause_boundary + 1:]
        
        if not NEGATION_REGEX.search(preceding):
            return True
    return False

# Materiality Rules
CRITICAL_MATERIALITY_TERMS: Set[str] = {
    "sole source", "single source", "exclusive", "asc 280", "10% of revenue",
    "primary foundry", "100% owned", "wholly-owned", "core operating",
}


@dataclass
class AnnotatedTriple:
    """Fully grounded 5-Axis financial triple."""
    # Required entity and relationship fields (non-defaults)
    source_name: str
    source_type: str  # Company, Subsidiary, Person, RegulatoryBody, Product, CommodityRisk
    target_name: str
    target_type: str
    rel_type: str

    # Optional Identifiers (Axis 1)
    source_ticker: Optional[str] = None
    source_cik: Optional[str] = None
    target_ticker: Optional[str] = None
    target_cik: Optional[str] = None

    # Directional Polarity (Axis 3)
    polarity: str = "NEUTRAL_STABLE"  # EXPANDING_BULLISH, NEUTRAL_STABLE, CONTRACTING_BEARISH, DISRUPTIVE_SHOCK

    # Financial Materiality (Axis 4)
    materiality: str = "MATERIAL_TIER_2"  # CRITICAL_TIER_1, MATERIAL_TIER_2, COMMODITY_TIER_3

    # Temporal & Provenance (Axis 5)
    status: str = "ACTIVE_CURRENT"  # ACTIVE_CURRENT, TERMINATED
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    provenance: str = "SEC_10K_ITEM1"
    confidence: float = 0.95
    nature: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_cypher_dsl(self) -> str:
        """Format as strict OpenCypher DSL string for student training target."""
        # Source node
        src_props = [f'name: "{self.source_name}"']
        if self.source_ticker:
            src_props.append(f'ticker: "{self.source_ticker}"')
        if self.source_cik:
            src_props.append(f'cik: "{self.source_cik}"')
        src_str = f"(:{self.source_type} {{{', '.join(src_props)}}})"

        # Edge properties (5-Axis)
        edge_props = [
            f'polarity: "{self.polarity}"',
            f'materiality: "{self.materiality}"',
            f'status: "{self.status}"',
            f'provenance: "{self.provenance}"',
            f'confidence: {self.confidence:.2f}',
        ]
        if self.valid_from:
            edge_props.append(f'valid_from: "{self.valid_from}"')
        if self.valid_to:
            edge_props.append(f'valid_to: "{self.valid_to}"')
        if self.nature:
            edge_props.append(f'nature: "{self.nature}"')
        edge_str = f"-[:{self.rel_type} {{{', '.join(edge_props)}}}]->"

        # Target node
        tgt_props = [f'name: "{self.target_name}"']
        if self.target_ticker:
            tgt_props.append(f'ticker: "{self.target_ticker}"')
        if self.target_cik:
            tgt_props.append(f'cik: "{self.target_cik}"')
        tgt_str = f"(:{self.target_type} {{{', '.join(tgt_props)}}})"

        return f"{src_str}{edge_str}{tgt_str}"


class FinancialTaxonomyAnnotator:
    """Enriches raw extraction candidates with full 5-Axis taxonomy metadata."""

    def __init__(self, universe_mgr: Optional[SP500UniverseManager] = None):
        self.universe_mgr = universe_mgr or SP500UniverseManager()
        self._init_entity_lookups()

    def _init_entity_lookups(self) -> None:
        """Pre-index S&P 500 company names, tickers, and CIKs."""
        constituents = self.universe_mgr.get_all_records()
        self.ticker_to_constituent = {c.ticker.upper(): c for c in constituents}
        self.name_to_constituent = {c.company_name.lower(): c for c in constituents}
        self.cik_to_constituent = {c.cik.zfill(10): c for c in constituents if c.cik}

    def infer_entity_typology(
        self,
        name: str,
        explicit_type: Optional[str] = None,
        event_date: Optional[str] = None,
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Axis 1: Determine Entity Typology, Ticker, and CIK with Point-in-Time and Best-Score Fuzzy Selection.
        Returns: (entity_type, ticker, cik)
        """
        if not name or is_generic_placeholder(name):
            return ("Entity", None, None)

        clean_name = name.strip()
        name_lower = clean_name.lower()
        clean_upper = clean_name.upper()

        # Build point-in-time constituent lookups if event_date provided
        if event_date:
            active_constituents = self.universe_mgr.get_constituents_at_date(event_date)
            ticker_map = {c.ticker.upper(): c for c in active_constituents}
            name_map = {c.company_name.lower(): c for c in active_constituents}
        else:
            ticker_map = self.ticker_to_constituent
            name_map = self.name_to_constituent

        # 1. Direct Ticker Match
        if clean_upper in ticker_map:
            c = ticker_map[clean_upper]
            return ("Company", c.ticker, c.cik)

        # 2. Direct Company Name Match
        if name_lower in name_map:
            c = name_map[name_lower]
            return ("Company", c.ticker, c.cik)

        # 3. Best-Score Fuzzy S&P 500 Match (Max similarity ranking, not dict iteration order)
        best_match = None
        best_score = 0.0

        for c_name, c in name_map.items():
            if len(c_name) >= 5 and (c_name in name_lower or name_lower in c_name):
                containment_score = len(c_name) / max(len(c_name), len(name_lower))
                if containment_score > best_score:
                    best_score = containment_score
                    best_match = c
            else:
                sim = _jaro_winkler_similarity(name_lower, c_name)
                if sim >= 0.88 and sim > best_score:
                    best_score = sim
                    best_match = c

        if best_match is not None and best_score >= 0.60:
            return ("Company", best_match.ticker, best_match.cik)

        # Fallback to general universe if event_date was restrictive
        if event_date and (clean_upper in self.ticker_to_constituent or name_lower in self.name_to_constituent):
            c = self.ticker_to_constituent.get(clean_upper) or self.name_to_constituent.get(name_lower)
            if c:
                return ("Company", c.ticker, c.cik)

        # 4. Infer Non-Company Typologies
        if explicit_type:
            exp_upper = explicit_type.strip().upper()
            if exp_upper in ["PERSON", "EXECUTIVE", "CEO", "CFO"]:
                return ("Person", None, None)
            if exp_upper in ["REGULATORYBODY", "GOVERNMENT", "AGENCY"]:
                return ("RegulatoryBody", None, None)
            if exp_upper in ["SUBSIDIARY", "DIVISION", "UNIT"]:
                return ("Subsidiary", None, None)
            if exp_upper in ["PRODUCT", "PLATFORM", "CHIP", "MODEL"]:
                return ("Product", None, None)
            if exp_upper in ["COMMODITY", "RISK", "CHOKEPOINT"]:
                return ("CommodityRisk", None, None)

        # Heuristic checks for Regulatory Bodies and People
        regulatory_names = {"sec", "sec edgar", "doj", "ftc", "fda", "epa", "cftc", "fed", "federal reserve", "treasury"}
        if name_lower in regulatory_names:
            return ("RegulatoryBody", None, None)

        # Default fallback
        return (explicit_type or "Company", None, None)

    def infer_directional_polarity(
        self,
        context_text: str,
        rel_type: str,
        market_sentiment: Optional[str] = None,
    ) -> str:
        """Axis 3: Classify Directional Polarity & Sentiment with Negation-Scope Protection."""
        if market_sentiment in ["EXPANDING_BULLISH", "NEUTRAL_STABLE", "CONTRACTING_BEARISH", "DISRUPTIVE_SHOCK"]:
            return market_sentiment

        clean_rel = rel_type.strip().upper()
        if clean_rel in ["DEFAULTED_ON", "TERMINATED", "CONTRACT_TERMINATION"]:
            return "DISRUPTIVE_SHOCK"

        text_val = context_text or ""

        # Check triggers with negation filtering: most severe to positive
        for shock_pat in POLARITY_TRIGGERS["DISRUPTIVE_SHOCK"]:
            if has_unnegated_pattern(shock_pat, text_val):
                return "DISRUPTIVE_SHOCK"

        for bear_pat in POLARITY_TRIGGERS["CONTRACTING_BEARISH"]:
            if has_unnegated_pattern(bear_pat, text_val):
                return "CONTRACTING_BEARISH"

        for bull_pat in POLARITY_TRIGGERS["EXPANDING_BULLISH"]:
            if has_unnegated_pattern(bull_pat, text_val):
                return "EXPANDING_BULLISH"

        return "NEUTRAL_STABLE"

    def infer_financial_materiality(
        self,
        context_text: str,
        rel_type: str,
        provenance: Optional[str] = None,
    ) -> str:
        """Axis 4: Determine Financial Materiality & Criticality."""
        clean_rel = rel_type.strip().upper()
        prov = (provenance or "").upper()
        text_lower = (context_text or "").lower()

        # Tier 1 Critical: Exhibit 21 subsidiaries, Sole-Source, or Form 8-K M&A
        if clean_rel in ["SOLE_SOURCE_DEPENDENT_ON", "ACQUIRED_BY", "DEFAULTED_ON"]:
            return "CRITICAL_TIER_1"
        if prov in ["SEC_EXHIBIT_21", "SEC_8K"]:
            return "CRITICAL_TIER_1"

        for term in CRITICAL_MATERIALITY_TERMS:
            if term in text_lower:
                return "CRITICAL_TIER_1"

        # Tier 3 Commodity: Generic off-the-shelf suppliers
        if any(w in text_lower for w in ["routine supplier", "standard vendor", "off-the-shelf", "minor supply"]):
            return "COMMODITY_TIER_3"

        return "MATERIAL_TIER_2"

    def annotate_triple(
        self,
        raw_source: str,
        raw_target: str,
        raw_rel: str,
        context_text: str = "",
        explicit_source_type: Optional[str] = None,
        explicit_target_type: Optional[str] = None,
        event_date: Optional[str] = None,
        provenance: Optional[str] = "SEC_10K_ITEM1",
        market_sentiment: Optional[str] = None,
        nature_summary: Optional[str] = None,
    ) -> Optional[AnnotatedTriple]:
        """
        Apply complete 5-Axis annotation pipeline to a candidate triple.
        Validates schema, links identifiers, calibrates confidence, and formats DSL.
        """
        # Axis 1: Entity Typology & Grounding
        src_type, src_ticker, src_cik = self.infer_entity_typology(raw_source, explicit_source_type, event_date=event_date)
        tgt_type, tgt_ticker, tgt_cik = self.infer_entity_typology(raw_target, explicit_target_type, event_date=event_date)

        # Axis 2: Relational Ontology Validation & Auto-Orientation
        validated = validate_and_orient_triple(
            src_id=raw_source,
            src_label=src_type,
            tgt_id=raw_target,
            tgt_label=tgt_type,
            rel_type=raw_rel,
        )
        if not validated:
            return None

        v_src, v_src_lbl, v_tgt, v_tgt_lbl, v_rel = validated

        # Re-assign if inverted during validation
        if v_src != raw_source:
            # Swapped
            final_src, final_src_type, final_src_ticker, final_src_cik = raw_target, tgt_type, tgt_ticker, tgt_cik
            final_tgt, final_tgt_type, final_tgt_ticker, final_tgt_cik = raw_source, src_type, src_ticker, src_cik
        else:
            final_src, final_src_type, final_src_ticker, final_src_cik = raw_source, src_type, src_ticker, src_cik
            final_tgt, final_tgt_type, final_tgt_ticker, final_tgt_cik = raw_target, tgt_type, tgt_ticker, tgt_cik

        # Axis 3: Directional Polarity
        polarity = self.infer_directional_polarity(context_text, v_rel, market_sentiment)

        # Axis 4: Financial Materiality
        materiality = self.infer_financial_materiality(context_text, v_rel, provenance)

        # Axis 5: Temporal Provenance & Confidence
        prov_key = provenance or "SEC_10K_ITEM1"
        confidence = compute_edge_confidence(provenance=prov_key, mention_count=1)

        # Decouple relationship lifecycle status from co-occurring sentence sentiment (ISSUE-10)
        terminal_relations = {"TERMINATED", "DEFAULTED_ON", "CONTRACT_TERMINATION", "ACQUIRED_BY"}
        if v_rel in terminal_relations:
            status = "TERMINATED"
        elif has_unnegated_pattern(r"\b(contract\s+terminated|partnership\s+dissolved|severed\s+relationship|terminated\s+agreement)\b", context_text):
            status = "TERMINATED"
        else:
            status = "ACTIVE_CURRENT"

        date_str = (event_date or datetime.now().strftime("%Y-%m-%d"))[:10]

        return AnnotatedTriple(
            source_name=final_src,
            source_type=final_src_type,
            source_ticker=final_src_ticker,
            source_cik=final_src_cik,
            target_name=final_tgt,
            target_type=final_tgt_type,
            target_ticker=final_tgt_ticker,
            target_cik=final_tgt_cik,
            rel_type=v_rel,
            polarity=polarity,
            materiality=materiality,
            status=status,
            valid_from=date_str,
            valid_to=date_str if status == "TERMINATED" else None,
            provenance=prov_key,
            confidence=confidence,
            nature=nature_summary or v_rel.lower().replace("_", " "),
        )


def main() -> None:
    annotator = FinancialTaxonomyAnnotator()
    sample_text = (
        "Apple Inc. relies on Taiwan Semiconductor Manufacturing Company (TSMC) on a sole source basis "
        "for its 3nm Apple Silicon processors. TSMC delivered record wafer volumes to Apple in Cupertino."
    )

    triple = annotator.annotate_triple(
        raw_source="TSMC",
        raw_target="Apple Inc.",
        raw_rel="SOLE_SOURCE_DEPENDENT_ON",
        context_text=sample_text,
        provenance="SEC_10K_ITEM1",
        nature_summary="3nm Apple Silicon wafer manufacturing",
    )

    if triple:
        print("\n--- 5-Axis Annotated Triple Object ---")
        print(json.dumps(triple.to_dict(), indent=2))
        print("\n--- OpenCypher DSL Target Output ---")
        print(triple.to_cypher_dsl())


if __name__ == "__main__":
    main()
