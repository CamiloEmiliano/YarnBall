"""
Multi-Axis Relational Reification Model.

Academic Foundations:
- Staab, S., & Studer, R. (2009): "Handbook on Ontologies" (Reification & Property Graph Modeling)
- US GAAP ASC 280: Segment Reporting (>10% revenue materiality disclosure rules)

Extends plain knowledge graph triples with rich reified dimensions:
- Axis 1: Entity Typology & Grounded SEC CIK / Ticker Identifiers
- Axis 2: Relational Predicate Ontology
- Axis 3: Directional Sentiment & Economic Polarity
- Axis 4: Financial Materiality & Criticality Tier
- Axis 5: Temporal Provenance & Regulatory Lifecycle
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Optional, Set


class PolarityAxis(str, Enum):
    """Axis 3: Directional Polarity & Market Impact."""
    EXPANDING_BULLISH = "EXPANDING_BULLISH"
    NEUTRAL_STABLE = "NEUTRAL_STABLE"
    CONTRACTING_BEARISH = "CONTRACTING_BEARISH"
    DISRUPTIVE_SHOCK = "DISRUPTIVE_SHOCK"


class MaterialityTier(str, Enum):
    """Axis 4: Financial Materiality & Criticality."""
    CRITICAL_TIER_1 = "CRITICAL_TIER_1"  # >10% revenue, sole source, primary foundry
    MATERIAL_TIER_2 = "MATERIAL_TIER_2"  # Major multi-year contracts, cross-licensing
    COMMODITY_TIER_3 = "COMMODITY_TIER_3"  # Standard commercial, interchangeable vendor


class LifecycleStatus(str, Enum):
    """Axis 5: Temporal Lifecycle Status."""
    ACTIVE_CURRENT = "ACTIVE_CURRENT"
    TERMINATED = "TERMINATED"


CRITICAL_MATERIALITY_TERMS: Set[str] = {
    "sole source", "single source", "exclusive", "asc 280", "10% of revenue",
    "primary foundry", "100% owned", "wholly-owned", "core operating",
}


@dataclass
class AnnotatedTriple:
    """
    Fully grounded 5-Axis financial triple representing an ontological edge.
    """
    # Core Entity Typologies & Relation
    source_name: str
    source_type: str  # Company, Subsidiary, Person, RegulatoryBody, Product, CommodityRisk
    target_name: str
    target_type: str
    rel_type: str

    # Optional Canonical Identifiers (Axis 1)
    source_ticker: Optional[str] = None
    source_cik: Optional[str] = None
    target_ticker: Optional[str] = None
    target_cik: Optional[str] = None

    # Directional Polarity (Axis 3)
    polarity: str = PolarityAxis.NEUTRAL_STABLE.value

    # Financial Materiality (Axis 4)
    materiality: str = MaterialityTier.MATERIAL_TIER_2.value

    # Temporal & Provenance (Axis 5)
    status: str = LifecycleStatus.ACTIVE_CURRENT.value
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    provenance: str = "SEC_10K_ITEM1"
    confidence: float = 0.95
    nature: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_cypher_dsl(self) -> str:
        """
        Format as strict OpenCypher DSL string for SFT targets and Memgraph ingestion.
        Example:
            (:Company {name: "Apple", ticker: "AAPL"})-[:SUPPLIES_TO {polarity: "NEUTRAL_STABLE", ...}]->(:Company {name: "TSMC"})
        """
        # Source node properties
        src_props = [f'name: "{self.source_name}"']
        if self.source_ticker:
            src_props.append(f'ticker: "{self.source_ticker}"')
        if self.source_cik:
            src_props.append(f'cik: "{self.source_cik}"')
        src_str = f"(:{self.source_type} {{{', '.join(src_props)}}})"

        # Edge reified properties (5-Axis)
        edge_props = [
            f'polarity: "{self.polarity}"',
            f'materiality: "{self.materiality}"',
            f'status: "{self.status}"',
            f'provenance: "{self.provenance}"',
            f'confidence: {self.confidence:.2f}',
        ]
        if self.nature:
            # Escape inner double quotes
            safe_nature = self.nature.replace('"', '\\"')
            edge_props.append(f'nature: "{safe_nature}"')
        if self.valid_from:
            edge_props.append(f'valid_from: "{self.valid_from}"')
        if self.valid_to:
            edge_props.append(f'valid_to: "{self.valid_to}"')

        edge_str = f"-[:{self.rel_type} {{{', '.join(edge_props)}}}]->"

        # Target node properties
        tgt_props = [f'name: "{self.target_name}"']
        if self.target_ticker:
            tgt_props.append(f'ticker: "{self.target_ticker}"')
        if self.target_cik:
            tgt_props.append(f'cik: "{self.target_cik}"')
        tgt_str = f"(:{self.target_type} {{{', '.join(tgt_props)}}})"

        return f"{src_str}{edge_str}{tgt_str}"
