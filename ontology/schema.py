"""
Formal Ontological Schema & Domain/Range Axiomatization (TBox).

Academic Foundations:
- Gruber, T. R. (1993): "A Translation Approach to Portable Ontology Specifications"
- Noy, N. F., & McGuinness, D. L. (2001): "Ontology Development 101: A Guide to Creating Your First Ontology"
- W3C RDF / OWL 2 Web Ontology Language Structural Specification

Defines the formal conceptual vocabulary, entity class typologies, predicate taxonomy,
and domain/range algebraic axioms for the YarnBall Financial Knowledge Graph.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Set, Tuple


class EntityClass(str, Enum):
    """Canonical Entity Classes (Concept Typology)."""
    COMPANY = "COMPANY"
    SUBSIDIARY = "SUBSIDIARY"
    PERSON = "PERSON"
    EXECUTIVE = "EXECUTIVE"
    REGULATORY_BODY = "REGULATORYBODY"
    COMMODITY = "COMMODITY"
    PRODUCT = "PRODUCT"
    GEOPOLITICAL_RISK = "GEOPOLITICALRISK"
    MARKET_EVENT = "MARKETEVENT"
    RISK = "RISK"
    FINANCIAL_INSTITUTION = "FINANCIALINSTITUTION"
    TECHNOLOGY = "TECHNOLOGY"
    SECTOR = "SECTOR"
    INDUSTRY = "INDUSTRY"
    METRIC = "METRIC"
    LOCATION = "LOCATION"
    NEWS = "NEWS"
    FORM = "FORM"
    FILING = "FILING"
    DOCUMENT = "DOCUMENT"


class RelationPredicate(str, Enum):
    """Canonical Relational Predicates (Object Properties)."""
    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    PARENT_OF = "PARENT_OF"
    SUPPLIES_TO = "SUPPLIES_TO"
    CUSTOMER_OF = "CUSTOMER_OF"
    SOLE_SOURCE_DEPENDENT_ON = "SOLE_SOURCE_DEPENDENT_ON"
    COMPETES_WITH = "COMPETES_WITH"
    PARTNERED_WITH = "PARTNERED_WITH"
    CEO_OF = "CEO_OF"
    EXECUTIVE_OF = "EXECUTIVE_OF"
    DIRECTOR_OF = "DIRECTOR_OF"
    INSIDER_OF = "INSIDER_OF"
    EXPOSED_TO_RISK = "EXPOSED_TO_RISK"
    LICENSES_FROM = "LICENSES_FROM"
    LICENSES_TO = "LICENSES_TO"
    ACQUIRED_BY = "ACQUIRED_BY"
    DEFAULTED_ON = "DEFAULTED_ON"
    SPUN_OFF_FROM = "SPUN_OFF_FROM"
    OWNS = "OWNS"
    INVESTS_IN = "INVESTS_IN"
    PRODUCES = "PRODUCES"
    IMPACTS = "IMPACTS"
    REPORTS = "REPORTS"
    FILED = "FILED"
    CO_INVESTS_WITH = "CO_INVESTS_WITH"
    PEER_OF = "PEER_OF"
    COLLABORATES_WITH = "COLLABORATES_WITH"


# Normalize UI/prompt PascalCase or underscored labels to TBox EntityClass tokens
LABEL_NORMALIZATION_MAP: Dict[str, str] = {
    "RISKFACTOR": "RISK",
    "RISK_FACTOR": "RISK",
    "GEOPOLITICAL_RISK": "GEOPOLITICALRISK",
    "REGULATORY_BODY": "REGULATORYBODY",
    "MARKET_EVENT": "MARKETEVENT",
    "FINANCIAL_INSTITUTION": "FINANCIALINSTITUTION",
    "FINANCIALMETRIC": "METRIC",
    "FINANCIAL_METRIC": "METRIC",
}

# Deterministic alias normalization: raw_predicate -> (canonical_predicate, swap_direction)
PREDICATE_ALIASES: Dict[str, Tuple[str, bool]] = {
    "ACQUIRED": ("ACQUIRED_BY", True),
    "HAS_SUBSIDIARY": ("SUBSIDIARY_OF", True),
    "SUPPLIED_BY": ("SUPPLIES_TO", True),
    "LEADS": ("EXECUTIVE_OF", False),
    "CONTROLS": ("OWNS", False),
    "MERGED_WITH": ("PARTNERED_WITH", False),
    "FILED_FORM": ("FILED", False),
    "EXPOSED_TO": ("EXPOSED_TO_RISK", False),
}


# ----------------------------------------------------------------------
# Ontological Domain & Range Schema Definition
# Format: rel_type -> (valid_source_types: Set[str], valid_target_types: Set[str], is_symmetric: bool)
# ----------------------------------------------------------------------
ONTOLOGY_SCHEMA: Dict[str, Tuple[Set[str], Set[str], bool]] = {
    "SUBSIDIARY_OF": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY"}, False),
    "PARENT_OF": ({"COMPANY"}, {"COMPANY", "SUBSIDIARY"}, False),
    "SUPPLIES_TO": ({"COMPANY", "SUBSIDIARY", "SUPPLIER"}, {"COMPANY", "CUSTOMER"}, False),
    "CUSTOMER_OF": ({"COMPANY", "CUSTOMER"}, {"COMPANY", "SUPPLIER"}, False),
    "SOLE_SOURCE_DEPENDENT_ON": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY", "SUBSIDIARY", "SUPPLIER"}, False),
    "COMPETES_WITH": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY", "SUBSIDIARY"}, True),
    "PARTNERED_WITH": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY", "SUBSIDIARY"}, True),
    "CEO_OF": ({"PERSON", "EXECUTIVE"}, {"COMPANY", "SUBSIDIARY"}, False),
    "EXECUTIVE_OF": ({"PERSON", "EXECUTIVE"}, {"COMPANY", "SUBSIDIARY"}, False),
    "DIRECTOR_OF": ({"PERSON", "EXECUTIVE"}, {"COMPANY", "SUBSIDIARY"}, False),
    "INSIDER_OF": ({"PERSON", "EXECUTIVE"}, {"COMPANY", "SUBSIDIARY"}, False),
    "EXPOSED_TO_RISK": (
        {"COMPANY", "SUBSIDIARY"},
        {"GEOPOLITICALRISK", "COMMODITY", "REGULATORYBODY", "MARKETEVENT", "RISK", "METRIC"},
        False,
    ),
    "LICENSES_FROM": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY", "SUBSIDIARY"}, False),
    "LICENSES_TO": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY", "SUBSIDIARY"}, False),
    "ACQUIRED_BY": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY"}, False),
    "DEFAULTED_ON": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY", "FINANCIALINSTITUTION"}, False),
    "SPUN_OFF_FROM": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY", "SUBSIDIARY"}, False),
    "OWNS": ({"COMPANY", "SUBSIDIARY", "FINANCIALINSTITUTION", "PERSON"}, {"COMPANY", "SUBSIDIARY"}, False),
    "INVESTS_IN": (
        {"COMPANY", "SUBSIDIARY", "FINANCIALINSTITUTION", "PERSON"},
        {"COMPANY", "SUBSIDIARY", "SECTOR", "INDUSTRY", "TECHNOLOGY", "PRODUCT"},
        False,
    ),
    "PRODUCES": ({"COMPANY", "SUBSIDIARY"}, {"PRODUCT", "TECHNOLOGY", "COMMODITY"}, False),
    "IMPACTS": (
        {"MARKETEVENT", "GEOPOLITICALRISK", "RISK", "REGULATORYBODY", "SECTOR", "COMPANY", "METRIC", "PRODUCT"},
        {"COMPANY", "SUBSIDIARY", "SECTOR", "PRODUCT", "METRIC"},
        False,
    ),
    "REPORTS": ({"COMPANY", "SUBSIDIARY"}, {"METRIC", "RISK", "FORM", "FILING"}, False),
    "FILED": ({"COMPANY", "SUBSIDIARY"}, {"FORM", "FILING", "DOCUMENT"}, False),
    "CO_INVESTS_WITH": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY", "SUBSIDIARY"}, True),
    "PEER_OF": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY", "SUBSIDIARY"}, True),
    "COLLABORATES_WITH": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY", "SUBSIDIARY"}, True),
}

# Symmetric relationship types that should be canonically ordered (R(a, b) == R(b, a))
SYMMETRIC_RELATIONSHIPS: Set[str] = {
    rel for rel, (_, _, is_sym) in ONTOLOGY_SCHEMA.items() if is_sym
}

# Standard allowed node labels for Cypher queries, prompts, and AST guards
DEFAULT_ALLOWED_LABELS: Set[str] = {
    "Company",
    "Person",
    "Product",
    "Technology",
    "Sector",
    "Metric",
    "Location",
    "Risk",
    "RegulatoryBody",
    "News",
}

EXTRACTION_PROMPT_REL_TYPES: Tuple[str, ...] = tuple(ONTOLOGY_SCHEMA.keys())


def format_extraction_prompt_vocab() -> Tuple[str, str]:
    """Return canonical comma-separated node labels and edge predicates for LLM prompts."""
    labels = ", ".join(sorted(DEFAULT_ALLOWED_LABELS - {"News"}))
    rels = ", ".join(sorted(EXTRACTION_PROMPT_REL_TYPES))
    return labels, rels
