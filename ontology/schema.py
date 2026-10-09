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
    METRIC = "METRIC"
    LOCATION = "LOCATION"
    NEWS = "NEWS"


class RelationPredicate(str, Enum):
    """Canonical Relational Predicates (Object Properties)."""
    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    PARENT_OF = "PARENT_OF"
    SUPPLIES_TO = "SUPPLIES_TO"
    CUSTOMER_OF = "CUSTOMER_OF"
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
    CO_INVESTS_WITH = "CO_INVESTS_WITH"
    PEER_OF = "PEER_OF"
    COLLABORATES_WITH = "COLLABORATES_WITH"


# ----------------------------------------------------------------------
# Ontological Domain & Range Schema Definition
# Format: rel_type -> (valid_source_types: Set[str], valid_target_types: Set[str], is_symmetric: bool)
# ----------------------------------------------------------------------
ONTOLOGY_SCHEMA: Dict[str, Tuple[Set[str], Set[str], bool]] = {
    "SUBSIDIARY_OF": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY"}, False),
    "PARENT_OF": ({"COMPANY"}, {"COMPANY", "SUBSIDIARY"}, False),
    "SUPPLIES_TO": ({"COMPANY", "SUBSIDIARY", "SUPPLIER"}, {"COMPANY", "CUSTOMER"}, False),
    "CUSTOMER_OF": ({"COMPANY", "CUSTOMER"}, {"COMPANY", "SUPPLIER"}, False),
    "COMPETES_WITH": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY", "SUBSIDIARY"}, True),
    "PARTNERED_WITH": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY", "SUBSIDIARY"}, True),
    "CEO_OF": ({"PERSON", "EXECUTIVE"}, {"COMPANY", "SUBSIDIARY"}, False),
    "EXECUTIVE_OF": ({"PERSON", "EXECUTIVE"}, {"COMPANY", "SUBSIDIARY"}, False),
    "DIRECTOR_OF": ({"PERSON", "EXECUTIVE"}, {"COMPANY", "SUBSIDIARY"}, False),
    "INSIDER_OF": ({"PERSON", "EXECUTIVE"}, {"COMPANY", "SUBSIDIARY"}, False),
    "EXPOSED_TO_RISK": ({"COMPANY", "SUBSIDIARY"}, {"GEOPOLITICALRISK", "COMMODITY", "REGULATORYBODY", "MARKETEVENT", "RISK"}, False),
    "LICENSES_FROM": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY", "SUBSIDIARY"}, False),
    "LICENSES_TO": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY", "SUBSIDIARY"}, False),
    "ACQUIRED_BY": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY"}, False),
    "DEFAULTED_ON": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY", "FINANCIALINSTITUTION"}, False),
    "CO_INVESTS_WITH": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY", "SUBSIDIARY"}, True),
    "PEER_OF": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY", "SUBSIDIARY"}, True),
    "COLLABORATES_WITH": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY", "SUBSIDIARY"}, True),
}

# Symmetric relationship types that should be canonically ordered (R(a, b) == R(b, a))
SYMMETRIC_RELATIONSHIPS: Set[str] = {
    rel for rel, (_, _, is_sym) in ONTOLOGY_SCHEMA.items() if is_sym
}

# Standard allowed node labels for Cypher queries and AST guards
DEFAULT_ALLOWED_LABELS: Set[str] = {
    "Company",
    "Person",
    "Product",
    "Technology",
    "Sector",
    "Metric",
    "Location",
    "News",
}
