"""Pydantic validation schemas for Linguistic CheckList fixtures.

Academic Foundation:
- Ribeiro, M. T., Wu, T., Guestrin, C., & Singh, S. (ACL 2020):
  "Beyond Accuracy: Behavioral Testing of NLP Models with CheckList"
"""

from __future__ import annotations

from enum import Enum
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class CapabilityEnum(str, Enum):
    """The 10 Ribeiro et al. Linguistic Capabilities adapted for Financial Knowledge Graphs."""
    SEMANTIC_ROLE_LABELING = "SEMANTIC_ROLE_LABELING"
    NEGATION_POLARITY = "NEGATION_POLARITY"
    TEMPORAL_ASPECT = "TEMPORAL_ASPECT"
    LOGIC_CONDITIONALS = "LOGIC_CONDITIONALS"
    NAMED_ENTITY_RECOGNITION = "NAMED_ENTITY_RECOGNITION"
    TAXONOMY_HIERARCHY = "TAXONOMY_HIERARCHY"
    COREFERENCE_ANAPHORA = "COREFERENCE_ANAPHORA"
    ROBUSTNESS_INVARIANCE = "ROBUSTNESS_INVARIANCE"
    VOCABULARY_MATERIALITY = "VOCABULARY_MATERIALITY"
    COORDINATION_SCOPE = "COORDINATION_SCOPE"


class BehavioralTestType(str, Enum):
    """Ribeiro et al. Behavioral Test Types."""
    MFT = "MFT"  # Minimum Functionality Test
    INV = "INV"  # Invariance Test (f(x) == f(x'))
    DIR = "DIR"  # Directional Expectation Test (f(x) -> shifted f(x'))

TestTypeEnum = BehavioralTestType
setattr(BehavioralTestType, "__test__", False)  # Prevent pytest from treating Enum as a test suite class


class ActionEnum(str, Enum):
    """Expected action on knowledge graph."""
    EMIT = "EMIT"          # Emit edge / entity
    REFUSE = "REFUSE"      # Suppress / filter out to (none)
    TERMINATE = "TERMINATE" # Mark edge status as TERMINATED, is_active=False
    ESCALATE = "ESCALATE"  # Escalate materiality tier to CRITICAL_TIER_1


class ExpectedEntity(BaseModel):
    """Expected entity attributes in extracted subgraph."""
    name: str
    ticker: Optional[str] = None
    cik: Optional[str] = None
    entity_class: str = "COMPANY"
    listing_exchange: Optional[str] = None

    @field_validator("cik")
    @classmethod
    def validate_cik(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            clean = v.strip()
            if not (clean.isdigit() and len(clean) == 10):
                raise ValueError(f"CIK must be a 10-digit zero-padded string, got '{v}'")
            return clean
        return None

    @field_validator("listing_exchange")
    @classmethod
    def validate_exchange(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            clean = v.strip().upper()
            valid = {"NASDAQ", "NYSE", "AMEX", "CBOE", "OTC", "UNKNOWN"}
            if clean not in valid:
                raise ValueError(f"Invalid listing exchange '{v}'; must be in {valid}")
            return clean
        return None


class ExpectedEdge(BaseModel):
    """Expected relational edge attributes."""
    source_name: str
    target_name: str
    rel_type: str
    is_active: bool = True
    status: Optional[str] = None
    materiality: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("rel_type")
    @classmethod
    def validate_rel_type(cls, v: str) -> str:
        clean = v.strip().upper()
        # Must match known canonical predicates
        allowed = {
            "SUPPLIES_TO",
            "CUSTOMER_OF",
            "PARTNERED_WITH",
            "SUBSIDIARY_OF",
            "PARENT_OF",
            "COMPETES_WITH",
            "LICENSES_FROM",
            "LICENSES_TO",
            "ACQUIRED_BY",
            "DEFAULTED_ON",
            "CO_INVESTS_WITH",
            "PEER_OF",
            "COLLABORATES_WITH",
            "EXPOSED_TO_RISK",
        }
        if clean not in allowed:
            raise ValueError(f"Relational predicate '{v}' not recognized in canonical ontology {allowed}")
        return clean


class LinguisticTestCase(BaseModel):
    """Schema for a single Ribeiro behavioral test case."""
    id: str
    capability: CapabilityEnum
    test_type: TestTypeEnum
    description: str
    input_text: str
    perturbed_text: Optional[str] = None
    expected_entities: List[ExpectedEntity] = Field(default_factory=list)
    expected_edges: List[ExpectedEdge] = Field(default_factory=list)
    expected_action: ActionEnum = ActionEnum.EMIT
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("perturbed_text")
    @classmethod
    def validate_perturbation(cls, v: Optional[str], info: Any) -> Optional[str]:
        test_type = info.data.get("test_type")
        if test_type in (TestTypeEnum.INV, TestTypeEnum.DIR):
            if not v or not v.strip():
                raise ValueError(f"Test type {test_type} requires a non-empty 'perturbed_text'")
        return v
