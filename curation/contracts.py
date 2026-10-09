"""
Data Contracts and Schema Definitions for Dataset Cartography and Active Learning.

Based on:
- Swayamdipta et al. (EMNLP 2020): "Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics"
- Toneva et al. (ICLR 2019): "An Empirical Study of Example Forgetting during Deep Neural Network Learning"
"""

from __future__ import annotations

from enum import Enum
from typing import List
from pydantic import BaseModel, Field


class CartographyRegion(str, Enum):
    """The three canonical regions defined in Swayamdipta et al. (2020)."""
    EASY_TO_LEARN = "easy_to_learn"       # High confidence, low variability (repetitive boilerplates)
    AMBIGUOUS = "ambiguous"               # High variability (decision boundary; generalization driver)
    HARD_TO_LEARN = "hard_to_learn"       # Low confidence, low variability (annotation noise or syntax error)


class CartographyCoordinate(BaseModel):
    """Data map coordinate for a single training instance."""
    guid: str = Field(..., description="Unique sample identifier (e.g., sample_id or GUID)")
    confidence: float = Field(..., description="Mean gold target probability across epochs (mu)")
    variability: float = Field(..., description="Standard deviation of target probability across epochs (sigma)")
    correctness: float = Field(..., description="Fraction of epochs target met syntax/confidence threshold")
    forgetfulness: int = Field(..., description="Number of times learned then forgotten (Toneva et al. 2019)")
    region: CartographyRegion = Field(..., description="Data map region assignment")


class ActiveLearningPartition(BaseModel):
    """The curated partition produced by the cartographic selection policy."""
    retained_guids: List[str] = Field(..., description="GUIDs retained for next-cycle training (ambiguous + easy subset)")
    pruned_easy_guids: List[str] = Field(..., description="GUIDs pruned from easy-to-learn cohort (redundant boilerplates)")
    quarantined_hard_guids: List[str] = Field(..., description="GUIDs quarantined for audit (label noise / syntax errors)")
    total_candidates: int = Field(..., description="Total candidate samples evaluated in the cartography map")
    retention_rate_pct: float = Field(..., description="Percentage of candidate samples retained for training")
