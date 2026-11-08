"""Financial document pre-segmentation and token-bounding package."""

from .chunker import (
    FinancialSectionSegmenter,
    FilingMetadata,
    FilingChunk,
    DEFAULT_SECTION_PATTERNS,
)

__all__ = [
    "FinancialSectionSegmenter",
    "FilingMetadata",
    "FilingChunk",
    "DEFAULT_SECTION_PATTERNS",
]

