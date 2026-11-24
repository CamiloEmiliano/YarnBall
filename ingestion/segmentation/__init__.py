"""Financial document pre-segmentation and token-bounding package."""

from .chunker import (
    DEFAULT_SECTION_PATTERNS,
    FilingChunk,
    FilingMetadata,
    FinancialSectionSegmenter,
    estimate_token_count,
    segment_filing,
    segment_news_article,
)

__all__ = [
    "DEFAULT_SECTION_PATTERNS",
    "FilingChunk",
    "FilingMetadata",
    "FinancialSectionSegmenter",
    "estimate_token_count",
    "segment_filing",
    "segment_news_article",
]

