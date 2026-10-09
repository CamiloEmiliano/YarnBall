"""
Corporate Earnings Call Transcripts Ingestion Subpackage.

Modules:
- `client`: `TranscriptSourceClient` (reads local archives & 8-K disclosures)
- `parser`: `EarningsTranscriptParser` (Prepared Remarks vs Q&A dialog segmenter)
- `harvester`: `EarningsTranscriptIngestor` (multi-quarter validation & Parquet stager)
"""

from .client import TranscriptSourceClient
from .parser import EarningsTranscriptParser
from .harvester import EarningsTranscriptIngestor, DEFAULT_TRANSCRIPT_DIR

__all__ = [
    "TranscriptSourceClient",
    "EarningsTranscriptParser",
    "EarningsTranscriptIngestor",
    "DEFAULT_TRANSCRIPT_DIR",
]
