"""
Corporate Earnings Call Transcript Source Client.

Provides utilities for reading and discovering transcript archives from:
- Local JSONL files
- Text / Markdown file directories
- SEC Form 8-K Item 2.02 (Results of Operations and Financial Condition) exhibits
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger("ingestion.transcripts.client")


class TranscriptSourceClient:
    """Discovers and streams raw transcript documents from local directories or JSONL files."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/transcripts")

    def stream_from_jsonl(self, jsonl_path: Path) -> Iterator[Dict[str, Any]]:
        """Stream raw transcript records from a JSONL file."""
        p = Path(jsonl_path)
        if not p.is_file():
            logger.warning(f"Transcript JSONL file not found: {p}")
            return

        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    yield json.loads(line_str)
                except Exception as e:
                    logger.debug(f"JSON decode error in {p}: {e}")

    def discover_local_transcripts(self, directory: Optional[Path] = None) -> List[Path]:
        """Scan directory for transcript text, markdown, or JSON files."""
        target = Path(directory or self.data_dir)
        if not target.is_dir():
            return []
        found = []
        for ext in ("*.txt", "*.md", "*.json", "*.jsonl"):
            found.extend(target.rglob(ext))
        return sorted(found)
