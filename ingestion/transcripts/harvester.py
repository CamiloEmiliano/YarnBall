"""
Earnings Call Transcript Harvester & Parquet Stager.

Processes earnings call transcripts and Form 8-K Item 2.02 disclosures, validates
against S&P 500 point-in-time constituent registry, and stages structured datasets into Parquet.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from tools.sp500_universe import SP500UniverseManager
from .parser import EarningsTranscriptParser
from .client import TranscriptSourceClient

logger = logging.getLogger("ingestion.transcripts.harvester")

DEFAULT_TRANSCRIPT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "transcripts"


class EarningsTranscriptIngestor:
    """Ingests, validates, and stages earnings call transcripts."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        universe_mgr: Optional[SP500UniverseManager] = None,
        parser: Optional[EarningsTranscriptParser] = None,
    ):
        self.output_dir = Path(output_dir or DEFAULT_TRANSCRIPT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.universe_mgr = universe_mgr or SP500UniverseManager()
        self.parser = parser or EarningsTranscriptParser()
        self.client = TranscriptSourceClient(data_dir=self.output_dir)

    def parse_transcript_text(self, raw_text: str) -> Dict[str, Any]:
        """Delegate to parser for backward compatibility."""
        return self.parser.parse_transcript_text(raw_text)

    def process_transcript_record(
        self,
        ticker: str,
        fiscal_year: int,
        fiscal_quarter: str,
        date_str: str,
        raw_text: str,
        source_url: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Process and validate an earnings transcript record against S&P 500 universe."""
        clean_ticker = ticker.strip().upper()
        clean_quarter = fiscal_quarter.strip().upper()

        if not self.universe_mgr.is_constituent(clean_ticker, target_date=date_str):
            logger.debug(f"Skipping transcript: {clean_ticker} was not in S&P 500 on {date_str}")
            return None

        parsed = self.parser.parse_transcript_text(raw_text)
        transcript_id = f"{clean_ticker}_{fiscal_year}_{clean_quarter}"
        source_hash = hashlib.sha256(f"{transcript_id}_{date_str}".encode("utf-8")).hexdigest()[:32]

        return {
            "transcript_id": transcript_id,
            "source_hash": source_hash,
            "ticker": clean_ticker,
            "fiscal_year": fiscal_year,
            "fiscal_quarter": clean_quarter,
            "event_date": date_str,
            "prepared_remarks": parsed["prepared_remarks"],
            "qa_dialogues_json": json.dumps(parsed["qa_dialogues"]),
            "executives": parsed["executives"],
            "analysts": parsed["analysts"],
            "source_url": source_url or f"urn:transcript:{transcript_id}",
        }

    def stage_transcripts_to_parquet(
        self,
        transcripts: List[Dict[str, Any]],
        filename: str = "sp500_transcripts.parquet",
    ) -> Path:
        """Stage processed transcript records to Parquet."""
        schema = pa.schema([
            ("transcript_id", pa.string()),
            ("source_hash", pa.string()),
            ("ticker", pa.string()),
            ("fiscal_year", pa.int32()),
            ("fiscal_quarter", pa.string()),
            ("event_date", pa.string()),
            ("prepared_remarks", pa.string()),
            ("qa_dialogues_json", pa.string()),
            ("executives", pa.list_(pa.string())),
            ("analysts", pa.list_(pa.string())),
            ("source_url", pa.string()),
        ])

        table = pa.Table.from_pylist(transcripts, schema=schema)
        out_path = self.output_dir / filename
        pq.write_table(table, out_path, compression="snappy")
        logger.info(f"Staged {len(transcripts)} transcripts to {out_path}")
        return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest earnings call transcripts.")
    parser.add_argument("--jsonl-path", type=str, default=None, help="Input raw transcripts JSONL")
    parser.add_argument("--output-file", type=str, default="transcripts_sp500.parquet", help="Output filename")
    args = parser.parse_args()

    ingestor = EarningsTranscriptIngestor()

    sample_records = [
        {
            "ticker": "AAPL",
            "fiscal_year": 2024,
            "fiscal_quarter": "Q3",
            "date_str": "2024-08-01",
            "raw_text": (
                "Good afternoon, everyone. Welcome to Apple's Q3 2024 earnings call. "
                "Today we are pleased to report revenue of $85.8 billion, up 5% year over year. "
                "We are seeing incredible customer enthusiasm for Apple Intelligence across our product line. "
                "TSMC remains our primary foundry partner for our 3nm M4 silicon. "
                "\n\nQuestion & Answer Session\n"
                "Toni Sacconaghi -- Bernstein: Could you elaborate on your supply chain ramp for the new chips?\n"
                "Tim Cook -- Chief Executive Officer: Thank you, Toni. Our manufacturing partners in Asia, including TSMC and Foxconn, are executing at peak efficiency."
            ),
        }
    ]

    processed = []
    for s in sample_records:
        rec = ingestor.process_transcript_record(
            ticker=s["ticker"],
            fiscal_year=s["fiscal_year"],
            fiscal_quarter=s["fiscal_quarter"],
            date_str=s["date_str"],
            raw_text=s["raw_text"],
        )
        if rec:
            processed.append(rec)

    if processed:
        p_path = ingestor.stage_transcripts_to_parquet(processed, filename=args.output_file)
        logger.info(f"Saved {len(processed)} transcripts to {p_path}")


if __name__ == "__main__":
    main()
