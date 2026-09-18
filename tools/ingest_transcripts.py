"""
Quarterly Earnings Call Transcript & Form 8-K Item 2.02 Ingestor.

Processes earnings call transcripts and Form 8-K Item 2.02 releases, segmenting
executive prepared remarks (strategic announcements) from analyst Q&A sessions
to supply high-fidelity conversational context for GraphRAG and multi-hop SFT.
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

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from tools.sp500_universe import SP500UniverseManager

logger = logging.getLogger("ingest_transcripts")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

DEFAULT_TRANSCRIPT_DIR = Path(__file__).resolve().parent.parent / "data" / "transcripts"


class EarningsTranscriptIngestor:
    """Ingests and parses earnings call transcripts and Item 2.02 disclosures."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        universe_mgr: Optional[SP500UniverseManager] = None,
    ):
        self.output_dir = output_dir or DEFAULT_TRANSCRIPT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.universe_mgr = universe_mgr or SP500UniverseManager()

    def parse_transcript_text(self, raw_text: str) -> Dict[str, Any]:
        """
        Segment transcript into prepared remarks and Q&A dialogues.
        Identifies executive speakers, analysts, and dialogue turns.
        """
        if not raw_text or not raw_text.strip():
            return {"prepared_remarks": "", "qa_dialogues": [], "executives": [], "analysts": []}

        # Patterns for section demarcation
        qa_split_pattern = re.compile(
            r"(?:question[s]?\s*(?:and|&)\s*answer[s]?\s*(?:session|period)?|q\s*&\s*a\s*session)",
            re.IGNORECASE,
        )

        parts = qa_split_pattern.split(raw_text, maxsplit=1)
        prepared_remarks = parts[0].strip() if parts else raw_text.strip()
        qa_text = parts[1].strip() if len(parts) > 1 else ""

        # Parse Q&A dialogues into speaker turns
        qa_dialogues: List[Dict[str, str]] = []
        executives: List[str] = []
        analysts: List[str] = []

        if qa_text:
            speaker_pattern = re.compile(
                r"(?:^|\n)([A-Z][a-zA-Z\.\s]+?)\s*(?:--|-|\—)\s*([A-Za-z\s,]+?)(?:\n|:)",
            )
            turns = re.split(r"\n(?=[A-Z][a-zA-Z\.\s]+?\s*(?:--|-|\—))", qa_text)
            for turn in turns:
                match = speaker_pattern.search(turn)
                if match:
                    speaker_name = match.group(1).strip().rstrip("-— ").strip()
                    speaker_role = match.group(2).strip().lstrip("-— ").strip()
                    speech_content = turn[match.end():].strip()

                    role_lower = speaker_role.lower()
                    if any(r in role_lower for r in ["ceo", "cfo", "chief", "president", "management", "officer"]):
                        if speaker_name not in executives:
                            executives.append(speaker_name)
                    else:
                        if speaker_name not in analysts:
                            analysts.append(speaker_name)

                    qa_dialogues.append({
                        "speaker": speaker_name,
                        "role": speaker_role,
                        "text": speech_content,
                    })

        return {
            "prepared_remarks": prepared_remarks,
            "qa_dialogues": qa_dialogues,
            "executives": executives,
            "analysts": analysts,
        }

    def process_transcript_record(
        self,
        ticker: str,
        fiscal_year: int,
        fiscal_quarter: str,
        date_str: str,
        raw_text: str,
        source_url: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Process and validate an earnings transcript record."""
        clean_ticker = ticker.strip().upper()
        clean_quarter = fiscal_quarter.strip().upper() # e.g. "Q1", "Q2", "Q3", "Q4"

        # Validate S&P 500 inclusion at date
        if not self.universe_mgr.is_constituent(clean_ticker, target_date=date_str):
            logger.debug(f"Skipping transcript: {clean_ticker} was not in S&P 500 on {date_str}")
            return None

        parsed = self.parse_transcript_text(raw_text)
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

    # Demonstration / synthetic seeding if no external file provided
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
