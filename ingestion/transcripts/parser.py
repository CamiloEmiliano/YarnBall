"""
Earnings Call Transcript Parser.

Segments raw earnings transcripts into:
- Executive Prepared Remarks (formal strategic announcements, forward guidance)
- Analyst Q&A Sessions (dialogue turns, adversarial questioning)
- Executive vs Analyst speaker attribution and role classification
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ingestion.transcripts.parser")


class EarningsTranscriptParser:
    """Parses earnings call transcript text into prepared remarks and Q&A dialogues."""

    QA_SPLIT_PATTERN = re.compile(
        r"(?:question[s]?\s*(?:and|&)\s*answer[s]?\s*(?:session|period)?|q\s*&\s*a\s*session)",
        re.IGNORECASE,
    )

    SPEAKER_PATTERN = re.compile(
        r"(?:^|\n)([A-Z][a-zA-Z\.\s]+?)\s*(?:--|-|\—)\s*([A-Za-z\s,]+?)(?:\n|:)",
    )

    EXECUTIVE_ROLES = ["ceo", "cfo", "chief", "president", "management", "officer", "treasurer", "coo"]

    def parse_transcript_text(self, raw_text: str) -> Dict[str, Any]:
        """Segment transcript into prepared remarks and Q&A dialogues."""
        if not raw_text or not raw_text.strip():
            return {
                "prepared_remarks": "",
                "qa_dialogues": [],
                "executives": [],
                "analysts": [],
            }

        parts = self.QA_SPLIT_PATTERN.split(raw_text, maxsplit=1)
        prepared_remarks = parts[0].strip() if parts else raw_text.strip()
        qa_text = parts[1].strip() if len(parts) > 1 else ""

        qa_dialogues: List[Dict[str, str]] = []
        executives: List[str] = []
        analysts: List[str] = []

        if qa_text:
            turns = re.split(r"\n(?=[A-Z][a-zA-Z\.\s]+?\s*(?:--|-|\—))", qa_text)
            for turn in turns:
                match = self.SPEAKER_PATTERN.search(turn)
                if match:
                    speaker_name = match.group(1).strip().rstrip("-— ").strip()
                    speaker_role = match.group(2).strip().lstrip("-— ").strip()
                    speech_content = turn[match.end():].strip()

                    role_lower = speaker_role.lower()
                    if any(r in role_lower for r in self.EXECUTIVE_ROLES):
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
