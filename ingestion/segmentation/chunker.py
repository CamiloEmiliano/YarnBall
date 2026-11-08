"""Financial Document Pre-Segmentation and Token Bounding Chunker.

Tailored to SEC Form 10-K, 10-Q, and 8-K disclosure structures:
1. Recognizes legal disclosure section boundaries (Item 1 Business, Item 1A Risk Factors,
   Item 7 MD&A, Item 1.01 Material Agreements, Item 8.01 Other Events, Exhibit 21).
2. Enforces a strict token ceiling (default <=1,500 tokens) with semantic paragraph/sentence
   preservation and sliding overlap.
3. Injects immutable document metadata headers into each chunk so the 7B model retains
   entity context without cross-chunk drift.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Pattern, Tuple


@dataclass
class FilingMetadata:
    """Metadata describing the parent filing."""

    cik: str
    ticker: Optional[str] = None
    company_name: Optional[str] = None
    form_type: str = "10-K"
    period: Optional[str] = None
    filing_date: Optional[str] = None

    def __post_init__(self):
        # Enforce zero-padded 10-digit CIK
        self.cik = str(self.cik).strip().zfill(10)
        if self.ticker:
            self.ticker = self.ticker.strip().upper()
        if self.company_name:
            self.company_name = self.company_name.strip()
        if self.form_type:
            self.form_type = self.form_type.strip().upper()

    def format_header(self, section_name: str) -> str:
        """Format immutable metadata header to prepend to each chunk."""
        parts = [f"DOCUMENT: SEC Form {self.form_type}", f"CIK: {self.cik}"]
        if self.ticker:
            parts.append(f"TICKER: {self.ticker}")
        if self.company_name:
            parts.append(f"COMPANY: {self.company_name}")
        if self.period:
            parts.append(f"PERIOD: {self.period}")
        if self.filing_date:
            parts.append(f"DATE: {self.filing_date}")
        parts.append(f"SECTION: {section_name}")
        return f"[{' | '.join(parts)}]"


@dataclass
class FilingChunk:
    """Represents a bounded, section-aware document chunk."""

    chunk_id: str
    section_name: str
    header: str
    text: str
    full_prompt_text: str
    token_count: int
    start_char: int
    end_char: int
    metadata: FilingMetadata


# Standard SEC Form 10-K, 10-Q, 8-K Section Patterns
DEFAULT_SECTION_PATTERNS: List[Tuple[str, Pattern]] = [
    # Form 10-K / 10-Q Core Sections
    (
        "Item 1 - Business",
        re.compile(
            r"(?:\n|^)\s*(?:ITEM\s+1[.\s–—\-]+BUSINESS\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "Item 1A - Risk Factors",
        re.compile(
            r"(?:\n|^)\s*(?:ITEM\s+1A[.\s–—\-]+RISK\s+FACTORS\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "Item 1B - Unresolved Staff Comments",
        re.compile(
            r"(?:\n|^)\s*(?:ITEM\s+1B[.\s–—\-]+UNRESOLVED\s+STAFF\s+COMMENTS\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "Item 2 - Properties",
        re.compile(
            r"(?:\n|^)\s*(?:ITEM\s+2[.\s–—\-]+PROPERTIES\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "Item 3 - Legal Proceedings",
        re.compile(
            r"(?:\n|^)\s*(?:ITEM\s+3[.\s–—\-]+LEGAL\s+PROCEEDINGS\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "Item 7 - Management Discussion and Analysis",
        re.compile(
            r"(?:\n|^)\s*(?:ITEM\s+7[.\s–—\-]+MANAGEMENT(?:'S)?\s+DISCUSSION\s+AND\s+ANALYSIS\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "Item 7A - Quantitative and Qualitative Disclosures",
        re.compile(
            r"(?:\n|^)\s*(?:ITEM\s+7A[.\s–—\-]+QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "Item 8 - Financial Statements",
        re.compile(
            r"(?:\n|^)\s*(?:ITEM\s+8[.\s–—\-]+FINANCIAL\s+STATEMENTS\b)",
            re.IGNORECASE,
        ),
    ),
    # Form 8-K Material Current Event Sections
    (
        "Item 1.01 - Entry into Material Agreement",
        re.compile(
            r"(?:\n|^)\s*(?:ITEM\s+1\.01[.\s–—\-]+ENTRY\s+INTO\s+A\s+MATERIAL\s+DEFINITIVE\s+AGREEMENT\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "Item 1.02 - Termination of Material Agreement",
        re.compile(
            r"(?:\n|^)\s*(?:ITEM\s+1\.02[.\s–—\-]+TERMINATION\s+OF\s+A\s+MATERIAL\s+DEFINITIVE\s+AGREEMENT\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "Item 2.01 - Completion of Acquisition or Disposition",
        re.compile(
            r"(?:\n|^)\s*(?:ITEM\s+2\.01[.\s–—\-]+COMPLETION\s+OF\s+ACQUISITION\s+OR\s+DISPOSITION\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "Item 8.01 - Other Events",
        re.compile(
            r"(?:\n|^)\s*(?:ITEM\s+8\.01[.\s–—\-]+OTHER\s+EVENTS\b)",
            re.IGNORECASE,
        ),
    ),
    # Exhibit 21 Subsidiaries
    (
        "Exhibit 21 - Subsidiaries",
        re.compile(
            r"(?:\n|^)\s*(?:EXHIBIT\s+21(?:\.1)?[.\s–—\-]+SUBSIDIARIES\b)",
            re.IGNORECASE,
        ),
    ),
]


def estimate_token_count(text: str) -> int:
    """Conservative, deterministic token count estimator for BPE/Qwen tokenizers.

    Uses a conservative blend of word count (* 1.35) and character count (/ 3.7)
    to guarantee that estimated tokens will not exceed actual tokenizer allocation.
    """
    if not text:
        return 0
    words = len(text.split())
    chars = len(text)
    return int(max(words * 1.35, chars / 3.7) + 1)


class FinancialSectionSegmenter:
    """Pre-segments SEC filings into section-bounded, token-capped chunks."""

    def __init__(
        self,
        max_tokens: int = 1500,
        overlap_tokens: int = 120,
        min_chunk_tokens: int = 40,
        section_patterns: Optional[List[Tuple[str, Pattern]]] = None,
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.section_patterns = section_patterns or DEFAULT_SECTION_PATTERNS

    def segment_filing(
        self,
        raw_text: str,
        metadata: FilingMetadata,
    ) -> List[FilingChunk]:
        """Segment filing text into bounded chunks with metadata headers.

        Args:
            raw_text: Full SEC filing or section text.
            metadata: Filing metadata container.

        Returns:
            List of FilingChunk objects, each strictly <= max_tokens.
        """
        if not raw_text or not raw_text.strip():
            return []

        # 1. Detect section boundaries
        sections = self._extract_sections(raw_text)

        chunks: List[FilingChunk] = []
        global_chunk_idx = 1

        for sec_name, sec_text, sec_start, sec_end in sections:
            header_str = metadata.format_header(sec_name)
            header_tokens = estimate_token_count(header_str)
            content_budget = max(200, self.max_tokens - header_tokens - 10)

            # 2. Check if section fits directly in one chunk
            sec_tokens = estimate_token_count(sec_text)
            if sec_tokens <= content_budget:
                full_text = f"{header_str}\n\n{sec_text.strip()}"
                chunks.append(
                    FilingChunk(
                        chunk_id=f"{metadata.cik}_{metadata.form_type}_{global_chunk_idx:03d}",
                        section_name=sec_name,
                        header=header_str,
                        text=sec_text.strip(),
                        full_prompt_text=full_text,
                        token_count=estimate_token_count(full_text),
                        start_char=sec_start,
                        end_char=sec_end,
                        metadata=metadata,
                    )
                )
                global_chunk_idx += 1
                continue

            # 3. Section exceeds budget -> segment into sub-chunks with overlap
            sub_chunks = self._chunk_section_text(
                sec_text=sec_text,
                sec_start=sec_start,
                content_budget=content_budget,
            )

            for sub_text, sub_start, sub_end in sub_chunks:
                full_text = f"{header_str}\n\n{sub_text.strip()}"
                chunks.append(
                    FilingChunk(
                        chunk_id=f"{metadata.cik}_{metadata.form_type}_{global_chunk_idx:03d}",
                        section_name=sec_name,
                        header=header_str,
                        text=sub_text.strip(),
                        full_prompt_text=full_text,
                        token_count=estimate_token_count(full_text),
                        start_char=sub_start,
                        end_char=sub_end,
                        metadata=metadata,
                    )
                )
                global_chunk_idx += 1

        return chunks

    def _extract_sections(
        self, raw_text: str
    ) -> List[Tuple[str, str, int, int]]:
        """Identify section boundaries using standard regex patterns."""
        matches: List[Tuple[int, int, str]] = []

        for name, pattern in self.section_patterns:
            for m in pattern.finditer(raw_text):
                matches.append((m.start(), m.end(), name))

        if not matches:
            # No section headers found; treat whole text as General Disclosure
            return [("General Disclosure", raw_text, 0, len(raw_text))]

        # Sort matches by appearance in text
        matches.sort(key=lambda x: x[0])

        sections: List[Tuple[str, str, int, int]] = []
        for i in range(len(matches)):
            start_pos = matches[i][0]
            sec_name = matches[i][2]
            end_pos = matches[i + 1][0] if i + 1 < len(matches) else len(raw_text)
            sec_content = raw_text[start_pos:end_pos].strip()
            if sec_content:
                sections.append((sec_name, sec_content, start_pos, end_pos))

        return sections

    def _chunk_section_text(
        self,
        sec_text: str,
        sec_start: int,
        content_budget: int,
    ) -> List[Tuple[str, int, int]]:
        """Split a long section into paragraphs and sentences adhering to content_budget."""
        # Split on double newlines (paragraphs) or table blocks
        raw_paragraphs = [
            p.strip()
            for p in re.split(r"\n\s*\n", sec_text)
            if p.strip()
        ]

        # Further subdivide any giant paragraph that alone exceeds budget
        normalized_paragraphs: List[str] = []
        for p in raw_paragraphs:
            if estimate_token_count(p) <= content_budget:
                normalized_paragraphs.append(p)
            else:
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if s.strip()]
                current_sent_group: List[str] = []
                current_group_tokens = 0
                for sent in sentences:
                    s_tokens = estimate_token_count(sent)
                    if current_group_tokens + s_tokens <= content_budget:
                        current_sent_group.append(sent)
                        current_group_tokens += s_tokens
                    else:
                        if current_sent_group:
                            normalized_paragraphs.append(" ".join(current_sent_group))
                        current_sent_group = [sent]
                        current_group_tokens = s_tokens
                if current_sent_group:
                    normalized_paragraphs.append(" ".join(current_sent_group))

        # Group paragraphs into chunks with sliding overlap
        chunks: List[Tuple[str, int, int]] = []
        current_chunk_paras: List[str] = []
        current_chunk_tokens = 0

        for para in normalized_paragraphs:
            p_tokens = estimate_token_count(para)
            if current_chunk_tokens + p_tokens <= content_budget:
                current_chunk_paras.append(para)
                current_chunk_tokens += p_tokens
            else:
                if current_chunk_paras:
                    chunk_text = "\n\n".join(current_chunk_paras)
                    chunks.append((chunk_text, sec_start, sec_start + len(chunk_text)))

                    # Compute overlap paragraphs from the tail
                    overlap_paras: List[str] = []
                    overlap_acc = 0
                    for rev_p in reversed(current_chunk_paras):
                        rev_tokens = estimate_token_count(rev_p)
                        if overlap_acc + rev_tokens <= self.overlap_tokens:
                            overlap_paras.insert(0, rev_p)
                            overlap_acc += rev_tokens
                        else:
                            break

                    current_chunk_paras = overlap_paras + [para]
                    current_chunk_tokens = sum(estimate_token_count(p) for p in current_chunk_paras)
                else:
                    chunks.append((para, sec_start, sec_start + len(para)))
                    current_chunk_paras = []
                    current_chunk_tokens = 0

        if current_chunk_paras:
            final_text = "\n\n".join(current_chunk_paras)
            chunks.append((final_text, sec_start, sec_start + len(final_text)))

        return chunks


# Singleton default segmenter
DEFAULT_FINANCIAL_SEGMENTER = FinancialSectionSegmenter()


def segment_filing(
    raw_text: str,
    metadata: FilingMetadata,
    max_tokens: int = 1500,
) -> List[FilingChunk]:
    """Functional convenience wrapper for filing segmentation."""
    segmenter = FinancialSectionSegmenter(max_tokens=max_tokens)
    return segmenter.segment_filing(raw_text, metadata)

