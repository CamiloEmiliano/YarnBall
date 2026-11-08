"""Unit tests for Stage 6.2: FinancialSectionSegmenter."""

import pytest
from ingestion.segmentation.chunker import (
    FinancialSectionSegmenter,
    FilingMetadata,
    FilingChunk,
    estimate_token_count,
    segment_filing,
)


@pytest.fixture
def metadata() -> FilingMetadata:
    return FilingMetadata(
        cik="1045810",  # Will be zero-padded to 0001045810
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        form_type="10-K",
        period="FY 2024",
        filing_date="2024-02-21",
    )


@pytest.fixture
def segmenter() -> FinancialSectionSegmenter:
    return FinancialSectionSegmenter(max_tokens=1500, overlap_tokens=120)


def test_filing_metadata_formatting(metadata: FilingMetadata):
    assert metadata.cik == "0001045810"
    header = metadata.format_header("Item 1A - Risk Factors")
    assert "DOCUMENT: SEC Form 10-K" in header
    assert "CIK: 0001045810" in header
    assert "TICKER: NVDA" in header
    assert "COMPANY: NVIDIA Corporation" in header
    assert "SECTION: Item 1A - Risk Factors" in header


def test_section_boundary_detection(segmenter: FinancialSectionSegmenter, metadata: FilingMetadata):
    text = (
        "ITEM 1. BUSINESS\n"
        "NVIDIA pioneered accelerated computing to tackle challenges ordinary computers cannot.\n\n"
        "ITEM 1A. RISK FACTORS\n"
        "Failure to meet customer delivery schedules could adversely affect our business and margins.\n\n"
        "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS\n"
        "Revenue for fiscal year 2024 increased significantly driven by Compute & Networking growth."
    )
    chunks = segmenter.segment_filing(text, metadata)
    assert len(chunks) == 3
    assert chunks[0].section_name == "Item 1 - Business"
    assert chunks[1].section_name == "Item 1A - Risk Factors"
    assert chunks[2].section_name == "Item 7 - Management Discussion and Analysis"

    for c in chunks:
        assert c.token_count <= 1500
        assert c.header in c.full_prompt_text
        assert metadata.cik == c.metadata.cik


def test_hard_token_ceiling_enforced(segmenter: FinancialSectionSegmenter, metadata: FilingMetadata):
    # Construct a long 6,000-token SEC section with 30 paragraphs
    paragraphs = [
        f"Paragraph {i}: The company relies on key wafer foundries including Taiwan Semiconductor Manufacturing Company (TSMC) "
        f"for the fabrication of all leading-edge high-bandwidth processors. Any disruption at TSMC cleanroom facilities in Hsinchu "
        f"or Tainan, whether resulting from seismic events, typhoons, power outages, or supply-chain allocation freezes, would severely "
        f"curtail product deliveries and reduce operating margins for multiple quarters. Secondary suppliers cannot be qualified "
        f"under nine months due to proprietary extreme ultraviolet lithography (EUV) packaging constraints."
        for i in range(40)
    ]
    long_section_text = "ITEM 1A. RISK FACTORS\n\n" + "\n\n".join(paragraphs)

    chunks = segmenter.segment_filing(long_section_text, metadata)
    assert len(chunks) > 1, "Long section must be partitioned into multiple chunks"

    for i, c in enumerate(chunks):
        assert c.token_count <= 1500, f"Chunk {i} exceeded 1500 tokens: {c.token_count}"
        assert c.section_name == "Item 1A - Risk Factors"
        assert c.header in c.full_prompt_text


def test_sliding_overlap_preserves_context(metadata: FilingMetadata):
    segmenter_small = FinancialSectionSegmenter(max_tokens=250, overlap_tokens=60)
    paras = [
        f"Paragraph {i}: Component supplier Alpha supplies critical substrate X to Beta Corporation. "
        f"Supply disruptions at Alpha affect downstream assembly at Gamma Inc."
        for i in range(12)
    ]
    text = "ITEM 1 - BUSINESS\n\n" + "\n\n".join(paras)

    chunks = segmenter_small.segment_filing(text, metadata)
    assert len(chunks) >= 2

    # Check that chunk 1 has content from chunk 0 due to sliding overlap
    chunk0_paras = set(chunks[0].text.split("\n\n"))
    chunk1_paras = set(chunks[1].text.split("\n\n"))
    intersection = chunk0_paras.intersection(chunk1_paras)
    assert len(intersection) >= 1, "Consecutive chunks must share overlapping paragraphs"


def test_giant_single_paragraph_sentence_subdivision(metadata: FilingMetadata):
    # A single continuous 2,000-token paragraph without any double newlines
    sentences = [
        f"Sentence {i}: Supplier Omega provides specialized connectors to focal manufacturer Delta under contract agreement Omega-Delta-2024."
        for i in range(50)
    ]
    single_giant_para = "ITEM 1 - BUSINESS\n" + " ".join(sentences)

    segmenter = FinancialSectionSegmenter(max_tokens=400, overlap_tokens=50)
    chunks = segmenter.segment_filing(single_giant_para, metadata)
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_count <= 400


def test_form_8k_current_events_segmentation():
    meta_8k = FilingMetadata(
        cik="0000320193",
        ticker="AAPL",
        company_name="Apple Inc.",
        form_type="8-K",
        filing_date="2024-05-02",
    )
    text = (
        "ITEM 1.01 ENTRY INTO A MATERIAL DEFINITIVE AGREEMENT\n"
        "Apple Inc. entered into an amended chip fabrication agreement with TSMC extending through 2028.\n\n"
        "ITEM 8.01 OTHER EVENTS\n"
        "The board of directors declared a cash dividend of $0.25 per share."
    )
    chunks = segment_filing(text, meta_8k, max_tokens=1500)
    assert len(chunks) == 2
    assert chunks[0].section_name == "Item 1.01 - Entry into Material Agreement"
    assert chunks[1].section_name == "Item 8.01 - Other Events"
    assert "DOCUMENT: SEC Form 8-K" in chunks[0].header


def test_empty_and_whitespace_input(segmenter: FinancialSectionSegmenter, metadata: FilingMetadata):
    assert segmenter.segment_filing("", metadata) == []
    assert segmenter.segment_filing("   \n\n  ", metadata) == []

