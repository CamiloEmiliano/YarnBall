# tests/test_taxonomy_annotator.py
# -*- coding: utf-8 -*-
"""Unit tests for tools/sft_taxonomy_annotator.py."""

from pathlib import Path
import pytest
from unittest.mock import MagicMock

from tools.sft_taxonomy_annotator import FinancialTaxonomyAnnotator, AnnotatedTriple
from tools.sp500_universe import SP500Constituent


@pytest.fixture
def mock_universe_mgr():
    mgr = MagicMock()
    mock_constituents = [
        SP500Constituent(ticker="AAPL", cik="0000320193", company_name="Apple Inc.", gics_sector="Information Technology"),
        SP500Constituent(ticker="NVDA", cik="0001045810", company_name="NVIDIA Corporation", gics_sector="Information Technology"),
        SP500Constituent(ticker="MSFT", cik="0000789019", company_name="Microsoft Corp", gics_sector="Information Technology"),
    ]
    mgr.get_all_records.return_value = mock_constituents
    return mgr


def test_entity_typology_inference_and_cik_linking(mock_universe_mgr):
    annotator = FinancialTaxonomyAnnotator(universe_mgr=mock_universe_mgr)

    # 1. Ticker resolution
    e_type, ticker, cik = annotator.infer_entity_typology("AAPL")
    assert e_type == "Company"
    assert ticker == "AAPL"
    assert cik == "0000320193"

    # 2. Company Name resolution
    e_type, ticker, cik = annotator.infer_entity_typology("Apple Inc.")
    assert e_type == "Company"
    assert ticker == "AAPL"
    assert cik == "0000320193"

    # 3. Regulatory Body
    e_type, ticker, cik = annotator.infer_entity_typology("SEC")
    assert e_type == "RegulatoryBody"
    assert ticker is None

    # 4. Person
    e_type, ticker, cik = annotator.infer_entity_typology("Tim Cook", explicit_type="Person")
    assert e_type == "Person"


def test_directional_polarity_classification(mock_universe_mgr):
    annotator = FinancialTaxonomyAnnotator(universe_mgr=mock_universe_mgr)

    # 1. Bullish text
    bullish_text = "NVIDIA expanded its multi-year partnership with TSMC with record orders."
    pol_bull = annotator.infer_directional_polarity(bullish_text, "SUPPLIES_TO")
    assert pol_bull == "EXPANDING_BULLISH"

    # 2. Bearish text
    bearish_text = "Customer cutbacks led to reduced orders and margin compression."
    pol_bear = annotator.infer_directional_polarity(bearish_text, "SUPPLIES_TO")
    assert pol_bear == "CONTRACTING_BEARISH"

    # 3. Disruptive shock
    shock_text = "The borrower defaulted on its credit covenants and declared Chapter 11 bankruptcy."
    pol_shock = annotator.infer_directional_polarity(shock_text, "DEFAULTED_ON")
    assert pol_shock == "DISRUPTIVE_SHOCK"

    # 4. Routine stable
    stable_text = "The company filed its annual report disclosing ordinary business operations."
    pol_stable = annotator.infer_directional_polarity(stable_text, "SUPPLIES_TO")
    assert pol_stable == "NEUTRAL_STABLE"

    # 5. Negated trigger (ISSUE-10): "avoided breach", "not defaulted"
    negated_text = "The Company avoided a covenant breach and has not defaulted on its obligations."
    pol_negated = annotator.infer_directional_polarity(negated_text, "SUPPLIES_TO")
    assert pol_negated == "NEUTRAL_STABLE"


def test_status_decoupled_from_sentiment_polarity(mock_universe_mgr):
    """Verify ISSUE-10 fix: negative macro context does NOT auto-terminate intact commercial edges."""
    annotator = FinancialTaxonomyAnnotator(universe_mgr=mock_universe_mgr)

    # Passage mentions supply relationship alongside negative stock/guidance shock
    passage = "Taiwan Semiconductor supplies advanced GPUs to NVIDIA; market sentiment suffered a disruptive shock on macro tariffs."
    triple = annotator.annotate_triple(
        raw_source="TSMC",
        raw_target="NVIDIA Corporation",
        raw_rel="SUPPLIES_TO",
        context_text=passage,
        provenance="FINANCIAL_NEWS_VERIFIED",
    )

    assert triple is not None
    # Relationship itself is active, not terminated
    assert triple.status == "ACTIVE_CURRENT"
    assert triple.rel_type == "SUPPLIES_TO"


def test_financial_materiality_scoring(mock_universe_mgr):
    annotator = FinancialTaxonomyAnnotator(universe_mgr=mock_universe_mgr)

    # 1. Sole source = Critical Tier 1
    mat_crit = annotator.infer_financial_materiality("Company relies on single source supplier.", "SOLE_SOURCE_DEPENDENT_ON")
    assert mat_crit == "CRITICAL_TIER_1"

    # 2. Exhibit 21 = Critical Tier 1
    mat_ex21 = annotator.infer_financial_materiality("Wholly-owned subsidiary.", "SUBSIDIARY_OF", provenance="SEC_EXHIBIT_21")
    assert mat_ex21 == "CRITICAL_TIER_1"

    # 3. Standard supply = Material Tier 2
    mat_tier2 = annotator.infer_financial_materiality("Ongoing commercial agreement.", "SUPPLIES_TO")
    assert mat_tier2 == "MATERIAL_TIER_2"


def test_5axis_triple_annotation_and_cypher_dsl_generation(mock_universe_mgr):
    annotator = FinancialTaxonomyAnnotator(universe_mgr=mock_universe_mgr)

    passage = (
        "Apple Inc. is expanding its strategic supply agreement with Taiwan Semiconductor Manufacturing Company (TSMC) "
        "for 3nm chips with record volume commitments."
    )

    triple = annotator.annotate_triple(
        raw_source="TSMC",
        raw_target="Apple Inc.",
        raw_rel="SUPPLIES_TO",
        context_text=passage,
        provenance="SEC_10K_ITEM1",
        nature_summary="3nm silicon fabrication",
    )

    assert triple is not None
    assert triple.source_name == "TSMC"
    assert triple.target_ticker == "AAPL"
    assert triple.target_cik == "0000320193"
    assert triple.polarity == "EXPANDING_BULLISH"
    assert triple.materiality in ["CRITICAL_TIER_1", "MATERIAL_TIER_2"]
    assert triple.status == "ACTIVE_CURRENT"
    assert triple.confidence == 0.98

    dsl = triple.to_cypher_dsl()
    assert "(:Company" in dsl
    assert '-[:SUPPLIES_TO {' in dsl
    assert 'polarity: "EXPANDING_BULLISH"' in dsl
    assert 'ticker: "AAPL"' in dsl


def test_inverted_triple_auto_orientation_in_annotation(mock_universe_mgr):
    annotator = FinancialTaxonomyAnnotator(universe_mgr=mock_universe_mgr)

    # Inverted: Company -> CEO_OF -> Person
    triple = annotator.annotate_triple(
        raw_source="Apple Inc.",
        raw_target="Tim Cook",
        raw_rel="CEO_OF",
        explicit_source_type="Company",
        explicit_target_type="Person",
        context_text="Tim Cook serves as Chief Executive Officer of Apple Inc.",
        provenance="SEC_10K_ITEM1",
    )

    assert triple is not None
    # Must be auto-oriented: Person -> CEO_OF -> Company
    assert triple.source_name == "Tim Cook"
    assert triple.source_type == "Person"
    assert triple.target_name == "Apple Inc."
    assert triple.target_ticker == "AAPL"
    assert triple.rel_type == "CEO_OF"
