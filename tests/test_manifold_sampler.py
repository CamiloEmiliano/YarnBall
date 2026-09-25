# tests/test_manifold_sampler.py
# -*- coding: utf-8 -*-
"""Unit tests for tools/sft_manifold_sampler.py."""

import json
from pathlib import Path
import pytest
from unittest.mock import MagicMock

from tools.sft_manifold_sampler import (
    ManifoldTargetedSampler,
    ManifoldSample,
    RARE_RELATION_FLOORS,
    DOMINANT_CLASS_CAP,
)
from tools.sp500_universe import SP500Constituent


@pytest.fixture
def mock_universe_mgr():
    mgr = MagicMock()
    mock_constituents = [
        SP500Constituent(ticker="AAPL", cik="0000320193", company_name="Apple Inc.", gics_sector="Information Technology"),
        SP500Constituent(ticker="MSFT", cik="0000789019", company_name="Microsoft Corp", gics_sector="Information Technology"),
        SP500Constituent(ticker="NVDA", cik="0001045810", company_name="NVIDIA Corporation", gics_sector="Information Technology"),
        SP500Constituent(ticker="TSLA", cik="0001318605", company_name="Tesla Inc.", gics_sector="Consumer Discretionary"),
        SP500Constituent(ticker="JPM", cik="0000019617", company_name="JPMorgan Chase", gics_sector="Financials"),
    ]
    mgr.get_current_constituents.return_value = mock_constituents
    mgr.is_constituent.side_effect = lambda t, target_date=None: t in {"AAPL", "MSFT", "NVDA", "TSLA", "JPM"}
    return mgr


def test_rare_relation_trigger_mining(mock_universe_mgr, tmp_path: Path):
    sampler = ManifoldTargetedSampler(output_dir=tmp_path, universe_mgr=mock_universe_mgr)

    passages = [
        {"raw_text": "The company relies on TSMC on a sole source basis for custom ASICs."},
        {"raw_text": "Qualcomm entered into a multi-year patent license from ARM Holdings."},
        {"raw_text": "The borrower received a formal notice of default from the syndicate."},
        {"raw_text": "General routine earnings report with regular quarterly metrics."},
    ]

    mined = sampler.mine_rare_relation_candidates(passages)

    assert "SOLE_SOURCE_DEPENDENT_ON" in mined
    assert len(mined["SOLE_SOURCE_DEPENDENT_ON"]) == 1

    assert "LICENSES_FROM" in mined
    assert len(mined["LICENSES_FROM"]) == 1

    assert "DEFAULTED_ON" in mined
    assert len(mined["DEFAULTED_ON"]) == 1


def test_hard_negative_synthesis(mock_universe_mgr, tmp_path: Path):
    sampler = ManifoldTargetedSampler(output_dir=tmp_path, universe_mgr=mock_universe_mgr)

    commentary = [
        {
            "raw_text": (
                "Equities traded mixed on Thursday. Both Apple Inc. and JPMorgan Chase saw modest outflows "
                "following the Federal Reserve's interest rate announcement. Tech stocks lagged while energy gained. "
                "Investors continued to assess macroeconomic inflation data and bond yields across major benchmark indices."
            ),
            "provider": "MARKET_PULSE",
        }
    ]

    # No economic relationship exists between AAPL and JPM
    known_active_pairs = {("AAPL", "NVDA"), ("MSFT", "NVDA")}

    negs = sampler.synthesize_hard_negatives(
        market_commentary_passages=commentary,
        known_active_pairs=known_active_pairs,
        target_count=5,
    )

    assert len(negs) == 1
    neg = negs[0]
    assert neg.is_hard_negative is True
    assert neg.grounded_triples == [] # Target is empty!
    assert neg.provenance == "HEURISTIC_HARD_NEGATIVE"
    assert neg.confidence == 0.65 # Capped heuristic negative confidence
    assert neg.difficulty_score >= 0.80


def test_ticker_word_collision_guard(mock_universe_mgr, tmp_path: Path):
    """Verify ISSUE-11 fix: common English words matching tickers (e.g. 'SO', 'ON', 'IT') do not mine false hard negatives without cashtags."""
    sampler = ManifoldTargetedSampler(output_dir=tmp_path, universe_mgr=mock_universe_mgr)

    # Passage with ordinary English words "so", "on", "it" but only 1 real company "Apple Inc."
    commentary = [
        {
            "raw_text": (
                "So, on Thursday the market focused on broader inflation data. It was clear that Apple Inc. "
                "outperformed peers while bond yields surged across global markets in extended trading."
            ),
        }
    ]

    negs = sampler.synthesize_hard_negatives(
        market_commentary_passages=commentary,
        known_active_pairs=set(),
        target_count=5,
    )

    # Since only 1 company (Apple) was mentioned without colliding 'SO'/'ON', no 2-company hard negative is synthesized
    assert len(negs) == 0

    # If cashtags are present: "$SO and $AAPL" -> correctly synthesized
    cashtag_commentary = [
        {
            "raw_text": (
                "Trading updates showed both $SO and $AAPL moved in opposite directions today following "
                "diverging sector rotation between regulated utilities and large cap consumer technology. "
                "Portfolio managers adjusted allocations across benchmark index components ahead of the closing bell."
            ),
        }
    ]

    mock_universe_mgr.get_current_constituents.return_value.append(
        SP500Constituent(ticker="SO", cik="0000092122", company_name="Southern Company", gics_sector="Utilities")
    )
    negs_cashtag = sampler.synthesize_hard_negatives(
        market_commentary_passages=cashtag_commentary,
        known_active_pairs=set(),
        target_count=5,
    )
    assert len(negs_cashtag) == 1


def test_manifold_class_balancing_floors_and_caps(mock_universe_mgr, tmp_path: Path):
    sampler = ManifoldTargetedSampler(output_dir=tmp_path, universe_mgr=mock_universe_mgr, null_sample_ratio=0.20)

    # 1. Create 800 dominant class samples (exceeds cap of 600)
    dominant_samples = [
        ManifoldSample(
            sample_id=f"DOM_{i}",
            text_passage=f"SubCorp {i} operates as a subsidiary of ParentCorp {i}.",
            grounded_triples=[{"source_id": f"SubCorp {i}", "target_id": f"ParentCorp {i}", "rel_type": "SUBSIDIARY_OF"}],
            entities_present=[],
            hop_count=1,
            is_hard_negative=False,
            gics_sector="Information Technology",
            provenance="SEC_EXHIBIT_21",
            confidence=1.0,
            difficulty_score=0.2,
        )
        for i in range(800)
    ]

    # 2. Create 10 rare class samples (below floor of 200, triggers augmentation)
    rare_samples = [
        ManifoldSample(
            sample_id=f"RARE_{i}",
            text_passage="Company relies on sole source supplier for mission-critical parts.",
            grounded_triples=[{"source_id": "Apple Inc.", "target_id": "NVIDIA Corporation", "rel_type": "SOLE_SOURCE_DEPENDENT_ON"}],
            entities_present=[],
            hop_count=1,
            is_hard_negative=False,
            gics_sector="Information Technology",
            provenance="SEC_10K_ITEM1",
            confidence=0.98,
            difficulty_score=0.6,
        )
        for i in range(10)
    ]

    # 3. Create hard negatives
    hard_negs = [
        ManifoldSample(
            sample_id=f"NEG_{i}",
            text_passage="Market commentary text...",
            grounded_triples=[],
            entities_present=[],
            hop_count=0,
            is_hard_negative=True,
            gics_sector="Cross-Sector",
            provenance="NEWS",
            confidence=1.0,
            difficulty_score=0.85,
        )
        for i in range(200)
    ]

    curated = sampler.balance_and_curate_manifold(
        positive_samples=dominant_samples + rare_samples,
        hard_negatives=hard_negs,
    )

    # Verify dominant class is capped at DOMINANT_CLASS_CAP (600)
    subsidiary_count = sum(
        1 for s in curated
        if s.grounded_triples and s.grounded_triples[0].get("rel_type") == "SUBSIDIARY_OF"
    )
    assert subsidiary_count == DOMINANT_CLASS_CAP

    # Verify rare class meets floor of 200 via augmentation
    sole_source_count = sum(
        1 for s in curated
        if s.grounded_triples and s.grounded_triples[0].get("rel_type") == "SOLE_SOURCE_DEPENDENT_ON"
    )
    assert sole_source_count == RARE_RELATION_FLOORS["SOLE_SOURCE_DEPENDENT_ON"]

    # Verify hard negatives are blended in
    neg_count = sum(1 for s in curated if s.is_hard_negative)
    assert neg_count > 0


def test_manifold_dataset_export_jsonl(mock_universe_mgr, tmp_path: Path):
    sampler = ManifoldTargetedSampler(output_dir=tmp_path, universe_mgr=mock_universe_mgr)

    samples = [
        ManifoldSample(
            sample_id="EXP_001",
            text_passage="Sample text",
            grounded_triples=[{"source_id": "A", "target_id": "B", "rel_type": "SUPPLIES_TO"}],
            entities_present=[{"name": "A", "ticker": "A", "type": "Company"}],
            hop_count=1,
            is_hard_negative=False,
            gics_sector="Information Technology",
            provenance="SEC_10K",
            confidence=0.95,
            difficulty_score=0.3,
        )
    ]

    out_file = sampler.export_manifold_dataset(samples, filename="test_export.jsonl")
    assert out_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["sample_id"] == "EXP_001"
        assert data["grounded_triples"][0]["rel_type"] == "SUPPLIES_TO"
