"""
Unit and integration tests for sft_pipeline package.

Tests:
1. Public API exports from sft_pipeline.
2. Legacy backward-compatibility adapters in tools/.
3. FullSFTDatasetBuilder initialization and multi-task record synthesis.
4. Pre-training curation: ColdStartCurationPipeline integration (MinHash & Coreset pruning).
5. Post-training curation: ActiveLearningPartition feedback & ManifoldTargetedSampler steering.
6. Hugging Face Hub dataset card template and upload utility behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Test direct imports from sft_pipeline package
from sft_pipeline import (
    FullSFTDatasetBuilder,
    ManifoldTargetedSampler,
    ManifoldSample,
    FinancialTaxonomyAnnotator,
    AnnotatedTriple,
    SFTDatasetExporter,
    SFTRecord,
    RARE_RELATION_FLOORS,
    DOMINANT_CLASS_CAP,
    upload_to_huggingface,
)
from sft_pipeline.builder import (
    MULTI_SECTOR_RELATION_TEMPLATES,
    COMMENTARY_TEMPLATES,
    OUTPUT_SFT_DIR,
)

from tools.sp500_universe import SP500Constituent
from curation.contracts import ActiveLearningPartition


@pytest.fixture
def mock_universe_mgr():
    mgr = MagicMock()
    mock_constituents = [
        SP500Constituent(ticker="AAPL", cik="0000320193", company_name="Apple Inc.", gics_sector="Information Technology"),
        SP500Constituent(ticker="MSFT", cik="0000789019", company_name="Microsoft Corp", gics_sector="Information Technology"),
        SP500Constituent(ticker="NVDA", cik="0001045810", company_name="NVIDIA Corporation", gics_sector="Information Technology"),
        SP500Constituent(ticker="TSLA", cik="0001318605", company_name="Tesla Inc.", gics_sector="Consumer Discretionary"),
        SP500Constituent(ticker="JPM", cik="0000019617", company_name="JPMorgan Chase", gics_sector="Financials"),
        SP500Constituent(ticker="PFE", cik="0000078003", company_name="Pfizer Inc.", gics_sector="Health Care"),
        SP500Constituent(ticker="XOM", cik="0000034088", company_name="Exxon Mobil Corp", gics_sector="Energy"),
        SP500Constituent(ticker="CAT", cik="0000018230", company_name="Caterpillar Inc.", gics_sector="Industrials"),
        SP500Constituent(ticker="LIN", cik="0001707925", company_name="Linde plc", gics_sector="Materials"),
        SP500Constituent(ticker="NEE", cik="0000753308", company_name="NextEra Energy Inc.", gics_sector="Utilities"),
        SP500Constituent(ticker="PLD", cik="0001045609", company_name="Prologis Inc.", gics_sector="Real Estate"),
    ]
    mgr.get_all_records.return_value = mock_constituents
    mgr.get_current_constituents.return_value = mock_constituents
    mgr.is_constituent.side_effect = lambda t, target_date=None: t in {c.ticker for c in mock_constituents}
    return mgr


def test_package_exports():
    """Verify that sft_pipeline exposes canonical classes and constants."""
    assert FullSFTDatasetBuilder is not None
    assert upload_to_huggingface is not None
    assert ManifoldTargetedSampler is not None
    assert FinancialTaxonomyAnnotator is not None
    assert SFTDatasetExporter is not None
    assert len(MULTI_SECTOR_RELATION_TEMPLATES) >= 40
    assert len(COMMENTARY_TEMPLATES) >= 4


def test_builder_initialization(mock_universe_mgr, tmp_path: Path):
    """Test builder initialization with mocked universe."""
    builder = FullSFTDatasetBuilder(
        output_dir=tmp_path,
        seed=42,
        universe_mgr=mock_universe_mgr,
        enable_cold_start_curation=False,
    )
    assert builder.output_dir == tmp_path
    assert builder.seed == 42
    assert len(builder.constituents) == 11
    assert "Information Technology" in builder.constituents_by_sector
    assert "Financials" in builder.constituents_by_sector


def test_builder_positive_and_negative_generation(mock_universe_mgr, tmp_path: Path):
    """Test generating positive and hard negative manifold samples."""
    builder = FullSFTDatasetBuilder(
        output_dir=tmp_path,
        seed=42,
        universe_mgr=mock_universe_mgr,
    )

    positives = builder.generate_positive_manifold_samples(target_per_template=2)
    assert len(positives) > 0
    for p in positives:
        assert isinstance(p, ManifoldSample)
        assert not p.is_hard_negative
        assert len(p.grounded_triples) > 0
        assert 0.0 <= p.confidence <= 1.0
        assert 0.0 <= p.difficulty_score <= 1.0

    negatives = builder.generate_hard_negative_samples(positives=positives, count=10)
    assert len(negatives) == 10
    for n in negatives:
        assert n.is_hard_negative
        assert n.grounded_triples == []


def test_builder_cold_start_curation_integration(mock_universe_mgr, tmp_path: Path):
    """Verify ColdStartCurationPipeline integration inside sft_pipeline.builder."""
    builder = FullSFTDatasetBuilder(
        output_dir=tmp_path,
        seed=42,
        universe_mgr=mock_universe_mgr,
    )

    # Create samples with near-duplicate text
    sample1 = ManifoldSample(
        sample_id="SAMPLE_001",
        text_passage="Apple Inc contracts with NVIDIA Corporation for advanced graphic processing units.",
        grounded_triples=[{"source_id": "Apple Inc.", "target_id": "NVIDIA Corporation", "rel_type": "LICENSES_FROM", "confidence": 0.9}],
        entities_present=[{"name": "Apple Inc."}, {"name": "NVIDIA Corporation"}],
        hop_count=1,
        is_hard_negative=False,
        gics_sector="Information Technology",
        provenance="TEST",
        confidence=0.95,
        difficulty_score=0.4,
    )
    # Near identical copy
    sample2 = ManifoldSample(
        sample_id="SAMPLE_002",
        text_passage="Apple Inc contracts with NVIDIA Corporation for advanced graphic processing units.",
        grounded_triples=[{"source_id": "Apple Inc.", "target_id": "NVIDIA Corporation", "rel_type": "LICENSES_FROM", "confidence": 0.9}],
        entities_present=[{"name": "Apple Inc."}, {"name": "NVIDIA Corporation"}],
        hop_count=1,
        is_hard_negative=False,
        gics_sector="Information Technology",
        provenance="TEST",
        confidence=0.95,
        difficulty_score=0.4,
    )
    # Distinct sample
    sample3 = ManifoldSample(
        sample_id="SAMPLE_003",
        text_passage="Exxon Mobil Corp announced exploration lease with deepwater offshore drilling operators.",
        grounded_triples=[{"source_id": "Exxon Mobil Corp", "target_id": "Drilling Operator", "rel_type": "JOINT_VENTURE_WITH", "confidence": 0.9}],
        entities_present=[{"name": "Exxon Mobil Corp"}],
        hop_count=1,
        is_hard_negative=False,
        gics_sector="Energy",
        provenance="TEST",
        confidence=0.90,
        difficulty_score=0.5,
    )

    retained = builder.apply_cold_start_curation(
        [sample1, sample2, sample3],
        minhash_threshold=0.80,
        coreset_retention_ratio=1.0,
    )

    # Near identical duplicate should be pruned
    assert len(retained) == 2
    retained_ids = {s.sample_id for s in retained}
    assert "SAMPLE_003" in retained_ids
    # Exactly one of SAMPLE_001 or SAMPLE_002 is retained
    assert ("SAMPLE_001" in retained_ids) ^ ("SAMPLE_002" in retained_ids)


def test_builder_active_learning_curation_integration(mock_universe_mgr, tmp_path: Path):
    """Verify ActiveLearningPartition integration and sampler steering inside builder."""
    builder = FullSFTDatasetBuilder(
        output_dir=tmp_path,
        seed=42,
        universe_mgr=mock_universe_mgr,
    )

    s_easy = ManifoldSample(
        sample_id="EASY_001",
        text_passage="Passage easy",
        grounded_triples=[{"source_id": "A", "target_id": "B", "rel_type": "SUPPLIER_TO", "confidence": 0.9}],
        entities_present=[],
        hop_count=1,
        is_hard_negative=False,
        gics_sector="Information Technology",
        provenance="TEST",
        confidence=0.99,
        difficulty_score=0.1,
    )
    s_ambiguous = ManifoldSample(
        sample_id="AMB_001",
        text_passage="Passage ambiguous",
        grounded_triples=[{"source_id": "A", "target_id": "B", "rel_type": "DEFAULTED_ON", "confidence": 0.9}],
        entities_present=[],
        hop_count=1,
        is_hard_negative=False,
        gics_sector="Financials",
        provenance="TEST",
        confidence=0.70,
        difficulty_score=0.8,
    )
    s_hard_quarantined = ManifoldSample(
        sample_id="HARD_001",
        text_passage="Passage hard quarantined",
        grounded_triples=[{"source_id": "A", "target_id": "B", "rel_type": "EXPOSED_TO_RISK", "confidence": 0.9}],
        entities_present=[],
        hop_count=1,
        is_hard_negative=False,
        gics_sector="Industrials",
        provenance="TEST",
        confidence=0.50,
        difficulty_score=0.95,
    )

    partition = ActiveLearningPartition(
        retained_guids=["EASY_001", "AMB_001"],
        pruned_easy_guids=[],
        quarantined_hard_guids=["HARD_001"],
        total_candidates=3,
        retention_rate_pct=66.67,
    )

    initial_floor = builder.sampler.relation_floors.get("DEFAULTED_ON", 200)

    filtered = builder.apply_active_learning_curation(
        [s_easy, s_ambiguous, s_hard_quarantined],
        partition_or_path=partition,
    )

    # Quarantined sample must be dropped
    filtered_ids = {s.sample_id for s in filtered}
    assert "HARD_001" not in filtered_ids
    assert "AMB_001" in filtered_ids
    assert "EASY_001" in filtered_ids
    assert len(filtered) == 2

    # Sampler must have steered DEFAULTED_ON floor upward
    steered_floor = builder.sampler.relation_floors.get("DEFAULTED_ON", 200)
    assert steered_floor > initial_floor
    assert steered_floor == int(initial_floor * 1.5)


def test_builder_task_d_and_e_records(mock_universe_mgr, tmp_path: Path):
    """Test generating high-cognitive reasoning records for Tasks D and E."""
    builder = FullSFTDatasetBuilder(
        output_dir=tmp_path,
        seed=42,
        universe_mgr=mock_universe_mgr,
    )

    recs_d = builder.generate_task_d_contagion_records(count=5)
    assert len(recs_d) == 5
    for r in recs_d:
        assert r.metadata["task_type"] == "CONTAGION_REASONING"
        assert "<|contagion_reasoning|>" in r.prompt
        assert "<think>" in r.target_completion
        assert "</think>" in r.target_completion

    recs_e = builder.generate_task_e_hedging_records(count=5)
    assert len(recs_e) == 5
    for r in recs_e:
        assert r.metadata["task_type"] == "PORTFOLIO_RECOMMENDATION"
        assert "<|portfolio_recommendation|>" in r.prompt
        assert "<think>" in r.target_completion
        assert "</think>" in r.target_completion


def test_hf_sync_behavior(tmp_path: Path):
    """Test HF sync upload logic and validation under mocked API."""
    import sft_pipeline.hf_sync as hf_mod

    # 1. Missing token raises ValueError when HfApi is present
    with patch.object(hf_mod, "HfApi", MagicMock()), patch.object(hf_mod, "create_repo", MagicMock()):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="No Hugging Face token"):
                upload_to_huggingface(data_dir=tmp_path, token="")

    # 2. Successful mock upload
    mock_api = MagicMock()
    mock_api.whoami.return_value = {"name": "test_user"}
    mock_create_repo = MagicMock()

    with patch.object(hf_mod, "HfApi", return_value=mock_api), patch.object(hf_mod, "create_repo", mock_create_repo):
        url = upload_to_huggingface(
            repo_name="mock-dataset",
            data_dir=tmp_path,
            token="hf_mock_token_12345",
            private=True,
        )
        assert url == "https://huggingface.co/datasets/test_user/mock-dataset"
        mock_create_repo.assert_called_once_with(
            repo_id="test_user/mock-dataset",
            repo_type="dataset",
            private=True,
            exist_ok=True,
            token="hf_mock_token_12345",
        )
        mock_api.upload_folder.assert_called_once()
