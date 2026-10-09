"""
Comprehensive unit tests for the Dataset Cartographer Engine.
Verifies Swayamdipta et al. (EMNLP 2020) and Toneva et al. (ICLR 2019) dynamics calculations.
"""

import json
from pathlib import Path
import numpy as np
import pytest

try:
    from curation.contracts import (
        ActiveLearningPartition,
        CartographyCoordinate,
        CartographyRegion,
    )
    from curation.active_learning.cartographer import DatasetCartographer
except ImportError:
    from graphrag_finance.curation.contracts import (
        ActiveLearningPartition,
        CartographyCoordinate,
        CartographyRegion,
    )
    from graphrag_finance.curation.active_learning.cartographer import DatasetCartographer


def test_contracts_validation():
    """Verify Pydantic models and Region enum constraints."""
    coord = CartographyCoordinate(
        guid="sample_001",
        confidence=0.8543,
        variability=0.1234,
        correctness=1.0,
        forgetfulness=0,
        region=CartographyRegion.EASY_TO_LEARN,
    )
    assert coord.guid == "sample_001"
    assert coord.confidence == 0.8543
    assert coord.region == CartographyRegion.EASY_TO_LEARN
    dumped = json.loads(coord.model_dump_json())
    assert dumped["region"] == "easy_to_learn"

    partition = ActiveLearningPartition(
        retained_guids=["a", "b"],
        pruned_easy_guids=["c"],
        quarantined_hard_guids=["d"],
        total_candidates=4,
        retention_rate_pct=50.0,
    )
    assert partition.retention_rate_pct == 50.0
    assert len(partition.retained_guids) == 2


def test_compute_forgetfulness():
    """Verify Toneva et al. (2019) example forgetfulness state transitions."""
    cartographer = DatasetCartographer()

    # Never learned -> sentinel 999
    assert cartographer.compute_forgetfulness([]) == 999
    assert cartographer.compute_forgetfulness([False, False, False]) == 999

    # Always correct -> 0
    assert cartographer.compute_forgetfulness([True, True, True]) == 0

    # Learned on epoch 3, never forgotten -> 0
    assert cartographer.compute_forgetfulness([False, False, True]) == 0

    # Learned on epoch 2, forgotten on epoch 3 -> 1
    assert cartographer.compute_forgetfulness([False, True, False]) == 1

    # Learned epoch 1, forgotten epoch 2, relearned epoch 3, forgotten epoch 4 -> 2
    assert cartographer.compute_forgetfulness([True, False, True, False]) == 2

    # Multiple consecutive correct epochs before forgetting
    assert cartographer.compute_forgetfulness([True, True, False, False, True]) == 1


def test_classify_region():
    """Verify boundary thresholds and classification logic."""
    cartographer = DatasetCartographer(
        conf_thresh_high=0.70,
        conf_thresh_low=0.30,
        var_thresh=0.15,
    )

    # 1. Ambiguous (Variability >= 0.15 regardless of confidence)
    assert cartographer.classify_region(confidence=0.85, variability=0.18) == CartographyRegion.AMBIGUOUS
    assert cartographer.classify_region(confidence=0.25, variability=0.15) == CartographyRegion.AMBIGUOUS
    assert cartographer.classify_region(confidence=0.50, variability=0.22) == CartographyRegion.AMBIGUOUS

    # 2. Easy-to-Learn (Variability < 0.15 and Confidence >= 0.70)
    assert cartographer.classify_region(confidence=0.92, variability=0.04) == CartographyRegion.EASY_TO_LEARN
    assert cartographer.classify_region(confidence=0.70, variability=0.14) == CartographyRegion.EASY_TO_LEARN

    # 3. Hard-to-Learn (Variability < 0.15 and Confidence < 0.70)
    assert cartographer.classify_region(confidence=0.12, variability=0.03) == CartographyRegion.HARD_TO_LEARN
    assert cartographer.classify_region(confidence=0.69, variability=0.14) == CartographyRegion.HARD_TO_LEARN
    assert cartographer.classify_region(confidence=0.28, variability=0.05) == CartographyRegion.HARD_TO_LEARN


def test_parse_dynamics_log_streaming(tmp_path: Path):
    """Verify streaming parser handles out-of-order epochs, sample_id alias, and blank lines."""
    log_file = tmp_path / "training_dynamics.jsonl"
    entries = [
        {"epoch": 2, "guid": "sample_A", "seq_prob": 0.85, "is_correct": True},
        {"epoch": 1, "guid": "sample_A", "seq_prob": 0.65, "is_correct": False},
        {"epoch": 3, "guid": "sample_A", "seq_prob": 0.95, "is_correct": True},
        # Uses sample_id alias
        {"epoch": 1, "sample_id": "sample_B", "seq_prob": 0.10, "is_correct": False},
        {"epoch": 2, "sample_id": "sample_B", "seq_prob": 0.12, "is_correct": False},
        # Empty whitespace line
        "",
    ]
    with open(log_file, "w", encoding="utf-8") as f:
        for item in entries:
            if isinstance(item, dict):
                f.write(json.dumps(item) + "\n")
            else:
                f.write("\n")

    cartographer = DatasetCartographer()
    trends = cartographer.parse_dynamics_log(log_file)

    assert "sample_A" in trends
    assert "sample_B" in trends

    # Verify epochs are sorted 1, 2, 3
    assert trends["sample_A"]["epochs"] == [1, 2, 3]
    assert trends["sample_A"]["probs"] == [0.65, 0.85, 0.95]
    assert trends["sample_A"]["correct"] == [False, True, True]

    assert trends["sample_B"]["epochs"] == [1, 2]
    assert trends["sample_B"]["probs"] == [0.10, 0.12]


def test_map_dataset_mathematical_precision(tmp_path: Path):
    """Verify statistical formulas match AllenAI paper implementation exactly."""
    log_file = tmp_path / "test_dynamics.jsonl"

    # Known samples
    # Sample Easy: [0.80, 0.85, 0.90] -> mean = 0.85, std = np.std([0.80, 0.85, 0.90]) ~ 0.0408
    # Sample Ambiguous: [0.30, 0.90, 0.40] -> mean ~ 0.5333, std = np.std([0.30, 0.90, 0.40]) ~ 0.2625
    # Sample Hard: [0.05, 0.08, 0.05] -> mean ~ 0.06, std ~ 0.0141
    records = [
        {"epoch": 1, "guid": "easy_1", "seq_prob": 0.80, "is_correct": True},
        {"epoch": 2, "guid": "easy_1", "seq_prob": 0.85, "is_correct": True},
        {"epoch": 3, "guid": "easy_1", "seq_prob": 0.90, "is_correct": True},
        {"epoch": 1, "guid": "ambig_1", "seq_prob": 0.30, "is_correct": False},
        {"epoch": 2, "guid": "ambig_1", "seq_prob": 0.90, "is_correct": True},
        {"epoch": 3, "guid": "ambig_1", "seq_prob": 0.40, "is_correct": False},
        {"epoch": 1, "guid": "hard_1", "seq_prob": 0.05, "is_correct": False},
        {"epoch": 2, "guid": "hard_1", "seq_prob": 0.08, "is_correct": False},
        {"epoch": 3, "guid": "hard_1", "seq_prob": 0.05, "is_correct": False},
    ]

    with open(log_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    cartographer = DatasetCartographer()
    coordinates = cartographer.map_dataset(log_file)
    coord_map = {c.guid: c for c in coordinates}

    # Verify easy_1
    c_easy = coord_map["easy_1"]
    assert c_easy.confidence == 0.85
    assert np.isclose(c_easy.variability, float(np.std([0.80, 0.85, 0.90])), atol=1e-4)
    assert c_easy.correctness == 1.0
    assert c_easy.forgetfulness == 0
    assert c_easy.region == CartographyRegion.EASY_TO_LEARN

    # Verify ambig_1
    c_ambig = coord_map["ambig_1"]
    assert np.isclose(c_ambig.confidence, float(np.mean([0.30, 0.90, 0.40])), atol=1e-4)
    assert c_ambig.variability > 0.15
    assert c_ambig.correctness == round(1 / 3, 4)
    assert c_ambig.forgetfulness == 1  # Correct at ep 2, wrong at ep 3
    assert c_ambig.region == CartographyRegion.AMBIGUOUS

    # Verify hard_1
    c_hard = coord_map["hard_1"]
    assert c_hard.confidence < 0.30
    assert c_hard.variability < 0.15
    assert c_hard.correctness == 0.0
    assert c_hard.forgetfulness == 999  # Never learned
    assert c_hard.region == CartographyRegion.HARD_TO_LEARN


def test_filter_active_learning_recipe():
    """Verify active learning partition policy: 100% ambiguous retained, 80% easy pruned, hard quarantined."""
    cartographer = DatasetCartographer()

    # Construct mock dataset: 50 easy, 30 ambiguous, 20 hard = 100 total
    coords = []
    for i in range(50):
        coords.append(
            CartographyCoordinate(
                guid=f"easy_{i}",
                confidence=0.90,
                variability=0.05,
                correctness=1.0,
                forgetfulness=0,
                region=CartographyRegion.EASY_TO_LEARN,
            )
        )
    for i in range(30):
        coords.append(
            CartographyCoordinate(
                guid=f"ambig_{i}",
                confidence=0.55,
                variability=0.25,
                correctness=0.5,
                forgetfulness=1,
                region=CartographyRegion.AMBIGUOUS,
            )
        )
    for i in range(20):
        coords.append(
            CartographyCoordinate(
                guid=f"hard_{i}",
                confidence=0.10,
                variability=0.02,
                correctness=0.0,
                forgetfulness=999,
                region=CartographyRegion.HARD_TO_LEARN,
            )
        )

    partition = cartographer.filter_active_learning(coords, prune_easy_ratio=0.80, random_seed=42)

    assert partition.total_candidates == 100
    # Ambiguous: all 30 must be in retained
    for i in range(30):
        assert f"ambig_{i}" in partition.retained_guids

    # Hard: all 20 must be in quarantined
    assert len(partition.quarantined_hard_guids) == 20
    for i in range(20):
        assert f"hard_{i}" in partition.quarantined_hard_guids
        assert f"hard_{i}" not in partition.retained_guids

    # Easy: 50 total * (1 - 0.80) = 10 retained, 40 pruned
    easy_retained = [g for g in partition.retained_guids if g.startswith("easy_")]
    assert len(easy_retained) == 10
    assert len(partition.pruned_easy_guids) == 40

    # Total retained: 30 ambiguous + 10 easy = 40
    assert len(partition.retained_guids) == 40
    assert partition.retention_rate_pct == 40.0


def test_export_coordinates_jsonl(tmp_path: Path):
    """Verify serializing coordinates to JSONL file."""
    cartographer = DatasetCartographer()
    coords = [
        CartographyCoordinate(
            guid="s1",
            confidence=0.95,
            variability=0.02,
            correctness=1.0,
            forgetfulness=0,
            region=CartographyRegion.EASY_TO_LEARN,
        ),
        CartographyCoordinate(
            guid="s2",
            confidence=0.45,
            variability=0.22,
            correctness=0.6,
            forgetfulness=1,
            region=CartographyRegion.AMBIGUOUS,
        ),
    ]

    out_file = tmp_path / "coords.jsonl"
    cartographer.export_coordinates_jsonl(coords, out_file)
    assert out_file.is_file()

    lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").strip().split("\n")]
    assert len(lines) == 2
    assert lines[0]["guid"] == "s1"
    assert lines[1]["region"] == "ambiguous"
