"""
Unit tests for Cold-Start Pre-Training Curation Pipeline (MinHash LSH, SemDeDup, Coreset Facility Location).
"""

import numpy as np
import pytest

from curation.cold_start.minhash_lsh import MinHashLSHDeduplicator
from curation.cold_start.semdedup import SemanticDeduplicator
from curation.cold_start.coreset import FacilityLocationCoresetSelector
from curation.cold_start.pipeline import ColdStartCurationPipeline


def test_minhash_near_duplicates():
    """Verify MinHash LSH detects near-duplicate syndicated news text."""
    dedup = MinHashLSHDeduplicator(jaccard_threshold=0.75, num_perm=128)

    # 1. Base regulatory filing risk disclosure
    doc1 = {
        "guid": "doc_1",
        "text": (
            "The Company depends on component suppliers and manufacturing partners located in East Asia. "
            "In particular, single source suppliers provide specialized sub-3nm semiconductor wafers, optical sensors, "
            "and battery assemblies. Any disruption from trade restrictions, natural disasters, or labor disputes could "
            "materially impact our financial condition, operating results, and gross margins."
        ),
    }
    # 2. Syndicated copy with one synonymous word variation ("trade restrictions" -> "export restrictions")
    doc2 = {
        "guid": "doc_2",
        "text": (
            "The Company depends on component suppliers and manufacturing partners located in East Asia. "
            "In particular, single source suppliers provide specialized sub-3nm semiconductor wafers, optical sensors, "
            "and battery assemblies. Any disruption from export restrictions, natural disasters, or labor disputes could "
            "materially impact our financial condition, operating results, and gross margins."
        ),
    }
    # 3. Completely unrelated passage
    doc3 = {
        "guid": "doc_3",
        "text": (
            "ExxonMobil finalized its multi-billion dollar offshore crude drilling exploration contract in the "
            "Permian Basin and Guyana offshore fields, expanding total deepwater barrels produced per operating day."
        ),
    }

    retained, duplicates = dedup.deduplicate([doc1, doc2, doc3])

    # Should retain 2 distinct topics and prune 1 duplicate
    assert len(retained) == 2
    assert len(duplicates) == 1
    retained_guids = {r["guid"] for r in retained}
    assert "doc_3" in retained_guids
    assert ("doc_1" in retained_guids) or ("doc_2" in retained_guids)


def test_minhash_identical_documents():
    """Verify exact duplicate documents are pruned."""
    dedup = MinHashLSHDeduplicator()
    text = "Taiwan Semiconductor Manufacturing Co supplies advanced 3nm chips to Nvidia Corporation for Blackwell server production."
    records = [{"guid": f"item_{i}", "text": text} for i in range(5)]

    retained, duplicates = dedup.deduplicate(records)
    assert len(retained) == 1
    assert len(duplicates) == 4


def test_semantic_dedup_clustering():
    """Verify SemDeDup groups high-cosine embeddings and retains centroid."""
    dedup = SemanticDeduplicator(cosine_threshold=0.88, embedding_dim=4)

    records = [{"guid": f"s_{i}", "text": f"sample text {i}"} for i in range(4)]

    # Pair 0 and 1 are nearly collinear (cos ~ 0.99)
    # Pair 2 and 3 are orthogonal
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.99, 0.05, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=np.float32)

    retained, duplicates = dedup.deduplicate_with_embeddings(records, embeddings)

    assert len(retained) == 3
    assert len(duplicates) == 1
    retained_guids = {r["guid"] for r in retained}
    assert "s_2" in retained_guids
    assert "s_3" in retained_guids
    assert ("s_0" in retained_guids) or ("s_1" in retained_guids)


def test_coreset_facility_location_geometric_coverage():
    """Verify Facility Location selects diverse representatives across clusters."""
    selector = FacilityLocationCoresetSelector(target_budget=2)

    records = [{"guid": f"p_{i}", "text": f"passage {i}"} for i in range(4)]

    # Two distinct clusters in 2D space:
    # Cluster A: points 0, 1 (near [1, 0])
    # Cluster B: points 2, 3 (near [0, 1])
    embeddings = np.array([
        [1.0, 0.02],
        [0.98, 0.01],
        [0.02, 1.0],
        [0.01, 0.99],
    ], dtype=np.float32)

    selected, pruned = selector.select_coreset(records, embeddings, budget=2)

    assert len(selected) == 2
    assert len(pruned) == 2

    selected_guids = {s["guid"] for s in selected}
    # Greedy facility location must pick one from Cluster A and one from Cluster B
    has_cluster_a = ("p_0" in selected_guids) or ("p_1" in selected_guids)
    has_cluster_b = ("p_2" in selected_guids) or ("p_3" in selected_guids)
    assert has_cluster_a and has_cluster_b


def test_cold_start_pipeline_orchestration():
    """Verify ColdStartCurationPipeline runs full chain with partitioning."""
    pipeline = ColdStartCurationPipeline(
        minhash_threshold=0.85,
        semdedup_threshold=0.88,
        coreset_retention_ratio=0.80,
    )

    records = [
        {"sample_id": f"rec_{i}", "prompt": f"Financial passage {i % 5} detailing quarterly performance."}
        for i in range(20)
    ]
    # Create distinct embeddings
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(20, 16)).astype(np.float32)

    result = pipeline.run(records, embeddings=embeddings)

    assert "retained" in result
    assert "stats" in result
    assert result["stats"]["initial_candidates"] == 20
    assert len(result["retained"]) <= 20

    # Test 80/10/10 partitioning
    train, val, test = pipeline.partition_train_val_test(result["retained"])
    assert len(train) + len(val) + len(test) == len(result["retained"])
    assert len(train) > 0
