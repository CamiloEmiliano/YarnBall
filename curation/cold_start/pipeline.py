"""
Unified Cold-Start Curation Pipeline.

Chains together:
1. MinHash LSH Deduplication (Broder 1997, Lee et al. ACL 2022)
2. Semantic Deduplication (SemDeDup, Abbas et al. 2023)
3. Facility Location Submodular Core-Set Selection (Mirzasoleiman et al. ICML 2020)
4. Stratified 80/10/10 Train/Val/Test Partitioning
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from curation.cold_start.minhash_lsh import MinHashLSHDeduplicator
from curation.cold_start.semdedup import SemanticDeduplicator
from curation.cold_start.coreset import FacilityLocationCoresetSelector


class ColdStartCurationPipeline:
    """
    End-to-end pipeline orchestrator for cold-start pre-training dataset curation.
    """

    def __init__(
        self,
        minhash_threshold: float = 0.85,
        semdedup_threshold: float = 0.88,
        coreset_retention_ratio: float = 0.80,
    ) -> None:
        self.minhash_dedup = MinHashLSHDeduplicator(jaccard_threshold=minhash_threshold)
        self.semantic_dedup = SemanticDeduplicator(cosine_threshold=semdedup_threshold)
        self.coreset_selector = FacilityLocationCoresetSelector(retention_fraction=coreset_retention_ratio)

    def run(
        self,
        records: List[Dict],
        embeddings: Optional[np.ndarray] = None,
        text_key: str = "prompt",
        id_key: str = "sample_id",
    ) -> Dict[str, Union[List[Dict], dict]]:
        """
        Executes the multi-stage cold-start pipeline.

        Returns:
            Dict containing:
                "retained": List of final curated records
                "pruned_minhash": Near-duplicate copies pruned
                "pruned_semdedup": Semantic duplicates pruned
                "pruned_coreset": Coreset non-selected instances pruned
                "stats": Pipeline telemetry summary
        """
        initial_count = len(records)
        if initial_count == 0:
            return {
                "retained": [],
                "pruned_minhash": [],
                "pruned_semdedup": [],
                "pruned_coreset": [],
                "stats": {"initial": 0, "final": 0},
            }

        # Stage 1: MinHash LSH Deduplication
        after_minhash, pruned_minhash = self.minhash_dedup.deduplicate(
            records, text_key=text_key, id_key=id_key
        )

        pruned_semdedup = []
        after_semdedup = after_minhash

        # Stage 2: Semantic Deduplication (if embeddings are provided)
        if embeddings is not None and len(embeddings) == initial_count:
            # Map remaining indices to embedding rows
            guid_to_row = {
                (rec.get(id_key) or rec.get("guid") or str(idx)): idx
                for idx, rec in enumerate(records)
            }
            active_rows = [
                guid_to_row.get(rec.get(id_key) or rec.get("guid") or "")
                for rec in after_minhash
                if (rec.get(id_key) or rec.get("guid") or "") in guid_to_row
            ]
            if len(active_rows) == len(after_minhash):
                sub_embeddings = embeddings[active_rows]
                after_semdedup, pruned_semdedup = self.semantic_dedup.deduplicate_with_embeddings(
                    after_minhash, sub_embeddings, id_key=id_key, text_key=text_key
                )

        pruned_coreset = []
        final_retained = after_semdedup

        # Stage 3: Submodular Coreset Selection (if embeddings available)
        if embeddings is not None and len(after_semdedup) > 0:
            guid_to_row = {
                (rec.get(id_key) or rec.get("guid") or str(idx)): idx
                for idx, rec in enumerate(records)
            }
            active_rows = [
                guid_to_row.get(rec.get(id_key) or rec.get("guid") or "")
                for rec in after_semdedup
                if (rec.get(id_key) or rec.get("guid") or "") in guid_to_row
            ]
            if len(active_rows) == len(after_semdedup):
                sub_embeddings = embeddings[active_rows]
                final_retained, pruned_coreset = self.coreset_selector.select_coreset(
                    after_semdedup, sub_embeddings
                )

        stats = {
            "initial_candidates": initial_count,
            "pruned_by_minhash": len(pruned_minhash),
            "pruned_by_semdedup": len(pruned_semdedup),
            "pruned_by_coreset": len(pruned_coreset),
            "final_retained": len(final_retained),
            "overall_retention_pct": round((len(final_retained) / max(1, initial_count)) * 100, 2),
        }

        return {
            "retained": final_retained,
            "pruned_minhash": pruned_minhash,
            "pruned_semdedup": pruned_semdedup,
            "pruned_coreset": pruned_coreset,
            "stats": stats,
        }

    def partition_train_val_test(
        self,
        records: List[Dict],
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        random_seed: int = 42,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Partitions curated records into stratified Train (80%), Val (10%), Test (10%).
        """
        rng = np.random.default_rng(random_seed)
        shuffled = list(records)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        train = shuffled[:n_train]
        val = shuffled[n_train : n_train + n_val]
        test = shuffled[n_train + n_val :]

        return train, val, test
