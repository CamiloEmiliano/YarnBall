"""
Dense Semantic Deduplication Engine (SemDeDup).

Academic Foundations:
- Abbas et al. (2023): "SemDeDup: Data-Efficient Learning at Scale through Semantic Deduplication"
- Song et al. (NeurIPS 2020): "Generalized Semantic Hashing"

Clusters redundant corporate governance and boilerplate disclosures using dense vector
representations and collapses semantic duplicates using cosine similarity >= 0.88.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


class SemanticDeduplicator:
    """
    Semantic Deduplicator using dense representation clustering and cosine thresholding.
    """

    def __init__(
        self,
        cosine_threshold: float = 0.88,
        embedding_dim: int = 768,
    ) -> None:
        """
        Args:
            cosine_threshold: Minimum cosine similarity to treat two passages as semantically identical. Default: 0.88.
            embedding_dim: Expected dimensionality of semantic vectors (e.g., 768 for all-mpnet-base-v2).
        """
        self.cosine_threshold = cosine_threshold
        self.embedding_dim = embedding_dim

    def deduplicate_with_embeddings(
        self,
        records: List[Dict],
        embeddings: np.ndarray,
        id_key: str = "guid",
        text_key: str = "text",
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Deduplicates records using precomputed dense vector embeddings.

        Args:
            records: List of sample dictionaries.
            embeddings: Numpy array of shape [N, D].
            id_key: Identifier key in record dictionaries.
            text_key: Text key in record dictionaries.

        Returns:
            Tuple: (retained_records, duplicate_records)
        """
        n_samples = len(records)
        if n_samples == 0:
            return [], []
        if n_samples != embeddings.shape[0]:
            raise ValueError(f"Mismatch: {n_samples} records but embeddings shape {embeddings.shape}")

        # 1. Normalize embeddings to unit sphere: ||v|| = 1
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed_emb = embeddings / norms

        # 2. Compute pairwise cosine similarity matrix: [N, N]
        sim_matrix = np.dot(normed_emb, normed_emb.T)

        # 3. Graph clustering via Union-Find for pairs with sim >= cosine_threshold
        parent = list(range(n_samples))

        def find(i: int) -> int:
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]

        def union(i: int, j: int) -> None:
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_j] = root_i

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if sim_matrix[i, j] >= self.cosine_threshold:
                    union(i, j)

        # 4. Group into clusters and pick representative centroids
        clusters = defaultdict(list)
        for idx in range(n_samples):
            root = find(idx)
            clusters[root].append(idx)

        retained_indices = set()
        duplicate_indices = set()

        for root, group in clusters.items():
            if len(group) == 1:
                retained_indices.add(group[0])
            else:
                # Compute centroid of cluster
                group_embs = normed_emb[group]
                centroid = np.mean(group_embs, axis=0, keepdims=True)
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm > 0:
                    centroid = centroid / centroid_norm

                # Pick exemplar with highest cosine similarity to cluster centroid
                similarities_to_centroid = np.dot(group_embs, centroid.T).flatten()
                best_within_group = group[int(np.argmax(similarities_to_centroid))]

                retained_indices.add(best_within_group)
                for member in group:
                    if member != best_within_group:
                        duplicate_indices.add(member)

        retained = [records[i] for i in sorted(retained_indices)]
        duplicates = [records[i] for i in sorted(duplicate_indices)]

        return retained, duplicates
