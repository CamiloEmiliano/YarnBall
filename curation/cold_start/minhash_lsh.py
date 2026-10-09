"""
Locality-Sensitive Hashing (MinHash LSH) Deduplication Engine.

Academic Foundations:
- Broder, A. Z. (1997): "On the resemblance and containment of documents"
- Lee et al. (ACL 2022): "Deduplicating Training Data Makes Language Models Better"

Eliminates syndicated news wire copies and duplicate SEC 10-K legal boilerplate passages
using 5-gram shingles, 128 permutation hashes, and Jaccard similarity >= 0.85.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple
import numpy as np


class MinHashLSHDeduplicator:
    """
    MinHash Locality-Sensitive Hashing (LSH) for syntactic near-duplicate elimination.
    """

    MERSENNE_PRIME = (1 << 31) - 1  # 2^31 - 1 for universal hashing

    def __init__(
        self,
        num_perm: int = 128,
        jaccard_threshold: float = 0.85,
        shingle_size: int = 5,
        num_bands: int = 16,
        random_seed: int = 42,
    ) -> None:
        """
        Args:
            num_perm: Number of permutation hashes (MinHash signature length). Default: 128.
            jaccard_threshold: Minimum estimated Jaccard similarity for duplicate pairs. Default: 0.85.
            shingle_size: Size of n-grams (in words) for shingling. Default: 5.
            num_bands: Number of LSH bands for bucket indexing (num_perm must be divisible by num_bands).
            random_seed: Deterministic seed for universal hash coefficients.
        """
        if num_perm % num_bands != 0:
            raise ValueError(f"num_perm ({num_perm}) must be divisible by num_bands ({num_bands})")

        self.num_perm = num_perm
        self.jaccard_threshold = jaccard_threshold
        self.shingle_size = shingle_size
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands

        # Initialize deterministic linear permutation hash coefficients: h_i(x) = (a * x + b) % MERSENNE_PRIME
        rng = np.random.default_rng(random_seed)
        self.hash_a = rng.integers(1, self.MERSENNE_PRIME, size=num_perm, dtype=np.int64)
        self.hash_b = rng.integers(0, self.MERSENNE_PRIME, size=num_perm, dtype=np.int64)

    def tokenize_shingles(self, text: str) -> Set[int]:
        """
        Extracts word n-grams and computes 32-bit integer hashes for each shingle.
        """
        # Normalize whitespace and lowercase
        words = re.findall(r"\w+", text.lower())
        if len(words) < self.shingle_size:
            # Fallback to single token or unigrams if text is very short
            shingle_str = " ".join(words)
            return {int(hashlib.md5(shingle_str.encode("utf-8")).hexdigest()[:8], 16)}

        shingles: Set[int] = set()
        for i in range(len(words) - self.shingle_size + 1):
            ngram = " ".join(words[i : i + self.shingle_size])
            # Hash to 32-bit integer
            h = int(hashlib.md5(ngram.encode("utf-8")).hexdigest()[:8], 16)
            shingles.add(h)
        return shingles

    def compute_minhash_signature(self, shingles: Set[int]) -> np.ndarray:
        """
        Computes the MinHash signature vector of length `num_perm`.
        signature[i] = min_{s in shingles} ((a_i * s + b_i) % MERSENNE_PRIME)
        """
        if not shingles:
            return np.zeros(self.num_perm, dtype=np.int64)

        shingle_arr = np.array(list(shingles), dtype=np.int64).reshape(-1, 1)  # [S, 1]
        a = self.hash_a.reshape(1, -1)  # [1, P]
        b = self.hash_b.reshape(1, -1)  # [1, P]

        # Vectorized hash computation: (S x P)
        hashes = (shingle_arr * a + b) % self.MERSENNE_PRIME
        return np.min(hashes, axis=0)  # [P]

    def estimate_jaccard(self, sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """Estimates Jaccard similarity as fraction of matching MinHash indices."""
        return float(np.mean(sig_a == sig_b))

    def deduplicate(
        self,
        records: List[Dict[str, str]],
        text_key: str = "text",
        id_key: str = "guid",
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Finds and removes near-duplicate documents.

        Args:
            records: List of dictionaries representing documents.
            text_key: Dictionary key containing text string.
            id_key: Dictionary key containing sample ID/GUID.

        Returns:
            Tuple: (retained_records, duplicate_records)
        """
        if not records:
            return [], []

        # 1. Compute signatures
        signatures: Dict[str, np.ndarray] = {}
        shingle_cache: Dict[str, Set[int]] = {}

        for rec in records:
            guid = rec.get(id_key) or rec.get("sample_id") or str(id(rec))
            text = rec.get(text_key, "")
            shingles = self.tokenize_shingles(text)
            shingle_cache[guid] = shingles
            signatures[guid] = self.compute_minhash_signature(shingles)

        # 2. LSH Band Bucketing
        # band_buckets[band_idx][bucket_hash] = list of guids
        band_buckets: List[Dict[int, List[str]]] = [defaultdict(list) for _ in range(self.num_bands)]

        for guid, sig in signatures.items():
            for b in range(self.num_bands):
                start = b * self.rows_per_band
                end = start + self.rows_per_band
                band_slice = tuple(sig[start:end])
                bucket_hash = hash(band_slice)
                band_buckets[b][bucket_hash].append(guid)

        # 3. Candidate Pair Verification
        candidate_pairs: Set[Tuple[str, str]] = set()
        for b in range(self.num_bands):
            for bucket in band_buckets[b].values():
                if len(bucket) > 1:
                    for i in range(len(bucket)):
                        for j in range(i + 1, len(bucket)):
                            pair = (min(bucket[i], bucket[j]), max(bucket[i], bucket[j]))
                            candidate_pairs.add(pair)

        # 4. Form connected duplicate clusters using Union-Find
        parent: Dict[str, str] = {}

        def find(u: str) -> str:
            if parent.get(u, u) != u:
                parent[u] = find(parent[u])
            return parent.get(u, u)

        def union(u: str, v: str) -> None:
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                parent[root_v] = root_u

        for u, v in candidate_pairs:
            sim = self.estimate_jaccard(signatures[u], signatures[v])
            if sim >= self.jaccard_threshold:
                union(u, v)

        # 5. Partition into retained and duplicate records
        # Keep the earliest or longest record per cluster
        guid_to_rec = {
            (rec.get(id_key) or rec.get("sample_id") or str(id(rec))): rec
            for rec in records
        }

        clusters = defaultdict(list)
        for guid in signatures:
            root = find(guid)
            clusters[root].append(guid)

        retained: List[Dict[str, str]] = []
        duplicates: List[Dict[str, str]] = []

        for root, group in clusters.items():
            # Pick the longest text as the cluster exemplar
            best_guid = max(group, key=lambda g: len(guid_to_rec[g].get(text_key, "")))
            retained.append(guid_to_rec[best_guid])

            for g in group:
                if g != best_guid:
                    duplicates.append(guid_to_rec[g])

        return retained, duplicates
