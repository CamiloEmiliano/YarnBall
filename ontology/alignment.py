"""
Ontology Alignment & Entity Resolution Utilities (Identity Axioms).

Academic Foundations:
- Christen, P. (2012): "Data Matching: Concepts and Techniques for Record Linkage, Entity Resolution, and Duplicate Detection"
- W3C OWL 2: Identity Axioms (owl:sameAs) and Canonical Edge Ordering

Provides normalization routines, corporate suffix stripping, string similarity metrics,
and symmetric relationship canonicalization.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

try:
    from .schema import SYMMETRIC_RELATIONSHIPS
except ImportError:
    try:
        from ontology.schema import SYMMETRIC_RELATIONSHIPS
    except ImportError:
        from graphrag_finance.ontology.schema import SYMMETRIC_RELATIONSHIPS

# Common corporate suffixes for name normalization
CORPORATE_SUFFIXES: List[str] = [
    "inc.", "inc", "corp.", "corp", "corporation", "co.", "co", "company",
    "ltd.", "ltd", "limited", "llc", "plc", "nv", "ag", "sa", "gmbh", "b.v.", "bv",
    "group", "holding", "holdings", "technology", "technologies", "tech",
    "system", "systems", "solution", "solutions", "semiconductor", "semiconductors",
    "international", "intl", "enterprises", "enterprise", "industries", "industry",
]

# Canonical set of all known media, news publishers, and syndication platforms
KNOWN_PUBLISHER_NAMES: Set[str] = {
    "cnbc", "bloomberg", "reuters", "associated press", "ap news", "dow jones",
    "pr newswire", "businesswire", "globe newswire", "accesswire", "marketwired",
    "yahoo finance", "barrons", "wall street journal", "wsj", "financial times",
    "ft.com", "forbes", "morningstar", "investopedia", "marketwatch",
    "motley fool", "the motley fool", "fool", "foolcom", "fool.com",
    "seeking alpha", "seekingalpha", "zacks", "zacks investment research",
    "benzinga", "investorplace", "thestreet", "tipranks", "insider monkey",
    "24/7 wall st", "alpha spreading", "fxstreet", "investing.com",
}


def normalize_entity_name(name: Optional[str]) -> str:
    """
    Normalize entity / company name for fuzzy matching and canonical linkage.
    Removes punctuation, lowercases, and strips trailing corporate suffixes.
    """
    if not name or not str(name).strip():
        return ""
    norm = str(name).lower().strip()
    norm = "".join(c for c in norm if c.isalnum() or c.isspace())
    tokens = norm.split()
    filtered = [t for t in tokens if t not in CORPORATE_SUFFIXES]
    return " ".join(filtered) if filtered else norm


def canonicalize_symmetric_edge(src: str, tgt: str, rel_type: str) -> Tuple[str, str]:
    """
    Canonicalizes endpoint ordering for symmetric ontological relationships (R(a, b) == R(b, a)).
    For example: (TSMC, COMPETES_WITH, Intel) and (Intel, COMPETES_WITH, TSMC) both
    canonicalize to (Intel, COMPETES_WITH, TSMC).
    """
    clean_rel = str(rel_type).strip().upper()
    if clean_rel in SYMMETRIC_RELATIONSHIPS:
        return tuple(sorted([src, tgt]))  # type: ignore
    return src, tgt


def jaro_winkler_similarity(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
    """
    Compute Jaro-Winkler similarity between two strings.
    """
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    max_dist = max(len1, len2) // 2 - 1
    if max_dist < 0:
        max_dist = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)

        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    transpositions //= 2
    jaro = (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0

    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * prefix_weight * (1.0 - jaro)
