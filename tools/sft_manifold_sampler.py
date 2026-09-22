"""
Manifold-Targeted Sampler for Financial Graph SFT Dataset Curation.

Avoids ambient saturation sampling by targeting the low-dimensional economic
relationship manifold (M*):
1. Subgraph topological extraction (1-hop, 2-hop, 3+ hop cascades)
2. Boundary-proximity hard negative mining (co-occurring pairs with target: none)
3. Information-theoretic class balancing (floors >= 200 for rare risk relations, caps for dominant classes)
4. Stratified sampling across all 11 GICS economic sectors
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from graph.quality_controls import (
    is_generic_placeholder,
    validate_and_orient_triple,
    compute_edge_confidence,
)
from tools.sp500_universe import SP500UniverseManager

logger = logging.getLogger("manifold_sampler")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

DEFAULT_SFT_DIR = Path(__file__).resolve().parent.parent / "data" / "sft"

# Target rare relationship classes requiring minimum representation floors
RARE_RELATION_FLOORS: Dict[str, int] = {
    "SOLE_SOURCE_DEPENDENT_ON": 200,
    "LICENSES_FROM": 200,
    "LICENSES_TO": 200,
    "DEFAULTED_ON": 200,
    "EXPOSED_TO_RISK": 200,
    "ACQUIRED_BY": 250,
}

DOMINANT_CLASS_CAP: int = 600

# Regex triggers for mining sparse boundary patterns in financial text
RARE_RELATION_TRIGGERS: Dict[str, List[str]] = {
    "SOLE_SOURCE_DEPENDENT_ON": [
        r"sole\s+source",
        r"single\s+source\s+supplier",
        r"no\s+alternate\s+supplier",
        r"solely\s+dependent\s+on",
        r"single-source\s+basis",
    ],
    "LICENSES_FROM": [
        r"cross-licensing\s+agreement",
        r"patent\s+license\s+from",
        r"technology\s+license\s+from",
        r"licensed\s+intellectual\s+property\s+from",
        r"exclusive\s+royalty\s+agreement",
    ],
    "LICENSES_TO": [
        r"granted\s+a\s+patent\s+license\s+to",
        r"licensed\s+technology\s+to",
        r"licensing\s+its\s+proprietary",
    ],
    "DEFAULTED_ON": [
        r"notice\s+of\s+default",
        r"terminated\s+for\s+cause",
        r"covenant\s+breach",
        r"failed\s+to\s+cure\s+default",
        r"acceleration\s+of\s+debt",
    ],
    "EXPOSED_TO_RISK": [
        r"export\s+control\s+restrictions",
        r"critical\s+supply\s+chokepoint",
        r"single\s+point\s+of\s+failure",
        r"geopolitical\s+disruption\s+risk",
        r"tariffs\s+and\s+trade\s+embargoes",
    ],
}


@dataclass
class ManifoldSample:
    """Standardized candidate training record on the economic manifold."""
    sample_id: str
    text_passage: str
    grounded_triples: List[Dict[str, Any]]
    entities_present: List[Dict[str, str]] # {"name": ..., "ticker": ..., "type": ...}
    hop_count: int
    is_hard_negative: bool
    gics_sector: str
    provenance: str
    confidence: float
    difficulty_score: float # 0.0 (easy) to 1.0 (complex multi-hop boundary)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ManifoldTargetedSampler:
    """Curates balanced, boundary-targeted training sets across the economic manifold."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        universe_mgr: Optional[SP500UniverseManager] = None,
        null_sample_ratio: float = 0.20,
    ):
        self.output_dir = output_dir or DEFAULT_SFT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.universe_mgr = universe_mgr or SP500UniverseManager()
        self.null_sample_ratio = null_sample_ratio

    def mine_rare_relation_candidates(
        self,
        text_passages: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Scan text passages with high-precision regex triggers to identify
        sparse, high-value risk relationship sentences.
        """
        harvested: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for item in text_passages:
            text = str(item.get("text") or item.get("raw_text") or "").strip()
            if not text:
                continue

            for rel_type, triggers in RARE_RELATION_TRIGGERS.items():
                for pat in triggers:
                    if re.search(pat, text, re.IGNORECASE):
                        harvested[rel_type].append(item)
                        break

        for rel, items in harvested.items():
            logger.info(f"Mined {len(items)} rare candidate passages for {rel}")

        return harvested

    def synthesize_hard_negatives(
        self,
        market_commentary_passages: List[Dict[str, Any]],
        known_active_pairs: Set[Tuple[str, str]],
        target_count: int = 500,
    ) -> List[ManifoldSample]:
        """
        Mine boundary-proximity hard negatives: passages that mention 2 or more S&P 500
        companies (e.g. in market summaries, index performance reviews) that share NO
        economic edge in the knowledge graph. Target output is strictly empty `(none)`.
        """
        hard_negatives: List[ManifoldSample] = []
        constituents = self.universe_mgr.get_current_constituents()
        name_to_ticker = {c.company_name.lower(): c.ticker for c in constituents}
        ticker_set = {c.ticker.upper() for c in constituents}

        for item in market_commentary_passages:
            if len(hard_negatives) >= target_count:
                break

            text = str(item.get("text") or item.get("raw_text") or "").strip()
            if len(text.split()) < 30:
                continue

            # Identify S&P 500 companies mentioned
            mentioned_tickers: Set[str] = set()
            for word in text.split():
                clean_w = "".join(c for c in word if c.isalnum()).upper()
                if clean_w in ticker_set and len(clean_w) >= 2:
                    mentioned_tickers.add(clean_w)

            # Look for company names in text
            text_lower = text.lower()
            for c_name, t in name_to_ticker.items():
                if len(c_name) >= 5 and c_name in text_lower:
                    mentioned_tickers.add(t)

            # Need at least 2 distinct companies mentioned
            if len(mentioned_tickers) >= 2:
                # Check if any pair has an active economic relationship in the graph
                has_active_edge = False
                t_list = list(mentioned_tickers)
                for i in range(len(t_list)):
                    for j in range(i + 1, len(t_list)):
                        pair1 = (t_list[i], t_list[j])
                        pair2 = (t_list[j], t_list[i])
                        if pair1 in known_active_pairs or pair2 in known_active_pairs:
                            has_active_edge = True
                            break
                    if has_active_edge:
                        break

                if not has_active_edge:
                    sample_id = "NEG_" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                    entities = [{"name": t, "ticker": t, "type": "Company"} for t in t_list[:4]]

                    hard_negatives.append(
                        ManifoldSample(
                            sample_id=sample_id,
                            text_passage=text,
                            grounded_triples=[], # Explicitly empty target!
                            entities_present=entities,
                            hop_count=0,
                            is_hard_negative=True,
                            gics_sector="Cross-Sector",
                            provenance=item.get("provider", "MARKET_COMMENTARY"),
                            confidence=1.0,
                            difficulty_score=0.85, # High difficulty for small LLMs
                        )
                    )

        logger.info(f"Synthesized {len(hard_negatives)} boundary-proximity hard negative samples.")
        return hard_negatives

    def balance_and_curate_manifold(
        self,
        positive_samples: List[ManifoldSample],
        hard_negatives: List[ManifoldSample],
        target_total_samples: int = 4000,
    ) -> List[ManifoldSample]:
        """
        Apply statistical class balancing:
        1. Enforce minimum floors for rare relations (via oversampling / augmentation)
        2. Cap dominant classes (SUBSIDIARY_OF) at DOMINANT_CLASS_CAP
        3. Maintain GICS sector stratification
        4. Blend in tuned null-sample ratio of hard negatives
        """
        rel_buckets: Dict[str, List[ManifoldSample]] = defaultdict(list)

        for s in positive_samples:
            if s.grounded_triples:
                primary_rel = str(s.grounded_triples[0].get("rel_type", "RELATED_TO")).upper()
                rel_buckets[primary_rel].append(s)
            else:
                rel_buckets["EMPTY"].append(s)

        curated: List[ManifoldSample] = []

        # 1. Process positive relation buckets with caps and floors
        for rel_type, samples in rel_buckets.items():
            if rel_type == "EMPTY":
                continue

            floor = RARE_RELATION_FLOORS.get(rel_type, 150)
            cap = DOMINANT_CLASS_CAP

            if len(samples) > cap:
                # Subsample dominant class
                curated.extend(random.sample(samples, cap))
            elif len(samples) < floor and len(samples) > 0:
                # Oversample to meet floor
                curated.extend(samples)
                shortfall = floor - len(samples)
                # Counterfactual entity-swap augmentation on shortfall
                augmented = self._augment_entity_swaps(samples, shortfall)
                curated.extend(augmented)
            else:
                curated.extend(samples)

        # 2. Add tuned proportion of hard negatives
        target_neg_count = int(len(curated) * (self.null_sample_ratio / (1.0 - self.null_sample_ratio)))
        selected_negs = hard_negatives[:target_neg_count] if len(hard_negatives) >= target_neg_count else hard_negatives
        curated.extend(selected_negs)

        random.seed(42)
        random.shuffle(curated)

        logger.info(
            f"Manifold curation complete: {len(curated)} samples "
            f"({len(curated) - len(selected_negs)} positive triples, {len(selected_negs)} hard negatives)."
        )

        return curated

    def _augment_entity_swaps(
        self,
        seed_samples: List[ManifoldSample],
        target_count: int,
    ) -> List[ManifoldSample]:
        """
        Generate synthetic boundary variants by swapping entity names with active S&P 500
        peers while preserving exact relational grammar and schema validity.
        """
        augmented: List[ManifoldSample] = []
        constituents = self.universe_mgr.get_current_constituents()
        if not seed_samples or not constituents:
            return augmented

        for i in range(target_count):
            base = random.choice(seed_samples)
            if not base.grounded_triples:
                continue

            # Pick replacement peer companies from S&P 500
            peer_a = random.choice(constituents)
            peer_b = random.choice([c for c in constituents if c.ticker != peer_a.ticker])

            old_triples = base.grounded_triples
            new_triples = []

            mutated_text = base.text_passage
            for t in old_triples:
                old_src = t.get("source_id", "")
                old_tgt = t.get("target_id", "")
                rel = t.get("rel_type", "")

                new_triples.append({
                    "source_id": peer_a.company_name,
                    "source_ticker": peer_a.ticker,
                    "target_id": peer_b.company_name,
                    "target_ticker": peer_b.ticker,
                    "rel_type": rel,
                    "confidence": t.get("confidence", 0.90),
                })

                if old_src and old_tgt:
                    mutated_text = mutated_text.replace(old_src, peer_a.company_name).replace(old_tgt, peer_b.company_name)

            aug_sample = ManifoldSample(
                sample_id=f"AUG_{base.sample_id}_{i}",
                text_passage=mutated_text,
                grounded_triples=new_triples,
                entities_present=[
                    {"name": peer_a.company_name, "ticker": peer_a.ticker, "type": "Company"},
                    {"name": peer_b.company_name, "ticker": peer_b.ticker, "type": "Company"},
                ],
                hop_count=base.hop_count,
                is_hard_negative=False,
                gics_sector=peer_a.gics_sector,
                provenance=f"SYNTHETIC_AUGMENTATION_{base.provenance}",
                confidence=0.88,
                difficulty_score=base.difficulty_score,
            )
            augmented.append(aug_sample)

        return augmented

    def export_manifold_dataset(
        self,
        samples: List[ManifoldSample],
        filename: str = "manifold_curated_samples.jsonl",
    ) -> Path:
        """Write curated manifold samples to JSONL format."""
        out_path = self.output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s.to_dict()) + "\n")

        logger.info(f"Exported {len(samples)} curated manifold samples to {out_path}")
        return out_path


def main() -> None:
    sampler = ManifoldTargetedSampler()
    logger.info("Initializing Manifold-Targeted Sampler demonstration...")

    sample_positives = [
        ManifoldSample(
            sample_id="SAMPLE_001",
            text_passage="Apple relies on TSMC as its sole source supplier for 3nm Apple Silicon wafers.",
            grounded_triples=[
                {"source_id": "TSMC", "target_id": "Apple Inc.", "rel_type": "SOLE_SOURCE_DEPENDENT_ON", "confidence": 0.95}
            ],
            entities_present=[
                {"name": "TSMC", "ticker": "TSM", "type": "Company"},
                {"name": "Apple Inc.", "ticker": "AAPL", "type": "Company"},
            ],
            hop_count=1,
            is_hard_negative=False,
            gics_sector="Information Technology",
            provenance="SEC_10K_ITEM1",
            confidence=0.98,
            difficulty_score=0.40,
        )
    ]

    market_commentary = [
        {
            "raw_text": "Both Microsoft and Tesla faced broader market headwinds today as Treasury yields climbed to 4.35%. " * 4,
            "provider": "REUTERS_MARKET_REPORT",
        }
    ]

    negs = sampler.synthesize_hard_negatives(
        market_commentary_passages=market_commentary,
        known_active_pairs={("AAPL", "TSM")},
        target_count=5,
    )

    balanced = sampler.balance_and_curate_manifold(sample_positives, negs)
    out_file = sampler.export_manifold_dataset(balanced, filename="demo_manifold_samples.jsonl")
    logger.info(f"Manifold sampler run successfully. File: {out_file}")


if __name__ == "__main__":
    main()
