"""
Dataset Cartography Engine for YarnBall SFT.

Directly implements and adapts:
- Swayamdipta et al. (EMNLP 2020): "Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics"
- Toneva et al. (ICLR 2019): "An Empirical Study of Example Forgetting during Deep Neural Network Learning"
- Reference repo: allenai/cartography (cartography/selection/train_dy_filtering.py)

Consumes multi-epoch training dynamics logs and partitions datasets into:
- Easy-to-Learn (high confidence, low variability): prune 80% redundant boilerplates
- Ambiguous (high variability): retain 100% boundary exemplars (drives generalization)
- Hard-to-Learn (low confidence, low variability): quarantine 100% for CIK/syntax audit
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

try:
    from curation.contracts import (
        ActiveLearningPartition,
        CartographyCoordinate,
        CartographyRegion,
    )
except ImportError:
    from graphrag_finance.curation.contracts import (
        ActiveLearningPartition,
        CartographyCoordinate,
        CartographyRegion,
    )


class DatasetCartographer:
    """
    Ripped and refactored from AllenAI's cartography/selection/train_dy_filtering.py.
    Provides typed, self-documenting data map generation and active learning filtering
    tailored for autoregressive financial SFT.
    """

    def __init__(
        self,
        conf_thresh_high: float = 0.70,
        conf_thresh_low: float = 0.30,
        var_thresh: float = 0.15,
    ) -> None:
        """
        Initialize cartography thresholds.

        Args:
            conf_thresh_high: Lower confidence bound for Easy-to-Learn cohort (default: 0.70).
            conf_thresh_low: Upper confidence bound for Hard-to-Learn cohort (default: 0.30).
            var_thresh: Variability boundary separating Ambiguous region from Easy/Hard (default: 0.15).
        """
        self.conf_thresh_high = conf_thresh_high
        self.conf_thresh_low = conf_thresh_low
        self.var_thresh = var_thresh

    def parse_dynamics_log(self, dynamics_file: Union[Path, str]) -> Dict[str, Dict[str, list]]:
        """
        Parses streaming training_dynamics.jsonl into per-guid probability and correctness trends.

        Expected record schema:
            {"epoch": int, "guid": str, "seq_prob": float, "is_correct": bool}
        (Also accepts "sample_id" as an alias for "guid").
        """
        path = Path(dynamics_file)
        if not path.is_file():
            raise FileNotFoundError(f"Training dynamics log not found at: {path}")

        # Store epoch along with prob/correctness to ensure proper temporal ordering
        epoch_records = defaultdict(list)

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as err:
                    raise ValueError(f"Malformed JSON on line {line_no} in {path}: {err}") from err

                guid = str(record.get("guid") or record.get("sample_id") or "")
                if not guid:
                    continue

                epoch = int(record.get("epoch", 0))
                seq_prob = float(record.get("seq_prob", 0.0))
                is_correct = bool(record.get("is_correct", False))

                epoch_records[guid].append((epoch, seq_prob, is_correct))

        trends: Dict[str, Dict[str, list]] = {}
        for guid, entries in epoch_records.items():
            # Sort by epoch ascending
            entries.sort(key=lambda item: item[0])
            trends[guid] = {
                "probs": [item[1] for item in entries],
                "correct": [item[2] for item in entries],
                "epochs": [item[0] for item in entries],
            }

        return trends

    def compute_forgetfulness(self, correctness_trend: List[bool]) -> int:
        """
        Calculates Toneva et al. (2019) forgetfulness metric.
        Counts how many times an example transitioned from correctly predicted in epoch e
        to incorrectly predicted in epoch e+1.

        Returns:
            int: Times forgotten, or 999 if the example was never learned across all epochs.
        """
        if not correctness_trend or not any(correctness_trend):
            return 999  # Never learned (Toneva / AllenAI sentinel)

        learnt = False
        times_forgotten = 0

        for is_correct in correctness_trend:
            if not learnt and is_correct:
                learnt = True
            elif learnt and not is_correct:
                learnt = False
                times_forgotten += 1

        return times_forgotten

    def classify_region(self, confidence: float, variability: float) -> CartographyRegion:
        """
        Assigns a sample to one of the 3 Data Map regions (Swayamdipta et al., 2020).

        1. Ambiguous: variability >= var_thresh (occupies the decision boundary)
        2. Easy-to-Learn: variability < var_thresh and confidence >= conf_thresh_high
        3. Hard-to-Learn: variability < var_thresh and confidence < conf_thresh_high
        """
        if variability >= self.var_thresh:
            return CartographyRegion.AMBIGUOUS
        elif confidence >= self.conf_thresh_high:
            return CartographyRegion.EASY_TO_LEARN
        else:
            return CartographyRegion.HARD_TO_LEARN

    def map_from_trends(self, trends: Dict[str, Dict[str, list]]) -> List[CartographyCoordinate]:
        """
        Calculates Data Map coordinates from a parsed trends dictionary.
        """
        coordinates: List[CartographyCoordinate] = []

        for guid, data in trends.items():
            probs = data.get("probs", [])
            correct = data.get("correct", [])

            if not probs:
                continue

            # Pure NumPy vector statistics matching AllenAI implementation
            conf = float(np.mean(probs))
            var = float(np.std(probs))  # population std (ddof=0) per AllenAI
            acc = float(np.mean(correct)) if correct else 0.0
            forg = self.compute_forgetfulness(correct)
            region = self.classify_region(conf, var)

            coordinates.append(
                CartographyCoordinate(
                    guid=guid,
                    confidence=round(conf, 4),
                    variability=round(var, 4),
                    correctness=round(acc, 4),
                    forgetfulness=forg,
                    region=region,
                )
            )

        return coordinates

    def map_dataset(self, dynamics_file: Union[Path, str]) -> List[CartographyCoordinate]:
        """
        Calculates Data Map coordinates for every training instance in a dynamics log.
        """
        trends = self.parse_dynamics_log(dynamics_file)
        return self.map_from_trends(trends)

    def filter_active_learning(
        self,
        coordinates: List[CartographyCoordinate],
        prune_easy_ratio: float = 0.80,
        random_seed: int = 42,
    ) -> ActiveLearningPartition:
        """
        Executes the Swayamdipta et al. optimal selection recipe:
        1. Ambiguous (100% retained): Core frontier of generalization.
        2. Easy-to-Learn (prune_easy_ratio pruned, default 80%): Keeps centroid exemplars.
        3. Hard-to-Learn (100% quarantined): Isolated for CIK and syntax audit.

        Args:
            coordinates: Evaluated sample coordinates.
            prune_easy_ratio: Fraction of Easy-to-Learn instances to discard (default: 0.80).
            random_seed: Random seed for reproducible subsampling.

        Returns:
            ActiveLearningPartition: Curated partitions and retention telemetry.
        """
        if not (0.0 <= prune_easy_ratio <= 1.0):
            raise ValueError(f"prune_easy_ratio must be between 0.0 and 1.0, got: {prune_easy_ratio}")

        rng = np.random.default_rng(random_seed)

        ambiguous = [c.guid for c in coordinates if c.region == CartographyRegion.AMBIGUOUS]
        hard = [c.guid for c in coordinates if c.region == CartographyRegion.HARD_TO_LEARN]
        easy = [c.guid for c in coordinates if c.region == CartographyRegion.EASY_TO_LEARN]

        # Subsample easy cohort
        if easy:
            num_easy_keep = max(1, int(round(len(easy) * (1.0 - prune_easy_ratio)))) if prune_easy_ratio < 1.0 else 0
            shuffled_easy = list(easy)
            rng.shuffle(shuffled_easy)
            retained_easy = shuffled_easy[:num_easy_keep]
            pruned_easy = shuffled_easy[num_easy_keep:]
        else:
            retained_easy = []
            pruned_easy = []

        retained = ambiguous + retained_easy
        total = len(coordinates)
        retention_rate = round((len(retained) / max(1, total)) * 100, 2)

        return ActiveLearningPartition(
            retained_guids=retained,
            pruned_easy_guids=pruned_easy,
            quarantined_hard_guids=hard,
            total_candidates=total,
            retention_rate_pct=retention_rate,
        )

    def export_coordinates_jsonl(
        self,
        coordinates: List[CartographyCoordinate],
        output_path: Union[Path, str],
    ) -> Path:
        """
        Serializes CartographyCoordinate items to JSONL.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for c in coordinates:
                f.write(c.model_dump_json() + "\n")
        return path
