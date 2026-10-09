"""
Data Map Visualizer for YarnBall Active Learning.

Directly implements and adapts:
- Swayamdipta et al. (EMNLP 2020): "Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics"
- Reference repo: allenai/cartography (cartography/selection/train_dy_filtering.py:plot_data_map)

Generates publication-quality diagnostic figures (canonical 4-panel GridSpec)
and interactive Plotly HTML inspection reports with outlier hover audit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

try:
    from curation.contracts import CartographyCoordinate, CartographyRegion
except ImportError:
    from graphrag_finance.curation.contracts import CartographyCoordinate, CartographyRegion


class DataMapVisualizer:
    """
    Ripped and refactored from AllenAI's cartography/selection/train_dy_filtering.py:plot_data_map.
    Generates publication-quality diagnostic figures and interactive HTML inspection reports.
    """

    def __init__(
        self,
        conf_thresh_high: float = 0.70,
        conf_thresh_low: float = 0.30,
        var_thresh: float = 0.15,
    ) -> None:
        self.conf_high = conf_thresh_high
        self.conf_low = conf_thresh_low
        self.var_thresh = var_thresh

    def plot_static_summary(
        self,
        coordinates: List[CartographyCoordinate],
        output_file: Union[Path, str],
        title: str = "YarnBall SFT Data Map (Qwen2.5-7B)",
        max_samples_to_plot: int = 15000,
    ) -> Path:
        """
        Generates AllenAI's canonical 4-panel diagnostic layout:
        - Panel 1 (Main, left 4.2/5.4 width): 2D Scatter plot with decision boundaries.
        - Panel 2 (Top Right): Marginal Confidence Density (KDE + Histogram).
        - Panel 3 (Middle Right): Marginal Variability Density (KDE + Histogram).
        - Panel 4 (Bottom Right): Marginal Correctness Distribution (Histogram).
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not coordinates:
            # Handle empty coordinates gracefully
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "No Cartography Coordinates Provided", ha="center", va="center")
            plt.savefig(output_path, bbox_inches="tight")
            plt.close()
            return output_path

        df = pd.DataFrame([c.model_dump() for c in coordinates])

        # Subsample if dataset exceeds limit to prevent visual overplotting
        if len(df) > max_samples_to_plot:
            df = df.sample(n=max_samples_to_plot, random_state=42)

        # Style configuration matching scientific standards
        sns.set_theme(style="whitegrid", font_scale=1.1)
        fig = plt.figure(figsize=(15, 9), dpi=300)
        gs = fig.add_gridspec(3, 2, width_ratios=[4.2, 1.2], hspace=0.32, wspace=0.22)

        # -----------------------------------------------------------------
        # Panel 1: Main 2D Data Map Scatter Plot
        # -----------------------------------------------------------------
        ax_main = fig.add_subplot(gs[:, 0])
        palette = {
            CartographyRegion.AMBIGUOUS: "#e67e22",      # Vibrant orange: Decision boundary
            CartographyRegion.EASY_TO_LEARN: "#27ae60",  # Forest green: Early converging
            CartographyRegion.HARD_TO_LEARN: "#c0392b",  # Crimson red: Noisy / quarantine
        }

        sns.scatterplot(
            data=df,
            x="variability",
            y="confidence",
            hue="region",
            palette=palette,
            alpha=0.65,
            s=28,
            edgecolor="none",
            ax=ax_main,
        )

        # Draw mathematical decision boundary lines
        ax_main.axvline(x=self.var_thresh, color="#7f8c8d", linestyle="--", linewidth=1.4, alpha=0.85)
        ax_main.axhline(y=self.conf_high, color="#7f8c8d", linestyle="--", linewidth=1.4, alpha=0.85)
        ax_main.axhline(y=self.conf_low, color="#7f8c8d", linestyle="--", linewidth=1.4, alpha=0.85)

        # Region callout annotations with styled bounding boxes
        bbox_style = lambda col: dict(boxstyle="round,pad=0.35", ec=col, lw=1.8, fc="white")
        ax_main.text(
            0.03, 0.94, "EASY-TO-LEARN\n(Prune 80% Boilerplate)", transform=ax_main.transAxes,
            fontsize=10.5, weight="bold", color="#27ae60", va="top", bbox=bbox_style("#27ae60")
        )
        ax_main.text(
            0.78, 0.52, "AMBIGUOUS\n(Retain 100% Boundary)", transform=ax_main.transAxes,
            fontsize=10.5, weight="bold", color="#e67e22", ha="center", bbox=bbox_style("#e67e22")
        )
        ax_main.text(
            0.03, 0.14, "HARD-TO-LEARN\n(Quarantine for Audit)", transform=ax_main.transAxes,
            fontsize=10.5, weight="bold", color="#c0392b", bbox=bbox_style("#c0392b")
        )

        max_var = float(df["variability"].max()) if not df.empty else 0.5
        ax_main.set_xlim(-0.02, max(0.50, max_var + 0.04))
        ax_main.set_ylim(-0.02, 1.04)
        ax_main.set_xlabel("Variability (sigma)", fontsize=13, weight="bold")
        ax_main.set_ylabel("Confidence (mu)", fontsize=13, weight="bold")
        ax_main.set_title(title, fontsize=15, weight="bold", pad=12)
        ax_main.legend(title="Data Map Region", loc="upper right", frameon=True, framealpha=0.92)

        # -----------------------------------------------------------------
        # Panel 2: Marginal Confidence Density (Top Right)
        # -----------------------------------------------------------------
        ax_conf = fig.add_subplot(gs[0, 1])
        sns.histplot(df["confidence"], kde=True, color="#2980b9", ax=ax_conf, bins=20)
        ax_conf.set_title("Confidence Density (mu)", fontsize=11, weight="bold")
        ax_conf.set_xlabel("")
        ax_conf.set_ylabel("Count")

        # -----------------------------------------------------------------
        # Panel 3: Marginal Variability Density (Middle Right)
        # -----------------------------------------------------------------
        ax_var = fig.add_subplot(gs[1, 1])
        sns.histplot(df["variability"], kde=True, color="#8e44ad", ax=ax_var, bins=20)
        ax_var.set_title("Variability Density (sigma)", fontsize=11, weight="bold")
        ax_var.set_xlabel("")
        ax_var.set_ylabel("Count")

        # -----------------------------------------------------------------
        # Panel 4: Marginal Correctness Distribution (Bottom Right)
        # -----------------------------------------------------------------
        ax_corr = fig.add_subplot(gs[2, 1])
        sns.histplot(df["correctness"], color="#16a085", ax=ax_corr, bins=10)
        ax_corr.set_title("Correctness Fraction", fontsize=11, weight="bold")
        ax_corr.set_xlabel("Accuracy Fraction")
        ax_corr.set_ylabel("Count")

        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        return output_path

    def export_interactive_html(
        self,
        coordinates: List[CartographyCoordinate],
        output_file: Union[Path, str],
        metadata_lookup: Optional[Dict[str, dict]] = None,
    ) -> Path:
        """
        Exports an interactive HTML scatter visualization (via Plotly) enabling hover
        inspection of outliers in the Hard-to-Learn quadrant.
        Shows sample_id, task, cik, correctness, forgetfulness, and gold target preview.
        """
        import plotly.express as px

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for c in coordinates:
            row = c.model_dump()
            # Ensure region is serialized as a string value for Plotly discrete color mapping
            row["region"] = c.region.value if hasattr(c.region, "value") else str(c.region)
            if metadata_lookup and c.guid in metadata_lookup:
                meta = metadata_lookup[c.guid]
                row["task"] = meta.get("task", "UNKNOWN")
                row["cik"] = meta.get("cik", "UNKNOWN")
                row["gold_preview"] = str(meta.get("target_completion") or meta.get("completion") or "")[:80]
            else:
                row["task"] = "N/A"
                row["cik"] = "N/A"
                row["gold_preview"] = "N/A"
            rows.append(row)

        df = pd.DataFrame(rows)

        color_map = {
            CartographyRegion.AMBIGUOUS.value: "#e67e22",
            CartographyRegion.EASY_TO_LEARN.value: "#27ae60",
            CartographyRegion.HARD_TO_LEARN.value: "#c0392b",
        }

        fig = px.scatter(
            df,
            x="variability",
            y="confidence",
            color="region",
            color_discrete_map=color_map,
            hover_data=["guid", "task", "cik", "correctness", "forgetfulness", "gold_preview"],
            title="YarnBall Interactive Data Map: Training Dynamics Outlier Audit",
        )
        fig.add_vline(x=self.var_thresh, line_dash="dash", line_color="gray")
        fig.add_hline(y=self.conf_high, line_dash="dash", line_color="gray")
        fig.add_hline(y=self.conf_low, line_dash="dash", line_color="gray")

        fig.write_html(str(output_path))
        return output_path
