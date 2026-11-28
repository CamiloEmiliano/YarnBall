"""Offline CLI for Controlled KG Form-Ablation & Embedder/Reranker Study (ADR-00047).

Reads a persisted Parquet snapshot (`nodes.parquet`, `edges.parquet`, `chunks.parquet`),
an optional `RuntimeRejectionQueue` JSONL directory, and an optional gold chunk-cited
query JSONL file, then emits `ablation_scorecard.json`, `ablation_report.md`, and
reproducible matplotlib comparison plots.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import List, Optional

from knowledge_graph.ablation import (
    load_gold_queries,
    render_ablation_artifacts,
    run_kg_form_ablation_study,
)


def _split_csv(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None or not raw.strip():
        return None
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ADR-00047 Controlled KG Form-Ablation & Embedder/Reranker Scaling Study."
    )
    parser.add_argument(
        "--snapshot-dir",
        type=str,
        required=True,
        help="Path to Parquet snapshot directory containing nodes.parquet, edges.parquet, chunks.parquet.",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=date.today().isoformat(),
        help="Point-in-Time historical evaluation cutoff date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--queries-jsonl",
        type=str,
        default=None,
        help="Optional path to gold chunk-cited query benchmark JSONL file.",
    )
    parser.add_argument(
        "--rejection-dir",
        type=str,
        default=None,
        help="Optional path to RuntimeRejectionQueue daily JSONL directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for scorecard JSON, Markdown report, and PNG plots (defaults under data/eval/adr00047/<ts>).",
    )
    parser.add_argument(
        "--forms",
        type=str,
        default=None,
        help="Optional comma-separated SEC form list (defaults to all forms present in snapshot).",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="layered,flat",
        help="Comma-separated graph isolation modes to compare (default: layered,flat).",
    )
    parser.add_argument(
        "--arms",
        type=str,
        default="E1,E2,E3",
        help="Comma-separated embedder/reranker arms (E1, E2, E2b, E3; default: E1,E2,E3).",
    )
    parser.add_argument(
        "--no-lofo",
        action="store_true",
        help="Skip Leave-One-Form-Out cells and evaluate full snapshots only.",
    )
    parser.add_argument(
        "--batch-gpu",
        action="store_true",
        help="Enable offline GPU batch embedding for document indexing (live query path remains CPU-pinned).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic offline mock embedder and mock cross-encoder reranker (CI/smoke mode).",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=1000,
        help="Number of paired bootstrap resamples for 95% confidence intervals (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for Leiden/Louvain partitioning and bootstrap CIs.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) if args.output_dir else (Path("data") / "eval" / "adr00047" / run_id)

    gold_queries = load_gold_queries(args.queries_jsonl)
    scorecard = run_kg_form_ablation_study(
        snapshot_dir=Path(args.snapshot_dir),
        as_of=args.as_of,
        gold_queries=gold_queries,
        rejection_dir=args.rejection_dir,
        forms=_split_csv(args.forms),
        graph_isolations=_split_csv(args.layers) or ["layered", "flat"],
        arm_ids=_split_csv(args.arms) or ["E1", "E2", "E3"],
        run_lofo=not args.no_lofo,
        allow_mock=bool(args.mock),
        batch_gpu=bool(args.batch_gpu),
        bootstrap_resamples=int(args.bootstrap_resamples),
        seed=int(args.seed),
    )
    artifacts = render_ablation_artifacts(scorecard, out_dir)
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(out_dir),
                "artifacts": artifacts,
                "form_promotion_verdicts": scorecard["form_promotion_verdicts"],
                "arm_promotion_verdicts": scorecard["arm_promotion_verdicts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
