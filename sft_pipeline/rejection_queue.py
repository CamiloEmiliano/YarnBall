"""Runtime Rejection Queue & Continuous Retraining Staging.

Captures, persists, and curates model generation failures, guardrail violations,
and malformed outputs into a non-blocking JSONL event bus:
1. Intercepts schema property violations, ontology prunings, malformed syntax,
   and runaway loops.
2. Persists structured failure records partitioned daily (data/rejection_queue/rejections_YYYYMMDD.jsonl).
3. Curates logged failures into contrastive DPO candidate pairs (Prompt, Chosen, Rejected)
   for continuous quarterly retraining and alignment refinement.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("rejection_queue")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REJECTION_DIR = PROJECT_ROOT / "data" / "rejection_queue"


class ErrorStage:
    """Standardized error codes for guardrail failure classification."""

    SCHEMA_AST_VIOLATION = "SCHEMA_AST_VIOLATION"  # Hallucinated Cypher properties/labels
    ONTOLOGY_PRUNING = "ONTOLOGY_PRUNING"          # Non-schema descriptive commentary predicates
    SYNTAX_MALFORMED = "SYNTAX_MALFORMED"          # Unbalanced brackets or invalid Cypher clauses
    RUNAWAY_LOOP = "RUNAWAY_LOOP"                  # Infinite token generation / n-gram repetition loops
    EPISTEMIC_HEDGING = "EPISTEMIC_HEDGING"        # Uncommitted / contradictory reasoning synthesis
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"          # Input prompts exceeding token ceiling before chunking
    UNKNOWN = "UNKNOWN"


from pydantic import BaseModel, Field


class RejectionRecord(BaseModel):
    """Structured record of a single guardrail failure or rejected generation."""

    task_id: str
    prompt: str
    raw_completion: str
    error_stage: str
    error_message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    sample_id: Optional[str] = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RejectionRecord:
        return cls(
            task_id=data.get("task_id", "UNKNOWN"),
            prompt=data.get("prompt", ""),
            raw_completion=data.get("raw_completion", ""),
            error_stage=data.get("error_stage", ErrorStage.UNKNOWN),
            error_message=data.get("error_message", ""),
            metadata=data.get("metadata", {}),
            sample_id=data.get("sample_id"),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )


class RuntimeRejectionQueue:
    """Thread-safe, non-blocking disk logger and DPO candidate exporter."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir or DEFAULT_REJECTION_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _get_daily_log_path(self, date_str: Optional[str] = None) -> Path:
        """Get path for daily partitioned JSONL file (e.g. rejections_20261108.jsonl)."""
        if not date_str:
            date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        return self.base_dir / f"rejections_{date_str}.jsonl"

    def log_rejection(self, record: RejectionRecord) -> Path:
        """Append a RejectionRecord to the active daily JSONL file thread-safely."""
        target_path = self._get_daily_log_path()
        line = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"

        with self._lock:
            with open(target_path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()

        logger.debug(
            "Logged failure to rejection queue: task=%s stage=%s file=%s",
            record.task_id,
            record.error_stage,
            target_path.name,
        )
        return target_path

    def record_failure(
        self,
        task_id: str,
        prompt: str,
        completion: str,
        error_stage: str,
        error_message: str,
        metadata: Optional[Dict[str, Any]] = None,
        sample_id: Optional[str] = None,
    ) -> Path:
        """Convenience method to construct and log a failure record."""
        record = RejectionRecord(
            task_id=task_id,
            prompt=prompt,
            raw_completion=completion,
            error_stage=error_stage,
            error_message=error_message,
            metadata=metadata or {},
            sample_id=sample_id,
        )
        return self.log_rejection(record)

    def list_log_files(self) -> List[Path]:
        """List all rejection log files in the directory sorted by date."""
        return sorted(self.base_dir.glob("rejections_*.jsonl"))

    def load_records(self, date_str: Optional[str] = None) -> List[RejectionRecord]:
        """Load records from a specific date or across all available log files."""
        records: List[RejectionRecord] = []
        if date_str:
            target_files = [self._get_daily_log_path(date_str)]
        else:
            target_files = self.list_log_files()

        for p in target_files:
            if not p.exists():
                continue
            with open(p, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, start=1):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        data = json.loads(line_str)
                        records.append(RejectionRecord.from_dict(data))
                    except json.JSONDecodeError as exc:
                        logger.warning("Skipping corrupt line %d in %s: %s", line_idx, p.name, exc)

        return records

    def get_summary_stats(self) -> Dict[str, Any]:
        """Compute aggregated statistics across all logged rejections."""
        records = self.load_records()
        total_count = len(records)
        by_stage: Dict[str, int] = {}
        by_task: Dict[str, int] = {}

        for r in records:
            by_stage[r.error_stage] = by_stage.get(r.error_stage, 0) + 1
            by_task[r.task_id] = by_task.get(r.task_id, 0) + 1

        return {
            "total_rejections": total_count,
            "by_error_stage": by_stage,
            "by_task_id": by_task,
            "available_log_files": len(self.list_log_files()),
        }

    def export_dpo_candidates(
        self,
        output_path: Optional[Path] = None,
        min_records: int = 1,
    ) -> List[Dict[str, Any]]:
        """Curate logged rejection samples into contrastive DPO candidate pairs.

        Produces paired preference records with:
        - prompt: The original input prompt
        - rejected: The raw model completion that tripped guardrails
        - error_stage: Category of failure for preference weighting
        - suggested_repair: Automated guidance or refusal target for the chosen response
        """
        records = self.load_records()
        if len(records) < min_records:
            logger.info("Not enough records to export (%d < %d)", len(records), min_records)
            return []

        candidates: List[Dict[str, Any]] = []

        for r in records:
            # Determine canonical chosen response target based on failure category
            suggested_chosen = "(none)"
            if r.error_stage == ErrorStage.SCHEMA_AST_VIOLATION:
                bad_props = r.metadata.get("invalid_properties", [])
                suggested_chosen = f"(none) // REFUSED: properties {bad_props} do not exist in graph schema"
            elif r.error_stage == ErrorStage.SYNTAX_MALFORMED:
                suggested_chosen = "(none) // REFUSED: malformed Cypher syntax"
            elif r.error_stage == ErrorStage.ONTOLOGY_PRUNING:
                suggested_chosen = "(none)"

            candidate = {
                "task": r.task_id,
                "error_stage": r.error_stage,
                "prompt": r.prompt,
                "rejected": r.raw_completion,
                "chosen": suggested_chosen,
                "error_message": r.error_message,
                "timestamp": r.timestamp,
                "metadata": r.metadata,
            }
            candidates.append(candidate)

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                for c in candidates:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
            logger.info("Exported %d DPO preference candidates to %s", len(candidates), out_p)

        return candidates


# Singleton default queue
DEFAULT_REJECTION_QUEUE = RuntimeRejectionQueue()


def log_runtime_rejection(
    task_id: str,
    prompt: str,
    completion: str,
    error_stage: str,
    error_message: str,
    metadata: Optional[Dict[str, Any]] = None,
    sample_id: Optional[str] = None,
) -> Path:
    """Convenience module-level logger."""
    return DEFAULT_REJECTION_QUEUE.record_failure(
        task_id=task_id,
        prompt=prompt,
        completion=completion,
        error_stage=error_stage,
        error_message=error_message,
        metadata=metadata,
        sample_id=sample_id,
    )


def main() -> None:
    """CLI utility for inspecting stats and exporting DPO candidate datasets."""
    parser = argparse.ArgumentParser(description="YarnBall Runtime Rejection Queue CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: stats
    subparsers.add_parser("stats", help="Show summary statistics of logged failures")

    # Subcommand: export-dpo-candidates
    export_parser = subparsers.add_parser("export-dpo", help="Export contrastive DPO candidate pairs")
    export_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(DEFAULT_REJECTION_DIR / "dpo_candidates.jsonl"),
        help="Destination path for curated DPO candidates JSONL",
    )

    args = parser.parse_args()
    queue = RuntimeRejectionQueue()

    if args.command == "stats":
        stats = queue.get_summary_stats()
        print("\n=== YarnBall Rejection Queue Summary ===")
        print(f"Total Rejections Logged: {stats['total_rejections']}")
        print(f"Daily Partition Files:   {stats['available_log_files']}")
        print("\nBreakdown by Error Stage:")
        for stage, count in sorted(stats["by_error_stage"].items(), key=lambda x: -x[1]):
            print(f"  {stage:<25}: {count}")
        print("\nBreakdown by Task ID:")
        for task, count in sorted(stats["by_task_id"].items(), key=lambda x: -x[1]):
            print(f"  {task:<25}: {count}")
        print("=========================================\n")

    elif args.command == "export-dpo":
        candidates = queue.export_dpo_candidates(output_path=Path(args.output))
        print(f"Exported {len(candidates)} DPO candidate pairs to {args.output}")


if __name__ == "__main__":
    main()

