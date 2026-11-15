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
import re
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, cast

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
    EXTRACTION_PARSE_ERROR = "EXTRACTION_PARSE_ERROR"  # Unparseable or schema-invalid JSON/OpenCypher extraction
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"            # Ollama connection, timeout, or transport failure
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


_CYPHER_PRED_RE = re.compile(r"-\[:([A-Za-z0-9_]+)")
_LOG_DATE_RE = re.compile(r"^rejections_(\d{8})\.jsonl$")


def _normalize_checkpoint_ymd(since_checkpoint: Optional[str]) -> Optional[str]:
    """Extract YYYYMMDD date prefix from a YYYYMMDD, YYYY-MM-DD, or ISO timestamp cursor."""
    if not since_checkpoint:
        return None
    digits = "".join(ch for ch in str(since_checkpoint)[:10] if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else None


def _extract_rejection_signature(
    record: RejectionRecord, fallback_doc_key: str
) -> Tuple[str, str, str, str, str, str, str]:
    """Extract normalized `(predicate, canonical_predicate, src_type, tgt_type, reason, doc_key, snippet)`."""
    meta = record.metadata or {}
    raw_edge = meta.get("edge")
    edge: Dict[str, Any] = raw_edge if isinstance(raw_edge, dict) else {}
    raw_pred = str(meta.get("predicate") or edge.get("type") or "").strip().upper()
    if not raw_pred:
        m = _CYPHER_PRED_RE.search(record.raw_completion or "")
        if m:
            raw_pred = m.group(1).strip().upper()
    pred = raw_pred or "UNKNOWN_PREDICATE"
    canonical_pred = str(meta.get("canonical_predicate") or pred).strip().upper()
    src_type = str(meta.get("source_type") or "Entity").strip() or "Entity"
    tgt_type = str(meta.get("target_type") or "Entity").strip() or "Entity"
    reason = str(meta.get("reason") or record.error_stage or "").strip()
    doc_key = str(record.sample_id or meta.get("source_id") or fallback_doc_key)
    snippet = str(meta.get("snippet") or record.prompt or "")[:300]
    return pred, canonical_pred, src_type, tgt_type, reason, doc_key, snippet


def _classify_predicate_bucket(
    predicate: str,
    canonical_predicate: str,
    reason: str,
    df: int,
    min_support: int,
) -> str:
    """Assign ADR-00031 triage bucket for an aggregated ONTOLOGY_PRUNING signature."""
    from ontology.guardrails import NON_SCHEMA_NOISE_PREDICATES
    from ontology.schema import ONTOLOGY_SCHEMA, PREDICATE_ALIASES

    if reason in ("GENERIC_PLACEHOLDER_NODE", "GENERIC_PLACEHOLDER_EDGE"):
        return "PLACEHOLDER_NOISE"
    if predicate in NON_SCHEMA_NOISE_PREDICATES:
        return "KNOWN_NOISE"
    if predicate in ONTOLOGY_SCHEMA or canonical_predicate in ONTOLOGY_SCHEMA:
        return "TYPING_ERROR"
    if predicate in PREDICATE_ALIASES:
        return "ALIASED"
    if df < min_support:
        return "LOW_SUPPORT"
    return "CANDIDATE_FOR_REVIEW"


class RuntimeRejectionQueue:
    """Thread-safe, non-blocking disk logger and DPO candidate exporter."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir or DEFAULT_REJECTION_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def checkpoint_file_path(self) -> Path:
        """Path to persisted ADR-00031 predicate convergence checkpoint cursor."""
        return self.base_dir / "predicate_convergence_checkpoint.json"

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

    def list_log_files(self, since_date: Optional[str] = None) -> List[Path]:
        """List rejection log files sorted chronologically, pruned by optional YYYYMMDD cursor."""
        files = sorted(self.base_dir.glob("rejections_*.jsonl"))
        since_ymd = _normalize_checkpoint_ymd(since_date)
        if not since_ymd:
            return files

        filtered: List[Path] = []
        for p in files:
            m = _LOG_DATE_RE.match(p.name)
            if not m or m.group(1) >= since_ymd:
                filtered.append(p)
        return filtered

    def load_records(
        self,
        date_str: Optional[str] = None,
        *,
        since_checkpoint: Optional[str] = None,
    ) -> List[RejectionRecord]:
        """Load records from a specific date or within a bounded `since_checkpoint` window."""
        records: List[RejectionRecord] = []
        if date_str:
            target_files = [self._get_daily_log_path(date_str)]
        else:
            target_files = self.list_log_files(since_date=since_checkpoint)

        has_subday_cutoff = bool(since_checkpoint and "T" in str(since_checkpoint))

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
                        rec = RejectionRecord.from_dict(data)
                        if has_subday_cutoff and rec.timestamp and rec.timestamp < str(since_checkpoint):
                            continue
                        records.append(rec)
                    except json.JSONDecodeError as exc:
                        logger.warning("Skipping corrupt line %d in %s: %s", line_idx, p.name, exc)

        return records

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load the latest predicate convergence checkpoint metadata if present."""
        if not self.checkpoint_file_path.exists():
            return None
        try:
            return cast(Dict[str, Any], json.loads(self.checkpoint_file_path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("Failed to load convergence checkpoint %s: %s", self.checkpoint_file_path, exc)
            return None

    def save_checkpoint(
        self,
        schema_version: str,
        *,
        since_checkpoint: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Persist a new schema release checkpoint advancing the `since_checkpoint` scan cursor."""
        cursor = since_checkpoint or datetime.now(timezone.utc).isoformat()
        payload: Dict[str, Any] = {
            "schema_version": schema_version,
            "since_checkpoint": cursor,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics or {},
        }
        self.checkpoint_file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.checkpoint_file_path

    def aggregate_predicate_rejections(
        self,
        *,
        min_support: int = 3,
        since_checkpoint: Optional[str] = None,
        max_samples: int = 5,
    ) -> List[Dict[str, Any]]:
        """Group ONTOLOGY_PRUNING rejections by `(predicate, source_type, target_type)` sorted by `df`."""
        records = self.load_records(since_checkpoint=since_checkpoint)
        groups: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        for idx, rec in enumerate(records):
            if rec.error_stage != ErrorStage.ONTOLOGY_PRUNING:
                continue
            pred, canonical_pred, src_type, tgt_type, reason, doc_key, snippet = _extract_rejection_signature(
                rec, fallback_doc_key=f"__row_{idx}"
            )
            key = (pred, src_type, tgt_type)
            entry = groups.get(key)
            if entry is None:
                entry = {
                    "predicate": pred,
                    "canonical_predicate": canonical_pred,
                    "source_type": src_type,
                    "target_type": tgt_type,
                    "reason": reason,
                    "count": 0,
                    "doc_ids": set(),
                    "sample_ids": [],
                    "snippets": [],
                }
                groups[key] = entry

            entry["count"] += 1
            entry["doc_ids"].add(doc_key)
            if doc_key not in entry["sample_ids"] and len(entry["sample_ids"]) < max_samples:
                entry["sample_ids"].append(doc_key)
            if snippet and snippet not in entry["snippets"] and len(entry["snippets"]) < max_samples:
                entry["snippets"].append(snippet)

        aggregated: List[Dict[str, Any]] = []
        for entry in groups.values():
            df = len(entry["doc_ids"])
            bucket = _classify_predicate_bucket(
                entry["predicate"],
                entry["canonical_predicate"],
                entry["reason"],
                df,
                min_support,
            )
            aggregated.append(
                {
                    "predicate": entry["predicate"],
                    "canonical_predicate": entry["canonical_predicate"],
                    "source_type": entry["source_type"],
                    "target_type": entry["target_type"],
                    "bucket": bucket,
                    "df": df,
                    "count": entry["count"],
                    "sample_ids": entry["sample_ids"],
                    "snippets": entry["snippets"],
                }
            )

        aggregated.sort(
            key=lambda item: (-item["df"], -item["count"], item["predicate"], item["source_type"], item["target_type"])
        )
        return aggregated

    def evaluate_predicate_convergence(
        self,
        *,
        total_candidate_triples: int = 0,
        min_support: int = 3,
        target_max_rate: float = 0.02,
        since_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute ADR-00031 convergence telemetry against holdout/batch thresholds."""
        aggregated = self.aggregate_predicate_rejections(
            min_support=min_support,
            since_checkpoint=since_checkpoint,
        )
        total_pruning = sum(item["count"] for item in aggregated)
        known_noise_count = sum(
            item["count"] for item in aggregated if item["bucket"] in ("KNOWN_NOISE", "PLACEHOLDER_NOISE")
        )
        typing_error_count = sum(item["count"] for item in aggregated if item["bucket"] == "TYPING_ERROR")
        non_noise_rejections = total_pruning - known_noise_count
        recurring_gaps = [item for item in aggregated if item["bucket"] == "CANDIDATE_FOR_REVIEW"]

        rate: Optional[float] = None
        rate_ok = True
        if total_candidate_triples > 0:
            rate = non_noise_rejections / total_candidate_triples
            rate_ok = rate <= target_max_rate

        return {
            "since_checkpoint": since_checkpoint,
            "total_candidate_triples": total_candidate_triples,
            "total_pruning_rejections": total_pruning,
            "known_noise_count": known_noise_count,
            "typing_error_count": typing_error_count,
            "non_noise_rejections": non_noise_rejections,
            "non_noise_rejection_rate": round(rate, 6) if rate is not None else None,
            "target_max_rate": target_max_rate,
            "min_support": min_support,
            "recurring_unknown_gaps": recurring_gaps,
            "converged": (len(recurring_gaps) == 0) and rate_ok,
            "signatures": aggregated,
        }

    def get_summary_stats(self, *, since_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """Compute aggregated statistics across logged rejections in the active window."""
        records = self.load_records(since_checkpoint=since_checkpoint)
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
            "available_log_files": len(self.list_log_files(since_date=since_checkpoint)),
            "since_checkpoint": since_checkpoint,
        }

    def export_dpo_candidates(
        self,
        output_path: Optional[Path] = None,
        min_records: int = 1,
        since_checkpoint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Curate logged rejection samples into contrastive DPO candidate pairs.

        Produces paired preference records with:
        - prompt: The original input prompt
        - rejected: The raw model completion that tripped guardrails
        - error_stage: Category of failure for preference weighting
        - suggested_repair: Automated guidance or refusal target for the chosen response
        """
        records = self.load_records(since_checkpoint=since_checkpoint)
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


def _resolve_since_cursor(queue: RuntimeRejectionQueue, since_arg: Optional[str]) -> Optional[str]:
    if since_arg == "auto":
        ckpt = queue.load_checkpoint()
        return ckpt.get("since_checkpoint") if ckpt else None
    return since_arg


def main() -> None:
    """CLI utility for inspecting stats, predicate convergence, and exporting DPO candidates."""
    parser = argparse.ArgumentParser(description="YarnBall Runtime Rejection Queue CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: stats
    stats_parser = subparsers.add_parser("stats", help="Show summary statistics of logged failures")
    stats_parser.add_argument("--since-checkpoint", type=str, default=None, help="YYYYMMDD, ISO timestamp, or 'auto'")

    # Subcommand: export-dpo
    export_parser = subparsers.add_parser("export-dpo", help="Export contrastive DPO candidate pairs")
    export_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(DEFAULT_REJECTION_DIR / "dpo_candidates.jsonl"),
        help="Destination path for curated DPO candidates JSONL",
    )
    export_parser.add_argument("--since-checkpoint", type=str, default=None, help="YYYYMMDD, ISO timestamp, or 'auto'")

    # Subcommand: predicate-report (ADR-00031 Step 3)
    pred_parser = subparsers.add_parser(
        "predicate-report",
        help="Group ONTOLOGY_PRUNING failures by (predicate, src_type, tgt_type) and distinct doc frequency",
    )
    pred_parser.add_argument("--min-support", type=int, default=3, help="Minimum distinct documents df(p)")
    pred_parser.add_argument("--since-checkpoint", type=str, default=None, help="YYYYMMDD, ISO timestamp, or 'auto'")
    pred_parser.add_argument("--json", action="store_true", help="Emit JSON output")

    # Subcommand: checkpoint (ADR-00031 Step 8)
    ckpt_parser = subparsers.add_parser(
        "checkpoint",
        help="Evaluate convergence metrics and advance the schema checkpoint cursor",
    )
    ckpt_parser.add_argument("--version", type=str, required=True, help="Release tag (e.g. v1)")
    ckpt_parser.add_argument("--total-candidates", type=int, default=0, help="Candidate triple count in holdout batch")
    ckpt_parser.add_argument("--target-rate", type=float, default=0.02, help="Max allowed non-noise rejection rate")
    ckpt_parser.add_argument("--min-support", type=int, default=3, help="Minimum distinct docs for recurring gap")
    ckpt_parser.add_argument("--since-checkpoint", type=str, default=None, help="YYYYMMDD, ISO timestamp, or 'auto'")
    ckpt_parser.add_argument("--enforce", action="store_true", help="Exit non-zero if convergence checks fail")

    args = parser.parse_args()
    queue = RuntimeRejectionQueue()
    since_cursor = _resolve_since_cursor(queue, getattr(args, "since_checkpoint", None))

    if args.command == "stats":
        stats = queue.get_summary_stats(since_checkpoint=since_cursor)
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
        candidates = queue.export_dpo_candidates(
            output_path=Path(args.output),
            since_checkpoint=since_cursor,
        )
        print(f"Exported {len(candidates)} DPO candidate pairs to {args.output}")

    elif args.command == "predicate-report":
        report = queue.aggregate_predicate_rejections(
            min_support=args.min_support,
            since_checkpoint=since_cursor,
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print("\n=== ADR-00031 Predicate Convergence Report ===")
            for item in report:
                print(
                    f"  {item['predicate']} ({item['source_type']} -> {item['target_type']}) "
                    f"| bucket={item['bucket']} | df={item['df']} | count={item['count']}"
                )
            print("==============================================\n")

    elif args.command == "checkpoint":
        summary = queue.evaluate_predicate_convergence(
            total_candidate_triples=args.total_candidates,
            min_support=args.min_support,
            target_max_rate=args.target_rate,
            since_checkpoint=since_cursor,
        )
        if args.enforce and not summary["converged"]:
            print(json.dumps(summary, indent=2))
            raise SystemExit(1)
        path = queue.save_checkpoint(args.version, metrics=summary)
        print(f"Saved checkpoint {args.version} to {path} (converged={summary['converged']})")


if __name__ == "__main__":
    main()

