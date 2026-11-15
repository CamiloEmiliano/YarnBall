"""Iterative Predicate Sufficiency & Calibration Batch Evaluator (ADR-00031).

Executes strict live-gate calibration over extracted batches without GPU or clustering
overhead, aggregating quarantine signatures by `(predicate, source_type, target_type)`
and verifying holdout non-noise rejection rate convergence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from ontology.validation import validate_graph_payload


def evaluate_calibration_batch(
    samples: Iterable[Dict[str, Any]],
    *,
    rejection_queue: Any,
    min_support: int = 3,
    target_max_rate: float = 0.02,
    since_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Steps 1-3 & Step 8 of ADR-00031 across a batch of extracted document payloads.

    Each item in `samples` expects keys:
      - `source_id`: document or chunk identifier
      - `source_text`: optional original prose snippet
      - `payload`: extracted graph dict with `nodes` and `edges`
    """
    window_start = since_checkpoint or datetime.now(timezone.utc).isoformat()
    total_candidates = 0
    accepted_edges = 0

    for item in samples:
        payload = item.get("payload") or {"nodes": [], "edges": []}
        total_candidates += len(payload.get("edges") or [])
        gated = validate_graph_payload(
            payload,
            source_id=item.get("source_id"),
            source_text=item.get("source_text"),
            rejection_queue=rejection_queue,
            strict=True,
        )
        accepted_edges += len(gated.get("edges") or [])

    summary: Dict[str, Any] = rejection_queue.evaluate_predicate_convergence(
        total_candidate_triples=total_candidates,
        min_support=min_support,
        target_max_rate=target_max_rate,
        since_checkpoint=window_start,
    )
    summary["accepted_edges"] = accepted_edges
    return summary
