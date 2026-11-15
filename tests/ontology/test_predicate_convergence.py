"""Unit tests verifying ADR-00031: Iterative predicate sufficiency & quarantine convergence."""

from __future__ import annotations

import json
from pathlib import Path

from ontology import evaluate_calibration_batch
from sft_pipeline.rejection_queue import (
    ErrorStage,
    RejectionRecord,
    RuntimeRejectionQueue,
)


def _write_jsonl_records(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )


def test_bounded_window_checkpoint_skips_historical_files(tmp_path: Path) -> None:
    old_file = tmp_path / "rejections_20261101.jsonl"
    recent_file = tmp_path / "rejections_20261114.jsonl"

    _write_jsonl_records(
        old_file,
        [
            RejectionRecord(
                task_id="LIVE_ONTOLOGY_GATE",
                prompt="source_id=old-doc",
                raw_completion='{"edge": {"type": "LEGACY_REL"}}',
                error_stage=ErrorStage.ONTOLOGY_PRUNING,
                error_message="Old rejected predicate",
                metadata={"predicate": "LEGACY_REL", "source_type": "Company", "target_type": "Company"},
                sample_id="old-doc",
                timestamp="2026-11-01T10:00:00+00:00",
            ).to_dict()
        ],
    )
    _write_jsonl_records(
        recent_file,
        [
            RejectionRecord(
                task_id="LIVE_ONTOLOGY_GATE",
                prompt="source_id=new-doc",
                raw_completion='{"edge": {"type": "CO_DEVELOPS_WITH"}}',
                error_stage=ErrorStage.ONTOLOGY_PRUNING,
                error_message="Recent rejected predicate",
                metadata={"predicate": "CO_DEVELOPS_WITH", "source_type": "Company", "target_type": "Company"},
                sample_id="new-doc-1",
                timestamp="2026-11-14T15:00:00+00:00",
            ).to_dict()
        ],
    )

    queue = RuntimeRejectionQueue(base_dir=tmp_path)
    window_files = queue.list_log_files(since_date="20261110")
    assert [p.name for p in window_files] == ["rejections_20261114.jsonl"]

    records = queue.load_records(since_checkpoint="2026-11-10")
    assert len(records) == 1
    assert records[0].sample_id == "new-doc-1"

    # Sub-day ISO timestamp checkpoint filters records earlier on the same day
    later_records = queue.load_records(since_checkpoint="2026-11-14T16:00:00+00:00")
    assert later_records == []


def test_calibration_batch_aggregates_buckets_and_detects_recurring_gaps(tmp_path: Path) -> None:
    queue = RuntimeRejectionQueue(base_dir=tmp_path)
    samples = []

    # 3 distinct docs with unknown recurring predicate -> CANDIDATE_FOR_REVIEW (df=3 >= K=3)
    for i in range(3):
        samples.append(
            {
                "source_id": f"doc-gap-{i}",
                "source_text": f"Company A co-develops next-gen lithography tools with Company B ({i}).",
                "payload": {
                    "nodes": [
                        {"id": f"Alpha{i}", "type": "Company", "properties": {}},
                        {"id": f"Beta{i}", "type": "Company", "properties": {}},
                    ],
                    "edges": [
                        {"source": f"Alpha{i}", "target": f"Beta{i}", "type": "CO_DEVELOPS_WITH", "properties": {}},
                        # Valid edge alongside rejected one
                        {"source": f"Alpha{i}", "target": f"Beta{i}", "type": "SUPPLIES_TO", "properties": {}},
                    ],
                },
            }
        )

    # 3 distinct docs with domain/range typing mismatch on canonical CEO_OF -> TYPING_ERROR
    for i in range(3):
        samples.append(
            {
                "source_id": f"doc-type-{i}",
                "source_text": f"CorpX{i} leads CorpY{i}.",
                "payload": {
                    "nodes": [
                        {"id": f"CorpX{i}", "type": "Company", "properties": {}},
                        {"id": f"CorpY{i}", "type": "Company", "properties": {}},
                    ],
                    "edges": [
                        {"source": f"CorpX{i}", "target": f"CorpY{i}", "type": "CEO_OF", "properties": {}},
                    ],
                },
            }
        )

    # 1 doc with one-off predicate -> LOW_SUPPORT; 1 doc with known noise -> KNOWN_NOISE
    samples.append(
        {
            "source_id": "doc-single",
            "source_text": "Gamma shares office space with Delta.",
            "payload": {
                "nodes": [
                    {"id": "Gamma Inc.", "type": "Company", "properties": {}},
                    {"id": "Delta Inc.", "type": "Company", "properties": {}},
                ],
                "edges": [
                    {"source": "Gamma Inc.", "target": "Delta Inc.", "type": "SHARES_OFFICE_WITH", "properties": {}},
                    {"source": "Gamma Inc.", "target": "Delta Inc.", "type": "CORRELATES_WITH", "properties": {}},
                ],
            },
        }
    )

    report = evaluate_calibration_batch(
        samples,
        rejection_queue=queue,
        min_support=3,
        target_max_rate=0.02,
    )

    assert report["accepted_edges"] == 3
    assert report["converged"] is False
    assert report["known_noise_count"] == 1
    assert report["typing_error_count"] == 3

    by_pred = {sig["predicate"]: sig for sig in report["signatures"]}
    assert by_pred["CO_DEVELOPS_WITH"]["bucket"] == "CANDIDATE_FOR_REVIEW"
    assert by_pred["CO_DEVELOPS_WITH"]["df"] == 3
    assert len(by_pred["CO_DEVELOPS_WITH"]["snippets"]) == 3
    assert by_pred["CEO_OF"]["bucket"] == "TYPING_ERROR"
    assert by_pred["SHARES_OFFICE_WITH"]["bucket"] == "LOW_SUPPORT"
    assert by_pred["CORRELATES_WITH"]["bucket"] == "KNOWN_NOISE"


def test_holdout_convergence_pass_and_checkpoint_persistence(tmp_path: Path) -> None:
    queue = RuntimeRejectionQueue(base_dir=tmp_path)

    # 50 valid edges across holdout docs, 1 noise rejection -> non_noise_rejection_rate == 0.0 <= 0.02
    holdout_samples = [
        {
            "source_id": f"holdout-{i}",
            "source_text": f"Supplier{i} Inc. supplies chips to Buyer{i} Corp.",
            "payload": {
                "nodes": [
                    {"id": f"Supplier{i} Inc.", "type": "Company", "properties": {}},
                    {"id": f"Buyer{i} Corp.", "type": "Company", "properties": {}},
                ],
                "edges": [
                    {"source": f"Supplier{i} Inc.", "target": f"Buyer{i} Corp.", "type": "SUPPLIES_TO", "properties": {}},
                ],
            },
        }
        for i in range(50)
    ]
    holdout_samples[0]["payload"]["edges"].append(
        {"source": "Supplier0 Inc.", "target": "Buyer0 Corp.", "type": "CORRELATES_WITH", "properties": {}}
    )

    summary = evaluate_calibration_batch(
        holdout_samples,
        rejection_queue=queue,
        min_support=3,
        target_max_rate=0.02,
    )
    assert summary["accepted_edges"] == 50
    assert summary["known_noise_count"] == 1
    assert summary["non_noise_rejections"] == 0
    assert summary["non_noise_rejection_rate"] == 0.0
    assert summary["recurring_unknown_gaps"] == []
    assert summary["converged"] is True

    ckpt_path = queue.save_checkpoint("v1", metrics=summary)
    assert ckpt_path.exists()
    loaded = queue.load_checkpoint()
    assert loaded is not None
    assert loaded["schema_version"] == "v1"
    assert loaded["metrics"]["converged"] is True
