"""
Unit tests for Phase 3 Cartography Inflection-Slice DPO Dataset Builder (`sft_pipeline/dpo_builder.py`).
"""

import json
import os
from pathlib import Path

os.environ.setdefault("EDGAR_LOCAL_DATA_DIR", "/tmp/.edgar")

from sft_pipeline.dpo_builder import InflectionSliceDPOBuilder


def test_inflection_slice_dpo_builder_outputs_and_zero_leakage():
    """Verify DPO dataset files, bucket coverage, non-identical pairs, and zero test leakage."""
    builder = InflectionSliceDPOBuilder()
    dpo_dir = builder.dpo_dir
    sft_dir = builder.sft_dir

    train_file = dpo_dir / "yarnball_dpo_train.jsonl"
    val_file = dpo_dir / "yarnball_dpo_val.jsonl"
    report_file = dpo_dir / "inflection_slice_report.json"
    summary_file = dpo_dir / "dpo_dataset_summary.json"

    assert train_file.is_file()
    assert val_file.is_file()
    assert report_file.is_file()
    assert summary_file.is_file()

    with open(report_file, "r", encoding="utf-8") as f:
        report = json.load(f)
    assert report["step_50_slow_inflection_cohort"]["total_guids"] == 353
    assert report["step_100_hardest_boundary_cohort"]["total_guids"] == 44

    # Load held-out test sample_ids
    test_ids = set()
    with open(sft_dir / "yarnball_sft_test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_ids.add(json.loads(line)["sample_id"])

    expected_buckets = {
        "BUCKET_1_TASK_D_MODE_BLEED",
        "BUCKET_1_TASK_D_COUNTERFACTUAL",
        "BUCKET_2_TASK_E_BOUNDARY_HEDGING",
        "BUCKET_3_TASK_AB_DISAMBIG_AND_CIK",
        "BUCKET_4_TASK_C_SCHEMA_CYPHER",
    }

    seen_buckets = set()
    for split_file in (train_file, val_file):
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                assert rec["prompt"].strip()
                assert rec["chosen"].strip()
                assert rec["rejected"].strip()
                assert rec["chosen"].strip() != rec["rejected"].strip()
                assert rec["metadata"]["source_sample_id"] not in test_ids
                seen_buckets.add(rec["metadata"]["dpo_bucket"])

    assert seen_buckets == expected_buckets
