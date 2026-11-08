"""
Phase 5 Diagnostic Adversarial Torture Runner & Architectural Decision Engine.

Executes the 150-sample Out-of-Distribution (OOD) Torture Suite (`TT-01` .. `TT-06`)
against the locally served `yarnball-qwen:7b` (`4-bit Q4_K_M` GGUF in Ollama),
profiles RTX 3070 Laptop GPU VRAM & prefill/decode latency across context lengths,
and emits the quantitative Phase 6 & Phase 7 Architectural Decision Matrix.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("EDGAR_LOCAL_DATA_DIR", "/tmp/.edgar")

from sft_pipeline.evaluation.phase5_torture_evaluators import evaluate_torture_sample
from sft_pipeline.evaluation.phase5_torture_suite_builder import (
    DEFAULT_SUITE_PATH,
    build_phase5_torture_suite,
)
from sft_pipeline.phase4_export_and_serve import (
    SYSTEM_PROMPT,
    apply_runtime_guardrails,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "data" / "evaluation" / "phase5_torture_results.jsonl"
DEFAULT_SCORECARD_PATH = PROJECT_ROOT / "data" / "evaluation" / "phase5_torture_scorecard.json"


def poll_gpu_vram_mib() -> Optional[int]:
    """Query active GPU memory usage in MiB via `nvidia-smi` (graceful fallback if unavailable)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=2.0,
            text=True,
        ).strip()
        first_line = out.splitlines()[0].strip()
        return int(first_line)
    except Exception:
        return None


def query_ollama_torture(
    prompt: str,
    task_type: str,
    model_tag: str = "yarnball-qwen:7b",
    ollama_url: str = "http://127.0.0.1:11434/api/generate",
    max_new_tokens: int = 768,
    num_ctx: int = 8192,
    enforce_think_prefix: bool = True,
) -> Dict[str, Any]:
    """
    Execute a single Phase 5 torture prompt against local Ollama with `num_ctx=8192`
    and return raw + Tier-4 guarded predictions alongside detailed prefill & decode telemetry.
    """
    assistant_prefix = (
        "<think>\n"
        if (enforce_think_prefix and task_type in ("CONTAGION_REASONING", "PORTFOLIO_RECOMMENDATION"))
        else ""
    )
    raw_chatml = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{prompt.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_prefix}"
    )

    payload = {
        "model": model_tag,
        "prompt": raw_chatml,
        "raw": True,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "num_ctx": num_ctx,
            "num_predict": max_new_tokens,
            "stop": ["<|im_end|>", "<|endoftext|>"],
        },
    }

    req = urllib.request.Request(
        ollama_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    wall_elapsed = time.time() - t0

    raw_response = body.get("response", "")
    raw_pred = f"{assistant_prefix}{raw_response}".strip()
    guarded_pred = apply_runtime_guardrails(prompt, raw_pred, task_type)

    prompt_tokens = int(body.get("prompt_eval_count") or max(1, len(raw_chatml.split())))
    prefill_ns = int(body.get("prompt_eval_duration") or 0)
    prefill_ms = round(prefill_ns / 1e6, 2) if prefill_ns > 0 else 0.0

    eval_count = int(body.get("eval_count") or max(1, len(guarded_pred.split())))
    eval_duration_ns = int(body.get("eval_duration") or 0)
    if eval_duration_ns > 0:
        tok_per_sec = eval_count / (eval_duration_ns / 1e9)
    else:
        tok_per_sec = eval_count / wall_elapsed if wall_elapsed > 0 else 0.0

    return {
        "raw_prediction": raw_pred,
        "guarded_prediction": guarded_pred,
        "wall_latency_sec": round(wall_elapsed, 3),
        "prefill_ms": prefill_ms,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": eval_count,
        "tokens_per_sec": round(tok_per_sec, 2),
    }


def _rate(passed: int, total: int) -> float:
    return round((passed / total) * 100.0, 2) if total > 0 else 0.0


def _context_bucket_label(prompt_tokens: int) -> str:
    if prompt_tokens < 500:
        return "SHORT_LT_500_TOK"
    if prompt_tokens < 1500:
        return "MEDIUM_500_TO_1500_TOK"
    return "LONG_1500_PLUS_TOK"


def compute_architectural_decisions(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Map empirical Phase 5 sub-cohort pass rates directly onto Phase 6 deterministic
    guardrail requirements and Phase 7 candidate paradigm triggers (per Section 5 of
    `notes/PHASE5_TORTURE_TEST_REPORT.md`).
    """
    by_sub: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    tt01_cypher_deep: List[Dict[str, Any]] = []
    tt01_cypher_cluttered: List[Dict[str, Any]] = []

    for rec in records:
        by_sub[rec["sub_cohort"]].append(rec)
        if rec["axis_id"] == "TT-01" and rec.get("metadata", {}).get("graph_format") == "RAW_CYPHER":
            hops = int(rec.get("metadata", {}).get("hops", 2))
            edges = int(rec.get("metadata", {}).get("edge_count", 10))
            if hops >= 4:
                tt01_cypher_deep.append(rec)
            if edges >= 25:
                tt01_cypher_cluttered.append(rec)

    def sub_pass_pct(key: str) -> float:
        rows = by_sub.get(key, [])
        return _rate(sum(1 for r in rows if r["functional_pass"]), len(rows))

    def list_pass_pct(rows: List[Dict[str, Any]]) -> float:
        return _rate(sum(1 for r in rows if r["functional_pass"]), len(rows))

    tt01_cluttered_pct = list_pass_pct(tt01_cypher_cluttered)
    tt01_deep_pct = list_pass_pct(tt01_cypher_deep)
    tt01_bfs_verbal_pct = sub_pass_pct("BFS_VERBALIZED_ABLATION")

    tt02_diamond_add_pct = sub_pass_pct("DIAMOND_ADDITIVE_SHOCK")
    tt02_terminated_pct = sub_pass_pct("STALE_TERMINATED_EDGE_TRAP")
    tt02_cycle_pct = sub_pass_pct("DIRECTED_CYCLE_FEEDBACK")

    tt03_rival_tailwind_pct = sub_pass_pct("ASYMMETRIC_RIVAL_TAILWIND")
    tt03_shared_bottleneck_pct = sub_pass_pct("SHARED_FOUNDRY_BOTTLENECK")
    tt03_combined_pct = list_pass_pct(
        by_sub.get("ASYMMETRIC_RIVAL_TAILWIND", []) + by_sub.get("SHARED_FOUNDRY_BOTTLENECK", [])
    )

    tt04_raw_dropout_pct = sub_pass_pct("RAW_EDGE_DROPOUT_BLINDSPOT")
    tt04_gnn_high_pct = sub_pass_pct("GNN_IMPLIED_HIGH_PROB")
    tt04_gnn_low_pct = sub_pass_pct("GNN_IMPLIED_LOW_PROB_CONTROL")
    tt04_gnn_combined_pct = list_pass_pct(
        by_sub.get("GNN_IMPLIED_HIGH_PROB", []) + by_sub.get("GNN_IMPLIED_LOW_PROB_CONTROL", [])
    )

    tt05_buffer_override_pct = sub_pass_pct("FILING_BUFFER_INSULATION_OVERRIDE")
    tt05_escalation_override_pct = sub_pass_pct("FILING_SOLE_SOURCE_ESCALATION_OVERRIDE")
    tt05_text_override_pct = list_pass_pct(
        by_sub.get("FILING_BUFFER_INSULATION_OVERRIDE", [])
        + by_sub.get("FILING_SOLE_SOURCE_ESCALATION_OVERRIDE", [])
    )

    tt06_schema_trap_pct = sub_pass_pct("ADVERSARIAL_SCHEMA_PROPERTY_TRAP")

    # Phase 6 deterministic infrastructure & guardrail triggers
    phase6_actions: List[str] = []
    if tt01_cluttered_pct < 90.0 or tt01_bfs_verbal_pct > tt01_deep_pct:
        phase6_actions.append(
            "HybridRetriever + GroundedSynthesizer: Enforce BFS hop-distance sorting, materiality pre-filtering, "
            "and verbalized arrow formatting (`A --> B [CRITICAL_TIER_1]`)."
        )
    if tt02_terminated_pct < 95.0:
        phase6_actions.append(
            "HybridRetriever: Enforce hard temporal lifecycle filter "
            "`WHERE coalesce(r.status, 'ACTIVE_CURRENT') = 'ACTIVE_CURRENT'` in Memgraph Cypher expansion."
        )
    if tt06_schema_trap_pct < 95.0:
        phase6_actions.append(
            "TextToCQL Guardrail: Enforce deterministic schema property-name AST whitelist rejecting "
            "unknown node/edge attributes before Memgraph execution."
        )

    # Phase 7 paradigm candidate evaluation
    trigger_candidate_a_sparse_cascade = bool(
        (tt01_bfs_verbal_pct < 80.0 and tt01_deep_pct < 80.0) or tt02_diamond_add_pct < 80.0
    )
    trigger_candidate_b_grpo_cot_rl = bool(
        tt03_combined_pct < 80.0 or tt05_text_override_pct < 80.0
    )
    trigger_candidate_c_link_pred_gnn = bool(
        tt04_raw_dropout_pct < 50.0 and tt04_gnn_combined_pct >= 75.0
    )

    return {
        "diagnostic_indicators": {
            "tt01_raw_cypher_cluttered_e25_plus_pct": tt01_cluttered_pct,
            "tt01_raw_cypher_deep_k4_plus_pct": tt01_deep_pct,
            "tt01_bfs_verbalized_k4_e80_pct": tt01_bfs_verbal_pct,
            "tt02_diamond_additive_shock_pct": tt02_diamond_add_pct,
            "tt02_stale_terminated_edge_trap_pct": tt02_terminated_pct,
            "tt02_directed_cycle_feedback_pct": tt02_cycle_pct,
            "tt03_asymmetric_rival_tailwind_pct": tt03_rival_tailwind_pct,
            "tt03_shared_foundry_bottleneck_pct": tt03_shared_bottleneck_pct,
            "tt03_combined_sign_inversion_pct": tt03_combined_pct,
            "tt04_raw_missing_edge_recall_pct": tt04_raw_dropout_pct,
            "tt04_gnn_high_prob_handshake_pct": tt04_gnn_high_pct,
            "tt04_gnn_low_prob_noise_control_pct": tt04_gnn_low_pct,
            "tt04_gnn_combined_handshake_pct": tt04_gnn_combined_pct,
            "tt05_buffer_override_pct": tt05_buffer_override_pct,
            "tt05_sole_source_escalation_override_pct": tt05_escalation_override_pct,
            "tt05_combined_filing_override_pct": tt05_text_override_pct,
            "tt06_adversarial_schema_trap_pct": tt06_schema_trap_pct,
        },
        "phase6_deterministic_guardrail_actions": phase6_actions,
        "phase7_paradigm_triggers": {
            "candidate_a_sparse_threshold_cascade": {
                "recommended": trigger_candidate_a_sparse_cascade,
                "complexity": "O(K * |E|) sparse forward flux (no O(N^3) Leontief inversion)",
                "rationale": (
                    f"TT-01 deep k>=4 pass={tt01_deep_pct}% (BFS verbalized={tt01_bfs_verbal_pct}%), "
                    f"TT-02A parallel diamond additive pass={tt02_diamond_add_pct}%."
                ),
            },
            "candidate_b_grpo_verifiable_cot_rl": {
                "recommended": trigger_candidate_b_grpo_cot_rl,
                "rationale": (
                    f"TT-03 competitive sign-inversion pass={tt03_combined_pct}%, "
                    f"TT-05A filing override vs SFT template pass={tt05_text_override_pct}%."
                ),
            },
            "candidate_c_self_supervised_link_prediction_gnn": {
                "recommended": trigger_candidate_c_link_pred_gnn,
                "rationale": (
                    f"TT-04A unobserved missing-edge recall={tt04_raw_dropout_pct}% vs "
                    f"TT-04B GNN implied-edge epistemic handshake pass={tt04_gnn_combined_pct}%."
                ),
            },
        },
    }


def run_phase5_torture_suite(
    suite_path: Path = DEFAULT_SUITE_PATH,
    results_path: Path = DEFAULT_RESULTS_PATH,
    scorecard_path: Path = DEFAULT_SCORECARD_PATH,
    model_tag: str = "yarnball-qwen:7b",
    rebuild_suite: bool = False,
) -> Dict[str, Any]:
    """Execute all 150 Phase 5 adversarial torture samples and emit granular diagnostics."""
    suite_path = Path(suite_path)
    results_path = Path(results_path)
    scorecard_path = Path(scorecard_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if rebuild_suite or not suite_path.is_file():
        logger.info("Building deterministic 150-sample Phase 5 torture suite -> %s", suite_path)
        suite = build_phase5_torture_suite(output_path=suite_path)
    else:
        with suite_path.open("r", encoding="utf-8") as f:
            suite = [json.loads(line) for line in f if line.strip()]

    logger.info("Loaded %d Phase 5 torture samples across 6 axes.", len(suite))

    # Pin model in Ollama VRAM before timing
    logger.info("Warming up %s in local Ollama VRAM...", model_tag)
    query_ollama_torture("Warmup ping", "TEXT_TO_CYPHER", model_tag=model_tag, max_new_tokens=8)
    baseline_vram_mib = poll_gpu_vram_mib()
    peak_vram_mib = baseline_vram_mib

    records: List[Dict[str, Any]] = []
    axis_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {
            "total": 0,
            "raw_pass": 0,
            "guarded_pass": 0,
            "verdict_pass": 0,
            "causal_trace_pass": 0,
            "epistemic_pass": 0,
            "schema_cik_pass": 0,
        }
    )
    sub_cohort_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "raw_pass": 0, "guarded_pass": 0}
    )
    tt01_grid_matrix: Dict[str, Dict[str, Any]] = {}
    context_buckets: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"prefill_ms": [], "wall_sec": [], "tok_s": [], "vram_mib": []}
    )

    t_run_start = time.time()
    with results_path.open("w", encoding="utf-8") as out_f:
        for idx, sample in enumerate(suite, 1):
            gen = query_ollama_torture(
                prompt=sample["prompt"],
                task_type=sample["task_type"],
                model_tag=model_tag,
            )
            cur_vram = poll_gpu_vram_mib()
            if cur_vram is not None:
                peak_vram_mib = max(peak_vram_mib or 0, cur_vram)

            raw_eval = evaluate_torture_sample(gen["raw_prediction"], sample)
            guarded_eval = evaluate_torture_sample(gen["guarded_prediction"], sample)

            axis = sample["axis_id"]
            sub = sample["sub_cohort"]
            ax = axis_stats[axis]
            ax["total"] += 1
            ax["raw_pass"] += int(raw_eval["functional_pass"])
            ax["guarded_pass"] += int(guarded_eval["functional_pass"])
            ax["verdict_pass"] += int(guarded_eval["verdict_pass"])
            ax["causal_trace_pass"] += int(guarded_eval["causal_trace_pass"])
            ax["epistemic_pass"] += int(guarded_eval["epistemic_pass"])
            ax["schema_cik_pass"] += int(guarded_eval["schema_cik_pass"])

            sc = sub_cohort_stats[sub]
            sc["total"] += 1
            sc["raw_pass"] += int(raw_eval["functional_pass"])
            sc["guarded_pass"] += int(guarded_eval["functional_pass"])

            if axis == "TT-01" and sample.get("metadata", {}).get("graph_format") == "RAW_CYPHER":
                meta = sample["metadata"]
                grid_key = f"k={meta['hops']}|E={meta['edge_count']:03d}|{meta['needle_placement']}"
                tt01_grid_matrix[grid_key] = {
                    "functional_pass": guarded_eval["functional_pass"],
                    "verdict_pass": guarded_eval["verdict_pass"],
                    "causal_trace_pass": guarded_eval["causal_trace_pass"],
                    "failure_summary": guarded_eval["failure_summary"],
                }

            bkt = context_buckets[_context_bucket_label(gen["prompt_tokens"])]
            bkt["prefill_ms"].append(gen["prefill_ms"])
            bkt["wall_sec"].append(gen["wall_latency_sec"])
            bkt["tok_s"].append(gen["tokens_per_sec"])
            if cur_vram is not None:
                bkt["vram_mib"].append(float(cur_vram))

            record = {
                "sample_id": sample["sample_id"],
                "axis_id": axis,
                "sub_cohort": sub,
                "task_type": sample["task_type"],
                "metadata": sample.get("metadata", {}),
                "prompt_tokens": gen["prompt_tokens"],
                "generated_tokens": gen["generated_tokens"],
                "prefill_ms": gen["prefill_ms"],
                "latency_sec": gen["wall_latency_sec"],
                "tokens_per_sec": gen["tokens_per_sec"],
                "gpu_vram_mib": cur_vram,
                "raw_prediction": gen["raw_prediction"],
                "prediction": gen["guarded_prediction"],
                "raw_functional_pass": raw_eval["functional_pass"],
                **guarded_eval,
            }
            records.append(record)
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            if idx % 10 == 0 or idx == len(suite):
                passed_so_far = sum(1 for r in records if r["functional_pass"])
                logger.info(
                    "[%3d/%3d] Latest axis=%s (%s) | Cumulative Guarded Pass: %d/%d (%.1f%%) | %.1f tok/s | VRAM=%s MiB",
                    idx,
                    len(suite),
                    axis,
                    sub,
                    passed_so_far,
                    idx,
                    _rate(passed_so_far, idx),
                    gen["tokens_per_sec"],
                    cur_vram if cur_vram is not None else "N/A",
                )

    total_elapsed = round(time.time() - t_run_start, 2)
    total_samples = len(records)
    total_raw_pass = sum(1 for r in records if r["raw_functional_pass"])
    total_guarded_pass = sum(1 for r in records if r["functional_pass"])
    total_gen_tokens = sum(r["generated_tokens"] for r in records)
    mean_tok_s = round(sum(r["tokens_per_sec"] for r in records) / max(1, total_samples), 2)

    per_axis_summary: Dict[str, Any] = {}
    for axis_id in sorted(axis_stats.keys()):
        st = axis_stats[axis_id]
        t = st["total"]
        per_axis_summary[axis_id] = {
            "samples": t,
            "raw_pass_count": st["raw_pass"],
            "raw_pass_pct": _rate(st["raw_pass"], t),
            "guarded_pass_count": st["guarded_pass"],
            "guarded_pass_pct": _rate(st["guarded_pass"], t),
            "verdict_pass_pct": _rate(st["verdict_pass"], t),
            "causal_trace_pass_pct": _rate(st["causal_trace_pass"], t),
            "epistemic_pass_pct": _rate(st["epistemic_pass"], t),
            "schema_cik_pass_pct": _rate(st["schema_cik_pass"], t),
        }

    per_sub_cohort_summary: Dict[str, Any] = {}
    for sub_id in sorted(sub_cohort_stats.keys()):
        st = sub_cohort_stats[sub_id]
        t = st["total"]
        per_sub_cohort_summary[sub_id] = {
            "samples": t,
            "raw_pass_pct": _rate(st["raw_pass"], t),
            "guarded_pass_count": st["guarded_pass"],
            "guarded_pass_pct": _rate(st["guarded_pass"], t),
        }

    telemetry_by_bucket: Dict[str, Any] = {}
    for bkt_name, vals in sorted(context_buckets.items()):
        n = len(vals["wall_sec"])
        telemetry_by_bucket[bkt_name] = {
            "samples": n,
            "mean_prefill_ms": round(sum(vals["prefill_ms"]) / n, 2) if n else 0.0,
            "mean_wall_latency_sec": round(sum(vals["wall_sec"]) / n, 3) if n else 0.0,
            "mean_decode_tok_per_sec": round(sum(vals["tok_s"]) / n, 2) if n else 0.0,
            "peak_gpu_vram_mib": int(max(vals["vram_mib"])) if vals["vram_mib"] else None,
        }

    decisions = compute_architectural_decisions(records)

    scorecard = {
        "phase": "Phase 5 — Diagnostic Adversarial Torture Test & Boundary Mapping",
        "model_tag": model_tag,
        "quantization": "4-bit Q4_K_M GGUF (Ollama local runtime, num_ctx=8192)",
        "total_samples": total_samples,
        "overall_raw_functional_pass_pct": _rate(total_raw_pass, total_samples),
        "overall_guarded_functional_pass_pct": _rate(total_guarded_pass, total_samples),
        "overall_guarded_pass_count": total_guarded_pass,
        "hardware_telemetry": {
            "total_wall_elapsed_sec": total_elapsed,
            "total_generated_tokens": total_gen_tokens,
            "mean_decode_tokens_per_sec": mean_tok_s,
            "baseline_gpu_vram_mib": baseline_vram_mib,
            "peak_gpu_vram_mib": peak_vram_mib,
            "by_context_length_bucket": telemetry_by_bucket,
        },
        "per_axis_breakdown": per_axis_summary,
        "per_sub_cohort_breakdown": per_sub_cohort_summary,
        "tt01_hop_clutter_grid": tt01_grid_matrix,
        "architectural_decision_matrix": decisions,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    scorecard_path.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
    logger.info(
        "Phase 5 Torture Suite complete in %.1fs | Overall Guarded Pass: %d/%d (%.2f%%) | Scorecard -> %s",
        total_elapsed,
        total_guarded_pass,
        total_samples,
        scorecard["overall_guarded_functional_pass_pct"],
        scorecard_path,
    )
    return scorecard


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5 Diagnostic Adversarial Torture Test.")
    parser.add_argument("--model-tag", default="yarnball-qwen:7b")
    parser.add_argument("--rebuild-suite", action="store_true")
    args = parser.parse_args()
    run_phase5_torture_suite(model_tag=args.model_tag, rebuild_suite=args.rebuild_suite)


if __name__ == "__main__":
    main()
