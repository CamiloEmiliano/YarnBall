"""Unit tests for the blinded multi-judge audit harness (parser, pass logic, statistics, packet building)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

os.environ.setdefault("EDGAR_LOCAL_DATA_DIR", "/tmp/.edgar")

from sft_pipeline.evaluation import phase5_judge_audit as ja


def _meta(**kw):
    base = {
        "kind": "CONTAGION", "sub_cohort": "X", "axis_id": "TT-02", "expected_verdict": "DISRUPTIVE_SHOCK",
        "epistemic": None, "mode": "last", "eoi": ["AAA"], "has_path": False,
    }
    base.update(kw)
    return base


def test_parse_answer_line_valid_and_invalid_values():
    oid, f = ja.parse_answer_line("a1b2c3 Q1=DISRUPTED E=AAA:HIT,BRK.B:PROTECTED F=NONE P=NA U=NA C=HIGH")
    assert oid == "a1b2c3"
    assert f["Q1"] == "DISRUPTED" and f["E"] == {"AAA": "HIT", "BRK.B": "PROTECTED"}
    # invalid option values are dropped rather than trusted
    _, bad = ja.parse_answer_line("a1b2c3 Q1=SORT_OF_BAD F=NONE")
    assert "Q1" not in bad and bad["F"] == "NONE"
    assert ja.parse_answer_line("not an answer line") is None


def test_judge_pass_disruptive_rival_and_contained():
    ok = {"Q1": "DISRUPTED", "E": {"AAA": "HIT"}, "F": "NONE", "P": "NA", "U": "NA"}
    assert ja.judge_pass(_meta(), ok) is True
    assert ja.judge_pass(_meta(), {**ok, "Q1": "CONTAINED", "E": {"AAA": "PROTECTED"}}) is False
    assert ja.judge_pass(_meta(), {**ok, "F": "MAJOR"}) is False  # misattribution fails
    assert ja.judge_pass(_meta(), {"Q1": "DISRUPTED"}) is None  # missing fields

    contained = _meta(expected_verdict="CONTAINED_MINIMAL_CONTAGION")
    assert ja.judge_pass(contained, {"Q1": "CONTAINED", "E": {"AAA": "PROTECTED"}, "F": "NONE"}) is True
    assert ja.judge_pass(contained, {"Q1": "CONTAINED", "E": {"AAA": "HIT"}, "F": "NONE"}) is False

    tail = _meta(expected_verdict="COMPETITOR_TAILWIND")
    assert ja.judge_pass(tail, {"Q1": "CONTAINED", "E": {"AAA": "GAINS_SHARE"}, "F": "NONE"}) is True
    assert ja.judge_pass(tail, {"Q1": "DISRUPTED", "E": {"AAA": "HIT"}, "F": "NONE"}) is False

    shared = _meta(expected_verdict="SHARED_BOTTLENECK_SHOCK")
    assert ja.judge_pass(shared, {"Q1": "DISRUPTED", "E": {"AAA": "HIT"}, "F": "NONE"}) is True
    assert ja.judge_pass(shared, {"Q1": "CONTAINED", "E": {"AAA": "GAINS_SHARE"}, "F": "NONE"}) is False


def test_judge_pass_epistemic_modes_modes_and_variant():
    meta = _meta(epistemic="HIGH_PROB_IMPLIED_EDGE")
    flagged = {"Q1": "EXPOSURE_FLAGGED", "E": {"AAA": "AT_RISK_UNCONFIRMED"}, "F": "NONE", "U": "YES"}
    assert ja.judge_pass(meta, flagged) is True
    assert ja.judge_pass(meta, {**flagged, "U": "NO"}) is False  # unqualified implied link fails
    # observed-graph variant: refusing to assert a confirmed shock is acceptable
    contained = {"Q1": "CONTAINED", "E": {"AAA": "PROTECTED"}, "F": "NONE", "U": "NO"}
    assert ja.judge_pass(meta, contained, "strict") is False
    assert ja.judge_pass(meta, contained, "observed_graph") is True
    assert ja.judge_pass(meta, {"Q1": "DISRUPTED", "E": {"AAA": "HIT"}, "F": "NONE", "U": "NO"}, "observed_graph") is False

    cyc = _meta(mode="all", eoi=["AAA", "BBB"])
    assert ja.judge_pass(cyc, {"Q1": "DISRUPTED", "E": {"AAA": "HIT", "BBB": "HIT"}, "F": "NONE"}) is True
    assert ja.judge_pass(cyc, {"Q1": "DISRUPTED", "E": {"AAA": "HIT"}, "F": "NONE"}) is False

    path = _meta(has_path=True)
    assert ja.judge_pass(path, {"Q1": "DISRUPTED", "E": {"AAA": "HIT"}, "F": "NONE", "P": "NO"}) is False


def test_judge_pass_portfolio_and_trap():
    hedge = {"kind": "PORTFOLIO", "expected_verdict": "PROTECTIVE_DOWNSIDE_HEDGE"}
    assert ja.judge_pass(hedge, {"R": "REDUCE_OR_HEDGE", "T": "YES"}) is True
    assert ja.judge_pass(hedge, {"R": "HOLD_OR_REBALANCE", "T": "YES"}) is False
    relief = {"kind": "PORTFOLIO", "expected_verdict": "RELIEF_OVERHANG_CLEARED"}
    assert ja.judge_pass(relief, {"R": "LIQUIDATE_ALL", "T": "YES"}) is False
    assert ja.judge_pass(relief, {"R": "HOLD_OR_REBALANCE", "T": "YES"}) is True
    trap = {"kind": "CYPHER_TRAP", "expected_verdict": None}
    assert ja.judge_pass(trap, {"B": "REFUSED_OR_FLAGGED"}) is True
    assert ja.judge_pass(trap, {"B": "SILENT_DROP"}) is False


def test_statistics():
    lo, hi = ja.wilson(3, 9)
    assert 0.11 < lo < 0.13 and 0.64 < hi < 0.67
    assert ja.wilson(0, 0) == (0.0, 0.0)
    assert ja.fleiss_kappa_binary([[True, True, True], [False, False, False], [True, True, False]]) is not None
    assert abs(ja.fleiss_kappa_binary([[True] * 3, [False] * 3, [True] * 3, [False] * 3]) - 1.0) < 1e-9


def test_controls_are_well_formed():
    ids = [c["id"] for c in ja.CONTROLS]
    assert len(ids) == len(set(ids)) and len(ids) >= 15
    for c in ja.CONTROLS:
        assert c["kind"] in ("CONTAGION", "PORTFOLIO", "CYPHER_TRAP")
        if c["kind"] == "CONTAGION":
            assert set(c["gold"]["E"]) <= {t for t, _ in c["entities"]}
            assert c["gold"]["Q1"] in ja.Q1_OPTIONS
        if c["kind"] == "CYPHER_TRAP":
            assert c["gold"]["B"] in ja.TRAP_OPTIONS


def test_build_audit_blinding_and_structure(tmp_path: Path):
    # Minimal synthetic suite and results to exercise the builder end to end.
    suite, results = [], []
    for i in range(30):
        sid = f"S{i:02d}"
        suite.append({
            "sample_id": sid, "axis_id": "TT-03", "sub_cohort": "ASYMMETRIC_RIVAL_TAILWIND",
            "task_type": "CONTAGION_REASONING", "metadata": {},
            "prompt": "<|contagion_reasoning|>\nScenario: SECRET_SCENARIO\nQuestion: q",
            "expected_assertions": {
                "expected_verdict": "COMPETITOR_TAILWIND", "queried_entity_ticker": f"T{i}",
                "required_bridge_entities": [f"H{i}|Hurt {i}", f"T{i}|Rival {i}"],
            },
        })
        results.append({"sample_id": sid, "raw_prediction": "<think>x</think> summary", "functional_pass": bool(i % 2)})
    sp, rp = tmp_path / "suite.jsonl", tmp_path / "results.jsonl"
    sp.write_text("\n".join(json.dumps(s) for s in suite), encoding="utf-8")
    rp.write_text("\n".join(json.dumps(r) for r in results), encoding="utf-8")

    out = tmp_path / "audit"
    info = ja.build_audit(out_dir=out, suite_path=sp, results_path=rp, seed=1, judges=["A", "B"], n_batches=3, n_duplicates=4)
    assert info["real_items"] == 30 and info["controls"] == len(ja.CONTROLS) and info["duplicates"] == 4

    key = json.loads((out / "key.json").read_text())
    for judge in ("A", "B"):
        assert len(key["judges"][judge]) == info["items_per_judge"]
        text = "".join(p.read_text() for p in sorted((out / "packets").glob(f"judge_{judge}_batch_*.md")))
        # blinding: no expected label, grader verdict, sample id or key in judge-facing text
        for forbidden in ("COMPETITOR_TAILWIND", "functional_pass", "expected_verdict", "expected_assertions", "S00", "dup:", "real:", "ctl:"):
            assert forbidden not in text
        assert text.count("### ITEM ") == info["items_per_judge"]
        # duplicates never share a batch with their original
        by_sample = {}
        for oid, v in key["judges"][judge].items():
            if v["sample_id"]:
                by_sample.setdefault(v["sample_id"], []).append(v["batch"])
        for batches in by_sample.values():
            if len(batches) == 2:
                assert batches[0] != batches[1]
    # judges see different orders and different opaque ids
    ids_a = list(key["judges"]["A"])
    ids_b = list(key["judges"]["B"])
    assert ids_a != ids_b and not set(ids_a) & set(ids_b)


def test_score_audit_end_to_end(tmp_path: Path):
    suite = [{
        "sample_id": "S1", "axis_id": "TT-03", "sub_cohort": "SHARED_FOUNDRY_BOTTLENECK", "task_type": "CONTAGION_REASONING",
        "metadata": {}, "prompt": "Scenario: s\nQuestion: q",
        "expected_assertions": {"expected_verdict": "SHARED_BOTTLENECK_SHOCK", "required_bridge_entities": ["F|Foundry", "R|Rival"]},
    }]
    results = [{"sample_id": "S1", "raw_prediction": "<think>x</think> both hit", "functional_pass": True}]
    sp, rp = tmp_path / "suite.jsonl", tmp_path / "results.jsonl"
    sp.write_text(json.dumps(suite[0]), encoding="utf-8")
    rp.write_text(json.dumps(results[0]), encoding="utf-8")
    out = tmp_path / "audit"
    ja.build_audit(out_dir=out, suite_path=sp, results_path=rp, seed=2, judges=["A", "B", "C"], n_batches=2, n_duplicates=1)
    key = json.loads((out / "key.json").read_text())
    (out / "answers").mkdir()
    controls = {c["id"]: c for c in ja.CONTROLS}
    for judge, jkey in key["judges"].items():
        lines = []
        for oid, info in jkey.items():
            if info["control_id"]:
                g = controls[info["control_id"]]["gold"]
                parts = [oid]
                if "Q1" in g:
                    parts += [f"Q1={g['Q1']}", "E=" + ",".join(f"{t}:{s}" for t, s in g["E"].items()), f"F={g.get('F', 'NONE')}", f"P=NA U={g.get('U', 'NA')}"]
                if "R" in g:
                    parts += [f"R={g['R']}", f"T={g['T']}"]
                if "B" in g:
                    parts += [f"B={g['B']}"]
                lines.append(" ".join(parts))
            else:
                # all judges agree the rival is hit, so the sample passes
                lines.append(f"{oid} Q1=DISRUPTED E=R:HIT F=NONE P=NA U=NA C=HIGH")
        (out / "answers" / f"judge_{judge}_batch_1.txt").write_text("\n".join(lines), encoding="utf-8")
    summary = ja.score_audit(out_dir=out, suite_path=sp, results_path=rp)
    assert all(s["gold_field_accuracy"] == 1.0 for s in summary["judges"].values())
    c = summary["cohorts"]["SHARED_FOUNDRY_BOTTLENECK"]
    assert c["consensus_pass"] == 1 and c["tp"] == 1 and c["unanimous"] == 1
    assert summary["human_review_queue"] == []
    assert re.search(r"SHARED_FOUNDRY_BOTTLENECK", ja.render_markdown(summary))
