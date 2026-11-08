"""
Blinded multi-judge LLM audit of the Phase 5 keyword graders.

Design (see notes/PHASE5_TORTURE_TEST_REPORT.md, Part IV and V):
  * Judges are asked to EXTRACT structured facts about each completion (what it concludes, how it
    treats each named entity, whether it contradicts the scenario). They never see the expected label
    or the automated grader's verdict. Pass/fail is then computed here, in code, from the extracted facts.
  * Panel of several independent judges; each gets its own item order, opaque item ids, batch
    partition and option-legend order, so position and id effects do not line up across judges.
  * Gold control items (known answers, including keyword-grader traps) measure each judge's accuracy.
  * Duplicate items (same completion under a new id, different batch) measure each judge's self-consistency.
  * Scoring reports inter-judge agreement (Fleiss kappa), per-cohort judge-consensus pass rates with
    Wilson intervals, grader precision/recall against the consensus, and a human review queue made of
    items where judges disagree or the grader and the consensus disagree.

Usage:
    python sft_pipeline/evaluation/phase5_judge_audit.py build
    python sft_pipeline/evaluation/phase5_judge_audit.py score
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation"
SUITE_PATH = EVAL_DIR / "phase5_torture_suite.jsonl"
RESULTS_PATH = EVAL_DIR / "phase5_torture_results.jsonl"
AUDIT_DIR = EVAL_DIR / "judge_audit"

JUDGES = ["A", "B", "C"]
N_BATCHES = 4
N_DUPLICATES = 12
DEFAULT_SEED = 20261108
GOLD_ACCURACY_FLOOR = 0.85

Q1_OPTIONS = ["DISRUPTED", "CONTAINED", "EXPOSURE_FLAGGED", "MIXED_UNCLEAR"]
STATUS_OPTIONS = ["HIT", "AT_RISK_UNCONFIRMED", "PROTECTED", "GAINS_SHARE", "NOT_ADDRESSED"]
FAITH_OPTIONS = ["NONE", "MINOR", "MAJOR"]
PATH_OPTIONS = ["YES", "PARTIAL", "NO", "NA"]
U_OPTIONS = ["YES", "NO", "NA"]
CONF_OPTIONS = ["HIGH", "LOW"]
REC_OPTIONS = ["REDUCE_OR_HEDGE", "HOLD_OR_REBALANCE", "INCREASE_EXPOSURE", "LIQUIDATE_ALL", "UNCLEAR"]
YESNO_OPTIONS = ["YES", "NO"]
TRAP_OPTIONS = ["REFUSED_OR_FLAGGED", "FAKE_PROPERTY", "SILENT_DROP", "OTHER"]

Q1_DEFS = {
    "DISRUPTED": "concludes the entities will suffer a material operational or financial hit, or that the shock propagates to them (even if magnitudes are uncertain).",
    "CONTAINED": "concludes the entities are not materially affected (buffered, insulated, shock does not reach them, terminated link is irrelevant). 'This is NOT a disruptive shock' is CONTAINED.",
    "EXPOSURE_FLAGGED": "flags possible or likely exposure but explicitly frames it as unconfirmed, implied, potential or conditional, without committing to a material hit.",
    "MIXED_UNCLEAR": "no committed conclusion, contradicts itself in its final synthesis, or does not address the entities.",
}
STATUS_DEFS = {
    "HIT": "said to suffer production, delivery or margin harm.",
    "AT_RISK_UNCONFIRMED": "said to be possibly or potentially exposed, explicitly framed as unconfirmed.",
    "PROTECTED": "said to be unaffected, insulated or buffered, with no market-share gain claimed.",
    "GAINS_SHARE": "said to be unaffected AND to gain market share or competitive advantage.",
    "NOT_ADDRESSED": "not mentioned, or no committed status given.",
}
REC_DEFS = {
    "REDUCE_OR_HEDGE": "trim, underweight, buy puts/collars, protective or defensive hedging.",
    "HOLD_OR_REBALANCE": "maintain or rebalance without a net cut in exposure.",
    "INCREASE_EXPOSURE": "add to or lever up the position.",
    "LIQUIDATE_ALL": "exit the entire position.",
    "UNCLEAR": "no clear direction.",
}
TRAP_DEFS = {
    "REFUSED_OR_FLAGGED": "output is '(none)' or explicitly says the requested attribute is not in the schema.",
    "FAKE_PROPERTY": "the query references a property that is not in the Schema line.",
    "SILENT_DROP": "a valid query using only schema properties that ignores the unanswerable part of the request without saying so.",
    "OTHER": "anything else.",
}

# ---------------------------------------------------------------------------
# Gold controls: known answers, several designed to fool keyword graders.
# ---------------------------------------------------------------------------
CONTROLS: List[Dict[str, Any]] = [
    dict(
        id="C01", kind="CONTAGION", entities=[("GAMM", "Gamma Inc")],
        scenario="Scenario: A port blockade halts shipments at both Alpha Corp (ALPH) and Beta Corp (BETA) for 6 months. Gamma Inc (GAMM) sources 40% of a critical substrate via Alpha and 45% via Beta and needs at least 75% of its substrate supply to stay above breakeven utilization.\nQuestion: Assess the impact on Gamma Inc (GAMM).",
        completion="<think>\n1. Both blockaded vendors feed GAMM: 40% + 45% = 85% of supply is interrupted.\n2. Only 15% of supply remains, far below the 75% needed.\n3. GAMM cannot sustain breakeven utilization for the 6 months.\n</think>\n**Summary**: GAMM faces a severe production shortfall and margin compression. Classification: `DISRUPTIVE_SHOCK`.",
        gold={"Q1": "DISRUPTED", "E": {"GAMM": "HIT"}, "F": "NONE"},
    ),
    dict(
        id="C02", kind="CONTAGION", entities=[("EPSL", "Epsilon Corp")],
        scenario="Scenario: A 3-week maintenance delay affects only secondary vendor Delta Co (DELT), which supplies 20% of the standard connectors used by Epsilon Corp (EPSL). Primary supplier Zeta Ltd (ZETL) supplies 80% at full capacity and EPSL holds 90 days of interchangeable buffer stock.\nQuestion: Does the outage at Delta Co halt production at Epsilon Corp (EPSL)?",
        completion="<think>\n1. DELT is a minor 20% source.\n2. ZETL covers the remaining 80% at full capacity.\n3. EPSL's 90-day buffer exceeds the 3-week outage.\n</think>\n**Summary**: The outage is absorbed. EPSL production is unaffected and the shock does not propagate. Verdict: `CONTAINED_MINIMAL_CONTAGION`.",
        gold={"Q1": "CONTAINED", "E": {"EPSL": "PROTECTED"}, "F": "NONE"},
    ),
    dict(
        id="C03", kind="CONTAGION", entities=[("THET", "Theta Corp")],
        scenario="Scenario: Plant maintenance at Eta Works (ETAW) pauses deliveries for 10 days. Theta Corp (THET) holds 6 months of buffer inventory of ETAW parts.\nQuestion: Is Theta Corp (THET) hit by a disruptive supply shock?",
        completion="<think>\n1. ETAW deliveries pause for 10 days.\n2. THET holds 6 months of buffer stock.\n3. The pause is far shorter than the buffer.\n</think>\n**Summary**: This does not amount to a DISRUPTIVE_SHOCK or any production halt. THET is insulated by its buffer inventory and no downstream impact is expected.",
        gold={"Q1": "CONTAINED", "E": {"THET": "PROTECTED"}, "F": "NONE"},
    ),
    dict(
        id="C04", kind="CONTAGION", entities=[("KAPI", "Kappa Inc")],
        scenario="Scenario: A fire cuts output at Iota Mills (IOTM) by 60% for two quarters. Kappa Inc (KAPI) depends on Iota Mills as its sole source for a critical alloy with no buffer stock.\nQuestion: Assess the impact on Kappa Inc (KAPI).",
        completion="<think>\n1. Initially the impact looks limited because the fire is localized.\n2. However KAPI has no buffer and no alternative source.\n</think>\n**Summary**: KAPI's output will be curtailed for two quarters and margins will compress. Classification: `DISRUPTIVE_SHOCK`.",
        gold={"Q1": "DISRUPTED", "E": {"KAPI": "HIT"}, "F": "NONE"},
    ),
    dict(
        id="C05", kind="CONTAGION", entities=[("MUIN", "Mu Inc")],
        scenario="Scenario: Lambda Corp (LAMB) supplies 30% of Mu Inc's (MUIN) inputs and is facing an unspecified delay.\nQuestion: What is the impact on Mu Inc (MUIN)?",
        completion="<think>\n1. The delay is unspecified.\n2. Impact depends on factors not given.\n</think>\n**Summary**: The effect on MUIN could be material or negligible depending on delay length, inventory and alternatives. Monitor developments before drawing conclusions.",
        gold={"Q1": "MIXED_UNCLEAR", "E": {"MUIN": "NOT_ADDRESSED"}, "F": "NONE"},
    ),
    dict(
        id="C06", kind="CONTAGION", entities=[("XILT", "Xi Ltd")],
        scenario="Scenario: A strike halts output at Nu Corp (NUCO) for 8 weeks. Xi Ltd (XILT) buys all its components from NUCO and holds no inventory.\nQuestion: Assess the impact on Xi Ltd (XILT).",
        completion="<think>\n1. The shock originates at Omicron Inc, whose plant was flooded.\n2. XILT holds 14 months of buffer inventory, which fully covers the disruption.\n</think>\n**Summary**: XILT is fully insulated by its 14-month buffer. No impact expected.",
        gold={"Q1": "CONTAINED", "E": {"XILT": "PROTECTED"}, "F": "MAJOR"},
    ),
    dict(
        id="C07", kind="CONTAGION", entities=[("SIGI", "Sigma Inc")],
        scenario="Scenario: Sole-source supplier Pi Foundry (PIFO) shuts down for 6 months, halting shipments to Rho Corp (RHOC). Rival Sigma Inc (SIGI) has no exposure to Pi Foundry and sources everything from unaffected Tau Fab (TAUF), which has spare capacity.\nQuestion: What is the impact on Sigma Inc (SIGI)?",
        completion="<think>\n1. Pi Foundry's shutdown hurts Rho Corp.\n2. SIGI sources from Tau Fab, which is unaffected and has spare capacity.\n3. Rho's shortfall lets SIGI take customers.\n</think>\n**Summary**: SIGI's supply chain is intact. It is well positioned to capture market share from Rho Corp.",
        gold={"Q1": "CONTAINED", "E": {"SIGI": "GAINS_SHARE"}, "F": "NONE"},
    ),
    dict(
        id="C08", kind="CONTAGION", entities=[("SIGI", "Sigma Inc")],
        scenario="Scenario: Sole-source supplier Pi Foundry (PIFO) shuts down for 6 months, halting shipments to Rho Corp (RHOC). Rival Sigma Inc (SIGI) has no exposure to Pi Foundry and sources everything from unaffected Tau Fab (TAUF), which has spare capacity.\nQuestion: What is the impact on Sigma Inc (SIGI)?",
        completion="<think>\n1. Pi Foundry's shutdown is a large industry shock.\n2. Rival firms are affected through the same supply base.\n</think>\n**Summary**: SIGI will also face component shortages and production delays. `DISRUPTIVE_SHOCK` across the sector.",
        gold={"Q1": "DISRUPTED", "E": {"SIGI": "HIT"}, "F": "MAJOR"},
    ),
    dict(
        id="C09", kind="CONTAGION", entities=[("CHIC", "Chi Corp")], uncertainty=True,
        scenario="Scenario: A 60% halt at Upsilon Corp (UPSC) curtails Phi Inc (PHII). A link predictor flags an implied, unconfirmed supply dependency from Phi Inc to Chi Corp (CHIC) with probability 0.88; no filing confirms it.\nQuestion: Analyze the potential contagion exposure to Chi Corp (CHIC), qualifying any unconfirmed link.",
        completion="<think>\n1. UPSC's halt curtails PHII (confirmed).\n2. The PHII to CHIC link is only an implied structural prior at 0.88.\n</think>\n**Summary**: CHIC may face secondary exposure, but this rests on an unconfirmed, model-inferred link and should be treated as a risk flag, not an established fact.",
        gold={"Q1": "EXPOSURE_FLAGGED", "E": {"CHIC": "AT_RISK_UNCONFIRMED"}, "F": "NONE", "U": "YES"},
    ),
    dict(
        id="C10", kind="CONTAGION", entities=[("CHIC", "Chi Corp")], uncertainty=True,
        scenario="Scenario: A 60% halt at Upsilon Corp (UPSC) curtails Phi Inc (PHII). A link predictor flags an implied, unconfirmed supply dependency from Phi Inc to Chi Corp (CHIC) with probability 0.88; no filing confirms it.\nQuestion: Analyze the potential contagion exposure to Chi Corp (CHIC), qualifying any unconfirmed link.",
        completion="<think>\n1. UPSC halts.\n2. PHII supplies CHIC per SEC filings.\n</think>\n**Summary**: CHIC will certainly suffer a severe component shortage; the supply link is confirmed in SEC filings. `DISRUPTIVE_SHOCK`.",
        gold={"Q1": "DISRUPTED", "E": {"CHIC": "HIT"}, "F": "MAJOR", "U": "NO"},
    ),
    dict(
        id="C11", kind="PORTFOLIO", entities=[],
        scenario="Entity: Psi Corp (PSIC)\nSector: Technology\nGraph Risk Factor: Management announced an upbeat AI partnership; however market telemetry shows a severe selloff (5-day CAR = -12.4%, volatility z-score = -3.6).\nObjective: Formulate an institutional hedging and rebalancing strategy.",
        completion="<think>\n1. The tape (CAR -12.4%, z -3.6) contradicts the upbeat headline.\n2. Distribution risk is elevated.\n</think>\n**Recommendation**: Reduce PSIC exposure by 300 bps and buy protective puts. The selloff outweighs the press release.",
        gold={"R": "REDUCE_OR_HEDGE", "T": "YES"},
    ),
    dict(
        id="C12", kind="PORTFOLIO", entities=[],
        scenario="Entity: Omega Corp (OMGC)\nSector: Health Care\nGraph Risk Factor: A final legacy settlement was announced (bearish headline); the settlement was 60% below consensus reserves, triggering a 5-day relief rally (CAR = +6.1%, z-score = +2.3).\nObjective: Formulate a disciplined positioning strategy that avoids panic liquidation.",
        completion="<think>\n1. The settlement is negative news.\n2. Exposure should be removed.\n</think>\n**Recommendation**: Liquidate the entire OMGC position immediately to avoid further losses.",
        gold={"R": "LIQUIDATE_ALL", "T": "NO"},
    ),
    dict(
        id="C13", kind="CYPHER_TRAP", entities=[],
        scenario="Schema: (:Company {name, ticker, cik})-[:SUPPLIES_TO {polarity, materiality, status, valid_from, valid_to}]->(:Company)\nQuery: List suppliers of Foo Corp (FOOC) sorted by CEO age.\n\nCypher:",
        completion="(none)",
        gold={"B": "REFUSED_OR_FLAGGED"},
    ),
    dict(
        id="C14", kind="CYPHER_TRAP", entities=[],
        scenario="Schema: (:Company {name, ticker, cik})-[:SUPPLIES_TO {polarity, materiality, status, valid_from, valid_to}]->(:Company)\nQuery: List suppliers of Foo Corp (FOOC) sorted by CEO age.\n\nCypher:",
        completion="MATCH (s:Company)-[r:SUPPLIES_TO]->(t:Company {ticker: 'FOOC'})\nRETURN s.name, s.ceo_age ORDER BY s.ceo_age DESC;",
        gold={"B": "FAKE_PROPERTY"},
    ),
    dict(
        id="C15", kind="CYPHER_TRAP", entities=[],
        scenario="Schema: (:Company {name, ticker, cik})-[:SUPPLIES_TO {polarity, materiality, status, valid_from, valid_to}]->(:Company)\nQuery: List suppliers of Foo Corp (FOOC) sorted by CEO age.\n\nCypher:",
        completion="MATCH (s:Company)-[r:SUPPLIES_TO]->(t:Company {ticker: 'FOOC'})\nRETURN s.name, s.ticker;",
        gold={"B": "SILENT_DROP"},
    ),
]


# ---------------------------------------------------------------------------
# Item construction (judge-facing view contains no expected label and no grader verdict)
# ---------------------------------------------------------------------------
def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _split_entity(entry: str) -> Tuple[str, str]:
    ticker, name = entry.split("|", 1)
    return ticker.strip(), name.strip()


def is_auditable(sample: Dict[str, Any]) -> bool:
    return sample["task_type"] in ("CONTAGION_REASONING", "PORTFOLIO_RECOMMENDATION") or (
        sample["sub_cohort"] == "ADVERSARIAL_SCHEMA_PROPERTY_TRAP"
    )


def _eoi_entries(sample: Dict[str, Any]) -> List[str]:
    a = sample.get("expected_assertions", {})
    req = a.get("required_bridge_entities", [])
    sub = sample["sub_cohort"]
    if sub == "ASYMMETRIC_RIVAL_TAILWIND":
        t = a.get("queried_entity_ticker", "")
        return [e for e in req if e.split("|")[0] == t]
    if sub in ("DIRECTED_CYCLE_FEEDBACK", "FILING_BUFFER_INSULATION_OVERRIDE", "FILING_SOLE_SOURCE_ESCALATION_OVERRIDE"):
        return list(req)
    if sub == "STALE_TERMINATED_EDGE_TRAP":
        return list(req[:1])
    return list(req[-1:])


def _scenario_text(sample: Dict[str, Any]) -> str:
    prompt = sample["prompt"]
    lines = [ln for ln in prompt.splitlines() if not ln.startswith("<|")]
    if sample["axis_id"] == "TT-01":
        keep = [ln for ln in lines if ln.startswith("Scenario:") or ln.startswith("Question:")]
        return "\n".join(keep)
    return "\n".join(lines).strip()


def _reference_path(sample: Dict[str, Any]) -> Optional[str]:
    if sample["axis_id"] != "TT-01":
        return None
    m = re.search(r"path from (.+?) \(([A-Z.\-]+)\)", sample["prompt"])
    origin = f"{m.group(2)} ({m.group(1)})" if m else "origin"
    chain = [origin] + [f"{t} ({n})" for t, n in map(_split_entity, sample["expected_assertions"]["required_bridge_entities"])]
    return " -> ".join(chain)


def expected_meta(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Expectation metadata used only by the scorer; never shown to judges."""
    a = sample.get("expected_assertions", {})
    sub = sample["sub_cohort"]
    mode = {"DIRECTED_CYCLE_FEEDBACK": "all", "FILING_SOLE_SOURCE_ESCALATION_OVERRIDE": "any"}.get(sub, "last")
    return {
        "kind": "CYPHER_TRAP" if sub == "ADVERSARIAL_SCHEMA_PROPERTY_TRAP" else (
            "PORTFOLIO" if sample["task_type"] == "PORTFOLIO_RECOMMENDATION" else "CONTAGION"
        ),
        "sub_cohort": sub,
        "axis_id": sample["axis_id"],
        "expected_verdict": a.get("expected_verdict"),
        "epistemic": a.get("epistemic_mode"),
        "mode": mode,
        "eoi": [_split_entity(e)[0] for e in _eoi_entries(sample)],
        "has_path": sample["axis_id"] == "TT-01",
    }


def make_item(sample: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    meta = expected_meta(sample)
    entities = [tuple(_split_entity(e)) for e in _eoi_entries(sample)] if meta["kind"] == "CONTAGION" else []
    return {
        "key": f"real:{sample['sample_id']}",
        "sample_id": sample["sample_id"],
        "kind": meta["kind"],
        "scenario": _scenario_text(sample) if meta["kind"] != "CYPHER_TRAP" else sample["prompt"],
        "entities": entities,
        "reference_path": _reference_path(sample),
        "uncertainty": bool(meta["epistemic"]),
        "completion": result["raw_prediction"].strip(),
        "meta": meta,
    }


def make_control_item(c: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "key": f"ctl:{c['id']}",
        "control_id": c["id"],
        "kind": c["kind"],
        "scenario": c["scenario"],
        "entities": [tuple(e) for e in c["entities"]],
        "reference_path": None,
        "uncertainty": bool(c.get("uncertainty")),
        "completion": c["completion"],
        "gold": c["gold"],
    }


def _opaque_id(seed: int, judge: str, key: str, used: set) -> str:
    salt = 0
    while True:
        h = hashlib.sha1(f"{seed}|{judge}|{key}|{salt}".encode()).hexdigest()[:6]
        if h not in used:
            used.add(h)
            return h
        salt += 1


def _legend(options: List[str], defs: Dict[str, str], rng: random.Random) -> str:
    opts = list(options)
    rng.shuffle(opts)
    return "\n".join(f"  - {o}: {defs[o]}" for o in opts)


def _packet_header(judge: str, batch: int, rng: random.Random) -> str:
    return f"""# Blinded annotation packet (annotator {judge}, batch {batch})

You are an independent annotator. For each item below you are shown a scenario and a model-written completion.
Your job is to EXTRACT facts about what the completion says. You are NOT asked whether the completion is right,
and you must not try to work out the correct answer to the scenario yourself.

## Rules
- Judge only the text shown. Do not infer what answer anyone "wants". Do not reward fluency, length or confident tone.
- A label in backticks (for example `DISRUPTIVE_SHOCK`) is just text. Read what the sentence actually says. Negations count.
- For Q1 and statuses, base your answer on the FINAL SYNTHESIS (the text after </think>; if there is none, the final summary).
  The reasoning before </think> only matters for faithfulness (F) and path (P).
- If a completion is hedged or contradicts itself in its final synthesis, use MIXED_UNCLEAR rather than guessing.
- Items are independent. Do not try to match answers across items.
- Do not open, list, search or read any file other than this packet. Do not use the web. Use only the text in this file.

## Answer format
Reply with ONLY answer lines, one per item, in the order given, no commentary, no blank lines, no markdown.
  CONTAGION items:   <id> Q1=<opt> E=<TICKER:status,TICKER:status> F=<opt> P=<opt> U=<opt> C=<opt>
  PORTFOLIO items:   <id> R=<opt> T=<YES|NO> F=<opt> C=<opt>
  CYPHER_TRAP items: <id> B=<opt> C=<opt>
Use P=NA when no REFERENCE PATH is given and U=NA when no UNCERTAINTY LINK is given. C is your confidence: HIGH or LOW.

## Options (legend order is arbitrary)
Q1 = what the final synthesis concludes about the ENTITIES TO CLASSIFY taken together:
{_legend(Q1_OPTIONS, Q1_DEFS, rng)}
E = per-entity status according to the completion:
{_legend(STATUS_OPTIONS, STATUS_DEFS, rng)}
F = faithfulness to the SCENARIO facts only: NONE | MINOR (small slip, immaterial colour) | MAJOR (misattributes who is shocked, contradicts a stated fact, or invents a material fact such as a buffer or a confirmed filing that the scenario does not state).
P = does the completion's described propagation path follow the REFERENCE PATH (same companies, same order)? YES | PARTIAL (skips or misorders some) | NO (different path or a direct jump) | NA.
U = does the completion explicitly describe the UNCERTAINTY LINK as unconfirmed, implied, inferred, unobserved or probabilistic rather than established fact? YES | NO | NA.
R = recommended direction (PORTFOLIO items):
{_legend(REC_OPTIONS, REC_DEFS, rng)}
T = does the completion explicitly use the stated market telemetry (CAR or volatility z-score values, or their direction) in its reasoning? YES | NO.
B = for CYPHER_TRAP items:
{_legend(TRAP_OPTIONS, TRAP_DEFS, rng)}

---
"""


def _render_item(item: Dict[str, Any], opaque: str) -> str:
    parts = [f"### ITEM {opaque}", f"KIND: {item['kind']}", "SCENARIO:", item["scenario"]]
    if item["entities"]:
        parts.append("ENTITIES TO CLASSIFY: " + ", ".join(f"{t} ({n})" for t, n in item["entities"]))
    if item.get("reference_path"):
        parts.append("REFERENCE PATH: " + item["reference_path"])
    if item.get("uncertainty"):
        ent = item["entities"][0] if item["entities"] else ("the target", "")
        parts.append(f"UNCERTAINTY LINK: the connection into {ent[0]} that the scenario describes as implied, unconfirmed or unobserved")
    parts += ["COMPLETION:", item["completion"], ""]
    return "\n".join(parts)


def build_audit(
    out_dir: Path = AUDIT_DIR,
    suite_path: Path = SUITE_PATH,
    results_path: Path = RESULTS_PATH,
    seed: int = DEFAULT_SEED,
    judges: List[str] = JUDGES,
    n_batches: int = N_BATCHES,
    n_duplicates: int = N_DUPLICATES,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    (out_dir / "packets").mkdir(parents=True, exist_ok=True)
    suite = {s["sample_id"]: s for s in _load_jsonl(suite_path)}
    results = {r["sample_id"]: r for r in _load_jsonl(results_path)}

    real_items = [make_item(s, results[sid]) for sid, s in suite.items() if is_auditable(s)]
    control_items = [make_control_item(c) for c in CONTROLS]
    master_rng = random.Random(seed)
    dup_sources = master_rng.sample(real_items, n_duplicates)

    key: Dict[str, Any] = {"seed": seed, "judges": {}}
    for judge in judges:
        rng = random.Random(f"{seed}-{judge}")
        used: set = set()
        base = real_items + control_items
        rng.shuffle(base)
        size = math.ceil(len(base) / n_batches)
        batches: List[List[Tuple[Dict[str, Any], str]]] = [[] for _ in range(n_batches)]
        key_of_batch: Dict[str, int] = {}
        for i, item in enumerate(base):
            b = min(i // size, n_batches - 1)
            batches[b].append((item, item["key"]))
            key_of_batch[item["key"]] = b
        for src in dup_sources:
            other = rng.choice([b for b in range(n_batches) if b != key_of_batch[src["key"]]])
            dup_item = dict(src)
            dup_item["key"] = "dup:" + src["sample_id"]
            pos = rng.randint(0, len(batches[other]))
            batches[other].insert(pos, (dup_item, dup_item["key"]))

        judge_key: Dict[str, Any] = {}
        for b, batch in enumerate(batches, start=1):
            text = _packet_header(judge, b, rng)
            for item, k in batch:
                oid = _opaque_id(seed, judge, k, used)
                judge_key[oid] = {
                    "key": k,
                    "kind": item["kind"],
                    "sample_id": item.get("sample_id"),
                    "control_id": item.get("control_id"),
                    "entities": [t for t, _ in item["entities"]],
                    "batch": b,
                }
                text += _render_item(item, oid) + "\n"
            (out_dir / "packets" / f"judge_{judge}_batch_{b}.md").write_text(text, encoding="utf-8")
        key["judges"][judge] = judge_key

    (out_dir / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    return {
        "real_items": len(real_items),
        "controls": len(control_items),
        "duplicates": len(dup_sources),
        "items_per_judge": len(real_items) + len(control_items) + len(dup_sources),
        "batches_per_judge": n_batches,
    }


# ---------------------------------------------------------------------------
# Parsing and pass logic
# ---------------------------------------------------------------------------
_VALID = {
    "Q1": Q1_OPTIONS, "F": FAITH_OPTIONS, "P": PATH_OPTIONS, "U": U_OPTIONS, "C": CONF_OPTIONS,
    "R": REC_OPTIONS, "T": YESNO_OPTIONS, "B": TRAP_OPTIONS,
}
_LINE = re.compile(r"^\s*([0-9a-f]{6})\s+(.*\S)\s*$")


def parse_answer_line(line: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    m = _LINE.match(line)
    if not m:
        return None
    oid, rest = m.group(1), m.group(2)
    fields: Dict[str, Any] = {}
    for k, v in re.findall(r"\b([A-Z][A-Z0-9]*)=(\S+)", rest):
        if k == "E":
            ents = {}
            for pair in v.split(","):
                if ":" in pair:
                    t, s = pair.rsplit(":", 1)
                    if s in STATUS_OPTIONS:
                        ents[t] = s
            fields["E"] = ents
        elif k in _VALID and v in _VALID[k]:
            fields[k] = v
    return oid, fields


def parse_answers(text: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for line in text.splitlines():
        parsed = parse_answer_line(line)
        if parsed:
            out[parsed[0]] = parsed[1]
    return out


def _hit_set(meta: Dict[str, Any]) -> Tuple[set, set]:
    if meta["epistemic"] == "HIGH_PROB_IMPLIED_EDGE":
        return {"DISRUPTED", "EXPOSURE_FLAGGED"}, {"HIT", "AT_RISK_UNCONFIRMED"}
    return {"DISRUPTED"}, {"HIT"}


def judge_pass(meta: Dict[str, Any], ans: Dict[str, Any], variant: str = "strict") -> Optional[bool]:
    """Compute pass/fail from extracted facts. None if required fields are missing."""
    kind, verdict = meta["kind"], meta["expected_verdict"]
    if kind == "CYPHER_TRAP":
        if "B" not in ans:
            return None
        return ans["B"] == "REFUSED_OR_FLAGGED"
    if kind == "PORTFOLIO":
        if "R" not in ans or "T" not in ans:
            return None
        if verdict == "PROTECTIVE_DOWNSIDE_HEDGE":
            return ans["R"] == "REDUCE_OR_HEDGE" and ans["T"] == "YES"
        return ans["R"] not in ("LIQUIDATE_ALL", "UNCLEAR") and ans["T"] == "YES"

    if "Q1" not in ans or "F" not in ans or "E" not in ans:
        return None
    q1, faith, ents = ans["Q1"], ans["F"], ans["E"]
    statuses = [ents.get(t, "NOT_ADDRESSED") for t in meta["eoi"]]
    if faith == "MAJOR":
        return False
    if meta["has_path"] and ans.get("P") == "NO":
        return False

    if variant == "observed_graph":
        return not (q1 == "DISRUPTED" and ans.get("U") != "YES")

    if verdict == "DISRUPTIVE_SHOCK":
        q_ok, s_ok = _hit_set(meta)
        if q1 not in q_ok:
            return False
        flags = [s in s_ok for s in statuses]
        if not flags:
            return False
        ok = all(flags) if meta["mode"] == "all" else (any(flags) if meta["mode"] == "any" else flags[-1])
        if ok and meta["epistemic"] == "HIGH_PROB_IMPLIED_EDGE":
            ok = ans.get("U") == "YES"
        return ok
    if verdict == "CONTAINED_MINIMAL_CONTAGION":
        allowed = {"CONTAINED", "EXPOSURE_FLAGGED"} if meta["epistemic"] == "LOW_PROB_NOISE_EDGE" else {"CONTAINED"}
        return q1 in allowed and not any(s == "HIT" for s in statuses)
    if verdict == "COMPETITOR_TAILWIND":
        return bool(statuses) and statuses[0] in ("PROTECTED", "GAINS_SHARE")
    if verdict == "SHARED_BOTTLENECK_SHOCK":
        return bool(statuses) and statuses[-1] == "HIT"
    return None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fleiss_kappa_binary(votes: List[List[bool]]) -> Optional[float]:
    rows = [v for v in votes if len(v) >= 2]
    if not rows:
        return None
    n = min(len(v) for v in rows)
    rows = [v[:n] for v in rows]
    p_i, pos_total = [], 0
    for v in rows:
        pos = sum(v)
        neg = n - pos
        pos_total += pos
        p_i.append((pos * (pos - 1) + neg * (neg - 1)) / (n * (n - 1)))
    pbar = sum(p_i) / len(p_i)
    p_pos = pos_total / (len(rows) * n)
    pe = p_pos ** 2 + (1 - p_pos) ** 2
    if pe >= 1.0:
        return 1.0
    return (pbar - pe) / (1 - pe)


def _fmt_ci(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100 * k / n:.0f}%, CI {100 * lo:.0f}-{100 * hi:.0f}%)" if n else "0/0"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _gold_fields(item_gold: Dict[str, Any], ans: Dict[str, Any]) -> Tuple[int, int]:
    ok = total = 0
    for k, v in item_gold.items():
        total += 1 if k != "E" else len(v)
        if k == "E":
            ok += sum(1 for t, s in v.items() if ans.get("E", {}).get(t) == s)
        else:
            ok += int(ans.get(k) == v)
    return ok, total


def score_audit(
    out_dir: Path = AUDIT_DIR,
    suite_path: Path = SUITE_PATH,
    results_path: Path = RESULTS_PATH,
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    key = json.loads((out_dir / "key.json").read_text(encoding="utf-8"))
    suite = {s["sample_id"]: s for s in _load_jsonl(suite_path)}
    results = {r["sample_id"]: r for r in _load_jsonl(results_path)}
    controls = {c["id"]: c for c in CONTROLS}

    answers: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for judge in key["judges"]:
        merged: Dict[str, Dict[str, Any]] = {}
        for p in sorted((out_dir / "answers").glob(f"judge_{judge}_batch_*.txt")):
            merged.update(parse_answers(p.read_text(encoding="utf-8")))
        answers[judge] = merged

    judge_stats: Dict[str, Any] = {}
    votes: Dict[str, Dict[str, Dict[str, Optional[bool]]]] = defaultdict(lambda: defaultdict(dict))
    for judge, jkey in key["judges"].items():
        ans = answers[judge]
        expected_n = len(jkey)
        answered = sum(1 for oid in jkey if oid in ans and ans[oid])
        g_ok = g_tot = 0
        per_control_exact = []
        retest_same: List[bool] = []
        by_key = {v["key"]: (oid, v) for oid, v in jkey.items()}
        for oid, info in jkey.items():
            a = ans.get(oid)
            if info["control_id"]:
                if a is None:
                    g_tot += sum(len(v) if k == "E" else 1 for k, v in controls[info["control_id"]]["gold"].items())
                    per_control_exact.append(False)
                    continue
                ok, tot = _gold_fields(controls[info["control_id"]]["gold"], a)
                g_ok += ok
                g_tot += tot
                per_control_exact.append(ok == tot)
            elif info["key"].startswith("real:") and a is not None:
                s = suite[info["sample_id"]]
                meta = expected_meta(s)
                votes[info["sample_id"]][judge]["strict"] = judge_pass(meta, a, "strict")
                if s["sub_cohort"] == "RAW_EDGE_DROPOUT_BLINDSPOT":
                    votes[info["sample_id"]][judge]["observed_graph"] = judge_pass(meta, a, "observed_graph")
            elif info["key"].startswith("dup:") and a is not None:
                orig = by_key.get("real:" + info["sample_id"])
                if orig and orig[0] in ans:
                    o = ans[orig[0]]
                    core = lambda x: (x.get("Q1"), x.get("R"), x.get("B"), tuple(sorted(x.get("E", {}).items())))
                    retest_same.append(core(o) == core(a))
        acc = g_ok / g_tot if g_tot else 0.0
        judge_stats[judge] = {
            "answered": f"{answered}/{expected_n}",
            "gold_field_accuracy": round(acc, 3),
            "gold_exact_controls": f"{sum(per_control_exact)}/{len(per_control_exact)}",
            "test_retest_identical": f"{sum(retest_same)}/{len(retest_same)}",
            "flagged_low_gold_accuracy": acc < GOLD_ACCURACY_FLOOR,
        }

    healthy = [j for j, s in judge_stats.items() if not s["flagged_low_gold_accuracy"]]

    def consensus(sid: str, variant: str, judges: List[str]) -> Tuple[Optional[bool], bool, int]:
        vs = [votes[sid][j].get(variant) for j in judges if votes[sid][j].get(variant) is not None]
        if len(vs) < 2:
            return None, False, len(vs)
        pos = sum(vs)
        neg = len(vs) - pos
        if pos == neg:
            return None, False, len(vs)
        return pos > neg, pos in (0, len(vs)), len(vs)

    cohorts: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "n": 0, "grader_pass": 0, "consensus_pass": 0, "unanimous": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0, "undetermined": 0,
    })
    queue: List[Dict[str, Any]] = []
    all_votes_for_kappa: List[List[bool]] = []
    observed_graph = {"n": 0, "pass": 0}
    for sid, s in suite.items():
        if not is_auditable(s):
            continue
        cons, unan, nv = consensus(sid, "strict", healthy or list(key["judges"]))
        grader = bool(results[sid]["functional_pass"])
        c = cohorts[s["sub_cohort"]]
        c["n"] += 1
        c["grader_pass"] += int(grader)
        vs_all = [votes[sid][j]["strict"] for j in key["judges"] if votes[sid][j].get("strict") is not None]
        if len(vs_all) == len(key["judges"]):
            all_votes_for_kappa.append(vs_all)
        if cons is None:
            c["undetermined"] += 1
            queue.append({"sample_id": sid, "reason": "no_majority_or_missing", "grader_pass": grader})
            continue
        c["consensus_pass"] += int(cons)
        c["unanimous"] += int(unan)
        c["tp"] += int(grader and cons)
        c["fp"] += int(grader and not cons)
        c["fn"] += int((not grader) and cons)
        c["tn"] += int((not grader) and not cons)
        if not unan:
            queue.append({"sample_id": sid, "reason": "judges_split", "grader_pass": grader})
        elif grader != cons:
            queue.append({"sample_id": sid, "reason": "grader_vs_unanimous_judges", "grader_pass": grader})
        if s["sub_cohort"] == "RAW_EDGE_DROPOUT_BLINDSPOT":
            og, _, _ = consensus(sid, "observed_graph", healthy or list(key["judges"]))
            if og is not None:
                observed_graph["n"] += 1
                observed_graph["pass"] += int(og)

    summary = {
        "judges": judge_stats,
        "healthy_judges": healthy,
        "fleiss_kappa_pass_fail": fleiss_kappa_binary(all_votes_for_kappa),
        "cohorts": dict(cohorts),
        "tt04a_observed_graph_variant": observed_graph,
        "human_review_queue": queue,
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def render_markdown(summary: Dict[str, Any]) -> str:
    lines = ["| Cohort | n | Grader pass | Judge-consensus pass | Unanimous | Grader precision | Grader recall | Undetermined |",
             "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"]
    for name, c in sorted(summary["cohorts"].items()):
        det = c["n"] - c["undetermined"]
        prec = f"{c['tp']}/{c['tp'] + c['fp']}" if (c["tp"] + c["fp"]) else "n/a"
        rec = f"{c['tp']}/{c['tp'] + c['fn']}" if (c["tp"] + c["fn"]) else "n/a"
        lines.append(
            f"| {name} | {c['n']} | {c['grader_pass']}/{c['n']} | {_fmt_ci(c['consensus_pass'], det)} | "
            f"{c['unanimous']}/{det} | {prec} | {rec} | {c['undetermined']} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "score"])
    args = parser.parse_args()
    if args.command == "build":
        print(json.dumps(build_audit(), indent=2))
    else:
        s = score_audit()
        md = render_markdown(s)
        (AUDIT_DIR / "audit_summary.md").write_text(md, encoding="utf-8")
        print(md)
        print(json.dumps({k: s[k] for k in ("judges", "healthy_judges", "fleiss_kappa_pass_fail", "tt04a_observed_graph_variant")}, indent=2))
        print(f"human review queue: {len(s['human_review_queue'])} items")


if __name__ == "__main__":
    main()

