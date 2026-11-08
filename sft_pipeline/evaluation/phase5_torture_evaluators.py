"""
Deterministic Semantic, Causal-Trace, Epistemic, and Schema Evaluators for Phase 5 Torture Suite.

Upgrades evaluation beyond syntax-only checks to measure:
1. Semantic Verdict & Entity Polarity Accuracy (`verdict_pass`)
2. Intermediate Multi-Hop Bridge Entity Recall & Distractor Isolation (`causal_trace_pass`)
3. Epistemic Provenance Qualification on Imputed Links (`epistemic_pass`)
4. Strict openCypher Schema Property Whitelist & 10-Digit CIK Integrity (`schema_cik_pass`)
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Set, Tuple

os.environ.setdefault("EDGAR_LOCAL_DATA_DIR", "/tmp/.edgar")

from sft_pipeline.phase4_export_and_serve import (
    validate_cypher_syntax,
    validate_reasoning_cot,
)

# Registered node and relationship property whitelist for Task C (`TEXT_TO_CYPHER`)
REGISTERED_SCHEMA_PROPERTIES: Set[str] = {
    "id",
    "name",
    "ticker",
    "cik",
    "sector",
    "gics_sector",
    "polarity",
    "materiality",
    "status",
    "provenance",
    "confidence",
    "nature",
    "valid_from",
    "valid_to",
    "effective_from",
    "source_hash",
}

DISRUPTIVE_PATTERNS = re.compile(
    r"\b("
    r"DISRUPTIVE[\s_-]+SHOCK|ACUTE[\s_-]+SHOCK|FULL[\s_-]+CONTAGION|"
    r"SEVERE[\s_-]+CONTAGION|CRITICAL[\s_-]+CONTAGION|HIGH[\s_-]+(DIRECT[\s_-]+)?CONTAGION|"
    r"PRODUCTION[\s_-]+(HALT|DELAYS?|CURTAILMENT)|DELAYED[\s_-]+(DELIVERIES|CRITICAL)|"
    r"CURTAILS?|CURTAILED|2[\s-]QUAD[\s_-]+DELAY|2[\s-]QUARTER[\s_-]+DELAYED|"
    r"(ACUTE|IMMEDIATE|SEVERE)?[\s_-]*COMPONENT[\s_-]+SHORTAGES?|"
    r"100%\s+COMPONENT\s+SHORTAGE|"
    r"MARGIN[\s_-]+(COMPRESSION|SHOCK|PRESSURE)|DOWNWARD[\s_-]+PRESSURE|"
    r"OPERATIONAL[\s_-]+DISRUPTION|MATERIAL[\s_-]+(SHOCK|HEADWIND)|"
    r"DISRUPTIVE[\s_-]+SUPPLY[\s_-]+(SHOCK|CONSTRAINT)|"
    r"SUPPLY[\s_-]+(CURTAILMENT|SHORTAGE|SHOCK|VULNERABILITY|BOTTLENECK)|"
    r"ALLOCATION[\s_-]+CUTS?|TERMINAL\s*\(FULL\s+MARGIN\s+SHOCK"
    r")\b",
    re.IGNORECASE,
)

DISRUPTIVE_NEGATION_OVERRIDES = re.compile(
    r"("
    r"\bCONTAINED[\s_-]+MINIMAL[\s_-]+(CONTAGION|CONTAINMENT)\b|"
    r"\bDOES\s+NOT\s+(PROPAGATE|TRANSMIT|DIRECTLY\s+SHOCK|DIRECTLY\s+IMPACT|DIRECTLY\s+TRIGGER|DIRECTLY\s+DISABLE)\b|"
    r"\bZERO\s+MATERIAL\s+(DISRUPTION|COMPONENT)\b|"
    r"\bMARGIN\s+STABILITY\s+PRESERVED\b|"
    r"\bCORE\s+OPERATIONAL\s+CHAIN\s+REMAINS\s+PROTECTED\b|"
    r"\bPARTIALLY\s+INSULATING\b[^.\n]{0,80}\bFROM\s+IMMEDIATE\s+OPERATIONAL\s+DISRUPTION\b|"
    r"\b(IS|ARE)\s+(FULLY\s+)?(CONTAINED|ISOLATED)\s+TO\s+(THAT|ITS\s+PRIMARY|ITS\s+DIRECT|THEIR\s+DIRECT|SECONDARY|MATERIAL_TIER_2|NON[\s-]CRITICAL)"
    r")",
    re.IGNORECASE,
)

CONTAINED_PATTERNS = re.compile(
    r"("
    r"\bCONTAINED[\s_-]+MINIMAL[\s_-]+(CONTAGION|CONTAINMENT)\b|"
    r"\bMINIMAL[\s_-]+CONTAGION\b|\bNO[\s_-]+MATERIAL[\s_-]+CONTAGION\b|\bZERO[\s_-]+MATERIAL[\s_-]+DISRUPTION\b|"
    r"\bDOES\s+NOT\s+(PROPAGATE|TRANSMIT|DIRECTLY\s+SHOCK|DIRECTLY\s+IMPACT)\b|"
    r"\b(FULLY|ENTIRELY|EFFECTIVELY)\s+(PROTECTED|INSULATED|INSULATES|HEDGED|SHIELDED|OPERATIONAL|FUNCTIONAL|PROTECTIVE)\b|"
    r"\bREMAINS?\s+(RESILIENT|INTACT|PROTECTED|OPERATIONAL|EFFECTIVE)\b|"
    r"\b(ISOLATED|CONTAINED)\s+TO\s+(ROUTINE|SCHEDULED|LEGACY|A\s+DEFUNCT|NON[\s-]CRITICAL)\b|"
    r"\bSHOCK\s+ISOLATION\b[^|\n]{0,90}\bCONTAINED\b|"
    r"\bPREVENTS?\s+ANY\s+SUPPLY\s+CHAIN\s+DISRUPTION\b|"
    r"\bMITIGATES?\s+ANY\s+DIRECT\s+SHOCK\b|"
    r"\bNO\s+IMMEDIATE\s+OPERATIONAL\s+CHANGE\s+REQUIRED\b"
    r")",
    re.IGNORECASE,
)

TAILWIND_PATTERNS = re.compile(
    r"("
    r"\bTAILWIND\b|"
    r"\b(CAPTURE|ACQUIRE|EXPAND|GAIN)S?\s+(ITS\s+)?MARKET[\s_-]?SHARE\b|"
    r"\bMARKET[\s_-]?SHARE\s+(GAIN|CAPTURE|EXPANSION|UPSIDE)\b|"
    r"\bCOMPETITIVE\s+(BENEFICIARY|EXPANSION|ADVANTAGE|OPPORTUNITY|MARKET\s+REACTION)\b|"
    r"\b(FULLY|ENTIRELY|COMPLETE)\s+(PROTECTED|INSULATED|SHIELDED|SUPPLY\s+CHAIN\s+PROTECTION)\b|"
    r"\bPROTECTED\s+(OPERATIONAL|FROM\s+OPERATIONAL)\b|"
    r"\bINSULATED\s+(FROM|DUE\s+TO)\b"
    r")",
    re.IGNORECASE,
)

SHARED_BOTTLENECK_PATTERNS = re.compile(
    r"("
    r"\bFULL[\s_-]+CONTAGION\b|\bBOTH[\s_-]+EXPOSED\b|"
    r"\bBOTH\b[^.\n]{0,100}\b(EXPOSED|DISRUPTED|SUFFER|FACE|CUT|CONSTRAINT|IMPACT|AFFECT|REDUCE)\b|"
    r"\bAFFECTS?\s+BOTH\b|\bIMPACTS?\s+BOTH\b|"
    r"\bIDENTICAL\s+(50%|9[\s-]MONTH|EXPOSURE|SUPPLY)\b|"
    r"\bSINGLE[\s,-]+SHARED\s+VALUE\s+CHAIN\b"
    r")",
    re.IGNORECASE,
)

IMPLIED_EXPOSURE_PATTERNS = re.compile(
    r"("
    r"\bPOTENTIALLY\s+DISRUPTED\b|"
    r"\b(SECONDARY|TIER\s*2|INDIRECT|HIGH|CONTINGENCY)\s+CONTAGION(\s+EXPOSURE)?\b|"
    r"\bCONTAGION\s+(EXPOSURE|TO|PATH)\b|"
    r"\bMATERIAL\s+HEADWIND\b|\bSTRATEGIC\s+RISK\b|\bSTRUCTURAL\s+GAP\b"
    r")",
    re.IGNORECASE,
)

DOWNSIDE_HEDGE_PATTERNS = re.compile(
    r"\b("
    r"PUT|COLLAR|REDUCE|UNDERWEIGHT|TRIM|PROTECTIVE|DOWNSIDE|SHORT|DEFENSIVE|HEDGE"
    r")\b",
    re.IGNORECASE,
)

EPISTEMIC_QUALIFIER_PATTERNS = re.compile(
    r"\b("
    r"IMPLIED|INFERRED|UNCONFIRMED|UNVERIFIED|STRUCTURAL_PRIOR|STRUCTURAL\s+PRIOR|"
    r"LATENT|POTENTIAL|PROBABLE|PROBABILITY|HYPOTHESIS|ESTIMATED|GNN"
    r")\b",
    re.IGNORECASE,
)


def _split_cot(prediction: str) -> Tuple[str, str]:
    """Return `(think_body, post_think_summary)` from a CoT completion."""
    if "<think>" in prediction and "</think>" in prediction:
        after_open = prediction.split("<think>", 1)[1]
        think_body, post_body = after_open.split("</think>", 1)
        return think_body.strip(), post_body.strip()
    return "", prediction.strip()


def evaluate_verdict_and_polarity(
    prediction: str, assertions: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Evaluate whether the model's post-`</think>` synthesis reaches the right causal/financial verdict.
    """
    expected_verdict = assertions.get("expected_verdict")
    if not expected_verdict:
        return True, "N/A"

    think_body, post_body = _split_cot(prediction)
    target_text = post_body if post_body else prediction
    full_upper = prediction.upper()
    post_upper = target_text.upper()

    if expected_verdict == "DISRUPTIVE_SHOCK":
        # When evaluating a high-probability GNN implied edge (TT-04B), calibrated outputs surface
        # implied/secondary contagion exposure rather than claiming hard verified DISRUPTIVE_SHOCK.
        if assertions.get("epistemic_mode") == "HIGH_PROB_IMPLIED_EDGE" and assertions.get("gnn_augmented_hint", False):
            if "CONTAINED_MINIMAL_CONTAGION" in post_upper:
                return False, "Falsely dismissed high-probability implied edge as CONTAINED_MINIMAL_CONTAGION"
            if IMPLIED_EXPOSURE_PATTERNS.search(target_text) or DISRUPTIVE_PATTERNS.search(target_text):
                return True, "Surfaced latent contagion exposure across GNN implied link"
            return False, "Failed to surface contagion exposure on high-probability GNN link"

        if DISRUPTIVE_NEGATION_OVERRIDES.search(target_text) and "DISRUPTIVE_SHOCK" not in post_upper:
            return False, "Summary concludes shock is contained/insulated despite expected DISRUPTIVE_SHOCK"
        if "DISRUPTIVE_SHOCK" in post_upper or DISRUPTIVE_PATTERNS.search(target_text):
            return True, "Matched DISRUPTIVE_SHOCK / active propagation verdict"
        return False, "Missing DISRUPTIVE_SHOCK or disruptive propagation verdict in summary"

    if expected_verdict == "CONTAINED_MINIMAL_CONTAGION":
        if "DISRUPTIVE_SHOCK" in post_upper and "CONTAINED_MINIMAL_CONTAGION" not in post_upper:
            if not re.search(r"\bCONTAINED\s+TO\s+THAT\s+ENTITY\b|\bDOES\s+NOT\s+PROPAGATE\b", target_text, re.I):
                return False, "Falsely classified contained/buffered scenario as DISRUPTIVE_SHOCK"
        if "CONTAINED_MINIMAL_CONTAGION" in post_upper or CONTAINED_PATTERNS.search(target_text):
            return True, "Matched CONTAINED_MINIMAL_CONTAGION"
        return False, "Failed to conclude shock is contained/insulated (memorized disruptive template)"

    if expected_verdict == "COMPETITOR_TAILWIND":
        # Must recognize rival is unaffected / gains share rather than suffering identical halt
        if "CONTAINED_MINIMAL_CONTAGION" in post_upper or TAILWIND_PATTERNS.search(target_text):
            # Make sure it doesn't claim the rival's own production is halted by the focal foundry
            rival_ticker = assertions.get("queried_entity_ticker", "")
            if rival_ticker and re.search(
                rf"\b{re.escape(rival_ticker)}\b[^.\n]{{0,60}}\b(halts?|curtails?\s+{re.escape(rival_ticker)}|faces?\s+immediate\s+shortage)",
                target_text,
                re.I,
            ):
                return False, f"Falsely propagated supply halt onto unaffected competitor {rival_ticker}"
            return True, "Matched COMPETITOR_TAILWIND / insulated rival"
        return False, "Failed to recognize unaffected competitor tailwind / insulation (smeared bearish shock)"

    if expected_verdict == "SHARED_BOTTLENECK_SHOCK":
        # Both competitors share the halted Tier-1 supplier -> neither can capture share unhindered
        if (
            "DISRUPTIVE_SHOCK" in full_upper
            or SHARED_BOTTLENECK_PATTERNS.search(target_text)
            or DISRUPTIVE_PATTERNS.search(target_text)
        ) and not re.search(
            r"\b(unaffected|fully\s+insulated|seamless\s+market[\s-]share\s+gain)\b",
            target_text,
            re.I,
        ):
            return True, "Matched SHARED_BOTTLENECK_SHOCK across both rivals"
        return False, "Failed to recognize shared upstream bottleneck constraining both competitors"

    if expected_verdict == "PROTECTIVE_DOWNSIDE_HEDGE":
        # Bearish tape (-CAR, -z) despite upbeat PR -> must recommend downside protection / trim, not naked long
        if re.search(r"\b(double\s+overweight|increase\s+long|add\s+leverage|unhedged\s+long)\b", target_text, re.I):
            return False, "Recommended leveraged long exposure despite severe negative CAR divergence"
        if DOWNSIDE_HEDGE_PATTERNS.search(target_text):
            return True, "Matched PROTECTIVE_DOWNSIDE_HEDGE"
        return False, "Missing protective downside hedge recommendation"

    if expected_verdict == "RELIEF_OVERHANG_CLEARED":
        # Bearish PR text already priced in (+CAR relief rally) -> avoid panic 100% liquidation / naked short
        if re.search(r"\b(liquidate\s+100%|immediate\s+full\s+liquidation|naked\s+short)\b", target_text, re.I):
            return False, "Recommended panic liquidation despite positive relief CAR"
        if validate_reasoning_cot(prediction)[0]:
            return True, "Matched balanced post-overhang rebalancing"
        return False, "Invalid CoT on relief scenario"

    return True, "Valid"


def evaluate_causal_hop_trace(
    prediction: str, assertions: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Verify `<think>...</think>` validity and recall of intermediate bridge entities along `k`-hop paths.
    """
    task_type = assertions.get("task_type", "")
    if task_type not in ("CONTAGION_REASONING", "PORTFOLIO_RECOMMENDATION"):
        return True, "N/A"

    cot_ok, cot_err = validate_reasoning_cot(prediction)
    if not cot_ok:
        return False, cot_err

    required_entities: List[str] = assertions.get("required_bridge_entities", [])
    if required_entities:
        pred_lower = prediction.lower()
        missing = []
        for ent in required_entities:
            # Accept either ticker or company name token match (e.g. "NVDA|NVIDIA")
            alternatives = [alt.strip().lower() for alt in ent.split("|") if alt.strip()]
            if not any(alt in pred_lower for alt in alternatives):
                missing.append(ent)
        if missing:
            return False, f"Dropped intermediate/target hop entities in trace: {missing}"

    return True, "Valid"


def evaluate_epistemic_provenance(
    prediction: str, assertions: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    For Axis 4 (`TT-04B`), verify that high-probability `IMPLIED_SUPPLIES_TO` edges are surfaced
    with explicit uncertainty qualifiers rather than fabricated SEC 10-K certainty.
    """
    epistemic_mode = assertions.get("epistemic_mode")
    if not epistemic_mode:
        return True, "N/A"

    if epistemic_mode == "HIGH_PROB_IMPLIED_EDGE":
        has_qualifier = bool(EPISTEMIC_QUALIFIER_PATTERNS.search(prediction))
        if not has_qualifier:
            return False, "Treated GNN implied link as hard confirmed fact without epistemic qualifier"
        if re.search(r"confirmed\s+in\s+SEC\s+Form\s+10-K\s+for\s+the\s+implied", prediction, re.I):
            return False, "Fabricated hard SEC 10-K provenance for implied edge"
        return True, "Properly qualified implied structural edge"

    if epistemic_mode == "LOW_PROB_NOISE_EDGE":
        # Low link_probability (e.g. 0.22) should be discounted / contained
        _, post_body = _split_cot(prediction)
        if "DISRUPTIVE_SHOCK" in post_body.upper() and not re.search(
            r"\b(low\s+probability|unconfirmed|unlikely|minimal|contained)\b", post_body, re.I
        ):
            return False, "Over-triggered DISRUPTIVE_SHOCK on low-probability (p < 0.30) noise edge"
        return True, "Properly discounted low-probability implied edge"

    return True, "Valid"


def evaluate_schema_and_cik_integrity(
    prediction: str, assertions: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Validate openCypher syntax, 10-digit CIK grounding, negation/termination accuracy, and
    Task C property-name whitelisting.
    """
    task_type = assertions.get("task_type", "")
    cleaned = prediction.strip()

    # 1. Check 10-digit CIKs against allowed ground-truth CIKs
    pred_ciks = set(re.findall(r"\b\d{10}\b", cleaned))
    allowed_ciks = set(assertions.get("allowed_ciks", []))
    if pred_ciks and allowed_ciks is not None:
        # When allowed_ciks is explicitly provided (or empty list for negative samples)
        if "allowed_ciks" in assertions:
            hallucinated = pred_ciks - allowed_ciks
            if hallucinated:
                return False, f"Hallucinated / ungrounded CIKs: {sorted(hallucinated)}"

    if task_type in ("EXTRACT_SEC_GRAPH", "EXTRACT_NEWS_EVENT"):
        expect_none = assertions.get("expect_none", False)
        if expect_none:
            if cleaned.lower() == "(none)":
                return True, "Matched (none) refusal"
            return False, f"Expected (none) on negated/non-causal passage, got: {cleaned[:120]}"

        ast_ok, ast_err = validate_cypher_syntax(cleaned)
        if not ast_ok:
            return False, ast_err
        if cleaned.lower() == "(none)":
            return False, "Unexpected (none) on positive long-context filing"

        expected_status = assertions.get("expected_edge_status")
        if expected_status and expected_status not in cleaned:
            return False, f"Expected lifecycle status '{expected_status}' not found in extracted triple"

        required_ciks = set(assertions.get("required_ciks", []))
        if required_ciks and not required_ciks.issubset(pred_ciks):
            return False, f"Missing required filing CIKs: {sorted(required_ciks - pred_ciks)}"

        return True, "Valid extraction"

    if task_type == "TEXT_TO_CYPHER":
        is_unanswerable_trap = assertions.get("unanswerable_schema_trap", False)
        if cleaned.lower() == "(none)":
            if is_unanswerable_trap:
                return True, "Cleanly refused unanswerable schema trap with (none)"
            return False, "Unexpected (none) on valid compositional query"

        ast_ok, ast_err = validate_cypher_syntax(cleaned)
        if not ast_ok:
            return False, ast_err

        # Strip quoted string literals first so company names like 'A. O. Smith' or 'Amazon.com'
        # do not trigger false dot-property matches.
        unquoted = re.sub(r"'[^']*'|\"[^\"]*\"", "''", cleaned)
        dot_props = re.findall(r"\b[a-zA-Z_]\w*\.([a-zA-Z_]\w*)\b", unquoted)
        map_props: List[str] = []
        for brace_body in re.findall(r"\{([^}]*)\}", unquoted):
            map_props.extend(re.findall(r"\b([a-zA-Z_]\w*)\s*:", brace_body))
        used_props = {p.lower() for p in (dot_props + map_props)}
        illegal_props = used_props - REGISTERED_SCHEMA_PROPERTIES
        if illegal_props:
            return False, f"Hallucinated non-existent schema properties: {sorted(illegal_props)}"

        required_clauses: List[str] = assertions.get("required_cypher_patterns", [])
        for pat in required_clauses:
            if not re.search(pat, cleaned, re.IGNORECASE):
                return False, f"Missing required compositional Cypher construct: /{pat}/"

        return True, "Valid schema-compliant Cypher"

    return True, "Valid"


def evaluate_torture_sample(prediction: str, sample: Dict[str, Any]) -> Dict[str, Any]:
    """Run all 4 Phase 5 evaluators on a single prediction and return granular diagnostics."""
    assertions = dict(sample.get("expected_assertions", {}))
    assertions.setdefault("task_type", sample["task_type"])
    if sample.get("metadata", {}).get("gnn_augmented"):
        assertions["gnn_augmented_hint"] = True

    verdict_ok, verdict_msg = evaluate_verdict_and_polarity(prediction, assertions)
    trace_ok, trace_msg = evaluate_causal_hop_trace(prediction, assertions)
    epistemic_ok, epistemic_msg = evaluate_epistemic_provenance(prediction, assertions)
    schema_ok, schema_msg = evaluate_schema_and_cik_integrity(prediction, assertions)

    overall_pass = bool(verdict_ok and trace_ok and epistemic_ok and schema_ok)
    failure_reasons = []
    if not verdict_ok:
        failure_reasons.append(f"verdict: {verdict_msg}")
    if not trace_ok:
        failure_reasons.append(f"causal_trace: {trace_msg}")
    if not epistemic_ok:
        failure_reasons.append(f"epistemic: {epistemic_msg}")
    if not schema_ok:
        failure_reasons.append(f"schema_cik: {schema_msg}")

    return {
        "functional_pass": overall_pass,
        "verdict_pass": verdict_ok,
        "verdict_detail": verdict_msg,
        "causal_trace_pass": trace_ok,
        "causal_trace_detail": trace_msg,
        "epistemic_pass": epistemic_ok,
        "epistemic_detail": epistemic_msg,
        "schema_cik_pass": schema_ok,
        "schema_cik_detail": schema_msg,
        "failure_summary": "; ".join(failure_reasons) if failure_reasons else None,
    }
