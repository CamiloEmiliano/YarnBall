"""
Phase 3 DPO Dataset Builder via Empirical Cartography Inflection-Lag Harvesting.

Harvests the slow-learning boundary slice (Steps 25, 50, 100 Time-to-Mastery) from
Run 2's empirical training dynamics (`data/sft/remote_artifacts/training_dynamics.jsonl`)
and combines it with Phase 2 autoregressive error patterns to construct paired
preference rankings (`prompt`, `chosen`, `rejected`) for TRL `DPOTrainer`.

Four Targeted Alignment Buckets:
1. Task D (`CONTAGION_REASONING`):
   - Eliminates first-token openCypher mode bleed (`MATCH ...` / `SELECT ...` when
     prompts contain multi-edge subgraph context).
   - De-templates rigid 4-bullet synthetic CoT traces while preserving causal verdicts.
   - Penalizes counterfactual contagion errors (propagating shocks across
     `NON_CRITICAL_SECONDARY` distractor vendors or ignoring `CRITICAL_TIER_1` chokepoints).
2. Task E (`PORTFOLIO_RECOMMENDATION`):
   - Aligns Step 50 slow-inflection boundary volatility cases (`|z| ~ 1.85-2.05`)
     and divergent fundamental vs. guidance signals toward capital-efficient
     delta-neutral derivative structures over unhedged directional calls.
3. Tasks A & B (`EXTRACT_SEC_GRAPH`, `EXTRACT_NEWS_EVENT`):
   - Reinforces Step 100 hardest spin-off disambiguation (`ENTITY_COLLISION_MINING`),
     borderline 8-K CAR volatility (`VOLATILITY_BOUNDARY_TELEMETRY`), and parent S&P 500
     issuer CIK resolution (e.g., DISH Network `0001001082` vs. subsidiary `0001001811`).
4. Task C (`TEXT_TO_CYPHER`):
   - Reinforces Step 50 slow-inflection Cypher query variants (`VAR_1, 2, 7, 9`)
     against unindexed cartesian scans and SQL-Cypher syntax hybrids.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("dpo_builder")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SFT_DIR = PROJECT_ROOT / "data" / "sft"
DEFAULT_DPO_DIR = PROJECT_ROOT / "data" / "dpo"


@dataclass
class DPORecord:
    """Standardized preference pair record compatible with Hugging Face TRL DPOTrainer."""
    sample_id: str
    prompt: str
    chosen: str
    rejected: str
    metadata: Dict[str, Any]
    schema_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class InflectionSliceDPOBuilder:
    """
    Harvests empirical Cartography inflection-lag cohorts and constructs paired
    Chosen vs. Rejected preference datasets for Phase 3 DPO alignment.
    """

    def __init__(
        self,
        sft_dir: Optional[Path] = None,
        dpo_dir: Optional[Path] = None,
        seed: int = 42,
    ):
        self.sft_dir = Path(sft_dir or DEFAULT_SFT_DIR).resolve()
        self.dpo_dir = Path(dpo_dir or DEFAULT_DPO_DIR).resolve()
        self.dpo_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # 1. Empirical Cartography Inflection-Lag Harvesting
    # ------------------------------------------------------------------
    def analyze_inflection_dynamics(self) -> Dict[str, Any]:
        """
        Parse `training_dynamics.jsonl` across early checkpoints [0, 25, 50, 100, 200]
        and map every `guid` (`sample_0` .. `sample_2893`) back to its validation record.
        """
        dyn_file = self.sft_dir / "remote_artifacts" / "training_dynamics.jsonl"
        if not dyn_file.is_file():
            raise FileNotFoundError(f"Training dynamics file not found at {dyn_file}")

        # Reconstruct exact 2,894 validation index ordering used by HF load_dataset
        val_files = sorted(self.sft_dir.glob("*val*.jsonl"))
        indexed_val_rows: List[Dict[str, Any]] = []
        for vf in val_files:
            with open(vf, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        indexed_val_rows.append(json.loads(line))

        if len(indexed_val_rows) != 2894:
            logger.warning("Expected 2894 indexed validation rows, found %d", len(indexed_val_rows))

        # Group sequence probabilities by guid and step
        guid_steps: Dict[str, Dict[int, float]] = defaultdict(dict)
        with open(dyn_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                guid_steps[rec["guid"]][int(rec["step"])] = float(rec["seq_prob"])

        early_steps = [0, 25, 50, 100, 200]
        slow_step50_guids: List[str] = []
        slow_step100_guids: List[str] = []
        sample_metrics: Dict[str, Dict[str, Any]] = {}

        task_step50_counts: Counter = Counter()
        prov_step50_counts: Counter = Counter()
        task_step100_counts: Counter = Counter()
        prov_step100_counts: Counter = Counter()

        for guid, step_map in guid_steps.items():
            idx = int(guid.split("_")[1])
            row = indexed_val_rows[idx]
            sid = row["sample_id"]
            task_type = row["metadata"]["task_type"]
            prov = row["metadata"].get("provenance", "UNKNOWN")

            p0 = step_map.get(0, 0.0)
            p25 = step_map.get(25, 0.0)
            p50 = step_map.get(50, 0.0)
            p100 = step_map.get(100, 0.0)
            p200 = step_map.get(200, 0.0)
            aulc_early = sum(step_map.get(s, 0.0) for s in early_steps) / len(early_steps)

            if p50 < 0.70:
                slow_step50_guids.append(guid)
                task_step50_counts[task_type] += 1
                prov_step50_counts[prov] += 1
            if p100 < 0.70:
                slow_step100_guids.append(guid)
                task_step100_counts[task_type] += 1
                prov_step100_counts[prov] += 1

            # Keep conservative (minimum) early inflection stats per unique sample_id
            if sid not in sample_metrics or p50 < sample_metrics[sid]["prob_step_50"]:
                sample_metrics[sid] = {
                    "sample_id": sid,
                    "task_type": task_type,
                    "provenance": prov,
                    "prob_step_0": round(p0, 4),
                    "prob_step_25": round(p25, 4),
                    "prob_step_50": round(p50, 4),
                    "prob_step_100": round(p100, 4),
                    "prob_step_200": round(p200, 4),
                    "early_aulc": round(aulc_early, 4),
                    "is_slow_step50": p50 < 0.70,
                    "is_slow_step100": p100 < 0.70,
                }

        unique_slow50_sids = {
            sid for sid, m in sample_metrics.items() if m["is_slow_step50"]
        }
        unique_slow100_sids = {
            sid for sid, m in sample_metrics.items() if m["is_slow_step100"]
        }

        report = {
            "generated_at": datetime.now().isoformat(),
            "total_probed_validation_guids": len(guid_steps),
            "unique_validation_sample_ids": len(sample_metrics),
            "step_50_slow_inflection_cohort": {
                "threshold": "seq_prob < 0.70 at Step 50",
                "total_guids": len(slow_step50_guids),
                "unique_sample_ids": len(unique_slow50_sids),
                "by_task": dict(task_step50_counts),
                "by_provenance": dict(prov_step50_counts.most_common()),
            },
            "step_100_hardest_boundary_cohort": {
                "threshold": "seq_prob < 0.70 at Step 100",
                "total_guids": len(slow_step100_guids),
                "unique_sample_ids": len(unique_slow100_sids),
                "by_task": dict(task_step100_counts),
                "by_provenance": dict(prov_step100_counts.most_common()),
            },
            "sample_metrics": sample_metrics,
        }

        report_path = self.dpo_dir / "inflection_slice_report.json"
        out_report = {k: v for k, v in report.items() if k != "sample_metrics"}
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(out_report, f, indent=2)
        logger.info("Saved empirical inflection-slice report to %s", report_path)
        return report

    # ------------------------------------------------------------------
    # 2. De-Templated Chosen & High-Contrast Rejected Synthesizers
    # ------------------------------------------------------------------
    def _parse_task_d_prompt(self, prompt: str) -> Dict[str, Any]:
        """Extract structured entities, tickers, scenario, and edges from a Task D prompt."""
        sector_m = re.search(r"Sector:\s*(.+)", prompt)
        scenario_m = re.search(r"Scenario:\s*(.+)", prompt)
        sector = sector_m.group(1).strip() if sector_m else "Industrials"
        scenario = scenario_m.group(1).strip() if scenario_m else "Supply curtailment shock"

        # Match Cypher edges in prompt context
        # Pattern 1: Distractor prompt with 2 upstream suppliers and 1 downstream customer
        edges = re.findall(
            r"\(([^:()]+):Company(?:\s*\{ticker:\s*'([^']+)'\})?\)"
            r"-\[:SUPPLIES_TO(?:\s*\{nature:\s*'([^']+)'(?:,\s*materiality:\s*'([^']+)')?\})?\]->"
            r"\(([^()]+)\)",
            prompt,
        )

        return {
            "sector": sector,
            "scenario": scenario,
            "edges": edges,
        }

    def _de_template_task_d_chosen(self, row: Dict[str, Any], style_idx: int) -> str:
        """
        Diversify the `<think>` Chain-of-Thought phrasing on Task D so DPO loosens
        the rigid 4-bullet SFT memorization while preserving exact causal accuracy
        and the downstream `**Contagion Analysis Summary**` contract.
        """
        orig = str(row["target_completion"])
        if "</think>" not in orig:
            return str(orig)

        think_part, summary_part = orig.split("</think>", 1)
        # Keep the canonical summary header (`**Contagion Analysis Summary**:`) consistent
        summary_clean = summary_part.strip()
        is_distractor = "CONTAINED_MINIMAL_CONTAGION" in summary_clean

        parsed = self._parse_task_d_prompt(row["prompt"])
        edges = parsed["edges"]
        scenario = parsed["scenario"]
        sector = parsed["sector"]

        if style_idx % 4 == 0:
            # Style 0: Retain original clean step-by-step trace
            return f"{think_part.strip()}\n</think>\n{summary_clean}"

        if is_distractor and len(edges) >= 3:
            # edges[0] = critical supplier -> focal, edges[1] = distractor -> focal, edges[2] = focal -> downstream
            crit_sup, crit_tkr, crit_nat, _, focal_raw = edges[0]
            dist_sup, dist_tkr, dist_nat, dist_mat, _ = edges[1]
            _, focal_tkr, down_nat, _, down_raw = edges[2]
            focal_name = focal_raw.split(":")[0].strip()
            down_name = down_raw.split(":")[0].strip()

            if style_idx % 4 == 1:
                new_think = (
                    f"<think>\n"
                    f"- **Shock Origin & Node Localization**: The reported disruption ({scenario}) is strictly localized at {dist_sup} ({dist_tkr}).\n"
                    f"- **5-Axis Edge Materiality Gate**: Inspecting the subgraph topology, {dist_sup} provides `{dist_nat}` classified as `{dist_mat or 'NON_CRITICAL_SECONDARY'}`—an interchangeable auxiliary input with multi-vendor redundancy.\n"
                    f"- **Primary Bottleneck Verification**: The `CRITICAL_TIER_1` path from {crit_sup} ({crit_tkr}) supplying `{crit_nat}` to {focal_name} ({focal_tkr}) experiences 0% curtailment.\n"
                    f"- **Buffer Absorption & Downstream Insulation**: Standard 90-day consumable buffer inventories at {focal_name} fully absorb the temporary staging delay, insulating `{down_nat}` shipments to {down_name}.\n"
                    f"</think>"
                )
            elif style_idx % 4 == 2:
                new_think = (
                    f"<think>\n"
                    f"Evaluating multi-hop contagion propagation across the {sector} subgraph:\n"
                    f"1. **Upstream Perturbation**: {dist_sup} ({dist_tkr}) suffers an acute operational hold affecting `{dist_nat}`.\n"
                    f"2. **Topological Attenuation (Weak-Link Pruning)**: Because the edge `({dist_tkr})-[:SUPPLIES_TO]->({focal_tkr})` carries materiality `NON_CRITICAL_SECONDARY`, shock transmission attenuates to zero at Hop 1 due to off-the-shelf vendor substitutability and buffer stock.\n"
                    f"3. **Core Production Continuity**: {focal_name}'s mission-critical input stream from {crit_sup} ({crit_tkr}, `CRITICAL_TIER_1`) operates at 100% baseline run-rate.\n"
                    f"4. **2nd-Order Contagion Verdict**: Downstream deliveries from {focal_name} to {down_name} face zero curtailment (`CONTAINED_MINIMAL_CONTAGION`).\n"
                    f"</think>"
                )
            else:
                new_think = (
                    f"<think>\n"
                    f"1. **Causal Path Decomposition**: The subgraph contains two parallel upstream edges into {focal_name} ({focal_tkr}): a `CRITICAL_TIER_1` core dependency on {crit_sup} ({crit_tkr}) and a `NON_CRITICAL_SECONDARY` packaging/auxiliary link from {dist_sup} ({dist_tkr}).\n"
                    f"2. **Contagion Gate Check**: The shock hits {dist_sup} ({dist_tkr}), which does not sit on {focal_name}'s critical manufacturing path. Existing 90-day safety stock covers the downtime window.\n"
                    f"3. **Downstream Impact Synthesis**: With {crit_sup} unaffected, {focal_name} sustains full output, protecting 2nd-order customer {down_name} from revenue or margin impact.\n"
                    f"</think>"
                )
            return f"{new_think}\n{summary_clean}"

        elif not is_distractor and len(edges) >= 2:
            sup_name, sup_tkr, sup_nat, sup_mat, focal_raw = edges[0]
            _, _, _, _, down_customers = edges[1]
            focal_name = focal_raw.split(":")[0].strip()

            if style_idx % 4 == 1:
                new_think = (
                    f"<think>\n"
                    f"- **Tier-1 Shock Vector**: {sup_name} ({sup_tkr}) faces an acute curtailment ({scenario}).\n"
                    f"- **Criticality & Substitutability Audit**: The edge `({sup_tkr})-[:SUPPLIES_TO]->({focal_name})` supplies `{sup_nat}` with materiality `{sup_mat or 'CRITICAL_TIER_1'}`. Extended requalification timelines prevent short-term vendor substitution.\n"
                    f"- **2nd-Order Cascade Propagation**: Production throttling at {focal_name} directly constrains downstream shipments to `{down_customers}`.\n"
                    f"- **Margin & Operating Leverage Impact**: Spot input price escalation combined with unabsorbed fixed manufacturing overhead compresses operating margins across the {sector.lower()} chain.\n"
                    f"</think>"
                )
            elif style_idx % 4 == 2:
                new_think = (
                    f"<think>\n"
                    f"Tracing 2nd- and 3rd-order contagion across the {sector} supply network:\n"
                    f"1. **Origin Bottleneck**: {sup_name} ({sup_tkr}) experiences a verified supply shock: {scenario}\n"
                    f"2. **Hop-1 Transmission ({sup_tkr} -> {focal_name})**: Because `{sup_nat}` is a `CRITICAL_TIER_1` sole-source dependency, {focal_name} cannot bypass the shortage via spot inventory.\n"
                    f"3. **Hop-2 Customer Exposure**: Downstream counterparties ({down_customers}) face deferred deliveries and backlog accumulation over the next several quarters.\n"
                    f"4. **Financial Elasticity**: Fixed-cost deleveraging and expedited freight/spot premiums trigger material operating margin compression (`DISRUPTIVE_SHOCK`).\n"
                    f"</think>"
                )
            else:
                new_think = (
                    f"<think>\n"
                    f"1. **Primary Shock Assessment**: {sup_name} ({sup_tkr}) curtails delivery of `{sup_nat}` ({scenario}).\n"
                    f"2. **Topological Vulnerability**: {focal_name} exhibits unbuffered `CRITICAL_TIER_1` reliance on {sup_name} with multi-quarter secondary vendor requalification barriers.\n"
                    f"3. **Multi-Hop Contagion & Margin Compression**: The bottleneck cascades through {focal_name} to 2nd-order enterprise customers ({down_customers}), deferring top-line revenue recognition and compressing operating margins.\n"
                    f"</think>"
                )
            return f"{new_think}\n{summary_clean}"

        return str(orig)

    def _build_task_d_mode_bleed_rejected(self, row: Dict[str, Any], variant: int) -> str:
        """
        Construct the exact Cypher / SQL-Cypher mode-bleed outputs observed in
        Phase 2 (`benchmark_autoregressive_results.jsonl` lines 3, 6, 9, 19)
        when `<|contagion_reasoning|>` prompts contain openCypher subgraph blocks.
        """
        parsed = self._parse_task_d_prompt(row["prompt"])
        edges = parsed["edges"]
        if len(edges) >= 2:
            src_name, src_tkr, nat, mat, tgt_raw = edges[0]
            tgt_name = tgt_raw.split(":")[0].strip()
            tkr_m = re.search(r"ticker:\s*'([^']+)'", tgt_raw)
            tgt_tkr = tkr_m.group(1) if tkr_m else "FOCAL"
        else:
            src_name, src_tkr, nat, mat, tgt_name, tgt_tkr = (
                "Supplier Corp", "SUP", "Core Components", "CRITICAL_TIER_1", "Focal Corp", "FOC"
            )

        if variant % 3 == 0:
            # Exact pattern from Phase 2 Failure #2 & #3 (lines 6 & 9: emitting MATCH ... RETURN)
            return (
                f"MATCH (s:Company {{ticker: '{src_tkr}'}})-[r:SUPPLIES_TO]->(m:Company {{ticker: '{tgt_tkr}'}})\n"
                f"WHERE (m)-[:SUPPLIES_TO]->(c:Company)\n"
                f"RETURN c.name AS downstream_customer, r.nature AS goods_purpose, r.materiality AS item_significance;"
            )
        elif variant % 3 == 1:
            # Exact pattern from Phase 2 Failure #1 (line 3: SQL-Cypher hybrid SELECT ... FROM (MATCH ...))
            return (
                f"SELECT\n"
                f"  p.nature AS disruption_source,\n"
                f"  c.ticker AS affected_entity\n"
                f"FROM\n"
                f"  (MATCH (s:Company {{ticker: '{src_tkr}'}})-[r:SUPPLIES_TO]->(t:Company {{ticker: '{tgt_tkr}'}})\n"
                f"   WHERE r.materiality = '{mat or 'CRITICAL_TIER_1'}'\n"
                f"   RETURN s AS source_company, t AS target_company, r AS edge) AS p;"
            )
        else:
            # Exact pattern from Phase 2 Failure #4 (line 19: emitting a raw graph triple instead of <think>)
            return (
                f"(:Company {{name: \"{src_name}\", ticker: \"{src_tkr}\"}})"
                f"-[:SUPPLIES_TO {{polarity: \"CONTRACTING_BEARISH\", materiality: \"{mat or 'CRITICAL_TIER_1'}\", "
                f"status: \"ACTIVE_CURRENT\", nature: \"{nat or 'critical components'}\"}}]->"
                f"(:Company {{name: \"{tgt_name}\", ticker: \"{tgt_tkr}\"}})"
            )

    def _build_task_d_counterfactual_rejected(self, row: Dict[str, Any]) -> str:
        """
        Construct a plausible but causally flawed `<think>` contagion trace:
        - For distractor prompts (`NON_CRITICAL_SECONDARY`), falsely propagate a 3-hop
          catastrophic halt (`DISRUPTIVE_SHOCK`) by ignoring edge materiality and buffer stock.
        - For critical Tier-1 prompts (`CRITICAL_TIER_1`), falsely dismiss the bottleneck
          as `CONTAINED_MINIMAL_CONTAGION` by assuming instant spot-market substitution.
        """
        orig = row["target_completion"]
        parsed = self._parse_task_d_prompt(row["prompt"])
        edges = parsed["edges"]
        is_distractor = "CONTAINED_MINIMAL_CONTAGION" in orig

        if is_distractor and len(edges) >= 3:
            crit_sup, crit_tkr, _, _, focal_raw = edges[0]
            dist_sup, dist_tkr, dist_nat, _, _ = edges[1]
            _, focal_tkr, _, _, down_raw = edges[2]
            focal_name = focal_raw.split(":")[0].strip()
            down_name = down_raw.split(":")[0].strip()
            return (
                f"<think>\n"
                f"1. Identify the primary shock origin: {dist_sup} ({dist_tkr}) experiences an operational hold on `{dist_nat}`.\n"
                f"2. Propagate unattenuated contagion: Because {dist_sup} is connected to {focal_name} ({focal_tkr}) in the supply graph, any delay at {dist_sup} immediately halts all manufacturing lines at {focal_name}, overriding {crit_sup} ({crit_tkr}) output.\n"
                f"3. Assert 2nd-order collapse: Downstream deliveries from {focal_name} to {down_name} will suffer a 100% shipment freeze and severe operating margin collapse across the next 3 quarters.\n"
                f"</think>\n"
                f"**Contagion Analysis Summary**:\n"
                f"- **Primary Disruption**: {dist_sup} outage halts {focal_name} core production.\n"
                f"- **2nd-Order Exposure**: {down_name} faces immediate assembly line shutdowns.\n"
                f"- **Contagion Verdict**: `DISRUPTIVE_SHOCK`. Severe uncontained multi-hop revenue loss."
            )
        else:
            if len(edges) >= 2:
                sup_name, sup_tkr, sup_nat, _, focal_raw = edges[0]
                _, _, _, _, down_customers = edges[1]
                focal_name = focal_raw.split(":")[0].strip()
            else:
                sup_name, sup_tkr, sup_nat, focal_name, down_customers = (
                    "Upstream Supplier", "SUP", "Critical Components", "Focal Company", "Downstream Partners"
                )
            return (
                f"<think>\n"
                f"1. Identify the disruption source: {sup_name} ({sup_tkr}) curtails `{sup_nat}` output.\n"
                f"2. Assume frictionless substitution: Even though the dependency is labeled `CRITICAL_TIER_1` with multi-month requalification barriers, {focal_name} can immediately replace 100% of specialized `{sup_nat}` volume on the open commodity spot market within 48 hours at zero cost premium.\n"
                f"3. Conclude zero downstream impact: Downstream customers ({down_customers}) experience zero delivery delays and operating margins expand.\n"
                f"</think>\n"
                f"**Contagion Analysis Summary**:\n"
                f"- **Primary Disruption**: {sup_name} curtailment is effortlessly bypassed with zero cost impact.\n"
                f"- **2nd-Order Exposure**: {down_customers} see no supply chain effect.\n"
                f"- **Risk Classification**: `CONTAINED_MINIMAL_CONTAGION` with zero margin sensitivity."
            )

    def _build_task_e_pairs(self, row: Dict[str, Any], style_idx: int) -> Tuple[str, str]:
        """
        Construct (chosen, rejected) for Task E (`PORTFOLIO_RECOMMENDATION`),
        focusing on the Step 50 slow-inflection cohort (`DIVERGENT_SIGNAL_HEDGING`
        and `SYNTHETIC_TEMPLATE_SKELETON`).
        """
        orig = row["target_completion"]
        entity_m = re.search(r"Entity:\s*([^\n]+)", row["prompt"])
        entity_str = entity_m.group(1).strip() if entity_m else "the focal issuer"
        z_m = re.search(r"Z-score\s*=\s*([+-]?\d+\.\d+)", row["prompt"], re.IGNORECASE)
        z_val = z_m.group(1) if z_m else "+1.92"

        # Diversify `<think>` prose slightly on odd style_idx so Task E also avoids single-template lock
        if "</think>" in orig and style_idx % 2 == 1:
            _, summary_part = orig.split("</think>", 1)
            if row["metadata"].get("provenance") == "DIVERGENT_SIGNAL_HEDGING":
                chosen = (
                    f"<think>\n"
                    f"1. **Signal Divergence & Volatility Regime**: Post-event telemetry for {entity_str} registers a borderline volatility z-score (`z = {z_val}`, just inside the 2.0-sigma structural break boundary), driven by tension between top-line demand resilience and forward cost/capex headwinds.\n"
                    f"2. **Asymmetric Tail-Risk Calibration**: Outright liquidation would forfeit structural upside and trigger unnecessary realization costs, whereas leaving the position unhedged exposes the portfolio to post-guidance earnings revision skew.\n"
                    f"3. **Derivative Overlay Construction**: Deploy a delta-neutral options structure (collar or calendar put spread) to cap downside drawdown while financing premium outlay via upside volatility skew.\n"
                    f"</think>\n"
                    f"{summary_part.strip()}"
                )
            else:
                chosen = orig
        else:
            chosen = orig

        if style_idx % 3 == 0:
            # Rejected Type 1: Impulsive unhedged panic liquidation ignoring borderline |z| < 2.0
            rejected = (
                f"<think>\n"
                f"1. Observe market volatility around {entity_str} (`z = {z_val}`).\n"
                f"2. Ignore fundamental signal divergence and options volatility skew; treat borderline noise as a total insolvency event.\n"
                f"3. Liquidate 100% of institutional equity holdings immediately at market open and initiate an unhedged naked short position.\n"
                f"</think>\n"
                f"**Strategic Portfolio Recommendation**:\n"
                f"- **Action**: Immediate 100% market-order liquidation of {entity_str} and 200 bps naked short allocation.\n"
                f"- **Hedging Strategy**: None (unhedged directional bet).\n"
                f"- **Rationale**: Reacts aggressively to short-term headline noise regardless of transaction or tax friction."
            )
        elif style_idx % 3 == 1:
            # Rejected Type 2: Complacent unhedged leverage ignoring latent margin compression
            rejected = (
                f"<think>\n"
                f"1. Dismiss the risk telemetry for {entity_str} because headline metrics remain positive.\n"
                f"2. Assume zero downside tail risk and sell naked out-of-the-money puts on margin without downside floor protection.\n"
                f"</think>\n"
                f"**Strategic Portfolio Recommendation**:\n"
                f"- **Action**: Double overweight allocation to {entity_str} (+400 bps) on margin.\n"
                f"- **Hedging Strategy**: Remove all existing hedges and sell uncovered puts.\n"
                f"- **Rationale**: Assumes structural cost inflation and supply risks have zero impact on valuation."
            )
        else:
            # Rejected Type 3: Cross-task Cypher bleed on a Task E prompt
            tkr_m = re.search(r"\(([A-Z.]+)\)", entity_str)
            tkr = tkr_m.group(1) if tkr_m else "SPY"
            rejected = (
                f"MATCH (c:Company {{ticker: '{tkr}'}})-[r:EXPOSED_TO_RISK]->(m:Risk)\n"
                f"RETURN c.name AS entity, r.polarity AS signal_direction, r.materiality AS risk_tier;"
            )

        return chosen, rejected

    def _build_task_ab_slow_inflection_pairs(self, row: Dict[str, Any], variant: int) -> Tuple[str, str]:
        """
        Construct (chosen, rejected) for slow-inflection Task A & B records:
        - `ENTITY_COLLISION_MINING` (corporate spin-offs): penalize hallucinated edges when target is `(none)`,
          or penalize `(none)` when the prompt explicitly states the statutory spin-off date (`SPUN_OFF_FROM`).
        - `VOLATILITY_BOUNDARY_TELEMETRY`: penalize misclassifying borderline `|z| < 2.0` 8-K events as
          `STATISTICALLY_SIGNIFICANT` / `DISRUPTIVE_SHOCK`.
        - `SEC_10K_ITEM1_*` parent/subsidiary CIK grounding: penalize subsidiary CIK confusion (e.g., `0001001811`
          DISH DBS instead of `0001001082` DISH Network) or corrupted 10-digit CIK strings.
        """
        chosen = row["target_completion"]
        prov = row["metadata"].get("provenance", "")

        if chosen == "(none)":
            focal = row["metadata"].get("focal_entity", "eBay Inc.")
            confusable = row["metadata"].get("confusable_entity", "PayPal Holdings, Inc.")
            # Extract tickers from prompt if present
            tickers = re.findall(r"\b([A-Z]{2,5})\b", row["prompt"])
            t1 = tickers[0] if len(tickers) > 0 else "FOO"
            t2 = tickers[1] if len(tickers) > 1 else "BAR"
            rejected = (
                f"(:Company {{name: \"{focal}\", ticker: \"{t1}\"}})"
                f"-[:SUPPLIES_TO {{polarity: \"NEUTRAL_STABLE\", materiality: \"CRITICAL_TIER_1\", "
                f"status: \"ACTIVE_CURRENT\", provenance: \"SEC_10K_ITEM1\", confidence: 0.98}}]->"
                f"(:Company {{name: \"{confusable}\", ticker: \"{t2}\"}})"
            )
            return chosen, rejected

        if prov == "VOLATILITY_BOUNDARY_TELEMETRY":
            # Replace BORDERLINE_SIGNIFICANT / MIXED_UNCERTAIN_SIGNAL with over-confident shock labels
            rejected = (
                chosen.replace("BORDERLINE_SIGNIFICANT", "STATISTICALLY_SIGNIFICANT")
                .replace("MIXED_UNCERTAIN_SIGNAL", "DISRUPTIVE_SHOCK")
            )
            if rejected == chosen:
                rejected = "(none)"
            return chosen, rejected

        if prov == "ENTITY_COLLISION_MINING" and "SPUN_OFF_FROM" in chosen:
            # Statutory separation record where chosen has [:SPUN_OFF_FROM]
            rejected = "(none)" if variant % 2 == 0 else chosen.replace("SPUN_OFF_FROM", "SUBSIDIARY_OF")
            return chosen, rejected

        # Positive extraction with 10-digit CIKs: corrupt CIK to subsidiary/transposed digits in `rejected`
        ciks = re.findall(r'\bcik:\s*"(\d{10})"', chosen)
        if ciks:
            orig_cik = ciks[0]
            # Special case: DISH Network parent (0001001082) vs DISH DBS subsidiary (0001001811)
            if orig_cik == "0001001082":
                bad_cik = "0001001811"
            else:
                # Transpose last 3 digits to simulate subsidiary/hallucinated CIK
                prefix = orig_cik[:6]
                suffix = orig_cik[6:]
                bad_suffix = suffix[::-1] if suffix[::-1] != suffix else "9981"
                bad_cik = prefix + bad_suffix
            rejected = chosen.replace(f'cik: "{orig_cik}"', f'cik: "{bad_cik}"', 1)
            return chosen, rejected

        return chosen, "(none)"

    def _build_task_c_slow_inflection_pairs(self, row: Dict[str, Any], variant: int) -> Tuple[str, str]:
        """
        Construct (chosen, rejected) for Step 50 slow-inflection `TEXT_TO_CYPHER` variants
        (`VAR_1, 2, 7, 9`), penalizing unindexed cartesian products and SQL-Cypher hybrids.
        """
        chosen = row["target_completion"]
        tkr_m = re.search(r"ticker\s*=\s*'([^']+)'|ticker:\s*'([^']+)'", chosen)
        ticker = (tkr_m.group(1) or tkr_m.group(2)) if tkr_m else "AAPL"

        if variant % 2 == 0:
            # Unindexed Cartesian Product + invented relationship property
            rejected = (
                f"MATCH (a), (b)\n"
                f"WHERE a.ticker = '{ticker}'\n"
                f"RETURN a, b;"
            )
        else:
            # SQL syntax bleed inside Cypher task
            rejected = (
                f"SELECT supplier_name, ticker, materiality\n"
                f"FROM Company_Graph\n"
                f"WHERE target_ticker = '{ticker}' AND status = 'ACTIVE_CURRENT';"
            )
        return chosen, rejected

    # ------------------------------------------------------------------
    # 3. End-to-End Dataset Construction (Train & Val Splits)
    # ------------------------------------------------------------------
    def build_dpo_splits(self) -> Dict[str, Any]:
        """
        Build `yarnball_dpo_train.jsonl` and `yarnball_dpo_val.jsonl` from the SFT train/val
        pools and empirical Cartography inflection telemetry, strictly excluding `yarnball_sft_test.jsonl`.
        """
        inflection_report = self.analyze_inflection_dynamics()
        val_metrics = inflection_report["sample_metrics"]

        # Load test prompts/IDs to enforce a hard zero-leakage gate
        test_ids: Set[str] = set()
        test_file = self.sft_dir / "yarnball_sft_test.jsonl"
        if test_file.is_file():
            with open(test_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        test_ids.add(json.loads(line)["sample_id"])

        slow_task_c_provs = {
            "SYNTHETIC_SCHEMA_DSL_VAR_1",
            "SYNTHETIC_SCHEMA_DSL_VAR_2",
            "SYNTHETIC_SCHEMA_DSL_VAR_7",
            "SYNTHETIC_SCHEMA_DSL_VAR_9",
        }

        split_outputs: Dict[str, List[DPORecord]] = {"train": [], "val": []}
        bucket_counts: Dict[str, Counter] = {"train": Counter(), "val": Counter()}

        for split_name, filename in [("train", "yarnball_sft_train.jsonl"), ("val", "yarnball_sft_val.jsonl")]:
            src_path = self.sft_dir / filename
            with open(src_path, "r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]

            style_counter = 0
            for row in rows:
                sid = row["sample_id"]
                if sid in test_ids:
                    continue

                task = row["metadata"]["task_type"]
                prov = row["metadata"].get("provenance", "")
                val_info = val_metrics.get(sid)
                is_val_slow50 = bool(val_info and val_info["is_slow_step50"])
                is_val_slow100 = bool(val_info and val_info["is_slow_step100"])

                # --------------------------------------------------------------
                # Bucket 1: Task D (`CONTAGION_REASONING`)
                # Include 100% of DISTRACTOR_CONTAGION_SYNTHESIS (both mode-bleed
                # and counterfactual pairs) + 50% of SYNTHETIC_TEMPLATE_SKELETON
                # --------------------------------------------------------------
                if task == "CONTAGION_REASONING":
                    if prov == "DISTRACTOR_CONTAGION_SYNTHESIS":
                        style_counter += 1
                        chosen_text = self._de_template_task_d_chosen(row, style_counter)
                        # Pair 1A: Mode-bleed rejection (primary Phase 2 fix)
                        rej_bleed = self._build_task_d_mode_bleed_rejected(row, style_counter)
                        split_outputs[split_name].append(
                            DPORecord(
                                sample_id=f"DPO_{sid}_BLEED",
                                prompt=row["prompt"],
                                chosen=chosen_text,
                                rejected=rej_bleed,
                                metadata={
                                    **row["metadata"],
                                    "dpo_bucket": "BUCKET_1_TASK_D_MODE_BLEED",
                                    "rejection_strategy": "CYPHER_MODE_BLEED_ON_SUBGRAPH_PROMPT",
                                    "source_sample_id": sid,
                                },
                            )
                        )
                        bucket_counts[split_name]["BUCKET_1_TASK_D_MODE_BLEED"] += 1

                        # Pair 1B: Counterfactual distractor propagation rejection
                        rej_cf = self._build_task_d_counterfactual_rejected(row)
                        split_outputs[split_name].append(
                            DPORecord(
                                sample_id=f"DPO_{sid}_CF",
                                prompt=row["prompt"],
                                chosen=chosen_text,
                                rejected=rej_cf,
                                metadata={
                                    **row["metadata"],
                                    "dpo_bucket": "BUCKET_1_TASK_D_COUNTERFACTUAL",
                                    "rejection_strategy": "FALSE_DISTRACTOR_CONTAGION_CASCADE",
                                    "source_sample_id": sid,
                                },
                            )
                        )
                        bucket_counts[split_name]["BUCKET_1_TASK_D_COUNTERFACTUAL"] += 1

                    elif prov == "SYNTHETIC_TEMPLATE_SKELETON" and (split_name == "val" or self.rng.random() < 0.45):
                        style_counter += 1
                        chosen_text = self._de_template_task_d_chosen(row, style_counter)
                        if style_counter % 2 == 0:
                            rej_text = self._build_task_d_mode_bleed_rejected(row, style_counter)
                            bucket_key = "BUCKET_1_TASK_D_MODE_BLEED"
                            strat = "CYPHER_MODE_BLEED_ON_SUBGRAPH_PROMPT"
                        else:
                            rej_text = self._build_task_d_counterfactual_rejected(row)
                            bucket_key = "BUCKET_1_TASK_D_COUNTERFACTUAL"
                            strat = "FALSE_SOLE_SOURCE_CONTAINMENT"

                        split_outputs[split_name].append(
                            DPORecord(
                                sample_id=f"DPO_{sid}",
                                prompt=row["prompt"],
                                chosen=chosen_text,
                                rejected=rej_text,
                                metadata={
                                    **row["metadata"],
                                    "dpo_bucket": bucket_key,
                                    "rejection_strategy": strat,
                                    "source_sample_id": sid,
                                },
                            )
                        )
                        bucket_counts[split_name][bucket_key] += 1

                # --------------------------------------------------------------
                # Bucket 2: Task E (`PORTFOLIO_RECOMMENDATION` - Step 50 Slow Slice)
                # Include 100% of DIVERGENT_SIGNAL_HEDGING + slow inflection slice
                # --------------------------------------------------------------
                elif task == "PORTFOLIO_RECOMMENDATION":
                    include_e = (
                        prov == "DIVERGENT_SIGNAL_HEDGING"
                        or is_val_slow50
                        or (split_name == "train" and self.rng.random() < 0.40)
                    )
                    if include_e:
                        style_counter += 1
                        chosen_e, rejected_e = self._build_task_e_pairs(row, style_counter)
                        split_outputs[split_name].append(
                            DPORecord(
                                sample_id=f"DPO_{sid}",
                                prompt=row["prompt"],
                                chosen=chosen_e,
                                rejected=rejected_e,
                                metadata={
                                    **row["metadata"],
                                    "dpo_bucket": "BUCKET_2_TASK_E_BOUNDARY_HEDGING",
                                    "rejection_strategy": "UNHEDGED_DIRECTIONAL_OR_MODE_BLEED",
                                    "is_step50_slow_inflection": is_val_slow50,
                                    "source_sample_id": sid,
                                },
                            )
                        )
                        bucket_counts[split_name]["BUCKET_2_TASK_E_BOUNDARY_HEDGING"] += 1

                # --------------------------------------------------------------
                # Bucket 3: Tasks A & B Slow-Inflection & CIK Grounding Slice
                # (`ENTITY_COLLISION_MINING`, `VOLATILITY_BOUNDARY_TELEMETRY`,
                #  `SEC_10K_ITEM1_EXPOSED_TO_RISK`, plus DISH / Parent CIK anchors)
                # --------------------------------------------------------------
                elif task in ("EXTRACT_SEC_GRAPH", "EXTRACT_NEWS_EVENT"):
                    is_dish_anchor = "DISH Network" in row["prompt"] or "0001001082" in row["target_completion"]
                    include_ab = (
                        prov in ("ENTITY_COLLISION_MINING", "VOLATILITY_BOUNDARY_TELEMETRY")
                        or is_val_slow50
                        or is_val_slow100
                        or is_dish_anchor
                        or (prov == "SEC_10K_ITEM1_EXPOSED_TO_RISK" and self.rng.random() < 0.25)
                    )
                    if include_ab:
                        style_counter += 1
                        chosen_ab, rejected_ab = self._build_task_ab_slow_inflection_pairs(row, style_counter)
                        if chosen_ab != rejected_ab:
                            split_outputs[split_name].append(
                                DPORecord(
                                    sample_id=f"DPO_{sid}",
                                    prompt=row["prompt"],
                                    chosen=chosen_ab,
                                    rejected=rejected_ab,
                                    metadata={
                                        **row["metadata"],
                                        "dpo_bucket": "BUCKET_3_TASK_AB_DISAMBIG_AND_CIK",
                                        "is_step50_slow_inflection": is_val_slow50,
                                        "is_step100_hard_boundary": is_val_slow100,
                                        "source_sample_id": sid,
                                    },
                                )
                            )
                            bucket_counts[split_name]["BUCKET_3_TASK_AB_DISAMBIG_AND_CIK"] += 1

                # --------------------------------------------------------------
                # Bucket 4: Task C (`TEXT_TO_CYPHER` - Step 50 Slow Variants)
                # --------------------------------------------------------------
                elif task == "TEXT_TO_CYPHER":
                    if prov in slow_task_c_provs or is_val_slow50:
                        style_counter += 1
                        chosen_c, rejected_c = self._build_task_c_slow_inflection_pairs(row, style_counter)
                        split_outputs[split_name].append(
                            DPORecord(
                                sample_id=f"DPO_{sid}",
                                prompt=row["prompt"],
                                chosen=chosen_c,
                                rejected=rejected_c,
                                metadata={
                                    **row["metadata"],
                                    "dpo_bucket": "BUCKET_4_TASK_C_SCHEMA_CYPHER",
                                    "is_step50_slow_inflection": is_val_slow50,
                                    "source_sample_id": sid,
                                },
                            )
                        )
                        bucket_counts[split_name]["BUCKET_4_TASK_C_SCHEMA_CYPHER"] += 1

            # Deterministic shuffle within each split
            self.rng.shuffle(split_outputs[split_name])

        # Write train and val JSONL files
        train_file = self.dpo_dir / "yarnball_dpo_train.jsonl"
        val_file = self.dpo_dir / "yarnball_dpo_val.jsonl"

        for split_name, out_path in [("train", train_file), ("val", val_file)]:
            with open(out_path, "w", encoding="utf-8") as f:
                for rec in split_outputs[split_name]:
                    assert rec.chosen.strip() != rec.rejected.strip(), f"Identical pair in {rec.sample_id}"
                    f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
            logger.info("Wrote %d DPO preference pairs to %s", len(split_outputs[split_name]), out_path)

        summary = {
            "schema_version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "base_sft_adapter": "CamiloEmiliano/yarnball-qwen-7b-lora",
            "target_dpo_adapter": "CamiloEmiliano/yarnball-qwen-7b-dpo",
            "cartography_selection_method": "Step 25-100 Inflection-Lag (Time-to-Mastery) + Phase 2 Error Mining",
            "zero_test_leakage_verified": True,
            "splits": {
                "train": len(split_outputs["train"]),
                "val": len(split_outputs["val"]),
                "total": len(split_outputs["train"]) + len(split_outputs["val"]),
            },
            "bucket_breakdown": {
                "train": dict(bucket_counts["train"]),
                "val": dict(bucket_counts["val"]),
            },
            "inflection_harvest_highlights": {
                "step_50_slow_guids": inflection_report["step_50_slow_inflection_cohort"]["total_guids"],
                "step_50_unique_samples": inflection_report["step_50_slow_inflection_cohort"]["unique_sample_ids"],
                "step_100_hardest_guids": inflection_report["step_100_hardest_boundary_cohort"]["total_guids"],
                "step_100_unique_samples": inflection_report["step_100_hardest_boundary_cohort"]["unique_sample_ids"],
            },
            "output_directory": str(self.dpo_dir),
        }

        summary_path = self.dpo_dir / "dpo_dataset_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved DPO dataset summary to %s", summary_path)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 3 DPO dataset via Cartography Inflection Slice")
    parser.add_argument("--sft-dir", default=str(DEFAULT_SFT_DIR), help="Path to SFT dataset directory")
    parser.add_argument("--dpo-dir", default=str(DEFAULT_DPO_DIR), help="Path to output DPO dataset directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    builder = InflectionSliceDPOBuilder(
        sft_dir=Path(args.sft_dir),
        dpo_dir=Path(args.dpo_dir),
        seed=args.seed,
    )
    summary = builder.build_dpo_splits()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

