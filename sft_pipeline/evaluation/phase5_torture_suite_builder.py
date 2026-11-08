"""
Deterministic Suite Builder for Phase 5 Diagnostic Adversarial Torture Test (150 OOD Samples).

Constructs 25 samples across each of the 6 orthogonal diagnostic stress axes (`TT-01` .. `TT-06`)
grounded strictly in real S&P 500 constituents and 10-digit SEC CIKs from
`data/sp500_constituents_historical.json`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UNIVERSE_PATH = PROJECT_ROOT / "data" / "sp500_constituents_historical.json"
DEFAULT_SUITE_PATH = PROJECT_ROOT / "data" / "evaluation" / "phase5_torture_suite.jsonl"


def _load_sp500_pool() -> List[Dict[str, Any]]:
    raw = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
    seen_tickers = set()
    clean_pool = []
    for row in raw:
        tkr = str(row.get("ticker", "")).strip()
        cik = str(row.get("cik", "")).strip()
        name = str(row.get("company_name", "")).strip()
        if tkr and len(cik) == 10 and cik.isdigit() and name and tkr not in seen_tickers:
            seen_tickers.add(tkr)
            clean_pool.append(
                {
                    "ticker": tkr,
                    "cik": cik,
                    "name": name,
                    "sector": row.get("gics_sector") or "Industrials",
                    "sub_industry": row.get("gics_sub_industry") or "Industrial Machinery",
                }
            )
    return clean_pool


def _build_boilerplate_block(rng: random.Random, target_words: int = 1600) -> str:
    """Generate realistic SEC Form 10-K / Item 7 GAAP financial and tabular boilerplate."""
    paragraphs = [
        (
            "In accordance with Accounting Standards Codification (ASC) Topic 606, Revenue from Contracts "
            "with Customers, the Company recognizes revenue upon transfer of control of promised goods or "
            "services to customers in an amount that reflects the consideration the Company expects to receive. "
            "Deferred revenue liabilities represent billings or cash received in excess of revenue recognized "
            "on multi-year maintenance and support arrangements."
        ),
        (
            "| Fiscal Quarter | Gross Margin (%) | R&D Expense (USD M) | SG&A (USD M) | Operating Cash Flow (USD M) |\n"
            "| :--- | :---: | :---: | :---: | :---: |\n"
            "| Q1 FY2024 | 54.2% | 612.4 | 418.9 | 1,104.2 |\n"
            "| Q2 FY2024 | 53.8% | 629.1 | 425.0 | 1,088.7 |\n"
            "| Q3 FY2024 | 55.1% | 645.8 | 431.2 | 1,192.5 |\n"
            "| Q4 FY2024 | 54.9% | 660.3 | 439.6 | 1,241.0 |"
        ),
        (
            "Management periodically evaluates long-lived tangible and finite-lived intangible assets for "
            "impairment whenever events or changes in circumstances indicate that the carrying amount of an "
            "asset group may not be recoverable under ASC Topic 360. Foreign currency translation adjustments "
            "arising from European and Asia-Pacific operating subsidiaries are recorded within accumulated "
            "other comprehensive income (loss) on the consolidated balance sheets."
        ),
        (
            "Interest rate swaps and foreign exchange forward contracts are designated as cash flow hedges "
            "under ASC Topic 815 where hedge effectiveness criteria are met at inception and on an ongoing "
            "quarterly basis. Counterparty credit risk is monitored across money-center banking institutions "
            "holding investment-grade ratings."
        ),
        (
            "Lease right-of-use assets and corresponding operating lease liabilities are recognized at the "
            "commencement date based on the present value of future minimum lease payments over the lease term "
            "utilizing the Company's incremental borrowing rate pursuant to ASC Topic 842."
        ),
    ]
    words: List[str] = []
    idx = 0
    while len(words) < target_words:
        block = paragraphs[idx % len(paragraphs)]
        if idx > 0 and idx % 5 == 0:
            rng_year = rng.choice([2021, 2022, 2023, 2024])
            block = f"[Note {rng.randint(3, 18)} — Supplemental Consolidated Schedule FY{rng_year}]\n" + block
        words.extend(block.split())
        words.append("\n\n")
        idx += 1
    return " ".join(words)


# ---------------------------------------------------------------------------
# Axis 1 (TT-01): Hop-Depth & "Needle-in-a-Subgraph" Clutter Cliff (25)
# ---------------------------------------------------------------------------
def _generate_tt01_samples(pool: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    hops_grid = [2, 3, 4, 5]
    edge_counts = [10, 25, 50, 80, 120]
    placements = ["HEAD", "MIDDLE", "TAIL"]
    components = [
        "EUV Lithography Optics",
        "High-Bandwidth Memory Substrates",
        "Sterile Bioreactor Filters",
        "High-Voltage SiC Power Modules",
        "Subsea Production Trees",
    ]

    idx = 0
    # 20 Cypher grid samples (4 hop depths x 5 edge counts)
    for k in hops_grid:
        for num_edges in edge_counts:
            placement = placements[idx % len(placements)]
            chosen = rng.sample(pool, k + 1 + min(len(pool) - (k + 1), max(12, num_edges * 2)))
            chain_nodes = chosen[: k + 1]
            distractor_pool = chosen[k + 1 :]
            comp = components[idx % len(components)]

            chain_edges = []
            for hop_i in range(k):
                u = chain_nodes[hop_i]
                v = chain_nodes[hop_i + 1]
                chain_edges.append(
                    f"({u['name']}:Company {{ticker: '{u['ticker']}'}})"
                    f"-[:SUPPLIES_TO {{nature: '{comp}', materiality: 'CRITICAL_TIER_1', status: 'ACTIVE_CURRENT'}}]->"
                    f"({v['name']}:Company {{ticker: '{v['ticker']}'}})"
                )

            num_distractors = max(0, num_edges - k)
            distractor_edges = []
            for d_i in range(num_distractors):
                du = distractor_pool[(d_i * 2) % len(distractor_pool)]
                dv = distractor_pool[(d_i * 2 + 1) % len(distractor_pool)]
                if du["ticker"] == dv["ticker"]:
                    dv = distractor_pool[(d_i * 2 + 3) % len(distractor_pool)]
                mat = "NON_CRITICAL_SECONDARY" if d_i % 2 == 0 else "COMMODITY_TIER_3"
                distractor_edges.append(
                    f"({du['name']}:Company {{ticker: '{du['ticker']}'}})"
                    f"-[:SUPPLIES_TO {{nature: 'Office & Packaging Consumables', materiality: '{mat}', status: 'ACTIVE_CURRENT'}}]->"
                    f"({dv['name']}:Company {{ticker: '{dv['ticker']}'}})"
                )

            if placement == "HEAD":
                all_edges = chain_edges + distractor_edges
            elif placement == "TAIL":
                all_edges = distractor_edges + chain_edges
            else:
                mid = len(distractor_edges) // 2
                shuffled_chain = list(chain_edges)
                rng.shuffle(shuffled_chain)
                all_edges = distractor_edges[:mid] + shuffled_chain + distractor_edges[mid:]

            origin = chain_nodes[0]
            target = chain_nodes[-1]
            req_entities = [f"{n['ticker']}|{n['name']}" for n in chain_nodes[1:]]

            prompt = (
                f"<|contagion_reasoning|>\n"
                f"Sector: {origin['sector']}\n"
                f"Scenario: A 55% acute production halt at {origin['name']} ({origin['ticker']}) following a cleanroom "
                f"fabrication fire, imposing a 9-month requalification freeze on {comp} with zero alternate Tier-1 capacity.\n"
                f"Knowledge Graph Subgraph Context:\n"
                + "\n".join(all_edges)
                + f"\n\nQuestion: Trace the {k}-hop supply chain contagion path from {origin['name']} ({origin['ticker']}) "
                f"and determine whether production and operating margins at {target['name']} ({target['ticker']}) suffer a disruptive shock."
            )

            samples.append(
                {
                    "sample_id": f"TT01_GRID_K{k}_E{num_edges:03d}_{placement}_{idx+1:02d}",
                    "axis_id": "TT-01",
                    "sub_cohort": f"CYPHER_K{k}_E{num_edges}_{placement}",
                    "task_type": "CONTAGION_REASONING",
                    "metadata": {
                        "hops": k,
                        "edge_count": num_edges,
                        "needle_placement": placement,
                        "graph_format": "RAW_CYPHER",
                    },
                    "prompt": prompt,
                    "expected_assertions": {
                        "expected_verdict": "DISRUPTIVE_SHOCK",
                        "required_bridge_entities": req_entities,
                        "allowed_ciks": [n["cik"] for n in chosen],
                    },
                }
            )
            idx += 1

    # 5 Paired BFS-Verbalized Arrow Ablation samples (k=4, |E|=80, middle-style distractor load)
    for v_idx in range(5):
        k = 4
        num_edges = 80
        chosen = rng.sample(pool, len(pool))
        chain_nodes = chosen[: k + 1]
        distractor_pool = chosen[k + 1 :]
        comp = components[v_idx % len(components)]

        verbal_lines = []
        for hop_i in range(k):
            u = chain_nodes[hop_i]
            v = chain_nodes[hop_i + 1]
            verbal_lines.append(
                f"Hop {hop_i+1} [CRITICAL_TIER_1 | ACTIVE_CURRENT]: "
                f"{u['name']} ({u['ticker']}) --> {v['name']} ({v['ticker']}) [Input: {comp}]"
            )
        for d_i in range(num_edges - k):
            du = distractor_pool[d_i % len(distractor_pool)]
            dv = distractor_pool[(d_i + 7 + (d_i // len(distractor_pool))) % len(distractor_pool)]
            if du["ticker"] == dv["ticker"]:
                dv = distractor_pool[(d_i + 1) % len(distractor_pool)]
            verbal_lines.append(
                f"Auxiliary Edge [NON_CRITICAL_SECONDARY]: "
                f"{du['name']} ({du['ticker']}) --> {dv['name']} ({dv['ticker']}) [Input: Packaging Consumables]"
            )

        origin = chain_nodes[0]
        target = chain_nodes[-1]
        prompt = (
            f"<|contagion_reasoning|>\n"
            f"Sector: {origin['sector']}\n"
            f"Scenario: A 55% acute production halt at {origin['name']} ({origin['ticker']}) following a cleanroom "
            f"fabrication fire, imposing a 9-month requalification freeze on {comp}.\n"
            f"Knowledge Graph Subgraph Context (BFS Topological Order):\n"
            + "\n".join(verbal_lines)
            + f"\n\nQuestion: Trace the {k}-hop supply chain contagion path from {origin['name']} ({origin['ticker']}) "
            f"and determine whether production and operating margins at {target['name']} ({target['ticker']}) suffer a disruptive shock."
        )
        samples.append(
            {
                "sample_id": f"TT01_BFS_VERBAL_K4_E080_{v_idx+1:02d}",
                "axis_id": "TT-01",
                "sub_cohort": "BFS_VERBALIZED_ABLATION",
                "task_type": "CONTAGION_REASONING",
                "metadata": {
                    "hops": k,
                    "edge_count": num_edges,
                    "needle_placement": "BFS_SORTED",
                    "graph_format": "BFS_VERBALIZED_ARROWS",
                },
                "prompt": prompt,
                "expected_assertions": {
                    "expected_verdict": "DISRUPTIVE_SHOCK",
                    "required_bridge_entities": [f"{n['ticker']}|{n['name']}" for n in chain_nodes[1:]],
                    "allowed_ciks": [n["cik"] for n in chosen],
                },
            }
        )

    return samples


# ---------------------------------------------------------------------------
# Axis 2 (TT-02): Diamond Convergence, Cycles & Stale TERMINATED Traps (25)
# ---------------------------------------------------------------------------
def _generate_tt02_samples(pool: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []

    # 9 Additive Parallel Diamond Shocks (35% + 45% = 80% combined loss -> DISRUPTIVE_SHOCK)
    for i in range(9):
        s1, s2, m1, m2, focal, downstream = rng.sample(pool, 6)
        prompt = (
            f"<|contagion_reasoning|>\n"
            f"Sector: {focal['sector']}\n"
            f"Scenario: A regional port and power blockade simultaneously halts shipments at BOTH {s1['name']} ({s1['ticker']}) "
            f"and {s2['name']} ({s2['ticker']}) for 6 months. {focal['name']} ({focal['ticker']}) splits its specialty substrate "
            f"procurement 40% via {m1['name']} ({m1['ticker']}) and 45% via {m2['name']} ({m2['ticker']}), and requires at least "
            f"75% substrate supply to keep assembly lines running above breakeven utilization.\n"
            f"Knowledge Graph Subgraph Context:\n"
            f"({s1['name']}:Company {{ticker: '{s1['ticker']}'}})-[:SUPPLIES_TO {{nature: 'Specialty Wafer Precursors (40% share)', materiality: 'MATERIAL_TIER_2', status: 'ACTIVE_CURRENT'}}]->({m1['name']}:Company {{ticker: '{m1['ticker']}'}})\n"
            f"({s2['name']}:Company {{ticker: '{s2['ticker']}'}})-[:SUPPLIES_TO {{nature: 'Specialty Wafer Precursors (45% share)', materiality: 'MATERIAL_TIER_2', status: 'ACTIVE_CURRENT'}}]->({m2['name']}:Company {{ticker: '{m2['ticker']}'}})\n"
            f"({m1['name']}:Company {{ticker: '{m1['ticker']}'}})-[:SUPPLIES_TO {{nature: 'Processed Substrates (40% share)', materiality: 'MATERIAL_TIER_2', status: 'ACTIVE_CURRENT'}}]->({focal['name']}:Company {{ticker: '{focal['ticker']}'}})\n"
            f"({m2['name']}:Company {{ticker: '{m2['ticker']}'}})-[:SUPPLIES_TO {{nature: 'Processed Substrates (45% share)', materiality: 'MATERIAL_TIER_2', status: 'ACTIVE_CURRENT'}}]->({focal['name']}:Company {{ticker: '{focal['ticker']}'}})\n"
            f"({focal['name']}:Company {{ticker: '{focal['ticker']}'}})-[:SUPPLIES_TO {{nature: 'Finished Systems', materiality: 'CRITICAL_TIER_1', status: 'ACTIVE_CURRENT'}}]->({downstream['name']}:Company {{ticker: '{downstream['ticker']}'}})\n\n"
            f"Question: Even though neither upstream branch is individually tagged CRITICAL_TIER_1, analyze the combined "
            f"diamond convergence impact on {focal['name']} ({focal['ticker']}) and {downstream['name']} ({downstream['ticker']})."
        )
        samples.append(
            {
                "sample_id": f"TT02A_DIAMOND_ADDITIVE_{i+1:02d}",
                "axis_id": "TT-02",
                "sub_cohort": "DIAMOND_ADDITIVE_SHOCK",
                "task_type": "CONTAGION_REASONING",
                "metadata": {"topology": "PARALLEL_DIAMOND_SUM"},
                "prompt": prompt,
                "expected_assertions": {
                    "expected_verdict": "DISRUPTIVE_SHOCK",
                    "required_bridge_entities": [
                        f"{m1['ticker']}|{m1['name']}",
                        f"{m2['ticker']}|{m2['name']}",
                        f"{focal['ticker']}|{focal['name']}",
                    ],
                },
            }
        )

    # 4 Single-Branch Buffered Diamond Controls (CONTAINED_MINIMAL_CONTAGION)
    for i in range(4):
        s1, s2, focal, downstream = rng.sample(pool, 4)
        prompt = (
            f"<|contagion_reasoning|>\n"
            f"Sector: {focal['sector']}\n"
            f"Scenario: A localized 3-week maintenance delay affects ONLY secondary vendor {s1['name']} ({s1['ticker']}, 20% share), "
            f"while primary dual-source supplier {s2['name']} ({s2['ticker']}, 80% share) operates at 100% capacity with 35% qualified "
            f"idle surge headroom and {focal['name']} ({focal['ticker']}) holds 90 days of interchangeable buffer stock.\n"
            f"Knowledge Graph Subgraph Context:\n"
            f"({s1['name']}:Company {{ticker: '{s1['ticker']}'}})-[:SUPPLIES_TO {{nature: 'Standard Connectors (20% share)', materiality: 'NON_CRITICAL_SECONDARY', status: 'ACTIVE_CURRENT'}}]->({focal['name']}:Company {{ticker: '{focal['ticker']}'}})\n"
            f"({s2['name']}:Company {{ticker: '{s2['ticker']}'}})-[:SUPPLIES_TO {{nature: 'Standard Connectors (80% share + surge headroom)', materiality: 'CRITICAL_TIER_1', status: 'ACTIVE_CURRENT'}}]->({focal['name']}:Company {{ticker: '{focal['ticker']}'}})\n"
            f"({focal['name']}:Company {{ticker: '{focal['ticker']}'}})-[:SUPPLIES_TO {{nature: 'Enterprise Assemblies', materiality: 'CRITICAL_TIER_1'}}]->({downstream['name']}:Company {{ticker: '{downstream['ticker']}'}})\n\n"
            f"Question: Analyze whether the outage at {s1['name']} ({s1['ticker']}) halts production at {focal['name']} ({focal['ticker']}) or delivery to {downstream['name']} ({downstream['ticker']})."
        )
        samples.append(
            {
                "sample_id": f"TT02A_DIAMOND_CONTROL_{i+1:02d}",
                "axis_id": "TT-02",
                "sub_cohort": "DIAMOND_DUAL_SOURCE_CONTROL",
                "task_type": "CONTAGION_REASONING",
                "metadata": {"topology": "DIAMOND_BUFFERED_CONTROL"},
                "prompt": prompt,
                "expected_assertions": {
                    "expected_verdict": "CONTAINED_MINIMAL_CONTAGION",
                    "required_bridge_entities": [f"{s2['ticker']}|{s2['name']}", f"{focal['ticker']}|{focal['name']}"],
                },
            }
        )

    # 6 Directed Cyclic Loops (A -> B -> C -> A)
    for i in range(6):
        a, b, c = rng.sample(pool, 3)
        prompt = (
            f"<|contagion_reasoning|>\n"
            f"Sector: {a['sector']}\n"
            f"Scenario: An acute 45% capacity curtailment hits {a['name']} ({a['ticker']}) for 2 quarters in a tightly coupled "
            f"3-company industrial feedback loop where each firm supplies critical capital sub-assemblies to the next.\n"
            f"Knowledge Graph Subgraph Context:\n"
            f"({a['name']}:Company {{ticker: '{a['ticker']}'}})-[:SUPPLIES_TO {{nature: 'Precision Metrology Lasers', materiality: 'CRITICAL_TIER_1', status: 'ACTIVE_CURRENT'}}]->({b['name']}:Company {{ticker: '{b['ticker']}'}})\n"
            f"({b['name']}:Company {{ticker: '{b['ticker']}'}})-[:SUPPLIES_TO {{nature: 'Custom Control ASICs', materiality: 'CRITICAL_TIER_1', status: 'ACTIVE_CURRENT'}}]->({c['name']}:Company {{ticker: '{c['ticker']}'}})\n"
            f"({c['name']}:Company {{ticker: '{c['ticker']}'}})-[:SUPPLIES_TO {{nature: 'Vacuum Chamber Actuators', materiality: 'CRITICAL_TIER_1', status: 'ACTIVE_CURRENT'}}]->({a['name']}:Company {{ticker: '{a['ticker']}'}})\n\n"
            f"Question: Analyze the multi-hop contagion and cyclic feedback impact across {b['name']} ({b['ticker']}) and {c['name']} ({c['ticker']})."
        )
        samples.append(
            {
                "sample_id": f"TT02B_CYCLE_LOOP_{i+1:02d}",
                "axis_id": "TT-02",
                "sub_cohort": "DIRECTED_CYCLE_FEEDBACK",
                "task_type": "CONTAGION_REASONING",
                "metadata": {"topology": "CYCLE_3_NODE"},
                "prompt": prompt,
                "expected_assertions": {
                    "expected_verdict": "DISRUPTIVE_SHOCK",
                    "required_bridge_entities": [f"{b['ticker']}|{b['name']}", f"{c['ticker']}|{c['name']}"],
                },
            }
        )

    # 6 Stale TERMINATED Bridge Edge Traps (CONTAINED_MINIMAL_CONTAGION)
    for i in range(6):
        a, b, c_active_alt = rng.sample(pool, 3)
        prompt = (
            f"<|contagion_reasoning|>\n"
            f"Sector: {b['sector']}\n"
            f"Scenario: On November 1, 2024, {a['name']} ({a['ticker']}) suffered a severe manufacturing halt. Note carefully the "
            f"contractual lifecycle status and effective dates on the supply edges below: {b['name']} ({b['ticker']}) exited its "
            f"relationship with {a['name']} in 2021 and migrated 100% of active procurement to {c_active_alt['name']} ({c_active_alt['ticker']}), "
            f"which remains fully operational at 100% capacity.\n"
            f"Knowledge Graph Subgraph Context:\n"
            f"({a['name']}:Company {{ticker: '{a['ticker']}'}})-[:SUPPLIES_TO {{nature: 'Legacy Baseband Modems', materiality: 'CRITICAL_TIER_1', status: 'TERMINATED', valid_from: '2018-01-01', valid_to: '2021-09-30'}}]->({b['name']}:Company {{ticker: '{b['ticker']}'}})\n"
            f"({c_active_alt['name']}:Company {{ticker: '{c_active_alt['ticker']}'}})-[:SUPPLIES_TO {{nature: 'Next-Gen RF Modems', materiality: 'CRITICAL_TIER_1', status: 'ACTIVE_CURRENT', valid_from: '2021-10-01'}}]->({b['name']}:Company {{ticker: '{b['ticker']}'}})\n\n"
            f"Question: Does the November 2024 halt at {a['name']} ({a['ticker']}) disrupt active 2024 production at {b['name']} ({b['ticker']})?"
        )
        samples.append(
            {
                "sample_id": f"TT02B_TERMINATED_EDGE_{i+1:02d}",
                "axis_id": "TT-02",
                "sub_cohort": "STALE_TERMINATED_EDGE_TRAP",
                "task_type": "CONTAGION_REASONING",
                "metadata": {"topology": "TEMPORAL_TERMINATED_BRIDGE"},
                "prompt": prompt,
                "expected_assertions": {
                    "expected_verdict": "CONTAINED_MINIMAL_CONTAGION",
                    "required_bridge_entities": [f"{b['ticker']}|{b['name']}", f"{c_active_alt['ticker']}|{c_active_alt['name']}"],
                },
            }
        )

    return samples


# ---------------------------------------------------------------------------
# Axis 3 (TT-03): Multi-Relation Sign Inversion / COMPETES_WITH Spillovers (25)
# ---------------------------------------------------------------------------
def _generate_tt03_samples(pool: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []

    # 13 Asymmetric Competitor Tailwind cases
    for i in range(13):
        foundry_halted, firm_hurt, rival_beneficiary, foundry_healthy = rng.sample(pool, 4)
        prompt = (
            f"<|contagion_reasoning|>\n"
            f"Sector: {rival_beneficiary['sector']}\n"
            f"Scenario: Sole-source supplier {foundry_halted['name']} ({foundry_halted['ticker']}) suffers a 6-month catastrophic "
            f"facility shutdown, halting shipments to {firm_hurt['name']} ({firm_hurt['ticker']}). Direct market competitor "
            f"{rival_beneficiary['name']} ({rival_beneficiary['ticker']}) has ZERO exposure to {foundry_halted['name']} and "
            f"sources 100% of its equivalent components from unaffected supplier {foundry_healthy['name']} ({foundry_healthy['ticker']}), "
            f"which has ample spare capacity.\n"
            f"Knowledge Graph Subgraph Context:\n"
            f"({foundry_halted['name']}:Company {{ticker: '{foundry_halted['ticker']}'}})-[:SOLE_SOURCE_DEPENDENT_ON {{materiality: 'CRITICAL_TIER_1', status: 'ACTIVE_CURRENT'}}]->({firm_hurt['name']}:Company {{ticker: '{firm_hurt['ticker']}'}})\n"
            f"({firm_hurt['name']}:Company {{ticker: '{firm_hurt['ticker']}'}})-[:COMPETES_WITH {{nature: 'Direct Enterprise Market Share Rivalry', status: 'ACTIVE_CURRENT'}}]->({rival_beneficiary['name']}:Company {{ticker: '{rival_beneficiary['ticker']}'}})\n"
            f"({foundry_healthy['name']}:Company {{ticker: '{foundry_healthy['ticker']}'}})-[:SUPPLIES_TO {{materiality: 'CRITICAL_TIER_1', status: 'ACTIVE_CURRENT'}}]->({rival_beneficiary['name']}:Company {{ticker: '{rival_beneficiary['ticker']}'}})\n\n"
            f"Question: Evaluate the specific operational and competitive market-share impact on {rival_beneficiary['name']} ({rival_beneficiary['ticker']}). "
            f"Is {rival_beneficiary['ticker']}'s own supply chain disrupted, or is it insulated and positioned as a competitive market-share beneficiary?"
        )
        samples.append(
            {
                "sample_id": f"TT03A_COMPETITOR_TAILWIND_{i+1:02d}",
                "axis_id": "TT-03",
                "sub_cohort": "ASYMMETRIC_RIVAL_TAILWIND",
                "task_type": "CONTAGION_REASONING",
                "metadata": {"relation_mix": "SUPPLIES_TO + COMPETES_WITH (ASYMMETRIC)"},
                "prompt": prompt,
                "expected_assertions": {
                    "expected_verdict": "COMPETITOR_TAILWIND",
                    "queried_entity_ticker": rival_beneficiary["ticker"],
                    "required_bridge_entities": [
                        f"{firm_hurt['ticker']}|{firm_hurt['name']}",
                        f"{rival_beneficiary['ticker']}|{rival_beneficiary['name']}",
                    ],
                },
            }
        )

    # 12 Shared Upstream Bottleneck cases (Both rivals hurt)
    for i in range(12):
        shared_foundry, firm_a, rival_b = rng.sample(pool, 3)
        prompt = (
            f"<|contagion_reasoning|>\n"
            f"Sector: {rival_b['sector']}\n"
            f"Scenario: Sole-source supplier {shared_foundry['name']} ({shared_foundry['ticker']}) suffers an industry-wide 50% "
            f"wafer allocation cut lasting 9 months. Both {firm_a['name']} ({firm_a['ticker']}) and its direct competitor "
            f"{rival_b['name']} ({rival_b['ticker']}) depend sole-source on {shared_foundry['name']} for core silicon.\n"
            f"Knowledge Graph Subgraph Context:\n"
            f"({shared_foundry['name']}:Company {{ticker: '{shared_foundry['ticker']}'}})-[:SUPPLIES_TO {{materiality: 'CRITICAL_TIER_1', status: 'ACTIVE_CURRENT'}}]->({firm_a['name']}:Company {{ticker: '{firm_a['ticker']}'}})\n"
            f"({shared_foundry['name']}:Company {{ticker: '{shared_foundry['ticker']}'}})-[:SUPPLIES_TO {{materiality: 'CRITICAL_TIER_1', status: 'ACTIVE_CURRENT'}}]->({rival_b['name']}:Company {{ticker: '{rival_b['ticker']}'}})\n"
            f"({firm_a['name']}:Company {{ticker: '{firm_a['ticker']}'}})-[:COMPETES_WITH {{status: 'ACTIVE_CURRENT'}}]->({rival_b['name']}:Company {{ticker: '{rival_b['ticker']}'}})\n\n"
            f"Question: Can {rival_b['name']} ({rival_b['ticker']}) avoid contagion and capture {firm_a['ticker']}'s lost volume, "
            f"or do both competitors suffer a disruptive supply shock from the shared upstream bottleneck?"
        )
        samples.append(
            {
                "sample_id": f"TT03B_SHARED_BOTTLENECK_{i+1:02d}",
                "axis_id": "TT-03",
                "sub_cohort": "SHARED_FOUNDRY_BOTTLENECK",
                "task_type": "CONTAGION_REASONING",
                "metadata": {"relation_mix": "SUPPLIES_TO + COMPETES_WITH (SHARED_CHOKEPOINT)"},
                "prompt": prompt,
                "expected_assertions": {
                    "expected_verdict": "SHARED_BOTTLENECK_SHOCK",
                    "required_bridge_entities": [
                        f"{shared_foundry['ticker']}|{shared_foundry['name']}",
                        f"{rival_b['ticker']}|{rival_b['name']}",
                    ],
                },
            }
        )

    return samples


# ---------------------------------------------------------------------------
# Axis 4 (TT-04): Incomplete News / Missing-Edge Partial Observability (25)
# ---------------------------------------------------------------------------
def _generate_tt04_samples(pool: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []

    # 12 Raw Edge-Dropout Blind Spot samples (intermediate link B -> C missing from observed news!)
    for i in range(12):
        origin, bridge_b, target_c, peer_d, shared_chem = rng.sample(pool, 5)
        prompt = (
            f"<|contagion_reasoning|>\n"
            f"Sector: {target_c['sector']}\n"
            f"Scenario: Breaking Form 8-K and wire reports disclose a 60% cleanroom production halt at {origin['name']} "
            f"({origin['ticker']}), immediately curtailing shipments to Tier-1 partner {bridge_b['name']} ({bridge_b['ticker']}). "
            f"Meanwhile, specialty chemical vendor {shared_chem['name']} ({shared_chem['ticker']}) operates normally at 100% capacity. "
            f"{target_c['name']} ({target_c['ticker']}) operates in the same {target_c['sub_industry']} cluster as {peer_d['name']} ({peer_d['ticker']}).\n"
            f"Knowledge Graph Subgraph Context:\n"
            f"({origin['name']}:Company {{ticker: '{origin['ticker']}'}})-[:SUPPLIES_TO {{materiality: 'CRITICAL_TIER_1', status: 'ACTIVE_CURRENT'}}]->({bridge_b['name']}:Company {{ticker: '{bridge_b['ticker']}'}})\n"
            f"({bridge_b['name']}:Company {{ticker: '{bridge_b['ticker']}'}})-[:SUPPLIES_TO {{materiality: 'CRITICAL_TIER_1', status: 'ACTIVE_CURRENT'}}]->({peer_d['name']}:Company {{ticker: '{peer_d['ticker']}'}})\n"
            f"({shared_chem['name']}:Company {{ticker: '{shared_chem['ticker']}'}})-[:SUPPLIES_TO {{materiality: 'MATERIAL_TIER_2', status: 'ACTIVE_CURRENT'}}]->({peer_d['name']}:Company {{ticker: '{peer_d['ticker']}'}})\n"
            f"({shared_chem['name']}:Company {{ticker: '{shared_chem['ticker']}'}})-[:SUPPLIES_TO {{materiality: 'MATERIAL_TIER_2', status: 'ACTIVE_CURRENT'}}]->({target_c['name']}:Company {{ticker: '{target_c['ticker']}'}})\n\n"
            f"Question: Analyze whether the 60% production halt at {origin['name']} ({origin['ticker']}) propagates through "
            f"the supply network to cause a disruptive shock at {target_c['name']} ({target_c['ticker']})."
        )
        # Real-world latent edge (bridge_b -> target_c) exists in ground truth, but is missing from G_observed!
        # Without GNN imputation, the LLM either misses the shock (false-negative blind spot) or fabricates
        # an unconfirmed link without epistemic qualification.
        samples.append(
            {
                "sample_id": f"TT04A_MISSING_EDGE_RAW_{i+1:02d}",
                "axis_id": "TT-04",
                "sub_cohort": "RAW_EDGE_DROPOUT_BLINDSPOT",
                "task_type": "CONTAGION_REASONING",
                "metadata": {"missing_edge": f"{bridge_b['ticker']}->{target_c['ticker']}", "gnn_augmented": False},
                "prompt": prompt,
                "expected_assertions": {
                    "expected_verdict": "DISRUPTIVE_SHOCK",
                    "epistemic_mode": "HIGH_PROB_IMPLIED_EDGE",
                    "required_bridge_entities": [
                        f"{bridge_b['ticker']}|{bridge_b['name']}",
                        f"{target_c['ticker']}|{target_c['name']}",
                    ],
                },
            }
        )

    # 9 High-Probability GNN Implied-Edge Handshake samples (p = 0.87)
    for i in range(9):
        origin, bridge_b, target_c = rng.sample(pool, 3)
        prob = round(0.84 + (i % 5) * 0.02, 2)
        prompt = (
            f"<|contagion_reasoning|>\n"
            f"Sector: {target_c['sector']}\n"
            f"Scenario: A 60% capacity halt hits {origin['name']} ({origin['ticker']}), curtailing {bridge_b['name']} ({bridge_b['ticker']}). "
            f"While direct news coverage is incomplete, the GNN Link Predictor flags a high-probability implied supply dependency "
            f"(probability = {prob}) from {bridge_b['name']} to {target_c['name']} ({target_c['ticker']}) based on shared Item 1A "
            f"component embeddings and historical volatility co-movement.\n"
            f"Knowledge Graph Subgraph Context (Verified Edges + GNN Structural Gap Priors):\n"
            f"({origin['name']}:Company {{ticker: '{origin['ticker']}'}})-[:SUPPLIES_TO {{materiality: 'CRITICAL_TIER_1', provenance: 'SEC_10K_VERIFIED'}}]->({bridge_b['name']}:Company {{ticker: '{bridge_b['ticker']}'}})\n"
            f"({bridge_b['name']}:Company {{ticker: '{bridge_b['ticker']}'}})-[:IMPLIED_SUPPLIES_TO {{materiality: 'CRITICAL_TIER_1', provenance: 'GNN_LINK_PREDICTOR', link_probability: {prob}, status: 'UNCONFIRMED_STRUCTURAL_PRIOR'}}]->({target_c['name']}:Company {{ticker: '{target_c['ticker']}'}})\n\n"
            f"Question: Analyze the potential contagion exposure to {target_c['name']} ({target_c['ticker']}), ensuring you explicitly "
            f"qualify the unconfirmed/implied GNN structural link rather than stating it as a verified SEC filing fact."
        )
        samples.append(
            {
                "sample_id": f"TT04B_GNN_IMPLIED_HIGH_{i+1:02d}",
                "axis_id": "TT-04",
                "sub_cohort": "GNN_IMPLIED_HIGH_PROB",
                "task_type": "CONTAGION_REASONING",
                "metadata": {"gnn_augmented": True, "link_probability": prob},
                "prompt": prompt,
                "expected_assertions": {
                    "expected_verdict": "DISRUPTIVE_SHOCK",
                    "epistemic_mode": "HIGH_PROB_IMPLIED_EDGE",
                    "required_bridge_entities": [
                        f"{bridge_b['ticker']}|{bridge_b['name']}",
                        f"{target_c['ticker']}|{target_c['name']}",
                    ],
                },
            }
        )

    # 4 Low-Probability GNN Noise Controls (p = 0.19 -> discount & contain)
    for i in range(4):
        origin, bridge_b, target_c = rng.sample(pool, 3)
        prob = round(0.17 + i * 0.02, 2)
        prompt = (
            f"<|contagion_reasoning|>\n"
            f"Sector: {target_c['sector']}\n"
            f"Scenario: A localized curtailment hits {origin['name']} ({origin['ticker']}). No verified supply link connects "
            f"{bridge_b['name']} ({bridge_b['ticker']}) to {target_c['name']} ({target_c['ticker']}), and the GNN Link Predictor "
            f"assigns only a weak, low-confidence noise score (link_probability = {prob}, below the 0.50 materiality threshold), "
            f"with {target_c['name']} maintaining independent domestic sourcing.\n"
            f"Knowledge Graph Subgraph Context:\n"
            f"({origin['name']}:Company {{ticker: '{origin['ticker']}'}})-[:SUPPLIES_TO {{materiality: 'CRITICAL_TIER_1', provenance: 'SEC_10K_VERIFIED'}}]->({bridge_b['name']}:Company {{ticker: '{bridge_b['ticker']}'}})\n"
            f"({bridge_b['name']}:Company {{ticker: '{bridge_b['ticker']}'}})-[:IMPLIED_SUPPLIES_TO {{provenance: 'GNN_LINK_PREDICTOR', link_probability: {prob}, status: 'LOW_CONFIDENCE_NOISE'}}]->({target_c['name']}:Company {{ticker: '{target_c['ticker']}'}})\n\n"
            f"Question: Should {target_c['name']} ({target_c['ticker']}) be classified as suffering a disruptive supply shock, or is "
            f"contagion minimal/contained given the low-probability unconfirmed link?"
        )
        samples.append(
            {
                "sample_id": f"TT04B_GNN_IMPLIED_LOW_NOISE_{i+1:02d}",
                "axis_id": "TT-04",
                "sub_cohort": "GNN_IMPLIED_LOW_PROB_CONTROL",
                "task_type": "CONTAGION_REASONING",
                "metadata": {"gnn_augmented": True, "link_probability": prob},
                "prompt": prompt,
                "expected_assertions": {
                    "expected_verdict": "CONTAINED_MINIMAL_CONTAGION",
                    "epistemic_mode": "LOW_PROB_NOISE_EDGE",
                    "required_bridge_entities": [f"{target_c['ticker']}|{target_c['name']}"],
                },
            }
        )

    return samples


# ---------------------------------------------------------------------------
# Axis 5 (TT-05): Filing Text Overrides & 3k-Token Long-Context Noise (25)
# ---------------------------------------------------------------------------
def _generate_tt05_samples(pool: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []

    # 8 Buffer Stockpile Text Overrides (Graph says CRITICAL_TIER_1, 8-K proves 14-month buffer -> CONTAINED)
    for i in range(8):
        sup, focal, cust = rng.sample(pool, 3)
        prompt = (
            f"<|contagion_reasoning|>\n"
            f"Sector: {focal['sector']}\n"
            f"Scenario: {sup['name']} ({sup['ticker']}) announces a 45-day temporary line maintenance shutdown. HOWEVER, per "
            f"Item 7.01 of {focal['name']}'s ({focal['ticker']}) Form 8-K filed this morning: '{focal['name']} holds 14 months "
            f"of pre-certified strategic buffer inventory of {sup['name']} components in domestic bonded warehouses and has "
            f"zero production or shipment curtailment across FY2025, fully protecting all downstream deliveries to {cust['name']}.'\n"
            f"Knowledge Graph Subgraph Context:\n"
            f"({sup['name']}:Company {{ticker: '{sup['ticker']}'}})-[:SUPPLIES_TO {{nature: 'Precision Optical Transceivers', materiality: 'CRITICAL_TIER_1', status: 'ACTIVE_CURRENT'}}]->({focal['name']}:Company {{ticker: '{focal['ticker']}'}})\n"
            f"({focal['name']}:Company {{ticker: '{focal['ticker']}'}})-[:SUPPLIES_TO {{nature: 'Cloud Switch Racks', materiality: 'CRITICAL_TIER_1'}}]->({cust['name']}:Company {{ticker: '{cust['ticker']}'}})\n\n"
            f"Question: Weighing the static CRITICAL_TIER_1 graph tag against the explicit Form 8-K 14-month buffer stockpile "
            f"disclosure, does the 45-day outage propagate to halt {focal['name']} ({focal['ticker']}) and {cust['name']} ({cust['ticker']})?"
        )
        samples.append(
            {
                "sample_id": f"TT05A_TEXT_BUFFER_OVERRIDE_{i+1:02d}",
                "axis_id": "TT-05",
                "sub_cohort": "FILING_BUFFER_INSULATION_OVERRIDE",
                "task_type": "CONTAGION_REASONING",
                "metadata": {"override_direction": "CRITICAL_GRAPH_TO_CONTAINED_TEXT"},
                "prompt": prompt,
                "expected_assertions": {
                    "expected_verdict": "CONTAINED_MINIMAL_CONTAGION",
                    "required_bridge_entities": [f"{focal['ticker']}|{focal['name']}", f"{cust['ticker']}|{cust['name']}"],
                },
            }
        )

    # 4 Hidden Sole-Source Escalation Overrides (Graph says NON_CRITICAL_SECONDARY, 8-K reveals 100% halt -> DISRUPTIVE)
    for i in range(4):
        sup, focal, cust = rng.sample(pool, 3)
        prompt = (
            f"<|contagion_reasoning|>\n"
            f"Sector: {focal['sector']}\n"
            f"Scenario: Although legacy graph metadata still tags {sup['name']} ({sup['ticker']}) as NON_CRITICAL_SECONDARY, "
            f"{focal['name']} ({focal['ticker']}) just filed an emergency Form 8-K Item 8.01 disclosing that {sup['name']} is "
            f"now the exclusive sole-source provider of a mandatory FDA-locked sterile valve with ZERO remaining buffer stock, "
            f"forcing an immediate 60% assembly halt at {focal['name']} and delaying shipments to {cust['name']} ({cust['ticker']}) for 2 quarters.\n"
            f"Knowledge Graph Subgraph Context:\n"
            f"({sup['name']}:Company {{ticker: '{sup['ticker']}'}})-[:SUPPLIES_TO {{nature: 'Sterile Assembly Valves', materiality: 'NON_CRITICAL_SECONDARY'}}]->({focal['name']}:Company {{ticker: '{focal['ticker']}'}})\n"
            f"({focal['name']}:Company {{ticker: '{focal['ticker']}'}})-[:SUPPLIES_TO {{nature: 'Infusion Systems', materiality: 'CRITICAL_TIER_1'}}]->({cust['name']}:Company {{ticker: '{cust['ticker']}'}})\n\n"
            f"Question: Accounting for the emergency Form 8-K sole-source halt disclosure overriding the stale graph tag, assess "
            f"the contagion impact on {focal['name']} ({focal['ticker']}) and {cust['name']} ({cust['ticker']})."
        )
        samples.append(
            {
                "sample_id": f"TT05A_TEXT_ESCALATION_OVERRIDE_{i+1:02d}",
                "axis_id": "TT-05",
                "sub_cohort": "FILING_SOLE_SOURCE_ESCALATION_OVERRIDE",
                "task_type": "CONTAGION_REASONING",
                "metadata": {"override_direction": "SECONDARY_GRAPH_TO_DISRUPTIVE_TEXT"},
                "prompt": prompt,
                "expected_assertions": {
                    "expected_verdict": "DISRUPTIVE_SHOCK",
                    "required_bridge_entities": [f"{focal['ticker']}|{focal['name']}", f"{cust['ticker']}|{cust['name']}"],
                },
            }
        )

    # 5 Long-Context (~2,200 words) Explicit Negations on Task A (expect (none))
    for i in range(5):
        c1, c2 = rng.sample(pool, 2)
        pad_top = _build_boilerplate_block(rng, target_words=750)
        pad_bot = _build_boilerplate_block(rng, target_words=750)
        needle = (
            f"Market Commentary Clarification: Despite third-party equity research commentary grouping {c1['name']} "
            f"(${c1['ticker']}) and {c2['name']} (${c2['ticker']}) within broader macro index baskets, {c1['name']} "
            f"has no supply, licensing, customer, or joint-venture agreements with {c2['name']} and conducts zero "
            f"commercial transactions with {c2['name']}."
        )
        prompt = (
            f"<|extract_sec_graph|>\n"
            f"Document: SEC 10-K / 10-Q Filing ({c1['sector']})\n"
            f"Text:\n{pad_top}\n\n{needle}\n\n{pad_bot}\n\nTriples:"
        )
        samples.append(
            {
                "sample_id": f"TT05B_LONG_NEGATION_{i+1:02d}",
                "axis_id": "TT-05",
                "sub_cohort": "LONG_CONTEXT_EXPLICIT_NEGATION",
                "task_type": "EXTRACT_SEC_GRAPH",
                "metadata": {"approx_words": 1600, "subtype": "NEGATION"},
                "prompt": prompt,
                "expected_assertions": {
                    "expect_none": True,
                    "allowed_ciks": [],
                },
            }
        )

    # 4 Long-Context (~2,200 words) Active Sole-Source Needle-in-Boilerplate on Task A
    for i in range(4):
        c1, c2 = rng.sample(pool, 2)
        pad_top = _build_boilerplate_block(rng, target_words=800)
        pad_bot = _build_boilerplate_block(rng, target_words=800)
        needle = (
            f"{c1['name']} ({c1['ticker']}, CIK {c1['cik']}) contracts with {c2['name']} ({c2['ticker']}, CIK {c2['cik']}) "
            f"as its sole source fabricator for radiation-hardened mixed-signal ASICs under a multi-year exclusive "
            f"capacity reservation agreement."
        )
        prompt = (
            f"<|extract_sec_graph|>\n"
            f"Document: SEC 10-K / 10-Q Filing ({c1['sector']})\n"
            f"Text:\n{pad_top}\n\n{needle}\n\n{pad_bot}\n\nTriples:"
        )
        samples.append(
            {
                "sample_id": f"TT05B_LONG_ACTIVE_NEEDLE_{i+1:02d}",
                "axis_id": "TT-05",
                "sub_cohort": "LONG_CONTEXT_NEEDLE_EXTRACTION",
                "task_type": "EXTRACT_SEC_GRAPH",
                "metadata": {"approx_words": 1700, "subtype": "ACTIVE_NEEDLE"},
                "prompt": prompt,
                "expected_assertions": {
                    "expect_none": False,
                    "allowed_ciks": [c1["cik"], c2["cik"]],
                    "required_ciks": [c1["cik"], c2["cik"]],
                },
            }
        )

    # 4 Long-Context (~2,000 words) Historical Terminated / News Extraction on Task B
    for i in range(4):
        c1, c2 = rng.sample(pool, 2)
        pad_top = _build_boilerplate_block(rng, target_words=700)
        pad_bot = _build_boilerplate_block(rng, target_words=700)
        needle = (
            f"Pursuant to Item 1.02 of Form 8-K, {c1['name']} ({c1['ticker']}, CIK {c1['cik']}) formally terminated "
            f"its exclusive patent cross-licensing agreement with {c2['name']} ({c2['ticker']}, CIK {c2['cik']}) "
            f"effective December 31, 2022, ending all ongoing royalty obligations."
        )
        prompt = (
            f"<|extract_news_event|>\n"
            f"Headline & Article:\n{pad_top}\n\n{needle}\n\n{pad_bot}\n\nDirectional Event:"
        )
        samples.append(
            {
                "sample_id": f"TT05B_LONG_TERMINATED_{i+1:02d}",
                "axis_id": "TT-05",
                "sub_cohort": "LONG_CONTEXT_TERMINATED_LIFECYCLE",
                "task_type": "EXTRACT_NEWS_EVENT",
                "metadata": {"approx_words": 1500, "subtype": "TERMINATED_CONTRACT"},
                "prompt": prompt,
                "expected_assertions": {
                    "expect_none": False,
                    "expected_edge_status": "TERMINATED",
                    "allowed_ciks": [c1["cik"], c2["cik"]],
                },
            }
        )

    return samples


# ---------------------------------------------------------------------------
# Axis 6 (TT-06): Compositional Cypher Schema Traps & Tape Divergence (25)
# ---------------------------------------------------------------------------
def _generate_tt06_samples(pool: List[Dict[str, Any]], rng: random.Random) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    schema_line = (
        "Schema: (:Company {name, ticker, cik})-"
        "[:SUPPLIES_TO|SOLE_SOURCE_DEPENDENT_ON|LICENSES_FROM|ACQUIRED_BY|DEFAULTED_ON "
        "{polarity, materiality, status, valid_from, valid_to}]->(:Company)"
    )

    # 8 Compositional Multi-Constraint Task C Queries
    for i in range(8):
        c1, c2 = rng.sample(pool, 2)
        if i % 2 == 0:
            nl_q = (
                f"Find all active CRITICAL_TIER_1 suppliers to {c1['name']} ({c1['ticker']}) "
                f"where the relationship status is 'ACTIVE_CURRENT', returning supplier name and ticker."
            )
            req_pats = [r"\bMATCH\b", r"\bWHERE\b", r"\bRETURN\b", rf"{c1['ticker']}"]
        else:
            nl_q = (
                f"Find 2-hop upstream suppliers that supply companies which in turn supply "
                f"{c2['name']} ({c2['ticker']}), filtering to ACTIVE_CURRENT edges."
            )
            req_pats = [r"\bMATCH\b", r"\bRETURN\b", rf"{c2['ticker']}"]

        prompt = f"<|text_to_cypher|>\n{schema_line}\nQuery: {nl_q}\n\nCypher:"
        samples.append(
            {
                "sample_id": f"TT06A_COMPOSITIONAL_CYPHER_{i+1:02d}",
                "axis_id": "TT-06",
                "sub_cohort": "COMPOSITIONAL_MULTI_HOP_CYPHER",
                "task_type": "TEXT_TO_CYPHER",
                "metadata": {"subtype": "COMPOSITIONAL"},
                "prompt": prompt,
                "expected_assertions": {
                    "unanswerable_schema_trap": False,
                    "required_cypher_patterns": req_pats,
                    "allowed_ciks": [c1["cik"], c2["cik"]],
                },
            }
        )

    # 5 Adversarial Unanswerable Schema Traps (CEO age, carbon score, Glassdoor rating, etc.)
    fake_requests = [
        "List all suppliers of Apple Inc. (AAPL) where the supplier's CEO age is above 60 and Scope-3 carbon emissions are below 50,000 tons.",
        "Find Tier-1 suppliers to Microsoft (MSFT) sorted by Glassdoor employee satisfaction score and board gender ratio.",
        "Return all companies supplying NVIDIA (NVDA) whose TikTok brand sentiment index exceeds 0.85.",
        "Find licensing partners of AbbVie (ABBV) filtered by Chief Medical Officer tenure years and patent citation h-index.",
        "List upstream vendors of Amazon (AMZN) where warehouse unionization percentage is greater than 30%.",
    ]
    for i, nl_q in enumerate(fake_requests):
        prompt = f"<|text_to_cypher|>\n{schema_line}\nQuery: {nl_q}\n\nCypher:"
        samples.append(
            {
                "sample_id": f"TT06A_SCHEMA_TRAP_{i+1:02d}",
                "axis_id": "TT-06",
                "sub_cohort": "ADVERSARIAL_SCHEMA_PROPERTY_TRAP",
                "task_type": "TEXT_TO_CYPHER",
                "metadata": {"subtype": "UNANSWERABLE_SCHEMA_TRAP"},
                "prompt": prompt,
                "expected_assertions": {
                    "unanswerable_schema_trap": True,
                    "allowed_ciks": [],
                },
            }
        )

    # 6 Bullish-PR / Severe Bearish-Tape Divergence on Task E (PROTECTIVE_DOWNSIDE_HEDGE)
    for i in range(6):
        ent = rng.choice(pool)
        car = round(-11.5 - i * 1.1, 1)
        z_val = round(-3.4 - i * 0.25, 2)
        prompt = (
            f"<|portfolio_recommendation|>\n"
            f"Entity: {ent['name']} ({ent['ticker']})\n"
            f"Sector: {ent['sector']}\n"
            f"Graph Risk Factor: Management issued an upbeat press release touting an 'EXPANDING_BULLISH' strategic AI alliance; "
            f"HOWEVER, post-filing market telemetry shows a severe institutional selloff (5-day CAR = {car}%, abnormal volatility "
            f"z-score = {z_val}, 30-day 25-delta put-call skew surging +420 bps) driven by gross-margin dilution and capex burn.\n"
            f"Objective: Formulate an institutional risk-hedging and portfolio rebalancing strategy reconciling the upbeat PR "
            f"headline against the severe negative market tape divergence."
        )
        samples.append(
            {
                "sample_id": f"TT06B_TAPE_DIVERGENCE_BEARISH_{i+1:02d}",
                "axis_id": "TT-06",
                "sub_cohort": "BULLISH_PR_BEARISH_TAPE_TRAP",
                "task_type": "PORTFOLIO_RECOMMENDATION",
                "metadata": {"car_pct": car, "vol_z": z_val},
                "prompt": prompt,
                "expected_assertions": {
                    "expected_verdict": "PROTECTIVE_DOWNSIDE_HEDGE",
                },
            }
        )

    # 6 Bearish-Headline / Priced-In Relief Rally on Task E (RELIEF_OVERHANG_CLEARED)
    for i in range(6):
        ent = rng.choice(pool)
        car = round(5.4 + i * 0.7, 1)
        z_val = round(2.2 + i * 0.15, 2)
        prompt = (
            f"<|portfolio_recommendation|>\n"
            f"Entity: {ent['name']} ({ent['ticker']})\n"
            f"Sector: {ent['sector']}\n"
            f"Graph Risk Factor: Form 8-K records a final legacy regulatory settlement ('CONTRACTING_BEARISH' headline fine); "
            f"HOWEVER, the settlement was 60% below worst-case consensus reserves, triggering a 5-day relief rally "
            f"(CAR = +{car}%, z-score = +{z_val}) and a sharp implied-volatility crush as the multi-year legal overhang cleared.\n"
            f"Objective: Formulate a disciplined institutional portfolio positioning and risk management strategy that avoids "
            f"panic liquidation on a priced-in historical headline while maintaining prudent tail-risk discipline."
        )
        samples.append(
            {
                "sample_id": f"TT06B_TAPE_DIVERGENCE_RELIEF_{i+1:02d}",
                "axis_id": "TT-06",
                "sub_cohort": "BEARISH_HEADLINE_RELIEF_RALLY",
                "task_type": "PORTFOLIO_RECOMMENDATION",
                "metadata": {"car_pct": car, "vol_z": z_val},
                "prompt": prompt,
                "expected_assertions": {
                    "expected_verdict": "RELIEF_OVERHANG_CLEARED",
                },
            }
        )

    return samples


def build_phase5_torture_suite(
    output_path: Path = DEFAULT_SUITE_PATH,
    seed: int = 20261107,
) -> List[Dict[str, Any]]:
    """Build and write the deterministic 150-sample Phase 5 OOD Torture Suite."""
    rng = random.Random(seed)
    pool = _load_sp500_pool()

    suite: List[Dict[str, Any]] = []
    suite.extend(_generate_tt01_samples(pool, rng))
    suite.extend(_generate_tt02_samples(pool, rng))
    suite.extend(_generate_tt03_samples(pool, rng))
    suite.extend(_generate_tt04_samples(pool, rng))
    suite.extend(_generate_tt05_samples(pool, rng))
    suite.extend(_generate_tt06_samples(pool, rng))

    assert len(suite) == 150, f"Expected 150 torture samples, got {len(suite)}"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in suite:
            f.write(json.dumps(row) + "\n")

    return suite


if __name__ == "__main__":
    rows = build_phase5_torture_suite()
    print(f"Wrote {len(rows)} Phase 5 torture-test samples to {DEFAULT_SUITE_PATH}")
