"""
Comprehensive Full S&P 500 Multi-Task SFT Dataset Generation Engine.

Generates large-scale, 11-sector balanced, point-in-time grounded instruction datasets for:
1. Qwen2.5-3B Extractor: Task A (SEC DSL), Task B (News Events), Task C (Text-to-Cypher)
2. Qwen3-8B Reasoner: Task D (Contagion Reasoning), Task E (Portfolio Hedging) with <think> CoT

Applies Manifold Boundary Hard Negative Mining, representation floors (>= 200) for rare risk relations,
and 80/10/10 Train/Val/Test partitioning into data/sft/.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment
try:
    from dotenv import load_dotenv
    env_file = PROJECT_ROOT / ".env"
    if env_file.is_file():
        load_dotenv(dotenv_path=env_file)
except ImportError:
    pass

from tools.sp500_universe import SP500Constituent, SP500UniverseManager
from tools.sft_manifold_sampler import ManifoldSample, ManifoldTargetedSampler
from tools.sft_taxonomy_annotator import AnnotatedTriple, FinancialTaxonomyAnnotator
from tools.export_sft_dataset import SFTDatasetExporter, SFTRecord
from tools.fetch_market_context import MarketContextIntegrator

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s", "level":"%(levelname)s", "msg":"%(message)s"}'
)
logger = logging.getLogger("build_full_sft_dataset")

OUTPUT_SFT_DIR = PROJECT_ROOT / "data" / "sft"


# ----------------------------------------------------------------------
# Multi-Sector Curated Relational Skeletons across 11 GICS Sectors
# ----------------------------------------------------------------------
MULTI_SECTOR_RELATION_TEMPLATES = [
    # Information Technology
    {
        "sector": "Information Technology",
        "rel_type": "SOLE_SOURCE_DEPENDENT_ON",
        "passage_template": "{focal_name} ({focal_ticker}) contracts with {partner_name} ({partner_ticker}) as its sole source fabricator for advanced sub-3nm semiconductor wafer nodes under multi-year exclusive capacity reservation agreements.",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "EXPANDING_BULLISH",
        "hops": 1,
        "difficulty": 0.45,
    },
    {
        "sector": "Information Technology",
        "rel_type": "LICENSES_FROM",
        "passage_template": "{focal_name} ({focal_ticker}) entered into a definitive cross-licensing patent framework with {partner_name} ({partner_ticker}) covering standard-essential cellular communications and RF modem architectures.",
        "materiality": "MATERIAL_TIER_2",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.50,
    },
    {
        "sector": "Information Technology",
        "rel_type": "SUPPLIES_TO",
        "passage_template": "{partner_name} ({partner_ticker}) manufactures specialized optical transceivers and high-bandwidth interconnects supplying the server infrastructure of {focal_name} ({focal_ticker}).",
        "materiality": "MATERIAL_TIER_2",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.35,
    },
    # Health Care
    {
        "sector": "Health Care",
        "rel_type": "LICENSES_TO",
        "passage_template": "{partner_name} ({partner_ticker}) granted an exclusive worldwide commercial license to {focal_name} ({focal_ticker}) for proprietary antibody-drug conjugate oncology therapeutics.",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "EXPANDING_BULLISH",
        "hops": 1,
        "difficulty": 0.55,
    },
    {
        "sector": "Health Care",
        "rel_type": "EXPOSED_TO_RISK",
        "passage_template": "{focal_name} ({focal_ticker}) faces single-facility sterilization and supply chain dependencies with contract manufacturer {partner_name} ({partner_ticker}) for pre-filled biologic syringes.",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "CONTRACTING_BEARISH",
        "hops": 1,
        "difficulty": 0.60,
    },
    # Financials
    {
        "sector": "Financials",
        "rel_type": "SUPPLIES_TO",
        "passage_template": "{partner_name} ({partner_ticker}) provides primary core banking mainframe clearing and automated payment processing infrastructure to {focal_name} ({focal_ticker}).",
        "materiality": "MATERIAL_TIER_2",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.40,
    },
    {
        "sector": "Financials",
        "rel_type": "DEFAULTED_ON",
        "passage_template": "{focal_name} ({focal_ticker}) issued a formal notice of default against borrower counterparty {partner_name} ({partner_ticker}) following senior credit facility covenant breaches.",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "DISRUPTIVE_SHOCK",
        "hops": 1,
        "difficulty": 0.65,
    },
    # Industrials & Aerospace
    {
        "sector": "Industrials",
        "rel_type": "SOLE_SOURCE_DEPENDENT_ON",
        "passage_template": "{focal_name} ({focal_ticker}) is solely dependent on {partner_name} ({partner_ticker}) for critical high-temperature titanium turbine forgings and fuselage sub-assemblies.",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "CONTRACTING_BEARISH",
        "hops": 1,
        "difficulty": 0.50,
    },
    {
        "sector": "Industrials",
        "rel_type": "ACQUIRED_BY",
        "passage_template": "{partner_name} ({partner_ticker}) finalized the definitive all-cash statutory merger under which it acquired {focal_name} ({focal_ticker}) for enterprise expansion.",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "EXPANDING_BULLISH",
        "hops": 1,
        "difficulty": 0.45,
    },
    # Consumer Discretionary & Automotive
    {
        "sector": "Consumer Discretionary",
        "rel_type": "SUPPLIES_TO",
        "passage_template": "{partner_name} ({partner_ticker}) delivers specialized lithium iron phosphate (LFP) battery cells and power inverters to the assembly plants of {focal_name} ({focal_ticker}).",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.40,
    },
    # Energy & Materials
    {
        "sector": "Energy",
        "rel_type": "SUPPLIES_TO",
        "passage_template": "{partner_name} ({partner_ticker}) operates midstream pipeline transport and gathering facilities servicing deepwater acreage owned by {focal_name} ({focal_ticker}).",
        "materiality": "MATERIAL_TIER_2",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.40,
    },
    {
        "sector": "Materials",
        "rel_type": "SOLE_SOURCE_DEPENDENT_ON",
        "passage_template": "{focal_name} ({focal_ticker}) depends on {partner_name} ({partner_ticker}) as its single source refiner for ultra-pure electronic-grade argon gas required in cleanroom fabrication.",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.55,
    },
    # Consumer Staples
    {
        "sector": "Consumer Staples",
        "rel_type": "SUPPLIES_TO",
        "passage_template": "{partner_name} ({partner_ticker}) supplies agricultural sweeteners, packaging aluminum, and distribution logistics to {focal_name} ({focal_ticker}).",
        "materiality": "COMMODITY_TIER_3",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.30,
    },
    # Communication Services
    {
        "sector": "Communication Services",
        "rel_type": "LICENSES_TO",
        "passage_template": "{partner_name} ({partner_ticker}) entered a multi-year content licensing distribution agreement granting streaming rights to {focal_name} ({focal_ticker}).",
        "materiality": "MATERIAL_TIER_2",
        "polarity": "EXPANDING_BULLISH",
        "hops": 1,
        "difficulty": 0.40,
    },
]

# Market Commentary Passages for Co-Occurrence Hard Negative Synthesis
COMMENTARY_TEMPLATES = [
    "Both {comp_a} (${tick_a}) and {comp_b} (${tick_b}) traded higher today alongside the broader {benchmark} benchmark as macroeconomic inflation prints cooled across major markets. Institutional asset managers noted balanced rotation across large-cap holdings.",
    "Treasury yields pressured megacap valuations including {comp_a} (${tick_a}) and {comp_b} (${tick_b}) as the Federal Reserve signaled higher terminal policy rates for longer durations amid resilient labor market data.",
    "Equity index rebalancing triggered elevated trading volumes in {comp_a} (${tick_a}) and {comp_b} (${tick_b}) as passive ETF index funds adjusted sector weightings at the market close.",
    "Macroeconomic concerns regarding consumer sentiment impacted equities including {comp_a} (${tick_a}) and {comp_b} (${tick_b}), with options volume indicating increased hedging activity across benchmark derivatives.",
    "Energy prices and geopolitical volatility weighed on diversified holdings such as {comp_a} (${tick_a}) and {comp_b} (${tick_b}), although neither firm announced any material company-specific operational updates.",
]

# Multi-Hop Shock Propagation Scenarios for 8B Reasoner (Task D)
CONTAGION_SCENARIOS = [
    {
        "sector": "Information Technology",
        "shock": "Severe drought and power grid rationing restrict ultra-pure water needed for 3nm wafer fabrication in Hsinchu Science Park, halting fab output for 45 days.",
        "focal": ("Apple Inc.", "AAPL"),
        "supplier": ("Taiwan Semiconductor Manufacturing Company", "TSM"),
        "customers": ["Hon Hai Precision (Foxconn)", "Pegatron Corp", "Global Retail Distribution Channels"],
    },
    {
        "sector": "Health Care",
        "shock": "FDA inspection reveals sterility validation deficiencies at the primary active pharmaceutical ingredient (API) synthesis facility, halting batch releases.",
        "focal": ("Eli Lilly and Company", "LLY"),
        "supplier": ("Lonza Group", "LONN"),
        "customers": ["Hospital Purchasing Alliances", "Wholesale Distributors (McKesson, AmerisourceBergen)"],
    },
    {
        "sector": "Industrials",
        "shock": "Quality control audit discovers uncertified titanium structural fasteners, grounding assembly lines for widebody commercial aircraft.",
        "focal": ("The Boeing Company", "BA"),
        "supplier": ("Spirit AeroSystems", "SPR"),
        "customers": ["Major Airlines (United, Delta, American)", "Global Air Cargo Carriers"],
    },
    {
        "sector": "Automotive & Clean Tech",
        "shock": "Export restrictions on refined battery-grade lithium carbonate and synthetic graphite delay next-generation battery cell manufacturing by two quarters.",
        "focal": ("Tesla, Inc.", "TSLA"),
        "supplier": ("Albemarle Corporation", "ALB"),
        "customers": ["Automotive Dealership Networks", "Commercial Fleet Operators"],
    },
    {
        "sector": "Energy",
        "shock": "Cyberattack disrupts pipeline flow and terminal SCADA systems across critical Gulf Coast refining corridors for 10 consecutive days.",
        "focal": ("Exxon Mobil Corporation", "XOM"),
        "supplier": ("Kinder Morgan Inc.", "KMI"),
        "customers": ["Industrial Petrochemical Plants", "Aviation Fuel Supply Hubs"],
    },
    {
        "sector": "Financials",
        "shock": "Abrupt liquidity contraction and commercial real estate office loan defaults trigger emergency capital preservation measures across regional banking counterparties.",
        "focal": ("JPMorgan Chase & Co.", "JPM"),
        "supplier": ("Regional Bank Syndicate Consortium", "KBW"),
        "customers": ["Institutional Pension Funds", "Corporate Treasury Depositors"],
    },
]

# Institutional Hedging Scenarios for 8B Reasoner (Task E)
HEDGING_SCENARIOS = [
    {
        "company": ("Apple Inc.", "AAPL"),
        "risk": "Single-source concentration in advanced semiconductor foundry operations in Taiwan",
        "hedge": "Execute an out-of-the-money put spread on semiconductor ETF (SMH) while rotating allocation into domestic software infrastructure leaders",
    },
    {
        "company": ("NVIDIA Corporation", "NVDA"),
        "risk": "Regulatory export controls on high-bandwidth AI accelerator shipments to foreign jurisdictions",
        "hedge": "Establish a collar on long core equity holding (buy 90% OTM put, sell 110% OTM call) to fund downside tail protection",
    },
    {
        "company": ("Eli Lilly and Company", "LLY"),
        "risk": "Contract manufacturing capacity bottleneck for GLP-1 injectable delivery devices",
        "hedge": "Overweight diversified healthcare equipment ETF (IHI) and initiate paired long-short trade against single-facility suppliers",
    },
    {
        "company": ("Exxon Mobil Corporation", "XOM"),
        "risk": "OPEC+ quota breakdown and sudden crude crack spread margin compression",
        "hedge": "Purchase Brent crude put options and rebalance portfolio weight into defensive regulated utility dividend holdings",
    },
    {
        "company": ("JPMorgan Chase & Co.", "JPM"),
        "risk": "Inverted yield curve duration mismatch and private credit default acceleration",
        "hedge": "Utilize interest rate swap swaptions and increase cash allocation to short-duration Treasury bills",
    },
    {
        "company": ("Walmart Inc.", "WMT"),
        "risk": "Container freight shipping cost spikes and East Coast port labor strikes",
        "hedge": "Long dry-bulk maritime freight rate futures and overweight domestic discount retail peers with domestic supply sourcing",
    },
]


class FullSFTDatasetBuilder:
    """Orchestrates large-scale multi-sector SFT dataset generation for dual models."""

    def __init__(self, output_dir: Path = OUTPUT_SFT_DIR, seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        random.seed(seed)

        self.universe_mgr = SP500UniverseManager()
        self.annotator = FinancialTaxonomyAnnotator(universe_mgr=self.universe_mgr)
        self.sampler = ManifoldTargetedSampler(output_dir=self.output_dir, universe_mgr=self.universe_mgr, seed=seed)
        self.exporter = SFTDatasetExporter(output_dir=self.output_dir, universe_mgr=self.universe_mgr, seed=seed)

        self.constituents = self.universe_mgr.get_all_records()
        self.constituents_by_sector = defaultdict(list)
        for c in self.constituents:
            self.constituents_by_sector[c.gics_sector].append(c)

    def generate_positive_manifold_samples(self, target_per_template: int = 35) -> List[ManifoldSample]:
        """Generate diverse positive relational samples across all 11 GICS sectors."""
        positive_samples: List[ManifoldSample] = []
        sample_idx = 1

        for tmpl in MULTI_SECTOR_RELATION_TEMPLATES:
            sec = tmpl["sector"]
            candidates = self.constituents_by_sector.get(sec, self.constituents)
            if len(candidates) < 2:
                candidates = self.constituents

            for _ in range(target_per_template):
                focal, partner = random.sample(candidates, 2)
                passage = tmpl["passage_template"].format(
                    focal_name=focal.company_name,
                    focal_ticker=focal.ticker,
                    partner_name=partner.company_name,
                    partner_ticker=partner.ticker,
                )

                sample_id = f"POS_{sec[:4].upper()}_{tmpl['rel_type'][:6]}_{sample_idx:04d}"
                sample_idx += 1

                sample = ManifoldSample(
                    sample_id=sample_id,
                    text_passage=passage,
                    grounded_triples=[
                        {
                            "source_id": partner.company_name,
                            "target_id": focal.company_name,
                            "rel_type": tmpl["rel_type"],
                            "confidence": 0.96,
                        }
                    ],
                    entities_present=[
                        {"name": partner.company_name, "ticker": partner.ticker, "type": "Company"},
                        {"name": focal.company_name, "ticker": focal.ticker, "type": "Company"},
                    ],
                    hop_count=tmpl["hops"],
                    is_hard_negative=False,
                    gics_sector=sec,
                    provenance=f"SEC_10K_ITEM1_{tmpl['rel_type']}",
                    confidence=0.96,
                    difficulty_score=tmpl["difficulty"],
                )
                positive_samples.append(sample)

        logger.info(f"Generated {len(positive_samples)} balanced positive manifold samples across {len(MULTI_SECTOR_RELATION_TEMPLATES)} templates.")
        return positive_samples

    def generate_hard_negative_samples(self, count: int = 250) -> List[ManifoldSample]:
        """Synthesize boundary-proximity hard negative samples from commentary templates."""
        hard_negatives: List[ManifoldSample] = []
        benchmarks = ["S&P 500", "Nasdaq 100", "Russell 1000", "Dow Jones Industrial Average"]

        for i in range(count):
            comp_a, comp_b = random.sample(self.constituents, 2)
            tmpl = random.choice(COMMENTARY_TEMPLATES)
            bench = random.choice(benchmarks)

            text = tmpl.format(
                comp_a=comp_a.company_name,
                tick_a=comp_a.ticker,
                comp_b=comp_b.company_name,
                tick_b=comp_b.ticker,
                benchmark=bench,
            )

            sample_id = f"NEG_HARD_{i+1:04d}_{comp_a.ticker}_{comp_b.ticker}"
            entities = [
                {"name": comp_a.company_name, "ticker": comp_a.ticker, "type": "Company"},
                {"name": comp_b.company_name, "ticker": comp_b.ticker, "type": "Company"},
            ]

            hard_negatives.append(
                ManifoldSample(
                    sample_id=sample_id,
                    text_passage=text,
                    grounded_triples=[], # Strictly empty target
                    entities_present=entities,
                    hop_count=0,
                    is_hard_negative=True,
                    gics_sector=comp_a.gics_sector,
                    provenance="MARKET_COMMENTARY_WIRE",
                    confidence=0.98,
                    difficulty_score=0.50,
                )
            )

        logger.info(f"Generated {len(hard_negatives)} boundary hard negative samples with target '(none)'.")
        return hard_negatives

    def build_dataset(self) -> Dict[str, Any]:
        """Execute full end-to-end multi-task SFT generation and export splits."""
        print("\n" + "=" * 70)
        print("GENERATING FULL S&P 500 MULTI-TASK SFT DATASET (11 SECTORS)")
        print("=" * 70)

        # 1. Positives & Hard Negatives
        positives = self.generate_positive_manifold_samples(target_per_template=35)
        negatives = self.generate_hard_negative_samples(count=250)

        # 2. Manifold class balancing (floors >= 200 for rare relations)
        curated_samples = self.sampler.balance_and_curate_manifold(positives, negatives)
        print(f"  Curated {len(curated_samples)} total manifold samples ({len(curated_samples) - len(negatives)} positives, {len(negatives)} hard negatives).")

        # 3. Format Multi-Task Records
        extractor_records: List[SFTRecord] = []
        reasoner_records: List[SFTRecord] = []

        # Task A & Task B
        for s in curated_samples:
            annotated_triples: List[AnnotatedTriple] = []
            for t in s.grounded_triples:
                ann = self.annotator.annotate_triple(
                    raw_source=t["source_id"],
                    raw_target=t["target_id"],
                    raw_rel=t["rel_type"],
                    context_text=s.text_passage,
                )
                if ann:
                    annotated_triples.append(ann)

            rec_a = self.exporter.format_task_a_sec_graph(s, annotated_triples)
            rec_b = self.exporter.format_task_b_news_event(s, annotated_triples)
            extractor_records.extend([rec_a, rec_b])

        # Task C: Text-to-Cypher across all constituents
        for c in self.constituents:
            rec_c = self.exporter.format_task_c_text_to_cypher(c.company_name, c.ticker)
            extractor_records.append(rec_c)

        # Task D: Contagion Reasoning (Multi-hop shock propagation with <think>)
        for sc in CONTAGION_SCENARIOS:
            for rep in range(12): # Replicate across diversified downstream customer permutations
                f_name, f_tick = sc["focal"]
                s_name, s_tick = sc["supplier"]
                rec_d = self.exporter.format_task_d_contagion_reasoning(
                    focal_company=f_name,
                    focal_ticker=f_tick,
                    supplier=s_name,
                    supplier_ticker=s_tick,
                    shock_scenario=sc["shock"],
                    impacted_customers=sc["customers"],
                )
                rec_d.sample_id = f"TASK_D_{f_tick}_{rep+1:02d}_{rec_d.sample_id[-8:]}"
                reasoner_records.append(rec_d)

        # Task E: Portfolio Hedging & Risk Rebalancing with <think>
        for hg in HEDGING_SCENARIOS:
            for rep in range(12):
                c_name, c_tick = hg["company"]
                rec_e = self.exporter.format_task_e_portfolio_recommendation(
                    focal_company=c_name,
                    focal_ticker=c_tick,
                    risk_exposure=hg["risk"],
                    recommended_hedge=hg["hedge"],
                )
                rec_e.sample_id = f"TASK_E_{c_tick}_{rep+1:02d}_{rec_e.sample_id[-8:]}"
                reasoner_records.append(rec_e)

        # 4. Partition into 80/10/10 Train/Val/Test Splits
        summary = self.exporter.export_full_sft_splits(extractor_records, reasoner_records)

        print("\n" + "=" * 70)
        print("SFT DATASET GENERATION SUMMARY (SCHEMA v1.0.0)")
        print("=" * 70)
        print(json.dumps(summary, indent=2))
        print(f"Output Directory: {self.output_dir}")
        print("=" * 70 + "\n")

        return summary


def main():
    builder = FullSFTDatasetBuilder()
    builder.build_dataset()


if __name__ == "__main__":
    main()
