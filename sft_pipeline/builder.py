"""
Comprehensive Full S&P 500 Multi-Task SFT Dataset Generation Engine.

Generates large-scale, 11-sector balanced, point-in-time grounded instruction datasets
for a unified Qwen2.5-7B Financial Intelligence Model spanning all 5 tasks:
- Task A: SEC OpenCypher Triples DSL (<|extract_sec_graph|>)
- Task B: Breaking News Event Edges (<|extract_news_event|>)
- Task C: Text-to-Cypher (<|text_to_cypher|>)
- Task D: Contagion Reasoning with <think> (<|contagion_reasoning|>)
- Task E: Portfolio Hedging & Allocation with <think> (<|portfolio_recommendation|>)

Applies Manifold Boundary Hard Negative Mining, representation floors (>= 200) for rare risk relations,
and 80/10/10 Train/Val/Test partitioning into data/sft/ (yarnball_sft_*.jsonl).
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
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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

try:
    from .sampler import ManifoldSample, ManifoldTargetedSampler
    from .annotator import AnnotatedTriple, FinancialTaxonomyAnnotator
    from .exporter import SFTDatasetExporter, SFTRecord
except ImportError:
    from sft_pipeline.sampler import ManifoldSample, ManifoldTargetedSampler
    from sft_pipeline.annotator import AnnotatedTriple, FinancialTaxonomyAnnotator
    from sft_pipeline.exporter import SFTDatasetExporter, SFTRecord

from tools.fetch_market_context import MarketContextIntegrator

# Formal Curation Engine Integration
try:
    from curation.cold_start.pipeline import ColdStartCurationPipeline
    from curation.active_learning.cartographer import DatasetCartographer
    from curation.contracts import ActiveLearningPartition
except ImportError:
    from graphrag_finance.curation.cold_start.pipeline import ColdStartCurationPipeline
    from graphrag_finance.curation.active_learning.cartographer import DatasetCartographer
    from graphrag_finance.curation.contracts import ActiveLearningPartition

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s", "level":"%(levelname)s", "msg":"%(message)s"}'
)
logger = logging.getLogger("build_full_sft_dataset")

OUTPUT_SFT_DIR = PROJECT_ROOT / "data" / "sft"


# ----------------------------------------------------------------------
# Multi-Sector Curated Relational Skeletons across all 11 GICS Sectors
# ----------------------------------------------------------------------
MULTI_SECTOR_RELATION_TEMPLATES = [
    # 1. Information Technology
    {
        "sector": "Information Technology",
        "rel_type": "SOLE_SOURCE_DEPENDENT_ON",
        "voice": "active",
        "source_role": "focal",
        "target_role": "partner",
        "passage_template": "{focal_name} ({focal_ticker}) contracts with {partner_name} ({partner_ticker}) as its sole source fabricator for advanced sub-3nm semiconductor wafer nodes under multi-year exclusive capacity reservation agreements.",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "EXPANDING_BULLISH",
        "hops": 1,
        "difficulty": 0.45,
    },
    {
        "sector": "Information Technology",
        "rel_type": "SOLE_SOURCE_DEPENDENT_ON",
        "voice": "passive",
        "source_role": "focal",
        "target_role": "partner",
        "passage_template": "{partner_name} ({partner_ticker}) is contracted by {focal_name} ({focal_ticker}) on an exclusive sole source basis for advanced sub-3nm semiconductor wafer fabrication under long-term capacity reservation agreements.",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "EXPANDING_BULLISH",
        "hops": 1,
        "difficulty": 0.50,
    },
    {
        "sector": "Information Technology",
        "rel_type": "LICENSES_FROM",
        "voice": "active",
        "source_role": "focal",
        "target_role": "partner",
        "passage_template": "{focal_name} ({focal_ticker}) entered into a definitive cross-licensing patent framework with {partner_name} ({partner_ticker}) covering standard-essential cellular communications and RF modem architectures.",
        "materiality": "MATERIAL_TIER_2",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.50,
    },
    {
        "sector": "Information Technology",
        "rel_type": "LICENSES_FROM",
        "voice": "passive",
        "source_role": "focal",
        "target_role": "partner",
        "passage_template": "Standard-essential cellular communications and RF modem patent architectures owned by {partner_name} ({partner_ticker}) are licensed to {focal_name} ({focal_ticker}) under a definitive multi-year cross-licensing framework.",
        "materiality": "MATERIAL_TIER_2",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.55,
    },
    {
        "sector": "Information Technology",
        "rel_type": "SUPPLIES_TO",
        "voice": "active",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "{partner_name} ({partner_ticker}) manufactures specialized optical transceivers and high-bandwidth interconnects supplying the server infrastructure of {focal_name} ({focal_ticker}).",
        "materiality": "MATERIAL_TIER_2",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.35,
    },
    {
        "sector": "Information Technology",
        "rel_type": "SUPPLIES_TO",
        "voice": "passive",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "Specialized optical transceivers and high-bandwidth interconnects are supplied to {focal_name} ({focal_ticker}) by {partner_name} ({partner_ticker}) for hyperscale AI server cluster infrastructure.",
        "materiality": "MATERIAL_TIER_2",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.45,
    },
    {
        "sector": "Information Technology",
        "rel_type": "EXPOSED_TO_RISK",
        "voice": "active",
        "source_role": "focal",
        "target_role": "partner",
        "passage_template": "Export control regulations and packaging substrate bottlenecks at {partner_name} ({partner_ticker}) create critical operational delivery risks for the accelerator division of {focal_name} ({focal_ticker}).",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "CONTRACTING_BEARISH",
        "hops": 1,
        "difficulty": 0.60,
    },
    {
        "sector": "Information Technology",
        "rel_type": "EXPOSED_TO_RISK",
        "voice": "passive",
        "source_role": "focal",
        "target_role": "partner",
        "passage_template": "The accelerator division of {focal_name} ({focal_ticker}) is severely exposed to delivery risk resulting from export control regulations and packaging substrate bottlenecks at {partner_name} ({partner_ticker}).",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "CONTRACTING_BEARISH",
        "hops": 1,
        "difficulty": 0.60,
    },

    # 2. Health Care
    {
        "sector": "Health Care",
        "rel_type": "LICENSES_TO",
        "voice": "active",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "{partner_name} ({partner_ticker}) granted an exclusive worldwide commercial license to {focal_name} ({focal_ticker}) for proprietary antibody-drug conjugate oncology therapeutics.",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "EXPANDING_BULLISH",
        "hops": 1,
        "difficulty": 0.55,
    },
    {
        "sector": "Health Care",
        "rel_type": "LICENSES_TO",
        "voice": "passive",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "An exclusive worldwide commercial license for proprietary antibody-drug conjugate oncology therapeutics was granted to {focal_name} ({focal_ticker}) by {partner_name} ({partner_ticker}).",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "EXPANDING_BULLISH",
        "hops": 1,
        "difficulty": 0.55,
    },
    {
        "sector": "Health Care",
        "rel_type": "EXPOSED_TO_RISK",
        "voice": "active",
        "source_role": "focal",
        "target_role": "partner",
        "passage_template": "{focal_name} ({focal_ticker}) faces single-facility sterilization and supply chain dependencies with contract manufacturer {partner_name} ({partner_ticker}) for pre-filled biologic syringes.",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "CONTRACTING_BEARISH",
        "hops": 1,
        "difficulty": 0.60,
    },
    {
        "sector": "Health Care",
        "rel_type": "EXPOSED_TO_RISK",
        "voice": "passive",
        "source_role": "focal",
        "target_role": "partner",
        "passage_template": "Critical operational dependency for pre-filled biologic syringes is concentrated at single-facility contract manufacturer {partner_name} ({partner_ticker}), exposing {focal_name} ({focal_ticker}) to potential regulatory halts.",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "CONTRACTING_BEARISH",
        "hops": 1,
        "difficulty": 0.60,
    },
    {
        "sector": "Health Care",
        "rel_type": "SOLE_SOURCE_DEPENDENT_ON",
        "voice": "active",
        "source_role": "focal",
        "target_role": "partner",
        "passage_template": "{focal_name} ({focal_ticker}) relies exclusively on {partner_name} ({partner_ticker}) for primary active pharmaceutical ingredient (API) synthesis under FDA-regulated purity specifications.",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.50,
    },
    {
        "sector": "Health Care",
        "rel_type": "SOLE_SOURCE_DEPENDENT_ON",
        "voice": "passive",
        "source_role": "focal",
        "target_role": "partner",
        "passage_template": "Primary active pharmaceutical ingredient (API) synthesis under FDA purity specifications is exclusively provided to {focal_name} ({focal_ticker}) by sole-source manufacturer {partner_name} ({partner_ticker}).",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.50,
    },

    # 3. Financials
    {
        "sector": "Financials",
        "rel_type": "SUPPLIES_TO",
        "voice": "active",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "{partner_name} ({partner_ticker}) provides primary core banking mainframe clearing and automated payment processing infrastructure to {focal_name} ({focal_ticker}).",
        "materiality": "MATERIAL_TIER_2",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.40,
    },
    {
        "sector": "Financials",
        "rel_type": "SUPPLIES_TO",
        "voice": "passive",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "Primary core banking mainframe clearing and automated payment processing infrastructure are provided to {focal_name} ({focal_ticker}) by {partner_name} ({partner_ticker}).",
        "materiality": "MATERIAL_TIER_2",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.40,
    },
    {
        "sector": "Financials",
        "rel_type": "DEFAULTED_ON",
        "voice": "active",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "{partner_name} ({partner_ticker}) breached senior credit facility covenants and defaulted on senior debt obligations owed to {focal_name} ({focal_ticker}).",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "DISRUPTIVE_SHOCK",
        "hops": 1,
        "difficulty": 0.65,
    },
    {
        "sector": "Financials",
        "rel_type": "DEFAULTED_ON",
        "voice": "passive",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "A formal notice of default was issued against borrower counterparty {partner_name} ({partner_ticker}) by {focal_name} ({focal_ticker}) following senior credit covenant breaches.",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "DISRUPTIVE_SHOCK",
        "hops": 1,
        "difficulty": 0.65,
    },
    {
        "sector": "Financials",
        "rel_type": "ACQUIRED_BY",
        "voice": "active",
        "source_role": "focal",
        "target_role": "partner",
        "passage_template": "{partner_name} ({partner_ticker}) completed the statutory all-cash acquisition of regional asset management assets from {focal_name} ({focal_ticker}).",
        "materiality": "MATERIAL_TIER_2",
        "polarity": "EXPANDING_BULLISH",
        "hops": 1,
        "difficulty": 0.45,
    },
    {
        "sector": "Financials",
        "rel_type": "ACQUIRED_BY",
        "voice": "passive",
        "source_role": "focal",
        "target_role": "partner",
        "passage_template": "Regional asset management subsidiary assets of {focal_name} ({focal_ticker}) were formally acquired by {partner_name} ({partner_ticker}) in a definitive statutory transaction.",
        "materiality": "MATERIAL_TIER_2",
        "polarity": "EXPANDING_BULLISH",
        "hops": 1,
        "difficulty": 0.45,
    },

    # 4. Industrials & Aerospace
    {
        "sector": "Industrials",
        "rel_type": "SOLE_SOURCE_DEPENDENT_ON",
        "voice": "active",
        "source_role": "focal",
        "target_role": "partner",
        "passage_template": "{focal_name} ({focal_ticker}) is solely dependent on {partner_name} ({partner_ticker}) for critical high-temperature titanium turbine forgings and fuselage sub-assemblies.",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "CONTRACTING_BEARISH",
        "hops": 1,
        "difficulty": 0.50,
    },
    {
        "sector": "Industrials",
        "rel_type": "SOLE_SOURCE_DEPENDENT_ON",
        "voice": "passive",
        "source_role": "focal",
        "target_role": "partner",
        "passage_template": "Critical high-temperature titanium turbine forgings and fuselage sub-assemblies are sourced exclusively by {focal_name} ({focal_ticker}) from single supplier {partner_name} ({partner_ticker}).",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "CONTRACTING_BEARISH",
        "hops": 1,
        "difficulty": 0.50,
    },
    {
        "sector": "Industrials",
        "rel_type": "ACQUIRED_BY",
        "voice": "active",
        "source_role": "focal",
        "target_role": "partner",
        "passage_template": "{partner_name} ({partner_ticker}) finalized the definitive statutory merger under which it acquired {focal_name} ({focal_ticker}) for enterprise industrial automation expansion.",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "EXPANDING_BULLISH",
        "hops": 1,
        "difficulty": 0.45,
    },
    {
        "sector": "Industrials",
        "rel_type": "ACQUIRED_BY",
        "voice": "passive",
        "source_role": "focal",
        "target_role": "partner",
        "passage_template": "{focal_name} ({focal_ticker}) was acquired by {partner_name} ({partner_ticker}) following unanimous regulatory approvals under a definitive statutory merger plan.",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "EXPANDING_BULLISH",
        "hops": 1,
        "difficulty": 0.45,
    },
    {
        "sector": "Industrials",
        "rel_type": "SUPPLIES_TO",
        "voice": "active",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "{partner_name} ({partner_ticker}) delivers precision hydraulic actuators and fly-by-wire flight control subsystems to {focal_name} ({focal_ticker}).",
        "materiality": "MATERIAL_TIER_2",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.40,
    },
    {
        "sector": "Industrials",
        "rel_type": "SUPPLIES_TO",
        "voice": "passive",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "Precision hydraulic actuators and fly-by-wire flight control subsystems are delivered to {focal_name} ({focal_ticker}) by {partner_name} ({partner_ticker}).",
        "materiality": "MATERIAL_TIER_2",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.40,
    },

    # 5. Consumer Discretionary & Automotive
    {
        "sector": "Consumer Discretionary",
        "rel_type": "SUPPLIES_TO",
        "voice": "active",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "{partner_name} ({partner_ticker}) delivers specialized lithium iron phosphate (LFP) battery cells and power inverters to the assembly plants of {focal_name} ({focal_ticker}).",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.40,
    },
    {
        "sector": "Consumer Discretionary",
        "rel_type": "SUPPLIES_TO",
        "voice": "passive",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "Specialized lithium iron phosphate (LFP) battery cells and power inverters are supplied to assembly plants of {focal_name} ({focal_ticker}) by {partner_name} ({partner_ticker}).",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.40,
    },
    {
        "sector": "Consumer Discretionary",
        "rel_type": "SOLE_SOURCE_DEPENDENT_ON",
        "voice": "active",
        "source_role": "focal",
        "target_role": "partner",
        "passage_template": "{focal_name} ({focal_ticker}) operates under an exclusive long-term agreement naming {partner_name} ({partner_ticker}) as its sole automotive sensor and solid-state lidar provider.",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "EXPANDING_BULLISH",
        "hops": 1,
        "difficulty": 0.50,
    },
    {
        "sector": "Consumer Discretionary",
        "rel_type": "SOLE_SOURCE_DEPENDENT_ON",
        "voice": "passive",
        "source_role": "focal",
        "target_role": "partner",
        "passage_template": "{partner_name} ({partner_ticker}) was designated by {focal_name} ({focal_ticker}) as its exclusive sole source provider of automotive sensors and solid-state lidar units under multi-year contracts.",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "EXPANDING_BULLISH",
        "hops": 1,
        "difficulty": 0.50,
    },

    # 6. Energy
    {
        "sector": "Energy",
        "rel_type": "SUPPLIES_TO",
        "voice": "active",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "{partner_name} ({partner_ticker}) operates midstream pipeline transport and gathering facilities servicing deepwater acreage owned by {focal_name} ({focal_ticker}).",
        "materiality": "MATERIAL_TIER_2",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.40,
    },
    {
        "sector": "Energy",
        "rel_type": "SUPPLIES_TO",
        "voice": "passive",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "Midstream pipeline transport and gathering services are provided to deepwater acreage of {focal_name} ({focal_ticker}) by {partner_name} ({partner_ticker}).",
        "materiality": "MATERIAL_TIER_2",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.40,
    },
    {
        "sector": "Energy",
        "rel_type": "DEFAULTED_ON",
        "voice": "active",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "{partner_name} ({partner_ticker}) defaulted on offshore service charter commitments owed to operator {focal_name} ({focal_ticker}) following deepwater operational failures.",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "DISRUPTIVE_SHOCK",
        "hops": 1,
        "difficulty": 0.65,
    },
    {
        "sector": "Energy",
        "rel_type": "DEFAULTED_ON",
        "voice": "passive",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "A formal contractual default declaration was issued against drilling contractor {partner_name} ({partner_ticker}) by {focal_name} ({focal_ticker}) after uncurable offshore drilling failures.",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "DISRUPTIVE_SHOCK",
        "hops": 1,
        "difficulty": 0.65,
    },

    # 7. Materials
    {
        "sector": "Materials",
        "rel_type": "SOLE_SOURCE_DEPENDENT_ON",
        "voice": "active",
        "source_role": "focal",
        "target_role": "partner",
        "passage_template": "{focal_name} ({focal_ticker}) depends on {partner_name} ({partner_ticker}) as its single source refiner for ultra-pure electronic-grade argon gas required in cleanroom fabrication.",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.55,
    },
    {
        "sector": "Materials",
        "rel_type": "SOLE_SOURCE_DEPENDENT_ON",
        "voice": "passive",
        "source_role": "focal",
        "target_role": "partner",
        "passage_template": "Ultra-pure electronic-grade argon gas for cleanroom fabrication is supplied exclusively to {focal_name} ({focal_ticker}) on a sole-source basis by {partner_name} ({partner_ticker}).",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.55,
    },
    {
        "sector": "Materials",
        "rel_type": "LICENSES_FROM",
        "voice": "active",
        "source_role": "focal",
        "target_role": "partner",
        "passage_template": "{focal_name} ({focal_ticker}) secured an exclusive intellectual property license from {partner_name} ({partner_ticker}) for proprietary specialty polymer membranes.",
        "materiality": "MATERIAL_TIER_2",
        "polarity": "EXPANDING_BULLISH",
        "hops": 1,
        "difficulty": 0.45,
    },
    {
        "sector": "Materials",
        "rel_type": "LICENSES_FROM",
        "voice": "passive",
        "source_role": "focal",
        "target_role": "partner",
        "passage_template": "Exclusive intellectual property rights for proprietary specialty polymer membranes were licensed to {focal_name} ({focal_ticker}) by {partner_name} ({partner_ticker}).",
        "materiality": "MATERIAL_TIER_2",
        "polarity": "EXPANDING_BULLISH",
        "hops": 1,
        "difficulty": 0.45,
    },

    # 8. Consumer Staples
    {
        "sector": "Consumer Staples",
        "rel_type": "SUPPLIES_TO",
        "voice": "active",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "{partner_name} ({partner_ticker}) supplies agricultural sweeteners, packaging aluminum, and distribution logistics to {focal_name} ({focal_ticker}).",
        "materiality": "COMMODITY_TIER_3",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.30,
    },
    {
        "sector": "Consumer Staples",
        "rel_type": "SUPPLIES_TO",
        "voice": "passive",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "Agricultural sweeteners, packaging aluminum, and distribution logistics are supplied to {focal_name} ({focal_ticker}) by {partner_name} ({partner_ticker}).",
        "materiality": "COMMODITY_TIER_3",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.30,
    },

    # 9. Communication Services
    {
        "sector": "Communication Services",
        "rel_type": "LICENSES_TO",
        "voice": "active",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "{partner_name} ({partner_ticker}) entered a multi-year content licensing distribution agreement granting streaming rights to {focal_name} ({focal_ticker}).",
        "materiality": "MATERIAL_TIER_2",
        "polarity": "EXPANDING_BULLISH",
        "hops": 1,
        "difficulty": 0.40,
    },
    {
        "sector": "Communication Services",
        "rel_type": "LICENSES_TO",
        "voice": "passive",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "Exclusive multi-year streaming and content distribution rights were granted to {focal_name} ({focal_ticker}) by {partner_name} ({partner_ticker}).",
        "materiality": "MATERIAL_TIER_2",
        "polarity": "EXPANDING_BULLISH",
        "hops": 1,
        "difficulty": 0.40,
    },
    {
        "sector": "Communication Services",
        "rel_type": "SUPPLIES_TO",
        "voice": "active",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "{partner_name} ({partner_ticker}) leases dark fiber backbone bandwidth and subsea transatlantic transit cables to {focal_name} ({focal_ticker}).",
        "materiality": "MATERIAL_TIER_2",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.35,
    },
    {
        "sector": "Communication Services",
        "rel_type": "SUPPLIES_TO",
        "voice": "passive",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "Dark fiber backbone bandwidth and subsea transatlantic transit capacity are leased to {focal_name} ({focal_ticker}) by {partner_name} ({partner_ticker}).",
        "materiality": "MATERIAL_TIER_2",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.35,
    },

    # 10. Utilities
    {
        "sector": "Utilities",
        "rel_type": "SOLE_SOURCE_DEPENDENT_ON",
        "voice": "active",
        "source_role": "focal",
        "target_role": "partner",
        "passage_template": "{focal_name} ({focal_ticker}) contracts with {partner_name} ({partner_ticker}) as its sole source enriched uranium fuel supplier for regulated nuclear power generation facilities.",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.50,
    },
    {
        "sector": "Utilities",
        "rel_type": "SOLE_SOURCE_DEPENDENT_ON",
        "voice": "passive",
        "source_role": "focal",
        "target_role": "partner",
        "passage_template": "Enriched uranium fuel for regulated nuclear facilities of {focal_name} ({focal_ticker}) is provided on an exclusive sole source basis by {partner_name} ({partner_ticker}).",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.50,
    },
    {
        "sector": "Utilities",
        "rel_type": "SUPPLIES_TO",
        "voice": "active",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "{partner_name} ({partner_ticker}) delivers high-voltage grid substation transformers and switchgear equipment to utility operator {focal_name} ({focal_ticker}).",
        "materiality": "MATERIAL_TIER_2",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.40,
    },
    {
        "sector": "Utilities",
        "rel_type": "SUPPLIES_TO",
        "voice": "passive",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "High-voltage grid substation transformers and switchgear equipment are delivered to utility operator {focal_name} ({focal_ticker}) by {partner_name} ({partner_ticker}).",
        "materiality": "MATERIAL_TIER_2",
        "polarity": "NEUTRAL_STABLE",
        "hops": 1,
        "difficulty": 0.40,
    },

    # 11. Real Estate
    {
        "sector": "Real Estate",
        "rel_type": "SUPPLIES_TO",
        "voice": "active",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "{partner_name} ({partner_ticker}) leases hyperscale data center colocation shell facilities and backup power infrastructure to {focal_name} ({focal_ticker}).",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "EXPANDING_BULLISH",
        "hops": 1,
        "difficulty": 0.45,
    },
    {
        "sector": "Real Estate",
        "rel_type": "SUPPLIES_TO",
        "voice": "passive",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "Hyperscale data center colocation shell facilities and backup power infrastructure are leased to {focal_name} ({focal_ticker}) by {partner_name} ({partner_ticker}).",
        "materiality": "CRITICAL_TIER_1",
        "polarity": "EXPANDING_BULLISH",
        "hops": 1,
        "difficulty": 0.45,
    },
    {
        "sector": "Real Estate",
        "rel_type": "DEFAULTED_ON",
        "voice": "active",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "{partner_name} ({partner_ticker}) breached commercial lease covenants and defaulted on office rental obligations owed to {focal_name} ({focal_ticker}).",
        "materiality": "MATERIAL_TIER_2",
        "polarity": "DISRUPTIVE_SHOCK",
        "hops": 1,
        "difficulty": 0.60,
    },
    {
        "sector": "Real Estate",
        "rel_type": "DEFAULTED_ON",
        "voice": "passive",
        "source_role": "partner",
        "target_role": "focal",
        "passage_template": "Formal eviction and foreclosure proceedings were initiated against commercial tenant {partner_name} ({partner_ticker}) by {focal_name} ({focal_ticker}) following rent covenant default.",
        "materiality": "MATERIAL_TIER_2",
        "polarity": "DISRUPTIVE_SHOCK",
        "hops": 1,
        "difficulty": 0.60,
    },
]

# Market Commentary Passages for Co-Occurrence Hard Negative Synthesis
COMMENTARY_TEMPLATES = [
    "Both {comp_a} (${tick_a}) and {comp_b} (${tick_b}) traded higher today alongside the broader {benchmark} benchmark as macroeconomic inflation prints cooled across major markets. Institutional asset managers noted balanced rotation across large-cap holdings.",
    "Treasury yields pressured megacap valuations including {comp_a} (${tick_a}) and {comp_b} (${tick_b}) as the Federal Reserve signaled higher terminal policy rates for longer durations amid resilient labor market data.",
    "Equity index rebalancing triggered elevated trading volumes in {comp_a} (${tick_a}) and {comp_b} (${tick_b}) as passive ETF index funds adjusted sector weightings at the market close.",
    "Macroeconomic concerns regarding consumer sentiment impacted equities including {comp_a} (${tick_a}) and {comp_b} (${tick_b}), with options volume indicating increased hedging activity across benchmark derivatives.",
    "Energy prices and geopolitical volatility weighed on diversified holdings such as {comp_a} (${tick_a}) and {comp_b} (${tick_b}), although neither firm announced any material company-specific operational updates.",
    "Sector rotation strategies led quantitative hedge funds to reallocate between {comp_a} (${tick_a}) and {comp_b} (${tick_b}), though no direct supplier or corporate partnership links exist between the firms.",
    "Analysts debated valuation multiples for large-cap peers {comp_a} (${tick_a}) and {comp_b} (${tick_b}) following quarterly economic GDP updates, noting divergent capital expenditure cycles.",
    "Foreign exchange headwinds and dollar strength impacted multinationals including {comp_a} (${tick_a}) and {comp_b} (${tick_b}) across international revenue reporting segments.",
]

# Multi-Hop Shock Propagation Scenarios for Reasoner (Task D)
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
    {
        "sector": "Utilities",
        "shock": "Unplanned outage at nuclear fuel enrichment centrifuge facility restricts uranium fuel rod deliveries for 6 months.",
        "focal": ("NextEra Energy", "NEE"),
        "supplier": ("Cameco Corporation", "CCJ"),
        "customers": ["Regional Grid Independent System Operators", "Industrial High-Load Manufacturing"],
    },
    {
        "sector": "Materials",
        "shock": "Geopolitical export embargo cuts titanium sponge production by 50%, choking aerospace forgings.",
        "focal": ("RTX Corporation", "RTX"),
        "supplier": ("Titanium Metals Corporation", "TMC"),
        "customers": ["Defense Logistics Agency", "Commercial Aircraft Operators"],
    },
]

# Institutional Hedging Scenarios for Reasoner (Task E)
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
    {
        "company": ("NextEra Energy", "NEE"),
        "risk": "Severe weather damage to solar generation assets and rising grid interconnection queue delays",
        "hedge": "Long utility sector ETF (XLU) put options and hedge via peak-load electricity forward swap contracts",
    },
    {
        "company": ("Tesla, Inc.", "TSLA"),
        "risk": "Lithium carbonate refining bottleneck delaying volume EV deliveries",
        "hedge": "Buy call options on lithium mining ETF (LIT) as a direct commodity input price hedge while hedging long delta via index collars",
    },
]


class FullSFTDatasetBuilder:
    """Orchestrates large-scale multi-sector SFT dataset generation for unified Qwen2.5-7B."""

    def __init__(
        self,
        output_dir: Path = OUTPUT_SFT_DIR,
        seed: int = 42,
        universe_mgr: Optional[SP500UniverseManager] = None,
        enable_cold_start_curation: bool = False,
        cold_start_pipeline: Optional[ColdStartCurationPipeline] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.rng = random.Random(seed)

        self.universe_mgr = universe_mgr or SP500UniverseManager()
        self.annotator = FinancialTaxonomyAnnotator(universe_mgr=self.universe_mgr)
        self.sampler = ManifoldTargetedSampler(output_dir=self.output_dir, universe_mgr=self.universe_mgr, seed=seed)
        self.exporter = SFTDatasetExporter(output_dir=self.output_dir, universe_mgr=self.universe_mgr, seed=seed)
        self.enable_cold_start_curation = enable_cold_start_curation
        self.cold_start_pipeline = cold_start_pipeline

        self.constituents = self.universe_mgr.get_all_records()
        self.constituents_by_sector = defaultdict(list)
        for c in self.constituents:
            self.constituents_by_sector[c.gics_sector].append(c)

    def generate_positive_manifold_samples(self, target_per_template: int = 80) -> List[ManifoldSample]:
        """Generate diverse positive relational samples across all 11 GICS sectors (both active and passive voice)."""
        positive_samples: List[ManifoldSample] = []
        sample_idx = 1
        seen_passages: Set[str] = set()

        qualifiers = [
            "under an executed multi-year commercial framework agreement",
            "pursuant to audited SEC regulatory disclosures",
            "governed by formal binding contractual specifications",
            "effective across multi-year fiscal operating cycles",
            "under an amended long-term master services arrangement",
            "following comprehensive counterparty qualification review",
        ]

        for tmpl in MULTI_SECTOR_RELATION_TEMPLATES:
            sec = tmpl["sector"]
            sec_candidates = self.constituents_by_sector.get(sec, self.constituents)
            if not sec_candidates:
                sec_candidates = self.constituents

            attempts = 0
            created_for_tmpl = 0
            max_attempts = target_per_template * 30

            while created_for_tmpl < target_per_template and attempts < max_attempts:
                attempts += 1
                focal = self.rng.choice(sec_candidates)

                in_sector_peers = [c for c in sec_candidates if c.ticker != focal.ticker]
                if in_sector_peers and self.rng.random() < 0.60:
                    partner = self.rng.choice(in_sector_peers)
                else:
                    universe_peers = [c for c in self.constituents if c.ticker != focal.ticker]
                    partner = self.rng.choice(universe_peers)

                qualifier = self.rng.choice(qualifiers)
                base_passage = tmpl["passage_template"].format(
                    focal_name=focal.company_name,
                    focal_ticker=focal.ticker,
                    partner_name=partner.company_name,
                    partner_ticker=partner.ticker,
                )

                passage = f"{base_passage.rstrip('.')} {qualifier}."

                if passage in seen_passages:
                    continue
                seen_passages.add(passage)

                voice_tag = tmpl.get("voice", "active")[:3].upper()
                sample_id = f"POS_{sec[:4].upper()}_{tmpl['rel_type'][:6]}_{voice_tag}_{sample_idx:05d}"
                sample_idx += 1

                # Topological invariant: assign source and target by semantic role, not surface word order
                src_ent = partner if tmpl.get("source_role", "partner") == "partner" else focal
                tgt_ent = focal if tmpl.get("target_role", "focal") == "focal" else partner

                sample = ManifoldSample(
                    sample_id=sample_id,
                    text_passage=passage,
                    grounded_triples=[
                        {
                            "source_id": src_ent.company_name,
                            "target_id": tgt_ent.company_name,
                            "rel_type": tmpl["rel_type"],
                            "confidence": 0.96,
                        }
                    ],
                    entities_present=[
                        {"name": src_ent.company_name, "ticker": src_ent.ticker, "type": "Company"},
                        {"name": tgt_ent.company_name, "ticker": tgt_ent.ticker, "type": "Company"},
                    ],
                    hop_count=tmpl["hops"],
                    is_hard_negative=False,
                    gics_sector=sec,
                    provenance=f"SEC_10K_ITEM1_{tmpl['rel_type']}",
                    confidence=0.96,
                    difficulty_score=tmpl["difficulty"],
                )
                positive_samples.append(sample)
                created_for_tmpl += 1

        logger.info(f"Generated {len(positive_samples)} balanced positive manifold samples across {len(MULTI_SECTOR_RELATION_TEMPLATES)} active/passive templates.")
        return positive_samples

    def generate_hard_negative_samples(self, positives: List[ManifoldSample], count: int = 1200) -> List[ManifoldSample]:
        """Synthesize verified boundary-proximity hard negative samples with target '(none)'."""
        known_active_pairs: Set[Tuple[str, str]] = set()
        for p in positives:
            for t in p.grounded_triples:
                s_id = str(t.get("source_id", "")).upper()
                t_id = str(t.get("target_id", "")).upper()
                if s_id and t_id:
                    known_active_pairs.add((s_id, t_id))
                    known_active_pairs.add((t_id, s_id))
            for e1 in p.entities_present:
                for e2 in p.entities_present:
                    tk1 = e1.get("ticker", "").upper()
                    tk2 = e2.get("ticker", "").upper()
                    if tk1 and tk2 and tk1 != tk2:
                        known_active_pairs.add((tk1, tk2))
                        known_active_pairs.add((tk2, tk1))

        return self.sampler.synthesize_template_hard_negatives(
            templates=COMMENTARY_TEMPLATES,
            known_active_pairs=known_active_pairs,
            count=count,
        )

    def generate_task_d_contagion_records(self, count: int = 400) -> List[SFTRecord]:
        """Generate unique, non-duplicative multi-hop shock propagation scenarios with <think> CoT."""
        records: List[SFTRecord] = []
        sectors = [s for s in self.constituents_by_sector.keys() if len(self.constituents_by_sector[s]) >= 2]
        if not sectors:
            sectors = ["Information Technology", "Health Care", "Industrials", "Financials", "Energy", "Materials"]

        shock_archetypes = [
            ("unscheduled fabrication downtime and wafer yield collapse", "Advanced Silicon Wafers", "1–2 quarters"),
            ("export control packaging substrate embargo and customs inspection hold", "High-Bandwidth Memory Packaging", "2–3 quarters"),
            ("sterility validation failure and regulatory inspection shutdown at primary API facility", "Active Pharmaceutical Ingredients", "2–4 quarters"),
            ("uncertified structural titanium fastener supply chokepoint", "Precision Aerospace Sub-Assemblies", "2–3 quarters"),
            ("refining bottleneck and port logistics interruption for battery-grade materials", "Lithium Battery Cells", "1–2 quarters"),
            ("critical cyberattack disrupting pipeline SCADA flow and terminal distribution", "Pipeline Transport Capacity", "1–2 quarters"),
            ("liquidity contraction and regional syndication office loan covenant breach", "Core Mainframe Clearing Services", "1–3 quarters"),
            ("centrifuge enrichment outage delaying enriched fuel rod qualification", "Enriched Nuclear Fuel Rods", "2–4 quarters"),
        ]

        seen_prompts: Set[str] = set()
        attempts = 0
        max_attempts = count * 20

        while len(records) < count and attempts < max_attempts:
            attempts += 1
            sec = self.rng.choice(sectors)
            candidates = self.constituents_by_sector.get(sec, self.constituents)
            if len(candidates) < 2:
                candidates = self.constituents

            focal, supplier = self.rng.sample(candidates, 2)
            shock_desc, supply_mat, timeline = self.rng.choice(shock_archetypes)
            cap_hit = self.rng.choice([15, 20, 25, 30, 35, 40, 50])
            lead_time = self.rng.choice([3, 4, 6, 8, 12, 18])

            full_shock = f"A {cap_hit}% quarterly supply curtailment resulting from {shock_desc}, imposing an estimated {lead_time}-month requalification timeline for secondary vendors."

            other_candidates = [c for c in self.constituents if c.ticker not in (focal.ticker, supplier.ticker)]
            sampled_customers = [c.company_name for c in self.rng.sample(other_candidates, self.rng.choice([2, 3]))]

            rec = self.exporter.format_task_d_contagion_reasoning(
                focal_company=focal.company_name,
                focal_ticker=focal.ticker,
                supplier=supplier.company_name,
                supplier_ticker=supplier.ticker,
                shock_scenario=full_shock,
                impacted_customers=sampled_customers,
                gics_sector=sec,
                supply_nature=supply_mat,
                timeline_quarters=timeline,
            )

            prompt_sig = rec.prompt.strip()
            if prompt_sig in seen_prompts:
                continue
            seen_prompts.add(prompt_sig)
            records.append(rec)

        logger.info(f"Generated {len(records)} unique Task D contagion reasoning records (0 duplicates).")
        return records

    def generate_task_e_hedging_records(self, count: int = 400) -> List[SFTRecord]:
        """Generate unique, non-duplicative portfolio hedging and risk allocation recommendations with <think> CoT."""
        records: List[SFTRecord] = []
        sectors = list(self.constituents_by_sector.keys())

        risk_hedges = [
            ("single-source concentration in foreign semiconductor foundry fabrication",
             "out-of-the-money put spread on semiconductor ETF (SMH) funded by call overwriting on mature software holdings"),
            ("regulatory export embargo on proprietary high-compute accelerators to international markets",
             "establish a zero-cost equity collar (90% strike floor put, 110% cap call) on core equity weight"),
            ("contract manufacturing fill-finish capacity bottleneck for biologic injectable devices",
             "overweight diversified medical equipment ETF (IHI) and initiate paired long-short equity hedge against single-facility suppliers"),
            ("abrupt commodity crack spread compression and refined product inventory accumulation",
             "acquire downside crude put options contracts while rotating capital into defensive regulated utility dividend holdings"),
            ("private credit duration mismatch and regional banking CRE delinquency acceleration",
             "execute interest rate swaptions and scale cash allocation into 3-month Treasury bills"),
            ("transoceanic maritime shipping rate spikes and container port labor strikes",
             "long dry-bulk maritime freight rate futures and overweight domestic discount retail peers with domestic logistics sourcing"),
            ("severe extreme weather damage to solar generation assets and rising grid interconnection queue delays",
             "long utility sector ETF (XLU) put options and hedge via peak-load electricity forward swap contracts"),
            ("critical raw material refining bottleneck and lithium supply chokepoint delaying volume deliveries",
             "purchase call options on lithium mining ETF (LIT) as a direct commodity input price hedge while hedging long delta via index collars"),
        ]

        seen_prompts: Set[str] = set()
        attempts = 0
        max_attempts = count * 20

        while len(records) < count and attempts < max_attempts:
            attempts += 1
            sec = self.rng.choice(sectors)
            candidates = self.constituents_by_sector.get(sec, self.constituents)
            if not candidates:
                candidates = self.constituents

            focal = self.rng.choice(candidates)
            risk, hedge = self.rng.choice(risk_hedges)
            bps = self.rng.choice([125, 150, 175, 200, 225, 250, 300])

            rec = self.exporter.format_task_e_portfolio_recommendation(
                focal_company=focal.company_name,
                focal_ticker=focal.ticker,
                risk_exposure=risk,
                recommended_hedge=hedge,
                gics_sector=sec,
                allocation_delta_bps=bps,
            )

            prompt_sig = rec.prompt.strip()
            if prompt_sig in seen_prompts:
                continue
            seen_prompts.add(prompt_sig)
            records.append(rec)

        logger.info(f"Generated {len(records)} unique Task E portfolio hedging records (0 duplicates).")
        return records

    def apply_cold_start_curation(
        self,
        samples: List[ManifoldSample],
        minhash_threshold: float = 0.85,
        coreset_retention_ratio: float = 0.85,
    ) -> List[ManifoldSample]:
        """
        Executes formal Cold-Start Curation (MinHash LSH & Facility Location Coreset Selection)
        via curation.cold_start.pipeline.ColdStartCurationPipeline.
        """
        pipeline = self.cold_start_pipeline or ColdStartCurationPipeline(
            minhash_threshold=minhash_threshold,
            coreset_retention_ratio=coreset_retention_ratio,
        )
        curation_payload = [
            {"sample_id": s.sample_id, "prompt": s.text_passage, "_obj": s}
            for s in samples
        ]
        result = pipeline.run(curation_payload, text_key="prompt", id_key="sample_id")
        retained = [item["_obj"] for item in result.get("retained", [])]
        logger.info(
            f"Cold-Start Curation: {len(samples)} -> {len(retained)} samples retained "
            f"(MinHash pruned: {len(result.get('pruned_minhash', []))}, "
            f"Coreset pruned: {len(result.get('pruned_coreset', []))})"
        )
        return retained

    def apply_active_learning_curation(
        self,
        samples: List[ManifoldSample],
        partition_or_path: Union[ActiveLearningPartition, Path, str],
    ) -> List[ManifoldSample]:
        """
        Applies Dataset Cartography Active Learning Partition from curation.active_learning:
        - Keeps 100% of Ambiguous samples (high variability, maximum learning signal).
        - Discards Hard/quarantined mislabeled samples.
        - Steers manifold sampler to boost representation of deficient/ambiguous relation classes.
        """
        if isinstance(partition_or_path, (str, Path)):
            p_path = Path(partition_or_path)
            if p_path.suffix == ".jsonl":
                cartographer = DatasetCartographer()
                dynamics = cartographer.parse_dynamics_log(p_path)
                partition = cartographer.filter_for_active_learning(dynamics)
            else:
                with open(p_path, "r", encoding="utf-8") as f:
                    partition = ActiveLearningPartition(**json.load(f))
        else:
            partition = partition_or_path

        retained_ids = set(partition.retained_guids)
        quarantined_ids = set(partition.quarantined_hard_guids)

        filtered: List[ManifoldSample] = []
        ambiguous_relations: Set[str] = set()

        for s in samples:
            if s.sample_id in quarantined_ids:
                continue
            if not retained_ids or s.sample_id in retained_ids:
                filtered.append(s)
                if s.grounded_triples:
                    ambiguous_relations.add(s.grounded_triples[0].get("rel_type", ""))

        if ambiguous_relations:
            self.sampler.steer_from_cartography(list(ambiguous_relations), boost_factor=1.5)

        logger.info(
            f"Active Learning Curation: {len(samples)} -> {len(filtered)} retained, "
            f"quarantined {len(quarantined_ids)} hard instances, steered {len(ambiguous_relations)} relations."
        )
        return filtered

    def build_dataset(
        self,
        enable_cold_start_curation: Optional[bool] = None,
        active_learning_partition: Optional[Union[ActiveLearningPartition, Path, str]] = None,
    ) -> Dict[str, Any]:
        """Execute full end-to-end multi-task SFT generation in the 12,000-15,000 range."""
        print("\n" + "=" * 70)
        print("GENERATING FULL S&P 500 MULTI-TASK SFT DATASET (12,000 - 15,000 SCALE)")
        print("=" * 70)

        # 1. Positives & Verified Hard Negatives
        positives = self.generate_positive_manifold_samples(target_per_template=75)
        negatives = self.generate_hard_negative_samples(positives=positives, count=1200)

        # 2. Manifold class balancing (floors >= 400 for rare relations, caps at 1500)
        curated_samples = self.sampler.balance_and_curate_manifold(positives, negatives, target_total_samples=8000)
        print(f"  Curated {len(curated_samples)} total manifold samples ({len(curated_samples) - len(negatives)} positives, {len(negatives)} hard negatives).")

        # Active Learning Feedback Integration (Cartography filtering & steering)
        if active_learning_partition:
            curated_samples = self.apply_active_learning_curation(curated_samples, active_learning_partition)

        # Cold-Start Curation Integration (MinHash LSH & Coreset selection)
        use_cold_start = self.enable_cold_start_curation if enable_cold_start_curation is None else enable_cold_start_curation
        if use_cold_start:
            curated_samples = self.apply_cold_start_curation(curated_samples)

        # 3. Format Multi-Task Records
        extractor_records: List[SFTRecord] = []
        reasoner_records: List[SFTRecord] = []

        # Task A & Task B: 2 records per curated sample
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

        # Task C: Multi-variation Text-to-Cypher across all constituents (10 query variants per company)
        for c in self.constituents:
            recs_c = self.exporter.format_task_c_variations(c.company_name, c.ticker, num_variations=10)
            extractor_records.extend(recs_c)

        # Task D: Contagion Reasoning (Multi-hop shock propagation with <think>, 0 duplicates)
        reasoner_records.extend(self.generate_task_d_contagion_records(count=400))

        # Task E: Portfolio Hedging & Risk Rebalancing with <think>, 0 duplicates
        reasoner_records.extend(self.generate_task_e_hedging_records(count=400))

        # 4. Partition into 80/10/10 Train/Val/Test Splits with zero split leakage
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
