"""
AAPL 20-Timestamp Dry Run Pipeline (2018-2025).

Executes end-to-end data harvesting, market context calculation, manifold sampling,
and dual-model SFT dataset export, monitoring for code-related failure modes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
try:
    from dotenv import load_dotenv
    env_file = PROJECT_ROOT / ".env"
    if env_file.is_file():
        load_dotenv(dotenv_path=env_file)
except ImportError:
    pass

from ingestion.sec import EdgarClient, HistoricalSECHarvester
from ingestion.market import MarketContextIntegrator
from sft_pipeline import (
    ManifoldSample,
    ManifoldTargetedSampler,
    FinancialTaxonomyAnnotator,
    SFTDatasetExporter,
    SFTRecord,
)
from tools.sp500_universe import SP500Constituent, SP500UniverseManager

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s", "level":"%(levelname)s", "msg":"%(message)s"}'
)
logger = logging.getLogger("dry_run_aapl")

import random

def generate_random_timestamps(n: int = 20, seed: int = 42) -> List[Dict[str, str]]:
    """Generate n random datetimes uniformly distributed between 2018-01-01 and 2025-06-30."""
    random.seed(seed)
    start_dt = datetime(2018, 1, 1)
    end_dt = datetime(2025, 6, 30)
    delta_days = (end_dt - start_dt).days

    sampled_days = sorted(random.sample(range(delta_days), n))
    timestamps = []
    for idx, day_offset in enumerate(sampled_days, 1):
        dt = start_dt + timedelta(days=day_offset)
        d_str = dt.strftime("%Y-%m-%d")
        timestamps.append({
            "date": d_str,
            "event": f"RANDOM_T_{idx:02d}",
            "desc": f"Random point-in-time manifold evaluation snapshot ({d_str})"
        })
    return timestamps

AAPL_TIMESTAMPS = generate_random_timestamps(n=20, seed=42)


def run_dry_run():
    staging_dir = PROJECT_ROOT / "data" / "dry_run_staging"
    sec_staging_dir = staging_dir / "sec"
    market_staging_dir = staging_dir / "market_context"
    sft_staging_dir = staging_dir / "sft"

    sec_staging_dir.mkdir(parents=True, exist_ok=True)
    market_staging_dir.mkdir(parents=True, exist_ok=True)
    sft_staging_dir.mkdir(parents=True, exist_ok=True)

    errors: List[str] = []

    print("\n" + "=" * 70)
    print("STARTING AAPL 20-TIMESTAMP HISTORICAL DRY RUN (2018-2025)")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Harvest Historical SEC filings for AAPL (2018-2024)
    # -------------------------------------------------------------
    print("\n[Step 1/4] Harvesting Historical SEC Filings (AAPL 2018-2024)...")
    sec_harvester = HistoricalSECHarvester(output_dir=sec_staging_dir)
    aapl_constituent = SP500Constituent(
        ticker="AAPL",
        cik="0000320193",
        company_name="Apple Inc.",
        gics_sector="Information Technology",
        gics_sub_industry="Technology Hardware, Storage & Peripherals",
        headquarters_location="Cupertino, California",
        date_added="1982-11-30",
        is_current=True,
    )

    harvest_years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    sec_stats = []
    for yr in harvest_years:
        try:
            print(f"  Harvesting AAPL Form 10-K & 8-K for FY{yr}...")
            res = sec_harvester.harvest_company_year(aapl_constituent, yr, forms=["10-K", "8-K"])
            sec_stats.append(res)
            print(f"    -> 10-K Downloaded: {res['10k_downloaded']}, 8-Ks: {res['8k_count']}, Subsidiaries: {res['subsidiaries_count']}")
        except Exception as e:
            err = f"SEC Harvest error for FY{yr}: {e}"
            logger.error(err)
            errors.append(err)

    # -------------------------------------------------------------
    # 2. Market Context Computation for 20 Timestamps
    # -------------------------------------------------------------
    print("\n[Step 2/4] Fetching Market Context & Computing CAR / Volatility Metrics...")
    market_integrator = MarketContextIntegrator(output_dir=market_staging_dir)
    
    # Fetch historical daily prices for AAPL and SPY benchmark (2018-01-01 to 2025-08-01)
    print("  Fetching daily OHLCV from yfinance for AAPL & SPY...")
    aapl_prices = market_integrator.fetch_historical_ohlcv("AAPL", "2018-01-01", "2025-08-01")
    spy_prices = market_integrator.fetch_historical_ohlcv("SPY", "2018-01-01", "2025-08-01")
    print(f"  Retrieved {len(aapl_prices)} trading days for AAPL, {len(spy_prices)} for SPY.")

    market_records = []
    for ts in AAPL_TIMESTAMPS:
        d_str = ts["date"]
        try:
            metrics = market_integrator.compute_event_window_metrics(
                ticker="AAPL",
                event_date_str=d_str,
                price_history=aapl_prices,
                benchmark_history=spy_prices,
                window_days=3,
            )
            metrics["event_tag"] = ts["event"]
            metrics["event_description"] = ts["desc"]
            market_records.append(metrics)
            print(f"  Date {d_str} [{ts['event']}]: CAR={metrics['car_abnormal_return_pct']}%, VolZ={metrics['volatility_zscore']}, Polarity={metrics['polarity_sentiment']}")
        except Exception as e:
            err = f"Market context error for date {d_str}: {e}"
            logger.error(err)
            errors.append(err)

    parquet_path = market_integrator.stage_market_context_to_parquet(
        market_records, filename="dry_run_market_context.parquet"
    )
    print(f"  -> Staged market context to {parquet_path}")

    # -------------------------------------------------------------
    # 3. Manifold Hard Negative Mining & Taxonomy Annotation
    # -------------------------------------------------------------
    print("\n[Step 3/4] Synthesizing Manifold Samples and Taxonomy Annotations...")
    sampler = ManifoldTargetedSampler(output_dir=sft_staging_dir)
    annotator = FinancialTaxonomyAnnotator()

    # Grounded positive relations across historical AAPL supply chain & partners
    positive_samples = [
        ManifoldSample(
            sample_id="AAPL_POS_001_TSMC",
            text_passage="Apple contracts with Taiwan Semiconductor Manufacturing Company (TSMC) as the exclusive fabricator of A-series and M-series silicon wafers utilizing advanced 3nm and 5nm FinFET process nodes.",
            grounded_triples=[
                {"source_id": "TSMC", "target_id": "Apple Inc.", "rel_type": "SOLE_SOURCE_DEPENDENT_ON", "confidence": 0.98}
            ],
            entities_present=[
                {"name": "TSMC", "ticker": "TSM", "type": "Company"},
                {"name": "Apple Inc.", "ticker": "AAPL", "type": "Company"},
            ],
            hop_count=1,
            is_hard_negative=False,
            gics_sector="Information Technology",
            provenance="SEC_10K_ITEM1_2023",
            confidence=0.98,
            difficulty_score=0.35,
        ),
        ManifoldSample(
            sample_id="AAPL_POS_002_FOXCONN",
            text_passage="Hon Hai Precision Industry (Foxconn) operates primary assembly facilities in Zhengzhou providing final assembly for over 60 percent of global iPhone shipments.",
            grounded_triples=[
                {"source_id": "Hon Hai Precision Industry", "target_id": "Apple Inc.", "rel_type": "SUPPLIES_TO", "confidence": 0.96}
            ],
            entities_present=[
                {"name": "Hon Hai Precision Industry", "ticker": "HNHPF", "type": "Company"},
                {"name": "Apple Inc.", "ticker": "AAPL", "type": "Company"},
            ],
            hop_count=1,
            is_hard_negative=False,
            gics_sector="Information Technology",
            provenance="SEC_10K_ITEM1_2022",
            confidence=0.96,
            difficulty_score=0.40,
        ),
        ManifoldSample(
            sample_id="AAPL_POS_003_QUALCOMM",
            text_passage="Under the 2019 settlement agreement, Qualcomm licenses its standard-essential cellular patents to Apple and supplies 5G Snapdragon modem-RF systems for iPhone hardware.",
            grounded_triples=[
                {"source_id": "Qualcomm Inc.", "target_id": "Apple Inc.", "rel_type": "LICENSES_TO", "confidence": 0.94},
                {"source_id": "Qualcomm Inc.", "target_id": "Apple Inc.", "rel_type": "SUPPLIES_TO", "confidence": 0.95}
            ],
            entities_present=[
                {"name": "Qualcomm Inc.", "ticker": "QCOM", "type": "Company"},
                {"name": "Apple Inc.", "ticker": "AAPL", "type": "Company"},
            ],
            hop_count=1,
            is_hard_negative=False,
            gics_sector="Information Technology",
            provenance="SEC_8K_20190416",
            confidence=0.95,
            difficulty_score=0.55,
        ),
        ManifoldSample(
            sample_id="AAPL_POS_004_CORNING",
            text_passage="Corning Incorporated develops and manufactures proprietary Ceramic Shield cover glass for Apple iPhone displays under long-term joint development agreements.",
            grounded_triples=[
                {"source_id": "Corning Inc.", "target_id": "Apple Inc.", "rel_type": "SUPPLIES_TO", "confidence": 0.93}
            ],
            entities_present=[
                {"name": "Corning Inc.", "ticker": "GLW", "type": "Company"},
                {"name": "Apple Inc.", "ticker": "AAPL", "type": "Company"},
            ],
            hop_count=1,
            is_hard_negative=False,
            gics_sector="Information Technology",
            provenance="SEC_10K_ITEM1_2021",
            confidence=0.93,
            difficulty_score=0.45,
        )
    ]

    # Hard negative commentary passages (entities mentioned together without causal/supply relationship)
    commentary = [
        {
            "raw_text": "Both Apple Inc. ($AAPL) and Microsoft Corporation ($MSFT) traded slightly higher today alongside the broader Nasdaq 100 benchmark as macroeconomic inflation prints cooled across major markets. " * 3,
            "provider": "REUTERS_FINANCIAL",
        },
        {
            "raw_text": "Treasury yields pressured megacap technology valuations including Apple Inc. ($AAPL) and Tesla Inc. ($TSLA) as the Federal Reserve signaled higher terminal policy rates for longer durations. " * 3,
            "provider": "BLOOMBERG_WIRE",
        }
    ]

    active_pairs = {("AAPL", "TSM"), ("AAPL", "HNHPF"), ("AAPL", "QCOM"), ("AAPL", "GLW")}
    hard_negs = sampler.synthesize_hard_negatives(
        market_commentary_passages=commentary,
        known_active_pairs=active_pairs,
        target_count=4,
    )

    curated_samples = sampler.balance_and_curate_manifold(positive_samples, hard_negs)
    print(f"  Curated {len(curated_samples)} manifold samples ({len(positive_samples)} positives, {len(hard_negs)} hard negatives).")

    # -------------------------------------------------------------
    # 4. Export Multi-Task SFT Datasets (Extractor 3B & Reasoner 8B)
    # -------------------------------------------------------------
    print("\n[Step 4/4] Formatting & Exporting Dual-Model SFT Splits...")
    exporter = SFTDatasetExporter(output_dir=sft_staging_dir)

    extractor_records: List[SFTRecord] = []
    reasoner_records: List[SFTRecord] = []

    for s in curated_samples:
        # Task A
        annotated_triples = []
        for t in s.grounded_triples:
            ann = annotator.annotate_triple(
                raw_source=t["source_id"],
                raw_target=t["target_id"],
                raw_rel=t["rel_type"],
                context_text=s.text_passage,
            )
            if ann:
                annotated_triples.append(ann)
        rec_a = exporter.format_task_a_sec_graph(s, annotated_triples)
        extractor_records.append(rec_a)

        # Task B
        rec_b = exporter.format_task_b_news_event(s, annotated_triples)
        extractor_records.append(rec_b)

    # Task C (Text-to-Cypher)
    c_queries = [
        ("Apple Inc.", "AAPL"),
        ("Taiwan Semiconductor Manufacturing", "TSM"),
        ("Qualcomm Inc.", "QCOM"),
        ("Corning Inc.", "GLW"),
    ]
    for comp, tick in c_queries:
        rec_c = exporter.format_task_c_text_to_cypher(comp, tick)
        extractor_records.append(rec_c)

    # Task D (Contagion Reasoning)
    rec_d1 = exporter.format_task_d_contagion_reasoning(
        focal_company="Apple Inc.",
        focal_ticker="AAPL",
        supplier="Taiwan Semiconductor Manufacturing Company (TSMC)",
        supplier_ticker="TSM",
        shock_scenario="Geopolitical tensions disrupt maritime transit and fab operations in the Taiwan Strait, halting 3nm wafer output for 45 days.",
        impacted_customers=["Foxconn", "Pegatron", "Global Consumer Hardware Channels"],
    )
    rec_d2 = exporter.format_task_d_contagion_reasoning(
        focal_company="Apple Inc.",
        focal_ticker="AAPL",
        supplier="Qualcomm Inc.",
        supplier_ticker="QCOM",
        shock_scenario="Supply chain allocation constraints on 5G modem baseband chips delay next-generation iPhone product rollout by one quarter.",
        impacted_customers=["Telecom Carriers (AT&T, Verizon, T-Mobile)", "Authorized Apple Resellers"],
    )
    reasoner_records.extend([rec_d1, rec_d2])

    # Task E (Portfolio Rebalancing / Hedging)
    rec_e1 = exporter.format_task_e_portfolio_recommendation(
        focal_company="Apple Inc.",
        focal_ticker="AAPL",
        risk_exposure="Supply chain single-source concentration in 3nm wafer manufacturing",
        recommended_hedge="Execute an out-of-the-money put spread on semiconductor ETF (SMH) while rotating capital allocation into domestic infrastructure software",
    )
    rec_e2 = exporter.format_task_e_portfolio_recommendation(
        focal_company="Apple Inc.",
        focal_ticker="AAPL",
        risk_exposure="Regulatory antitrust enforcement on App Store services margin",
        recommended_hedge="Underweight consumer hardware concentration and increase allocation to defensive enterprise SaaS leaders with high pricing power",
    )
    reasoner_records.extend([rec_e1, rec_e2])

    summary = exporter.export_full_sft_splits(extractor_records, reasoner_records)

    # -------------------------------------------------------------
    # 5. Validation & Summary Reporting
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("AAPL 20-TIMESTAMP HISTORICAL DRY RUN AUDIT & VALIDATION")
    print("=" * 70)
    print(f"Total Errors Encountered: {len(errors)}")
    if errors:
        for err in errors:
            print(f"  [ERROR] {err}")
    else:
        print("  [SUCCESS] All components executed with ZERO code-related errors!")

    print("\nGenerated Artifacts:")
    print(f"  1. SEC Staging Directory   : {sec_staging_dir} (Files: {len(list(sec_staging_dir.glob('**/*')))})")
    print(f"  2. Market Context Parquet  : {parquet_path} (Size: {parquet_path.stat().st_size} bytes, Rows: {len(market_records)})")
    print(f"  3. Extractor 3B Train SFT  : {sft_staging_dir / 'extractor_3b_train.jsonl'}")
    print(f"  4. Reasoner 8B Train SFT   : {sft_staging_dir / 'reasoner_8b_train.jsonl'}")
    print(f"  5. SFT Summary Metadata    : {sft_staging_dir / 'dataset_summary.json'}")
    print("\nDataset Split Summary:")
    print(json.dumps(summary, indent=2))
    print("=" * 70 + "\n")

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(run_dry_run())
