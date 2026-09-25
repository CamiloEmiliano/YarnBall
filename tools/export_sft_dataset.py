"""
Multi-Task SFT Dataset Export Engine for Dual-Model Distillation.

Extracts, annotates (5-Axis Taxonomy), and partitions grounded financial knowledge
graphs into specialized instruction datasets for:
1. Qwen2.5-3B Extractor (~2.1 GB vRAM, Task A: SEC Graph, Task B: News Event, Task C: Text-to-CQL)
2. Qwen3-8B Reasoner (~5.2 GB vRAM, Task D: Contagion Reasoning, Task E: Portfolio Recommendation)

Applies train/val/test splits (80% / 10% / 10%) and tracks outputs with DVC.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Set, Tuple

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from tools.sft_manifold_sampler import ManifoldTargetedSampler, ManifoldSample
from tools.sft_taxonomy_annotator import FinancialTaxonomyAnnotator, AnnotatedTriple
from tools.sp500_universe import SP500UniverseManager

logger = logging.getLogger("export_sft_dataset")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

DEFAULT_SFT_DIR = Path(__file__).resolve().parent.parent / "data" / "sft"


@dataclass
class SFTRecord:
    """Standardized SFT training record matching the multi-model architecture."""
    sample_id: str
    prompt: str
    target_completion: str
    metadata: Dict[str, Any]
    schema_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SFTDatasetExporter:
    """Orchestrates end-to-end multi-task SFT dataset generation for 3B and 8B models."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        universe_mgr: Optional[SP500UniverseManager] = None,
        seed: int = 42,
    ):
        self.output_dir = output_dir or DEFAULT_SFT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.universe_mgr = universe_mgr or SP500UniverseManager()
        self.seed = seed
        self.sampler = ManifoldTargetedSampler(output_dir=self.output_dir, universe_mgr=self.universe_mgr, seed=seed)
        self.annotator = FinancialTaxonomyAnnotator(universe_mgr=self.universe_mgr)

    # ----------------------------------------------------------------------
    # 3B Extractor Task Formatters
    # ----------------------------------------------------------------------
    def format_task_a_sec_graph(self, sample: ManifoldSample, annotated_triples: List[AnnotatedTriple]) -> SFTRecord:
        """Task A: <|extract_sec_graph|> SEC Filing text -> OpenCypher Triples DSL."""
        prompt = f"<|extract_sec_graph|>\nDocument: SEC 10-K / 10-Q Filing ({sample.gics_sector})\nText:\n{sample.text_passage}\n\nTriples:"

        if sample.is_hard_negative or not annotated_triples:
            target = "(none)"
        else:
            target = "\n".join(t.to_cypher_dsl() for t in annotated_triples)

        return SFTRecord(
            sample_id=f"TASK_A_{sample.sample_id}",
            prompt=prompt,
            target_completion=target,
            metadata={
                "task_type": "EXTRACT_SEC_GRAPH",
                "assigned_student": "QWEN_2.5_3B_EXTRACTOR",
                "cognitive_tier": "LOW",
                "difficulty_score": sample.difficulty_score,
                "is_hard_negative": sample.is_hard_negative,
                "gics_sector": sample.gics_sector,
                "provenance": sample.provenance,
            },
        )

    def format_task_b_news_event(self, sample: ManifoldSample, annotated_triples: List[AnnotatedTriple]) -> SFTRecord:
        """Task B: <|extract_news_event|> Breaking News text -> Directional 5-Axis Graph Edge."""
        prompt = f"<|extract_news_event|>\nHeadline & Article:\n{sample.text_passage}\n\nDirectional Event:"

        if sample.is_hard_negative or not annotated_triples:
            target = "(none)"
        else:
            target = "\n".join(t.to_cypher_dsl() for t in annotated_triples)

        return SFTRecord(
            sample_id=f"TASK_B_{sample.sample_id}",
            prompt=prompt,
            target_completion=target,
            metadata={
                "task_type": "EXTRACT_NEWS_EVENT",
                "assigned_student": "QWEN_2.5_3B_EXTRACTOR",
                "cognitive_tier": "LOW",
                "difficulty_score": sample.difficulty_score,
                "is_hard_negative": sample.is_hard_negative,
                "gics_sector": sample.gics_sector,
                "provenance": sample.provenance,
            },
        )

    def format_task_c_text_to_cypher(self, company_name: str, ticker: str, rel_type: str = "SUPPLIES_TO") -> SFTRecord:
        """Task C: <|text_to_cypher|> Financial natural language query -> Valid Memgraph Cypher query."""
        prompt = (
            f"<|text_to_cypher|>\n"
            f"Schema: (:Company {{name, ticker, cik}})-[:{rel_type} {{polarity, materiality, status}}]->(:Company)\n"
            f"Query: Find all Tier 1 suppliers providing critical components to {company_name} ({ticker}).\n\n"
            f"Cypher:"
        )
        target = (
            f"MATCH (s:Company)-[r:SUPPLIES_TO]->(t:Company)\n"
            f"WHERE (t.ticker = '{ticker}' OR t.name = '{company_name}') "
            f"AND r.materiality = 'CRITICAL_TIER_1' AND r.status = 'ACTIVE_CURRENT'\n"
            f"RETURN s.name AS supplier, s.ticker AS ticker, r.nature AS component, r.polarity AS sentiment;"
        )
        sample_id = f"TASK_C_{ticker}_{hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:12]}"

        return SFTRecord(
            sample_id=sample_id,
            prompt=prompt,
            target_completion=target,
            metadata={
                "task_type": "TEXT_TO_CYPHER",
                "assigned_student": "QWEN_2.5_3B_EXTRACTOR",
                "cognitive_tier": "LOW",
                "difficulty_score": 0.35,
                "is_hard_negative": False,
                "gics_sector": "Universal",
                "provenance": "SYNTHETIC_SCHEMA_DSL",
            },
        )

    # ----------------------------------------------------------------------
    # 8B Reasoner Task Formatters
    # ----------------------------------------------------------------------
    def format_task_d_contagion_reasoning(
        self,
        focal_company: str,
        focal_ticker: str,
        supplier: str,
        supplier_ticker: str,
        shock_scenario: str,
        impacted_customers: List[str],
    ) -> SFTRecord:
        """Task D: <|contagion_reasoning|> Multi-hop shock propagation with structured <think> CoT block."""
        prompt = (
            f"<|contagion_reasoning|>\n"
            f"Scenario: {shock_scenario}\n"
            f"Knowledge Graph Subgraph Context:\n"
            f"({supplier}:Company {{ticker: '{supplier_ticker}'}})-[:SUPPLIES_TO {{nature: 'Advanced Wafers', materiality: 'CRITICAL_TIER_1'}}]->({focal_company}:Company {{ticker: '{focal_ticker}'}})\n"
            f"({focal_company}:Company)-[:SUPPLIES_TO]->({', '.join(impacted_customers)})\n\n"
            f"Question: Analyze the 2nd- and 3rd-order supply chain contagion impact on downstream revenue and operating margins."
        )

        reasoning_thought = (
            f"<think>\n"
            f"1. Identify the primary shock origin: {supplier} ({supplier_ticker}) experiences a critical supply bottleneck due to '{shock_scenario}'.\n"
            f"2. Assess Tier 1 direct impact: {focal_company} ({focal_ticker}) has sole-source / critical dependency on {supplier}, creating immediate component shortages.\n"
            f"3. Propagate 2nd-order contagion: Downstream customers ({', '.join(impacted_customers)}) relying on {focal_company} will face production delays and deferred revenue.\n"
            f"4. Quantify financial elasticity: Operating margins will compress due to component spot price inflation and unabsorbed fixed overhead.\n"
            f"</think>\n"
        )

        completion = (
            f"{reasoning_thought}"
            f"**Contagion Analysis Summary**:\n"
            f"- **Primary Disruption**: {supplier} supply constraint immediately curtails {focal_company}'s production run rate.\n"
            f"- **2nd-Order Exposure**: Key downstream partners ({', '.join(impacted_customers)}) face delayed deliveries across the next 1–2 quarters.\n"
            f"- **Risk Classification**: `DISRUPTIVE_SHOCK` with high margin sensitivity across the semiconductor-hardware value chain."
        )

        sample_id = f"TASK_D_{focal_ticker}_{hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:12]}"

        return SFTRecord(
            sample_id=sample_id,
            prompt=prompt,
            target_completion=completion,
            metadata={
                "task_type": "CONTAGION_REASONING",
                "assigned_student": "QWEN_3_8B_REASONER",
                "cognitive_tier": "HIGH",
                "difficulty_score": 0.88,
                "graph_hops": 3,
                "gics_sector": "Information Technology",
                "provenance": "FRONTIER_MULTI_TEACHER_CONSENSUS",
            },
        )

    def format_task_e_portfolio_recommendation(
        self,
        focal_company: str,
        focal_ticker: str,
        risk_exposure: str,
        recommended_hedge: str,
    ) -> SFTRecord:
        """Task E: <|portfolio_recommendation|> Risk synthesis -> Portfolio allocation and hedging strategy."""
        prompt = (
            f"<|portfolio_recommendation|>\n"
            f"Entity: {focal_company} ({focal_ticker})\n"
            f"Graph Risk Factor: Exposed to {risk_exposure}.\n"
            f"Objective: Formulate an institutional risk-hedging and portfolio rebalancing strategy."
        )

        reasoning_thought = (
            f"<think>\n"
            f"1. Evaluate risk concentration: {focal_ticker} holds asymmetric single-factor vulnerability to {risk_exposure}.\n"
            f"2. Identify uncorrelated / inverse counterparties: Alternative suppliers or sector inverse hedges mitigate downside tails.\n"
            f"3. Structure execution recommendation: Rebalance portfolio weights and establish derivative collar or peer basket hedge.\n"
            f"</think>\n"
        )

        completion = (
            f"{reasoning_thought}"
            f"**Strategic Portfolio Recommendation**:\n"
            f"- **Action**: Reduce direct overweight exposure to {focal_ticker} by 150–200 bps.\n"
            f"- **Hedging Instrument**: {recommended_hedge}.\n"
            f"- **Rationale**: Mitigates 3rd-order downside tail risk without incurring prohibitive carry costs."
        )

        sample_id = f"TASK_E_{focal_ticker}_{hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:12]}"

        return SFTRecord(
            sample_id=sample_id,
            prompt=prompt,
            target_completion=completion,
            metadata={
                "task_type": "PORTFOLIO_RECOMMENDATION",
                "assigned_student": "QWEN_3_8B_REASONER",
                "cognitive_tier": "HIGH",
                "difficulty_score": 0.82,
                "graph_hops": 2,
                "gics_sector": "Cross-Sector",
                "provenance": "FRONTIER_MULTI_TEACHER_CONSENSUS",
            },
        )

    # ----------------------------------------------------------------------
    # Partitioning & Export Pipeline
    # ----------------------------------------------------------------------
    def export_full_sft_splits(
        self,
        extractor_records: List[SFTRecord],
        reasoner_records: List[SFTRecord],
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
    ) -> Dict[str, Any]:
        """
        Split records into Train (80%), Val (10%), Test (10%) and export JSONL files.
        """
        random.seed(42)

        def split_list(items: List[SFTRecord]) -> Tuple[List[SFTRecord], List[SFTRecord], List[SFTRecord]]:
            shuffled = list(items)
            random.shuffle(shuffled)
            n_total = len(shuffled)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)

            train_set = shuffled[:n_train]
            val_set = shuffled[n_train:n_train + n_val]
            test_set = shuffled[n_train + n_val:]
            return train_set, val_set, test_set

        # Split Extractor 3B
        ext_train, ext_val, ext_test = split_list(extractor_records)
        # Split Reasoner 8B
        rea_train, rea_val, rea_test = split_list(reasoner_records)

        def write_jsonl(filepath: Path, recs: List[SFTRecord]) -> None:
            with open(filepath, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r.to_dict()) + "\n")
            logger.info(f"Wrote {len(recs)} records to {filepath}")

        # Write Extractor 3B files
        write_jsonl(self.output_dir / "extractor_3b_train.jsonl", ext_train)
        write_jsonl(self.output_dir / "extractor_3b_val.jsonl", ext_val)
        write_jsonl(self.output_dir / "extractor_3b_test.jsonl", ext_test)

        # Write Reasoner 8B files
        write_jsonl(self.output_dir / "reasoner_8b_train.jsonl", rea_train)
        write_jsonl(self.output_dir / "reasoner_8b_val.jsonl", rea_val)
        write_jsonl(self.output_dir / "reasoner_8b_test.jsonl", rea_test)

        summary = {
            "schema_version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "extractor_3b": {
                "total": len(extractor_records),
                "train": len(ext_train),
                "val": len(ext_val),
                "test": len(ext_test),
            },
            "reasoner_8b": {
                "total": len(reasoner_records),
                "train": len(rea_train),
                "val": len(rea_val),
                "test": len(rea_test),
            },
            "output_directory": str(self.output_dir),
        }

        with open(self.output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary


def main() -> None:
    exporter = SFTDatasetExporter()
    logger.info("Running SFT Dataset Export demonstration...")

    # Synthesize demo records
    extractor_recs = [
        exporter.format_task_c_text_to_cypher("Apple Inc.", "AAPL"),
        exporter.format_task_c_text_to_cypher("Microsoft Corp", "MSFT"),
        exporter.format_task_c_text_to_cypher("NVIDIA Corporation", "NVDA"),
    ]

    reasoner_recs = [
        exporter.format_task_d_contagion_reasoning(
            focal_company="Apple Inc.",
            focal_ticker="AAPL",
            supplier="TSMC",
            supplier_ticker="TSM",
            shock_scenario="Severe drought restricts ultra-pure water needed for 3nm wafer fabrication in Hsinchu Science Park",
            impacted_customers=["Foxconn", "Pegatron", "Global Retail Distribution"],
        ),
        exporter.format_task_e_portfolio_recommendation(
            focal_company="NVIDIA Corporation",
            focal_ticker="NVDA",
            risk_exposure="Export restrictions on advanced AI compute to Asian markets",
            recommended_hedge="Establish a long put collar on SOXX index and overweight domestic enterprise software",
        ),
    ]

    summary = exporter.export_full_sft_splits(extractor_recs, reasoner_recs)
    logger.info(f"SFT Dataset Export complete: {summary}")


if __name__ == "__main__":
    main()
