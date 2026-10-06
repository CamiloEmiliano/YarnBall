---
annotations_creators:
- machine-generated
- verified-ground-truth
language:
- en
license: mit
multilinguality:
- monolingual
size_categories:
- 10K<n<100K
source_datasets:
- SEC EDGAR (Form 10-K, 10-Q, 8-K, Exhibit 21)
- Yahoo Finance Historical OHLCV
task_categories:
- text-generation
- text-retrieval
- question-answering
- graph-ml
pretty_name: "YarnBall S&P 500 Financial Graph & Contagion SFT Dataset"
tags:
- finance
- graphrag
- opencypher
- knowledge-graph-extraction
- contagion-analysis
- portfolio-hedging
- qwen2.5
- sft
- memgraph
configs:
- config_name: default
  data_files:
  - split: train
    path: yarnball_sft_train.jsonl
  - split: validation
    path: yarnball_sft_val.jsonl
  - split: test
    path: yarnball_sft_test.jsonl
- config_name: extraction
  data_files:
  - split: train
    path: extraction_tasks_train.jsonl
  - split: validation
    path: extraction_tasks_val.jsonl
  - split: test
    path: extraction_tasks_test.jsonl
- config_name: reasoning
  data_files:
  - split: train
    path: reasoning_tasks_train.jsonl
  - split: validation
    path: reasoning_tasks_val.jsonl
  - split: test
    path: reasoning_tasks_test.jsonl
---

# YarnBall S&P 500 Financial Graph & Contagion SFT Dataset (v1.0.0)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/CamiloEmiliano/yarnball-sft)
[![Target Model](https://img.shields.io/badge/Target%20Model-Qwen2.5--7B--Instruct-purple)](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

The **YarnBall Financial Graph & Contagion SFT Dataset** is a large-scale, point-in-time grounded instruction-tuning corpus containing **12,470 curated financial samples** spanning all **11 GICS economic sectors**. 

It is engineered specifically to fine-tune a unified **`Qwen2.5-7B-Instruct`** model capable of acting simultaneously as a high-throughput **information extraction engine** (converting raw SEC disclosures and breaking market news into validated OpenCypher graph DSL) and an advanced **cognitive reasoning engine** (simulating multi-hop contagion shock cascades and generating institutional portfolio hedging strategies using step-by-step `<think>` Chain-of-Thought).

---

## 1. Quickstart & Usage

The dataset provides a unified split as well as modular task configurations via the `datasets` library:

```python
from datasets import load_dataset

# 1. Load the Unified Dataset (12,470 records across all 5 tasks)
dataset = load_dataset("CamiloEmiliano/yarnball-sft")

train_set = dataset["train"]        # 9,976 records (80%)
val_set = dataset["validation"]     # 1,247 records (10%)
test_set = dataset["test"]          # 1,247 records (10%)

# 2. Alternatively, load modular subsets
# Tasks A, B, C: Extraction & Cypher (11,670 records)
extraction_ds = load_dataset("CamiloEmiliano/yarnball-sft", "extraction")

# Tasks D, E: Multi-Hop Reasoning with <think> (800 records)
reasoning_ds = load_dataset("CamiloEmiliano/yarnball-sft", "reasoning")

# Inspect a sample
sample = train_set[0]
print("--- PROMPT ---")
print(sample["prompt"])
print("\n--- COMPLETION ---")
print(sample["completion"])
```

---

## 2. Multi-Task Taxonomy & Schema Specifications

The dataset unifies 5 core financial intelligence tasks under standard ChatML prompt tags:

| Task | Category | Prompt Tag | Output Format | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Task A** | Extractor | `<|extract_sec_graph|>` | OpenCypher Triples DSL | Parses Form 10-K Item 1, 1A, and Exhibit 21 into verified entity & relationship triples with materiality tiers. |
| **Task B** | Extractor | `<|extract_news_event|>` | OpenCypher Event Edge | Ingests real-time financial news releases, extracting point-in-time event edges, temporal validity, and 5-axis directional polarity. |
| **Task C** | Extractor | `<|text_to_cypher|>` | Executable Cypher Query | 10 natural language query variations per constituent mapped to Memgraph schema paths. |
| **Task D** | Reasoner | `<|contagion_reasoning|>` | `<think>` CoT + Cascade | Simulates multi-hop supply disruptions, calculating counterparty revenue shocks, requalification delays, and downstream impacts. |
| **Task E** | Reasoner | `<|portfolio_recommendation|>` | `<think>` CoT + Allocation | Generates risk-mitigating portfolio recommendations (zero-cost collars, put spreads, swaptions) with specific basis-point deltas. |

### Prompt Loss Masking Contract
During fine-tuning, cross-entropy loss should be computed **strictly on target completion tokens**. Prompt tokens preceding the target delimiters must be masked with label `-100`:
- **Task A**: Mask before `Triples:`
- **Task B**: Mask before `Directional Event:`
- **Task C**: Mask before `Cypher Query:`
- **Task D & E**: Mask before `<think>`

---

## 3. Curational Methodology & Topological Invariants

Unlike generic web-scraped LLM datasets, every sample in this corpus satisfies strict regulatory, topological, and quantitative invariants:

### A. Point-in-Time Regulatory Ground Truth
- **SEC EDGAR Public Filings**: Primary ground truth originates from Form 10-K (business descriptions, risk factors, Exhibit 21 subsidiary directories) and Form 8-K material event disclosures.
- **Constituent Registry**: Point-in-time S&P 500 membership tracking (`sp500_constituents_historical.json`) eliminates survivorship bias.
- **Quantitative Market Anchoring**: Historical daily OHLCV prices, Cumulative Abnormal Returns (CAR), and volatility z-scores are computed via Yahoo Finance across 2018–2025 to determine empirical edge materiality and directional polarities (`EXPANDING_BULLISH`, `CONTRACTING_BEARISH`, `DISRUPTIVE_SHOCK`, `NEUTRAL_STABLE`).

### B. Manifold Boundary Hard Negative Mining
Class balance naturally skews toward generic mentions. The dataset pipeline applies `ManifoldTargetedSampler` to enforce:
- **Dominant Class Caps**: Pervasive relations (`SUBSIDIARY_OF`) are capped at 1,500 samples to prevent model collapse.
- **Rare Relation Representation Floors**: High-impact credit and supply chokepoints (`SOLE_SOURCE_DEPENDENT_ON`, `LICENSES_FROM`, `DEFAULTED_ON`) are enforced at $\ge 400$ samples via entity-swap augmentation across peer constituents.
- **Boundary Hard Negatives**: Synthesizes market commentary referencing two or more S&P 500 companies that share no economic relationship. The target completion is strictly set to `(none)` with capped heuristic confidence (0.65), training the model to resist co-occurrence hallucinations.

### C. Directional Orientation Guard (Passive-Voice Invariance)
To prevent directional inversion (e.g. mistakenly reversing supplier and customer), the dataset incorporates **52 balanced active- and passive-voice syntactic templates** across all 11 GICS sectors:
- Explicit `source_role` and `target_role` fields guarantee invariant edge directionality regardless of surface syntax word order.

### D. Zero Cross-Split Leakage
All prompt signatures across the **80% Train / 10% Validation / 10% Test** splits are formally asserted to be disjoint sets prior to export:
```python
assert len(train_signatures & val_signatures) == 0
assert len(train_signatures & test_signatures) == 0
assert len(val_signatures & test_signatures) == 0
```

---

## 4. Dataset Partitioning & Statistics

```
Total Curated Records: 12,470
├── Unified Splits:
│   ├── Train:      9,976 (80.0%)
│   ├── Validation: 1,247 (10.0%)
│   └── Test:       1,247 (10.0%)
└── Modular Subsets:
    ├── Extraction Tasks (A, B, C): 11,670 (9,336 train / 1,167 val / 1,167 test)
    └── Reasoning Tasks (D, E):        800 (  640 train /    80 val /    80 test)
```

### Sector Coverage (All 11 GICS Sectors)
- Information Technology
- Health Care
- Financials
- Consumer Discretionary
- Communication Services
- Industrials
- Consumer Staples
- Energy
- Utilities
- Real Estate
- Materials

---

## 5. Record Schema Example

```json
{
  "sample_id": "POS_INFO_SOLE_S_ACT_00001",
  "task": "sec_graph_extraction",
  "gics_sector": "Information Technology",
  "prompt": "Extract all verified financial entity relationships and corporate parentage from the following SEC regulatory disclosure into validated OpenCypher triples.\n\nContext:\nApple Inc. (AAPL) contracts with NVIDIA Corporation (NVDA) as its sole source fabricator for advanced sub-3nm semiconductor wafer nodes under multi-year exclusive capacity reservation agreements under an executed multi-year commercial framework agreement.\n\nTriples:",
  "completion": " (:Company {name: \"Apple Inc.\", ticker: \"AAPL\"})-[:SOLE_SOURCE_DEPENDENT_ON {materiality: \"CRITICAL_TIER_1\", polarity: \"EXPANDING_BULLISH\", confidence: 0.96}]->(:Company {name: \"NVIDIA Corporation\", ticker: \"NVDA\"})",
  "metadata": {
    "provenance": "SEC_10K_ITEM1_SOLE_SOURCE_DEPENDENT_ON",
    "difficulty_score": 0.45,
    "confidence": 0.96,
    "is_hard_negative": false
  }
}
```

---

## 6. Citation & Disclaimers

### Financial Disclaimer
> **Important Notice**: This dataset is curated exclusively for academic research, benchmark evaluation, and natural language processing in financial graph extraction and systemic contagion modeling. It does not constitute investment advice, legal counsel, or financial trading recommendations.

### Citation
```bibtex
@dataset{yarnball_sft_2026,
  author    = {Calderon, Camilo Emiliano},
  title     = {YarnBall S&P 500 Financial Graph & Contagion SFT Dataset},
  year      = {2026},
  publisher = {Hugging Face},
  version   = {1.0.0},
  url       = {https://huggingface.co/datasets/CamiloEmiliano/yarnball-sft}
}
```
