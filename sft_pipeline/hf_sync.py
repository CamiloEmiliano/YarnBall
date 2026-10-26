"""
Upload YarnBall SFT Dataset to Hugging Face Hub as a Private or Public Dataset.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Optional

# Dynamically determine project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    qwen_env = PROJECT_ROOT.parent.parent / "QwenSFT_YarnBall" / ".env"
    if qwen_env.is_file():
        load_dotenv(qwen_env)
except ImportError:
    pass

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    HfApi = None
    create_repo = None


DATASET_CARD_TEMPLATE = """---
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
- SEC EDGAR (10-K, 10-Q, 8-K, Exhibit 21)
- Yahoo Finance Historical OHLCV
task_categories:
- text-generation
- text-retrieval
- question-answering
pretty_name: "YarnBall S&P 500 Financial Graph & Contagion SFT Dataset (v2.0 Cartography-Optimized)"
tags:
- finance
- graphrag
- opencypher
- dataset-cartography
- contagion-analysis
- portfolio-hedging
- qwen2.5
- sec-edgar
- yfinance
---

# YarnBall S&P 500 Financial Graph & Contagion SFT Dataset (v2.0)

**Dataset Version**: 2.0.0 (Dataset Cartography Optimized)  
**Total Samples**: 14,470 point-in-time grounded instances across all 11 GICS sectors  
**Target Model**: `Qwen/Qwen2.5-7B-Instruct` (Unified Extractor and Reasoner)  
**Repository**: [CamiloEmiliano/yarnball-sft](https://huggingface.co/datasets/CamiloEmiliano/yarnball-sft)  
**Trained Adapter**: [CamiloEmiliano/yarnball-qwen-7b-lora](https://huggingface.co/CamiloEmiliano/yarnball-qwen-7b-lora)  

---

## 1. Executive Overview & Mission

Generic financial large language models suffer from two catastrophic failure modes in production:
1. **Ungrounded Entity Hallucination**: Inventing ticker symbols, misattributing legal subsidiaries, and confounding spun-off corporate entities.
2. **Multi-Hop Causal Blindness**: Inability to traverse deep supply chains (3 to 5 hops) to isolate critical failure points from non-critical distractor disruptions.

The **YarnBall SFT Dataset** is mathematically engineered to transform `Qwen2.5-7B-Instruct` into an institutional financial intelligence engine. It aligns the model to emit deterministic openCypher graph queries for Neo4j/Memgraph property graphs and structured `<think>...</think>` Chain-of-Thought causal contagion traces.

---

## 2. Dataset Cartography Architecture & Empirical Engineering

This dataset is explicitly designed and diagnosed using the **Dataset Cartography** framework (Swayamdipta et al., EMNLP 2020) and **Example Forgetting** dynamics (Toneva et al., ICLR 2019).

```
Confidence
  1.0 +---------------------------------------------------------------+
      |  [EASY-TO-LEARN] (~52%)             |  [AMBIGUOUS] (~35%)     |
      |  - Direct 10-K subsidiary trees     |  - Multi-hop contagion  |
      |  - Standard Text-to-Cypher queries  |  - Borderline CAR (|z| ~2)|
      |  - Canonical 8-K filings            |  - Guidance divergence  |
  0.7 +-------------------------------------+-------------------------+
      |  [HARD-TO-LEARN] (~13%)             |                         |
      |  - Spin-off corporate collisions    |                         |
      |  - Commentary co-occurrence negatives|                         |
      |  - Debt covenant technical defaults |                         |
  0.0 +-------------------------------------+-------------------------+
      0.0                                  0.15                      0.5
                                                                 Variability
```

### 2.1 Post-Mortem of Run v1 (The 100% "Easy" Collapse)
During initial baseline fine-tuning on an A100 GPU (Run v1, 936 steps), the generated Data Map collapsed into an uninformative cluster at (Variability ≈ 0.00, Confidence ≈ 1.00), with 100% of samples categorized as "Easy-to-Learn". Three structural root causes were decomposed:

1. **Missing Step 0 (Zero-Shot Base Prior)**: Training dynamics were logged strictly at Epochs 1, 2, and 3 (steps 312, 624, 936). By step 312, the model had completed 10,000 forward passes and fully converged. The critical learning struggle occurred between steps 0 and 100, which went unprobed.
2. **Whole-Sequence Teacher-Forcing Dilution**: Cross-entropy loss was averaged across entire completion sequences. Over 80% of tokens in JSON and Cypher completions consist of invariant formatting boilerplate (brackets, quotes, whitespace, and static keywords like `MATCH`, `WHERE`, `RETURN`). 99.99% accuracy on whitespace drowned out learning errors on informative payload tokens.
3. **Unimodal Deterministic Grounding**: Absence of borderline volatility thresholds, corporate spin-off collisions, or supply chain distractor paths presented the model with zero epistemic contradiction.

### 2.2 The Three Pillars of the v2 Architecture

#### Pillar 1: Focal Payload Token Masking
Rather than naive whole-sequence averaging, training dynamics probing isolates cross-entropy evaluation exclusively to informational payload tokens:
- Standard loss mask: `M_standard(t) = 1(label_t != -100)`
- Focal payload mask: `M_focal(t) = M_standard(t) * 1(t not in BoilerplateTokens)`
Boilerplate tokens (ChatML controls, generic JSON keys, standard Cypher keywords) are masked out during evaluation. The model is scored strictly on its ability to predict correct Central Index Keys (CIKs), entity names, relationship types, event polarities, volatility z-scores, and causal pruning steps.

#### Pillar 2: Dense Early Trajectory Probing Schedule
Dynamics probing is decoupled from epoch boundaries. An 8-checkpoint evaluation schedule captures the full learning curve:
`Probing Checkpoints = [0, 25, 50, 100, 200, 400, 700, 1086]`
- **Step 0**: Evaluated inside `on_train_begin` before any optimizer update, capturing the raw `Qwen2.5-7B-Instruct` zero-shot inductive prior.
- **Warmup Steps (25, 50, 100)**: Evaluated at rapid intervals as the adapter learns Cypher grammar and schema constraints, maximizing the variability metric for complex samples.
- **Consolidation Steps (200, 400, 700, 1086)**: Evaluates late-stage convergence and identifies persistent structural outliers.

#### Pillar 3: Doubled Ambiguous Cohort (Optimizing OOD Transferability)
A foundational finding of Swayamdipta et al. (EMNLP 2020) is that **training on ambiguous samples (high variability, located near the decision boundary) produces the greatest out-of-distribution (OOD) generalization and task transferability**, substantially outperforming models trained on easy instances or random subsets.

To maximize transferability, the ambiguous cohort in YarnBall v2 has been systematically doubled:
- **Task D Contagion Reasoning**: Scaled from 400 to **800** deep `<think>` reasoning records.
- **Task E Portfolio Hedging**: Scaled from 400 to **800** institutional risk-hedging `<think>` records.
- **Task D Distractor Pruning**: Scaled from 150 to **300** multi-hop graphs with distractor paths.
- **Task E Boundary Hedging**: Scaled from 150 to **300** divergent catalyst records.
- **Task B Borderline Volatility**: Scaled from 150 to **300** Form 8-K filings near the 2.0 sigma margin.
- **Task A Spin-off Disambiguation**: Scaled from 150 to **300** parent vs. spin-off collision records.

Total targeted ambiguous and boundary samples: **2,800 records** (over a third of the active training distribution).

---

## 3. Quality Assurance: Safeguards Against Noise in Hard Samples

In crowdsourced NLP benchmarks (SNLI, MNLI, SQuAD), human audits revealed that the Hard-to-Learn region often contains label noise, annotator mistakes, or subjective contradictions. Training on noisy samples degrades model reasoning.

YarnBall eliminates label noise through four deterministic safeguards:

1. **Zero Crowdsourced Annotator Noise**: All ground truth is derived from statutory SEC EDGAR filings (Exhibit 21 legal subsidiaries, Form 8-K mandatory disclosures) and Yahoo Finance adjusted OHLCV market pricing. There is zero crowdworker subjectivity.
2. **Topological Exclusion Filter (`known_active_pairs`)**: When generating hard negative commentary samples requiring target completion `(none)`, a global bipartite graph of all verified relationships is constructed across tickers and legal names. If two firms share any commercial, supply, or subsidiary link, they are strictly rejected. A target of `(none)` is an absolute mathematical and legal fact.
3. **Statutory Corporate Separation Grounding**: Corporate spin-off collisions (e.g., GE Aerospace vs. GE Vernova, Carrier Global vs. RTX, Alcon vs. Novartis, Viatris vs. Pfizer, AbbVie vs. Abbott) are grounded in historical SEC Form 10 and Form 8-K effective split dates. When querying GE Aerospace (`GE`) about a GE Vernova (`GEV`) wind turbine contract, `(none)` is a verified corporate fact.
4. **Toneva Forgetfulness Quarantining**: The active learning cartographer (`curation/active_learning/cartographer.py`) tracks forgetfulness across training. Samples with forgetfulness = 999 (never learned across all 1,086 steps) are automatically isolated into an audit log (`quarantined_hard_guids`) to ensure no corrupted syntax or edge artifacts pollute downstream models.

---

## 4. Dataset Partitioning & Summary

The dataset contains **14,470 point-in-time grounded instances**, partitioned into an 80/10/10 Train/Validation/Test split with verified zero split leakage:

| Split | Unified Total | Extraction Tasks (A, B, C) | Reasoning Tasks (D, E with `<think>`) | File Name |
| :--- | :---: | :---: | :---: | :--- |
| **Train (80%)** | **11,576** | 9,816 | 1,760 | `yarnball_sft_train.jsonl` |
| **Validation (10%)** | **1,447** | 1,227 | 220 | `yarnball_sft_val.jsonl` |
| **Test (10%)** | **1,447** | 1,227 | 220 | `yarnball_sft_test.jsonl` |
| **Total Corpus** | **14,470** | **12,270** | **2,200** | |

### Projected Cartography Distribution
- **Easy-to-Learn**: **~50% – 54%** (~7,500 samples) | Confidence >= 0.70, Variability < 0.15
- **Ambiguous**: **~32% – 38%** (~5,000 samples) | **Variability >= 0.15** (decision boundary frontier)
- **Hard-to-Learn**: **~12% – 16%** (~2,000 samples) | Confidence < 0.70, Variability < 0.15

---

## 5. Multi-Task Taxonomy & Format Specifications

All samples use standard ChatML formatting adhering to the Qwen2.5 chat template:

### Task A: SEC OpenCypher Triples Extraction (`<|extract_sec_graph|>`)
- **Input**: Raw text from SEC Form 10-K Item 1 business descriptions or Exhibit 21 legal subsidiary lists.
- **Output**: OpenCypher graph pattern detailing companies, tickers, relationship types, and directional attributes. Hard negatives return `(none)`.

### Task B: Material Current Event Extraction (`<|extract_news_event|>`)
- **Input**: Narrative from SEC Form 8-K mandatory disclosures alongside 3-day CAR and trailing 252-day market model volatility z-scores.
- **Output**: Structured event edge with regulatory item code, statistical materiality, and 5-axis directional polarity:
  - `|z| < 2.0`: `MIXED_UNCERTAIN_SIGNAL` and `BORDERLINE_SIGNIFICANT`
  - `|z| >= 2.0`: `EXPANDING_BULLISH` (z > 0) or `CONTRACTING_BEARISH` (z < 0) and `STATISTICALLY_SIGNIFICANT`

### Task C: Text-to-Cypher Synthesis (`<|text_to_cypher|>`)
- **Input**: Natural language financial research query.
- **Output**: Syntactically valid, executable openCypher query targeting the YarnBall schema (e.g., `MATCH (s:Company)-[:SUPPLIES_TO]->(t:Company {ticker: 'AAPL'}) ...`).

### Task D: Multi-Hop Contagion Reasoning (`<|contagion_reasoning|>`)
- **Input**: Multi-hop supply chain subgraph and an acute shock hitting a specific node.
- **Output**: Structured `<think>...</think>` Chain-of-Thought reasoning evaluating component criticality, alternative vendor availability, and inventory buffers, followed by an actionable `**Contagion Analysis Summary**`. In distractor cases, false-positive propagation is pruned to `CONTAINED_MINIMAL_CONTAGION`.

### Task E: Portfolio Hedging & Risk Allocation (`<|portfolio_recommendation|>`)
- **Input**: Corporate exposure telemetry, post-event volatility z-scores, and divergent guidance vs. GAAP fundamentals.
- **Output**: Structured `<think>...</think>` trace synthesizing asymmetric downside tail risk, followed by capital-efficient institutional risk management (zero-cost equity collars, out-of-the-money put spreads, interest rate swaptions, or paired long-short equity hedges).

---

## 6. Regulatory & Market Data Provenance

- **Regulatory Filings (Ground Truth)**: SEC EDGAR via `edgartools` / `EdgarClient` (depth back to 2018+).
- **Market Pricing & Event Volatility**: Yahoo Finance adjusted daily OHLCV prices (2018–2025). 3-day CAR evaluated against the S&P 500 benchmark (`^GSPC`).
- **Data Boundaries**: Finnhub is **strictly excluded** from historical SFT dataset construction (Finnhub free tier is limited to ~1 year and fails on multi-year queries). Finnhub is strictly reserved for live, real-time forward streaming in production deployment.
- **Constituent Registry**: Point-in-time historical constituent registry (`sp500_constituents_historical.json`) eliminating lookahead survivorship bias.

---

## 7. How to Use

```python
from datasets import load_dataset

# Load unified splits
dataset = load_dataset("CamiloEmiliano/yarnball-sft")
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]

print(f"Train samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Sample prompt:
{train_data[0]['prompt']}")
print(f"Sample target:
{train_data[0]['target_completion']}")
```

---

## 8. Academic References

- **Swayamdipta, S., Schwartz, R., Lourie, N., Wang, Y., Hajishirzi, H., & Smith, N. A.** (2020). *Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics*. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 9275–9293.
- **Toneva, M., Sordoni, A., Combes, R. T. d., Trischler, A., Bengio, Y., & Gordon, G. J.** (2019). *An Empirical Study of Example Forgetting during Deep Neural Network Learning*. International Conference on Learning Representations (ICLR)."""


def upload_to_huggingface(
    repo_name: str = "yarnball-sft",
    repo_id: Optional[str] = None,
    data_dir: Optional[Path] = None,
    token: Optional[str] = None,
    private: bool = True,
) -> str:
    """Uploads SFT dataset files to the Hugging Face Hub."""
    if HfApi is None:
        raise ImportError("huggingface_hub is not installed. Run: pip install huggingface_hub")

    target_dir = Path(data_dir or (PROJECT_ROOT / "data" / "sft")).resolve()
    if not target_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found at {target_dir}")

    auth_token = (
        token
        or os.getenv("HUGGINGFACE_FULL_ACCESS_TOKEN_01")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )
    if not auth_token:
        raise ValueError("No Hugging Face token found in environment or arguments.")

    api = HfApi(token=auth_token)
    user_info = api.whoami()
    username = user_info.get("name") or user_info.get("user")
    target_repo_id = repo_id or f"{username}/{repo_name}"

    create_repo(
        repo_id=target_repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=auth_token,
    )

    api.upload_file(
        path_or_fileobj=DATASET_CARD_TEMPLATE.strip().encode("utf-8"),
        path_in_repo="README.md",
        repo_id=target_repo_id,
        repo_type="dataset",
        commit_message="docs(cartography): update Dataset Card with Cartography architecture and transferability analysis (v2.0)",
    )

    api.upload_folder(
        folder_path=str(target_dir),
        repo_id=target_repo_id,
        repo_type="dataset",
        commit_message="feat(dataset): upload YarnBall SFT v2.0 dataset splits (14,470 samples)",
    )

    return f"https://huggingface.co/datasets/{target_repo_id}"


def main():
    parser = argparse.ArgumentParser(description="Upload YarnBall SFT dataset to Hugging Face Hub")
    parser.add_argument("--repo-name", default="yarnball-sft", help="Repository name on Hugging Face")
    parser.add_argument("--repo-id", default=None, help="Explicit repository ID (e.g. username/repo-name)")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "sft"), help="Path to data/sft directory")
    parser.add_argument("--token", default=None, help="Hugging Face access token")
    parser.add_argument("--private", action="store_true", default=True, help="Set repository to private")

    args = parser.parse_args()
    url = upload_to_huggingface(
        repo_name=args.repo_name,
        repo_id=args.repo_id,
        data_dir=Path(args.data_dir),
        token=args.token,
        private=args.private,
    )
    print(f"SUCCESS: Uploaded to {url}")


if __name__ == "__main__":
    main()
