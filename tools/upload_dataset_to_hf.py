"""
Upload YarnBall SFT Dataset to Hugging Face Hub as a Private Dataset.

Reads HUGGINGFACE_FULL_ACCESS_TOKEN_01 from .env, validates authentication,
creates the private dataset repository if it doesn't already exist, and
uploads all dataset splits and metadata.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Attempt dotenv loading from current repo or QwenSFT_YarnBall
try:
    from dotenv import load_dotenv
    qwen_env = Path("/home/caspe/practice/QwenSFT_YarnBall/.env")
    local_env = PROJECT_ROOT / ".env"
    if qwen_env.is_file():
        load_dotenv(qwen_env)
    elif local_env.is_file():
        load_dotenv(local_env)
except ImportError:
    pass

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("Error: huggingface_hub is not installed. Run: pip install huggingface_hub")
    sys.exit(1)


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
- information-retrieval
- question-answering
task_ids:
- knowledge-graph-extraction
- reasoning
pretty_name: "YarnBall S&P 500 Financial Graph & Contagion SFT Dataset"
tags:
- finance
- graphrag
- opencypher
- contagion-analysis
- portfolio-hedging
- qwen2.5
---

# YarnBall S&P 500 Financial Graph & Contagion SFT Dataset (v1.0.0)

This is the official Supervised Fine-Tuning (SFT) dataset for **YarnBall**, designed to train a unified `Qwen2.5-7B-Instruct` financial intelligence model spanning 5 core extraction and high-cognitive reasoning tasks:

- **Task A (`<|extract_sec_graph|>`)**: SEC Form 10-K Item 1, 1A, and Exhibit 21 OpenCypher Triples DSL Extraction.
- **Task B (`<|extract_news_event|>`)**: Breaking financial news event extraction with temporal edge validity and 5-axis directional polarity (`EXPANDING_BULLISH`, `CONTRACTING_BEARISH`, `DISRUPTIVE_SHOCK`, `NEUTRAL_STABLE`).
- **Task C (`<|text_to_cypher|>`)**: High-accuracy natural language questions mapped to executable Cypher queries across Memgraph / Neo4j schemas.
- **Task D (`<|contagion_reasoning|>`)**: Multi-hop shock propagation and counterparty contagion reasoning using `<think>` Chain-of-Thought.
- **Task E (`<|portfolio_recommendation|>`)**: Portfolio risk hedging and exposure rebalancing (collars, puts, swaptions) with `<think>` Chain-of-Thought.

## Dataset Structure

- `yarnball_sft_train.jsonl`: 9,976 records (80%)
- `yarnball_sft_val.jsonl`: 1,247 records (10%)
- `yarnball_sft_test.jsonl`: 1,247 records (10%)
- `extraction_tasks_*.jsonl`: Modular splits for Tasks A, B, and C (11,670 total records)
- `reasoning_tasks_*.jsonl`: Modular splits for Tasks D and E (800 total records)
- `dataset_summary.json`: Formal schema specifications and generation metrics

## Grounding & Invariants
- **Point-in-Time Regulatory Truth**: Sourced strictly from SEC EDGAR filings and S&P 500 constituent histories.
- **Hard Negative Boundary Refusal**: Rigorously synthesized non-relational market co-occurrences targeting `(none)` with capped heuristic confidence (0.65).
- **Directional Orientation Guard**: 52 active and passive voice syntactic templates across all 11 GICS sectors with invariant semantic role assignment.
- **Zero Cross-Split Leakage**: 100% disjoint prompt signatures verified prior to export.
"""


def main():
    parser = argparse.ArgumentParser(description="Upload YarnBall SFT dataset to Hugging Face Hub")
    parser.add_argument("--repo-name", default="yarnball-sft", help="Repository name on Hugging Face (default: yarnball-sft)")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "sft"), help="Path to data/sft directory")
    parser.add_argument("--private", action="store_true", default=True, help="Set repository to private (default: True)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"Error: Dataset directory not found at {data_dir}")
        sys.exit(1)

    # Token lookup
    token = (
        os.getenv("HUGGINGFACE_FULL_ACCESS_TOKEN_01")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        print("Error: No Hugging Face token found in environment or .env file.")
        print("Set HUGGINGFACE_FULL_ACCESS_TOKEN_01 in /home/caspe/practice/QwenSFT_YarnBall/.env or export HF_TOKEN.")
        sys.exit(1)

    api = HfApi(token=token)
    user_info = api.whoami()
    username = user_info.get("name")
    repo_id = f"{username}/{args.repo_name}"

    print(f"Authenticated as Hugging Face user: {username}")
    print(f"Target Dataset Repository: {repo_id} (Private: {args.private})")

    # 1. Ensure README.md / dataset card exists in data/sft
    readme_path = data_dir / "README.md"
    if not readme_path.exists():
        print("Creating Dataset Card (README.md) in data/sft/...")
        readme_path.write_text(DATASET_CARD_TEMPLATE, encoding="utf-8")

    # 2. Create repo on Hub if needed
    print(f"Ensuring repository {repo_id} exists on Hugging Face Hub...")
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
        token=token,
    )

    # 3. Upload entire folder
    print(f"Uploading files from {data_dir} to https://huggingface.co/datasets/{repo_id} ...")
    api.upload_folder(
        folder_path=str(data_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload YarnBall v1.0 SFT dataset (12,470 point-in-time records)",
    )

    print("\n" + "=" * 65)
    print("SUCCESS: Dataset uploaded to Hugging Face Hub!")
    print(f"URL: https://huggingface.co/datasets/{repo_id}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
