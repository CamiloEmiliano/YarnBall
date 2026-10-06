"""
Upload YarnBall SFT Dataset to Hugging Face Hub as a Private Dataset.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

# Dynamically determine project root and runtime working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURRENT_WORKING_DIR = Path(os.environ.get("PWD", os.getcwd())).resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Attempt dynamic dotenv loading from runtime CWD, project root, or sibling repos
try:
    from dotenv import load_dotenv

    candidate_env_paths = [
        CURRENT_WORKING_DIR / ".env",
        PROJECT_ROOT / ".env",
        CURRENT_WORKING_DIR.parent / "QwenSFT_YarnBall" / ".env",
        PROJECT_ROOT.parent / "QwenSFT_YarnBall" / ".env",
        PROJECT_ROOT.parent.parent / "QwenSFT_YarnBall" / ".env",
    ]

    for env_path in candidate_env_paths:
        if env_path.is_file():
            load_dotenv(env_path)
            break
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
    parser.add_argument("--repo-id", default=None, help="Explicit repository ID (e.g. username/repo-name). Overrides --repo-name.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "sft"), help="Path to data/sft directory")
    parser.add_argument("--env-file", default=None, help="Explicit path to a .env file to load credentials from")
    parser.add_argument("--token", default=None, help="Hugging Face access token (overrides environment variables)")
    
    # Python 3.9+ BooleanOptionalAction for clean --private / --no-private flags
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True, help="Set repository to private (default: True)")
    else:
        parser.add_argument("--private", action="store_true", default=True, help="Set repository to private (default: True)")
        parser.add_argument("--public", dest="private", action="store_false", help="Set repository to public")

    args = parser.parse_args()

    # Load custom env file if specified
    if args.env_file:
        custom_env = Path(args.env_file).resolve()
        if custom_env.is_file():
            try:
                from dotenv import load_dotenv
                load_dotenv(custom_env, override=True)
            except ImportError:
                pass
        else:
            print(f"Warning: Specified --env-file not found: {custom_env}")

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        print(f"Error: Dataset directory not found at {data_dir}")
        sys.exit(1)

    # Token lookup: CLI flag > environment variables
    token = (
        args.token
        or os.getenv("HUGGINGFACE_FULL_ACCESS_TOKEN_01")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )
    if not token:
        print("Error: No Hugging Face token found in environment, CLI flags, or loaded .env file.")
        print("Provide via --token, export HF_TOKEN, or set HUGGINGFACE_FULL_ACCESS_TOKEN_01 in your .env.")
        sys.exit(1)

    api = HfApi(token=token)
    user_info = api.whoami()
    username = user_info.get("name") or user_info.get("user")

    if args.repo_id:
        repo_id = args.repo_id
    else:
        repo_id = f"{username}/{args.repo_name}"

    print(f"Authenticated as Hugging Face user: {username}")
    print(f"Target Dataset Repository: {repo_id} (Private: {args.private})")

    # 1. Ensure repository exists on Hub
    print(f"Ensuring repository {repo_id} exists on Hugging Face Hub...")
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
        token=token,
    )

    # 2. Upload dataset card (README.md) in-memory without modifying local DVC tracked directory
    print("Uploading Dataset Card (README.md)...")
    api.upload_file(
        path_or_fileobj=DATASET_CARD_TEMPLATE.strip().encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add formal Dataset Card for YarnBall SFT v1.0",
    )

    # 3. Upload dataset files from local directory
    print(f"Uploading dataset files from {data_dir} to https://huggingface.co/datasets/{repo_id} ...")
    api.upload_folder(
        folder_path=str(data_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload YarnBall v1.0 SFT dataset splits (12,470 records)",
    )

    print("\n" + "=" * 65)
    print("SUCCESS: Dataset uploaded to Hugging Face Hub!")
    print(f"URL: https://huggingface.co/datasets/{repo_id}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
