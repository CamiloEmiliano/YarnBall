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
"""


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
        commit_message="Add formal Dataset Card for YarnBall SFT v1.0",
    )

    api.upload_folder(
        folder_path=str(target_dir),
        repo_id=target_repo_id,
        repo_type="dataset",
        commit_message="Upload YarnBall SFT dataset splits",
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
