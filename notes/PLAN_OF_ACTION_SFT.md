# Master Plan of Action: Multi-Source Financial Graph & Qwen3:8b_YarnBall SFT Distillation

## Executive Overview
This document outlines the end-to-end plan for building **Qwen3:8b_YarnBall**—a multi-task financial intelligence model trained via task-specific supervised fine-tuning (SFT) and teacher distillation across a multi-source financial corpus spanning 2018 to 2025+.

The plan decouples the **GraphRAG production runtime** (`YarnBall`) from the **dedicated ML training lab** (`Qwen3_YarnBall_SFT`), establishes multi-year S&P 500 point-in-time snapshot archives tracked by DVC, incorporates 4 complementary data layers (SEC Filings, Historical News, Earnings Transcripts, Market/Macro series), and enforces a rigorous 5-tier Ground Truth verification protocol.

---

## The 4-Source Multi-Layer Data Matrix

```
┌────────────────────────────────────────────────────────────────────────┐
│ DATA SOURCES & INGESTION STREAMS                                       │
│                                                                        │
│ 1. SEC EDGAR Archive (2018–2025)                                       │
│    - Form 10-K (Item 1 Business, 1A Risks, Exhibit 21 Subsidiaries)    │
│    - Form 10-Q (Quarterly Segment & Supply Updates)                    │
│    - Form 8-K (Material Events, M&A, Defaults, Leadership Departures)  │
│    - Form 4 (Insider Transactions & Options Exercises)                 │
│                                                                        │
│ 2. Historical & Real-Time Financial News (2018–2025)                   │
│    - Historical News Archives (Kaggle/HuggingFace 2018–2025 financial) │
│    - Real-time Finnhub, RSS Feeds (Reuters, Investing.com, Yahoo)      │
│    - GDELT Event Stream (Global supply chain & geopolitical events)    │
│                                                                        │
│ 3. Earnings Call Transcripts & Corporate Guidance                      │
│    - Quarterly Earnings Conference Calls (Executive Remarks + Q&A)     │
│    - Form 8-K Item 2.02 Earnings Releases & Guidance Statements        │
│                                                                        │
│ 4. Quantitative Market & Macro Grounding                               │
│    - Historical Daily Price Series (OHLCV via yfinance / OpenBB)       │
│    - FRED Economic Data (Federal Funds Rate, CPI, Commodity Indices)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ REPOSITORY 1: YarnBall (Production GraphRAG Platform)                  │
│ - Historical Point-in-Time Snapshot Generator (SP500_YYYY_QN via DVC)  │
│ - PostgreSQL (10,422 CIK Master Registry, News, Embeddings)            │
│ - Memgraph Temporal Graph Database (Nodes, Temporal Weighted Edges)    │
│ - Hybrid Retriever, Guarded Text-to-CQL, and Chainlit UI               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
       [Exports Multi-Source Corpus & Gold Graph Snapshots]
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ REPOSITORY 2: Qwen3_YarnBall_SFT (Dedicated Training & MLOps Lab)      │
│ - 5-Tier Ground Truth Protocol (Deterministic Anchors, Triangulation,  │
│   Multi-Teacher Consensus, CIK Gates, Human Audit)                     │
│ - Multi-Task Dataset Curation: Extraction, Sentiment, Text-to-CQL, Rec │
│ - Statistical Balancing: GICS Sector Stratification, Negative Mining   │
│ - LoRA / QLoRA Cloud Training on Vast.ai / RunPod (Unsloth / TRL)      │
│ - 4-Tier Monitoring: W&B Tracking, Checkpoint F1, Downstream A/B Gate │
│ - Quantization to GGUF (Q4_K_M) & Ollama Modelfile Packaging           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Phase Breakdown & Workflows

### Phase 1: Multi-Source Historical Data Ingestion & Snapshot Engine (in `YarnBall`)
- **Objective**: Ingest multi-year historical filings, news, transcripts, and market data for the S&P 500 (2018–2025) with zero paid API limits.
- **Key Deliverables**:
  1. **Historical Constituent Registry**: Map point-in-time S&P 500 membership (2018–2025) to avoid survivorship bias.
  2. **SEC EDGAR Downloader (`tools/download_historical_sec.py`)**:
     - Batch fetch 10-K, 10-Q, 8-K, Form 4 across years; extract Item 1, 1A, and Exhibit 21.
  3. **Financial News Archive Ingestion (`tools/ingest_historical_news.py`)**:
     - Ingest 2018–2025 historical financial news corpora into PostgreSQL `financial_news_queue`.
  4. **Earnings Call Transcript Ingestion (`tools/ingest_transcripts.py`)**:
     - Download and stage quarterly earnings call transcripts and 8-K earnings releases.
  5. **Market Context Integrator (`tools/fetch_market_context.py`)**:
     - Pull historical OHLCV and event volatility windows via `yfinance` to ground price contagion.
  6. **Multi-Interval Snapshot Exporter (`graph/snapshot_manager.py`)**:
     - Export point-in-time Parquet snapshots (`data/snapshots/SP500_YYYY_QN/`).
     - Track all raw datasets and snapshots with **DVC** (`data/multi_source_historical.dvc`).

---

### Phase 2: Dedicated Training Repository Setup (`Qwen3_YarnBall_SFT`)
- **Objective**: Create an isolated repository for multi-task dataset curation, cloud training, and model evaluation.
- **Key Deliverables**:
  1. **Workspace Initialization**:
     - Independent `pyproject.toml` with PyTorch, CUDA 12, `unsloth`, `transformers`, `peft`, `trl`, `bitsandbytes`, `wandb`.
  2. **Inter-Repo Data Pipeline**:
     - Data loader consuming `YarnBall`'s DVC-tracked multi-source corpus.

---

### Phase 3: Ground Truth Verification & Multi-Task Dataset Curation
- **Objective**: Generate a 100% verified, balanced training dataset of 5,000–10,000 golden samples.

#### 3.1 The 5-Tier Ground Truth Verification Protocol
1. **Tier 1: Deterministic SEC Anchors (100% Confidence)**:
   - Form 10-K Exhibit 21 tables (subsidiary trees & jurisdictions).
   - Form 4 XML filings (insider buys/sells, executive titles, dates).
   - Form 8-K Item mappings (Item 1.01 Material Agreements, Item 2.01 M&A).
   - Master CIK registry (10,422 official SEC companies).
2. **Tier 2: Dual-Counterparty Triangulation**:
   - Under US GAAP ASC 280 (>10% revenue customer disclosure), cross-verify disclosures: If Supplier A reports Customer B, and Customer B lists Supplier A, confidence = 1.0.
3. **Tier 3: Multi-Teacher Consensus (Frontier Committee Voting)**:
   - For unstructured text, run 3 independent Frontier models from distinct AI labs: **Google Gemini** (Gemini 1.5 Pro / 2.0), **Anthropic Claude** (Claude 3.5 Sonnet), and **OpenAI** (GPT-4o).
   - *Design Decision*: Qwen is intentionally excluded from the teacher voting committee to prevent self-distillation bias and echo-chamber effects, ensuring the student model (`Qwen3:8b_YarnBall`) learns strictly from an external consensus of top frontier systems.
   - Require 2-out-of-3 agreement on exact entity pair, relationship type, and directionality; reject or flag disagreements.
4. **Tier 4: Programmatic Constraint & CIK Grounding Gates**:
   - Automated code gate: Source and target must resolve to valid CIKs in PostgreSQL `sec_companies` via trigram matching score > 0.85. Unlinked strings are automatically pruned.
   - Enforce closed relationship ontology and temporal metadata integrity.
5. **Tier 5: Human Golden Set Calibration**:
   - Expert audit on a statistically stratified cohort of 300–500 samples across all 11 GICS sectors to calibrate automated consensus accuracy (target Cohen's Kappa > 0.90).

#### 3.2 The 4 Core Training Tasks
1. **Task A (SEC Graph Extraction)**: Filing text -> Grounded entities & temporal edges.
2. **Task B (News Event & Sentiment Extraction)**: Breaking news article -> Directional event edge (`IMPACTS_NEGATIVELY`, `EXPANDS_PARTNERSHIP`).
3. **Task C (Guarded Text-to-CQL)**: Financial query -> Syntactically valid Memgraph Cypher query.
4. **Task D (Contagion & Action Recommendation)**: Subgraph + News context -> Multi-hop financial reasoning and portfolio action recommendations.

#### 3.3 Statistical Balancing Pipeline
- Stratified sampling across all 11 GICS economic sectors.
- Embedding clustering (`all-mpnet-base-v2`) to deduplicate legal/journalistic boilerplate.
- 15–20% hard-negative injection (macro text with ground truth `{"nodes": [], "edges": []}`).
- Train / Validation / Test split (80% / 10% / 10%).

---

### Phase 4: Cloud LoRA Training, Hardware Specs & Experiment Tracking
- **Objective**: Fine-tune `Qwen3:8b_YarnBall` on cloud GPU compute (Vast.ai / RunPod).
- **GPU Hardware & Compute Estimates**:
  - **Workload**: Base 8B model (`Qwen2.5-8B-Instruct` or `Qwen3-8B`), ~5,000–10,000 multi-task samples, 3 epochs (~25–30M tokens).
  - **Memory Footprint (4-bit QLoRA via Unsloth)**: ~4.0 GB base weights + ~0.1 GB LoRA adapters (r=32, alpha=64) + ~0.3 GB AdamW optimizer + ~5.0 GB activation memory with checkpointing = **~9.5 to 11.5 GB peak vRAM**.
  - **Memory Footprint (16-bit BF16 LoRA)**: ~16.0 GB base weights + ~0.6 GB LoRA/optimizer + ~6.0 GB activations = **~22.5 to 24.0 GB peak vRAM**.
  - **Recommended Hardware Tier**:
    - **Primary Target**: **1x Nvidia RTX 4090 (24 GB vRAM)** — Training Time: **~1.0 to 1.5 hours**, Cost: **~$0.50 to $0.75 USD** (via Vast.ai/RunPod).
    - **Enterprise Alternative**: **1x Nvidia A100 (80 GB SXM4)** — Training Time: **~35 to 45 minutes**, Cost: **~$0.80 to $1.20 USD**.
    - **System Specs**: 8 vCPUs, 32 GB System RAM, 50 GB NVMe storage.
- **Key Deliverables**:
  1. **Multi-Task LoRA Training (`train_multitask_lora.py`)**:
     - Target: LoRA rank r=32, alpha=64, targeting attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP projections (`gate_proj`, `up_proj`, `down_proj`).
     - Sequence Length: Clamped to 2,048 tokens.
  2. **Experiment Tracking (Weights & Biases)**:
     - Real-time logging of train/eval loss per task, learning rate decay, gradient norms, and GPU memory telemetry.
  3. **Multi-Task Checkpoint Evaluation (`eval_checkpoint.py`)**:
     - Evaluated every 100 steps on held-out test set: JSON Validity %, Entity/Relation F1, Cypher Syntax Accuracy, Sentiment F1, Directional Accuracy.

---

### Phase 5: Hugging Face Model Registry, GGUF Quantization & YarnBall Integration
- **Objective**: Host the trained model on Hugging Face Hub, quantize to GGUF, package for Ollama, and integrate into `YarnBall`.
- **Hugging Face Hub Private Model Registry (`hf.co/<username>/Qwen3-8B-YarnBall`)**:
  - **Multi-Format Artifact Storage**:
    - `adapter_model.safetensors` (~150 MB): Lightweight LoRA weights for incremental updates and fine-tuning.
    - `model.safetensors` (~16 GB): Merged 16-bit BF16 weights for high-throughput cloud endpoints (vLLM / TGI).
    - `Qwen3-8b-YarnBall-Q4_K_M.gguf` (~5.2 GB): 4-bit quantized binary for local laptop execution.
    - `Qwen3-8b-YarnBall-Q8_0.gguf` (~8.5 GB): 8-bit high-precision quantized binary.
  - **Model Card & Provenance**: Auto-generate `README.md` on HF with dataset statistics, training hyperparameters, loss curves, and evaluation benchmark scores.
  - **Release Versioning**: Tag releases via Git LFS tags (`v1.0-pilot`, `v2.0-sp500-multitask`).
- **Key Deliverables**:
  1. **Quantization Pipeline**: Export `Q4_K_M` and `Q8_0` GGUF binaries using `llama.cpp`.
  2. **Hugging Face Hub Deployment**: Automated push of adapter, merged weights, and GGUF files to private HF repository.
  3. **Ollama Integration**:
     - Direct pull from Hugging Face: `ollama run hf.co/<username>/Qwen3-8B-YarnBall:Q4_K_M`.
     - Local `Modelfile` configuration (`num_ctx 2048`, `num_predict 512`, `temperature 0.0`).
  4. **End-to-End Regression Verification in `YarnBall`**:
     - Update `YarnBall/.env` (`QWEN_MODEL_NAME=Qwen3:8b_YarnBall`).
     - Run full 84-test regression suite and 25-query golden benchmark harness (`rag/eval_harness.py`).

---

### Phase 6: Continuous Monitoring & Data Flywheel
- **Objective**: Maintain an active learning feedback loop to continuously retrain and improve.
- **Key Deliverables**:
  1. **Runtime Ingestion Telemetry**: Track JSON syntax errors, unlinked entity rates, and sentiment confidence scores.
  2. **Active Learning Rejection Queue**: Automatically stage failed or low-confidence samples to `data/rejection_queue.jsonl`.
  3. **Quarterly Batch Retraining**: Periodic incremental LoRA updates on newly released quarterly 10-Qs and earnings events.

---

## Verification & Acceptance Gates

| Gate | Target | Verification Method |
| :--- | :---: | :--- |
| **1. Dataset Ground Truth** | **100% JSON valid, >90% CIK linked, Kappa >0.90** | 5-Tier Ground Truth Protocol (CIK gates + Committee Consensus) |
| **2. Extraction F1** | **> 90% Entity F1, > 85% Relation F1** | 200-sample held-out golden test set |
| **3. Text-to-CQL Accuracy** | **> 95% Executable Cypher** | Automated Memgraph execution test suite |
| **4. GraphRAG A/B Gate** | **Zero PRG regression** | `YarnBall` 25-query golden evaluation harness |
| **5. Hardware Footprint** | **< 5.5 GB vRAM, < 3s latency** | Clamped context execution on local Ollama |
