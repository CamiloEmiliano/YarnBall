# Master Plan of Action: Multi-Source Financial Graph & Quantized Dual-Model SFT Distillation (Qwen2.5-3B & Qwen3-8B)

## Executive Overview
This document outlines the end-to-end plan for building **YarnBall's Dual-Model Intelligence Engine**—a pair of task-specialized, quantized financial models trained via supervised fine-tuning (SFT) and teacher distillation across a multi-source financial corpus spanning 2018 to 2025+:

1. **`Qwen2.5-3b_YarnBall_Extractor` (4-bit Q4_K_M)**: Low-cognitive workhorse for high-throughput, syntax-constrained entity extraction, CIK normalization, and Text-to-Cypher generation (**~2.1 GB vRAM**, **~110-130 tok/s**).
2. **`Qwen3-8b_YarnBall_Reasoner` (4-bit Q4_K_M)**: High-cognitive analyst for multi-hop shock propagation, systemic risk contagion, and portfolio hedging synthesis (**~5.2 GB vRAM**, **~35-48 tok/s**).

The plan decouples the **GraphRAG production runtime** (`YarnBall`) from the **dedicated ML training lab** (`Qwen_YarnBall_SFT`), establishes multi-year S&P 500 point-in-time snapshot archives tracked by DVC, incorporates 4 complementary data layers (SEC Filings, Historical News, Earnings Transcripts, Market/Macro series), and enforces a rigorous 5-tier Ground Truth verification protocol.

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
│ - Dual-Model Router (Dispatches k <= 2 to 3B, k >= 3 to 8B)            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
       [Exports Multi-Source Corpus & Gold Graph Snapshots]
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ REPOSITORY 2: Qwen_YarnBall_SFT (Dedicated Training & MLOps Lab)       │
│ - 5-Tier Ground Truth Protocol (Deterministic Anchors, Triangulation,  │
│   Multi-Teacher Consensus, CIK Gates, Human Audit)                     │
│ - Multi-Task Dataset Curation: Extraction, Sentiment, Text-to-CQL, Rec │
│ - 30/50/20 Difficulty Curriculum & Rare Relationship Over-Sampling     │
│ - Dual QLoRA Training: Qwen2.5-3B & Qwen3-8B on Cloud GPUs             │
│ - GGUF 4-bit Quantization (Q4_K_M) & Hugging Face Hub Registry         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Phase Breakdown & Workflows

### Phase 1: S&P 500 Multi-Source Historical Data Ingestion & Snapshot Engine (in `YarnBall`)
- **Objective**: Ingest multi-year historical filings, news, transcripts, and market data specifically bounded to the **S&P 500** corporate universe (2018–2025) with zero paid API limits.
- **Scope & Universe Boundary**:
  - Primary universe is strictly bounded to the **500 S&P 500 constituents** per year (~80% of total US market capitalization).
  - External counterparties (e.g. key international foundries like `TSM`, `ASML` or major private partners like `OpenAI`) are linked when explicitly disclosed by S&P 500 filers.
- **Key Deliverables**:
  1. **S&P 500 Historical Constituent Registry**: Map point-in-time S&P 500 membership (2018–2025) to account for index additions/deletions and avoid survivorship bias.
  2. **SEC EDGAR Downloader (`tools/download_historical_sec.py`)**:
     - Batch fetch 10-K, 10-Q, 8-K, Form 4 across 2018–2025 for S&P 500 constituents (~500 10-Ks and ~1,500 10-Qs per year).
     - Extract plain-text sections (`item_1_business.txt`, `item_1a_risk_factors.txt`, `subsidiaries.json`).
  3. **Financial News Archive Ingestion (`tools/ingest_historical_news.py`)**:
     - Ingest 2018–2025 historical financial news corpora into PostgreSQL `financial_news_queue` filtered by S&P 500 tickers.
  4. **Earnings Call Transcript Ingestion (`tools/ingest_transcripts.py`)**:
     - Download and stage quarterly earnings call transcripts and 8-K earnings releases.
  5. **Market Context Integrator (`tools/fetch_market_context.py`)**:
     - Pull historical OHLCV and event volatility windows via `yfinance` to ground price contagion.
  6. **Multi-Interval Snapshot Exporter (`graph/snapshot_manager.py`)**:
     - Export point-in-time Parquet snapshots (`data/snapshots/SP500_YYYY_QN/`).
     - Track all raw datasets and snapshots with **DVC** (`data/multi_source_historical.dvc`).

---

### Phase 2: Dedicated Training Repository Setup (`Qwen_YarnBall_SFT`)
- **Objective**: Create an isolated repository for multi-task dataset curation, dual-model cloud training, and benchmark evaluation.
- **Key Deliverables**:
  1. **Workspace Initialization**:
     - Independent `pyproject.toml` with PyTorch, CUDA 12, `unsloth`, `transformers`, `peft`, `trl`, `bitsandbytes`, `wandb`.
  2. **Inter-Repo Data Pipeline**:
     - Data loader consuming `YarnBall`'s DVC-tracked multi-source corpus.

---

### Phase 3: Ground Truth Verification, 5-Axis Taxonomy & Multi-Task Dataset Curation
- **Objective**: Generate a 100% verified, balanced, multi-task dataset of 5,000–10,000 golden training samples partitioned for dual-model specialization.

#### 3.1 The 5-Tier Ground Truth Verification Protocol
1. **Tier 1: Deterministic SEC Anchors (100% Confidence)**:
   - Form 10-K Exhibit 21 tables (subsidiary trees & jurisdictions).
   - Form 4 XML filings (insider buys/sells, executive titles, dates).
   - Form 8-K Item mappings (Item 1.01 Material Agreements, Item 2.01 M&A).
   - Master CIK registry (10,422 official SEC companies).
2. **Tier 2: Dual-Counterparty Triangulation**:
   - Under US GAAP ASC 280 (>10% revenue customer disclosure), cross-verify disclosures: If Supplier A reports Customer B, and Customer B lists Supplier A, confidence = 1.0.
3. **Tier 3: Multi-Teacher Consensus (Frontier Committee Voting)**:
   - For unstructured text, run 3 independent Frontier models: **Google Gemini** (Gemini 1.5 Pro / 2.0), **Anthropic Claude** (Claude 3.5 Sonnet), and **OpenAI** (GPT-4o).
   - *Design Decision*: Qwen is intentionally excluded from the teacher voting committee to prevent self-distillation bias and echo-chamber effects, ensuring the student models learn strictly from an external consensus of top frontier systems.
   - Require 2-out-of-3 agreement on exact entity pair, relationship type, and directionality; reject or flag disagreements.
4. **Tier 4: Programmatic Constraint & CIK Grounding Gates**:
   - Automated code gate: Source and target must resolve to valid CIKs in PostgreSQL `sec_companies` via trigram matching score > 0.85. Unlinked strings are automatically pruned.
   - Enforce closed relationship ontology and temporal metadata integrity.
5. **Tier 5: Human Golden Set Calibration**:
   - Expert audit on a statistically stratified cohort of 300–500 samples across all 11 GICS sectors to calibrate automated consensus accuracy (target Cohen's Kappa > 0.90).

#### 3.2 The 5-Axis Financial Labeling Taxonomy
Every extracted triple is classified across 5 orthogonal dimensions:
- **Axis 1 (Entity Typology)**: `Company` (S&P 500 CIK/Ticker), `Subsidiary`, `Product/Platform`, `Commodity/Component`, `RegulatoryBody`, `Person`.
- **Axis 2 (Relational Ontology)**: `SUPPLIES_TO`, `CUSTOMER_OF`, `PARTNERED_WITH`, `SUBSIDIARY_OF`, `COMPETES_WITH`, `LICENSES_FROM`, `EXPOSED_TO_RISK`.
- **Axis 3 (Directional Polarity & Sentiment)**: `EXPANDING_BULLISH` (+1), `NEUTRAL_STABLE` (0), `CONTRACTING_BEARISH` (-1), `DISRUPTIVE_SHOCK` (-2).
- **Axis 4 (Financial Materiality & Criticality)**: `CRITICAL_TIER_1` (>10% revenue / sole-source), `MATERIAL_TIER_2` (major multi-year), `COMMODITY_TIER_3` (interchangeable off-the-shelf).
- **Axis 5 (Temporal Provenance & Lifecycle)**: Status (`ACTIVE_CURRENT`, `TERMINATED`), `valid_from`, `valid_to`, `source_provenance` (`SEC_10K_ITEM1`, `SEC_8K`, `REUTERS_NEWS`).

#### 3.3 Cognitive Load Task Classification & Quantized Dual-Model Architecture
Tasks are partitioned by **Cognitive Load** and topological hop distance (k <= 2 vs k >= 3):

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. LOW COGNITIVE LOAD WORKHORSE: Qwen2.5:3b_YarnBall_Extractor (Q4_K_M)│
│ ────────────────────────────────────────────────────────────────────── │
│ Role: High-throughput, deterministic syntax & parsing (k <= 2 hops)    │
│ Quantization: 4-bit GGUF (Q4_K_M)                                      │
│ Memory Footprint: ~2.1 GB vRAM (100% in GPU, 110-130 tokens/sec)       │
│ Grammar Enforcement: GBNF Cypher Triples DSL Grammars                  │
│ Teacher Pipeline: Gemini 2.0 Flash + Deterministic CIK Constraint Gates│
│                                                                        │
│ - Task A (`<|extract_sec_graph|>`): SEC 10-K/10-Q/8-K -> Triples DSL   │
│ - Task B (`<|extract_news_event|>`): Breaking News -> Directional Edge │
│ - Task C (`<|text_to_cypher|>`): Query + Schema -> Valid Memgraph CQL  │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│ 2. HIGH COGNITIVE LOAD ANALYST: Qwen3:8b_YarnBall_Reasoner (Q4_K_M)    │
│ ────────────────────────────────────────────────────────────────────── │
│ Role: Multi-hop reasoning, shock propagation, hedging (k >= 3 hops)    │
│ Quantization: 4-bit GGUF (Q4_K_M)                                      │
│ Memory Footprint: ~5.2 GB vRAM (35-48 tokens/sec, fits 8GB vRAM)      │
│ Reasoning Structure: Structured Chain-of-Thought (<think> blocks)      │
│ Teacher Pipeline: Frontier Committee (Gemini 1.5 Pro + Claude 3.5 + O1)│
│                                                                        │
│ - Task D (`<|contagion_reasoning|>`): Subgraph -> 2nd/3rd order impact │
│ - Task E (`<|portfolio_recommendation|>`): Risk synthesis -> Strategy  │
└────────────────────────────────────────────────────────────────────────┘
```

#### 3.4 The 30 / 50 / 20 Difficulty Curriculum & Metadata Envelope
To prevent catastrophic forgetting and ensure balanced capability:
- **30% Easy (Syntax & Structural Baseline)**: 1-hop direct relationships, clean legal names, anchor Cypher syntax.
- **50% Medium (Multi-Entity & Transitive 2-Hop)**: Paragraphs with 3–5 overlapping entities, brand aliases, and directional sentiment.
- **20% Hard (Multi-Hop Contagion & Hard Negatives)**: Complex 3+ hop supply disruptions, revenue elasticity, and hard negatives (zero corporate links with empty graph output `(none)`).

**Standard Dataset Record Metadata Envelope (JSONL)**:
```json
{
  "sample_id": "SFT_SP500_2024_0842",
  "metadata": {
    "task_type": "EXTRACT_SEC_GRAPH",
    "cognitive_tier": "LOW",
    "difficulty": "MEDIUM",
    "graph_hops": 2,
    "gics_sector": "Information Technology",
    "assigned_student": "QWEN_2.5_3B_EXTRACTOR",
    "teacher_origin": "GEMINI_FLASH_CIK_GATE",
    "curriculum_weight": 1.0
  },
  "prompt": "<|extract_sec_graph|>The Company relies on TSMC for 3nm wafer fabrication...",
  "target_completion": "(TSMC:Company {ticker:\"TSM\"})-[:SUPPLIES_TO {nature:\"3nm wafer fabrication\", polarity:\"NEUTRAL_STABLE\", materiality:\"CRITICAL_TIER_1\"}]->(Apple Inc.:Company {ticker:\"AAPL\"})"
}
```

#### 3.5 Statistical Balancing Pipeline
- Stratified sampling across all 11 GICS economic sectors.
- Embedding clustering (`all-mpnet-base-v2`) to deduplicate legal/journalistic boilerplate.
- Automated Python partitioning into Qwen 2.5 (3B) and Qwen 3 (8B) training sets.
- Train / Validation / Test split (80% / 10% / 10%).

#### 3.6 Rare Relationship Over-Sampling & Lexical Trigger Harvesting
To prevent class collapse where the model only predicts common relationships (`SUBSIDIARY_OF`, `SUPPLIES_TO`) while missing high-value risk edges:
1. **Targeted Lexical Harvesting (Regex Trigger Mining)**:
   - `SOLE_SOURCE_DEPENDENT_ON`: Harvest paragraphs with `"sole source"`, `"single source supplier"`, `"no alternate supplier"`, `"solely dependent on"`.
   - `LICENSES_FROM / LICENSES_TO`: Harvest `"cross-licensing agreement"`, `"patent license"`, `"exclusive royalty"`, `"technology licensing"`.
   - `EXPOSED_TO_CHOKEPOINT`: Harvest `"export restriction"`, `"critical bottleneck"`, `"ITAR compliance"`, `"single point of failure"`.
   - `DEFAULTED_ON / TERMINATED`: Harvest `"terminated for cause"`, `"notice of default"`, `"covenant breach"`, `"contract cancellation"`.
2. **Inverse-Frequency Floor Capping**:
   - Cap common relations (`SUBSIDIARY_OF`, generic `SUPPLIES_TO`) at a maximum of ~500 samples.
   - Enforce a hard minimum floor of at least **250 verified training samples** for every rare relationship class.
3. **Targeted Form 8-K Event Item Mining**:
   - Specifically pull rare corporate disruption filings: Item 1.02 (Termination of Material Agreement), Item 1.03 (Bankruptcy), Item 2.04 (Acceleration of Direct Financial Obligation / Default).
4. **Counterfactual Entity-Swap Augmentation**:
   - For ultra-rare legal/settlement patterns, generate 2–3 synthetic variants by swapping entity names while preserving identical Cypher triple relationship grammar.

---

### Phase 4: Cloud LoRA Training, Hardware Specs & Experiment Tracking
- **Objective**: Fine-tune both `Qwen2.5-3B` and `Qwen3-8B` on cloud GPU compute (Vast.ai / RunPod).
- **GPU Hardware & Compute Estimates**:
  - **Workload**:
    - Extractor: Base 3B model (`Qwen2.5-3B-Instruct`), ~4,000 extraction samples, 3 epochs.
    - Reasoner: Base 8B model (`Qwen3-8B`), ~4,000 multi-hop reasoning samples, 3 epochs.
  - **Training Compute Requirements (4-bit QLoRA via Unsloth)**:
    - 3B Model Peak vRAM: **~5.5 to 6.5 GB vRAM** (Training time: **~30 to 45 min** on RTX 4090).
    - 8B Model Peak vRAM: **~9.5 to 11.5 GB vRAM** (Training time: **~1.0 to 1.5 hours** on RTX 4090).
  - **Recommended Hardware Tier**:
    - **Primary Target**: **1x Nvidia RTX 4090 (24 GB vRAM)** — Total Combined Training Time: **~2.0 hours**, Total Cost: **~$1.00 to $1.50 USD** (via Vast.ai/RunPod).
    - **System Specs**: 8 vCPUs, 32 GB System RAM, 50 GB NVMe storage.
- **Key Deliverables**:
  1. **Dual-Model LoRA Training Scripts (`train_extractor_3b.py`, `train_reasoner_8b.py`)**:
     - Target: LoRA rank r=32, alpha=64, targeting attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP projections (`gate_proj`, `up_proj`, `down_proj`).
     - Sequence Length: Clamped to 2,048 tokens.
  2. **Experiment Tracking (Weights & Biases)**:
     - Real-time logging of train/eval loss per task, learning rate decay, gradient norms, and GPU memory telemetry.
  3. **Multi-Task Checkpoint Evaluation (`eval_checkpoint.py`)**:
     - Evaluated every 100 steps on held-out test set: JSON Validity %, Entity/Relation F1, Cypher Syntax Accuracy, Sentiment F1, Directional Accuracy.

---

### Phase 5: Hugging Face Model Registry, GGUF Quantization & YarnBall Integration
- **Objective**: Host the trained models on Hugging Face Hub, quantize to 4-bit GGUF (`Q4_K_M`), package for Ollama, and integrate the dual-model router into `YarnBall`.
- **Hugging Face Hub Private Model Registry**:
  - `hf.co/<username>/Qwen2.5-3B-YarnBall-Extractor`
  - `hf.co/<username>/Qwen3-8B-YarnBall-Reasoner`
- **Multi-Format Artifact Storage**:
  - `adapter_model.safetensors` (~75MB for 3B, ~150MB for 8B): Lightweight LoRA weights.
  - `Qwen2.5-3b-YarnBall-Extractor-Q4_K_M.gguf` (**~2.1 GB**): Fast extraction workhorse.
  - `Qwen3-8b-YarnBall-Reasoner-Q4_K_M.gguf` (**~5.2 GB**): Deep multi-hop reasoner.
- **Key Deliverables**:
  1. **Quantization Pipeline**: Export `Q4_K_M` GGUF binaries using `llama.cpp`.
  2. **Hugging Face Hub Deployment**: Automated push of adapter, merged weights, and GGUF files to private HF repository.
  3. **Ollama Integration**:
     - Pull commands:
       - `ollama run hf.co/<username>/Qwen2.5-3B-YarnBall-Extractor:Q4_K_M`
       - `ollama run hf.co/<username>/Qwen3-8B-YarnBall-Reasoner:Q4_K_M`
     - Dedicated `Modelfile` configurations with GBNF grammar constraints for the 3B extractor.
  4. **Dual-Model Router in `YarnBall` (`rag/llm_router.py`)**:
     - Dispatches ingestion workers, filing parsers, and simple 1-hop lookups to `Qwen2.5-3B-Extractor` (~2.1 GB vRAM).
     - Dispatches multi-hop contagion questions, supply chain shocks, and portfolio hedging to `Qwen3-8B-Reasoner` (~5.2 GB vRAM).
  5. **End-to-End Regression Verification in `YarnBall`**:
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
| **2. Extraction F1 (3B Workhorse)** | **> 92% Entity F1, > 88% Relation F1** | 200-sample held-out golden test set with GBNF grammar |
| **3. Text-to-CQL Accuracy (3B Workhorse)** | **> 96% Executable Cypher** | Automated Memgraph execution test suite |
| **4. Contagion Reasoning (8B Analyst)** | **> 85% Directional & Propagation Accuracy** | Golden multi-hop scenario evaluation benchmark |
| **5. Local Hardware Footprint** | **Extractor: ~2.1 GB vRAM (>100 tok/s)<br>Reasoner: ~5.2 GB vRAM (>35 tok/s)** | Measured under local Ollama on 8 GB vRAM laptop GPU |

