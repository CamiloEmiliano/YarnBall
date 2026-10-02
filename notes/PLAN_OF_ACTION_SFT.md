# Master Plan of Action: Multi-Source Financial Graph & Unified Qwen2.5-7B SFT Distillation

## Executive Overview
This document outlines the end-to-end plan for building **YarnBall's Unified Financial Intelligence Model**—a task-specialized, quantized 7B financial model (`Qwen2.5-7B-Instruct`) trained via supervised fine-tuning (SFT) and teacher distillation across a multi-source financial corpus spanning 2018 to 2025+:

- **`yarnball-qwen:7b` (`Qwen2.5-7B-Instruct` 4-bit Q4_K_M GGUF)**: Unified high-throughput extraction and high-cognitive reasoning engine (**~4.7 GB vRAM**, **~60-75 tok/s**, fitting 100% inside 8 GB laptop GPU vRAM) spanning all 5 core tasks:
  - **Task A (`<|extract_sec_graph|>`)**: SEC Form 10-K/10-Q/8-K parsing into validated OpenCypher DSL.
  - **Task B (`<|extract_news_event|>`)**: Breaking news and press releases into 5-axis directional graph edges.
  - **Task C (`<|text_to_cypher|>`)**: Natural language financial queries into executable Memgraph CQL.
  - **Task D (`<|contagion_reasoning|>`)**: Multi-hop supply chain contagion and shock propagation with `<think>` Chain-of-Thought (CoT).
  - **Task E (`<|portfolio_recommendation|>`)**: Risk synthesis and institutional portfolio hedging strategies with `<think>` CoT.

The plan decouples the **GraphRAG production runtime** (`graphrag_finance` / `YarnBall`) from the **dedicated ML training lab** (`Qwen_YarnBall_SFT`), establishes multi-year S&P 500 point-in-time snapshot archives tracked by DVC, incorporates 4 complementary data layers (SEC Filings, Historical News, Earnings Transcripts, Market/Macro series), and enforces a rigorous 5-tier Ground Truth verification protocol.

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
│ 2. Uncapped Historical & Real-Time Financial News (2018–2025)          │
│    - FNSPID Hugging Face Dataset (>29M S&P 500 articles 2010–2024)     │
│    - SEC Form 8-K Material Event Disclosures (Item 1.01/2.01 press)    │
│    - Real-Time Harvesters: `yfinance` Ticker News, corporate wire feeds, │
│      and direct publisher disclosures                                  │
│    - GDELT 2.0 Global Corporate & Supply Chain Event Stream            │
│    - Auxiliary Fallback: Finnhub API (ticker metadata & quote lookup)  │
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
  1. **S&P 500 Historical Constituent Registry (`tools/sp500_universe.py`)**: Map point-in-time S&P 500 membership (2018–2025) to account for index additions/deletions and avoid survivorship bias.
  2. **SEC EDGAR Downloader (`tools/download_historical_sec.py`)**:
     - Batch fetch 10-K, 10-Q, 8-K, Form 4 across 2018–2025 for S&P 500 constituents (~500 10-Ks and ~1,500 10-Qs per year).
     - Extract plain-text sections (`item_1_business.txt`, `item_1a_risk_factors.txt`, `subsidiaries.json`, `form8k_events.json`).
  3. **Generous Multi-Source News Ingestion (`tools/ingest_historical_news.py` & `ingest/news_harvester.py`)**:
     - **Bulk Historical (2018–2025)**: Stream and filter the open **FNSPID** Hugging Face dataset (>29M financial articles mapped to S&P 500 tickers) and SEC Form 8-K material event releases directly into PostgreSQL `financial_news_queue` with zero API rate limits.
     - **Live & Ongoing Ingestion**: Direct ticker news and corporate wire streams via `yfinance` and publisher feeds (with Finnhub relegated to an auxiliary metadata fallback).
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

#### 3.3 Cognitive Load Task Classification & Unified Model Architecture
All 5 financial tasks are unified under a single **Qwen2.5-7B** backbone using distinct control tokens:

```
┌────────────────────────────────────────────────────────────────────────┐
│ UNIFIED FINANCIAL INTELLIGENCE ENGINE: Qwen2.5-7B-Instruct (Q4_K_M)    │
│ ────────────────────────────────────────────────────────────────────── │
│ Architecture: 7.6B Parameters, 28 Layers, 128k Context Window          │
│ Quantization: 4-bit GGUF (Q4_K_M, ~4.7 GB Binary)                      │
│ Hardware Fit: 100% in Laptop 8 GB vRAM (~60-75 tok/s, Zero Offload)    │
│ Format: Task Control Tokens + Prompt Loss Masking                      │
│                                                                        │
│ - Task A (`<|extract_sec_graph|>`): SEC 10-K/10-Q/8-K -> Triples DSL   │
│ - Task B (`<|extract_news_event|>`): Breaking News -> Directional Edge │
│ - Task C (`<|text_to_cypher|>`): Query + Schema -> Valid Memgraph CQL  │
│ - Task D (`<|contagion_reasoning|>`): Subgraph -> Multi-hop <think>    │
│ - Task E (`<|portfolio_recommendation|>`): Risk -> Hedging <think>     │
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
    "assigned_student": "QWEN_2.5_7B_UNIFIED",
    "teacher_origin": "GEMINI_FLASH_CIK_GATE",
    "curriculum_weight": 1.0
  },
  "prompt": "<|extract_sec_graph|>The Company relies on TSMC for 3nm wafer fabrication...",
  "target_completion": "(TSMC:Company {ticker:\"TSM\"})-[:SUPPLIES_TO {nature:\"3nm wafer fabrication\", polarity:\"NEUTRAL_STABLE\", materiality:\"CRITICAL_TIER_1\"}]->(Apple Inc.:Company {ticker:\"AAPL\"})"
}
```

#### 3.5 Statistical Balancing, Deduplication & Data Cartography Pipeline
To eliminate redundant boilerplates and optimize sample efficiency without losing tail risks:
1. **MinHash LSH Deduplication** (*Broder, 1997; Lee et al., ACL 2022*): 5-gram tokenization with 128 permutation hashes (Jaccard >= 0.85) to eliminate syndicated news wire copies and identical 10-K legal text.
2. **Dense Semantic Embedding Clustering** (*Song et al., NeurIPS 2020; Abbas et al., 2023 - SemDeDup*): 768-dim `all-mpnet-base-v2` clustering (cosine threshold >= 0.88) collapsing boilerplate disclosures into centroid exemplars.
3. **Dataset Cartography & Training Dynamics** (*Swayamdipta et al., EMNLP 2020; Toneva et al., ICLR 2019*): Prunes ~80% of "Easy-to-Learn" samples, retains 100% of "Ambiguous / Boundary" samples, and quarantines "Hard-to-Learn" noisy data.
4. **Submodular Core-Set Selection** (*Mirzasoleiman et al., ICML 2020; Sener & Savarese, ICLR 2018*): Facility location optimization guaranteeing uniform geometric coverage across all 11 GICS economic sectors and 5-axis types.
5. **Deterministic Split**: Train / Validation / Test partitioning (`yarnball_sft_train.jsonl` 80%, `yarnball_sft_val.jsonl` 10%, `yarnball_sft_test.jsonl` 10%).

#### 3.6 Rare Relationship Over-Sampling & Lexical Trigger Harvesting
To prevent class collapse where the model only predicts common relationships (`SUBSIDIARY_OF`, `SUPPLIES_TO`) while missing high-value risk edges:
1. **Targeted Lexical Harvesting (Regex Trigger Mining)**:
   - `SOLE_SOURCE_DEPENDENT_ON`: Harvest paragraphs with `"sole source"`, `"single source supplier"`, `"no alternate supplier"`, `"solely dependent on"`.
   - `LICENSES_FROM / LICENSES_TO`: Harvest `"cross-licensing agreement"`, `"patent license"`, `"exclusive royalty"`, `"technology licensing"`.
   - `EXPOSED_TO_CHOKEPOINT`: Harvest `"export restriction"`, `"critical bottleneck"`, `"ITAR compliance"`, `"single point of failure"`.
   - `DEFAULTED_ON / TERMINATED`: Harvest `"terminated for cause"`, `"notice of default"`, `"covenant breach"`, `"contract cancellation"`.
2. **Inverse-Frequency Floor Capping**:
   - Cap common relations (`SUBSIDIARY_OF`, generic `SUPPLIES_TO`) at a maximum of ~500 samples.
   - Enforce a hard minimum floor of at least **200 verified training samples** for every rare relationship class.
3. **Targeted Form 8-K Event Item Mining**:
   - Specifically pull rare corporate disruption filings: Item 1.02 (Termination of Material Agreement), Item 1.03 (Bankruptcy), Item 2.04 (Acceleration of Direct Financial Obligation / Default).
4. **Counterfactual Entity-Swap Augmentation**:
   - For ultra-rare legal/settlement patterns, generate 2–3 synthetic variants by swapping entity names while preserving identical Cypher triple relationship grammar.

---

### Phase 4: Cloud LoRA Training, Hardware Specs & Experiment Tracking
- **Objective**: Fine-tune `Qwen2.5-7B-Instruct` on cloud GPU compute (Vast.ai / RunPod).
- **GPU Hardware & Compute Estimates**:
  - **Workload**: Base model `Qwen/Qwen2.5-7B-Instruct`, ~3,000–5,000 unified multi-task samples, 3 epochs.
  - **Training Compute Requirements (4-bit QLoRA via Unsloth / Hugging Face PEFT)**:
    - 7B Model Peak vRAM during training: **~8.5 to 11.5 GB vRAM** (Training time: **~45 to 60 min** on RTX 4090).
  - **Recommended Hardware Tier**:
    - **Primary Target**: **1x Nvidia RTX 4090 (24 GB vRAM)** — Total Training Time: **~1.0 hour**, Total Cost: **~$0.40 to $0.75 USD** (via Vast.ai).
    - **System Specs**: 8 vCPUs, 32 GB System RAM, 50 GB NVMe storage.
- **Key Deliverables**:
  1. **Unified LoRA Training Script (`train_lora_vastai.py`)**:
     - Target: LoRA rank r=32, alpha=64, targeting attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP projections (`gate_proj`, `up_proj`, `down_proj`).
     - Prompt loss masking with label `-100` on input prompt tokens.
     - Sequence Length: Clamped to 2,048 tokens.
  2. **Experiment Tracking (Weights & Biases)**:
     - Real-time logging of train/eval loss across tasks, learning rate decay, gradient norms, and GPU memory telemetry.
  3. **Multi-Task Checkpoint Evaluation (`eval_checkpoint.py`)**:
     - Evaluated every 100 steps on held-out test set: JSON Validity %, Entity/Relation F1, Cypher Syntax Accuracy, Sentiment F1, Directional Accuracy.

---

### Phase 5: Hugging Face Model Registry, GGUF Quantization & YarnBall Integration
- **Objective**: Host the trained adapter on Hugging Face Hub, quantize merged weights to 4-bit GGUF (`Q4_K_M`), package for Ollama, and serve in `graphrag_finance`.
- **Hugging Face Hub Private Model Registry**:
  - `hf.co/<username>/yarnball-qwen-7b`
- **Multi-Format Artifact Storage**:
  - `adapter_model.safetensors` (~140MB): Lightweight LoRA weights.
  - `yarnball-qwen-7b-Q4_K_M.gguf` (**~4.7 GB**): Fast, unified extraction & reasoning engine.
- **Key Deliverables**:
  1. **Quantization Pipeline**: Export `Q4_K_M` GGUF binaries using `llama.cpp`.
  2. **Hugging Face Hub Deployment**: Automated push of adapter, merged weights, and GGUF files to private HF repository.
  3. **Ollama Integration**:
     - Register local model: `ollama create yarnball-qwen:7b -f Modelfile`
     - Test direct inference: `ollama run yarnball-qwen:7b "<|extract_sec_graph|>..."`
  4. **GraphRAG Platform Integration (`rag/` & `kafka_pipeline/`)**:
     - Ingestion workers route incoming SEC filings and news streams through `yarnball-qwen:7b`.
     - Hybrid retriever and text-to-cypher queries invoke `yarnball-qwen:7b` for Cypher generation and multi-hop graph reasoning.
  5. **End-to-End Regression Verification in `YarnBall`**:
     - Run full 134-test regression suite and 25-query golden benchmark harness (`rag/eval_harness.py`).

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
| **2. Extraction F1 (Task A & B)** | **> 92% Entity F1, > 88% Relation F1** | 200-sample held-out golden test set with GBNF grammar |
| **3. Text-to-CQL Accuracy (Task C)** | **> 96% Executable Cypher** | Automated Memgraph execution test suite |
| **4. Contagion Reasoning (Task D & E)** | **> 85% Directional & Propagation Accuracy** | Golden multi-hop scenario evaluation benchmark |
| **5. Local Hardware Footprint** | **~4.7 GB vRAM (~60-75 tok/s, 100% GPU)** | Measured under local Ollama on 8 GB vRAM laptop GPU |

