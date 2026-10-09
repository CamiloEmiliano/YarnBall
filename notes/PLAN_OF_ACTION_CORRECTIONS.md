# Master Audit & Sequential Corrections for PLAN_OF_ACTION_SFT.md

**Document Under Audit**: [notes/PLAN_OF_ACTION_SFT.md](notes/PLAN_OF_ACTION_SFT.md)  
**Status**: Executed & Synchronized (First Iteration Complete)  
**Objective**: Systematically eliminate internal contradictions, obsolete legacy specifications, and mathematical pipeline paradoxes from the master plan before modifying the codebase.

---

## Executive Summary of Audit Findings

The master plan in [notes/PLAN_OF_ACTION_SFT.md](notes/PLAN_OF_ACTION_SFT.md) contains five foundational discrepancies caused by the natural evolution of the project:
1. **Model Architecture Contradiction**: The document opens with the Unified Qwen2.5-7B model, but still contains legacy diagrams and sections describing a Dual-Model (3B Extractor + 8B Reasoner) router.
2. **Data Source Policy Breach**: Lines 32–35 specify Finnhub and live web scrapers for historical SFT dataset creation, directly violating the repository's deterministic data source rules.
3. **Teacher Committee Feasibility**: Tier 3 requires 37,000+ proprietary frontier LLM calls across all samples, which is cost-prohibitive and unnecessary given deterministic SEC filings.
4. **The Dataset Cartography Paradox (Section 3.5)**: Step 3 places Dataset Cartography *before* model training has ever occurred—a temporal impossibility because cartography requires epoch training dynamics.
5. **Hardware & Precision Mismatch (Phase 4)**: The document specifies 4-bit QLoRA on a consumer RTX 4090, while the production training engine is built for full `bfloat16` LoRA on an enterprise Datacenter A100 (80GB).

Below is the logical, step-by-step sequence of corrections to execute.

---

## Logical Correction Sequence

```
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Harmonize Model Architecture to Unified Qwen2.5-7B             │
│ (Eliminates obsolete 3B/8B Dual-Model router references across doc)    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Enforce Strict Data Source Compliance (Rule data_sources.md)   │
│ (Removes Finnhub & web scrapers; anchors on SEC 8-K, FNSPID, OHLCV)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Restructure Tier 3 Ground Truth Verification Protocol          │
│ (Anchors truth to SEC disclosures; reserves LLM voting for rare news)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Resolve the Dataset Cartography Paradox (Section 3.5)          │
│ (Splits into Cold-Start Curation v1.0 and Active Learning Loop v1.1)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 5: Update Phase 4 Cloud Hardware & Precision Specs                │
│ (Updates RTX 4090 QLoRA to Datacenter A100 80GB in native bfloat16)    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 6: Harmonize Verification & Canary Gatekeeping Targets            │
│ (Aligns acceptance criteria with the unified 5-task 7B model)          │
└────────────────────────────────────────────────────────────────────────┘
```

---

### Step 1: Harmonize Model Architecture to Unified Qwen2.5-7B

#### Context & Problem
In the early project design, the team considered routing simple 1-hop and 2-hop extraction to a small 3B student model and reserving multi-hop reasoning for an 8B student. However, the project successfully transitioned to a **Unified Qwen2.5-7B** model (`yarnball-qwen:7b`) that handles all 5 tasks under a single context window. 

The document currently contradicts itself:
- Lines 6–12 define the Unified 7B model.
- Lines 53, 65, and 108 still mandate a Dual-Model 3B/8B router.

#### Affected Locations in `PLAN_OF_ACTION_SFT.md`
- **Line 53**: `Dual-Model Router (Dispatches k <= 2 to 3B, k >= 3 to 8B)`
- **Line 65**: `Dual QLoRA Training: Qwen2.5-3B & Qwen3-8B on Cloud GPUs`
- **Line 97**: Title refers to `Qwen_YarnBall_SFT` instead of `QwenSFT_YarnBall`
- **Line 108**: Mentions `partitioned for dual-model specialization`

#### Proposed Correction
1. In the Phase 1 diagram (Line 53), replace `Dual-Model Router (Dispatches k <= 2 to 3B, k >= 3 to 8B)` with:
   `Unified Model Engine: Serves all 5 extraction and reasoning tasks via yarnball-qwen:7b`
2. In the Repository 2 diagram box (Line 65), replace `Dual QLoRA Training: Qwen2.5-3B & Qwen3-8B on Cloud GPUs` with:
   `Unified LoRA SFT: Qwen2.5-7B-Instruct on Datacenter A100 GPUs (bfloat16)`
3. In Phase 2 heading (Line 97), standardize the repository name to `QwenSFT_YarnBall`.
4. In Phase 3 objective (Line 108), replace `partitioned for dual-model specialization` with:
   `curated for unified multi-task specialization across Tasks A through E`.

---

### Step 2: Enforce Strict Data Source Compliance (`.agents/rules/data_sources.md`)

#### Context & Problem
The repository has a strict user rule in [.agents/rules/data_sources.md](.agents/rules/data_sources.md) prohibiting Finnhub and live web scrapers from historical SFT dataset creation or dry runs. 

Lines 32–35 in `PLAN_OF_ACTION_SFT.md` currently contradict this rule by listing:
- `yfinance Ticker News, corporate wire feeds, and direct publisher disclosures`
- `Auxiliary Fallback: Finnhub API (ticker metadata & quote lookup)`

#### Affected Locations in `PLAN_OF_ACTION_SFT.md`
- **Lines 31–36**: News stream specifications in "The 4-Source Multi-Layer Data Matrix"
- **Line 86**: `(with Finnhub relegated to an auxiliary metadata fallback)`

#### Proposed Correction
1. Rewrite Lines 31–36 to explicitly align with `data_sources.md`:
   - Primary historical events: SEC Form 8-K material disclosures (Items 1.01, 1.02, 2.01, 2.04) and local FNSPID parquet archives.
   - Market pricing & volatility context: Historical daily OHLCV bars and CAR z-scores via `yfinance` (2018–2025 depth).
   - Finnhub policy: Strictly relegated to production real-time forward streaming (zero historical SFT or dry-run usage).
   - Web scraping policy: Ephemeral web scrapers and `yfinance.Ticker.news` are excluded due to survivorship bias, paywalls, and collinearity with FNSPID.
2. In Line 86, clarify that Finnhub is exclusively reserved for live forward streaming in production.

---

### Step 3: Restructure Tier 3 Ground Truth Verification Protocol

#### Context & Problem
Section 3.1 Tier 3 specifies a "Frontier Committee Voting" protocol requiring consensus across Google Gemini 1.5/2.0 Pro, Anthropic Claude 3.5 Sonnet, and OpenAI GPT-4o for every sample.
- For 12,470 samples, this requires **37,410 API calls**, costing hundreds of dollars in proprietary tokens and introducing vendor billing dependencies.
- Furthermore, for SEC Form 10-K Exhibit 21 subsidiaries, Form 4 insider filings, and Form 8-K disclosures, deterministic legal text parsers already provide **100% factual accuracy** without calling an LLM.

#### Affected Locations in `PLAN_OF_ACTION_SFT.md`
- **Lines 118–121**: Tier 3 description under "The 5-Tier Ground Truth Verification Protocol"

#### Proposed Correction
Restructure Tier 3 to distinguish deterministic regulatory filings from unstructured news text:
1. **Deterministic Regulatory Filings (Tiers 1 & 2)**: All corporate relationships from Form 10-K Exhibit 21, Item 1/1A regex anchors, and Form 8-K filings bypass frontier LLMs completely. They are verified via programmatic schema gates and CIK resolution.
2. **Unstructured Market Commentary & News (Tier 3)**: Multi-teacher frontier consensus is deployed **selectively** on ambiguous, syntactically complex market headlines, using a single primary validator with secondary tie-breaking rather than brute-force 3-model committee calls on the entire corpus.

---

### Step 4: Resolve the "Dataset Cartography" Temporal Paradox (Section 3.5)

#### Context & Problem
Section 3.5 currently lists the curation pipeline as:
1. MinHash LSH Deduplication
2. Dense Semantic Embedding Clustering (SemDeDup)
3. Dataset Cartography & Training Dynamics
4. Submodular Core-Set Selection
5. Deterministic Split (80/10/10)

**The Mathematical Paradox**:
Dataset Cartography (*Swayamdipta et al., 2020*) measures the mean confidence and variability of a model across training epochs. **You cannot calculate training dynamics on a dataset before the model has ever trained.** Putting it in Step 3 of the pre-training cold-start pipeline is a temporal impossibility.

#### Affected Locations in `PLAN_OF_ACTION_SFT.md`
- **Lines 181–188**: Section 3.5 title and numbered steps

#### Proposed Correction
Split Section 3.5 into two explicitly separated, chronological phases:

#### 1. Cold-Start Generation Pipeline (Dataset v1.0.0 - Pre-Training):
```
Raw Manifold Candidates (SEC + FNSPID + 8-K)
       │
       ▼
[Stage 1: MinHash LSH Deduplication] ── 5-gram tokenization, 128 permutation hashes (Jaccard >= 0.85)
       │
       ▼
[Stage 2: Dense Semantic Clustering (SemDeDup)] ── 768-dim all-mpnet-base-v2 (cosine >= 0.88)
       │
       ▼
[Stage 3: Submodular Core-Set Selection] ── Facility Location coverage across all 11 GICS sectors
       │
       ▼
[Stage 4: Deterministic Partitioning] ── 80% Train, 10% Validation, 10% Test
```

#### 2. Active Learning Iteration Pipeline (Dataset v(N) -> v(N+1) - Post-Training):
```
Model Training Run v(N) in QwenSFT_YarnBall
       │
       ▼
[TrainingDynamicsCallback] ── Emits epoch confidence & variability to training_dynamics.jsonl
       │
       ▼
[DatasetCartographer Analysis in graphrag_finance]
  • Easy-to-Learn (High confidence, low variability): Prune 80% of repetitive boilerplates
  • Ambiguous / Boundary (High variability): Retain 100% (the core learning engine)
  • Hard-to-Learn (Low confidence, low variability): Quarantine for label audit
       │
       ▼
Export Refined Dataset v(N+1) -> Retrain on Vast.ai
```

---

### Step 5: Update Phase 4 Cloud Hardware & Precision Specs

#### Context & Problem
Phase 4 (Lines 206–224) was written when the plan assumed 4-bit QLoRA on a single consumer RTX 4090 (24GB). 

However:
- In financial extraction, 4-bit NormalFloat quantization during training introduces rounding noise that can corrupt strict OpenCypher syntax (unclosed brackets, escaped quotes).
- In `QwenSFT_YarnBall/config.yaml`, the training engine is configured for **native `bfloat16` precision with LoRA rank 32**.
- On Vast.ai, a verified datacenter **NVIDIA A100 (80GB SXM)** costs only **~$1.07/hr**, and an entire 3-epoch run on 12,470 records completes in ~35 minutes for **under $0.80 total cost**.

#### Affected Locations in `PLAN_OF_ACTION_SFT.md`
- **Lines 209–214**: Hardware specs, vRAM requirements, and RTX 4090 recommendations

#### Proposed Correction
1. Replace 4-bit QLoRA references with **Native `bfloat16` LoRA Adaptation** (no base quantization artifacts).
2. Update the recommended hardware tier to:
   - **Primary Target**: **1x NVIDIA A100 (80GB SXM)** or **A100 PCIe (80GB)** in verified commercial datacenters (`datacenter=true verified=true`).
   - Sequence length: 2,048 tokens.
   - Batch size: 4 per device with 4 gradient accumulation steps (effective global batch size: 16).
   - Training time: ~35–45 minutes; total compute cost: ~$0.75–$1.20 USD.

---

### Step 6: Harmonize Verification & Canary Gatekeeping Targets

#### Context & Problem
The acceptance gates in Section 6 (Lines 258–266) include legacy targets that need alignment with the unified 5-task model and our Stage 3 diagnostic evaluation probes.

#### Affected Locations in `PLAN_OF_ACTION_SFT.md`
- **Lines 258–266**: "Verification & Acceptance Gates" table

#### Proposed Correction
Harmonize the table to reflect the 5 formal Stage 3 diagnostic probes:

| Gate | Minimum Threshold | Evaluation Metric & Probe Scope |
| :--- | :---: | :--- |
| **1. Cypher Syntax Execution** | **>= 99.5%** | Formal Memgraph AST parser validation across 500 holdout samples (Task A, C). |
| **2. Hard Negative Refusal** | **>= 96.0%** | Rejection accuracy strictly returning `(none)` on co-occurring non-causal entities. |
| **3. Relational Macro F1** | **>= 0.90** | Macro F1 across all 11 GICS economic sectors (Tasks A, B). |
| **4. Entity & CIK Grounding** | **>= 95.0%** | Source/target alignment against official S&P 500 master constituent registry. |
| **5. CoT Reasoning Validity** | **>= 92.0%** | Directional and propagation validity on multi-hop supply contagion (Tasks D, E). |
| **6. Local Inference Footprint** | **~4.7 GB vRAM** | Fits 100% in local 8GB laptop GPU vRAM under 4-bit GGUF (`Q4_K_M`) at ~60–75 tok/s. |

## Iteration 1 Completion Status

All 6 sequential corrections have been applied directly to [notes/PLAN_OF_ACTION_SFT.md](notes/PLAN_OF_ACTION_SFT.md):
- [x] **Step 1**: Harmonized model architecture to Unified Qwen2.5-7B (`yarnball-qwen:7b`).
- [x] **Step 2**: Enforced strict data source compliance (SEC 8-K, FNSPID, OHLCV; Finnhub strictly production streaming).
- [x] **Step 3**: Grounded Tier 3 verification in deterministic SEC filings; selective frontier teacher voting on ambiguous news.
- [x] **Step 4**: Resolved Dataset Cartography paradox by decoupling Cold-Start Curation (v1.0.0) from the Active Learning Loop (vN -> vN+1).
- [x] **Step 5**: Updated hardware specs to Datacenter NVIDIA A100 (80GB) in native `bfloat16` LoRA.
- [x] **Step 6**: Harmonized acceptance gates with the 6 Stage 3 diagnostic probes.

The master design document is now mathematically consistent and aligned with repository policies. Next, we can consider code self-documenting qualities and structuring the curation pipeline modules in executable code.
