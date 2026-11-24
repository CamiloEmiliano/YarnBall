# ADR-00012: Upgraded Qwen 9B Linear-Attention & Task-Selective SFT + RL Alignment Plan

- **ADR ID**: ADR-00012
- **Filename**: `ADR-00012-qwen-model-upgrade-and-fine-tuning-plan-.md`
- **Date**: 2026-11-23 23:08 UTC
- **Status**: PROPOSED
- **Target Subsystem**: `sft_pipeline` / `retrieval` / `ontology`
- **Linked Commits**: [`a26bc0c`](../../sft_pipeline/serving/model_client.py)

---

## 1. Context & Problem Statement

YarnBall's neural reasoning and Text-to-Cypher interface is currently parameterized around a 7B dense transformer (`yarnball-qwen:7b`) with traditional softmax attention trained via uniform Supervised Fine-Tuning (SFT) plus offline Direct Preference Optimization (DPO) across five tasks (`Tasks A–E`).

Three architectural and post-training insights motivate updating both the **backbone architecture** and a **task-selective SFT + RL alignment ladder**:

1. **8 GB VRAM KV-Cache Ceiling on Softmax 7B**:
   In financial GraphRAG pipelines, chunked extraction with gleaning ([`ADR-00035`](ADR-00035-text-unit-chunk-layer-and-evidence-links.md)), multi-hop subgraphs, and community map-reduce ([`ADR-00037`](ADR-00037-pit-community-layer-and-impact-ranking--.md)) produce long prompt contexts ($8\text{k}\text{–}16\text{k}+$ tokens). Under standard softmax attention, KV-cache VRAM scales linearly $O(L)$ with sequence length:
   $$\text{KV Cache VRAM} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times L \times \text{bytes\_per\_element}$$
   Upgrading to a ~9B hybrid linear-attention architecture (e.g., Qwen 3.5 9B with Gated DeltaNet / recurrent linear layers + sparse full-attention anchors) condenses history into a fixed-size $O(1)$ recurrent state $S_t \in \mathbb{R}^{d \times d}$ while pinning `Qwen3-Embedding-0.6B` to host CPU RAM (`num_gpu: 0` per [`ADR-00035`](ADR-00035-text-unit-chunk-layer-and-evidence-links.md)) so 100% of VRAM stays dedicated to the 9B generator.

2. **Domain 1 Curriculum Realignment**:
   - **Retire Task E (`PORTFOLIO_RECOMMENDATION`)**: Per [`ADR-00010`](ADR-00010-directional-basket-peer-cluster-mapping-.md) (`REJECTED`) and [`ADR-00034`](ADR-00034-single-bi-read-facade-and-pipeline-merge.md), Domain 2 portfolio/options hedging is deleted from YarnBall, freeing model capacity.
   - **Add Grounded Synthesis & Community Reporting (`Task F`)**: [`ADR-00034`](ADR-00034-single-bi-read-facade-and-pipeline-merge.md), [`ADR-00035`](ADR-00035-text-unit-chunk-layer-and-evidence-links.md), and [`ADR-00037`](ADR-00037-pit-community-layer-and-impact-ranking--.md) introduce mandatory `[E# ← chunk_id]` passage citations and hierarchical Leiden `COMMUNITY_REPORT` summarization (while keeping `local` vs. `global` query routing deterministic rules-first in Python).

3. **Task-Asymmetric Strengths of SFT vs. RLVR vs. RLAIF**:
   Neither pure SFT nor blanket RL is universally superior across all tasks:
   - **SFT excels at injecting closed-schema conventions** (`ONTOLOGY_SCHEMA`, `<|task|>` headers, 5-axis JSON extraction, and structured synthesis/community templates) 5–10× faster and more stably than RL exploration.
   - **RLVR (Reinforcement Learning with Verifiable Rewards) excels at sequence-level constraint satisfaction** where a single wrong token causes a 100% runtime failure (`CypherASTWhitelistGuardrail`, verbatim cascade numbers, mandatory citations).
   - **RLAIF (AI-Judge RL) carries Goodhart / reward-hacking risk** (verbosity and style bias) and must be treated strictly as an **evidence-gated optional stage** rather than an assumed improvement over SFT + RLVR.

Furthermore, YarnBall preserves the **LLM Swapability Invariant**:
> The Large Language Model in YarnBall is strictly an exchangeable verbalizer and DSL translator behind [`sft_pipeline/serving/model_client.py`](../../sft_pipeline/serving/model_client.py) (`YarnBallModelClient`). Deterministic cascade physics, regulatory schemas ([`ADR-00011`](ADR-00011-operational-contagion-report-schema-----.md)), and AST query guardrails remain completely independent of the neural backbone.

---

## 2. Decision Drivers & Constraints

1. **8 GB Inference VRAM Envelope (with CPU-Pinned Embedder)**:
   - 9B model weights at 4-bit (`Q4_K_M`): $\approx 5.1\text{–}5.3\text{ GB}$; hybrid linear state + anchor KV cache: $\approx 0.2\text{–}0.5\text{ GB}$; CPU-pinned `qwen3-embedding:0.6b` (`num_gpu: 0`): $0.0\text{ GB}$ VRAM $\implies \approx 5.8\text{–}6.2\text{ GB}$ peak VRAM ($\ge 1.8\text{ GB}$ safety buffer).
2. **Task-Selective Training in a Single Merged Checkpoint (Zero SFT Forgetting)**:
   - Schema-led tasks (Tasks A, B, F) and RL-sharpened tasks (Tasks C, D) must inhabit **one** merged `yarnball-qwen:9b` GGUF so Ollama never swaps models in VRAM, using an SFT replay anchor (or validated `TIES`/`DARE` LoRA merge) so targeted RL never degrades SFT-led tasks.
3. **Evidence-Gated Promotion**:
   - Every post-SFT stage (RLVR, RLAIF) must empirically outperform the preceding checkpoint on held-out verification (`sft_pipeline` torture suite + citation/guardrail audit) before promotion.
4. **Prefix-Forced Reasoning Preservation & Decoupled Engineering Pace**:
   - Maintain `<think>\n` prefix forcing ([`ADR-00004`](ADR-00004-sit-flow-5-model-foundry-alignment-smoke.md)) for reasoning/synthesis tasks and zero-think direct output for structured DSL tasks; `ADR-00033`–`ADR-00037` proceed without blocking on rental-GPU training runs.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Retain 7B Softmax Model**:
  - *Rationale for Rejection*: Keeps $O(L)$ KV-cache OOM risk on long GraphRAG contexts.
- **Alternative B: 14B Model Quantized to 2-Bit/3-Bit**:
  - *Rationale for Rejection*: Sub-4-bit perplexity degradation breaks Cypher syntax fidelity (Task C) and verbatim metric quoting ([`ADR-00011`](ADR-00011-operational-contagion-report-schema-----.md)).
- **Alternative C: External Cloud LLM APIs for Runtime Serving**:
  - *Rationale for Rejection*: Violates local air-gapped serving and deterministic latency invariants.
- **Alternative D: Replace SFT Completely with Zero-Shot RLAIF Across All Tasks**:
  - *Rationale for Rejection*: Severe cold-start reward sparsity on YarnBall's custom `<|task|>` headers and closed 26-predicate `ONTOLOGY_SCHEMA`, plus AI-judge Goodhart risk.
- **Alternative E: Multi-Task PPO with a 9B Value/Critic Network**:
  - *Rationale for Rejection*: Critic network inflates rental-GPU VRAM and training instability; critic-free **GRPO** eliminates the value network.
- **Alternative F: Un-anchored Targeted RL on Tasks C/D Only**:
  - *Rationale for Rejection*: Because RL KL penalties are evaluated only on prompts inside the RL batch, running GRPO solely on Tasks C/D without an SFT replay anchor allows shared MLP/attention weights to drift and degrade SFT-led Tasks A, B, and F.

---

## 4. Decision Outcome: Evidence-Gated Task-Selective SFT + Anchored RL Roadmap

```
Stage 1: 9B Hybrid Linear-Attention Base & ChatML Verification
   │
   ▼
Stage 2 (Gate 1 Baseline): Shared Multi-Task SFT (+ Verifier-Filtered DPO) Across Tasks A, B, C, D, F
   │
   ▼
Stage 3a (Gate 2 — Targeted RLVR): Critic-Free GRPO on Tasks C & D + SFT Replay Anchor on Tasks A, B, F
   │
   ▼
Stage 3b (Gate 3 — Optional Gated RLAIF): Multiplicatively Gated AI-Judge Polish on Task D (if Gate 3 > Gate 2)
   │
   ▼
Stage 4 & 5: BF16 Fusion, Q4_K_M GGUF (`yarnball-qwen:9b`) & Zero-Forgetting / Citation Regression Audit
```

### Stage 1: Base Model & Tokenizer Verification
- Verify target ~9B hybrid linear-attention base checkpoint (e.g., Qwen 3.5 9B), `flash-linear-attention` (`fla`) training kernels, GGUF support in Ollama, and ChatML `<|task|>` tokenization.

### Stage 2 (Gate 1 Baseline): Shared Multi-Task SFT (+ Verifier-Filtered Offline DPO)
- Train across all active Domain 1 tasks (retiring Task E `PORTFOLIO_RECOMMENDATION`):
  - **SFT-Led Tasks (Heavy Schema & Format Mastery)**:
    - **Task A (`EXTRACT_SEC_GRAPH`)**: 800-token chunk extraction + Pass-2 gleaning ([`ADR-00035`](ADR-00035-text-unit-chunk-layer-and-evidence-links.md)).
    - **Task B (`EXTRACT_NEWS_EVENT`)**: Directional 5-axis event extraction & `(none)` hard negatives.
    - **Task F (`GROUNDED_SYNTHESIS` / `COMMUNITY_REPORT`)**: Citation-enforced synthesis ([`ADR-00034`](ADR-00034-single-bi-read-facade-and-pipeline-merge.md)) and Leiden cluster summaries ([`ADR-00037`](ADR-00037-pit-community-layer-and-impact-ranking--.md)).
  - **RL-Candidate Seed Tasks**:
    - **Task C (`TEXT_TO_CYPHER`)** and **Task D (`CONTAGION_REASONING`)**.
- **LoRA Configuration**: Rank $r=16$, $\alpha=32$, dropout $0.05$, targeting attention projections, **linear recurrent state/gate projections**, and MLP layers; prompt tokens masked with `-100`.
- Apply 1 pass of verifier-filtered offline DPO via [`sft_pipeline/dpo_builder.py`](../../sft_pipeline/dpo_builder.py) to establish the **Gate 1 benchmark score**.

### Stage 3a (Gate 2): Targeted RLVR on Tasks C & D with SFT Replay Anchor
- Run critic-free **GRPO** ($G = 4\text{–}8$ rollouts/prompt, token-length-normalized Dr. GRPO loss) **selectively on Tasks C & D** using pure zero-cost **deterministic code rewards ($R_{\text{RLVR}}$)**:
  - **Task C (`TEXT_TO_CYPHER`)**: [`retrieval/guardrails/schema_guardrail.py`](../../retrieval/guardrails/schema_guardrail.py) `CypherASTWhitelistGuardrail` pass + fixture subgraph execution equivalence.
  - **Task D (`CONTAGION_REASONING`)**: Verbatim cascade metric match ([`ADR-00011`](ADR-00011-operational-contagion-report-schema-----.md)) + valid `[E# ← chunk_id]` citations ([`ADR-00035`](ADR-00035-text-unit-chunk-layer-and-evidence-links.md)) + `<think>` boundary compliance + zero Domain 2 trading bleed.
- **Anti-Forgetting Pin on SFT-Led Tasks**: Optimize the joint objective mixing ~10–15% SFT cross-entropy replay batches on Tasks A, B, and F (plus optional cheap [`ontology/validation.py`](../../ontology/validation.py) `validate_graph_payload` checks):
  $$\mathcal{L}_{\text{step}} = \mathcal{L}_{\text{GRPO}}(\text{Tasks C, D}) + \lambda_{\text{anchor}} \, \mathcal{L}_{\text{SFT}}(\text{Tasks A, B, F})$$
  *(Alternative fallback: train Stage-2 SFT LoRA and Stage-3a RL LoRA separately and fuse via validation-tuned `TIES`/`DARE` merging.)*
- **Promotion Gate 2**: Keep Stage 3a only if Task C/D guardrail & citation pass rates beat Gate 1 with $\le 0.5\%$ F1 regression on SFT-led Tasks A, B, and F.

### Stage 3b (Gate 3 — Optional): Multiplicatively Gated RLAIF on Reasoning
- Executed **only if** human/torture-suite review after Gate 2 identifies residual prose reasoning weaknesses on Task D (or Task F):
  $$R_{\text{total}}(x, y) = G_{\text{hard}}(x, y) \times \left(w_{\text{ver}} R_{\text{RLVR}}(x, y) + w_{\text{judge}} R_{\text{RLAIF}}(x, y)\right)$$
  where $G_{\text{hard}} \in \{0, 1\}$ zeroes out reward on any deterministic schema, citation, numeric, or Domain-1-isolation violation so the model cannot reward-hack the AI judge.
- **Promotion Gate 3**: Promoted only if held-out reasoning quality beats Gate 2 with **zero** regression on deterministic code-gate pass rates; otherwise Gate 2 (or Gate 1) ships directly to Stage 4.

### Stages 4 & 5: Quantization, Ollama Registration & Full Audit
- Fuse winning adapter in BF16, quantize to `Q4_K_M` GGUF, register as `yarnball-qwen:9b` (`YARNBALL_MODEL_TAG=yarnball-qwen:9b`), and run full regression + [`tools/dry_run_sec_8k_contagion.py`](../../tools/dry_run_sec_8k_contagion.py) citation verification.

---

## 5. Invariants & Guardrails

1. **LLM Swapability Invariant**: All callers interact exclusively through `YarnBallModelClient`; switching `yarnball-qwen:7b` ↔ `yarnball-qwen:9b` requires zero code changes in `retrieval/`, `ingestion/`, or `knowledge_graph/`.
2. **Strict 8 GB Hardware Envelope**: Peak inference VRAM at 16k context must remain $\le 7.5\text{ GB}$, with embeddings pinned to host CPU RAM during live serving ([`ADR-00035`](ADR-00035-text-unit-chunk-layer-and-evidence-links.md)).
3. **Zero-Forgetting & Evidence-Gated Promotion Invariant**: No RLVR or RLAIF stage may be promoted over the Stage-2 SFT baseline if it regresses held-out performance on SFT-led extraction/synthesis tasks (Tasks A, B, F) or deterministic guardrail compliance.
4. **Multiplicative Deterministic Gate Invariant**: Whenever AI-judge RLAIF is enabled, $G_{\text{hard}} = 0 \implies R_{\text{total}} = 0$.
5. **Domain 1 Curriculum Purity**: No training task or reward rubric may reintroduce Domain 2 trading, option strikes, or portfolio-allocation outputs ([`ADR-00010`](ADR-00010-directional-basket-peer-cluster-mapping-.md), [`ADR-00034`](ADR-00034-single-bi-read-facade-and-pipeline-merge.md)).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Uses each post-training method strictly where it has proven mathematical leverage—SFT for closed-schema mastery, RLVR for hard code-verified constraints, and gated RLAIF only if empirically earned—inside a single resident 9B model.
  - SFT replay anchoring prevents targeted RL on Cypher/reasoning from degrading extraction accuracy.
  - $O(1)$ linear-attention state eliminates long-context KV-cache OOMs on 8 GB GPUs.
- **Negative / Trade-offs**:
  - Gate 2/3 GRPO rollouts require rental-GPU compute (~$1.5\text{–}2.5\times$ SFT alone on an RTX 4090 / A100), bounded cleanly because Gate 1 (SFT + DPO) already provides a shippable checkpoint if RL stages yield diminishing returns.
