# ADR-00012: Upgraded Qwen Model Fine-Tuning & Serving Plan

- **ADR ID**: ADR-00012
- **Filename**: `ADR-00012-qwen-model-upgrade-and-fine-tuning-plan-.md`
- **Date**: 2026-11-12 05:50 UTC
- **Status**: PROPOSED
- **Target Subsystem**: `sft_pipeline` / `retrieval`
- **Linked Commits**: [`a26bc0c`](../../sft_pipeline/serving/model_client.py)

---

## 1. Context & Problem Statement

YarnBall's neural reasoning and Text-to-Cypher interface is currently parameterized around a 7B dense transformer (`yarnball-qwen:7b`) with traditional softmax attention.

In financial GraphRAG pipelines, expanding topological subgraphs (multi-hop entity chains, regulatory disclosure passages, inventory buffer matrices) yields long prompt contexts ($8\text{k}-16\text{k}+$ tokens). Under standard softmax attention, the key-value (KV) cache memory footprint scales strictly linearly $O(L)$ with sequence length:
$$\text{KV Cache VRAM} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times L \times \text{bytes\_per\_element}$$

On an 8 GB consumer GPU budget:
- A 7B model at 4-bit (`Q4_K_M`) consumes $\approx 4.3 \text{ GB}$ base weights.
- At 16k context, an FP16 KV cache consumes an additional $1.5 - 2.0 \text{ GB}$ of VRAM. Combined with the CUDA runtime overhead ($\approx 0.6 \text{ GB}$) and desktop framebuffers, standard attention pushes an 8 GB card to the verge of an Out-Of-Memory (OOM) crash during multi-hop graph expansion.

Upgrading to an advanced ~9B architecture featuring linear attention (or hybrid linear/recurrent state mechanisms, e.g. Qwen 3.5 9B) resolves this constraint by condensing past context into a fixed-size $O(1)$ recurrent state matrix $S_t \in \mathbb{R}^{d \times d}$. The memory footprint of the attention state during autoregressive generation becomes sub-quadratic and constant-bounded, decoupling VRAM growth from GraphRAG context size.

Furthermore, a foundational principle of the YarnBall platform is the **LLM Swapability Invariant**:
> The Large Language Model in YarnBall is strictly an exchangeable verbalizer and DSL translator. The deterministic graph cascade physics (`SparseNetworkCascadeEngine`), regulatory schemas (ADR-00011), and AST query guardrails (`CypherASTWhitelistGuardrail`) are completely independent of the underlying neural backbone. Consequently, all pipeline and integration tests executed against the current fine-tuned model are functionally equivalent to runs on future upgraded models, differing only in parametric hallucination rates and syntax fidelity.

---

## 2. Decision Drivers & Constraints

1. **8 GB VRAM Envelope**:
   - Model weights for a 9B model quantized at 4-bit (`Q4_K_M`): $\approx 5.1 - 5.3 \text{ GB}$.
   - Linear attention recurrent state: $\approx 0.1 - 0.3 \text{ GB}$ (fixed-size).
   - Total runtime footprint: $\approx 5.8 - 6.2 \text{ GB}$, preserving a safe $\approx 1.8 \text{ GB}$ buffer for CUDA runtime, memory fragmentation, and host graphics.
2. **Zero SFT Dataset Modification**:
   - The supervised datasets (`data/sft/train_active_learning.jsonl`, `data/sft/val_curated.jsonl`), data cartography splits, rejection queue candidates, and ChatML prompt structures (`<|im_start|>`, `<|im_end|>`) are 100% token-agnostic and must be reused without modification.
3. **Prefix-Forced Reasoning Preservation**:
   - The `<think>\n` prefix forcing protocol established in ADR-00004 for reasoning tasks (Tasks D and E) and strict zero-think prefix for extraction/Cypher tasks (Tasks A, B, and C) must be maintained.
4. **Decoupled Verification Pace**:
   - Operational testing, integration harnesses, and dry-run CLI tools must remain fully operational using existing mock clients and current model checkpoints without blocking on new training runs.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Retain 7B Softmax Model**:
  - *Rationale for Rejection*: Fails to solve the $O(L)$ KV cache scaling dilemma on long GraphRAG context windows, causing OOM risk on 8 GB cards when verbalizing deep supply trees.
- **Alternative B: 14B Model Quantized to 2-Bit/3-Bit**:
  - *Rationale for Rejection*: Severe perplexity degradation at ultra-low bit widths produces syntax errors in Cypher DSL generation (Task C) and numerical errors when quoting verbatim cascade metrics (Task D/ADR-00011).
- **Alternative C: External Cloud LLM APIs (e.g., GPT-4o, Claude 3.5)**:
  - *Rationale for Rejection*: Violates YarnBall's air-gapped data sovereignty and low-latency offline execution invariants.

---

## 4. Decision Outcome: 5-Stage Fine-Tuning & Serving Roadmap

```
Stage 1: Base Model & Tokenizer Verification (VRAM budget & ChatML validation)
   │
   ▼
Stage 2: Supervised Fine-Tuning (SFT) (PEFT/LoRA on train_active_learning.jsonl)
   │
   ▼
Stage 3: Direct Preference Optimization (DPO) (Alignment on mode bleed & domain isolation)
   │
   ▼
Stage 4: Weight Fusion, GGUF Q4_K_M Quantization & Local Ollama Registration
   │
   ▼
Stage 5: Test Suite Regression & Verbatim Citation Validation (ADR-00011 compliance)
```

### Stage 1: Base Model & Tokenizer Verification
- Download base model weights for target 9B linear-attention architecture (e.g. Qwen 3.5 9B).
- Validate tokenizer compatibility with ChatML turn formatting (`<|im_start|>system...`, `<|im_start|>user...`, `<|im_start|>assistant...`).
- Verify base 4-bit memory consumption under Ollama / vLLM.

### Stage 2: Supervised Fine-Tuning (SFT)
- **Input Corpus**: Direct ingestion of existing `data/sft/train_active_learning.jsonl` containing the balanced 5-task manifold:
  - Task A: `EXTRACT_SEC_GRAPH`
  - Task B: `EXTRACT_NEWS_EVENT`
  - Task C: `TEXT_TO_CYPHER`
  - Task D: `CONTAGION_REASONING`
  - Task E: `PORTFOLIO_RECOMMENDATION` (Domain 1 supply hedging only)
- **LoRA Configuration**:
  - Rank $r=16$, $\alpha=32$, dropout $0.05$.
  - Target modules: Attention projections, recurrent linear state matrices, and feed-forward MLP layers.
- **Loss Masking**:
  - Prompt tokens masked with label `-100`; only assistant completions contribute to backpropagation gradients.
- **Training Dynamics Logging**:
  - Log per-example confidence and variability across epochs for continuous data cartography auditing.

### Stage 3: Direct Preference Optimization (DPO)
- Fine-tune the SFT adapter using candidate pairs generated by `sft_pipeline/dpo_builder.py`.
- Primary alignment objectives:
  1. **Mode-Bleed Suppression**: Eliminate first-token Cypher syntax leaks in Tasks D and E.
  2. **Domain Isolation Enforcement**: Penalize any generation of speculative trading recommendations or market-making jargon (per rejection of ADR-00010).

### Stage 4: Fusion, Quantization & Serving
- Fuse LoRA adapter into base weights in BF16 precision.
- Convert fused safetensors to GGUF format and quantize using `Q4_K_M`.
- Register the quantized model with the local Ollama daemon under tag `yarnball-qwen:9b` using canonical serving parameters (temperature 0.1, stop token `<|im_end|>`, and the standardized YarnBall system prompt).
- Configure `sft_pipeline/serving/model_client.py` via `YARNBALL_MODEL_TAG=yarnball-qwen:9b` (or pass model tag explicitly).

### Stage 5: Test Suite Regression & Citation Audit
- Execute the full test suite (`pytest tests/retrieval/ tests/integration/`).
- Run dry-run computational reading (`tools/dry_run_sec_8k_contagion.py --validate-citations`) to verify that the upgraded model achieves zero verbatim citation violations against `SparseNetworkCascadeEngine` metrics.

---

## 5. Invariants & Guardrails

1. **The LLM Swapability Invariant**:
   - The underlying language model must remain strictly decoupled behind `YarnBallModelClient`. No graph traversal, cascade algorithm, or reporting contract may introduce model-specific dependencies.
2. **Strict 8 GB Hardware Envelope**:
   - Peak VRAM allocation for batch size 1 inference at 16k sequence length must never exceed $7.5 \text{ GB}$.
3. **Zero Dataset Re-Engineering**:
   - The SFT training and validation corpora remain immutable across model upgrades. Model improvements must stem from superior base pretraining, linear attention scaling, and increased parameter density.
4. **Verbatim Metric Citation Invariant (ADR-00011)**:
   - When synthesizing operational disruption narratives (Task D), the model must cite all cascade numerical metrics (onset days, downtime days, buffer days, capacity contraction) verbatim from the deterministic engine.

---

## 6. Consequences & Downstream Impact

### Positive Guarantees
- **Linear Attention Scaling**: Memory footprint during GraphRAG graph verbalization becomes constant-bounded ($O(1)$ recurrent state), completely eliminating OOM risks on 8 GB cards.
- **Higher Cypher Fidelity (Task C)**: Increased parametric capacity (~9B vs 7B) reduces syntax hallucinations and AST guardrail tripping on complex multi-hop path queries.
- **Improved Verbatim Numerical Precision**: Better instruction-following fidelity lowers verbalization citation violations on ADR-00011 operational reports.

### Negative / Accepted Trade-offs
- **Compute Cost for Retraining**: LoRA fine-tuning and DPO alignment on a 9B model will require approximately $1.5\times$ to $2.0\times$ more GPU compute time than the 7B baseline on rental hardware (e.g. Vast.ai RTX 4090 / A100).
- **Inference Latency Trade-off**: Slightly larger matrix multiplications will increase decode latency by $\approx 15-25\%$, though this is partially offset by faster linear attention prefill on long contexts.
