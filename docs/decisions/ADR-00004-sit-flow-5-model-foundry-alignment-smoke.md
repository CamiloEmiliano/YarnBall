# ADR-00004: SIT Flow 5 Model Foundry Alignment Smoke

<!--
STRICT FILENAME CONVENTION:
- Format: ADR-[5_DIGIT_ID]-[40_CHARACTER_SLUG].md
- Total Filename Length: Exactly 53 characters (including '.md')
- Slug Specification: Exactly 40 characters using lowercase alphanumeric and hyphens.
- Padding Rule: If the natural slug is under 40 characters, pad with trailing hyphens ('-').
Filename: ADR-00004-sit-flow-5-model-foundry-alignment-smoke.md (53 chars)
-->

- **ADR ID**: ADR-00004
- **Filename**: `ADR-00004-sit-flow-5-model-foundry-alignment-smoke.md`
- **Date**: 2026-11-10 01:05
- **Status**: ACCEPTED
- **Target Subsystem**: `sft_pipeline` / `evaluation`
- **Linked Commits**: `a788815` (Flow 5 test harness in `tests/integration/pipelines/test_model_foundry_smoke_flow.py`)

---

## 1. Context & Problem Statement

The YarnBall financial reasoning engine (`yarnball-qwen:7b`) depends on a multi-stage post-training pipeline:
1. **Multi-Task SFT Synthesis**: Generating supervised fine-tuning corpora across five specialized financial reasoning tasks:
   - Task A: Entity & CIK Normalization (10-digit zero-padded CIK grounding).
   - Task B: SEC Regulatory Graph Extraction (triple extraction with ontology constraints).
   - Task C: Text-to-Cypher Compilation (openCypher generation with AST safety).
   - Task D: Contagion Shock Reasoning (structured `<think>...</think>` CoT traces).
   - Task E: Portfolio Hedging Directives (options overlay recommendations).
2. **DPO Preference Pair Alignment**: Mining chosen/rejected pairs to eliminate common LLM failure modes:
   - Inflection Lag: LLM assuming delayed transmission when instantaneous contract defaults occur.
   - Mode-Bleed: LLM responding in generic chat style rather than emitting structured `<think>` reasoning traces.
   - Schema Traps: LLM querying hallucinated graph properties (e.g. `c.ceo_age`).
3. **Streaming LoRA Fusion & Serving Export**: Fusing LoRA adapter weights into base SafeTensors shards via FP32 accumulation, followed by BF16 conversion, GGUF conversion, and Ollama Modelfile manifest generation.

Individual unit tests verify isolated math operations or string parsers, but fail to guarantee cross-subsystem pipeline integrity:
- ChatML formatting errors and unescaped special tokens slip into training datasets.
- Temporal split leakage allows future corporate events into training splits.
- Weight fusion tensor mapping mismatches produce corrupted GGUF weights.

System Integration Test (SIT) **Flow 5** integrates and validates the entire post-training alignment stack:
`Task Sampling -> SFT Corpus Construction -> DPO Preference Pair Synthesis -> Streaming SafeTensors Adapter Fusion -> GGUF/Ollama Packaging & Validator Verification`.

---

## 2. Decision Drivers & Constraints

1. **ChatML Formatting Integrity**:
   - All generated training pairs must strictly follow the ChatML standard (`<|im_start|>system...<|im_start|>user...<|im_start|>assistant...<|im_end|>`) without unescaped delimiters.
2. **Strict Point-in-Time Split Isolation**:
   - Datasets must enforce strict temporal boundaries (Train: <= 2023-12-31, Test/Eval: >= 2024-01-01) with zero event leakage.
3. **Bit-Exact Streaming Adapter Fusion**:
   - LoRA weight merging must accumulate in FP32 before casting to BF16, preventing underflow and numeric instability.
4. **Offline Micro-Harness Latency Budget**:
   - Tier 2 SIT smoke tests must run 100% offline using synthetic 2-layer micro-checkpoints in `< 5.0 seconds` without requiring live GPU hardware or external Hugging Face network downloads.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Full-Scale Model Training in Integration Tests**:
  - *Rejected*: Fine-tuning even a 7B model takes hours on dedicated GPU nodes. SIT integration tests must execute deterministically in CI/CD within seconds.
- **Alternative B: Black-Box HuggingFace PEFT Merge Script**:
  - *Rejected*: In-memory monolithic merging (`merge_and_unload`) triggers OOM on non-GPU environments and masks GGUF tensor naming regressions.
- **Alternative C: Unvalidated Post-Training Export**:
  - *Rejected*: Models deployed without automated validator checks (`validate_cik_integrity`, `validate_cypher_syntax`, `validate_reasoning_cot`) frequently fail in production due to subtle tokenization regressions.

---

## 4. Decision Outcome

We establish **Flow 5** as an automated end-to-end pipeline integration test in `tests/integration/pipelines/test_model_foundry_smoke_flow.py` covering:

1. **SFT Multi-Task Mini-Batch Synthesis**:
   - Generates a synthetic mini-batch across Tasks A-E and verifies valid ChatML token framing and temporal boundary partitions.
2. **DPO Preference Pair Generation**:
   - Synthesizes chosen/rejected pairs over mode-bleed and inflection-lag scenarios, asserting that rejected pairs explicitly contain targeted failure modes.
3. **SafeTensors Streaming LoRA Merger**:
   - Executes `fuse_safetensors_shard()` over synthetic base and LoRA adapter weights with FP32 accumulation, verifying numerical fidelity against analytical ground truth.
4. **Export & Validator Verification**:
   - Runs `validate_cik_integrity()`, `validate_cypher_syntax()`, and `validate_reasoning_cot()` across generated outputs to assert production readiness.

---

## 5. Invariants & Guardrails

1. **ChatML Boundary Invariant**:
   - Every synthesized sample contains exactly one system block, one user block, and one assistant block terminated by `<|im_end|>`.
2. **Temporal Split Invariant**:
   - For all training samples `S`, `S.as_of_date <= '2023-12-31'`. No future event dates exist in the training partition.
3. **DPO Divergence Invariant**:
   - In every preference pair, the chosen response passes schema and CoT validation, while the rejected response fails at least one validator check.
4. **Numerical Fusion Invariant**:
   - Merged adapter tensor elements must satisfy `|W_merged - (W_base + scaling * (B @ A))| <= 1e-3`. This tolerance is not arbitrary: it represents the theoretical precision floor of bfloat16 (BF16), which possesses only 7 mantissa bits (machine epsilon 2^-7 ≈ 7.8e-3; standard PyTorch bfloat16 comparison atol=1e-3). While intermediate accumulation is executed in FP32, final serialization to BF16 introduces truncation error bounded by this 1e-3 threshold.
5. **Offline Isolation Invariant**:
   - Zero network requests to Hugging Face Hub, WandB, or remote model repositories.

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Guarantees that prompt formatting, DPO preference pair mining, adapter weight merging, and Ollama serving packaging operate harmoniously across the post-training pipeline.
  - Catches regressions in tokenizer boundaries and tensor name mapping before expensive cloud GPU training runs.
- **Negative / Trade-offs**:
  - Requires maintaining synthetic micro-checkpoint fixtures (`config.json` and tiny BF16 tensor arrays) in test utilities.

