# ADR-00044: Chapter 4 Technical Report — Curation, Cartography & Model Foundry

- **ADR ID**: ADR-00044
- **Filename**: `ADR-00044-ch4-curation-cartography-model-foundry--.md`
- **Date**: 2026-11-25 04:29 UTC
- **Status**: PROPOSED
- **Target Subsystem**: `docs` / `curation` / `sft_pipeline`
- **Linked Commits**: Pending acceptance

---

## 1. Context & Problem Statement

1. **YarnBall's richest concentration of pre-ADR data-science and optimization algorithms lives in `curation/` and `sft_pipeline/`.** Because the ADR registry began at [`ADR-00001`](ADR-00001-sit-flow-2-event-shock-contagion-harness.md) during system-integration harness verification, almost the entire mathematical data-curation, active-learning, manifold-sampling, preference-alignment, adapter-fusion, and multi-judge evaluation stack predates the ADR log:
   - **Pre-ADR foundational subsystems**: MinHash LSH syntactic deduplication ([`curation/cold_start/minhash_lsh.py`](../../curation/cold_start/minhash_lsh.py)), SemDeDup dense semantic deduplication ([`curation/cold_start/semdedup.py`](../../curation/cold_start/semdedup.py)), submodular Facility Location core-set selection ([`curation/cold_start/coreset.py`](../../curation/cold_start/coreset.py)), AllenAI-adapted Dataset Cartography & Toneva forgetfulness mapping ([`curation/active_learning/cartographer.py`](../../curation/active_learning/cartographer.py)), economic manifold-targeted sampling & hard-negative synthesis ([`sft_pipeline/sampler.py`](../../sft_pipeline/sampler.py)), multi-task SFT dataset construction & leak-free split export ([`sft_pipeline/builder.py`](../../sft_pipeline/builder.py), [`sft_pipeline/exporter.py`](../../sft_pipeline/exporter.py)), empirical inflection-lag DPO pair construction & TRL training ([`sft_pipeline/dpo_builder.py`](../../sft_pipeline/dpo_builder.py), [`sft_pipeline/train_dpo.py`](../../sft_pipeline/train_dpo.py)), sequential Safetensors BF16/FP32 LoRA adapter fusion & GGUF export ([`sft_pipeline/phase4_export_and_serve.py`](../../sft_pipeline/phase4_export_and_serve.py)), and the Phase-5 adversarial torture suite & multi-judge audit (`sft_pipeline/evaluation/`).
   - **Post-`ADR-00001` governance & forward roadmap**: SIT Flow 5 alignment smoke verification ([`ADR-00004`](ADR-00004-sit-flow-5-model-foundry-alignment-smoke.md)), runtime rejection quarantine & iterative predicate convergence ([`ADR-00029`](ADR-00029-structured-llm-extraction-and-quarantine.md), [`ADR-00031`](ADR-00031-iterative-predicate-convergence-protocol.md)), unified `YarnBallModelClient` serving & non-FT fallback purge ([`ADR-00030`](ADR-00030-purge-non-ft-qwen-runtime-dependencies--.md)), Domain-1 task scope isolation ([`ADR-00010`](ADR-00010-directional-basket-peer-cluster-mapping-.md), [`ADR-00034`](ADR-00034-single-bi-read-facade-and-pipeline-merge.md)), plus the `PROPOSED` forward evolution in [`ADR-00012`](ADR-00012-qwen-model-upgrade-and-fine-tuning-plan-.md) (Qwen3.5-9B Gated DeltaNet + Task-Selective SFT/GRPO) and [`ADR-00038`](ADR-00038-event-study-backtest-and-signal-decay---.md) (First-Instance Event-Study Backtest).
2. **Legacy `sft_pipeline/README.md` suffers from pre-`ADR-00034` drift.** Existing notes in `sft_pipeline/` do not contextualize historical Task E (`portfolio_recommendation`) against the Domain-1 operational boundary (`ADR-00010`) or connect runtime `RuntimeRejectionQueue` telemetry back into dataset curation.
3. **Need for Blog Part 4 + AST Model-Foundry Reference.** Chapter 4 (`docs/technical-report/04-curation-and-model-foundry.md`) closes the 5-chapter series by documenting how YarnBall curates, aligns, fuses, serves, and audits domain-specialized 7B/9B models on an 8 GB VRAM budget.

---

## 2. Decision Drivers & Constraints

1. **Recover and formalize pre-ADR optimization & learning-dynamics math**: derive MinHash universal hashing & LSH banding probabilities, SemDeDup centroid retention, submodular Facility Location greedy maximization, Dataset Cartography $(\hat{\mu}_i, \hat{\sigma}_i)$ & forgetfulness dynamics, manifold rare-class floors / hard negatives, inflection-lag DPO (`seq_prob < 0.70` at Steps 50/100), and LoRA weight fusion.
2. **Explicitly distinguish active runtime serving from historical vs. `PROPOSED` foundry artifacts**: clearly separate (a) active `yarnball-qwen:7b` serving via [`YarnBallModelClient`](../../sft_pipeline/serving/model_client.py), (b) historical SFT/DPO Task E training slices (offline research/alignment only—excluded from Domain-1 runtime per [`ADR-00010`](ADR-00010-directional-basket-peer-cluster-mapping-.md)), and (c) `PROPOSED` next-generation designs ([`ADR-00012`](ADR-00012-qwen-model-upgrade-and-fine-tuning-plan-.md), [`ADR-00038`](ADR-00038-event-study-backtest-and-signal-decay---.md)).
3. **Dual-level visual intuition (`ADR-00040` §4.1)**: pair a **Part A Topic-Level UML Diagram** (`classDiagram`) of the closed-loop data-curation, training-dynamics, alignment, serving, and evaluation flywheel with **Part B AST Call Graphs** (`flowchart`) and artifact contracts.
4. **Honest Pre-ADR vs. ADR provenance & mechanized drift verification**: attribute every pre-ADR algorithm and ADR refinement in the Theory-to-Code Crosswalk Table, validated by `tests/docs/test_docs_integrity.py`.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Treat `curation/` and `sft_pipeline/` as offline ML scripts out of scope for the core platform report.** Rejected. YarnBall's runtime extraction (`EXTRACT_SEC_GRAPH`), Cypher translation (`TEXT_TO_CYPHER`), contagion reasoning (`CONTAGION_REASONING`), and community summarization (`COMMUNITY_REPORT`) depend directly on the fine-tuned `yarnball-qwen` model ([`ADR-00030`](ADR-00030-purge-non-ft-qwen-runtime-dependencies--.md)), and `curation/` + `sft_pipeline/` contain core data-science contributions central to the technical blog series.
- **Alternative B: Split `curation/` + `sft_pipeline/` and `sft_pipeline/evaluation/` into two separate chapters (creating a 6-chapter series).** Rejected. Dataset Cartography (`curation/active_learning/`), inflection-lag DPO (`dpo_builder.py`), and the Phase-5 torture & multi-judge evaluation suite (`sft_pipeline/evaluation/`) form one continuous closed-loop alignment lifecycle; keeping them together preserves the approved 5-chapter (`Ch. 0`–`Ch. 4`) structure.
- **Alternative C: Document `ADR-00012` (Qwen3.5-9B + GRPO) and `ADR-00038` (Event-Study Backtest) as already-deployed runtime code.** Rejected. Both carry `Status: PROPOSED`; Chapter 4 must document the implemented Qwen 2.5 7B SFT + Inflection-Slice DPO + GGUF pipeline as the present-tense system and clearly demarcate `ADR-00012` and `ADR-00038` as proposed next-stage architecture.

---

## 4. Decision Outcome

Author **`docs/technical-report/04-curation-and-model-foundry.md`** (*Part 4: Submodular Curation, Dataset Cartography & Task-Selective Model Alignment*) and register it in `mkdocs.yml` and `tests/docs/test_docs_integrity.py` with the following exact structure:

### 4.1 Part A — Domain & Theoretical Technical Report (*What & Why* / Blog Part 4)

1. **Cold-Start Deduplication & Submodular Facility Location Core-Set Selection (`curation/cold_start/`)**:
   - Why raw SEC filings and syndicated financial news wires induce catastrophic boilerplate memorization and sector skew in 7B parameter models.
   - Stage 1 — Syntactic deduplication ([`MinHashLSHDeduplicator`](../../curation/cold_start/minhash_lsh.py)): 5-gram shingling, universal linear permutations $h_i(s) = (a_i s + b_i) \bmod (2^{31}-1)$ ($P = 128$, $b = 16$ bands $\times r = 8$ rows), S-curve candidate probability $1 - (1 - J^r)^b$, and Jaccard cutoff $\ge 0.85$.
   - Stage 2 — Dense semantic deduplication ([`SemanticDeduplicator`](../../curation/cold_start/semdedup.py)): unit-sphere cosine graph clustering ($\cos(u, v) \ge 0.88$) retaining cluster-centroid exemplars $\arg\max_{i \in C} \langle \hat{v}_i, \hat{\mu}_C \rangle$.
   - Stage 3 — Submodular core-set selection ([`FacilityLocationCoresetSelector`](../../curation/cold_start/coreset.py)): maximizing the monotone submodular Facility Location objective under cardinality budget $K = \lfloor 0.80\,N \rfloor$:
     $$F(S) = \sum_{i \in V} \max_{j \in S}\, \text{sim}_+(i, j), \qquad \Delta(c \mid S) = \sum_{i \in V} \max\!\big(0,\; \text{sim}_+(i, c) - \alpha_i\big)$$
     guaranteeing a $(1 - 1/e)$-approximate geometric cover across all 11 GICS sectors.
2. **Economic Manifold Sampling & Hard-Negative Boundary Mining (`sft_pipeline/sampler.py`)**:
   - Escaping ambient `SUPPLIES_TO` class dominance on the economic manifold $\mathcal{M}^*$: enforcing rare risk-predicate floors (`RARE_RELATION_FLOORS`: `400` for `SOLE_SOURCE_DEPENDENT_ON`, `LICENSES_FROM`, `LICENSES_TO`, `DEFAULTED_ON`, `EXPOSED_TO_RISK`; `500` for `ACQUIRED_BY`) and `DOMINANT_CLASS_CAP = 1500`.
   - Boundary-proximity hard negatives (`null_sample_ratio = 0.20`): multi-issuer market commentary passages co-mentioning S&P 500 tickers with zero active KG edge mapped strictly to `(none)`, plus corporate spin-off separation grounding (`GE` vs. `GEV`, `RTX` vs. `CARR`, `ABT` vs. `ABBV`).
3. **Dataset Cartography, Learning Dynamics & Closed-Loop Steering (`curation/active_learning/`)**:
   - Mapping per-instance sequence-probability trajectories across training checkpoints in [`DatasetCartographer`](../../curation/active_learning/cartographer.py) (Swayamdipta et al., 2020; Toneva et al., 2019):
     $$\hat{\mu}_i = \frac{1}{E}\sum_{e=1}^E p_{\theta^{(e)}}(y_i^* \mid x_i), \qquad \hat{\sigma}_i = \sqrt{\frac{1}{E}\sum_{e=1}^E \Big(p_{\theta^{(e)}}(y_i^* \mid x_i) - \hat{\mu}_i\Big)^2}$$
   - Active-learning partition rule (`var_thresh = 0.15`, `conf_thresh_high = 0.70`, `conf_thresh_low = 0.30`): retain `100%` **Ambiguous** boundary instances, prune `80%` redundant **Easy-to-Learn** boilerplate, quarantine `100%` **Hard-to-Learn** / never-learned (`forgetfulness = 999`) instances for CIK/label audit, and feed ambiguous relations back into `ManifoldTargetedSampler.steer_from_cartography` ($1.5\times$ floor boost).
4. **Multi-Task SFT, Empirical Inflection-Lag DPO, Adapter Fusion & 8 GB VRAM Serving (`ADR-00030`)**:
   - Multi-task corpus formulation ([`FullSFTDatasetBuilder`](../../sft_pipeline/builder.py), [`SFTDatasetExporter`](../../sft_pipeline/exporter.py)), explicitly documenting why historical Task E (`portfolio_recommendation`) is quarantined to offline alignment research and excluded from Domain-1 runtime serving ([`ADR-00010`](ADR-00010-directional-basket-peer-cluster-mapping-.md), [`ADR-00034`](ADR-00034-single-bi-read-facade-and-pipeline-merge.md)).
   - Inflection-lag preference alignment ([`InflectionSliceDPOBuilder`](../../sft_pipeline/dpo_builder.py), [`train_dpo.py`](../../sft_pipeline/train_dpo.py)): harvesting slow-mastery cohorts (`seq_prob < 0.70` at Steps 50 & 100, early AULC over $\{0, 25, 50, 100, 200\}$) across 4 failure modes (eliminating Task-D OpenCypher mode-bleed, subsidiary-to-parent CIK rollup, borderline volatility $z$-score calibration, and indexed Cypher generation).
   - Sequential LoRA weight fusion ($W_{\text{fused}} = W_0 + \frac{\alpha_{\text{SFT}}}{r_{\text{SFT}}} B_{\text{SFT}} A_{\text{SFT}} \rightarrow + \frac{\alpha_{\text{DPO}}}{r_{\text{DPO}}} B_{\text{DPO}} A_{\text{DPO}}$ in FP32 before BF16/GGUF serialization in [`phase4_export_and_serve.py`](../../sft_pipeline/phase4_export_and_serve.py)) and single-client Ollama serving via [`YarnBallModelClient`](../../sft_pipeline/serving/model_client.py) (`ADR-00030`, runaway repetition detection, structured JSON/telemetry).
5. **Adversarial Torture Evaluation, Multi-Judge Audit & Forward Alignment Roadmap (`ADR-00004`, `ADR-00012`, `ADR-00029`, `ADR-00031`, `ADR-00038`)**:
   - 4-axis deterministic torture evaluation ([`phase5_torture_evaluators.py`](../../sft_pipeline/evaluation/phase5_torture_evaluators.py): verdict/polarity, causal hop trace, epistemic provenance, schema/CIK integrity), multi-judge inter-annotator audit ([`phase5_judge_audit.py`](../../sft_pipeline/evaluation/phase5_judge_audit.py)), [`EvaluationHarness`](../../synthesis/eval_harness.py), and closed-loop runtime quarantine ([`RuntimeRejectionQueue`](../../sft_pipeline/rejection_queue.py)).
   - Clearly marked **Forward Architecture Roadmap (`PROPOSED`)**: how [`ADR-00012`](ADR-00012-qwen-model-upgrade-and-fine-tuning-plan-.md) (Qwen3.5-9B Gated DeltaNet + Task-Selective SFT/GRPO) and [`ADR-00038`](ADR-00038-event-study-backtest-and-signal-decay---.md) (First-Instance Event-Study Backtest & Signal-Decay Calibration) build on this foundation.
6. **Topic-Level UML Diagram & Theory-to-Code Crosswalk Table**:
   - Mermaid `classDiagram` modeling `ColdStartSubmodularCurator`, `ManifoldTargetedSampler`, `DatasetCartographer`, `InflectionSliceDPOFoundry`, `LoRAFusionAndGGUFServing`, `RuntimeRejectionFlywheel`, and `TortureAndJudgeAudit`.
   - Full Crosswalk Table mapping every objective function, threshold, and alignment step to `module:symbol`, provenance (`Pre-ADR Foundation` vs. `ADR-00004`/`00012`/`00029`–`00031`/`00038`), and verification tests.

### 4.2 Part B — Software Architecture, Call Graphs & Entry Points (*Where & How*)

1. **Subsystem Module Inventory**: complete file-by-file breakdown across `curation/` (7 modules: `contracts.py`, `cold_start/*`, `active_learning/*`), `sft_pipeline/` (18 modules including `serving/model_client.py` and `evaluation/*`), [`synthesis/eval_harness.py`](../../synthesis/eval_harness.py), and [`tools/dry_run_aapl_pipeline.py`](../../tools/dry_run_aapl_pipeline.py).
2. **AST-Verified Execution Call Graphs (3 Mermaid `flowchart` pipelines)**:
   - *Pipeline 1 — Cold-Start Curation, Manifold Sampling & Split Export*: `ColdStartCurationPipeline` (`MinHashLSHDeduplicator` $\rightarrow$ `SemanticDeduplicator` $\rightarrow$ `FacilityLocationCoresetSelector`) $\rightarrow$ `ManifoldTargetedSampler` $\rightarrow$ `FullSFTDatasetBuilder` $\rightarrow$ `SFTDatasetExporter.export_full_sft_splits`.
   - *Pipeline 2 — Cartography Dynamics, Inflection-Slice DPO, LoRA Fusion & Serving*: `DatasetCartographer.map_dataset` & `filter_active_learning` $\rightarrow$ `InflectionSliceDPOBuilder.analyze_inflection_dynamics` $\rightarrow$ `run_dpo_training` $\rightarrow$ `merge_sft_and_dpo_adapters` / GGUF export $\rightarrow$ `YarnBallModelClient` $\leftrightarrow$ `RuntimeRejectionQueue` (`ADR-00029` / `ADR-00031` feedback loop).
   - *Pipeline 3 — Phase-5 Adversarial Torture Suite & Multi-Judge Calibration Audit*: `build_phase5_torture_suite` $\rightarrow$ `run_phase5_torture_suite` (`evaluate_torture_sample` + `poll_gpu_vram_mib`) $\rightarrow$ `build_audit` (`phase5_judge_audit.py`).
3. **CLI Entry Points, Training/Serving Artifacts & JSONL Contracts**:
   - CLI scripts across `sft_pipeline/` and `tools/dry_run_aapl_pipeline.py`, plus exact JSONL schemas for `SFTRecord`, `DPORecord`, `training_dynamics.jsonl`, `CartographyCoordinate`, `RejectionRecord`, and Phase-5 torture scorecards.

---

## 5. Invariants & Guardrails

1. **All `ADR-00040` Series Invariants Apply**: zero host-specific paths, pure GFM + `$...$`/`$$...$$` + `mermaid` in `04-curation-and-model-foundry.md` (zero `mkdocstrings` `:::` directives in narrative prose), AST-verified call-graph edges, and honest `Pre-ADR Foundation` vs. `ADR-XXXXX` attribution.
2. **Chapter-4 Constant & Symbol Verification**: `tests/docs/test_docs_integrity.py` must verify all `module:symbol` references in Chapter 4's Crosswalk Table alongside live constants (`num_perm = 128`, `jaccard_threshold = 0.85`, `shingle_size = 5`, `num_bands = 16`, `cosine_threshold = 0.88`, `retention_fraction = 0.80`, `conf_thresh_high = 0.70`, `conf_thresh_low = 0.30`, `var_thresh = 0.15`, `prune_easy_ratio = 0.80`, `DOMINANT_CLASS_CAP = 1500`, `null_sample_ratio = 0.20`).
3. **Strict `PROPOSED` vs. `ACCEPTED` Demarcation**: Chapter 4 must never describe `PROPOSED` records (`ADR-00012`, `ADR-00038`) as active runtime behavior.

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Completes the 5-chapter / 5-ADR Dual-Lens Technical Report specification (`ADR-00040`–`ADR-00044`), surfacing YarnBall's entire pre-ADR curation, cartography, manifold-sampling, DPO, and multi-judge evaluation architecture into a publication-ready *Blog Part 4* and engineering reference.
  - Explicitly resolves the historical Task E documentation drift by contextualizing offline training artifacts against the Domain-1 operational wall.
- **Negative / Trade-offs**:
  - Updating curation thresholds, manifold class floors/caps, or model-client constants (e.g., when implementing `ADR-00012`) will require synchronized updates to `04-curation-and-model-foundry.md` to pass `test_docs_integrity.py`.
