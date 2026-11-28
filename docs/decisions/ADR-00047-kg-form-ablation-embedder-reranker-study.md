# ADR-00047: Controlled KG Form-Ablation Scorecard & Embedder/Reranker Scaling Study

- **ADR ID**: ADR-00047
- **Filename**: `ADR-00047-kg-form-ablation-embedder-reranker-study.md`
- **Date**: 2026-11-28 04:46 UTC
- **Status**: ACCEPTED
- **Target Subsystem**: `knowledge_graph` / `embedding` / `retrieval` / `tools` / `tests`
- **Linked Commits**: [`7d6df08`](../../knowledge_graph/ablation.py), [`7d6df08`](../../knowledge_graph/chunk_store.py), [`7d6df08`](../../embedding/embedder.py), [`7d6df08`](../../embedding/reranker.py), [`7d6df08`](../../tools/eval_kg_form_ablation.py), [`7d6df08`](../../tests/knowledge_graph/test_kg_form_ablation_adr00047.py)

---

## 1. Context & Problem Statement

[`ADR-00045`](ADR-00045-canonical-sec-edgar-filing-registry-----.md) established the exhaustive 33-form SEC EDGAR canon and 4-layer graph partition (`DOMAIN1_OPERATIONAL`, `GOVERNANCE_INTERLOCK`, `CAPITAL_OWNERSHIP`, `FINANCING_OFFERING`) in [`ingestion/sec/canon.py`](../../ingestion/sec/canon.py), and [`ADR-00046`](ADR-00046-sec-core-sections-amendments-exhibit-991.md) expanded core statutory extraction across `10-K`/`10-K/A` (6 sections + `Exhibit 21`), part-qualified `10-Q`/`10-Q/A`, `8-K`/`8-K/A` + `Exhibit 99.1`, and sector-aware Pass-2 gleaning.

Before committing further engineering and extraction budgets to Phase 2b/2c/2d forms (`Exhibit 10`, XBRL `companyfacts`, `Form 10`, `S-4`, `20-F`, `6-K`, `DEF 14A`, `SD`), optional ownership/financing forms (`4`, `13F-HR`, `SC 13D/13G`, `424B2`, `S-3`), or a heavier retrieval stack, YarnBall must replace architectural intuition with **controlled, reproducible empirical measurement** across three coupled questions:

1. **Marginal Knowledge Graph Quality per SEC Form**:
   Which SEC form types genuinely improve supply-chain edge coverage, epistemic provenance grounding (`FILING_GROUNDED` vs. `PARTIALLY_ASSUMED` vs. `STRESS_UPPER_BOUND`), community modularity $Q$, and evidence retrieval—and which forms degrade the graph via institutional hub collapse, `MAX_SUBGRAPH_EDGES = 25` truncation (`dropped_impact_share`), or extraction quarantine noise?
2. **Flat Ingestion vs. `SECGraphLayer` Partitioning**:
   When non-operational forms (`CAPITAL_OWNERSHIP` such as `13F-HR`/`SC 13G`/`Form 4`, or `FINANCING_OFFERING` such as `424B2`/`S-3`) enter the repository, how much topological degradation is avoided by strictly isolating them outside `DOMAIN1_OPERATIONAL`?
3. **Embedder Scaling vs. Second-Stage Cross-Encoder Reranking Under Form-Volume Dilution**:
   As expanded SEC forms and sections multiply `document_chunks`, does upgrading YarnBall's default [`qwen3-embedding:0.6b`](../../embedding/embedder.py) (`1024`-d, CPU-pinned `num_gpu: 0`) to a larger Matryoshka bi-encoder (`qwen3-embedding:4b`, MRL-truncated & L2 re-normalized to `1024`-d) or adding a compact second-stage cross-encoder reranker over Hybrid RRF candidates deliver statistically significant gains in Recall@K / MRR / nDCG within YarnBall's 8 GB VRAM serving envelope ([`ADR-00012`](ADR-00012-qwen-model-upgrade-and-fine-tuning-plan-.md))?

An audit of the live codebase reveals three concrete plumbing gaps blocking this study today:

- **Missing `form_type` & `sec_graph_layer` columns on text chunks**: While `kg_edges` / `edges.parquet` record `form_type`, `DocumentChunkRecord` in [`knowledge_graph/chunk_store.py`](../../knowledge_graph/chunk_store.py), PostgreSQL `document_chunks` ([`tools/init_db.py`](../../tools/init_db.py)), and `chunks.parquet` ([`knowledge_graph/snapshot_manager.py`](../../knowledge_graph/snapshot_manager.py)) do not persist `form_type` or `sec_graph_layer`, preventing fast, lossless chunk-level Leave-One-Form-Out (LOFO) slicing.
- **No Matryoshka (MRL) truncation or pluggable reranker hook**: [`embedding/embedder.py`](../../embedding/embedder.py) validates exact `EMBEDDING_DIM = 1024` parity without an explicit MRL slice-and-renormalize path for higher-dimensional embedders (`2560`-d $\rightarrow$ `1024`-d), and `search_chunks_hybrid_rrf` in [`knowledge_graph/chunk_store.py`](../../knowledge_graph/chunk_store.py) has no pluggable `RerankerPort` hook over its top-$M$ RRF candidates.
- **No unified offline KG-quality & form-ablation scorecard harness**: Topology metrics (Leiden modularity $Q$, degree Gini, 2-hop ego expansion), cascade provenance metrics (`dropped_impact_share`, `ProvenanceTier`), quarantine rates ([`sft_pipeline/rejection_queue.py`](../../sft_pipeline/rejection_queue.py)), and retrieval metrics (`Recall@K`, `MRR@10`, `nDCG@10` with paired bootstrap confidence intervals) are not yet aggregated into a single reproducible ablation runner and plot generator.

---

## 2. Decision Drivers & Constraints

1. **Empirical Discipline Over Dogma**: No SEC form promotion/exclusion and no embedder/reranker swap may be adopted into production defaults without measured numbers, paired confidence intervals, and archived plots from a reproducible offline run.
2. **Zero Dimensionality Confound (`VECTOR(1024)` Parity)**: All first-stage embedding arms (`0.6B` native vs. `4B` MRL) must emit unit-norm `1024`-d vectors so `pgvector` HNSW index geometry and RAM footprint remain strictly identical across arms.
3. **8 GB VRAM Live-Serving Preservation (`ADR-00012` / `ADR-00035`)**: Production serving reserves GPU VRAM for the generative backbone (`yarnball-qwen` 7B/9B). Live query embedding and any promoted reranker must satisfy explicit CPU-pinned (`num_gpu: 0`) latency and host-RAM budgets with zero generative-model eviction (while offline bulk indexing may continue using `batch_gpu=True` after `unload_ollama_model`).
4. **Zero Extraction Variance in Form-Removal (LOFO) Ablations**: Leave-One-Form-Out comparisons must slice a single persisted, `form_type`-tagged snapshot rather than re-running stochastic LLM extraction for every subset.
5. **Strict Scope Boundaries**:
   - This ADR owns **KG quality assessment, SEC form-removal/inclusion ablation, and embedder/reranker scaling**.
   - Generative 9B backbone fine-tuning, RLVR, and optional gated RLAIF remain governed by [`ADR-00012`](ADR-00012-qwen-model-upgrade-and-fine-tuning-plan-.md).
   - Market-return event studies and signal-decay backtesting remain governed by [`ADR-00038`](ADR-00038-event-study-backtest-and-signal-decay---.md) (and never leak trading/return fields into Domain 1 runtime schemas per [`ADR-00010`](ADR-00010-directional-basket-peer-cluster-mapping-.md)).

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Unilaterally replace `qwen3-embedding:0.6b` with `qwen3-embedding:4b` (or `8b`) at native `2560`-d / `4096`-d without a controlled ablation.** Rejected. Mutating `VECTOR(1024)` inflates HNSW RAM $2.5\times$–$4\times$, requires full database re-migration, risks exceeding live CPU latency or 8 GB VRAM budgets, and cannot separate embedding-quality bottlenecks from low-signal SEC form noise.
- **Alternative B: Rank SEC forms solely by raw extracted edge count or uncalibrated LLM-as-a-Judge prose ratings.** Rejected. Raw edge counts reward high-volume noise and institutional ownership cliques (`13F-HR`, `424B2`) that collapse community modularity $Q$ and inflate `dropped_impact_share`; uncalibrated LLM judges exhibit verbosity bias.
- **Alternative C: Re-run full LLM graph extraction independently for each Leave-One-Form-Out cell.** Rejected. Multiplies extraction compute $O(|\mathcal{F}|)$ and confounds true form effects with LLM sampling variance. Single-pass extraction tagged with canonical `(form_type, section, sec_graph_layer)` followed by deterministic snapshot filtering yields exact, zero-variance LOFO deltas $\Delta M(f)$.

---

## 4. Decision Outcome

### 4.1 Chunk-Level Form & Layer Provenance Extension

Extend text-unit persistence so every chunk carries first-class canonical form and graph-layer provenance alongside `kg_edges`:

1. **`DocumentChunkRecord` ([`knowledge_graph/chunk_store.py`](../../knowledge_graph/chunk_store.py))**:
   - Add backward-compatible optional fields `form_type: Optional[str] = None` and `sec_graph_layer: Optional[str] = None`.
   - Normalize `form_type` via `normalize_form_type` and auto-derive `sec_graph_layer` from `get_form_spec(form_type).graph_layer.value` when omitted.
2. **PostgreSQL & Parquet Snapshot Persistence**:
   - Add idempotent `ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS form_type TEXT;` and `sec_graph_layer TEXT;` (plus composite index on `(form_type, valid_from)`) in [`tools/init_db.py`](../../tools/init_db.py).
   - Persist `form_type` and `sec_graph_layer` in `chunks.parquet` via [`knowledge_graph/snapshot_manager.py`](../../knowledge_graph/snapshot_manager.py) and stamp both fields in [`ingestion/sec/harvester.py`](../../ingestion/sec/harvester.py) (`EdgarWorker`) and [`tools/reextract_sp500_chunks.py`](../../tools/reextract_sp500_chunks.py).
3. **Form-Filtered Hybrid Retrieval**:
   - Add optional `allowed_form_types: Optional[Sequence[str]] = None` and `allowed_graph_layers: Optional[Sequence[str]] = None` filters to `search_chunks_hybrid_rrf` (and its offline Parquet equivalent used by the evaluation harness) so every retrieval arm evaluates exact LOFO chunk slices deterministically.

### 4.2 Three-Axis Controlled Experimental Matrix

All experiments operate over a **fixed, stratified multi-sector S&P 500 benchmark cohort $\mathcal{C}_{11}$** covering all 11 GICS sectors (at least 2–3 constituents per sector, e.g., $22\text{–}33$ issuers over a fixed multi-year window anchored at a single `PiTContext(as_of=t^*, mode="historical")` per [`ADR-00033`](ADR-00033-bitemporal-pit-contract-for-read-paths--.md)):

1. **Axis 1 — SEC Form-Inclusion Ladder & Leave-One-Form-Out (LOFO) Ablation**:
   - **Additive Inclusion Ladder**:
     - `F0`: `10-K` (`Item 1` + `Item 1A` + `Exhibit 21` — pre-Phase-2a baseline)
     - `F1`: `F0` + Phase 2a `10-K` sections (`Item 1C`, `Item 2`, `Item 3`, `Item 7`)
     - `F2`: `F1` + `8-K` (primary text only)
     - `F3`: `F2` + `8-K` `Exhibit 99.1` press releases
     - `F4`: `F3` + part-qualified `10-Q` + amended filings (`10-K/A`, `10-Q/A`, `8-K/A`)
     - `F5+`: Incremental Phase 2b/2c/2d additions (`Exhibit 10`, XBRL `companyfacts`, `Form 10`/`S-4`/`20-F`/`6-K`, `DEF 14A`, `SD`) as each lands
     - `F_stress`: `F4` + `OPTIONAL_ABLATION` forms (`4`, `13F-HR`, `SC 13D/13G`, `424B2`, `S-3`)
   - **Exact Leave-One-Form-Out Delta**: For any active form set $\mathcal{F}$ and member $f \in \mathcal{F}$, compute:

     $$\Delta M(f) = M(\mathcal{F}) - M(\mathcal{F} \setminus \{f\})$$

     over a single deterministic filtered view of the persisted snapshot (`nodes.parquet`, `edges.parquet`, `chunks.parquet`), isolating the pure structural and retrieval contribution of form $f$.

2. **Axis 2 — Graph-Layer Isolation (`FLAT` vs. `LAYERED`)**:
   - **`FLAT`**: All ingested forms share one undifferentiated graph during community detection and cascade traversal.
   - **`LAYERED`**: Cascade propagation and structural communities restrict strictly to `sec_graph_layer == "DOMAIN1_OPERATIONAL"`, isolating `CAPITAL_OWNERSHIP`, `FINANCING_OFFERING`, and `GOVERNANCE_INTERLOCK` into dedicated partitions.

3. **Axis 3 — Embedder & Second-Stage Reranker Scaling Arms (`E1` vs. `E2` vs. `E2b` vs. `E3`)**:

| Arm | First-Stage Bi-Encoder | Vector Dim | Second-Stage Reranker | Local 8 GB Hardware Posture | Purpose in Study |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **`E1` (Baseline)** | `qwen3-embedding:0.6b` | `1024` (native) | None (Hybrid RRF, $k=60$) | `0 MB` live VRAM (`num_gpu: 0`) | Production baseline ([`ADR-00035`](ADR-00035-text-unit-chunk-layer-and-evidence-links.md)) |
| **`E2` (MRL Larger Bi-Encoder)** | `qwen3-embedding:4b` | `1024` (MRL slice + L2 norm) | None (Hybrid RRF, $k=60$) | `0 MB` live VRAM (`num_gpu: 0`) / ~`2.5 GB` (`Q4_K_M`) offline `batch_gpu` | Tests $6.7\times$ bi-encoder scale at constant `VECTOR(1024)` |
| **`E2b` (Offline Diagnostic Ceiling)** | `qwen3-embedding:8b` (`Q4_K_M`) | `1024` (MRL slice + L2 norm) | None (Hybrid RRF, $k=60$) | Offline-only (~`4.7 GB` VRAM via `unload_ollama_model()` + `batch_gpu=True`); **non-deployable live** | Zero-shot upper-bound diagnostic ruler isolating bi-encoder capacity vs. corpus/chunking bottlenecks (no fine-tuning implied) |
| **`E3` (Two-Stage Compact Reranker)** | `qwen3-embedding:0.6b` | `1024` (native) | Compact Cross-Encoder (`RerankerPort`, default `Qwen/Qwen3-Reranker-0.6B`, top-$M=40 \rightarrow K=10$) | `0 MB` live VRAM (`num_gpu: 0`) | Tests second-stage cross-attention vs. heavier bi-encoder under form dilution |

   Cross all configured arms (`--arms E1,E2,E3` by default; `--arms E1,E2,E2b,E3` when measuring the ceiling) against both a clean operational index (`F4`) and a form-diluted stress index (`F4 + F_stress`) locally on 8 GB VRAM (zero neocloud required). Comparing `E1` / `E2` / `E3` against the `E2b` (`8B`) ceiling distinguishes three concrete regimes:
   - *Flat ceiling (`E1` $\approx$ `E2` $\approx$ `E2b`)*: Bi-encoder capacity is not the bottleneck; prioritize form curation, chunking, or `E3` cross-attention.
   - *Saturating at `4B` (`E1` $<$ `E2` $\approx$ `E2b`)*: `4B` captures nearly all dense representation headroom.
   - *Large uncaptured headroom (`E1` $<$ `E2` $\ll$ `E2b`, not closed by `E3`)*: Provides the sole empirical trigger for considering a *future* ADR on offline contrastive distillation of `0.6B`.

### 4.3 Four-Pillar Quantitative KG Quality Scorecard

Every `(form_subset, graph_isolation, embedder_arm)` cell is evaluated deterministically across four orthogonal metric pillars:

1. **Pillar 1 — Topology & Community Health** (over PiT structural graph $G_t = (V, E_t)$ filtered by `filter_structural_edges_pit` in [`knowledge_graph/communities.py`](../../knowledge_graph/communities.py)):
   - **Weighted Leiden Modularity $Q$**:
     $$Q = \frac{1}{2m} \sum_{i,j \in V} \left[A_{ij} - \frac{k_i k_j}{2m}\right] \mathbf{1}[c_i = c_j]$$
     where $A_{ij}$ is the undirected edge weight (`share`, default `1.0`), $k_i = \sum_j A_{ij}$, $2m = \sum_{i,j} A_{ij}$, and $c_i$ is node $i$'s community assignment from `build_communities` ([`ADR-00037`](ADR-00037-pit-community-layer-and-impact-ranking--.md)).
   - **Hub-Collapse & Degree Skew**: Max degree $k_{\max}$, top-1% degree Gini coefficient $G_k$, and mean 2-hop ego-graph edge count $\bar{e}_2$ relative to `MAX_SUBGRAPH_EDGES = 25` ([`retrieval/pipeline.py`](../../retrieval/pipeline.py)).
   - **Cascade Context Truncation Loss**: Mean and P95 `dropped_impact_share` across canonical shock origins.
   - **Structural Coverage**: Active operational edges per S&P 500 issuer, largest connected component (LCC) share, and singleton-node rate.
2. **Pillar 2 — Epistemic Provenance Grounding**:
   - Distribution of downstream cascade reports across [`ProvenanceTier`](../../retrieval/reporting.py): `FILING_GROUNDED` ($\le 0.10$), `PARTIALLY_ASSUMED` ($0.10 < s < 0.50$), and `STRESS_UPPER_BOUND` ($\ge 0.50$).
   - **Evidence Link Integrity**: Fraction of active edges with `evidence_status == "chunked"` whose `evidence_chunk_ids` resolve cleanly in `chunks.parquet` / `document_chunks`.
3. **Pillar 3 — Extraction & Reasoning Fidelity**:
   - [`RuntimeRejectionQueue`](../../sft_pipeline/rejection_queue.py) quarantine rate stratified by `(form_type, ErrorStage)` (`ONTOLOGY_PRUNING`, `EXTRACTION_PARSE_ERROR`, `GLEANING_PARSE_FAILED`, `RUNAWAY_LOOP`).
   - Holdout ontology predicate rejection rate ([`ADR-00031`](ADR-00031-iterative-predicate-convergence-protocol.md)).
   - Optional 4-axis Phase-5 torture accuracy (`Verdict`, `HopTrace`, `Provenance`, `CIK` via [`sft_pipeline/evaluation/phase5_torture_evaluators.py`](../../sft_pipeline/evaluation/phase5_torture_evaluators.py)) when comparing downstream neural checkpoints ([`ADR-00012`](ADR-00012-qwen-model-upgrade-and-fine-tuning-plan-.md)).
4. **Pillar 4 — Retrieval Precision & Systems Envelope** (over a fixed gold chunk-cited query suite $\mathcal{Q}_{\text{gold}}$):
   - **Ranking Quality**: $\text{Recall@5}$, $\text{Recall@10}$, $\text{MRR@10}$, and $\text{nDCG@10}$ with **paired bootstrap 95% confidence intervals** ($B = 1{,}000$ resamples over queries, seeded deterministically).
   - **Systems Budget**: Live CPU-pinned query latency (`P50` / `P95` ms at `num_gpu: 0`), offline batch throughput (`chunks/s` at `batch_gpu=True`), peak host RAM and GPU VRAM (`MB`), total chunk count, and `pgvector` HNSW index size.

### 4.4 MRL Embedder Extension & Pluggable Reranker Protocol

1. **Matryoshka Truncation in [`embedding/embedder.py`](../../embedding/embedder.py)**:
   - Support opt-in MRL projection (`EMBEDDER_MRL_TRUNCATE=1` or explicit `target_dim: Optional[int]`): given a raw model vector $v \in \mathbb{R}^{D}$ with $D > d = \text{EMBEDDING\_DIM}$, compute:
     $$\hat{v}_{1:d} = \frac{v_{1:d}}{\|v_{1:d}\|_2}$$
     before running `validate_embedding_vectors` ([`ADR-00020`](ADR-00020-fail-fast-embedder-and-zero-vector-guard.md)), rejecting any degenerate zero-norm prefix or $D < d$ mismatch.
   - Pass Ollama `dimensions` when supported while enforcing client-side truncation + L2 re-normalization as the deterministic source of truth.
2. **Pluggable Cross-Encoder Reranker ([`embedding/reranker.py`](../../embedding/reranker.py))**:
   - Define `RerankerPort` (`Protocol`) with `score_pairs(query: str, passages: Sequence[str]) -> List[float]`, plus:
     - `DeterministicMockReranker` (for offline CI / unit tests under `YARNBALL_RERANKER_MOCK=1`),
     - `LocalCrossEncoderReranker` (lazy `sentence_transformers.CrossEncoder` / local HuggingFace reranker defaulting to `Qwen/Qwen3-Reranker-0.6B`, configurable via `RERANKER_MODEL` to alternatives such as `BAAI/bge-reranker-v2-m3`, CPU-pinned by default for live serving).
   - Wire optional `reranker: Optional[RerankerPort] = None` and `rerank_top_m: int = 40` into `search_chunks_hybrid_rrf` ([`knowledge_graph/chunk_store.py`](../../knowledge_graph/chunk_store.py)) with zero behavior change when `reranker is None`.

### 4.5 Offline Evaluation Harness & Artifacts

Implement a zero-database-capable offline harness ([`knowledge_graph/ablation.py`](../../knowledge_graph/ablation.py) + CLI [`tools/eval_kg_form_ablation.py`](../../tools/eval_kg_form_ablation.py)):

- Consumes a persisted Parquet snapshot directory (`nodes.parquet`, `edges.parquet`, `chunks.parquet`), an optional rejection-queue JSONL directory, and a gold chunk-cited query JSONL file (`queries_gold.jsonl`).
- Computes all four scorecard pillars across the requested form subsets (`--ladder`, `--lofo`), graph-isolation modes (`--layers flat,layered`), and embedder/reranker arms (`--arms E1,E2,E3` or `--arms E1,E2,E2b,E3`, plus `--mock` for CI).
- Emits four deterministic artifacts under `data/eval/adr00047/<run_id>/`:
  1. `ablation_scorecard.json` (full structured cell metrics, LOFO $\Delta M(f)$, paired bootstrap 95% CIs, and gate verdicts),
  2. `ablation_report.md` (concise Markdown summary tables),
  3. `lofo_delta_heatmap.png` (marginal $\Delta M(f)$ across SEC forms $\times$ quality pillars),
  4. `embedder_recall_vs_latency.png` & `modularity_vs_dropped_impact.png` (Pareto frontier plots).

### 4.6 Pre-Registered Empirical Promotion Gates

1. **SEC Form Promotion Gate (`PLANNED` / `OPTIONAL_ABLATION` $\rightarrow$ Default `DOMAIN1_OPERATIONAL`)**:
   A form $f$ is promoted into default operational batch harvesting **if and only if**:
   - It produces a positive gain ($\Delta > 0$) on at least one of: operational edge coverage per issuer, `FILING_GROUNDED` cascade share, or retrieval `Recall@10` / `MRR@10`; **AND**
   - It does **not** reduce Leiden modularity $Q$ by $> 0.03$, does **not** increase mean `dropped_impact_share` by $> +0.02$ (2 percentage points), and does **not** raise per-form `RuntimeRejectionQueue` quarantine rate by $> +0.05$.
   Forms that add signal only when isolated remain partitioned in their non-operational `SECGraphLayer`; forms failing both gates remain `OPTIONAL_ABLATION`.
2. **Embedder / Reranker Promotion Gate (Replacing `E1` Production Default)**:
   A live-deployable arm (`E2` `4B` MRL or `E3` `0.6B` + reranker; `E2b` `8B` is strictly a non-deployable offline diagnostic ceiling) replaces `E1` as default **if and only if**:
   - Paired bootstrap 95% CI on $\Delta\text{Recall@10}$ or $\Delta\text{nDCG@10}$ is strictly $> 0$ with practical lift $\ge +0.03$ on $\mathcal{Q}_{\text{gold}}$; **AND**
   - Live CPU-pinned (`num_gpu: 0`) query latency remains $\le 250\text{ ms}$ P95 with `0 MB` GPU VRAM eviction against `yarnball-qwen` (7B/9B).

---

## 5. Invariants & Guardrails

1. **Zero-Variance LOFO Slicing**: Leave-One-Form-Out ablations must derive every subset $\mathcal{F} \setminus \{f\}$ deterministically from a single canonical persisted snapshot tagged with normalized `form_type` and `sec_graph_layer`—never by re-running stochastic LLM extraction across cells.
2. **Strict Point-in-Time Consistency**: All topology, community, cascade, and retrieval metrics in a single study run must share one immutable `PiTContext(as_of=t^*, mode="historical")` ([`ADR-00033`](ADR-00033-bitemporal-pit-contract-for-read-paths--.md)).
3. **MRL Unit-Norm & Dimensional Parity**: Any MRL-truncated vector must be L2 re-normalized after slicing to `get_embedding_dim()` and validated through `validate_embedding_vectors` ([`ADR-00020`](ADR-00020-fail-fast-embedder-and-zero-vector-guard.md)).
4. **Zero-Eviction Live Serving Guarantee**: `E1`, `E2`, and `E3` live-serving benchmarks must report both CPU-pinned (`num_gpu: 0`) latency and peak GPU VRAM, prohibiting production defaults that trigger Ollama model swaps against `yarnball-qwen` inside 8 GB VRAM.
5. **No Unmeasured Promotion Claims**: Updating `default_batch_harvest` in [`ingestion/sec/canon.py`](../../ingestion/sec/canon.py) or `_DEFAULT_MODEL` in [`embedding/embedder.py`](../../embedding/embedder.py) based on this study requires referencing a committed `ablation_scorecard.json` run ID that passes Section 4.6's promotion gates.

---

## 6. Verification Plan

1. **New Offline Ablation & Embedder Unit/Integration Suite** ([`tests/knowledge_graph/test_kg_form_ablation_adr00047.py`](../../tests/knowledge_graph/test_kg_form_ablation_adr00047.py)):
   - **Chunk Provenance**: Verify `DocumentChunkRecord` stamps normalized `form_type` and auto-derived `sec_graph_layer` into both `chunks.parquet` and `store_document_chunks`, and that `search_chunks_hybrid_rrf` filters accurately by `allowed_form_types` / `allowed_graph_layers`.
   - **LOFO & Hub-Collapse Detection**: Construct a deterministic synthetic multi-form snapshot (`10-K`, `8-K`, `10-Q`, plus a synthetic `13F-HR` institutional-holder hub connecting all issuers) and assert:
     - `FLAT` ingestion measurably drops Leiden modularity $Q$, spikes $k_{\max}$ / degree Gini $G_k$, and raises `dropped_impact_share`;
     - `LAYERED` isolation and LOFO removal of `13F-HR` restore `DOMAIN1_OPERATIONAL` modularity and low truncation loss;
     - Exact identity $\Delta M(f) = M(\mathcal{F}) - M(\mathcal{F} \setminus \{f\})$ holds across all metrics.
   - **MRL Truncation & Validation**: Verify `2560`-d $\rightarrow$ `1024`-d MRL projection preserves unit L2 norm, matches `get_embedding_dim()`, and rejects zero-norm prefixes or $D < d$ vectors via `InvalidEmbeddingError`.
   - **Reranker Hook & Bootstrap CIs**: Verify `RerankerPort` cleanly reorders top-$M$ Hybrid RRF candidates, leaves default `reranker=None` output byte-for-byte unchanged, and computes exact `Recall@K`, `MRR@10`, `nDCG@10`, and seeded paired bootstrap 95% CIs.
   - **CLI Smoke Test**: Run `tools/eval_kg_form_ablation.py --mock` on a temporary Parquet snapshot and verify `ablation_scorecard.json`, `ablation_report.md`, and PNG plot artifacts render without error.
2. **Full-Corpus Static & Regression Gate**:
   - `.venv/bin/mypy` (0 errors across full corpus + new modules/tests).
   - `.venv/bin/pytest --ignore=tests/knowledge_graph/test_embedding_store.py -v` and `.venv/bin/pytest tests/docs/test_adr_integrity.py`.
   - `.venv/bin/graphify update .`.

---

## 7. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Gives YarnBall a rigorous, quantitative feedback loop showing *which* SEC forms improve the Knowledge Graph and *which* degrade topology—directly guiding Phase 2b/2c/2d prioritization.
  - Evaluates `qwen3-embedding:0.6b` vs. `4B` MRL (`1024`-d) vs. `0.6B` + compact cross-encoder reranker (`Qwen/Qwen3-Reranker-0.6B`) against an optional offline `8B` (`Q4_K_M`) MRL ceiling locally on 8 GB VRAM—zero neocloud spend, zero `VECTOR(1024)` schema churn, and zero generative-model VRAM eviction.
  - Keeps neural generator alignment ([`ADR-00012`](ADR-00012-qwen-model-upgrade-and-fine-tuning-plan-.md)) and downstream financial market backtesting ([`ADR-00038`](ADR-00038-event-study-backtest-and-signal-decay---.md)) cleanly decoupled from KG data-substrate and retrieval ablation.
- **Negative / Trade-offs**:
  - Adds two nullable columns (`form_type`, `sec_graph_layer`) to `document_chunks` and `chunks.parquet` (fully backward-compatible for existing rows via `NULL` defaults).
  - Running live multi-arm benchmarks over a 22–33 issuer multi-year S&P 500 cohort requires curating a gold chunk-cited query file (`queries_gold.jsonl`) and running offline GPU batch embedding (`batch_gpu=True`) per arm.
