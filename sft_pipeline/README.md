# `sft_pipeline`: Synthetic SFT Generation & Manifold Balancing Engine

The `sft_pipeline` package orchestrates the end-to-end generation, 5-axis annotation, manifold balancing, and multi-task formatting of instruction datasets for the YarnBall financial intelligence models (`Qwen2.5-7B-Instruct`).

---

## 1. Architectural Mission

Standard LLM fine-tuning datasets frequently suffer from **ambient saturation** (excessive routine boilerplate) and **catastrophic hallucinations** on boundary concepts. `sft_pipeline` addresses this by:
1. **Targeting the Economic Manifold ($M^*$)**: Stratified sampling across all 11 GICS sectors, ensuring minimum representation floors for sparse risk relations (`DEFAULTED_ON`, `SOLE_SOURCE_DEPENDENT_ON`, `LICENSES_FROM`).
2. **Boundary-Proximity Hard Negative Mining**: Synthesizing co-occurring non-relational entity pairs with strictly empty completion targets `(none)`.
3. **Formal Ontological Grounding**: Enforcing 5-axis relational reification and OpenCypher DSL targets via `ontology/`.
4. **Curation Engine Feedback Loop**: Interfacing directly with `curation/` for MinHash/Coreset cold-start filtering and Dataset Cartography active learning steering.

---

## 2. Package Architecture

```
sft_pipeline/
├── __init__.py      # Clean public API exports
├── README.md        # Formal package documentation
├── sampler.py       # ManifoldTargetedSampler, ManifoldSample, class balancing & active learning steering
├── annotator.py     # FinancialTaxonomyAnnotator (5-axis triple grounder and DSL compiler adapter)
├── exporter.py      # SFTDatasetExporter (Tasks A-E formatting, zero-leakage 80/10/10 split partitioning)
├── builder.py       # FullSFTDatasetBuilder (end-to-end pipeline with Cold-Start & Active Learning hooks)
└── hf_sync.py       # upload_to_huggingface hub synchronization utility
```

---

## 3. Communication with `curation`

The pipeline bridges synthetic generation with formal dataset selection through two hooks:

### A. Pre-Training Cold-Start Curation
```python
from sft_pipeline import FullSFTDatasetBuilder

builder = FullSFTDatasetBuilder(enable_cold_start_curation=True)
summary = builder.build_dataset()
```
- Ingests raw candidate samples.
- Executes `curation.cold_start.pipeline.ColdStartCurationPipeline`:
  1. **MinHash LSH**: Prunes near-duplicate legal boilerplates.
  2. **Facility Location Coreset**: Maximizes geometric coverage across diverse features.
- Exports the deduplicated, geometrically representative curriculum.

### B. Post-Training Active Learning Steering
```python
from sft_pipeline import FullSFTDatasetBuilder

builder = FullSFTDatasetBuilder()
summary = builder.build_dataset(
    active_learning_partition="data/cartography/active_learning_partition.json"
)
```
- Filters training instances based on Swayamdipta Dataset Cartography coordinates (`mu`, `sigma`, `F`):
  - **Ambiguous** instances (high variability) are 100% retained.
  - **Hard** instances (mislabeled) are quarantined and discarded.
- Automatically steers `ManifoldTargetedSampler.steer_from_cartography` to boost sampling targets for deficient relation manifolds.

---

## 4. Multi-Task Curriculum (Tasks A through E)

- **Task A (`<|extract_sec_graph|>`)**: SEC Form 10-K / 10-Q Item 1 & Exhibit 21 OpenCypher Triples DSL extraction.
- **Task B (`<|extract_news_event|>`)**: Breaking news events mapped to directional 5-axis property graph edges.
- **Task C (`<|text_to_cypher|>`)**: Natural language financial questions mapped to valid Memgraph Cypher queries.
- **Task D (`<|contagion_reasoning|>`)**: Multi-hop shock propagation through supply chains with `<think>` Chain-of-Thought.
- **Task E (`<|portfolio_recommendation|>`)**: Factor exposure hedging and allocation rebalancing with `<think>` Chain-of-Thought.
