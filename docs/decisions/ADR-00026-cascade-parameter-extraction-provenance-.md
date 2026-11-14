# ADR-00026: Supply-Chain Cascade Parameter Extraction & Empirical Bounds

- **ADR ID**: ADR-00026
- **Filename**: `ADR-00026-cascade-parameter-extraction-provenance-.md`
- **Date**: 2026-11-14 19:25 UTC
- **Status**: PROPOSED
- **Target Subsystem**: `contagion` / `retrieval` / `ingestion` / `explorer`
- **Linked Commits**: Pending acceptance

---

## 1. Context & Problem Statement

Inspecting [`contagion/cascade.py`](../../contagion/cascade.py), [`retrieval/offline_store.py`](../../retrieval/offline_store.py), [`retrieval/pipeline.py`](../../retrieval/pipeline.py), [`ingestion/sec/harvester.py`](../../ingestion/sec/harvester.py), [`knowledge_graph/graph_store.py`](../../knowledge_graph/graph_store.py), and `data/snapshots/G_resolved/edges.parquet` showed an acute gap between cascade physics inputs and KG generation outputs:

- `SparseNetworkCascadeEngine` relies on per-edge operational parameters (`share` $\in [0, 1]$, `buffer_days` $\ge 0$, `input_group`).
- `_build_cascade_graph` in `retrieval/pipeline.py` falls back silently to worst-case defaults (`share=1.0`, `buffer_days=0.0`, `input_group="default"`) whenever those attributes are absent.
- Neither `EdgarWorker`, `graph_store.py`, nor `data/snapshots/G_resolved/edges.parquet` currently extracts or populates `share`, `buffer_days`, or `input_group`. Consequently, any real generated KG behaves identically to a `100%` unbuffered single-source dependency across every edge, and [`explorer/`](../../explorer/workbench_api.py) cannot distinguish empirically measured concentration from fallback defaults.

---

## 2. Decision Drivers & Constraints

1. **Distinguish Extracted Evidence From Structural Defaults**: Downstream reports ([`retrieval/reporting.py`](../../retrieval/reporting.py)) and UI provenance panels must explicitly label whether `share` / `buffer_days` were extracted from disclosures or defaulted.
2. **Preserve Fail-Closed Safety Bound**: When an SEC filing confirms sole-source dependence without disclosing exact inventory days, conservative defaults must remain explicit and auditable rather than inventing optimistic buffers.
3. **Consistent Edge Schema Across Memgraph, Parquet & JSONL**: Parameter fields and provenance tags must flow cleanly from extraction $\rightarrow$ `SnapshotManager` Parquet exports $\rightarrow$ `InMemorySubgraphStore` $\rightarrow$ `TopologicalEdgeDescriptor`.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Keep Silent `share=1.0`, `buffer_days=0.0` Fallback Without Provenance Flagging**:
  - *Rationale for Rejection*: Misleads analysts inspecting **Event Audit** or **PiT KG Explorer** into believing uncalibrated default assumptions represent disclosed SEC metrics.
- **Alternative B: Refuse Cascade Execution Unless All Edges Carry Quantitative Disclosures**:
  - *Rationale for Rejection*: Most 10-K supplier mentions omit exact inventory days; refusing traversal altogether blocks qualitative topology inspection across real multi-hop graphs.

---

## 4. Decision Outcome

Introduce **Cascade Parameter Extraction & Provenance Tagging** across extraction prompts, [`retrieval/offline_store.py`](../../retrieval/offline_store.py), [`retrieval/pipeline.py`](../../retrieval/pipeline.py), [`retrieval/reporting.py`](../../retrieval/reporting.py), and the [`explorer/`](../../explorer/static/app.js) provenance inspector:

1. **Extraction Prompt & Schema Support**:
   - Extend SEC/news extraction schemas to capture quantitative operational cues when explicitly stated in filings (`share` from GAAP revenue/procurement share disclosures or `"sole source"` $\rightarrow$ `1.0`, `buffer_days` from inventory/weeks-of-supply text, `input_group` / `dependency_domain` from product/component category).
2. **Explicit `param_provenance` Metadata**:
   - Attach `param_provenance` (e.g., `"EXTRACTED_QUANTITATIVE"`, `"EXTRACTED_QUALITATIVE_SOLE_SOURCE"`, `"DEFAULT_CONSERVATIVE_BOUND"`) on normalized edge records in `InMemorySubgraphStore` and `TopologicalEdgeDescriptor`.
3. **Sensitivity / Default Flagging in Reports & Workbench UI**:
   - Surface `param_provenance` in `OperationalContagionReport` telemetry and in the **PiT KG Explorer** Edge Provenance panel so users see immediately whether `share=100%` / `buffer=0d` reflects verified filing text or an uncalibrated default.

---

## 5. Invariants & Guardrails

1. **No Unmarked Default Fabrication**: Missing quantitative parameters MUST carry explicit default provenance (`DEFAULT_CONSERVATIVE_BOUND`) and never masquerade as filing-extracted metrics.
2. **Strict Domain 1 Operational Scope**: `share` and `buffer_days` model physical procurement concentration and inventory buffer duration only (consistent with [`ADR-00006`](ADR-00006-phase-7-sparse-network-cascade-engine---.md) and [`ADR-00010`](ADR-00010-directional-basket-peer-cluster-mapping-.md)).
3. **Zero Absolute Paths Invariant**: All references remain strictly repository-relative (`../../<path>`).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Makes worst-case cascade defaults transparent across CLI dry runs, cohort exports, and the Workbench dashboard.
  - Creates a clean integration contract for quantitative 10-K Item 1/Item 7 and 8-K extraction improvements.
- **Negative / Trade-offs**:
  - Adds optional provenance fields to `TopologicalEdgeDescriptor` and Workbench edge tables, requiring backward-compatible defaults when reading legacy snapshots like `G_resolved`.
