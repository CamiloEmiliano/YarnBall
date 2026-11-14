# ADR-00028: Cross-Source Entity Linking & Live-to-Snapshot Graph Sync

- **ADR ID**: ADR-00028
- **Filename**: `ADR-00028-cross-source-entity-linking-and-kg-sync-.md`
- **Date**: 2026-11-14 19:25 UTC
- **Status**: PROPOSED
- **Target Subsystem**: `knowledge_graph` / `ingestion` / `explorer`
- **Linked Commits**: Pending acceptance

---

## 1. Context & Problem Statement

End-to-end inspection of [`ingestion/sec/harvester.py`](../../ingestion/sec/harvester.py), [`knowledge_graph/graph_store.py`](../../knowledge_graph/graph_store.py), [`knowledge_graph/entity_resolver.py`](../../knowledge_graph/entity_resolver.py), [`knowledge_graph/snapshot_manager.py`](../../knowledge_graph/snapshot_manager.py), and [`explorer/workbench_api.py`](../../explorer/workbench_api.py) highlighted three cross-pipeline disconnects:

1. **Asymmetric Entity Linking & Provenance Across SEC vs. News**: `EdgarWorker` resolves company strings through `self.linker.link_entity(...)` and stamps edges with `fiscal_year`, `valid_from`, `is_current`, `form_type`, `section`, and `accession_number`. By contrast, `store_graph_entities` (`graph_store.py`) merges raw LLM surface strings (`"Apple"`, `"TSMC"`) directly into Memgraph without canonical entity linking and stamps only `r.source_hash`—omitting `source_type` and `published_at`/`valid_from`.
2. **Offline-Only `EntityResolver`**: [`knowledge_graph/entity_resolver.py`](../../knowledge_graph/entity_resolver.py) is not wired into the live news/SEC ingestion loop, allowing split nodes (`"Apple"` vs. `"Apple Inc."`) to accumulate until manual resolution.
3. **Dashboard Visibility Lag (Parquet-Only Workbench)**: `workbench_api.py` (`_resolve_graph_source`) loads exclusively from static Parquet snapshots (`data/snapshots/*`) or CSV/JSON edge manifests. Live writes into Memgraph by `EdgarWorker` or `memgraph_consumer` never appear in **PiT KG Explorer** or **Event Audit** until `SnapshotManager.export_snapshot(...)` is run explicitly, and no CLI step currently runs `EdgarWorker` + `SnapshotManager` over `data/sec_filings/`.

---

## 2. Decision Drivers & Constraints

1. **Canonical Node Identity Across Modalities**: A news article mentioning `"Apple"` or `"TSMC"` must merge onto the same canonical Memgraph node (`id` / `ticker` / `cik`) created by SEC Form 10-K/8-K ingestion.
2. **Statutory vs. Media Provenance Separation**: Memgraph edges must carry explicit `source_type` (`"sec_filing"` vs. `"news"`) and PiT timestamps (`valid_from` / `published_at`) so retrieval and cascade queries can weight, audit, or exclude uncorroborated news edges cleanly.
3. **Reproducible Read-Only Dashboard Freshness**: Analysts using `tools/serve_explorer.py` must be able to export freshly built Memgraph graphs into `data/snapshots/` automatically via one reproducible pipeline step (preserving offline Parquet fast reads).

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Keep News Edges Unlinked Until Offline Batch `EntityResolver`**:
  - *Rationale for Rejection*: Breaks real-time hybrid retrieval (`HybridRetriever`) and causes multi-hop SEC+news traversals to miss links between `"TSMC"` and `"Taiwan Semiconductor Manufacturing Company"`.
- **Alternative B: Replace Parquet Snapshots Entirely With Mandatory Live Memgraph Queries in Workbench**:
  - *Rationale for Rejection*: Violates [`ADR-00015`](ADR-00015-dynamic-in-memory-offline-subgraph-store.md) and [`ADR-00017`](ADR-00017-local-fastapi-kg-contagion-workbench-app.md) offline zero-DB workbench guarantees; automated snapshot export preserves offline speed while keeping dashboard graphs current.

---

## 4. Decision Outcome

Establish **Inline Cross-Source Entity Linking & Deterministic Snapshot Sync**:

1. **Entity Linking + Temporal/Source Stamping in `store_graph_entities`**:
   - Route extracted news entities (`Company` / organization nodes and edge endpoints) in [`knowledge_graph/graph_store.py`](../../knowledge_graph/graph_store.py) through canonical `EntityResolver` / `EdgarEntityLinker` lookup before `MERGE`.
   - Accept optional `published_at` and `source_type` (`"news"` default for `memgraph_consumer`), stamping `r.source_type`, `r.valid_from`, `r.is_current = true`, and `n.published_at` on `:News` nodes and derived edges.
2. **Reproducible Build-and-Export Snapshot Command**:
   - Provide a single CLI entrypoint (e.g., `tools/build_kg_snapshot.py` or a flag on ingestion workers) that ingests local `data/sec_filings/` (+ optional staged news), runs `EntityResolver`, and calls `SnapshotManager.export_snapshot(...)` to publish timestamped snapshots under `data/snapshots/`, immediately discoverable by `GET /api/catalog`.

---

## 5. Invariants & Guardrails

1. **Mandatory Source Provenance on Edges**: Every Memgraph edge MUST carry `source_type` (`"sec_filing"` or `"news"`) and a temporal anchor (`valid_from` or explicit `null` flagged as unanchored).
2. **Offline Workbench Preservation**: `explorer/workbench_api.py` MUST continue functioning cleanly with zero running database containers by reading exported `data/snapshots/*` Parquet tables.
3. **Zero Absolute Paths Invariant**: All references remain strictly repository-relative (`../../<path>`).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Connects news and SEC supply-chain assertions onto shared canonical entities without sacrificing provenance separation.
  - Bridges the gap between local `data/sec_filings/` files on disk and visible subgraphs inside **PiT KG Explorer**.
- **Negative / Trade-offs**:
  - Adds a lightweight entity-linker cache lookup per extracted node/edge during `memgraph_consumer` throughput.
