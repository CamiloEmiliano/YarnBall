# ADR-00015: Dynamic In-Memory Offline Subgraph Store

- **ADR ID**: ADR-00015
- **Filename**: `ADR-00015-dynamic-in-memory-offline-subgraph-store.md`
- **Date**: 2026-11-13 07:15 UTC
- **Status**: ACCEPTED
- **Target Subsystem**: `retrieval` / `knowledge_graph` / `tools`
- **Linked Commits**: [`8bdda14`](../../retrieval/offline_store.py), [`8bdda14`](../../tools/dry_run_sec_8k_contagion.py)

---

## 1. Context & Problem Statement

[`tools/dry_run_sec_8k_contagion.py`](../../tools/dry_run_sec_8k_contagion.py) currently embeds a minimal `OfflineGraphStore` stub whose `session().run()` ignores root ticker constraints, hop bounds, relationship types, lifecycle status, and Point-in-Time (`as_of_date`) filters injected by [`TemporalQueryFilter`](../../retrieval/temporal.py), returning every stored edge on any `"MATCH"` query. Furthermore, when `graph_driver` is omitted in `run_sec_8k_dry_run`, topology resolution falls back to hardcoded `CRUS` / `FABCO` mock edge lists or a synthetic single-edge stub to `AAPL`.

For offline multi-year event cohorts (2018–2026) exported to downstream econometric workflows ([`ADR-00013`](ADR-00013-contagion-event-time-series-join-schema-.md)), the pipeline needs a reusable in-memory graph provider capable of:

1. **Parquet Snapshot & Manifest Loading**: Restoring real multi-tier company networks directly from [`SnapshotManager`](../../knowledge_graph/snapshot_manager.py) exports (`nodes.parquet`, `edges.parquet` in `data/snapshots/*`) or portable JSON/JSONL edge manifests without requiring a running Memgraph/Neo4j container.
2. **Root-Scoped Multi-Hop Traversal**: Executing deterministic $k$-hop downstream BFS scoped to the queried origin ticker and relationship type (`SUPPLIES_TO`), computing accurate hop distances instead of leaking unrelated graph edges.
3. **Point-in-Time (PiT) Contract Filtering**: Evaluating `status` and `[valid_from, valid_to]` edge validity against `as_of_date` so future or already-terminated supply contracts never leak into historical event windows.

---

## 2. Decision Drivers & Constraints

1. **Zero Database Infrastructure Dependency**:
   - Offline batch evaluation and CLI dry runs must execute strictly in-memory against local Parquet snapshots or JSON/JSONL manifests with zero running Memgraph/Neo4j sockets.
2. **Deterministic Multi-Hop & Relationship Scoping**:
   - `MATCH (s:Company {ticker: '...'})-[r:SUPPLIES_TO*1..k]->(t:Company)` queries must execute deterministic sorted BFS downstream traversal up to `k` hops and tag each edge with its exact shortest hop distance.
3. **Strict Point-in-Time & Lifecycle Parity**:
   - Query evaluation must respect `as_of_date` (via `$as_of_date` parameter or inline `'YYYY-MM-DD'` literals injected by [`TemporalQueryFilter`](../../retrieval/temporal.py)) against `status`, `valid_from`, and `valid_to`.
4. **Compatibility with `SnapshotManager` Parquet Schemas**:
   - Must load `nodes.parquet` (`id`, `label`, `ticker`, `source_hash`, `properties_json`) and `edges.parquet` (`source_id`, `target_id`, `rel_type`, `source_hash`, `properties_json`) directly into normalized edge records consumable by [`GraphRAGPipelineOrchestrator`](../../retrieval/pipeline.py).
5. **Backward Compatibility with Canonical Dry-Run Smoke Fixtures**:
   - Existing callers passing `custom_network_edges` or running canonical fixtures (`crus_8k_disruption.json`, `aapl_8k_disruption.json`) without external snapshots must continue functioning unchanged.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Require a Live Memgraph Container for All Dry Runs & Backtests**:
  - *Rationale for Rejection*: Breaks zero-infrastructure offline execution, slows batch 2018–2026 historical sweeps with container startup/network round-trips, and cannot run inside network-isolated sandboxes.
- **Alternative B: Embed an External Third-Party Cypher Runtime**:
  - *Rationale for Rejection*: Unnecessary dependency footprint when [`GraphRAGPipelineOrchestrator`](../../retrieval/pipeline.py) and [`CypherASTWhitelistGuardrail`](../../retrieval/guardrails/schema_guardrail.py) constrain retrieval to a compact, deterministic family of read-only `MATCH` path patterns.
- **Alternative C: Keep Global Unfiltered Edge Dump in `OfflineGraphStore`**:
  - *Rationale for Rejection*: Leaks unrelated issuers and post-`as_of_date` edges into [`SparseNetworkCascadeEngine`](../../contagion/cascade.py) propagation and [`HopOrderedSubgraphVerbalizer`](../../retrieval/pipeline.py) edge budgets (`MAX_SUBGRAPH_EDGES`).

---

## 4. Decision Outcome

Introduce a dedicated in-memory graph provider module [`retrieval/offline_store.py`](../../retrieval/offline_store.py) (exported via [`retrieval/__init__.py`](../../retrieval/__init__.py) and integrated into [`tools/dry_run_sec_8k_contagion.py`](../../tools/dry_run_sec_8k_contagion.py)):

### 4.1 Read-Only Cypher Subgraph Inspector (`CypherSubgraphQuery`)
Using pre-compiled linear-time patterns, parses read-only `MATCH` queries (including `WHERE` predicates injected by `TemporalQueryFilter.inject_active_coalescence` or `inject_audit_filter`) to extract:
- `root_ticker`: Extracted from inline `{ticker: '...'}` or parameterized `$ticker` (falls back to unscoped scan if absent).
- `rel_types`: Extracted relationship labels (e.g., `SUPPLIES_TO`; unconstrained if anonymous).
- `max_hops`: Variable path bound `*1..k` or fixed `*k` (defaults cleanly when omitted).
- `as_of_date` and `required_status`: Extracted from query `params["as_of_date"]`, inline `'YYYY-MM-DD'` literals, and `coalesce(r.status, 'ACTIVE_CURRENT') = '...'` or explicit audit status clauses.

### 4.2 Adjacency-Indexed PiT BFS Store (`InMemorySubgraphStore`)
Implements Neo4j/Memgraph driver duck-typing (`store.session()` context manager returning `.run(cypher, **params)` records compatible with `GraphRAGPipelineOrchestrator.execute_graph_query`):
- Normalizes incoming edges (`source`, `target`, `rel_type`, `status`, `share`/`exposure_pct`, `buffer_days`, `nature`/`input_group`, `materiality_tier`, `valid_from`, `valid_to`).
- Filters edges deterministically on `rel_types`, `required_status`, and PiT interval (`valid_from <= as_of_date <= valid_to` when bounds are non-null).
- Executes cycle-safe sorted BFS from `root_ticker` up to `max_hops`, annotating each emitted edge dictionary with exact `hop` distance.
- `OfflineGraphStore` in [`tools/dry_run_sec_8k_contagion.py`](../../tools/dry_run_sec_8k_contagion.py) delegates to / aliases `InMemorySubgraphStore` for seamless backward compatibility.

### 4.3 Snapshot & Manifest Loaders
Provide clean classmethods on `InMemorySubgraphStore`:
- `from_parquet_snapshot(snapshot_dir)`: Reads `nodes.parquet` and `edges.parquet` matching [`SnapshotManager`](../../knowledge_graph/snapshot_manager.py), resolves node `id` $\to$ uppercase `ticker` (checking explicit `ticker` column, `properties_json["ticker"]`, and normalized entity IDs), and unpacks edge `properties_json` (`share`, `exposure_pct`, `buffer_days`, `nature`, `valid_from`, `valid_to`, `status`).
- `from_edges_file(edges_path)`: Loads normalized edge records from `.json` (`list` or `{"edges": [...]}`) or `.jsonl` manifests.

### 4.4 CLI & Dry-Run Precedence Integration
Extend [`tools/dry_run_sec_8k_contagion.py`](../../tools/dry_run_sec_8k_contagion.py) with `--snapshot-dir` and `--edges-file` flags (and `snapshot_dir` / `edges_path` parameters on `run_sec_8k_dry_run`). When `graph_driver` is `None`, topology resolution follows strict deterministic precedence:
1. Explicit `custom_network_edges`
2. Explicit `--snapshot-dir` Parquet directory (`InMemorySubgraphStore.from_parquet_snapshot`)
3. Explicit `--edges-file` JSON/JSONL manifest (`InMemorySubgraphStore.from_edges_file`)
4. Canonical fixture smoke-test fallback (`CRUS`, `FABCO`, or single-edge fallback)

---

## 5. Invariants & Guardrails

1. **Zero Look-Ahead & Lifecycle Leakage**:
   - Edges with `valid_from > as_of_date`, `valid_to < as_of_date`, or non-matching `status` (e.g., `TERMINATED` on active queries) MUST NEVER be emitted when an active PiT query specifies `as_of_date`.
2. **Root & Hop Isolation Invariant**:
   - Queries specifying `{ticker: 'X'}` and `*[1..H]` MUST emit only edges reachable on valid active directed paths starting at `X` within $\le H$ hops.
3. **Read-Only Immutability**:
   - `InMemorySubgraphStore` strictly rejects or no-ops mutating Cypher statements (`CREATE`, `MERGE`, `DELETE`, `SET`, `DETACH`) and never modifies snapshot files on disk.
4. **Zero Absolute Paths Invariant**:
   - All document links remain strictly repository-relative (`../../<path>`).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Unlocks zero-Docker, Point-in-Time offline contagion sweeps over real supply networks (`data/snapshots/*` or extracted JSONL edge datasets) across all 2018–2026 Form 8-K events prior to downstream econometric testing.
- **Negative / Trade-offs**:
  - Complex openCypher features beyond bounded `MATCH` traversals and temporal/status `WHERE` clauses (e.g., nested aggregations or `OPTIONAL MATCH` subqueries) remain out of scope for the in-memory store and continue to execute against live Memgraph instances via [`get_memgraph_driver`](../../knowledge_graph/memgraph_driver.py).

