# ADR-00033: Point-in-Time Read Contract & Edge Temporal Semantics

- **ADR ID**: ADR-00033
- **Filename**: `ADR-00033-bitemporal-pit-contract-for-read-paths--.md`
- **Date**: 2026-11-22 08:48 UTC
- **Status**: ACCEPTED
- **Target Subsystem**: `retrieval` / `knowledge_graph` / `tools`
- **Linked Commits**: [`bcb33bd`](../../retrieval/temporal.py), [`bcb33bd`](../../retrieval/hybrid.py), [`bcb33bd`](../../retrieval/offline_store.py), [`bcb33bd`](../../knowledge_graph/embedding_store.py), [`bcb33bd`](../../ingestion/sec/harvester.py), [`bcb33bd`](../../tools/init_db.py), [`bcb33bd`](../../tests/retrieval/test_pit_invariants.py)

---

## 1. Context & Problem Statement

YarnBall's contract with Domain 2 is *leakage-free evidence as of time T*. The 2026-11-22 architecture review, with every item below confirmed in source, found that only one read path honors that contract, and that the temporal predicate it uses has gaps of its own.

1. **Path A predicate gaps** ([`retrieval/temporal.py`](../../retrieval/temporal.py)). `TemporalQueryFilter.inject_active_coalescence` emits:
   ```
   coalesce(r.status, 'ACTIVE_CURRENT') = 'ACTIVE_CURRENT'
   AND (r.valid_from IS NULL OR r.valid_from <= <as_of>)
   AND (r.valid_to IS NULL OR r.valid_to >= <as_of>)
   ```
   - **Unanchored edges pass every historical filter** because of `valid_from IS NULL`.
   - **`status` is a mutable, current-state attribute**, so it cannot be used to answer a historical question.
   - **No production writer sets `status` or `valid_to`** (grep across `knowledge_graph/`, `ingestion/`, `streaming/`, `tools/`, `retrieval/`, `contagion/`). Once written, an edge stays active forever. The graph has no termination or supersession semantics, so relationships that ended still show up in later as-of queries.
2. **Path B has no temporal enforcement** ([`retrieval/hybrid.py`](../../retrieval/hybrid.py)):
   - `expand_subgraph` runs `MATCH (s)-[r]->(t) WHERE s.id IN $seeds OR t.id IN $seeds ... LIMIT 50` with no temporal predicate. It computes `h = hops or self.subgraph_hops` and never uses it. `LIMIT 50` has no `ORDER BY`, and exceptions are swallowed at `DEBUG` level.
   - `search_dense_passages(..., as_of_date: Optional[str] = None)` and `retrieve(..., as_of_date: Optional[str] = None)` treat the cutoff as optional, so the default behavior leaks.
3. **The `/search` service has no as-of concept** ([`retrieval/service.py`](../../retrieval/service.py), [`retrieval/enricher.py`](../../retrieval/enricher.py)). It calls `enrich_nodes(ids, hops=1)`, which interpolates `OPTIONAL MATCH (n)-[r*1..{hops}]-(m)` into an f-string. The match is undirected, has no temporal predicate, and has no `LIMIT`.
4. **Node embeddings aren't point-in-time** ([`tools/init_db.py`](../../tools/init_db.py)). `node_embeddings` is keyed by `node_id TEXT PRIMARY KEY` and overwritten on every upsert (`updated_at`). A dense entity search for 2019 scores against vectors built from mentions up to today. Lineage in `node_embedding_sources` can't rebuild historical vectors because `first_seen_at TIMESTAMPTZ DEFAULT now()` records *ingestion* wall-clock time, not when the source was published.
5. **Valid time and knowledge time are the same field.** SEC edges stamp `valid_from = filing_date` ([`ingestion/sec/harvester.py`](../../ingestion/sec/harvester.py)) and news edges stamp `published_at` ([`ADR-00028`](ADR-00028-cross-source-entity-linking-and-kg-sync-.md)). In practice `valid_from` already means "publicly knowable date", which is the right anchor for trading, but it's undocumented and only has day resolution. An 8-K accepted after the 16:00 ET close is attributed to the same trading session. `edgartools` exposes `acceptance_datetime` in SEC headers, but YarnBall doesn't capture it.

---

## 2. Decision Drivers & Constraints

1. **Backtest and SFT validity**: every historical label and backtest built through a read path is invalid if future edges, future mentions, or ended relationships can reach it.
2. **Fail closed**: missing temporal information must exclude evidence, not include it. Leakage has to be impossible by default, not merely avoidable by careful callers.
3. **Trading-session correctness**: Domain 2 acts per session, so the knowledge timestamp has to resolve after-close disclosures to the next session.
4. **Offline parity**: `InMemorySubgraphStore` ([`ADR-00015`](ADR-00015-dynamic-in-memory-offline-subgraph-store.md)) and Memgraph must apply identical semantics.
5. **Free public data only** (`.agents/rules/data_sources.md`): knowledge timestamps come from EDGAR headers and FNSPID/8-K metadata, never Finnhub, for historical work.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Post-filter in Python after retrieval.** Rejected. `LIMIT` before the filter starves results non-deterministically, it moves more data, and each caller would have to remember to apply it.
- **Alternative B: One Memgraph database or graph version per as-of date.** Rejected. Storage grows linearly with snapshots, and Parquet snapshots ([`ADR-00028`](ADR-00028-cross-source-entity-linking-and-kg-sync-.md)) already serve that role offline.
- **Alternative C: Full bitemporal model (valid time + transaction time, versioned property history).** Deferred. Trading only needs *when knowable* and *when superseded*. Full transaction-time history adds write amplification without a current consumer.
- **Alternative D: Keep the `status` predicate for historical queries.** Rejected. A current-state flag can't answer a historical question. Once any writer sets `TERMINATED`, it would wrongly hide edges that were active at `as_of`.
- **Alternative E: Close edges when a later 10-K stops mentioning them.** Rejected. 10-K disclosure is annual and driven by materiality, so a missing mention isn't evidence the relationship ended. Material terminations have to be disclosed on Form 8-K within four business days.
- **Alternative F: Use the EDGAR 17:30 ET filing-date cutoff as the session boundary.** Rejected. YarnBall's consumers trade only in the regular session, so anything accepted after the 16:00 ET close can only be acted on in the next session.
- **Alternative G: Periodic snapshot copies for point-in-time node embeddings (`node_embeddings_pit` keyed by `snapshot_id`).** Rejected. Copying every unchanged entity vector on a fixed monthly/quarterly cadence forces an arbitrary cadence and portfolio-scoping choice while still losing daily precision compared to append-only validity intervals.

---

## 4. Decision Outcome

Adopt a **mandatory point-in-time read contract** with explicit edge temporal semantics.

1. **`PiTContext` in [`retrieval/temporal.py`](../../retrieval/temporal.py)**:
   ```python
   @dataclass(frozen=True)
   class PiTContext:
       as_of: date
       mode: Literal["historical", "live"] = "historical"
       include_unanchored: bool = False
   ```
   Every public read API takes `ctx: PiTContext` as a **required** argument: `HybridRetriever.retrieve`, `search_dense_entities`, `search_dense_passages`, `expand_subgraph`, `enrich_nodes`, the `/search` request model, and `GraphRAGPipelineOrchestrator.run` through `AnalystQuery`.
2. **Formalized edge temporal fields (write side)**: `store_graph_entities` in [`knowledge_graph/graph_store.py`](../../knowledge_graph/graph_store.py) and `EdgarWorker._persist_sec_graph_data` in [`ingestion/sec/harvester.py`](../../ingestion/sec/harvester.py):
   - `known_at: datetime (UTC)` is new. It comes from EDGAR `acceptance_datetime` and from news `published_at` timestamps. When the source has no timestamp it's set to null, with `known_at_imputed = true`.
   - `valid_from: date` means **the first U.S. equity regular trading session (NYSE / NASDAQ / Cboe) in which the assertion is actionable**, derived by `actionable_session(known_at)` (in [`retrieval/temporal.py`](../../retrieval/temporal.py)):
     - If `known_at` is on an open U.S. equity trading day **at or before the regular-session close** (16:00 ET; 13:00 ET on scheduled early-close half-days), the edge is actionable in that session.
     - If it's after the close, or on a weekend or exchange holiday, the edge is actionable in the **next** U.S. equity trading session.
     - The EDGAR 17:30 ET filing-date cutoff is ignored for this purpose. Extended-hours trading is out of scope, so information that lands after the regular close can only be acted on in the next regular session.
     - When `known_at_imputed = true`, `valid_from` is the next trading session after the document date (conservative: never earlier than the true knowable session).
   - `valid_to: date | null` is set **only by an explicit Form 8-K termination disclosure** (for example Item 1.02, *Termination of a Material Definitive Agreement*, due within four business days of the event). A relationship missing from a later 10-K is **not** treated as terminated, because 10-K disclosure is annual and driven by materiality, not an affirmative statement that a relationship ended.
     - `valid_to = actionable_session(known_at of the 8-K)`, **not** the event date stated in the 8-K. The edge stays visible to historical queries until the market could have known it ended. Filtering by event date would leak the termination into the gap between the event and the filing.
     - The event date is kept for audit as `terminated_on: date` and `termination_accession: str`, and is never used in read predicates.
     - Termination edges go through the same `validate_graph_payload(strict=True, source_text=...)` gate; an unresolved counterparty quarantines the termination rather than closing an arbitrary edge.
   - `last_affirmed_at: date` (new, informational only) is updated whenever a later filing restates the relationship. It's never used in read predicates; it exists so reports can show how old an edge's last confirmation is ([`ADR-00037`](ADR-00037-pit-community-layer-and-impact-ranking--.md) coverage).
   - `status` is kept for `live` mode and audit queries (`inject_audit_filter`) only.
   - **Trading calendar**: an NYSE session calendar with holidays and early closes is required. Neither `exchange_calendars` nor `pandas_market_calendars` is installed today; picking the library (or a static vendored calendar for 2018 onward) is an implementation-phase decision.
3. **Historical predicate** (replaces the coalescence when `ctx.mode == "historical"`):
   ```
   r.valid_from IS NOT NULL AND r.valid_from <= $as_of
   AND (r.valid_to IS NULL OR r.valid_to > $as_of)
   ```
   Unanchored edges are excluded unless `ctx.include_unanchored`. Either way, the count of excluded unanchored edges is returned to the caller. The existing `status` coalescence still applies in `live` mode.
4. **Path B conformance (minimal, before [`ADR-00034`](ADR-00034-single-bi-read-facade-and-pipeline-merge.md) consolidates the paths)**:
   - `expand_subgraph(seed_ids, hops, ctx)` uses a directed `[r*1..H]`. `H` is a validated `int` in `[1, 10]` before interpolation. The query goes through `TemporalQueryFilter` and has a deterministic `ORDER BY` before `LIMIT`.
   - `search_dense_passages` always applies `published_at <= ctx.as_of`.
   - `enrich_nodes(node_ids, hops, ctx, limit)` validates `hops`, adds the temporal predicate and `LIMIT`, and stays undirected only for node-detail display, documented as such.
5. **Snapshot-free point-in-time node embeddings (SCD Type 2)** ([`tools/init_db.py`](../../tools/init_db.py), [`knowledge_graph/embedding_store.py`](../../knowledge_graph/embedding_store.py)):
   - Replace the destructive `PRIMARY KEY (node_id)` overwrite in `node_embeddings` with append-only versioned rows using the same `[valid_from, valid_to)` interval contract as graph edges:
     - Columns: `(node_id TEXT NOT NULL, valid_from DATE NOT NULL, valid_to DATE NULL, entity_type VARCHAR(64), embedding vector(EMBEDDING_DIM), source_hash VARCHAR(64), known_at TIMESTAMPTZ, PRIMARY KEY (node_id, valid_from))`.
     - Indexes: partial HNSW index `WHERE valid_to IS NULL` for fast live queries, plus B-tree index on `(valid_from, valid_to)` for historical filtering.
   - On write: when an entity's contextual descriptor changes on actionable session date `D`, close the prior open row (`SET valid_to = D`) and insert the new version (`valid_from = D, valid_to = NULL`). Structural SEC updates refresh `node_embeddings`; transient news context remains in `financial_news_queue` (and `document_chunks` under [`ADR-00035`](ADR-00035-text-unit-chunk-layer-and-evidence-links.md)) so unchanged companies do not churn versions.
   - Add `source_published_at TIMESTAMPTZ` to `node_embedding_sources` (populated from the source document's knowable timestamp rather than `now()`) for audit lineage.
   - On read:
     - `historical` mode filters `WHERE valid_from <= $as_of AND (valid_to IS NULL OR valid_to > $as_of)`.
     - `live` mode filters `WHERE valid_to IS NULL`.
     - If no version exists for an entity at `ctx.as_of`, it is omitted rather than falling back to a later version.
6. **Migration**: existing edges keep `valid_from` and get `known_at_imputed = true`. Existing `node_embeddings` and `node_embedding_sources` rows have `valid_from` / `source_published_at` backfilled by joining `source_hash` to `sec_filings_queue.filing_date` and `financial_news_queue.published_at`. Legacy rows that cannot be joined receive a sentinel/excluded status so `historical` mode never reads them.

---

## 5. Invariants & Guardrails

1. **No defaulted as-of**: no public function in `retrieval/`, `synthesis/`, or `explorer/` may declare `ctx`/`as_of` as `Optional` or give it a default. This is enforced by an AST test in `tests/retrieval/test_pit_invariants.py`.
2. **Single temporal gate**: every Cypher string executed against Memgraph from `retrieval/` passes through `TemporalQueryFilter` or appears on an explicit, reviewed allowlist in the same AST test.
3. **Future exclusion**: an edge with `valid_from > as_of` (or a passage with `published_at > as_of`) is absent from every public read API and from both store implementations.
4. **Historical inclusion**: an edge active at `as_of` is present in `historical` mode no matter its current `status`.
5. **No future vectors in history**: `historical` mode queries `node_embeddings` strictly with `valid_from <= $as_of AND (valid_to IS NULL OR valid_to > $as_of)` and never reads open rows whose `valid_from > $as_of` or unanchored legacy rows.
6. **Validated traversal depth**: hop counts are interpolated only after validation as `int` in `[1, 10]`.
7. **Knowability-dated termination**: `valid_to` is always derived from the terminating 8-K's `known_at`, never from the event date it reports. An edge terminated by an 8-K filed after the close on day D is still visible to an as-of-D query.
8. **Termination requires an 8-K**: no code path sets `valid_to` from 10-K/10-Q non-reaffirmation, news, or elapsed time.

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Backtests and SFT labels built through any read path are protected against lookahead by construction.
  - Time semantics resolve to the regular NYSE trading session, matching Domain 2 execution constraints.
  - Append-only versioned `node_embeddings` deliver exact session-level precision without maintaining periodic embedding snapshots or portfolio-scoped builds.
  - Leaks show up as test failures instead of silent bias.
- **Negative / Trade-offs**:
  - Every caller has to construct a `PiTContext`, which breaks existing call signatures; the `adr-implementation` caller-fanout sweep is required.
  - Excluding unanchored edges by default may visibly reduce historical coverage until anchoring improves.
  - Append-only `node_embeddings` rows grow when entity descriptions change (~1–2 rows/entity/year for structural SEC updates) and require a schema migration on `node_embeddings`.

---

## 7. Open Questions for Review

1. ~~**Supersession policy**~~ **Resolved (2026-11-22)**: edges stay open until an explicit Form 8-K termination disclosure. 10-K non-reaffirmation never closes an edge (see Section 4.2).
2. ~~**After-close cutoff**~~ **Resolved (2026-11-23)**: regular-session close (16:00 ET, and 13:00 ET on NYSE early-close half-days) decides whether a disclosure moves to the next session. Extended-hours trading is out of scope (see Section 4.2). Empirical validation of reaction vs. drift across this boundary is deferred to [`ADR-00038`](ADR-00038-event-study-backtest-and-signal-decay---.md).
3. ~~**Snapshot scope and cadence for point-in-time embeddings**~~ **Resolved (2026-11-23)**: eliminated by adopting append-only `[valid_from, valid_to)` versioned `node_embeddings` (Section 4.5). Ephemeral Memgraph RAM retention and durable Postgres KG persistence are governed by [`ADR-00036`](ADR-00036-fail-fast-infra-and-deployable-topology-.md).
