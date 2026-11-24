# ADR-00035: Text-Unit Chunk Layer & Edge Evidence Links

- **ADR ID**: ADR-00035
- **Filename**: `ADR-00035-text-unit-chunk-layer-and-evidence-links.md`
- **Date**: 2026-11-22 08:48 UTC
- **Status**: PROPOSED
- **Target Subsystem**: `ingestion` / `knowledge_graph` / `ontology` / `retrieval` / `tools`
- **Linked Commits**: Pending acceptance

---

## 1. Context & Problem Statement

1. **The segmenter is dead code at runtime.** `FinancialSectionSegmenter` ([`ingestion/segmentation/chunker.py`](../../ingestion/segmentation/chunker.py)) produces `<=1500`-token overlapping chunks with `[DOCUMENT | CIK | TICKER | SECTION]` headers. Outside its own package it's imported only by tests (`tests/ingestion/test_section_segmenter.py`, `tests/integration/pipelines/test_filing_to_graph_flow.py`, `tests/integration/test_unified_sit_contract.py`).
2. **SEC extraction uses one heuristic window.** `select_salient_sec_context(text, max_chars=2500)` ([`ingestion/sec/parser.py`](../../ingestion/sec/parser.py)) keyword-scores paragraphs and keeps only what fits in 2500 characters per section, falling back to `text[:max_chars]`. Supplier and customer disclosures spread across Item 1, Item 1A, and Item 7 of large filers compete for that one window. Anything outside it is never extracted, so recall is capped hardest for the most complex, highest-weight S&P 500 constituents.
3. **News extraction sends raw article text** (`extract_entities` in [`knowledge_graph/graph_store.py`](../../knowledge_graph/graph_store.py)), so long articles depend on silent model-side truncation.
4. **No claim-level provenance.** Edges cite a document (`accession_number`, `source_hash`), not a passage. Neither the Workbench audit view ([`ADR-00018`](ADR-00018-workbench-ui-shell-and-8k-audit-view----.md)) nor synthesis can show the sentence behind an edge. Clause-scoped negation ([`ADR-00024`](ADR-00024-clause-scoped-negation-and-hedge-filter-.md)) runs against the selected excerpt rather than the evidence span.
5. **No text-unit layer.** Canonical GraphRAG makes text units first-class (document ↔ text unit ↔ entity). That layer is the backbone of both local retrieval and community summarization ([`ADR-00037`](ADR-00037-pit-community-layer-and-impact-ranking--.md)), and it's missing here.

---

## 2. Decision Drivers & Constraints

1. **Audit-ready BI**: every edge that reaches Domain 2 must trace back to a citable passage.
2. **Recall on large filers**: extraction must cover the full relevant sections, not a single 2500-character window.
3. **Local compute budget**: extraction runs on local Ollama (`yarnball-qwen:7b`), so chunk count directly drives batch wall-clock time.
4. **Ontology gate unchanged**: `validate_graph_payload(strict=True, source_text=...)` stays the single write gate ([`ADR-00023`](ADR-00023-unified-live-ontology-gate-and-schema---.md)).
5. **Point-in-time**: chunks inherit the document's `valid_from` / `known_at` ([`ADR-00033`](ADR-00033-bitemporal-pit-contract-for-read-paths--.md)).
6. **Thin in-memory graph discipline** ([`ADR-00036`](ADR-00036-fail-fast-infra-and-deployable-topology-.md)): Memgraph is intentionally pure in-memory and must stay thin (structural entities and edges only, `< 1–2 GB` RAM). Storing millions of chunk or document text nodes inside Memgraph RAM would defeat that memory bound.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Raise `max_chars`.** Rejected. It moves the ceiling without removing it, still has no provenance, and widens the context negation filtering has to work across.
- **Alternative B: Materialize `(:Chunk)` and `(:Document)` nodes inside Memgraph RAM.** Rejected. Across 2018–2026 filings and news, millions of chunk nodes and `EVIDENCED_BY` edges would bloat Memgraph RAM to tens of gigabytes and slow boot hydration ([`ADR-00036`](ADR-00036-fail-fast-infra-and-deployable-topology-.md)). Because the verbalizer clamps context to 25 edges before synthesis, resolving passage text from Postgres via a single batched `WHERE chunk_id = ANY(%s)` lookup adds negligible latency while keeping Memgraph RAM compact.
- **Alternative C: Full-document extraction with long-context models.** Rejected. It goes beyond the fine-tuned 7B serving envelope and gives up span-level provenance.

---

## 4. Decision Outcome

1. **Wire in the segmenter (`max_tokens=800`, `overlap_tokens=120`)**:
   - `EdgarWorker` ([`ingestion/sec/harvester.py`](../../ingestion/sec/harvester.py)) segments each section with `FinancialSectionSegmenter(max_tokens=800, overlap_tokens=120)`, sized to stay comfortably within `yarnball-qwen:7b`'s 512-token extraction output cap and produce tight ~1-page audit citations.
   - `extract_entities` segments news articles longer than the token budget.
   - `select_salient_sec_context` becomes a **chunk ranker** used to order and prioritize extraction, never a filter that drops text.
2. **High-accuracy financial embedder & hybrid lexical index** ([`embedding/embedder.py`](../../embedding/embedder.py), [`tools/init_db.py`](../../tools/init_db.py)):
   - Upgrade the default embedder from `all-mpnet-base-v2` (384-token ceiling, symmetric 2021 pretraining) to **`Qwen3-Embedding-0.6B`** (`qwen3-embedding:0.6b`, Apache 2.0, **32K context**, native **`EMBEDDING_DIM=1024`**—preserving full dimensionality within `pgvector` HNSW's 2,000-dim limit without truncation loss).
   - Add explicit asymmetric entry points `embed_query(text)` (instruction-prefixed for financial/supply-chain retrieval) and `embed_documents(texts)` in [`embedding/embedder.py`](../../embedding/embedder.py).
   - Serve via **Ollama `/api/embed`** pinned to CPU (`options: {"num_gpu": 0}`) during live serving so `yarnball-qwen:7b` retains 100% of 8 GB VRAM with zero model eviction, and run GPU-accelerated Phase-2 batch embedding offline (after unloading `yarnball-qwen:7b` via `keep_alive: 0`).
   - Add a Postgres `GIN` `tsvector` full-text index alongside `embedding vector(1024)` on `document_chunks` so chunk retrieval combines dense semantic similarity with exact lexical matching (tickers, CIKs, SEC Item codes, fab names) via Reciprocal Rank Fusion (RRF).
3. **Storage (Postgres text store + thin Memgraph pointers)**:
   - Postgres: `document_chunks(chunk_id TEXT PRIMARY KEY, doc_id, source_type, cik, section, char_start, char_end, text, tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED, valid_from, known_at, embedding vector(EMBEDDING_DIM))` plus `chunk_entity_links(chunk_id, node_id, valid_from)` created in [`tools/init_db.py`](../../tools/init_db.py).
   - Durable KG ledger (`kg_edges`, [`ADR-00036`](ADR-00036-fail-fast-infra-and-deployable-topology-.md)) and Memgraph typed edges carry compact `evidence_chunk_ids: list[str]` pointers—**no `(:Chunk)` or raw prose nodes are stored in Memgraph RAM**.
   - `chunk_id` format is deterministic: `<accession_or_source_hash>#<section_slug>:c<ordinal>`.
4. **Per-chunk extraction & targeted gleaning (`ENABLE_TARGETED_GLEANING=1`)**:
   - Run Pass 1 via `run_structured_extraction` per chunk followed by `validate_graph_payload(strict=True, source_text=chunk.text)`.
   - **Targeted 1-pass gleaning (`should_glean_chunk`)**: run at most one follow-up `EXTRACT_SEC_GRAPH` (`temperature=0.0`) pass when `chunk.section_name` is a high-signal section (`Item 1 - Business`, `Item 1A - Risk Factors`, or Form 8-K `Item 1.01` / `1.02` / `2.01` / `8.01`), `not resp.runaway_detected`, and **either** Pass 1 produced $\ge 1$ valid relation **or** `resp.generated_tokens >= int(0.85 * 512)` (near the 512-token output cap).
   - Pass 2 receives `chunk.full_prompt_text` plus a compact `(src, predicate, dst)` summary of Pass-1 triples with an explicit empty-JSON stop condition, runs through the identical `validate_graph_payload(strict=True, source_text=chunk.text)` gate, quarantines Pass-2 failures (`GLEANING_PARSE_FAILED`) non-destructively without discarding Pass 1, tags rescued triples with `extraction_pass=2`, and logs `gleaning_new_edges_count`.
   - Cross-chunk merge dedupes on `(src, predicate, dst)`, unions `evidence_chunk_ids`, and resolves `ParameterProvenance` by highest-confidence chunk.
5. **Contract changes**:
   - `ExtractionGraphPayload` gains `chunk_id`.
   - `ParameterProvenance` ([`contagion/parameters.py`](../../contagion/parameters.py)) gains `evidence_chunk_ids`.
   - `SnapshotManager` exports `chunks.parquet` next to `nodes.parquet` and `edges.parquet` for zero-DB offline Workbench citation lookup.
6. **Citation handles**: `HopOrderedSubgraphVerbalizer` emits `[E<n> ← <chunk_id>]` per edge and fetches passage snippets in one batched Postgres (or offline `chunks.parquet`) lookup for the clamped edges only. `GroundedSynthesizer` ([`ADR-00034`](ADR-00034-single-bi-read-facade-and-pipeline-merge.md)) must cite them, and outputs with uncited claims go to `RuntimeRejectionQueue`.
7. **Migration & incremental re-extraction**: bump `embedding_schema_metadata` to `qwen3-embedding:0.6b` (`EMBEDDING_DIM=1024`) and re-embed `node_embeddings` / `financial_news_queue` cleanly before populating `document_chunks`. Existing edges get `evidence_chunk_ids = []` and `evidence_status = "legacy_unchunked"`, which shows up in report coverage ([`ADR-00037`](ADR-00037-pit-community-layer-and-impact-ranking--.md)). Re-extraction runs incrementally via a resumable batch job under `tools/` starting with **active Point-in-Time S&P 500 constituents** ([`universe/sp500.py`](../../universe/sp500.py)), with broader expansion calibrated later via [`ADR-00038`](ADR-00038-event-study-backtest-and-signal-decay---.md).

---

## 5. Invariants & Guardrails

1. **Evidence on new edges**: every edge written after this ADR has a non-empty `evidence_chunk_ids` list whose IDs resolve to `document_chunks`.
2. **Thin Memgraph invariant**: no `(:Chunk)` nodes or raw filing/article prose properties are written into Memgraph RAM.
3. **Negation scope**: ontology gate negation and hedge checks run against the chunk text that produced the assertion.
4. **No text dropped by selection**: no runtime extractor truncates section text by a character budget. Ranking only orders the chunks.
5. **Chunk point-in-time inheritance**: `document_chunks.valid_from` equals the parent document's `valid_from`.

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Claim-level provenance from Domain 2 reports back to exact SEC/news passages without bloating Memgraph RAM.
  - Higher extraction recall on large filers.
  - Fast Memgraph boot hydration preserved ([`ADR-00036`](ADR-00036-fail-fast-infra-and-deployable-topology-.md)), with one batched Postgres chunk fetch per query.
  - The text-unit backbone needed for community summarization ([`ADR-00037`](ADR-00037-pit-community-layer-and-impact-ranking--.md)).
- **Negative / Trade-offs**:
  - Extraction LLM calls grow roughly with chunk count (several times more per 10-K).
  - Extra disk storage in Postgres for chunk text and vectors.
  - Cross-chunk merge can introduce conflicting parameter values that need resolution rules.

---

## 7. Open Questions for Review

1. ~~**Chunk size & embedder**~~ **Resolved (2026-11-23)**: default `max_tokens=800` (`overlap_tokens=120`) for extraction precision within the 512-token output cap, paired with upgrading the embedder to **`Qwen3-Embedding-0.6B`** (`qwen3-embedding:0.6b`, 32K context, native 1024 dims, CPU-pinned via Ollama `num_gpu: 0` during live queries / GPU Phase-2 batch offline) plus Postgres `tsvector` hybrid RRF (see Sections 4.1–4.2).
2. ~~**Gleaning**~~ **Resolved (2026-11-23)**: enable targeted 1-pass gleaning (`ENABLE_TARGETED_GLEANING=1`) gated to high-signal sections (10-K Item 1/1A and 8-K disruption items) when Pass 1 either extracted $\ge 1$ relation or reached $\ge 85\%$ of the 512-token output cap, preserving Pass 1 non-destructively on Pass-2 failure (see Section 4.4).
3. ~~**Re-extraction scope**~~ **Resolved (2026-11-23)**: run incrementally via a resumable batch job starting with active Point-in-Time S&P 500 constituents (`universe/sp500.py`), flagging untouched legacy edges as `legacy_unchunked` (see Section 4.7).
