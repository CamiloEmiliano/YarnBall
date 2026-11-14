# ADR-00021: Contextualized Node Embeddings & Composite `pgvector` Keying

- **ADR ID**: ADR-00021
- **Filename**: `ADR-00021-contextual-node-embeddings-and-pgvector-.md`
- **Date**: 2026-11-14 19:25 UTC
- **Status**: PROPOSED
- **Target Subsystem**: `knowledge_graph` / `retrieval` / `tools`
- **Linked Commits**: Pending acceptance

---

## 1. Context & Problem Statement

Inspecting [`knowledge_graph/embedding_store.py`](../../knowledge_graph/embedding_store.py), [`tools/init_db.py`](../../tools/init_db.py), [`tools/utils.py`](../../tools/utils.py), [`retrieval/hybrid.py`](../../retrieval/hybrid.py), and [`retrieval/service.py`](../../retrieval/service.py) revealed three structural fragilities in how graph nodes are embedded and stored in PostgreSQL `pgvector`:

1. **Context-Free Node String Embedding**: `store_node_embeddings` builds `texts = [node["id"] for node in nodes]`, discarding `entity_type`, `ticker`, `cik`, and surrounding properties/context. Short or polysemous names (`"Apple"`, `"Delta"`, `"Alphabet"`, `"ARM"`) carry zero domain disambiguation in vector space.
2. **Destructive Single-Row Overwrite on `node_id`**: `node_embeddings` uses `node_id TEXT PRIMARY KEY` and upserts with `ON CONFLICT (node_id) DO UPDATE SET embedding = EXCLUDED.embedding, entity_type = EXCLUDED.entity_type, source_hash = EXCLUDED.source_hash`. Every new article mentioning an existing node overwrites its embedding and erases prior `source_hash` provenance.
3. **Inconsistent Wire Serialization for `%s::vector`**: `tools/utils.py` (`store_embedding`) formats vectors as `"[" + ",".join(map(str, vector)) + "]"` before `%s::vector`, whereas `store_node_embeddings`, `HybridRetriever.search_dense_entities`, and `retrieval/service.py` pass raw Python `list[float]` directly to `%s::vector` without guaranteed `pgvector` psycopg type registration.

---

## 2. Decision Drivers & Constraints

1. **Semantic Disambiguation**: Embeddings for financial graph nodes must encode entity role (`entity_type`), canonical identifiers (`ticker`, `cik`), and concise role context so polysemous surface strings separate cleanly.
2. **Idempotent Provenance Preservation**: Multiple filings and news articles referencing the same entity must not overwrite each other's `source_hash` lineage.
3. **Uniform `psycopg2` / `pgvector` Wire Format**: All `INSERT`, `UPDATE`, and `ORDER BY embedding <=> %s::vector` SQL paths must use one shared serializer compatible with vanilla `psycopg2` (even when `pgvector.psycopg2.register_vector` is not called).

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Keep `node_id` Alone With Post-Query Filtering**:
  - *Rationale for Rejection*: Dense top-$k$ retrieval happens *before* post-filtering; if `"Delta"` embeds near mathematical or river concepts rather than `"Delta Air Lines (DAL)"`, relevant candidate nodes never enter top-$k$.
- **Alternative B: Store Multiple Duplicate Embedding Rows per `node_id` in `node_embeddings`**:
  - *Rationale for Rejection*: Without separation, a single frequently mentioned entity (`Apple Inc.`) dominates all `top_k` slots in `ORDER BY embedding <=> %s::vector LIMIT k`, crowding out distinct neighbors.

---

## 4. Decision Outcome

Standardize contextual node representation, provenance preservation, and vector serialization across [`knowledge_graph/embedding_store.py`](../../knowledge_graph/embedding_store.py), [`tools/init_db.py`](../../tools/init_db.py), and [`retrieval/`](../../retrieval/hybrid.py):

1. **Canonical Contextual Embedding Formatter (`format_node_embedding_text`)**:
   - Format each node before calling `embed_texts`:
     `"{id} | type={entity_type} | ticker={ticker or 'N/A'} | cik={cik or 'N/A'} | context={context_snippet}"`.
   - Include short extracted descriptions or properties when available, bounded cleanly to model token limits.
2. **Split Canonical Node Vector vs. Provenance Mentions**:
   - Keep `node_embeddings` (`node_id PRIMARY KEY`) focused on canonical node retrieval (storing `last_source_hash` + `mention_count` or weighted rolling embedding, never blindingly losing mention lineage).
   - Record `(node_id, source_hash, first_seen_at)` in a normalized `node_embedding_sources` link table (or JSONB array capped cleanly) so every citing document remains auditable.
3. **Shared `format_pgvector_literal(vector)` Helper**:
   - Centralize `"[f1,f2,...]"` literal formatting (or explicit `pgvector` adapter registration on `pg_connection()`) and apply it uniformly in `store_node_embeddings`, `store_embedding`, `HybridRetriever.search_dense_entities`, and `retrieval/service.py`.

---

## 5. Invariants & Guardrails

1. **No Bare-ID Embeddings for Typed Nodes**: `store_node_embeddings` MUST include at least `entity_type` and available `ticker`/`cik` metadata in the embedded string.
2. **Deterministic Vector Wire Format**: Every raw SQL `%s::vector` parameter MUST pass through one shared serialization helper tested against live PostgreSQL `pgvector`.
3. **Provenance Non-Destruction**: Re-ingesting a node from a new `source_hash` MUST preserve prior source references rather than silently discarding them.
4. **Zero Absolute Paths Invariant**: All references remain strictly repository-relative (`../../<path>`).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Improves cosine top-$k$ entity disambiguation in [`retrieval/hybrid.py`](../../retrieval/hybrid.py) and [`retrieval/service.py`](../../retrieval/service.py).
  - Eliminates driver-specific `%s::vector` casting failures across `psycopg2` queries.
- **Negative / Trade-offs**:
  - Requires re-embedding existing `node_embeddings` rows once implemented so stored vectors match query-time embedding formatting.
