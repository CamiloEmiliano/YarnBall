"""
Hybrid Retriever Engine (PostgreSQL pgvector + Memgraph Subgraph Traversal).

Executes deterministic Point-in-Time multi-modal retrieval (ADR-00021 / ADR-00022 / ADR-00033):
1. Translates intent to Cypher and executes against Memgraph via TextToCQL with PiT filtering.
2. Performs Point-in-Time dense semantic vector search over SCD Type 2 `node_embeddings`.
3. Performs Point-in-Time dense passage search over `financial_news_queue.embedding`.
4. Expands validated directed k-hop subgraph neighborhoods in Memgraph via TemporalQueryFilter.
5. Retrieves original news provenance snippets bounded by `ctx.cutoff_utc`.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from knowledge_graph.db import pg_connection
from knowledge_graph.memgraph_driver import get_memgraph_driver
from .temporal import PiTContext, TemporalQueryFilter, validate_hops
from .text_to_cql import TextToCQL

logger = logging.getLogger("retrieval.hybrid")


class HybridRetriever:
    """Combines Point-in-Time dense vector search with guarded multi-hop graph expansion."""

    def __init__(
        self,
        text_to_cql_engine: Optional[TextToCQL] = None,
        top_k_dense: int = 5,
        subgraph_hops: int = 1,
        temporal_filter: Optional[TemporalQueryFilter] = None,
    ):
        self.text_to_cql = text_to_cql_engine or TextToCQL()
        self.top_k_dense = top_k_dense
        self.subgraph_hops = validate_hops(subgraph_hops)
        self.temporal_filter = temporal_filter or TemporalQueryFilter()

    def _embed_query_vector_literal(self, query: str) -> Optional[str]:
        from embedding.embedder import (
            EmbedderUnavailableError,
            InvalidEmbeddingError,
            embed_texts,
            format_pgvector_literal,
        )

        try:
            embeddings = embed_texts([query])
            if not embeddings or not embeddings[0]:
                return None
            return format_pgvector_literal(embeddings[0])
        except (EmbedderUnavailableError, InvalidEmbeddingError):
            logger.error("Dense query embedding unavailable or invalid for query '%s'", query[:80])
            raise
        except Exception as exc:
            logger.debug("Embedding generation failed (%s); returning empty dense results", exc)
            return None

    # ----------------------------------------------------------------------
    # 1. Point-in-Time Dense Semantic Entity Search (SCD Type 2 pgvector)
    # ----------------------------------------------------------------------
    def search_dense_entities(
        self,
        query: str,
        ctx: PiTContext,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search PostgreSQL versioned `node_embeddings` strictly as of `ctx` (ADR-00033)."""
        if not isinstance(ctx, PiTContext):
            raise TypeError(f"ctx must be a PiTContext instance, got {type(ctx).__name__}")

        k = top_k or self.top_k_dense
        results: List[Dict[str, Any]] = []
        query_vector_literal = self._embed_query_vector_literal(query)
        if query_vector_literal is None:
            return results

        if ctx.mode == "live":
            sql = """
                SELECT node_id, entity_type, source_hash,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM node_embeddings
                WHERE valid_to IS NULL
                  AND valid_from IS NOT NULL
                  AND valid_from <= %s::date
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """
            params: Tuple[Any, ...] = (query_vector_literal, ctx.as_of_iso, query_vector_literal, k)
        else:
            sql = """
                SELECT node_id, entity_type, source_hash,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM node_embeddings
                WHERE COALESCE(is_unanchored, FALSE) = FALSE
                  AND valid_from IS NOT NULL
                  AND valid_from <= %s::date
                  AND (valid_to IS NULL OR valid_to > %s::date)
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """
            params = (
                query_vector_literal,
                ctx.as_of_iso,
                ctx.as_of_iso,
                query_vector_literal,
                k,
            )

        try:
            with pg_connection() as conn:
                cur = conn.cursor()
                cur.execute(sql, params)
                rows = cur.fetchall()
                for row in rows:
                    results.append(
                        {
                            "node_id": row[0],
                            "entity_type": row[1],
                            "source_hash": row[2],
                            "similarity": float(row[3]) if row[3] is not None else 1.0,
                        }
                    )
                if hasattr(cur, "close"):
                    cur.close()
        except Exception as exc:
            logger.warning("pgvector similarity search query failed: %s", exc)

        return results

    def search_dense_passages(
        self,
        query: str,
        ctx: PiTContext,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search PostgreSQL `financial_news_queue.embedding` bounded strictly by `ctx` (ADR-00033)."""
        if not isinstance(ctx, PiTContext):
            raise TypeError(f"ctx must be a PiTContext instance, got {type(ctx).__name__}")

        k = top_k or self.top_k_dense
        passages: List[Dict[str, Any]] = []
        query_vector_literal = self._embed_query_vector_literal(query)
        if query_vector_literal is None:
            return passages

        sql = """
            SELECT source_hash, title, source_url, published_at, raw_text, author,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM financial_news_queue
            WHERE embedding IS NOT NULL
              AND published_at IS NOT NULL
              AND published_at <= %s::timestamptz
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        params: Tuple[Any, ...] = (query_vector_literal, ctx.cutoff_utc_iso, query_vector_literal, k)

        try:
            with pg_connection() as conn:
                cur = conn.cursor()
                cur.execute(sql, params)
                for row in cur.fetchall():
                    raw_text = str(row[4] or "")
                    snippet = raw_text[:600] + ("..." if len(raw_text) > 600 else "")
                    passages.append(
                        {
                            "source_hash": row[0],
                            "title": row[1] or "Financial News Update",
                            "url": row[2] or "",
                            "published_at": str(row[3]) if row[3] else None,
                            "snippet": snippet,
                            "author": row[5] or "Financial News Desk",
                            "similarity": float(row[6]) if row[6] is not None else 1.0,
                        }
                    )
                if hasattr(cur, "close"):
                    cur.close()
        except Exception as exc:
            logger.warning("pgvector passage similarity search query failed: %s", exc)

        return passages

    def expand_passage_entities(
        self,
        source_hashes: List[str],
        ctx: PiTContext,
        limit: int = 20,
    ) -> List[str]:
        """Resolve matched news source_hashes back to Memgraph :News-[:MENTIONS]->entity IDs."""
        if not isinstance(ctx, PiTContext):
            raise TypeError(f"ctx must be a PiTContext instance, got {type(ctx).__name__}")

        valid_hashes = [h for h in source_hashes if h]
        if not valid_hashes:
            return []
        driver = get_memgraph_driver()
        entity_ids: List[str] = []
        base_query = """
            MATCH (n:News)-[:MENTIONS]->(e)
            WHERE n.id IN $hashes
            RETURN DISTINCT e.id AS entity_id
            ORDER BY entity_id ASC
            LIMIT $limit
        """
        query = self.temporal_filter.inject_news_cutoff(base_query, ctx, news_var="n")
        try:
            with driver.session() as session:
                res = session.run(query, hashes=valid_hashes, limit=limit)
                for r in res:
                    eid = r.get("entity_id")
                    if eid:
                        entity_ids.append(str(eid))
        except Exception as exc:
            logger.warning("Memgraph passage entity lookup failed: %s", exc)
        return entity_ids

    # ----------------------------------------------------------------------
    # 2. Multi-Hop Directed Point-in-Time Subgraph Expansion (Memgraph)
    # ----------------------------------------------------------------------
    def expand_subgraph(
        self,
        seed_entity_ids: List[str],
        ctx: PiTContext,
        hops: Optional[int] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Expand seed entities along directed 1..hops paths filtered by `ctx` (ADR-00033)."""
        if not isinstance(ctx, PiTContext):
            raise TypeError(f"ctx must be a PiTContext instance, got {type(ctx).__name__}")

        if not seed_entity_ids:
            return []

        h = validate_hops(self.subgraph_hops if hops is None else hops)
        safe_limit = max(1, min(int(limit), 500))
        driver = get_memgraph_driver()
        triples: List[Dict[str, Any]] = []

        base_query = f"""
            MATCH p = (seed)-[r*1..{h}]->(end_node)
            WHERE seed.id IN $seeds
            WITH relationships(p) AS path_rels, length(p) AS hop_dist
            UNWIND path_rels AS rel
            RETURN DISTINCT startNode(rel).id AS source,
                   type(rel) AS rel_type,
                   endNode(rel).id AS target,
                   properties(rel) AS props,
                   rel.source_hash AS source_hash,
                   min(hop_dist) AS hop
            ORDER BY hop ASC, source ASC, rel_type ASC, target ASC
            LIMIT $limit
        """
        query = self.temporal_filter.inject_active_coalescence(base_query, ctx)

        try:
            with driver.session() as session:
                res = session.run(
                    query,
                    seeds=seed_entity_ids,
                    limit=safe_limit,
                    as_of_date=ctx.as_of_iso,
                )
                for r in res:
                    source = r.get("source")
                    target = r.get("target")
                    rel_type = r.get("rel_type")
                    if source and target and rel_type:
                        props = r.get("props", {}) or {}
                        sh = r.get("source_hash") or props.get("source_hash")
                        triples.append(
                            {
                                "source": source,
                                "relation": rel_type,
                                "target": target,
                                "properties": props,
                                "source_hash": sh or None,
                                "hop": int(r.get("hop", 1) or 1),
                            }
                        )
        except Exception as exc:
            logger.warning("Memgraph subgraph expansion failed: %s", exc)

        return triples

    # ----------------------------------------------------------------------
    # 3. Point-in-Time Article Context Enrichment (financial_news_queue)
    # ----------------------------------------------------------------------
    def fetch_article_context(
        self,
        source_hashes: List[str],
        ctx: PiTContext,
        max_articles: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve original article metadata and text snippets bounded strictly by `ctx`."""
        if not isinstance(ctx, PiTContext):
            raise TypeError(f"ctx must be a PiTContext instance, got {type(ctx).__name__}")

        valid_hashes = [h for h in source_hashes if h]
        if not valid_hashes:
            return []

        articles: List[Dict[str, Any]] = []
        try:
            with pg_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT source_hash, title, source_url, published_at, raw_text, author
                    FROM financial_news_queue
                    WHERE source_hash = ANY(%s)
                      AND published_at IS NOT NULL
                      AND published_at <= %s::timestamptz
                    ORDER BY published_at DESC, source_hash ASC
                    LIMIT %s;
                    """,
                    (valid_hashes[:max_articles], ctx.cutoff_utc_iso, max_articles),
                )
                rows = cur.fetchall()
                for row in rows:
                    raw_text = str(row[4] or "")
                    snippet = raw_text[:600] + ("..." if len(raw_text) > 600 else "")
                    articles.append(
                        {
                            "source_hash": row[0],
                            "title": row[1] or "Financial News Update",
                            "url": row[2] or "",
                            "published_at": str(row[3]) if row[3] else None,
                            "snippet": snippet,
                            "author": row[5] or "Financial News Desk",
                        }
                    )
                if hasattr(cur, "close"):
                    cur.close()
        except Exception as exc:
            logger.warning("Could not fetch article context from PostgreSQL: %s", exc)

        return articles

    # ----------------------------------------------------------------------
    # 4. End-to-End Hybrid Retrieval
    # ----------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        ctx: PiTContext,
    ) -> Dict[str, Any]:
        """Execute dual-index Point-in-Time hybrid retrieval strictly as of `ctx` (ADR-00033)."""
        if not isinstance(ctx, PiTContext):
            raise TypeError(f"ctx must be a PiTContext instance, got {type(ctx).__name__}")

        start_time = time.perf_counter()

        # 1. Execute Guarded & Temporalized Text-to-CQL
        cql_res = self.text_to_cql.execute_query(query, ctx=ctx)
        structured_records = cql_res.get("records", [])

        # 2. Dual-Index Point-in-Time Dense Search (Versioned Nodes + Financial News Passages)
        dense_entities = self.search_dense_entities(query, ctx=ctx)
        dense_passages = self.search_dense_passages(query, ctx=ctx)

        seed_ids = [e["node_id"] for e in dense_entities]
        passage_hashes = [p["source_hash"] for p in dense_passages if p.get("source_hash")]
        seed_ids.extend(self.expand_passage_entities(passage_hashes, ctx=ctx))

        # Extract entity IDs returned from Cypher results to expand if needed
        for r in structured_records:
            for val in r.values():
                if isinstance(val, str) and len(val) < 64:
                    seed_ids.append(val)

        unique_seeds = list(dict.fromkeys(seed_ids))[:8]

        # 3. Directed, Hop-Validated, Temporalized Subgraph Expansion
        subgraph_triples = self.expand_subgraph(unique_seeds, ctx=ctx)

        # 4. Collect all unique source_hash values
        all_hashes: Set[str] = set()
        for e in dense_entities:
            if e.get("source_hash"):
                all_hashes.add(e["source_hash"])
        for p in dense_passages:
            if p.get("source_hash"):
                all_hashes.add(p["source_hash"])
        for t in subgraph_triples:
            if t.get("source_hash"):
                all_hashes.add(t["source_hash"])
        for r in structured_records:
            if "source_hash" in r and r["source_hash"]:
                all_hashes.add(r["source_hash"])

        # 5. Merge article context from PostgreSQL and dense passages without duplicates
        fetched_articles = self.fetch_article_context(list(all_hashes), ctx=ctx, max_articles=5)
        seen_article_hashes: Set[str] = {
            str(a["source_hash"]) for a in fetched_articles if a.get("source_hash")
        }
        articles = list(fetched_articles)
        for p in dense_passages:
            sh = str(p.get("source_hash") or "")
            if sh and sh not in seen_article_hashes and len(articles) < 5:
                seen_article_hashes.add(sh)
                articles.append(p)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "query": query,
            "as_of": ctx.as_of_iso,
            "pit_mode": ctx.mode,
            "cql_result": cql_res,
            "dense_entities": dense_entities,
            "dense_passages": dense_passages,
            "subgraph_triples": subgraph_triples,
            "articles": articles,
            "source_hashes": list(all_hashes),
            "latency_ms": round(latency_ms, 2),
        }
