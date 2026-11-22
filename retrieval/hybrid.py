"""
Hybrid Retriever Engine (PostgreSQL pgvector + Memgraph Subgraph Traversal).

Executes deterministic multi-modal retrieval:
1. Translates intent to Cypher and executes against Memgraph via TextToCQL.
2. Performs dense semantic vector search over PostgreSQL `node_embeddings`.
3. Expands k-hop local subgraph neighborhoods in Memgraph.
4. Retrieves original news provenance snippets from PostgreSQL `financial_news_queue`.
"""

from __future__ import annotations

import json
import logging
import os
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
from .text_to_cql import TextToCQL

logger = logging.getLogger("retrieval.hybrid")


class HybridRetriever:
    """Combines dense vector search with multi-hop graph expansion and Text-to-CQL."""

    def __init__(
        self,
        text_to_cql_engine: Optional[TextToCQL] = None,
        top_k_dense: int = 5,
        subgraph_hops: int = 1,
    ):
        self.text_to_cql = text_to_cql_engine or TextToCQL()
        self.top_k_dense = top_k_dense
        self.subgraph_hops = subgraph_hops

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
    # 1. Dense Semantic Entity Search (PostgreSQL pgvector)
    # ----------------------------------------------------------------------
    def search_dense_entities(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search PostgreSQL node_embeddings using cosine distance via pgvector."""
        k = top_k or self.top_k_dense
        results: List[Dict[str, Any]] = []
        query_vector_literal = self._embed_query_vector_literal(query)
        if query_vector_literal is None:
            return results

        try:
            with pg_connection() as conn:
                cur = conn.cursor()
                # Query nearest neighbor nodes using pgvector cosine distance operator <=>
                cur.execute(
                    """
                    SELECT node_id, entity_type, source_hash,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM node_embeddings
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (query_vector_literal, query_vector_literal, k),
                )
                rows = cur.fetchall()
                for row in rows:
                    results.append({
                        "node_id": row[0],
                        "entity_type": row[1],
                        "source_hash": row[2],
                        "similarity": float(row[3]) if row[3] is not None else 1.0,
                    })
                if hasattr(cur, "close"):
                    cur.close()
        except Exception as exc:
            logger.debug("pgvector similarity search query failed: %s", exc)

        return results

    def search_dense_passages(
        self,
        query: str,
        top_k: Optional[int] = None,
        as_of_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search PostgreSQL financial_news_queue.embedding with optional PiT cutoff (ADR-00022)."""
        k = top_k or self.top_k_dense
        passages: List[Dict[str, Any]] = []
        query_vector_literal = self._embed_query_vector_literal(query)
        if query_vector_literal is None:
            return passages

        cutoff = str(as_of_date).strip() if as_of_date else ""
        pit_clause = (
            "AND published_at IS NOT NULL AND published_at <= %s::timestamptz"
            if cutoff
            else ""
        )
        sql = f"""
            SELECT source_hash, title, source_url, published_at, raw_text, author,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM financial_news_queue
            WHERE embedding IS NOT NULL
            {pit_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        params: Tuple[Any, ...] = (
            (query_vector_literal, cutoff, query_vector_literal, k)
            if cutoff
            else (query_vector_literal, query_vector_literal, k)
        )

        try:
            with pg_connection() as conn:
                cur = conn.cursor()
                cur.execute(sql, params)
                for row in cur.fetchall():
                    raw_text = str(row[4] or "")
                    snippet = raw_text[:600] + ("..." if len(raw_text) > 600 else "")
                    passages.append({
                        "source_hash": row[0],
                        "title": row[1] or "Financial News Update",
                        "url": row[2] or "",
                        "published_at": str(row[3]) if row[3] else None,
                        "snippet": snippet,
                        "author": row[5] or "Financial News Desk",
                        "similarity": float(row[6]) if row[6] is not None else 1.0,
                    })
                if hasattr(cur, "close"):
                    cur.close()
        except Exception as exc:
            logger.debug("pgvector passage similarity search query failed: %s", exc)

        return passages

    def expand_passage_entities(
        self,
        source_hashes: List[str],
        limit: int = 20,
    ) -> List[str]:
        """Resolve matched news source_hashes back to Memgraph :News-[:MENTIONS]->entity IDs."""
        valid_hashes = [h for h in source_hashes if h]
        if not valid_hashes:
            return []
        driver = get_memgraph_driver()
        entity_ids: List[str] = []
        query = """
            MATCH (n:News)-[:MENTIONS]->(e)
            WHERE n.id IN $hashes
            RETURN DISTINCT e.id AS entity_id
            LIMIT $limit
        """
        try:
            with driver.session() as session:
                res = session.run(query, hashes=valid_hashes, limit=limit)
                for r in res:
                    eid = r.get("entity_id")
                    if eid:
                        entity_ids.append(str(eid))
        except Exception as exc:
            logger.debug("Memgraph passage entity lookup failed: %s", exc)
        return entity_ids

    # ----------------------------------------------------------------------
    # 2. Multi-Hop Subgraph Neighborhood Expansion (Memgraph)
    # ----------------------------------------------------------------------
    def expand_subgraph(
        self, seed_entity_ids: List[str], hops: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Expand seed entities by 1-to-2 hops in Memgraph to retrieve connecting triples."""
        if not seed_entity_ids:
            return []

        h = hops or self.subgraph_hops
        driver = get_memgraph_driver()
        triples: List[Dict[str, Any]] = []

        query = f"""
            MATCH (s)-[r]->(t)
            WHERE s.id IN $seeds OR t.id IN $seeds
            RETURN s.id AS source, type(r) AS rel_type, t.id AS target,
                   properties(r) AS props, r.source_hash AS source_hash
            LIMIT 50
        """

        try:
            with driver.session() as session:
                res = session.run(query, seeds=seed_entity_ids)
                for r in res:
                    source = r.get("source")
                    target = r.get("target")
                    rel_type = r.get("rel_type")
                    if source and target and rel_type:
                        props = r.get("props", {}) or {}
                        sh = r.get("source_hash") or props.get("source_hash")
                        triples.append({
                            "source": source,
                            "relation": rel_type,
                            "target": target,
                            "properties": props,
                            "source_hash": sh or None,
                        })
        except Exception as exc:
            logger.debug("Memgraph subgraph expansion failed: %s", exc)

        return triples

    # ----------------------------------------------------------------------
    # 3. Article Context Enrichment (financial_news_queue)
    # ----------------------------------------------------------------------
    def fetch_article_context(
        self, source_hashes: List[str], max_articles: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve original article metadata and text snippets by source_hash."""
        valid_hashes = [h for h in source_hashes if h]
        if not valid_hashes:
            return []

        articles = []
        try:
            with pg_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT source_hash, title, source_url, published_at, raw_text, author
                    FROM financial_news_queue
                    WHERE source_hash = ANY(%s)
                    LIMIT %s;
                    """,
                    (valid_hashes[:max_articles], max_articles),
                )
                rows = cur.fetchall()
                for row in rows:
                    raw_text = str(row[4] or "")
                    snippet = raw_text[:600] + ("..." if len(raw_text) > 600 else "")
                    articles.append({
                        "source_hash": row[0],
                        "title": row[1] or "Financial News Update",
                        "url": row[2] or "",
                        "published_at": str(row[3]) if row[3] else None,
                        "snippet": snippet,
                        "author": row[5] or "Financial News Desk",
                    })
                if hasattr(cur, "close"):
                    cur.close()
        except Exception as exc:
            logger.debug("Could not fetch article context from PostgreSQL: %s", exc)

        return articles

    # ----------------------------------------------------------------------
    # 4. End-to-End Hybrid Retrieval
    # ----------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        as_of_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute dual-index hybrid retrieval combining Text-to-CQL, entity, and passage search."""
        start_time = time.perf_counter()

        # 1. Execute Guarded Text-to-CQL
        cql_res = self.text_to_cql.execute_query(query)
        structured_records = cql_res.get("records", [])

        # 2. Dual-Index Dense Search (Nodes + Financial News Passages)
        dense_entities = self.search_dense_entities(query)
        dense_passages = self.search_dense_passages(query, as_of_date=as_of_date)

        seed_ids = [e["node_id"] for e in dense_entities]
        passage_hashes = [p["source_hash"] for p in dense_passages if p.get("source_hash")]
        seed_ids.extend(self.expand_passage_entities(passage_hashes))

        # Extract entity IDs returned from Cypher results to expand if needed
        for r in structured_records:
            for val in r.values():
                if isinstance(val, str) and len(val) < 64:
                    seed_ids.append(val)

        unique_seeds = list(dict.fromkeys(seed_ids))[:8]

        # 3. Subgraph Neighborhood Expansion
        subgraph_triples = self.expand_subgraph(unique_seeds)

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
        fetched_articles = self.fetch_article_context(list(all_hashes), max_articles=5)
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
            "cql_result": cql_res,
            "dense_entities": dense_entities,
            "dense_passages": dense_passages,
            "subgraph_triples": subgraph_triples,
            "articles": articles,
            "source_hashes": list(all_hashes),
            "latency_ms": round(latency_ms, 2),
        }
