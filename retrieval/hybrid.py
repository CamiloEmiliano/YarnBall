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

    # ----------------------------------------------------------------------
    # 1. Dense Semantic Entity Search (PostgreSQL pgvector)
    # ----------------------------------------------------------------------
    def search_dense_entities(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Search PostgreSQL node_embeddings using cosine distance via pgvector."""
        k = top_k or self.top_k_dense
        results = []

        try:
            from embedding.embedder import embed_texts
            embeddings = embed_texts([query])
            if not embeddings or not embeddings[0]:
                return []
            query_vector = embeddings[0]
        except Exception as exc:
            logger.debug("Embedding generation failed (%s); returning empty dense results", exc)
            return []

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
                    (query_vector, query_vector, k),
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
    def retrieve(self, query: str) -> Dict[str, Any]:
        """Execute hybrid retrieval combining Text-to-CQL and Dense Vector Expansion."""
        start_time = time.perf_counter()

        # 1. Execute Guarded Text-to-CQL
        cql_res = self.text_to_cql.execute_query(query)
        structured_records = cql_res.get("records", [])

        # 2. Dense Vector Entity Search
        dense_entities = self.search_dense_entities(query)
        seed_ids = [e["node_id"] for e in dense_entities]

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
        for t in subgraph_triples:
            if t.get("source_hash"):
                all_hashes.add(t["source_hash"])
        for r in structured_records:
            if "source_hash" in r and r["source_hash"]:
                all_hashes.add(r["source_hash"])

        # 5. Fetch Article Context from PostgreSQL
        articles = self.fetch_article_context(list(all_hashes), max_articles=5)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "query": query,
            "cql_result": cql_res,
            "dense_entities": dense_entities,
            "subgraph_triples": subgraph_triples,
            "articles": articles,
            "source_hashes": list(all_hashes),
            "latency_ms": round(latency_ms, 2),
        }
