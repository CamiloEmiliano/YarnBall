# rag/hybrid_retriever.py
# -*- coding: utf-8 -*-
"""Hybrid Retriever & Grounded Synthesis Engine.

Combines PostgreSQL pgvector dense semantic search with Memgraph multi-hop subgraph
expansion and guarded Text-to-CQL, retrieving article provenance via source_hash
links and synthesizing grounded financial answers with clickable citations using Ollama (qwen3:8b).
"""

import os
import re
import json
import logging
import time
from typing import Optional, Dict, Any, List, Tuple, Set
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from graph.db import pg_connection
from graph.memgraph_driver import get_memgraph_driver
from rag.text_to_cql import TextToCQL

logger = logging.getLogger(__name__)

# LLM Configuration
def _default_ollama_url() -> str:
    env_url = os.getenv("QWEN_API_BASE")
    if env_url:
        if "host.docker.internal" in env_url and not os.path.exists("/.dockerenv"):
            return env_url.replace("host.docker.internal", "localhost")
        return env_url
    if os.path.exists("/.dockerenv"):
        return "http://host.docker.internal:11434/api/generate"
    return "http://localhost:11434/api/generate"

OLLAMA_API_BASE = _default_ollama_url()
OLLAMA_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "qwen3:8b")


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


class GroundedSynthesizer:
    """Synthesizes factual answers grounded strictly in retrieved graph & article context."""

    def __init__(
        self,
        ollama_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.ollama_url = ollama_url or OLLAMA_API_BASE
        self.model_name = model_name or OLLAMA_MODEL_NAME

    def _build_synthesis_prompt(
        self, query: str, context: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Construct the grounded prompt with verified facts and anti-hallucination rules."""
        triples = context.get("subgraph_triples", [])
        cql_records = context.get("cql_result", {}).get("records", [])
        articles = context.get("articles", [])

        # Format Graph Relationships
        graph_lines = []
        for t in triples[:15]:
            graph_lines.append(f"- ({t['source']}) -[{t['relation']}]-> ({t['target']})")
        for r in cql_records[:10]:
            graph_lines.append(f"- Structured Result: {json.dumps(r)}")

        graph_context = "\n".join(graph_lines) if graph_lines else "No direct graph relationships found."

        # Format Supporting Article Context
        article_lines = []
        for idx, a in enumerate(articles, 1):
            title = a.get("title", "News Article")
            url = a.get("url", "")
            date = a.get("published_at", "Recent")
            snippet = a.get("snippet", "")
            article_lines.append(f"[{idx}] Title: {title}\n    Date: {date}\n    URL: {url}\n    Excerpt: {snippet}")

        articles_context = "\n\n".join(article_lines) if article_lines else "No supporting articles found."

        system_prompt = """You are YarnBall, an expert financial intelligence assistant.
Your goal is to answer the user's financial question based STRICTLY and ONLY on the provided graph facts and news articles.

ANTI-HALLUCINATION GUIDELINES:
1. Every factual statement must be directly substantiated by the provided Graph Relationships or News Articles.
2. For every fact mentioned, provide a markdown citation pointing to the supporting article: `[Article Title](URL)` or `[Source: Title]`.
3. If the context does not contain sufficient facts to answer the question, state clearly: "Based on the current financial knowledge graph, no verified relationships were found for this query." Do NOT speculate or invent connections.
4. Keep the answer structured, concise, and professional."""

        user_prompt = f"""USER QUESTION:
{query}

VERIFIED GRAPH RELATIONSHIPS:
{graph_context}

SUPPORTING NEWS CONTEXT & CITATIONS:
{articles_context}

Synthesize a grounded answer with citations:"""

        return system_prompt, user_prompt

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call local Ollama model for answer generation."""
        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed; returning fallback response")
            return "Based on the current financial knowledge graph, no verified relationships were found for this query."

        api_url = self.ollama_url
        if "/v1" in api_url:
            endpoint = api_url.rstrip("/") + "/chat/completions"
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.0,
                "stream": False,
            }
            try:
                with httpx.Client(timeout=45.0) as client:
                    resp = client.post(endpoint, json=payload)
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                logger.warning("Ollama /v1 API call failed: %s; trying /api/generate", exc)

        gen_url = self.ollama_url
        if "/v1" in gen_url:
            gen_url = gen_url.split("/v1")[0].rstrip("/") + "/api/generate"

        try:
            with httpx.Client(timeout=45.0) as client:
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                resp = client.post(
                    gen_url,
                    json={
                        "model": self.model_name,
                        "prompt": full_prompt,
                        "stream": False,
                    },
                )
                resp.raise_for_status()
                return resp.json().get("response", "").strip()
        except Exception as exc:
            logger.warning("Ollama generation request failed: %s", exc)
            return "Based on the current financial knowledge graph, no verified relationships were found for this query."

    def extract_citations(
        self, answer: str, articles: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Extract referenced citations and URLs from the generated answer and article set."""
        citations = []
        for a in articles:
            url = a.get("url", "")
            title = a.get("title", "")
            if url and (url in answer or title in answer or len(articles) <= 3):
                citations.append({
                    "title": title,
                    "url": url,
                    "published_at": a.get("published_at", ""),
                })
        return citations

    def synthesize(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize a grounded answer with citations from the retrieval context."""
        triples = context.get("subgraph_triples", [])
        cql_records = context.get("cql_result", {}).get("records", [])
        articles = context.get("articles", [])

        # Negative Grounding Check: If no graph relations and no articles exist
        if not triples and not cql_records and not articles:
            return {
                "query": query,
                "answer": "Based on the current financial knowledge graph, no verified relationships or news evidence were found for this query.",
                "is_grounded": False,
                "citations": [],
                "subgraph_triples": [],
            }

        sys_prompt, usr_prompt = self._build_synthesis_prompt(query, context)
        raw_answer = self._call_llm(sys_prompt, usr_prompt)
        citations = self.extract_citations(raw_answer, articles)

        return {
            "query": query,
            "answer": raw_answer,
            "is_grounded": True,
            "citations": citations,
            "subgraph_triples": triples,
            "cql_records": cql_records,
        }


class HybridGraphRAGEngine:
    """Unified end-to-end interface for Hybrid Retrieval and Grounded Synthesis."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        synthesizer: Optional[GroundedSynthesizer] = None,
    ):
        self.retriever = retriever or HybridRetriever()
        self.synthesizer = synthesizer or GroundedSynthesizer()

    def answer_query(self, query: str) -> Dict[str, Any]:
        """Execute the full Hybrid GraphRAG pipeline from natural language to grounded response."""
        context = self.retriever.retrieve(query)
        synthesis = self.synthesizer.synthesize(query, context)

        return {
            "query": query,
            "answer": synthesis["answer"],
            "is_grounded": synthesis["is_grounded"],
            "citations": synthesis["citations"],
            "cypher_query": context.get("cql_result", {}).get("cypher", ""),
            "subgraph_triples": context.get("subgraph_triples", []),
            "articles": context.get("articles", []),
            "retrieval_latency_ms": context.get("latency_ms", 0.0),
        }

    def query(self, query: str) -> Dict[str, Any]:
        """Alias for answer_query."""
        return self.answer_query(query)
