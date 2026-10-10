"""
Grounded Synthesis Engine (LLM Reasoner & Anti-Hallucination Citation Enforcer).

Synthesizes factual financial intelligence grounded strictly in retrieved
graph relationships and verified article disclosures.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("synthesis.synthesizer")

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
