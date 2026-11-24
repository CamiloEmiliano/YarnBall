"""
Grounded Synthesis Engine (LLM Reasoner & Anti-Hallucination Citation Enforcer).

Synthesizes factual financial intelligence grounded strictly in retrieved
graph relationships and verified article disclosures via `YarnBallModelClient`
and logs refusals / transport failures to `RuntimeRejectionQueue` (ADR-00030 / ADR-00034).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from sft_pipeline.rejection_queue import ErrorStage, RuntimeRejectionQueue
from sft_pipeline.serving.model_client import ModelResponse, YarnBallModelClient

logger = logging.getLogger("synthesis.synthesizer")

NO_VERIFIED_EVIDENCE_MESSAGE = (
    "Based on the current financial knowledge graph, no verified relationships "
    "or news evidence were found for this query."
)


class GroundedSynthesizer:
    """Synthesizes factual answers grounded strictly in retrieved graph & article context."""

    def __init__(
        self,
        model_client: Optional[YarnBallModelClient] = None,
        rejection_queue: Optional[RuntimeRejectionQueue] = None,
    ) -> None:
        self.model_client = model_client or YarnBallModelClient()
        self.rejection_queue = rejection_queue or RuntimeRejectionQueue()

    def _build_synthesis_prompt(
        self, query: str, context: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Construct the grounded prompt with verified facts and anti-hallucination rules."""
        triples = context.get("subgraph_triples", [])
        cql_records = context.get("cql_result", {}).get("records", [])
        articles = context.get("articles", [])

        graph_lines: List[str] = []
        for t in triples[:15]:
            rel = t.get("relation") or t.get("rel_type") or "RELATED_TO"
            graph_lines.append(f"- ({t['source']}) -[{rel}]-> ({t['target']})")
        for r in cql_records[:10]:
            graph_lines.append(f"- Structured Result: {json.dumps(r)}")

        graph_context = "\n".join(graph_lines) if graph_lines else "No direct graph relationships found."

        article_lines: List[str] = []
        for idx, a in enumerate(articles, 1):
            title = a.get("title", "News Article")
            url = a.get("url", "")
            pub_date = a.get("published_at", "Recent")
            snippet = a.get("snippet", "")
            article_lines.append(
                f"[{idx}] Title: {title}\n    Date: {pub_date}\n    URL: {url}\n    Excerpt: {snippet}"
            )

        articles_context = "\n\n".join(article_lines) if article_lines else "No supporting articles found."

        system_prompt = (
            "You are YarnBall, an expert financial intelligence assistant.\n"
            "Your goal is to answer the user's financial question based STRICTLY and ONLY on the "
            "provided graph facts and news articles.\n\n"
            "ANTI-HALLUCINATION GUIDELINES:\n"
            "1. Every factual statement must be directly substantiated by the provided Graph Relationships "
            "or News Articles.\n"
            "2. For every fact mentioned, provide a markdown citation pointing to the supporting article: "
            "`[Article Title](URL)` or `[Source: Title]`.\n"
            "3. If the context does not contain sufficient facts to answer the question, state clearly: "
            '"Based on the current financial knowledge graph, no verified relationships were found for this query." '
            "Do NOT speculate or invent connections.\n"
            "4. Keep the answer structured, concise, and professional."
        )

        user_prompt = (
            f"USER QUESTION:\n{query}\n\n"
            f"VERIFIED GRAPH RELATIONSHIPS:\n{graph_context}\n\n"
            f"SUPPORTING NEWS CONTEXT & CITATIONS:\n{articles_context}\n\n"
            f"Synthesize a grounded answer with citations:"
        )

        return system_prompt, user_prompt

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Tuple[str, bool]:
        """Execute grounded synthesis strictly through `YarnBallModelClient` (ADR-00034)."""
        resp: ModelResponse = self.model_client.generate(
            prompt=user_prompt,
            task="GROUNDED_SYNTHESIS",
            system_prompt=system_prompt,
            temperature=0.0,
            enforce_think_prefix=False,
        )
        if resp.error:
            logger.warning("YarnBallModelClient grounded synthesis failed: %s", resp.error)
            self.rejection_queue.record_failure(
                task_id=f"grounded_synthesis_{int(time.time())}",
                prompt=user_prompt,
                completion=resp.raw_text or resp.final_output,
                error_stage=ErrorStage.LLM_UNAVAILABLE,
                error_message=f"Grounded synthesis transport error: {resp.error}",
            )
            return NO_VERIFIED_EVIDENCE_MESSAGE, False

        cleaned = resp.final_output.strip()
        if not cleaned or cleaned == "(none)":
            self.rejection_queue.record_failure(
                task_id=f"grounded_synthesis_refusal_{int(time.time())}",
                prompt=user_prompt,
                completion=resp.raw_text or cleaned,
                error_stage=ErrorStage.EPISTEMIC_HEDGING,
                error_message="Grounded synthesis returned empty or refusal output.",
            )
            return NO_VERIFIED_EVIDENCE_MESSAGE, False

        return cleaned, True

    def extract_citations(
        self, answer: str, articles: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Extract referenced citations and URLs from the generated answer and article set."""
        citations: List[Dict[str, str]] = []
        for a in articles:
            url = str(a.get("url", ""))
            title = str(a.get("title", ""))
            if url and (url in answer or title in answer or len(articles) <= 3):
                citations.append(
                    {
                        "title": title,
                        "url": url,
                        "published_at": str(a.get("published_at", "")),
                    }
                )
        return citations

    def synthesize(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize a grounded answer with citations from the retrieval context."""
        triples = context.get("subgraph_triples", [])
        cql_records = context.get("cql_result", {}).get("records", [])
        articles = context.get("articles", [])

        if not triples and not cql_records and not articles:
            self.rejection_queue.record_failure(
                task_id=f"grounded_synthesis_empty_{int(time.time())}",
                prompt=query,
                completion="(none)",
                error_stage=ErrorStage.EPISTEMIC_HEDGING,
                error_message="Negative grounding refusal: zero verified graph triples or articles.",
            )
            return {
                "query": query,
                "answer": NO_VERIFIED_EVIDENCE_MESSAGE,
                "is_grounded": False,
                "citations": [],
                "subgraph_triples": [],
            }

        sys_prompt, usr_prompt = self._build_synthesis_prompt(query, context)
        raw_answer, ok = self._call_llm(sys_prompt, usr_prompt)
        citations = self.extract_citations(raw_answer, articles) if ok else []

        return {
            "query": query,
            "answer": raw_answer,
            "is_grounded": ok,
            "citations": citations,
            "subgraph_triples": triples,
            "cql_records": cql_records,
        }
