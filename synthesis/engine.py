"""
Financial Analysis Engine (Coordinator of Point-in-Time Retrieval & Grounded Synthesis).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from retrieval.hybrid import HybridRetriever
from retrieval.temporal import PiTContext
from .synthesizer import GroundedSynthesizer


class FinancialAnalysisEngine:
    """Unified end-to-end interface coordinating Point-in-Time Retrieval and Grounded Synthesis."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        synthesizer: Optional[GroundedSynthesizer] = None,
    ):
        self.retriever = retriever or HybridRetriever()
        self.synthesizer = synthesizer or GroundedSynthesizer()

    def answer_query(self, query: str, ctx: PiTContext) -> Dict[str, Any]:
        """Execute the full pipeline from natural language to grounded response as of `ctx` (ADR-00033)."""
        if not isinstance(ctx, PiTContext):
            raise TypeError(f"ctx must be a PiTContext instance, got {type(ctx).__name__}")
        context = self.retriever.retrieve(query, ctx=ctx)
        synthesis = self.synthesizer.synthesize(query, context)

        return {
            "query": query,
            "as_of": ctx.as_of_iso,
            "pit_mode": ctx.mode,
            "answer": synthesis["answer"],
            "is_grounded": synthesis["is_grounded"],
            "citations": synthesis["citations"],
            "cypher_query": context.get("cql_result", {}).get("cypher", ""),
            "subgraph_triples": context.get("subgraph_triples", []),
            "articles": context.get("articles", []),
            "retrieval_latency_ms": context.get("latency_ms", 0.0),
        }

    def query(self, query: str, ctx: PiTContext) -> Dict[str, Any]:
        """Alias for answer_query requiring explicit `PiTContext`."""
        return self.answer_query(query, ctx=ctx)


# Symmetrical and backward-compatible alias
HybridGraphRAGEngine = FinancialAnalysisEngine
