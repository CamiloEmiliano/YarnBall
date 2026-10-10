"""
Financial Analysis Engine (Coordinator of Retrieval & Grounded Synthesis).
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from retrieval.hybrid import HybridRetriever
from .synthesizer import GroundedSynthesizer


class FinancialAnalysisEngine:
    """Unified end-to-end interface coordinating Hybrid Retrieval and Grounded Synthesis."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        synthesizer: Optional[GroundedSynthesizer] = None,
    ):
        self.retriever = retriever or HybridRetriever()
        self.synthesizer = synthesizer or GroundedSynthesizer()

    def answer_query(self, query: str) -> Dict[str, Any]:
        """Execute the full pipeline from natural language to grounded response."""
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


# Symmetrical and backward-compatible alias
HybridGraphRAGEngine = FinancialAnalysisEngine
