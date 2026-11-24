"""
Financial Analysis Engine (Deprecated Shim over `retrieval.facade.query_bi`).

Per ADR-00034 Section 4.4, `FinancialAnalysisEngine` and its alias
`HybridGraphRAGEngine` are retained for one release as a thin shim over
`retrieval.facade.query_bi` before removal.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
import warnings

from retrieval.facade import SubgraphStorePort, query_bi
from retrieval.hybrid import HybridSeeder
from retrieval.temporal import PiTContext
from sft_pipeline.serving.model_client import YarnBallModelClient
from .synthesizer import GroundedSynthesizer


class FinancialAnalysisEngine:
    """Deprecated one-release shim delegating directly to `retrieval.facade.query_bi` (ADR-00034)."""

    def __init__(
        self,
        store: Optional[SubgraphStorePort] = None,
        seeder: Optional[HybridSeeder] = None,
        model_client: Optional[YarnBallModelClient] = None,
        synthesizer: Optional[GroundedSynthesizer] = None,
    ) -> None:
        warnings.warn(
            "FinancialAnalysisEngine / HybridGraphRAGEngine are deprecated under ADR-00034 "
            "and will be removed in the next release; call retrieval.facade.query_bi directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.store = store
        self.seeder = seeder
        self.model_client = model_client
        self.synthesizer = synthesizer

    def answer_query(self, query: str, ctx: PiTContext) -> Dict[str, Any]:
        """Delegate query execution to `retrieval.facade.query_bi` strictly as of `ctx`."""
        if not isinstance(ctx, PiTContext):
            raise TypeError(f"ctx must be a PiTContext instance, got {type(ctx).__name__}")

        result = query_bi(
            query,
            ctx=ctx,
            store=self.store,
            seeder=self.seeder,
            model_client=self.model_client,
        )

        if self.synthesizer is not None:
            context = {
                "subgraph_triples": result.retrieved_subgraph,
                "cql_result": {"cypher": result.cypher_query or "", "records": result.retrieved_subgraph},
                "articles": result.articles,
            }
            synth = self.synthesizer.synthesize(query, context)
            answer = synth["answer"]
            is_grounded = bool(synth["is_grounded"])
            citations = list(synth["citations"])
        else:
            answer = result.contagion_verdict
            is_grounded = bool(result.is_valid_query and result.contagion_verdict != "(none)")
            citations = [
                {
                    "title": str(a.get("title", "")),
                    "url": str(a.get("url", "")),
                    "published_at": str(a.get("published_at", "")),
                }
                for a in result.articles
                if a.get("url")
            ]

        return {
            "query": query,
            "as_of": ctx.as_of_iso,
            "pit_mode": ctx.mode,
            "answer": answer,
            "is_grounded": is_grounded,
            "citations": citations,
            "cypher_query": result.cypher_query or "",
            "subgraph_triples": result.retrieved_subgraph,
            "seed_hits": result.seed_hits,
            "articles": result.articles,
            "retrieval_latency_ms": result.total_latency_ms,
        }

    def query(self, query: str, ctx: PiTContext) -> Dict[str, Any]:
        """Alias for answer_query requiring explicit `PiTContext`."""
        return self.answer_query(query, ctx=ctx)


# Deprecated one-release alias
HybridGraphRAGEngine = FinancialAnalysisEngine
