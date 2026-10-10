"""
Generative / Neural Grounded Reasoning & Evaluation Package.

Submodules:
- `synthesizer`: `GroundedSynthesizer` (anti-hallucination citation enforcer)
- `engine`: `FinancialAnalysisEngine`, `HybridGraphRAGEngine` (end-to-end coordinator)
- `eval_harness`: `EvaluationHarness`, `BENCHMARK_QUERIES` (golden multi-hop benchmark)
"""

from .synthesizer import GroundedSynthesizer
from .engine import FinancialAnalysisEngine, HybridGraphRAGEngine
from .eval_harness import EvaluationHarness, BENCHMARK_QUERIES

__all__ = [
    "GroundedSynthesizer",
    "FinancialAnalysisEngine",
    "HybridGraphRAGEngine",
    "EvaluationHarness",
    "BENCHMARK_QUERIES",
]
