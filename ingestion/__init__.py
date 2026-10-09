"""
Unified Data Ingestion Subsystem for YarnBall.

Provides tripartite (client -> parser -> harvester) ingestion engines across four core financial domains:
- `sec`: Regulatory filings (Form 10-K, 10-Q, 8-K, Exhibit 21) via SEC EDGAR
- `news`: Live streaming news (Finnhub) & historical archives (FNSPID, Form 8-K releases)
- `transcripts`: Corporate quarterly earnings call transcripts & Form 8-K Item 2.02
- `market`: Historical daily OHLCV prices, CAR metrics, and volatility z-scores (Yahoo Finance)
"""

from typing import Callable, Dict

# Central task registry mapping task names to fetch functions
registry: Dict[str, Callable] = {}


def ingest_task(name: str):
    """Decorator that registers an ingestion function in the central registry."""
    def wrapper(fn: Callable) -> Callable:
        registry[name] = fn
        return fn
    return wrapper


from .sec import (
    DEFAULT_SP500_BENCHMARK,
    EdgarClient,
    EdgarEntityLinker,
    EdgarWorker,
    HistoricalSECHarvester,
    normalize_company_name,
)
from .news import (
    HistoricalNewsIngestor,
    MultiSourceNewsHarvester,
    TokenBucket,
    compute_source_hash,
    strip_publisher_boilerplates,
)
from .transcripts import (
    EarningsTranscriptIngestor,
    EarningsTranscriptParser,
    TranscriptSourceClient,
)
from .market import (
    MarketContextIntegrator,
    MarketMetricsCalculator,
    YahooFinanceClient,
)

__all__ = [
    "registry",
    "ingest_task",
    # SEC
    "EdgarClient",
    "DEFAULT_SP500_BENCHMARK",
    "EdgarEntityLinker",
    "EdgarWorker",
    "HistoricalSECHarvester",
    "normalize_company_name",
    # News
    "TokenBucket",
    "MultiSourceNewsHarvester",
    "HistoricalNewsIngestor",
    "strip_publisher_boilerplates",
    "compute_source_hash",
    # Transcripts
    "EarningsTranscriptIngestor",
    "EarningsTranscriptParser",
    "TranscriptSourceClient",
    # Market
    "YahooFinanceClient",
    "MarketMetricsCalculator",
    "MarketContextIntegrator",
]
