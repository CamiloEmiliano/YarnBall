"""
Market Pricing, Volatility, and Event Returns Ingestion Subpackage.

Modules:
- `client`: `YahooFinanceClient` (OHLCV prices and benchmark synchronization)
- `parser`: `MarketMetricsCalculator` (CAR, volatility z-scores, return shocks)
- `harvester`: `MarketContextIntegrator` (constituent staging and Parquet export)
"""

from .client import YahooFinanceClient
from .parser import MarketMetricsCalculator
from .harvester import MarketContextIntegrator, DEFAULT_MARKET_DIR

__all__ = [
    "YahooFinanceClient",
    "MarketMetricsCalculator",
    "MarketContextIntegrator",
    "DEFAULT_MARKET_DIR",
]
