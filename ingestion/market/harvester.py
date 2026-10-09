"""
Market Context & Event-Window Volatility Harvester.

Orchestrates:
- Fetching historical OHLCV data across S&P 500 constituents
- Computing event-window Cumulative Abnormal Return (CAR) and volatility z-scores
- Staging structured market context tables into Parquet archives
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from tools.sp500_universe import SP500UniverseManager
from .client import YahooFinanceClient
from .parser import MarketMetricsCalculator

logger = logging.getLogger("ingestion.market.harvester")

DEFAULT_MARKET_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "market_context"


class MarketContextIntegrator:
    """Computes event-window price dynamics and returns for knowledge graph grounding."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        universe_mgr: Optional[SP500UniverseManager] = None,
        client: Optional[YahooFinanceClient] = None,
        calculator: Optional[MarketMetricsCalculator] = None,
    ):
        self.output_dir = Path(output_dir or DEFAULT_MARKET_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.universe_mgr = universe_mgr or SP500UniverseManager()
        self.client = client or YahooFinanceClient()
        self.calculator = calculator or MarketMetricsCalculator()

    def fetch_historical_ohlcv(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> List[Dict[str, Any]]:
        """Fetch daily price history using client."""
        return self.client.fetch_historical_ohlcv(ticker, start_date, end_date)

    def compute_event_window_metrics(
        self,
        ticker: str,
        event_date_str: str,
        price_history: List[Dict[str, Any]],
        benchmark_history: Optional[List[Dict[str, Any]]] = None,
        window_days: int = 3,
    ) -> Dict[str, Any]:
        """Delegate metric calculation to calculator."""
        return self.calculator.compute_event_window_metrics(
            ticker=ticker,
            event_date_str=event_date_str,
            price_history=price_history,
            benchmark_history=benchmark_history,
            window_days=window_days,
        )

    def stage_market_context_to_parquet(
        self,
        records: List[Dict[str, Any]],
        filename: str = "sp500_market_context.parquet",
    ) -> Path:
        """Stage market context records to Parquet archive."""
        schema = pa.schema([
            ("ticker", pa.string()),
            ("event_date", pa.string()),
            ("window_days", pa.int32()),
            ("window_truncated", pa.bool_()),
            ("event_return_pct", pa.float64()),
            ("car_abnormal_return_pct", pa.float64()),
            ("volatility_zscore", pa.float64()),
            ("polarity_sentiment", pa.string()),
        ])

        table = pa.Table.from_pylist(records, schema=schema)
        out_path = self.output_dir / filename
        pq.write_table(table, out_path, compression="snappy")
        logger.info(f"Staged {len(records)} market context rows to {out_path}")
        return out_path


def main() -> None:
    integrator = MarketContextIntegrator()
    sample_ticker = "AAPL"
    event_date = "2024-08-01"

    logger.info(f"Fetching market context for {sample_ticker} around {event_date}...")
    prices = [
        {"date": "2024-07-28", "open": 218.0, "high": 220.0, "low": 217.0, "close": 218.5, "volume": 50000000},
        {"date": "2024-08-01", "open": 219.0, "high": 222.0, "low": 218.0, "close": 220.0, "volume": 60000000},
        {"date": "2024-08-04", "open": 222.0, "high": 225.0, "low": 221.0, "close": 224.2, "volume": 55000000},
    ]

    metrics = integrator.compute_event_window_metrics(sample_ticker, event_date, prices)
    logger.info(f"Computed metrics: {metrics}")

    p_path = integrator.stage_market_context_to_parquet([metrics])
    logger.info(f"Output saved to {p_path}")


if __name__ == "__main__":
    main()
