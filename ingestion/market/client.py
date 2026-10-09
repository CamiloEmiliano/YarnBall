"""
Yahoo Finance Market Pricing Transport Client.

Fetches daily OHLCV price histories for individual equities and benchmark indices (SPY).
Handles multi-index column flattening from modern yfinance formats.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ingestion.market.client")


class YahooFinanceClient:
    """Historical OHLCV data client wrapping yfinance."""

    def __init__(self, benchmark_ticker: str = "SPY"):
        self.benchmark_ticker = benchmark_ticker.strip().upper()

    def fetch_historical_ohlcv(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> List[Dict[str, Any]]:
        """Fetch daily price history using yfinance."""
        clean_ticker = ticker.strip().upper()
        prices: List[Dict[str, Any]] = []

        try:
            import yfinance as yf
            df = yf.download(clean_ticker, start=start_date, end=end_date, progress=False)
            if df is None or df.empty:
                return prices

            for idx, row in df.iterrows():
                d_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
                close_val = float(row["Close"].iloc[0] if hasattr(row["Close"], "iloc") else row["Close"])
                vol_val = float(row["Volume"].iloc[0] if hasattr(row["Volume"], "iloc") else row["Volume"])
                open_val = float(row["Open"].iloc[0] if hasattr(row["Open"], "iloc") else row["Open"])
                high_val = float(row["High"].iloc[0] if hasattr(row["High"], "iloc") else row["High"])
                low_val = float(row["Low"].iloc[0] if hasattr(row["Low"], "iloc") else row["Low"])

                prices.append({
                    "date": d_str,
                    "open": round(open_val, 4),
                    "high": round(high_val, 4),
                    "low": round(low_val, 4),
                    "close": round(close_val, 4),
                    "volume": vol_val,
                })
        except Exception as exc:
            logger.debug(f"Failed to fetch OHLCV for {clean_ticker}: {exc}")

        return prices

    def fetch_benchmark_ohlcv(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Fetch historical prices for benchmark index (default SPY)."""
        return self.fetch_historical_ohlcv(self.benchmark_ticker, start_date, end_date)
