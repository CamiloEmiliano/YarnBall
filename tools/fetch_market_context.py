"""
Market Context & Event-Window Volatility Grounding Tool.

Fetches historical OHLCV data across S&P 500 constituents and computes
event-window price shock and Cumulative Abnormal Return (CAR) metrics
around corporate disclosure dates (SEC filings, earnings calls, M&A releases).
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

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from tools.sp500_universe import SP500UniverseManager

logger = logging.getLogger("market_context")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

DEFAULT_MARKET_DIR = Path(__file__).resolve().parent.parent / "data" / "market_context"


class MarketContextIntegrator:
    """Computes event-window price dynamics and returns for knowledge graph grounding."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        universe_mgr: Optional[SP500UniverseManager] = None,
    ):
        self.output_dir = output_dir or DEFAULT_MARKET_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.universe_mgr = universe_mgr or SP500UniverseManager()

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
                # Flatten potential multi-index columns in modern yfinance
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

    def compute_event_window_metrics(
        self,
        ticker: str,
        event_date_str: str,
        price_history: List[Dict[str, Any]],
        benchmark_history: Optional[List[Dict[str, Any]]] = None,
        window_days: int = 3,
    ) -> Dict[str, Any]:
        """
        Calculate event window return and Cumulative Abnormal Return (CAR) around event date.
        Window: [T - window_days, T + window_days].
        """
        clean_ticker = ticker.strip().upper()
        if not price_history:
            return {
                "ticker": clean_ticker,
                "event_date": event_date_str,
                "window_days": window_days,
                "window_truncated": True,
                "event_return_pct": 0.0,
                "car_abnormal_return_pct": 0.0,
                "volatility_zscore": 0.0,
                "polarity_sentiment": "NEUTRAL_STABLE",
            }

        date_to_price = {p["date"]: p["close"] for p in price_history}
        bench_to_price = {p["date"]: p["close"] for p in (benchmark_history or [])}

        sorted_dates = sorted(date_to_price.keys())
        if event_date_str not in date_to_price:
            # Find closest date
            closest_dates = [d for d in sorted_dates if d <= event_date_str]
            anchor_date = closest_dates[-1] if closest_dates else sorted_dates[0]
        else:
            anchor_date = event_date_str

        anchor_idx = sorted_dates.index(anchor_date)
        window_truncated = (anchor_idx - window_days < 0) or (anchor_idx + window_days >= len(sorted_dates))

        pre_idx = max(0, anchor_idx - window_days)
        post_idx = min(len(sorted_dates) - 1, anchor_idx + window_days)

        p_pre = date_to_price[sorted_dates[pre_idx]]
        p_post = date_to_price[sorted_dates[post_idx]]

        raw_return = (p_post - p_pre) / max(0.0001, p_pre)

        # Compute benchmark market return
        market_return = 0.0
        if bench_to_price:
            if sorted_dates[pre_idx] in bench_to_price and sorted_dates[post_idx] in bench_to_price:
                b_pre = bench_to_price[sorted_dates[pre_idx]]
                b_post = bench_to_price[sorted_dates[post_idx]]
                market_return = (b_post - b_pre) / max(0.0001, b_pre)

        car = raw_return - market_return

        # Compute daily return baseline volatility for z-score scaling
        daily_returns: List[float] = []
        for i in range(1, len(sorted_dates)):
            p_prev = date_to_price[sorted_dates[i - 1]]
            p_curr = date_to_price[sorted_dates[i]]
            if p_prev > 0:
                daily_returns.append((p_curr - p_prev) / p_prev)

        if len(daily_returns) >= 2:
            mean_ret = sum(daily_returns) / len(daily_returns)
            variance = sum((r - mean_ret) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
            daily_vol = math.sqrt(max(1e-8, variance))
        else:
            daily_vol = 0.02  # Fallback 2% daily volatility baseline

        k_days = max(1, post_idx - pre_idx)
        window_vol = daily_vol * math.sqrt(k_days)
        volatility_zscore = car / window_vol if window_vol > 0 else 0.0

        # Assign 5-Axis directional polarity scaled by volatility z-score
        if volatility_zscore >= 2.0:
            polarity = "EXPANDING_BULLISH"
        elif volatility_zscore <= -3.0:
            polarity = "DISRUPTIVE_SHOCK"
        elif volatility_zscore <= -1.5:
            polarity = "CONTRACTING_BEARISH"
        else:
            polarity = "NEUTRAL_STABLE"

        return {
            "ticker": clean_ticker,
            "event_date": event_date_str,
            "window_days": window_days,
            "window_truncated": window_truncated,
            "event_return_pct": round(raw_return * 100.0, 2),
            "car_abnormal_return_pct": round(car * 100.0, 2),
            "volatility_zscore": round(volatility_zscore, 2),
            "polarity_sentiment": polarity,
        }

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
