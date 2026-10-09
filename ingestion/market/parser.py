"""
Market Context & Event-Window Volatility Metric Calculator.

Computes:
- Event-window return shock [T - k, T + k]
- Benchmark market return and Cumulative Abnormal Return (CAR)
- Daily return variance and volatility z-score scaling
- 5-Axis directional polarity assignment
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ingestion.market.parser")


class MarketMetricsCalculator:
    """Calculates CAR, volatility z-scores, and directional polarities."""

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

        # Benchmark market return
        market_return = 0.0
        if bench_to_price:
            if sorted_dates[pre_idx] in bench_to_price and sorted_dates[post_idx] in bench_to_price:
                b_pre = bench_to_price[sorted_dates[pre_idx]]
                b_post = bench_to_price[sorted_dates[post_idx]]
                market_return = (b_post - b_pre) / max(0.0001, b_pre)

        car = raw_return - market_return

        # Daily return baseline volatility for z-score scaling
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
            daily_vol = 0.02

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
