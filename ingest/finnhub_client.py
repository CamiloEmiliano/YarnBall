# ingest/finnhub_client.py
"""Finnhub ingestion client.

Provides `fetch_finnhub()` which respects the 60 calls/minute rate limit using `TokenBucket`.
"""
import os
import json
import time
import datetime
from threading import Lock
from typing import Any, List, Dict

from kafka_pipeline.kafka_driver import send_record
from tools.utils import logger
from . import ingest_task  # noqa: F401

# ------------------------------------------------------------------------------
class TokenBucket:
    """Token bucket rate limiter for API call quotas.
    Respects the free-tier 60 calls/minute limit by default.
    """
    def __init__(self, capacity: int = 60, refill_seconds: int = 60):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = capacity / float(refill_seconds)
        self.last_refill = time.monotonic()
        self.lock = Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        added = elapsed * self.refill_rate
        if added >= 1:
            self.tokens = min(self.capacity, self.tokens + added)
            self.last_refill = now

    def consume(self, n: int = 1) -> bool:
        with self.lock:
            self._refill()
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False

    def wait(self, n: int = 1) -> None:
        while not self.consume(n):
            time.sleep(0.2)

# ------------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# Load tickers from environment variable (comma‑separated)
FINNHUB_TICKERS = [t.strip() for t in os.getenv("FINNHUB_TICKERS", "").split(",") if t.strip()]
HIST_DATETIME_FORMAT = os.getenv("HIST_DATETIME_FORMAT", "%Y-%m-%d %H:%M:%S")

try:
    import httpx
except ImportError:
    class _Response:
        def __init__(self, data=None, status_code=200):
            self._data = data or {}
            self.status_code = status_code
        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP error {self.status_code}")
        def json(self):
            return self._data
    class _Client:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def get(self, url, params=None, headers=None, timeout=None):
            return _Response()
        def head(self, url, headers=None, timeout=None):
            return _Response()
    class httpx:
        Client = _Client
        HTTPError = Exception

try:
    from tenacity import retry, stop_after_attempt, wait_exponential
except ImportError:
    def retry(*dargs, **dkwargs):
        def decorator(fn):
            return fn
        return decorator
    def stop_after_attempt(n):
        return None
    def wait_exponential(*args, **kwargs):
        return None

# ------------------------------------------------------------------------------
def _date_range() -> dict[str, str]:
    """Return normalized start/end window values for upstream APIs.

    By default we fetch the last 30 days. Override with HIST_START and
    HIST_END environment variables (format %Y-%m-%d %H:%M:%S in UTC).
    """
    end_env = os.getenv("HIST_END")
    start_env = os.getenv("HIST_START")

    if end_env:
        end_dt = datetime.datetime.strptime(
            end_env, HIST_DATETIME_FORMAT
        ).replace(tzinfo=datetime.timezone.utc)
    else:
        end_dt = datetime.datetime.now(datetime.timezone.utc).replace(
            microsecond=0
        )

    if start_env:
        start_dt = datetime.datetime.strptime(
            start_env, HIST_DATETIME_FORMAT
        ).replace(tzinfo=datetime.timezone.utc)
    else:
        start_dt = (end_dt - datetime.timedelta(days=30)).replace(microsecond=0)

    return {
        "from_date": start_dt.date().isoformat(),
        "to_date": end_dt.date().isoformat(),
        "from_iso_utc": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "to_iso_utc": end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

# ------------------------------------------------------------------------------
def _finnhub_params(ticker: str) -> dict[str, Any]:
    dr = _date_range()
    return {
        "symbol": ticker,
        "from": dr["from_date"],
        "to": dr["to_date"],
        "token": os.getenv("FINNHUB_API_KEY"),
    }

# ------------------------------------------------------------------------------
def _process_finnhub(data: Any, ticker: str) -> List[dict[str, str]]:
    """Transform Finnhub raw items into the payload format expected downstream.
    Filters articles to the exact HIST_START and HIST_END timestamps if set.
    """
    results: List[dict[str, str]] = []
    if not isinstance(data, list):
        logger.warning(f"Unexpected Finnhub payload shape for {ticker}; expected list")
        return results

    start_env = os.getenv("HIST_START")
    end_env = os.getenv("HIST_END")
    start_ts = (
        datetime.datetime.strptime(start_env, HIST_DATETIME_FORMAT)
        .replace(tzinfo=datetime.timezone.utc)
        .timestamp()
        if start_env
        else 0
    )
    end_ts = (
        datetime.datetime.strptime(end_env, HIST_DATETIME_FORMAT)
        .replace(tzinfo=datetime.timezone.utc)
        .timestamp()
        if end_env
        else float("inf")
    )

    for item in data:
        if not isinstance(item, dict):
            continue
        item_ts = item.get("datetime", 0)
        # Apply exact timestamp window filtering
        if (start_env or end_env) and not (start_ts <= item_ts <= end_ts):
            continue

        payload = json.dumps({
            "title": item.get("headline"),
            "url": item.get("url"),
            "summary": item.get("summary"),
            "published": datetime.datetime.fromtimestamp(
                item_ts, tz=datetime.timezone.utc
            ).isoformat(),
            "ticker": ticker,
        })
        results.append({"source": "Finnhub", "payload": payload})
    return results

# ------------------------------------------------------------------------------
@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=20),
)
def _get_json(
    url: str,
    params: Dict[str, Any] | None = None,
    headers: Dict[str, str] | None = None,
) -> Any:
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(url, params=params or {}, headers=headers or {})
        resp.raise_for_status()
        return resp.json()

# ------------------------------------------------------------------------------
def _is_api_available(
    url: str,
    headers: Dict[str, str] | None = None,
) -> bool:
    """Check whether an API endpoint is reachable before the main request."""
    try:
        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            resp = client.head(url, headers=headers or {})
            if resp.status_code == 405:
                resp = client.get(url, headers=headers or {})
            return resp.status_code < 500
    except httpx.HTTPError as exc:
        logger.warning(f"API availability check failed for {url}: {exc}")
        return False

# ------------------------------------------------------------------------------
def _fetch_json_if_available(
    service_name: str,
    url: str,
    params: Dict[str, Any] | None = None,
    headers: Dict[str, str] | None = None,
) -> Any | None:
    return _get_json(url, params=params, headers=headers)

# ------------------------------------------------------------------------------
def _test_finnhub_endpoint(ticker: str) -> bool:
    """Verify the Finnhub endpoint is reachable before making a full request."""
    url = "https://finnhub.io/api/v1/company-news"
    return _is_api_available(url)

# ------------------------------------------------------------------------------
@ingest_task("finnhub")
def fetch_finnhub() -> None:
    """Entry point called by the task runner.
    Iterates over configured tickers, respects the rate limit, fetches data and pushes to Kafka.
    """
    if not os.getenv("FINNHUB_API_KEY"):
        logger.warning("FINNHUB_API_KEY not set - skipping Finnhub ingestion")
        return
    tickers = [t.strip() for t in os.getenv("FINNHUB_TICKERS", "").split(",") if t.strip()] or FINNHUB_TICKERS
    logger.info("Fetching Finnhub news for tickers: %s", ", ".join(tickers))
    bucket = TokenBucket()
    for ticker in tickers:
        bucket.wait()  # enforce 60 calls/minute limit
        try:
            data = _fetch_json_if_available(
                "Finnhub",
                "https://finnhub.io/api/v1/company-news",
                params=_finnhub_params(ticker),
            )
            if data is not None:
                records = _process_finnhub(data, ticker)
                logger.info("Fetched %d news records for %s", len(records), ticker)
                for record in records:
                    send_record(record["source"], record["payload"])
        except Exception as exc:
            logger.error(f"Finnhub fetch failed for {ticker}: {exc}")

# ------------------------------------------------------------------------------
def check_api() -> None:
    """Sanity check and verification function for Finnhub API connectivity and data."""
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        print("ERROR: FINNHUB_API_KEY not set. Set it in .env or the environment.")
        return

    tickers = [t.strip() for t in os.getenv("FINNHUB_TICKERS", "").split(",") if t.strip()] or FINNHUB_TICKERS
    if not tickers:
        print("ERROR: FINNHUB_TICKERS not set or empty. Provide a comma-separated list in .env.")
        return

    print(f"Checking Finnhub API for tickers: {', '.join(tickers)}")
    bucket = TokenBucket()
    for ticker in tickers:
        bucket.wait()
        if not _test_finnhub_endpoint(ticker):
            print(f"ERROR: Finnhub endpoint not responsive for ticker {ticker}")
            continue
        try:
            data = _fetch_json_if_available(
                "Finnhub",
                "https://finnhub.io/api/v1/company-news",
                params=_finnhub_params(ticker),
            )
            if data is None:
                print(f"WARNING: No data returned for ticker {ticker}")
                continue
            records = _process_finnhub(data, ticker)
            print(f"OK: Retrieved {len(records)} news items for ticker {ticker}")
            if records:
                print("First item preview:")
                first_payload = json.loads(records[0]["payload"])
                print(json.dumps(first_payload, indent=2))
        except Exception as exc:
            print(f"ERROR: Request for ticker {ticker} failed: {exc}")
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    check_api()

