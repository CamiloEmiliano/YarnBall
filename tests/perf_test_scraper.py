# tests/perf_test_scraper.py
# -*- coding: utf-8 -*-
"""
Performance test for the scraping worker (ingest/scraping_worker.py).

* Pulls a few news items from Finnhub (real API call)
* Sends them to the Kafka topic used by the scraper
* Runs the scraper’s `process_scraping_message` on a limited number of messages
* Reports total time and throughput (records / second).

The test stops after `MAX_RECORDS` items – it never reaches the retriever or Memgraph stages.
"""

import json
import os
import time
from typing import List

# Finnhub client helpers
from ingest.finnhub_client import (
    _fetch_json_if_available,
    _finnhub_params,
    FINNHUB_TICKERS,
)

# Kafka producer (uses the same driver the scheduler uses)
from kafka_pipeline.kafka_driver import send_record

# Scraper processing function (the code we want to evaluate)
from ingest.scraping_worker import process_scraping_message

# ----------------------------------------------------------------------
# Configuration (adjust via environment or edit here)
# ----------------------------------------------------------------------
MAX_TICKERS = int(os.getenv("PERF_MAX_TICKERS", "2"))      # how many tickers to fetch
MAX_PER_TICKER = int(os.getenv("PERF_MAX_PER_TICKER", "5"))  # how many news items per ticker
MAX_RECORDS = int(os.getenv("PERF_MAX_RECORDS", "10"))       # total records to process

# ----------------------------------------------------------------------
def fetch_finnhub_items() -> List[dict]:
    """Fetch a handful of news items from Finnhub for the configured tickers."""
    items: List[dict] = []
    tickers = FINNHUB_TICKERS or ["AAPL"]
    for ticker in tickers[:MAX_TICKERS]:
        data = _fetch_json_if_available(
            "Finnhub",
            "https://finnhub.io/api/v1/company-news",
            params=_finnhub_params(ticker),
        )
        if not data:
            continue
        # Keep only items that contain a URL and respect the limits
        url_items = [itm for itm in data[:MAX_PER_TICKER] if itm.get('url')]
        items.extend(url_items)
        if len(items) >= MAX_RECORDS:
            break
    return items[:MAX_RECORDS]

# ----------------------------------------------------------------------
def produce_to_kafka(items: List[dict]) -> None:
    """Push the raw Finnhub payloads onto the Kafka topic the scraper reads."""
    for item in items:
        payload = json.dumps(item)
        # The `send_record` helper mirrors what the scheduler does:
        #   source = "Finnhub", payload = JSON string
        send_record(source="Finnhub", payload=payload)

# ----------------------------------------------------------------------
def run_scraper_test(items: List[dict]) -> None:
    """Feed the same payloads directly to the scraper (bypassing the consumer) and time it."""
    start = time.time()
    success = 0
    for item in items:
        # The scraper expects a dict with the same shape the Kafka consumer yields:
        # {"_source": "Finnhub", "raw_payload": "<json string>"}
        msg = {"_source": "Finnhub", "raw_payload": json.dumps(item)}
        if process_scraping_message(msg):
            success += 1
    elapsed = time.time() - start
    print("\n--- Scraper performance report ---")
    print(f"Processed {success}/{len(items)} records")
    print(f"Total time: {elapsed:.2f}s")
    if elapsed > 0:
        print(f"Throughput: {success/elapsed:.2f} records / second")
    else:
        print("Throughput: ∞ (instant)")

# ----------------------------------------------------------------------
def main() -> None:
    print("🔎 Fetching Finnhub news items …")
    items = fetch_finnhub_items()
    if not items:
        print("No items retrieved - check FINNHUB_API_KEY / FINNHUB_TICKERS in .env")
        return

    print(f"Got {len(items)} items - pushing to Kafka …")
    produce_to_kafka(items)

    # Give the scraper a moment to see the messages (optional)
    time.sleep(2)

    print("Running scraper on the same payloads (direct call) …")
    run_scraper_test(items)

if __name__ == "__main__":
    main()
