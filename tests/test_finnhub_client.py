import json
import os
import unittest
from unittest.mock import patch

from ingest import finnhub_client


class FinnhubClientContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.maxDiff = None

    def test_finnhub_processes_top_level_list_payload(self) -> None:
        payload = [
            {
                "headline": "AAPL launches new product",
                "url": "https://example.com/aapl",
                "summary": "Great news",
                "datetime": 1679920200,
            },
            "not-a-dict",
        ]

        with patch.dict(
            os.environ,
            {
                "HIST_START": "2023-03-27 00:00:00",
                "HIST_END": "2023-03-28 00:00:00",
            },
            clear=False,
        ):
            result = finnhub_client._process_finnhub(payload, "AAPL")

        self.assertEqual(len(result), 1)
        parsed = json.loads(result[0]["payload"])
        self.assertEqual(parsed["ticker"], "AAPL")
        self.assertEqual(parsed["title"], "AAPL launches new product")
        self.assertEqual(parsed["url"], "https://example.com/aapl")

    def test_finnhub_params_contract(self) -> None:
        with patch.dict(
            os.environ,
            {
                "HIST_START": "2026-01-01 00:00:00",
                "HIST_END": "2026-01-01 01:00:00",
                "FINNHUB_API_KEY": "test_key",
            },
            clear=False,
        ), patch.object(
            finnhub_client, "HIST_DATETIME_FORMAT", "%Y-%m-%d %H:%M:%S"
        ):
            params = finnhub_client._finnhub_params("AAPL")

        self.assertEqual(params["symbol"], "AAPL")
        self.assertEqual(params["from"], "2026-01-01")
        self.assertEqual(params["to"], "2026-01-01")
        self.assertEqual(params["token"], "test_key")

    def test_fetch_finnhub_sends_records(self) -> None:
        mock_data = [
            {
                "headline": "AAPL Report",
                "url": "https://example.com/report",
                "summary": "Financial report summary",
                "datetime": 1679920200,
            }
        ]
        with patch.dict(
            os.environ,
            {
                "FINNHUB_API_KEY": "test_key",
                "FINNHUB_TICKERS": "AAPL",
                "HIST_START": "2023-03-27 00:00:00",
                "HIST_END": "2023-03-28 00:00:00",
            },
            clear=False,
        ), patch.object(
            finnhub_client, "_fetch_json_if_available", return_value=mock_data
        ), patch.object(
            finnhub_client, "send_record"
        ) as send_record_mock:
            finnhub_client.fetch_finnhub()

        self.assertEqual(send_record_mock.call_count, 1)
        args = send_record_mock.call_args[0]
        self.assertEqual(args[0], "Finnhub")
        payload = json.loads(args[1])
        self.assertEqual(payload["ticker"], "AAPL")


if __name__ == "__main__":
    unittest.main()
