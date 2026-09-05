import json
from datetime import datetime, timezone
from unittest import TestCase, mock

from tools.utils import insert_raw_message


class InsertRawMessageTests(TestCase):
    def setUp(self):
        # Mock DB connection and cursor
        self.mock_cursor = mock.MagicMock()
        self.mock_conn = mock.MagicMock()
        # Ensure the mock connection works as a context manager and returns a cursor that also works as a context manager
        self.mock_conn.__enter__.return_value = self.mock_conn
        self.mock_conn.cursor.return_value.__enter__.return_value = (
            self.mock_cursor
        )
        self.mock_conn.__exit__.return_value = None
        self.mock_cursor.__exit__.return_value = None
        self.patcher = mock.patch(
            "tools.utils._db_connection", lambda: self.mock_conn
        )
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_full_payload_is_inserted(self):
        payload = json.dumps(
            {
                "title": "Markets surge",
                "url": "https://example.com/article",
                "published": "2026-08-08T12:00:00Z",
                "author": "John Doe",
                "category": "Finance",
                "tags": ["stocks", "earnings"],
                "ticker_symbols": ["AAPL", "MSFT"],
                "sentiment": 0.92,
                "source": "NewsAPI",
            }
        )
        fetched = datetime(2026, 8, 8, 12, 1, tzinfo=timezone.utc)
        hash_ = "abc123"

        insert_raw_message(
            raw_payload=payload, fetched_at=fetched, payload_hash=hash_
        )

        # Verify SQL contains new column names
        executed_sql = self.mock_cursor.execute.call_args[0][0]
        for col in [
            "source_url",
            "title",
            "published_at",
            "author",
            "category",
            "tags",
            "ticker_symbols",
            "sentiment_score",
            "provider",
        ]:
            self.assertIn(col, executed_sql)

        # Verify values order matches the INSERT statement
        values = self.mock_cursor.execute.call_args[0][1]
        self.assertEqual(values[3], hash_)  # source_hash
        self.assertEqual(values[4], "https://example.com/article")  # source_url
        self.assertEqual(values[5], "Markets surge")  # title
        self.assertEqual(
            values[6], datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
        )  # published_at
        self.assertEqual(values[7], "John Doe")  # author
        self.assertEqual(values[8], "Finance")  # category
        self.assertEqual(json.loads(values[9]), ["stocks", "earnings"])  # tags
        self.assertEqual(
            json.loads(values[10]), ["AAPL", "MSFT"]
        )  # ticker_symbols
        self.assertEqual(values[11], 0.92)  # sentiment_score
        self.assertEqual(values[12], "NewsAPI")  # provider

    def test_missing_optional_fields(self):
        payload = json.dumps(
            {"title": "Only title", "url": "https://example.com/minimal"}
        )
        fetched = datetime.now(timezone.utc)
        insert_raw_message(
            raw_payload=payload, fetched_at=fetched, payload_hash="hash"
        )
        values = self.mock_cursor.execute.call_args[0][1]
        # source_url is taken from payload url, title is present, others should be None
        self.assertEqual(values[4], "https://example.com/minimal")
        self.assertIsNone(values[6])  # published_at
        self.assertIsNone(values[7])  # author
        self.assertIsNone(values[8])  # category
        self.assertIsNone(values[9])  # tags
        self.assertIsNone(values[10])  # ticker_symbols
        self.assertIsNone(values[11])  # sentiment_score
        self.assertIsNone(values[12])  # provider
