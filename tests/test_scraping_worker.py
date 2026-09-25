# tests/test_scraping_worker.py
# -*- coding: utf-8 -*-
"""Unit tests for ingest/scraping_worker.py."""

import unittest
from unittest.mock import MagicMock, patch

from ingest.scraping_worker import (
    compute_source_hash,
    extract_text_from_html,
    get_domain_from_url,
    is_domain_blacklisted,
    is_paywall_or_stub,
    process_scraping_message,
    record_domain_failure,
    record_domain_success,
)


class ScrapingWorkerTests(unittest.TestCase):
    def test_get_domain_from_url(self):
        self.assertEqual(get_domain_from_url("https://finance.yahoo.com/news/article-123.html"), "finance.yahoo.com")
        self.assertEqual(get_domain_from_url("http://www.bloomberg.com/news/articles/2026"), "bloomberg.com")
        self.assertEqual(get_domain_from_url("https://sub.domain.co.uk:8080/path"), "sub.domain.co.uk")
        self.assertEqual(get_domain_from_url(""), "")

    def test_compute_source_hash(self):
        hash1 = compute_source_hash("https://example.com/news/1")
        hash2 = compute_source_hash("https://example.com/news/1")
        hash3 = compute_source_hash("https://example.com/news/2")
        self.assertEqual(hash1, hash2)
        self.assertNotEqual(hash1, hash3)
        self.assertEqual(len(hash1), 32)

    def test_is_paywall_or_stub(self):
        self.assertTrue(is_paywall_or_stub(None))
        self.assertTrue(is_paywall_or_stub(""))
        # Too short (< 40 words)
        short_text = "This is a very short preview of a financial article."
        self.assertTrue(is_paywall_or_stub(short_text))

        # Paywall phrase trigger
        long_paywalled_text = (
            "Apple announced record quarterly earnings today beating all Wall Street estimates. " * 5
            + "Subscribe to continue reading full market analysis and forecast."
        )
        self.assertTrue(is_paywall_or_stub(long_paywalled_text))

        # Valid full article
        valid_text = (
            "Apple announced record quarterly earnings today beating all Wall Street estimates. "
            "Revenue reached $120 billion driven by strong iPhone 17 and Vision Pro sales. "
            "CEO Tim Cook highlighted emerging market growth across India and Southeast Asia. "
            "Operating margin expanded by 150 basis points year over year. "
            "Analysts raised their price targets to $280 citing AI services adoption."
        )
        self.assertFalse(is_paywall_or_stub(valid_text))

    def test_extract_text_from_html(self):
        sample_html = """
        <html>
            <head><title>Test Article</title></head>
            <body>
                <script>var x = 1;</script>
                <h1>Headline</h1>
                <p>This is paragraph one of the article body with financial figures and facts.</p>
                <p>This is paragraph two providing additional context on market earnings.</p>
            </body>
        </html>
        """
        extracted = extract_text_from_html(sample_html)
        self.assertIsNotNone(extracted)
        self.assertIn("This is paragraph one", extracted)
        self.assertNotIn("var x = 1", extracted)

    def test_is_domain_blacklisted(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        # When status is blacklisted
        mock_cur.fetchone.return_value = ("blacklisted",)
        self.assertTrue(is_domain_blacklisted("bloomberg.com", conn=mock_conn))

        # When status is allowed
        mock_cur.fetchone.return_value = ("allowed",)
        self.assertFalse(is_domain_blacklisted("yahoo.com", conn=mock_conn))

        # When not in db
        mock_cur.fetchone.return_value = None
        self.assertFalse(is_domain_blacklisted("newdomain.com", conn=mock_conn))

    def test_record_domain_success(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        record_domain_success("yahoo.com", conn=mock_conn)
        mock_cur.execute.assert_called()
        mock_conn.commit.assert_called()

    def test_record_domain_failure(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        record_domain_failure("paywall.com", failure_threshold=3, conn=mock_conn)
        mock_cur.execute.assert_called()
        mock_conn.commit.assert_called()

    @patch("ingest.scraping_worker.fetch_and_extract")
    def test_process_scraping_message_success(self, mock_fetch):
        mock_fetch.return_value = (
            "A long valid article text about tech earnings with plenty of words and facts... " * 10,
            {},
        )
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_cur.fetchone.return_value = None  # domain not blacklisted

        msg = {
            "url": "https://finance.yahoo.com/news/aapl-earnings.html",
            "title": "Apple Q3 Earnings Surge",
            "published": 1772409600,
            "ticker_symbols": ["AAPL"],
        }
        result = process_scraping_message(msg, conn=mock_conn)
        self.assertTrue(result)

    @patch("ingest.scraping_worker.fetch_and_extract")
    def test_process_scraping_message_paywall_fails(self, mock_fetch):
        mock_fetch.return_value = (None, {})
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_cur.fetchone.return_value = None  # not blacklisted yet

        msg = {
            "url": "https://wsj.com/news/subscriber-only-analysis.html",
            "title": "WSJ Exclusive Analysis",
        }
        result = process_scraping_message(msg, conn=mock_conn)
        self.assertFalse(result)

    def test_process_scraping_message_blacklisted_domain_skipped(self):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_cur.fetchone.return_value = ("blacklisted",)

        msg = {
            "url": "https://blocked-paywall.com/news/story.html",
            "title": "Blocked Story",
        }
        result = process_scraping_message(msg, conn=mock_conn)
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
