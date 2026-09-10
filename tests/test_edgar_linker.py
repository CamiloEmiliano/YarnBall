"""
Unit tests for EdgarEntityLinker and name normalization.
"""

import unittest
from unittest.mock import MagicMock

from ingest.edgar_linker import (
    EdgarEntityLinker,
    normalize_company_name,
)


class TestEdgarEntityLinker(unittest.TestCase):
    def test_normalize_company_name(self):
        self.assertEqual(normalize_company_name("Apple Inc."), "apple")
        self.assertEqual(normalize_company_name("Microsoft Corporation"), "microsoft")
        self.assertEqual(normalize_company_name("Alphabet Inc - Class A"), "alphabet a")
        self.assertEqual(normalize_company_name("NVIDIA CORP"), "nvidia")
        self.assertEqual(normalize_company_name("Amazon.com, Inc."), "amazon com")
        self.assertEqual(normalize_company_name("Taiwan Semiconductor Manufacturing Co., Ltd."), "taiwan semiconductor manufacturing")
        self.assertEqual(normalize_company_name(""), "")
        self.assertEqual(normalize_company_name(None), "")

    def test_tier1_ticker_resolution(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        # Mock query return for Tier 1 Ticker lookup
        mock_cursor.fetchone.return_value = (
            "0000320193", "AAPL", "Apple Inc.", "apple", "3571", "Electronic Computers", True
        )

        linker = EdgarEntityLinker(pg_conn=mock_conn)
        result = linker.link_entity(ticker="AAPL")

        self.assertIsNotNone(result)
        self.assertEqual(result["cik"], "0000320193")
        self.assertEqual(result["ticker"], "AAPL")
        self.assertEqual(result["match_tier"], "tier_1_ticker")
        self.assertEqual(result["match_score"], 1.0)
        self.assertTrue(result["is_sp500"])

    def test_tier2_exact_name_resolution(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        mock_cursor.fetchone.return_value = (
            "0001018724", "AMZN", "Amazon Com Inc", "amazon com", "5961", "Catalog & Mail-Order Houses", True
        )

        linker = EdgarEntityLinker(pg_conn=mock_conn)
        result = linker.link_entity(name="Amazon.com, Inc.")

        self.assertIsNotNone(result)
        self.assertEqual(result["cik"], "0001018724")
        self.assertEqual(result["match_tier"], "tier_2_exact_name")
        self.assertEqual(result["resolved_via"], "normalized_name")

    def test_tier3_trigram_fuzzy_resolution(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        # First call (Tier 2 exact name) returns None, second call (Tier 3 trigram) returns candidate
        mock_cursor.fetchone.side_effect = [
            None,  # Tier 2 exact match
            ("0001045810", "NVDA", "NVIDIA Corp", "nvidia", "3674", "Semiconductors", True, 0.92),  # Tier 3 trigram match
        ]

        linker = EdgarEntityLinker(pg_conn=mock_conn, trigram_threshold=0.85)
        result = linker.link_entity(name="Nvidia Technologies")

        self.assertIsNotNone(result)
        self.assertEqual(result["ticker"], "NVDA")
        self.assertEqual(result["match_tier"], "tier_3_trigram_fuzzy")
        self.assertAlmostEqual(result["match_score"], 0.92)

    def test_tier4_subsidiary_resolution(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        mock_cursor.fetchone.side_effect = [
            None,  # Tier 2 exact match
            None,  # Tier 3 trigram match
            ("0001652044", "GOOGL", "Alphabet Inc.", "alphabet", "7370", "Services-Computer Programming", True, "DeepMind Technologies Ltd"),  # Tier 4 subsidiary
        ]

        linker = EdgarEntityLinker(pg_conn=mock_conn)
        result = linker.link_entity(name="DeepMind Technologies Ltd")

        self.assertIsNotNone(result)
        self.assertEqual(result["cik"], "0001652044")
        self.assertEqual(result["ticker"], "GOOGL")
        self.assertEqual(result["match_tier"], "tier_4_subsidiary")
        self.assertIn("DeepMind Technologies Ltd", result["resolved_via"])

    def test_caching_behavior(self):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        mock_cursor.fetchone.return_value = (
            "0000320193", "AAPL", "Apple Inc.", "apple", "3571", "Electronic Computers", True
        )

        linker = EdgarEntityLinker(pg_conn=mock_conn)
        res1 = linker.link_entity(ticker="AAPL")
        res2 = linker.link_entity(ticker="AAPL")

        self.assertEqual(res1, res2)
        # Database query should only run once due to cache
        self.assertEqual(mock_cursor.execute.call_count, 1)


if __name__ == "__main__":
    unittest.main()
