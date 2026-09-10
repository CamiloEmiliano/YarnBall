"""
Unit tests for EdgarClient and 10-K section parsing.
"""

import unittest
from unittest.mock import MagicMock, patch

from ingest.edgar_client import EdgarClient, DEFAULT_SP500_BENCHMARK


class TestEdgarClient(unittest.TestCase):
    def test_client_init(self):
        client = EdgarClient(user_agent="TestAgent test@test.org")
        self.assertEqual(client.user_agent, "TestAgent test@test.org")

    def test_default_sp500_benchmark_list(self):
        self.assertGreaterEqual(len(DEFAULT_SP500_BENCHMARK), 50)
        self.assertIn("AAPL", DEFAULT_SP500_BENCHMARK)
        self.assertIn("MSFT", DEFAULT_SP500_BENCHMARK)
        self.assertIn("NVDA", DEFAULT_SP500_BENCHMARK)

    def test_parse_subsidiaries_text(self):
        client = EdgarClient()
        raw_text = """
        Subsidiaries of Apple Inc.
        Apple Operations International\tIreland
        Apple Operations Europe\tIreland
        Beats Electronics LLC\tDelaware, USA
        Claris International Inc.\tCalifornia, USA
        """
        subs = client._parse_subsidiaries_text(raw_text)
        self.assertGreaterEqual(len(subs), 4)
        names = [s["name"] for s in subs]
        self.assertIn("Apple Operations International", names)
        self.assertIn("Beats Electronics LLC", names)

    def test_extract_10k_sections_mock(self):
        client = EdgarClient()
        mock_filing = MagicMock()
        mock_filing.accession_number = "0000320193-24-000106"
        mock_filing.filing_date = "2024-11-01"
        mock_filing.report_date = "2024-09-28"

        mock_tenk = MagicMock()
        mock_tenk.item_1 = "Apple designs, manufactures and markets smartphones, personal computers, tablets, wearables and accessories."
        mock_tenk.item_1a = "Global economic conditions, supply chain disruptions, and intense competition could impact our business."
        mock_filing.obj.return_value = mock_tenk

        mock_att = MagicMock()
        mock_att.document_type = "EX-21.1"
        mock_att.description = "Subsidiaries of the Registrant"
        mock_att.text.return_value = "Apple Sales International\tIreland\nApple Japan Inc.\tJapan"
        mock_filing.attachments = [mock_att]

        sections = client.extract_10k_sections(mock_filing)

        self.assertEqual(sections["accession_number"], "0000320193-24-000106")
        self.assertEqual(sections["filing_date"], "2024-11-01")
        self.assertIn("Apple designs", sections["item_1_business"])
        self.assertIn("supply chain disruptions", sections["item_1a_risk_factors"])
        self.assertEqual(len(sections["subsidiaries"]), 2)
        self.assertEqual(sections["subsidiaries"][0]["name"], "Apple Sales International")


if __name__ == "__main__":
    unittest.main()
