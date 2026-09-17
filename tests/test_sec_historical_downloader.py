"""
Unit tests for Multi-Year SEC Historical Batch Harvester.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from tools.download_historical_sec import HistoricalSECHarvester
from tools.sp500_universe import SP500Constituent


class TestHistoricalSECHarvester(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name) / "sec_historical"
        self.mock_client = MagicMock()
        self.harvester = HistoricalSECHarvester(
            output_dir=self.output_dir,
            client=self.mock_client,
            rate_limit_delay_sec=0.0,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_harvest_company_year_10k_and_sections(self):
        # Mock 10-K filing and section extraction
        mock_filing = MagicMock()
        mock_filing.accession_number = "0000320193-23-000106"
        self.mock_client.fetch_latest_10k.return_value = mock_filing
        self.mock_client.extract_10k_sections.return_value = {
            "accession_number": "0000320193-23-000106",
            "filing_date": "2023-11-03",
            "report_date": "2023-09-30",
            "item_1_business": "Apple designs, manufactures and markets smartphones...",
            "item_1a_risk_factors": "Global economic conditions and supply chain dependencies...",
            "subsidiaries": [
                {"name": "Apple Operations International", "jurisdiction": "Ireland"},
                {"name": "Apple Sales International", "jurisdiction": "Ireland"},
            ],
        }
        self.mock_client.fetch_10q_filings.return_value = []
        self.mock_client.fetch_recent_8k.return_value = []
        self.mock_client.fetch_form4_transactions.return_value = []

        constituent = SP500Constituent(
            ticker="AAPL",
            cik="0000320193",
            company_name="Apple Inc.",
            gics_sector="Information Technology",
        )

        stats = self.harvester.harvest_company_year(constituent, year=2023, forms=["10-K"])

        self.assertTrue(stats["10k_downloaded"])
        self.assertEqual(stats["subsidiaries_count"], 2)

        # Verify staged directory and files
        acc_dir = self.output_dir / "AAPL" / "2023" / "10K_0000320193-23-000106"
        self.assertTrue(acc_dir.exists())
        self.assertTrue((acc_dir / "metadata.json").exists())
        self.assertTrue((acc_dir / "item_1_business.txt").exists())
        self.assertTrue((acc_dir / "item_1a_risk_factors.txt").exists())
        self.assertTrue((acc_dir / "subsidiaries.json").exists())

        with open(acc_dir / "subsidiaries.json", "r") as f:
            subs = json.load(f)
        self.assertEqual(len(subs), 2)
        self.assertEqual(subs[0]["name"], "Apple Operations International")

    def test_harvest_company_year_8k_and_10q(self):
        self.mock_client.fetch_latest_10k.return_value = None
        self.mock_client.fetch_10q_filings.return_value = [
            {
                "accession_number": "0000320193-23-000050",
                "filing_date": "2023-05-04",
                "form": "10-Q",
                "text": "Quarterly revenues increased...",
            }
        ]
        self.mock_client.fetch_recent_8k.return_value = [
            {
                "accession_number": "0000320193-23-000080",
                "filing_date": "2023-08-03",
                "form": "8-K",
                "items": ["Item 2.02 Results of Operations"],
                "text": "Earnings release statement...",
            }
        ]
        self.mock_client.fetch_form4_transactions.return_value = [
            {
                "accession_number": "0000320193-23-000099",
                "filing_date": "2023-10-01",
                "form": "4",
            }
        ]

        constituent = SP500Constituent(
            ticker="AAPL",
            cik="0000320193",
            company_name="Apple Inc.",
            gics_sector="Information Technology",
        )

        stats = self.harvester.harvest_company_year(constituent, year=2023, forms=["10-Q", "8-K", "4"])

        self.assertEqual(stats["10q_count"], 1)
        self.assertEqual(stats["8k_count"], 1)
        self.assertEqual(stats["form4_count"], 1)

        comp_dir = self.output_dir / "AAPL" / "2023"
        self.assertTrue((comp_dir / "10Q_0000320193-23-000050" / "quarterly_text.txt").exists())
        self.assertTrue((comp_dir / "8K_0000320193-23-000080" / "form8k_events.json").exists())
        self.assertTrue((comp_dir / "Form4_Transactions" / "form4_transactions.json").exists())

    def test_harvest_year_with_ticker_filter_and_limit(self):
        self.mock_client.fetch_latest_10k.return_value = None
        self.mock_client.fetch_10q_filings.return_value = []
        self.mock_client.fetch_recent_8k.return_value = []
        self.mock_client.fetch_form4_transactions.return_value = []

        results = self.harvester.harvest_year(
            year=2023,
            tickers=["AAPL", "ABT"],
            forms=["10-K"],
            max_companies=1,
        )

        self.assertEqual(len(results), 1)
        self.assertIn(results[0]["ticker"], ["AAPL", "ABT"])


if __name__ == "__main__":
    unittest.main()
