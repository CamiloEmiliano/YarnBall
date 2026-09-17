"""
Unit tests for S&P 500 Historical Constituent Universe Manager.
"""

from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from tools.sp500_universe import SP500Constituent, SP500UniverseManager


class TestSP500UniverseManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_file = Path(self.temp_dir.name) / "sp500_test.json"
        self.manager = SP500UniverseManager(cache_file=self.cache_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cache_created_on_init(self):
        self.assertTrue(self.cache_file.exists())
        with open(self.cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertGreater(len(data), 50)

    def test_point_in_time_active_filtering(self):
        # Twitter (TWTR) was added 2018-06-07 and removed 2022-10-28
        self.assertTrue(self.manager.is_sp500("TWTR", "2020-01-01"))
        self.assertTrue(self.manager.is_sp500("TWTR", "2022-01-01"))
        self.assertFalse(self.manager.is_sp500("TWTR", "2023-01-01"))
        self.assertFalse(self.manager.is_sp500("TWTR", "2017-01-01"))

        # Tesla (TSLA) was added 2020-12-21
        self.assertFalse(self.manager.is_sp500("TSLA", "2020-01-01"))
        self.assertTrue(self.manager.is_sp500("TSLA", "2021-01-01"))
        self.assertTrue(self.manager.is_sp500("TSLA", "2024-01-01"))

        # SVB Financial (SIVB) was added 2018-12-24 and removed 2023-03-15
        self.assertTrue(self.manager.is_sp500("SIVB", "2022-06-01"))
        self.assertFalse(self.manager.is_sp500("SIVB", "2024-01-01"))

    def test_cik_lookup(self):
        # Apple CIK is 0000320193 / 320193
        self.assertTrue(self.manager.is_sp500("0000320193"))
        self.assertTrue(self.manager.is_sp500("320193"))
        self.assertFalse(self.manager.is_sp500("9999999999"))

    def test_gics_sector_distribution(self):
        dist = self.manager.get_sector_distribution()
        # Ensure all 11 GICS sectors are represented
        expected_sectors = {
            "Information Technology", "Financials", "Health Care",
            "Consumer Discretionary", "Communication Services", "Industrials",
            "Consumer Staples", "Energy", "Utilities", "Real Estate", "Materials"
        }
        for sector in expected_sectors:
            self.assertIn(sector, dist, f"Sector {sector} should be in distribution")
            self.assertGreater(dist[sector], 0, f"Sector {sector} should have constituents")

    def test_get_constituents_at_date(self):
        c_2019 = self.manager.get_constituents_at_date("2019-06-01")
        tickers_2019 = {c.ticker for c in c_2019}
        self.assertIn("TWTR", tickers_2019)
        self.assertNotIn("TSLA", tickers_2019)

        c_2024 = self.manager.get_constituents_at_date("2024-10-01")
        tickers_2024 = {c.ticker for c in c_2024}
        self.assertIn("TSLA", tickers_2024)
        self.assertIn("PLTR", tickers_2024)
        self.assertNotIn("TWTR", tickers_2024)

    @patch("tools.sp500_universe._connect_db")
    def test_sync_to_postgres(self, mock_connect):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_connect.return_value = mock_conn

        synced = self.manager.sync_to_postgres("financial_rag")
        self.assertGreater(synced, 50)
        mock_cur.execute.assert_called()
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
