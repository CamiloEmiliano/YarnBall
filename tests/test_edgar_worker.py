"""
Unit tests for EdgarWorker end-to-end 10-K processing and graph extraction.
"""

import unittest
from unittest.mock import MagicMock, patch

from ingest.edgar_worker import EdgarWorker, extract_sec_relationships


class TestEdgarWorker(unittest.TestCase):
    def setUp(self):
        self.mock_pg_conn = MagicMock()
        self.mock_cursor = MagicMock()
        self.mock_pg_conn.cursor.return_value = self.mock_cursor

        self.mock_driver = MagicMock()
        self.mock_session = MagicMock()
        self.mock_driver.session.return_value.__enter__.return_value = self.mock_session

        self.worker = EdgarWorker(pg_conn=self.mock_pg_conn, memgraph_driver=self.mock_driver)

    def test_extract_sec_relationships_fallback(self):
        # Empty text should return empty dict
        res = extract_sec_relationships("")
        self.assertEqual(res, {"nodes": [], "edges": []})

    def test_persist_subsidiaries(self):
        subs = [
            {"name": "Apple Operations Europe", "jurisdiction": "Ireland"},
            {"name": "Beats Electronics LLC", "jurisdiction": "Delaware"},
        ]
        nodes, edges = self.worker._persist_subsidiaries(
            parent_cik="0000320193",
            parent_name="Apple Inc.",
            parent_ticker="AAPL",
            subsidiaries=subs,
            accession_number="0000320193-24-000106",
            filing_date="2024-11-01",
        )

        self.assertIn("Apple Inc.", nodes)
        self.assertIn("Apple Operations Europe", nodes)
        self.assertEqual(len(edges), 2)
        self.assertEqual(edges[0]["type"], "SUBSIDIARY_OF")

        # Verify Memgraph session executed MERGE statements
        self.assertGreater(self.mock_session.run.call_count, 0)
        # Verify PostgreSQL executemany called
        self.mock_cursor.executemany.assert_called()

    def test_persist_sec_graph_data(self):
        graph_data = {
            "nodes": [
                {"id": "Taiwan Semiconductor Manufacturing", "type": "Company", "properties": {"ticker": "TSM"}},
                {"id": "Foxconn", "type": "Company", "properties": {}},
            ],
            "edges": [
                {"source": "Taiwan Semiconductor Manufacturing", "target": "Apple Inc.", "type": "SUPPLIES_TO", "properties": {"nature": "chips"}},
            ],
        }

        nodes, edges = self.worker._persist_sec_graph_data(
            primary_name="Apple Inc.",
            primary_ticker="AAPL",
            graph_data=graph_data,
            accession_number="0000320193-24-000106",
            filing_date="2024-11-01",
            form_type="10-K",
            section="Item 1 Business",
        )

        self.assertIn("Apple Inc.", nodes)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["type"], "SUPPLIES_TO")

        # Verify Memgraph session was called with temporal parameters
        self.assertGreater(self.mock_session.run.call_count, 0)

    @patch.object(EdgarWorker, "_persist_sec_graph_data")
    @patch.object(EdgarWorker, "_persist_subsidiaries")
    def test_process_company_10k_flow(self, mock_subs, mock_graph):
        mock_subs.return_value = (["Apple Operations"], [{"source": "Apple Operations", "target": "Apple Inc.", "type": "SUBSIDIARY_OF"}])
        mock_graph.return_value = (["TSM"], [{"source": "TSM", "target": "Apple Inc.", "type": "SUPPLIES_TO"}])

        mock_filing = MagicMock()
        mock_filing.company = "Apple Inc."
        mock_filing.accession_number = "0000320193-24-000106"
        mock_filing.filing_date = "2024-11-01"
        mock_filing.report_date = "2024-09-28"

        self.worker.client.fetch_latest_10k = MagicMock(return_value=mock_filing)
        self.worker.client.extract_10k_sections = MagicMock(return_value={
            "accession_number": "0000320193-24-000106",
            "filing_date": "2024-11-01",
            "report_date": "2024-09-28",
            "item_1_business": "Apple designs and markets consumer electronics.",
            "item_1a_risk_factors": "Supply chain risks and geopolitical factors.",
            "subsidiaries": [{"name": "Apple Operations", "jurisdiction": "Ireland"}],
        })

        result = self.worker.process_company_10k("AAPL")

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["accession_number"], "0000320193-24-000106")
        self.assertGreaterEqual(result["nodes"], 1)
        self.assertGreaterEqual(result["edges"], 1)


if __name__ == "__main__":
    unittest.main()
