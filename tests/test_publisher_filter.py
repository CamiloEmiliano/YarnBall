"""
Unit tests for Publisher Stoplist & Editorial Disclaimer Filtering.
"""

import unittest

from graph.entity_resolver import EntityResolver, is_blacklisted_publisher, PUBLISHER_STOPLIST
from ingest.scraping_worker import strip_publisher_boilerplates


class TestPublisherFilter(unittest.TestCase):
    def test_is_blacklisted_publisher(self):
        # Known publishers
        self.assertTrue(is_blacklisted_publisher("Motley Fool"))
        self.assertTrue(is_blacklisted_publisher("The Motley Fool"))
        self.assertTrue(is_blacklisted_publisher("fool.com"))
        self.assertTrue(is_blacklisted_publisher("Seeking Alpha"))
        self.assertTrue(is_blacklisted_publisher("seekingalpha"))
        self.assertTrue(is_blacklisted_publisher("Zacks Investment Research"))
        self.assertTrue(is_blacklisted_publisher("Benzinga"))
        self.assertTrue(is_blacklisted_publisher("PR Newswire"))
        self.assertTrue(is_blacklisted_publisher("BusinessWire"))
        self.assertTrue(is_blacklisted_publisher("InvestorPlace"))

        # Real companies must NOT be blacklisted
        self.assertFalse(is_blacklisted_publisher("Apple Inc."))
        self.assertFalse(is_blacklisted_publisher("Microsoft Corporation"))
        self.assertFalse(is_blacklisted_publisher("NVIDIA Corporation"))
        self.assertFalse(is_blacklisted_publisher("Tesla, Inc."))
        self.assertFalse(is_blacklisted_publisher("Taiwan Semiconductor"))
        self.assertFalse(is_blacklisted_publisher("Amazon.com"))

    def test_entity_resolver_filters_publisher_nodes(self):
        resolver = EntityResolver()
        raw_nodes = [
            {"id": "Apple Inc.", "label": "Company", "ticker": "AAPL"},
            {"id": "The Motley Fool", "label": "Company", "ticker": None},
            {"id": "Seeking Alpha", "label": "Company", "ticker": None},
            {"id": "Taiwan Semiconductor", "label": "Company", "ticker": "TSM"},
        ]

        node_mapping, resolved_nodes = resolver.resolve_nodes(raw_nodes)

        resolved_ids = {n["id"] for n in resolved_nodes}
        self.assertIn("Apple Inc.", resolved_ids)
        self.assertIn("Taiwan Semiconductor", resolved_ids)
        self.assertNotIn("The Motley Fool", resolved_ids)
        self.assertNotIn("Seeking Alpha", resolved_ids)

    def test_entity_resolver_drops_publisher_edges(self):
        resolver = EntityResolver()
        raw_nodes = [
            {"id": "Apple Inc.", "label": "Company", "ticker": "AAPL"},
            {"id": "The Motley Fool", "label": "Company", "ticker": None},
            {"id": "Taiwan Semiconductor", "label": "Company", "ticker": "TSM"},
        ]
        node_mapping, _ = resolver.resolve_nodes(raw_nodes)

        raw_edges = [
            # Valid commercial relationship
            {"source_id": "Taiwan Semiconductor", "target_id": "Apple Inc.", "rel_type": "SUPPLIES_TO"},
            # Spurious publisher recommendation edges
            {"source_id": "The Motley Fool", "target_id": "Apple Inc.", "rel_type": "RECOMMENDS"},
            {"source_id": "The Motley Fool", "target_id": "Taiwan Semiconductor", "rel_type": "OWNS_SHARES_OF"},
        ]

        rewired_edges = resolver.rewire_edges(raw_edges, node_mapping)

        self.assertEqual(len(rewired_edges), 1)
        self.assertEqual(rewired_edges[0]["source_id"], "Taiwan Semiconductor")
        self.assertEqual(rewired_edges[0]["target_id"], "Apple Inc.")
        self.assertEqual(rewired_edges[0]["rel_type"], "SUPPLIES_TO")

    def test_strip_publisher_boilerplates(self):
        article_text = (
            "Apple reported strong quarterly earnings driven by iPhone and Mac sales.\n"
            "The company saw increased enterprise adoption.\n\n"
            "The Motley Fool has positions in and recommends Apple, Microsoft, and Tesla. "
            "Disclosure: The author owns shares of Apple."
        )

        cleaned = strip_publisher_boilerplates(article_text)

        self.assertIn("Apple reported strong quarterly earnings", cleaned)
        self.assertIn("increased enterprise adoption", cleaned)
        self.assertNotIn("The Motley Fool has positions in", cleaned)
        self.assertNotIn("Disclosure: The author owns shares", cleaned)


if __name__ == "__main__":
    unittest.main()
