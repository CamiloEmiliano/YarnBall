# -*- coding: utf-8 -*-
import json
import unittest
from unittest.mock import MagicMock, patch

from graph.memgraph_consumer import process_graph_message, _deserialize_message


class TestMemgraphConsumer(unittest.TestCase):
    def test_deserialize_message(self):
        self.assertIsNone(_deserialize_message(None))
        data = {"key": "val"}
        raw = json.dumps(data).encode("utf-8")
        self.assertEqual(_deserialize_message(raw), data)

    def test_process_graph_message_malformed_payload(self):
        self.assertFalse(process_graph_message({}))
        self.assertFalse(process_graph_message({"source_hash": "abc"}))
        self.assertFalse(process_graph_message({"raw_payload": "text"}))

    @patch("graph.memgraph_consumer.store_graph_entities")
    def test_process_graph_message_json_payload(self, mock_store):
        payload = {
            "source_hash": "hash123",
            "raw_payload": json.dumps({"title": "Apple News", "summary": "Q3 record profits"}),
        }
        result = process_graph_message(payload)
        self.assertTrue(result)
        mock_store.assert_called_once_with("hash123", "Apple News Q3 record profits")

    @patch("graph.memgraph_consumer.store_graph_entities")
    def test_process_graph_message_raw_string(self, mock_store):
        payload = {
            "source_hash": "hash456",
            "raw_payload": "Raw text news article",
        }
        result = process_graph_message(payload)
        self.assertTrue(result)
        mock_store.assert_called_once_with("hash456", "Raw text news article")


if __name__ == "__main__":
    unittest.main()
