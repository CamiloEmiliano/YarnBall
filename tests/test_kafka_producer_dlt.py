# -*- coding: utf-8 -*-
"""Tests for the Kafka producer DLT (dead‑letter‑topic) handling.
These tests verify that a malformed payload triggers the DLT path and that
the dead‑letter topic is auto‑created when a producer is obtained.
"""

import unittest
from unittest.mock import patch, MagicMock

# Import the module under test
from kafka_pipeline.kafka_producer import publish_message, get_producer, _init_topic, send_to_dlt

class KafkaProducerDLTTests(unittest.TestCase):
    def test_malformed_payload_triggers_dlt(self):
        """A payload that cannot be JSON‑encoded (e.g., a set) should cause
        publish_message to call ``send_to_dlt`` and return ``False``.
        """
        malformed = {"unserializable": {1, 2, 3}}  # sets are not JSON serialisable
        # Patch the low‑level producer so that ``send`` raises an exception
        mock_producer = MagicMock()
        mock_producer.send.side_effect = Exception("serialization error")
        # Patch ``_producer`` to return our mock (so get_producer uses it)
        with patch("kafka_pipeline.kafka_producer._producer", return_value=mock_producer):
            # Spy on send_to_dlt
            with patch("kafka_pipeline.kafka_producer.send_to_dlt") as mock_send_to_dlt:
                result = publish_message(malformed)
                self.assertFalse(result)
                # send_to_dlt should be called exactly once with the mock producer
                mock_send_to_dlt.assert_called_once()
                args, kwargs = mock_send_to_dlt.call_args
                self.assertIs(args[0], mock_producer)  # producer argument
                self.assertEqual(args[2], malformed)   # payload argument
                self.assertEqual(kwargs.get("reason"), "publish_failed")

    def test_get_producer_initialises_dlt_topic(self):
        """Calling ``get_producer`` should attempt to create both the main
        topic and the dead‑letter topic via ``_init_topic``.
        """
        with patch("kafka_pipeline.kafka_producer._init_topic") as mock_init_topic:
            # Patch the actual Kafka producer creation to avoid network calls
            with patch("kafka_pipeline.kafka_producer._producer") as mock_prod_factory:
                mock_prod_factory.return_value = MagicMock()
                prod = get_producer()
                # Ensure a producer instance is returned
                self.assertIsNotNone(prod)
                # ``_init_topic`` should be called twice – once for each topic
                self.assertEqual(mock_init_topic.call_count, 2)
                # Verify the topic names that were passed (main then DLT)
                called_topics = [call.args[0] for call in mock_init_topic.call_args_list]
                self.assertTrue(any("news" in t for t in called_topics))
                self.assertTrue(any(".dlt" in t for t in called_topics))

if __name__ == "__main__":
    unittest.main()
