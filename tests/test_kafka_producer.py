# -*- coding: utf-8 -*-
"""
Unit tests for the kafka/kafka_producer module.
These tests verify producer creation, fallback handling, and the dead‑letter
routing helper `send_to_dlt`.
"""

import json
import os
import unittest
from collections import namedtuple
from unittest.mock import MagicMock, patch

# Import the functions under test
from kafka_pipeline.kafka_producer import _producer, send_to_dlt, KafkaProducerFallback


class TestKafkaProducer(unittest.TestCase):
    def setUp(self):
        # Ensure a clean environment for each test
        self.original_env = os.environ.copy()
        os.environ.clear()
        os.environ.update(self.original_env)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.original_env)

    @patch("kafka_pipeline.kafka_producer.KafkaProducer")
    def test_producer_created_with_env_vars(self, mock_kafka_producer):
        """_producer should instantiate KafkaProducer with the bootstrap server
        from the environment and a JSON serializer."""
        os.environ["KAFKA_BOOTSTRAP_SERVERS"] = "localhost:9092"
        prod = _producer()
        mock_kafka_producer.assert_called_once()
        args, kwargs = mock_kafka_producer.call_args
        self.assertEqual(kwargs["bootstrap_servers"], "localhost:9092")
        # The serializer should be a callable that returns JSON‑encoded bytes
        serializer = kwargs["value_serializer"]
        self.assertTrue(callable(serializer))
        sample = {"x": 1}
        self.assertEqual(serializer(sample), json.dumps(sample).encode("utf-8"))
        # The returned object should be the mock instance
        self.assertIs(prod, mock_kafka_producer.return_value)

    @patch("kafka_pipeline.kafka_producer.KafkaProducer", side_effect=ImportError)
    def test_producer_fallback_when_kafka_not_installed(self, _):
        """When the kafka library cannot be imported, the fallback class is used.
        Instantiating the fallback should raise a clear RuntimeError."""
        # Force import error path – the module already substituted the class
        # at import time, so we directly instantiate the fallback.
        with self.assertRaises(RuntimeError) as ctx:
            KafkaProducerFallback()
        self.assertIn("kafka-python is required", str(ctx.exception))

    @patch("tools.utils.logger")
    def test_send_to_dlt_no_producer_logs_and_returns_false(self, mock_logger):
        """When producer is None, send_to_dlt logs a skip event and returns False."""
        os.environ["KAFKA_DLT_TOPIC"] = "test_dlt"
        result = send_to_dlt(
            producer=None,
            message=None,
            payload={"a": 1},
            reason="test_reason",
            error="test_error",
        )
        self.assertFalse(result)
        mock_logger.warning.assert_called_once()
        logged_json = json.loads(mock_logger.warning.call_args[0][0])
        self.assertEqual(logged_json["event"], "dlt_skipped")
        self.assertEqual(logged_json["reason"], "test_reason")
        self.assertEqual(logged_json["error"], "test_error")

    @patch("tools.utils.logger")
    def test_send_to_dlt_successful_send_returns_true(self, mock_logger):
        """A healthy producer should send the event, log success and return True."""
        mock_producer = MagicMock()
        mock_future = MagicMock()
        mock_future.get.return_value = None
        mock_producer.send.return_value = mock_future

        os.environ["KAFKA_DLT_TOPIC"] = "my_dlt"
        os.environ["KAFKA_TOPIC"] = "my_topic"
        dummy_message = namedtuple("Msg", ["partition", "offset"])(partition=5, offset=10)
        payload = {"foo": "bar"}

        result = send_to_dlt(
            producer=mock_producer,
            message=dummy_message,
            payload=payload,
            reason="test_success",
            error=None,
        )
        self.assertTrue(result)
        mock_producer.send.assert_called_once()
        # Verify the topic used is the env value
        args, kwargs = mock_producer.send.call_args
        self.assertEqual(args[0], "my_dlt")
        event = kwargs["value"]
        self.assertEqual(event["reason"], "test_success")
        self.assertEqual(event["original_topic"], "my_topic")
        self.assertEqual(event["partition"], 5)
        self.assertEqual(event["offset"], 10)
        self.assertEqual(event["payload"], payload)
        mock_logger.error.assert_called_once()
        logged_json = json.loads(mock_logger.error.call_args[0][0])
        self.assertEqual(logged_json["event"], "message_sent_to_dlt")
        self.assertEqual(logged_json["topic"], "my_dlt")
        self.assertEqual(logged_json["reason"], "test_success")

    @patch("tools.utils.logger")
    def test_send_to_dlt_send_raises_exception_returns_false(self, mock_logger):
        """If producer.send().get() raises, the function logs failure and returns False."""
        mock_producer = MagicMock()
        mock_future = MagicMock()
        mock_future.get.side_effect = RuntimeError("boom")
        mock_producer.send.return_value = mock_future

        os.environ["KAFKA_DLT_TOPIC"] = "dl_topic"
        os.environ["KAFKA_TOPIC"] = "src_topic"
        dummy_message = namedtuple("Msg", ["partition", "offset"])(partition=1, offset=2)
        payload = {"x": 123}

        result = send_to_dlt(
            producer=mock_producer,
            message=dummy_message,
            payload=payload,
            reason="failure_case",
            error="oops",
        )
        self.assertFalse(result)
        mock_producer.send.assert_called_once()
        mock_logger.error.assert_called_once()
        logged_json = json.loads(mock_logger.error.call_args[0][0])
        self.assertEqual(logged_json["event"], "dlt_publish_failed")
        self.assertEqual(logged_json["topic"], "dl_topic")
        self.assertEqual(logged_json["reason"], "failure_case")
        self.assertIn("boom", logged_json["error"])


if __name__ == "__main__":
    unittest.main()
