import json
import os
import unittest
from unittest.mock import patch

from kafka_pipeline.kafka_consumer import ConsumerError, ingest_message, process_message


class KafkaConsumerTests(unittest.TestCase):
    def test_process_message_rejects_malformed_payload(self) -> None:
        with self.assertRaises(ConsumerError):
            process_message({"raw_payload": None})

    def test_ingest_message_routes_invalid_json_to_dlt(self) -> None:
        with patch.dict(
            os.environ, {"DLT_TOPIC": "financial_news_dlt"}, clear=False
        ), patch("kafka_pipeline.kafka_consumer.send_to_dlt") as dlt_mock, patch(
            "kafka_pipeline.kafka_consumer.store_raw"
        ) as store_mock:
            ingest_message({"raw_payload": "not-json", "source_hash": "hash"})

        dlt_mock.assert_called_once()
        self.assertEqual(store_mock.call_count, 0)

    def test_ingest_message_writes_valid_payload(self) -> None:
        valid_payload = json.dumps({"title": "abc", "url": "https://x"})
        with patch("kafka_pipeline.kafka_consumer.store_raw") as store_mock:
            ingest_message(
                {
                    "raw_payload": valid_payload,
                    "fetched_at": "2026-01-01T00:00:00+00:00",
                    "source_hash": "hash",
                }
            )

        store_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
