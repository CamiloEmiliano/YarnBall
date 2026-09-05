# graph/memgraph_consumer.py
# -*- coding: utf-8 -*-
"""Standalone Kafka consumer worker for Memgraph entity & relationship extraction.

Continuously consumes news messages from the Kafka topic and runs LLM-driven
entity/relationship extraction to build the knowledge graph in Memgraph.
Runs with its own dedicated consumer group so it can scale and operate
independently of raw PostgreSQL ingestion.
"""

import json
import os
import sys
from datetime import datetime
from typing import Any

# Fallback class when kafka-python is not installed in local environment
class KafkaConsumerFallback:
    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("kafka-python is required for consumer execution")

    def __iter__(self):
        return iter(())

    def commit(self) -> None:
        return None

    def close(self) -> None:
        return None

try:
    from kafka import KafkaConsumer as KafkaConsumerType
except ImportError:
    KafkaConsumerType = KafkaConsumerFallback

KafkaConsumer = KafkaConsumerType

from tools.utils import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC, logger
from .graph_store import store_graph_entities

# Dedicated consumer group for Memgraph to allow independent scaling & offset tracking
KAFKA_MEMGRAPH_GROUP_ID = os.getenv(
    "KAFKA_MEMGRAPH_GROUP_ID",
    f"{os.getenv('KAFKA_GROUP_ID', 'financial_rag')}-memgraph",
)


def _deserialize_message(value: bytes | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return json.loads(value.decode("utf-8"))


def process_graph_message(payload: dict[str, Any]) -> bool:
    """Extract raw payload text and persist entities/relations to Memgraph."""
    source_hash = payload.get("source_hash")
    raw_payload = payload.get("raw_payload")

    if not source_hash or not raw_payload:
        logger.warning("Skipping malformed payload in Memgraph consumer: %s", payload)
        return False

    # Extract text content (e.g. title + summary if JSON, or raw string)
    try:
        data = json.loads(raw_payload) if isinstance(raw_payload, str) else raw_payload
        text_parts = []
        if isinstance(data, dict):
            if data.get("title"):
                text_parts.append(data["title"])
            if data.get("summary"):
                text_parts.append(data["summary"])
        free_text = " ".join(text_parts) if text_parts else str(raw_payload)
    except Exception:
        free_text = str(raw_payload)

    store_graph_entities(source_hash, free_text)
    return True


def run_memgraph_consumer() -> None:
    """Continuously consume messages from Kafka and persist extracted entities to Memgraph."""
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=KAFKA_MEMGRAPH_GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=_deserialize_message,
    )

    logger.info(
        "Memgraph standalone consumer started (group: %s, topic: %s)",
        KAFKA_MEMGRAPH_GROUP_ID,
        KAFKA_TOPIC,
    )

    try:
        for message in consumer:
            payload = message.value
            if payload is None:
                consumer.commit()
                continue

            try:
                success = process_graph_message(payload)
                if success:
                    logger.info(
                        "Extracted and stored Memgraph entities for source_hash: %s",
                        payload.get("source_hash"),
                    )
                consumer.commit()
            except Exception as exc:
                logger.error(
                    "Failed to process/store graph entities for %s: %s",
                    payload.get("source_hash"),
                    exc,
                )
                # Do not commit offset so the message can be retried on next consumer run
    finally:
        consumer.close()
        logger.info("Memgraph consumer shut down")


if __name__ == "__main__":
    try:
        run_memgraph_consumer()
    except KeyboardInterrupt:
        sys.exit(0)
