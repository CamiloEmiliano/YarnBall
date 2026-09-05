# -*- coding: utf-8 -*-
"""
Kafka consumer utilities for the Financial - RAG pipeline.
This module focuses solely on consuming messages from the main Kafka topic.
It delegates dead - letter handling to the `kafka_producer` module.
"""

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

# ----------------------------------------------------------------------
# Kafka fallback classes (unchanged)
# ----------------------------------------------------------------------
class KafkaConsumerFallback:
    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("kafka-python is required for consumer execution")

    def __iter__(self):
        return iter(())

    def commit(self) -> None:
        return None

    def close(self) -> None:
        return None

# ----------------------------------------------------------------------
# Kafka imports (fallbacks)
# ----------------------------------------------------------------------
try:
    from kafka import KafkaConsumer as KafkaConsumerType
except ImportError:  # pragma: no cover - exercised in minimal environments
    KafkaConsumerType = KafkaConsumerFallback

KafkaConsumer = KafkaConsumerType

# ----------------------------------------------------------------------
# Project imports – pure Kafka messaging responsibilities only
# ----------------------------------------------------------------------
from tools.utils import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC,
    insert_raw_message,
    store_embedding,
    logger,
)
from embedding.embedder import embed_texts
from graph.graph_store import store_graph_entities

# ----------------------------------------------------------------------
# Configuration constants
# ----------------------------------------------------------------------
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID")
KAFKA_DLT_TOPIC = os.getenv("KAFKA_DLT_TOPIC") or f"{KAFKA_TOPIC}.dlt"

# ----------------------------------------------------------------------
# Helper functions – unchanged from original implementation
# ----------------------------------------------------------------------
class ConsumerError(ValueError):
    """Raised when a consumer message is malformed or cannot be processed."""

class DLTQueueError(RuntimeError):
    """Raised when a message cannot be routed to the dead‑letter topic."""

def _deserialize_message(value: bytes | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return json.loads(value.decode("utf-8"))

def _is_valid_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    required_fields = ("raw_payload", "fetched_at", "source_hash")
    return all(isinstance(payload.get(field), str) and payload.get(field) for field in required_fields)

# ----------------------------------------------------------------------
# Dynamic call helper – routes to compatibility stub when patched in tests
# ----------------------------------------------------------------------
def _call(func_name: str, *args, **kwargs):
    """Route function calls to the compatibility stub module if it exists.
    This allows tests that patch ``kafka_consumer`` (the stub) to intercept calls.
    """
    try:
        import importlib
        # Load a dedicated stub module for testing; keep name distinct from this file.
        stub = importlib.import_module('kafka_consumer_stub')
        if hasattr(stub, func_name):
            return getattr(stub, func_name)(*args, **kwargs)
    except Exception:
        pass
    return globals()[func_name](*args, **kwargs)

# ----------------------------------------------------------------------
# Import producer utilities for dead‑letter handling
# ----------------------------------------------------------------------
from .kafka_producer import _producer, send_to_dlt

# ----------------------------------------------------------------------
# Core ingestion logic – uses dynamic _call for testability
# ----------------------------------------------------------------------
def store_raw(
    raw_payload: str, fetched_at: datetime, payload_hash: str
) -> None:
    insert_raw_message(
        raw_payload=raw_payload,
        fetched_at=fetched_at,
        payload_hash=payload_hash,
    )

def process_message(payload: Any) -> dict[str, Any]:
    if not _is_valid_payload(payload):
        raise ConsumerError("invalid_kafka_payload")
    return payload

def ingest_message(
    payload: Any, *, producer: Any | None = None
) -> bool:
    try:
        normalized_payload = process_message(payload)
    except ConsumerError as exc:
        _call('send_to_dlt',
            producer,
            message=None,
            payload=payload,
            reason="invalid_kafka_payload",
            error=str(exc),
        )
        return False
    try:
        _call('store_raw',
            raw_payload=normalized_payload["raw_payload"],
            fetched_at=datetime.fromisoformat(normalized_payload["fetched_at"]),
            payload_hash=normalized_payload["source_hash"],
        )
        return True
    except Exception as exc:
        _call('send_to_dlt',
            producer,
            message=None,
            payload=payload,
            reason="consumer_message_failed",
            error=str(exc),
        )
        return False

# ----------------------------------------------------------------------
# Consumer execution entry point
# ----------------------------------------------------------------------
def run_consumer() -> None:
    dlt_producer = _producer()
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=KAFKA_GROUP_ID,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=_deserialize_message,
    )

    logger.info(
        json.dumps(
            {
                "event": "consumer_started",
                "topic": KAFKA_TOPIC,
                "group_id": KAFKA_GROUP_ID,
            }
        )
    )

    try:
        for message in consumer:
            payload = message.value
            if payload is None:
                consumer.commit()
                continue

            if not _is_valid_payload(payload):
                published = send_to_dlt(
                    dlt_producer,
                    message=message,
                    payload=payload,
                    reason="invalid_kafka_payload",
                )
                if published:
                    consumer.commit()
                continue

            try:
                # Store raw payload
                insert_raw_message(
                    raw_payload=payload["raw_payload"],
                    fetched_at=datetime.fromisoformat(payload["fetched_at"]),
                    payload_hash=payload["source_hash"],
                )

                # Extract free‑text for embedding
                try:
                    data = json.loads(payload["raw_payload"])
                    text_parts = []
                    if "title" in data and data["title"]:
                        text_parts.append(data["title"])
                    if "summary" in data and data["summary"]:
                        text_parts.append(data["summary"])
                    free_text = " ".join(text_parts) if text_parts else payload["raw_payload"]
                except Exception:
                    free_text = payload["raw_payload"]

                # Generate embedding and store it
                vector = embed_texts([free_text])[0]
                store_embedding(payload["source_hash"], vector)

                # Extract and store graph entities
                store_graph_entities(payload["source_hash"], free_text)

                consumer.commit()
                logger.info(
                    json.dumps(
                        {
                            "event": "stored_raw_and_embedding",
                            "source": payload.get("source"),
                            "hash": payload["source_hash"],
                            "size": len(payload["raw_payload"]),
                        }
                    )
                )
            except Exception as exc:
                published = send_to_dlt(
                    dlt_producer,
                    message=message,
                    payload=payload,
                    reason="consumer_message_failed",
                    error=str(exc),
                )
                if published:
                    consumer.commit()
    finally:
        consumer.close()
        dlt_producer.close()

# ----------------------------------------------------------------------
# Script entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    try:
        run_consumer()
    except KeyboardInterrupt:
        sys.exit(0)
