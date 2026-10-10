# streaming/__init__.py
"""Streaming ingestion and event broker pipeline."""

from streaming.driver import send_record
from streaming.producer import (
    KafkaProducer,
    KafkaProducerFallback,
    _init_topic,
    _producer,
    get_producer,
    publish_message,
    send_to_dlt,
)
from streaming.consumer import (
    ConsumerError,
    DLTQueueError,
    ingest_message,
    process_message,
    run_consumer,
    store_raw,
)

__all__ = [
    "send_record",
    "publish_message",
    "send_to_dlt",
    "get_producer",
    "_init_topic",
    "_producer",
    "KafkaProducer",
    "KafkaProducerFallback",
    "ConsumerError",
    "DLTQueueError",
    "ingest_message",
    "process_message",
    "run_consumer",
    "store_raw",
]
