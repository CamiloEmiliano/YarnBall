# -*- coding: utf-8 -*-
"""
Kafka producer utilities for the Financial - RAG pipeline.
Provides a fallback implementation when `kafka-python` is not installed
and a `send_to_dlt` helper that routes malformed/failed messages to the
dead - letter topic.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any

# Kafka admin imports for topic creation
try:
    from kafka.admin import KafkaAdminClient, NewTopic
except ImportError:  # pragma: no cover
    KafkaAdminClient = None
    NewTopic = None

def _ensure_topics() -> None:
    """
    Create the primary data topic and the dead‑letter topic if they do not
    already exist.  It is idempotent – any “already exists” error is ignored.
    """
    # Main topic (default is “news”)
    _init_topic(os.getenv("KAFKA_TOPIC"))

    # Dead‑letter topic – default is "<main>.dlt"
    dlt_topic = os.getenv("KAFKA_DLT_TOPIC") or f"{os.getenv('KAFKA_TOPIC')}.dlt"
    _init_topic(dlt_topic)

# ----------------------------------------------------------------------
# Kafka fallback class (unchanged)
# ----------------------------------------------------------------------
class KafkaProducerFallback:
    def __init__(self, *args, **kwargs) -> None:
        # In environments without kafka‑python we raise a clear error.
        from tools.utils import logger
        logger.error("KafkaProducerFallback instantiated – kafka‑python is required.")
        raise RuntimeError("kafka-python is required for KafkaProducer")

    def send(self, *args, **kwargs):  # pragma: no cover
        # Return an object with a ``get`` method so callers can ``.get()`` safely.
        class _NoOpFuture:
            def get(self, timeout=None):
                return None
        return _NoOpFuture()

    def close(self) -> None:
        # Nothing to close in the no‑op implementation.
        return None

# ----------------------------------------------------------------------
# Kafka import with fallback
# ----------------------------------------------------------------------
try:
    from kafka import KafkaProducer as KafkaProducerType
except ImportError:  # pragma: no cover - exercised in minimal environments
    KafkaProducerType = KafkaProducerFallback

KafkaProducer = KafkaProducerType

# ----------------------------------------------------------------------
# Helper to create a producer instance
# ----------------------------------------------------------------------
def _producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    )

# ----------------------------------------------------------------------
# Send a message to the dead‑letter topic (DLT)
# ----------------------------------------------------------------------
def send_to_dlt(
    producer: KafkaProducer | None,
    message: Any,
    payload: Any,
    reason: str,
    error: str | None = None,
) -> bool:
    if producer is None:
        # In testing environments the producer may be omitted – we log and skip.
        from tools.utils import logger
        logger.warning(
            json.dumps(
                {
                    "event": "dlt_skipped",
                    "topic": os.getenv("KAFKA_DLT_TOPIC"),
                    "reason": reason,
                    "error": error,
                }
            )
        )
        return False
    event = {
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "error": error,
        "original_topic": os.getenv("KAFKA_TOPIC"),
        "partition": getattr(message, "partition", None),
        "offset": getattr(message, "offset", None),
        "payload": payload,
    }
    try:
        producer.send(os.getenv("KAFKA_DLT_TOPIC"), value=event).get(timeout=30)
        from tools.utils import logger
        logger.error(
            json.dumps(
                {
                    "event": "message_sent_to_dlt",
                    "topic": os.getenv("KAFKA_DLT_TOPIC"),
                    "reason": reason,
                }
            )
        )
        return True
    except Exception as exc:
        from tools.utils import logger
        logger.error(
            json.dumps(
                {
                    "event": "dlt_publish_failed",
                    "topic": os.getenv("KAFKA_DLT_TOPIC"),
                    "reason": reason,
                    "error": str(exc),
                }
            )
        )
        return False

# ----------------------------------------------------------------------
# Topic initialization and convenience wrapper
# ----------------------------------------------------------------------

def _init_topic(topic: str) -> None:
    """Create the given topic if it does not already exist.
    Uses KafkaAdminClient; no‑op if admin client unavailable.
    """
    if KafkaAdminClient is None:
        # Admin client not available (fallback mode)
        return
    admin = KafkaAdminClient(bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS"))
    try:
        admin.create_topics([NewTopic(name=topic, num_partitions=1, replication_factor=1)])
    except Exception:
        # Topic may already exist; ignore errors
        pass
    finally:
        admin.close()

def get_producer() -> KafkaProducer:
    """Factory that returns a ready‑to‑use producer and ensures both main and dead‑letter topics exist."""
    prod = _producer()
    main_topic = os.getenv("KAFKA_TOPIC")
    dlt_topic = os.getenv("KAFKA_DLT_TOPIC") or f"{main_topic}.dlt"
    _init_topic(main_topic)
    _init_topic(dlt_topic)
    return prod

def publish_message(
    payload: Any,
    topic: str | None = None,
    producer: KafkaProducer | None = None,
) -> bool:
    """Convenient wrapper for sending a JSON‑serialised message.

    Arguments:
        payload: The Python object to send (will be JSON‑encoded).
        topic:   Optional explicit topic; defaults to KAFKA_TOPIC env var.
        producer: Optional pre‑created producer; if omitted a new one is created.
    Returns:
        True on success, False on failure (also logs via send_to_dlt).
    """
    if producer is None:
        producer = get_producer()
    target_topic = topic or os.getenv("KAFKA_TOPIC")
    try:
        # The producer serialises via the value_serializer defined in _producer
        producer.send(target_topic, value=payload).get(timeout=30)
        return True
    except Exception as exc:
        # On failure, route to dead‑letter topic
        send_to_dlt(producer, None, payload, reason="publish_failed", error=str(exc))
        return False
