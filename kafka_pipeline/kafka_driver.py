# kafka/kafka_driver.py
# -*- coding: utf-8 -*-

"""Thin wrapper used by the ingestion scripts.
It converts a `(source, payload)` pair into the envelope expected
by the consumer and forwards it to the Kafka producer."""

import json
import os
import hashlib
from datetime import datetime, timezone
from typing import Any

from .kafka_producer import publish_message

def _make_envelope(source: str, raw_payload: str) -> dict[str, Any]:
    """
    Build the internal message format that kafka_consumer._is_valid_payload
    expects:
        {
            "raw_payload": <json‑string>,
            "fetched_at":  <ISO‑8601 timestamp>,
            "source_hash": <deterministic short hash>,
            "_source":     <source name>,
        }
    """
    fetched_at = datetime.now(timezone.utc).isoformat()
    source_hash = hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()[:16]

    envelope = {
        "raw_payload": raw_payload,
        "fetched_at": fetched_at,
        "source_hash": source_hash,
        "_source": source,
    }
    return envelope


def send_record(source: str, payload: str) -> bool:
    """
    Called by the ingestion modules.
    Returns True on successful publish, False otherwise (the producer already
    logs the reason and will route to the dead‑letter topic if needed).
    """
    envelope = _make_envelope(source, payload)
    return publish_message(envelope)