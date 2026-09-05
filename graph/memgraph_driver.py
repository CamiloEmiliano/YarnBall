# graph/memgraph_driver.py
# -*- coding: utf-8 -*-
"""Memgraph database driver utility.

Provides `get_memgraph_driver()` to connect to a live Memgraph instance via the
Bolt protocol (`neo4j` driver), with an automatic in-memory fallback for offline
development and unit testing.
"""

import os
import logging
import atexit
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Neo4j / Memgraph Bolt driver imports with fallback
try:
    from neo4j import GraphDatabase, Driver
    from neo4j.exceptions import ServiceUnavailable, Neo4jError
except ImportError:  # pragma: no cover
    GraphDatabase = None
    Driver = None
    ServiceUnavailable = Exception
    Neo4jError = Exception


class _DummySession:
    """In-memory dummy session for testing without a live Memgraph instance."""

    def __init__(self, ids=None):
        self.ids = ids or []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query, *args, **kwargs):
        class _Record(dict):
            def __getitem__(self, key):
                return dict.__getitem__(self, key)

        records = []
        ids = kwargs.get("ids", self.ids)
        if ids:
            for nid in ids:
                records.append(_Record({
                    "nid": nid,
                    "nodes": [{"id": nid}],
                    "rels": [],
                    "neighbors": [],
                }))
        return records


class _DummyDriver:
    """Dummy driver compatible with the Neo4j/Memgraph driver interface."""

    def session(self, *args, **kwargs):
        return _DummySession()

    def verify_connectivity(self):
        return None

    def close(self):
        return None


_driver_instance = None


def get_memgraph_driver(uri: str | None = None, auth: tuple | None = None):
    """Return a live Memgraph driver if reachable, or a dummy driver for unit testing."""
    global _driver_instance
    if _driver_instance is not None:
        return _driver_instance

    if GraphDatabase is None:
        logger.warning("neo4j library not available; using dummy Memgraph driver.")
        return _DummyDriver()

    host = os.getenv("MEMGRAPH_HOST", "localhost")
    # In local development outside Docker, adjust 'memgraph' host to localhost if needed
    if host == "memgraph":
        import socket
        try:
            with socket.create_connection(("memgraph", 7687), timeout=0.5):
                pass
        except Exception:
            host = "localhost"

    port = os.getenv("MEMGRAPH_PORT", "7687")
    target_uri = uri or f"bolt://{host}:{port}"

    user = os.getenv("MEMGRAPH_USER", "")
    password = os.getenv("MEMGRAPH_PASSWORD", "")
    auth_tuple = auth if auth is not None else ((user, password) if (user or password) else None)

    try:
        driver = GraphDatabase.driver(target_uri, auth=auth_tuple)
        driver.verify_connectivity()
        logger.info("Connected to live Memgraph instance at %s", target_uri)
        _driver_instance = driver
        return driver
    except Exception as exc:
        logger.info("Memgraph not reachable at %s (%s); using fallback dummy driver.", target_uri, exc)
        return _DummyDriver()


def close_memgraph_driver() -> None:
    """Close the active Memgraph driver instance if open."""
    global _driver_instance
    if _driver_instance is not None:
        try:
            _driver_instance.close()
        except Exception:
            pass
        _driver_instance = None


atexit.register(close_memgraph_driver)
