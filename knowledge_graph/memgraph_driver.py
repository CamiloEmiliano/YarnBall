# knowledge_graph/memgraph_driver.py
# -*- coding: utf-8 -*-
"""Memgraph database driver utility (ADR-00036 fail-fast contract).

Provides `get_memgraph_driver()` to connect to a live Memgraph instance via the
Bolt protocol (`neo4j` driver). When Memgraph is unreachable, raises
`GraphStoreUnavailableError` unless `YARNBALL_GRAPH_MOCK=1` or explicit
`allow_mock=True` is set.
"""

from __future__ import annotations

import atexit
import logging
import os
from pathlib import Path
from typing import Any, Optional, Tuple

from knowledge_graph.errors import GraphStoreUnavailableError

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Neo4j / Memgraph Bolt driver imports with optional fallback
try:
    from neo4j import Driver, GraphDatabase
    from neo4j.exceptions import Neo4jError, ServiceUnavailable
except ImportError:  # pragma: no cover
    GraphDatabase = None  # type: ignore[assignment,misc,no-redef]
    Driver = None  # type: ignore[assignment,misc,no-redef]
    ServiceUnavailable = Exception  # type: ignore[assignment,misc,no-redef]
    Neo4jError = Exception  # type: ignore[assignment,misc,no-redef]


def is_graph_mock_enabled() -> bool:
    """Return True iff explicit offline graph mock mode (`YARNBALL_GRAPH_MOCK=1`) is enabled."""
    return os.getenv("YARNBALL_GRAPH_MOCK", "").strip().lower() in ("1", "true", "yes")


class _DummySession:
    """In-memory dummy session for explicit offline mock testing (`YARNBALL_GRAPH_MOCK=1`)."""

    def __init__(self, ids: Optional[list[Any]] = None):
        self.ids = ids or []

    def __enter__(self) -> "_DummySession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def run(self, query: str, *args: Any, **kwargs: Any) -> list[Any]:
        class _Record(dict):
            def __getitem__(self, key: Any) -> Any:
                return dict.__getitem__(self, key)

        records = []
        ids = kwargs.get("ids", self.ids)
        if ids:
            for nid in ids:
                records.append(
                    _Record(
                        {
                            "nid": nid,
                            "nodes": [{"id": nid}],
                            "rels": [],
                            "neighbors": [],
                        }
                    )
                )
        return records


class _DummyDriver:
    """Dummy driver compatible with the Neo4j/Memgraph driver interface (explicit opt-in only)."""

    def session(self, *args: Any, **kwargs: Any) -> _DummySession:
        return _DummySession()

    def verify_connectivity(self) -> None:
        return None

    def close(self) -> None:
        return None


_driver_instance: Optional[Any] = None


def reset_memgraph_driver_cache() -> None:
    """Clear the module-level cached Memgraph driver reference without raising."""
    global _driver_instance
    _driver_instance = None


def get_memgraph_driver(
    uri: Optional[str] = None,
    auth: Optional[Tuple[str, str]] = None,
    *,
    allow_mock: bool = False,
) -> Any:
    """Return a live Memgraph Bolt driver or raise `GraphStoreUnavailableError` (ADR-00036).

    Explicit offline mock mode is enabled strictly via `YARNBALL_GRAPH_MOCK=1` or `allow_mock=True`.
    """
    global _driver_instance
    mock_allowed = allow_mock or is_graph_mock_enabled()

    if _driver_instance is not None:
        return _driver_instance

    if mock_allowed and os.getenv("YARNBALL_GRAPH_MOCK_FORCE", "").strip().lower() in ("1", "true", "yes"):
        return _DummyDriver()

    if GraphDatabase is None:
        if mock_allowed:
            logger.warning("neo4j library not available; returning explicit mock _DummyDriver.")
            return _DummyDriver()
        raise GraphStoreUnavailableError(
            "neo4j Bolt driver is not installed and YARNBALL_GRAPH_MOCK is not set. "
            "Install `neo4j` or explicitly set YARNBALL_GRAPH_MOCK=1 for offline tests."
        )

    host = os.getenv("MEMGRAPH_HOST", "localhost")
    # In local development outside Docker, adjust 'memgraph' host to localhost if needed
    if host == "memgraph" and not os.path.exists("/.dockerenv"):
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
        if mock_allowed:
            logger.warning(
                "Memgraph unreachable at %s (%s); returning explicit mock _DummyDriver (YARNBALL_GRAPH_MOCK=1).",
                target_uri,
                exc,
            )
            return _DummyDriver()
        raise GraphStoreUnavailableError(
            f"Memgraph unreachable at {target_uri}: {exc}. "
            "Set YARNBALL_GRAPH_MOCK=1 or allow_mock=True for explicit offline mock mode."
        ) from exc


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
