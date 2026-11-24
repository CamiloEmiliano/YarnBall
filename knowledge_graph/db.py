"""PostgreSQL connection helper with fail-fast store availability semantics (ADR-00036).

Raises `DocumentStoreUnavailableError` whenever PostgreSQL is unreachable unless
explicit offline mock mode (`YARNBALL_PG_MOCK=1` or `allow_mock=True`) is enabled.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional
from urllib.parse import urlparse, urlunparse

from knowledge_graph.errors import DocumentStoreUnavailableError

logger = logging.getLogger(__name__)

try:
    import psycopg
except ImportError:
    try:
        import psycopg2 as psycopg  # type: ignore[no-redef]
    except ImportError:
        psycopg = None  # type: ignore[assignment]

OperationalError = getattr(psycopg, "OperationalError", Exception) if psycopg else Exception


def is_pg_mock_enabled() -> bool:
    """Return True iff explicit offline Postgres mock mode (`YARNBALL_PG_MOCK=1`) is enabled."""
    return os.getenv("YARNBALL_PG_MOCK", "").strip().lower() in ("1", "true", "yes")


class _MockCursor:
    """Explicit opt-in offline cursor returned only when `YARNBALL_PG_MOCK=1` or `allow_mock=True`."""

    def __enter__(self) -> "_MockCursor":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def execute(self, *args: Any, **kwargs: Any) -> None:
        return None

    def executemany(self, *args: Any, **kwargs: Any) -> None:
        return None

    def fetchall(self) -> List[Any]:
        return []

    def fetchone(self) -> Optional[Any]:
        return None

    def close(self) -> None:
        return None


class _MockConnection:
    """Explicit opt-in offline connection returned only when `YARNBALL_PG_MOCK=1` or `allow_mock=True`."""

    def __enter__(self) -> "_MockConnection":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def cursor(self) -> _MockCursor:
        return _MockCursor()

    def execute(self, *args: Any, **kwargs: Any) -> None:
        return None

    def fetchall(self) -> List[Any]:
        return []

    def fetchone(self) -> Optional[Any]:
        return None

    def close(self) -> None:
        return None

    def commit(self) -> None:
        return None


def _build_dsn(database: Optional[str] = None) -> str:
    """Construct a clean DSN for PostgreSQL, respecting custom database names."""
    target_db = database or os.getenv("FINANCIAL_RAG_DB", "financial_rag")
    url = os.getenv("POSTGRES_URL")
    if url:
        parsed = urlparse(url)
        new_path = f"/{target_db}"
        return urlunparse(parsed._replace(path=new_path))

    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    host = os.getenv("POSTGRES_HOST", "localhost")
    if host == "postgres" and not os.path.exists("/.dockerenv"):
        host = "localhost"
    port = os.getenv("POSTGRES_PORT", "5432")
    return f"postgresql://{user}:{password}@{host}:{port}/{target_db}"


def pg_connection(database: Optional[str] = None, *, allow_mock: bool = False) -> Any:
    """Return a live PostgreSQL connection or raise `DocumentStoreUnavailableError` (ADR-00036).

    Prefers `tools.init_db._connect_db`, then falls back to direct DSN connection.
    Returns `_MockConnection` strictly when `YARNBALL_PG_MOCK=1` or `allow_mock=True`.
    """
    mock_allowed = allow_mock or is_pg_mock_enabled()
    target_db = database or os.getenv("FINANCIAL_RAG_DB") or "financial_rag"
    last_exc: Optional[Exception] = None

    try:
        from tools.init_db import _connect_db

        return _connect_db(target_db)
    except Exception as exc:
        last_exc = exc

    dsn = _build_dsn(target_db)
    if psycopg is not None:
        try:
            return psycopg.connect(dsn)
        except Exception as exc:
            last_exc = exc
            if mock_allowed:
                logger.warning(
                    "Could not connect to PostgreSQL at %s (%s); returning explicit mock connection (YARNBALL_PG_MOCK=1).",
                    dsn,
                    exc,
                )
                return _MockConnection()
            raise DocumentStoreUnavailableError(
                f"PostgreSQL unreachable at {dsn}: {exc}. "
                "Set YARNBALL_PG_MOCK=1 or allow_mock=True for explicit offline mock mode."
            ) from exc

    if mock_allowed:
        logger.warning("Neither psycopg nor psycopg2 is installed; returning explicit mock connection.")
        return _MockConnection()

    raise DocumentStoreUnavailableError(
        f"PostgreSQL driver unavailable ({last_exc}) and YARNBALL_PG_MOCK is not set. "
        "Install `psycopg` or explicitly set YARNBALL_PG_MOCK=1 for offline tests."
    )


def get_connection(database: Optional[str] = None, *, allow_mock: bool = False) -> Any:
    """Alias for `pg_connection` satisfying the ADR-00036 fail-fast connection contract."""
    return pg_connection(database=database, allow_mock=allow_mock)


__all__ = [
    "OperationalError",
    "_MockConnection",
    "_MockCursor",
    "get_connection",
    "is_pg_mock_enabled",
    "pg_connection",
]
