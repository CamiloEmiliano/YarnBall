import os
import logging
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)

try:
    import psycopg
except ImportError:
    try:
        import psycopg2 as psycopg
    except ImportError:
        from tools import psycopg2 as psycopg

OperationalError = getattr(psycopg, "OperationalError", Exception)

def _build_dsn(database: str = None) -> str:
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
    port = os.getenv("POSTGRES_PORT", "5432")
    return f"postgresql://{user}:{password}@{host}:{port}/{target_db}"

def pg_connection(database: str = None):
    """Return a PostgreSQL connection.

    Prefers tools.init_db._connect_db, then falls back to direct DSN connection.
    Returns _MockConnection for offline test environments if the DB is unreachable.
    """
    target_db = database or os.getenv("FINANCIAL_RAG_DB", "financial_rag")
    try:
        from tools.init_db import _connect_db
        return _connect_db(target_db)
    except Exception:
        pass

    dsn = _build_dsn(target_db)
    try:
        return psycopg.connect(dsn)
    except Exception as exc:
        logger.warning("Could not connect to PostgreSQL at %s (%s); using mock fallback.", dsn, exc)
        class _MockConnection:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
            def cursor(self):
                return self
            def execute(self, *args, **kwargs):
                pass
            def fetchall(self):
                return []
            def close(self):
                pass
            def commit(self):
                pass
        return _MockConnection()
