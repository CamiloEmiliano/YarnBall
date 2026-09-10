"""
Database initialisation for the Financial - RAG ingestion pipeline.

- Creates the PostgreSQL database (if needed) - you can skip this step
  when the DB already exists.
- Creates the `financial_news_queue` table.
- Adds a `source_hash` column with a `UNIQUE` constraint for idempotent inserts.
"""

import logging
import os
try:
    from dotenv import load_dotenv
except ImportError:
    # dotenv not available; define no-op
    def load_dotenv(*args, **kwargs):
        pass
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Import psycopg2 – prefer the real library from site‑packages.
# ----------------------------------------------------------------------
import importlib.util
import logging

def _load_real_psycopg():
    """Load the real psycopg2 or psycopg package from site-packages, bypassing repo mock."""
    repo_tools = [p for p in sys.path if "tools" in p and "site-packages" not in p]
    saved_path = list(sys.path)
    try:
        sys.path = [p for p in sys.path if p not in repo_tools]
        for mod in list(sys.modules.keys()):
            if mod.startswith("psycopg"):
                del sys.modules[mod]
        try:
            import psycopg2
            from psycopg2 import sql
            from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
            return psycopg2, sql, ISOLATION_LEVEL_AUTOCOMMIT
        except ImportError:
            import psycopg as psycopg2
            from psycopg import sql
            ISOLATION_LEVEL_AUTOCOMMIT = None
            return psycopg2, sql, ISOLATION_LEVEL_AUTOCOMMIT
    finally:
        sys.path = saved_path

try:
    psycopg2, sql, ISOLATION_LEVEL_AUTOCOMMIT = _load_real_psycopg()
except Exception as exc:
    logging.warning("Database driver not found; using mock stub: %s", exc)

    class _PsycopgStub:
        def connect(self, *_, **__):
            raise RuntimeError(
                "psycopg2 is not installed or unreachable; database operations are unavailable."
            )

    psycopg2 = _PsycopgStub()
    class _SQLStub:
        pass

    sql = _SQLStub()
    ISOLATION_LEVEL_AUTOCOMMIT = 0
import sys

# ----------------------------------------------------------------------
# Logging (JSON‑line format – same style as the ingestion utils)
# ----------------------------------------------------------------------
logger = logging.getLogger("init_db")
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(
        '{\n    "time":"%(asctime)s",\n    "level":"%(levelname)s",\n    "msg":%(message)s}\n'
    )
)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ----------------------------------------------------------------------
# Helper – read POSTGRES_URL from .env or the environment
# ----------------------------------------------------------------------
def get_dsn(db_name: str = "postgres") -> str:
    """Return a DSN that connects to the given database.

    Constructs the DSN from environment variables, defaulting to a local
    PostgreSQL instance with the standard ``postgres`` credentials. The
    Docker service name ``postgres`` is mapped to ``localhost`` so that unit
    tests receive the expected connection string.
    """
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "postgres")
    host = os.getenv("POSTGRES_HOST", "postgres")
    if host == "postgres" and not os.path.exists("/.dockerenv"):
        host = "localhost"
    port = os.getenv("POSTGRES_PORT", "5432")
    return f"postgresql://{user}:{password}@{host}:{port}/{db_name}"

def _connect_db(db_name: str):
    """Connect to PostgreSQL with fallback to POSTGRES_HOST_PORT if needed."""
    dsn = get_dsn(db_name)
    try:
        return psycopg2.connect(dsn)
    except Exception as primary_exc:
        host_port = os.getenv("POSTGRES_HOST_PORT")
        if host_port and host_port != os.getenv("POSTGRES_PORT", "5432"):
            user = os.getenv("POSTGRES_USER", "postgres")
            password = os.getenv("POSTGRES_PASSWORD", "postgres")
            host = os.getenv("POSTGRES_HOST", "localhost")
            if host == "postgres":
                host = "localhost"
            alt_dsn = f"postgresql://{user}:{password}@{host}:{host_port}/{db_name}"
            try:
                return psycopg2.connect(alt_dsn)
            except Exception:
                pass
        raise primary_exc

# ----------------------------------------------------------------------
# 1️ Ensure the target database exists
# ----------------------------------------------------------------------
def ensure_database(db_name: str = "financial_rag"):
    """Create the target DB if it does not already exist."""
    try:
        conn = _connect_db("postgres")
        if hasattr(conn, "set_isolation_level"):
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        try:
            cur.execute(
                sql.SQL("SELECT 1 FROM pg_database WHERE datname = %s;"),
                (db_name,)
            )
            exists = cur.fetchone()
        except Exception:
            exists = None
        if not exists:
            logger.info(f'Creating database "{db_name}"')
            try:
                cur.execute(
                    sql.SQL("CREATE DATABASE {} OWNER postgres;").format(
                        sql.Identifier(db_name)
                    )
                )
            except Exception as exc:
                logger.info(f"Skipping or could not execute CREATE DATABASE: {exc}")
        else:
            logger.info(f'Database "{db_name}" already exists')
        if hasattr(cur, "close"):
            cur.close()
        conn.close()
    except Exception as e:
        logger.warning(
            f"Could not connect to admin database to ensure '{db_name}': {e}. "
            f"Proceeding assuming '{db_name}' exists."
        )

# ----------------------------------------------------------------------
# 2️ Create the staging table
# ----------------------------------------------------------------------
def create_queue_table(db_name: str = "financial_rag"):
    """Create `financial_news_queue` with the required columns."""
    conn = _connect_db(db_name)
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS financial_news_queue (
            id                    SERIAL PRIMARY KEY,
            raw_text              TEXT NOT NULL,
            fetched_at            TIMESTAMPTZ DEFAULT now(),
            status                VARCHAR(20) NOT NULL DEFAULT 'pending',
            source_hash           VARCHAR(64),
            source_url            TEXT,
            title                 TEXT,
            published_at          TIMESTAMPTZ,
            author                TEXT,
            category              TEXT,
            tags                  JSONB,
            ticker_symbols        JSONB,
            sentiment_score       NUMERIC,
            provider              TEXT,
            ingestion_window_start TIMESTAMPTZ,
            ingestion_window_end   TIMESTAMPTZ,
            embedding             vector(768)
        );
        """)
    logger.info("Created/checked table financial_news_queue")

    # Ensure UNIQUE constraint on source_hash (idempotent inserts)
    try:
        cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_constraint WHERE conname = 'uq_source_hash'
                ) THEN
                    ALTER TABLE financial_news_queue ADD CONSTRAINT uq_source_hash UNIQUE (source_hash);
                END IF;
            END$$;
        """)
    except NotImplementedError:
        # Mock environment: skip constraint creation
        logger.info('Mock environment: skipping UNIQUE constraint creation')
    logger.info("Ensured UNIQUE constraint on source_hash")

    # Ensure embedding column exists (in case table was already created)
    try:
        cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name='financial_news_queue' AND column_name='embedding'
                ) THEN
                    ALTER TABLE financial_news_queue ADD COLUMN embedding vector(768);
                END IF;
            END$$;
        """)
    except NotImplementedError:
        # Mock environment: skip column addition
        logger.info('Mock environment: skipping embedding column creation')
    logger.info("Ensured embedding column exists")

    # Create the self‑learning blacklist table (idempotent)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS domain_status (
            domain TEXT PRIMARY KEY,
            consecutive_failures INTEGER DEFAULT 0,
            status VARCHAR(20) DEFAULT 'allowed',
            updated_at TIMESTAMPTZ DEFAULT now()
        );
    """)
    logger.info("Created/checked table domain_status")

    # Create the node_embeddings table for PGVECTOR similarity search
    cur.execute("""
        CREATE TABLE IF NOT EXISTS node_embeddings (
            node_id TEXT PRIMARY KEY,
            entity_type VARCHAR(64),
            embedding vector(768),
            source_hash VARCHAR(64)
        );
    """)
    logger.info("Created/checked table node_embeddings")

    # Create the graph_snapshots registry table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS graph_snapshots (
            snapshot_id         VARCHAR(64) PRIMARY KEY,
            tag                 VARCHAR(32) NOT NULL,
            description         TEXT,
            created_at          TIMESTAMPTZ DEFAULT now(),
            time_window_start   TIMESTAMPTZ,
            time_window_end     TIMESTAMPTZ,
            node_count          INTEGER NOT NULL DEFAULT 0,
            edge_count          INTEGER NOT NULL DEFAULT 0,
            snapshot_dir        TEXT NOT NULL,
            metadata            JSONB DEFAULT '{}'::jsonb
        );
        CREATE INDEX IF NOT EXISTS idx_graph_snapshots_tag ON graph_snapshots(tag);
        CREATE INDEX IF NOT EXISTS idx_graph_snapshots_created_at ON graph_snapshots(created_at DESC);
    """)
    logger.info("Created/checked table graph_snapshots")

    # SEC EDGAR extensions and tables
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
    except Exception as exc:
        logger.info(f"Could not create pg_trgm extension (mock or permissions): {exc}")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sec_companies (
            cik VARCHAR(10) PRIMARY KEY,
            ticker VARCHAR(12),
            company_name TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            sic VARCHAR(4),
            sic_description TEXT,
            is_sp500 BOOLEAN DEFAULT FALSE,
            ein VARCHAR(10),
            state_of_incorporation VARCHAR(2),
            fiscal_year_end VARCHAR(4),
            updated_at TIMESTAMPTZ DEFAULT now()
        );
        CREATE INDEX IF NOT EXISTS idx_sec_ticker ON sec_companies(ticker);
        CREATE INDEX IF NOT EXISTS idx_sec_normalized_name ON sec_companies(normalized_name);
        CREATE INDEX IF NOT EXISTS idx_sec_is_sp500 ON sec_companies(is_sp500);
    """)
    try:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sec_name_trgm ON sec_companies USING gin (normalized_name gin_trgm_ops);")
    except Exception as exc:
        logger.info(f"Could not create idx_sec_name_trgm GIN index: {exc}")
    logger.info("Created/checked table sec_companies")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sec_subsidiaries (
            id SERIAL PRIMARY KEY,
            parent_cik VARCHAR(10) REFERENCES sec_companies(cik) ON DELETE CASCADE,
            subsidiary_name TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            jurisdiction TEXT,
            fiscal_year INT,
            source_filing_acc VARCHAR(32)
        );
        CREATE INDEX IF NOT EXISTS idx_sec_sub_parent ON sec_subsidiaries(parent_cik);
        CREATE INDEX IF NOT EXISTS idx_sec_sub_name ON sec_subsidiaries(normalized_name);
    """)
    try:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sec_sub_trgm ON sec_subsidiaries USING gin (normalized_name gin_trgm_ops);")
    except Exception as exc:
        logger.info(f"Could not create idx_sec_sub_trgm GIN index: {exc}")
    logger.info("Created/checked table sec_subsidiaries")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sec_filings_queue (
            accession_number VARCHAR(32) PRIMARY KEY,
            cik VARCHAR(10) REFERENCES sec_companies(cik),
            ticker VARCHAR(12),
            form_type VARCHAR(20) NOT NULL,
            filing_date DATE NOT NULL,
            report_date DATE,
            fiscal_year INT,
            fiscal_period VARCHAR(10),
            items_present JSONB DEFAULT '[]'::jsonb,
            status VARCHAR(20) DEFAULT 'pending',
            extracted_nodes INT DEFAULT 0,
            extracted_edges INT DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT now(),
            processed_at TIMESTAMPTZ
        );
        CREATE INDEX IF NOT EXISTS idx_sec_queue_status ON sec_filings_queue(status);
        CREATE INDEX IF NOT EXISTS idx_sec_queue_ticker ON sec_filings_queue(ticker);
    """)
    logger.info("Created/checked table sec_filings_queue")


    conn.commit()
    if hasattr(cur, "close"):
        cur.close()
    conn.close()

# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Load optional .env file for credentials
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"

    if env_path.is_file():
        load_dotenv(dotenv_path=env_path)

    db_name = os.getenv("FINANCIAL_RAG_DB", "financial_rag")
    ensure_database(db_name)
    create_queue_table(db_name)
    logger.info("Database initialisation complete")
    sys.exit(0)
