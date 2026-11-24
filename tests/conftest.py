import os
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load environment variables early for test run
load_dotenv(dotenv_path=ROOT / ".env")

# Build the host DSN pointing to the exposed port to avoid docker-to-host conflicts
user = os.getenv("POSTGRES_USER", "postgres")
password = os.getenv("POSTGRES_PASSWORD", "postgres")
host = "localhost"
port = os.getenv("POSTGRES_HOST_PORT", "5432")
database = os.getenv("FINANCIAL_RAG_DB", "financial_rag")
host_dsn = f"postgresql://{user}:{password}@{host}:{port}/{database}"

# Override POSTGRES_URL for all tests running on the host system
os.environ["POSTGRES_URL"] = host_dsn

# Ensure EDGAR library writes to /tmp instead of read-only ~/.edgar
os.environ["EDGAR_LOCAL_DATA_DIR"] = "/tmp/edgar"
Path("/tmp/edgar").mkdir(parents=True, exist_ok=True)

import pytest


@pytest.fixture(autouse=True)
def _adr00036_offline_mock_opt_in(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enable explicit store mocks ONLY for tests marked `@pytest.mark.offline` (ADR-00036)."""
    from knowledge_graph.memgraph_driver import reset_memgraph_driver_cache

    reset_memgraph_driver_cache()
    if request.node.get_closest_marker("offline") is not None:
        monkeypatch.setenv("YARNBALL_GRAPH_MOCK", "1")
        monkeypatch.setenv("YARNBALL_PG_MOCK", "1")
    else:
        monkeypatch.delenv("YARNBALL_GRAPH_MOCK", raising=False)
        monkeypatch.delenv("YARNBALL_PG_MOCK", raising=False)
