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
