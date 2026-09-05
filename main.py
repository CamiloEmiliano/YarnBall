import asyncio
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from ingest import task_runner

if sys.version_info < (3, 13):
    raise RuntimeError("Python 3.13 or greater is required.")


async def run_all():
    """Execute all registered ingestion tasks concurrently."""
    await task_runner.run_all()


if __name__ == "__main__":
    asyncio.run(run_all())
