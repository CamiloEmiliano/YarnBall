# ingest/task_runner.py
# -*- coding: utf-8 -*-
"""Task runner for ingestion clients.

Discovers and executes all registered ingestion tasks concurrently in a thread pool.
"""

import asyncio
import logging

# Import registry and client modules to ensure @ingest_task decorators execute
from . import registry
from . import finnhub_client

logger = logging.getLogger(__name__)


async def _run_sync(func):
    """Run a synchronous fetch function in a thread executor."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, func)


async def run_all():
    """Execute all fetch functions registered via the @ingest_task decorator.

    Each fetch function is executed concurrently via `_run_sync`.
    """
    tasks = [_run_sync(fn) for fn in registry.values()]
    if not tasks:
        logger.warning("No ingestion fetch functions registered.")
        return
    await asyncio.gather(*tasks)
