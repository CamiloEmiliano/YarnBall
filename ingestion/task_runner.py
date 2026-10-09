"""
Task runner for ingestion clients.

Discovers and executes all registered ingestion tasks concurrently in a thread pool.
"""

from __future__ import annotations

import asyncio
import logging

from . import registry

logger = logging.getLogger("ingestion.task_runner")


async def _run_sync(func):
    """Run a synchronous fetch function in a thread executor."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, func)


async def run_all():
    """Execute all fetch functions registered via the @ingest_task decorator.

    Each fetch function is executed concurrently via `_run_sync`.
    """
    if not registry:
        logger.warning("No ingestion fetch functions registered.")
        return
    tasks = [_run_sync(fn) for fn in registry.values()]
    await asyncio.gather(*tasks)
