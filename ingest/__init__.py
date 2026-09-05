"""Ingestion package - entry point for the collection of micro-scripts."""
from typing import Callable, Dict

# Central registry mapping task names to fetch functions
registry: Dict[str, Callable] = {}

def ingest_task(name: str):
    """Decorator that registers a fetch_* function in the central registry.

    Example::
        @ingest_task("finnhub")
        def fetch_finnhub():
            ...
    """
    def wrapper(fn: Callable) -> Callable:
        registry[name] = fn
        return fn
    return wrapper

__all__ = ["registry", "ingest_task"]