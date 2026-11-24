"""Fail-fast storage availability exceptions for YarnBall GraphRAG (ADR-00036).

An unreachable graph or relational store must never masquerade as an empty
knowledge graph or "no supply-chain exposure" downstream.
"""

from __future__ import annotations


class StoreUnavailableError(RuntimeError):
    """Base exception raised when a backing store (Memgraph or PostgreSQL) is unreachable."""


class GraphStoreUnavailableError(StoreUnavailableError):
    """Raised when Memgraph cannot be reached and `YARNBALL_GRAPH_MOCK=1` is not set."""


class DocumentStoreUnavailableError(StoreUnavailableError):
    """Raised when PostgreSQL cannot be reached and `YARNBALL_PG_MOCK=1` is not set."""


__all__ = [
    "StoreUnavailableError",
    "GraphStoreUnavailableError",
    "DocumentStoreUnavailableError",
]
