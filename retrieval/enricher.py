"""
Graph Neighborhood Enrichment Module (ADR-00033 / ADR-00034).

Extracts bounded, Point-in-Time k-hop ego-networks for specified entity node IDs
strictly for node-detail display via `MemgraphSubgraphStore.fetch_node_details`
(no longer used to construct LLM retrieval context under ADR-00034).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .facade import MemgraphSubgraphStore
from .temporal import PiTContext, validate_hops


def enrich_nodes(
    node_ids: List[str],
    ctx: PiTContext,
    hops: int = 1,
    limit: int = 50,
    *,
    store: Optional[MemgraphSubgraphStore] = None,
) -> List[Dict[str, Any]]:
    """Pull each node plus its bounded, Point-in-Time `hops`-hop neighborhood for node-detail display.

    Note (ADR-00033 Section 4.4 / ADR-00034 Section 4.5): Traversal remains undirected strictly
    for node-detail display, delegates Memgraph execution to `MemgraphSubgraphStore`, and is
    never used to build LLM context.
    """
    if not isinstance(ctx, PiTContext):
        raise TypeError(f"ctx must be a PiTContext instance, got {type(ctx).__name__}")
    if not node_ids:
        return []

    validated_hops = validate_hops(hops)
    safe_limit = max(1, min(int(limit), 500))
    active_store = store or MemgraphSubgraphStore()
    return active_store.fetch_node_details(
        node_ids=node_ids,
        ctx=ctx,
        hops=validated_hops,
        limit=safe_limit,
    )


def enrich_node(
    node_id: str,
    ctx: PiTContext,
    hops: int = 1,
    limit: int = 50,
    *,
    store: Optional[MemgraphSubgraphStore] = None,
) -> List[Dict[str, Any]]:
    """Convenience single-node wrapper around `enrich_nodes` requiring `PiTContext`."""
    return enrich_nodes([node_id], ctx=ctx, hops=hops, limit=limit, store=store)
