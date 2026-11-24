"""
Graph Neighborhood Enrichment Module (ADR-00033).

Extracts bounded, Point-in-Time k-hop ego-networks for specified entity node IDs
from Memgraph for node-detail display.
"""

from __future__ import annotations

from typing import Any, Dict, List

import knowledge_graph.memgraph_driver as memgraph_driver
from .temporal import DEFAULT_TEMPORAL_FILTER, PiTContext, validate_hops


def enrich_nodes(
    node_ids: List[str],
    ctx: PiTContext,
    hops: int = 1,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Pull each node plus its bounded, Point-in-Time `hops`-hop neighborhood for node-detail display.

    Note (ADR-00033 Section 4.4): Traversal remains undirected strictly for node-detail display
    and is bounded by validated `hops` (`int` in `[1, 10]`), `TemporalQueryFilter`, and `LIMIT`.
    """
    if not isinstance(ctx, PiTContext):
        raise TypeError(f"ctx must be a PiTContext instance, got {type(ctx).__name__}")
    if not node_ids:
        return []

    validated_hops = validate_hops(hops)
    safe_limit = max(1, min(int(limit), 500))
    driver = memgraph_driver.get_memgraph_driver()

    base_query = f"""
        UNWIND $ids AS nid
        MATCH (n {{id: nid}})
        OPTIONAL MATCH (n)-[r*1..{validated_hops}]-(m)
        RETURN nid,
               collect(DISTINCT n) AS nodes,
               collect(DISTINCT r) AS rels,
               collect(DISTINCT m) AS neighbors
        ORDER BY nid ASC
        LIMIT $limit;
    """
    query = DEFAULT_TEMPORAL_FILTER.inject_active_coalescence(base_query, ctx)

    with driver.session() as session:
        result = session.run(query, ids=node_ids, limit=safe_limit, as_of_date=ctx.as_of_iso)
        payload: List[Dict[str, Any]] = []
        for record in result:
            payload.append(
                {
                    "root_id": record["nid"],
                    "nodes": [dict(r) for r in record["nodes"]],
                    "relationships": [dict(r) for r in record["rels"]],
                    "neighbors": [dict(r) for r in record["neighbors"]],
                }
            )
        return payload


def enrich_node(
    node_id: str,
    ctx: PiTContext,
    hops: int = 1,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Convenience single-node wrapper around `enrich_nodes` requiring `PiTContext`."""
    return enrich_nodes([node_id], ctx=ctx, hops=hops, limit=limit)
