"""
Graph Neighborhood Enrichment Module.

Extracts k-hop subgraphs and ego-networks for specified entity node IDs
from Memgraph to furnish structured context for retrieval.
"""

from __future__ import annotations

from typing import Any, Dict, List
import graph.memgraph_driver as memgraph_driver


def enrich_nodes(node_ids: List[str], hops: int = 1) -> List[Dict[str, Any]]:
    """Given a list of node IDs, pull each node plus its `hops`-hop neighborhood
    from Memgraph, returning a JSON-serializable structure.
    """
    if not node_ids:
        return []

    driver = memgraph_driver.get_memgraph_driver()
    query = f"""
        UNWIND $ids AS nid
        MATCH (n {{id: nid}})
        OPTIONAL MATCH (n)-[r*1..{hops}]-(m)
        RETURN nid,
               collect(DISTINCT n) AS nodes,
               collect(DISTINCT r) AS rels,
               collect(DISTINCT m) AS neighbors;
    """
    with driver.session() as session:
        result = session.run(query, ids=node_ids)
        payload = []
        for record in result:
            payload.append({
                "root_id": record["nid"],
                "nodes": [dict(r) for r in record["nodes"]],
                "relationships": [dict(r) for r in record["rels"]],
                "neighbors": [dict(r) for r in record["neighbors"]],
            })
        return payload


def enrich_node(node_id: str, hops: int = 1) -> List[Dict[str, Any]]:
    """Convenience single-node wrapper around enrich_nodes."""
    return enrich_nodes([node_id], hops=hops)
