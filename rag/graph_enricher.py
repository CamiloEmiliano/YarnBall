from graph.memgraph_driver import get_memgraph_driver

def enrich_nodes(node_ids: list[str], hops: int = 1):
    """Given a list of node IDs, pull each node plus its `hops`‑hop neighborhood
    from Memgraph, returning a JSON‑serialisable structure.
    """
    driver = get_memgraph_driver()
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
