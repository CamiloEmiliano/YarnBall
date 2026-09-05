from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from embedding.embedder import embed_texts
from graph.db import pg_connection
from rag.graph_enricher import enrich_nodes

app = FastAPI()

class SearchRequest(BaseModel):
    query: str
    k: int = 10  # number of nearest neighbours to return

@app.post("/search")
async def search(req: SearchRequest):
    # 1️⃣ Embed the user query
    query_vec = embed_texts([req.query])[0]

    # 2️⃣ Retrieve top‑k node IDs from PGVECTOR (cosine distance)
    sql = """
        SELECT node_id FROM node_embeddings
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """
    with pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (query_vec, req.k))
            ids = [row[0] for row in cur.fetchall()]

    if not ids:
        raise HTTPException(status_code=404, detail="No similar nodes found")

    # 3️⃣ Enrich those IDs with graph context from Memgraph
    payload = enrich_nodes(ids, hops=1)   # hop depth = 1 (user‑chosen)
    return payload
