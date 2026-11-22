"""
FastAPI Microservice for Hybrid Retrieval & Node Enrichment.
"""

from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import embedding.embedder as _embedder
from knowledge_graph.db import pg_connection
from .enricher import enrich_nodes

app = FastAPI(title="YarnBall Hybrid Retrieval Service")


class SearchRequest(BaseModel):
    query: str
    k: int = 10  # number of nearest neighbours to return


@app.get("/health")
async def health() -> JSONResponse:
    """Surface embedder readiness telemetry without degrading silently."""
    status = _embedder.verify_embedder_ready(raise_on_error=False)
    return JSONResponse(status_code=200 if status["ready"] else 503, content=status)


@app.post("/search")
async def search(req: SearchRequest) -> List[Dict[str, Any]]:
    # 1. Embed and validate the user query (prohibit zero vectors / unavailable embedder)
    try:
        raw_vectors = _embedder.embed_texts([req.query])
        query_vec = _embedder.validate_embedding_vectors(raw_vectors)[0]
    except _embedder.InvalidEmbeddingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except _embedder.EmbedderUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # 2. Retrieve top-k node IDs from PGVECTOR (cosine distance)
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

    # 3. Enrich those IDs with graph context from Memgraph
    payload = enrich_nodes(ids, hops=1)
    return payload
