"""
FastAPI Microservice for Point-in-Time Hybrid Retrieval & Node Enrichment (ADR-00033).
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Literal, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import embedding.embedder as _embedder
from knowledge_graph.db import pg_connection
from .enricher import enrich_nodes
from .temporal import PiTContext, validate_hops

app = FastAPI(title="YarnBall Hybrid Retrieval Service")


class SearchRequest(BaseModel):
    """Point-in-Time semantic entity search request requiring explicit `as_of` (ADR-00033)."""

    query: str
    as_of: date
    mode: Literal["historical", "live"] = "historical"
    include_unanchored: bool = False
    k: int = 10
    hops: int = 1


@app.get("/health")
async def health() -> JSONResponse:
    """Surface embedder readiness telemetry without degrading silently."""
    status = _embedder.verify_embedder_ready(raise_on_error=False)
    return JSONResponse(status_code=200 if status["ready"] else 503, content=status)


@app.post("/search")
async def search(req: SearchRequest) -> List[Dict[str, Any]]:
    """Execute Point-in-Time dense entity search over SCD Type 2 `node_embeddings` + enrichment."""
    try:
        validated_hops = validate_hops(req.hops)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    ctx = PiTContext(
        as_of=req.as_of,
        mode=req.mode,
        include_unanchored=req.include_unanchored,
    )

    # 1. Embed, validate, and serialize the user query (ADR-00020 & ADR-00021)
    try:
        raw_vectors = _embedder.embed_texts([req.query])
        query_vec_literal = _embedder.format_pgvector_literal(raw_vectors[0] if raw_vectors else [])
    except _embedder.InvalidEmbeddingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except _embedder.EmbedderUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # 2. Retrieve top-k Point-in-Time node IDs from SCD Type 2 `node_embeddings` (ADR-00033)
    if ctx.mode == "live":
        sql = """
            SELECT node_id FROM node_embeddings
            WHERE valid_to IS NULL
              AND valid_from IS NOT NULL
              AND valid_from <= %s::date
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        params: Tuple[Any, ...] = (ctx.as_of_iso, query_vec_literal, req.k)
    else:
        sql = """
            SELECT node_id FROM node_embeddings
            WHERE COALESCE(is_unanchored, FALSE) = FALSE
              AND valid_from IS NOT NULL
              AND valid_from <= %s::date
              AND (valid_to IS NULL OR valid_to > %s::date)
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        params = (ctx.as_of_iso, ctx.as_of_iso, query_vec_literal, req.k)

    with pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            ids = [row[0] for row in cur.fetchall()]

    if not ids:
        raise HTTPException(status_code=404, detail="No similar nodes found")

    # 3. Enrich those IDs with Point-in-Time graph context from Memgraph
    payload = enrich_nodes(ids, ctx=ctx, hops=validated_hops)
    return payload
