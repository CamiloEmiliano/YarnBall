"""
FastAPI Microservice for Consolidated Point-in-Time BI Retrieval & Reporting (ADR-00033 / ADR-00034 / ADR-00036).

Endpoints:
- `GET /health`: Lightweight container liveness check (ADR-00036).
- `GET /ready`: Deep readiness probe checking Memgraph (plus hydration completeness),
  PostgreSQL, `embedding_schema_metadata` alignment (ADR-00022), `verify_embedder_ready`,
  and Ollama model presence (`yarnball-qwen:7b` and `qwen3-embedding:0.6b` per ADR-00035).
- `POST /search`: Delegates directly to `retrieval.facade.query_bi` with mandatory `as_of` and `mode`.
- `POST /report`: Delegates directly to `retrieval.facade.query_bi` and returns `OperationalContagionReport`.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import embedding.embedder as _embedder
from knowledge_graph.errors import StoreUnavailableError
from knowledge_graph.ledger import (
    check_memgraph_readiness,
    check_postgres_and_schema_metadata as _check_postgres_and_schema_metadata,
)
from sft_pipeline.serving.model_client import check_ollama_models_ready
from .facade import MemgraphSubgraphStore, query_bi
from .hybrid import HybridSeeder
from .reporting import OperationalContagionReport
from .temporal import PiTContext, validate_hops

app = FastAPI(title="YarnBall BI Retrieval Service")


class SearchRequest(BaseModel):
    """Point-in-Time BI query request requiring explicit `as_of` and `mode` (ADR-00033 / ADR-00034)."""

    query: str
    as_of: date
    mode: Literal["historical", "live"] = "historical"
    include_unanchored: bool = False
    k: int = 10
    hops: int = 2
    target_ticker: Optional[str] = None
    shock_origin: Optional[str] = None
    shock_magnitude_pct: Optional[float] = None


class ReportRequest(SearchRequest):
    """Point-in-Time operational contagion report request returning `OperationalContagionReport`."""

    event_id: str
    sec_item: str = "Item 8.01"
    disrupted_entity_ticker: str
    filing_date: Optional[str] = None
    disruption_category: str = "OTHER_UNCLASSIFIED_MATERIAL_HALT"
    event_summary: str
    shock_duration_days: float = 30.0
    shock_magnitude_pct: float = 100.0
    disrupted_entity_name: Optional[str] = None
    disrupted_entity_cik: Optional[str] = None
    report_id: Optional[str] = None
    filing_timestamp: Optional[str] = None
    model_mode: str = "LIVE_LLM"


@app.get("/health")
async def health() -> JSONResponse:
    """Lightweight liveness probe (ADR-00036 Section 4.5)."""
    return JSONResponse(status_code=200, content={"status": "alive", "service": "retrieval"})


@app.get("/ready")
async def ready() -> JSONResponse:
    """Deep readiness probe checking Postgres, schema metadata, Memgraph hydration, embedder & Ollama."""
    reasons: List[str] = []

    pg_checks = _check_postgres_and_schema_metadata()
    pg_status = pg_checks["postgres"]
    meta_status = pg_checks["embedding_schema_metadata"]
    if not pg_status["ready"]:
        reasons.append(f"postgres: {pg_status.get('detail') or 'unavailable'}")
    if not meta_status["ready"]:
        reasons.append(f"embedding_schema_metadata: {meta_status.get('detail') or 'misaligned'}")

    memgraph_status = check_memgraph_readiness()
    if not memgraph_status["ready"]:
        reasons.append(f"memgraph: {memgraph_status.get('detail') or 'unavailable'}")

    embedder_status = _embedder.verify_embedder_ready(raise_on_error=False)
    if not embedder_status.get("ready"):
        reasons.append(f"embedder: {embedder_status.get('detail') or 'unavailable'}")

    ollama_status = check_ollama_models_ready()
    if not ollama_status.get("ready"):
        reasons.append(f"ollama: {ollama_status.get('detail') or 'models unavailable'}")

    payload = {
        "ready": len(reasons) == 0,
        "reasons": reasons,
        "checks": {
            "postgres": pg_status,
            "embedding_schema_metadata": meta_status,
            "memgraph": memgraph_status,
            "embedder": embedder_status,
            "ollama": ollama_status,
        },
    }
    return JSONResponse(status_code=200 if not reasons else 503, content=payload)


@app.post("/search")
async def search(req: SearchRequest) -> Dict[str, Any]:
    """Execute consolidated Point-in-Time BI retrieval via `query_bi` (ADR-00034 / ADR-00036)."""
    try:
        validated_hops = validate_hops(req.hops)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    ctx = PiTContext(
        as_of=req.as_of,
        mode=req.mode,
        include_unanchored=req.include_unanchored,
    )

    try:
        result = query_bi(
            req.query,
            ctx=ctx,
            store=MemgraphSubgraphStore(),
            seeder=HybridSeeder(top_k_dense=req.k),
            hops=validated_hops,
            target_ticker=req.target_ticker,
            shock_origin=req.shock_origin,
            shock_magnitude_pct=req.shock_magnitude_pct,
        )
    except _embedder.InvalidEmbeddingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except _embedder.EmbedderUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except StoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if result.degraded:
        raise HTTPException(
            status_code=503,
            detail={
                "degraded": True,
                "reasons": result.degraded_reasons,
                "refusal_reason": result.refusal_reason,
            },
        )

    return result.to_dict()


@app.post("/report", response_model=OperationalContagionReport)
async def report(req: ReportRequest) -> OperationalContagionReport:
    """Execute `query_bi` and serialize a validated `OperationalContagionReport` (ADR-00034 / ADR-00036)."""
    try:
        validated_hops = validate_hops(req.hops)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    ctx = PiTContext(
        as_of=req.as_of,
        mode=req.mode,
        include_unanchored=req.include_unanchored,
    )
    origin = req.shock_origin or req.disrupted_entity_ticker
    filing_date = req.filing_date or ctx.as_of_iso

    try:
        pipeline_result = query_bi(
            req.query,
            ctx=ctx,
            store=MemgraphSubgraphStore(),
            seeder=HybridSeeder(top_k_dense=req.k),
            hops=validated_hops,
            target_ticker=req.target_ticker,
            shock_origin=origin,
            shock_magnitude_pct=req.shock_magnitude_pct,
            initial_shocks={origin: req.shock_magnitude_pct},
            initial_durations={origin: req.shock_duration_days},
            initial_onsets={origin: 0.0},
        )
    except _embedder.InvalidEmbeddingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except _embedder.EmbedderUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except StoreUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if pipeline_result.degraded:
        raise HTTPException(
            status_code=503,
            detail={
                "degraded": True,
                "reasons": pipeline_result.degraded_reasons,
                "refusal_reason": pipeline_result.refusal_reason,
            },
        )

    op_report: OperationalContagionReport = pipeline_result.to_operational_report(
        event_id=req.event_id,
        sec_item=req.sec_item,
        disrupted_entity_ticker=req.disrupted_entity_ticker,
        filing_date=filing_date,
        disruption_category=req.disruption_category,
        event_summary=req.event_summary,
        shock_duration_days=req.shock_duration_days,
        shock_magnitude_pct=req.shock_magnitude_pct,
        disrupted_entity_name=req.disrupted_entity_name,
        disrupted_entity_cik=req.disrupted_entity_cik,
        report_id=req.report_id,
        filing_timestamp=req.filing_timestamp,
        model_mode=req.model_mode,
    )
    return op_report
