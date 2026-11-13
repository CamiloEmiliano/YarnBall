"""Local FastAPI workbench backend exposing read-only PiT KG, 8-K audit, and cohort APIs (ADR-00017)."""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from ingestion.sec.disruption_gating import (
    SanitizedDisruptionAssessment,
    assess_sec_8k_disruption,
)
from retrieval.offline_store import InMemorySubgraphStore
from retrieval.reporting import (
    ContagionTimeSeriesEventRecord,
    OperationalContagionReport,
    flatten_report_to_timeseries_records,
    validate_verbatim_citations,
)
from sft_pipeline.serving.model_client import YarnBallModelClient
from tools.dry_run_sec_8k_contagion import run_sec_8k_dry_run

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATIC_DIR = Path(__file__).resolve().parent / "static"


# ==============================================================================
# Pydantic Request / Response Contracts (ADR-00017 Section 4.1)
# ==============================================================================


class SnapshotEntry(BaseModel):
    """Discovered local Parquet snapshot directory."""

    snapshot_id: str
    path: str
    has_nodes: bool
    has_edges: bool
    edge_count: int
    tickers: List[str] = Field(default_factory=list)


class FilingEntry(BaseModel):
    """Discovered local SEC Form 8-K JSON disclosure fixture."""

    name: str
    path: str
    ticker: Optional[str] = None
    company_name: Optional[str] = None
    filing_date: Optional[str] = None
    items: List[str] = Field(default_factory=list)
    headline: Optional[str] = None


class EdgeManifestEntry(BaseModel):
    """Discovered local JSON/JSONL supply-chain edge manifest."""

    name: str
    path: str
    edge_count: int
    tickers: List[str] = Field(default_factory=list)


class CohortExportEntry(BaseModel):
    """Discovered ADR-00013 1NF JSONL cohort export."""

    name: str
    path: str
    record_count: int


class OllamaStatus(BaseModel):
    """Local Ollama daemon availability telemetry."""

    healthy: bool
    detail: str


class CatalogResponse(BaseModel):
    """Catalog of read-only local snapshots, fixtures, edge manifests, cohorts, and LLM health."""

    snapshots: List[SnapshotEntry] = Field(default_factory=list)
    filings: List[FilingEntry] = Field(default_factory=list)
    edge_manifests: List[EdgeManifestEntry] = Field(default_factory=list)
    cohorts: List[CohortExportEntry] = Field(default_factory=list)
    ollama: OllamaStatus


class SubgraphNode(BaseModel):
    """Node entry returned by bidirectional PiT ego-subgraph query."""

    id: str
    ticker: str
    hop: int
    role: str


class SubgraphResponse(BaseModel):
    """Response payload for `GET /api/subgraph`."""

    source_kind: str
    source_path: str
    root_ticker: Optional[str] = None
    direction: str
    max_hops: int
    as_of_date: Optional[str] = None
    nodes: List[SubgraphNode] = Field(default_factory=list)
    edges: List[Dict[str, Any]] = Field(default_factory=list)
    available_tickers: List[str] = Field(default_factory=list)


class InlineForm8KPayload(BaseModel):
    """Ephemeral inline SEC Form 8-K payload for `POST /api/evaluate-8k`."""

    cik: str = "0000000000"
    ticker: str
    company_name: str = "Unknown Corporate Entity"
    form_type: str = "8-K"
    filing_date: str = "2024-10-28"
    filing_timestamp: Optional[str] = None
    items: List[str] = Field(default_factory=lambda: ["Item 8.01 Other Events"])
    headline: str = ""
    content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Evaluate8KRequest(BaseModel):
    """Request body for `POST /api/evaluate-8k`."""

    filing_path: Optional[str] = None
    inline_filing: Optional[InlineForm8KPayload] = None
    snapshot_id: Optional[str] = None
    snapshot_dir: Optional[str] = None
    edges_file: Optional[str] = None
    target_ticker: Optional[str] = None
    use_mock: bool = True

    @model_validator(mode="after")
    def _validate_filing_source(self) -> "Evaluate8KRequest":
        has_path = bool(self.filing_path and self.filing_path.strip())
        has_inline = self.inline_filing is not None
        if has_path == has_inline:
            raise ValueError("Provide exactly one of `filing_path` or `inline_filing`.")
        return self


class Evaluate8KResponse(BaseModel):
    """Full audit output for observed Form 8-K evaluation."""

    assessment: SanitizedDisruptionAssessment
    report: OperationalContagionReport
    citation_violations: List[str] = Field(default_factory=list)
    timeseries_rows: List[ContagionTimeSeriesEventRecord] = Field(default_factory=list)


class CohortFilterRequest(BaseModel):
    """Request body for `POST /api/cohort`."""

    cohort_path: str
    verdict: Optional[str] = None
    disruption_category: Optional[str] = None
    sec_item: Optional[str] = None
    ticker: Optional[str] = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class CohortFilterResponse(BaseModel):
    """Filtered `ContagionTimeSeriesEventRecord` rows and aggregate counts."""

    cohort_path: str
    total_records: int
    filtered_records: int
    verdict_counts: Dict[str, int] = Field(default_factory=dict)
    category_counts: Dict[str, int] = Field(default_factory=dict)
    sec_item_counts: Dict[str, int] = Field(default_factory=dict)
    records: List[ContagionTimeSeriesEventRecord] = Field(default_factory=list)


# ==============================================================================
# Path Confinement & Workspace Discovery Helpers
# ==============================================================================


@dataclass(frozen=True)
class _ResolvedGraphSource:
    source_kind: str
    path: Path
    display_path: str


def _dedupe_resolved_paths(paths: Sequence[Path]) -> Tuple[Path, ...]:
    seen: set[Path] = set()
    ordered: List[Path] = []
    for item in paths:
        resolved = Path(item).resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    return tuple(ordered)


def _to_repo_display_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(project_root):
        rel = resolved.relative_to(project_root).as_posix()
        return rel or "."
    return resolved.name


def _resolve_allowlisted_path(
    raw_path: str,
    allowed_roots: Sequence[Path],
    base_dir: Path,
    must_exist: bool = True,
    expect_dir: Optional[bool] = None,
    required_parent: Optional[Path] = None,
) -> Path:
    """Resolves `raw_path` and enforces strict containment within allowlisted roots."""
    cleaned = (raw_path or "").strip()
    if not cleaned or "\x00" in cleaned:
        raise HTTPException(status_code=400, detail="Invalid empty or malformed filesystem path.")

    candidate = Path(cleaned)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    resolved = candidate.resolve()

    if not any(resolved.is_relative_to(root) for root in allowed_roots):
        raise HTTPException(
            status_code=403,
            detail="Requested path resolves outside allowlisted workspace directories.",
        )

    if required_parent is not None and not resolved.is_relative_to(required_parent.resolve()):
        raise HTTPException(
            status_code=403,
            detail="Requested resource identifier escapes its designated directory.",
        )

    if must_exist:
        if expect_dir is True and not resolved.is_dir():
            raise HTTPException(status_code=404, detail=f"Directory not found: {cleaned}")
        if expect_dir is False and not resolved.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {cleaned}")
        if expect_dir is None and not resolved.exists():
            raise HTTPException(status_code=404, detail=f"Path not found: {cleaned}")

    return resolved


def _resolve_graph_source(
    snapshot_id: Optional[str],
    snapshot_dir: Optional[str],
    edges_file: Optional[str],
    *,
    project_root: Path,
    snapshots_root: Path,
    allowed_roots: Sequence[Path],
    allow_default: bool,
) -> Optional[_ResolvedGraphSource]:
    """Resolves `snapshot_id`, `snapshot_dir`, or `edges_file` strictly under allowlisted roots."""
    if edges_file and edges_file.strip():
        resolved = _resolve_allowlisted_path(
            edges_file,
            allowed_roots=allowed_roots,
            base_dir=project_root,
            must_exist=True,
            expect_dir=False,
        )
        return _ResolvedGraphSource("edges_file", resolved, _to_repo_display_path(resolved, project_root))

    if snapshot_dir and snapshot_dir.strip():
        resolved = _resolve_allowlisted_path(
            snapshot_dir,
            allowed_roots=allowed_roots,
            base_dir=project_root,
            must_exist=True,
            expect_dir=True,
        )
        if not (resolved / "edges.parquet").is_file():
            raise HTTPException(status_code=404, detail=f"Snapshot missing edges.parquet: {snapshot_dir}")
        return _ResolvedGraphSource("snapshot_dir", resolved, _to_repo_display_path(resolved, project_root))

    if snapshot_id and snapshot_id.strip():
        if not any(snapshots_root.resolve().is_relative_to(root) for root in allowed_roots):
            raise HTTPException(status_code=403, detail="Snapshots directory is outside allowlisted roots.")
        resolved = _resolve_allowlisted_path(
            snapshot_id,
            allowed_roots=allowed_roots,
            base_dir=snapshots_root,
            must_exist=True,
            expect_dir=True,
            required_parent=snapshots_root,
        )
        if not (resolved / "edges.parquet").is_file():
            raise HTTPException(status_code=404, detail=f"Snapshot missing edges.parquet: {snapshot_id}")
        return _ResolvedGraphSource("snapshot_id", resolved, _to_repo_display_path(resolved, project_root))

    if allow_default:
        default_dir = snapshots_root / "G_resolved"
        if (default_dir / "edges.parquet").is_file() and any(
            default_dir.resolve().is_relative_to(root) for root in allowed_roots
        ):
            resolved = default_dir.resolve()
            return _ResolvedGraphSource("snapshot_id", resolved, _to_repo_display_path(resolved, project_root))
        raise HTTPException(
            status_code=400,
            detail="No snapshot_id, snapshot_dir, or edges_file specified and default G_resolved snapshot is absent.",
        )

    return None


def _load_store(source: _ResolvedGraphSource) -> InMemorySubgraphStore:
    if source.source_kind == "edges_file":
        return InMemorySubgraphStore.from_edges_file(source.path)
    return InMemorySubgraphStore.from_parquet_snapshot(source.path)


def _is_within_roots(path: Path, allowed_roots: Sequence[Path]) -> bool:
    resolved = path.resolve()
    return any(resolved.is_relative_to(root) for root in allowed_roots)


def _discover_snapshots(
    snapshots_root: Path,
    project_root: Path,
    allowed_roots: Sequence[Path],
) -> List[SnapshotEntry]:
    if not snapshots_root.is_dir() or not _is_within_roots(snapshots_root, allowed_roots):
        return []
    entries: List[SnapshotEntry] = []
    for child in sorted(snapshots_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or not (child / "edges.parquet").is_file():
            continue
        if not _is_within_roots(child, allowed_roots):
            continue
        try:
            store = InMemorySubgraphStore.from_parquet_snapshot(child)
        except Exception:
            continue
        entries.append(
            SnapshotEntry(
                snapshot_id=child.name,
                path=_to_repo_display_path(child, project_root),
                has_nodes=(child / "nodes.parquet").is_file(),
                has_edges=True,
                edge_count=len(store._edges_by_key),
                tickers=store.list_tickers(),
            )
        )
    return entries


def _discover_filings(
    filings_root: Path,
    project_root: Path,
    allowed_roots: Sequence[Path],
) -> List[FilingEntry]:
    if not filings_root.is_dir() or not _is_within_roots(filings_root, allowed_roots):
        return []
    entries: List[FilingEntry] = []
    for json_file in sorted(filings_root.glob("*.json"), key=lambda p: p.name):
        if not _is_within_roots(json_file, allowed_roots):
            continue
        try:
            payload = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        items = [str(item) for item in payload.get("items", [])] if isinstance(payload.get("items"), list) else []
        entries.append(
            FilingEntry(
                name=json_file.name,
                path=_to_repo_display_path(json_file, project_root),
                ticker=str(payload["ticker"]).upper() if payload.get("ticker") else None,
                company_name=str(payload["company_name"]) if payload.get("company_name") else None,
                filing_date=str(payload["filing_date"]) if payload.get("filing_date") else None,
                items=items,
                headline=str(payload["headline"]) if payload.get("headline") else None,
            )
        )
    return entries


def _try_parse_edge_manifest(path: Path) -> Optional[InMemorySubgraphStore]:
    try:
        store = InMemorySubgraphStore.from_edges_file(path)
    except Exception:
        return None
    if not store._edges_by_key:
        return None
    return store


def _discover_edge_manifests(
    project_root: Path,
    snapshots_root: Path,
    allowed_roots: Sequence[Path],
) -> List[EdgeManifestEntry]:
    candidate_dirs = [
        snapshots_root,
        project_root / "data" / "manifests",
        project_root / "data" / "dry_run_staging",
        project_root / "tests" / "fixtures" / "graphs",
    ]
    seen: set[Path] = set()
    entries: List[EdgeManifestEntry] = []
    for directory in candidate_dirs:
        if not directory.is_dir() or not _is_within_roots(directory, allowed_roots):
            continue
        for ext in ("*.json", "*.jsonl"):
            for candidate in sorted(directory.glob(ext), key=lambda p: p.name):
                resolved = candidate.resolve()
                if resolved in seen or not _is_within_roots(resolved, allowed_roots):
                    continue
                seen.add(resolved)
                store = _try_parse_edge_manifest(resolved)
                if store is None:
                    continue
                entries.append(
                    EdgeManifestEntry(
                        name=candidate.name,
                        path=_to_repo_display_path(resolved, project_root),
                        edge_count=len(store._edges_by_key),
                        tickers=store.list_tickers(),
                    )
                )
    return entries


def _count_cohort_records(path: Path) -> Optional[int]:
    count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                ContagionTimeSeriesEventRecord.model_validate_json(line)
                count += 1
    except Exception:
        return None
    return count if count > 0 else None


def _discover_cohorts(
    project_root: Path,
    allowed_roots: Sequence[Path],
) -> List[CohortExportEntry]:
    candidate_dirs = [
        project_root / "data",
        project_root / "data" / "cohorts",
        project_root / "data" / "evaluation",
        project_root / "data" / "dry_run_staging",
        project_root / "tests" / "fixtures" / "cohorts",
    ]
    seen: set[Path] = set()
    entries: List[CohortExportEntry] = []
    for directory in candidate_dirs:
        if not directory.is_dir() or not _is_within_roots(directory, allowed_roots):
            continue
        for candidate in sorted(directory.glob("*.jsonl"), key=lambda p: p.name):
            resolved = candidate.resolve()
            if resolved in seen or not _is_within_roots(resolved, allowed_roots):
                continue
            seen.add(resolved)
            record_count = _count_cohort_records(resolved)
            if record_count is None:
                continue
            entries.append(
                CohortExportEntry(
                    name=candidate.name,
                    path=_to_repo_display_path(resolved, project_root),
                    record_count=record_count,
                )
            )
    return entries


def _parse_cohort_file(cohort_path: Path) -> List[ContagionTimeSeriesEventRecord]:
    records: List[ContagionTimeSeriesEventRecord] = []
    with cohort_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(ContagionTimeSeriesEventRecord.model_validate_json(line))
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid ContagionTimeSeriesEventRecord at line {line_no}: {exc}",
                ) from exc
    return records


def _record_matches_cohort_filters(
    record: ContagionTimeSeriesEventRecord,
    req: CohortFilterRequest,
) -> bool:
    if req.verdict and record.verdict.upper() != req.verdict.strip().upper():
        return False
    if req.disruption_category and record.disruption_category.upper() != req.disruption_category.strip().upper():
        return False
    if req.sec_item and req.sec_item.strip().lower() not in record.sec_item.lower():
        return False
    if req.ticker:
        target = req.ticker.strip().upper()
        if (
            record.disrupted_entity_ticker.upper() != target
            and record.target_entity_ticker.upper() != target
        ):
            return False
    return True


# ==============================================================================
# Application Factory & Route Registration
# ==============================================================================


def create_workbench_app(
    project_root: Optional[Union[str, Path]] = None,
    allowlisted_roots: Optional[Sequence[Union[str, Path]]] = None,
    snapshots_dir: Optional[Union[str, Path]] = None,
    filings_dir: Optional[Union[str, Path]] = None,
    static_dir: Optional[Union[str, Path]] = None,
) -> FastAPI:
    """Creates a configured read-only FastAPI backend for local YarnBall exploration."""
    resolved_root = Path(project_root).resolve() if project_root else DEFAULT_PROJECT_ROOT
    resolved_snapshots = (
        Path(snapshots_dir).resolve()
        if snapshots_dir
        else (resolved_root / "data" / "snapshots").resolve()
    )
    resolved_filings = (
        Path(filings_dir).resolve()
        if filings_dir
        else (resolved_root / "tests" / "fixtures" / "filings").resolve()
    )
    resolved_static = Path(static_dir).resolve() if static_dir else DEFAULT_STATIC_DIR

    if allowlisted_roots is not None:
        allowed_roots = _dedupe_resolved_paths([Path(p) for p in allowlisted_roots])
    else:
        allowed_roots = _dedupe_resolved_paths([resolved_root, resolved_snapshots, resolved_filings])

    fastapi_app = FastAPI(
        title="YarnBall Local Workbench API",
        version="0.1.0",
        description=(
            "Read-only local backend exposing Point-in-Time KG queries, "
            "SEC Form 8-K disruption gating & cascade evaluation, and 1NF cohort filtering."
        ),
    )

    if resolved_static.is_dir():
        fastapi_app.mount("/static", StaticFiles(directory=str(resolved_static)), name="static")

    @fastapi_app.get("/", response_model=None)
    def workbench_readiness():
        index_html = resolved_static / "index.html"
        if index_html.is_file():
            return FileResponse(index_html, media_type="text/html")

        html = (
            "<!doctype html>"
            "<html lang='en'>"
            "<head><meta charset='utf-8'><title>YarnBall Local Workbench API</title></head>"
            "<body>"
            "<h1>YarnBall Local Workbench API</h1>"
            "<p>Status: <strong>ready</strong> (static workbench shell not present).</p>"
            "<ul>"
            "<li><a href='/docs'>OpenAPI Interactive Docs (/docs)</a></li>"
            "<li><a href='/api/catalog'>Workspace Catalog (/api/catalog)</a></li>"
            "</ul>"
            "</body>"
            "</html>"
        )
        return HTMLResponse(content=html, status_code=200)

    @fastapi_app.get("/api/catalog", response_model=CatalogResponse)
    def get_catalog() -> CatalogResponse:
        healthy, detail = YarnBallModelClient().check_health(timeout_sec=1.0)
        return CatalogResponse(
            snapshots=_discover_snapshots(resolved_snapshots, resolved_root, allowed_roots),
            filings=_discover_filings(resolved_filings, resolved_root, allowed_roots),
            edge_manifests=_discover_edge_manifests(resolved_root, resolved_snapshots, allowed_roots),
            cohorts=_discover_cohorts(resolved_root, allowed_roots),
            ollama=OllamaStatus(healthy=healthy, detail=detail),
        )

    @fastapi_app.get("/api/subgraph", response_model=SubgraphResponse)
    def get_subgraph(
        snapshot_id: Optional[str] = Query(default=None),
        snapshot_dir: Optional[str] = Query(default=None),
        edges_file: Optional[str] = Query(default=None),
        ticker: Optional[str] = Query(default=None),
        max_hops: int = Query(default=2, ge=1, le=5),
        direction: Literal["both", "downstream", "upstream"] = Query(default="both"),
        as_of_date: Optional[str] = Query(default=None),
        rel_types: Optional[str] = Query(default=None),
        required_status: Optional[str] = Query(default="ACTIVE_CURRENT"),
    ) -> SubgraphResponse:
        source = _resolve_graph_source(
            snapshot_id=snapshot_id,
            snapshot_dir=snapshot_dir,
            edges_file=edges_file,
            project_root=resolved_root,
            snapshots_root=resolved_snapshots,
            allowed_roots=allowed_roots,
            allow_default=True,
        )
        assert source is not None
        store = _load_store(source)
        parsed_rels = (
            [item.strip() for item in rel_types.split(",") if item.strip()]
            if rel_types
            else None
        )
        ego = store.query_ego_subgraph(
            ticker=ticker,
            max_hops=max_hops,
            direction=direction,
            as_of_date=as_of_date,
            rel_types=parsed_rels,
            required_status=required_status or None,
        )
        return SubgraphResponse(
            source_kind=source.source_kind,
            source_path=source.display_path,
            root_ticker=ego["root_ticker"],
            direction=direction,
            max_hops=max_hops,
            as_of_date=as_of_date,
            nodes=[SubgraphNode(**node) for node in ego["nodes"]],
            edges=ego["edges"],
            available_tickers=ego["available_tickers"],
        )

    @fastapi_app.post("/api/evaluate-8k", response_model=Evaluate8KResponse)
    def evaluate_8k(request: Evaluate8KRequest) -> Evaluate8KResponse:
        graph_source = _resolve_graph_source(
            snapshot_id=request.snapshot_id,
            snapshot_dir=request.snapshot_dir,
            edges_file=request.edges_file,
            project_root=resolved_root,
            snapshots_root=resolved_snapshots,
            allowed_roots=allowed_roots,
            allow_default=False,
        )
        snap_arg = graph_source.path if graph_source and graph_source.source_kind != "edges_file" else None
        edges_arg = graph_source.path if graph_source and graph_source.source_kind == "edges_file" else None

        if request.filing_path is not None:
            filing_file = _resolve_allowlisted_path(
                request.filing_path,
                allowed_roots=allowed_roots,
                base_dir=resolved_root,
                must_exist=True,
                expect_dir=False,
            )
            try:
                filing_data = json.loads(filing_file.read_text(encoding="utf-8"))
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Malformed SEC Form 8-K JSON: {exc}") from exc
            if not isinstance(filing_data, dict):
                raise HTTPException(status_code=400, detail="SEC Form 8-K payload must be a JSON object.")

            assessment = assess_sec_8k_disruption(
                items=filing_data.get("items", ["Item 8.01 Other Events"]),
                headline=str(filing_data.get("headline", "")),
                content=str(filing_data.get("content", "")),
                metadata=filing_data.get("metadata", {}),
            )
            report, _ = run_sec_8k_dry_run(
                filing_path=filing_file,
                target_ticker=request.target_ticker,
                offline_isolation=True,
                use_mock=request.use_mock,
                snapshot_dir=snap_arg,
                edges_path=edges_arg,
            )
        else:
            assert request.inline_filing is not None
            payload_dict = request.inline_filing.model_dump()
            assessment = assess_sec_8k_disruption(
                items=payload_dict["items"],
                headline=payload_dict["headline"],
                content=payload_dict["content"],
                metadata=payload_dict["metadata"],
            )
            with tempfile.TemporaryDirectory(prefix="yarnball_8k_") as tmpdir:
                temp_filing = Path(tmpdir) / "inline_8k.json"
                temp_filing.write_text(json.dumps(payload_dict), encoding="utf-8")
                report, _ = run_sec_8k_dry_run(
                    filing_path=temp_filing,
                    target_ticker=request.target_ticker,
                    offline_isolation=True,
                    use_mock=request.use_mock,
                    snapshot_dir=snap_arg,
                    edges_path=edges_arg,
                )

        return Evaluate8KResponse(
            assessment=assessment,
            report=report,
            citation_violations=validate_verbatim_citations(report),
            timeseries_rows=flatten_report_to_timeseries_records(report),
        )

    @fastapi_app.post("/api/cohort", response_model=CohortFilterResponse)
    def filter_cohort(request: CohortFilterRequest) -> CohortFilterResponse:
        cohort_file = _resolve_allowlisted_path(
            request.cohort_path,
            allowed_roots=allowed_roots,
            base_dir=resolved_root,
            must_exist=True,
            expect_dir=False,
        )
        all_records = _parse_cohort_file(cohort_file)
        filtered = [rec for rec in all_records if _record_matches_cohort_filters(rec, request)]
        verdict_counts = dict(sorted(Counter(rec.verdict for rec in filtered).items()))
        category_counts = dict(sorted(Counter(rec.disruption_category for rec in filtered).items()))
        sec_item_counts = dict(sorted(Counter(rec.sec_item for rec in filtered).items()))
        page = filtered[request.offset : request.offset + request.limit]
        return CohortFilterResponse(
            cohort_path=_to_repo_display_path(cohort_file, resolved_root),
            total_records=len(all_records),
            filtered_records=len(filtered),
            verdict_counts=verdict_counts,
            category_counts=category_counts,
            sec_item_counts=sec_item_counts,
            records=page,
        )

    return fastapi_app


app = create_workbench_app()

