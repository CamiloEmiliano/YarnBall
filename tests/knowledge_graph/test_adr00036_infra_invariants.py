"""ADR-00036 Fail-Fast Storage, Ephemeral Memgraph Lifecycle & Deployable Topology Invariant Tests.

Covers all 6 Invariants in ADR-00036 Section 5 plus `/health` & `/ready` probes:
1. Invariant #1 (No implicit dummies): unreachable Memgraph / PostgreSQL raise
   `GraphStoreUnavailableError` / `DocumentStoreUnavailableError` unless explicit
   `YARNBALL_GRAPH_MOCK=1` / `YARNBALL_PG_MOCK=1` opt-in (or `@pytest.mark.offline`).
2. Invariant #2 (No empty-on-error reads AST check + runtime): no `except Exception` in
   `retrieval/` silently returns an empty collection without re-raising or marking degraded;
   `query_bi` turns store errors into explicit degraded outcomes and never into an empty report.
3. Invariant #3 (Durable-before-ephemeral write order): both `store_graph_entities` and
   `EdgarWorker._persist_sec_graph_data` commit to PostgreSQL `kg_nodes`/`kg_edges` BEFORE
   projecting into Memgraph; wiping Memgraph + running `hydrate_memgraph_from_ledger`
   reproduces 100% of structural topology deterministically.
4. Invariant #4 (Thin graph & bounded `:News` event RAM TTL): raw prose never enters
   Memgraph; `MEMGRAPH_EVENT_TTL_DAYS` (default 90) evicts expired `:News` nodes from RAM
   without touching PostgreSQL.
5. Invariant #5 (Packaging & container import smoke): `pyproject.toml` extras, selective
   `retrieval/Dockerfile` COPY (no `COPY . .`, no PyTorch/`sentence-transformers`), and
   import smoke verification (`retrieval.service`, `retrieval.pipeline`, `synthesis.synthesizer`).
6. Invariant #6 (Compose references exist): `docker-compose.yml` has no `scraper` service,
   no persistent volume on `memgraph`, includes `memgraph_hydrator`, `stream_consumer`, and
   `memgraph_consumer`, and every `python -m <module>` resolves to an importable module.
"""

from __future__ import annotations

import ast
from datetime import date, datetime, timezone
import importlib
import importlib.util
import json
from pathlib import Path
import tomllib
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pytest
import yaml

import knowledge_graph.db as kg_db
import knowledge_graph.memgraph_driver as kg_mg
from knowledge_graph.errors import (
    DocumentStoreUnavailableError,
    GraphStoreUnavailableError,
    StoreUnavailableError,
)
from knowledge_graph.graph_store import store_graph_entities
from knowledge_graph.ledger import (
    DEFAULT_MEMGRAPH_EVENT_TTL_DAYS,
    evict_expired_news_nodes,
    get_memgraph_event_ttl_days,
    hydrate_memgraph_from_ledger,
)
from ingestion.sec.harvester import EdgarWorker
from retrieval.facade import MemgraphSubgraphStore, query_bi
from retrieval.hybrid import HybridSeeder
from retrieval.service import app as retrieval_app
from retrieval.temporal import PiTContext
from sft_pipeline.serving.model_client import ModelResponse, YarnBallModelClient

REPO_ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------
# Invariant #1: Fail-Fast Stores & Explicit Mock Opt-In
# ----------------------------------------------------------------------
def test_adr00036_invariant1_fail_fast_stores_and_explicit_mock_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unreachable Memgraph and PostgreSQL must raise unless explicit mock env flags are set."""
    kg_mg.reset_memgraph_driver_cache()
    monkeypatch.delenv("YARNBALL_GRAPH_MOCK", raising=False)
    monkeypatch.delenv("YARNBALL_PG_MOCK", raising=False)

    # 1. Memgraph unreachable without YARNBALL_GRAPH_MOCK -> raises GraphStoreUnavailableError
    with patch.object(kg_mg, "GraphDatabase") as mock_gdb:
        mock_gdb.driver.side_effect = ConnectionRefusedError("bolt refused")
        with pytest.raises(GraphStoreUnavailableError):
            kg_mg.get_memgraph_driver(uri="bolt://127.0.0.1:19999")

        # With explicit YARNBALL_GRAPH_MOCK=1 -> returns _DummyDriver
        monkeypatch.setenv("YARNBALL_GRAPH_MOCK", "1")
        drv = kg_mg.get_memgraph_driver(uri="bolt://127.0.0.1:19999")
        assert isinstance(drv, kg_mg._DummyDriver)

    # 2. PostgreSQL unreachable without YARNBALL_PG_MOCK -> raises DocumentStoreUnavailableError
    with patch("tools.init_db._connect_db", side_effect=ConnectionRefusedError("pg refused")), \
         patch.object(kg_db, "psycopg") as mock_psycopg:
        mock_psycopg.connect.side_effect = ConnectionRefusedError("pg refused")
        with pytest.raises(DocumentStoreUnavailableError):
            kg_db.get_connection()
        with pytest.raises(DocumentStoreUnavailableError):
            kg_db.pg_connection()

        # With explicit YARNBALL_PG_MOCK=1 -> returns _MockConnection
        monkeypatch.setenv("YARNBALL_PG_MOCK", "1")
        conn = kg_db.get_connection()
        assert isinstance(conn, kg_db._MockConnection)


@pytest.mark.offline
def test_adr00036_offline_marker_enables_explicit_store_mocks() -> None:
    """`tests/conftest.py` opts `@pytest.mark.offline` tests into explicit store mock env flags."""
    assert kg_mg.is_graph_mock_enabled() is True
    assert kg_db.is_pg_mock_enabled() is True


# ----------------------------------------------------------------------
# Invariant #2: No Empty-on-Error Reads (AST Check + Runtime Degraded Facade)
# ----------------------------------------------------------------------
def _is_empty_collection_return(node: ast.Return) -> bool:
    val = node.value
    if val is None:
        return True
    if isinstance(val, ast.Constant) and val.value is None:
        return True
    if isinstance(val, (ast.List, ast.Tuple, ast.Set)) and len(val.elts) == 0:
        return True
    if isinstance(val, ast.Dict) and len(val.keys) == 0:
        return True
    return False


def _handler_marks_degraded_or_raises(handler: ast.ExceptHandler) -> bool:
    for sub in ast.walk(handler):
        if isinstance(sub, ast.Raise):
            return True
        if isinstance(sub, ast.Name) and "degraded" in sub.id:
            return True
        if isinstance(sub, ast.keyword) and sub.arg and "degraded" in sub.arg:
            return True
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and "degraded" in sub.func.id:
            return True
    return False


def test_adr00036_invariant2_no_empty_on_error_reads_ast() -> None:
    """AST Invariant #2: No `except Exception` in `retrieval/` returns an empty collection or swallows silently."""
    offenders: List[str] = []
    retrieval_dir = REPO_ROOT / "retrieval"

    for py_path in sorted(retrieval_dir.rglob("*.py")):
        rel = str(py_path.relative_to(REPO_ROOT))
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            is_broad = node.type is None or (
                isinstance(node.type, ast.Name) and node.type.id in ("Exception", "BaseException")
            )
            if not is_broad:
                continue

            if _handler_marks_degraded_or_raises(node):
                continue

            # Check for silent empty return or bare pass/continue/log-only swallow
            has_empty_ret = any(
                isinstance(sub, ast.Return) and _is_empty_collection_return(sub)
                for sub in ast.walk(node)
            )
            has_non_empty_return = any(
                isinstance(sub, ast.Return) and not _is_empty_collection_return(sub)
                for sub in ast.walk(node)
            )
            if has_empty_ret or not has_non_empty_return:
                # Permit explicit retry-loop error recording only if handler sets last_error
                sets_last_error = any(
                    isinstance(sub, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "last_error" for t in sub.targets)
                    for sub in ast.walk(node)
                )
                if not sets_last_error:
                    offenders.append(f"{rel}:{node.lineno} broad except Exception without re-raise or degraded flag")

    assert offenders == [], f"ADR-00036 Invariant #2 AST violations: {offenders}"


def test_adr00036_facade_turns_store_outages_into_degraded_outcome_never_empty_report() -> None:
    """`query_bi` turns store errors into explicit degraded outcomes and refuses empty operational reports."""
    ctx = PiTContext(as_of=date(2025, 5, 1), mode="historical")

    class _MockCypherClient(YarnBallModelClient):
        def generate(
            self,
            prompt: str,
            task: str = "TEXT_TO_CYPHER",
            system_prompt: Optional[str] = None,
            temperature: float = 0.0,
            max_new_tokens: Optional[int] = None,
            enforce_think_prefix: bool = True,
            num_ctx: int = 8192,
        ) -> ModelResponse:
            out = "MATCH (s:Company {ticker: 'TSM'})-[r:SUPPLIES_TO]->(t:Company) RETURN s, r, t LIMIT 25"
            return ModelResponse(
                raw_text=out,
                think_trace=None,
                final_output=out,
                task=task,
                latency_sec=0.01,
                prefill_ms=1.0,
                tokens_per_sec=100.0,
                prompt_tokens=10,
                generated_tokens=10,
            )

    # 1. Memgraph outage during query_bi -> explicit degraded PipelineResult
    with patch(
        "knowledge_graph.memgraph_driver.get_memgraph_driver",
        side_effect=GraphStoreUnavailableError("Memgraph down"),
    ):
        degraded_res = query_bi(
            "What is Apple's exposure to TSMC?",
            ctx=ctx,
            store=MemgraphSubgraphStore(),
            model_client=_MockCypherClient(),
        )
    assert degraded_res.degraded is True
    assert degraded_res.is_valid_query is False
    assert any("Memgraph down" in r for r in degraded_res.degraded_reasons)
    assert "DEGRADED" in degraded_res.contagion_verdict

    # Degraded result MUST refuse conversion into an empty OperationalContagionReport
    with pytest.raises(StoreUnavailableError, match="degraded retrieval result"):
        degraded_res.to_operational_report(
            event_id="evt-1",
            sec_item="Item 8.01",
            disrupted_entity_ticker="TSM",
            filing_date="2025-05-01",
            disruption_category="FACILITY_HALT",
            event_summary="Fab outage",
            shock_duration_days=30.0,
            shock_magnitude_pct=100.0,
        )

    # 2. Postgres outage during HybridSeeder -> explicit degraded PipelineResult
    failing_seeder = HybridSeeder()
    with patch("embedding.embedder.embed_query", return_value=[0.5, 0.5]), \
         patch("retrieval.hybrid.pg_connection", side_effect=DocumentStoreUnavailableError("Postgres down")):
        pg_degraded = query_bi(
            "What is Apple's exposure to TSMC?",
            ctx=ctx,
            store=MemgraphSubgraphStore(),
            seeder=failing_seeder,
            model_client=_MockCypherClient(),
        )
    assert pg_degraded.degraded is True
    assert any("Postgres down" in r for r in pg_degraded.degraded_reasons)

    # 3. FastAPI `/search` and `/report` return HTTP 503 on store outage (never 200 empty exposure)
    client = TestClient(retrieval_app)
    with patch("embedding.embedder.embed_query", return_value=[0.5, 0.5]), \
         patch("retrieval.hybrid.pg_connection", side_effect=DocumentStoreUnavailableError("Postgres down")):
        resp_search = client.post("/search", json={"query": "TSMC risk", "as_of": "2025-05-01"})
        assert resp_search.status_code == 503
        assert resp_search.json()["detail"]["degraded"] is True

        resp_report = client.post(
            "/report",
            json={
                "query": "TSMC risk",
                "as_of": "2025-05-01",
                "event_id": "evt-1",
                "disrupted_entity_ticker": "TSM",
                "event_summary": "Fab outage",
            },
        )
        assert resp_report.status_code == 503


# ----------------------------------------------------------------------
# Invariant #3: Durable-Before-Ephemeral Write Order & Deterministic Re-Hydration
# ----------------------------------------------------------------------
class _InMemoryLedgerPGConn:
    """In-memory relational ledger stub recording exact SQL writes and timeline order."""

    def __init__(self, timeline: List[str]) -> None:
        self.timeline = timeline
        self.nodes: Dict[str, Tuple[Any, ...]] = {}
        self.edges: Dict[Tuple[str, str, str, str], Tuple[Any, ...]] = {}

    def __enter__(self) -> "_InMemoryLedgerPGConn":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def cursor(self) -> "_InMemoryLedgerCursor":
        return _InMemoryLedgerCursor(self)

    def commit(self) -> None:
        self.timeline.append("pg:commit")


class _InMemoryLedgerCursor:
    def __init__(self, conn: _InMemoryLedgerPGConn) -> None:
        self.conn = conn
        self._last_rows: List[Tuple[Any, ...]] = []

    def __enter__(self) -> "_InMemoryLedgerCursor":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        sql_norm = " ".join(sql.split())
        if "INSERT INTO kg_nodes" in sql_norm:
            self.conn.timeline.append(f"pg:kg_nodes:{params[0]}")
            self.conn.nodes[params[0]] = tuple(params)
        elif "INSERT INTO kg_edges" in sql_norm:
            key = (params[0], params[1], params[2], params[3])
            self.conn.timeline.append(f"pg:kg_edges:{params[0]}->{params[1]}:{params[2]}")
            self.conn.edges[key] = tuple(params)
        elif "FROM kg_nodes" in sql_norm:
            self._last_rows = [self.conn.nodes[k] for k in sorted(self.conn.nodes.keys())]
        elif "FROM kg_edges" in sql_norm:
            self._last_rows = [self.conn.edges[k] for k in sorted(self.conn.edges.keys())]
        else:
            self._last_rows = []

    def fetchall(self) -> List[Tuple[Any, ...]]:
        return list(self._last_rows)

    def fetchone(self) -> Optional[Tuple[Any, ...]]:
        return self._last_rows[0] if self._last_rows else None

    def close(self) -> None:
        return None


class _EphemeralMemgraphStub:
    """In-memory ephemeral Memgraph stub that supports full wipe + deterministic re-hydration."""

    def __init__(self, timeline: List[str]) -> None:
        self.timeline = timeline
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.news: Dict[str, str] = {}

    def wipe(self) -> None:
        self.nodes.clear()
        self.edges.clear()
        self.news.clear()

    def verify_connectivity(self) -> None:
        return None

    def session(self) -> "_EphemeralMemgraphSession":
        return _EphemeralMemgraphSession(self)


class _EphemeralMemgraphSession:
    def __init__(self, driver: _EphemeralMemgraphStub) -> None:
        self.driver = driver

    def __enter__(self) -> "_EphemeralMemgraphSession":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def run(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        q = " ".join(query.split())
        self.driver.timeline.append(f"mg:{q[:40]}")
        if "DETACH DELETE n" in q and "MATCH (n:News)" not in q:
            self.driver.wipe()
            return []
        if "MATCH (n:News)" in q and "DETACH DELETE n" in q:
            cutoff = str(kwargs["cutoff_iso"])
            stale = [nid for nid, pub in self.driver.news.items() if pub < cutoff]
            for nid in stale:
                del self.driver.news[nid]
            return [{"evicted": len(stale)}]
        if "MERGE (n:News" in q:
            self.driver.news[str(kwargs["source_hash"])] = str(kwargs.get("published_at") or "")
            return []
        if "MERGE (p:Company" in q or "MERGE (n:" in q:
            nid = str(kwargs["id"])
            props = dict(kwargs.get("props") or {})
            if "ticker" in kwargs:
                props["ticker"] = kwargs["ticker"]
            self.driver.nodes[nid] = {**self.driver.nodes.get(nid, {}), **props}
            return []
        if "MERGE (s)-[r:" in q:
            rel_type = q.split("MERGE (s)-[r:", 1)[1].split("]", 1)[0].split(" ", 1)[0]
            src = str(kwargs["source"])
            tgt = str(kwargs["target"])
            props = dict(kwargs.get("props") or {})
            for extra in ("valid_from", "evidence_chunk_ids", "evidence_status", "source_type"):
                if extra in kwargs and kwargs[extra] is not None:
                    props[extra] = kwargs[extra]
            self.driver.edges[(src, tgt, rel_type)] = props
            return []
        return []


def test_adr00036_invariant3_durable_before_ephemeral_and_deterministic_rehydration() -> None:
    """Both writers persist to Postgres kg_nodes/kg_edges BEFORE Memgraph; wipe + re-hydrate matches 100%."""
    timeline: List[str] = []
    pg_conn = _InMemoryLedgerPGConn(timeline)
    mg_driver = _EphemeralMemgraphStub(timeline)

    # 1. Writer A: store_graph_entities (news)
    with patch("knowledge_graph.embedding_store.store_node_embeddings"):
        store_graph_entities(
            source_hash="news_h1",
            text="TSMC supplies advanced wafers to Apple Inc.",
            graph_data={
                "nodes": [
                    {"id": "TSMC", "type": "Company", "properties": {"ticker": "TSM"}},
                    {"id": "Apple Inc.", "type": "Company", "properties": {"ticker": "AAPL"}},
                ],
                "edges": [
                    {
                        "source": "TSMC",
                        "target": "Apple Inc.",
                        "type": "SUPPLIES_TO",
                        "properties": {"share": 0.45, "buffer_days": 30.0},
                    }
                ],
            },
            driver=mg_driver,
            published_at="2024-05-01T18:00:00Z",
            pg_conn=pg_conn,
        )

    first_mg_idx = next(i for i, ev in enumerate(timeline) if ev.startswith("mg:"))
    last_pg_ledger_idx = max(i for i, ev in enumerate(timeline) if ev.startswith("pg:kg_"))
    assert last_pg_ledger_idx < first_mg_idx, f"Durable Postgres write did not precede Memgraph: {timeline}"

    # 2. Writer B: EdgarWorker._persist_sec_graph_data (SEC)
    timeline.clear()
    worker = EdgarWorker(pg_conn=pg_conn, memgraph_driver=mg_driver)
    worker.linker.link_entity = MagicMock(return_value=None)
    with patch("knowledge_graph.embedding_store.store_node_embeddings"):
        worker._persist_sec_graph_data(
            primary_name="Apple Inc.",
            primary_ticker="AAPL",
            graph_data={
                "nodes": [
                    {"id": "Foxconn", "type": "Company", "properties": {}},
                    {"id": "Apple Inc.", "type": "Company", "properties": {"ticker": "AAPL"}},
                ],
                "edges": [
                    {
                        "source": "Apple Inc.",
                        "target": "Foxconn",
                        "type": "SOLE_SOURCE_DEPENDENT_ON",
                        "properties": {"share": 0.60, "buffer_days": 14.0},
                    }
                ],
            },
            accession_number="0000320193-24-000106",
            filing_date="2024-11-01",
            form_type="10-K",
            section="Item 1 Business",
            acceptance_datetime="2024-11-01T15:00:00Z",
        )

    sec_first_mg_idx = next(i for i, ev in enumerate(timeline) if ev.startswith("mg:"))
    sec_last_pg_idx = max(i for i, ev in enumerate(timeline) if ev.startswith("pg:kg_"))
    assert sec_last_pg_idx < sec_first_mg_idx, f"SEC durable write did not precede Memgraph: {timeline}"

    before_wipe_edges = set(mg_driver.edges.keys())
    assert ("TSMC", "Apple Inc.", "SUPPLIES_TO") in before_wipe_edges
    assert ("Apple Inc.", "Foxconn", "SOLE_SOURCE_DEPENDENT_ON") in before_wipe_edges

    # 3. Simulate Memgraph container restart (total RAM wipe) + deterministic re-hydration
    mg_driver.wipe()
    assert mg_driver.edges == {}
    summary = hydrate_memgraph_from_ledger(pg_conn=pg_conn, driver=mg_driver, clear_first=True)
    assert summary["nodes_hydrated"] == 3
    assert summary["edges_hydrated"] == 2
    assert set(mg_driver.edges.keys()) == before_wipe_edges
    assert mg_driver.edges[("TSMC", "Apple Inc.", "SUPPLIES_TO")]["share"] == 0.45
    assert mg_driver.edges[("Apple Inc.", "Foxconn", "SOLE_SOURCE_DEPENDENT_ON")]["buffer_days"] == 14.0
    assert len(mg_driver.edges[("Apple Inc.", "Foxconn", "SOLE_SOURCE_DEPENDENT_ON")]["evidence_chunk_ids"]) == 1


# ----------------------------------------------------------------------
# Invariant #4: Bounded :News RAM TTL Eviction (MEMGRAPH_EVENT_TTL_DAYS)
# ----------------------------------------------------------------------
def test_adr00036_invariant4_news_ttl_eviction_leaves_recent_and_postgres_intact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient :News nodes older than MEMGRAPH_EVENT_TTL_DAYS (default 90) are evicted from Memgraph RAM."""
    monkeypatch.delenv("MEMGRAPH_EVENT_TTL_DAYS", raising=False)
    assert get_memgraph_event_ttl_days() == DEFAULT_MEMGRAPH_EVENT_TTL_DAYS == 90

    monkeypatch.setenv("MEMGRAPH_EVENT_TTL_DAYS", "30")
    assert get_memgraph_event_ttl_days() == 30

    mg_driver = _EphemeralMemgraphStub([])
    mg_driver.news["old_news"] = "2024-01-01T12:00:00+00:00"
    mg_driver.news["fresh_news"] = "2024-05-15T12:00:00+00:00"

    now_utc = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    evicted = evict_expired_news_nodes(driver=mg_driver, now_utc=now_utc)
    assert evicted == 1
    assert "old_news" not in mg_driver.news
    assert "fresh_news" in mg_driver.news


# ----------------------------------------------------------------------
# Readiness Endpoint (`/health` vs `/ready`)
# ----------------------------------------------------------------------
def test_adr00036_health_liveness_and_ready_probe_checks_all_five_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/health` returns 200 liveness; `/ready` checks all 5 subsystems and returns 503 with reasons on failure."""
    client = TestClient(retrieval_app)
    assert client.get("/health").status_code == 200

    # Failure case: all 5 subsystems report failures -> 503 with 5 distinct reasons
    with patch(
        "retrieval.service._check_postgres_and_schema_metadata",
        return_value={
            "postgres": {"ready": False, "detail": "pg down"},
            "embedding_schema_metadata": {"ready": False, "detail": "dim mismatch"},
        },
    ), patch(
        "retrieval.service.check_memgraph_readiness",
        return_value={"ready": False, "connected": True, "hydrated": False, "detail": "memgraph_hydration_incomplete"},
    ), patch(
        "embedding.embedder.verify_embedder_ready",
        return_value={"ready": False, "detail": "embedder offline"},
    ), patch(
        "retrieval.service.check_ollama_models_ready",
        return_value={
            "ready": False,
            "required": ["yarnball-qwen:7b", "qwen3-embedding:0.6b"],
            "present": [],
            "missing": ["yarnball-qwen:7b", "qwen3-embedding:0.6b"],
            "detail": "missing_ollama_models: yarnball-qwen:7b, qwen3-embedding:0.6b",
        },
    ):
        resp_fail = client.get("/ready")
        assert resp_fail.status_code == 503
        body_fail = resp_fail.json()
        assert body_fail["ready"] is False
        assert len(body_fail["reasons"]) == 5

    # Healthy case: all 5 checks pass -> 200 ready=True
    with patch(
        "retrieval.service._check_postgres_and_schema_metadata",
        return_value={
            "postgres": {"ready": True, "detail": None},
            "embedding_schema_metadata": {"ready": True, "detail": None},
        },
    ), patch(
        "retrieval.service.check_memgraph_readiness",
        return_value={"ready": True, "connected": True, "hydrated": True, "detail": None},
    ), patch(
        "embedding.embedder.verify_embedder_ready",
        return_value={"ready": True, "detail": None},
    ), patch(
        "retrieval.service.check_ollama_models_ready",
        return_value={
            "ready": True,
            "required": ["yarnball-qwen:7b", "qwen3-embedding:0.6b"],
            "present": ["yarnball-qwen:7b", "qwen3-embedding:0.6b"],
            "missing": [],
            "detail": None,
        },
    ):
        resp_ok = client.get("/ready")
        assert resp_ok.status_code == 200
        assert resp_ok.json()["ready"] is True
        assert resp_ok.json()["reasons"] == []


# ----------------------------------------------------------------------
# Invariants #5 & #6: Packaging, Dockerfile Selective COPY & Compose Integrity
# ----------------------------------------------------------------------
def test_adr00036_invariants_5_and_6_packaging_dockerfile_and_compose_modules() -> None:
    """Verify pyproject.toml extras, retrieval/Dockerfile selective COPY + import smoke, and docker-compose modules."""
    # 1. pyproject.toml validation
    pyproject_path = REPO_ROOT / "pyproject.toml"
    assert pyproject_path.is_file()
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    for extra in ("retrieval", "ingestion", "dev"):
        assert extra in extras, f"Missing optional dependency extra [{extra}] in pyproject.toml"

    # 2. retrieval/Dockerfile selective COPY & lightweight deps (Invariant #5)
    dockerfile_text = (REPO_ROOT / "retrieval" / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY . ." not in dockerfile_text
    assert "COPY pyproject.toml" in dockerfile_text
    assert "pip install --no-cache-dir --no-deps ." in dockerfile_text
    assert "import retrieval.service, retrieval.pipeline, synthesis.synthesizer" in dockerfile_text
    for pkg in (
        "retrieval",
        "knowledge_graph",
        "embedding",
        "ontology",
        "contagion",
        "synthesis",
        "curation",
        "sft_pipeline",
        "universe",
        "tools",
    ):
        assert f"COPY {pkg} ./{pkg}" in dockerfile_text, f"Missing selective COPY for {pkg} in retrieval/Dockerfile"

    ret_reqs = (REPO_ROOT / "retrieval" / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "torch" not in ret_reqs
    assert "sentence-transformers" not in ret_reqs

    # In-process container entrypoint import smoke check
    for mod_name in ("retrieval.service", "retrieval.pipeline", "synthesis.synthesizer"):
        assert importlib.import_module(mod_name) is not None

    # 3. docker-compose.yml topology & module resolution (Invariant #6)
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services: Dict[str, Any] = compose["services"]

    # Dead scraper service must be deleted; memgraph must stay purely ephemeral (no volumes)
    assert "scraper" not in services
    assert "ingest.scraping_worker" not in json.dumps(compose)
    assert not services["memgraph"].get("volumes"), "Memgraph must remain purely in-memory with no persistent volume"

    # Required new deployable services must exist with health/completion dependencies
    for required_svc in ("memgraph_hydrator", "stream_consumer", "memgraph_consumer", "retriever"):
        assert required_svc in services, f"Missing service {required_svc} in docker-compose.yml"
        assert "depends_on" in services[required_svc]

    # Every `python -m <module>` or `python <script.py>` command must resolve in the repo
    checked_modules: List[str] = []
    for svc_name, svc_cfg in services.items():
        cmd = svc_cfg.get("command")
        if not cmd:
            continue
        tokens = cmd if isinstance(cmd, list) else str(cmd).split()
        if len(tokens) >= 3 and tokens[0] in ("python", "python3") and tokens[1] == "-m":
            mod = tokens[2]
            spec = importlib.util.find_spec(mod)
            assert spec is not None, f"Compose service '{svc_name}' references non-existent module '{mod}'"
            checked_modules.append(mod)
        elif len(tokens) >= 2 and tokens[0] in ("python", "python3") and tokens[1].endswith(".py"):
            script_path = REPO_ROOT / tokens[1]
            assert script_path.is_file(), f"Compose service '{svc_name}' references missing script '{tokens[1]}'"
            checked_modules.append(tokens[1])

    assert "tools.hydrate_memgraph" in checked_modules
    assert "streaming.consumer" in checked_modules
    assert "knowledge_graph.memgraph_consumer" in checked_modules
