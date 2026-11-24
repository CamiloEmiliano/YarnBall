"""ADR-00033 Point-in-Time Read Contract, U.S. Session Calendar & AST Invariant Tests."""

from __future__ import annotations

import ast
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, List, Set, Tuple
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError

from ingestion.sec.harvester import EdgarWorker
from knowledge_graph.embedding_store import backfill_node_embedding_validity, store_node_embeddings
from retrieval.enricher import enrich_nodes
from retrieval.facade import MemgraphSubgraphStore
from retrieval.hybrid import HybridSeeder
from retrieval.offline_store import InMemorySubgraphStore
from retrieval.pipeline import AnalystQuery, GraphRAGPipelineOrchestrator
from retrieval.service import app as retrieval_app
from retrieval.temporal import (
    PiTContext,
    TemporalQueryFilter,
    actionable_session,
    is_us_equity_trading_day,
    next_us_equity_trading_day,
    us_equity_regular_close_et,
    validate_hops,
)
from retrieval.text_to_cql import TextToCQL

REPO_ROOT = Path(__file__).resolve().parents[2]

# Reviewed allowlist for Cypher execution without edge-level TemporalQueryFilter (Invariant #2):
# Schema introspection reads only distinct label/rel-type names, never entity or edge facts.
ALLOWED_UNGUARDED_SESSION_RUN: Set[Tuple[str, str]] = {
    ("retrieval/facade.py", "introspect_schema"),
}

# Read-entrypoint parameters governed by Invariant #1 ("no Optional / defaulted ctx or as_of")
PIT_PARAM_NAMES: Set[str] = {"ctx", "as_of", "as_of_date"}


def _is_optional_annotation(ann: ast.expr | None) -> bool:
    if ann is None:
        return False
    if isinstance(ann, ast.Subscript) and isinstance(ann.value, ast.Name) and ann.value.id == "Optional":
        return True
    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
        sides = (ann.left, ann.right)
        return any(isinstance(s, ast.Constant) and s.value is None for s in sides)
    return False


def _is_required_fastapi_query(default_node: ast.expr) -> bool:
    """Return True for `Query(...)` (required FastAPI parameter with Ellipsis default)."""
    if not isinstance(default_node, ast.Call):
        return False
    fn = default_node.func
    name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")
    if name != "Query":
        return False
    if default_node.args and isinstance(default_node.args[0], ast.Constant) and default_node.args[0].value is Ellipsis:
        return True
    for kw in default_node.keywords:
        if kw.arg == "default" and isinstance(kw.value, ast.Constant) and kw.value.value is Ellipsis:
            return True
    return False


# ----------------------------------------------------------------------
# Invariant #1 & #2: Static AST Architecture Invariants
# ----------------------------------------------------------------------
def test_adr00033_invariant1_no_optional_or_defaulted_pit_params_in_read_packages() -> None:
    """Invariant #1: No public function in retrieval/, synthesis/, or explorer/ defaults ctx/as_of."""
    offenders: List[str] = []
    # Internal/DTO helpers where as_of_date is metadata (not a read cutoff) are excluded explicitly
    exempt_functions: Set[Tuple[str, str]] = {
        ("retrieval/reporting.py", "build_operational_contagion_report"),
        ("retrieval/reporting.py", "flatten_report_to_timeseries_records"),
        ("retrieval/offline_store.py", "_edge_matches_filters"),
        ("retrieval/offline_store.py", "_bfs_traverse"),
    }

    for pkg in ("retrieval", "synthesis", "explorer"):
        pkg_dir = REPO_ROOT / pkg
        for py_path in sorted(pkg_dir.rglob("*.py")):
            rel = str(py_path.relative_to(REPO_ROOT))
            tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=rel)
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if node.name.startswith("_") or (rel, node.name) in exempt_functions:
                    continue

                args = node.args
                pos_args = list(args.posonlyargs) + list(args.args)
                pos_defaults_offset = len(pos_args) - len(args.defaults)
                for idx, arg in enumerate(pos_args):
                    if arg.arg not in PIT_PARAM_NAMES:
                        continue
                    if _is_optional_annotation(arg.annotation):
                        offenders.append(f"{rel}:{node.lineno} {node.name}({arg.arg}) is Optional")
                    if idx >= pos_defaults_offset:
                        default_node = args.defaults[idx - pos_defaults_offset]
                        if not _is_required_fastapi_query(default_node):
                            offenders.append(f"{rel}:{node.lineno} {node.name}({arg.arg}) has a default value")

                for kw_arg, kw_default in zip(args.kwonlyargs, args.kw_defaults):
                    if kw_arg.arg not in PIT_PARAM_NAMES:
                        continue
                    if _is_optional_annotation(kw_arg.annotation):
                        offenders.append(f"{rel}:{node.lineno} {node.name}(*, {kw_arg.arg}) is Optional")
                    if kw_default is not None and not _is_required_fastapi_query(kw_default):
                        offenders.append(f"{rel}:{node.lineno} {node.name}(*, {kw_arg.arg}) has a default value")

    assert not offenders, f"ADR-00033 Invariant #1 violations (Optional/defaulted PiT params): {offenders}"


def test_adr00033_invariant2_all_retrieval_cypher_passes_through_temporal_filter() -> None:
    """Invariant #2: Every Memgraph session.run in retrieval/ passes through TemporalQueryFilter."""
    offenders: List[str] = []
    retrieval_dir = REPO_ROOT / "retrieval"
    temporal_methods = {"inject_active_coalescence", "inject_audit_filter", "inject_news_cutoff", "inject_active_temporal_filter"}

    for py_path in sorted(retrieval_dir.rglob("*.py")):
        rel = str(py_path.relative_to(REPO_ROOT))
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            has_session_run = False
            has_temporal_gate = False
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Call):
                    continue
                fn = sub.func
                if isinstance(fn, ast.Attribute) and fn.attr == "run":
                    if isinstance(fn.value, ast.Name) and fn.value.id == "session":
                        has_session_run = True
                called = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else "")
                if called in temporal_methods:
                    has_temporal_gate = True

            if has_session_run and not has_temporal_gate:
                if (rel, node.name) not in ALLOWED_UNGUARDED_SESSION_RUN:
                    offenders.append(f"{rel}:{node.lineno} {node.name} calls session.run without TemporalQueryFilter")

    assert not offenders, f"ADR-00033 Invariant #2 violations (unguarded Cypher execution): {offenders}"


# ----------------------------------------------------------------------
# Invariant #3 & #4: Future Exclusion & Historical Inclusion of Later-Terminated Edges
# ----------------------------------------------------------------------
def test_pit_future_exclusion_and_terminated_later_inclusion() -> None:
    """Invariants #3 & #4: Historical mode excludes future/unanchored edges and keeps later-terminated edges."""
    store = InMemorySubgraphStore(
        [
            {
                "source": "TSM",
                "target": "AAPL",
                "rel_type": "SUPPLIES_TO",
                "nature": "wafers",
                "status": "TERMINATED",  # Terminated later in 2024!
                "valid_from": "2020-01-02",
                "valid_to": "2024-06-03",
                "share": 0.40,
            },
            {
                "source": "TSM",
                "target": "NVDA",
                "rel_type": "SUPPLIES_TO",
                "nature": "future_edge",
                "status": "ACTIVE_CURRENT",
                "valid_from": "2025-01-02",
                "valid_to": None,
                "share": 0.50,
            },
            {
                "source": "TSM",
                "target": "UNANCHORED_CO",
                "rel_type": "SUPPLIES_TO",
                "nature": "no_date",
                "status": "ACTIVE_CURRENT",
                "valid_from": None,
                "valid_to": None,
                "share": 0.25,
            },
        ]
    )

    tf = TemporalQueryFilter()
    base_cypher = "MATCH (s:Company {ticker: 'TSM'})-[r:SUPPLIES_TO*1..2]->(t:Company) RETURN s, r, t"

    # Historical as of 2023-05-01:
    # - AAPL MUST be present even though its current status today is TERMINATED (Invariant #4)
    # - NVDA (valid_from 2025) MUST be excluded (Invariant #3)
    # - UNANCHORED_CO (valid_from None) MUST be excluded and counted
    ctx_2023 = PiTContext(as_of=date(2023, 5, 1), mode="historical")
    cypher_2023 = tf.inject_active_coalescence(base_cypher, ctx=ctx_2023)
    assert "ACTIVE_CURRENT" not in cypher_2023
    assert "valid_from IS NOT NULL" in cypher_2023
    assert "valid_to > '2023-05-01'" in cypher_2023

    with store.session() as session:
        rows_2023 = session.run(cypher_2023, as_of_date=ctx_2023.as_of_iso)
    assert [r["target"] for r in rows_2023] == ["AAPL"]
    assert store.last_unanchored_excluded == 1

    # Same assertion via query_ego_subgraph
    ego_2023 = store.query_ego_subgraph(ctx=ctx_2023, ticker="TSM", max_hops=1, direction="downstream")
    assert [e["target"] for e in ego_2023["edges"]] == ["AAPL"]
    assert ego_2023["unanchored_excluded"] == 1

    # Half-open boundary [valid_from, valid_to): on exact valid_to date (2024-06-03), AAPL is excluded!
    ctx_boundary = PiTContext(as_of=date(2024, 6, 3), mode="historical")
    ego_boundary = store.query_ego_subgraph(ctx=ctx_boundary, ticker="TSM", max_hops=1, direction="downstream")
    assert ego_boundary["edges"] == []

    # Live mode on 2023-05-01 DOES enforce status = ACTIVE_CURRENT -> AAPL (TERMINATED) is excluded
    ctx_live = PiTContext(as_of=date(2023, 5, 1), mode="live")
    cypher_live = tf.inject_active_coalescence(base_cypher, ctx=ctx_live)
    assert "coalesce(_rel.status, 'ACTIVE_CURRENT') = 'ACTIVE_CURRENT'" in cypher_live
    with store.session() as session:
        rows_live = session.run(cypher_live, as_of_date=ctx_live.as_of_iso)
    assert rows_live == []

    # Opt-in include_unanchored=True surfaces UNANCHORED_CO
    ctx_opt_in = PiTContext(as_of=date(2023, 5, 1), mode="historical", include_unanchored=True)
    ego_opt_in = store.query_ego_subgraph(ctx=ctx_opt_in, ticker="TSM", max_hops=1, direction="downstream")
    assert sorted(e["target"] for e in ego_opt_in["edges"]) == ["AAPL", "UNANCHORED_CO"]
    assert ego_opt_in["unanchored_excluded"] == 0


def test_analyst_query_and_search_endpoint_reject_missing_as_of() -> None:
    """Invariant #1 runtime check: AnalystQuery and POST /search refuse calls without as_of."""
    with pytest.raises(ValidationError):
        AnalystQuery(query="Who supplies Apple?")  # type: ignore[call-arg]

    client = TestClient(retrieval_app)
    resp = client.post("/search", json={"query": "TSMC", "k": 3})
    assert resp.status_code == 422


# ----------------------------------------------------------------------
# Invariant #5: SCD Type 2 Versioned Node Embeddings (No Future Vectors in History)
# ----------------------------------------------------------------------
def test_scd_type2_node_embeddings_write_and_pit_read() -> None:
    """Invariant #5: Writes close prior open rows; historical reads filter [valid_from, valid_to)."""
    executed: List[Tuple[str, Any]] = []

    class _Cur:
        rowcount = 2

        def __enter__(self) -> "_Cur":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def execute(self, sql: str, params: Any = None) -> None:
            executed.append((sql, params))

        def fetchall(self) -> List[Tuple[Any, ...]]:
            return [("Apple Inc.", "Company", "h_2020", 0.92)]

        def close(self) -> None:
            return None

    class _Conn:
        def __enter__(self) -> "_Conn":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def cursor(self) -> _Cur:
            return _Cur()

        def commit(self) -> None:
            return None

        def close(self) -> None:
            return None

    conn = _Conn()
    with patch("tools.init_db._connect_db", return_value=conn), \
         patch("embedding.embedder.embed_texts", return_value=[[0.6, 0.8]]):
        store_node_embeddings(
            [{"id": "Apple Inc.", "type": "Company"}],
            source_hash="h_2020",
            valid_from="2020-02-01",
            known_at="2020-01-31T21:30:00+00:00",
        )

    close_stmts = [p for sql, p in executed if "UPDATE node_embeddings" in sql]
    insert_stmts = [p for sql, p in executed if "INSERT INTO node_embeddings " in sql]
    src_stmts = [p for sql, p in executed if "INSERT INTO node_embedding_sources " in sql]

    assert close_stmts == [("2020-02-01", "Apple Inc.", "2020-02-01", "2020-02-01")]
    assert insert_stmts == [
        ("Apple Inc.", "2020-02-01", "Company", "[0.6,0.8]", "h_2020", "2020-01-31T21:30:00+00:00", False)
    ]
    assert src_stmts == [("Apple Inc.", "h_2020", "Company", "2020-01-31T21:30:00+00:00")]

    # Unanchored write (no valid_from, no known_at) fails closed with is_unanchored=True
    executed.clear()
    with patch("tools.init_db._connect_db", return_value=conn), \
         patch("embedding.embedder.embed_texts", return_value=[[0.6, 0.8]]):
        store_node_embeddings([{"id": "Legacy Corp.", "type": "Company"}], source_hash="h_legacy")
    unanchored_inserts = [p for sql, p in executed if "INSERT INTO node_embeddings " in sql]
    assert len(unanchored_inserts) == 1
    assert unanchored_inserts[0][6] is True

    # Backfill migration helper executes cleanly across sources, embeddings, and unanchored sentinel
    executed.clear()
    bf = backfill_node_embedding_validity(conn)
    assert bf == {
        "updated_news_sources": 2,
        "updated_sec_sources": 2,
        "updated_news_embeddings": 2,
        "updated_sec_embeddings": 2,
        "marked_unanchored_embeddings": 2,
    }
    assert any("SET is_unanchored = TRUE" in sql for sql, _ in executed)

    # Historical dense entity search filters strictly on [valid_from, valid_to) AND excludes unanchored rows
    executed.clear()
    seeder = HybridSeeder(text_to_cql_engine=MagicMock())
    ctx_hist = PiTContext(as_of=date(2020, 6, 1), mode="historical")
    with patch("retrieval.hybrid.pg_connection", return_value=conn), \
         patch("embedding.embedder.embed_texts", return_value=[[0.6, 0.8]]):
        res = seeder.search_dense_entities("Apple suppliers", ctx=ctx_hist, top_k=3)

    assert len(res) == 1
    sql_hist, params_hist = executed[0]
    assert "COALESCE(is_unanchored, FALSE) = FALSE" in sql_hist
    assert "valid_from IS NOT NULL" in sql_hist
    assert "valid_from <= %s::date" in sql_hist
    assert "(valid_to IS NULL OR valid_to > %s::date)" in sql_hist
    assert params_hist == ("[0.6,0.8]", "2020-06-01", "2020-06-01", "[0.6,0.8]", 3)


# ----------------------------------------------------------------------
# Invariant #6: Validated Traversal Depth & Directed Multi-Hop Expansion
# ----------------------------------------------------------------------
def test_validated_traversal_hops_and_directed_temporal_expansion() -> None:
    """Invariant #6: Invalid hop bounds raise ValueError / 422; valid hops honor [r*1..H]."""
    assert validate_hops(1) == 1
    assert validate_hops(10) == 10
    for bad in (0, -1, 11, True, "2; MATCH (n) DELETE n"):
        with pytest.raises(ValueError):
            validate_hops(bad)

    ctx = PiTContext(as_of=date(2024, 5, 1), mode="historical")
    mock_session = MagicMock()
    mock_session.run.return_value = []
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    store = MemgraphSubgraphStore(driver=mock_driver)
    with pytest.raises(ValueError):
        store.traverse(["AAPL"], hops=15, ctx=ctx)

    store.traverse(["TSM"], hops=3, ctx=ctx, limit=25)

    executed_cypher = mock_session.run.call_args[0][0]
    assert "[r*1..3]->" in executed_cypher
    assert "ALL(_rel IN r WHERE _rel.valid_from IS NOT NULL AND _rel.valid_from <= '2024-05-01'" in executed_cypher
    assert "ORDER BY hop ASC" in executed_cypher
    assert "LIMIT $limit" in executed_cypher

    # enrich_nodes also validates hops and injects temporal predicate + LIMIT via MemgraphSubgraphStore
    mock_session.reset_mock()
    with patch("knowledge_graph.memgraph_driver.get_memgraph_driver", return_value=mock_driver):
        with pytest.raises(ValueError):
            enrich_nodes(["TSM"], ctx=ctx, hops=0)
        enrich_nodes(["TSM"], ctx=ctx, hops=2, limit=15)

    enrich_cypher = mock_session.run.call_args[0][0]
    assert "[r*1..2]-" in enrich_cypher
    assert "ALL(_rel IN r WHERE" in enrich_cypher
    assert "LIMIT $limit" in enrich_cypher

    # POST /search rejects hops > 10 with 422
    client = TestClient(retrieval_app)
    bad_hops_resp = client.post("/search", json={"query": "TSM", "as_of": "2024-05-01", "hops": 12})
    assert bad_hops_resp.status_code == 422


# ----------------------------------------------------------------------
# Invariants #7 & #8: U.S. Regular-Session Close Calendar & Form 8-K Termination
# ----------------------------------------------------------------------
def test_us_equity_regular_session_calendar_and_actionable_session() -> None:
    """Invariant #7 calendar checks: 16:00 ET regular close, 13:00 ET half-days, holidays, mourning days."""
    # Standard weekday before vs after 16:00 ET (EDT = UTC-4 in May)
    d_before, dt_before, imp_before = actionable_session("2024-05-01T19:59:00Z")  # 15:59 EDT
    assert d_before == date(2024, 5, 1) and dt_before is not None and imp_before is False

    d_exact, _, _ = actionable_session("2024-05-01T20:00:00Z")  # 16:00 EDT inclusive
    assert d_exact == date(2024, 5, 1)

    d_after, _, _ = actionable_session("2024-05-01T20:01:00Z")  # 16:01 EDT -> next session
    assert d_after == date(2024, 5, 2)

    # Early-close half-day: July 3, 2024 closes at 13:00 EDT (17:00 UTC); July 4 is holiday -> July 5
    assert us_equity_regular_close_et(date(2024, 7, 3)) is not None
    assert us_equity_regular_close_et(date(2024, 7, 3)).hour == 13  # type: ignore[union-attr]
    assert is_us_equity_trading_day(date(2024, 7, 4)) is False

    d_half_before, _, _ = actionable_session("2024-07-03T16:55:00Z")  # 12:55 EDT
    assert d_half_before == date(2024, 7, 3)

    d_half_after, _, _ = actionable_session("2024-07-03T17:05:00Z")  # 13:05 EDT -> skips Jul 4 -> Jul 5
    assert d_half_after == date(2024, 7, 5)

    # Day after Thanksgiving 2024 (Nov 29) early close at 13:00 EST (18:00 UTC) -> rolls to Mon Dec 2
    d_bf_after, _, _ = actionable_session("2024-11-29T18:05:00Z")  # 13:05 EST
    assert d_bf_after == date(2024, 12, 2)

    # Unscheduled national days of mourning: Dec 5, 2018 (Bush) and Jan 9, 2025 (Carter)
    assert is_us_equity_trading_day(date(2018, 12, 5)) is False
    assert next_us_equity_trading_day(date(2018, 12, 4)) == date(2018, 12, 6)
    assert is_us_equity_trading_day(date(2025, 1, 9)) is False
    assert next_us_equity_trading_day(date(2025, 1, 8)) == date(2025, 1, 10)

    # Juneteenth observed starting 2022 only (not 2021)
    assert is_us_equity_trading_day(date(2021, 6, 18)) is True
    assert is_us_equity_trading_day(date(2022, 6, 20)) is False

    # Imputed timestamp (date-only): conservatively maps to next trading session
    d_imp, dt_imp, is_imp = actionable_session(None, fallback_doc_date="2024-05-01")
    assert d_imp == date(2024, 5, 2) and dt_imp is None and is_imp is True


def test_form_8k_termination_enforces_invariants_7_and_8() -> None:
    """Invariants #7 & #8: Only 8-K may set valid_to; valid_to derives from known_at, not event date."""
    mock_session = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    worker = EdgarWorker(memgraph_driver=mock_driver, pg_conn=None)
    worker.linker.link_entity = MagicMock(return_value=None)

    # Invariant #8: 10-K, 10-Q, and news are forbidden from closing relationships
    for bad_form in ("10-K", "10-Q", "news"):
        with pytest.raises(ValueError, match="Invariant #8"):
            worker.terminate_sec_relationship(
                source_name="TSMC",
                target_name="Apple Inc.",
                rel_type="SUPPLIES_TO",
                form_type=bad_form,
                accession_number="0000320193-24-000001",
                filing_date="2024-05-06",
            )

    # Unresolved generic placeholder ("Customer A") is quarantined, never closing arbitrary edges
    quarantined = worker.terminate_sec_relationship(
        source_name="TSMC",
        target_name="Customer A",
        rel_type="SUPPLIES_TO",
        form_type="8-K",
        accession_number="0000320193-24-000002",
        filing_date="2024-05-06",
        acceptance_datetime="2024-05-06T20:15:00Z",
        terminated_on="2024-05-01",
        source_text="TSMC terminated its supply agreement with Customer A.",
    )
    assert quarantined["terminated"] is False
    mock_session.run.assert_not_called()

    # Invariant #7: 8-K accepted at 16:15 EDT (20:15 UTC) on 2024-05-06 for an event on 2024-05-01
    # must set valid_to = 2024-05-07 (next U.S. session), preserving visibility on 2024-05-06!
    res = worker.terminate_sec_relationship(
        source_name="TSMC",
        target_name="Apple Inc.",
        rel_type="SUPPLIES_TO",
        form_type="8-K",
        accession_number="0000320193-24-000003",
        filing_date="2024-05-06",
        acceptance_datetime="2024-05-06T20:15:00Z",
        terminated_on="2024-05-01",
        source_text="TSMC terminated its wafer supply agreement with Apple Inc.",
    )
    assert res["terminated"] is True
    assert res["valid_to"] == "2024-05-07"
    assert res["terminated_on"] == "2024-05-01"
    mock_session.run.assert_called_once()
    run_kwargs = mock_session.run.call_args.kwargs
    assert run_kwargs["valid_to"] == "2024-05-07"
    assert run_kwargs["terminated_on"] == "2024-05-01"
