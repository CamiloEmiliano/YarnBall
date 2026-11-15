"""Unit tests verifying ADR-00025: Pattern-based entity stoplist & dynamic SEC subsidiary rollup."""

from __future__ import annotations

from typing import Any, List, Tuple
from unittest.mock import patch

from knowledge_graph.entity_resolver import EntityResolver
from ontology import (
    SEED_SUBSIDIARY_TO_PARENT_CIK_MAP,
    Tier4OntologyGate,
    invalidate_subsidiary_registry_cache,
    is_generic_placeholder,
    load_dynamic_subsidiary_registry,
    validate_graph_payload,
)


class _StubCursor:
    def __init__(self, cik_rows: List[Tuple[str, str]], name_rows: List[Tuple[str, str, str, str]]):
        self._cik_rows = cik_rows
        self._name_rows = name_rows
        self._last_result: List[Tuple[Any, ...]] = []
        self.closed = False

    def execute(self, query: str, params: Any = None) -> None:
        if "sub_co.cik, s.parent_cik" in query:
            self._last_result = list(self._cik_rows)
        elif "s.normalized_name, c.company_name" in query:
            self._last_result = list(self._name_rows)
        else:
            self._last_result = []

    def fetchall(self) -> List[Tuple[Any, ...]]:
        return list(self._last_result)

    def close(self) -> None:
        self.closed = True


class _StubPgConn:
    def __init__(self, cik_rows: List[Tuple[str, str]], name_rows: List[Tuple[str, str, str, str]]):
        self._cik_rows = cik_rows
        self._name_rows = name_rows
        self.query_calls = 0

    def cursor(self) -> _StubCursor:
        self.query_calls += 1
        return _StubCursor(self._cik_rows, self._name_rows)


def test_anonymized_counterparties_and_indefinite_role_descriptors_flagged() -> None:
    # SEC ASC 280 anonymized counterparties
    for label in (
        "Customer A",
        "Supplier X",
        "Vendor 1",
        "Client #2",
        "Partner B",
        "Distributor 12",
    ):
        assert is_generic_placeholder(label) is True, f"Expected placeholder for {label!r}"

    # Indefinite / possessive role phrases lacking proper corporate suffixes or tickers
    for phrase in (
        "a major foundry partner",
        "our primary OEM partner",
        "one of our largest customers",
        "the sole supplier",
        "several external manufacturers",
    ):
        assert is_generic_placeholder(phrase) is True, f"Expected placeholder for {phrase!r}"

    # Pronouns & SEC boilerplate subject references
    for ref in ("it", "It", "they", "them", "its affiliates", "our affiliates", "the registrant"):
        assert is_generic_placeholder(ref) is True, f"Expected placeholder for {ref!r}"

    # Real corporate names, proper-suffix entities, ticker-qualified names, and uppercase 'IT' ticker must survive
    for real_name in (
        "Apple Inc.",
        "Taiwan Semiconductor Manufacturing Company",
        "The Boeing Company",
        "Partner Re Ltd.",
        "Vendor Tech Corp",
        "Our Partner (AAPL)",
        "IT",
        "",
        None,
    ):
        assert is_generic_placeholder(real_name) is False, f"Unexpected placeholder for {real_name!r}"


def test_validate_graph_payload_drops_anonymized_and_indefinite_nodes() -> None:
    payload = {
        "nodes": [
            {"id": "Customer A", "type": "Company", "properties": {}},
            {"id": "a major foundry partner", "type": "Company", "properties": {}},
            {"id": "Apple Inc.", "type": "Company", "properties": {"ticker": "AAPL"}},
            {"id": "TSMC", "type": "Company", "properties": {"ticker": "TSM"}},
        ],
        "edges": [
            {"source": "Apple Inc.", "target": "Customer A", "type": "SUPPLIES_TO", "properties": {}},
            {"source": "a major foundry partner", "target": "Apple Inc.", "type": "SUPPLIES_TO", "properties": {}},
            {"source": "TSMC", "target": "Apple Inc.", "type": "SUPPLIES_TO", "properties": {}},
        ],
    }
    gated = validate_graph_payload(payload, strict=True)
    node_ids = [n["id"] for n in gated["nodes"]]
    assert node_ids == ["Apple Inc.", "TSMC"]
    assert len(gated["edges"]) == 1
    assert gated["edges"][0]["source"] == "TSMC"
    assert gated["edges"][0]["target"] == "Apple Inc."


def test_offline_test_determinism_without_postgres() -> None:
    invalidate_subsidiary_registry_cache()
    with patch("knowledge_graph.db.pg_connection", side_effect=RuntimeError("postgres offline")):
        gate = Tier4OntologyGate()
        assert gate.subsidiary_map == SEED_SUBSIDIARY_TO_PARENT_CIK_MAP
        raw = "(:Company {name: 'DISH DBS', cik: '0001001811'})-[:SUPPLIES_TO]->(:Company {name: 'EchoStar', cik: '0001415404'})"
        repaired = gate.apply_guardrails(raw, "EXTRACT_SEC_GRAPH")
        assert "0001001082" in repaired
        assert "0001001811" not in repaired
    invalidate_subsidiary_registry_cache()


def test_dynamic_subsidiary_registry_db_load_cache_and_rollup() -> None:
    invalidate_subsidiary_registry_cache()
    stub_conn = _StubPgConn(
        cik_rows=[("0009998888", "0000320193")],
        name_rows=[("beats electronics", "Apple Inc.", "0000320193", "AAPL")],
    )

    with patch("knowledge_graph.db.pg_connection", return_value=stub_conn):
        reg1 = load_dynamic_subsidiary_registry()
        reg2 = load_dynamic_subsidiary_registry()  # served from TTL cache
        assert reg1.from_database is True
        assert reg1.cik_map["0009998888"] == "0000320193"
        assert reg1.cik_map["0001001811"] == "0001001082"  # seed retained
        assert reg2.name_to_parent["beats electronics"] == ("Apple Inc.", "0000320193", "AAPL")
        assert stub_conn.query_calls == 1

        # Invalidate cache and verify fresh reload works
        invalidate_subsidiary_registry_cache()
        gate = Tier4OntologyGate()
        assert gate.subsidiary_map["0009998888"] == "0000320193"
        cypher = "(:Company {name: 'Beats Sub', cik: '0009998888'})-[:SUPPLIES_TO]->(:Company {name: 'Foxconn'})"
        out = gate.apply_guardrails(cypher, "EXTRACT_SEC_GRAPH")
        assert "0000320193" in out
        assert "0009998888" not in out

        json_gated = gate.gate_json_payload(
            {
                "nodes": [
                    {"id": "Beats Electronics, LLC", "type": "Company", "properties": {}},
                ],
                "edges": [],
            }
        )
        assert json_gated["nodes"][0]["properties"]["cik"] == "0000320193"
        assert json_gated["nodes"][0]["properties"]["parent_company"] == "Apple Inc."
    invalidate_subsidiary_registry_cache()


def test_entity_resolver_merges_dynamic_and_seed_subsidiaries_to_parent() -> None:
    resolver = EntityResolver(
        subsidiary_cik_map={"0001061219": "0001045810"},
        subsidiary_name_map={
            "mellanox": ("NVIDIA Corporation", "0001045810", "NVDA"),
            "beats electronics": ("Apple Inc.", "0000320193", "AAPL"),
        },
    )
    raw_nodes = [
        {"id": "Mellanox Technologies", "label": "Company", "ticker": "MLNX", "cik": "0001061219", "source_hash": "h1"},
        {"id": "NVIDIA Corporation", "label": "Company", "ticker": "NVDA", "cik": "0001045810", "source_hash": "h2"},
        {"id": "Beats Electronics LLC", "label": "Company", "ticker": None, "source_hash": "h3"},
        {"id": "Apple Inc.", "label": "Company", "ticker": "AAPL", "cik": "0000320193", "source_hash": "h4"},
        {"id": "Customer A", "label": "Company", "ticker": None, "source_hash": "h5"},
        {"id": "our primary OEM partner", "label": "Company", "ticker": None, "source_hash": "h6"},
    ]

    mapping, resolved = resolver.resolve_nodes(raw_nodes)
    assert "Customer A" not in mapping
    assert "our primary OEM partner" not in mapping
    assert mapping["Mellanox Technologies"] == "NVIDIA Corporation"
    assert mapping["Beats Electronics LLC"] == "Apple Inc."
    assert len(resolved) == 2

    nvda = next(n for n in resolved if n["id"] == "NVIDIA Corporation")
    assert nvda["ticker"] == "NVDA"
    aapl = next(n for n in resolved if n["id"] == "Apple Inc.")
    assert aapl["ticker"] == "AAPL"
