"""Unit tests verifying ADR-00024: Clause-scoped negation, hedge filtering, and per-edge noise gating."""

import json

from ontology import (
    apply_runtime_guardrails,
    classify_polarity_from_text,
    has_unnegated_pattern,
    infer_terminated_relation,
    is_triple_hedged,
    split_financial_clauses,
    validate_graph_payload,
)


def test_clause_splitter_preserves_abbreviations_and_decimals() -> None:
    text = (
        "Under Item 1.01, Apple Inc. avoided supply delays, "
        "and Taiwan Semiconductor Co. expanded 3.5nm wafer production."
    )
    clauses = split_financial_clauses(text)
    assert len(clauses) == 2
    assert "Item 1.01" in clauses[0]
    assert "Apple Inc." in clauses[0]
    assert "Taiwan Semiconductor Co." in clauses[1]
    assert "3.5nm wafer production" in clauses[1]


def test_negation_isolated_to_clause_and_non_negation_lifecycle_verbs_fixed() -> None:
    # Previous clause has negation ("avoided"), subsequent coordinated clause asserts expansion.
    text_avoided = "Apple avoided supply chain delays, and expanded production capacity at TSMC."
    assert has_unnegated_pattern(r"expand(?:s|ed|ing)?", text_avoided) is True
    assert classify_polarity_from_text(text_avoided) == "EXPANDING_BULLISH"

    # Terminal lifecycle verb ("terminated") is no longer classified as negation of other verbs.
    text_terminated = (
        "Apple terminated its agreement with LegacyChip and expanded orders with TSMC."
    )
    assert has_unnegated_pattern(r"expand(?:s|ed|ing)?", text_terminated) is True
    assert classify_polarity_from_text(text_terminated) == "EXPANDING_BULLISH"

    # Cross-clause negation followed by positive assertion.
    text_but = "Intel did not expand in Europe, but expanded packaging facilities in Malaysia."
    assert has_unnegated_pattern(r"expand(?:s|ed|ing)?", text_but) is True

    # Parenthetical interruption within a single negated clause must remain suppressed.
    text_single_negated = "Apple did not, during fiscal 2023, expand capacity in Europe."
    assert has_unnegated_pattern(r"expand(?:s|ed|ing)?", text_single_negated) is False


def test_evidence_scoped_hedging_preserves_factual_risk_factor_triples() -> None:
    prompt = (
        "Intel manufactures wafers for Apple under a 2025 contract. "
        "Amazon may from time to time consider alternative cloud suppliers such as Oracle."
    )
    assert is_triple_hedged(prompt, ["Intel", "Apple"]) is False
    assert is_triple_hedged(prompt, ["Amazon", "Oracle"]) is True

    mixed_cypher = (
        "(:Company {name: 'Intel'})-[:SUPPLIES_TO]->(:Company {name: 'Apple'})\n"
        "(:Company {name: 'Amazon'})-[:PARTNERED_WITH]->(:Company {name: 'Oracle'})"
    )
    repaired = apply_runtime_guardrails(prompt, mixed_cypher, "EXTRACT_SEC_GRAPH")
    assert "[:SUPPLIES_TO]" in repaired
    assert "Intel" in repaired and "Apple" in repaired
    assert "PARTNERED_WITH" not in repaired
    assert "Oracle" not in repaired


def test_mixed_schema_noise_edges_pruned_individually() -> None:
    prompt = "TSMC manufactures chips for Apple. Apple stock correlated with Microsoft."
    mixed_cypher = (
        "(:Company {name: 'TSMC'})-[:SUPPLIES_TO]->(:Company {name: 'Apple'})\n"
        "(:Company {name: 'Apple'})-[:CORRELATES_WITH]->(:Company {name: 'MSFT'})"
    )
    repaired = apply_runtime_guardrails(prompt, mixed_cypher, "EXTRACT_SEC_GRAPH")
    assert "[:SUPPLIES_TO]" in repaired
    assert "CORRELATES_WITH" not in repaired

    json_payload = {
        "nodes": [
            {"id": "TSMC", "type": "Company", "properties": {"name": "TSMC"}},
            {"id": "Apple", "type": "Company", "properties": {"name": "Apple"}},
            {"id": "MSFT", "type": "Company", "properties": {"name": "MSFT"}},
        ],
        "edges": [
            {"source": "TSMC", "target": "Apple", "type": "SUPPLIES_TO", "properties": {}},
            {"source": "Apple", "target": "MSFT", "type": "CORRELATES_WITH", "properties": {}},
        ],
    }
    validated = validate_graph_payload(json_payload, source_text=prompt, strict=True)
    assert len(validated["edges"]) == 1
    assert validated["edges"][0]["type"] == "SUPPLIES_TO"
    assert validated["edges"][0]["source"] == "TSMC"


def test_brace_aware_parsing_and_context_aware_termination_repair() -> None:
    # Nested/quoted braces must parse intact without regex truncation.
    prompt_license = "Apple terminated its patent royalty licensing agreement with Qualcomm."
    cypher_nested = (
        '(:Company {name: "Apple"})-[:TERMINATED {note: "{royalty-tier}", status: "TERMINATED"}]->'
        '(:Company {name: "Qualcomm"})'
    )
    repaired_license = apply_runtime_guardrails(prompt_license, cypher_nested, "EXTRACT_SEC_GRAPH")
    assert "[:LICENSES_FROM" in repaired_license
    assert 'note: "{royalty-tier}"' in repaired_license
    assert "TERMINATED" in repaired_license
    assert "is_active: false" in repaired_license

    # Context inference variants
    assert infer_terminated_relation("Ended joint venture partnership between A and B") == "PARTNERED_WITH"
    assert infer_terminated_relation("Customer purchasing contract with Beta Corp cancelled") == "CUSTOMER_OF"
    assert infer_terminated_relation("Terminated wafer manufacturing supply agreement") == "SUPPLIES_TO"

    # End-to-end JSON payload context-aware termination repair
    jv_payload = {
        "nodes": [
            {"id": "Sony", "type": "Company", "properties": {"name": "Sony"}},
            {"id": "Honda", "type": "Company", "properties": {"name": "Honda"}},
        ],
        "edges": [
            {"source": "Sony", "target": "Honda", "type": "DISSOLVED", "properties": {}},
        ],
    }
    validated_jv = validate_graph_payload(
        jv_payload,
        source_text="Sony and Honda dissolved their electric vehicle joint venture partnership.",
        strict=True,
    )
    assert len(validated_jv["edges"]) == 1
    assert validated_jv["edges"][0]["type"] == "PARTNERED_WITH"
    assert validated_jv["edges"][0]["properties"]["status"] == "TERMINATED"
    assert validated_jv["edges"][0]["properties"]["is_active"] is False

    # Check JSON string repair via apply_runtime_guardrails as well
    json_str_out = apply_runtime_guardrails(
        prompt_license,
        json.dumps(
            {
                "nodes": [
                    {"id": "Apple", "type": "Company"},
                    {"id": "Qualcomm", "type": "Company"},
                ],
                "edges": [
                    {"source": "Apple", "target": "Qualcomm", "type": "TERMINATED", "properties": {}}
                ],
            }
        ),
        "EXTRACT_SEC_GRAPH",
    )
    parsed = json.loads(json_str_out)
    assert parsed["edges"][0]["type"] == "LICENSES_FROM"


def test_edgar_worker_propagates_source_text_for_hedge_and_termination_gating() -> None:
    from unittest.mock import MagicMock

    from ingestion.sec.harvester import EdgarWorker

    mock_driver = MagicMock()
    worker = EdgarWorker(memgraph_driver=mock_driver, pg_conn=None)
    worker.linker.link_entity = MagicMock(return_value=None)

    section_text = (
        "Intel manufactures wafers for Apple under a 2025 supply agreement. "
        "Amazon may from time to time consider partnering with Oracle."
    )
    _nodes, edges = worker._persist_sec_graph_data(
        primary_name="Apple Inc.",
        primary_ticker="AAPL",
        graph_data={
            "nodes": [
                {"id": "Intel", "type": "Company", "properties": {}},
                {"id": "Apple", "type": "Company", "properties": {}},
                {"id": "Amazon", "type": "Company", "properties": {}},
                {"id": "Oracle", "type": "Company", "properties": {}},
            ],
            "edges": [
                {"source": "Intel", "target": "Apple", "type": "SUPPLIES_TO", "properties": {}},
                {"source": "Amazon", "target": "Oracle", "type": "PARTNERED_WITH", "properties": {}},
            ],
        },
        accession_number="0000320193-26-000011",
        filing_date="2026-10-30",
        form_type="10-K",
        section="Item 1A Risk Factors",
        source_text=section_text,
    )
    assert edges == [{"source": "Intel", "target": "Apple", "type": "SUPPLIES_TO"}]
