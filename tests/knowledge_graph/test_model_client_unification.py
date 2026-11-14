"""Verification tests for ADR-00030: Purge non-FT Qwen runtime dependencies."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

from ingestion.sec.parser import extract_sec_relationships
from knowledge_graph.graph_store import extract_entities
from retrieval.text_to_cql import TextToCQL
from sft_pipeline.serving.model_client import YarnBallModelClient, resolve_ollama_urls

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUNTIME_DIRS = (
    "knowledge_graph",
    "ingestion",
    "retrieval",
    "explorer",
    "contagion",
    "streaming",
)


def test_runtime_modules_contain_no_legacy_qwen3_or_env_vars():
    """Runtime inference modules and .env.example must not reference qwen3 or QWEN_MODEL_NAME."""
    forbidden = ("qwen3", "QWEN_MODEL_NAME", "QWEN_API_BASE")
    violations = []

    for rel_dir in RUNTIME_DIRS:
        for py_file in (PROJECT_ROOT / rel_dir).rglob("*.py"):
            if "__pycache__" in py_file.parts:
                continue
            text = py_file.read_text(encoding="utf-8")
            for token in forbidden:
                if token in text:
                    violations.append((str(py_file.relative_to(PROJECT_ROOT)), token))

    env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    for token in forbidden:
        if token in env_example:
            violations.append((".env.example", token))

    assert violations == [], f"Found forbidden non-FT Qwen references: {violations}"


def test_resolve_ollama_urls_normalizes_host_and_paths():
    """resolve_ollama_urls strips /v1 or /api/generate and rewrites host.docker.internal outside Docker."""
    with patch("os.path.exists", return_value=False):
        base, gen = resolve_ollama_urls("http://host.docker.internal:11434/v1")
        assert base == "http://127.0.0.1:11434"
        assert gen == "http://127.0.0.1:11434/api/generate"

        base2, gen2 = resolve_ollama_urls("http://127.0.0.1:11434/api/generate")
        assert base2 == "http://127.0.0.1:11434"
        assert gen2 == "http://127.0.0.1:11434/api/generate"

    with patch("os.path.exists", return_value=True), patch.dict(os.environ, {}, clear=False):
        os.environ.pop("OLLAMA_API_BASE", None)
        base3, gen3 = resolve_ollama_urls(None)
        assert base3 == "http://host.docker.internal:11434"
        assert gen3 == "http://host.docker.internal:11434/api/generate"


def test_graph_store_routes_through_yarnball_model_client():
    """graph_store.extract_entities invokes YarnBallModelClient targeting yarnball-qwen:7b."""
    seen_payloads = []

    def fake_transport(payload):
        seen_payloads.append(payload)
        return {
            "response": json.dumps({
                "nodes": [{"id": "Apple Inc.", "type": "Company", "properties": {"ticker": "AAPL"}}],
                "edges": [],
            })
        }

    client = YarnBallModelClient(transport_fn=fake_transport)
    result = extract_entities("Apple Inc. expanded chip production.", model_client=client)

    assert len(seen_payloads) == 1
    assert seen_payloads[0]["model"] == "yarnball-qwen:7b"
    assert result["nodes"][0]["id"] == "Apple Inc."


def test_sec_parser_routes_through_yarnball_model_client():
    """extract_sec_relationships invokes YarnBallModelClient targeting yarnball-qwen:7b."""
    seen_payloads = []

    def fake_transport(payload):
        seen_payloads.append(payload)
        return {
            "response": json.dumps({
                "nodes": [{"id": "TSMC", "type": "Company", "properties": {"ticker": "TSM"}}],
                "edges": [],
            })
        }

    client = YarnBallModelClient(transport_fn=fake_transport)
    result = extract_sec_relationships("We rely on TSMC as our foundry supplier.", model_client=client)

    assert len(seen_payloads) == 1
    assert seen_payloads[0]["model"] == "yarnball-qwen:7b"
    assert result["nodes"][0]["id"] == "TSMC"


def test_text_to_cql_routes_through_yarnball_model_client():
    """TextToCQL._call_llm delegates to YarnBallModelClient targeting yarnball-qwen:7b."""
    seen_payloads = []

    def fake_transport(payload):
        seen_payloads.append(payload)
        return {"response": "MATCH (c:Company {ticker: 'AAPL'}) RETURN c.id LIMIT 50;"}

    client = YarnBallModelClient(transport_fn=fake_transport)
    engine = TextToCQL(model_client=client)
    cypher = engine._call_llm("Find Apple", "System prompt")

    assert len(seen_payloads) == 1
    assert seen_payloads[0]["model"] == "yarnball-qwen:7b"
    assert cypher == "MATCH (c:Company {ticker: 'AAPL'}) RETURN c.id LIMIT 50"
