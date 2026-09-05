# graph/graph_store.py
# -*- coding: utf-8 -*-
"""Entity & relationship extraction and persistence to Memgraph.

Extracts structured knowledge graphs (nodes and edges) from financial news text
using a local LLM (e.g., Qwen via Ollama) and merges them into Memgraph.
"""

import os
import re
import json
import logging
import atexit
from typing import Optional, Dict, Any, List
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from .memgraph_driver import get_memgraph_driver

logger = logging.getLogger(__name__)

# Constants for safe Cypher identifier lengths
MAX_LABEL_LENGTH = 64
MAX_REL_TYPE_LENGTH = 64

# LLM / Ollama Configuration
def _default_ollama_url() -> str:
    """Detect whether running inside Docker container or host."""
    env_url = os.getenv("QWEN_API_BASE")
    if env_url:
        return env_url
    if os.path.exists("/.dockerenv"):
        return "http://host.docker.internal:11434/api/generate"
    return "http://localhost:11434/api/generate"

OLLAMA_API_BASE = _default_ollama_url()
OLLAMA_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "qwen3:8b")

SYSTEM_EXTRACTION_PROMPT = """You are an expert financial knowledge graph extractor.
Read the provided financial news article and extract key entities and relationships.
Output ONLY a valid JSON object matching this schema:
{
    "nodes": [
        {"id": "<Entity Name>", "type": "<NodeType>", "properties": {"ticker": "<TICKER if applicable>", "<key>": "<value>"}}
    ],
    "edges": [
        {"source": "<Source Entity Name>", "target": "<Target Entity Name>", "type": "<EDGE_TYPE>", "properties": {}}
    ]
}
Allowed node types: Company, Person, Product, Technology, Sector, Metric, Location.
Allowed edge types: ACQUIRED, INVESTS_IN, PARTNERED_WITH, COMPETES_WITH, LEADS, PRODUCES, IMPACTS, REPORTS.
Do not include any explanation or markdown formatting. Output raw JSON only."""


def _clean_json_response(raw_text: str) -> dict:
    """Extract and parse JSON from LLM output, handling markdown fences or reasoning blocks."""
    if not raw_text:
        return {"nodes": [], "edges": []}

    # 1. Remove markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)

    # 2. Try direct JSON parse
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 3. Extract JSON object substring between { and }
    match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    logger.warning("Could not parse LLM output as JSON: %s", raw_text[:200])
    return {"nodes": [], "edges": []}


def _extract_entities_via_ollama(text: str) -> dict:
    """Query local Ollama server for entity extraction."""
    try:
        import httpx
    except ImportError:
        logger.warning("httpx not installed; skipping Ollama extraction")
        return {"nodes": [], "edges": []}

    prompt = f"{SYSTEM_EXTRACTION_PROMPT}\n\nArticle:\n{text}\n\nJSON:"

    # Normalize endpoint URL (support both /v1/chat/completions and /api/generate)
    api_url = OLLAMA_API_BASE
    if "/v1" in api_url:
        endpoint = api_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": OLLAMA_MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_EXTRACTION_PROMPT},
                {"role": "user", "content": text},
            ],
            "temperature": 0.0,
            "stream": False,
        }
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(endpoint, json=payload)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return _clean_json_response(content)
        except Exception as exc:
            logger.warning("Ollama /v1 API call failed: %s; trying /api/generate", exc)

    # Default to native /api/generate
    gen_url = OLLAMA_API_BASE
    if "/v1" in gen_url:
        gen_url = gen_url.split("/v1")[0].rstrip("/") + "/api/generate"
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                gen_url,
                json={
                    "model": OLLAMA_MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return _clean_json_response(data.get("response", ""))
    except Exception as exc:
        logger.error("Ollama extraction request failed: %s", exc)
        return {"nodes": [], "edges": []}


def extract_entities(text: str) -> dict:
    """Extract entities and relationships from text using local LLM."""
    if not text or not text.strip():
        return {"nodes": [], "edges": []}
    return _extract_entities_via_ollama(text)


def store_graph_entities(source_hash: str, text: str) -> None:
    """Extract entities and relationships and persist them to Memgraph."""
    graph_data = extract_entities(text)

    nodes: List[Dict[str, Any]] = graph_data.get("nodes", [])
    edges: List[Dict[str, Any]] = graph_data.get("edges", [])

    if not nodes and not edges:
        return

    driver = get_memgraph_driver()

    with driver.session() as session:
        # 1. Insert/Merge Article News Node
        session.run(
            "MERGE (n:News {id: $source_hash}) SET n.updated_at = datetime()",
            source_hash=source_hash,
        )

        # 2. Insert/Merge Entity Nodes
        for node in nodes:
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                continue

            raw_type = str(node.get("type", "Entity")).strip()
            # Clean label to ensure valid Cypher identifier
            label = "".join(c for c in raw_type if c.isalnum())[:MAX_LABEL_LENGTH] or "Entity"
            props = node.get("properties", {}) or {}

            query = f"MERGE (n:{label} {{id: $id}}) SET n += $props"
            session.run(query, id=node_id, props=props)

            # Link news article to mentioned entity
            mention_query = f"MATCH (news:News {{id: $source_hash}}), (e:{label} {{id: $id}}) MERGE (news)-[:MENTIONS]->(e)"
            session.run(mention_query, source_hash=source_hash, id=node_id)

        # 3. Insert/Merge Relationship Edges
        for edge in edges:
            source = str(edge.get("source", "")).strip()
            target = str(edge.get("target", "")).strip()
            if not source or not target:
                continue

            raw_type = str(edge.get("type", "RELATED_TO")).strip().upper()
            rel_type = "".join(c for c in raw_type if c.isalnum() or c == "_")[:MAX_REL_TYPE_LENGTH] or "RELATED_TO"
            props = edge.get("properties", {}) or {}

            query = f"""
            MATCH (s {{id: $source}})
            MATCH (t {{id: $target}})
            MERGE (s)-[r:{rel_type}]->(t)
            SET r += $props, r.source_hash = $source_hash
            """
            session.run(query, source=source, target=target, props=props, source_hash=source_hash)

    logger.info(
        "Stored %d nodes and %d edges in Memgraph for source_hash %s",
        len(nodes),
        len(edges),
        source_hash,
    )

    # 4. Persist node embeddings to PGVECTOR
    if nodes:
        try:
            from .embedding_store import store_node_embeddings
            store_node_embeddings(nodes, source_hash=source_hash)
        except Exception as exc:
            logger.warning("Failed to store node embeddings for %s: %s", source_hash, exc)
