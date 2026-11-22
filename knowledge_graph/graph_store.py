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
from typing import TYPE_CHECKING, Optional, Dict, Any, List, Tuple
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

if TYPE_CHECKING:
    from sft_pipeline.serving.model_client import YarnBallModelClient

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

from ontology.schema import format_extraction_prompt_vocab
from ontology.validation import validate_graph_payload

_ALLOWED_NODE_TYPES_STR, _ALLOWED_REL_TYPES_STR = format_extraction_prompt_vocab()

SYSTEM_EXTRACTION_PROMPT = f"""You are an expert financial knowledge graph extractor.
Read the provided financial news article and extract key entities and relationships.
Output ONLY a valid JSON object matching this schema:
{{
    "nodes": [
        {{"id": "<Entity Name>", "type": "<NodeType>", "properties": {{"ticker": "<TICKER if applicable>", "<key>": "<value>"}}}}
    ],
    "edges": [
        {{"source": "<Source Entity Name>", "target": "<Target Entity Name>", "type": "<EDGE_TYPE>", "properties": {{"input_group": "<component or input group>", "share": "<optional 0.0-1.0 procurement/revenue share if stated>", "buffer_days": "<optional inventory buffer days if stated>", "sole_source": "<true if explicitly sole/single source>"}}}}
    ]
}}
Allowed node types: {_ALLOWED_NODE_TYPES_STR}.
Allowed edge types: {_ALLOWED_REL_TYPES_STR}.
Do not include any explanation or markdown formatting. Output raw JSON only."""


class GraphExtractionError(RuntimeError):
    """Raised when LLM graph extraction fails transport, parsing, or schema validation."""

    def __init__(self, message: str, error_stage: str = "EXTRACTION_PARSE_ERROR"):
        super().__init__(message)
        self.error_stage = error_stage


class ExtractionNode(BaseModel):
    """Schema-validated extracted graph entity."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(..., min_length=1)
    type: str = Field(default="Entity", min_length=1)
    properties: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "type", mode="before")
    @classmethod
    def _strip_non_empty(cls, value: Any) -> str:
        text = str(value).strip() if value is not None else ""
        if not text:
            raise ValueError("must be a non-empty string")
        return text


class ExtractionEdge(BaseModel):
    """Schema-validated extracted relationship edge."""

    model_config = ConfigDict(extra="allow")

    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    type: str = Field(default="RELATED_TO", min_length=1)
    properties: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("source", "target", "type", mode="before")
    @classmethod
    def _strip_non_empty(cls, value: Any) -> str:
        text = str(value).strip() if value is not None else ""
        if not text:
            raise ValueError("must be a non-empty string")
        return text


class ExtractionGraphPayload(BaseModel):
    """Canonical structured extraction payload consumed by Memgraph writers."""

    model_config = ConfigDict(extra="forbid")

    nodes: List[ExtractionNode] = Field(default_factory=list)
    edges: List[ExtractionEdge] = Field(default_factory=list)


_CYPHER_IDENT_RE = re.compile(r":\s*([A-Za-z_][A-Za-z0-9_]*)")


def _strip_think_and_fences(raw_text: str) -> str:
    """Strip <think> reasoning blocks and markdown code fences from LLM output."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"```(?:json|cypher)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned)
    return cleaned.strip()


def _skip_quoted_string(text: str, pos: int) -> int:
    """Advance index past a single- or double-quoted string literal."""
    quote = text[pos]
    idx = pos + 1
    n = len(text)
    while idx < n:
        ch = text[idx]
        if ch == "\\" and idx + 1 < n:
            idx += 2
            continue
        if ch == quote:
            return idx + 1
        idx += 1
    raise ValueError("Unterminated string literal in Cypher property map")


def _find_balanced_block(text: str, start: int, open_ch: str, close_ch: str) -> Tuple[str, int]:
    """Extract content inside balanced delimiters respecting quoted strings."""
    if start >= len(text) or text[start] != open_ch:
        raise ValueError(f"Expected '{open_ch}' at index {start}")
    depth = 0
    idx = start
    n = len(text)
    while idx < n:
        ch = text[idx]
        if ch in ('"', "'"):
            idx = _skip_quoted_string(text, idx)
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start + 1 : idx], idx + 1
        idx += 1
    raise ValueError(f"Unbalanced '{open_ch}' in Cypher output")


def _parse_cypher_scalar(raw_val: str) -> Any:
    """Decode a Cypher literal value into Python primitives."""
    val = raw_val.strip()
    if not val:
        raise ValueError("Empty property value in Cypher map")
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        if val.startswith("'"):
            inner = val[1:-1].replace("\\'", "'").replace('"', '\\"')
            return json.loads(f'"{inner}"')
        return json.loads(val)
    lower = val.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower == "null":
        return None
    try:
        return int(val)
    except ValueError:
        return float(val)


def _parse_cypher_prop_map(raw_map: str) -> Dict[str, Any]:
    """Parse Cypher '{key: value, ...}' map body cleanly across quoted commas/colons."""
    props: Dict[str, Any] = {}
    n = len(raw_map)
    idx = 0
    while idx < n:
        while idx < n and raw_map[idx] in " \t\r\n,":
            idx += 1
        if idx >= n:
            break
        key_start = idx
        while idx < n and raw_map[idx] != ":":
            idx += 1
        if idx >= n:
            raise ValueError("Missing ':' in Cypher property map")
        key = raw_map[key_start:idx].strip().strip('"').strip("'")
        if not key:
            raise ValueError("Empty property key in Cypher map")
        idx += 1  # skip ':'
        while idx < n and raw_map[idx] in " \t\r\n":
            idx += 1
        val_start = idx
        while idx < n and raw_map[idx] != ",":
            if raw_map[idx] in ('"', "'"):
                idx = _skip_quoted_string(raw_map, idx)
                continue
            idx += 1
        props[key] = _parse_cypher_scalar(raw_map[val_start:idx])
        idx += 1  # skip ','
    return props


def _scan_cypher_entity(text: str, pos: int, open_ch: str, close_ch: str) -> Tuple[str, Dict[str, Any], int]:
    """Parse '(:Label {props})' or '[:REL {props}]' returning (type, props, next_pos)."""
    inner, next_pos = _find_balanced_block(text, pos, open_ch, close_ch)
    match = _CYPHER_IDENT_RE.match(inner.strip())
    if not match:
        raise ValueError(f"Missing Cypher label/type in '{inner}'")
    entity_type = match.group(1)
    rest = inner.strip()[match.end() :].strip()
    if not rest:
        return entity_type, {}, next_pos
    prop_body, after_brace = _find_balanced_block(rest, 0, "{", "}")
    if rest[after_brace:].strip():
        raise ValueError(f"Unexpected trailing content in Cypher entity: {rest[after_brace:]}")
    return entity_type, _parse_cypher_prop_map(prop_body), next_pos


def _parse_opencypher_triples(cleaned: str) -> Optional[Dict[str, Any]]:
    """Parse fine-tuned YarnBall OpenCypher triple clauses into structured nodes/edges."""
    n = len(cleaned)
    idx = 0
    nodes_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    triple_count = 0

    try:
        while idx < n:
            while idx < n and cleaned[idx] in " \t\r\n;":
                idx += 1
            if idx >= n:
                break

            src_label, src_props, idx = _scan_cypher_entity(cleaned, idx, "(", ")")
            while idx < n and cleaned[idx] in " \t\r\n":
                idx += 1
            if idx >= n or cleaned[idx] != "-":
                return None
            idx += 1
            while idx < n and cleaned[idx] in " \t\r\n":
                idx += 1

            rel_type, rel_props, idx = _scan_cypher_entity(cleaned, idx, "[", "]")
            while idx < n and cleaned[idx] in " \t\r\n":
                idx += 1
            if cleaned[idx : idx + 2] != "->":
                return None
            idx += 2
            while idx < n and cleaned[idx] in " \t\r\n":
                idx += 1

            dst_label, dst_props, idx = _scan_cypher_entity(cleaned, idx, "(", ")")

            for label, props in ((src_label, src_props), (dst_label, dst_props)):
                node_id = str(props.get("id") or props.get("name") or props.get("ticker") or "").strip()
                if not node_id:
                    return None
                node_props = {k: v for k, v in props.items() if k != "id"}
                key = (label, node_id)
                if key not in nodes_by_key:
                    nodes_by_key[key] = {"id": node_id, "type": label, "properties": node_props}
                else:
                    nodes_by_key[key]["properties"].update(node_props)

            src_id = str(src_props.get("id") or src_props.get("name") or src_props.get("ticker")).strip()
            dst_id = str(dst_props.get("id") or dst_props.get("name") or dst_props.get("ticker")).strip()
            edges.append({"source": src_id, "target": dst_id, "type": rel_type, "properties": rel_props})
            triple_count += 1
    except ValueError:
        return None

    if triple_count == 0:
        return None
    return {"nodes": list(nodes_by_key.values()), "edges": edges}


def _extract_balanced_json_payload(cleaned: str) -> Optional[Dict[str, Any]]:
    """Scan balanced JSON objects via JSONDecoder.raw_decode and validate ExtractionGraphPayload."""
    decoder = json.JSONDecoder()

    def _validate(candidate: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(candidate, dict):
            return None
        if "nodes" not in candidate and "edges" not in candidate:
            return None
        try:
            validated = ExtractionGraphPayload.model_validate(candidate)
            return validated.model_dump()
        except ValidationError:
            return None

    try:
        direct = _validate(json.loads(cleaned))
        if direct is not None:
            return direct
    except Exception:
        pass

    pos = 0
    while True:
        brace_idx = cleaned.find("{", pos)
        if brace_idx == -1:
            break
        try:
            obj, end_idx = decoder.raw_decode(cleaned, brace_idx)
            validated = _validate(obj)
            if validated is not None:
                return validated
            pos = end_idx
        except json.JSONDecodeError:
            pos = brace_idx + 1
    return None


def parse_extraction_payload(raw_text: str) -> Dict[str, Any]:
    """Parse and Pydantic-validate LLM output (JSON, OpenCypher triples, or '(none)')."""
    if not raw_text or not raw_text.strip():
        raise GraphExtractionError("Empty LLM extraction output", error_stage="EXTRACTION_PARSE_ERROR")

    cleaned = _strip_think_and_fences(raw_text)
    if not cleaned:
        raise GraphExtractionError(
            "LLM extraction output contained no payload outside reasoning/fences",
            error_stage="EXTRACTION_PARSE_ERROR",
        )

    if cleaned == "(none)":
        return {"nodes": [], "edges": []}

    json_payload = _extract_balanced_json_payload(cleaned)
    if json_payload is not None:
        return json_payload

    cypher_payload = _parse_opencypher_triples(cleaned)
    if cypher_payload is not None:
        try:
            return ExtractionGraphPayload.model_validate(cypher_payload).model_dump()
        except ValidationError as exc:
            raise GraphExtractionError(
                f"Schema validation failed for OpenCypher extraction: {exc}",
                error_stage="EXTRACTION_PARSE_ERROR",
            ) from exc

    raise GraphExtractionError(
        f"Unparseable or schema-invalid LLM extraction output: {cleaned[:200]}",
        error_stage="EXTRACTION_PARSE_ERROR",
    )


def _clean_json_response(raw_text: str) -> dict:
    """Parse and validate LLM output (supports balanced JSON, OpenCypher triples, and '(none)')."""
    return parse_extraction_payload(raw_text)


def _quarantine_extraction_failure(
    *,
    task_id: str,
    prompt: str,
    raw_completion: str,
    error_stage: str,
    error_message: str,
    source_id: Optional[str] = None,
    rejection_queue: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist failed extraction telemetry to RuntimeRejectionQueue without masking exceptions."""
    try:
        queue = rejection_queue
        if queue is None:
            from sft_pipeline.rejection_queue import RuntimeRejectionQueue

            queue = RuntimeRejectionQueue()
        merged_meta = dict(metadata or {})
        if source_id:
            merged_meta["source_id"] = source_id
        queue.record_failure(
            task_id=task_id,
            prompt=prompt,
            completion=raw_completion,
            error_stage=error_stage,
            error_message=error_message,
            metadata=merged_meta,
            sample_id=source_id,
        )
    except Exception as exc:
        logger.warning("Failed to write extraction failure to RuntimeRejectionQueue: %s", exc)


def run_structured_extraction(
    *,
    prompt: str,
    task: str,
    system_prompt: str,
    model_client: Optional["YarnBallModelClient"] = None,
    rejection_queue: Optional[Any] = None,
    source_id: Optional[str] = None,
    timeout_sec: float = 60.0,
) -> Dict[str, Any]:
    """Execute YarnBallModelClient inference, validate output, and quarantine failures."""
    if model_client is None:
        from sft_pipeline.serving.model_client import YarnBallModelClient

        client = YarnBallModelClient(timeout_sec=timeout_sec)
    else:
        client = model_client

    resp = client.generate(
        prompt=prompt,
        task=task,
        system_prompt=system_prompt,
        temperature=0.0,
    )
    if resp.error:
        logger.error("YarnBall extraction request failed (%s): %s", task, resp.error)
        _quarantine_extraction_failure(
            task_id=task,
            prompt=prompt,
            raw_completion=resp.raw_text or "",
            error_stage="LLM_UNAVAILABLE",
            error_message=resp.error,
            source_id=source_id,
            rejection_queue=rejection_queue,
        )
        raise GraphExtractionError(resp.error, error_stage="LLM_UNAVAILABLE")

    try:
        return parse_extraction_payload(resp.final_output)
    except GraphExtractionError as exc:
        logger.warning("Quarantining invalid extraction output (%s): %s", task, exc)
        _quarantine_extraction_failure(
            task_id=task,
            prompt=prompt,
            raw_completion=resp.raw_text or resp.final_output or "",
            error_stage=exc.error_stage,
            error_message=str(exc),
            source_id=source_id,
            rejection_queue=rejection_queue,
            metadata={"final_output_preview": (resp.final_output or "")[:500]},
        )
        raise


def _extract_entities_via_ollama(
    text: str,
    model_client: Optional["YarnBallModelClient"] = None,
    rejection_queue: Optional[Any] = None,
    source_id: Optional[str] = None,
) -> dict:
    """Query fine-tuned YarnBall model via YarnBallModelClient for entity extraction."""
    return run_structured_extraction(
        prompt=f"Article:\n{text}\n\nJSON:",
        task="EXTRACT_NEWS_EVENT",
        system_prompt=SYSTEM_EXTRACTION_PROMPT,
        model_client=model_client,
        rejection_queue=rejection_queue,
        source_id=source_id,
    )


def extract_entities(
    text: str,
    model_client: Optional["YarnBallModelClient"] = None,
    rejection_queue: Optional[Any] = None,
    source_id: Optional[str] = None,
) -> dict:
    """Extract entities and relationships from text using YarnBallModelClient."""
    if not text or not text.strip():
        return {"nodes": [], "edges": []}
    return _extract_entities_via_ollama(
        text,
        model_client=model_client,
        rejection_queue=rejection_queue,
        source_id=source_id,
    )


def store_graph_entities(
    source_hash: str,
    text: str,
    graph_data: Optional[Dict[str, Any]] = None,
    driver: Optional[Any] = None,
    rejection_queue: Optional[Any] = None,
) -> None:
    """Extract entities, enforce live ontology gate, and persist them to Memgraph."""
    if graph_data is None:
        graph_data = extract_entities(text, source_id=source_hash, rejection_queue=rejection_queue)

    gated = validate_graph_payload(
        graph_data,
        source_id=source_hash,
        rejection_queue=rejection_queue,
        strict=True,
        default_node_type="Entity",
        source_text=text,
    )
    nodes: List[Dict[str, Any]] = gated.get("nodes", [])
    edges: List[Dict[str, Any]] = gated.get("edges", [])

    if not nodes and not edges:
        return

    active_driver = driver if driver is not None else get_memgraph_driver()

    with active_driver.session() as session:
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
        from contagion.parameters import resolve_edge_cascade_parameters

        for edge in edges:
            source = str(edge.get("source", "")).strip()
            target = str(edge.get("target", "")).strip()
            if not source or not target:
                continue

            raw_type = str(edge.get("type", "RELATED_TO")).strip().upper()
            rel_type = "".join(c for c in raw_type if c.isalnum() or c == "_")[:MAX_REL_TYPE_LENGTH] or "RELATED_TO"
            props = dict(edge.get("properties", {}) or {})
            resolved_params = resolve_edge_cascade_parameters({**edge, "rel_type": rel_type})
            props.update(resolved_params)

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
        from embedding.embedder import InvalidEmbeddingError

        try:
            from .embedding_store import store_node_embeddings
            store_node_embeddings(nodes, source_hash=source_hash)
        except InvalidEmbeddingError:
            raise
        except Exception as exc:
            logger.warning("Failed to store node embeddings for %s: %s", source_hash, exc)
