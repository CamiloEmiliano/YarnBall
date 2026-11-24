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
    """Canonical structured extraction payload consumed by Memgraph writers (ADR-00029 / ADR-00035)."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: Optional[str] = None
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


def parse_extraction_payload(raw_text: str, *, chunk_id: Optional[str] = None) -> Dict[str, Any]:
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
        out_empty: Dict[str, Any] = {"nodes": [], "edges": []}
        if chunk_id is not None:
            out_empty["chunk_id"] = chunk_id
        return out_empty

    json_payload = _extract_balanced_json_payload(cleaned)
    if json_payload is not None:
        if chunk_id is not None:
            json_payload["chunk_id"] = chunk_id
        elif json_payload.get("chunk_id") is None:
            json_payload.pop("chunk_id", None)
        return json_payload

    cypher_payload = _parse_opencypher_triples(cleaned)
    if cypher_payload is not None:
        try:
            validated = ExtractionGraphPayload.model_validate(cypher_payload).model_dump()
            if chunk_id is not None:
                validated["chunk_id"] = chunk_id
            elif validated.get("chunk_id") is None:
                validated.pop("chunk_id", None)
            return validated
        except ValidationError as exc:
            raise GraphExtractionError(
                f"Schema validation failed for OpenCypher extraction: {exc}",
                error_stage="EXTRACTION_PARSE_ERROR",
            ) from exc

    raise GraphExtractionError(
        f"Unparseable or schema-invalid LLM extraction output: {cleaned[:200]}",
        error_stage="EXTRACTION_PARSE_ERROR",
    )


def merge_chunk_extraction_payloads(chunk_payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge per-chunk extraction payloads: dedupe edges, union evidence_chunk_ids, keep highest-confidence provenance."""
    from contagion.parameters import PROVENANCE_CONFIDENCE_RANK, resolve_edge_cascade_parameters

    nodes_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    edges_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    edge_rank_by_key: Dict[Tuple[str, str, str], Tuple[int, float]] = {}

    for payload in chunk_payloads:
        chunk_id = payload.get("chunk_id")
        for node in payload.get("nodes", []) or []:
            nid = str(node.get("id", "")).strip()
            ntype = str(node.get("type", "Entity")).strip() or "Entity"
            if not nid:
                continue
            nkey = (ntype.lower(), nid.lower())
            nprops = dict(node.get("properties") or {})
            if nkey not in nodes_by_key:
                nodes_by_key[nkey] = {"id": nid, "type": ntype, "properties": nprops}
            else:
                merged_nprops = dict(nodes_by_key[nkey]["properties"])
                merged_nprops.update({k: v for k, v in nprops.items() if v is not None})
                nodes_by_key[nkey]["properties"] = merged_nprops

        for edge in payload.get("edges", []) or []:
            src = str(edge.get("source", "")).strip()
            tgt = str(edge.get("target", "")).strip()
            rel = str(edge.get("type", "RELATED_TO")).strip().upper() or "RELATED_TO"
            if not src or not tgt:
                continue
            ekey = (src.lower(), rel, tgt.lower())
            eprops = dict(edge.get("properties") or {})
            cids = set(str(c).strip() for c in (eprops.get("evidence_chunk_ids") or []) if str(c).strip())
            if isinstance(chunk_id, str) and chunk_id.strip():
                cids.add(chunk_id.strip())
            eprops["evidence_chunk_ids"] = sorted(cids)

            resolved = resolve_edge_cascade_parameters({**edge, "properties": eprops, "rel_type": rel})
            prov_rank = PROVENANCE_CONFIDENCE_RANK.get(str(resolved["param_provenance"]), 0)
            conf_raw = eprops.get("confidence", 0.0)
            try:
                conf_score = float(conf_raw) if conf_raw is not None else 0.0
            except (TypeError, ValueError):
                conf_score = 0.0
            candidate_rank = (prov_rank, conf_score)

            if ekey not in edges_by_key:
                edges_by_key[ekey] = {
                    "source": src,
                    "target": tgt,
                    "type": rel,
                    "properties": eprops,
                }
                edge_rank_by_key[ekey] = candidate_rank
            else:
                existing = edges_by_key[ekey]
                existing_props = dict(existing.get("properties") or {})
                unioned_cids = sorted(
                    set(existing_props.get("evidence_chunk_ids") or []).union(cids)
                )
                prev_pass = existing_props.get("extraction_pass")
                new_pass = eprops.get("extraction_pass")
                if candidate_rank > edge_rank_by_key[ekey]:
                    merged_props = {**existing_props, **eprops}
                    edge_rank_by_key[ekey] = candidate_rank
                else:
                    merged_props = {**eprops, **existing_props}
                merged_props["evidence_chunk_ids"] = unioned_cids
                if prev_pass is not None and new_pass is not None:
                    merged_props["extraction_pass"] = min(int(prev_pass), int(new_pass))
                existing["properties"] = merged_props

    return {
        "nodes": list(nodes_by_key.values()),
        "edges": list(edges_by_key.values()),
    }


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


def run_structured_extraction_with_telemetry(
    *,
    prompt: str,
    task: str,
    system_prompt: str,
    model_client: Optional["YarnBallModelClient"] = None,
    rejection_queue: Optional[Any] = None,
    source_id: Optional[str] = None,
    chunk_id: Optional[str] = None,
    timeout_sec: float = 60.0,
    quarantine_on_error: bool = True,
) -> Tuple[Dict[str, Any], Any]:
    """Execute YarnBallModelClient inference, validate output, and return `(payload, ModelResponse)`."""
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
        if quarantine_on_error:
            _quarantine_extraction_failure(
                task_id=task,
                prompt=prompt,
                raw_completion=resp.raw_text or "",
                error_stage="LLM_UNAVAILABLE",
                error_message=resp.error,
                source_id=source_id,
                rejection_queue=rejection_queue,
                metadata={"chunk_id": chunk_id} if chunk_id else None,
            )
        raise GraphExtractionError(resp.error, error_stage="LLM_UNAVAILABLE")

    try:
        payload = parse_extraction_payload(resp.final_output, chunk_id=chunk_id)
        return payload, resp
    except GraphExtractionError as exc:
        logger.warning("Quarantining invalid extraction output (%s): %s", task, exc)
        if quarantine_on_error:
            meta: Dict[str, Any] = {"final_output_preview": (resp.final_output or "")[:500]}
            if chunk_id:
                meta["chunk_id"] = chunk_id
            _quarantine_extraction_failure(
                task_id=task,
                prompt=prompt,
                raw_completion=resp.raw_text or resp.final_output or "",
                error_stage=exc.error_stage,
                error_message=str(exc),
                source_id=source_id,
                rejection_queue=rejection_queue,
                metadata=meta,
            )
        raise


def run_structured_extraction(
    *,
    prompt: str,
    task: str,
    system_prompt: str,
    model_client: Optional["YarnBallModelClient"] = None,
    rejection_queue: Optional[Any] = None,
    source_id: Optional[str] = None,
    chunk_id: Optional[str] = None,
    timeout_sec: float = 60.0,
) -> Dict[str, Any]:
    """Execute YarnBallModelClient inference, validate output, and quarantine failures."""
    payload, _ = run_structured_extraction_with_telemetry(
        prompt=prompt,
        task=task,
        system_prompt=system_prompt,
        model_client=model_client,
        rejection_queue=rejection_queue,
        source_id=source_id,
        chunk_id=chunk_id,
        timeout_sec=timeout_sec,
    )
    return payload


def _extract_entities_via_ollama(
    text: str,
    model_client: Optional["YarnBallModelClient"] = None,
    rejection_queue: Optional[Any] = None,
    source_id: Optional[str] = None,
    chunk_id: Optional[str] = None,
) -> dict:
    """Query fine-tuned YarnBall model via YarnBallModelClient for entity extraction."""
    return run_structured_extraction(
        prompt=f"Article:\n{text}\n\nJSON:",
        task="EXTRACT_NEWS_EVENT",
        system_prompt=SYSTEM_EXTRACTION_PROMPT,
        model_client=model_client,
        rejection_queue=rejection_queue,
        source_id=source_id,
        chunk_id=chunk_id,
    )


def extract_entities(
    text: str,
    model_client: Optional["YarnBallModelClient"] = None,
    rejection_queue: Optional[Any] = None,
    source_id: Optional[str] = None,
    *,
    max_tokens: int = 800,
    overlap_tokens: int = 120,
) -> dict:
    """Segment news articles exceeding token budget, extract per chunk, and merge (ADR-00035)."""
    if not text or not text.strip():
        return {"nodes": [], "edges": []}

    from ingestion.segmentation.chunker import segment_news_article

    chunks = segment_news_article(
        text,
        source_hash=source_id or "news",
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    if len(chunks) <= 1:
        cid = chunks[0].chunk_id if chunks else None
        payload = _extract_entities_via_ollama(
            text,
            model_client=model_client,
            rejection_queue=rejection_queue,
            source_id=source_id,
            chunk_id=cid,
        )
        if cid:
            for edge in payload.get("edges", []):
                eprops = dict(edge.get("properties") or {})
                eprops["evidence_chunk_ids"] = sorted({*(eprops.get("evidence_chunk_ids") or []), cid})
                edge["properties"] = eprops
        payload.pop("chunk_id", None)
        return payload

    chunk_payloads: List[Dict[str, Any]] = []
    first_error: Optional[GraphExtractionError] = None
    for chunk in chunks:
        try:
            raw_payload = _extract_entities_via_ollama(
                chunk.text,
                model_client=model_client,
                rejection_queue=rejection_queue,
                source_id=source_id or chunk.chunk_id,
                chunk_id=chunk.chunk_id,
            )
            gated_chunk = validate_graph_payload(
                raw_payload,
                source_id=chunk.chunk_id,
                rejection_queue=rejection_queue,
                strict=True,
                default_node_type="Entity",
                source_text=chunk.text,
            )
            chunk_payloads.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "nodes": gated_chunk.get("nodes", []),
                    "edges": gated_chunk.get("edges", []),
                }
            )
        except GraphExtractionError as exc:
            if first_error is None:
                first_error = exc
            continue

    if not chunk_payloads and first_error is not None:
        raise first_error

    return merge_chunk_extraction_payloads(chunk_payloads)


_FORBIDDEN_MEMGRAPH_PROSE_KEYS = frozenset({"raw_text", "full_text", "chunk_text", "evidence_text", "body_text"})


def store_graph_entities(
    source_hash: str,
    text: str,
    graph_data: Optional[Dict[str, Any]] = None,
    driver: Optional[Any] = None,
    rejection_queue: Optional[Any] = None,
    published_at: Optional[str] = None,
    source_type: str = "news",
    linker: Optional[Any] = None,
    pg_conn: Optional[Any] = None,
) -> None:
    """Extract entities per chunk, enforce live ontology gate, persist text units in Postgres, and link thin Memgraph pointers (ADR-00028 / ADR-00035)."""
    from ingestion.segmentation.chunker import segment_news_article
    from knowledge_graph.chunk_store import (
        DocumentChunkRecord,
        make_chunk_id,
        store_chunk_entity_links,
        store_document_chunks,
    )
    from retrieval.temporal import actionable_session

    clean_source_type = str(source_type or "news").strip().lower() or "news"
    session_date, known_at_utc, known_at_imputed = actionable_session(published_at)
    valid_from: Optional[str] = session_date.isoformat() if session_date else None
    known_at_iso: Optional[str] = known_at_utc.isoformat() if known_at_utc else None
    is_temporal_unanchored = valid_from is None

    news_chunks = segment_news_article(
        text,
        source_hash=source_hash,
        published_at=published_at,
        max_tokens=800,
        overlap_tokens=120,
    )
    default_chunk_ids = [c.chunk_id for c in news_chunks] if news_chunks else (
        [make_chunk_id(source_hash, "body", 1)] if (text and text.strip()) else []
    )
    chunk_records = [
        DocumentChunkRecord(
            chunk_id=c.chunk_id,
            doc_id=source_hash,
            source_type=clean_source_type,
            cik=None,
            section=c.section_name,
            char_start=c.start_char,
            char_end=c.end_char,
            text=c.text,
            valid_from=valid_from,
            known_at=known_at_iso,
        )
        for c in news_chunks
    ]

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

    from ingestion.sec.parser import EdgarEntityLinker, normalize_company_name

    active_linker = linker if linker is not None else EdgarEntityLinker(pg_conn=pg_conn)
    canonical_id_map: Dict[str, str] = {}
    linked_nodes_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for node in nodes:
        raw_id = str(node.get("id", "")).strip()
        if not raw_id:
            continue
        raw_type = str(node.get("type", "Entity")).strip()
        label = "".join(c for c in raw_type if c.isalnum())[:MAX_LABEL_LENGTH] or "Entity"
        props = {
            k: v
            for k, v in dict(node.get("properties", {}) or {}).items()
            if k not in _FORBIDDEN_MEMGRAPH_PROSE_KEYS
        }
        canonical_id = raw_id

        if label in ("Company", "Organization", "Entity"):
            resolved = active_linker.link_entity(
                name=raw_id,
                ticker=props.get("ticker"),
                event_date=str(published_at)[:10] if published_at else None,
            )
            if resolved and resolved.get("company_name"):
                canonical_id = str(resolved["company_name"]).strip()
                if resolved.get("ticker") and not props.get("ticker"):
                    props["ticker"] = resolved["ticker"]
                if resolved.get("cik") and not props.get("cik"):
                    props["cik"] = resolved["cik"]

        canonical_id_map[raw_id] = canonical_id
        norm_key = normalize_company_name(raw_id)
        if norm_key and norm_key not in canonical_id_map:
            canonical_id_map[norm_key] = canonical_id

        dedup_key = (label, canonical_id)
        if dedup_key not in linked_nodes_by_key:
            linked_nodes_by_key[dedup_key] = {
                "id": canonical_id,
                "label": label,
                "properties": props,
            }
        else:
            existing_props = linked_nodes_by_key[dedup_key]["properties"]
            merged_props = {**existing_props, **props}
            linked_nodes_by_key[dedup_key]["properties"] = merged_props

    # Persist text units and entity links to PostgreSQL (thin Memgraph invariant #2)
    chunk_entity_links: List[Tuple[str, str, Optional[str]]] = []
    for cid in default_chunk_ids:
        for node_entry in linked_nodes_by_key.values():
            chunk_entity_links.append((cid, node_entry["id"], valid_from))
    if chunk_records:
        try:
            store_document_chunks(chunk_records, conn=pg_conn)
            if chunk_entity_links:
                store_chunk_entity_links(chunk_entity_links, conn=pg_conn)
        except Exception as exc:
            logger.debug("PostgreSQL chunk persistence skipped/unavailable for %s: %s", source_hash, exc)

    active_driver = driver if driver is not None else get_memgraph_driver()

    with active_driver.session() as session:
        # 1. Insert/Merge Article News Node with published_at provenance (no raw prose in Memgraph)
        session.run(
            "MERGE (n:News {id: $source_hash}) SET n.updated_at = datetime(), n.published_at = $published_at, n.source_type = $source_type",
            source_hash=source_hash,
            published_at=published_at,
            source_type=clean_source_type,
        )

        # 2. Insert/Merge Canonical Entity Nodes
        for node_entry in linked_nodes_by_key.values():
            node_id = node_entry["id"]
            label = node_entry["label"]
            props = node_entry["properties"]

            query = f"MERGE (n:{label} {{id: $id}}) SET n += $props"
            session.run(query, id=node_id, props=props)

            mention_query = f"MATCH (news:News {{id: $source_hash}}), (e:{label} {{id: $id}}) MERGE (news)-[:MENTIONS]->(e)"
            session.run(mention_query, source_hash=source_hash, id=node_id)

        # 3. Insert/Merge Relationship Edges with compact evidence_chunk_ids pointers (ADR-00035)
        from contagion.parameters import resolve_edge_cascade_parameters

        for edge in edges:
            source_raw = str(edge.get("source", "")).strip()
            target_raw = str(edge.get("target", "")).strip()
            if not source_raw or not target_raw:
                continue

            source = canonical_id_map.get(source_raw) or canonical_id_map.get(normalize_company_name(source_raw))
            if not source:
                res_s = active_linker.link_entity(name=source_raw)
                source = str(res_s["company_name"]).strip() if res_s and res_s.get("company_name") else source_raw

            target = canonical_id_map.get(target_raw) or canonical_id_map.get(normalize_company_name(target_raw))
            if not target:
                res_t = active_linker.link_entity(name=target_raw)
                target = str(res_t["company_name"]).strip() if res_t and res_t.get("company_name") else target_raw

            raw_type = str(edge.get("type", "RELATED_TO")).strip().upper()
            rel_type = "".join(c for c in raw_type if c.isalnum() or c == "_")[:MAX_REL_TYPE_LENGTH] or "RELATED_TO"
            props = {
                k: v
                for k, v in dict(edge.get("properties", {}) or {}).items()
                if k not in _FORBIDDEN_MEMGRAPH_PROSE_KEYS
            }
            resolved_params = resolve_edge_cascade_parameters({**edge, "properties": props, "rel_type": rel_type})
            props.update(resolved_params)

            edge_cids = list(props.get("evidence_chunk_ids") or [])
            if not edge_cids and default_chunk_ids:
                edge_cids = list(default_chunk_ids)
            edge_cids = sorted({str(c).strip() for c in edge_cids if str(c).strip()})
            evidence_status = "chunked" if edge_cids else "legacy_unchunked"
            props["evidence_chunk_ids"] = edge_cids
            props["evidence_status"] = evidence_status

            query = f"""
            MATCH (s {{id: $source}})
            MATCH (t {{id: $target}})
            MERGE (s)-[r:{rel_type}]->(t)
            SET r += $props,
                r.source_hash = $source_hash,
                r.source_type = $source_type,
                r.evidence_chunk_ids = $evidence_chunk_ids,
                r.evidence_status = $evidence_status,
                r.valid_from = coalesce(r.valid_from, $valid_from),
                r.known_at = coalesce(r.known_at, $known_at),
                r.known_at_imputed = coalesce(r.known_at_imputed, $known_at_imputed),
                r.last_affirmed_at = $valid_from,
                r.is_current = true,
                r.is_temporal_unanchored = (coalesce(r.valid_from, $valid_from) IS NULL)
            """
            session.run(
                query,
                source=source,
                target=target,
                props=props,
                source_hash=source_hash,
                source_type=clean_source_type,
                evidence_chunk_ids=edge_cids,
                evidence_status=evidence_status,
                valid_from=valid_from,
                known_at=known_at_iso,
                known_at_imputed=known_at_imputed,
                is_temporal_unanchored=is_temporal_unanchored,
            )

    logger.info(
        "Stored %d canonical nodes and %d edges in Memgraph for source_hash %s",
        len(linked_nodes_by_key),
        len(edges),
        source_hash,
    )

    # 4. Persist node embeddings to PGVECTOR
    if linked_nodes_by_key:
        from embedding.embedder import InvalidEmbeddingError

        embed_nodes = [
            {"id": n["id"], "type": n["label"], "properties": n["properties"]}
            for n in linked_nodes_by_key.values()
        ]
        try:
            from .embedding_store import store_node_embeddings
            store_node_embeddings(
                embed_nodes,
                source_hash=source_hash,
                valid_from=valid_from,
                known_at=known_at_iso,
            )
        except InvalidEmbeddingError:
            raise
        except Exception as exc:
            logger.warning("Failed to store node embeddings for %s: %s", source_hash, exc)
