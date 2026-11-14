# ADR-00029: Structured LLM Extraction Outputs & Failure Quarantine Queue

- **ADR ID**: ADR-00029
- **Filename**: `ADR-00029-structured-llm-extraction-and-quarantine.md`
- **Date**: 2026-11-14 19:25 UTC
- **Status**: PROPOSED
- **Target Subsystem**: `knowledge_graph` / `streaming` / `ingestion`
- **Linked Commits**: Pending acceptance

---

## 1. Context & Problem Statement

Inspecting [`knowledge_graph/graph_store.py`](../../knowledge_graph/graph_store.py) (`_clean_json_response`, `_extract_entities_via_ollama`, `store_graph_entities`) and [`knowledge_graph/memgraph_consumer.py`](../../knowledge_graph/memgraph_consumer.py) revealed fragile handling of LLM extraction outputs:

1. **Greedy Brace Regex Parsing**: `_clean_json_response` falls back to `re.search(r"(\{.*\})", raw_text, re.DOTALL)`. When reasoning models (`qwen3:8b`, `yarnball-qwen:7b`) emit `<think>...</think>` reasoning blocks or surrounding prose containing curly braces `{...}`, the greedy match spans from the first `{` to the last `}` across non-JSON text, failing `json.loads(...)`.
2. **Silent Data Loss on Parse / Model Failures**: When `_clean_json_response` fails or Ollama raises an HTTP/timeout error, `_extract_entities_via_ollama` logs a warning/error and returns `{"nodes": [], "edges": []}`. Because `store_graph_entities` returns `None` cleanly on empty graphs, `memgraph_consumer.py` logs `"Extracted and stored Memgraph entities"` / returns `True` and **commits the Kafka offset**—silently dropping articles without routing them to `send_to_dlt` or a local quarantine file.

---

## 2. Decision Drivers & Constraints

1. **Reasoning-Block Safe JSON Parsing**: Models emitting `<think>...</think>` trace blocks must have reasoning stripped and isolated cleanly before JSON schema validation.
2. **Explicit Failure Propagation Over Silent Success**: Unreachable Ollama endpoints, HTTP 5xx/timeouts, and unparseable JSON responses must raise or return structured failure status so Kafka consumers route the payload to DLT/quarantine rather than committing lost work.
3. **Schema-Validated Extraction Payloads**: Extracted `{nodes, edges}` structures must satisfy typed field checks prior to Cypher execution.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Retain Silent `{"nodes": [], "edges": []}` Return on Exceptions**:
  - *Rationale for Rejection*: Makes genuine outages (e.g., Ollama down or prompt breakage) indistinguishable from empty-entity articles and causes irreversible offset commits in Kafka.
- **Alternative B: Disable Reasoning / `<think>` Tokens Globally via Prompting Alone**:
  - *Rationale for Rejection*: Reasoning models frequently emit `<think>` blocks or markdown fences despite prompt instructions; parser resilience and structured response constraints are both required.

---

## 4. Decision Outcome

Harden LLM extraction parsing and failure accounting in [`knowledge_graph/graph_store.py`](../../knowledge_graph/graph_store.py) and [`knowledge_graph/memgraph_consumer.py`](../../knowledge_graph/memgraph_consumer.py):

1. **`<think>`-Aware Balanced JSON Extractor & Pydantic Schema (`ExtractionGraphPayload`)**:
   - Strip `<think>...</think>` blocks and code fences first.
   - Extract candidate JSON via balanced brace scanning (`json.JSONDecoder.raw_decode`) rather than greedy regex across the whole output, then validate against a Pydantic model (`ExtractionNode`, `ExtractionEdge`, `ExtractionGraphPayload`).
   - Prefer Ollama structured JSON schema enforcement (`format: schema` or `response_format`) where supported.
2. **Distinguish Empty Valid Extraction From Extraction Failure**:
   - Raise `GraphExtractionError` on HTTP/connection/timeout errors or unparseable model output (and optionally record raw response snippets under `data/rejection_queue/`).
   - Update `memgraph_consumer.py` (`process_graph_message`) so `GraphExtractionError` triggers `send_to_dlt(..., reason="llm_extraction_failed")` rather than silently committing as success.

---

## 5. Invariants & Guardrails

1. **Zero Silent Offset Commits on Broken LLM Output**: Connection failures and malformed JSON outputs MUST NOT be reported as successful graph extraction.
2. **`<think>` Trace Isolation**: Reasoning traces MUST be stripped cleanly before `json.loads` / `raw_decode` scanning.
3. **Quarantine Auditability**: Unparseable responses routed to DLT or `data/rejection_queue/` MUST retain `source_hash` and error reason for replay after prompt/model fixes.
4. **Zero Absolute Paths Invariant**: All references remain strictly repository-relative (`../../<path>`).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Prevents silent data loss across live Kafka runs (`memgraph_consumer.py`) when Ollama restarts or emits `<think>` blocks with braces.
  - Creates a replayable quarantine queue for broken extractions.
- **Negative / Trade-offs**:
  - Callers of `extract_entities` / `store_graph_entities` in tests must explicitly mock or handle `GraphExtractionError` when simulating LLM failures rather than assuming `{}` empty dict returns.
