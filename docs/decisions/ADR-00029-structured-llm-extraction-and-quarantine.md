# ADR-00029: Structured LLM Extraction Outputs & Failure Quarantine Queue

- **ADR ID**: ADR-00029
- **Filename**: `ADR-00029-structured-llm-extraction-and-quarantine.md`
- **Date**: 2026-11-14 19:25 UTC
- **Status**: ACCEPTED
- **Target Subsystem**: `knowledge_graph` / `streaming` / `ingestion` / `sft_pipeline`
- **Linked Commits**: [`e5d1028`](../../knowledge_graph/graph_store.py), [`e5d1028`](../../knowledge_graph/memgraph_consumer.py), [`e5d1028`](../../ingestion/sec/parser.py), [`e5d1028`](../../sft_pipeline/rejection_queue.py)

---

## 1. Context & Problem Statement

Following [`ADR-00030`](ADR-00030-purge-non-ft-qwen-runtime-dependencies--.md), all runtime graph extraction (`knowledge_graph/graph_store.py`, `ingestion/sec/parser.py`) routes exclusively through [`YarnBallModelClient`](../../sft_pipeline/serving/model_client.py) targeting the fine-tuned `yarnball-qwen:7b` adapter. Inspecting [`knowledge_graph/graph_store.py`](../../knowledge_graph/graph_store.py) (`_clean_json_response`, `_extract_entities_via_ollama`, `store_graph_entities`), [`ingestion/sec/parser.py`](../../ingestion/sec/parser.py) (`extract_sec_relationships`), and [`knowledge_graph/memgraph_consumer.py`](../../knowledge_graph/memgraph_consumer.py) revealed three structural gaps:

1. **Greedy Brace Regex Parsing**: `_clean_json_response` falls back to `re.search(r"(\{.*\})", raw_text, re.DOTALL)`. When a model emits `<think>...</think>` reasoning blocks or surrounding prose containing curly braces `{...}`, the greedy match spans from the first `{` to the last `}` across non-JSON text, failing `json.loads(...)`.
2. **Format Divergence With Fine-Tuned `yarnball-qwen:7b`**: SFT training data (`data/sft/extraction_tasks_val.jsonl`) trains `yarnball-qwen:7b` to emit either `"(none)"` (valid empty extraction) or OpenCypher triples `(:Label {...})-[:REL {...}]->(:Label {...})` in addition to structured JSON payloads in test/schema fixtures. A JSON-only parser rejects valid `"(none)"` and OpenCypher triples.
3. **Silent Data Loss on Parse / Model Failures**: When extraction parsing fails or `YarnBallModelClient` returns `resp.error` (e.g., Ollama down/timeout), `_extract_entities_via_ollama` and `extract_sec_relationships` log an error and return `{"nodes": [], "edges": []}`. Because `store_graph_entities` returns `None` cleanly on empty graphs, `memgraph_consumer.py` treats outages and broken outputs as success, **commits the Kafka offset**, and writes zero records to [`RuntimeRejectionQueue`](../../sft_pipeline/rejection_queue.py) or Kafka DLT.

---

## 2. Decision Drivers & Constraints

1. **Reasoning-Block Safe & Dual-Format Parsing**: Strip `<think>...</think>` trace blocks and markdown fences, accept `"(none)"` as a valid empty extraction, and parse both `<think>`-safe balanced JSON (`ExtractionGraphPayload`) and `yarnball-qwen:7b` OpenCypher triples into a canonical `{nodes, edges}` schema.
2. **Explicit Failure Propagation Over Silent Success**: Unreachable Ollama endpoints (`resp.error`), empty model responses, and unparseable/schema-invalid completions must record a structured failure into `RuntimeRejectionQueue` (`data/rejection_queue/`) and raise `GraphExtractionError` so Kafka consumers route messages to DLT rather than committing lost work.
3. **Resilient Batch Isolation for SEC Filings**: `EdgarWorker` must quarantine failed SEC chunks into `RuntimeRejectionQueue` and continue persisting non-failing filing sections (e.g., Exhibit 21 subsidiaries) rather than aborting the entire 10-K filing.
4. **Zero Circular Imports**: `knowledge_graph/graph_store.py` must import `RuntimeRejectionQueue` / `ErrorStage` lazily inside functions (matching `YarnBallModelClient`) to preserve clean module loading with `sft_pipeline` and `ingestion`.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Retain Silent `{"nodes": [], "edges": []}` Return on Exceptions**:
  - *Rationale for Rejection*: Makes genuine Ollama outages, timeouts, and broken completions indistinguishable from valid `"(none)"` articles and causes irreversible Kafka offset commits.
- **Alternative B: Support Only JSON or Only OpenCypher Parsing**:
  - *Rationale for Rejection*: Fine-tuned `yarnball-qwen:7b` emits OpenCypher triples and `"(none)"` natively per its SFT dataset, while schema-constrained prompts and swapped local Qwen models may emit JSON. Normalizing both into `ExtractionGraphPayload` ensures seamless model swappability.

---

## 4. Decision Outcome

Harden LLM extraction parsing, schema validation, and failure quarantine across [`knowledge_graph/graph_store.py`](../../knowledge_graph/graph_store.py), [`ingestion/sec/parser.py`](../../ingestion/sec/parser.py), [`ingestion/sec/harvester.py`](../../ingestion/sec/harvester.py), [`knowledge_graph/memgraph_consumer.py`](../../knowledge_graph/memgraph_consumer.py), and [`sft_pipeline/rejection_queue.py`](../../sft_pipeline/rejection_queue.py):

1. **`<think>`-Aware Dual-Format Parser & Pydantic Schema (`ExtractionGraphPayload`)**:
   - Define `ExtractionNode`, `ExtractionEdge`, and `ExtractionGraphPayload` Pydantic models (`extra="forbid"` at payload root) and `GraphExtractionError(RuntimeError)` in [`knowledge_graph/graph_store.py`](../../knowledge_graph/graph_store.py).
   - Strip `<think>...</think>` blocks and markdown fences, return `{"nodes": [], "edges": []}` cleanly on explicit `"(none)"`, scan balanced JSON objects via `json.JSONDecoder.raw_decode` (requiring `"nodes"` or `"edges"`), and fall back cleanly to a quote-aware OpenCypher triple parser `(:Label {...})-[:REL {...}]->(:Label {...})`.
2. **Unified Execution & Quarantine Helper (`run_structured_extraction`)**:
   - Route both news extraction (`_extract_entities_via_ollama`) and SEC extraction (`extract_sec_relationships`) through `run_structured_extraction(...)`, which calls `YarnBallModelClient`, rejects `resp.error` (`ErrorStage.LLM_UNAVAILABLE`) and invalid completions (`ErrorStage.EXTRACTION_PARSE_ERROR`), records failed attempts into `RuntimeRejectionQueue` (`data/rejection_queue/rejections_YYYYMMDD.jsonl`) with `source_id`, and raises `GraphExtractionError`.
3. **Kafka DLT & SEC Worker Propagation**:
   - Update [`knowledge_graph/memgraph_consumer.py`](../../knowledge_graph/memgraph_consumer.py) (`dispatch_memgraph_message` / `run_memgraph_consumer`) to catch `GraphExtractionError` and route failed messages to `send_to_dlt(..., reason="llm_extraction_failed")` without reporting extraction success.
   - Update [`ingestion/sec/harvester.py`](../../ingestion/sec/harvester.py) (`EdgarWorker.process_company_10k`) to catch `GraphExtractionError` per extracted 10-K section after quarantine logging so unaffected filing sections remain intact.

---

## 5. Invariants & Guardrails

1. **Zero Silent Offset Commits on Broken LLM Output**: Connection failures (`resp.error`) and malformed/empty LLM outputs MUST raise `GraphExtractionError` instead of returning empty `{"nodes": [], "edges": []}`.
2. **`(none)` Equivalence**: Explicit `"(none)"` from fine-tuned `yarnball-qwen:7b` MUST resolve cleanly to `{"nodes": [], "edges": []}` with zero quarantine records or errors.
3. **`<think>` Trace Isolation**: Reasoning traces MUST be stripped before balanced JSON `raw_decode` or OpenCypher scanning.
4. **Quarantine Auditability**: Unparseable responses and model transport errors MUST persist to `RuntimeRejectionQueue` (`data/rejection_queue/`) with task ID, `source_id`, raw completion, and `error_stage`, alongside DLT emission (`reason="llm_extraction_failed"`).
5. **Zero Absolute Paths Invariant**: All references remain strictly repository-relative (`../../<path>`).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Prevents silent data loss across Kafka consumers (`memgraph_consumer.py`, `streaming/consumer.py`) and captures failed generations for DPO/retraining replay (`RuntimeRejectionQueue`).
  - Bridges fine-tuned OpenCypher/`(none)` completions and JSON schema outputs under one validated contract (`ExtractionGraphPayload`).
- **Negative / Trade-offs**:
  - Callers invoking `extract_entities`, `extract_sec_relationships`, or `store_graph_entities` with broken mock outputs or unreachable Ollama instances receive `GraphExtractionError` instead of `{}` unless handled explicitly.
