# ADR-00030: Purge Non-Fine-Tuned Qwen Runtime Dependencies & Unify on `YarnBallModelClient`

- **ADR ID**: ADR-00030
- **Filename**: `ADR-00030-purge-non-ft-qwen-runtime-dependencies--.md`
- **Date**: 2026-11-14 20:22 UTC
- **Status**: ACCEPTED
- **Target Subsystem**: `sft_pipeline` / `knowledge_graph` / `ingestion` / `retrieval`
- **Linked Commits**: [`47681f4`](../../sft_pipeline/serving/model_client.py), [`47681f4`](../../knowledge_graph/graph_store.py), [`47681f4`](../../ingestion/sec/parser.py), [`47681f4`](../../retrieval/text_to_cql.py), [`47681f4`](../../tests/knowledge_graph/test_model_client_unification.py)

---

## 1. Context & Problem Statement

End-to-end audit of runtime LLM callers uncovered a destructive split-brain between the fine-tuned production model (`yarnball-qwen:7b`, served via [`sft_pipeline/serving/model_client.py`](../../sft_pipeline/serving/model_client.py)) and legacy extraction/Cypher modules that still call pre-fine-tuned `qwen3:8b` over raw HTTP:

1. **Legacy Non-FT `qwen3:8b` Callers**:
   - [`knowledge_graph/graph_store.py`](../../knowledge_graph/graph_store.py) defaults `QWEN_MODEL_NAME` to `"qwen3:8b"`, reads `QWEN_API_BASE`, and sends raw `httpx` `/api/generate` (`"format": "json"`) prompts.
   - [`ingestion/sec/parser.py`](../../ingestion/sec/parser.py) (`extract_sec_relationships`) imports `OLLAMA_API_BASE`, `OLLAMA_MODEL_NAME`, and `_clean_json_response` directly from `knowledge_graph/graph_store.py`, bypassing `YarnBallModelClient`.
   - [`retrieval/text_to_cql.py`](../../retrieval/text_to_cql.py) duplicates `_default_ollama_url()`, defaults `QWEN_MODEL_NAME` to `"qwen3:8b"`, and uses raw `httpx` prompting rather than `YarnBallModelClient` (`TEXT_TO_CYPHER`).
   - [`.env.example`](../../.env.example) still documents `QWEN_MODEL_NAME=qwen3:8b` and `QWEN_API_BASE`.
2. **8 GB VRAM Thrashing & Lost SFT/DPO Alignment**:
   - Under `Q4_K_M`, `yarnball-qwen:7b` (`data/merged_model/yarnball-qwen-7b-q4_k_m.gguf`) consumes ~4.3 GB VRAM, fits cleanly inside an 8 GB GPU budget, and achieved `100.0%` functional accuracy at `74.89 tok/s` across all 5 tasks (`EXTRACT_SEC_GRAPH`, `EXTRACT_NEWS_EVENT`, `TEXT_TO_CYPHER`, `CONTAGION_REASONING`, `PORTFOLIO_RECOMMENDATION`) in [`data/dpo/benchmark_post_dpo_scorecard.json`](../../data/dpo/benchmark_post_dpo_scorecard.json).
   - Keeping non-FT `qwen3:8b` active for KG extraction and Text-to-Cypher while `yarnball-qwen:7b` handles reasoning forces Ollama to evict and reload ~4.5 GB models inside 8 GB VRAM, and forfeits DPO negative-control rejection (`"(none)"`), token budgets, and runaway-loop protection.
3. **Training vs. Runtime Separation**:
   - Offline SFT/DPO training scripts under `sft_pipeline/` legitimately reference upstream HuggingFace base checkpoints to train future adapters (e.g., linear-attention Qwen 3.5 9B per [`ADR-00012`](ADR-00012-qwen-model-upgrade-and-fine-tuning-plan-.md)), whereas live runtime services (`ingestion/`, `knowledge_graph/`, `retrieval/`, `explorer/`) must depend solely on the deployed fine-tuned `yarnball-qwen` runtime interface.

---

## 2. Decision Drivers & Constraints

1. **Single Hot Model in 8 GB VRAM**: Every runtime inference task (Tasks A–E) must route to the same active `YARNBALL_MODEL_TAG` (`yarnball-qwen:7b` default) so one `Q4_K_M` model stays resident in VRAM without Ollama model swapping.
2. **Preserve Future Checkpoint Swapability (`ADR-00012`)**: Centralizing all Ollama calls in `YarnBallModelClient` ensures upgrading to a fine-tuned 9B linear-attention checkpoint requires updating only `YARNBALL_MODEL_TAG` / Ollama `Modelfile`.
3. **Single URL & Guardrail Implementation**: Consolidate Docker-vs-host Ollama URL resolution (`host.docker.internal` inside `/.dockerenv`, `127.0.0.1` on host), prefix forcing, token caps, and runaway n-gram loop truncation inside `YarnBallModelClient` instead of duplicating `httpx` clients in three modules.
4. **Zero Regressions in Existing Unit Tests**: `tests/retrieval/test_text_to_cql.py`, `tests/retrieval/test_hybrid_retriever.py`, and `tests/knowledge_graph/test_memgraph_consumer.py` must continue passing cleanly via mock transport or clean method injection.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Keep `qwen3:8b` for Extraction / Cypher and `yarnball-qwen:7b` Only for Contagion Reasoning**:
  - *Rationale for Rejection*: Causes VRAM model thrashing on 8 GB GPUs, ignores trained Task A/B/C capabilities (`100.0%` benchmark accuracy), and preserves three divergent Ollama HTTP/prompt implementations.
- **Alternative B: Purge Upstream Base Model References Inside `sft_pipeline/` Training Scripts Too**:
  - *Rationale for Rejection*: `sft_pipeline/` offline training scripts need upstream base checkpoint IDs to train new LoRA/DPO adapters (including future Qwen 3.5 9B upgrades under [`ADR-00012`](ADR-00012-qwen-model-upgrade-and-fine-tuning-plan-.md)). Only runtime inference paths should purge non-FT defaults.

---

## 4. Decision Outcome

Purge all runtime non-FT `qwen3:8b` defaults and duplicate Ollama HTTP clients prior to downstream Wave 1 (`ADR-00029`, `ADR-00023`) changes:

1. **Docker/Host URL Normalization in `YarnBallModelClient` ([`sft_pipeline/serving/model_client.py`](../../sft_pipeline/serving/model_client.py))**:
   - Absorb `_default_ollama_url()` resolution (`OLLAMA_API_BASE`, honoring Docker `/.dockerenv` vs host `127.0.0.1` translation; optionally reading legacy `QWEN_API_BASE` only with a deprecation warning).
   - Default `YARNBALL_MODEL_TAG` to `"yarnball-qwen:7b"` everywhere.
2. **Route All Runtime Modules Through `YarnBallModelClient`**:
   - [`knowledge_graph/graph_store.py`](../../knowledge_graph/graph_store.py): Remove `QWEN_MODEL_NAME` / `"qwen3:8b"` and duplicate `_default_ollama_url` + `httpx` request code; invoke `YarnBallModelClient` (`task="EXTRACT_NEWS_EVENT"`).
   - [`ingestion/sec/parser.py`](../../ingestion/sec/parser.py): Update `extract_sec_relationships` to invoke `YarnBallModelClient` (`task="EXTRACT_SEC_GRAPH"`) rather than borrowing `OLLAMA_MODEL_NAME="qwen3:8b"` from `graph_store.py`.
   - [`retrieval/text_to_cql.py`](../../retrieval/text_to_cql.py): Replace `QWEN_MODEL_NAME="qwen3:8b"` and raw `_call_llm` `httpx` requests with `YarnBallModelClient` (`task="TEXT_TO_CYPHER"`), removing `qwen3:8b` docstrings/comments and keeping `TextToCQL.clean_cypher` + `CypherASTWhitelistGuardrail` intact.
3. **Environment & Configuration Cleanup**:
   - Update [`.env.example`](../../.env.example) to replace `QWEN_MODEL_NAME=qwen3:8b` and `QWEN_API_BASE` with `YARNBALL_MODEL_TAG=yarnball-qwen:7b` and `OLLAMA_API_BASE`.
   - Enforce a CI/test check confirming no runtime module (`knowledge_graph/`, `ingestion/`, `retrieval/`, `explorer/`, `contagion/`, `streaming/`) references `qwen3` or `QWEN_MODEL_NAME`.

---

## 5. Invariants & Guardrails

1. **Zero Non-FT Model Strings in Runtime Inference**: Runtime packages (`ingestion/`, `knowledge_graph/`, `retrieval/`, `explorer/`, `contagion/`, `streaming/`) MUST NOT reference `qwen3` or default to un-fine-tuned checkpoints.
2. **Single Inference Gateway**: All local Ollama / vLLM completions across Tasks A–E MUST route through `YarnBallModelClient` (allowing dependency injection / mock `transport_fn` in tests).
3. **Training-Runtime Separation**: Upstream base model identifiers are permitted solely in offline training/export scripts under `sft_pipeline/`.
4. **Zero Absolute Paths Invariant**: All references remain strictly repository-relative (`../../<path>`).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Guarantees a single `Q4_K_M` model stays warm in 8 GB VRAM across SEC extraction, news extraction, Text-to-Cypher, and contagion reasoning.
  - Simplifies [`ADR-00029`](ADR-00029-structured-llm-extraction-and-quarantine.md) (`Structured LLM Extraction Outputs & Failure Quarantine Queue`) and [`ADR-00023`](ADR-00023-unified-live-ontology-gate-and-schema---.md) (`Unified Live Ontology Gate`) because extraction now speaks `yarnball-qwen`'s ChatML / OpenCypher-triple contract directly.
- **Negative / Trade-offs**:
  - Non-mock live ingestion and Text-to-Cypher require `yarnball-qwen:7b` (or another fine-tuned `YARNBALL_MODEL_TAG` registered in Ollama from `data/merged_model/Modelfile`) rather than pulling stock `qwen3:8b`.
