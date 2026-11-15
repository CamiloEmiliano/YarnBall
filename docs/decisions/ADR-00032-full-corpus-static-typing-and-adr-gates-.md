# ADR-00032: Full-Corpus Static Typing, Deterministic ADR Integrity & Architecture Gates

- **ADR ID**: ADR-00032
- **Filename**: `ADR-00032-full-corpus-static-typing-and-adr-gates-.md`
- **Date**: 2026-11-15 07:39 UTC
- **Status**: ACCEPTED
- **Target Subsystem**: `mypy.ini` / `docs/decisions` / `tests/docs` / `tests/ontology` / `.agents/skills` / `streaming` / `retrieval` / `ontology` / `sft_pipeline` / `tools`
- **Linked Commits**: [`68cc394`](../../mypy.ini), [`68cc394`](../../tests/docs/test_adr_integrity.py), [`68cc394`](../../tests/ontology/test_architecture_invariants.py), [`68cc394`](../../.agents/skills/adr-implementation/SKILL.md)

---

## 1. Context & Problem Statement

Following completion of Wave 1 ([`ADR-00023`](ADR-00023-unified-live-ontology-gate-and-schema---.md), [`ADR-00024`](ADR-00024-clause-scoped-negation-and-hedge-filter-.md), [`ADR-00025`](ADR-00025-entity-stoplist-and-dynamic-cik-rollup--.md), [`ADR-00029`](ADR-00029-structured-llm-extraction-and-quarantine.md), [`ADR-00030`](ADR-00030-purge-non-ft-qwen-runtime-dependencies--.md), [`ADR-00031`](ADR-00031-iterative-predicate-convergence-protocol.md)), a post-wave audit revealed two classes of systemic omission risk:

1. **Non-Deterministic Documentation & Interface Fanout Drift**:
   - [`docs/decisions/index.md`](index.md) omitted the Markdown hyperlink on `ADR-00029`'s `Linked Commits` cell (`e5d1028` plain backticks) and `ADR-00029` referenced a pre-normalization filename for `ADR-00030`.
   - When `source_text` clause-scoping ([`ADR-00024`](ADR-00024-clause-scoped-negation-and-hedge-filter-.md)) was added to [`validate_graph_payload`](../../ontology/validation.py), callers outside [`store_graph_entities`](../../knowledge_graph/graph_store.py)—specifically `EdgarWorker._persist_sec_graph_data` in [`ingestion/sec/harvester.py`](../../ingestion/sec/harvester.py)—did not automatically trigger a failure because no static type or AST invariant gate checked parameter/caller propagation across the full corpus.
2. **Incomplete `mypy` Scope & Unresolved Type Debt**:
   - [`mypy.ini`](../../mypy.ini) checked only 9 packages (`curation, ingestion, knowledge_graph, ontology, retrieval, sft_pipeline, synthesis, universe, tools`), omitting active Python packages `contagion`, `embedding`, `explorer`, and `streaming`.
   - Running `mypy` across the full repository exposed **21 pre-existing type errors across 6 files** ([`sft_pipeline/rejection_queue.py`](../../sft_pipeline/rejection_queue.py), [`ontology/convergence.py`](../../ontology/convergence.py), [`retrieval/offline_store.py`](../../retrieval/offline_store.py), [`tools/dry_run_sec_8k_contagion.py`](../../tools/dry_run_sec_8k_contagion.py), [`streaming/producer.py`](../../streaming/producer.py), [`streaming/consumer.py`](../../streaming/consumer.py)).

Relying on ad-hoc visual inspection or prompt discipline alone cannot deterministically prevent silent regressions when interfaces or ADRs evolve. Structural invariants must move into automated static and test gates, while procedural discipline (caller fanout checks and commit backfill order) must be codified as an explicit implementation skill.

---

## 2. Decision Drivers & Constraints

1. **Deterministic Over Procedural**: Every invariant checkable by AST inspection, regular expressions, filesystem checks, or static type checking MUST execute automatically inside `.venv/bin/pytest` and `.venv/bin/mypy`.
2. **Full-Corpus Static Typing Without False Positives**: Every production Python package (`contagion`, `curation`, `embedding`, `explorer`, `ingestion`, `knowledge_graph`, `ontology`, `retrieval`, `sft_pipeline`, `streaming`, `synthesis`, `tools`, `universe`) must be checked by a bare `.venv/bin/mypy` invocation with zero errors.
3. **Preserve Legacy Intentional History**: Link integrity verification in `docs/decisions/` must respect historical decommissioning records such as [`ADR-00016`](ADR-00016-decommission-legacy-chainlit-ui-frontend.md), which intentionally documents deleted legacy UI paths, as well as early `ADR-00001`–`ADR-00007` inline header formatting while enforcing strict hyperlink and hash discipline on all `index.md` entries and post-`ADR-00007` ADRs.
4. **Preserve Runtime Monkeypatching Compatibility**: Fixing typing in [`streaming/producer.py`](../../streaming/producer.py) must keep the module-level `KafkaProducer` attribute intact so existing streaming unit tests continue to monkeypatch cleanly.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Prompt-Only Rules Without Automated Test/Type Gates**:
  - *Rationale for Rejection*: LLM attention is stochastic across multi-file edits; rules without executable assertions still allow silent link rot, missing parameters, and type drift.
- **Alternative B: Enforce Full `strict = true` Globally Across Legacy Modules**:
  - *Rationale for Rejection*: Existing strict per-package overrides (`retrieval`, `sft_pipeline`, `ontology`, `curation`, `universe`) plus full-corpus checking (`mypy.ini` base config) already catch untyped defs in core subsystems and signature/return-type bugs everywhere else; rewriting every test helper or prototype script globally adds churn without improving interface safety.

---

## 4. Decision Outcome

### 4.1 Full-Corpus `mypy` Coverage & Zero-Error Remediation
1. Update [`mypy.ini`](../../mypy.ini) to set `explicit_package_bases = true` and expand `files =` to include all 13 Python packages:
   `contagion, curation, embedding, explorer, ingestion, knowledge_graph, ontology, retrieval, sft_pipeline, synthesis, streaming, tools, universe`.
2. Remediate all 21 type errors cleanly:
   - [`sft_pipeline/rejection_queue.py`](../../sft_pipeline/rejection_queue.py): bind checked `dict` before `.get("type")` and cast `json.loads(...)` return in `load_checkpoint`.
   - [`ontology/convergence.py`](../../ontology/convergence.py): annotate `summary` dict return from `evaluate_predicate_convergence`.
   - [`retrieval/offline_store.py`](../../retrieval/offline_store.py): add full type annotations to `_InMemorySubgraphStore.session` and `_InMemorySession` methods (`__enter__`, `__exit__`, `run`) matching `[mypy-retrieval.*]` strictness.
   - [`tools/dry_run_sec_8k_contagion.py`](../../tools/dry_run_sec_8k_contagion.py): align `DeterministicMockModelClient.generate` signature with [`YarnBallModelClient.generate`](../../sft_pipeline/serving/model_client.py).
   - [`streaming/producer.py`](../../streaming/producer.py) and [`streaming/consumer.py`](../../streaming/consumer.py): type `KafkaProducer` cleanly via a `Protocol` / `TYPE_CHECKING` separation without breaking the runtime module attribute `KafkaProducer`, allow `topic: str | None` in `_init_topic`, return typed producers from `_producer()`/`get_producer()`, and cast JSON/validated payloads cleanly.

### 4.2 Deterministic ADR & Architecture Invariant Gates
1. **`tests/docs/test_adr_integrity.py`** verifies across all files matching `docs/decisions/ADR-*.md` and [`docs/decisions/index.md`](index.md):
   - Every ADR filename has exact length `53` characters (`ADR-XXXXX-<40-char-slug>.md`).
   - Zero absolute filesystem paths, home-directory prefixes, or file URI schemes appear anywhere in `docs/decisions/`.
   - Every local Markdown link resolves on disk relative to the source document (exempting intentionally removed legacy UI paths documented in [`ADR-00016`](ADR-00016-decommission-legacy-chainlit-ui-frontend.md)).
   - Every ADR is listed in `docs/decisions/index.md` with matching `Status` (`PROPOSED` vs. `ACCEPTED`), and every `ACCEPTED` row in `index.md` carries hyperlinked commit links (`[<7-hex-hash>]` targeting a valid repository-relative path) pointing to valid non-`Pending` git commit prefixes (plus matching header metadata for post-`ADR-00007` ADRs).
2. **`tests/ontology/test_architecture_invariants.py`** enforces key runtime AST invariants across active packages:
   - **`ADR-00030` Single Model Boundary**: Zero runtime modules outside `sft_pipeline/` import `ollama` directly.
   - **`ADR-00023` / `ADR-00024` Ontology Gate & `source_text` Propagation**: Every active Memgraph writer (`store_graph_entities` in [`knowledge_graph/graph_store.py`](../../knowledge_graph/graph_store.py) and `EdgarWorker._persist_sec_graph_data` in [`ingestion/sec/harvester.py`](../../ingestion/sec/harvester.py)) calls `validate_graph_payload` with explicit keyword argument `source_text=...`.

### 4.3 Procedural Caller-Fanout & Verification Skill
Create workspace skill [`.agents/skills/adr-implementation/SKILL.md`](../../.agents/skills/adr-implementation/SKILL.md) codifying:
- Mandatory caller/reference fanout search (`rg` + `.venv/bin/graphify query`) before modifying shared function signatures, filenames, or ADRs.
- Explicit two-step commit sequence for ADRs (`implementation commit` -> `backfill commit hash into ADR header + index.md` -> `backfill commit`).
- Mandatory unified verification gate (`.venv/bin/mypy` + `.venv/bin/pytest` + `.venv/bin/graphify update .`) before declaring any phase complete.

---

## 5. Verification & Acceptance Criteria

- Bare `.venv/bin/mypy` reports `Success: no issues found` across all 13 configured packages.
- `.venv/bin/pytest --ignore=tests/knowledge_graph/test_embedding_store.py` passes with zero failures, including `tests/docs/test_adr_integrity.py` and `tests/ontology/test_architecture_invariants.py`.
- Zero host filesystem paths or `file` URI schemes appear anywhere in `docs/decisions/`.
- Every `ACCEPTED` row in [`docs/decisions/index.md`](index.md) carries clickable Markdown links (`[<7-hex-hash>]` pointing to a repository-relative target) to real git commits.
- `.venv/bin/graphify update .` succeeds and leaves the knowledge graph synchronized.

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**: Converts ADR link integrity, zero-absolute-path rules, cross-module type safety, single-model-client boundaries, and ontology-gate parameter propagation into fast, repeatable regression checks runnable in seconds.
- **Negative / Trade-offs**: Any future Python package or `ACCEPTED` ADR that omits `index.md` backfill hyperlinks, breaks relative paths, or introduces untyped signatures in strict modules fails CI/pytest immediately until resolved.
