# Architecture Decision Records (ADR) Registry

This directory maintains the immutable, chronological record of all high-consequence architectural decisions, design trade-offs, and invariants for the YarnBall Financial GraphRAG Platform.

---

## Filename & Formatting Conventions

To preserve visual discipline, strict columnar alignment in directory listings, and predictable sorting, every ADR filename MUST adhere to the following specification:

- **Format**: `ADR-[5_DIGIT_ID]-[40_CHARACTER_SLUG].md`
- **Total Filename Length**: Exactly **53 characters** (`4 + 5 + 1 + 40 + 3 = 53`).
- **Slug Padding Rule**: The descriptive slug must be exactly **40 characters** in length using lowercase alphanumeric characters and hyphens. If the natural description is under 40 characters, pad it with trailing hyphens (`-`).

Example:
```
ADR-00001-record-architecture-decisions-in-repo--.md  (53 chars)
ADR-00002-dual-exchange-sp500-universe-grounding-.md  (53 chars)
```

---

## Lifecycle Statuses

- **`PROPOSED`**: Under active discussion and review; not yet accepted into code.
- **`ACCEPTED`**: Approved and actively governing the codebase.
- **`SUPERSEDED BY ADR-XXXXX`**: Formally replaced by a subsequent decision record.
- **`DEPRECATED`**: Obsolete; no longer enforced or maintained.

---

## Master Decision Index

| ID | Title | Date (YYYY-MM-DD HH:MM) | Status | Subsystem | Linked Commits |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [ADR-00001](ADR-00001-sit-flow-2-event-shock-contagion-harness.md) | SIT Flow 2 Event Shock Vector & Contagion | 2026-11-09 22:24 | `ACCEPTED` | `retrieval` / `ingestion` | [`5d95c75`](../../tests/fixtures/filings/aapl_8k_disruption.json), [`307c79e`](../../retrieval/shock_propagation.py), [`ad1b83a`](../../tests/integration/pipelines/test_event_shock_propagation_flow.py) |
| [ADR-00002](ADR-00002-sit-flow-3-nl-to-cypher-rag-e2e-harness-.md) | SIT Flow 3 NL-to-Cypher RAG E2E Harness | 2026-11-09 22:58 | `ACCEPTED` | `retrieval` / `sft_pipeline` | [`4ae944b`](../../tests/integration/e2e/test_nl_to_cypher_rag_e2e.py), [`e9a3ef9`](../../tests/integration/e2e/test_nl_to_cypher_rag_e2e.py) |
| [ADR-00003](ADR-00003-sit-flow-4-streaming-event-graph-update-.md) | SIT Flow 4 Streaming Event Graph Update | 2026-11-10 00:44 | `ACCEPTED` | `streaming` / `knowledge_graph` | [`af5be21`](../../tests/integration/pipelines/test_streaming_event_loop_flow.py) |
| [ADR-00004](ADR-00004-sit-flow-5-model-foundry-alignment-smoke.md) | SIT Flow 5 Model Foundry Alignment Smoke | 2026-11-10 01:05 | `ACCEPTED` | `sft_pipeline` / `evaluation` | [`a788815`](../../tests/integration/pipelines/test_model_foundry_smoke_flow.py) |


