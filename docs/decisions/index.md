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

| ID | Title | Date (YYYY-MM-DD HH:MM UTC) | Status | Subsystem | Linked Commits |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [ADR-00001](ADR-00001-sit-flow-2-event-shock-contagion-harness.md) | SIT Flow 2 Event Shock Vector & Contagion | 2026-11-10 03:24 UTC | `ACCEPTED` | `retrieval` / `ingestion` | [`5d95c75`](../../tests/fixtures/filings/aapl_8k_disruption.json), [`307c79e`](../../retrieval/shock_propagation.py), [`ad1b83a`](../../tests/integration/pipelines/test_event_shock_propagation_flow.py) |
| [ADR-00002](ADR-00002-sit-flow-3-nl-to-cypher-rag-e2e-harness-.md) | SIT Flow 3 NL-to-Cypher RAG E2E Harness | 2026-11-10 03:58 UTC | `ACCEPTED` | `retrieval` / `sft_pipeline` | [`4ae944b`](../../tests/integration/e2e/test_nl_to_cypher_rag_e2e.py), [`e9a3ef9`](../../tests/integration/e2e/test_nl_to_cypher_rag_e2e.py) |
| [ADR-00003](ADR-00003-sit-flow-4-streaming-event-graph-update-.md) | SIT Flow 4 Streaming Event Graph Update | 2026-11-10 05:44 UTC | `ACCEPTED` | `streaming` / `knowledge_graph` | [`af5be21`](../../tests/integration/pipelines/test_streaming_event_loop_flow.py) |
| [ADR-00004](ADR-00004-sit-flow-5-model-foundry-alignment-smoke.md) | SIT Flow 5 Model Foundry Alignment Smoke | 2026-11-10 06:05 UTC | `ACCEPTED` | `sft_pipeline` / `evaluation` | [`ef47afa`](../../tests/integration/pipelines/test_model_foundry_smoke_flow.py) |
| [ADR-00005](ADR-00005-sit-flow-1-pit-sec-filing-to-kg-ingest--.md) | SIT Flow 1 PiT SEC Filing to KG Ingest | 2026-11-10 08:12 UTC | `ACCEPTED` | `universe` / `ontology` | [`4ae944b`](../../tests/integration/pipelines/test_filing_to_graph_flow.py), [`041e541`](../../universe/sp500.py) |
| [ADR-00006](ADR-00006-phase-7-sparse-network-cascade-engine---.md) | Phase 7 Sparse Network Cascade Engine | 2026-11-10 08:32 UTC | `ACCEPTED` | `contagion` / `retrieval` | [`e0bc6cc`](../../contagion/cascade.py), [`4aecdc8`](../../contagion/cascade.py), [`ceaeac7`](../../retrieval/pipeline.py), [`953b832`](../../tests/contagion/test_phase5_tt02a_verification.py) |
| [ADR-00007](ADR-00007-cascade-engine-adversarial-verification-.md) | Cascade Engine Verification (10 Canonical + 8 Pathological) | 2026-11-11 04:05 UTC | `ACCEPTED` | `contagion` | [`63f377d`](../../contagion/cascade.py), [`923352f`](../../tests/contagion/test_cascade_pathologies.py) |
| [ADR-00008](ADR-00008-sit-harness-audit-and-unified-contract--.md) | SIT Harness Audit & Unified Contract | 2026-11-11 04:35 UTC | `ACCEPTED` | `evaluation` / `integration` | [`dd37ae3`](../../tests/integration/test_unified_sit_contract.py), [`dd37ae3`](../../tests/integration/pipelines/test_event_shock_propagation_flow.py), [`dd37ae3`](../../contagion/cascade.py) |
| [ADR-00009](ADR-00009-sec-8k-computational-reading-dry-run----.md) | SEC 8-K Computational Reading Dry Run | 2026-11-11 04:35 UTC | `ACCEPTED` | `retrieval` / `evaluation` | [`41b38db`](../../tools/dry_run_sec_8k_contagion.py) |
| [ADR-00010](ADR-00010-directional-basket-peer-cluster-mapping-.md) | Directional Basket & Peer Cluster Mapping | 2026-11-11 04:35 UTC | `REJECTED` | `ontology` / `retrieval` | Formally rejected: Out of domain scope (trading/market-making) |
| [ADR-00011](ADR-00011-operational-contagion-report-schema-----.md) | Operational Contagion Report Schema & Language | 2026-11-11 23:30 UTC | `ACCEPTED` | `retrieval` / `contagion` | [`7e6aa39`](../../retrieval/reporting.py) |
| [ADR-00012](ADR-00012-qwen-model-upgrade-and-fine-tuning-plan-.md) | Upgraded Qwen Model Fine-Tuning & Serving Plan | 2026-11-12 05:50 UTC | `PROPOSED` | `sft_pipeline` / `retrieval` | [`a26bc0c`](../../sft_pipeline/serving/model_client.py) |
| [ADR-00013](ADR-00013-contagion-event-time-series-join-schema-.md) | Contagion Event Telemetry & Time-Series Join Contract | 2026-11-13 06:30 UTC | `ACCEPTED` | `retrieval` / `tools` | [`retrieval/reporting.py`](../../retrieval/reporting.py), [`tools/dry_run_sec_8k_contagion.py`](../../tools/dry_run_sec_8k_contagion.py) |
| [ADR-00014](ADR-00014-sec-8k-boilerplate-sanitization-gating--.md) | SEC 8-K Boilerplate Sanitization & Disruption Gating | 2026-11-13 06:55 UTC | `ACCEPTED` | `ingestion` / `retrieval` | [`ingestion/sec/disruption_gating.py`](../../ingestion/sec/disruption_gating.py), [`tools/dry_run_sec_8k_contagion.py`](../../tools/dry_run_sec_8k_contagion.py) |
| [ADR-00015](ADR-00015-dynamic-in-memory-offline-subgraph-store.md) | Dynamic In-Memory Offline Subgraph Store | 2026-11-13 07:15 UTC | `ACCEPTED` | `retrieval` / `knowledge_graph` | [`retrieval/offline_store.py`](../../retrieval/offline_store.py), [`tools/dry_run_sec_8k_contagion.py`](../../tools/dry_run_sec_8k_contagion.py) |
| [ADR-00016](ADR-00016-decommission-legacy-chainlit-ui-frontend.md) | Decommission Legacy Chainlit UI Frontend | 2026-11-13 08:47 UTC | `ACCEPTED` | `interactive` / `requirements` | [`requirements.txt`](../../requirements.txt), [`.gitignore`](../../.gitignore) |
| [ADR-00017](ADR-00017-local-fastapi-kg-contagion-workbench-app.md) | Local FastAPI Workbench Backend & API Contracts | 2026-11-13 09:46 UTC | `ACCEPTED` | `explorer` / `retrieval` | [`explorer/workbench_api.py`](../../explorer/workbench_api.py), [`tools/serve_explorer.py`](../../tools/serve_explorer.py) |




