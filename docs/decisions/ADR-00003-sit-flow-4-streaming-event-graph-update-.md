# ADR-00003: SIT Flow 4 Streaming Event Graph Update

<!--
STRICT FILENAME CONVENTION:
- Format: ADR-[5_DIGIT_ID]-[40_CHARACTER_SLUG].md
- Total Filename Length: Exactly 53 characters (including '.md')
- Slug Specification: Exactly 40 characters using lowercase alphanumeric and hyphens.
- Padding Rule: If the natural slug is under 40 characters, pad with trailing hyphens ('-').
Filename: ADR-00003-sit-flow-4-streaming-event-graph-update-.md (53 chars)
-->

- **ADR ID**: ADR-00003
- **Filename**: `ADR-00003-sit-flow-4-streaming-event-graph-update-.md`
- **Date**: 2026-11-10 00:44
- **Status**: ACCEPTED
- **Target Subsystem**: `streaming` / `curation` / `knowledge_graph`
- **Linked Commits**: `af5be21` (Flow 4 implementation and SIT test harness)

---

## 1. Context & Problem Statement

In an enterprise Financial GraphRAG platform, material corporate events (SEC Form 8-K filings, earnings releases, credit alerts) stream asynchronously via distributed messaging brokers (Apache Kafka). When these events trigger graph updates, three critical failure modes threaten system integrity:

1. **Duplicate Message Replay & Topology Inflation**: In distributed messaging systems, consumer group rebalancing, offset re-reads, and network retries frequently cause duplicate message delivery (at-least-once semantics). If the graph ingestion layer uses naive insertion (`CREATE`), duplicate company nodes and supply-chain edges proliferate, exponentially distorting multi-hop centrality and contagion shock diffusion.
2. **Historical Timestamp & Property Corruption**: Replaying historical events must never shift `as_of_date` or `created_at` timestamps forward to the replay time. Updating existing edges must follow strict idempotent update semantics (`ON CREATE SET` vs. `ON MATCH SET`) to preserve temporal auditability.
3. **Stream Poisoning from Malformed Regulatory Disclosures**: Corrupted, partial, or schema-violating event payloads must never crash the streaming consumer or write partial, unindexed nodes to the property graph. They must be quarantined immediately to a Dead-Letter Topic (DLT).

System Integration Test (SIT) **Flow 4** formalizes the contract across `streaming/`, `curation/`, and `knowledge_graph/`:
`Kafka Event Payload -> Consumer Ingestion -> Curation Parsing -> Graph MERGE Mutation -> Dead-Letter Routing & Idempotency Verification`.

---

## 2. Decision Drivers & Constraints

1. **Strict Write Idempotency**:
   - Replaying the identical event payload N times must result in bit-exact graph equivalence: exactly zero duplicate nodes, zero duplicate edges, and zero timestamp drift.
2. **Offline Hermetic Execution**:
   - Tier 2 SIT integration tests must execute without external Kafka brokers or Docker daemons, utilizing in-memory transport fallbacks running in `< 2.0 seconds`.
3. **Dead-Letter Quarantine (DLT)**:
   - Any message failing Pydantic schema validation, lacking mandatory CIK identifiers, or containing invalid payload hashes must route deterministically to the DLT without halting consumer processing.
4. **Point-in-Time Temporal Preservation**:
   - Graph mutations must preserve the original disclosure timestamp (`period_of_report`) rather than the processing timestamp (`fetched_at`), preventing temporal leakage into historical backtests.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Live Kafka Daemon Requirement in CI/CD**:
  - *Rejected*: Requiring a running Kafka/Zookeeper/KRaft container causes environment-dependent test failures, port collisions, and execution latency exceeding the 90-second Tier 2 budget.
- **Alternative B: Periodic Offline Graph Deduplication Jobs**:
  - *Rejected*: Asynchronous deduplication leaves transient windows where duplicate paths exist, corrupting real-time analytical queries and shock propagation traversals.
- **Alternative C: Discarding Malformed Messages Silently**:
  - *Rejected*: Violates regulatory auditability and prevents pipeline operators from identifying upstream feed schema drift.

---

## 4. Decision Outcome

We establish **Flow 4** as an architectural contract implemented in `tests/integration/pipelines/test_streaming_event_loop_flow.py` with the following components:

1. **In-Memory Deterministic Stream Harness**:
   - Replays sequenced 8-K disclosures through `streaming.consumer.ingest_message()` using local mock buffers.
2. **Graph Store Idempotent MERGE Engine**:
   - Compiles parsed events into Cypher `MERGE` patterns keyed on unique entity identifiers (10-digit zero-padded CIK) and directed relationship keys `(source_cik, target_cik, rel_type)`.
3. **Dead-Letter Topic (DLT) Queue Verification**:
   - Verifies that malformed payloads (missing hash, corrupted JSON) invoke `send_to_dlt()` with `reason='invalid_kafka_payload'` and leave the primary graph store completely untouched.
4. **Idempotency Assertions**:
   - Measures pre-ingest and post-ingest node/edge counts across repeated replays to guarantee `delta_nodes == 0` and `delta_edges == 0`.

---

## 5. Invariants & Guardrails

1. **Idempotency Invariant**:
   - For any message `M`, executing `ingest_message(M)` twice yields identical graph state:
     `len(nodes_run_2) == len(nodes_run_1)` and `len(edges_run_2) == len(edges_run_1)`.
2. **Zero Primary Graph Poisoning Invariant**:
   - Malformed payloads rejected by `_is_valid_payload()` or Pydantic validators write zero nodes and zero edges to the database store.
3. **DLT Routing Invariant**:
   - Every rejected message generates a corresponding record in the Dead-Letter Queue containing the original payload, rejection timestamp, and error classification.
4. **Offline Isolation Invariant**:
   - Test execution runs 100% offline with zero live broker dependencies.

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Guarantees streaming resilience against network retries, consumer crashes, and message duplication.
  - Ensures clean separation between valid business events and corrupted upstream inputs via Dead-Letter isolation.
  - Secures the streaming foundation required before deploying Phase 7 continuous market-shock ingestion.
- **Negative / Trade-offs**:
  - Requires maintaining Cypher `MERGE` statement templates with `ON CREATE` and `ON MATCH` clauses for all streaming graph mutations.

