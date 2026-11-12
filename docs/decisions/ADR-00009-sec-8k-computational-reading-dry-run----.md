# ADR-00009: SEC 8-K Computational Reading & Contagion Dry Run

<!--
STRICT FILENAME CONVENTION:
- Format: ADR-[5_DIGIT_ID]-[40_CHARACTER_SLUG].md
- Total Filename Length: Exactly 53 characters (including '.md')
- Slug Specification: Exactly 40 characters using lowercase alphanumeric and hyphens.
- Padding Rule: If the natural slug is under 40 characters, pad with trailing hyphens ('-').
Filename: ADR-00009-sec-8k-computational-reading-dry-run----.md (53 chars)
-->

- **ADR ID**: ADR-00009
- **Filename**: `ADR-00009-sec-8k-computational-reading-dry-run----.md`
- **Date**: 2026-11-11 04:35 UTC
- **Status**: ACCEPTED
- **Target Subsystem**: `retrieval` / `contagion` / `evaluation`
- **Linked Commits**: [`41b38db`](../../tools/dry_run_sec_8k_contagion.py)

---

## 1. Context & Problem Statement

YarnBall's core capability is the **computational reading of unstructured corporate regulatory disclosures** (SEC Form 8-K, Item 1.01, 1.02, 1.03, or 8.01) to trace supply-chain contagion and operational disruption across interconnected networks of companies.

While isolated unit tests prove functional correctness across individual modules, we lack a cohesive, push-button **end-to-end dry-run tool** that:
1. Ingests a raw SEC Form 8-K text payload.
2. Extracts the material disruption event (loss ratio, duration, affected components).
3. Atomically merges the event into the live knowledge graph.
4. Traverses active supply-chain paths and executes `SparseNetworkCascadeEngine` to calculate multi-tier capacity loss, inventory buffer delays, and unbuffered downtime.
5. Emits the platform's native [`OperationalContagionReport`](../../retrieval/reporting.py) (via [`PipelineResult.to_operational_report(...)`](../../retrieval/pipeline.py#L88)) containing hop-ordered subgraph topology, deterministic shock summaries, and the aligned model's Chain-of-Thought contagion assessment per [ADR-00011](ADR-00011-operational-contagion-report-schema-----.md).

Downstream trading decisions, directional basket construction, and portfolio hedging belong to a separate, downstream decision domain. This ADR focuses strictly on proving the upstream intelligence pipeline works reliably on authentic regulatory text.

---

## 2. Decision Drivers & Constraints

1. **End-to-End Processing Throughput (< 30 Seconds)**:
   - The end-to-end Python pipeline must complete ingestion, graph mutation, cascade diffusion, and model verbalization within **30 seconds** of receiving the 8-K payload.
   - Microsecond HFT latency is explicitly non-goal; depth of extraction, mathematical grounding, and operational assessment fidelity are the primary criteria.
2. **Authentic SEC 8-K Regulatory Grounding**:
   - The dry run must parse authentic corporate disclosure schemas conforming to `.agents/rules/data_sources.md` (using local SEC filing fixtures such as `tests/fixtures/filings/aapl_8k_disruption.json` or real Form 8-K filings via `edgartools`).
3. **Native Platform Schema Grounding (`OperationalContagionReport`)**:
   - The dry-run output must strictly utilize existing, verified platform contracts:
     - [`OperationalContagionReport`](../../retrieval/reporting.py) conforming to [ADR-00011](ADR-00011-operational-contagion-report-schema-----.md).
     - [`PipelineResult`](../../retrieval/pipeline.py#L62) from the retrieval orchestrator.
     - [`ConvergentShockSummary`](../../contagion/cascade.py#L211) from the cascade engine.
   - Avoid premature, ungrounded external trading signal schemas.
4. **Strict Offline Isolation**:
   - Zero live broker APIs, external HTTP endpoints, or remote sockets (`socket.socket.connect` blocked).

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Synthetic Mock Event Generation**:
  - Invent fictional tickers and hypothetical shock payloads.
  - *Rejection Rationale*: Fails to test authentic SEC legal language, item numbering, and realistic company ticker mapping.
- **Alternative B: Coupling to Downstream Trading / Portfolio Schemas**:
  - Forcing the dry-run tool to emit speculative trade signals (long/short baskets, option strikes).
  - *Rejection Rationale*: Violates domain boundaries. Downstream strategy and portfolio construction constitute a distinct domain that consumes YarnBall's output. YarnBall's core responsibility is objective, factual computational reading and operational contagion assessment.

---

## 4. Proposed Architecture: Dry-Run Pipeline

```mermaid
flowchart TD
    A["Raw SEC Form 8-K<br/>(Item 8.01 Other Events)"] --> B["Filing Ingestion<br/>& Event Extractor"]
    B --> C["Atomic Graph Upsert<br/>(Memgraph Driver)"]
    C --> D["Automated Query Generation<br/>(Task C Cypher AST)"]
    D --> E["Sparse Cascade Engine<br/>(contagion/cascade.py)"]
    E --> F["yarnball-qwen:7b<br/>(Task D Contagion Reasoning)"]
    F --> G["Verified Output:<br/>OperationalContagionReport"]
```

### Native Output Contract

The dry-run tool executes `GraphRAGPipelineOrchestrator.run()` and emits:
1. **Machine-Readable Artifact**: An [`OperationalContagionReport`](../../retrieval/reporting.py) generated via `PipelineResult.to_operational_report(...)` conforming to the schema established in [ADR-00011](ADR-00011-operational-contagion-report-schema-----.md).
2. **Visual Inspection Report**: A standardized Section 4.5B visual ASCII/Markdown report for terminal inspection and human review.

### Concrete Deliverables

1. **Tool Implementation**: Author `tools/dry_run_sec_8k_contagion.py`.
   - Accepts an SEC Form 8-K JSON payload or raw text path.
   - Merges the event into the graph.
   - Executes the full orchestrator flow and prints a clean, structured inspection summary.
2. **Automated Integration Test**: Author `tests/integration/test_dry_run_8k_contagion.py`.
   - Runs `tools/dry_run_sec_8k_contagion.py` under offline isolation.
   - Asserts end-to-end execution $< 30\text{s}$, exact numerical metric preservation, and valid CoT `<think>` traces.

---

## 5. Invariants & Guardrails

1. **Deterministic Metric Preservation**:
   - The textual explanation emitted by `yarnball-qwen:7b` in `contagion_verdict` must cite the exact numerical percentages, downtime days, and onset days computed by `SparseNetworkCascadeEngine`.
2. **Buffer Absorption Invariant**:
   - Downstream nodes whose inventory buffer covers the shock duration ($B \ge h_u D_u$) must be classified as `BUFFER_ABSORBED` (or `UNAFFECTED`) with $0.0$ unbuffered downtime days.
3. **Domain Boundary Invariant**:
   - The dry-run harness operates strictly within the computational reading and contagion domain. It does not emit speculative trading signals, broker orders, or portfolio allocations.

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Provides a single, reproducible push-button dry run proving the core computational reading test works on authentic SEC 8-K data.
  - Generates verified, factual Operational Contagion Reports ready for downstream strategy and backtesting modules.
  - Closes the feedback loop between raw unstructured legal disclosures and deterministic network math.
- **Negative / Trade-offs**:
  - Requires maintaining high-fidelity SEC 8-K fixtures in the test suite.
