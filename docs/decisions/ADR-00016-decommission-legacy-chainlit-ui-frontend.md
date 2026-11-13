# ADR-00016: Decommission Legacy Chainlit UI Frontend

- **ADR ID**: ADR-00016
- **Filename**: `ADR-00016-decommission-legacy-chainlit-ui-frontend.md`
- **Date**: 2026-11-13 08:47 UTC
- **Status**: ACCEPTED
- **Target Subsystem**: `interactive` / `requirements`
- **Linked Commits**: [`b198a35`](../../requirements.txt), [`b198a35`](../../.gitignore)

---

## 1. Context & Problem Statement

[`interactive/app.py`](../../interactive/app.py) is an obsolete Chainlit + PyVis conversational interface wired to the pre-contagion `HybridGraphRAGEngine` stack (displaying a stale hardcoded `qwen3:8b` welcome banner alongside live Memgraph and `pgvector` assumptions), along with orphaned Chainlit configuration artifacts ([`chainlit.md`](../../chainlit.md), `.chainlit/config.toml`) and broken CLI paths (e.g., `chainlit run rag/app.py -w`). Meanwhile, YarnBall's active architecture centers on deterministic supply-chain cascade calculation, SEC Form 8-K sanitization/gating, Point-in-Time in-memory subgraphs, and 1NF telemetry ([`ADR-00011`](ADR-00011-operational-contagion-report-schema-----.md), [`ADR-00013`](ADR-00013-contagion-event-time-series-join-schema-.md), [`ADR-00014`](ADR-00014-sec-8k-boilerplate-sanitization-gating--.md), [`ADR-00015`](ADR-00015-dynamic-in-memory-offline-subgraph-store.md)).

Keeping and refactoring `interactive/` risks carrying mismatched abstractions (`HybridGraphRAGEngine`, live-DB assumptions, and generic triple rendering) into upcoming contagion visualization work. Rather than refactoring legacy UI code, this ADR formally removes the obsolete Chainlit UI stack cleanly before designing a fresh visualization layer in a subsequent ADR.

---

## 2. Decision Drivers & Constraints

1. **Clean-Slate Architecture**:
   - Remove dead UI modules, Chainlit scaffolding, and stale runtime dependencies (`chainlit`, `pyvis`) rather than building adapters on top of incompatible legacy code.
2. **Zero Impact on Active Subsystems**:
   - Retain active modules still exercised by the test suite, specifically [`synthesis/`](../../synthesis/__init__.py) (`FinancialAnalysisEngine`, `GroundedSynthesizer`, `EvaluationHarness`) and [`curation/active_learning/visualizer.py`](../../curation/active_learning/visualizer.py) (`DataMapVisualizer`), as well as historical ADRs under [`docs/decisions/`](index.md).
3. **Verifiable Cleanup via Repository-Wide References**:
   - Ensure zero broken imports across `tests/`, `tools/`, `retrieval/`, `ingestion/`, and Docker/Compose manifests after removal.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Refactor `interactive/app.py` In-Place**:
  - *Rationale for Rejection*: Inherits Chainlit boilerplate, legacy `HybridGraphRAGEngine` coupling, and generic unstyled triple graphs poorly suited for buffer timelines, severity states, or offline self-contained reports.
- **Alternative B: Leave `interactive/` Dormant in the Repository**:
  - *Rationale for Rejection*: Pollutes dependency specs (`requirements.txt`), graph indexes (`graphify-out/`), and onboarding surface area with non-functional code referencing deleted paths (`rag/app.py`).

---

## 4. Decision Outcome

### 4.1 Artifacts to Remove
Decommission and delete the following obsolete UI assets:
- [`interactive/__init__.py`](../../interactive/__init__.py) and [`interactive/app.py`](../../interactive/app.py)
- [`tests/interactive/__init__.py`](../../tests/interactive/__init__.py) and [`tests/interactive/test_app.py`](../../tests/interactive/test_app.py)
- Root [`chainlit.md`](../../chainlit.md) and `.chainlit/config.toml`
- Remove `chainlit>=1.1.0` and `pyvis>=0.3.2` from [`requirements.txt`](../../requirements.txt) and remove the `.chainlit/translations/` rule from [`.gitignore`](../../.gitignore).

### 4.2 Scope Boundaries (Explicitly Preserved)
- Active retrieval, contagion, ingestion, and SFT curation modules—including [`synthesis/`](../../synthesis/__init__.py), [`curation/active_learning/visualizer.py`](../../curation/active_learning/visualizer.py), [`retrieval/reporting.py`](../../retrieval/reporting.py), and [`tools/dry_run_sec_8k_contagion.py`](../../tools/dry_run_sec_8k_contagion.py)—remain untouched.
- A follow-up ADR (**ADR-00017**) will specify the clean-slate visualization layer directly atop [`OperationalContagionReport`](../../retrieval/reporting.py) and [`ContagionTimeSeriesEventRecord`](../../retrieval/reporting.py).

---

## 5. Invariants & Guardrails

1. **Zero Dangling Imports**:
   - No Python module or test file outside deleted `tests/interactive/` may import `interactive`, `chainlit`, or `pyvis`.
2. **Regression Suite Integrity**:
   - All remaining unit and integration tests under `tests/` must pass after removal.
3. **Zero Absolute Paths Invariant**:
   - All document references remain strictly repository-relative (`../../<path>` or peer ADR links).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Eliminates obsolete Chainlit/PyVis surface area, simplifies dependencies in `requirements.txt`, and gives the upcoming visualization layer (ADR-00017) a clean architectural slate.
- **Negative / Trade-offs**:
  - Removes the legacy Chainlit conversational entrypoint; anyone needing interactive exploration waits until the replacement visualization ADR is designed and implemented (prior snapshots remain recoverable in git history).

