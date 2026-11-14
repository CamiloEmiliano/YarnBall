# ADR-00031: Iterative Predicate Sufficiency & Quarantine Convergence Protocol

- **ADR ID**: ADR-00031
- **Filename**: `ADR-00031-iterative-predicate-convergence-protocol.md`
- **Date**: 2026-11-14 23:05 UTC
- **Status**: PROPOSED
- **Target Subsystem**: `ontology` / `sft_pipeline` / `knowledge_graph` / `retrieval`
- **Linked Commits**: Pending acceptance

---

## 1. Context & Problem Statement

[`ADR-00023`](ADR-00023-unified-live-ontology-gate-and-schema---.md) closed the open-world predicate bypass and converged Memgraph writers onto a single canonical TBox ([`ontology/schema.py`](../../ontology/schema.py)) backed by strict write-time gating (`validate_graph_payload`) and quarantine logging (`RuntimeRejectionQueue`).

However, static inspection alone cannot prove that a closed predicate vocabulary is **empirically sufficient** across diverse SEC 10-K/8-K filings, financial news streams, and swapped local Qwen models:

1. Over-narrow ontologies silently starve downstream contagion/Cypher queries if recurring financial relationships get dropped at the gate without review.
2. Uncurated schema expansion in response to isolated model phrasings fragments graph edge types and hurts Text-to-CQL accuracy.
3. Because `ADR-00024` (clause-scoped negation/hedging) and `ADR-00025` (entity placeholder stoplist and dynamic CIK rollup) also sit inside the pre-write quality gate, empirical predicate convergence must run *after* those noise filters stabilize so quarantine metrics reflect genuine predicate gaps rather than upstream boilerplate noise.

---

## 2. Decision Drivers & Constraints

1. **Evidence-Driven Over Guesswork**: Ontology growth and alias additions must be driven by observed quarantine telemetry (`data/rejection_queue/`) over representative financial corpora rather than manual speculation.
2. **Model-Swap Resilience**: Upgrading to newer local Qwen weights behind [`YarnBallModelClient`](../../sft_pipeline/serving/model_client.py) requires a deterministic calibration loop that absorbs benign phrasing shifts via `PREDICATE_ALIASES` before requiring full SFT retraining.
3. **Zero Schema Bloat**: Synonyms and inverted verbs map deterministically onto existing canonical predicates (`ONTOLOGY_SCHEMA`); new predicates are added only when downstream analytical competency questions (supply-chain, ownership, credit/governance, risk exposure) require them.
4. **Sequence Dependency**: Execution of this protocol depends on `ADR-00023` (strict gate & quarantine), plus completion of `ADR-00024` and `ADR-00025` so hedged/placeholder noise does not distort predicate rejection frequencies.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Freeze `ONTOLOGY_SCHEMA` Permanently After `ADR-00023`**:
  - *Rationale for Rejection*: Risks blind spots on newly observed financial contract structures or upgraded model vocabularies without an empirical feedback channel.
- **Alternative B: Allow Dynamic Runtime Addition of Arbitrary Predicates**:
  - *Rationale for Rejection*: Reintroduces the open-world drift eliminated by `ADR-00023` and breaks [`retrieval/guardrails/schema_guardrail.py`](../../retrieval/guardrails/schema_guardrail.py) WHITELIST invariants.

---

## 4. Decision Outcome

Establish a bounded, multi-pass **Iterative Predicate Convergence Loop** executed over representative 10-K, 8-K, and news calibration batches once Wave 1 (`ADR-00023`–`ADR-00025`) is active:

1. **Calibration Sweep & Telemetry Collection**:
   - Run dry-run/replay extraction across a stratified sample of SEC 10-K/8-K filings and historical news records with strict write-time gating enabled (`LIVE_ONTOLOGY_GATE` -> [`RuntimeRejectionQueue`](../../sft_pipeline/rejection_queue.py)).
   - Aggregate rejected triples by `(raw_predicate, source_type, target_type)` frequency using `sft_pipeline/rejection_queue.py` (`stats` / dedicated calibration triage summary).
2. **Three-Bucket Triage Policy**:
   - **Bucket 1 — Deterministic Synonym / Inverse Alias**: If a recurring rejected predicate expresses an existing canonical relation under a different surface label or swapped direction (e.g., `PURCHASED`, `SUPPLIED_BY`), add a normalized mapping `(canonical_predicate, swap_direction)` to `PREDICATE_ALIASES` in [`ontology/schema.py`](../../ontology/schema.py). Zero schema bloat and zero retraining needed.
   - **Bucket 2 — Genuine Competency Gap**: If a recurring rejected predicate represents a distinct financial relationship required by downstream Text-to-CQL queries, risk reporting ([`retrieval/reporting.py`](../../retrieval/reporting.py)), or materiality tiering ([`ontology/compiler.py`](../../ontology/compiler.py)), promote it into `RelationPredicate` and `ONTOLOGY_SCHEMA` with explicit domain/range and symmetry axioms. Prompts and Cypher AST guardrails inherit the addition automatically.
   - **Bucket 3 — Off-Domain Noise**: Non-structural prose predicates (e.g., `COMPARES_TO`, `CORRELATES_WITH`) stay rejected and are exported via `RuntimeRejectionQueue.export_dpo_candidates(chosen="(none)")` for contrastive DPO refinement ([`ADR-00012`](ADR-00012-qwen-model-upgrade-and-fine-tuning-plan-.md)).
3. **Convergence & Stop Criteria**:
   - Repeat triage passes until two convergence gates hold simultaneously on an unseen holdout batch:
     1. Zero recurring unmapped predicates represent actionable financial/supply-chain competency gaps (only noise or domain-mismatch pairs hit Bucket 3).
     2. Non-noise `ONTOLOGY_PRUNING` predicate rejection rate stabilizes below a configured operational ceiling (`<= 2%` of total candidate triples per batch).

---

## 5. Invariants & Guardrails

1. **Downward-Propagating Single Registry**: All promotions in Bucket 2 MUST be applied solely in [`ontology/schema.py`](../../ontology/schema.py) so prompts, Tier-4 repair, and Cypher AST guardrails stay synchronized automatically.
2. **No Unaliased Ad-Hoc Writes During Calibration**: Calibration runs MUST inspect rejections from Memgraph write-time quarantine rather than disabling `strict=True`.
3. **Holdout Validation Before Closing Loop**: Convergence MUST be verified on an unseen validation split distinct from the tuning sample used to derive `PREDICATE_ALIASES`.
4. **Zero Absolute Paths Invariant**: All references remain strictly repository-relative (`../../<path>`).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Replaces guesswork with measurable, repeatable convergence criteria grounded in production-format data.
  - Creates a clean upgrade playbook whenever `yarnball-qwen:7b` is swapped to a newer local Qwen architecture.
  - Feeds verified contrastive DPO negatives directly from quarantine into the model foundry ([`ADR-00012`](ADR-00012-qwen-model-upgrade-and-fine-tuning-plan-.md)).
- **Negative / Trade-offs**:
  - Requires periodic human-in-the-loop review of quarantine summaries during calibration passes (or when onboarding new base model weights).
