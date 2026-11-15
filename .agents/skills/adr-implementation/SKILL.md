---
name: adr-implementation
description: Mandatory caller-fanout inspection, two-commit backfill discipline, and full-corpus verification workflow when implementing or modifying Architecture Decision Records (ADRs) and shared interfaces.
---

# ADR Implementation & Interface Fanout Discipline

Follow this procedural checklist on every ADR implementation or shared interface change to prevent silent omissions across documentation and runtime callers.

## 1. Pre-Edit Caller & Reference Fanout Sweep

Before modifying any shared function signature, parameter semantics, ADR filename, or module dependency:

1. Search every caller and test referencing the symbol:
   - Use `grep -rn "<symbol>"` or query `.venv/bin/graphify query "<symbol>"`.
   - Inspect **all** runtime callers across `contagion/`, `curation/`, `embedding/`, `explorer/`, `ingestion/`, `knowledge_graph/`, `ontology/`, `retrieval/`, `sft_pipeline/`, `streaming/`, `synthesis/`, `tools/`, and `universe/`.
2. When adding optional gating or provenance arguments (such as `source_text`), update **every** production call site in the same commit rather than relying on defaults alone.
3. If renaming or creating an ADR under `docs/decisions/`, verify the exact **53-character** filename (`ADR-XXXXX-<40-char-slug>.md`) and search existing ADRs (`grep -rn "ADR-XXXXX" docs/decisions/`) so no peer ADR references an outdated slug.

## 2. Two-Step ADR Acceptance & Commit Backfill Sequence

`tests/docs/test_adr_integrity.py` strictly forbids `Pending` commit text on any `ACCEPTED` ADR. Follow this exact two-commit sequence:

1. **Implementation Commit**:
   - Keep the new/updated ADR header `- **Status**: PROPOSED` and `docs/decisions/index.md` status `` `PROPOSED` `` with `Pending acceptance`.
   - Run all verification checks (Section 3) so all tests pass cleanly.
   - Commit implementation code, tests, and the `PROPOSED` ADR. Record the resulting 7-character commit hash (`git rev-parse --short=7 HEAD`).
2. **Backfill Commit**:
   - Update the ADR `- **Status**: ACCEPTED` and `- **Linked Commits**:` with clickable Markdown commit links (``[`<hash>`](../../<modified-file>)``).
   - Update the matching row in `docs/decisions/index.md` to `` `ACCEPTED` `` with hyperlinked commit links (``[`<hash>`](../../<modified-file>)``).
   - Re-run `.venv/bin/pytest tests/docs/test_adr_integrity.py` to confirm parity and git hash resolution pass.
   - Commit the backfill update (never run `git push`).

## 3. Mandatory Unified Verification Gate

Before reporting completion of any phase or ADR:

1. Run `.venv/bin/mypy` (full-corpus static check defined in `mypy.ini`).
2. Run `.venv/bin/pytest --ignore=tests/knowledge_graph/test_embedding_store.py -v` (or include `test_embedding_store.py` when Postgres is running).
3. Run `.venv/bin/graphify update .` after Python code modifications so `graphify-out/` stays synchronized.
4. Confirm zero absolute host paths (`file:///`, `/home/`, `/Users/`) appear in modified files.
