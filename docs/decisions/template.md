# ADR-[00000]: [Title of Decision]

<!--
STRICT FILENAME CONVENTION:
- Format: ADR-[5_DIGIT_ID]-[40_CHARACTER_SLUG].md
- Total Filename Length: Exactly 53 characters (including '.md')
- Slug Specification: Exactly 40 characters using lowercase alphanumeric and hyphens.
- Padding Rule: If the natural slug is under 40 characters, pad with trailing hyphens ('-').
Example: ADR-00001-record-architecture-decisions-in-repo--.md
-->

- **ADR ID**: ADR-[00000]
- **Filename**: `ADR-[00000]-[exact-40-character-slug-padded-w-hyphen].md`
- **Date**: YYYY-MM-DD
- **Status**: [PROPOSED | ACCEPTED | SUPERSEDED BY ADR-XXXXX | DEPRECATED]
- **Target Subsystem**: [`universe` | `ontology` | `knowledge_graph` | `retrieval` | `sft_pipeline` | `contagion`]
- **Linked Commits**: [`commit-hash`]

---

## 1. Context & Problem Statement
What was the specific operational, mathematical, or architectural dilemma? What prompted this decision?

## 2. Decision Drivers & Constraints
What were the non-negotiable constraints (e.g., broker routing APIs, regulatory data availability, survivorship bias, execution latency)?

## 3. Considered Alternatives & Rejection Rationale
- **Alternative A**: Description. Why was this rejected?
- **Alternative B**: Description. Why was this rejected?

## 4. Decision Outcome
What concrete architecture, pattern, or policy was chosen, and how is it implemented?

## 5. Invariants & Guardrails
What immutable rules or invariants are established by this decision that future code, pipelines, and tests must never violate?

## 6. Consequences & Downstream Impact
- **Positive / Guarantees**: What capabilities, safety guarantees, or architectural clarity are gained?
- **Negative / Trade-offs**: What operational costs, complexity, or performance trade-offs are accepted?


