# ADR-00025: Financial Entity Placeholder Stoplist & Dynamic SEC Alias Expansion

- **ADR ID**: ADR-00025
- **Filename**: `ADR-00025-entity-stoplist-and-dynamic-cik-rollup--.md`
- **Date**: 2026-11-14 19:25 UTC
- **Status**: PROPOSED
- **Target Subsystem**: `ontology` / `knowledge_graph` / `ingestion`
- **Linked Commits**: Pending acceptance

---

## 1. Context & Problem Statement

Probing [`ontology/validation.py`](../../ontology/validation.py) (`is_generic_placeholder`) and [`ontology/guardrails.py`](../../ontology/guardrails.py) (`SUBSIDIARY_TO_PARENT_CIK_MAP`) revealed two brittle static lists when confronted with authentic SEC filings and news prose:

1. **Anonymized & Descriptive Noun Leaks in `is_generic_placeholder`**: While exact phrases in `GENERIC_NOUN_STOPLIST` (`"the supplier"`) return `True`, common SEC ASC 280 anonymized counterparty designations and indefinite partner descriptions—verified on `Customer A`, `Supplier X`, `a major foundry partner`, and `our primary OEM partner`—all returned `False`, allowing non-unique placeholder entities to enter Memgraph and collide across unrelated companies.
2. **Hardcoded 4-Pair Subsidiary Rollup Map**: `SUBSIDIARY_TO_PARENT_CIK_MAP` inside `ontology/guardrails.py` currently hardcodes only four `(subsidiary_cik -> parent_cik)` pairs (`DISH DBS`, `Mellanox`, `Audible`, `Meta`), while hundreds of Exhibit 21 subsidiaries harvested by `EdgarWorker` ([`ingestion/sec/harvester.py`](../../ingestion/sec/harvester.py)) reside in PostgreSQL `sec_subsidiaries` ([`tools/init_db.py`](../../tools/init_db.py)) without feeding back into `Tier4OntologyGate` or `EntityResolver`.

---

## 2. Decision Drivers & Constraints

1. **Prevent Cross-Filing Hub Collisions**: If Company 1 and Company 2 both disclose `"Customer A"`, merging them onto a single `"Customer A"` node in Memgraph fabricates false 2-hop contagion paths between unrelated firms.
2. **Preserve Scoped Anonymized Disclosures Where Needed Without Node Merging**: Either reject anonymized nouns as graph anchors (`is_generic_placeholder == True`) or namespace them strictly by filer (`{filer_cik}::Customer A`), never as a shared global entity.
3. **Database-Backed Subsidiary & Alias Scale**: Exhibit 21 parent-subsidiary rollup and SEC alias resolution must scale dynamically from ingested `sec_subsidiaries` / `sec_companies` tables rather than growing a hardcoded Python dict.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Expand `GENERIC_NOUN_STOPLIST` Purely With Static Phrases**:
  - *Rationale for Rejection*: Combinatorial noun phrases (`"a major [tier-1] [foundry/OEM/logistics] partner"`, `"Customer [A-Z0-9]"`, `"Vendor [1-9]"`) cannot be enumerated exhaustively by exact string matching.
- **Alternative B: Keep Hardcoding Subsidiary CIKs in `ontology/guardrails.py`**:
  - *Rationale for Rejection*: Requires code edits for every M&A closing or Exhibit 21 update and leaves live `sec_subsidiaries` data unused.

---

## 4. Decision Outcome

Upgrade placeholder detection and parent-subsidiary resolution across [`ontology/validation.py`](../../ontology/validation.py), [`ontology/guardrails.py`](../../ontology/guardrails.py), and [`knowledge_graph/entity_resolver.py`](../../knowledge_graph/entity_resolver.py):

1. **Pattern-Based Anonymized & Indefinite Entity Filtering (`is_generic_placeholder`)**:
   - Extend `is_generic_placeholder` beyond exact `GENERIC_NOUN_STOPLIST` lookups to flag:
     - Single-letter / numeric anonymized counterparties (`^(?:customer|supplier|vendor|client|partner|manufacturer|distributor|counterparty)\s+[a-z0-9]{1,2}$`),
     - Indefinite/possessive role descriptors lacking proper nouns (`^(?:a|an|the|our|one|one\s+of\s+our|several|major|primary|sole)\s+.*\b(?:supplier|customer|vendor|partner|manufacturer|foundry|distributor|client|competitor)s?\b$` when no corporate suffix or ticker/CIK signal is present), and
     - Pronouns and boilerplate subject references (`it`, `they`, `its affiliates`, `the registrant`).
2. **Dynamic Subsidiary Rollup Registry**:
   - Load `SUBSIDIARY_TO_PARENT_CIK_MAP` dynamically via a repository/DB loader reading `sec_subsidiaries` + `sec_companies` (with caching/refresh), keeping the existing static dict strictly as an offline deterministic fallback seed for unit tests.

---

## 5. Invariants & Guardrails

1. **Zero Unscoped Placeholder Hubs**: Anonymized identifiers (`Customer A`, `Supplier X`, `Vendor 1`) and generic noun phrases MUST NEVER merge as shared cross-company nodes in Memgraph.
2. **Offline Test Determinism**: `Tier4OntologyGate()` instantiated without PostgreSQL MUST gracefully use the offline seed map without throwing connection errors.
3. **Zero Absolute Paths Invariant**: All references remain strictly repository-relative (`../../<path>`).

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Prevents high-degree spurious hub nodes created by SEC ASC 280 anonymized customer/supplier disclosures.
  - Automatically propagates Exhibit 21 ownership updates from Postgres ingestion into runtime ontology gates.
- **Negative / Trade-offs**:
  - Quantitative SEC concentration disclosures referencing `"Customer A"` (e.g., >10% revenue dependency) will need explicit filer-scoped attribution if modeled later rather than bare global company nodes.
