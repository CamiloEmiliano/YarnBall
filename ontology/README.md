# YarnBall Financial Ontology Specification & Architecture

## 1. Executive Summary & Foundational Purpose

In formal Knowledge Representation (*Gruber, 1993; Studer et al., 1998; Noy & McGuinness, 2001*), an ontology is defined as **"a formal, explicit specification of a shared conceptualization."**

The `ontology/` package defines and enforces the terminological component (**TBox**) and assertion component (**ABox**) constraints for the YarnBall Financial GraphRAG platform. Rather than treating knowledge graphs as raw, unstructured triple stores, YarnBall treats the graph as a formally typed, algebraically constrained, and temporally grounded financial knowledge ontology.

---

## 2. Theoretical Grounding & Literature Foundations

The design of this package directly implements established standards from the Semantic Web (W3C RDF / OWL 2 / SHACL) and academic empirical finance:

1. **Portable Ontology Specifications (*Gruber, 1993*)**:
   - Explicit domain and range axiomatization separating entity typologies (`Company`, `Person`, `Product`) from relational predicates (`SUPPLIES_TO`, `CUSTOMER_OF`).
2. **Reified Multi-Axis Property Graphs (*Staab & Studer, 2009; ISO/IEC 39075 GQL*)**:
   - Reification of relationship edges into 5 orthogonal semantic dimensions: Typology, Predicate, Polarity, Materiality, and Temporal Provenance.
3. **Algebraic Property Axioms (Symmetry, Anti-Reflexivity, DAG Constraints)**:
   - Symmetrical relationship canonicalization ($R(a, b) \equiv R(b, a)$) for `COMPETES_WITH` and `PARTNERED_WITH`.
   - Strict Acyclicity and Anti-Reflexivity for corporate hierarchy (`PARENT_OF` / `SUBSIDIARY_OF`), verified via DFS cycle breaking.
4. **Identity Axiomatization & Probabilistic Entity Resolution (*Christen, 2012 - Data Matching*)**:
   - Disambiguation of surface variants into unique persistent SEC CIK identifiers via probabilistic linkage and Jaro-Winkler string metrics (`owl:sameAs`).
5. **Frame Semantics & Lexical Trigger Mining (*Fillmore, 1982*)**:
   - Extraction of directional market sentiment and supply chain disruption cues from financial discourse with syntactic negation boundary handling.

---

## 3. Package Architecture & Module Decomposition

```
graphrag_finance/ontology/
├── __init__.py          # Unified namespace exporting core classes, schemas, and validators
├── README.md            # Formal architectural specification and literature references
├── schema.py            # Concept typologies, closed relational predicates, and domain/range axioms
├── reification.py       # 5-Axis AnnotatedTriple model (Polarity, Materiality, Temporal Provenance)
├── validation.py        # SHACL-style domain/range checking, DAG ownership cycle breaker, stoplist guards
├── alignment.py         # Entity normalization, symmetric edge canonicalization, Jaro-Winkler linkage
└── compiler.py          # Lexical trigger mining, negation-boundary handling, OpenCypher DSL compiler
```

---

## 4. The 5-Axis Relational Reification Model

Every extracted financial edge is reified across five orthogonal dimensions:

- **Axis 1 (Entity Typology & Grounded Identifier)**:
  - Source and Target entities resolve to closed classes: `Company`, `Subsidiary`, `Person`, `RegulatoryBody`, `Product`, `CommodityRisk`.
  - Grounded against official SEC CIK and ticker registries.
- **Axis 2 (Relational Ontology)**:
  - Closed predicate vocabulary: `SUPPLIES_TO`, `CUSTOMER_OF`, `PARTNERED_WITH`, `COMPETES_WITH`, `PARENT_OF`, `SUBSIDIARY_OF`, `LICENSES_FROM`, `DEFAULTED_ON`, `EXPOSED_TO_RISK`.
- **Axis 3 (Directional Sentiment & Economic Polarity)**:
  - `EXPANDING_BULLISH` (+1): Commercial expansion, capacity agreements, partnership growth.
  - `NEUTRAL_STABLE` (0): Baseline recurring operations, standard cross-licensing.
  - `CONTRACTING_BEARISH` (-1): Volume cutbacks, margin compression, contract cancellations.
  - `DISRUPTIVE_SHOCK` (-2): Defaults, bankruptcies, export bans, litigation terminations.
- **Axis 4 (Financial Materiality & Criticality)**:
  - `CRITICAL_TIER_1`: Under US GAAP ASC 280 (>10% revenue customer disclosures), single-source foundries, Exhibit 21 wholly-owned subsidiaries.
  - `MATERIAL_TIER_2`: Major multi-year contracts, standard patent cross-licensing.
  - `COMMODITY_TIER_3`: Interchangeable off-the-shelf vendor relationships.
- **Axis 5 (Temporal Provenance & Lifecycle)**:
  - State: `ACTIVE_CURRENT` or `TERMINATED`.
  - Validity window: `valid_from` and `valid_to` ISO dates.
  - Source provenance: `SEC_EXHIBIT_21` (confidence: 1.00), `SEC_10K_ITEM1` (0.98), `SEC_8K` (0.95), `FINANCIAL_NEWS_VERIFIED` (0.75).

---

## 5. Usage Example

```python
from ontology import (
    AnnotatedTriple,
    validate_and_orient_triple,
    canonicalize_symmetric_edge,
    is_generic_placeholder,
    classify_polarity_from_text,
)

# 1. Ontological specificity check
assert is_generic_placeholder("the company") is True
assert is_generic_placeholder("Apple Inc.") is False

# 2. Domain & range checking with auto-orientation
validated = validate_and_orient_triple(
    src_id="AAPL", src_label="Company",
    tgt_id="Tim Cook", tgt_label="Person",
    rel_type="CEO_OF",
)
# Automatically inverts to: ("Tim Cook", "Person", "AAPL", "Company", "CEO_OF")

# 3. Symmetric relationship canonicalization
src, tgt = canonicalize_symmetric_edge("TSMC", "Intel", "COMPETES_WITH")
# Produces: ("Intel", "TSMC") ordered alphabetically

# 4. Constructing reified 5-axis triple and OpenCypher DSL
triple = AnnotatedTriple(
    source_name="TSMC",
    source_type="Company",
    source_ticker="TSM",
    target_name="Apple Inc.",
    target_type="Company",
    target_ticker="AAPL",
    rel_type="SUPPLIES_TO",
    polarity="NEUTRAL_STABLE",
    materiality="CRITICAL_TIER_1",
)
cypher_str = triple.to_cypher_dsl()
```
