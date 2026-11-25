# ADR-00040: Chapter 0 Technical Report — System Atlas & Docs Integrity Gate

- **ADR ID**: ADR-00040
- **Filename**: `ADR-00040-ch0-system-atlas-and-docs-integrity-----.md`
- **Date**: 2026-11-24 17:58 UTC
- **Status**: ACCEPTED
- **Target Subsystem**: `docs` / `pyproject.toml` / `tests`
- **Linked Commits**: [`a5f4732`](../../docs/technical-report/00-system-atlas.md), [`a5f4732`](../../README.md), [`a5f4732`](../../mkdocs.yml), [`a5f4732`](../../pyproject.toml), [`a5f4732`](../../docs/index.md), [`a5f4732`](../../docs/reference/index.md), [`a5f4732`](../../tests/docs/test_docs_integrity.py)

---

## 1. Context & Problem Statement

1. **Pre-ADR foundational knowledge gap & chronological fragmentation.** The ADR discipline ([`index.md`](index.md)) was introduced relatively late in YarnBall's lifecycle—beginning at system-integration harness verification ([`ADR-00001`](ADR-00001-sit-flow-2-event-shock-contagion-harness.md)). Major foundational subsystems—including the S&P 500 constituent builder ([`universe/`](../../universe/sp500.py)), news/market/transcript harvesters ([`ingestion/`](../../ingestion/task_runner.py)), Kafka streaming pipeline ([`streaming/`](../../streaming/consumer.py)), 5-axis formal ontology ([`ontology/`](../../ontology/schema.py)), submodular cold-start curation & Dataset Cartography (`curation/`), multi-judge evaluation (`evaluation/`), and SFT manifold balancing ([`sft_pipeline/`](../../sft_pipeline/sampler.py))—predate the ADR log entirely, while root [`README.md`](../../README.md) remains a single-line stub (`# YarnBall`). Consequently, the ADR registry is an incomplete record of the system: pre-ADR foundations lack formal narrative documentation, and post-`ADR-00001` changes exist only as chronological deltas rather than a unified present-tense model.
2. **Dual audience & blog-instrumentalizable narrative requirement.** Standard API docstrings miss the financial, statistical, and mathematical foundations of the platform, whereas abstract theory detached from code rots quickly and omits call graphs, entry points, and schemas. Furthermore, the non-programming narrative must read as a cohesive, publication-grade technical report that can be directly instrumentalized into a 5-part technical blog series (`Part 0` through `Part 4`) without rewriting from scratch.
3. **Need for a dedicated architectural spine before deep-dive chapters.** Before documenting individual pipelines in detail, both readers and subsequent chapters need a single System Atlas mapping all packages, entry points, storage schemas, Domain 1 vs. Domain 2 boundaries, and Pre-ADR vs. ADR provenance—paired with an automated `pytest` documentation drift gate.

---

## 2. Decision Drivers & Constraints

1. **Close the pre-ADR gap from ground-truth code**: recover and formalize the theoretical, mathematical, and financial design of pre-ADR subsystems directly from live source modules, tests, and domain literature rather than assuming ADRs cover the whole codebase.
2. **Publication-ready narrative + engineering reference in one artifact**: each chapter must pair a blog-ready *Domain & Quantitative Technical Report* (*What & Why*) with an *AST-Verified Software Anatomy & Call-Graph Reference* (*Where & How*).
3. **Clean 5-chapter / 5-ADR series (`Ch. 0` through `Ch. 4`)**: dedicate Chapter 0 (`ADR-00040`) to the whole-system atlas and drift-test harness so Chapters 1–4 (`ADR-00041` through `ADR-00044`) remain tightly focused on their respective pipelines without cross-chapter duplication.
4. **Forward ADR governance & mechanized zero-drift verification**: file links, cited `module:symbol` identifiers, and key numeric constants in documentation must be validated automatically in `pytest`, while any *new* architectural changes continue to be recorded as ADRs under [`docs/decisions/`](index.md).

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Inline docstring sweep + expanded `README.md` only.** Rejected. Docstrings scatter multi-module mathematical derivations and regulatory failure modes across dozens of files and cannot serve as a coherent technical report or blog narrative.
- **Alternative B: Separate "Theory Monograph" books and "Code Reference" manuals.** Rejected. Describing the same pipeline twice (once mathematically, once structurally) bloats into a repetitive multi-hundred-page book and guarantees drift between equations and variable names.
- **Alternative C: Fold the System Atlas into Chapter 1 (`4-chapter` compression).** Rejected. Overloading Chapter 1 with whole-repo package maps, Docker/FastAPI catalogs, and all-ADR provenance blurs the narrative focus of both the architecture overview post and the ingestion/temporal deep dive.
- **Alternative D: Embed full long-form chapters inside `docs/decisions/ADR-*.md` directly.** Rejected. ADRs follow a strict decision-record schema ([`template.md`](template.md)); dedicated chapters under `docs/technical-report/` keep ADRs concise while making each chapter directly portable to long-form blog posts.
- **Alternative E: Sphinx (`sphinx` + `myst-parser` + `autodoc`) for site & API generation.** Rejected. Sphinx's reStructuredText heritage and MyST `{eval-rst}` / `:py:func:` role directives introduce framework-specific syntax lock-in that pollutes raw Markdown chapters when repurposing Part A prose into blog posts; furthermore, Sphinx `autodoc` imports target Python modules at build time, requiring fragile mock registries whenever optional runtime extras (`graspologic`, `vllm`, `torch`, or live database services) are not installed.

---

## 4. Decision Outcome

1. **Establish the 5-Chapter Dual-Lens Technical Report Series (`docs/technical-report/`)**:
   - Every chapter follows a strict two-part layout:
     - **Part A — Domain & Theoretical Technical Report (*What & Why*)**: written in publication/blog-ready prose from the perspective of quantitative finance, network economics, and data science—covering real-world failure modes, a **Topic-Level UML Diagram** (Mermaid `classDiagram` capturing domain bounded contexts, mathematical state, and directional topic-to-topic transformations for high-level intuition), formal LaTeX equations, statutory/statistical rules, and a **Theory-to-Code Crosswalk Table** (`Concept / Symbol` $\rightarrow$ `Module & Function` $\rightarrow$ `Provenance (Pre-ADR Foundation or ADR-XXXXX)` $\rightarrow$ `Verification / Test`).
     - **Part B — Software Architecture, Call Graphs & Entry Points (*Where & How*)**: AST-verified Mermaid call/dependency graphs, CLI / FastAPI / Kafka entry points, SQL/Parquet/Pydantic data contracts, and module inventories.
   - Five-chapter / five-ADR roadmap:
     - **`ADR-00040` — Chapter 0**: `docs/technical-report/00-system-atlas.md` + [`README.md`](../../README.md) + `mkdocs.yml` + `docs/reference/` + `tests/docs/test_docs_integrity.py` (*Part 0: Architecture of a Point-in-Time Financial GraphRAG Engine*)
     - **`ADR-00041` — Chapter 1**: `docs/technical-report/01-pit-universe-and-ingestion.md` (*Part 1: Point-in-Time Equity Universes, Market Time & Statutory Ingestion*)
     - **`ADR-00042` — Chapter 2**: `docs/technical-report/02-ontology-linkage-and-ledger.md` (*Part 2: Formal Financial Ontology, Entity Linkage & Durable-vs-Ephemeral Graph Storage*)
     - **`ADR-00043` — Chapter 3**: `docs/technical-report/03-retrieval-contagion-and-reporting.md` (*Part 3: Hybrid GraphRAG Retrieval, Network Contagion Algebra & Provenance-Graded Reporting*)
     - **`ADR-00044` — Chapter 4**: `docs/technical-report/04-curation-and-model-foundry.md` (*Part 4: Submodular Curation, Dataset Cartography & Task-Selective Model Alignment*)

2. **Deliver Chapter 0 (`docs/technical-report/00-system-atlas.md`)**:
   - **Part A (Domain & Architectural Technical Report — Blog Part 0)**:
     1. *Why Naive RAG Fails in Institutional Finance*: temporal look-ahead leakage, survivorship and corporate-action split bias ([`ADR-00039`](ADR-00039-sp500-pit-stock-split-corporate-actions-.md)), hallucinated causal edges, unbounded in-memory graph bloat, silent outage fallbacks, and conflating physical supply-chain shocks with portfolio trading rules.
     2. *The Domain 1 vs. Domain 2 Architectural Wall*: why YarnBall strictly separates operational supply-chain contagion intelligence (Domain 1) from downstream portfolio construction and order sizing (Domain 2) ([`ADR-00010`](ADR-00010-directional-basket-peer-cluster-mapping-.md), [`ADR-00011`](ADR-00011-operational-contagion-report-schema-----.md), [`ADR-00034`](ADR-00034-single-bi-read-facade-and-pipeline-merge.md)).
     3. *End-to-End Quantitative & Systems Synthesis (with Topic-Level UML Diagram)*: unified walkthrough and Mermaid `classDiagram` of the interacting domain topics—`PiT Market Universe & Time` $\rightarrow$ `Statutory & Event Ingestion` $\rightarrow$ `5-Axis Financial Ontology` $\rightarrow$ `Bitemporal Knowledge Ledger` $\rightarrow$ `Hybrid Retrieval & Contagion Algebra` $\rightarrow$ `Domain-1 Intelligence & Audit`, closed by the `Curation & Model Alignment Foundry` feedback loop—providing high-level conceptual intuition unclouded by low-level utility imports.
     4. *Pre-ADR Foundations & Thematic ADR Provenance Atlas*: capability-grouped map explicitly distinguishing pre-`ADR-00001` subsystems from ADR-governed evolutions (`ADR-00001`–`ADR-00039`) and supersede/refinement chains.
   - **Part B (Software Anatomy, Package Graph, Entry Points & Schemas)**:
     - Complete 13-package runtime/pipeline inventory (`contagion`, `curation`, `embedding`, `explorer`, `ingestion`, `knowledge_graph`, `ontology`, `retrieval`, `sft_pipeline` [including `sft_pipeline/evaluation/` and `sft_pipeline/serving/`], `streaming`, `synthesis`, `tools`, `universe`) classified by **functional orthogonality vs. AST import coupling**: contrasting Part A's clean topic-level UML flow against the physical AST dependency graph—highlighting zero-dependency algorithmic leaves (`contagion`, `curation`), pure consumer shells (`explorer`, `synthesis`), and the 12 bidirectional package-import couplings induced by shared-kernel modules ([`sft_pipeline/serving/model_client.py`](../../sft_pipeline/serving/model_client.py), [`sft_pipeline/rejection_queue.py`](../../sft_pipeline/rejection_queue.py), [`retrieval/temporal.py`](../../retrieval/temporal.py), [`tools/init_db.py`](../../tools/init_db.py)).
     - Master **Entry-Point & Service Catalog**: every FastAPI endpoint ([`retrieval/service.py`](../../retrieval/service.py), [`explorer/workbench_api.py`](../../explorer/workbench_api.py)), Kafka consumer/scheduler ([`streaming/consumer.py`](../../streaming/consumer.py), [`knowledge_graph/memgraph_consumer.py`](../../knowledge_graph/memgraph_consumer.py), [`tools/scheduler.py`](../../tools/scheduler.py)), CLI tool (`tools/*.py`), and Docker Compose service ([`docker-compose.yml`](../../docker-compose.yml)).
     - Master **Storage & Artifact Schema Map**: every PostgreSQL table in [`tools/init_db.py`](../../tools/init_db.py) (including `sp500_stock_splits` per [`ADR-00039`](ADR-00039-sp500-pit-stock-split-corporate-actions-.md)), pure in-memory Memgraph projection + `MEMGRAPH_EVENT_TTL_DAYS` ([`ADR-00036`](ADR-00036-fail-fast-infra-and-deployable-topology-.md)), and offline JSON/Parquet snapshot artifacts (`nodes.parquet`, `edges.parquet`, `chunks.parquet`, `communities.parquet`, `sp500_constituents_historical.json`, `sp500_stock_splits_historical.json`).

3. **Deliver Concise Root [`README.md`](../../README.md)**:
   - Replace the 1-line `# YarnBall` stub with a concise project landing page: mission statement, high-level architecture diagram, navigation table for the 5-chapter technical report series and [`docs/decisions/index.md`](index.md), local verification/CLI/Docker/MkDocs quickstart commands, and data-source / Domain-1 governance summary.

4. **Adopt MkDocs Material + Static AST API Reference (`mkdocs.yml`, `docs/index.md`, `docs/reference/`, [`pyproject.toml`](../../pyproject.toml))**:
   - Add a `docs` optional dependency group (`mkdocs-material`, `mkdocstrings[python]`) in [`pyproject.toml`](../../pyproject.toml) and ensure namespace-package compatibility for `embedding/`.
   - Author `mkdocs.yml` (`docs_dir: docs`) rendering `docs/index.md`, `docs/technical-report/*.md`, `docs/decisions/*.md`, KaTeX/MathJax equations (`pymdownx.arithmatex`), Mermaid diagrams (`pymdownx.superfences`), and Griffe-backed static AST Python API reference pages under `docs/reference/` for all 13 runtime/pipeline packages without importing heavy runtime dependencies at build time.

5. **Deliver Mechanized Documentation Drift Suite (`tests/docs/test_docs_integrity.py`)**:
   - Automated `pytest` suite scanning `README.md`, `mkdocs.yml`, `docs/index.md`, `docs/reference/*.md`, and `docs/technical-report/*.md` to enforce zero host-specific paths, valid local Markdown links and `mkdocs.yml` navigation targets, valid repository file and `module.path:symbol` crosswalk references via Python AST, live constant parity (`MAX_SUBGRAPH_EDGES=25`, `EMBEDDING_DIM=1024`, `MEMGRAPH_EVENT_TTL_DAYS=90`, chunk `800`/`120`, ASC 280 `0.10`/`0.50` thresholds), and conditional `mkdocs build` verification when the `docs` extra is installed.

---

## 5. Invariants & Guardrails

1. **Zero Absolute Paths**: neither `README.md`, `mkdocs.yml`, nor any Markdown file under `docs/` may contain host-specific filesystem paths, file URI schemes, or Windows drive letters.
2. **Link, Nav & Symbol Resolution**: every relative Markdown link, `mkdocs.yml` nav entry, `mkdocstrings` target package, and explicit `module.path:symbol` crosswalk entry in `README.md` and `docs/` must resolve against the live repository tree and Python AST; enforced by `tests/docs/test_docs_integrity.py`.
3. **AST-Faithful Dependency & Call Graphs**: every Mermaid dependency or call-graph edge in Part B must correspond to a real import or function/method call in the codebase.
4. **Honest Provenance Attribution & Forward ADR Governance**: chapters must explicitly distinguish pre-ADR foundational implementations (documented directly from live source code and domain literature) from ADR-governed evolutions, without fabricating retroactive decision history; any *new* architectural modification still requires its own ADR under [`docs/decisions/`](index.md).
5. **Pure Markdown Blog Portability (Zero Directive Lock-In in Narrative Chapters)**: `docs/technical-report/*.md` must remain pure GitHub-Flavored Markdown (`$...$` / `$$...$$` math and ```` ```mermaid ```` fences only); `mkdocstrings` `:::` directives are strictly isolated under `docs/reference/` so every chapter's Part A lifts directly into external blog publishing platforms without stripping framework syntax.

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Gives the project a clean, blog-ready **Part 0: Architecture Overview** (`00-system-atlas.md`), replaces the 1-line `README.md` stub, and puts the automated `test_docs_integrity.py` harness in place before Chapters 1–4 (`ADR-00041`–`ADR-00044`) are written.
  - Keeps Chapter 1 (`ADR-00041`) purely focused on S&P 500 PiT universe, corporate split actions, `16:00 ET` market time, statutory ingestion, chunking, and gleaning.
- **Negative / Trade-offs**:
  - Renaming or deleting public modules, CLI tools, or constants referenced in `00-system-atlas.md` or `README.md` will intentionally fail `test_docs_integrity.py` until updated in the same commit.
