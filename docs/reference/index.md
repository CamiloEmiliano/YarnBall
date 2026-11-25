# Python API Reference Overview

These reference pages expose static AST signatures, type annotations, and docstrings across all **13 YarnBall runtime and pipeline packages** via `mkdocstrings[python]` (Griffe) without importing runtime database or GPU dependencies during site builds.

> **Blog Portability Note ([ADR-00040](../decisions/ADR-00040-ch0-system-atlas-and-docs-integrity-----.md) Invariant #5)**: All `mkdocstrings` `:::` directives are strictly isolated within `docs/reference/` so narrative chapters in `docs/technical-report/` remain pure GitHub-Flavored Markdown.

## Package-to-Reference Map

| Reference Section | Covered Top-Level Packages | Corresponding Report Chapter |
| :--- | :--- | :---: |
| [Universe, Ingestion & Streaming](ingestion-and-universe.md) | `universe`, `ingestion`, `streaming` | Ch. 1 ([ADR-00041](../decisions/ADR-00041-ch1-pit-universe-and-statutory-ingestion.md)) |
| [Ontology, Embeddings & Ledger](ontology-and-ledger.md) | `ontology`, `embedding`, `knowledge_graph` | Ch. 2 ([ADR-00042](../decisions/ADR-00042-ch2-ontology-entity-linkage-and-ledger--.md)) |
| [Contagion, Retrieval & Explorer](retrieval-and-contagion.md) | `contagion`, `retrieval`, `synthesis`, `explorer` | Ch. 3 ([ADR-00043](../decisions/ADR-00043-ch3-retrieval-contagion-and-reporting---.md)) |
| [Curation, SFT Foundry & Tools](curation-and-foundry.md) | `curation`, `sft_pipeline`, `tools` | Ch. 4 ([ADR-00044](../decisions/ADR-00044-ch4-curation-cartography-model-foundry--.md)) & Ch. 0 ([System Atlas](../technical-report/00-system-atlas.md)) |
