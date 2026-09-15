# YarnBall Active Session Checkpoint

## Project State & Context
- **Workspace**: `/home/caspe/practice/rag_samples/graphrag_finance`
- **Master Roadmap**: [PLAN_OF_ACTION.md](file:///home/caspe/practice/rag_samples/graphrag_finance/PLAN_OF_ACTION.md)
- **Phase 2 Design**: [phase2_entity_resolution_design.md](file:///home/caspe/.gemini/antigravity-ide/brain/e461f87a-63c3-4226-8b4c-61de3eb9e162/phase2_entity_resolution_design.md)

## Key Architecture & Design Decisions
1. **Single Modular Repo**: Ingestion, Memgraph storage, Splink entity resolution, evaluation harness, Text-to-CQL, and Chainlit UI all reside in this repository.
2. **Automated A/B Quality Loop**: Evaluates G_raw vs G_resolved on 25 golden multi-hop benchmark queries (`rag/eval_harness.py`) to measure Path Recovery Gain, compression ratio, and grounding fidelity.
3. **Conversational UI**: Chainlit app with streaming tokens, Cypher step-tracing accordions, PyVis interactive network visualization, and snapshot hot-swapping.
4. **Text-to-CQL Guardrails**: AST/Regex read-only enforcement, schema whitelisting, resource limits (`LIMIT 50`, 3s timeout), and self-correction loop with Ollama `qwen3:8b`.
5. **Phase 7 (GNN Extension)**: PyTorch Geometric (R-GCN / GAT) reserved as a future module once independent reading is complete.

## Next Immediate Action on Resume
- **Core MVP (Phases 1–6) Complete & Verified (33/33 Unit Tests Passing)**:
  - All core layers implemented: Ingestion & Parquet Snapshots (`graph/snapshot_manager.py`), Splink Entity Resolution (`graph/entity_resolver.py`), Automated A/B Evaluation (`rag/eval_harness.py`), Guarded Text-to-CQL (`rag/text_to_cql.py`), Hybrid Retriever (`rag/hybrid_retriever.py`), and Chainlit Conversational UI (`rag/app.py`).
  - To launch the conversational UI locally: `chainlit run rag/app.py -w --port 8000`.
  - Future Phase 7 (GNN Extension): PyTorch Geometric (R-GCN / GAT) reserved once independent reading is complete.

## Resume Command
When starting a new session, simply prompt:
> **"recover chat"** or **"continue from checkpoint"**
