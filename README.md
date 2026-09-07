# YarnBall

> **Untangling financial tall tales into grounded, multi-hop knowledge graphs.**

Financial news is full of people *spinning yarn* — and the global market is a giant, tangled ball of interconnected supply chains, competitors, acquisitions, and regulatory filings. 

**YarnBall** is an end-to-end Financial GraphRAG intelligence system that ingests noisy real-time market news, untangles duplicate entities using probabilistic linkage (Splink + DuckDB), stores structured relationships in Memgraph, and provides hallucination-free (hopefully) answers via guarded Text-to-CQL and a conversational Chainlit UI.

---

## Key Architecture & Highlights

```
  [ Finnhub Streaming & Scraping ] (Kafka Ingestion Pipeline)
                 │
                 ▼
       [ Raw Graph State ] ──────────────► Snapshot: "G_raw" (Parquet)
                 │
                 ▼
     [ Splink Entity Resolution ]
     (DuckDB + Fellegi-Sunter Probabilistic Linkage)
                 │
                 ▼
       [ Resolved Graph State ] ─────────► Snapshot: "G_resolved" (Parquet)
                 │
                 ▼
┌────────────────────────────────────────────────────────────────────────┐
│ AUTOMATED EVALUATION HARNESS (rag/eval_harness.py)                     │
│                                                                        │
│ Executes 25 golden multi-hop benchmark queries across both graphs:     │
│   1. Path Recovery Gain (% increase in valid multi-hop paths)          │
│   2. Entity Duplicate Reduction (Raw node count vs Canonical count)    │
│   3. Query Execution Rate (QER) & Non-Empty Return Rate (NER)          │
│   4. LLM-as-a-Judge Faithfulness & Grounding Score                     │
└────────────────────────────────────────────────────────────────────────┘
                 │
                 ▼
[ Guarded Text-to-CQL & Hybrid Retriever ] (Ollama qwen3:8b + pgvector)
                 │
                 ▼
[ Chainlit Conversational UI ] (Multi-turn chat, Step tracing, PyVis visualization)
```

1. **Parquet-Native Snapshot Engine (`graph/snapshot_manager.py`)**:
   - Zero-overhead point-in-time graph serialization into compressed `nodes.parquet` and `edges.parquet`.
   - High-throughput restoration into Memgraph using vectorized reads and parameterized `UNWIND` Cypher batch insertions.
   - Central PostgreSQL catalog (`graph_snapshots`) tracking historical news windows and graph topology.

2. **Splink Entity Resolution & Edge Canonicalization (`graph/entity_resolver.py`)**:
   - Resolves messy aliases (e.g., *"Apple"*, *"Apple Inc."*, *"AAPL"*) into canonical entities.
   - Normalizes symmetric relationships (`COMPETES_WITH`, `PARTNERED_WITH`) to eliminate inverse duplicate edges while maintaining bi-directional Cypher queries.

3. **Automated A/B Quality Loop (`rag/eval_harness.py`)**:
   - Evaluates $G_{\text{raw}}$ vs $G_{\text{resolved}}$ on 25 golden multi-hop financial benchmark queries to measure Path Recovery Gain and compression ratio before UI exposure.

4. **Guarded Text-to-CQL Engine (`rag/text_to_cql.py`)**:
   - Compiles natural language questions to Cypher queries with local Ollama `qwen3:8b`.
   - Enforces read-only AST safety, active schema whitelisting, query timeouts (3s), and automatic syntax error self-correction.

5. **Chainlit Conversational Interface (`rag/app.py`)**:
   - Streaming responses with expandable Cypher step-tracing accordions and interactive draggable network visualizations (`pyvis`).

---

## Quick Start

### 1. Prerequisites
- Python 3.13+
- Docker & Docker Compose
- [Ollama](https://ollama.ai/) with `qwen3:8b` (for local entity extraction and Text-to-CQL)

### 2. Setup Environment
```bash
# Clone repository
git clone https://github.com/CamiloEmiliano/YarnBall.git
cd YarnBall

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
```

### 3. Launch Core Infrastructure
```bash
# Start Memgraph, PostgreSQL (pgvector), and Kafka
docker compose up -d postgres memgraph kafka
```

### 4. Initialize Database & Run Tests
```bash
# Initialize PostgreSQL tables and pgvector extension
python tools/init_db.py

# Run unit tests
pytest tests/ -v
```

---

## Project Roadmap

- [x] **Phase 1: Ingestion Scaling & Parquet Snapshot Engine**
  - Multi-day Mag7 historical window configuration.
  - Parquet export/restore engine (`nodes.parquet`, `edges.parquet`) with parameterized `UNWIND`.
  - PostgreSQL `graph_snapshots` metadata catalog.
- [ ] **Phase 2: Splink Entity Resolution & Edge Canonicalization**
  - DuckDB + Splink Fellegi-Sunter probabilistic linkage model.
  - Entity clustering and symmetric edge canonicalization.
- [ ] **Phase 3: Automated A/B Evaluation Harness**
  - 25 multi-hop golden financial benchmark queries.
  - Statistical scorecards ($G_{\text{raw}}$ vs $G_{\text{resolved}}$).
- [ ] **Phase 4: Guarded Text-to-CQL Engine**
  - Schema-aware prompt compilation with Ollama `qwen3:8b`.
  - Read-only AST validators and self-correction loop.
- [ ] **Phase 5: Hybrid Retriever & Grounded Synthesis**
  - Dense pgvector search + multi-hop Memgraph traversal.
- [ ] **Phase 6: Chainlit Conversational UI & PyVis Visualizer**
  - Interactive multi-turn chat, Cypher tracing, and draggable graph visualizer.
- [ ] Phase 7: GNN-Readiness (Extension)
  - PyTorch Geometric export (R-GCN / GAT link prediction).

---

## License
MIT License