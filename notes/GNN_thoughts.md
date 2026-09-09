# Graph Neural Networks (GNNs) for Financial GraphRAG: Concepts & Roadmap

## 1. Motivation & Vision

While Large Language Models (LLMs) excel at natural language parsing and summarization, they often struggle with complex, multi-hop topological reasoning over interconnected market ecosystems. 

In financial markets, critical risks and opportunities manifest as chain reactions across multi-layered graphs (e.g., semiconductor supply shortages impacting component assemblers, ultimately affecting OEM earnings and downstream retailers). 

By leveraging the knowledge graphs extracted by our pipeline as structured training datasets for **Graph Neural Networks (GNNs)**, we can achieve high-precision topological inference, latent link prediction, and dynamic risk propagation that complement the LLM chatbot.

---

## 2. Core GNN Capabilities & Applications

### 2.1. Multi-Hop Risk & Sentiment Propagation (Message Passing)
- **Architecture**: Relational Graph Convolutional Networks (R-GCN) or Graph Attention Networks (GAT).
- **Mechanism**: Shocks, sentiment shifts, or regulatory changes originating at a specific entity node propagate along typed dependency edges (`SUPPLIES_TO`, `COMPETES_WITH`, `CREDITOR_OF`).
- **Use Case**: Calculating the "blast radius" or exposure score of an enterprise when a partner or competitor is hit by unforeseen events.

### 2.2. Latent Link Prediction & Discovery
- **Architecture**: Heterogeneous Link Prediction (e.g., HeteroGraphConv, DistMult, ComplEx).
- **Mechanism**: Scoring the probability of unstated or emerging edges between existing entities based on structural proximity and semantic feature similarity.
- **Use Case**: Predicting undisclosed supply-chain dependencies, covert competitive threats, or high-probability M&A targets before they are widely published.

### 2.3. Temporal & Dynamic Graph Learning (TGNs)
- **Architecture**: Temporal Graph Networks (TGN) / Temporal Graph Attention (TGAT).
- **Mechanism**: Modeling edge creation, modification, and deletion over sequential historical snapshots:
  `G(t0), G(t1), ..., G(tk)`
- **Use Case**: Tracking structural phase transitions in market sectors (e.g., shifts in market leadership, sudden cluster decoupling, or contagion dynamics).

### 2.4. Topology-Aware Graph Embeddings for GraphRAG
- **Architecture**: GraphSAGE or GNN Autoencoders.
- **Mechanism**: Combining raw text embeddings (from `sentence-transformers`) with topological neighborhood features to create structural entity embeddings:
  `h_v = AGGREGATE({h_u : u in Neighbors(v)})`
- **Use Case**: Enabling the GraphRAG retriever to perform topology-aware dense search (retrieving nodes that are structurally central or interconnected, not just lexically similar).

---

## 3. Designing the Current Pipeline for GNN Readiness

To ensure the knowledge graph constructed in Memgraph and PostgreSQL is immediately consumable by frameworks like **PyTorch Geometric (PyG)** and **Deep Graph Library (DGL)**, the following architectural conventions are maintained:

1. **Heterogeneous Typed Graph Schema**:
   - Explicit node labels: `Company`, `Executive`, `Product`, `Sector`, `RegulatoryBody`.
   - Explicit directed edge types: `SUPPLIES_TO`, `COMPETES_WITH`, `ACQUIRED`, `REGULATES`, `INVESTED_IN`.
2. **Initial Node Feature Matrices**:
   - Dense embeddings (768-dimensional `all-mpnet-base-v2` stored in pgvector) serve directly as the initial feature matrix **X** for GNN layers.
3. **Edge Attributes & Provenance**:
   - Each edge stores `sentiment_score`, `confidence`, `timestamp`, and `article_hash`, enabling weighted, temporal, and attributed message passing.
4. **Snapshot Compatibility**:
   - The graph snapshot and recovery system provides the dataset partitions needed for training, validation, and historical backtesting.

---

## 4. Integration Roadmap

```
[Memgraph / Postgres Snapshots]
             │
             ▼
[PyTorch Geometric Data Exporter] (torch_geometric.data.HeteroData)
             │
             ▼
[GNN Training & Evaluation] (Link Prediction / Node Classification)
             │
             ▼
[Inference Service / GNN Model Server]
             │
             ▼
[GraphRAG Chatbot Tool Call] ("Get GNN Risk Exposure Score for NVDA")
```

1. **Phase 1 (Data Exporter)**: Implement high-speed export from Memgraph snapshots to `torch_geometric.data.HeteroData` or Parquet edge/node lists.
2. **Phase 2 (Baseline Models)**: Train a baseline R-GCN / GAT for link prediction and node risk classification on historical market graphs.
3. **Phase 3 (Chatbot Tooling)**: Register GNN inference endpoints as callable tools within the GraphRAG chatbot engine.
