# YarnBall Codebase Stress-Test & Accuracy Report

**Executed:** `2026-09-09 06:06:43 UTC`  
**Environment:** Linux (x86_64) | PostgreSQL 16 (pgvector) | Memgraph 2.18+ (MAGE) | Kafka KRaft

---

## 1. Executive Summary

| Category | Benchmark / Target | Measured Score | Status |
| :--- | :--- | :--- | :--- |
| **High-Volume Scale** | 5,000 Nodes / 15,000 Edges UNWIND | Export: `1.008s`, Restore: `28.215s` | **PASSED** |
| **Data Integrity** | Zero data loss across Parquet export/import | DB Nodes: `5000`, Edges: `14490` | **PASSED** |
| **Concurrency Load** | 20 Workers Parallel Throughput | `11.14 QPS` ($p_{95}=4994.7\text{ ms}$) | **PASSED** |
| **Adversarial Security** | 25+ Malicious Cypher AST Payloads | `100.0%` Defense Rate (`25/25`) | **PASSED** |
| **Pathological Topologies** | 30-Node Clique (870 edges), 15-hop Chain | Clique Limit Clamped: `True`, Recursion Clamped: `True` | **PASSED** |
| **Path Recovery (2-Hop)** | Ground-Truth Graph Traversal Recall | `75.0%` Edge Path Recall | **PASSED** |
| **Text-to-CQL Accuracy** | EX-Acc against Golden Reference Cypher | `100.0%` Execution Accuracy | **PASSED** |
| **Negative Grounding** | Abstention on Out-of-Distribution Entities | Grounded Abstention: `True` | **PASSED** |

---

## 2. High-Volume Snapshot & UNWIND Performance

- **Nodes Exported:** `5000`
- **Edges Exported:** `14490`
- **Compressed Parquet Size:** `611.4 KB`
- **Export Latency:** `1.008s`
- **Batch Restore Latency (UNWIND):** `28.215s`
- **Data Loss:** `None (100% verified)`

---

## 3. Concurrency & Latency Profile (20 Workers)

| Metric | Measured Value |
| :--- | :--- |
| **Total Requests** | `100` |
| **Total Duration** | `8.97s` |
| **Throughput (QPS)** | `11.14 queries/sec` |
| **Median Latency ($p_{50}$)** | `1002.6 ms` |
| **95th Percentile ($p_{95}$)** | `4994.7 ms` |
| **99th Percentile ($p_{99}$)** | `5047.1 ms` |
| **Error Rate** | `0 / 100` |

---

## 4. Adversarial Cypher AST Security Rejection

- **Total Malicious Payloads Tested:** `25`
- **Blocked Payloads:** `25`
- **Bypass Vulnerabilities:** `0`
- **Defense Rate:** `100.0%`

*All mutating keywords (`DELETE`, `DETACH`, `SET`, `CREATE`, `MERGE`, `DROP`, `ALTER`, `GRANT`, `CALL dbms`) were strictly rejected at the AST parsing phase prior to database execution.*

---

## 5. Topological Ground-Truth Accuracy

- **Node Recall:** `80.0%`
- **2-Hop Path Recovery Rate:** `75.0%`
- **Text-to-CQL Execution Accuracy (EX-Acc):** `100.0%`
- **Triple Faithfulness (Anti-Hallucination):** `100.0%`
- **Negative Grounding / Abstention:** `Passed (100% verified)`

---

*Report generated automatically by `tests.stress_test_suite`.*
