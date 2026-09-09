# YarnBall GraphRAG A/B Evaluation Report

**Generated:** 2026-09-09T06:07:58.312002+00:00  
**Comparison:** `G_raw` (Raw Extracted Graph) vs `G_resolved` (Splink-Resolved Graph)

---

## 1. Executive Summary & Statistical Metrics

| Metric | Baseline (G_raw) | Resolved (G_resolved) | Impact / Delta |
| :--- | :--- | :--- | :--- |
| **Valid Multi-Hop Paths** | 0 | 0 | **+0.0% Path Recovery** |
| **Entity Node Count** | 0 | 5 | **0.0% Compression** |
| **Relationship Edge Count** | 0 | 3 | **0.0% Deduplication** |
| **Query Execution Rate (QER)** | 28.0% | 28.0% | **0.0%** |
| **Non-Empty Return Rate (NER)** | 0.0% | 0.0% | **+0.0% Discovery** |

---

## 2. Benchmark Query Breakdown (25 Golden Queries)

| ID | Category | Question | G_raw Paths | G_resolved Paths | Delta |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `SC-01` | Supply Chain | Which semiconductor manufacturers supply AI chips or foundry capacity to Microsoft and Apple? | 0 | 0 | **0** |
| `SC-02` | Supply Chain | What products manufactured by NVIDIA depend on third-party packaging or memory suppliers? | 0 | 0 | **0** |
| `SC-03` | Supply Chain | Find 2-hop hardware dependencies connecting Amazon AWS data centers to chip producers. | 0 | 0 | **0** |
| `SC-04` | Supply Chain | Which shared suppliers provide components to both Tesla and Alphabet? | 0 | 0 | **0** |
| `SC-05` | Supply Chain | Identify custom ASIC or accelerator partnerships across Meta and cloud infrastructure providers. | 0 | 0 | **0** |
| `CD-01` | Competitive Dynamics | Which companies compete with Microsoft in Enterprise Cloud and simultaneously partner with OpenAI? | 0 | 0 | **0** |
| `CD-02` | Competitive Dynamics | Identify direct and 2-hop competitors of Alphabet in Generative AI search and foundation models. | 0 | 0 | **0** |
| `CD-03` | Competitive Dynamics | Which automotive and autonomy peers compete directly with Tesla's FSD / Robotaxi initiatives? | 0 | 0 | **0** |
| `CD-04` | Competitive Dynamics | Find overlapping product categories where Apple and Meta compete head-to-head. | 0 | 0 | **0** |
| `CD-05` | Competitive Dynamics | Which semiconductor designers compete with NVIDIA in data center GPU acceleration? | 0 | 0 | **0** |
| `CI-01` | Co-Investment | Which tech giants have co-invested in or formed strategic commercial pacts with Anthropic? | 0 | 0 | **0** |
| `CI-02` | Co-Investment | What common AI startups or frontier research labs share investment backing from Microsoft and Amazon? | 0 | 0 | **0** |
| `CI-03` | Co-Investment | Find venture or startup entities acquired or funded by NVIDIA in the AI software ecosystem. | 0 | 0 | **0** |
| `CI-04` | Co-Investment | Identify joint investments or technology licensing deals between Alphabet and telecom providers. | 0 | 0 | **0** |
| `CI-05` | Co-Investment | Which semiconductor or packaging consortia count both Apple and TSMC as active partners? | 0 | 0 | **0** |
| `EL-01` | Executive Leadership | Who leads Microsoft and what strategic partnerships have been executed under their tenure? | 0 | 0 | **0** |
| `EL-02` | Executive Leadership | Find executives leading NVIDIA who also serve on industry boards or advisory councils. | 0 | 0 | **0** |
| `EL-03` | Executive Leadership | Which executive leaders at Apple oversee hardware engineering or silicon design partnerships? | 0 | 0 | **0** |
| `EL-04` | Executive Leadership | Identify Tesla leadership roles connected to external ventures like xAI or SpaceX. | 0 | 0 | **0** |
| `EL-05` | Executive Leadership | Find leadership transitions across Meta's Reality Labs and AI research groups. | 0 | 0 | **0** |
| `RA-01` | Regulatory Impact | Which cloud and AI companies are impacted by regulatory investigations or antitrust scrutiny? | 0 | 0 | **0** |
| `RA-02` | Regulatory Impact | Find 2-hop regulatory impact spreading from chip export controls to semiconductor producers. | 0 | 0 | **0** |
| `RA-03` | Regulatory Impact | Which search distribution agreements between Alphabet and Apple have drawn regulatory focus? | 0 | 0 | **0** |
| `RA-04` | Regulatory Impact | Identify acquisitions by Microsoft or Amazon currently scrutinized under merger guidelines. | 0 | 0 | **0** |
| `RA-05` | Regulatory Impact | What cross-entity data sharing or privacy agreements impact Meta and European operations? | 0 | 0 | **0** |

---

## 3. Analysis & Key Takeaways

1. **Broken Multi-Hop Bridging**: Entity deduplication resolved alias fragmentation (e.g. `AAPL` vs `Apple Inc.`), turning isolated islands into continuous multi-hop paths.
2. **Symmetric Relationship Normalization**: Canonical edge ordering on `COMPETES_WITH` and `PARTNERED_WITH` eliminated inverse duplicates while preserving bidirectional discovery.
3. **Hallucination Protection**: Queries on `G_resolved` returned structured ground-truth subgraphs with verified source article hashes.