# Implementation Plan: Graph-Aware Financial Sentiment Analysis Engine

This document specifies the technical design, architecture, and mathematical formulation for adding a **Graph-Aware Financial Sentiment Analysis Engine** to YarnBall.

---

## 1. Executive Overview

When answering sentiment questions (e.g., *"What do you feel about AAPL?"*, *"Is market sentiment on NVDA bullish or bearish?"*), the system goes beyond naive keyword matching by combining **direct article sentiment** with **multi-hop knowledge graph topological propagation**.

---

## 2. Core Architecture

```
                                  +---------------------------------------+
                                  |         User Sentiment Query          |
                                  |    "What is market sentiment on AAPL?"|
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |      Entity Extractor / Disambiguator |
                                  |       Target: AAPL (Apple Inc.)       |
                                  +-------------------+-------------------+
                                                      |
                         +----------------------------+----------------------------+
                         |                                                         |
                         v                                                         v
          +------------------------------+                          +------------------------------+
          |    Direct Sentiment Engine   |                          |  Graph Sentiment Propagation |
          |   (Postgres News Evidence)   |                          |    (Memgraph 1-2 Hop Walks)  |
          +--------------+---------------+                          +--------------+---------------+
                         |                                                         |
                         | - Aggregates recent articles                            | - SUPPLIES_TO (TSMC -> AAPL)
                         | - Extracted polarity in [-1.0, 1.0]                     | - COMPETES_WITH (MSFT, GOOG)
                         | - Conviction weight in [0.0, 1.0]                       | - PARTNERED_WITH, INVESTS_IN
                         | - Calculates S_direct                                   | - Calculates S_graph
                         |                                                         |
                         +----------------------------+----------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |       Composite Scoring Engine        |
                                  | S_composite = a*S_dir + (1-a)*S_graph |
                                  | Classification: Bullish / Bearish     |
                                  +-------------------+-------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |         Grounded LLM Synthesis        |
                                  | - Executive Sentiment Gauge           |
                                  | - Direct News Catalysts / Headwinds   |
                                  | - Supply-Chain & Partner Ripple Effects|
                                  | - Clickable Provenance Citations      |
                                  +---------------------------------------+
```

---

## 3. Mathematical Formulation

### 3.1. Direct Article Sentiment ($S_{\text{direct}}$)
Let $\mathcal{A}(e) = \{a_1, a_2, \dots, a_k\}$ be the set of recent news articles mentioning entity $e$, where each article $a_i$ has a sentiment polarity $s(a_i) \in [-1.0, +1.0]$ and confidence $c(a_i) \in [0.0, 1.0]$:

$$S_{\text{direct}}(e) = \frac{\sum_{a_i \in \mathcal{A}(e)} s(a_i) \cdot c(a_i) \cdot \lambda^{\Delta t_i}}{\sum_{a_i \in \mathcal{A}(e)} c(a_i) \cdot \lambda^{\Delta t_i}}$$

where $\lambda^{\Delta t_i}$ is an exponential time-decay factor discounting older news.

---

### 3.2. Topological Sentiment Propagation ($S_{\text{graph}}$)
Sentiment propagates along typed dependency edges in the Memgraph knowledge graph. Let $\mathcal{N}(e)$ denote the 1-hop and 2-hop neighbor entities of $e$:

$$S_{\text{graph}}(e) = \frac{\sum_{n \in \mathcal{N}(e)} w(r_{e, n}) \cdot S_{\text{direct}}(n) \cdot \text{conf}(r_{e, n})}{\sum_{n \in \mathcal{N}(e)} |w(r_{e, n})| \cdot \text{conf}(r_{e, n})}$$

#### Edge Type Weight Matrix $w(r)$:
| Relationship Type | Propagation Weight $w(r)$ | Financial Rationale |
| :--- | :--- | :--- |
| `SUPPLIES_TO` | $+0.85$ | Supplier bottlenecks or momentum directly impact downstream OEM delivery. |
| `PARTNERED_WITH` | $+0.75$ | Mutual upside on joint ventures and strategic alliances. |
| `INVESTS_IN` | $+0.70$ | Direct portfolio balance sheet exposure. |
| `SUBSIDIARY_OF` | $+0.90$ | Structural corporate ownership. |
| `COMPETES_WITH` | $-0.40$ | Competitor negative shocks can create market share tailwinds; competitor dominance poses headwinds. |

---

### 3.3. Composite Sentiment Score ($S_{\text{composite}}$)
The overall sentiment is a convex combination:

$$S_{\text{composite}}(e) = \alpha \cdot S_{\text{direct}}(e) + (1 - \alpha) \cdot S_{\text{graph}}(e)$$

*(Default baseline: $\alpha = 0.65$)*.

---

## 4. Rating Classification Scale

| Score Range ($S_{\text{composite}}$) | Sentiment Classification | Visual Gauge |
| :--- | :--- | :--- |
| $+0.50 \le S \le +1.00$ | **Strongly Bullish** | `[++++] Bullish conviction` |
| $+0.15 \le S < +0.50$ | **Moderately Bullish** | `[++] Positive bias` |
| $-0.15 \le S < +0.15$ | **Neutral / Balanced** | `[==] Mixed / No clear trend` |
| $-0.50 < S \le -0.15$ | **Moderately Bearish** | `[--] Cautious / Negative bias` |
| $-1.00 \le S \le -0.50$ | **Strongly Bearish** | `[----] High risk / Negative conviction` |

---

## 5. Implementation Roadmap (Shelved for Phase 8)

1. **`rag/sentiment_engine.py`**:
   - `FinancialSentimentEngine` class implementing direct aggregation, graph traversal, and composite scoring.
2. **`tools/sentiment_extractor.py`**:
   - LLM-assisted / FinBERT sentiment scoring on ingested article text.
3. **Chainlit UI Cards**:
   - Visual sentiment meters, catalyst breakdown accordions, and supplier contagion badges.
4. **Unit & Integration Tests**:
   - `tests/test_sentiment_engine.py` validating weighted propagation math, cycle handling, and synthesis formatting.
