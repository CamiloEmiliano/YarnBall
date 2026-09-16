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

### 3.1. Direct Article Sentiment (S_direct)
Let A(e) = {a1, a2, ..., ak} be the set of recent news articles mentioning entity e, where each article a_i has a sentiment polarity s(a_i) in [-1.0, +1.0] and confidence c(a_i) in [0.0, 1.0]:

```text
S_direct(e) = Sum(s(a_i) * c(a_i) * lambda^(delta_t_i)) / Sum(c(a_i) * lambda^(delta_t_i))
```

where lambda^(delta_t_i) is an exponential time-decay factor discounting older news.

---

### 3.2. Topological Sentiment Propagation (S_graph)
Sentiment propagates along typed dependency edges in the Memgraph knowledge graph. Let N(e) denote the 1-hop and 2-hop neighbor entities of e:

```text
S_graph(e) = Sum(w(r_e,n) * S_direct(n) * conf(r_e,n)) / Sum(|w(r_e,n)| * conf(r_e,n))
```

#### Edge Type Weight Matrix w(r):
| Relationship Type | Propagation Weight w(r) | Financial Rationale |
| :--- | :--- | :--- |
| `SUPPLIES_TO` | +0.85 | Supplier bottlenecks or momentum directly impact downstream OEM delivery. |
| `PARTNERED_WITH` | +0.75 | Mutual upside on joint ventures and strategic alliances. |
| `INVESTS_IN` | +0.70 | Direct portfolio balance sheet exposure. |
| `SUBSIDIARY_OF` | +0.90 | Structural corporate ownership. |
| `COMPETES_WITH` | -0.40 | Competitor negative shocks can create market share tailwinds; competitor dominance poses headwinds. |

---

### 3.3. Composite Sentiment Score (S_composite)
The overall sentiment is a weighted combination:

```text
S_composite(e) = alpha * S_direct(e) + (1 - alpha) * S_graph(e)
```

*(Default baseline: alpha = 0.65)*.

---

## 4. Rating Classification Scale

| Score Range (S_composite) | Sentiment Classification | Visual Gauge |
| :--- | :--- | :--- |
| +0.50 <= S <= +1.00 | **Strongly Bullish** | `[++++] Bullish conviction` |
| +0.15 <= S < +0.50 | **Moderately Bullish** | `[++] Positive bias` |
| -0.15 <= S < +0.15 | **Neutral / Balanced** | `[==] Mixed / No clear trend` |
| -0.50 < S <= -0.15 | **Moderately Bearish** | `[--] Cautious / Negative bias` |
| -1.00 <= S <= -0.50 | **Strongly Bearish** | `[----] High risk / Negative conviction` |

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
