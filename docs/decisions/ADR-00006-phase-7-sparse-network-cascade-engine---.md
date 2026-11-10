# ADR-00006: Phase 7 Sparse Network Cascade Engine

<!--
STRICT FILENAME CONVENTION:
- Format: ADR-[5_DIGIT_ID]-[40_CHARACTER_SLUG].md
- Total Filename Length: Exactly 53 characters (including '.md')
- Slug Specification: Exactly 40 characters using lowercase alphanumeric and hyphens.
- Padding Rule: If the natural slug is under 40 characters, pad with trailing hyphens ('-').
Filename: ADR-00006-phase-7-sparse-network-cascade-engine---.md (53 chars)
-->

- **ADR ID**: ADR-00006
- **Filename**: `ADR-00006-phase-7-sparse-network-cascade-engine---.md`
- **Date**: 2026-11-10 03:32
- **Status**: PROPOSED
- **Target Subsystem**: `contagion` / `retrieval`
- **Linked Commits**: *Pending implementation*

---

## 1. Context & Problem Statement

In institutional financial supply networks, operational and geopolitical disruptions (e.g., semiconductor foundry halts, critical mineral export bans, logistics strikes) do not propagate in isolated linear chains. Instead, they diffuse through multi-branch convergent topologies (diamond networks) and circular supply dependencies.

During the Phase 5 Adversarial Torture Test (`TT-02A`), the 7B language model (`yarnball-qwen:7b`) was presented with a convergent diamond scenario:
- Two parallel Tier-2 suppliers (Supplier A with 40% capacity exposure, Supplier B with 45% capacity exposure) both experienced simultaneous disruptions.
- Both suppliers converged on a single focal hardware manufacturer (Customer C), resulting in a combined, catastrophic 85% capacity contraction.

The language model failed decisively, achieving a consensus pass rate of only **1/9 (11%)**. The LLM evaluated each branch in isolated cognitive silos (*"Supplier A has a 40% disruption which is manageable; Supplier B has a 45% disruption which is manageable"*), completely failing to compute the additive arithmetic sum.

Language models exhibit persistent mode-bleed, hallucinations, and attention diffusion when required to calculate multi-hop graph arithmetic. Relying on an LLM to calculate multi-branch network flux produces dangerously uncalibrated risk assessments and flawed portfolio hedging directives.

---

## 2. Decision Drivers & Constraints

1. **Deterministic Additive Diamond Compounding**:
   - When multiple upstream suppliers fail simultaneously and converge on a single downstream customer, incoming capacity losses must sum additively rather than being evaluated in isolation.
2. **Strict Symbolic Division of Labor**:
   - Mathematical graph shock diffusion must be computed entirely by a deterministic, symbolic Python algorithm running in O(K * |E|) time before prompting the language model. The LLM must receive exact pre-computed numbers and focus exclusively on qualitative narrative synthesis and options overlay structuring.
3. **Inventory Buffer Horizons & Temporal Delays**:
   - Downstream finished goods and component inventory buffers delay shock onset dates by exactly B calendar days and absorb disruption duration:
     Unbuffered_Duration = max(0, Disruption_Duration - Buffer_Days).
4. **Cycle Damping & Guaranteed Loop Termination**:
   - Circular supply dependencies (e.g. A -> B -> C -> A) must be detected and dampened geometrically (gamma = 0.5 attenuation per cycle traversal) with a strict maximum iteration cap K, guaranteeing algorithm termination.
5. **No LaTeX Formatting (`.agents/rules/style.md`)**:
   - All mathematical formulas and variables must use plain text and Unicode characters (e.g., delta, gamma, arrows).
6. **Ultra-Low Latency Budget**:
   - The cascade calculation must execute purely in-memory in < 10 milliseconds for subgraphs up to 100 nodes and 500 edges.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Qualitative LLM Prompting with Chain-of-Thought (CoT)**:
  - *Rejected*: Torture testing TT-02A proved that even with <think> reasoning tokens, 7B models fail on parallel flux addition (11% pass rate). Prompt engineering cannot reliably overcome arithmetic limitations on multi-branch graphs.
- **Alternative B: End-to-End Deep Graph Neural Network (GNN: GIN/RGAT)**:
  - *Rejected*: Full deep GNNs introduce significant inference latency, require GPU compute, suffer from over-smoothing beyond 3-4 hops, and act as black boxes lacking the auditable mathematical transparency demanded by institutional compliance.
- **Alternative C: Unattenuated All-Pairs Shortest Path (BFS)**:
  - *Rejected*: Ignores real-world operational absorption where finished goods stockpiles buffer shocks.

---

## 4. Decision Outcome

We establish **Phase 7** by creating a dedicated package `contagion/` centered around the `SparseNetworkCascadeEngine` in `contagion/cascade.py` with the following architectural components:

1. **Graph Data Structures**:
   - `CascadeNode`: Represents an entity with baseline revenue/capacity, active buffers, and accumulated shock state.
   - `CascadeEdge`: Directed relationship carrying `exposure_pct`, `buffer_days`, and `materiality_tier`.
   - `CascadeGraph`: Directed graph representation supporting fast in-degree indexing and cycle detection.
2. **Deterministic Additive Flux Algorithm**:
   - For each target node v, incoming flux from upstream parents u is calculated as:
     Absorption_Ratio = max(0.0, 1.0 - (Buffer_Days(u -> v) / Duration(u)))
     Flux(u -> v) = Shock(u) * (Exposure_Pct(u -> v) / 100.0) * Absorption_Ratio
     Total_Accumulated_Shock(v) = min(100.0, sum(Flux(u -> v) for u in In(v)))
3. **Temporal Shock Propagation**:
   - Effective Onset Date at node v is delayed by the buffer:
     Effective_Onset(v) = Onset(u) + Buffer_Days(u -> v)
   - Unbuffered Duration:
     Unbuffered_Duration(v) = max(0, Duration(u) - Buffer_Days(u -> v))
4. **Cycle Damping Protocol**:
   - Traversal tracks path history. When an already-visited node is re-encountered in a cycle, its transmitted shock is attenuated by damping factor gamma = 0.5. Iterations terminate when delta_shock < 0.01 or iteration count reaches K = 5.
5. **LLM Handshake Interface**:
   - Outputs a structured `ConvergentShockSummary` object injected into the prompt verbalizer in `retrieval/pipeline.py` for Tasks D and E.

---

## 5. Invariants & Guardrails

1. **Additive Diamond Convergence Invariant**:
   - For two parallel suppliers A and B converging on C with zero buffer:
     Shock(C) == min(100.0, Flux(A -> C) + Flux(B -> C)).
     Parallel losses must compound additively with zero branch omission.
2. **Buffer Delay & Absorption Invariant**:
   - An active inventory buffer of B days strictly shifts downstream shock onset by B calendar days:
     Effective_Onset(v) == Event_Date + B days.
   - If Buffer_Days >= Disruption_Duration, unbuffered duration must equal 0, and effective shock collapses to the baseline residual friction (0.05).
3. **Cycle Termination Invariant**:
   - In graphs containing directed cycles, the cascade engine must terminate in at most K = 5 iterations with total flux strictly bounded and non-divergent.
4. **Monotonic Path Attenuation Invariant**:
   - Along any single linear (non-convergent) path, propagated shock intensity must strictly decrease:
     Shock(Hop N+1) < Shock(Hop N).
5. **Zero External Network Dependency**:
   - The engine must execute 100% offline using pure Python and NumPy with zero live network or database calls.

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Permanently eliminates the 1/9 parallel diamond compounding failure mode discovered in Phase 5 torture testing.
  - Offloads arithmetic computation from the probabilistic LLM to a deterministic O(K * |E|) graph algorithm.
  - Delivers mathematically auditable shock metrics directly into portfolio hedging directives.
- **Negative / Trade-offs**:
  - Requires downstream customers to maintain capacity exposure percentages and inventory buffer day attributes on graph edges.
