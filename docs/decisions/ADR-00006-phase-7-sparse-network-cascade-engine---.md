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
   - Mathematical graph shock diffusion must be computed entirely by a deterministic, symbolic Python algorithm running in $\mathcal{O}(K \times |E|)$ time before prompting the language model. The LLM must receive exact pre-computed numbers and focus exclusively on qualitative narrative synthesis and options overlay structuring.
3. **Inventory Buffer Horizons & Temporal Delays**:
   - Downstream finished goods and component inventory buffers delay shock onset dates by exactly $B$ calendar days and absorb disruption duration:
     $$D_{\text{unbuffered}} = \max(0,\, D_{\text{disruption}} - B)$$
4. **Cycle Damping & Guaranteed Loop Termination**:
   - Circular supply dependencies (e.g., $A \to B \to C \to A$) must be detected and dampened geometrically ($\gamma = 0.5$ attenuation per cycle traversal) with a strict maximum iteration cap $K = 5$, guaranteeing algorithm termination.
5. **Formal Mathematical Specification**:
   - Diffusion mechanics, temporal horizons, and loop damping must follow explicit, verifiable formulas rendered in LaTeX for unambiguous verification against code implementations.
6. **Ultra-Low Latency Budget**:
   - The cascade calculation must execute purely in-memory in $< 10$ milliseconds for subgraphs up to 100 nodes and 500 edges.

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
   - `CascadeEdge`: Directed relationship carrying capacity exposure $\omega \in [0.0, 100.0]$, inventory buffer days $B \ge 0$, and materiality tier.
   - `CascadeGraph`: Directed graph representation supporting fast in-degree indexing and cycle detection.
2. **Deterministic Additive Flux Algorithm**:
   - For each target node $v$, incoming flux from upstream parent $u$ is calculated as:
     $$\text{AbsorptionRatio}(u \to v) = \max\left(0.0,\, 1.0 - \frac{B(u \to v)}{D(u)}\right)$$
     $$\text{Flux}(u \to v) = S(u) \times \left(\frac{\omega(u \to v)}{100.0}\right) \times \text{AbsorptionRatio}(u \to v)$$
     $$S(v) = \min\left(100.0,\, \sum_{u \in \text{In}(v)} \text{Flux}(u \to v)\right)$$
3. **Temporal Shock Propagation**:
   - Effective Onset Date at node $v$ is delayed by the buffer:
     $$T_{\text{onset}}(v) = T_{\text{onset}}(u) + B(u \to v)$$
   - Unbuffered Duration:
     $$D(v) = \max\left(0,\, D(u) - B(u \to v)\right)$$
4. **Cycle Damping Protocol**:
   - Traversal tracks path history. When an already-visited node is re-encountered in a cycle, its transmitted shock is attenuated by damping factor $\gamma = 0.5$:
     $$\text{Flux}_{\text{cycle}}(u \to v) = \gamma \times \text{Flux}(u \to v)$$
   - Iterations terminate when $\Delta S < 0.01$ or iteration count reaches $K = 5$. Because $\gamma = 0.5 < 1.0$, the geometric cycle series is strictly bounded:
     $$\sum_{k=0}^{K} \gamma^k < \frac{1}{1 - \gamma} = 2.0$$
5. **LLM Handshake Interface**:
   - Outputs a structured `ConvergentShockSummary` object injected into the prompt verbalizer in `retrieval/pipeline.py` for Tasks D and E.

---

## 5. Invariants & Guardrails

1. **Additive Diamond Convergence Invariant**:
   - For two parallel suppliers $A$ and $B$ converging on $C$ with zero buffer:
     $$S(C) = \min\left(100.0,\, \text{Flux}(A \to C) + \text{Flux}(B \to C)\right)$$
   - Parallel losses must compound additively with zero branch omission.
2. **Buffer Delay & Absorption Invariant**:
   - An active inventory buffer of $B$ days strictly shifts downstream shock onset by $B$ calendar days:
     $$T_{\text{onset}}(v) = T_{\text{event}} + B$$
   - If $B \ge D_{\text{disruption}}$, unbuffered duration must equal $0$, and effective shock collapses to baseline residual friction ($S_{\text{friction}} = 0.05$).
3. **Cycle Termination Invariant**:
   - In graphs containing directed cycles, the cascade engine must terminate in at most $K = 5$ iterations with total flux strictly bounded and non-divergent.
4. **Monotonic Path Attenuation Invariant**:
   - Along any single linear (non-convergent) path, propagated shock intensity must strictly decrease:
     $$S(\text{Hop}_{N+1}) < S(\text{Hop}_N)$$
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

