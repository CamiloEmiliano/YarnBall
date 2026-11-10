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
- **Status**: ACCEPTED
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
   - When multiple upstream suppliers fail simultaneously and converge on a single downstream customer, incoming capacity losses must sum additively rather than being evaluated in isolation, following multi-supplier Leontief input-deficit aggregation [7].
2. **Strict Symbolic Division of Labor**:
   - Mathematical graph shock diffusion must be computed entirely by a deterministic, symbolic Python algorithm running in $\mathcal{O}(K \times |E|)$ time before prompting the language model. The LLM must receive exact pre-computed numbers and focus exclusively on qualitative narrative synthesis and options overlay structuring.
3. **Inventory Buffer Horizons & Temporal Delays**:
   - Downstream finished goods and component inventory buffers delay shock onset dates by exactly $B$ calendar days and absorb disruption duration under the Time-to-Survive (TTS) framework [6]:
     $$D_{\text{unbuffered}} = \max(0,\, D_{\text{disruption}} - B)$$
4. **Cycle Damping & Guaranteed Loop Termination**:
   - Formulates systemic shock propagation under the DebtRank recursive distress framework [2], where node impairment $h_v(t) \in [0.0, 1.0]$ compounds incoming parent distress up to a non-linear capacity ceiling.
   - Circular supply dependencies (e.g., $A \to B \to C \to A$) are dampened under Bardoscia et al.'s spectral radius contraction condition ($\lambda_{\max} < 1.0$) [3], applying geometric cycle attenuation $\gamma = 0.50$ grounded in Acemoglu's intermediate input pass-through share $\mu \approx 0.50$ [1].
   - Consistent with empirical disaster propagation studies showing approximately 50% disruption attenuation beyond immediate Tier-1 links [4, 5].
   - Numerically, $\gamma = 0.5 = 2^{-1}$ provides exact IEEE-754 dyadic floating-point precision, while ensuring the residual shock at $K = 5$ ($\gamma^5 = 3.125\%$) decays strictly below the institutional materiality threshold ($\epsilon = 0.05$) established under SEC SAB 99 [8].
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
2. **Deterministic Additive Flux Algorithm (DebtRank Formulation)**:
   - Node operational distress is parameterized as $h_v(t) \in [0.0, 1.0]$ (where percentage capacity shock is $S(v) = 100.0 \times h_v(t)$).
   - The directed inter-firm impact matrix element $W_{uv}$ integrates intermediate input capacity exposure [1, 4] with the Time-to-Survive (TTS) inventory absorption ratio [6]:
     $$W_{uv} = \left(\frac{\omega(u \to v)}{100.0}\right) \times \max\left(0.0,\, 1.0 - \frac{B(u \to v)}{D(u)}\right)$$
   - Following the recursive DebtRank dynamical equation [2], incoming parallel shocks compound additively across upstream suppliers up to a non-linear capacity ceiling [7]:
     $$h_v(t) = \min\left(1.0,\, h_v(t-1) + \sum_{u \in \text{In}(v)} W_{uv} \cdot \Delta h_u(t-1)\right)$$
     where $\Delta h_u(t-1) = h_u(t-1) - h_u(t-2)$ is the incremental new distress transmitted from round $t-1$ (with initial condition $\Delta h_{\text{shock}}(0) = h_{\text{shock}}(0) = S_0 / 100.0$).
3. **Temporal Shock Propagation**:
   - Effective Onset Date at node $v$ is delayed by the active inventory buffer duration (TTS) [6]:
     $$T_{\text{onset}}(v) = T_{\text{onset}}(u) + B(u \to v)$$
   - Unbuffered Duration represents net exposure exceeding the inventory horizon [6]:
     $$D(v) = \max\left(0,\, D(u) - B(u \to v)\right)$$
4. **Cycle Damping Protocol**:
   - In accordance with Bardoscia et al.'s spectral stability criterion ($\rho(\mathbf{W}) = \lambda_{\max} < 1.0$) [3], cyclical multi-round reverberation must decay geometrically rather than amplifying into runaway contagion.
   - When an edge completes a directed cycle (re-encountering an active ancestor in path history), its transmitted incremental distress is attenuated by geometric damping factor $\gamma = 0.50$:
     $$\Delta h_v^{\text{cycle}}(t) = \gamma \sum_{u \in \text{In}(v) \cap \text{Cycle}} W_{uv} \cdot \Delta h_u(t-1)$$
   - Across $k$ successive loop traversals, cyclical feedback decays as $\mathcal{O}(\gamma^k)$. Iterations terminate when network incremental distress $\sum_v \Delta h_v(t) < \delta = 0.01$ (1 basis point) or iteration count reaches $K = 5$.
   - At $K = 5$, the remaining loop reverberation $\gamma^5 = (0.5)^5 = 3.125\% < \epsilon = 0.05$ (the SEC SAB 99 materiality floor [8]), guaranteeing numerical extinction and execution in $< 10$ milliseconds.
5. **LLM Handshake Interface**:
   - Outputs a structured `ConvergentShockSummary` object injected into the prompt verbalizer in `retrieval/pipeline.py` for Tasks D and E.

---

## 5. Invariants & Guardrails

1. **Additive Diamond Convergence Invariant**:
   - For two parallel suppliers $A$ and $B$ converging on $C$ with zero buffer:
     $$h_C = \min\left(1.0,\, W_{AC} h_A + W_{BC} h_B\right) \iff S(C) = \min\left(100.0,\, \text{Flux}(A \to C) + \text{Flux}(B \to C)\right)$$
   - Parallel losses must compound additively with zero branch omission [2, 7].
2. **Buffer Delay & Absorption Invariant**:
   - An active inventory buffer of $B$ days strictly shifts downstream shock onset by $B$ calendar days [6]:
     $$T_{\text{onset}}(v) = T_{\text{event}} + B$$
   - If $B \ge D_{\text{disruption}}$, unbuffered duration must equal $0$, and effective shock collapses to baseline residual friction ($S_{\text{friction}} = 0.05$), grounded in the SEC 5% materiality floor [8].
3. **Cycle Termination Invariant**:
   - In graphs containing directed cycles, the cascade engine must terminate in at most $K = 5$ iterations with total flux strictly bounded and non-divergent [2, 3].
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

---

## 7. References

1. Acemoglu, D., Carvalho, V. M., Ozdaglar, A., & Tahbaz-Salehi, A. (2012). The Network Origins of Aggregate Fluctuations. *Econometrica*, 80(5), 1977–2016. https://doi.org/10.3982/ECTA9623
2. Battiston, S., Puliga, M., Kaushik, R., Tasca, P., & Caldarelli, G. (2012). DebtRank: Too Central to Fail? Financial Networks, the FED and Systemic Risk. *Scientific Reports*, 2, 541. https://doi.org/10.1038/srep00541
3. Bardoscia, M., Battiston, S., Caccioli, F., & Caldarelli, G. (2017). Pathways towards instability in financial networks. *Nature Communications*, 8, 14416. https://doi.org/10.1038/ncomms14416
4. Barrot, J.-N., & Sauvagnat, J. (2016). Input Specificity and the Propagation of Idiosyncratic Shocks in Production Networks. *The Quarterly Journal of Economics*, 131(3), 1543–1592. https://doi.org/10.1093/qje/qjw018
5. Carvalho, V. M., Nirei, M., Saito, Y. U., & Tahbaz-Salehi, A. (2021). Supply Chain Disruptions: Evidence from the Great East Japan Earthquake. *The Quarterly Journal of Economics*, 136(2), 1255–1321. https://doi.org/10.1093/qje/qjaa044
6. Gao, S. Y., Simchi-Levi, D., Teo, C.-P., & Yan, Z. (2019). Disruption Risk Mitigation in Supply Chains—The Risk Exposure Index Revisited. *Operations Research*, 67(3), 831–852. https://doi.org/10.1287/opre.2018.1776
7. Inoue, H., & Todo, Y. (2019). Firm-level propagation of shocks through supply-chain networks. *Nature Sustainability*, 2(9), 841–847. https://doi.org/10.1038/s41893-019-0351-x
8. U.S. Securities and Exchange Commission. (1999). Staff Accounting Bulletin No. 99: Materiality. *SEC Docket*, 70(16), 1490–1499. Codified at 17 C.F.R. Part 211. https://www.sec.gov/interps/account/sab99.htm

