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

Language models exhibit persistent mode-bleed, hallucinations, and attention diffusion when required to calculate multi-hop graph arithmetic. Relying on an LLM to calculate multi-branch network shock produces dangerously uncalibrated risk assessments and flawed portfolio hedging directives.

---

## 2. Decision Drivers & Constraints

1. **Grouped-Leontief Input Deficit Aggregation**:
   - When multiple upstream suppliers fail simultaneously and converge on a single downstream customer, incoming capacity shortfalls sum additively within interchangeable input groups and combine via Leontief fixed-proportions maximization across distinct input groups [7].
2. **Strict Symbolic Division of Labor**:
   - Mathematical graph shock diffusion must be computed entirely by a deterministic, symbolic Python algorithm running in $\mathcal{O}(K \times |E|)$ time before prompting the language model. The LLM must receive exact pre-computed numbers and focus exclusively on qualitative narrative synthesis and options overlay structuring.
3. **Inventory Buffer Horizons & Temporal Delays**:
   - Downstream finished goods and component inventory buffers delay shock onset dates and absorb disruption duration. Edge inventory cover $B(u \to v)$ (in days) acts as a conservative proxy for Time-to-Survive (TTS) [6], absorbing lost input-days under material balance:
     $$D(u \to v) = \max\left(0,\, D(u) - \frac{B(u \to v)}{h(u)}\right)$$
     $$L(u \to v) = \max\left(0,\, h(u) D(u) - B(u \to v)\right)$$
4. **Cycle Damping & Guaranteed Loop Termination**:
   - Formulates systemic shock propagation under a grouped-Leontief fixed-point framework [7], where node impairment $h_v(t) \in [0.0, 1.0]$ compounds incoming parent distress up to a non-linear capacity ceiling.
   - Circular supply dependencies (e.g., $A \to B \to C \to A$) are dampened under Bardoscia et al.'s spectral radius contraction condition ($\lambda_{\max} < 1.0$) [3], applying geometric cycle attenuation $\gamma = 0.50$ grounded in Acemoglu's intermediate input pass-through share $\mu \approx 0.50$ [1].
   - Consistent with empirical disaster propagation studies showing approximately 50% disruption attenuation beyond immediate Tier-1 links [4, 5].
   - Numerically, $\gamma = 0.5 = 2^{-1}$ provides exact IEEE-754 dyadic floating-point precision, while ensuring the residual shock at $K = 5$ ($\gamma^5 = 3.125\%$) decays strictly below the institutional materiality threshold ($\epsilon = 0.05$) established under SEC SAB 99 [8].
   - Nodes located beyond the $K = 5$ hop horizon are flagged as `OUT_OF_HORIZON` rather than silently assigned zero.
5. **Formal Mathematical Specification**:
   - Diffusion mechanics, temporal horizons, and loop damping must follow explicit, verifiable formulas rendered in LaTeX for unambiguous verification against code implementations.
6. **Ultra-Low Latency Budget**:
   - The cascade calculation must execute purely in-memory in $< 10$ milliseconds for subgraphs up to 100 nodes and 500 edges.

---

## 3. Considered Alternatives & Rejection Rationale

- **Alternative A: Qualitative LLM Prompting with Chain-of-Thought (CoT)**:
  - *Rejected*: Torture testing TT-02A proved that even with <think> reasoning tokens, 7B models fail on parallel shock addition (11% pass rate). Prompt engineering cannot reliably overcome arithmetic limitations on multi-branch graphs.
- **Alternative B: End-to-End Deep Graph Neural Network (GNN: GIN/RGAT)**:
  - *Rejected*: Full deep GNNs introduce significant inference latency, require GPU compute, suffer from over-smoothing beyond 3-4 hops, and act as black boxes lacking the auditable mathematical transparency demanded by institutional compliance.
- **Alternative C: Unattenuated Reachability (BFS)**:
  - *Rejected*: Ignores real-world operational absorption where finished goods stockpiles buffer shocks.

---

## 4. Decision Outcome

We establish **Phase 7** by creating a dedicated package `contagion/` centered around the `SparseNetworkCascadeEngine` in `contagion/cascade.py` with the following architectural components:

1. **Graph Data Structures**:
   - `CascadeNode`: Represents an entity with baseline revenue/capacity, active buffers, accumulated shock state $h_v$, onset $T_{\text{onset}}(v)$, downtime $D(v)$, and horizon status.
   - `CascadeEdge`: Directed relationship $(u \to v)$ carrying input group identifier $k(u \to v)$, procurement share $a_{uv} = \omega(u \to v) / 100.0 \in [0.0, 1.0]$ subject to within-group normalization $\sum_{u \in \text{In}_k(v)} a_{uv} \le 1.0$, inventory buffer days $B(u \to v) \ge 0$, and non-computational metadata (`materiality_tier`).
   - `CascadeGraph`: Directed graph representation supporting fast in-degree indexing, strongly connected component (SCC) decomposition (Tarjan's algorithm), and hop distance indexing from shock sources.
2. **Deterministic Grouped-Leontief Propagation Algorithm**:
   - Node operational distress is parameterized as $h_v(t) \in [0.0, 1.0]$ (where percentage capacity shock is $S(v) = 100.0 \times h_v(t)$).
   - Edge transmission is gated by inventory exhaustion: edge $(u \to v)$ only transmits shock once disruption exceeds the inventory buffer cover:
     $$e_{uv} = \mathbb{1}\left[h_u D_u > B(u \to v)\right]$$
   - Effective transmission weight incorporates cycle damping on feedback edges (defined in Section 4.4):
     $$\widetilde{a}_{uv} = \begin{cases} \gamma \cdot a_{uv} & \text{if } (u \to v) \in E_{\text{feedback}} \\ a_{uv} & \text{otherwise} \end{cases}$$
   - In each round $t \in \{1, \dots, K\}$, incoming shortfalls sum additively within each input group $k$, and downstream node capacity impairment is governed by the Leontief bottleneck across all required input groups [7]:
     $$g_{v,k}(t) = \min\left(1.0,\, \sum_{u \in \text{In}_k(v)} \widetilde{a}_{uv} \cdot e_{uv} \cdot h_u(t-1)\right)$$
     $$h_v(t) = \max\left(h_v^0,\, \max_k g_{v,k}(t)\right)$$
     initialized with exogenous shock vector $h^0 \in [0.0, 1.0]^N$ at $t = 0$.
3. **Temporal Shock Propagation (Option A Formulation)**:
   - Effective per-edge onset date, unbuffered downtime, and lost input-days:
     $$T_{uv} = T_u + \frac{B(u \to v)}{h_u} \quad (h_u > 0)$$
     $$D_{uv} = \max\left(0,\, D_u - \frac{B(u \to v)}{h_u}\right)$$
     $$L_{uv} = \max\left(0,\, h_u D_u - B(u \to v)\right)$$
   - Multi-parent convergent aggregation: For downstream node $v$ with active unbuffered incoming edges $\text{In}^*(v) = \{u \in \text{In}(v) \mid e_{uv} = 1\}$:
     $$T_{\text{onset}}(v) = \min_{u \in \text{In}^*(v)} T_{uv}$$
     $$T_{\text{end}}(v) = \max_{u \in \text{In}^*(v)} \left(T_{uv} + D_{uv}\right)$$
     $$D(v) = \max\left(0,\, T_{\text{end}}(v) - T_{\text{onset}}(v)\right)$$
   - If all incoming edges are fully buffered ($\text{In}^*(v) = \emptyset$), then $D(v) = 0$ and $T_{\text{onset}}(v) = \text{None}$.
   - *Option A Temporal Assumption*: In scalar aggregation, asynchronous non-overlapping disruptions within the same input group are evaluated by peak additive capacity sum.
4. **Cycle Damping & Iteration Control Protocol**:
   - Edge feedback classification: An edge $(u \to v)$ is defined as a feedback edge ($(u \to v) \in E_{\text{feedback}}$) if:
     1. $u$ and $v$ belong to the same non-trivial Strongly Connected Component ($\text{SCC}$), and
     2. $\text{dist}(v) \le \text{dist}(u)$, where $\text{dist}(\cdot)$ is the shortest hop distance from the initial exogenous shock sources.
   - On all feedback edges, transmission weight is attenuated by $\gamma = 0.50$: $\widetilde{a}_{uv} = \gamma \cdot a_{uv}$.
   - Iteration stops when the maximum node distress update falls below numerical tolerance:
     $$\max_v |h_v(t) - h_v(t-1)| < \text{tol} = 10^{-9}$$
     or when iteration count reaches horizon limit $K = 5$.
   - Any reachable node at topological distance $> 5$ hops is flagged with status `OUT_OF_HORIZON`.
   - At $K = 5$, the remaining loop reverberation $\gamma^5 = (0.5)^5 = 3.125\% < \epsilon = 0.05$ (the SEC SAB 99 materiality floor [8]), guaranteeing numerical extinction and execution in $< 10$ milliseconds.
5. **LLM Handshake Interface**:
   - Outputs a structured `ConvergentShockSummary` object injected into the prompt verbalizer in `retrieval/pipeline.py` for Tasks D and E.

---

## 5. Invariants & Guardrails

1. **Grouped-Leontief Diamond Convergence Invariant**:
   - For two convergent suppliers $A$ and $B$ supplying Customer $C$ with zero buffer:
     - If $A$ and $B$ supply the **same input group** $k$ (interchangeable inputs, e.g., TT-02A substrate procurement):
       $$h_C = \min\left(1.0,\, a_{AC} h_A + a_{BC} h_B\right) \iff S(C) = \min\left(100.0,\, a_{AC} S(A) + a_{BC} S(B)\right)$$
       Under the literal TT-02A two-tier graph topology (where upstream Tier-2 edges carry $a_{s1,m1} = 0.40$ and $a_{s2,m2} = 0.45$, feeding into $a_{m1,C} = 0.40$ and $a_{m2,C} = 0.45$), the transmitted impairment at Customer $C$ is $(0.40 \times 40.0\%) + (0.45 \times 45.0\%) = 16.0\% + 20.25\% = \mathbf{36.25\%}$ (surpassing the 25% threshold to trigger `DISRUPTIVE_SHOCK`). If intermediary suppliers $m1$ and $m2$ suffer complete 100% outages ($h_{m1} = h_{m2} = 1.0$), combined capacity loss reaches $\mathbf{85.0\%}$.
     - If $A$ and $B$ supply **distinct input groups** $k_1 \neq k_2$ (strict Leontief complementarity):
       $$h_C = \max\left(\min(1.0,\, a_{AC} h_A),\, \min(1.0,\, a_{BC} h_B)\right)$$
       For complete outages ($h_A = h_B = 1.0$) with $a_{AC} = 0.40$ and $a_{BC} = 0.45$: $S(C) = \max(40.0\%,\, 45.0\%) = \mathbf{45.0\%}$.
2. **Buffer Delay, Absorption, and Residual Friction Invariant**:
   - Edge buffer delay: $T_{uv} = T_u + B(u \to v) / h_u$. For a linear chain of fully disrupted nodes ($h = 1.0$), shock onset accumulates additively: $T_{\text{onset}}(v_m) = T_0 + \sum_{i=1}^m B(v_{i-1} \to v_i)$.
   - Residual Friction Floor: For any node $v$ that receives incoming shock ($g_v > 0$) but whose disruptions are 100% absorbed by inventory buffers ($D(v) = 0$):
     $$h_{\text{friction}}(v) = 0.05 \iff S_{\text{friction}}(v) = 5.0\%$$
     grounded in the SEC SAB 99 5% materiality floor [8]. This friction represents expedited freight, safety stock replenishment, and administrative overhead. Crucially, $h_{\text{friction}}$ is applied strictly as a **reported terminal floor** for node $v$ after convergence and does **not** propagate downstream to subsequent tiers.
3. **Cycle Termination & Boundedness Invariant**:
   - In graphs containing directed cycles, the engine terminates in at most $K = 5$ iterations with total shock strictly bounded and non-divergent: $\max_v h_v \le \max_s h_s^0 \le 1.0$ [2, 3].
4. **Monotonic Path Attenuation Invariant**:
   - Along any single linear (non-convergent, acyclic) path with $e_{uv} = 1$:
     $$S(\text{Hop}_{N+1}) \le S(\text{Hop}_N)$$
     with strict inequality $S(\text{Hop}_{N+1}) < S(\text{Hop}_N)$ holding if and only if $a_{uv} < 1.0$.
5. **Zero External Network Dependency**:
   - The engine must execute 100% offline using pure Python and NumPy with zero live network or database calls.

---

## 6. Consequences & Downstream Impact

- **Positive / Guarantees**:
  - Permanently eliminates the 1/9 parallel diamond compounding failure mode discovered in Phase 5 torture testing.
  - Offloads arithmetic computation from the probabilistic LLM to a deterministic $\mathcal{O}(K \times |E|) + \mathcal{O}(V + E)$ graph algorithm.
  - Correctly captures both interchangeable supplier sums (within-group additive) and multi-component assembly bottlenecks (across-group Leontief max).
  - Delivers mathematically auditable shock metrics directly into portfolio hedging directives.
- **Negative / Trade-offs**:
  - Requires edges to specify input group identifiers and procurement shares.
  - Option A scalar aggregation does not model daily staggered disruption timelines within an input group.

---

## 7. References

1. Acemoglu, D., Carvalho, V. M., Ozdaglar, A., & Tahbaz-Salehi, A. (2012). The Network Origins of Aggregate Fluctuations. *Econometrica*, 80(5), 1977–2016. https://doi.org/10.3982/ECTA9623
2. Battiston, S., Puliga, M., Kaushik, R., Tasca, P., & Caldarelli, G. (2012). DebtRank: Too Central to Fail? Financial Networks, the FED and Systemic Risk. *Scientific Reports*, 2, 541. https://doi.org/10.1038/srep00541
3. Bardoscia, M., Battiston, S., Caccioli, F., & Caldarelli, G. (2017). Pathways towards instability in financial networks. *Nature Communications*, 8, 14416. https://doi.org/10.1038/ncomms14416
4. Barrot, J.-N., & Sauvagnat, J. (2016). Input Specificity and the Propagation of Idiosyncratic Shocks in Production Networks. *The Quarterly Journal of Economics*, 131(3), 1543–1592. https://doi.org/10.1093/qje/qjw018
5. Carvalho, V. M., Nirei, M., Saito, Y. U., & Tahbaz-Salehi, A. (2021). Supply Chain Disruptions: Evidence from the Great East Japan Earthquake. *The Quarterly Journal of Economics*, 136(2), 1255–1321. https://doi.org/10.1093/qje/qjaa044
6. Simchi-Levi, D., Schmidt, W., Wei, Y., Zhang, P. Y., Combs, K., Ge, Y., Gusikhin, O., Sanders, M., & Zhang, D. (2015). Identifying Risks and Mitigating Disruptions in the Automotive Supply Chain. *Interfaces*, 45(5), 375–390. https://doi.org/10.1287/inte.2015.0804
7. Inoue, H., & Todo, Y. (2019). Firm-level propagation of shocks through supply-chain networks. *Nature Sustainability*, 2(9), 841–847. https://doi.org/10.1038/s41893-019-0351-x
8. U.S. Securities and Exchange Commission. (1999). Staff Accounting Bulletin No. 99: Materiality. *SEC Docket*, 70(16), 1490–1499. Codified at 17 C.F.R. Part 211. https://www.sec.gov/interps/account/sab99.htm
