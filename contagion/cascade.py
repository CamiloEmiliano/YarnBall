"""Sparse Network Cascade Engine Data Models & Graph Primitives.

Academic References:
- ADR-00006: Phase 7 Sparse Network Cascade Engine
- Inoue & Todo (2019): "Firm-level propagation of shocks through supply-chain networks", Nature Sustainability
- Simchi-Levi et al. (2015): "Identifying Risks and Mitigating Disruptions in the Automotive Supply Chain", Interfaces
- Bardoscia et al. (2017): "Pathways towards instability in financial networks", Nature Communications
"""

from __future__ import annotations

from collections import deque
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger("contagion.cascade")


class CascadeNode(BaseModel):
    """Represents an entity carrying operational capacity, accumulated distress, and temporal metrics."""

    entity_id: str
    name: str = ""
    sector: str = ""
    baseline_capacity: float = Field(default=1.0, ge=0.0)
    accumulated_shock: float = Field(default=0.0, ge=0.0, le=1.0)  # h_v in [0.0, 1.0]
    onset_day: Optional[float] = None
    downtime_days: float = Field(default=0.0, ge=0.0)
    lost_capacity_days: float = Field(default=0.0, ge=0.0)
    is_out_of_horizon: bool = False
    is_friction_floor: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def percentage_shock(self) -> float:
        """Returns capacity shock as a percentage S(v) in [0.0, 100.0]."""
        return round(self.accumulated_shock * 100.0, 4)


class CascadeEdge(BaseModel):
    """Directed supply-chain edge carrying intermediate procurement share and inventory cover."""

    source: str
    target: str
    input_group: str = "default"  # Interchangeable input grouping k
    share: float = Field(default=1.0, ge=0.0, le=1.0)  # a_uv = omega(u -> v) / 100.0 in [0.0, 1.0]
    buffer_days: float = Field(default=0.0, ge=0.0)  # B(u -> v) in calendar days (TTS proxy)
    materiality_tier: str = "NON_CRITICAL"  # Non-computational audit metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def capacity_exposure_pct(self) -> float:
        """Returns capacity exposure percentage omega in [0.0, 100.0]."""
        return round(self.share * 100.0, 4)


class CascadeGraph:
    """Directed sparse graph supporting fast in-degree indexing, cycle detection, and distance metrics."""

    def __init__(self) -> None:
        self.nodes: Dict[str, CascadeNode] = {}
        self.edges: List[CascadeEdge] = []
        self._in_edges: Dict[str, List[CascadeEdge]] = {}
        self._out_edges: Dict[str, List[CascadeEdge]] = {}
        self._in_edges_by_group: Dict[str, Dict[str, List[CascadeEdge]]] = {}

    def add_node(self, node: CascadeNode) -> None:
        """Adds or updates a node in the graph."""
        self.nodes[node.entity_id] = node
        if node.entity_id not in self._in_edges:
            self._in_edges[node.entity_id] = []
            self._out_edges[node.entity_id] = []
            self._in_edges_by_group[node.entity_id] = {}

    def add_edge(self, edge: CascadeEdge) -> None:
        """Adds a directed edge, auto-creating source and target nodes if needed."""
        if edge.source not in self.nodes:
            self.add_node(CascadeNode(entity_id=edge.source))
        if edge.target not in self.nodes:
            self.add_node(CascadeNode(entity_id=edge.target))

        self.edges.append(edge)
        self._in_edges[edge.target].append(edge)
        self._out_edges[edge.source].append(edge)

        target_groups = self._in_edges_by_group[edge.target]
        if edge.input_group not in target_groups:
            target_groups[edge.input_group] = []
        target_groups[edge.input_group].append(edge)

    def in_edges(self, node_id: str) -> List[CascadeEdge]:
        """Returns all incoming directed edges targeting node_id."""
        return self._in_edges.get(node_id, [])

    def out_edges(self, node_id: str) -> List[CascadeEdge]:
        """Returns all outgoing directed edges originating from node_id."""
        return self._out_edges.get(node_id, [])

    def in_edges_by_group(self, node_id: str) -> Dict[str, List[CascadeEdge]]:
        """Returns incoming edges for node_id grouped by interchangeable input_group k."""
        return self._in_edges_by_group.get(node_id, {})

    def validate_group_shares(self, tolerance: float = 1e-6) -> None:
        """Validates that for every node v and input group k, sum_{u in In_k(v)} a_uv <= 1.0 + tol."""
        for target_id, groups in self._in_edges_by_group.items():
            for group_k, edges in groups.items():
                total_share = sum(e.share for e in edges)
                if total_share > 1.0 + tolerance:
                    raise ValueError(
                        f"Target node '{target_id}' input group '{group_k}' has total procurement "
                        f"share {total_share:.4f} > 1.0 (violates Leontief capacity constraint)."
                    )

    def compute_sccs(self) -> List[Set[str]]:
        """Computes Strongly Connected Components (SCCs) using Tarjan's algorithm in O(V + E) time."""
        index = 0
        indices: Dict[str, int] = {}
        lowlinks: Dict[str, int] = {}
        stack: List[str] = []
        on_stack: Set[str] = set()
        sccs: List[Set[str]] = []

        def strongconnect(v: str) -> None:
            nonlocal index
            indices[v] = index
            lowlinks[v] = index
            index += 1
            stack.append(v)
            on_stack.add(v)

            for edge in self._out_edges.get(v, []):
                w = edge.target
                if w not in indices:
                    strongconnect(w)
                    lowlinks[v] = min(lowlinks[v], lowlinks[w])
                elif w in on_stack:
                    lowlinks[v] = min(lowlinks[v], indices[w])

            if lowlinks[v] == indices[v]:
                scc: Set[str] = set()
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    scc.add(w)
                    if w == v:
                        break
                sccs.append(scc)

        for node_id in self.nodes:
            if node_id not in indices:
                strongconnect(node_id)

        return sccs

    def compute_hop_distances(self, source_nodes: Set[str]) -> Dict[str, int]:
        """Computes shortest topological hop distance from shock source nodes via BFS."""
        distances: Dict[str, int] = {}
        queue: deque[str] = deque()

        for s in source_nodes:
            if s in self.nodes:
                distances[s] = 0
                queue.append(s)

        while queue:
            curr = queue.popleft()
            curr_dist = distances[curr]
            for edge in self._out_edges.get(curr, []):
                nxt = edge.target
                if nxt not in distances:
                    distances[nxt] = curr_dist + 1
                    queue.append(nxt)

        return distances

    def identify_feedback_edges(self, source_nodes: Set[str]) -> Set[Tuple[str, str, str]]:
        """Identifies feedback edges (source, target, input_group).

        Defined in ADR-00006 Section 4.4:
        An edge (u -> v) is a feedback edge iff:
        1. u and v belong to the same non-trivial Strongly Connected Component (SCC).
        2. dist(v) <= dist(u) relative to initial shock sources.
        """
        sccs = self.compute_sccs()
        node_to_scc: Dict[str, int] = {}
        for scc_id, scc in enumerate(sccs):
            if len(scc) > 1:
                for node in scc:
                    node_to_scc[node] = scc_id
            elif len(scc) == 1:
                # Check for self-loop
                node = next(iter(scc))
                if any(e.target == node for e in self._out_edges.get(node, [])):
                    node_to_scc[node] = scc_id

        distances = self.compute_hop_distances(source_nodes)
        feedback_edges: Set[Tuple[str, str, str]] = set()

        for edge in self.edges:
            u, v = edge.source, edge.target
            if u in node_to_scc and v in node_to_scc and node_to_scc[u] == node_to_scc[v]:
                dist_u = distances.get(u, float("inf"))
                dist_v = distances.get(v, float("inf"))
                if dist_v <= dist_u:
                    feedback_edges.add((u, v, edge.input_group))

        return feedback_edges


class ConvergentShockSummary(BaseModel):
    """Deterministic summary of cascade shock impact on a target entity."""

    target_entity: str
    peak_shock_pct: float = Field(ge=0.0, le=100.0)
    impairment_score: float = Field(ge=0.0, le=1.0)
    onset_day: Optional[float] = None
    downtime_days: float = Field(default=0.0, ge=0.0)
    lost_capacity_days: float = Field(default=0.0, ge=0.0)
    is_out_of_horizon: bool = False
    is_friction_floor: bool = False
    verdict: str  # DISRUPTIVE_SHOCK, CONTAINED_MINIMAL_CONTAGION, TOLERATED_BUFFERED, NEUTRAL_UNAFFECTED
    contributing_edges: List[Dict[str, Any]] = Field(default_factory=list)
    spectral_radius: Optional[float] = None
    iteration_count: int = 0


class SparseNetworkCascadeEngine:
    """Deterministic Sparse Network Cascade Engine implementing Grouped-Leontief fixed-point propagation.

    Academic References:
    - ADR-00006: Phase 7 Sparse Network Cascade Engine
    - Inoue & Todo (2019): Grouped-Leontief input aggregation with inventory absorption.
    - Simchi-Levi et al. (2015): Time-to-Survive (TTS) and unbuffered downtime dynamics.
    - Bardoscia et al. (2017): Spectral stability and feedback loop attenuation.
    - Acemoglu et al. (2012): Intermediate input pass-through share gamma = 0.50.
    """

    def __init__(
        self,
        gamma: float = 0.50,
        friction_floor: float = 0.05,
        max_iterations: int = 5,
        tolerance: float = 1e-9,
        disruption_threshold_pct: float = 25.0,
    ) -> None:
        self.gamma = gamma
        self.friction_floor = friction_floor
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.disruption_threshold_pct = disruption_threshold_pct

    def compute_spectral_radius(self, graph: CascadeGraph, feedback_edges: Set[Tuple[str, str, str]]) -> float:
        """Computes spectral radius rho(W) of the effective weighted adjacency matrix."""
        import numpy as np

        node_list = sorted(list(graph.nodes.keys()))
        n = len(node_list)
        if n == 0:
            return 0.0

        node_to_idx = {nid: i for i, nid in enumerate(node_list)}
        w_mat = np.zeros((n, n), dtype=float)

        for edge in graph.edges:
            u_idx = node_to_idx.get(edge.source)
            v_idx = node_to_idx.get(edge.target)
            if u_idx is not None and v_idx is not None:
                is_feedback = (edge.source, edge.target, edge.input_group) in feedback_edges
                w_eff = (self.gamma * edge.share) if is_feedback else edge.share
                w_mat[u_idx, v_idx] += w_eff

        try:
            eigvals = np.linalg.eigvals(w_mat)
            return float(np.max(np.abs(eigvals)))
        except Exception as exc:
            logger.warning("Failed to compute spectral radius: %s", exc)
            return 0.0

    def propagate(
        self,
        graph: CascadeGraph,
        initial_shocks: Dict[str, float],
        initial_durations: Optional[Dict[str, float]] = None,
        initial_onsets: Optional[Dict[str, float]] = None,
    ) -> Dict[str, ConvergentShockSummary]:
        """Executes multi-round deterministic shock propagation.

        Args:
            graph: CascadeGraph containing nodes, procurement edges, and buffer metadata.
            initial_shocks: Mapping of entity_id -> initial shock magnitude in [0.0, 1.0] or [0.0, 100.0].
            initial_durations: Mapping of entity_id -> initial disruption duration in calendar days.
            initial_onsets: Mapping of entity_id -> initial shock onset calendar day.

        Returns:
            Mapping of entity_id -> ConvergentShockSummary.
        """
        # Validate Leontief group share constraint sum_{u} a_uv <= 1.0
        graph.validate_group_shares()

        durations_in = initial_durations or {}
        onsets_in = initial_onsets or {}

        # Normalize initial shocks to h_v in [0.0, 1.0]
        norm_h0: Dict[str, float] = {}
        for nid, s in initial_shocks.items():
            val = float(s)
            if val > 1.0:
                val = val / 100.0
            norm_h0[nid] = max(0.0, min(1.0, val))

        # Identify shock sources
        seed_nodes = {nid for nid, shock in norm_h0.items() if shock > 0.0}

        # BFS hop distances & feedback edge identification
        distances = graph.compute_hop_distances(seed_nodes)
        feedback_edges = graph.identify_feedback_edges(seed_nodes)
        spectral_rad = self.compute_spectral_radius(graph, feedback_edges)

        # State tracking
        h_prev: Dict[str, float] = {nid: norm_h0.get(nid, 0.0) for nid in graph.nodes}
        h_curr: Dict[str, float] = dict(h_prev)
        node_durations: Dict[str, float] = {
            nid: float(durations_in.get(nid, 0.0)) for nid in graph.nodes
        }
        node_onsets: Dict[str, Optional[float]] = {
            nid: float(onsets_in.get(nid, 0.0)) if nid in seed_nodes else None for nid in graph.nodes
        }

        iterations_executed = 0

        # Iterative Grouped-Leontief Propagation Loop
        for round_idx in range(1, self.max_iterations + 1):
            iterations_executed = round_idx
            max_change = 0.0

            for target_id, node in graph.nodes.items():
                in_groups = graph.in_edges_by_group(target_id)
                if not in_groups:
                    # Source or leaf node without incoming edges
                    continue

                group_shortfalls: List[float] = []

                for group_k, edges in in_groups.items():
                    group_sum = 0.0
                    for edge in edges:
                        u = edge.source
                        h_u = h_prev.get(u, 0.0)
                        d_u = node_durations.get(u, 0.0)

                        # Edge gating: buffer exhaustion
                        if edge.buffer_days <= 0.0:
                            is_active = (h_u > 0.0)
                        else:
                            is_active = (h_u * d_u > edge.buffer_days)

                        if is_active and h_u > 0.0:
                            is_feedback = (edge.source, edge.target, edge.input_group) in feedback_edges
                            a_eff = (self.gamma * edge.share) if is_feedback else edge.share
                            group_sum += a_eff * h_u

                    group_shortfalls.append(min(1.0, group_sum))

                # Leontief bottleneck across distinct input groups
                g_max = max(group_shortfalls) if group_shortfalls else 0.0
                h_new = max(norm_h0.get(target_id, 0.0), g_max)

                change = abs(h_new - h_prev[target_id])
                if change > max_change:
                    max_change = change
                h_curr[target_id] = h_new

            h_prev = dict(h_curr)
            if max_change < self.tolerance:
                break

        # Temporal Shock Propagation (Option A Formulation)
        for target_id, node in graph.nodes.items():
            in_groups = graph.in_edges_by_group(target_id)
            active_timings: List[Tuple[float, float, float, CascadeEdge]] = []

            for group_k, edges in in_groups.items():
                for edge in edges:
                    u = edge.source
                    h_u = h_curr.get(u, 0.0)
                    d_u = node_durations.get(u, 0.0)
                    t_u = node_onsets.get(u)

                    if edge.buffer_days <= 0.0:
                        is_active = (h_u > 0.0)
                    else:
                        is_active = (h_u * d_u > edge.buffer_days)

                    if is_active and h_u > 0.0:
                        t_base = t_u if t_u is not None else 0.0
                        t_uv = t_base + (edge.buffer_days / h_u)
                        d_uv = max(0.0, d_u - (edge.buffer_days / h_u))
                        l_uv = max(0.0, (h_u * d_u) - edge.buffer_days)
                        if d_uv > 0.0:
                            active_timings.append((t_uv, d_uv, l_uv, edge))

            if active_timings:
                t_onset = min(item[0] for item in active_timings)
                t_end = max(item[0] + item[1] for item in active_timings)
                d_net = max(0.0, t_end - t_onset)
                node_onsets[target_id] = t_onset
                node_durations[target_id] = d_net
            else:
                if target_id in seed_nodes and durations_in.get(target_id, 0.0) > 0.0:
                    node_onsets[target_id] = float(onsets_in.get(target_id, 0.0))
                    node_durations[target_id] = float(durations_in.get(target_id, 0.0))
                else:
                    node_onsets[target_id] = None
                    node_durations[target_id] = 0.0

        # Post-Processing: Friction Floor, Horizon Flagging, and Summaries
        summaries: Dict[str, ConvergentShockSummary] = {}

        for target_id, node in graph.nodes.items():
            final_h = h_curr.get(target_id, 0.0)
            d_final = node_durations.get(target_id, 0.0)
            t_final = node_onsets.get(target_id)
            is_friction = False
            is_out_of_horizon = distances.get(target_id, float("inf")) > self.max_iterations

            # Check if target had incoming disruption absorbed by inventory buffer
            had_buffered_incoming = False
            for edge in graph.in_edges(target_id):
                u_shock = h_curr.get(edge.source, 0.0)
                u_dur = node_durations.get(edge.source, 0.0)
                if u_shock > 0.0 and edge.buffer_days > 0.0:
                    if (u_shock * u_dur) <= edge.buffer_days:
                        had_buffered_incoming = True
                        break

            # Friction floor: buffered incoming shock with zero downtime
            if d_final == 0.0 and target_id not in seed_nodes:
                if had_buffered_incoming or final_h > 0.0:
                    is_friction = True
                    final_h = max(final_h, self.friction_floor)

            lost_cap_days = round(final_h * d_final, 4)
            peak_pct = round(final_h * 100.0, 4)

            # Determine institutional verdict
            if peak_pct >= self.disruption_threshold_pct:
                verdict = "DISRUPTIVE_SHOCK"
            elif is_friction or (peak_pct > 0.0 and d_final == 0.0):
                verdict = "TOLERATED_BUFFERED"
            elif peak_pct > 0.0:
                verdict = "CONTAINED_MINIMAL_CONTAGION"
            else:
                verdict = "NEUTRAL_UNAFFECTED"

            # Update node in-place
            node.accumulated_shock = min(1.0, final_h)
            node.onset_day = t_final
            node.downtime_days = d_final
            node.lost_capacity_days = lost_cap_days
            node.is_friction_floor = is_friction
            node.is_out_of_horizon = is_out_of_horizon

            # Collect active contributing edges
            contribs: List[Dict[str, Any]] = []
            for edge in graph.in_edges(target_id):
                u_shock = h_curr.get(edge.source, 0.0)
                if u_shock > 0.0:
                    contribs.append(
                        {
                            "source": edge.source,
                            "input_group": edge.input_group,
                            "share_pct": edge.capacity_exposure_pct,
                            "buffer_days": edge.buffer_days,
                            "source_shock_pct": round(u_shock * 100.0, 4),
                        }
                    )

            summaries[target_id] = ConvergentShockSummary(
                target_entity=target_id,
                peak_shock_pct=peak_pct,
                impairment_score=round(final_h, 6),
                onset_day=t_final,
                downtime_days=d_final,
                lost_capacity_days=lost_cap_days,
                is_out_of_horizon=is_out_of_horizon,
                is_friction_floor=is_friction,
                verdict=verdict,
                contributing_edges=contribs,
                spectral_radius=round(spectral_rad, 4),
                iteration_count=iterations_executed,
            )

        return summaries

