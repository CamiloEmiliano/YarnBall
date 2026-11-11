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
    verdict: str  # DISRUPTIVE_SHOCK, CONTAINED_MINIMAL_CONTAGION, TOLERATED_BUFFERED
    contributing_edges: List[Dict[str, Any]] = Field(default_factory=list)
    spectral_radius: Optional[float] = None
    iteration_count: int = 0
