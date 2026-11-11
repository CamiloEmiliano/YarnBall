"""Unit tests for Phase 7 Cascade Data Models and Graph Primitives.

Verifies:
1. CascadeNode and CascadeEdge validation and properties.
2. CascadeGraph in-degree indexing, group share constraints.
3. Tarjan SCC decomposition and cycle identification.
4. BFS hop distance calculations and feedback edge tagging.
5. ConvergentShockSummary structure.
"""

import pytest
from pydantic import ValidationError

from contagion.cascade import (
    CascadeEdge,
    CascadeGraph,
    CascadeNode,
    ConvergentShockSummary,
)


class TestCascadeNode:
    """Tests for CascadeNode creation, defaults, and boundary constraints."""

    def test_default_initialization(self) -> None:
        node = CascadeNode(entity_id="AAPL", name="Apple Inc.", sector="Technology")
        assert node.entity_id == "AAPL"
        assert node.name == "Apple Inc."
        assert node.accumulated_shock == 0.0
        assert node.percentage_shock == 0.0
        assert node.onset_day is None
        assert node.downtime_days == 0.0
        assert not node.is_out_of_horizon
        assert not node.is_friction_floor

    def test_accumulated_shock_bounds(self) -> None:
        node = CascadeNode(entity_id="NVDA", accumulated_shock=0.45)
        assert node.percentage_shock == 45.0

        with pytest.raises(ValidationError):
            CascadeNode(entity_id="BAD", accumulated_shock=-0.1)

        with pytest.raises(ValidationError):
            CascadeNode(entity_id="BAD", accumulated_shock=1.05)


class TestCascadeEdge:
    """Tests for CascadeEdge creation, share constraints, and properties."""

    def test_edge_initialization(self) -> None:
        edge = CascadeEdge(
            source="TSM",
            target="AAPL",
            input_group="semiconductors",
            share=0.40,
            buffer_days=30.0,
            materiality_tier="CRITICAL_TIER_1",
        )
        assert edge.source == "TSM"
        assert edge.target == "AAPL"
        assert edge.input_group == "semiconductors"
        assert edge.share == 0.40
        assert edge.capacity_exposure_pct == 40.0
        assert edge.buffer_days == 30.0
        assert edge.materiality_tier == "CRITICAL_TIER_1"

    def test_edge_share_bounds(self) -> None:
        with pytest.raises(ValidationError):
            CascadeEdge(source="A", target="B", share=-0.05)

        with pytest.raises(ValidationError):
            CascadeEdge(source="A", target="B", share=1.1)

        with pytest.raises(ValidationError):
            CascadeEdge(source="A", target="B", buffer_days=-5.0)


class TestCascadeGraph:
    """Tests for CascadeGraph indexing, group validation, and topology algorithms."""

    def test_auto_node_creation_and_indexing(self) -> None:
        graph = CascadeGraph()
        e1 = CascadeEdge(source="A", target="C", input_group="wafers", share=0.40)
        e2 = CascadeEdge(source="B", target="C", input_group="wafers", share=0.45)
        graph.add_edge(e1)
        graph.add_edge(e2)

        assert "A" in graph.nodes
        assert "B" in graph.nodes
        assert "C" in graph.nodes

        in_c = graph.in_edges("C")
        assert len(in_c) == 2

        out_a = graph.out_edges("A")
        assert len(out_a) == 1
        assert out_a[0].target == "C"

        groups = graph.in_edges_by_group("C")
        assert "wafers" in groups
        assert len(groups["wafers"]) == 2

    def test_validate_group_shares_success(self) -> None:
        graph = CascadeGraph()
        graph.add_edge(CascadeEdge(source="A", target="C", input_group="wafers", share=0.40))
        graph.add_edge(CascadeEdge(source="B", target="C", input_group="wafers", share=0.45))
        # 0.40 + 0.45 = 0.85 <= 1.0 -> should pass
        graph.validate_group_shares()

    def test_validate_group_shares_violation(self) -> None:
        graph = CascadeGraph()
        graph.add_edge(CascadeEdge(source="A", target="C", input_group="wafers", share=0.60))
        graph.add_edge(CascadeEdge(source="B", target="C", input_group="wafers", share=0.55))
        # 0.60 + 0.55 = 1.15 > 1.0 -> must raise ValueError
        with pytest.raises(ValueError, match="violates Leontief capacity constraint"):
            graph.validate_group_shares()

    def test_tarjan_scc_dag(self) -> None:
        # A -> B -> C (DAG)
        graph = CascadeGraph()
        graph.add_edge(CascadeEdge(source="A", target="B"))
        graph.add_edge(CascadeEdge(source="B", target="C"))

        sccs = graph.compute_sccs()
        # In a DAG, every SCC has exactly 1 node
        assert len(sccs) == 3
        for scc in sccs:
            assert len(scc) == 1

    def test_tarjan_scc_cycle(self) -> None:
        # S -> A -> B -> C -> A (cycle among A, B, C)
        graph = CascadeGraph()
        graph.add_edge(CascadeEdge(source="S", target="A"))
        graph.add_edge(CascadeEdge(source="A", target="B"))
        graph.add_edge(CascadeEdge(source="B", target="C"))
        graph.add_edge(CascadeEdge(source="C", target="A"))

        sccs = graph.compute_sccs()
        # Should have {S} and {A, B, C}
        cycle_scc = [scc for scc in sccs if len(scc) > 1]
        assert len(cycle_scc) == 1
        assert cycle_scc[0] == {"A", "B", "C"}

    def test_compute_hop_distances(self) -> None:
        # S -> A -> B -> C
        graph = CascadeGraph()
        graph.add_edge(CascadeEdge(source="S", target="A"))
        graph.add_edge(CascadeEdge(source="A", target="B"))
        graph.add_edge(CascadeEdge(source="B", target="C"))

        dists = graph.compute_hop_distances({"S"})
        assert dists == {"S": 0, "A": 1, "B": 2, "C": 3}

    def test_identify_feedback_edges(self) -> None:
        # S -> A -> B -> C -> A
        # Distances from S: S=0, A=1, B=2, C=3.
        # Edges in cycle:
        # A -> B: dist(A)=1, dist(B)=2 (forward: dist(B) > dist(A))
        # B -> C: dist(B)=2, dist(C)=3 (forward: dist(C) > dist(B))
        # C -> A: dist(C)=3, dist(A)=1 (feedback: dist(A) <= dist(C))
        graph = CascadeGraph()
        graph.add_edge(CascadeEdge(source="S", target="A", input_group="g1"))
        graph.add_edge(CascadeEdge(source="A", target="B", input_group="g1"))
        graph.add_edge(CascadeEdge(source="B", target="C", input_group="g1"))
        graph.add_edge(CascadeEdge(source="C", target="A", input_group="g1"))

        feedback = graph.identify_feedback_edges({"S"})
        assert feedback == {("C", "A", "g1")}

    def test_identify_feedback_edges_acyclic_diamond(self) -> None:
        # S1 -> M1 -> C, S2 -> M2 -> C (pure DAG diamond)
        graph = CascadeGraph()
        graph.add_edge(CascadeEdge(source="S1", target="M1"))
        graph.add_edge(CascadeEdge(source="S2", target="M2"))
        graph.add_edge(CascadeEdge(source="M1", target="C"))
        graph.add_edge(CascadeEdge(source="M2", target="C"))

        feedback = graph.identify_feedback_edges({"S1", "S2"})
        assert len(feedback) == 0  # No cycles, zero feedback edges


class TestConvergentShockSummary:
    """Tests for ConvergentShockSummary structure."""

    def test_summary_creation(self) -> None:
        summary = ConvergentShockSummary(
            target_entity="C",
            peak_shock_pct=36.25,
            impairment_score=0.3625,
            onset_day=15.0,
            downtime_days=45.0,
            lost_capacity_days=16.3125,
            verdict="DISRUPTIVE_SHOCK",
            iteration_count=2,
        )
        assert summary.target_entity == "C"
        assert summary.peak_shock_pct == 36.25
        assert summary.verdict == "DISRUPTIVE_SHOCK"
        assert not summary.is_out_of_horizon
        assert not summary.is_friction_floor

