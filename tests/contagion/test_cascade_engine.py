"""Unit and integration tests for SparseNetworkCascadeEngine (Step 7.2).

Verifies:
1. TT-02A literal graph convergence (36.25% impairment, DISRUPTIVE_SHOCK).
2. Intermediary 100% complete failure diamond compounding (85.0% impairment).
3. Leontief bottleneck across distinct input groups (max over groups).
4. Inventory buffer absorption and friction floor (S = 5.0%, TOLERATED_BUFFERED).
5. Temporal delay and unbuffered downtime dynamics (Option A).
6. Multi-parent convergent onset and downtime aggregation.
7. Feedback cycle damping (gamma = 0.50) and spectral radius calculation.
8. 5-hop horizon boundary enforcement (OUT_OF_HORIZON).
9. Latency budget compliance (< 10ms for 100-node graph).
"""

import time
import pytest

from contagion.cascade import (
    CascadeEdge,
    CascadeGraph,
    CascadeNode,
    SparseNetworkCascadeEngine,
)


class TestSparseNetworkCascadeEngine:
    """Test suite for the deterministic cascade engine."""

    def test_tt02a_literal_diamond_compounding(self) -> None:
        """TT-02A scenario: Two parallel branches converging on Customer C.

        Upstream:
        s1 -> m1 (40% share, wafers)
        s2 -> m2 (45% share, wafers)
        Midstream to focal:
        m1 -> C (40% share, substrates)
        m2 -> C (45% share, substrates)

        Literal shock: s1=100%, s2=100%
        m1 = 40%, m2 = 45%
        C = (0.40 * 40%) + (0.45 * 45%) = 16.0% + 20.25% = 36.25%
        """
        graph = CascadeGraph()
        graph.add_edge(CascadeEdge(source="s1", target="m1", input_group="wafers", share=0.40))
        graph.add_edge(CascadeEdge(source="s2", target="m2", input_group="wafers", share=0.45))
        graph.add_edge(CascadeEdge(source="m1", target="C", input_group="substrates", share=0.40))
        graph.add_edge(CascadeEdge(source="m2", target="C", input_group="substrates", share=0.45))

        engine = SparseNetworkCascadeEngine(disruption_threshold_pct=25.0)
        results = engine.propagate(graph, initial_shocks={"s1": 1.0, "s2": 1.0})

        summary_c = results["C"]
        assert summary_c.peak_shock_pct == pytest.approx(36.25, abs=1e-3)
        assert summary_c.impairment_score == pytest.approx(0.3625, abs=1e-4)
        assert summary_c.verdict == "DISRUPTIVE_SHOCK"
        assert not summary_c.is_out_of_horizon

    def test_tt02a_complete_tier2_halt_85pct(self) -> None:
        """If intermediate suppliers m1 and m2 suffer complete 100% shutdowns,

        Customer C incurs 40% + 45% = 85.0% capacity contraction.
        """
        graph = CascadeGraph()
        graph.add_edge(CascadeEdge(source="m1", target="C", input_group="substrates", share=0.40))
        graph.add_edge(CascadeEdge(source="m2", target="C", input_group="substrates", share=0.45))

        engine = SparseNetworkCascadeEngine(disruption_threshold_pct=25.0)
        results = engine.propagate(graph, initial_shocks={"m1": 1.0, "m2": 1.0})

        summary_c = results["C"]
        assert summary_c.peak_shock_pct == pytest.approx(85.0, abs=1e-3)
        assert summary_c.impairment_score == pytest.approx(0.85, abs=1e-4)
        assert summary_c.verdict == "DISRUPTIVE_SHOCK"

    def test_distinct_input_groups_leontief_bottleneck(self) -> None:
        """If A and B supply distinct, non-interchangeable components,

        downstream capacity loss is governed by the Leontief bottleneck max across groups.
        """
        graph = CascadeGraph()
        graph.add_edge(CascadeEdge(source="A", target="C", input_group="microcontrollers", share=0.40))
        graph.add_edge(CascadeEdge(source="B", target="C", input_group="displays", share=0.45))

        engine = SparseNetworkCascadeEngine()
        results = engine.propagate(graph, initial_shocks={"A": 1.0, "B": 1.0})

        summary_c = results["C"]
        # Max across groups: max(0.40, 0.45) = 0.45
        assert summary_c.peak_shock_pct == pytest.approx(45.0, abs=1e-3)
        assert summary_c.impairment_score == pytest.approx(0.45, abs=1e-4)

    def test_inventory_buffer_full_absorption_and_friction_floor(self) -> None:
        """When an inventory buffer completely absorbs supplier downtime,

        downtime is 0 and effective shock collapses to the 5% SAB 99 friction floor.
        """
        graph = CascadeGraph()
        # u -> v: 80% share, 60 days buffer
        graph.add_edge(CascadeEdge(source="u", target="v", share=0.80, buffer_days=60.0))

        engine = SparseNetworkCascadeEngine(friction_floor=0.05)
        # u has 50% shock for 60 days: h_u * D_u = 0.5 * 60 = 30 <= 60 -> fully absorbed!
        results = engine.propagate(
            graph,
            initial_shocks={"u": 0.50},
            initial_durations={"u": 60.0},
        )

        summary_v = results["v"]
        assert summary_v.downtime_days == 0.0
        assert summary_v.lost_capacity_days == 0.0
        assert summary_v.is_friction_floor
        assert summary_v.peak_shock_pct == pytest.approx(5.0, abs=1e-3)
        assert summary_v.verdict == "TOLERATED_BUFFERED"

    def test_inventory_buffer_partial_absorption_and_delay(self) -> None:
        """When disruption exceeds inventory cover:

        Downtime = D_u - B / h_u, Onset = T_u + B / h_u.
        """
        graph = CascadeGraph()
        # u -> v: 70% share, 30 days buffer
        graph.add_edge(CascadeEdge(source="u", target="v", share=0.70, buffer_days=30.0))

        engine = SparseNetworkCascadeEngine()
        # u has 100% shock for 90 days, onset day 0
        results = engine.propagate(
            graph,
            initial_shocks={"u": 1.0},
            initial_durations={"u": 90.0},
            initial_onsets={"u": 0.0},
        )

        summary_v = results["v"]
        # Onset: 0 + 30 / 1.0 = 30
        assert summary_v.onset_day == pytest.approx(30.0, abs=1e-3)
        # Downtime: 90 - 30 = 60
        assert summary_v.downtime_days == pytest.approx(60.0, abs=1e-3)
        # Lost capacity days: 0.70 * 60 = 42
        assert summary_v.lost_capacity_days == pytest.approx(42.0, abs=1e-3)
        assert summary_v.peak_shock_pct == pytest.approx(70.0, abs=1e-3)
        assert summary_v.verdict == "DISRUPTIVE_SHOCK"

    def test_multi_parent_convergent_temporal_aggregation(self) -> None:
        """Customer C has two incoming disrupted suppliers with staggered buffer exhaustion:

        A -> C: buffer 15 days, duration 60 days
        B -> C: buffer 30 days, duration 90 days
        """
        graph = CascadeGraph()
        graph.add_edge(CascadeEdge(source="A", target="C", input_group="wafers", share=0.40, buffer_days=15.0))
        graph.add_edge(CascadeEdge(source="B", target="C", input_group="wafers", share=0.45, buffer_days=30.0))

        engine = SparseNetworkCascadeEngine()
        results = engine.propagate(
            graph,
            initial_shocks={"A": 1.0, "B": 1.0},
            initial_durations={"A": 60.0, "B": 90.0},
            initial_onsets={"A": 0.0, "B": 0.0},
        )

        summary_c = results["C"]
        # Earliest onset: min(15, 30) = 15
        assert summary_c.onset_day == pytest.approx(15.0, abs=1e-3)
        # Latest end: max(15 + 45, 30 + 60) = 90
        # Downtime: 90 - 15 = 75
        assert summary_c.downtime_days == pytest.approx(75.0, abs=1e-3)

    def test_feedback_cycle_damping(self) -> None:
        """Directed cycle S -> A -> B -> A.

        B -> A is identified as a feedback edge and dampened by gamma = 0.50.
        """
        graph = CascadeGraph()
        graph.add_edge(CascadeEdge(source="S", target="A", share=0.50))
        graph.add_edge(CascadeEdge(source="A", target="B", share=0.50))
        graph.add_edge(CascadeEdge(source="B", target="A", share=0.50))

        engine = SparseNetworkCascadeEngine(gamma=0.50)
        results = engine.propagate(graph, initial_shocks={"S": 1.0})

        # Distances: S=0, A=1, B=2. B->A has dist(A)=1 <= dist(B)=2, so dampened by 0.50
        # Effective weight B->A = 0.50 * 0.50 = 0.25
        assert results["A"].spectral_radius is not None
        assert results["A"].spectral_radius > 0.0

    def test_out_of_horizon_tagging(self) -> None:
        """Nodes beyond max_iterations (5 hops) are flagged as OUT_OF_HORIZON."""
        graph = CascadeGraph()
        nodes = [f"N{i}" for i in range(7)]
        for i in range(6):
            graph.add_edge(CascadeEdge(source=nodes[i], target=nodes[i + 1], share=0.80))

        engine = SparseNetworkCascadeEngine(max_iterations=5)
        results = engine.propagate(graph, initial_shocks={"N0": 1.0})

        assert not results["N5"].is_out_of_horizon  # 5 hops = within horizon
        assert results["N6"].is_out_of_horizon  # 6 hops = OUT_OF_HORIZON

    def test_latency_budget(self) -> None:
        """Engine must process a 100-node, 300-edge graph in < 10 milliseconds."""
        graph = CascadeGraph()
        num_nodes = 100
        for i in range(num_nodes - 1):
            graph.add_edge(CascadeEdge(source=f"Node_{i}", target=f"Node_{i+1}", share=0.30))
            if i + 2 < num_nodes:
                graph.add_edge(CascadeEdge(source=f"Node_{i}", target=f"Node_{i+2}", share=0.20))
            if i + 3 < num_nodes:
                graph.add_edge(CascadeEdge(source=f"Node_{i}", target=f"Node_{i+3}", share=0.10))

        engine = SparseNetworkCascadeEngine(max_iterations=5)
        start = time.perf_counter()
        results = engine.propagate(graph, initial_shocks={"Node_0": 1.0})
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        assert len(results) == num_nodes
        assert elapsed_ms < 10.0, f"Execution took {elapsed_ms:.2f}ms > 10ms budget"
