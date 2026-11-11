"""ADR-00007 Deterministic Pathological Attack Suite for SparseNetworkCascadeEngine.

Executes closed-form verification of all 8 adversarial network topologies:
1. Zero-Buffer / Duration-Less Shock (No false friction floor demotion)
2. Micro-Share Float Leak (1e-6 noise suppression)
3. Self-Loop Singularity (A -> A trivial cycle damping)
4. Disconnected Ghost Island (Zero-leakage proof)
5. Multi-Pumping Resonant Cycle (S1 -> A <-> B <- S2)
6. Fractal Nested Diamond (Multi-tier associativity)
7. Extreme Buffer Asymmetry (B=0 vs B=1e6 days)
8. High-Degree Fan-Out / Fan-In Bottleneck (1 to 50 to 1)
"""

import time
import pytest

from contagion.cascade import (
    CascadeEdge,
    CascadeGraph,
    SparseNetworkCascadeEngine,
)


class TestCascadeEnginePathologies:
    """Verifies SparseNetworkCascadeEngine against ADR-00007 pathological attack topologies."""

    def test_pathology_1_zero_buffer_durationless_shock(self) -> None:
        """Topology 1: Omitted duration on unbuffered edge must NOT trigger friction floor."""
        graph = CascadeGraph()
        graph.add_edge(CascadeEdge(source="U", target="V", share=0.20, buffer_days=0.0))
        graph.add_edge(CascadeEdge(source="U", target="W", share=0.40, buffer_days=0.0))

        engine = SparseNetworkCascadeEngine(disruption_threshold_pct=25.0)
        # Duration omitted (defaults to 0.0)
        results = engine.propagate(graph, initial_shocks={"U": 1.0})

        # Node V: 20% shock, unbuffered, D=0
        v = results["V"]
        assert v.peak_shock_pct == pytest.approx(20.0, abs=1e-3)
        assert v.downtime_days == 0.0
        assert not v.is_friction_floor
        assert v.verdict == "CONTAINED_MINIMAL_CONTAGION"

        # Node W: 40% shock, unbuffered, D=0
        w = results["W"]
        assert w.peak_shock_pct == pytest.approx(40.0, abs=1e-3)
        assert w.downtime_days == 0.0
        assert not w.is_friction_floor
        assert w.verdict == "DISRUPTIVE_SHOCK"

    def test_pathology_2_micro_share_noise_suppression(self) -> None:
        """Topology 2: Microscopic shares (1e-6) must not cause float explosion or false alarms."""
        graph = CascadeGraph()
        graph.add_edge(CascadeEdge(source="U", target="V1", share=1e-6, buffer_days=0.0))
        graph.add_edge(CascadeEdge(source="V1", target="V2", share=1e-6, buffer_days=0.0))
        graph.add_edge(CascadeEdge(source="V2", target="V3", share=1e-6, buffer_days=0.0))

        engine = SparseNetworkCascadeEngine(disruption_threshold_pct=25.0)
        results = engine.propagate(
            graph,
            initial_shocks={"U": 1.0},
            initial_durations={"U": 30.0},
        )

        # V1: 1e-6 = 0.0001%
        v1 = results["V1"]
        assert v1.peak_shock_pct == pytest.approx(0.0001, abs=1e-4)
        assert v1.lost_capacity_days == 0.0000
        assert v1.verdict == "CONTAINED_MINIMAL_CONTAGION"

        # V2: 1e-12 = 0.0000%
        v2 = results["V2"]
        assert v2.peak_shock_pct == 0.0000
        assert v2.lost_capacity_days == 0.0000
        assert v2.verdict == "NEUTRAL_UNAFFECTED"

        # V3: 1e-18 = 0.0000%
        v3 = results["V3"]
        assert v3.peak_shock_pct == 0.0000
        assert v3.lost_capacity_days == 0.0000
        assert v3.verdict == "NEUTRAL_UNAFFECTED"

    def test_pathology_3_self_loop_singularity_damping(self) -> None:
        """Topology 3: Self-loop A -> A with gamma=0.50 damping must converge geometrically."""
        graph = CascadeGraph()
        # S -> A: 0.50, A -> A: 0.40 (total raw group share = 0.90 <= 1.0)
        graph.add_edge(CascadeEdge(source="S", target="A", share=0.50, buffer_days=0.0))
        graph.add_edge(CascadeEdge(source="A", target="A", share=0.40, buffer_days=0.0))

        feedback_edges = graph.identify_feedback_edges({"S"})
        assert ("A", "A", "default") in feedback_edges, "A -> A must be identified as feedback edge"

        engine = SparseNetworkCascadeEngine(disruption_threshold_pct=25.0)
        results = engine.propagate(graph, initial_shocks={"S": 1.0})

        a = results["A"]
        # Closed form in 5 iters with gamma=0.50:
        # iter 1: 0.5000, iter 2: 0.6000, iter 3: 0.6200, iter 4: 0.6240, iter 5: 0.6248
        # Infinite limit = 0.50 / (1 - 0.20) = 0.6250 (62.50%)
        assert a.peak_shock_pct == pytest.approx(62.48, abs=0.05)
        assert a.verdict == "DISRUPTIVE_SHOCK"

    def test_pathology_4_disconnected_ghost_island_zero_leakage(self) -> None:
        """Topology 4: Disjoint components must receive strictly zero leakage."""
        graph = CascadeGraph()
        # Component 1
        graph.add_edge(CascadeEdge(source="S", target="A", share=0.60, buffer_days=0.0))
        graph.add_edge(CascadeEdge(source="A", target="B", share=0.50, buffer_days=0.0))
        # Component 2 (Disjoint)
        graph.add_edge(CascadeEdge(source="X", target="Y", share=0.70, buffer_days=0.0))
        graph.add_edge(CascadeEdge(source="Y", target="Z", share=0.80, buffer_days=0.0))

        engine = SparseNetworkCascadeEngine(disruption_threshold_pct=25.0)
        results = engine.propagate(
            graph,
            initial_shocks={"S": 1.0},
            initial_durations={"S": 30.0},
        )

        # Component 1 impacted
        assert results["A"].peak_shock_pct == pytest.approx(60.0, abs=1e-3)
        assert results["B"].peak_shock_pct == pytest.approx(30.0, abs=1e-3)

        # Component 2 must be strictly zero-leakage
        for node in ["X", "Y", "Z"]:
            res = results[node]
            assert res.peak_shock_pct == 0.0000, f"Node {node} leaked energy"
            assert res.downtime_days == 0.0
            assert res.lost_capacity_days == 0.0000
            assert res.is_out_of_horizon, f"Node {node} should be out of horizon"
            assert res.verdict == "NEUTRAL_UNAFFECTED"

    def test_pathology_5_multi_pumping_resonant_cycle(self) -> None:
        """Topology 5: Multi-source pumping into coupled cycle S1 -> A <-> B <- S2."""
        graph = CascadeGraph()
        # S1 -> A (0.40), B -> A (0.40) in group "raw"
        graph.add_edge(CascadeEdge(source="S1", target="A", share=0.40, input_group="raw"))
        graph.add_edge(CascadeEdge(source="B", target="A", share=0.40, input_group="raw"))
        # S2 -> B (0.30), A -> B (0.50) in group "raw"
        graph.add_edge(CascadeEdge(source="S2", target="B", share=0.30, input_group="raw"))
        graph.add_edge(CascadeEdge(source="A", target="B", share=0.50, input_group="raw"))

        engine = SparseNetworkCascadeEngine(disruption_threshold_pct=25.0)
        results = engine.propagate(
            graph,
            initial_shocks={"S1": 1.0, "S2": 1.0},
            initial_durations={"S1": 20.0, "S2": 20.0},
        )

        # Under ADR-00006, both nodes are dist=1 from seeds, so dist(v) <= dist(u) (1 <= 1)
        # applies to both edges in the cycle, applying gamma=0.50 damping to both:
        # Effective weights: B -> A: 0.40 * 0.50 = 0.20; A -> B: 0.50 * 0.50 = 0.25.
        # Iteration 5 produces A = 48.40%, B = 42.075%.
        a = results["A"]
        b = results["B"]
        assert a.peak_shock_pct == pytest.approx(48.40, abs=0.05)
        assert b.peak_shock_pct == pytest.approx(42.08, abs=0.05)
        assert a.verdict == "DISRUPTIVE_SHOCK"
        assert b.verdict == "DISRUPTIVE_SHOCK"

    def test_pathology_6_fractal_nested_diamond_associativity(self) -> None:
        """Topology 6: Multi-tier nested diamond compounding distributes associatively."""
        graph = CascadeGraph()
        # Tier 3 -> Tier 2
        graph.add_edge(CascadeEdge(source="S1", target="M1", share=0.40))
        graph.add_edge(CascadeEdge(source="S2", target="M1", share=0.40))
        graph.add_edge(CascadeEdge(source="S1", target="M2", share=0.50))
        graph.add_edge(CascadeEdge(source="S2", target="M2", share=0.30))
        # Tier 2 -> Tier 1
        graph.add_edge(CascadeEdge(source="M1", target="C", share=0.50))
        graph.add_edge(CascadeEdge(source="M2", target="C", share=0.50))

        engine = SparseNetworkCascadeEngine(disruption_threshold_pct=25.0)
        results = engine.propagate(graph, initial_shocks={"S1": 1.0, "S2": 0.0})

        # M1 = 1.0 * 0.40 = 40.0%
        # M2 = 1.0 * 0.50 = 50.0%
        # C = 0.40 * 0.50 + 0.50 * 0.50 = 20% + 25% = 45.0%
        assert results["M1"].peak_shock_pct == pytest.approx(40.0, abs=1e-3)
        assert results["M2"].peak_shock_pct == pytest.approx(50.0, abs=1e-3)
        c = results["C"]
        assert c.peak_shock_pct == pytest.approx(45.0, abs=1e-3)
        assert c.verdict == "DISRUPTIVE_SHOCK"

    def test_pathology_7_extreme_buffer_asymmetry(self) -> None:
        """Topology 7: Parallel convergent branches with B=0 vs B=1,000,000 days."""
        graph = CascadeGraph()
        # Branch 1: unbuffered (B=0)
        graph.add_edge(CascadeEdge(source="S", target="B1", share=1.0, buffer_days=0.0))
        # Branch 2: immense buffer (B=1,000,000 days)
        graph.add_edge(CascadeEdge(source="S", target="B2", share=1.0, buffer_days=1_000_000.0))
        # Convergence onto F
        graph.add_edge(CascadeEdge(source="B1", target="F", share=0.50, buffer_days=0.0))
        graph.add_edge(CascadeEdge(source="B2", target="F", share=0.50, buffer_days=0.0))

        engine = SparseNetworkCascadeEngine(disruption_threshold_pct=25.0)
        results = engine.propagate(
            graph,
            initial_shocks={"S": 1.0},
            initial_durations={"S": 60.0},
        )

        b1 = results["B1"]
        assert b1.peak_shock_pct == pytest.approx(100.0, abs=1e-3)
        assert b1.downtime_days == 60.0
        assert b1.verdict == "DISRUPTIVE_SHOCK"

        b2 = results["B2"]
        assert b2.peak_shock_pct == pytest.approx(5.0, abs=1e-3)
        assert b2.downtime_days == 0.0
        assert b2.is_friction_floor
        assert b2.verdict == "TOLERATED_BUFFERED"

        f = results["F"]
        # Branch 2 is completely buffered, continuing uninterrupted production.
        # Target F experiences capacity loss exclusively from unbuffered Branch 1 (0.50 * 100% = 50.0%)
        # with downtime matching Branch 1 (60.0 days) -> 30.0 lost capacity days.
        assert f.peak_shock_pct == pytest.approx(50.0, abs=1e-3)
        assert f.downtime_days == pytest.approx(60.0, abs=1e-3)
        assert f.lost_capacity_days == pytest.approx(30.0, abs=1e-3)
        assert f.verdict == "DISRUPTIVE_SHOCK"

    def test_pathology_8_high_degree_fan_out_fan_in_bottleneck(self) -> None:
        """Topology 8: 1 to 50 to 1 fan-out/fan-in bottleneck with exactly 1.0 sum."""
        graph = CascadeGraph()
        num_intermediates = 50
        share_per_intermediate = 1.0 / num_intermediates  # 0.02 (2.0%)

        for i in range(num_intermediates):
            m_id = f"M_{i:02d}"
            # S -> M_i: 100% share
            graph.add_edge(CascadeEdge(source="S", target=m_id, share=1.0, buffer_days=0.0))
            # M_i -> C: 2% share
            graph.add_edge(CascadeEdge(source=m_id, target="C", share=share_per_intermediate, buffer_days=0.0))

        engine = SparseNetworkCascadeEngine(disruption_threshold_pct=25.0)

        start = time.perf_counter()
        results = engine.propagate(
            graph,
            initial_shocks={"S": 1.0},
            initial_durations={"S": 30.0},
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        c = results["C"]
        # 50 * (1.0 * 0.02) = 1.00 (100.0%)
        assert c.peak_shock_pct == pytest.approx(100.0, abs=1e-3)
        assert c.downtime_days == pytest.approx(30.0, abs=1e-3)
        assert c.lost_capacity_days == pytest.approx(30.0, abs=1e-3)
        assert c.verdict == "DISRUPTIVE_SHOCK"

        # Check latency is under 5.0 ms
        assert elapsed_ms < 5.0, f"Execution took {elapsed_ms:.2f}ms > 5.0ms budget"
