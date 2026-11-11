"""ADR-00007 Canonical Ground-Truth Industrial Archetypes Suite for SparseNetworkCascadeEngine.

Executes closed-form verification of all 10 well-behaved canonical supply-chain topologies:
1. 3-Tier Compound Diamond (EV Battery Supply Chain)
2. Dual-Source with Buffer & Net Deficit (Ford / Simchi-Levi Model)
3. Interlocking Butterfly / Shared Bottleneck (TSMC Packaging Model)
4. Hybrid Multi-Input Aerospace Assembly (Complements + Substitutes)
5. Nested Coupled Feedback Loops (Foundry & Lithography Toolmaker)
6. Staggered Asymmetric Temporal Waves
7. Deep 7-Hop Linear Chain with Dyadic Decay (2^-k)
8. Dense Directed Web (K3 Strongly Connected Clique)
9. Negative Control: Friction Floor Non-Propagation
10. Shared Maritime Logistics Chokepoint (Suez / Panama Canal)
"""

import pytest

from contagion.cascade import (
    CascadeEdge,
    CascadeGraph,
    SparseNetworkCascadeEngine,
)


class TestCascadeCanonicalTopologies:
    """Verifies SparseNetworkCascadeEngine against ADR-00007 canonical industrial archetypes."""

    def test_canonical_1_three_tier_compound_diamond(self) -> None:
        """Canonical 1: 3-tier compound diamond (8 distinct paths) with additive Leontief aggregation."""
        graph = CascadeGraph()
        # Tier 3 (Mines) -> Tier 2 (Refiners)
        for mine in ["L1", "L2"]:
            for refiner in ["R1", "R2"]:
                graph.add_edge(CascadeEdge(source=mine, target=refiner, share=0.50))
        # Tier 2 (Refiners) -> Tier 1 (Gigafactories)
        for refiner in ["R1", "R2"]:
            for giga in ["G1", "G2"]:
                graph.add_edge(CascadeEdge(source=refiner, target=giga, share=0.50))
        # Tier 1 (Gigafactories) -> OEM
        for giga in ["G1", "G2"]:
            graph.add_edge(CascadeEdge(source=giga, target="OEM", share=0.50))

        engine = SparseNetworkCascadeEngine(disruption_threshold_pct=25.0)

        # Simultaneous 100% outage on both raw material sources
        results = engine.propagate(
            graph,
            initial_shocks={"L1": 1.0, "L2": 1.0},
            initial_durations={"L1": 30.0, "L2": 30.0},
        )

        oem = results["OEM"]
        assert oem.peak_shock_pct == pytest.approx(100.0, abs=1e-3)
        assert oem.downtime_days == pytest.approx(30.0, abs=1e-3)
        assert oem.lost_capacity_days == pytest.approx(30.0, abs=1e-3)
        assert oem.verdict == "DISRUPTIVE_SHOCK"

        # Single mine shock test: 50% linear transmission preserved
        results_single = engine.propagate(
            graph,
            initial_shocks={"L1": 1.0, "L2": 0.0},
            initial_durations={"L1": 30.0, "L2": 0.0},
        )
        assert results_single["OEM"].peak_shock_pct == pytest.approx(50.0, abs=1e-3)

    def test_canonical_2_dual_source_surge_and_buffer(self) -> None:
        """Canonical 2: Dual-source assembly with inventory buffer absorption and net downtime."""
        graph = CascadeGraph()
        # Primary supplier: 75% share, unbuffered
        graph.add_edge(CascadeEdge(source="S1", target="OEM", share=0.75, buffer_days=0.0))
        # Secondary supplier: 25% share, 20 days inventory buffer
        graph.add_edge(CascadeEdge(source="S2", target="OEM", share=0.25, buffer_days=20.0))

        engine = SparseNetworkCascadeEngine(disruption_threshold_pct=25.0)
        # S2 halts for 60 days
        results = engine.propagate(
            graph,
            initial_shocks={"S2": 1.0},
            initial_durations={"S2": 60.0},
        )

        oem = results["OEM"]
        # Capacity loss = 25% share * 100% = 25.0%
        assert oem.peak_shock_pct == pytest.approx(25.0, abs=1e-3)
        # 60d - 20d buffer = 40 days downtime
        assert oem.downtime_days == pytest.approx(40.0, abs=1e-3)
        assert oem.onset_day == pytest.approx(20.0, abs=1e-3)
        # Lost capacity days = 0.25 * 40 = 10.0 lost days
        assert oem.lost_capacity_days == pytest.approx(10.0, abs=1e-3)
        assert oem.verdict == "DISRUPTIVE_SHOCK"

    def test_canonical_3_interlocking_butterfly_shared_bottleneck(self) -> None:
        """Canonical 3: Shared foundry M supplies two competing OEMs with disparate reliance ratios."""
        graph = CascadeGraph()
        graph.add_edge(CascadeEdge(source="A1", target="OEM1", share=0.40))
        graph.add_edge(CascadeEdge(source="M", target="OEM1", share=0.60))
        graph.add_edge(CascadeEdge(source="M", target="OEM2", share=0.30))
        graph.add_edge(CascadeEdge(source="A2", target="OEM2", share=0.70))

        engine = SparseNetworkCascadeEngine(disruption_threshold_pct=25.0)
        results = engine.propagate(
            graph,
            initial_shocks={"M": 0.50},
            initial_durations={"M": 90.0},
        )

        # OEM1 reliance is 60%: 0.50 * 0.60 = 30% -> DISRUPTIVE_SHOCK
        oem1 = results["OEM1"]
        assert oem1.peak_shock_pct == pytest.approx(30.0, abs=1e-3)
        assert oem1.verdict == "DISRUPTIVE_SHOCK"

        # OEM2 reliance is 30%: 0.50 * 0.30 = 15% -> CONTAINED_MINIMAL_CONTAGION
        oem2 = results["OEM2"]
        assert oem2.peak_shock_pct == pytest.approx(15.0, abs=1e-3)
        assert oem2.verdict == "CONTAINED_MINIMAL_CONTAGION"

    def test_canonical_4_aerospace_hybrid_assembly(self) -> None:
        """Canonical 4: Distinct complementary input groups combine via Leontief maximization."""
        graph = CascadeGraph()
        # Airframe group
        graph.add_edge(CascadeEdge(source="T1", target="OEM", share=0.50, input_group="airframe"))
        graph.add_edge(CascadeEdge(source="T2", target="OEM", share=0.50, input_group="airframe"))
        # Engines group
        graph.add_edge(CascadeEdge(source="E1", target="OEM", share=0.60, input_group="engines"))
        graph.add_edge(CascadeEdge(source="E2", target="OEM", share=0.40, input_group="engines"))
        # Avionics group
        graph.add_edge(CascadeEdge(source="AV", target="OEM", share=1.00, input_group="avionics"))

        engine = SparseNetworkCascadeEngine(disruption_threshold_pct=25.0)
        results = engine.propagate(
            graph,
            initial_shocks={"T1": 0.60, "E1": 0.50, "AV": 0.20},
        )

        # Airframe: 0.60 * 0.50 = 30%
        # Engines: 0.50 * 0.60 = 30%
        # Avionics: 0.20 * 1.00 = 20%
        # Leontief max: max(30%, 30%, 20%) = 30%
        oem = results["OEM"]
        assert oem.peak_shock_pct == pytest.approx(30.0, abs=1e-3)
        assert oem.verdict == "DISRUPTIVE_SHOCK"

    def test_canonical_5_coupled_feedback_loops(self) -> None:
        """Canonical 5: Multi-loop cyclic feedback between Foundry F, Toolmaker T, and Wafer W."""
        graph = CascadeGraph()
        # Seed S -> F (0.40)
        graph.add_edge(CascadeEdge(source="S", target="F", share=0.40))
        # Forward F -> T (0.50)
        graph.add_edge(CascadeEdge(source="F", target="T", share=0.50))
        # Feedback T -> F (0.30)
        graph.add_edge(CascadeEdge(source="T", target="F", share=0.30))
        # Forward T -> W (0.30)
        graph.add_edge(CascadeEdge(source="T", target="W", share=0.30))
        # Feedback W -> F (0.30)
        graph.add_edge(CascadeEdge(source="W", target="F", share=0.30))

        engine = SparseNetworkCascadeEngine(disruption_threshold_pct=25.0)
        results = engine.propagate(graph, initial_shocks={"S": 1.0})

        # At iteration 5 with gamma=0.50 damping on back-edges:
        # F ~= 44.125%, T ~= 21.95%, W ~= 6.45%
        assert results["F"].peak_shock_pct == pytest.approx(44.125, abs=0.05)
        assert results["T"].peak_shock_pct == pytest.approx(21.95, abs=0.05)
        assert results["W"].peak_shock_pct == pytest.approx(6.45, abs=0.05)
        assert results["F"].verdict == "DISRUPTIVE_SHOCK"

    def test_canonical_6_staggered_temporal_waves(self) -> None:
        """Canonical 6: Multi-parent temporal window union under staggered disruption onsets."""
        graph = CascadeGraph()
        graph.add_edge(CascadeEdge(source="S1", target="C", share=0.50, buffer_days=15.0))
        graph.add_edge(CascadeEdge(source="S2", target="C", share=0.50, buffer_days=10.0))

        engine = SparseNetworkCascadeEngine(disruption_threshold_pct=25.0)
        # S1: T=0, D=60, B=15 -> on at T=15, end at T=60
        # S2: T=30, D=60, B=10 -> on at T=40, end at T=90
        # Convergent C: onset = min(15, 40) = 15; end = max(60, 90) = 90 -> downtime = 75 days
        results = engine.propagate(
            graph,
            initial_shocks={"S1": 1.0, "S2": 1.0},
            initial_durations={"S1": 60.0, "S2": 60.0},
            initial_onsets={"S1": 0.0, "S2": 30.0},
        )

        c = results["C"]
        assert c.onset_day == pytest.approx(15.0, abs=1e-3)
        assert c.downtime_days == pytest.approx(75.0, abs=1e-3)
        assert c.lost_capacity_days == pytest.approx(75.0, abs=1e-3)
        assert c.peak_shock_pct == pytest.approx(100.0, abs=1e-3)
        assert c.verdict == "DISRUPTIVE_SHOCK"

    def test_canonical_7_deep_7hop_dyadic_decay(self) -> None:
        """Canonical 7: Deep 7-hop linear chain with 2^-k dyadic decay and out-of-horizon tagging."""
        graph = CascadeGraph()
        nodes = ["S"] + [f"T{i}" for i in range(1, 8)]
        for i in range(len(nodes) - 1):
            graph.add_edge(CascadeEdge(source=nodes[i], target=nodes[i + 1], share=0.50))

        engine = SparseNetworkCascadeEngine(disruption_threshold_pct=25.0, max_iterations=5)
        results = engine.propagate(
            graph,
            initial_shocks={"S": 1.0},
            initial_durations={"S": 30.0},
        )

        # Hops 1 to 5: exact powers of 0.5
        assert results["T1"].peak_shock_pct == pytest.approx(50.0, abs=1e-3)
        assert results["T2"].peak_shock_pct == pytest.approx(25.0, abs=1e-3)
        assert results["T3"].peak_shock_pct == pytest.approx(12.5, abs=1e-3)
        assert results["T4"].peak_shock_pct == pytest.approx(6.25, abs=1e-3)
        assert results["T5"].peak_shock_pct == pytest.approx(3.125, abs=1e-3)

        # Hops 6 & 7: beyond 5-iteration horizon
        assert results["T6"].is_out_of_horizon
        assert results["T6"].peak_shock_pct == 0.0000
        assert results["T6"].verdict == "NEUTRAL_UNAFFECTED"

        assert results["T7"].is_out_of_horizon
        assert results["T7"].peak_shock_pct == 0.0000
        assert results["T7"].verdict == "NEUTRAL_UNAFFECTED"

    def test_canonical_8_k3_strongly_connected_clique(self) -> None:
        """Canonical 8: Dense directed web (K3 clique) terminates stably without overflow."""
        graph = CascadeGraph()
        graph.add_edge(CascadeEdge(source="S", target="A", share=0.34))
        graph.add_edge(CascadeEdge(source="A", target="B", share=0.33))
        graph.add_edge(CascadeEdge(source="B", target="A", share=0.33))
        graph.add_edge(CascadeEdge(source="B", target="C", share=0.33))
        graph.add_edge(CascadeEdge(source="C", target="B", share=0.33))
        graph.add_edge(CascadeEdge(source="C", target="A", share=0.33))
        graph.add_edge(CascadeEdge(source="A", target="C", share=0.33))

        engine = SparseNetworkCascadeEngine(disruption_threshold_pct=25.0)
        results = engine.propagate(graph, initial_shocks={"S": 1.0})

        # All nodes terminate stably strictly below 1.0
        assert results["A"].peak_shock_pct == pytest.approx(38.82, abs=0.1)
        assert results["B"].peak_shock_pct == pytest.approx(15.05, abs=0.1)
        assert results["C"].peak_shock_pct == pytest.approx(15.05, abs=0.1)
        assert results["A"].verdict == "DISRUPTIVE_SHOCK"
        assert results["B"].verdict == "CONTAINED_MINIMAL_CONTAGION"

    def test_canonical_9_friction_floor_non_propagation(self) -> None:
        """Canonical 9: Restocking friction floor on buffered supplier B must NOT leak to customer C."""
        graph = CascadeGraph()
        # S -> B: shock 30d absorbed by 60d buffer
        graph.add_edge(CascadeEdge(source="S", target="B", share=1.0, buffer_days=60.0))
        # B -> C: unbuffered link
        graph.add_edge(CascadeEdge(source="B", target="C", share=1.0, buffer_days=0.0))

        engine = SparseNetworkCascadeEngine(disruption_threshold_pct=25.0)
        results = engine.propagate(
            graph,
            initial_shocks={"S": 1.0},
            initial_durations={"S": 30.0},
        )

        # Supplier B absorbed shock -> friction floor 5%
        b = results["B"]
        assert b.peak_shock_pct == pytest.approx(5.0, abs=1e-3)
        assert b.downtime_days == 0.0
        assert b.is_friction_floor
        assert b.verdict == "TOLERATED_BUFFERED"

        # Customer C experiences strictly ZERO downtime and ZERO capacity shock
        c = results["C"]
        assert c.peak_shock_pct == 0.0000
        assert c.downtime_days == 0.0
        assert c.lost_capacity_days == 0.0000
        assert not c.is_friction_floor
        assert c.verdict == "NEUTRAL_UNAFFECTED"

    def test_canonical_10_shared_logistics_chokepoint(self) -> None:
        """Canonical 10: Multi-supplier freight consolidation through shared logistics hub."""
        graph = CascadeGraph()
        for s in ["S1", "S2", "S3", "S4"]:
            graph.add_edge(CascadeEdge(source=s, target="L", share=0.25))
        graph.add_edge(CascadeEdge(source="L", target="C", share=1.00))

        engine = SparseNetworkCascadeEngine(disruption_threshold_pct=25.0)
        # Logistics hub suffers 30% throughput slowdown for 45 days
        results = engine.propagate(
            graph,
            initial_shocks={"L": 0.30},
            initial_durations={"L": 45.0},
        )

        c = results["C"]
        assert c.peak_shock_pct == pytest.approx(30.0, abs=1e-3)
        assert c.downtime_days == pytest.approx(45.0, abs=1e-3)
        assert c.lost_capacity_days == pytest.approx(13.5, abs=1e-3)
        assert c.verdict == "DISRUPTIVE_SHOCK"
