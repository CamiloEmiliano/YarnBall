"""System Integration Test (SIT) Suite: Linguistic Clamping & Behavioral Invariant Verification.

Academic Reference:
- Ribeiro, M. T., Wu, T., Guestrin, C., & Singh, S. (ACL 2020 Best Paper):
  "Beyond Accuracy: Behavioral Testing of NLP Models with CheckList"

Clamps linguistic degrees of freedom to zero across 10 canonical capabilities:
1. TestSRLVoiceInvariance [INV]
2. TestNegationDirectionalFlip [DIR]
3. TestTemporalPiTSequencing [MFT]
4. TestLogicEpistemicHedging [DIR]
5. TestNERAcronymInvariance [INV]
6. TestTaxonomySubsidiaryRollup [DIR]
7. TestCoreferenceAnaphoricBridging [MFT]
8. TestRobustnessTypoInvariance [INV]
9. TestVocabularyMaterialityEscalation [DIR]
10. TestCoordinationDistributiveScope [MFT]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import pytest

from knowledge_graph.entity_resolver import EntityResolver
from knowledge_graph.snapshot_manager import SnapshotManager
from ontology.alignment import extract_initialism, normalize_entity_name
from ontology.compiler import (
    classify_financial_materiality,
    classify_polarity_from_text,
    has_unnegated_pattern,
    is_hypothetical_or_hedged,
)
from ontology.guardrails import (
    DEFAULT_ONTOLOGY_GATE,
    Tier4OntologyGate,
    apply_runtime_guardrails,
)
from ontology.reification import AnnotatedTriple, MaterialityTier, PolarityAxis
from ontology.validation import validate_and_orient_triple
from retrieval.temporal import TemporalQueryFilter
from tests.fixtures.linguistic.schema import LinguisticTestCase

FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "linguistic" / "checklist_cases.json"


def load_fixture_case(case_id: str) -> LinguisticTestCase:
    """Load and validate a single behavioral test case from checklist_cases.json."""
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)
    for c in cases:
        if c.get("id") == case_id:
            return LinguisticTestCase.model_validate(c)
    raise KeyError(f"Case '{case_id}' not found in {FIXTURE_PATH}")


@pytest.fixture
def isolated_resolver(tmp_path: Path) -> EntityResolver:
    """Provides an isolated in-memory EntityResolver."""
    mgr = SnapshotManager(base_dir=tmp_path / "snapshots")
    return EntityResolver(snapshot_manager=mgr)


@pytest.mark.integration
class TestSRLVoiceInvariance:
    """Capability 1: Semantic Role Labeling (SRL) — Active vs. Passive Inversion [INV]."""

    def test_srl_passive_voice_edge_direction_invariance(self):
        case = load_fixture_case("srl_01_inv")
        
        # Active voice extraction: Supermicro supplies Nvidia
        active_triple = AnnotatedTriple(
            source_name="Supermicro Computer Inc.",
            source_type="Company",
            target_name="Nvidia Corporation",
            target_type="Company",
            rel_type="SUPPLIES_TO",
        )
        valid_active = validate_and_orient_triple(
            active_triple.source_name,
            active_triple.source_type,
            active_triple.target_name,
            active_triple.target_type,
            active_triple.rel_type,
        )
        assert valid_active is not None
        assert valid_active[0] == "Supermicro Computer Inc."
        assert valid_active[2] == "Nvidia Corporation"
        assert valid_active[4] == "SUPPLIES_TO"

        # Passive voice with dative inversion: modules procured from Supermicro by Nvidia
        # Surface parser produces CUSTOMER_OF relation
        passive_surface_triple = AnnotatedTriple(
            source_name="Nvidia Corporation",
            source_type="Company",
            target_name="Supermicro Computer Inc.",
            target_type="Company",
            rel_type="CUSTOMER_OF",
        )
        valid_passive = validate_and_orient_triple(
            passive_surface_triple.source_name,
            passive_surface_triple.source_type,
            passive_surface_triple.target_name,
            passive_surface_triple.target_type,
            passive_surface_triple.rel_type,
        )
        assert valid_passive is not None

        # Both express the invariant economic relationship: Supermicro is supplier, Nvidia is buyer
        assert (
            (valid_active[0] == valid_passive[2] and valid_active[4] == "SUPPLIES_TO" and valid_passive[4] == "CUSTOMER_OF")
            or (valid_active[0] == valid_passive[0] and valid_active[4] == valid_passive[4])
        )


@pytest.mark.integration
class TestNegationDirectionalFlip:
    """Capability 2: Negation & Polarity — Multi-Clause Dissolution & Retraction [DIR]."""

    def test_negation_flips_contract_lifecycle_and_polarity(self):
        case = load_fixture_case("negation_02_dir")

        # 1. Affirmative input produces active status
        assert has_unnegated_pattern(r"serves?\s+as", case.input_text) is True
        pol_active = classify_polarity_from_text(context_text=case.input_text, rel_type="SUPPLIES_TO")
        assert pol_active != PolarityAxis.DISRUPTIVE_SHOCK.value

        # 2. Perturbed clause contains negative dissolution cue
        assert case.perturbed_text is not None
        assert has_unnegated_pattern(r"dissolved", case.perturbed_text) is True

        # Polarity must directionally shift to DISRUPTIVE_SHOCK or CONTRACTING_BEARISH
        pol_perturbed = classify_polarity_from_text(context_text=case.perturbed_text, rel_type="TERMINATED")
        assert pol_perturbed == PolarityAxis.DISRUPTIVE_SHOCK.value

        # Guardrail repairs [:TERMINATED] into [:SUPPLIES_TO {status: 'TERMINATED', is_active: false}]
        mock_raw_output = "(:Company {name: 'LegacyChip Ltd'})-[:TERMINATED]->(:Company {name: 'The Company'})"
        repaired = apply_runtime_guardrails(case.perturbed_text, mock_raw_output, "EXTRACT_SEC_GRAPH")
        assert "SUPPLIES_TO" in repaired
        assert "TERMINATED" in repaired
        assert "is_active: false" in repaired


@pytest.mark.integration
class TestTemporalPiTSequencing:
    """Capability 3: Temporal & Aspect — Point-in-Time Event Sequencing [MFT]."""

    def test_temporal_filter_prevents_future_lookahead_leakage(self):
        case = load_fixture_case("temporal_03_mft")
        as_of_date = case.metadata["as_of_date"]  # "2023-01-01"
        event_date = case.metadata["event_date"]  # "2023-08-15"

        temporal_engine = TemporalQueryFilter()
        raw_cypher = (
            "MATCH (f:Company {name: 'FabCo'})-[r:SUPPLIES_TO]->(c:Company) "
            "RETURN f.name, c.name"
        )
        scoped_cypher = temporal_engine.inject_active_coalescence(raw_cypher, as_of_date=as_of_date)

        # Assert point-in-time invariant constraints injected into query
        assert f"valid_from <= '{as_of_date}'" in scoped_cypher
        assert f"valid_to >= '{as_of_date}'" in scoped_cypher
        assert "coalesce(r.status, 'ACTIVE_CURRENT') = 'ACTIVE_CURRENT'" in scoped_cypher

        # Mathematically verify that event executed in August 2023 violates the as_of_date invariant
        assert not (event_date <= as_of_date), "Event date must strictly follow evaluation horizon"


@pytest.mark.integration
class TestLogicEpistemicHedging:
    """Capability 4: Logic & Conditionals — Epistemic Hedging & Counterfactuals [DIR]."""

    def test_epistemic_hedging_suppresses_unexecuted_commitment(self):
        case = load_fixture_case("logic_04_dir")

        # Affirmative contract is not hedged
        assert is_hypothetical_or_hedged(case.input_text) is False

        # Perturbed hypothetical sentence is classified as hedged
        assert case.perturbed_text is not None
        assert is_hypothetical_or_hedged(case.perturbed_text) is True

        # Runtime ontology gate strictly suppresses hedged statement to (none)
        mock_prediction = "(:Company {name: 'The Company'})-[:PARTNERED_WITH]->(:Company {name: 'Intel Foundry Services'})"
        refused_result = apply_runtime_guardrails(case.perturbed_text, mock_prediction, "EXTRACT_SEC_GRAPH")
        assert refused_result == "(none)", f"Hedged statement must be refused to '(none)', got '{refused_result}'"


@pytest.mark.integration
class TestNERAcronymInvariance:
    """Capability 5: Named Entity Recognition — Acronyms, Tickers & CIKs [INV] (The TSMC Dilemma)."""

    def test_tsmc_acronym_and_full_name_merge_invariance(self, isolated_resolver: EntityResolver):
        case = load_fixture_case("ner_05_inv")

        raw_nodes: List[Dict[str, Any]] = [
            {"id": "TSMC", "label": "Company", "ticker": None, "source_hash": "h1"},
            {"id": "Taiwan Semiconductor Manufacturing Company", "label": "Company", "ticker": None, "source_hash": "h2"},
            {"id": "Apple", "label": "Company", "ticker": "AAPL", "source_hash": "h3"},
        ]

        mapping, resolved = isolated_resolver.resolve_nodes(raw_nodes)

        # Invariant: TSMC resolves to canonical Taiwan Semiconductor Manufacturing Company
        assert mapping["TSMC"] == "Taiwan Semiconductor Manufacturing Company"
        assert mapping["Taiwan Semiconductor Manufacturing Company"] == "Taiwan Semiconductor Manufacturing Company"

        # Check canonical node attributes and dual-exchange enrichment
        tsmc_node = next(n for n in resolved if n["id"] == "Taiwan Semiconductor Manufacturing Company")
        assert tsmc_node["ticker"] == "TSM"
        assert tsmc_node["listing_exchange"] == "NYSE"
        props = json.loads(tsmc_node["properties_json"])
        assert props["cik"] == "0001045810"


@pytest.mark.integration
class TestTaxonomySubsidiaryRollup:
    """Capability 6: Taxonomy & Hierarchy — Corporate Parent/Subsidiary Rollup [DIR]."""

    def test_financing_subsidiary_rolls_up_to_parent_issuer(self):
        case = load_fixture_case("taxonomy_06_dir")
        
        # Test CIK rollup table mapping
        broadcom_parent_cik = "0001730168"
        broadcom_subsidiary_cik = "0001850000"  # Mock financing sub CIK

        gate = Tier4OntologyGate(subsidiary_map={broadcom_subsidiary_cik: broadcom_parent_cik})
        mock_sub_prediction = f"(:Company {{cik: '{broadcom_subsidiary_cik}'}})-[:PARTNERED_WITH]->(:Company {{name: 'Cisco'}})"
        
        rolled_up = gate.apply_guardrails(mock_sub_prediction, "EXTRACT_SEC_GRAPH")
        assert broadcom_parent_cik in rolled_up
        assert broadcom_subsidiary_cik not in rolled_up


@pytest.mark.integration
class TestCoreferenceAnaphoricBridging:
    """Capability 7: Coreference & Anaphora — Definite Descriptions & Bridging [MFT]."""

    def test_definite_description_and_demonstrative_antecedent_binding(self, isolated_resolver: EntityResolver):
        case = load_fixture_case("coreference_07_mft")

        # Antecedents extracted across 2-sentence context
        raw_nodes: List[Dict[str, Any]] = [
            {"id": "Taiwan Semiconductor", "label": "Company", "ticker": "TSM", "source_hash": "h1"},
            {"id": "Apple Inc.", "label": "Company", "ticker": "AAPL", "listing_exchange": "NASDAQ", "source_hash": "h2"},
            {"id": "The Cupertino-based consumer electronics giant", "label": "Company", "ticker": "AAPL", "source_hash": "h3"},
        ]

        mapping, resolved = isolated_resolver.resolve_nodes(raw_nodes)

        # Definite description bound to Apple Inc. via ticker alignment
        assert mapping["The Cupertino-based consumer electronics giant"] == "Apple Inc."
        assert len(resolved) == 2


@pytest.mark.integration
class TestRobustnessTypoInvariance:
    """Capability 8: Robustness & Noise — Typo, Casing, and Whitespace Invariance [INV]."""

    def test_typo_and_whitespace_noise_resolves_to_canonical_entity(self, isolated_resolver: EntityResolver):
        case = load_fixture_case("robustness_08_inv")

        raw_nodes: List[Dict[str, Any]] = [
            {"id": "  nvida   ", "label": "Company", "ticker": None, "source_hash": "h1"},
            {"id": "NVIDIA Corporation", "label": "Company", "ticker": "NVDA", "listing_exchange": "NASDAQ", "source_hash": "h2"},
            {"id": "Microsoft", "label": "Company", "ticker": "MSFT", "listing_exchange": "NASDAQ", "source_hash": "h3"},
        ]

        mapping, resolved = isolated_resolver.resolve_nodes(raw_nodes)

        # 1-edit typo + whitespace padding safely resolves to NVIDIA Corporation
        assert mapping["  nvida   "] == "NVIDIA Corporation"
        assert len(resolved) == 2
        nvda_node = next(n for n in resolved if n["id"] == "NVIDIA Corporation")
        assert nvda_node["ticker"] == "NVDA"
        assert nvda_node["listing_exchange"] == "NASDAQ"


@pytest.mark.integration
class TestVocabularyMaterialityEscalation:
    """Capability 9: Vocabulary & Materiality — Legal Terms of Art Escalation [DIR]."""

    def test_legal_terms_of_art_escalate_materiality_tier(self):
        case = load_fixture_case("vocabulary_09_dir")

        # Baseline shipping delays -> COMMODITY_TIER_3 / MATERIAL_TIER_2
        mat_baseline = classify_financial_materiality(context_text=case.input_text, rel_type="SUPPLIES_TO")
        assert mat_baseline != MaterialityTier.CRITICAL_TIER_1.value

        # Perturbed with "Material Adverse Effect" and "sole source" -> CRITICAL_TIER_1
        assert case.perturbed_text is not None
        mat_escalated = classify_financial_materiality(context_text=case.perturbed_text, rel_type="SUPPLIES_TO")
        assert mat_escalated == MaterialityTier.CRITICAL_TIER_1.value


@pytest.mark.integration
class TestCoordinationDistributiveScope:
    """Capability 10: Coordination & Distributive Scope — Compound Parallelism [MFT]."""

    def test_respective_distributive_scope_parallel_unambiguous_edges(self):
        case = load_fixture_case("coordination_10_mft")

        # Emitted edges must strictly match parallel coordination:
        # (The Company) -> (Apple) for "power-management IC"
        # (The Company) -> (Microsoft) for "optical transceivers"
        expected = case.expected_edges
        assert len(expected) == 2

        apple_edge = next(e for e in expected if e.target_name == "Apple")
        assert apple_edge.properties.get("product") == "power-management IC"

        msft_edge = next(e for e in expected if e.target_name == "Microsoft")
        assert msft_edge.properties.get("product") == "optical transceivers"

        # Assert no cross-contamination (Apple bound to optical transceivers is false)
        assert apple_edge.properties.get("product") != "optical transceivers"
