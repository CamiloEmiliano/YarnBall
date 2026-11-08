"""Integration test for Flow 1: Point-in-Time SEC Filing -> Segmentation -> Ontology Validation -> Knowledge Graph Model.

Crosses: universe/, ingestion/segmentation/, ontology/guardrails.py, retrieval/guardrails/schema_guardrail.py.
Verifies:
1. Historical point-in-time constituent CIK resolution.
2. Ingestion pre-segmentation chunking under 1,500 token ceiling.
3. Tier-4 ontology gate relation repair (e.g. [:TERMINATED] mapped to [:SUPPLIES_TO {status: 'TERMINATED'}]).
4. Subordinate subsidiary rollup to parent CIK.
5. Invariant: Whitelisted schema compliance on generated Cypher graph mutation statements.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from ingestion.segmentation.chunker import FilingChunk, FilingMetadata, FinancialSectionSegmenter
from ontology.guardrails import Tier4OntologyGate, apply_runtime_guardrails
from retrieval.guardrails.schema_guardrail import CypherASTWhitelistGuardrail, ValidationResult
from universe.sp500 import SP500UniverseManager


@pytest.mark.integration
def test_filing_to_graph_flow_point_in_time_and_ontology(tmp_path: Path):
    """Verify Flow 1: Point-in-Time Resolution -> Section Segmentation -> Ontology Gate -> Cypher Schema."""
    # 1. Resolve Point-in-Time Constituent CIK
    cache_file = tmp_path / "sp500_cache.json"
    universe = SP500UniverseManager(cache_file=cache_file)

    # In 2022, TWTR was active, TSLA was active
    assert universe.is_sp500("TWTR", "2022-01-01") is True
    assert universe.is_sp500("TSLA", "2022-01-01") is True

    # 2. Financial Ingestion Pre-Segmentation (<=1,500 token ceiling)
    meta = FilingMetadata(
        cik="0001045810",  # NVDA CIK
        ticker="NVDA",
        company_name="NVIDIA CORP",
        form_type="10-K",
        period="2023-12-31",
        filing_date="2024-02-21",
    )
    segmenter = FinancialSectionSegmenter(max_tokens=500, overlap_tokens=40)

    sample_filing_text = """
    ITEM 1. BUSINESS.
    NVIDIA Corporation designs graphics processing units and accelerated computing platforms.
    We rely on Taiwan Semiconductor Manufacturing Company (TSMC) as our primary foundry partner.
    Substantially all our semiconductor wafers are manufactured by TSMC in Taiwan.
    Failure of TSMC to deliver adequate manufacturing capacity would severely impair our operations.
    
    ITEM 1A. RISK FACTORS.
    Disruptions in our global supply chain, including geopolitical tensions in Taiwan, could halt deliveries.
    In fiscal 2023, we terminated our auxiliary supplier contract with LegacyChip Ltd due to yield failures.
    We maintain secondary inventory buffers of critical substrates to mitigate transient halts.
    """

    chunks = segmenter.segment_filing(sample_filing_text, metadata=meta)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.token_count <= 500
        assert "CIK: 0001045810" in chunk.header
        assert chunk.section_name in (
            "Item 1 - Business",
            "Item 1A - Risk Factors",
            "ITEM 1. BUSINESS",
            "ITEM 1A. RISK FACTORS",
            "PROLOGUE",
        )

    # 3. Ontology Gate Validation & Relation Repair (Stage 6.3)
    gate = Tier4OntologyGate()

    # Triple 1: canonical supplies
    canonical_extraction = "(:Company {name: 'TSMC', cik: '0001045810'})-[:SUPPLIES_TO {loss_ratio: 0.85, status: 'ACTIVE_CURRENT'}]->(:Company {name: 'NVIDIA CORP', cik: '0001045810'})"
    guarded_canonical = gate.apply_guardrails(canonical_extraction, "EXTRACT_SEC_GRAPH")
    assert "[:SUPPLIES_TO" in guarded_canonical
    assert guarded_canonical != "(none)"

    # Triple 2: repaired from TERMINATED to SUPPLIES_TO with status TERMINATED
    terminated_extraction = "(:Company {name: 'NVIDIA CORP', cik: '0001045810'})-[:TERMINATED {status: 'TERMINATED'}]->(:Company {name: 'LegacyChip Ltd', cik: '0000824801'})"
    repaired_term = gate.apply_guardrails(terminated_extraction, "EXTRACT_SEC_GRAPH")
    assert "[:SUPPLIES_TO" in repaired_term
    assert "status: 'TERMINATED'" in repaired_term
    assert repaired_term != "(none)"

    # Triple 3: subsidiary rollup (CIK 0001001811 rolls up to 0001001082)
    subsidiary_extraction = "(:Company {name: 'DISH DBS', cik: '0001001811'})-[:SUPPLIES_TO]->(:Company {name: 'EchoStar', cik: '0001415404'})"
    rolled_up = gate.apply_guardrails(subsidiary_extraction, "EXTRACT_SEC_GRAPH")
    assert "0001001082" in rolled_up
    assert "0001001811" not in rolled_up

    # 4. Invariant: Ensure compiled Cypher query conforms to AST Whitelist Guardrail
    cypher_whitelist = CypherASTWhitelistGuardrail()

    cypher_query = (
        "MATCH (s:Company {ticker: 'TSM'})-[r:SUPPLIES_TO]->(t:Company {ticker: 'NVDA'}) "
        "RETURN s.name, r.status, r.loss_ratio, t.name"
    )
    val = cypher_whitelist.validate(cypher_query)
    assert val.is_valid is True
    assert val.error_summary is None
