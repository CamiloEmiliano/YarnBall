"""Mechanized integrity & zero-drift tests for the SEC EDGAR Filing Canon (ADR-00045)."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from ingestion.sec import canon
from ingestion.sec.canon import (
    SEC_8K_PRIORITY_TIERS,
    SEC_EXHIBIT_CANON,
    SEC_FORM_CANON,
    SEC_ITEM_DESCRIPTIONS,
    SEC_SECTION_CANON,
    SECForm8KItem,
    SECFormCategory,
    SECFormStatus,
    SECFormType,
    SECGraphLayer,
    SECIngestionMode,
    get_default_harvest_forms,
    get_forms_by_layer,
    get_forms_by_status,
    get_sec_provenance_confidence_map,
    get_section_patterns,
    is_high_signal_section,
    is_termination_eligible_form,
    normalize_form_type,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CANON_PATH = REPO_ROOT / "ingestion" / "sec" / "canon.py"
RUNTIME_PACKAGES = frozenset(
    {
        "contagion",
        "curation",
        "embedding",
        "explorer",
        "ingestion",
        "knowledge_graph",
        "ontology",
        "retrieval",
        "sft_pipeline",
        "streaming",
        "synthesis",
        "tools",
        "universe",
    }
)


def test_canon_has_zero_internal_package_imports() -> None:
    """ADR-00045 Invariant #1: ingestion/sec/canon.py must depend strictly on Python stdlib."""
    tree = ast.parse(CANON_PATH.read_text(encoding="utf-8"), filename=str(CANON_PATH))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "Relative internal import forbidden inside ingestion/sec/canon.py"
            if node.module:
                imported_roots.add(node.module.split(".", 1)[0])

    forbidden = imported_roots & RUNTIME_PACKAGES
    assert not forbidden, f"ingestion/sec/canon.py must not import internal packages: {sorted(forbidden)}"


def test_sec_form_canon_complete_bijection_and_bounds() -> None:
    """ADR-00045 Invariant #2: Every SECFormType has a valid SECFormSpec with valid exhibits and bounds."""
    assert set(SEC_FORM_CANON.keys()) == set(SECFormType)
    assert len(SEC_FORM_CANON) == 33
    assert len(SEC_EXHIBIT_CANON) == 9
    assert len(SEC_SECTION_CANON) == 25

    for ft, spec in SEC_FORM_CANON.items():
        assert spec.form_type is ft
        assert isinstance(spec.category, SECFormCategory)
        assert isinstance(spec.graph_layer, SECGraphLayer)
        assert isinstance(spec.ingestion_mode, SECIngestionMode)
        assert isinstance(spec.status, SECFormStatus)
        assert 0.0 < spec.base_confidence <= 1.0
        for ex_key in spec.tracked_exhibits:
            assert ex_key in SEC_EXHIBIT_CANON, f"Unknown exhibit '{ex_key}' referenced by {ft.value}"

    assert SECFormType.FORM_13F_HR in get_forms_by_status(SECFormStatus.OPTIONAL_ABLATION)
    assert SECFormType.FORM_424B2 in get_forms_by_layer(SECGraphLayer.FINANCING_OFFERING)
    assert SECFormType.FORM_20F in get_forms_by_status(SECFormStatus.PLANNED)


def test_sec_8k_items_exhaustive_descriptions_and_tier_bijection() -> None:
    """ADR-00045 Invariant #3: All 32 SECForm8KItem members map cleanly to descriptions and exactly one tier."""
    from ingestion.sec import disruption_gating

    assert len(SECForm8KItem) == 32
    all_values = [item.value for item in SECForm8KItem]
    assert set(SEC_ITEM_DESCRIPTIONS.keys()) == set(all_values)
    for item_val, desc in SEC_ITEM_DESCRIPTIONS.items():
        assert isinstance(desc, str) and len(desc.strip()) > 0, f"Empty description for {item_val}"

    tier_items: list[str] = []
    tier_numbers: list[int] = []
    for tier, items in SEC_8K_PRIORITY_TIERS:
        tier_numbers.append(tier)
        tier_items.extend(items)

    assert tier_numbers == [1, 2, 3, 4]
    assert len(tier_items) == len(set(tier_items)), "Duplicate Form 8-K item across priority tiers"
    assert set(tier_items) == set(all_values), "SEC_8K_PRIORITY_TIERS must cover all 32 Form 8-K items"
    assert disruption_gating._TIER_ITEMS is SEC_8K_PRIORITY_TIERS


def test_termination_eligibility_strict_adr00033() -> None:
    """ADR-00045 Invariant #4 / ADR-00033 Invariant #8: Only 8-K and 8-K/A may close edge valid_to."""
    for allowed in ("8-K", "8-K/A", "8k", "8k/a", "Form 8-K", "SEC Form 8-K/A"):
        assert is_termination_eligible_form(allowed) is True, f"Expected {allowed!r} to allow termination"

    for disallowed in ("10-K", "10-K/A", "10-Q", "4", "20-F", "6-K", "13F-HR", "424B2", "", None):
        assert is_termination_eligible_form(disallowed) is False, f"Expected {disallowed!r} to block termination"


def test_default_harvest_forms_domain1_active_only() -> None:
    """ADR-00045 Invariant #7 / ADR-00046: Default Domain-1 batch forms cover all 6 core active forms."""
    defaults = get_default_harvest_forms()
    assert defaults == ("10-K", "10-K/A", "10-Q", "10-Q/A", "8-K", "8-K/A")
    for form_code in defaults:
        spec = canon.get_form_spec(form_code)
        assert spec is not None
        assert spec.status == SECFormStatus.ACTIVE
        assert spec.graph_layer == SECGraphLayer.DOMAIN1_OPERATIONAL
    assert SEC_EXHIBIT_CANON["Exhibit 99.1"].status == SECFormStatus.ACTIVE
    assert "Exhibit 99.1" in SEC_FORM_CANON[SECFormType.FORM_8K].tracked_exhibits
    assert "Exhibit 99.1" in SEC_FORM_CANON[SECFormType.FORM_8KA].tracked_exhibits


def test_ontology_provenance_confidence_parity_without_cycle() -> None:
    """ADR-00045 Invariant #5: Exact confidence parity with ontology/validation.py without ontology->ingestion import."""
    from ontology.validation import PROVENANCE_CONFIDENCE_MAP

    sec_map = get_sec_provenance_confidence_map()
    assert set(sec_map.keys()) == {"SEC_10K_ITEM1", "SEC_10Q", "SEC_8K", "SEC_FORM_4", "SEC_EXHIBIT_21"}
    for key, conf in sec_map.items():
        assert key in PROVENANCE_CONFIDENCE_MAP
        assert PROVENANCE_CONFIDENCE_MAP[key] == pytest.approx(conf)

    # Confirm no ontology/*.py file imports ingestion (zero new package coupling).
    for py_file in (REPO_ROOT / "ontology").rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".", 1)[0] != "ingestion", f"{py_file} must not import ingestion"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".", 1)[0] != "ingestion", f"{py_file} must not import ingestion"


def test_backward_compatible_reexports_and_cold_start_import_orders() -> None:
    """Backward-compatible re-exports stay identical and cold-start imports succeed in either order."""
    from ingestion.segmentation.chunker import DEFAULT_SECTION_PATTERNS
    from retrieval import reporting

    assert reporting.SECForm8KItem is SECForm8KItem
    assert reporting.SEC_ITEM_DESCRIPTIONS is SEC_ITEM_DESCRIPTIONS
    assert DEFAULT_SECTION_PATTERNS == get_section_patterns()

    for snippet in (
        "import retrieval.reporting; import ingestion.sec.disruption_gating",
        "import ingestion.sec.disruption_gating; import retrieval.reporting",
    ):
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"Cold-start import order failed ({snippet}):\n{proc.stderr}"


def test_unified_section_and_gleaning_alignment() -> None:
    """ADR-00045 Invariant #6: Chunker segments Tier-1 8-K & 10-K 1C headers; parser delegates gleaning to canon."""
    from ingestion.sec.parser import is_high_signal_gleaning_section
    from ingestion.segmentation.chunker import FilingMetadata, FinancialSectionSegmenter

    high_signal_cases = (
        "Item 1 - Business",
        "Item 1",
        "Item 1A - Risk Factors",
        "Item 1C - Cybersecurity",
        "Item 1.01 - Entry into Material Agreement",
        "Item 1.02 - Termination of Material Agreement",
        "Item 1.03 - Bankruptcy or Receivership",
        "Item 1.04 - Mine Safety Shutdowns",
        "Item 1.05 - Material Cybersecurity Incidents",
        "Item 2.01 - Completion of Acquisition or Disposition",
        "Item 2.04 - Acceleration of Financial Obligations",
        "Item 2.05 - Exit or Disposal Activities",
        "Item 2.06 - Material Impairments",
        "Item 8.01 - Other Events",
    )
    low_signal_cases = (
        "Item 1B - Unresolved Staff Comments",
        "Item 1B",
        "Item 2 - Properties",
        "Item 7 - Management Discussion and Analysis",
        "Item 2.02 - Results of Operations",
        "Item 7.01 - Regulation FD Disclosure",
        "Exhibit 21 - Subsidiaries",
        "",
        None,
    )
    for sec in high_signal_cases:
        assert is_high_signal_section(sec) is True, f"Expected high-signal: {sec!r}"
        assert is_high_signal_gleaning_section(sec) is True, f"Parser mismatch: {sec!r}"
    for low_sec in low_signal_cases:
        assert is_high_signal_section(low_sec) is False, f"Expected non-high-signal: {low_sec!r}"
        assert is_high_signal_gleaning_section(low_sec) is False, f"Parser mismatch: {low_sec!r}"

    segmenter = FinancialSectionSegmenter()
    sample_8k = (
        "ITEM 1.05. MATERIAL CYBERSECURITY INCIDENTS\n"
        "On March 12, 2025, an operational ransomware event halted production across primary lines "
        "serving multiple S&P 500 manufacturing customers for fourteen consecutive days.\n\n"
        "ITEM 2.05. COSTS ASSOCIATED WITH EXIT OR DISPOSAL ACTIVITIES\n"
        "The Board approved permanent closure of the regional assembly plant and related tooling disposal.\n"
    )
    meta = FilingMetadata(cik="320193", ticker="AAPL", company_name="Apple Inc.", form_type="Form 8-K")
    assert meta.form_type == "8-K"
    chunks = segmenter.segment_filing(sample_8k, meta)
    section_names = {c.section_name for c in chunks}
    assert "Item 1.05 - Material Cybersecurity Incidents" in section_names
    assert "Item 2.05 - Exit or Disposal Activities" in section_names


def test_normalize_form_type_and_harvester_default_forms(tmp_path: Path) -> None:
    """Form aliases normalize deterministically and default harvester omits opt-in Form 4."""
    from ingestion.sec.harvester import HistoricalSECHarvester
    from universe.sp500 import SP500Constituent

    assert normalize_form_type("SEC Form 10-K") == "10-K"
    assert normalize_form_type("form4") == "4"
    assert normalize_form_type("10-12B") == "10"
    assert normalize_form_type("DEF14A") == "DEF 14A"
    assert normalize_form_type("13F") == "13F-HR"
    assert normalize_form_type("8k/a") == "8-K/A"
    assert normalize_form_type(None, default="10-K") == "10-K"

    mock_client = MagicMock()
    mock_client.fetch_latest_10k.return_value = None
    mock_client.fetch_10k_filings.return_value = []
    mock_client.fetch_10q_filings.return_value = []
    mock_client.fetch_recent_8k.return_value = []
    mock_client.fetch_form4_transactions.return_value = []

    harvester = HistoricalSECHarvester(client=mock_client, output_dir=tmp_path, rate_limit_delay_sec=0.0)
    constituent = SP500Constituent(
        ticker="AAPL",
        cik="0000320193",
        company_name="Apple Inc.",
        gics_sector="Information Technology",
        gics_sub_industry="Technology Hardware",
    )

    harvester.harvest_company_year(constituent, year=2024)
    mock_client.fetch_latest_10k.assert_called_once()
    mock_client.fetch_10k_filings.assert_called_once()
    assert mock_client.fetch_10q_filings.call_count == 2  # 10-Q + 10-Q/A
    assert mock_client.fetch_recent_8k.call_count == 2  # 8-K + 8-K/A
    mock_client.fetch_form4_transactions.assert_not_called()

    # Explicit opt-in still invokes Form 4 fetcher cleanly.
    harvester.harvest_company_year(constituent, year=2024, forms=["form4"])
    mock_client.fetch_form4_transactions.assert_called_once()
