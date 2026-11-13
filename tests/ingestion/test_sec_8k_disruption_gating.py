"""Unit tests for SEC Form 8-K Boilerplate Sanitization & Disruption Gating (ADR-00014)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestion.sec.disruption_gating import (
    SanitizedDisruptionAssessment,
    assess_sec_8k_disruption,
    get_sec_8k_item_tier,
    sanitize_sec_8k_text,
    select_primary_sec_8k_item,
)
from retrieval.reporting import OperationalDisruptionCategory, SECForm8KItem

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "filings"


def test_sanitize_strips_cover_safe_harbor_exhibits_and_signatures():
    """Strips EDGAR cover headers, PSLRA Safe Harbor risk lists, signatures, and Item 9.01."""
    raw = (
        "Check the appropriate box below if the Form 8-K filing is intended to simultaneously "
        "satisfy the filing obligation of the registrant. Commission File Number 001-12345. "
        "Emerging growth company [ ].\n\n"
        "Item 2.02 Results of Operations and Financial Condition\n"
        "On November 4, 2025, Example Semiconductor Corp. reported quarterly revenue of $1.8 billion "
        "and strong gross margins across automotive microcontrollers.\n\n"
        "Forward-Looking Statements: This report contains forward-looking statements within the "
        "meaning of the Private Securities Litigation Reform Act of 1995. Actual results may differ "
        "materially due to risks including fires, explosions, earthquakes, floods, cyberattacks, "
        "ransomware, power outages, port closures, and equipment breakdowns.\n\n"
        "Item 9.01 Financial Statements and Exhibits\n"
        "Exhibit No.    Description\n"
        "99.1           Earnings Press Release covering cyber and termination risk factors\n\n"
        "SIGNATURES\n"
        "Pursuant to the requirements of the Securities Exchange Act of 1934, the registrant has "
        "duly caused this report to be signed on its behalf by the undersigned, discussing fire risks."
    )

    cleaned = sanitize_sec_8k_text(raw)

    assert "Commission File Number" not in cleaned
    assert "Private Securities Litigation Reform Act" not in cleaned
    assert "cyberattacks" not in cleaned
    assert "Exhibit No." not in cleaned
    assert "Securities Exchange Act of 1934" not in cleaned
    assert "reported quarterly revenue of $1.8 billion" in cleaned


def test_sanitize_handles_single_newline_and_monolithic_inline_safe_harbor():
    """Preserves substantive operational prose when Safe Harbor trails inside a monolithic paragraph."""
    monolithic = (
        "Check the appropriate box below. Commission File Number 001-99999.\n"
        "Item 8.01 Other Events.\n"
        "On May 12, 2024, FabLine suffered an industrial fire at its Austin wafer facility. "
        "Forward-Looking Statements: Actual results may differ materially due to ransomware, "
        "earthquakes, or maritime port congestion.\n"
        "Pursuant to the requirements of the Securities Exchange Act of 1934, signed by CFO."
    )

    cleaned = sanitize_sec_8k_text(monolithic)

    assert "industrial fire at its Austin wafer facility" in cleaned
    assert "ransomware" not in cleaned
    assert "earthquakes" not in cleaned
    assert "Commission File Number" not in cleaned
    assert "signed by CFO" not in cleaned


def test_select_primary_sec_8k_item_respects_priority_tiers():
    """Ranks Tier 1 acute items > Tier 2 events > Tier 3 financial > Tier 4 procedural."""
    assert select_primary_sec_8k_item(["Item 9.01", "Item 1.05"]) == SECForm8KItem.ITEM_1_05.value
    assert select_primary_sec_8k_item(["Item 8.01", "Item 1.02"]) == SECForm8KItem.ITEM_1_02.value
    assert select_primary_sec_8k_item(["Item 9.01 Financial Statements", "Item 5.02"]) == SECForm8KItem.ITEM_5_02.value
    assert select_primary_sec_8k_item(["Item 9.01", "Item 2.02 Results of Operations"]) == SECForm8KItem.ITEM_2_02.value
    assert select_primary_sec_8k_item([]) == SECForm8KItem.ITEM_8_01.value

    assert get_sec_8k_item_tier("Item 1.05") == 1
    assert get_sec_8k_item_tier("Item 8.01") == 2
    assert get_sec_8k_item_tier("Item 2.02") == 3
    assert get_sec_8k_item_tier("Item 5.02") == 4
    assert get_sec_8k_item_tier("Item 9.01") == 4


def test_assess_gates_safe_harbor_earnings_release_to_zero_shock_negative_control():
    """Routine Item 2.02 / Item 5.02 with PSLRA risk enumeration gates to 0.0% / 0.0 days."""
    assessment = assess_sec_8k_disruption(
        items=["Item 9.01 Exhibits", "Item 2.02 Results of Operations"],
        headline="Quarterly Financial Results for Q3 Fiscal 2025",
        content=(
            "Revenue grew 11% year over year on strong cloud demand.\n\n"
            "Cautionary Note Regarding Forward-Looking Statements: Actual results may differ materially "
            "as a result of facility fires, floods, ransomware attacks, power grid failures, or supply halt."
        ),
        metadata={"primary_exchange": "NASDAQ"},
    )

    assert isinstance(assessment, SanitizedDisruptionAssessment)
    assert assessment.primary_sec_item == SECForm8KItem.ITEM_2_02.value
    assert assessment.item_priority_tier == 3
    assert assessment.is_material_operational_disruption is False
    assert assessment.shock_magnitude_pct == 0.0
    assert assessment.shock_duration_days == 0.0
    assert assessment.stripped_paragraph_count >= 1


def test_assess_item_1_02_distinguishes_supply_termination_from_credit_or_officer_notice():
    """Item 1.02 triggers supply termination with supplier context, otherwise gates cleanly."""
    supply_term = assess_sec_8k_disruption(
        items=["Item 1.02 Termination of a Material Definitive Agreement"],
        headline="Termination of Master Wafer Foundry Supply Agreement",
        content="The company terminated its sole-source silicon wafer manufacturing contract with SupplierX.",
    )
    assert supply_term.is_material_operational_disruption is True
    assert supply_term.disruption_category == OperationalDisruptionCategory.PRIMARY_SUPPLY_CONTRACT_TERMINATED
    assert supply_term.shock_magnitude_pct == 100.0

    non_supply_term = assess_sec_8k_disruption(
        items=["Item 1.02 Termination of a Material Definitive Agreement"],
        headline="Termination of Unutilized Revolving Credit Agreement",
        content="The company voluntarily retired its $250M credit facility prior to maturity with zero penalties.",
    )
    assert non_supply_term.is_material_operational_disruption is False
    assert non_supply_term.shock_magnitude_pct == 0.0
    assert non_supply_term.shock_duration_days == 0.0


def test_assess_item_1_05_anchors_cyber_and_tier3_with_real_fire_triggers():
    """Tier 1 Item 1.05 anchors cyber halt; Tier 3 Item 2.02 with genuine fab fire still triggers."""
    cyber = assess_sec_8k_disruption(
        items=["Item 9.01", "Item 1.05 Material Cybersecurity Incidents"],
        headline="Material Cybersecurity Disclosure",
        content="The company detected unauthorized access to enterprise ERP infrastructure causing fulfillment delays.",
    )
    assert cyber.primary_sec_item == SECForm8KItem.ITEM_1_05.value
    assert cyber.is_material_operational_disruption is True
    assert cyber.disruption_category == OperationalDisruptionCategory.RANSOMWARE_CYBER_INFRASTRUCTURE_HALT

    genuine_fire_in_2_02 = assess_sec_8k_disruption(
        items=["Item 2.02"],
        headline="Preliminary Q4 Update Following Austin Fab Fire",
        content="Production at the primary semiconductor fab was halted following an electrical fire in Cleanroom B.",
    )
    assert genuine_fire_in_2_02.is_material_operational_disruption is True
    assert genuine_fire_in_2_02.disruption_category == OperationalDisruptionCategory.FACILITY_FIRE_OR_EXPLOSION


@pytest.mark.parametrize(
    "fixture_name,expected_cat,expected_mag,expected_dur",
    [
        (
            "crus_8k_disruption.json",
            OperationalDisruptionCategory.FACILITY_FIRE_OR_EXPLOSION,
            100.0,
            60.0,
        ),
        (
            "aapl_8k_disruption.json",
            OperationalDisruptionCategory.EQUIPMENT_BREAKDOWN_FAB_HALT,
            100.0,
            90.0,
        ),
    ],
)
def test_canonical_fixtures_strict_backward_compatibility(fixture_name, expected_cat, expected_mag, expected_dur):
    """Canonical test fixtures retain identical operational classifications under ADR-00014."""
    payload = json.loads((FIXTURES_DIR / fixture_name).read_text(encoding="utf-8"))
    assessment = assess_sec_8k_disruption(
        items=payload["items"],
        headline=payload["headline"],
        content=payload["content"],
        metadata=payload["metadata"],
    )

    assert assessment.primary_sec_item == SECForm8KItem.ITEM_8_01.value
    assert assessment.is_material_operational_disruption is True
    assert assessment.disruption_category == expected_cat
    assert assessment.shock_magnitude_pct == expected_mag
    assert assessment.shock_duration_days == expected_dur

