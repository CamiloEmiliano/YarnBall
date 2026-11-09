"""Gate 1: Automated integrity test for Linguistic CheckList fixtures.

Verifies:
1. 100% of fixture entries parse cleanly into the LinguisticTestCase Pydantic model.
2. All 10 Ribeiro et al. capabilities are covered.
3. All corporate CIKs are 10-digit zero-padded strings.
4. All relational predicates exist in the closed ontology schema.
5. All INV and DIR test cases contain valid non-empty perturbation strings.
6. Unique test IDs with no duplicate keys.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from tests.fixtures.linguistic.schema import (
    CapabilityEnum,
    LinguisticTestCase,
    TestTypeEnum,
)

FIXTURES_PATH = Path(__file__).parent / "checklist_cases.json"


def test_fixture_file_exists():
    """Verify that checklist_cases.json exists and is readable."""
    assert FIXTURES_PATH.is_file(), f"Fixture file not found at {FIXTURES_PATH}"


def test_all_cases_validate_against_pydantic_schema():
    """Verify that every test case cleanly parses into LinguisticTestCase."""
    with open(FIXTURES_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    assert isinstance(raw_data, list), "checklist_cases.json must contain a list of test cases"
    assert len(raw_data) >= 10, f"Expected at least 10 test cases, got {len(raw_data)}"

    test_ids = set()
    validated_cases: list[LinguisticTestCase] = []

    for item in raw_data:
        case = LinguisticTestCase.model_validate(item)
        validated_cases.append(case)

        # ID uniqueness
        assert case.id not in test_ids, f"Duplicate test ID '{case.id}'"
        test_ids.add(case.id)

        # Non-empty input text
        assert case.input_text.strip(), f"Empty input_text in test '{case.id}'"

        # Perturbation requirement for INV and DIR
        if case.test_type in (TestTypeEnum.INV, TestTypeEnum.DIR):
            assert case.perturbed_text and case.perturbed_text.strip(), (
                f"Test '{case.id}' of type '{case.test_type}' must have non-empty perturbed_text"
            )

        # CIK zero-padding and format
        for entity in case.expected_entities:
            if entity.cik:
                assert len(entity.cik) == 10 and entity.cik.isdigit(), (
                    f"Entity '{entity.name}' has invalid CIK '{entity.cik}' in case '{case.id}'"
                )

    # All 10 capabilities must be represented
    covered_capabilities = {c.capability for c in validated_cases}
    all_capabilities = set(CapabilityEnum)
    missing = all_capabilities - covered_capabilities
    assert not missing, f"Missing test cases for capabilities: {missing}"


def test_dual_exchange_universe_consistency():
    """Verify that equity entities in the fixtures specify valid primary exchanges."""
    with open(FIXTURES_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    for item in raw_data:
        case = LinguisticTestCase.model_validate(item)
        for entity in case.expected_entities:
            if entity.ticker:
                assert entity.listing_exchange in {"NASDAQ", "NYSE", "AMEX", "CBOE"}, (
                    f"Entity '{entity.name}' ({entity.ticker}) in case '{case.id}' has invalid exchange '{entity.listing_exchange}'"
                )

