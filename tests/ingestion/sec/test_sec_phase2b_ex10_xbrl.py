"""Unit & integration tests for Phase 2b Operational Exhibit 10 & Offline XBRL Priors (ADR-00048)."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from contagion.parameters import (
    MAX_PLAUSIBLE_XBRL_DIO_DAYS,
    PHYSICAL_INVENTORY_GICS_SECTORS,
    PROVENANCE_CONFIDENCE_RANK,
    ParameterProvenance,
    is_physical_inventory_gics_sector,
    resolve_edge_cascade_parameters,
)
from ingestion.sec.client import (
    EdgarClient,
    classify_exhibit_10_attachment,
    extract_operational_contract_clauses,
)
from ingestion.sec.disruption_gating import DEFAULT_MAX_FILING_CHARS
from ingestion.sec.harvester import EdgarWorker, HistoricalSECHarvester
from ingestion.sec.xbrl import (
    METRIC_ASC280_CONCENTRATION_SHARE,
    METRIC_XBRL_DIO_DAYS,
    XBRLPriorStore,
    extract_asc280_concentration_from_companyfacts,
    extract_xbrl_dio_priors_from_companyfacts,
    fetch_companyfacts_json,
    is_physical_inventory_sector,
    lookup_pit_xbrl_dio,
)
from universe.sp500 import SP500Constituent

REPO_ROOT = Path(__file__).resolve().parents[3]


class _FakeAttachment:
    def __init__(
        self,
        document_type: str,
        description: str,
        document: str = "ex10.htm",
        body: str = "",
        *,
        forbid_text_fetch: bool = False,
    ) -> None:
        self.document_type = document_type
        self.description = description
        self.document = document
        self._body = body
        self._forbid_text_fetch = forbid_text_fetch
        self.text_calls = 0

    def text(self) -> str:
        self.text_calls += 1
        if self._forbid_text_fetch:
            raise AssertionError(f"Rejected attachment {self.document_type} ({self.description}) must not fetch text")
        return self._body


def _build_sample_companyfacts() -> Dict[str, Any]:
    """Deterministic multi-year + quarterly + 10-K/A + outlier EDGAR companyfacts fixture."""
    return {
        "cik": 320193,
        "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": {
                "InventoryNet": {
                    "units": {
                        "USD": [
                            # FY2023 10-K (DIO = 365 * 6e9 / 219e9 = 10.0 days)
                            {
                                "end": "2023-09-30",
                                "val": 6_000_000_000.0,
                                "accn": "0000320193-23-000106",
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2023-11-03",
                            },
                            # Q1 2024 10-Q (DIO = 365 * 7.3e9 / (4 * 66.6125e9) = 10.0 days? Let's make 15.0 days:
                            # 4 * 45.625e9 = 182.5e9; 365 * 7.5e9 / 182.5e9 = 15.0 days)
                            {
                                "end": "2023-12-30",
                                "val": 7_500_000_000.0,
                                "accn": "0000320193-24-000006",
                                "fy": 2024,
                                "fp": "Q1",
                                "form": "10-Q",
                                "filed": "2024-02-02",
                            },
                            # FY2024 10-K reporting BOTH FY2024 (20.0 days) AND comparative FY2023 (12.0 days)
                            {
                                "end": "2024-09-28",
                                "val": 12_000_000_000.0,
                                "accn": "0000320193-24-000123",
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-11-01",
                            },
                            {
                                "end": "2023-09-30",
                                "val": 7_200_000_000.0,
                                "accn": "0000320193-24-000123",
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-11-01",
                            },
                            # FY2024 10-K/A restatement filed 2025-01-15 (25.0 days)
                            {
                                "end": "2024-09-28",
                                "val": 15_000_000_000.0,
                                "accn": "0000320193-25-000002",
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K/A",
                                "filed": "2025-01-15",
                            },
                            # Implausible outlier DIO > 365 days (must be skipped)
                            {
                                "end": "2022-09-24",
                                "val": 500_000_000_000.0,
                                "accn": "0000320193-22-000108",
                                "fy": 2022,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2022-10-28",
                            },
                        ]
                    }
                },
                "CostOfGoodsAndServicesSold": {
                    "units": {
                        "USD": [
                            # FY2023 10-K annual flow (365 days)
                            {
                                "start": "2022-10-01",
                                "end": "2023-09-30",
                                "val": 219_000_000_000.0,
                                "accn": "0000320193-23-000106",
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2023-11-03",
                            },
                            # Q1 2024 10-Q quarterly flow (91 days) -> annualized = 182.5e9 -> DIO = 15.0
                            {
                                "start": "2023-10-01",
                                "end": "2023-12-30",
                                "val": 45_625_000_000.0,
                                "accn": "0000320193-24-000006",
                                "fy": 2024,
                                "fp": "Q1",
                                "form": "10-Q",
                                "filed": "2024-02-02",
                            },
                            # 6-month YTD flow (182 days) -> must be ignored
                            {
                                "start": "2023-07-01",
                                "end": "2023-12-30",
                                "val": 99_000_000_000.0,
                                "accn": "0000320193-24-000006",
                                "fy": 2024,
                                "fp": "Q1",
                                "form": "10-Q",
                                "filed": "2024-02-02",
                            },
                            # FY2024 10-K: both Q4 standalone (90d) AND full year (364d) -> ANNUAL must win (DIO=20.0)
                            {
                                "start": "2024-06-30",
                                "end": "2024-09-28",
                                "val": 100_000_000_000.0,
                                "accn": "0000320193-24-000123",
                                "fy": 2024,
                                "fp": "Q4",
                                "form": "10-K",
                                "filed": "2024-11-01",
                            },
                            {
                                "start": "2023-10-01",
                                "end": "2024-09-28",
                                "val": 219_000_000_000.0,
                                "accn": "0000320193-24-000123",
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-11-01",
                            },
                            # Comparative FY2023 inside FY2024 10-K
                            {
                                "start": "2022-10-01",
                                "end": "2023-09-30",
                                "val": 219_000_000_000.0,
                                "accn": "0000320193-24-000123",
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-11-01",
                            },
                            # FY2024 10-K/A restatement (DIO = 25.0)
                            {
                                "start": "2023-10-01",
                                "end": "2024-09-28",
                                "val": 219_000_000_000.0,
                                "accn": "0000320193-25-000002",
                                "fy": 2024,
                                "fp": "FY",
                                "form": "10-K/A",
                                "filed": "2025-01-15",
                            },
                            # FY2022 outlier denominator (DIO = 365 * 500e9 / 100e9 = 1825 > 365)
                            {
                                "start": "2021-09-26",
                                "end": "2022-09-24",
                                "val": 100_000_000_000.0,
                                "accn": "0000320193-22-000108",
                                "fy": 2022,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2022-10-28",
                            },
                        ]
                    }
                },
                "ConcentrationRiskPercentage1": {
                    "units": {
                        "pure": [
                            {
                                "end": "2023-09-30",
                                "val": 0.14,
                                "accn": "0000320193-23-000106",
                                "form": "10-K",
                                "filed": "2023-11-03",
                                "frame": "CY2023",
                            },
                            {
                                "end": "2024-09-28",
                                "val": 0.22,
                                "accn": "0000320193-24-000123",
                                "form": "10-K",
                                "filed": "2024-11-01",
                                "frame": "CY2024",
                            },
                            # Below ASC 280 10% threshold (must skip)
                            {
                                "end": "2024-09-28",
                                "val": 0.05,
                                "accn": "0000320193-24-000123",
                                "form": "10-K",
                                "filed": "2024-11-01",
                            },
                            # Invalid > 1.0 (must skip)
                            {
                                "end": "2024-09-28",
                                "val": 1.45,
                                "accn": "0000320193-24-000123",
                                "form": "10-K",
                                "filed": "2024-11-01",
                            },
                        ]
                    }
                },
            }
        },
    }


def test_exhibit_10_classifier_accepts_operational_and_rejects_comp_and_debt() -> None:
    """ADR-00048 Invariant #1: Pre-fetch classifier keeps operational contracts and rejects comp/debt noise."""
    accepted = [
        ("EX-10.1", "Wafer Supply and Foundry Capacity Reservation Agreement", "ex10-1.htm", "SUPPLY_MANUFACTURING"),
        ("EX-10.4", "Commercial CDMO Collaboration and License Agreement", "cdmo.htm", "PHARMA_LICENSING_CDMO"),
        ("EX-10.12", "Long-Term LNG Off-Take and Pipeline Transportation Agreement", "lng.htm", "ENERGY_PIPELINE_PPA"),
        ("EX-10.2", "Solar Power Purchase Agreement (PPA)", "ppa.htm", "ENERGY_PIPELINE_PPA"),
        ("EX-10.7", "Pharmacy Benefit Management (PBM) Services Agreement", "pbm.htm", "COMMERCIAL_DISTRIBUTION_PBM"),
        ("EX-10.9", "Cloud Services and Data Center Colocation Agreement", "cloud.htm", "CLOUD_INFRA_LOGISTICS"),
        ("EX-10.15", "Spin-Off Transition Services Agreement (TSA)", "tsa.htm", "TSA_JOINT_VENTURE"),
    ]
    for doc_type, desc, fname, expected_cat in accepted:
        assert classify_exhibit_10_attachment(doc_type, desc, fname) == expected_cat

    rejected = [
        ("EX-10.1", "Executive Employment Agreement with Chief Executive Officer", "emp.htm"),
        ("EX-10.2", "Form of Restricted Stock Unit (RSU) Award Agreement", "rsu.htm"),
        ("EX-10.3", "Non-Employee Director Compensation Plan", "dircomp.htm"),
        ("EX-10.4", "Form of Indemnification Agreement for Officers and Directors", "indem.htm"),
        ("EX-10.5", "Second Amended and Restated Credit Agreement", "credit.htm"),
        ("EX-10.6", "Senior Unsecured Loan Agreement and Promissory Note", "loan.htm"),
        ("EX-10.7", "2024 Omnibus Equity Incentive Plan and Stock Option Agreement", "eip.htm"),
        ("EX-10.8", "Share Purchase Agreement and Registration Rights Agreement", "spa.htm"),
        ("EX-10.9", "Senior Note Purchase Agreement", "npa.htm"),
        ("EX-101.INS", "XBRL Instance Document Supply", "aapl.xml"),
        ("EX-104", "Cover Page Interactive Data File", "aapl.xml"),
        ("EX-99.1", "Press Release on Wafer Supply Agreement", "pr.htm"),
        ("EX-21.1", "Subsidiaries of the Registrant", "ex21.htm"),
        ("EX-10.1", "Wafer Supply Agreement", "diagram.pdf"),
    ]
    for doc_type, desc, fname in rejected:
        assert classify_exhibit_10_attachment(doc_type, desc, fname) is None, f"Should reject: {doc_type} | {desc}"


def test_exhibit_10_clause_windowing_and_pre_fetch_gate() -> None:
    """ADR-00048 §4.1: Preserves recitals + operational clauses under 50k chars and skips rejected text fetches."""
    contract_text = "\n\n".join(
        [
            (
                "WAFER SUPPLY AGREEMENT\n"
                "This Wafer Supply Agreement is entered into between Taiwan Semiconductor Manufacturing "
                "Company Limited and NVIDIA Corporation effective January 1, 2024. "
                "WHEREAS Supplier operates Fab 18 producing advanced 3nm logic wafers."
            ),
            (
                "SECTION 1. DEFINITIONS.\n"
                "Affiliate means any corporation or entity controlling or controlled by a party. "
                "Business Day means any day other than Saturday or Sunday in New York."
            ),
            (
                "SECTION 2. EXCLUSIVE SUPPLY, MINIMUM VOLUME AND SAFETY STOCK.\n"
                "Supplier shall serve as sole-source supplier for 3nm wafers under a minimum purchase "
                "commitment of 50,000 wafers per quarter and shall maintain 30 days of safety stock "
                "inventory buffer with a 45-day delivery lead time."
            ),
            (
                "SECTION 12. NOTICES.\n"
                "All formal legal notices hereunder shall be sent by registered mail to corporate counsel "
                "in Delaware with duplicate courier delivery."
            ),
            (
                "SECTION 13. MISCELLANEOUS.\n"
                "This Agreement may be executed in counterparts and governed by Delaware law without "
                "regard to conflict of laws principles."
            ),
            "IN WITNESS WHEREOF, the parties have executed this Agreement by their duly authorized officers.",
        ]
    )

    windowed = extract_operational_contract_clauses(contract_text)
    assert "Taiwan Semiconductor Manufacturing" in windowed
    assert "NVIDIA Corporation" in windowed
    assert "sole-source supplier" in windowed
    assert "30 days of safety stock" in windowed
    assert "SECTION 1. DEFINITIONS." not in windowed
    assert "SECTION 12. NOTICES." not in windowed
    assert "SECTION 13. MISCELLANEOUS." not in windowed
    assert "IN WITNESS WHEREOF" not in windowed

    # Hard cap <= DEFAULT_MAX_FILING_CHARS (50_000) on oversized contracts
    huge_contract = contract_text + "\n\n" + ("\n\n".join(f"SECTION {i}. CAPACITY SUPPLY CLAUSE. " + ("supply volume capacity " * 80) for i in range(20, 90)))
    assert len(huge_contract) > DEFAULT_MAX_FILING_CHARS
    bounded = extract_operational_contract_clauses(huge_contract)
    assert len(bounded) <= DEFAULT_MAX_FILING_CHARS

    # Pre-fetch gate: rejected EX-10.1 must never even call .text()
    rejected_comp = _FakeAttachment(
        "EX-10.1",
        "Executive Employment and Severance Agreement",
        forbid_text_fetch=True,
    )
    accepted_supply = _FakeAttachment(
        "EX-10.2",
        "Master Wafer Supply and Foundry Agreement",
        body=contract_text,
    )
    fake_filing = MagicMock()
    fake_filing.attachments = [rejected_comp, accepted_supply]

    client = EdgarClient()
    contracts = client.extract_exhibit_10_contracts(fake_filing)
    assert rejected_comp.text_calls == 0
    assert accepted_supply.text_calls == 1
    assert len(contracts) == 1
    assert contracts[0]["exhibit_key"] == "EX-10.2"
    assert contracts[0]["matched_category"] == "SUPPLY_MANUFACTURING"
    assert contracts[0]["filtered_char_count"] < contracts[0]["raw_char_count"]


def test_xbrl_dio_and_asc280_pit_extraction_and_sector_guards(tmp_path: Path) -> None:
    """ADR-00048 §4.2: Exact DIO math, outlier skipping, ASC 280, O(log K) PiT isolation & physical-sector guard."""
    facts = _build_sample_companyfacts()

    # Non-physical sectors return empty immediately
    for excluded_sector in ("Financials", "Real Estate", "Utilities", "Communication Services"):
        assert is_physical_inventory_sector(excluded_sector) is False
        assert (
            extract_xbrl_dio_priors_from_companyfacts(
                facts,
                ticker="JPM",
                gics_sector=excluded_sector,
            )
            == []
        )

    for physical_sector in PHYSICAL_INVENTORY_GICS_SECTORS:
        assert is_physical_inventory_sector(physical_sector) is True

    dio_recs = extract_xbrl_dio_priors_from_companyfacts(
        facts,
        ticker="AAPL",
        gics_sector="Information Technology",
    )
    # Outlier FY2022 (>365d) skipped; 5 valid observations remain:
    # FY2023 10-K, Q1 2024 10-Q, FY2024 10-K (FY24 + comparative FY23), FY2024 10-K/A
    assert len(dio_recs) == 5
    assert all(0.0 < r.dio_days <= MAX_PLAUSIBLE_XBRL_DIO_DAYS for r in dio_recs)

    conc_recs = extract_asc280_concentration_from_companyfacts(
        facts,
        ticker="AAPL",
        gics_sector="Information Technology",
    )
    assert len(conc_recs) == 2
    assert [r.concentration_share for r in conc_recs] == [pytest.approx(0.14), pytest.approx(0.22)]

    store = XBRLPriorStore(dio_records=dio_recs, concentration_records=conc_recs)

    # Strict Point-in-Time queries (no lookahead!)
    assert store.lookup_pit_dio("AAPL", "2023-10-15") is None

    rec_fy23 = store.lookup_pit_dio("AAPL", "2023-11-10")
    assert rec_fy23 is not None
    assert rec_fy23.dio_days == pytest.approx(10.0)
    assert rec_fy23.period_basis == "ANNUAL"
    assert rec_fy23.period_end == "2023-09-30"

    # Q1 2024 10-Q filed 2024-02-02 refreshes DIO to 15.0 days via 4x annualized quarterly COGS
    rec_q1 = store.lookup_pit_dio("0000320193", "2024-03-01")
    assert rec_q1 is not None
    assert rec_q1.dio_days == pytest.approx(15.0)
    assert rec_q1.period_basis == "QUARTERLY_ANNUALIZED"
    assert rec_q1.period_end == "2023-12-30"

    # FY2024 10-K filed 2024-11-01 selects FY2024 ANNUAL (20.0d) over both Q4 standalone and comparative FY2023
    rec_fy24 = store.lookup_pit_dio("AAPL", "2024-12-01")
    assert rec_fy24 is not None
    assert rec_fy24.dio_days == pytest.approx(20.0)
    assert rec_fy24.period_basis == "ANNUAL"
    assert rec_fy24.period_end == "2024-09-28"
    assert rec_fy24.form_type == "10-K"

    # FY2024 10-K/A restatement filed 2025-01-15 becomes visible only on/after its actionable valid_from
    rec_fy24_amended = store.lookup_pit_dio("AAPL", "2025-01-20")
    assert rec_fy24_amended is not None
    assert rec_fy24_amended.dio_days == pytest.approx(25.0)
    assert rec_fy24_amended.form_type == "10-K/A"

    # Sector guard at lookup time
    assert store.lookup_pit_dio("AAPL", "2025-01-20", gics_sector="Financials") is None

    # ASC 280 PiT concentration lookup
    conc_early = store.lookup_pit_max_concentration("AAPL", "2024-06-01")
    assert conc_early is not None and conc_early.concentration_share == pytest.approx(0.14)
    conc_late = store.lookup_pit_max_concentration("AAPL", "2024-12-01")
    assert conc_late is not None and conc_late.concentration_share == pytest.approx(0.22)

    # Parquet round-trip + top-level lookup_pit_xbrl_dio convenience helper
    pq_path = tmp_path / "sec_xbrl_priors.parquet"
    store.save_parquet(pq_path)
    reloaded = lookup_pit_xbrl_dio("AAPL", "2024-12-01", gics_sector="Information Technology", parquet_path=pq_path)
    assert reloaded is not None
    assert reloaded.dio_days == pytest.approx(20.0)

    # JSON round-trip + local cache loader
    json_cache = tmp_path / "CIK0000320193.json"
    json_cache.write_text(json.dumps(facts), encoding="utf-8")
    loaded_raw = fetch_companyfacts_json("320193", cache_dir=tmp_path)
    assert loaded_raw is not None and loaded_raw["cik"] == 320193

    # Postgres upsert sync (mock connection)
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    synced = store.sync_to_postgres(mock_conn)
    assert synced == 7  # 5 DIO + 2 ASC 280
    mock_cur.executemany.assert_called_once()
    mock_conn.commit.assert_called_once()


def test_xbrl_component_inventory_and_cost_of_revenue_fallback() -> None:
    """When InventoryNet is omitted, sum FinishedGoods+WIP+RawMaterials and fall back to CostOfRevenue."""
    facts: Dict[str, Any] = {
        "cik": "0001045810",
        "facts": {
            "us-gaap": {
                "InventoryFinishedGoods": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-01-28",
                                "val": 2_000_000_000.0,
                                "accn": "0001045810-24-000029",
                                "form": "10-K",
                                "filed": "2024-02-21",
                            }
                        ]
                    }
                },
                "InventoryWorkInProcess": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-01-28",
                                "val": 1_500_000_000.0,
                                "accn": "0001045810-24-000029",
                                "form": "10-K",
                                "filed": "2024-02-21",
                            }
                        ]
                    }
                },
                "InventoryRawMaterialsAndSupplies": {
                    "units": {
                        "USD": [
                            {
                                "end": "2024-01-28",
                                "val": 1_500_000_000.0,
                                "accn": "0001045810-24-000029",
                                "form": "10-K",
                                "filed": "2024-02-21",
                            }
                        ]
                    }
                },
                "CostOfRevenue": {
                    "units": {
                        "USD": [
                            {
                                "start": "2023-01-30",
                                "end": "2024-01-28",
                                "val": 18_250_000_000.0,
                                "accn": "0001045810-24-000029",
                                "form": "10-K",
                                "filed": "2024-02-21",
                            }
                        ]
                    }
                },
            }
        },
    }
    recs = extract_xbrl_dio_priors_from_companyfacts(facts, ticker="NVDA", gics_sector="Information Technology")
    assert len(recs) == 1
    assert recs[0].inventory_usd == pytest.approx(5_000_000_000.0)
    assert recs[0].dio_days == pytest.approx(100.0)
    assert recs[0].cogs_tag == "CostOfRevenue"
    assert recs[0].metric_name == METRIC_XBRL_DIO_DAYS


def test_cascade_parameter_provenance_xbrl_precedence_and_leaf_invariant() -> None:
    """ADR-00048 Invariants #2, #3 & #5: Strict edge-specific precedence, sector/domain guard & stdlib leaf."""
    assert PROVENANCE_CONFIDENCE_RANK[ParameterProvenance.XBRL_ISSUER_DIO_PROXY.value] == 2
    assert (
        PROVENANCE_CONFIDENCE_RANK[ParameterProvenance.EXTRACTED_QUANTITATIVE.value]
        > PROVENANCE_CONFIDENCE_RANK[ParameterProvenance.XBRL_ISSUER_DIO_PROXY.value]
        > PROVENANCE_CONFIDENCE_RANK[ParameterProvenance.DEFAULT_CONSERVATIVE_BOUND.value]
    )

    # 1. Edge with explicit buffer_days=45 ignores target_xbrl_dio_days=62 and stays EXTRACTED_QUANTITATIVE
    res_explicit = resolve_edge_cascade_parameters(
        {"rel_type": "SUPPLIES_TO", "buffer_days": 45.0, "nature": "wafers"},
        target_xbrl_dio_days=62.0,
        target_gics_sector="Information Technology",
    )
    assert res_explicit["buffer_days"] == pytest.approx(45.0)
    assert res_explicit["param_provenance"] == ParameterProvenance.EXTRACTED_QUANTITATIVE.value
    assert res_explicit["buffer_days_is_default"] is False

    # 2. Edge with text-extracted buffer also overrides XBRL DIO proxy
    res_text = resolve_edge_cascade_parameters(
        {"rel_type": "SUPPLIES_TO", "nature": "wafers"},
        source_text="The customer maintains 30 days of safety stock for advanced substrates.",
        target_xbrl_dio_days=62.0,
        target_gics_sector="Information Technology",
    )
    assert res_text["buffer_days"] == pytest.approx(30.0)
    assert res_text["param_provenance"] == ParameterProvenance.EXTRACTED_QUANTITATIVE.value

    # 3. Physical edge without buffer upgrades to XBRL_ISSUER_DIO_PROXY while keeping share_is_default honest
    res_proxy = resolve_edge_cascade_parameters(
        {"rel_type": "SUPPLIES_TO", "nature": "silicon_wafers"},
        target_xbrl_dio_days=62.0,
        target_gics_sector="Information Technology",
    )
    assert res_proxy["buffer_days"] == pytest.approx(62.0)
    assert res_proxy["buffer_days_is_default"] is False
    assert res_proxy["share"] == pytest.approx(1.0)
    assert res_proxy["share_is_default"] is True
    assert res_proxy["param_provenance"] == ParameterProvenance.XBRL_ISSUER_DIO_PROXY.value

    # 4. Stored previously-defaulted physical edge upgrades cleanly when XBRL DIO is supplied
    res_upgraded = resolve_edge_cascade_parameters(
        {
            "rel_type": "SUPPLIES_TO",
            "nature": "jet_turbines",
            "share": 1.0,
            "buffer_days": 0.0,
            "share_is_default": True,
            "buffer_days_is_default": True,
            "param_provenance": ParameterProvenance.DEFAULT_CONSERVATIVE_BOUND.value,
        },
        target_xbrl_dio_days=48.5,
        target_gics_sector="Industrials",
    )
    assert res_upgraded["buffer_days"] == pytest.approx(48.5)
    assert res_upgraded["buffer_days_is_default"] is False
    assert res_upgraded["param_provenance"] == ParameterProvenance.XBRL_ISSUER_DIO_PROXY.value

    # 5. Non-physical GICS sectors reject XBRL DIO proxy
    for bad_sector in ("Financials", "Real Estate", "Utilities", "Communication Services"):
        assert is_physical_inventory_gics_sector(bad_sector) is False
        res_bad_sec = resolve_edge_cascade_parameters(
            {"rel_type": "SUPPLIES_TO", "nature": "components"},
            target_xbrl_dio_days=62.0,
            target_gics_sector=bad_sector,
        )
        assert res_bad_sec["buffer_days"] == pytest.approx(0.0)
        assert res_bad_sec["buffer_days_is_default"] is True
        assert res_bad_sec["param_provenance"] == ParameterProvenance.DEFAULT_CONSERVATIVE_BOUND.value

    # 6. DIGITAL_INFRASTRUCTURE & COMMERCIAL_ENTERPRISE edges reject XBRL DIO proxy even on IT sector
    for non_physical_domain in ("DIGITAL_INFRASTRUCTURE", "COMMERCIAL_ENTERPRISE"):
        res_bad_dom = resolve_edge_cascade_parameters(
            {"rel_type": "SUPPLIES_TO", "dependency_domain": non_physical_domain},
            target_xbrl_dio_days=62.0,
            target_gics_sector="Information Technology",
        )
        assert res_bad_dom["buffer_days"] == pytest.approx(0.0)
        assert res_bad_dom["buffer_days_is_default"] is True
        assert res_bad_dom["param_provenance"] == ParameterProvenance.DEFAULT_CONSERVATIVE_BOUND.value

    # 7. Implausible DIO bounds rejected
    for bad_dio in (0.0, -15.0, 366.0, float("nan"), float("inf")):
        res_bad_dio = resolve_edge_cascade_parameters(
            {"rel_type": "SUPPLIES_TO", "nature": "wafers"},
            target_xbrl_dio_days=bad_dio,
            target_gics_sector="Information Technology",
        )
        assert res_bad_dio["buffer_days"] == pytest.approx(0.0)
        assert res_bad_dio["param_provenance"] == ParameterProvenance.DEFAULT_CONSERVATIVE_BOUND.value

    # 8. contagion/parameters.py stays a pure stdlib zero-dependency leaf
    params_path = REPO_ROOT / "contagion" / "parameters.py"
    tree = ast.parse(params_path.read_text(encoding="utf-8"), filename=str(params_path))
    internal_pkgs = {
        "contagion", "curation", "embedding", "explorer", "ingestion", "knowledge_graph",
        "ontology", "retrieval", "sft_pipeline", "streaming", "synthesis", "tools", "universe",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".", 1)[0] not in internal_pkgs
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".", 1)[0] not in internal_pkgs


def test_harvester_and_worker_opt_in_exhibit_10_staging(tmp_path: Path) -> None:
    """ADR-00048 §4.1 & Invariant #6: Exhibit 10 harvesting is opt-in and persists DOMAIN1_OPERATIONAL chunks."""
    contract_entry = {
        "exhibit_key": "EX-10.2",
        "description": "Wafer Supply Agreement",
        "matched_category": "SUPPLY_MANUFACTURING",
        "text": (
            "WAFER SUPPLY AGREEMENT between Taiwan Semiconductor Manufacturing and NVIDIA Corporation. "
            "Supplier is sole-source supplier of 3nm wafers with 30 days of safety stock."
        ),
        "raw_char_count": 400,
        "filtered_char_count": 170,
    }

    mock_client = MagicMock()
    mock_filing = MagicMock()
    mock_client.fetch_latest_10k.return_value = mock_filing
    mock_client.fetch_10k_filings.return_value = []
    mock_client.fetch_10q_filings.return_value = []
    mock_client.fetch_recent_8k.return_value = []

    def _fake_extract_10k(filing: Any, *, include_exhibit_10: bool = False) -> Dict[str, Any]:
        del filing
        payload: Dict[str, Any] = {
            "accession_number": "0001045810-24-000029",
            "filing_date": "2024-02-21",
            "acceptance_datetime": "2024-02-21T16:30:00-05:00",
            "report_date": "2024-01-28",
            "form": "10-K",
            "item_1_business": "NVIDIA Corporation designs GPUs.",
            "item_1a_risk_factors": "",
            "item_1c_cybersecurity": "",
            "item_2_properties": "",
            "item_3_legal_proceedings": "",
            "item_7_mda": "",
            "subsidiaries": [],
        }
        if include_exhibit_10:
            payload["exhibit_10_contracts"] = [contract_entry]
        return payload

    mock_client.extract_10k_sections.side_effect = _fake_extract_10k

    constituent = SP500Constituent(
        ticker="NVDA",
        cik="0001045810",
        company_name="NVIDIA Corporation",
        gics_sector="Information Technology",
        gics_sub_industry="Semiconductors",
    )
    harvester = HistoricalSECHarvester(client=mock_client, output_dir=tmp_path, rate_limit_delay_sec=0.0)

    # Default include_exhibit_10=False: no exhibit_10_contracts.json staged
    stats_default = harvester.harvest_company_year(constituent, year=2024, forms=["10-K"])
    assert stats_default["exhibit_10_count"] == 0
    acc_dir = tmp_path / "NVDA" / "2024" / "10K_0001045810-24-000029"
    assert not (acc_dir / "exhibit_10_contracts.json").exists()

    # Opt-in include_exhibit_10=True: stages exhibit_10_contracts.json cleanly
    stats_opt_in = harvester.harvest_company_year(
        constituent,
        year=2024,
        forms=["10-K"],
        include_exhibit_10=True,
    )
    assert stats_opt_in["exhibit_10_count"] == 1
    ex10_file = acc_dir / "exhibit_10_contracts.json"
    assert ex10_file.is_file()
    staged_ex10 = json.loads(ex10_file.read_text(encoding="utf-8"))
    assert staged_ex10[0]["exhibit_key"] == "EX-10.2"

    # EdgarWorker processes Exhibit 10 contracts under section="Exhibit 10 - Material Contract"
    mock_driver = MagicMock()
    worker = EdgarWorker(pg_conn=None, memgraph_driver=mock_driver)
    worker.client = mock_client
    worker.linker = MagicMock()
    worker.linker.link_entity.return_value = {
        "cik": "0001045810",
        "ticker": "NVDA",
        "company_name": "NVIDIA Corporation",
    }

    seen_sections: List[str] = []
    persisted_layers: List[str] = []

    def _fake_extract_rels(text: str, **kwargs: Any) -> Dict[str, Any]:
        del text
        sec_name = str(kwargs.get("section_name") or "")
        seen_sections.append(sec_name)
        if sec_name == "Exhibit 10 - Material Contract":
            return {
                "nodes": [
                    {"id": "Taiwan Semiconductor Manufacturing", "type": "Company", "properties": {}},
                    {"id": "NVIDIA Corporation", "type": "Company", "properties": {}},
                ],
                "edges": [
                    {
                        "source": "Taiwan Semiconductor Manufacturing",
                        "target": "NVIDIA Corporation",
                        "type": "SUPPLIES_TO",
                        "properties": {"nature": "3nm_wafers", "buffer_days": 30.0},
                    }
                ],
            }
        return {"nodes": [], "edges": []}

    def _capture_persist(**kwargs: Any) -> tuple[list[str], list[dict[str, Any]]]:
        for edge in kwargs["graph_data"].get("edges", []):
            persisted_layers.append(str((edge.get("properties") or {}).get("sec_graph_layer") or ""))
        return ["NVIDIA Corporation"], [{"source": "TSM", "target": "NVDA", "type": "SUPPLIES_TO"}]

    with (
        patch("ingestion.sec.harvester.extract_sec_relationships", side_effect=_fake_extract_rels),
        patch.object(worker, "_persist_sec_graph_data", side_effect=_capture_persist) as mock_persist,
    ):
        res = worker.process_company_10k("NVDA", year=2024, include_exhibit_10=True)

    assert res["status"] == "completed"
    assert "Exhibit 10 - Material Contract" in seen_sections
    assert persisted_layers == ["DOMAIN1_OPERATIONAL"]
    assert mock_persist.call_args.kwargs["section"] == "Exhibit 10 - Material Contract"
    assert METRIC_ASC280_CONCENTRATION_SHARE == "ASC280_CONCENTRATION_SHARE"
