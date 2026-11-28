"""Unit & integration tests for Phase 2c-i Foreign SEC Filings (20-F, 40-F, 6-K, Ex. 8) (ADR-00049)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from ingestion.sec.canon import (
    get_default_harvest_forms,
    get_foreign_harvest_forms,
    is_termination_eligible_form,
)
from ingestion.sec.client import EdgarClient
from ingestion.sec.foreign import (
    DEFAULT_TIER1_FOREIGN_COUNTERPARTIES,
    assess_sec_6k_disruption,
    filter_english_operational_text,
    filter_foreign_counterparties,
    is_foreign_counterparty,
    lookup_foreign_counterparty,
    sanitize_sec_6k_text,
)
from ingestion.sec.harvester import EdgarWorker, HistoricalSECHarvester
from ingestion.sec.parser import normalize_company_name
from ingestion.sec.xbrl import XBRLPriorStore, extract_xbrl_dio_priors_from_companyfacts
from retrieval.reporting import OperationalDisruptionCategory


class _FakeAtt:
    def __init__(self, doc_type: str, desc: str, doc: str, body: str) -> None:
        self.document_type = doc_type
        self.description = desc
        self.document = doc
        self._body = body

    def text(self) -> str:
        return self._body


def test_foreign_counterparty_registry_and_domestic_isolation() -> None:
    """ADR-00049 Invariant #1 & #4: Foreign catalog is isolated from domestic defaults and cannot terminate edges."""
    assert len(DEFAULT_TIER1_FOREIGN_COUNTERPARTIES) >= 20
    assert get_default_harvest_forms() == ("10-K", "10-K/A", "10-Q", "10-Q/A", "8-K", "8-K/A")
    assert get_foreign_harvest_forms() == ("20-F", "20-F/A", "40-F", "6-K")

    for non_term in ("20-F", "20-F/A", "40-F", "6-K", "Form 6-K"):
        assert is_termination_eligible_form(non_term) is False

    tsm = lookup_foreign_counterparty("tsm")
    assert tsm is not None
    assert tsm.cik == "0001046179"
    assert tsm.annual_form == "20-F"
    assert lookup_foreign_counterparty("1046179") == tsm
    assert is_foreign_counterparty("ASML") is True
    assert is_foreign_counterparty("AAPL") is False

    canadian_40f = filter_foreign_counterparties(annual_form="40-F")
    canadian_tickers = {c.ticker for c in canadian_40f}
    assert {"ENB", "TRP", "CNI", "CP", "NTR"}.issubset(canadian_tickers)


def test_rule_306_english_and_bilingual_filter() -> None:
    """ADR-00049 Invariant #2: Deterministic O(N) Rule 306 guard strips CJK and French bilingual blocks."""
    # 1. Bilingual English + Traditional Chinese (e.g. Taiwan 6-K)
    bilingual_tw = (
        "台灣積體電路製造股份有限公司公告本公司地震影響說明\n"
        "Taiwan Semiconductor Manufacturing Company Limited reported that production lines "
        "at Fab 18 resumed inspection following the earthquake in Hualien.\n\n"
        "本公司已啟動緊急應變程序並通知主要客戶最新產能恢復進度。"
    )
    filtered_tw, report_tw = filter_english_operational_text(bilingual_tw)
    assert report_tw.is_usable_english is True
    assert "Fab 18" in filtered_tw
    assert "台灣積體電路" not in filtered_tw
    assert "緊急應變" not in filtered_tw

    # 2. Bilingual English + French (e.g. Canadian 40-F / 6-K)
    bilingual_ca = (
        "Enbridge Inc. reported a temporary curtailment along Line 3 following a pipeline inspection "
        "in partnership with Société Générale advisory engineers.\n\n"
        "Communiqué de presse : La société annonce les résultats pour le trimestre clos et informe "
        "les actionnaires dans le rapport trimestriel avec notre conseil."
    )
    filtered_ca, report_ca = filter_english_operational_text(bilingual_ca)
    assert report_ca.is_usable_english is True
    assert "Line 3" in filtered_ca
    assert "Société Générale" in filtered_ca  # English prose naming French entity preserved
    assert "Communiqué de presse" not in filtered_ca
    assert report_ca.dropped_paragraphs >= 1

    # 3. Pure non-English appendix rejected
    pure_cjk = "台灣積體電路製造股份有限公司二零二四年三月營收報告公告全文如下。"
    filtered_cjk, report_cjk = filter_english_operational_text(pure_cjk)
    assert filtered_cjk == ""
    assert report_cjk.is_usable_english is False


def test_form_6k_sanitization_and_zero_shock_negative_control_gating() -> None:
    """ADR-00049 Invariant #3: Routine 6-K filings emit zero shock; acute foreign shocks fire cleanly."""
    # Routine TSMC monthly revenue report 6-K -> negative control (0.0 shock)
    routine_6k = (
        "UNITED STATES SECURITIES AND EXCHANGE COMMISSION\n"
        "Form 6-K\n"
        "Report of Foreign Private Issuer Pursuant to Rule 13a-16 or 15d-16\n"
        "Indicate by check mark whether the registrant files or will file annual reports under cover of Form 20-F or Form 40-F.\n\n"
        "TSMC March 2024 Revenue Report: Net revenue for March 2024 was approximately NT$195.21 billion, "
        "an increase of 7.5 percent from February 2024 and an increase of 34.3 percent from March 2023.\n\n"
        "Pursuant to the requirements of the Securities Exchange Act of 1934, the registrant has duly caused this report to be signed."
    )
    cleaned_routine, stripped_cnt, lang_rep = sanitize_sec_6k_text(routine_6k)
    assert lang_rep.is_usable_english is True
    assert stripped_cnt >= 2
    assert "Rule 13a-16" not in cleaned_routine
    assert "March 2024 Revenue Report" in cleaned_routine

    routine_assessment = assess_sec_6k_disruption("TSMC March 2024 Revenue Report", routine_6k)
    assert routine_assessment.is_material_operational_disruption is False
    assert routine_assessment.shock_magnitude_pct == pytest.approx(0.0)
    assert routine_assessment.shock_duration_days == pytest.approx(0.0)
    assert routine_assessment.gating_reason == "NON_DISRUPTIVE_6K_NEGATIVE_CONTROL"

    # Non-English 6-K -> skipped with zero shock
    non_en_assessment = assess_sec_6k_disruption("", "本公司董事會決議股利分派案公告。")
    assert non_en_assessment.is_material_operational_disruption is False
    assert non_en_assessment.shock_magnitude_pct == pytest.approx(0.0)
    assert non_en_assessment.gating_reason == "NON_ENGLISH_OR_EMPTY_6K_SKIPPED"

    # Acute fab earthquake 6-K via attached Exhibit 99.1 -> fires NATURAL_DISASTER_FORCE_MAJEURE
    quake_exhibit = (
        "Exhibit 99.1 Press Release: Following the magnitude 7.2 earthquake in Taiwan, "
        "wafer fabrication tools were temporarily evacuated and production lines paused for safety inspection."
    )
    quake_assessment = assess_sec_6k_disruption(
        "TSMC Operational Update",
        "Report of Foreign Private Issuer Pursuant to Rule 13a-16 or 15d-16",
        exhibit_texts=[quake_exhibit],
    )
    assert quake_assessment.is_material_operational_disruption is True
    assert quake_assessment.disruption_category == OperationalDisruptionCategory.NATURAL_DISASTER_FORCE_MAJEURE
    assert quake_assessment.shock_magnitude_pct > 0.0

    # Export-control restriction 6-K -> fires GOVERNMENT_EXPORT_BAN_OR_SANCTION
    export_assessment = assess_sec_6k_disruption(
        "ASML Export Licensing Update",
        "The Dutch government announced an export license revocation and new export control restrictions "
        "impacting shipments of NXT:2050i lithography systems to certain customers.",
    )
    assert export_assessment.is_material_operational_disruption is True
    assert export_assessment.disruption_category == OperationalDisruptionCategory.GOVERNMENT_EXPORT_BAN_OR_SANCTION

    # Canadian pipeline rupture 6-K -> fires EQUIPMENT_BREAKDOWN_FAB_HALT
    pipe_assessment = assess_sec_6k_disruption(
        "Enbridge Mainline Operational Notice",
        "Operations on the mainline crude corridor experienced a pipeline rupture and temporary shutdown "
        "affecting deliveries to Midwest U.S. refineries.",
    )
    assert pipe_assessment.is_material_operational_disruption is True
    assert pipe_assessment.disruption_category == OperationalDisruptionCategory.EQUIPMENT_BREAKDOWN_FAB_HALT


def test_international_company_name_normalization_and_diacritics() -> None:
    """ADR-00049 §4.3: Diacritic folding and non-U.S. Exhibit 8 corporate suffixes normalize cleanly."""
    assert normalize_company_name("Société Industrielle S.A.") == "societe industrielle"
    assert normalize_company_name("ASML Netherlands B.V.") == "asml netherlands"
    assert normalize_company_name("Murata Manufacturing K.K.") == "murata manufacturing"
    assert normalize_company_name("Novo Nordisk A/S") == "novo nordisk"
    assert normalize_company_name("STMicroelectronics S.r.l.") == "stmicroelectronics"
    assert normalize_company_name("Ferrari S.p.A.") == "ferrari"
    assert normalize_company_name("Sanofi S.A.S.") == "sanofi"
    assert normalize_company_name("BHP Group Pty Ltd") == "bhp"
    assert normalize_company_name("Nokia Oyj") == "nokia"
    assert normalize_company_name("Merck KGaA") == "merck"
    # Preserve domestic cases and avoid stripping normal English words like 'as'
    assert normalize_company_name("Alphabet Inc - Class A") == "alphabet a"
    assert normalize_company_name("Texas Instruments as Supplier") == "texas instruments as supplier"


def test_client_20f_40f_6k_and_exhibit8_extraction() -> None:
    """EdgarClient extracts 20-F Items 3.D/4/5 + Exhibit 8, 40-F AIF, and English-filtered 6-K."""
    client = EdgarClient()

    # 1. Form 20-F with plain-text section fallback and Exhibit 8 attachment
    fake_20f = MagicMock()
    fake_20f.form = "20-F"
    fake_20f.accession_number = "0001046179-24-000012"
    fake_20f.filing_date = "2024-04-18"
    fake_20f.report_date = "2023-12-31"
    fake_20f.acceptance_datetime = "2024-04-18T08:15:00-04:00"
    fake_20f.obj.return_value = None
    fake_20f.text.return_value = (
        "ITEM 3.D. RISK FACTORS\n"
        "We rely on ASML Holding N.V. as sole-source supplier of EUV lithography tools.\n\n"
        "ITEM 4. INFORMATION ON THE COMPANY\n"
        "TSMC operates Fab 18 in Tainan supplying advanced 3nm wafers to Apple and NVIDIA.\n\n"
        "ITEM 5. OPERATING AND FINANCIAL REVIEW AND PROSPECTS\n"
        "Capital expenditures totaled US$30.4 billion in fiscal 2023.\n"
    )
    fake_20f.attachments = [
        _FakeAtt(
            "EX-8.1",
            "List of Significant Subsidiaries",
            "ex8-1.htm",
            "TSMC Arizona Corporation    Delaware\nJapan Advanced Semiconductor Manufacturing K.K.    Japan\n",
        ),
        _FakeAtt("EX-101.INS", "XBRL Instance", "tsm.xml", "ignored xml"),
    ]

    sec_20f = client.extract_20f_sections(fake_20f)
    assert sec_20f["form"] == "20-F"
    assert "EUV lithography" in sec_20f["item_3d_risk_factors"]
    assert "Fab 18" in sec_20f["item_4_company_info"]
    assert "Capital expenditures" in sec_20f["item_5_operating_review"]
    assert len(sec_20f["subsidiaries"]) == 2
    assert sec_20f["subsidiaries"][0]["name"] == "TSMC Arizona Corporation"

    # 2. Form 40-F with Exhibit 99.1 Annual Information Form (bilingual French stripped)
    fake_40f = MagicMock()
    fake_40f.form = "40-F"
    fake_40f.accession_number = "0000895728-24-000010"
    fake_40f.filing_date = "2024-02-09"
    fake_40f.report_date = "2023-12-31"
    fake_40f.acceptance_datetime = "2024-02-09T17:10:00-05:00"
    fake_40f.attachments = [
        _FakeAtt(
            "EX-99.1",
            "Annual Information Form",
            "aif.htm",
            (
                "ANNUAL INFORMATION FORM\n"
                "Enbridge Inc. operates the Mainline crude oil pipeline system delivering Canadian heavy crude "
                "to U.S. Midwest and Gulf Coast refineries.\n\n"
                "Communiqué de presse : La société annonce les résultats pour le trimestre clos dans le rapport trimestriel."
            ),
        )
    ]
    sec_40f = client.extract_40f_sections(fake_40f)
    assert sec_40f["form"] == "40-F"
    assert "Mainline crude oil pipeline" in sec_40f["annual_information_form"]
    assert "Communiqué de presse" not in sec_40f["annual_information_form"]


def test_currency_invariant_ifrs_xbrl_dio_extraction() -> None:
    """ADR-00049 Invariant #5: ifrs-full DIO pairs strictly identical currencies and rejects currency mismatch."""
    tsm_ifrs_facts: Dict[str, Any] = {
        "cik": 1046179,
        "entityName": "Taiwan Semiconductor Manufacturing Company Limited",
        "facts": {
            "ifrs-full": {
                "Inventories": {
                    "units": {
                        "TWD": [
                            {
                                "end": "2023-12-31",
                                "val": 250_000_000_000.0,
                                "accn": "0001046179-24-000012",
                                "form": "20-F",
                                "filed": "2024-04-18",
                            }
                        ],
                        # Mismatched USD inventory with NO USD CostOfSales (must NOT divide against TWD COGS)
                        "USD": [
                            {
                                "end": "2022-12-31",
                                "val": 8_000_000_000.0,
                                "accn": "0001046179-23-000010",
                                "form": "20-F",
                                "filed": "2023-04-20",
                            }
                        ],
                    }
                },
                "CostOfSales": {
                    "units": {
                        "TWD": [
                            {
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "val": 1_825_000_000_000.0,
                                "accn": "0001046179-24-000012",
                                "form": "20-F",
                                "filed": "2024-04-18",
                            },
                            # TWD COGS for 2022-12-31 while inventory was USD -> must NOT match!
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "val": 1_000_000_000_000.0,
                                "accn": "0001046179-23-000010",
                                "form": "20-F",
                                "filed": "2023-04-20",
                            },
                        ]
                    }
                },
            }
        },
    }

    recs = extract_xbrl_dio_priors_from_companyfacts(
        tsm_ifrs_facts,
        ticker="TSM",
        gics_sector="Information Technology",
    )
    assert len(recs) == 1
    assert recs[0].dio_days == pytest.approx(50.0)
    assert recs[0].inventory_tag == "ifrs-full:Inventories[TWD]"
    assert recs[0].cogs_tag == "ifrs-full:CostOfSales[TWD]"
    assert recs[0].form_type == "20-F"

    store = XBRLPriorStore(dio_records=recs)
    assert store.lookup_pit_dio("TSM", "2024-04-01") is None
    pit = store.lookup_pit_dio("TSM", "2024-05-01")
    assert pit is not None and pit.dio_days == pytest.approx(50.0)

    # Non-physical foreign financial institution rejected
    assert extract_xbrl_dio_priors_from_companyfacts(tsm_ifrs_facts, ticker="HSBC", gics_sector="Financials") == []


def test_harvester_and_worker_foreign_counterparty_integration(tmp_path: Path) -> None:
    """Opt-in foreign harvesting stages under FOREIGN/{TICKER}/{YEAR}/ and worker persists Exhibit 8 + 20-F."""
    tsm = lookup_foreign_counterparty("TSM")
    assert tsm is not None

    mock_client = MagicMock()
    mock_20f = MagicMock()
    mock_client.fetch_20f_filings.side_effect = lambda ticker, year=None, form_type="20-F", limit=2: (
        [mock_20f] if form_type == "20-F" else []
    )
    mock_client.extract_20f_sections.return_value = {
        "accession_number": "0001046179-24-000012",
        "filing_date": "2024-04-18",
        "acceptance_datetime": "2024-04-18T08:15:00-04:00",
        "report_date": "2023-12-31",
        "form": "20-F",
        "item_3d_risk_factors": "ASML Holding N.V. is our sole-source supplier for EUV lithography tools.",
        "item_4_company_info": "TSMC supplies 3nm wafers to NVIDIA Corporation.",
        "item_5_operating_review": "",
        "subsidiaries": [{"name": "TSMC Arizona Corporation", "jurisdiction": "Delaware"}],
    }
    mock_client.fetch_recent_6k.return_value = [
        {
            "accession_number": "0001046179-24-000020",
            "filing_date": "2024-04-04",
            "acceptance_datetime": "2024-04-04T09:00:00-04:00",
            "form": "6-K",
            "text": "Following the Hualien earthquake, production lines at Fab 18 paused briefly for safety inspection.",
            "exhibits": {},
            "is_usable_english": True,
            "latin_letter_ratio": 1.0,
        }
    ]

    harvester = HistoricalSECHarvester(client=mock_client, output_dir=tmp_path, rate_limit_delay_sec=0.0)
    stats = harvester.harvest_foreign_counterparty_year(tsm, year=2024)
    assert stats["20f_count"] == 1
    assert stats["6k_count"] == 1
    assert stats["exhibit_8_subsidiaries_count"] == 1

    foreign_dir = tmp_path / "FOREIGN" / "TSM" / "2024"
    assert (foreign_dir / "20F_0001046179-24-000012" / "exhibit_8_subsidiaries.json").is_file()
    staged_6k = json.loads((foreign_dir / "6K_0001046179-24-000020" / "form6k_events.json").read_text(encoding="utf-8"))
    assert staged_6k["disruption_assessment"]["is_material_operational_disruption"] is True
    assert staged_6k["disruption_assessment"]["disruption_category"] == "NATURAL_DISASTER_FORCE_MAJEURE"

    # EdgarWorker.process_foreign_annual persists Exhibit 8 + 20-F narrative sections cleanly
    mock_driver = MagicMock()
    worker = EdgarWorker(pg_conn=None, memgraph_driver=mock_driver)
    worker.client = mock_client
    worker.linker = MagicMock()
    worker.linker.link_entity.return_value = None

    seen_sections: List[str] = []

    def _fake_extract(text: str, **kwargs: Any) -> Dict[str, Any]:
        del text
        seen_sections.append(str(kwargs.get("section_name") or ""))
        return {
            "nodes": [
                {"id": "ASML Holding N.V.", "type": "Company", "properties": {}},
                {"id": "Taiwan Semiconductor Manufacturing Company Limited", "type": "Company", "properties": {}},
            ],
            "edges": [
                {
                    "source": "ASML Holding N.V.",
                    "target": "Taiwan Semiconductor Manufacturing Company Limited",
                    "type": "SUPPLIES_TO",
                    "properties": {"nature": "euv_lithography"},
                }
            ],
        }

    with (
        patch("ingestion.sec.harvester.extract_sec_relationships", side_effect=_fake_extract),
        patch.object(worker, "_persist_subsidiaries", return_value=(["TSMC Arizona Corporation"], [])) as mock_subs,
        patch.object(worker, "_persist_sec_graph_data", return_value=(["ASML Holding N.V."], [{"type": "SUPPLIES_TO"}])),
    ):
        res = worker.process_foreign_annual("TSM", year=2024, form_type="20-F")

    assert res["status"] == "completed"
    assert res["form_type"] == "20-F"
    assert mock_subs.call_args.kwargs["section"] == "Exhibit 8 - Significant Subsidiaries (20-F)"
    assert "Item 3.D - Risk Factors (20-F)" in seen_sections
    assert "Item 4 - Information on the Company (20-F)" in seen_sections
