"""Unit & integration tests for Phase 2a Core SEC Expansion (ADR-00046)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from ingestion.sec.canon import is_high_signal_section
from ingestion.sec.client import EdgarClient
from ingestion.sec.disruption_gating import (
    DEFAULT_MAX_FILING_CHARS,
    assess_sec_8k_disruption,
)
from ingestion.sec.harvester import EdgarWorker, HistoricalSECHarvester
from ingestion.sec.parser import is_high_signal_gleaning_section, should_glean_chunk
from ingestion.segmentation.chunker import FilingMetadata, FinancialSectionSegmenter
from retrieval.reporting import OperationalDisruptionCategory
from universe.sp500 import SP500Constituent


class _DictBackedReport:
    """Minimal deterministic stand-in for edgartools TenK / TenQ objects."""

    def __init__(self, items: Dict[str, str], **attrs: str) -> None:
        self._items = dict(items)
        for key, val in attrs.items():
            setattr(self, key, val)

    def __getitem__(self, key: str) -> Optional[str]:
        return self._items.get(key)


class _FakeAttachment:
    def __init__(
        self,
        document_type: str,
        description: str,
        body: str,
        document: str = "exhibit.htm",
    ) -> None:
        self.document_type = document_type
        self.description = description
        self._body = body
        self.document = document

    def text(self) -> str:
        return self._body


class _FakeFiling:
    def __init__(
        self,
        *,
        form: str,
        accession_number: str,
        filing_date: str,
        report_date: str = "",
        acceptance_datetime: Optional[str] = None,
        items: Optional[List[str]] = None,
        raw_text: str = "",
        obj: Any = None,
        attachments: Optional[List[Any]] = None,
        company: str = "Example Corp.",
        cik: str = "0000320193",
    ) -> None:
        self.form = form
        self.accession_number = accession_number
        self.filing_date = filing_date
        self.report_date = report_date or filing_date
        self.acceptance_datetime = acceptance_datetime
        self.items = items or []
        self._raw_text = raw_text
        self._obj = obj
        self.attachments = attachments or []
        self.company = company
        self.cik = cik

    def text(self) -> str:
        return self._raw_text

    def obj(self) -> Any:
        return self._obj


def test_extract_10k_sections_captures_all_six_sections_and_amendment_form() -> None:
    """extract_10k_sections populates Items 1, 1A, 1C, 2, 3, 7 + Ex. 21 and preserves 10-K/A."""
    client = EdgarClient()
    report = _DictBackedReport(
        {
            "Item 1C": "Board cybersecurity committee oversees vendor incident response.",
            "Item 2": "Operates 14 wafer fabs and 3.2M sq ft of logistics centers.",
            "Item 3": "Subject to FDA consent decree and patent injunction proceedings.",
            "Item 7": "MD&A: Revenue concentration with primary foundry reached 31%.",
        },
        business="Designs high-precision analog semiconductors for automotive OEMs.",
        risk_factors="Relies on single-source wafer fabrication in Taiwan.",
    )
    filing = _FakeFiling(
        form="10-K/A",
        accession_number="0000320193-24-000210",
        filing_date="2025-02-28",
        report_date="2024-09-28",
        acceptance_datetime="2025-02-28T16:35:00-05:00",
        obj=report,
        attachments=[
            _FakeAttachment("EX-21.1", "Subsidiaries of the Registrant", "Delayed Sub LLC\tDelaware, USA\n"),
        ],
    )

    sections = client.extract_10k_sections(filing)

    assert sections["form"] == "10-K/A"
    assert sections["acceptance_datetime"] == "2025-02-28T16:35:00-05:00"
    assert "analog semiconductors" in sections["item_1_business"]
    assert "single-source wafer" in sections["item_1a_risk_factors"]
    assert "cybersecurity committee" in sections["item_1c_cybersecurity"]
    assert "14 wafer fabs" in sections["item_2_properties"]
    assert "FDA consent decree" in sections["item_3_legal_proceedings"]
    assert "reached 31%" in sections["item_7_mda"]
    assert sections["subsidiaries"] == [{"name": "Delayed Sub LLC", "jurisdiction": "Delaware, USA"}]


def test_extract_10q_sections_part_qualified_disambiguation_and_segmentation() -> None:
    """extract_10q_sections uses part-qualified keys, excludes Part I Item 1 GAAP tables, and segments cleanly."""
    client = EdgarClient()
    tenq_obj = _DictBackedReport(
        {
            "Part I, Item 1": "CONDENSED CONSOLIDATED BALANCE SHEETS $100,000,000 GAAP TABLE NOISE",
            "Item 1": "CONDENSED CONSOLIDATED BALANCE SHEETS $100,000,000 GAAP TABLE NOISE",
            "Part I, Item 2": "Quarterly revenue grew 12% while wafer lead times stretched across Tier-1 fabs.",
            "Part I, Item 3": "Foreign exchange and copper commodity derivative hedges cover 65% of exposure.",
            "Part II, Item 1": "Pending EPA administrative enforcement action at the Baton Rouge refinery.",
            "Part II, Item 1A": "Updated supply-chain risk: port congestion delayed Q2 component arrivals.",
        }
    )
    filing = _FakeFiling(
        form="10-Q/A",
        accession_number="0000320193-24-000145",
        filing_date="2024-08-19",
        report_date="2024-06-29",
        acceptance_datetime="2024-08-19T14:15:00-04:00",
        obj=tenq_obj,
    )

    extracted = client.extract_10q_sections(filing)

    assert extracted["form"] == "10-Q/A"
    assert "GAAP TABLE NOISE" not in extracted["text"]
    assert "GAAP TABLE NOISE" not in extracted["part2_item1_legal_proceedings"]
    assert "wafer lead times" in extracted["part1_item2_mda"]
    assert "copper commodity" in extracted["part1_item3_market_risk"]
    assert "Baton Rouge refinery" in extracted["part2_item1_legal_proceedings"]
    assert "port congestion" in extracted["part2_item1a_risk_factors"]

    # Segmenter maps part-qualified 10-Q headings cleanly to canonical SEC sections.
    segmenter = FinancialSectionSegmenter()
    meta = FilingMetadata(cik="0000320193", ticker="AAPL", company_name="Apple Inc.", form_type="10-Q/A")
    chunks = segmenter.segment_filing(extracted["text"], meta)
    sec_names = {c.section_name for c in chunks}
    assert "Item 7 - Management Discussion and Analysis" in sec_names
    assert "Item 7A - Quantitative and Qualitative Disclosures" in sec_names
    assert "Item 3 - Legal Proceedings" in sec_names
    assert "Item 1A - Risk Factors" in sec_names
    assert "Item 1 - Business" not in sec_names
    assert "Item 2 - Properties" not in sec_names

    # Fallback path when .obj() yields no structured sections uses 50K char bound (not 15K).
    long_fallback_body = ("A" * 20_000) + " TAIL_OPERATIONAL_SIGNAL " + ("B" * 40_000)
    fallback_filing = _FakeFiling(
        form="10-Q",
        accession_number="0000320193-24-000146",
        filing_date="2024-05-03",
        raw_text=long_fallback_body,
        obj=None,
    )
    fallback = client.extract_10q_sections(fallback_filing)
    assert len(fallback["text"]) == DEFAULT_MAX_FILING_CHARS
    assert "TAIL_OPERATIONAL_SIGNAL" in fallback["text"]


def test_fetch_recent_8k_extracts_exhibit_99_skips_xbrl_and_feeds_disruption_gate() -> None:
    """fetch_recent_8k lifts cap to 50K, extracts EX-99.x, skips XBRL/binaries, supports 8-K/A, and gates disruptions."""
    client = EdgarClient()
    primary_body = (
        "Item 8.01 Other Events.\n\n"
        + ("Routine regulatory filing cover framing. " * 400)
        + "\nSee press release attached hereto as Exhibit 99.1.\n\n"
        "SIGNATURES\n"
        "Pursuant to the requirements of the Securities Exchange Act of 1934, duly signed.\n"
        "By: /s/ CFO"
    )
    assert len(primary_body) > 10_000

    ex99_1_body = (
        "FOR IMMEDIATE RELEASE\n\n"
        "On September 4, 2024, an explosion and industrial fire forced an emergency halt at our "
        "primary Gulf Coast petrochemical cracking unit.\n\n"
        "Forward-Looking Statements: Actual results may differ materially due to ransomware."
    )
    ex99_2_body = "Supplemental customer shipment schedule (" + ("X" * 60_000) + ")"

    filing_8ka = _FakeFiling(
        form="8-K/A",
        accession_number="0000320193-24-000333",
        filing_date="2024-09-05",
        acceptance_datetime="2024-09-05T16:20:00-04:00",
        items=["8.01", "9.01"],
        raw_text=primary_body,
        attachments=[
            _FakeAttachment("EX-99.1", "Press Release dated September 4, 2024", ex99_1_body, "ex99-1.htm"),
            _FakeAttachment("EX-99.2", "Supplemental Schedule", ex99_2_body, "ex99-2.htm"),
            _FakeAttachment("EX-101.INS", "XBRL Instance Document", "<xbrl>ignored</xbrl>", "inst.xml"),
            _FakeAttachment("EX-104", "Cover Page Inline XBRL", "<xbrl>cover</xbrl>", "cover.xml"),
            _FakeAttachment("GRAPHIC", "Company Logo", "PNGDATA", "logo.png"),
        ],
    )

    mock_company = MagicMock()
    mock_company.get_filings.return_value = [filing_8ka]

    with patch.object(client, "get_company", return_value=mock_company):
        records = client.fetch_recent_8k("AAPL", year=2024, form_type="8-K/A")

    mock_company.get_filings.assert_called_once_with(form="8-K/A")
    assert len(records) == 1
    rec = records[0]
    assert rec["form"] == "8-K/A"
    assert len(rec["text"]) > 10_000  # No longer truncated at legacy 10K limit
    assert set(rec["exhibits"].keys()) == {"EX-99.1", "EX-99.2"}
    assert len(rec["exhibits"]["EX-99.2"]) == DEFAULT_MAX_FILING_CHARS
    assert "explosion and industrial fire" in rec["exhibit_99_1"]

    # End-to-end flow into ADR-00014 disruption gating detects the fire exclusively inside Exhibit 99.1.
    assessment = assess_sec_8k_disruption(
        items=rec["items"],
        headline="Form 8-K/A Current Report",
        content=rec["text"],
        exhibit_texts=list(rec["exhibits"].values()),
    )
    assert assessment.is_material_operational_disruption is True
    assert assessment.disruption_category == OperationalDisruptionCategory.FACILITY_FIRE_OR_EXPLOSION
    assert "ransomware" not in assessment.sanitized_content.lower()


def test_harvester_stages_all_six_core_forms_and_enforces_strict_ablation_filter(tmp_path: Path) -> None:
    """Default harvest stages all 6 core forms + Ex. 99.1; explicit `forms` strictly blocks unrequested forms."""
    mock_client = MagicMock()
    mock_client.fetch_latest_10k.return_value = _FakeFiling(
        form="10-K", accession_number="0000320193-24-000100", filing_date="2024-11-01"
    )
    mock_client.fetch_10k_filings.return_value = [
        _FakeFiling(form="10-K/A", accession_number="0000320193-24-000101", filing_date="2024-12-10")
    ]
    mock_client.extract_10k_sections.side_effect = [
        {
            "accession_number": "0000320193-24-000100",
            "filing_date": "2024-11-01",
            "acceptance_datetime": "2024-11-01T16:15:00-04:00",
            "report_date": "2024-09-28",
            "form": "10-K",
            "item_1_business": "Item 1 text",
            "item_1a_risk_factors": "Item 1A text",
            "item_1c_cybersecurity": "Item 1C cyber text",
            "item_2_properties": "Item 2 properties text",
            "item_3_legal_proceedings": "Item 3 legal text",
            "item_7_mda": "Item 7 MD&A text",
            "subsidiaries": [{"name": "Sub A", "jurisdiction": "Ireland"}],
        },
        {
            "accession_number": "0000320193-24-000101",
            "filing_date": "2024-12-10",
            "acceptance_datetime": "2024-12-10T17:05:00-05:00",
            "report_date": "2024-09-28",
            "form": "10-K/A",
            "item_1_business": "",
            "item_1a_risk_factors": "",
            "item_1c_cybersecurity": "",
            "item_2_properties": "",
            "item_3_legal_proceedings": "",
            "item_7_mda": "Amended Item 7 MD&A note.",
            "subsidiaries": [{"name": "Delayed Sub B", "jurisdiction": "Delaware"}],
        },
    ]

    def _mock_10q(_ticker: str, year: Optional[int] = None, limit: int = 4, form_type: str = "10-Q") -> List[Dict[str, Any]]:
        if form_type == "10-Q/A":
            return [
                {
                    "accession_number": "0000320193-24-000201",
                    "filing_date": "2024-08-22",
                    "acceptance_datetime": "2024-08-22T15:00:00-04:00",
                    "form": "10-Q/A",
                    "part1_item2_mda": "Amended quarterly MD&A.",
                    "text": "PART I, ITEM 2 - MANAGEMENT'S DISCUSSION AND ANALYSIS\n\nAmended quarterly MD&A.",
                }
            ]
        return [
            {
                "accession_number": "0000320193-24-000200",
                "filing_date": "2024-08-02",
                "acceptance_datetime": "2024-08-02T16:30:00-04:00",
                "form": "10-Q",
                "part1_item2_mda": "Original quarterly MD&A.",
                "part2_item1_legal_proceedings": "Pending patent litigation.",
                "text": "PART I, ITEM 2 - MANAGEMENT'S DISCUSSION AND ANALYSIS\n\nOriginal quarterly MD&A.",
            }
        ]

    def _mock_8k(_ticker: str, limit: int = 5, year: Optional[int] = None, form_type: str = "8-K") -> List[Dict[str, Any]]:
        if form_type == "8-K/A":
            return [
                {
                    "accession_number": "0000320193-24-000301",
                    "filing_date": "2024-09-12",
                    "acceptance_datetime": "2024-09-12T16:45:00-04:00",
                    "form": "8-K/A",
                    "items": ["8.01", "9.01"],
                    "text": "Item 8.01 Amendment referencing Exhibit 99.1.",
                    "exhibit_99_1": "Amended press release: plant outage extended to 90 days.",
                    "exhibits": {"EX-99.1": "Amended press release: plant outage extended to 90 days."},
                }
            ]
        return [
            {
                "accession_number": "0000320193-24-000300",
                "filing_date": "2024-09-05",
                "acceptance_datetime": "2024-09-05T08:00:00-04:00",
                "form": "8-K",
                "items": ["8.01", "9.01"],
                "text": "Item 8.01 Current Report referencing Exhibit 99.1.",
                "exhibit_99_1": "Press release announcing temporary plant shutdown.",
                "exhibits": {"EX-99.1": "Press release announcing temporary plant shutdown."},
            }
        ]

    mock_client.fetch_10q_filings.side_effect = _mock_10q
    mock_client.fetch_recent_8k.side_effect = _mock_8k

    constituent = SP500Constituent(
        ticker="AAPL",
        cik="0000320193",
        company_name="Apple Inc.",
        gics_sector="Information Technology",
    )
    harvester = HistoricalSECHarvester(output_dir=tmp_path, client=mock_client, rate_limit_delay_sec=0.0)

    stats = harvester.harvest_company_year(constituent, year=2024)
    assert stats["10k_downloaded"] is True
    assert stats["10ka_count"] == 1
    assert stats["10q_count"] == 1
    assert stats["10qa_count"] == 1
    assert stats["8k_count"] == 1
    assert stats["8ka_count"] == 1
    assert stats["exhibit_99_1_count"] == 2
    assert stats["subsidiaries_count"] == 2

    yr_dir = tmp_path / "AAPL" / "2024"
    for fname in (
        "item_1_business.txt",
        "item_1a_risk_factors.txt",
        "item_1c_cybersecurity.txt",
        "item_2_properties.txt",
        "item_3_legal_proceedings.txt",
        "item_7_mda.txt",
        "subsidiaries.json",
    ):
        assert (yr_dir / "10K_0000320193-24-000100" / fname).is_file()

    meta_10ka = json.loads((yr_dir / "10KA_0000320193-24-000101" / "metadata.json").read_text(encoding="utf-8"))
    assert meta_10ka["form"] == "10-K/A"
    assert meta_10ka["acceptance_datetime"] == "2024-12-10T17:05:00-05:00"

    assert (yr_dir / "10Q_0000320193-24-000200" / "part1_item2_mda.txt").is_file()
    assert (yr_dir / "10Q_0000320193-24-000200" / "part2_item1_legal_proceedings.txt").is_file()
    assert (yr_dir / "10QA_0000320193-24-000201" / "part1_item2_mda.txt").is_file()

    staged_8ka = json.loads((yr_dir / "8KA_0000320193-24-000301" / "form8k_events.json").read_text(encoding="utf-8"))
    assert staged_8ka["form_type"] == "8-K/A"
    assert staged_8ka["filing_timestamp"] == "2024-09-12T16:45:00-04:00"
    assert staged_8ka["exhibit_texts"] == ["Amended press release: plant outage extended to 90 days."]

    # Strict LOFO ablation filter: requesting only ["10-K", "8-K"] must never fetch /A amendments or 10-Q.
    ablation_client = MagicMock()
    ablation_client.fetch_latest_10k.return_value = None
    ablation_client.fetch_recent_8k.return_value = []
    ablation_harvester = HistoricalSECHarvester(output_dir=tmp_path / "ablation", client=ablation_client, rate_limit_delay_sec=0.0)
    ablation_harvester.harvest_company_year(constituent, year=2024, forms=["10-K", "8-K"])
    ablation_client.fetch_latest_10k.assert_called_once()
    ablation_client.fetch_10k_filings.assert_not_called()
    ablation_client.fetch_10q_filings.assert_not_called()
    ablation_client.fetch_recent_8k.assert_called_once_with("AAPL", limit=5, year=2024)


def test_edgar_worker_processes_10ka_all_six_sections_with_amendment_pit_and_sector() -> None:
    """EdgarWorker stamps 10-K/A with amendment known_at, processes all 6 sections, and forwards gics_sector."""
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    worker = EdgarWorker(pg_conn=None, memgraph_driver=mock_driver)
    filing_10ka = _FakeFiling(
        form="10-K/A",
        accession_number="0001045609-25-000019",
        filing_date="2025-03-14",
        report_date="2024-12-31",
        acceptance_datetime="2025-03-14T16:40:00-04:00",  # Friday after 16:00 ET -> actionable Monday 2025-03-17
        company="Prologis, Inc.",
        cik="0001045609",
    )
    worker.client.fetch_10k_filings = MagicMock(return_value=[filing_10ka])
    worker.client.extract_10k_sections = MagicMock(
        return_value={
            "accession_number": "0001045609-25-000019",
            "filing_date": "2025-03-14",
            "acceptance_datetime": "2025-03-14T16:40:00-04:00",
            "report_date": "2024-12-31",
            "form": "10-K/A",
            "item_1_business": "Prologis operates logistics real estate warehouses.",
            "item_1a_risk_factors": "Tenant concentration and regional vacancy risks.",
            "item_1c_cybersecurity": "Cybersecurity incident monitoring across tenant portals.",
            "item_2_properties": "Owns 1.2 billion sq ft of distribution facilities leased to Amazon.",
            "item_3_legal_proceedings": "No material environmental remediation orders.",
            "item_7_mda": "MD&A occupancy averaged 96.4% across North American markets.",
            "subsidiaries": [{"name": "Prologis L.P.", "jurisdiction": "Delaware"}],
        }
    )

    seen_sections: List[str] = []
    seen_sectors: List[Optional[str]] = []
    seen_forms: List[str] = []

    def _fake_extract(text: str, **kwargs: Any) -> Dict[str, Any]:
        seen_sections.append(str(kwargs.get("section_name")))
        seen_sectors.append(kwargs.get("gics_sector"))
        seen_forms.append(str(kwargs.get("form_type")))
        return {
            "nodes": [{"id": "Amazon", "type": "Company", "properties": {}}],
            "edges": [{"source": "Prologis, Inc.", "target": "Amazon", "type": "SUPPLIES_TO", "properties": {}}],
        }

    with (
        patch("ingestion.sec.harvester.extract_sec_relationships", side_effect=_fake_extract),
        patch.object(worker, "_persist_sec_graph_data", return_value=(["Amazon"], [{"type": "SUPPLIES_TO"}])) as mock_persist,
        patch.object(worker, "_persist_subsidiaries", return_value=(["Prologis L.P."], [{"type": "SUBSIDIARY_OF"}])) as mock_subs,
    ):
        result = worker.process_company_10k("PLD", year=2024, form_type="10-K/A", gics_sector="Real Estate")

    assert result["status"] == "completed"
    assert result["form_type"] == "10-K/A"
    assert len(seen_sections) == 6
    assert set(seen_sections) == {
        "Item 1 - Business",
        "Item 1A - Risk Factors",
        "Item 1C - Cybersecurity",
        "Item 2 - Properties",
        "Item 3 - Legal Proceedings",
        "Item 7 - Management Discussion and Analysis",
    }
    assert all(f == "10-K/A" for f in seen_forms)
    assert all(s == "Real Estate" for s in seen_sectors)

    # Confirm both Exhibit 21 and section persistence receive amendment form_type and amendment acceptance_datetime.
    assert mock_subs.call_args.kwargs["form_type"] == "10-K/A"
    assert mock_subs.call_args.kwargs["acceptance_datetime"] == "2025-03-14T16:40:00-04:00"
    assert mock_persist.call_count == 6
    for call in mock_persist.call_args_list:
        assert call.kwargs["form_type"] == "10-K/A"
        assert call.kwargs["acceptance_datetime"] == "2025-03-14T16:40:00-04:00"


def test_sector_aware_gleaning_and_part_qualified_disambiguation() -> None:
    """Sector-scoped sections glean strictly for their declared GICS sectors; Item 1B stays globally excluded."""
    # Item 2 - Properties -> Real Estate only (and must NOT confuse 10-Q Part I, Item 2 MD&A)
    assert is_high_signal_section("Item 2 - Properties") is False
    assert is_high_signal_section("Item 2 - Properties", gics_sector="Information Technology") is False
    assert is_high_signal_section("Item 2 - Properties", gics_sector="Real Estate") is True
    assert is_high_signal_gleaning_section("Item 2", gics_sector="real estate") is True
    assert is_high_signal_section("Part I, Item 2 - Management's Discussion and Analysis", gics_sector="Real Estate") is False

    # Item 4 - Mine Safety Disclosures -> Materials & Energy only
    assert is_high_signal_section("Item 4 - Mine Safety Disclosures", gics_sector="Materials") is True
    assert is_high_signal_section("Item 4 - Mine Safety Disclosures", gics_sector="Energy") is True
    assert is_high_signal_section("Item 4 - Mine Safety Disclosures", gics_sector="Health Care") is False

    # Item 7 MD&A (and 10-Q Part I, Item 2 MD&A) -> Financials only
    assert is_high_signal_section("Item 7 - Management Discussion and Analysis", gics_sector="Financials") is True
    assert is_high_signal_section("Part I, Item 2 - Management's Discussion and Analysis", gics_sector="Financials") is True
    assert is_high_signal_section("Item 7 - Management Discussion and Analysis", gics_sector="Industrials") is False
    assert is_high_signal_section("Item 7A - Quantitative and Qualitative Disclosures", gics_sector="Financials") is False

    # 10-Q Part II, Item 1 (Legal Proceedings) must NOT falsely match 10-K Item 1 (Business)
    assert is_high_signal_section("Part II, Item 1 - Legal Proceedings") is False
    assert is_high_signal_section("Part I, Item 1 - Financial Statements") is False

    # Item 1B remains excluded regardless of sector
    for sec in (None, "Real Estate", "Financials", "Materials", "Energy", "Information Technology"):
        assert is_high_signal_section("Item 1B - Unresolved Staff Comments", gics_sector=sec) is False

    # End-to-end should_glean_chunk respects gics_sector
    assert should_glean_chunk(
        "Item 2 - Properties",
        pass1_valid_edge_count=1,
        generated_tokens=120,
        runaway_detected=False,
        enabled=True,
        gics_sector="Real Estate",
    ) is True
    assert should_glean_chunk(
        "Item 2 - Properties",
        pass1_valid_edge_count=1,
        generated_tokens=120,
        runaway_detected=False,
        enabled=True,
        gics_sector="Information Technology",
    ) is False
