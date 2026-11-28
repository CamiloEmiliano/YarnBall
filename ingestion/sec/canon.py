"""Canonical SEC EDGAR Filing Registry, Statutory Item Canon & Section Grammar (ADR-00045).

Single, zero-internal-dependency source of truth (Python standard library only) for:
1. All 33 canonical SEC EDGAR form types across 9 regulatory categories (`SECFormType`, `SEC_FORM_CANON`).
2. All 9 tracked statutory exhibits & structured XBRL feeds (`SEC_EXHIBIT_CANON`).
3. All 32 statutory SEC Form 8-K items (`SECForm8KItem`), human-readable descriptions
   (`SEC_ITEM_DESCRIPTIONS`), and 4-tier disruption precedence lattice (`SEC_8K_PRIORITY_TIERS`).
4. All 26 unified statutory filing section regexes & Pass-2 gleaning rules (`SEC_SECTION_CANON`).
5. Deterministic normalization, default-harvest, termination-eligibility (`ADR-00033` Invariant #8),
   and epistemic confidence helpers supporting Phase-2 form-ablation studies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Dict, List, Optional, Pattern, Tuple


class SECFormCategory(str, Enum):
    """High-level regulatory taxonomy grouping SEC EDGAR filings by disclosure role."""

    CORE_PERIODIC = "CORE_PERIODIC"
    CURRENT_EVENT = "CURRENT_EVENT"
    CORPORATE_ACTION_MA_SPINOFF = "CORPORATE_ACTION_MA_SPINOFF"
    FOREIGN_COUNTERPARTY = "FOREIGN_COUNTERPARTY"
    GOVERNANCE_PROXY = "GOVERNANCE_PROXY"
    SECTOR_RAW_MATERIALS = "SECTOR_RAW_MATERIALS"
    DISTRESS_LATE_FILING = "DISTRESS_LATE_FILING"
    OWNERSHIP_INSIDER = "OWNERSHIP_INSIDER"
    CAPITAL_MARKETS_OFFERING = "CAPITAL_MARKETS_OFFERING"


class SECGraphLayer(str, Enum):
    """Target Knowledge Graph partition preventing cross-domain topology pollution."""

    DOMAIN1_OPERATIONAL = "DOMAIN1_OPERATIONAL"
    GOVERNANCE_INTERLOCK = "GOVERNANCE_INTERLOCK"
    CAPITAL_OWNERSHIP = "CAPITAL_OWNERSHIP"
    FINANCING_OFFERING = "FINANCING_OFFERING"


class SECIngestionMode(str, Enum):
    """Primary parsing modality for an SEC filing or exhibit."""

    TEXT_LLM_EXTRACTION = "TEXT_LLM_EXTRACTION"
    STRUCTURED_XML_OR_XBRL = "STRUCTURED_XML_OR_XBRL"
    HYBRID = "HYBRID"


class SECFormStatus(str, Enum):
    """Lifecycle posture in YarnBall's ingestion pipeline and ablation roadmap."""

    ACTIVE = "ACTIVE"
    PLANNED = "PLANNED"
    OPTIONAL_ABLATION = "OPTIONAL_ABLATION"


class SECFormType(str, Enum):
    """Exhaustive canonical SEC EDGAR form codes tracked or reserved for S&P 500 studies."""

    # Core Domestic Periodic
    FORM_10K = "10-K"
    FORM_10KA = "10-K/A"
    FORM_10Q = "10-Q"
    FORM_10QA = "10-Q/A"

    # Material Current Events
    FORM_8K = "8-K"
    FORM_8KA = "8-K/A"

    # Corporate Restructuring, Spin-Offs, IPOs & M&A
    FORM_10 = "10"
    FORM_S1 = "S-1"
    FORM_F1 = "F-1"
    FORM_S4 = "S-4"
    FORM_F4 = "F-4"
    FORM_DEFM14A = "DEFM14A"
    FORM_SC_TO_T = "SC TO-T"
    FORM_SC_14D9 = "SC 14D-9"

    # Foreign Private Issuer & Canadian Upstream Counterparties
    FORM_20F = "20-F"
    FORM_20FA = "20-F/A"
    FORM_40F = "40-F"
    FORM_6K = "6-K"

    # Corporate Governance & Proxy
    FORM_DEF14A = "DEF 14A"

    # Sector-Specific Supply Chain & Supplemental Annuals
    FORM_SD = "SD"
    FORM_ARS = "ARS"

    # Distress, Late Filings & Statutory Delistings
    FORM_NT_10K = "NT 10-K"
    FORM_NT_10Q = "NT 10-Q"
    FORM_25 = "25"

    # Ownership, Activism & Insider Holdings (Ablation Candidates)
    FORM_4 = "4"
    FORM_13F_HR = "13F-HR"
    FORM_SC_13D = "SC 13D"
    FORM_SC_13G = "SC 13G"

    # Capital Markets & Financing Offerings (Ablation Candidates)
    FORM_S3 = "S-3"
    FORM_424B2 = "424B2"
    FORM_424B5 = "424B5"
    FORM_FWP = "FWP"
    FORM_D = "D"


@dataclass(frozen=True)
class SECFormSpec:
    """Declarative specification governing a single SEC EDGAR form type."""

    form_type: SECFormType
    category: SECFormCategory
    graph_layer: SECGraphLayer
    ingestion_mode: SECIngestionMode
    status: SECFormStatus
    default_batch_harvest: bool
    allow_contract_termination: bool
    tracked_exhibits: Tuple[str, ...]
    provenance_key: Optional[str]
    base_confidence: float
    roadmap_phase: str
    sector_scope: str
    description: str


@dataclass(frozen=True)
class SECExhibitSpec:
    """Declarative specification governing an attached SEC exhibit or XBRL feed."""

    exhibit_key: str
    status: SECFormStatus
    applicable_forms: Tuple[SECFormType, ...]
    ingestion_mode: SECIngestionMode
    provenance_key: Optional[str]
    base_confidence: float
    roadmap_phase: str
    description: str


@dataclass(frozen=True)
class SECSectionSpec:
    """Declarative specification unifying section boundary regexes and Pass-2 gleaning rules."""

    canonical_name: str
    header_pattern: Pattern[str]
    applicable_forms: Tuple[SECFormType, ...]
    high_signal_gleaning: bool
    sector_gleaning_only: Tuple[str, ...] = ()
    gleaning_match_pattern: Optional[Pattern[str]] = None


# ==============================================================================
# 1. EXHAUSTIVE SEC FORM CANON (33 FORMS ACROSS 9 CATEGORIES)
# ==============================================================================

SEC_FORM_CANON: Dict[SECFormType, SECFormSpec] = {
    SECFormType.FORM_10K: SECFormSpec(
        form_type=SECFormType.FORM_10K,
        category=SECFormCategory.CORE_PERIODIC,
        graph_layer=SECGraphLayer.DOMAIN1_OPERATIONAL,
        ingestion_mode=SECIngestionMode.HYBRID,
        status=SECFormStatus.ACTIVE,
        default_batch_harvest=True,
        allow_contract_termination=False,
        tracked_exhibits=("Exhibit 21",),
        provenance_key="SEC_10K_ITEM1",
        base_confidence=0.98,
        roadmap_phase="Phase 1 / 2a",
        sector_scope="All 11 S&P 500 GICS sectors",
        description="Annual report covering Items 1, 1A, 1C, 2, 3, 7 MD&A, and Exhibit 21.",
    ),
    SECFormType.FORM_10KA: SECFormSpec(
        form_type=SECFormType.FORM_10KA,
        category=SECFormCategory.CORE_PERIODIC,
        graph_layer=SECGraphLayer.DOMAIN1_OPERATIONAL,
        ingestion_mode=SECIngestionMode.HYBRID,
        status=SECFormStatus.ACTIVE,
        default_batch_harvest=True,
        allow_contract_termination=False,
        tracked_exhibits=("Exhibit 21",),
        provenance_key="SEC_10K_ITEM1",
        base_confidence=0.98,
        roadmap_phase="Phase 2a",
        sector_scope="All 11 S&P 500 GICS sectors",
        description="Amended annual report capturing restatements, delayed Exhibit 21 schedules, and Part III updates.",
    ),
    SECFormType.FORM_10Q: SECFormSpec(
        form_type=SECFormType.FORM_10Q,
        category=SECFormCategory.CORE_PERIODIC,
        graph_layer=SECGraphLayer.DOMAIN1_OPERATIONAL,
        ingestion_mode=SECIngestionMode.HYBRID,
        status=SECFormStatus.ACTIVE,
        default_batch_harvest=True,
        allow_contract_termination=False,
        tracked_exhibits=(),
        provenance_key="SEC_10Q",
        base_confidence=0.95,
        roadmap_phase="Phase 1 / 2a",
        sector_scope="All 11 S&P 500 GICS sectors",
        description="Quarterly report disclosing part-qualified MD&A, market risk, legal proceedings, and risk updates.",
    ),
    SECFormType.FORM_10QA: SECFormSpec(
        form_type=SECFormType.FORM_10QA,
        category=SECFormCategory.CORE_PERIODIC,
        graph_layer=SECGraphLayer.DOMAIN1_OPERATIONAL,
        ingestion_mode=SECIngestionMode.HYBRID,
        status=SECFormStatus.ACTIVE,
        default_batch_harvest=True,
        allow_contract_termination=False,
        tracked_exhibits=(),
        provenance_key="SEC_10Q",
        base_confidence=0.95,
        roadmap_phase="Phase 2a",
        sector_scope="All 11 S&P 500 GICS sectors",
        description="Amended quarterly report stamped strictly at amendment known_at.",
    ),
    SECFormType.FORM_8K: SECFormSpec(
        form_type=SECFormType.FORM_8K,
        category=SECFormCategory.CURRENT_EVENT,
        graph_layer=SECGraphLayer.DOMAIN1_OPERATIONAL,
        ingestion_mode=SECIngestionMode.TEXT_LLM_EXTRACTION,
        status=SECFormStatus.ACTIVE,
        default_batch_harvest=True,
        allow_contract_termination=True,
        tracked_exhibits=("Exhibit 99.1",),
        provenance_key="SEC_8K",
        base_confidence=0.95,
        roadmap_phase="Phase 1 / 2a",
        sector_scope="All 11 S&P 500 GICS sectors",
        description="Current report and Exhibit 99.1 disclosures covering material agreements, operational halts, cyber incidents, and terminations.",
    ),
    SECFormType.FORM_8KA: SECFormSpec(
        form_type=SECFormType.FORM_8KA,
        category=SECFormCategory.CURRENT_EVENT,
        graph_layer=SECGraphLayer.DOMAIN1_OPERATIONAL,
        ingestion_mode=SECIngestionMode.TEXT_LLM_EXTRACTION,
        status=SECFormStatus.ACTIVE,
        default_batch_harvest=True,
        allow_contract_termination=True,
        tracked_exhibits=("Exhibit 99.1",),
        provenance_key="SEC_8K",
        base_confidence=0.95,
        roadmap_phase="Phase 1 / 2a",
        sector_scope="All 11 S&P 500 GICS sectors",
        description="Amended Form 8-K correcting material event facts, Exhibit 99.1 disclosures, or contract termination dates at amendment known_at.",
    ),
    SECFormType.FORM_10: SECFormSpec(
        form_type=SECFormType.FORM_10,
        category=SECFormCategory.CORPORATE_ACTION_MA_SPINOFF,
        graph_layer=SECGraphLayer.DOMAIN1_OPERATIONAL,
        ingestion_mode=SECIngestionMode.TEXT_LLM_EXTRACTION,
        status=SECFormStatus.PLANNED,
        default_batch_harvest=False,
        allow_contract_termination=False,
        tracked_exhibits=("Exhibit 99.1", "Exhibit 10"),
        provenance_key=None,
        base_confidence=0.98,
        roadmap_phase="Phase 2c",
        sector_scope="Industrials, Health Care, Staples, Materials, Energy",
        description="Spin-off registration & Information Statement detailing TSAs, shared plants, and separation dates.",
    ),
    SECFormType.FORM_S1: SECFormSpec(
        form_type=SECFormType.FORM_S1,
        category=SECFormCategory.CORPORATE_ACTION_MA_SPINOFF,
        graph_layer=SECGraphLayer.DOMAIN1_OPERATIONAL,
        ingestion_mode=SECIngestionMode.TEXT_LLM_EXTRACTION,
        status=SECFormStatus.PLANNED,
        default_batch_harvest=False,
        allow_contract_termination=False,
        tracked_exhibits=("Exhibit 21", "Exhibit 10"),
        provenance_key=None,
        base_confidence=0.98,
        roadmap_phase="Phase 2c",
        sector_scope="All 11 S&P 500 GICS sectors",
        description="Domestic IPO registration statement closing the pre-first-10-K disclosure window.",
    ),
    SECFormType.FORM_F1: SECFormSpec(
        form_type=SECFormType.FORM_F1,
        category=SECFormCategory.CORPORATE_ACTION_MA_SPINOFF,
        graph_layer=SECGraphLayer.DOMAIN1_OPERATIONAL,
        ingestion_mode=SECIngestionMode.TEXT_LLM_EXTRACTION,
        status=SECFormStatus.PLANNED,
        default_batch_harvest=False,
        allow_contract_termination=False,
        tracked_exhibits=("Exhibit 8",),
        provenance_key=None,
        base_confidence=0.98,
        roadmap_phase="Phase 2c",
        sector_scope="Non-U.S. newly listed suppliers",
        description="Foreign Private Issuer IPO registration statement.",
    ),
    SECFormType.FORM_S4: SECFormSpec(
        form_type=SECFormType.FORM_S4,
        category=SECFormCategory.CORPORATE_ACTION_MA_SPINOFF,
        graph_layer=SECGraphLayer.DOMAIN1_OPERATIONAL,
        ingestion_mode=SECIngestionMode.TEXT_LLM_EXTRACTION,
        status=SECFormStatus.PLANNED,
        default_batch_harvest=False,
        allow_contract_termination=False,
        tracked_exhibits=("Exhibit 2",),
        provenance_key=None,
        base_confidence=0.95,
        roadmap_phase="Phase 2c",
        sector_scope="Energy, Health Care, Financials, Industrials, Consumer",
        description="Domestic stock-merger registration detailing combined facility footprints and divestitures.",
    ),
    SECFormType.FORM_F4: SECFormSpec(
        form_type=SECFormType.FORM_F4,
        category=SECFormCategory.CORPORATE_ACTION_MA_SPINOFF,
        graph_layer=SECGraphLayer.DOMAIN1_OPERATIONAL,
        ingestion_mode=SECIngestionMode.TEXT_LLM_EXTRACTION,
        status=SECFormStatus.PLANNED,
        default_batch_harvest=False,
        allow_contract_termination=False,
        tracked_exhibits=("Exhibit 2",),
        provenance_key=None,
        base_confidence=0.95,
        roadmap_phase="Phase 2c",
        sector_scope="Cross-border M&A counterparties",
        description="Cross-border merger registration statement.",
    ),
    SECFormType.FORM_DEFM14A: SECFormSpec(
        form_type=SECFormType.FORM_DEFM14A,
        category=SECFormCategory.CORPORATE_ACTION_MA_SPINOFF,
        graph_layer=SECGraphLayer.DOMAIN1_OPERATIONAL,
        ingestion_mode=SECIngestionMode.TEXT_LLM_EXTRACTION,
        status=SECFormStatus.PLANNED,
        default_batch_harvest=False,
        allow_contract_termination=False,
        tracked_exhibits=(),
        provenance_key=None,
        base_confidence=0.95,
        roadmap_phase="Phase 2c",
        sector_scope="All 11 S&P 500 GICS sectors",
        description="Definitive merger proxy detailing FTC/DOJ antitrust remedies, closing dates, and breakup triggers.",
    ),
    SECFormType.FORM_SC_TO_T: SECFormSpec(
        form_type=SECFormType.FORM_SC_TO_T,
        category=SECFormCategory.CORPORATE_ACTION_MA_SPINOFF,
        graph_layer=SECGraphLayer.DOMAIN1_OPERATIONAL,
        ingestion_mode=SECIngestionMode.TEXT_LLM_EXTRACTION,
        status=SECFormStatus.PLANNED,
        default_batch_harvest=False,
        allow_contract_termination=False,
        tracked_exhibits=(),
        provenance_key=None,
        base_confidence=0.95,
        roadmap_phase="Phase 2c",
        sector_scope="Health Care/Biotech, Industrials, IT",
        description="Third-party cash tender offer statement for corporate acquisitions.",
    ),
    SECFormType.FORM_SC_14D9: SECFormSpec(
        form_type=SECFormType.FORM_SC_14D9,
        category=SECFormCategory.CORPORATE_ACTION_MA_SPINOFF,
        graph_layer=SECGraphLayer.DOMAIN1_OPERATIONAL,
        ingestion_mode=SECIngestionMode.TEXT_LLM_EXTRACTION,
        status=SECFormStatus.PLANNED,
        default_batch_harvest=False,
        allow_contract_termination=False,
        tracked_exhibits=(),
        provenance_key=None,
        base_confidence=0.95,
        roadmap_phase="Phase 2c",
        sector_scope="Health Care/Biotech, Industrials, IT",
        description="Target board solicitation/recommendation statement detailing tender-offer counterparty terms.",
    ),
    SECFormType.FORM_20F: SECFormSpec(
        form_type=SECFormType.FORM_20F,
        category=SECFormCategory.FOREIGN_COUNTERPARTY,
        graph_layer=SECGraphLayer.DOMAIN1_OPERATIONAL,
        ingestion_mode=SECIngestionMode.HYBRID,
        status=SECFormStatus.PLANNED,
        default_batch_harvest=False,
        allow_contract_termination=False,
        tracked_exhibits=("Exhibit 8",),
        provenance_key=None,
        base_confidence=0.98,
        roadmap_phase="Phase 2c",
        sector_scope="Non-U.S. Tier-1 suppliers across Pharma, Energy, Materials, Semis",
        description="Foreign Private Issuer annual report (Item 3.D Risks, Item 4 Company, Exhibit 8 Subsidiaries).",
    ),
    SECFormType.FORM_20FA: SECFormSpec(
        form_type=SECFormType.FORM_20FA,
        category=SECFormCategory.FOREIGN_COUNTERPARTY,
        graph_layer=SECGraphLayer.DOMAIN1_OPERATIONAL,
        ingestion_mode=SECIngestionMode.HYBRID,
        status=SECFormStatus.PLANNED,
        default_batch_harvest=False,
        allow_contract_termination=False,
        tracked_exhibits=("Exhibit 8",),
        provenance_key=None,
        base_confidence=0.98,
        roadmap_phase="Phase 2c",
        sector_scope="Non-U.S. Tier-1 suppliers",
        description="Amended Foreign Private Issuer annual report stamped at amendment known_at.",
    ),
    SECFormType.FORM_40F: SECFormSpec(
        form_type=SECFormType.FORM_40F,
        category=SECFormCategory.FOREIGN_COUNTERPARTY,
        graph_layer=SECGraphLayer.DOMAIN1_OPERATIONAL,
        ingestion_mode=SECIngestionMode.HYBRID,
        status=SECFormStatus.PLANNED,
        default_batch_harvest=False,
        allow_contract_termination=False,
        tracked_exhibits=(),
        provenance_key=None,
        base_confidence=0.98,
        roadmap_phase="Phase 2c",
        sector_scope="Canadian MJDS pipelines, energy, fertilizer, mining, and Class-I rail",
        description="Canadian MJDS annual report for upstream North American infrastructure and commodity partners.",
    ),
    SECFormType.FORM_6K: SECFormSpec(
        form_type=SECFormType.FORM_6K,
        category=SECFormCategory.FOREIGN_COUNTERPARTY,
        graph_layer=SECGraphLayer.DOMAIN1_OPERATIONAL,
        ingestion_mode=SECIngestionMode.TEXT_LLM_EXTRACTION,
        status=SECFormStatus.PLANNED,
        default_batch_harvest=False,
        allow_contract_termination=False,
        tracked_exhibits=("Exhibit 99.1",),
        provenance_key=None,
        base_confidence=0.95,
        roadmap_phase="Phase 2c",
        sector_scope="Non-U.S. Tier-1 suppliers across all sectors",
        description="Foreign Private Issuer current report covering plant halts, natural disasters, export controls, and output updates.",
    ),
    SECFormType.FORM_DEF14A: SECFormSpec(
        form_type=SECFormType.FORM_DEF14A,
        category=SECFormCategory.GOVERNANCE_PROXY,
        graph_layer=SECGraphLayer.GOVERNANCE_INTERLOCK,
        ingestion_mode=SECIngestionMode.TEXT_LLM_EXTRACTION,
        status=SECFormStatus.PLANNED,
        default_batch_harvest=False,
        allow_contract_termination=False,
        tracked_exhibits=(),
        provenance_key=None,
        base_confidence=0.95,
        roadmap_phase="Phase 2d",
        sector_scope="All 11 S&P 500 GICS sectors",
        description="Definitive proxy statement covering board/executive interlocks and Reg S-K Item 404 related-party contracts.",
    ),
    SECFormType.FORM_SD: SECFormSpec(
        form_type=SECFormType.FORM_SD,
        category=SECFormCategory.SECTOR_RAW_MATERIALS,
        graph_layer=SECGraphLayer.DOMAIN1_OPERATIONAL,
        ingestion_mode=SECIngestionMode.TEXT_LLM_EXTRACTION,
        status=SECFormStatus.PLANNED,
        default_batch_harvest=False,
        allow_contract_termination=False,
        tracked_exhibits=(),
        provenance_key=None,
        base_confidence=0.95,
        roadmap_phase="Phase 2d",
        sector_scope="Materials, Industrials, Aerospace, Electronics",
        description="Specialized Disclosure (Dodd-Frank Section 1502) listing Tier-2/3 3TG conflict-mineral smelters and origins.",
    ),
    SECFormType.FORM_NT_10K: SECFormSpec(
        form_type=SECFormType.FORM_NT_10K,
        category=SECFormCategory.DISTRESS_LATE_FILING,
        graph_layer=SECGraphLayer.DOMAIN1_OPERATIONAL,
        ingestion_mode=SECIngestionMode.TEXT_LLM_EXTRACTION,
        status=SECFormStatus.PLANNED,
        default_batch_harvest=False,
        allow_contract_termination=False,
        tracked_exhibits=(),
        provenance_key=None,
        base_confidence=0.95,
        roadmap_phase="Phase 2d",
        sector_scope="All 11 S&P 500 GICS sectors",
        description="Notification of inability to timely file Form 10-K (audit, accounting, or cyber distress indicator).",
    ),
    SECFormType.FORM_NT_10Q: SECFormSpec(
        form_type=SECFormType.FORM_NT_10Q,
        category=SECFormCategory.DISTRESS_LATE_FILING,
        graph_layer=SECGraphLayer.DOMAIN1_OPERATIONAL,
        ingestion_mode=SECIngestionMode.TEXT_LLM_EXTRACTION,
        status=SECFormStatus.PLANNED,
        default_batch_harvest=False,
        allow_contract_termination=False,
        tracked_exhibits=(),
        provenance_key=None,
        base_confidence=0.95,
        roadmap_phase="Phase 2d",
        sector_scope="All 11 S&P 500 GICS sectors",
        description="Notification of inability to timely file Form 10-Q.",
    ),
    SECFormType.FORM_25: SECFormSpec(
        form_type=SECFormType.FORM_25,
        category=SECFormCategory.DISTRESS_LATE_FILING,
        graph_layer=SECGraphLayer.DOMAIN1_OPERATIONAL,
        ingestion_mode=SECIngestionMode.STRUCTURED_XML_OR_XBRL,
        status=SECFormStatus.PLANNED,
        default_batch_harvest=False,
        allow_contract_termination=False,
        tracked_exhibits=(),
        provenance_key=None,
        base_confidence=1.00,
        roadmap_phase="Phase 2d",
        sector_scope="All 11 S&P 500 GICS sectors",
        description="Statutory exchange delisting notification grounding historical index removals.",
    ),
    SECFormType.FORM_4: SECFormSpec(
        form_type=SECFormType.FORM_4,
        category=SECFormCategory.OWNERSHIP_INSIDER,
        graph_layer=SECGraphLayer.CAPITAL_OWNERSHIP,
        ingestion_mode=SECIngestionMode.STRUCTURED_XML_OR_XBRL,
        status=SECFormStatus.OPTIONAL_ABLATION,
        default_batch_harvest=False,
        allow_contract_termination=False,
        tracked_exhibits=(),
        provenance_key="SEC_FORM_4",
        base_confidence=1.00,
        roadmap_phase="Ablation",
        sector_scope="All 11 S&P 500 GICS sectors",
        description="Statement of changes in insider beneficial ownership XML; opt-in via explicit forms=['4'].",
    ),
    SECFormType.FORM_13F_HR: SECFormSpec(
        form_type=SECFormType.FORM_13F_HR,
        category=SECFormCategory.OWNERSHIP_INSIDER,
        graph_layer=SECGraphLayer.CAPITAL_OWNERSHIP,
        ingestion_mode=SECIngestionMode.STRUCTURED_XML_OR_XBRL,
        status=SECFormStatus.OPTIONAL_ABLATION,
        default_batch_harvest=False,
        allow_contract_termination=False,
        tracked_exhibits=(),
        provenance_key=None,
        base_confidence=1.00,
        roadmap_phase="Ablation",
        sector_scope="Institutional asset managers",
        description="Quarterly institutional holdings XML (ablation target for flat-vs-layered hub-collapse study).",
    ),
    SECFormType.FORM_SC_13D: SECFormSpec(
        form_type=SECFormType.FORM_SC_13D,
        category=SECFormCategory.OWNERSHIP_INSIDER,
        graph_layer=SECGraphLayer.CAPITAL_OWNERSHIP,
        ingestion_mode=SECIngestionMode.TEXT_LLM_EXTRACTION,
        status=SECFormStatus.OPTIONAL_ABLATION,
        default_batch_harvest=False,
        allow_contract_termination=False,
        tracked_exhibits=(),
        provenance_key=None,
        base_confidence=0.95,
        roadmap_phase="Ablation",
        sector_scope="All 11 S&P 500 GICS sectors",
        description="Activist >5% beneficial ownership report (breakup/spin-off campaigns vs. ownership noise ablation).",
    ),
    SECFormType.FORM_SC_13G: SECFormSpec(
        form_type=SECFormType.FORM_SC_13G,
        category=SECFormCategory.OWNERSHIP_INSIDER,
        graph_layer=SECGraphLayer.CAPITAL_OWNERSHIP,
        ingestion_mode=SECIngestionMode.TEXT_LLM_EXTRACTION,
        status=SECFormStatus.OPTIONAL_ABLATION,
        default_batch_harvest=False,
        allow_contract_termination=False,
        tracked_exhibits=(),
        provenance_key=None,
        base_confidence=0.95,
        roadmap_phase="Ablation",
        sector_scope="All 11 S&P 500 GICS sectors",
        description="Passive >5% institutional beneficial ownership report (hub-degree ablation control).",
    ),
    SECFormType.FORM_ARS: SECFormSpec(
        form_type=SECFormType.FORM_ARS,
        category=SECFormCategory.SECTOR_RAW_MATERIALS,
        graph_layer=SECGraphLayer.DOMAIN1_OPERATIONAL,
        ingestion_mode=SECIngestionMode.TEXT_LLM_EXTRACTION,
        status=SECFormStatus.OPTIONAL_ABLATION,
        default_batch_harvest=False,
        allow_contract_termination=False,
        tracked_exhibits=(),
        provenance_key=None,
        base_confidence=0.90,
        roadmap_phase="Ablation",
        sector_scope="All 11 S&P 500 GICS sectors",
        description="Annual Report to Security Holders (redundancy vs. Form 10-K ablation control).",
    ),
    SECFormType.FORM_S3: SECFormSpec(
        form_type=SECFormType.FORM_S3,
        category=SECFormCategory.CAPITAL_MARKETS_OFFERING,
        graph_layer=SECGraphLayer.FINANCING_OFFERING,
        ingestion_mode=SECIngestionMode.TEXT_LLM_EXTRACTION,
        status=SECFormStatus.OPTIONAL_ABLATION,
        default_batch_harvest=False,
        allow_contract_termination=False,
        tracked_exhibits=(),
        provenance_key=None,
        base_confidence=0.90,
        roadmap_phase="Ablation",
        sector_scope="All 11 S&P 500 GICS sectors",
        description="Shelf registration statement (capital-markets volume vs. precision ablation control).",
    ),
    SECFormType.FORM_424B2: SECFormSpec(
        form_type=SECFormType.FORM_424B2,
        category=SECFormCategory.CAPITAL_MARKETS_OFFERING,
        graph_layer=SECGraphLayer.FINANCING_OFFERING,
        ingestion_mode=SECIngestionMode.TEXT_LLM_EXTRACTION,
        status=SECFormStatus.OPTIONAL_ABLATION,
        default_batch_harvest=False,
        allow_contract_termination=False,
        tracked_exhibits=(),
        provenance_key=None,
        base_confidence=0.85,
        roadmap_phase="Ablation",
        sector_scope="Financials structured-note issuers",
        description="Structured-note pricing supplement (high-volume boilerplate & embedder-scaling ablation control).",
    ),
    SECFormType.FORM_424B5: SECFormSpec(
        form_type=SECFormType.FORM_424B5,
        category=SECFormCategory.CAPITAL_MARKETS_OFFERING,
        graph_layer=SECGraphLayer.FINANCING_OFFERING,
        ingestion_mode=SECIngestionMode.TEXT_LLM_EXTRACTION,
        status=SECFormStatus.OPTIONAL_ABLATION,
        default_batch_harvest=False,
        allow_contract_termination=False,
        tracked_exhibits=(),
        provenance_key=None,
        base_confidence=0.85,
        roadmap_phase="Ablation",
        sector_scope="All 11 S&P 500 GICS sectors",
        description="Debt and equity shelf takedown prospectus supplement (financing-noise ablation control).",
    ),
    SECFormType.FORM_FWP: SECFormSpec(
        form_type=SECFormType.FORM_FWP,
        category=SECFormCategory.CAPITAL_MARKETS_OFFERING,
        graph_layer=SECGraphLayer.FINANCING_OFFERING,
        ingestion_mode=SECIngestionMode.TEXT_LLM_EXTRACTION,
        status=SECFormStatus.OPTIONAL_ABLATION,
        default_batch_harvest=False,
        allow_contract_termination=False,
        tracked_exhibits=(),
        provenance_key=None,
        base_confidence=0.85,
        roadmap_phase="Ablation",
        sector_scope="All 11 S&P 500 GICS sectors",
        description="Free Writing Prospectus marketing term sheet (ablation control).",
    ),
    SECFormType.FORM_D: SECFormSpec(
        form_type=SECFormType.FORM_D,
        category=SECFormCategory.CAPITAL_MARKETS_OFFERING,
        graph_layer=SECGraphLayer.FINANCING_OFFERING,
        ingestion_mode=SECIngestionMode.STRUCTURED_XML_OR_XBRL,
        status=SECFormStatus.OPTIONAL_ABLATION,
        default_batch_harvest=False,
        allow_contract_termination=False,
        tracked_exhibits=(),
        provenance_key=None,
        base_confidence=0.90,
        roadmap_phase="Ablation",
        sector_scope="Private exempt offerings",
        description="Notice of exempt offering of securities (low-signal ablation control).",
    ),
}


# ==============================================================================
# 2. TRACKED EXHIBITS & STRUCTURED XBRL FEEDS (9 ENTRIES)
# ==============================================================================

SEC_EXHIBIT_CANON: Dict[str, SECExhibitSpec] = {
    "Exhibit 21": SECExhibitSpec(
        exhibit_key="Exhibit 21",
        status=SECFormStatus.ACTIVE,
        applicable_forms=(SECFormType.FORM_10K, SECFormType.FORM_10KA, SECFormType.FORM_S1),
        ingestion_mode=SECIngestionMode.HYBRID,
        provenance_key="SEC_EXHIBIT_21",
        base_confidence=1.00,
        roadmap_phase="Phase 1",
        description="Sworn statutory list of domestic operating subsidiaries backing CIK rollup.",
    ),
    "Exhibit 99.1": SECExhibitSpec(
        exhibit_key="Exhibit 99.1",
        status=SECFormStatus.ACTIVE,
        applicable_forms=(SECFormType.FORM_8K, SECFormType.FORM_8KA, SECFormType.FORM_6K, SECFormType.FORM_10),
        ingestion_mode=SECIngestionMode.TEXT_LLM_EXTRACTION,
        provenance_key=None,
        base_confidence=0.95,
        roadmap_phase="Phase 2a / 2c",
        description="Attached operational press releases on Form 8-K/8-K/A/6-K and spin-off Information Statements on Form 10.",
    ),
    "Exhibit 10": SECExhibitSpec(
        exhibit_key="Exhibit 10",
        status=SECFormStatus.ACTIVE,
        applicable_forms=(
            SECFormType.FORM_10K,
            SECFormType.FORM_10KA,
            SECFormType.FORM_10Q,
            SECFormType.FORM_10QA,
            SECFormType.FORM_8K,
            SECFormType.FORM_8KA,
            SECFormType.FORM_10,
            SECFormType.FORM_S1,
        ),
        ingestion_mode=SECIngestionMode.TEXT_LLM_EXTRACTION,
        provenance_key="SEC_EXHIBIT_10",
        base_confidence=0.98,
        roadmap_phase="Phase 2b",
        description="Material contracts (supply, licensing, LNG/pipeline off-take, DoD/airline, PPA, PBM, co-brand cards).",
    ),
    "Exhibit 101 / XBRL": SECExhibitSpec(
        exhibit_key="Exhibit 101 / XBRL",
        status=SECFormStatus.ACTIVE,
        applicable_forms=(
            SECFormType.FORM_10K,
            SECFormType.FORM_10KA,
            SECFormType.FORM_10Q,
            SECFormType.FORM_10QA,
            SECFormType.FORM_20F,
        ),
        ingestion_mode=SECIngestionMode.STRUCTURED_XML_OR_XBRL,
        provenance_key="SEC_XBRL_COMPANYFACTS",
        base_confidence=1.00,
        roadmap_phase="Phase 2b",
        description="Inline XBRL / EDGAR companyfacts (Days Inventory Outstanding, ASC 280 customer concentration, split ratios).",
    ),
    "Exhibit 8": SECExhibitSpec(
        exhibit_key="Exhibit 8",
        status=SECFormStatus.PLANNED,
        applicable_forms=(SECFormType.FORM_20F, SECFormType.FORM_20FA, SECFormType.FORM_F1),
        ingestion_mode=SECIngestionMode.HYBRID,
        provenance_key=None,
        base_confidence=1.00,
        roadmap_phase="Phase 2c",
        description="Foreign Private Issuer significant subsidiaries schedule (FPI equivalent of Exhibit 21).",
    ),
    "Exhibit 2": SECExhibitSpec(
        exhibit_key="Exhibit 2",
        status=SECFormStatus.PLANNED,
        applicable_forms=(SECFormType.FORM_8K, SECFormType.FORM_S4, SECFormType.FORM_F4, SECFormType.FORM_10K),
        ingestion_mode=SECIngestionMode.TEXT_LLM_EXTRACTION,
        provenance_key=None,
        base_confidence=0.98,
        roadmap_phase="Phase 2c",
        description="Plan of acquisition, reorganization, merger, divestiture, or succession.",
    ),
    "Exhibit 95": SECExhibitSpec(
        exhibit_key="Exhibit 95",
        status=SECFormStatus.PLANNED,
        applicable_forms=(SECFormType.FORM_10K, SECFormType.FORM_10Q),
        ingestion_mode=SECIngestionMode.HYBRID,
        provenance_key=None,
        base_confidence=0.98,
        roadmap_phase="Phase 2d",
        description="Mine Safety Disclosures (Dodd-Frank Section 1503, paired with Form 8-K Item 1.04).",
    ),
    "Exhibit 96": SECExhibitSpec(
        exhibit_key="Exhibit 96",
        status=SECFormStatus.PLANNED,
        applicable_forms=(SECFormType.FORM_10K, SECFormType.FORM_20F),
        ingestion_mode=SECIngestionMode.TEXT_LLM_EXTRACTION,
        provenance_key=None,
        base_confidence=0.95,
        roadmap_phase="Phase 2d",
        description="Mining Technical Report Summaries (Reg S-K Subpart 1300) for Materials & Energy.",
    ),
    "Exhibit 4": SECExhibitSpec(
        exhibit_key="Exhibit 4",
        status=SECFormStatus.OPTIONAL_ABLATION,
        applicable_forms=(SECFormType.FORM_10K, SECFormType.FORM_8K),
        ingestion_mode=SECIngestionMode.TEXT_LLM_EXTRACTION,
        provenance_key=None,
        base_confidence=0.95,
        roadmap_phase="Ablation",
        description="Instruments defining rights of security holders and debt indentures/covenants.",
    ),
}


# ==============================================================================
# 3. EXHAUSTIVE STATUTORY FORM 8-K ITEM CANON & 4-TIER SEVERITY LATTICE
# ==============================================================================

class SECForm8KItem(str, Enum):
    """Exhaustive statutory SEC Form 8-K Item canon across all 9 statutory sections."""

    # Section 1: Registrant's Business and Operations
    ITEM_1_01 = "Item 1.01"  # Entry into a Material Definitive Agreement
    ITEM_1_02 = "Item 1.02"  # Termination of a Material Definitive Agreement
    ITEM_1_03 = "Item 1.03"  # Bankruptcy or Receivership
    ITEM_1_04 = "Item 1.04"  # Mine Safety - Reporting of Shutdowns & Violations
    ITEM_1_05 = "Item 1.05"  # Material Cybersecurity Incidents
    # Section 2: Financial Information
    ITEM_2_01 = "Item 2.01"  # Completion of Acquisition or Disposition of Assets
    ITEM_2_02 = "Item 2.02"  # Results of Operations and Financial Condition
    ITEM_2_03 = "Item 2.03"  # Creation of a Direct Financial Obligation
    ITEM_2_04 = "Item 2.04"  # Acceleration of Obligations (Debt/Credit Default)
    ITEM_2_05 = "Item 2.05"  # Costs Associated with Exit/Disposal (Plant Closures)
    ITEM_2_06 = "Item 2.06"  # Material Impairments (Asset/Plant Write-Offs)
    # Section 3: Securities and Trading Markets
    ITEM_3_01 = "Item 3.01"  # Notice of Delisting / Failure to Satisfy Listing Rule
    ITEM_3_02 = "Item 3.02"  # Unregistered Sales of Equity Securities
    ITEM_3_03 = "Item 3.03"  # Material Modification to Rights of Security Holders
    # Section 4: Accountants and Financial Statements
    ITEM_4_01 = "Item 4.01"  # Changes in Registrant's Certifying Accountant
    ITEM_4_02 = "Item 4.02"  # Non-Reliance on Previously Issued Financial Statements
    # Section 5: Corporate Governance and Management
    ITEM_5_01 = "Item 5.01"  # Changes in Control of Registrant
    ITEM_5_02 = "Item 5.02"  # Departure/Election of Directors or Principal Officers
    ITEM_5_03 = "Item 5.03"  # Amendments to Articles of Incorporation or Bylaws
    ITEM_5_04 = "Item 5.04"  # Temporary Suspension of Trading Under Benefit Plans
    ITEM_5_05 = "Item 5.05"  # Amendments/Waivers to Code of Ethics
    ITEM_5_06 = "Item 5.06"  # Change in Shell Company Status
    ITEM_5_07 = "Item 5.07"  # Submission of Matters to a Vote of Security Holders
    ITEM_5_08 = "Item 5.08"  # Shareholder Director Nominations
    # Section 6: Asset-Backed Securities
    ITEM_6_01 = "Item 6.01"  # ABS Informational and Computational Material
    ITEM_6_02 = "Item 6.02"  # Change of Servicer or Trustee
    ITEM_6_03 = "Item 6.03"  # Change in Credit Enhancement or External Support
    ITEM_6_04 = "Item 6.04"  # Failure to Make a Required Distribution
    ITEM_6_05 = "Item 6.05"  # Securities Act Updating Disclosure
    # Section 7: Regulation FD
    ITEM_7_01 = "Item 7.01"  # Regulation FD Disclosure
    # Section 8: Other Material Events
    ITEM_8_01 = "Item 8.01"  # Other Events (Statutory General Catch-All)
    # Section 9: Financial Statements and Exhibits
    ITEM_9_01 = "Item 9.01"  # Financial Statements and Exhibits


SEC_ITEM_DESCRIPTIONS: Dict[str, str] = {
    "Item 1.01": "Entry into a Material Definitive Agreement",
    "Item 1.02": "Termination of a Material Definitive Agreement",
    "Item 1.03": "Bankruptcy or Receivership",
    "Item 1.04": "Mine Safety - Reporting of Shutdowns and Violations",
    "Item 1.05": "Material Cybersecurity Incidents",
    "Item 2.01": "Completion of Acquisition or Disposition of Assets",
    "Item 2.02": "Results of Operations and Financial Condition",
    "Item 2.03": "Creation of a Direct Financial Obligation",
    "Item 2.04": "Acceleration of Obligations (Debt/Credit Default)",
    "Item 2.05": "Costs Associated with Exit/Disposal Activities (Plant Closures)",
    "Item 2.06": "Material Impairments (Asset/Plant Write-Offs)",
    "Item 3.01": "Notice of Delisting or Failure to Satisfy a Continued Listing Rule",
    "Item 3.02": "Unregistered Sales of Equity Securities",
    "Item 3.03": "Material Modification to Rights of Security Holders",
    "Item 4.01": "Changes in Registrant's Certifying Accountant",
    "Item 4.02": "Non-Reliance on Previously Issued Financial Statements",
    "Item 5.01": "Changes in Control of Registrant",
    "Item 5.02": "Departure of Directors or Certain Officers; Election of Directors",
    "Item 5.03": "Amendments to Articles of Incorporation or Bylaws",
    "Item 5.04": "Temporary Suspension of Trading Under Employee Benefit Plans",
    "Item 5.05": "Amendments to the Registrant's Code of Ethics",
    "Item 5.06": "Change in Shell Company Status",
    "Item 5.07": "Submission of Matters to a Vote of Security Holders",
    "Item 5.08": "Shareholder Director Nominations",
    "Item 6.01": "ABS Informational and Computational Material",
    "Item 6.02": "Change of Servicer or Trustee",
    "Item 6.03": "Change in Credit Enhancement or External Support",
    "Item 6.04": "Failure to Make a Required Distribution",
    "Item 6.05": "Securities Act Updating Disclosure",
    "Item 7.01": "Regulation FD Disclosure",
    "Item 8.01": "Other Events",
    "Item 9.01": "Financial Statements and Exhibits",
}


# Canonical Priority Tiers (ADR-00014 Section 4.2, consolidated by ADR-00045)
SEC_8K_PRIORITY_TIERS: Tuple[Tuple[int, Tuple[str, ...]], ...] = (
    (
        1,
        (
            SECForm8KItem.ITEM_1_05.value,
            SECForm8KItem.ITEM_1_04.value,
            SECForm8KItem.ITEM_1_03.value,
            SECForm8KItem.ITEM_1_02.value,
            SECForm8KItem.ITEM_2_05.value,
            SECForm8KItem.ITEM_2_06.value,
            SECForm8KItem.ITEM_2_04.value,
        ),
    ),
    (
        2,
        (
            SECForm8KItem.ITEM_1_01.value,
            SECForm8KItem.ITEM_8_01.value,
        ),
    ),
    (
        3,
        (
            SECForm8KItem.ITEM_2_01.value,
            SECForm8KItem.ITEM_2_02.value,
            SECForm8KItem.ITEM_2_03.value,
            SECForm8KItem.ITEM_7_01.value,
        ),
    ),
    (
        4,
        (
            SECForm8KItem.ITEM_3_01.value,
            SECForm8KItem.ITEM_3_02.value,
            SECForm8KItem.ITEM_3_03.value,
            SECForm8KItem.ITEM_4_01.value,
            SECForm8KItem.ITEM_4_02.value,
            SECForm8KItem.ITEM_5_01.value,
            SECForm8KItem.ITEM_5_02.value,
            SECForm8KItem.ITEM_5_03.value,
            SECForm8KItem.ITEM_5_04.value,
            SECForm8KItem.ITEM_5_05.value,
            SECForm8KItem.ITEM_5_06.value,
            SECForm8KItem.ITEM_5_07.value,
            SECForm8KItem.ITEM_5_08.value,
            SECForm8KItem.ITEM_6_01.value,
            SECForm8KItem.ITEM_6_02.value,
            SECForm8KItem.ITEM_6_03.value,
            SECForm8KItem.ITEM_6_04.value,
            SECForm8KItem.ITEM_6_05.value,
            SECForm8KItem.ITEM_9_01.value,
        ),
    ),
)


# ==============================================================================
# 4. UNIFIED STATUTORY SECTION & PASS-2 GLEANING CANON (26 SECTIONS)
# ==============================================================================

_PERIODIC_FORMS: Tuple[SECFormType, ...] = (
    SECFormType.FORM_10K,
    SECFormType.FORM_10KA,
    SECFormType.FORM_10Q,
    SECFormType.FORM_10QA,
)
_CURRENT_8K_FORMS: Tuple[SECFormType, ...] = (
    SECFormType.FORM_8K,
    SECFormType.FORM_8KA,
)
_FPI_20F_FORMS: Tuple[SECFormType, ...] = (
    SECFormType.FORM_20F,
    SECFormType.FORM_20FA,
)

SEC_SECTION_CANON: Tuple[SECSectionSpec, ...] = (
    # Form 10-K / 10-Q Core Sections
    SECSectionSpec(
        canonical_name="Item 1 - Business",
        header_pattern=re.compile(r"(?:\n|^)\s*(?:ITEM\s+1[.\s–—\-]+BUSINESS\b)", re.IGNORECASE),
        applicable_forms=_PERIODIC_FORMS,
        high_signal_gleaning=True,
        gleaning_match_pattern=re.compile(r"(?:^\s*item\s*1\s*$|\bitem\s*1\s*[-–—.\s]+business\b)", re.IGNORECASE),
    ),
    SECSectionSpec(
        canonical_name="Item 1A - Risk Factors",
        header_pattern=re.compile(
            r"(?:\n|^)\s*(?:(?:PART\s+II[,\s–—\-]+)?ITEM\s+1A[.\s–—\-]+RISK\s+FACTORS\b)",
            re.IGNORECASE,
        ),
        applicable_forms=_PERIODIC_FORMS,
        high_signal_gleaning=True,
        gleaning_match_pattern=re.compile(r"\bitem\s*1a\b", re.IGNORECASE),
    ),
    SECSectionSpec(
        canonical_name="Item 1B - Unresolved Staff Comments",
        header_pattern=re.compile(
            r"(?:\n|^)\s*(?:ITEM\s+1B[.\s–—\-]+UNRESOLVED\s+STAFF\s+COMMENTS\b)",
            re.IGNORECASE,
        ),
        applicable_forms=_PERIODIC_FORMS,
        high_signal_gleaning=False,
    ),
    SECSectionSpec(
        canonical_name="Item 1C - Cybersecurity",
        header_pattern=re.compile(r"(?:\n|^)\s*(?:ITEM\s+1C[.\s–—\-]+CYBERSECURITY\b)", re.IGNORECASE),
        applicable_forms=(SECFormType.FORM_10K, SECFormType.FORM_10KA),
        high_signal_gleaning=True,
        gleaning_match_pattern=re.compile(r"\bitem\s*1c\b", re.IGNORECASE),
    ),
    SECSectionSpec(
        canonical_name="Item 2 - Properties",
        header_pattern=re.compile(r"(?:\n|^)\s*(?:ITEM\s+2[.\s–—\-]+PROPERTIES\b)", re.IGNORECASE),
        applicable_forms=_PERIODIC_FORMS,
        high_signal_gleaning=False,
        sector_gleaning_only=("Real Estate",),
        gleaning_match_pattern=re.compile(
            r"(?:^\s*item\s*2\s*$|\bitem\s*2\s*[-–—.\s]+properties\b)",
            re.IGNORECASE,
        ),
    ),
    SECSectionSpec(
        canonical_name="Item 3 - Legal Proceedings",
        header_pattern=re.compile(
            r"(?:\n|^)\s*(?:(?:ITEM\s+3|PART\s+II[,\s–—\-]+ITEM\s+1)[.\s–—\-]+LEGAL\s+PROCEEDINGS\b)",
            re.IGNORECASE,
        ),
        applicable_forms=_PERIODIC_FORMS,
        high_signal_gleaning=False,
    ),
    SECSectionSpec(
        canonical_name="Item 4 - Mine Safety Disclosures",
        header_pattern=re.compile(r"(?:\n|^)\s*(?:ITEM\s+4[.\s–—\-]+MINE\s+SAFETY\b)", re.IGNORECASE),
        applicable_forms=_PERIODIC_FORMS,
        high_signal_gleaning=False,
        sector_gleaning_only=("Materials", "Energy"),
        gleaning_match_pattern=re.compile(
            r"(?:^\s*item\s*4\s*$|\bitem\s*4\s*[-–—.\s]+mine\s+safety\b|\bmine\s+safety\b)",
            re.IGNORECASE,
        ),
    ),
    SECSectionSpec(
        canonical_name="Item 7 - Management Discussion and Analysis",
        header_pattern=re.compile(
            r"(?:\n|^)\s*(?:(?:ITEM\s+7|PART\s+I[,\s–—\-]+ITEM\s+2)[.\s–—\-]+MANAGEMENT(?:'S)?\s+DISCUSSION\s+AND\s+ANALYSIS\b)",
            re.IGNORECASE,
        ),
        applicable_forms=_PERIODIC_FORMS,
        high_signal_gleaning=False,
        sector_gleaning_only=("Financials",),
        gleaning_match_pattern=re.compile(
            r"(?:\bitem\s*7\b(?!a\b)(?!\.\d)|\bpart\s*i\s*,?\s*item\s*2\b.*\b(?:management|md&a)\b|\bmanagement(?:'s)?\s+discussion\s+and\s+analysis\b)",
            re.IGNORECASE,
        ),
    ),
    SECSectionSpec(
        canonical_name="Item 7A - Quantitative and Qualitative Disclosures",
        header_pattern=re.compile(
            r"(?:\n|^)\s*(?:(?:ITEM\s+7A|PART\s+I[,\s–—\-]+ITEM\s+3)[.\s–—\-]+QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES\b)",
            re.IGNORECASE,
        ),
        applicable_forms=_PERIODIC_FORMS,
        high_signal_gleaning=False,
    ),
    SECSectionSpec(
        canonical_name="Item 8 - Financial Statements",
        header_pattern=re.compile(r"(?:\n|^)\s*(?:ITEM\s+8[.\s–—\-]+FINANCIAL\s+STATEMENTS\b)", re.IGNORECASE),
        applicable_forms=_PERIODIC_FORMS,
        high_signal_gleaning=False,
    ),
    # Form 8-K Material Current Event Sections
    SECSectionSpec(
        canonical_name="Item 1.01 - Entry into Material Agreement",
        header_pattern=re.compile(
            r"(?:\n|^)\s*(?:ITEM\s+1\.01[.\s–—\-]+ENTRY\s+INTO\s+A\s+MATERIAL\s+DEFINITIVE\s+AGREEMENT\b)",
            re.IGNORECASE,
        ),
        applicable_forms=_CURRENT_8K_FORMS,
        high_signal_gleaning=True,
        gleaning_match_pattern=re.compile(r"\bitem\s*1\.01\b", re.IGNORECASE),
    ),
    SECSectionSpec(
        canonical_name="Item 1.02 - Termination of Material Agreement",
        header_pattern=re.compile(
            r"(?:\n|^)\s*(?:ITEM\s+1\.02[.\s–—\-]+TERMINATION\s+OF\s+A\s+MATERIAL\s+DEFINITIVE\s+AGREEMENT\b)",
            re.IGNORECASE,
        ),
        applicable_forms=_CURRENT_8K_FORMS,
        high_signal_gleaning=True,
        gleaning_match_pattern=re.compile(r"\bitem\s*1\.02\b", re.IGNORECASE),
    ),
    SECSectionSpec(
        canonical_name="Item 1.03 - Bankruptcy or Receivership",
        header_pattern=re.compile(
            r"(?:\n|^)\s*(?:ITEM\s+1\.03[.\s–—\-]+BANKRUPTCY\s+OR\s+RECEIVERSHIP\b)",
            re.IGNORECASE,
        ),
        applicable_forms=_CURRENT_8K_FORMS,
        high_signal_gleaning=True,
        gleaning_match_pattern=re.compile(r"\bitem\s*1\.03\b", re.IGNORECASE),
    ),
    SECSectionSpec(
        canonical_name="Item 1.04 - Mine Safety Shutdowns",
        header_pattern=re.compile(r"(?:\n|^)\s*(?:ITEM\s+1\.04[.\s–—\-]+MINE\s+SAFETY\b)", re.IGNORECASE),
        applicable_forms=_CURRENT_8K_FORMS,
        high_signal_gleaning=True,
        gleaning_match_pattern=re.compile(r"\bitem\s*1\.04\b", re.IGNORECASE),
    ),
    SECSectionSpec(
        canonical_name="Item 1.05 - Material Cybersecurity Incidents",
        header_pattern=re.compile(
            r"(?:\n|^)\s*(?:ITEM\s+1\.05[.\s–—\-]+MATERIAL\s+CYBERSECURITY\s+INCIDENTS?\b)",
            re.IGNORECASE,
        ),
        applicable_forms=_CURRENT_8K_FORMS,
        high_signal_gleaning=True,
        gleaning_match_pattern=re.compile(r"\bitem\s*1\.05\b", re.IGNORECASE),
    ),
    SECSectionSpec(
        canonical_name="Item 2.01 - Completion of Acquisition or Disposition",
        header_pattern=re.compile(
            r"(?:\n|^)\s*(?:ITEM\s+2\.01[.\s–—\-]+COMPLETION\s+OF\s+ACQUISITION\s+OR\s+DISPOSITION\b)",
            re.IGNORECASE,
        ),
        applicable_forms=_CURRENT_8K_FORMS,
        high_signal_gleaning=True,
        gleaning_match_pattern=re.compile(r"\bitem\s*2\.01\b", re.IGNORECASE),
    ),
    SECSectionSpec(
        canonical_name="Item 2.02 - Results of Operations",
        header_pattern=re.compile(r"(?:\n|^)\s*(?:ITEM\s+2\.02[.\s–—\-]+RESULTS\s+OF\s+OPERATIONS\b)", re.IGNORECASE),
        applicable_forms=_CURRENT_8K_FORMS,
        high_signal_gleaning=False,
    ),
    SECSectionSpec(
        canonical_name="Item 2.04 - Acceleration of Financial Obligations",
        header_pattern=re.compile(
            r"(?:\n|^)\s*(?:ITEM\s+2\.04[.\s–—\-]+TRIGGERING\s+EVENTS\b)",
            re.IGNORECASE,
        ),
        applicable_forms=_CURRENT_8K_FORMS,
        high_signal_gleaning=True,
        gleaning_match_pattern=re.compile(r"\bitem\s*2\.04\b", re.IGNORECASE),
    ),
    SECSectionSpec(
        canonical_name="Item 2.05 - Exit or Disposal Activities",
        header_pattern=re.compile(
            r"(?:\n|^)\s*(?:ITEM\s+2\.05[.\s–—\-]+COSTS\s+ASSOCIATED\s+WITH\s+EXIT\b)",
            re.IGNORECASE,
        ),
        applicable_forms=_CURRENT_8K_FORMS,
        high_signal_gleaning=True,
        gleaning_match_pattern=re.compile(r"\bitem\s*2\.05\b", re.IGNORECASE),
    ),
    SECSectionSpec(
        canonical_name="Item 2.06 - Material Impairments",
        header_pattern=re.compile(r"(?:\n|^)\s*(?:ITEM\s+2\.06[.\s–—\-]+MATERIAL\s+IMPAIRMENTS?\b)", re.IGNORECASE),
        applicable_forms=_CURRENT_8K_FORMS,
        high_signal_gleaning=True,
        gleaning_match_pattern=re.compile(r"\bitem\s*2\.06\b", re.IGNORECASE),
    ),
    SECSectionSpec(
        canonical_name="Item 7.01 - Regulation FD Disclosure",
        header_pattern=re.compile(r"(?:\n|^)\s*(?:ITEM\s+7\.01[.\s–—\-]+REGULATION\s+FD\b)", re.IGNORECASE),
        applicable_forms=_CURRENT_8K_FORMS,
        high_signal_gleaning=False,
    ),
    SECSectionSpec(
        canonical_name="Item 8.01 - Other Events",
        header_pattern=re.compile(r"(?:\n|^)\s*(?:ITEM\s+8\.01[.\s–—\-]+OTHER\s+EVENTS\b)", re.IGNORECASE),
        applicable_forms=_CURRENT_8K_FORMS,
        high_signal_gleaning=True,
        gleaning_match_pattern=re.compile(r"\bitem\s*8\.01\b", re.IGNORECASE),
    ),
    # Exhibits (10 & 21) & Foreign 20-F Reserved Boundaries
    SECSectionSpec(
        canonical_name="Exhibit 10 - Material Contract",
        header_pattern=re.compile(
            r"(?:\n|^)\s*(?:EXHIBIT\s+10(?:\.\d+)?\b[.\s–—\-]+MATERIAL\s+CONTRACT\b)",
            re.IGNORECASE,
        ),
        applicable_forms=(
            SECFormType.FORM_10K,
            SECFormType.FORM_10KA,
            SECFormType.FORM_10Q,
            SECFormType.FORM_10QA,
            SECFormType.FORM_8K,
            SECFormType.FORM_8KA,
            SECFormType.FORM_10,
            SECFormType.FORM_S1,
        ),
        high_signal_gleaning=True,
        gleaning_match_pattern=re.compile(r"\bexhibit\s*10(?:\.\d+)?\b", re.IGNORECASE),
    ),
    SECSectionSpec(
        canonical_name="Exhibit 21 - Subsidiaries",
        header_pattern=re.compile(r"(?:\n|^)\s*(?:EXHIBIT\s+21(?:\.1)?[.\s–—\-]+SUBSIDIARIES\b)", re.IGNORECASE),
        applicable_forms=(SECFormType.FORM_10K, SECFormType.FORM_10KA, SECFormType.FORM_S1),
        high_signal_gleaning=False,
    ),
    SECSectionSpec(
        canonical_name="Item 3.D - Risk Factors (20-F)",
        header_pattern=re.compile(r"(?:\n|^)\s*(?:ITEM\s+3\.?D[.\s–—\-]+RISK\s+FACTORS\b)", re.IGNORECASE),
        applicable_forms=_FPI_20F_FORMS,
        high_signal_gleaning=True,
        gleaning_match_pattern=re.compile(r"\bitem\s*3\.?d\b", re.IGNORECASE),
    ),
    SECSectionSpec(
        canonical_name="Item 4 - Information on the Company (20-F)",
        header_pattern=re.compile(
            r"(?:\n|^)\s*(?:ITEM\s+4[.\s–—\-]+INFORMATION\s+ON\s+THE\s+COMPANY\b)",
            re.IGNORECASE,
        ),
        applicable_forms=_FPI_20F_FORMS,
        high_signal_gleaning=True,
        gleaning_match_pattern=re.compile(r"\bitem\s*4\s*[-–—.]+\s*information\s+on\s+the\s+company\b", re.IGNORECASE),
    ),
)

_ITEM_1B_EXCLUSION_RE = re.compile(r"\bitem\s*1b\b", re.IGNORECASE)

_FORM_ALIASES: Dict[str, str] = {
    "10K": SECFormType.FORM_10K.value,
    "10K/A": SECFormType.FORM_10KA.value,
    "10Q": SECFormType.FORM_10Q.value,
    "10Q/A": SECFormType.FORM_10QA.value,
    "8K": SECFormType.FORM_8K.value,
    "8K/A": SECFormType.FORM_8KA.value,
    "20F": SECFormType.FORM_20F.value,
    "20F/A": SECFormType.FORM_20FA.value,
    "40F": SECFormType.FORM_40F.value,
    "6K": SECFormType.FORM_6K.value,
    "FORM4": SECFormType.FORM_4.value,
    "10-12B": SECFormType.FORM_10.value,
    "10-12G": SECFormType.FORM_10.value,
    "DEF14A": SECFormType.FORM_DEF14A.value,
    "SC13D": SECFormType.FORM_SC_13D.value,
    "SC13G": SECFormType.FORM_SC_13G.value,
    "13F": SECFormType.FORM_13F_HR.value,
    "NT10-K": SECFormType.FORM_NT_10K.value,
    "NT10-Q": SECFormType.FORM_NT_10Q.value,
}

_FORM_VALUE_MAP: Dict[str, SECFormType] = {ft.value: ft for ft in SECFormType}


# ==============================================================================
# 5. CANONICAL HELPER FUNCTIONS
# ==============================================================================

def normalize_form_type(raw: Optional[str], *, default: str = "") -> str:
    """Normalize raw EDGAR form strings (e.g., 'SEC Form 8-K', 'form4', '10k') to canonical codes."""
    if raw is None:
        return default
    cleaned = re.sub(r"\s+", " ", str(raw).strip().upper())
    if cleaned.startswith("SEC "):
        cleaned = cleaned[4:].strip()
    if cleaned.startswith("FORM "):
        cleaned = cleaned[5:].strip()
    if not cleaned:
        return default
    return _FORM_ALIASES.get(cleaned, cleaned)


def get_form_spec(form_type: Optional[str]) -> Optional[SECFormSpec]:
    """Look up the canonical SECFormSpec for a form type string or enum, if registered."""
    if isinstance(form_type, SECFormType):
        return SEC_FORM_CANON.get(form_type)
    normalized = normalize_form_type(form_type)
    enum_val = _FORM_VALUE_MAP.get(normalized)
    return SEC_FORM_CANON.get(enum_val) if enum_val is not None else None


def get_default_harvest_forms() -> Tuple[str, ...]:
    """Return canonical form codes enabled for default Domain-1 batch harvesting."""
    return tuple(
        spec.form_type.value
        for spec in SEC_FORM_CANON.values()
        if spec.status == SECFormStatus.ACTIVE and spec.default_batch_harvest
    )


def get_forms_by_status(status: SECFormStatus) -> Tuple[SECFormType, ...]:
    """Return all canonical SECFormType entries carrying the requested lifecycle status."""
    return tuple(spec.form_type for spec in SEC_FORM_CANON.values() if spec.status == status)


def get_forms_by_layer(layer: SECGraphLayer) -> Tuple[SECFormType, ...]:
    """Return all canonical SECFormType entries assigned to a given Knowledge Graph layer."""
    return tuple(spec.form_type for spec in SEC_FORM_CANON.values() if spec.graph_layer == layer)


def is_termination_eligible_form(form_type: Optional[str]) -> bool:
    """Return True iff `form_type` is legally permitted to close edge `valid_to` (ADR-00033 Invariant #8)."""
    spec = get_form_spec(form_type)
    return bool(spec and spec.allow_contract_termination)


def get_section_patterns(form_type: Optional[str] = None) -> List[Tuple[str, Pattern[str]]]:
    """Return compiled `(canonical_name, header_pattern)` tuples for FinancialSectionSegmenter."""
    if form_type is None:
        return [(spec.canonical_name, spec.header_pattern) for spec in SEC_SECTION_CANON]
    normalized = normalize_form_type(form_type)
    target = _FORM_VALUE_MAP.get(normalized)
    if target is None:
        return [(spec.canonical_name, spec.header_pattern) for spec in SEC_SECTION_CANON]
    return [
        (spec.canonical_name, spec.header_pattern)
        for spec in SEC_SECTION_CANON
        if target in spec.applicable_forms
    ]


def is_high_signal_section(
    section_name: Optional[str],
    *,
    gics_sector: Optional[str] = None,
) -> bool:
    """Return True when `section_name` qualifies for targeted Pass-2 gleaning (ADR-00035 / ADR-00045 / ADR-00046)."""
    if not section_name:
        return False
    raw = str(section_name).strip()
    if _ITEM_1B_EXCLUSION_RE.search(raw):
        return False
    norm_sector = str(gics_sector).strip().lower() if gics_sector else ""
    for spec in SEC_SECTION_CANON:
        sector_allowed = bool(
            norm_sector
            and any(sec.strip().lower() == norm_sector for sec in spec.sector_gleaning_only)
        )
        if not spec.high_signal_gleaning and not sector_allowed:
            continue
        if raw.lower() == spec.canonical_name.lower():
            return True
        if spec.gleaning_match_pattern is not None and spec.gleaning_match_pattern.search(raw):
            return True
    return False


def get_sec_provenance_confidence_map() -> Dict[str, float]:
    """Return canonical SEC provenance-key -> base-confidence weights (parity-checked with ontology/validation.py)."""
    mapping: Dict[str, float] = {}
    for spec in SEC_FORM_CANON.values():
        if spec.provenance_key:
            mapping[spec.provenance_key] = spec.base_confidence
    for ex_spec in SEC_EXHIBIT_CANON.values():
        if ex_spec.provenance_key:
            mapping[ex_spec.provenance_key] = ex_spec.base_confidence
    return mapping


__all__ = [
    "SECFormCategory",
    "SECGraphLayer",
    "SECIngestionMode",
    "SECFormStatus",
    "SECFormType",
    "SECFormSpec",
    "SECExhibitSpec",
    "SECSectionSpec",
    "SEC_FORM_CANON",
    "SEC_EXHIBIT_CANON",
    "SECForm8KItem",
    "SEC_ITEM_DESCRIPTIONS",
    "SEC_8K_PRIORITY_TIERS",
    "SEC_SECTION_CANON",
    "normalize_form_type",
    "get_form_spec",
    "get_default_harvest_forms",
    "get_forms_by_status",
    "get_forms_by_layer",
    "is_termination_eligible_form",
    "get_section_patterns",
    "is_high_signal_section",
    "get_sec_provenance_confidence_map",
]
