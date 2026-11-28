"""
SEC EDGAR Regulatory Disclosures Ingestion Subpackage.

Modules:
- `canon`: Zero-dependency canonical SEC EDGAR form, exhibit, 8-K item & section registry (ADR-00045)
- `client`: `EdgarClient`, `DEFAULT_SP500_BENCHMARK`, operational Exhibit 10 filter & filing downloader
- `xbrl`: Offline US-GAAP XBRL `companyfacts` DIO & ASC 280 concentration priors store (ADR-00048)
- `parser`: `EdgarEntityLinker`, `normalize_company_name`, section extractor, LLM prompts
- `harvester`: `HistoricalSECHarvester`, `EdgarWorker`, batch queue pipeline
- `disruption_gating`: Form 8-K boilerplate sanitization & disruption gating (ADR-00014)
"""

from .canon import (
    SEC_8K_PRIORITY_TIERS,
    SEC_EXHIBIT_CANON,
    SEC_FORM_CANON,
    SEC_ITEM_DESCRIPTIONS,
    SEC_SECTION_CANON,
    SECExhibitSpec,
    SECForm8KItem,
    SECFormCategory,
    SECFormSpec,
    SECFormStatus,
    SECFormType,
    SECGraphLayer,
    SECIngestionMode,
    SECSectionSpec,
    get_default_harvest_forms,
    get_form_spec,
    get_forms_by_layer,
    get_forms_by_status,
    get_sec_provenance_confidence_map,
    get_section_patterns,
    is_high_signal_section,
    is_termination_eligible_form,
    normalize_form_type,
)
from .client import (
    DEFAULT_SP500_BENCHMARK,
    EDGAR_AVAILABLE,
    EdgarClient,
    classify_exhibit_10_attachment,
    download_filings_for_ticker,
    extract_operational_contract_clauses,
)
from .xbrl import (
    PHYSICAL_INVENTORY_GICS_SECTORS,
    XBRLConcentrationRecord,
    XBRLDIOPriorRecord,
    XBRLPriorRecord,
    XBRLPriorStore,
    extract_asc280_concentration_from_companyfacts,
    extract_xbrl_dio_priors_from_companyfacts,
    fetch_companyfacts_json,
    is_physical_inventory_sector,
    lookup_pit_xbrl_dio,
)
from .parser import (
    CORP_SUFFIXES,
    EdgarEntityLinker,
    SEC_EXTRACTION_PROMPT,
    extract_sec_relationships,
    is_high_signal_gleaning_section,
    normalize_company_name,
    rank_salient_chunks,
    score_chunk_salience,
    select_salient_sec_context,
    should_glean_chunk,
    sync_sec_companies_from_sec,
)
from .harvester import (
    EdgarWorker,
    HistoricalSECHarvester,
    run_sec_ingestion,
)
from .disruption_gating import (
    SanitizedDisruptionAssessment,
    assess_sec_8k_disruption,
    extract_8k_exhibit_texts,
    get_sec_8k_item_tier,
    normalize_staged_8k_payload,
    sanitize_sec_8k_text,
    select_primary_sec_8k_item,
)

__all__ = [
    # Canon (ADR-00045)
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
    # Client
    "EdgarClient",
    "DEFAULT_SP500_BENCHMARK",
    "EDGAR_AVAILABLE",
    "classify_exhibit_10_attachment",
    "download_filings_for_ticker",
    "extract_operational_contract_clauses",
    # XBRL Priors (ADR-00048)
    "PHYSICAL_INVENTORY_GICS_SECTORS",
    "XBRLConcentrationRecord",
    "XBRLDIOPriorRecord",
    "XBRLPriorRecord",
    "XBRLPriorStore",
    "extract_asc280_concentration_from_companyfacts",
    "extract_xbrl_dio_priors_from_companyfacts",
    "fetch_companyfacts_json",
    "is_physical_inventory_sector",
    "lookup_pit_xbrl_dio",
    # Parser
    "EdgarEntityLinker",
    "normalize_company_name",
    "CORP_SUFFIXES",
    "SEC_EXTRACTION_PROMPT",
    "score_chunk_salience",
    "rank_salient_chunks",
    "is_high_signal_gleaning_section",
    "should_glean_chunk",
    "select_salient_sec_context",
    "extract_sec_relationships",
    "sync_sec_companies_from_sec",
    # Harvester
    "HistoricalSECHarvester",
    "EdgarWorker",
    "run_sec_ingestion",
    # Disruption Gating (ADR-00014)
    "SanitizedDisruptionAssessment",
    "assess_sec_8k_disruption",
    "extract_8k_exhibit_texts",
    "get_sec_8k_item_tier",
    "normalize_staged_8k_payload",
    "sanitize_sec_8k_text",
    "select_primary_sec_8k_item",
]
