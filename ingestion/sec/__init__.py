"""
SEC EDGAR Regulatory Disclosures Ingestion Subpackage.

Modules:
- `client`: `EdgarClient`, `DEFAULT_SP500_BENCHMARK`, filing downloader
- `parser`: `EdgarEntityLinker`, `normalize_company_name`, section extractor, LLM prompts
- `harvester`: `HistoricalSECHarvester`, `EdgarWorker`, batch queue pipeline
"""

from .client import (
    DEFAULT_SP500_BENCHMARK,
    EDGAR_AVAILABLE,
    EdgarClient,
    download_filings_for_ticker,
)
from .parser import (
    CORP_SUFFIXES,
    EdgarEntityLinker,
    SEC_EXTRACTION_PROMPT,
    extract_sec_relationships,
    normalize_company_name,
    select_salient_sec_context,
    sync_sec_companies_from_sec,
)
from .harvester import (
    EdgarWorker,
    HistoricalSECHarvester,
    run_sec_ingestion,
)

__all__ = [
    # Client
    "EdgarClient",
    "DEFAULT_SP500_BENCHMARK",
    "EDGAR_AVAILABLE",
    "download_filings_for_ticker",
    # Parser
    "EdgarEntityLinker",
    "normalize_company_name",
    "CORP_SUFFIXES",
    "SEC_EXTRACTION_PROMPT",
    "select_salient_sec_context",
    "extract_sec_relationships",
    "sync_sec_companies_from_sec",
    # Harvester
    "HistoricalSECHarvester",
    "EdgarWorker",
    "run_sec_ingestion",
]
