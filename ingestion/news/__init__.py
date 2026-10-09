"""
Financial News & Material Disclosures Ingestion Subpackage.

Modules:
- `client`: `FinnhubClient`, `TokenBucket`, news endpoints
- `parser`: `strip_publisher_boilerplates`, `compute_source_hash`, domain blacklist
- `harvester`: `MultiSourceNewsHarvester`, `HistoricalNewsIngestor`, `ScrapingWorker`
"""

from .client import (
    FINNHUB_TICKERS,
    TokenBucket,
    check_api,
    fetch_finnhub,
)
from .parser import (
    compute_source_hash,
    extract_text_from_html,
    get_domain_from_url,
    is_domain_blacklisted,
    is_paywall_or_stub,
    record_domain_failure,
    record_domain_success,
    strip_publisher_boilerplates,
)
from .harvester import (
    HistoricalNewsIngestor,
    MultiSourceNewsHarvester,
    fetch_and_extract,
    insert_extracted_article,
    process_scraping_message,
    run_worker,
)

__all__ = [
    # Client
    "TokenBucket",
    "fetch_finnhub",
    "check_api",
    "FINNHUB_TICKERS",
    # Parser
    "strip_publisher_boilerplates",
    "compute_source_hash",
    "get_domain_from_url",
    "is_paywall_or_stub",
    "extract_text_from_html",
    "is_domain_blacklisted",
    "record_domain_success",
    "record_domain_failure",
    # Harvester
    "MultiSourceNewsHarvester",
    "HistoricalNewsIngestor",
    "fetch_and_extract",
    "insert_extracted_article",
    "process_scraping_message",
    "run_worker",
]
