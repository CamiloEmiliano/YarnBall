"""
News Text Parser, HTML Sanitizer, and Publisher Filter.

Provides:
- Publisher disclaimer & tail stock recommendation stripping (`strip_publisher_boilerplates`)
- Paywall and stub detection (`is_paywall_or_stub`)
- Domain extraction and deterministic source hash calculation (`compute_source_hash`)
- Dynamic domain blacklisting and cooldown handling
- Trafilatura / regex HTML text extraction
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

try:
    import trafilatura
except ImportError:
    trafilatura = None

try:
    from tools.init_db import _connect_db
except ImportError:
    _connect_db = None

logger = logging.getLogger("ingestion.news.parser")

PAYWALL_PHRASES = [
    "subscribe to continue reading",
    "this article is exclusive to subscribers",
    "sign in for full access",
    "subscribe for unlimited access",
    "already a subscriber",
    "you have reached your limit of free articles",
    "to read the full story",
    "subscriber-only content",
    "join now to read",
]

DISCLAIMER_BOILERPLATE_PATTERNS = [
    re.compile(r"(?:the\s+)?motley\s+fool\s+(?:has\s+positions\s+in|owns\s+shares\s+of|recommends|transacts).*", re.IGNORECASE | re.DOTALL),
    re.compile(r"disclosure:\s*(?:the\s+author|the\s+motley\s+fool|seeking\s+alpha|zacks|investorplace|fool\.com).*", re.IGNORECASE | re.DOTALL),
    re.compile(r"disclaimer:\s*(?:past\s+performance\s+is\s+no\s+guarantee|the\s+opinions\s+expressed\s+herein|this\s+article\s+represents\s+the\s+opinion).*", re.IGNORECASE | re.DOTALL),
    re.compile(r"seeking\s+alpha\s+contributor.*", re.IGNORECASE | re.DOTALL),
    re.compile(r"zacks\s+investment\s+research\s+disclaimer.*", re.IGNORECASE | re.DOTALL),
    re.compile(r"the\s+views\s+and\s+opinions\s+expressed\s+herein\s+are\s+the\s+views\s+and\s+opinions\s+of\s+the\s+author.*", re.IGNORECASE | re.DOTALL),
]


def get_domain_from_url(url: str) -> str:
    """Extract clean lower-case domain from URL."""
    if not url:
        return ""
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    if ":" in netloc:
        netloc = netloc.split(":")[0]
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def compute_source_hash(url: str, raw_payload: Optional[str] = None) -> str:
    """Compute deterministic SHA-256 hash for deduplication."""
    identifier = url.strip() if url else (raw_payload or "")
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:32]


def is_paywall_or_stub(text: Optional[str], min_word_count: int = 40) -> bool:
    """Check if extracted text is a paywall barrier or truncated stub."""
    if not text:
        return True
    words = text.split()
    if len(words) < min_word_count:
        return True
    lower_text = text.lower()
    for phrase in PAYWALL_PHRASES:
        if phrase in lower_text:
            return True
    return False


def strip_publisher_boilerplates(text: str) -> str:
    """Strip editorial disclosures and publisher stock recommendations from article tail."""
    if not text:
        return ""
    cleaned = text
    for pattern in DISCLAIMER_BOILERPLATE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned.strip()


def extract_text_from_html(html: str) -> Optional[str]:
    """Extract clean body text from HTML using Trafilatura or regex fallback with boilerplate stripping."""
    if not html:
        return None
    raw_text = None
    if trafilatura is not None:
        extracted = trafilatura.extract(
            html,
            output_format="txt",
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
        if extracted:
            raw_text = extracted.strip()

    if not raw_text:
        clean = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r"<[^>]+>", " ", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        raw_text = clean if clean else None

    if raw_text:
        cleaned_text = strip_publisher_boilerplates(raw_text)
        return cleaned_text if cleaned_text else None
    return None


def is_domain_blacklisted(domain: str, conn=None, cooldown_minutes: int = 60) -> bool:
    """Check whether a domain is currently blacklisted in PostgreSQL domain_status with cool-down support."""
    if not domain:
        return False
    close_after = False
    if conn is None:
        if _connect_db is None:
            return False
        try:
            conn = _connect_db(os.getenv("FINANCIAL_RAG_DB", "financial_rag"))
            close_after = True
        except Exception as exc:
            logger.warning(f"DB connection failed while checking domain {domain}: {exc}")
            return False

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT status, updated_at, 
                       (now() > updated_at + ( %s || ' minutes')::interval) AS cooldown_expired
                FROM domain_status 
                WHERE domain = %s;
                """,
                (str(cooldown_minutes), domain),
            )
            row = cur.fetchone()
            if row:
                status = row[0]
                cooldown_expired = row[2] if len(row) > 2 else False
                if status == "blacklisted":
                    if cooldown_expired:
                        logger.info(f"Domain {domain} blacklist cooldown expired; allowing half-open probe.")
                        return False
                    return True
            return False
    except Exception as exc:
        logger.warning(f"Error checking domain status for {domain}: {exc}")
        return False
    finally:
        if close_after and conn:
            conn.close()


def record_domain_success(domain: str, conn=None) -> None:
    """Reset consecutive failures to 0 on successful scrape."""
    if not domain or _connect_db is None:
        return
    close_after = False
    if conn is None:
        try:
            conn = _connect_db(os.getenv("FINANCIAL_RAG_DB", "financial_rag"))
            close_after = True
        except Exception:
            return

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO domain_status (domain, consecutive_failures, status, updated_at)
                VALUES (%s, 0, 'allowed', now())
                ON CONFLICT (domain) DO UPDATE
                SET consecutive_failures = 0, status = 'allowed', updated_at = now();
                """,
                (domain,),
            )
        conn.commit()
    except Exception:
        pass
    finally:
        if close_after and conn:
            conn.close()


def record_domain_failure(domain: str, failure_threshold: int = 3, conn=None) -> None:
    """Increment domain failure count and blacklist if threshold exceeded."""
    if not domain or _connect_db is None:
        return
    close_after = False
    if conn is None:
        try:
            conn = _connect_db(os.getenv("FINANCIAL_RAG_DB", "financial_rag"))
            close_after = True
        except Exception:
            return

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO domain_status (domain, consecutive_failures, status, updated_at)
                VALUES (%s, 1, 'allowed', now())
                ON CONFLICT (domain) DO UPDATE
                SET consecutive_failures = domain_status.consecutive_failures + 1,
                    status = CASE 
                        WHEN domain_status.consecutive_failures + 1 >= %s THEN 'blacklisted' 
                        ELSE 'allowed' 
                    END,
                    updated_at = now();
                """,
                (domain, failure_threshold),
            )
        conn.commit()
    except Exception as exc:
        logger.warning(f"Failed to record domain failure for {domain}: {exc}")
    finally:
        if close_after and conn:
            conn.close()
