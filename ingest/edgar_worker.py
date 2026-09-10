"""
SEC EDGAR Ingestion & Graph Extraction Worker.

Orchestrates:
- 10-K / 8-K / Form 4 filing downloads via `EdgarClient`
- Entity resolution via `EdgarEntityLinker` (4-tier cascade)
- LLM relationship extraction on Item 1 (Business) and Item 1A (Risk Factors)
- Exhibit 21 subsidiary persistence to PostgreSQL & Memgraph
- Temporal edge insertion into Memgraph with fiscal timestamps and accession provenance
- Staging queue tracking in `sec_filings_queue`
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from graph.memgraph_driver import get_memgraph_driver
from graph.graph_store import MAX_LABEL_LENGTH, MAX_REL_TYPE_LENGTH, _clean_json_response, OLLAMA_API_BASE, OLLAMA_MODEL_NAME
from .edgar_client import EdgarClient, DEFAULT_SP500_BENCHMARK
from .edgar_linker import EdgarEntityLinker, normalize_company_name

logger = logging.getLogger("edgar_worker")

SEC_EXTRACTION_PROMPT = """You are an expert financial SEC filing knowledge graph extractor.
Analyze the following section from an SEC filing (e.g. Form 10-K Item 1 Business / Item 1A Risks, or Form 8-K).
Extract key corporate entities, suppliers, customers, strategic partners, competitors, key products, and material risks.

Output ONLY a valid JSON object matching this schema:
{
    "nodes": [
        {"id": "<Full Company or Entity Name>", "type": "Company|Product|Technology|RiskFactor|Person|RegulatoryBody", "properties": {"ticker": "<TICKER if known>"}}
    ],
    "edges": [
        {"source": "<Source Entity>", "target": "<Target Entity>", "type": "<RELATIONSHIP_TYPE>", "properties": {"nature": "<brief context>"}}
    ]
}

Allowed relationship types:
- SUPPLIES_TO (source sells/supplies goods/services to target)
- CUSTOMER_OF (source buys goods/services from target)
- COMPETES_WITH (source competes directly with target)
- PARTNERED_WITH (strategic alliance, joint venture, distribution agreement)
- LICENSES_TO / LICENSES_FROM (IP or patent licensing)
- OWNS / CONTROLS (equity stake, joint venture)
- EXPOSED_TO_RISK (company exposed to specific operational/geopolitical/supply risk)
- ACQUIRED / MERGED_WITH (M&A events)

Do not include any explanation or markdown formatting. Output raw JSON only."""


def extract_sec_relationships(text: str) -> Dict[str, Any]:
    """Extract entities and relationships from SEC text chunk using local Ollama LLM."""
    if not text or not text.strip():
        return {"nodes": [], "edges": []}

    try:
        import httpx
    except ImportError:
        logger.warning("httpx not available; skipping LLM extraction")
        return {"nodes": [], "edges": []}

    # Limit text chunk size to fit model context safely (e.g. first 6,000 chars of section)
    chunk = text[:6000]
    prompt = f"{SEC_EXTRACTION_PROMPT}\n\nSEC Text Section:\n{chunk}\n\nJSON:"

    gen_url = OLLAMA_API_BASE
    if "/v1" in gen_url:
        gen_url = gen_url.split("/v1")[0].rstrip("/") + "/api/generate"

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                gen_url,
                json={
                    "model": OLLAMA_MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return _clean_json_response(data.get("response", ""))
    except Exception as e:
        logger.error(f"Failed SEC LLM extraction call: {e}")
        return {"nodes": [], "edges": []}


class EdgarWorker:
    """End-to-End SEC EDGAR Filing Processor."""

    def __init__(self, pg_conn=None, memgraph_driver=None):
        self.pg_conn = pg_conn
        self.driver = memgraph_driver or get_memgraph_driver()
        self.client = EdgarClient()
        self.linker = EdgarEntityLinker(pg_conn=self.pg_conn)

    def process_company_10k(
        self,
        ticker_or_cik: str,
        year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Ingest, extract, and persist a company's 10-K filing into PostgreSQL and Memgraph."""
        logger.info(f"Processing 10-K for {ticker_or_cik} (Year: {year or 'latest'})")

        filing = self.client.fetch_latest_10k(ticker_or_cik, year=year)
        if not filing:
            logger.warning(f"No 10-K filing found for {ticker_or_cik}")
            return {"status": "not_found", "nodes": 0, "edges": 0}

        sections = self.client.extract_10k_sections(filing)
        acc_num = sections.get("accession_number", "UNKNOWN")
        filing_date_str = sections.get("filing_date") or datetime.utcnow().strftime("%Y-%m-%d")
        report_date_str = sections.get("report_date") or filing_date_str

        # Resolve primary company
        primary_entity = self.linker.link_entity(ticker=ticker_or_cik, name=getattr(filing, "company", ticker_or_cik))
        cik = primary_entity["cik"] if primary_entity else getattr(filing, "cik", "")
        primary_name = primary_entity["company_name"] if primary_entity else ticker_or_cik
        ticker = primary_entity["ticker"] if primary_entity else ticker_or_cik

        # Record queue entry in PostgreSQL
        if self.pg_conn:
            self._record_filing_queue(
                accession_number=acc_num,
                cik=cik,
                ticker=ticker,
                form_type="10-K",
                filing_date=filing_date_str,
                report_date=report_date_str,
                items_present=["Item 1", "Item 1A", "Exhibit 21"],
                status="processing",
            )

        extracted_nodes = []
        extracted_edges = []

        # 1. Process Exhibit 21 Subsidiaries
        subs = sections.get("subsidiaries", [])
        if subs:
            sub_nodes, sub_edges = self._persist_subsidiaries(
                parent_cik=cik,
                parent_name=primary_name,
                parent_ticker=ticker,
                subsidiaries=subs,
                accession_number=acc_num,
                filing_date=filing_date_str,
            )
            extracted_nodes.extend(sub_nodes)
            extracted_edges.extend(sub_edges)

        # 2. Extract relationships from Item 1 Business
        item1_text = sections.get("item_1_business", "")
        if item1_text:
            graph_data = extract_sec_relationships(item1_text)
            n, e = self._persist_sec_graph_data(
                primary_name=primary_name,
                primary_ticker=ticker,
                graph_data=graph_data,
                accession_number=acc_num,
                filing_date=filing_date_str,
                form_type="10-K",
                section="Item 1 Business",
            )
            extracted_nodes.extend(n)
            extracted_edges.extend(e)

        # 3. Extract risks from Item 1A Risk Factors
        item1a_text = sections.get("item_1a_risk_factors", "")
        if item1a_text:
            graph_data = extract_sec_relationships(item1a_text)
            n, e = self._persist_sec_graph_data(
                primary_name=primary_name,
                primary_ticker=ticker,
                graph_data=graph_data,
                accession_number=acc_num,
                filing_date=filing_date_str,
                form_type="10-K",
                section="Item 1A Risk Factors",
            )
            extracted_nodes.extend(n)
            extracted_edges.extend(e)

        # Update queue status
        if self.pg_conn:
            self._update_filing_queue(
                accession_number=acc_num,
                status="completed",
                node_count=len(extracted_nodes),
                edge_count=len(extracted_edges),
            )

        logger.info(
            f"Completed 10-K processing for {ticker_or_cik}: {len(extracted_nodes)} nodes, {len(extracted_edges)} edges"
        )
        return {
            "status": "completed",
            "accession_number": acc_num,
            "nodes": len(extracted_nodes),
            "edges": len(extracted_edges),
        }

    def _persist_subsidiaries(
        self,
        parent_cik: str,
        parent_name: str,
        parent_ticker: str,
        subsidiaries: List[Dict[str, str]],
        accession_number: str,
        filing_date: str,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Persist Exhibit 21 subsidiaries to PostgreSQL sec_subsidiaries and Memgraph."""
        nodes = []
        edges = []

        # 1. PostgreSQL insertion
        if self.pg_conn and parent_cik:
            cur = self.pg_conn.cursor()
            try:
                records = [
                    (
                        parent_cik,
                        s.get("name", ""),
                        normalize_company_name(s.get("name", "")),
                        s.get("jurisdiction", "Unknown"),
                        accession_number,
                    )
                    for s in subsidiaries
                    if s.get("name")
                ]
                if records:
                    cur.executemany(
                        """
                        INSERT INTO sec_subsidiaries (parent_cik, subsidiary_name, normalized_name, jurisdiction, source_filing_acc)
                        VALUES (%s, %s, %s, %s, %s);
                        """,
                        records,
                    )
                    self.pg_conn.commit()
            except Exception as e:
                logger.warning(f"Error saving subsidiaries to PostgreSQL: {e}")
            finally:
                cur.close()

        # 2. Memgraph edge insertion
        fiscal_year = int(filing_date[:4]) if filing_date and len(filing_date) >= 4 else datetime.utcnow().year
        with self.driver.session() as session:
            # Ensure parent node
            session.run(
                "MERGE (p:Company {id: $id}) SET p.ticker = $ticker",
                id=parent_name,
                ticker=parent_ticker,
            )
            nodes.append(parent_name)

            for sub in subsidiaries:
                sub_name = sub.get("name", "").strip()
                if not sub_name:
                    continue

                session.run(
                    "MERGE (s:Company {id: $id}) SET s.jurisdiction = $jurisdiction",
                    id=sub_name,
                    jurisdiction=sub.get("jurisdiction", "Unknown"),
                )
                nodes.append(sub_name)

                session.run(
                    """
                    MATCH (p:Company {id: $parent_id})
                    MATCH (s:Company {id: $sub_id})
                    MERGE (s)-[r:SUBSIDIARY_OF]->(p)
                    SET r.fiscal_year = $fiscal_year,
                        r.valid_from = $valid_from,
                        r.is_current = true,
                        r.form_type = '10-K',
                        r.accession_number = $acc_num,
                        r.jurisdiction = $jurisdiction
                    """,
                    parent_id=parent_name,
                    sub_id=sub_name,
                    fiscal_year=fiscal_year,
                    valid_from=filing_date,
                    acc_num=accession_number,
                    jurisdiction=sub.get("jurisdiction", "Unknown"),
                )
                edges.append({"source": sub_name, "target": parent_name, "type": "SUBSIDIARY_OF"})

        return nodes, edges

    def _persist_sec_graph_data(
        self,
        primary_name: str,
        primary_ticker: str,
        graph_data: Dict[str, Any],
        accession_number: str,
        filing_date: str,
        form_type: str,
        section: str,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Link entities and persist structured relationships into Memgraph with temporal metadata."""
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        if not nodes and not edges:
            return [], []

        fiscal_year = int(filing_date[:4]) if filing_date and len(filing_date) >= 4 else datetime.utcnow().year

        inserted_nodes = []
        inserted_edges = []

        with self.driver.session() as session:
            # Ensure primary entity exists
            session.run(
                "MERGE (p:Company {id: $id}) SET p.ticker = $ticker",
                id=primary_name,
                ticker=primary_ticker,
            )
            inserted_nodes.append(primary_name)

            # Insert / Merge extracted nodes with Entity Linker resolution
            for node in nodes:
                raw_id = str(node.get("id", "")).strip()
                if not raw_id:
                    continue

                raw_type = str(node.get("type", "Company")).strip()
                label = "".join(c for c in raw_type if c.isalnum())[:MAX_LABEL_LENGTH] or "Company"
                props = node.get("properties", {}) or {}

                # Entity Linking resolution
                resolved = self.linker.link_entity(name=raw_id, ticker=props.get("ticker"))
                canonical_id = resolved["company_name"] if resolved else raw_id
                if resolved and resolved.get("ticker"):
                    props["ticker"] = resolved["ticker"]
                if resolved and resolved.get("cik"):
                    props["cik"] = resolved["cik"]

                query = f"MERGE (n:{label} {{id: $id}}) SET n += $props"
                session.run(query, id=canonical_id, props=props)
                inserted_nodes.append(canonical_id)

            # Insert / Merge temporal relationship edges
            for edge in edges:
                source_raw = str(edge.get("source", "")).strip()
                target_raw = str(edge.get("target", "")).strip()
                if not source_raw or not target_raw:
                    continue

                # Resolve source and target
                res_s = self.linker.link_entity(name=source_raw)
                res_t = self.linker.link_entity(name=target_raw)

                source_id = res_s["company_name"] if res_s else source_raw
                target_id = res_t["company_name"] if res_t else target_raw

                raw_type = str(edge.get("type", "RELATED_TO")).strip().upper()
                rel_type = "".join(c for c in raw_type if c.isalnum() or c == "_")[:MAX_REL_TYPE_LENGTH] or "RELATED_TO"
                props = edge.get("properties", {}) or {}

                query = f"""
                MATCH (s {{id: $source}})
                MATCH (t {{id: $target}})
                MERGE (s)-[r:{rel_type}]->(t)
                SET r += $props,
                    r.fiscal_year = $fiscal_year,
                    r.valid_from = $valid_from,
                    r.is_current = true,
                    r.form_type = $form_type,
                    r.section = $section,
                    r.accession_number = $acc_num
                """
                session.run(
                    query,
                    source=source_id,
                    target=target_id,
                    props=props,
                    fiscal_year=fiscal_year,
                    valid_from=filing_date,
                    form_type=form_type,
                    section=section,
                    acc_num=accession_number,
                )
                inserted_edges.append({"source": source_id, "target": target_id, "type": rel_type})

        return inserted_nodes, inserted_edges

    def _record_filing_queue(
        self,
        accession_number: str,
        cik: str,
        ticker: str,
        form_type: str,
        filing_date: str,
        report_date: str,
        items_present: List[str],
        status: str = "pending",
    ):
        """Insert or update filing record in PostgreSQL sec_filings_queue."""
        if not self.pg_conn:
            return
        cur = self.pg_conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO sec_filings_queue (
                    accession_number, cik, ticker, form_type, filing_date, report_date,
                    items_present, status, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (accession_number) DO UPDATE SET
                    status = EXCLUDED.status;
                """,
                (
                    accession_number,
                    cik or None,
                    ticker or None,
                    form_type,
                    filing_date or None,
                    report_date or None,
                    json.dumps(items_present),
                    status,
                ),
            )
            self.pg_conn.commit()
        except Exception as e:
            logger.warning(f"Failed to record sec_filings_queue: {e}")
        finally:
            cur.close()

    def _update_filing_queue(
        self,
        accession_number: str,
        status: str,
        node_count: int,
        edge_count: int,
    ):
        """Update processed status in PostgreSQL sec_filings_queue."""
        if not self.pg_conn:
            return
        cur = self.pg_conn.cursor()
        try:
            cur.execute(
                """
                UPDATE sec_filings_queue
                SET status = %s,
                    extracted_nodes = %s,
                    extracted_edges = %s,
                    processed_at = now()
                WHERE accession_number = %s;
                """,
                (status, node_count, edge_count, accession_number),
            )
            self.pg_conn.commit()
        except Exception as e:
            logger.warning(f"Failed to update sec_filings_queue: {e}")
        finally:
            cur.close()
