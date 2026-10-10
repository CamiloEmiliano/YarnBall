"""
Multi-Year SEC Historical Batch Harvester & Graph Worker.

Provides:
- `HistoricalSECHarvester`: Multi-threaded batch harvesting across S&P 500 constituents (Form 10-K, 10-Q, 8-K, Form 4)
- `EdgarWorker`: Ingestion pipeline extracting entities, subsidiaries, and risk edges into PostgreSQL & Memgraph
- CLI entry points for historical bulk downloading and pipeline execution
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from knowledge_graph.memgraph_driver import get_memgraph_driver
from knowledge_graph.graph_store import MAX_LABEL_LENGTH, MAX_REL_TYPE_LENGTH
from universe import SP500Constituent, SP500UniverseManager
from .client import EdgarClient, DEFAULT_SP500_BENCHMARK
from .parser import (
    EdgarEntityLinker,
    extract_sec_relationships,
    normalize_company_name,
    select_salient_sec_context,
    sync_sec_companies_from_sec,
)

logger = logging.getLogger("ingestion.sec.harvester")

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sec_historical"


class HistoricalSECHarvester:
    """Batch harvester for historical S&P 500 SEC filings with rate limiting and section extraction."""

    def __init__(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        client: Optional[EdgarClient] = None,
        rate_limit_delay_sec: float = 0.1,  # 10 req/s compliance
    ):
        self.output_dir = Path(output_dir)
        self.client = client or EdgarClient()
        self.universe_manager = SP500UniverseManager()
        self.rate_limit_delay = rate_limit_delay_sec

    def harvest_ticker_history(
        self,
        ticker: str,
        start_year: int = 2018,
        end_year: int = 2025,
        include_8k: bool = True,
        include_10q: bool = False,
        include_form4: bool = False,
    ) -> Dict[str, Any]:
        """Harvest full historical filing trajectory for a single constituent ticker."""
        clean_ticker = ticker.strip().upper()
        ticker_dir = self.output_dir / clean_ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "ticker": clean_ticker,
            "years_processed": [],
            "10k_count": 0,
            "8k_count": 0,
            "10q_count": 0,
            "form4_count": 0,
            "subsidiaries_captured": 0,
        }

        company = self.client.get_company(clean_ticker)
        if not company:
            logger.warning(f"Could not resolve SEC company for {clean_ticker}")
            return summary

        for yr in range(start_year, end_year + 1):
            time.sleep(self.rate_limit_delay)
            filing_10k = self.client.fetch_latest_10k(clean_ticker, year=yr)
            if filing_10k:
                sections = self.client.extract_10k_sections(filing_10k)
                acc = sections.get("accession_number", "UNKNOWN").replace("-", "")
                yr_dir = ticker_dir / f"{yr}_10K_{acc}"
                yr_dir.mkdir(parents=True, exist_ok=True)

                with open(yr_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump({
                        "ticker": clean_ticker,
                        "fiscal_year": yr,
                        "accession_number": sections.get("accession_number"),
                        "filing_date": sections.get("filing_date"),
                        "report_date": sections.get("report_date"),
                        "subsidiaries_count": len(sections.get("subsidiaries", [])),
                    }, f, indent=2)

                if sections.get("item_1_business"):
                    with open(yr_dir / "item_1_business.txt", "w", encoding="utf-8") as f:
                        f.write(sections["item_1_business"])

                if sections.get("item_1a_risk_factors"):
                    with open(yr_dir / "item_1a_risk_factors.txt", "w", encoding="utf-8") as f:
                        f.write(sections["item_1a_risk_factors"])

                if sections.get("subsidiaries"):
                    with open(yr_dir / "exhibit_21_subsidiaries.json", "w", encoding="utf-8") as f:
                        json.dump(sections["subsidiaries"], f, indent=2)

                summary["10k_count"] += 1
                summary["subsidiaries_captured"] += len(sections.get("subsidiaries", []))
                summary["years_processed"].append(yr)

            if include_8k:
                time.sleep(self.rate_limit_delay)
                eight_ks = self.client.fetch_recent_8k(clean_ticker, limit=10, year=yr)
                if eight_ks:
                    eight_dir = ticker_dir / f"{yr}_8K"
                    eight_dir.mkdir(parents=True, exist_ok=True)
                    for item in eight_ks:
                        acc = item.get("accession_number", "unknown").replace("-", "")
                        with open(eight_dir / f"{acc}.json", "w", encoding="utf-8") as f:
                            json.dump(item, f, indent=2)
                    summary["8k_count"] += len(eight_ks)

        return summary

    def harvest_company_year(
        self,
        constituent: SP500Constituent,
        year: int,
        forms: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Harvest all requested filings for a specific company and fiscal year."""
        forms_to_fetch = [f.upper() for f in (forms or ["10-K", "8-K", "10-Q", "4"])]
        ticker = constituent.ticker.upper()
        company_year_dir = self.output_dir / ticker / str(year)
        company_year_dir.mkdir(parents=True, exist_ok=True)

        stats: Dict[str, Any] = {
            "ticker": ticker,
            "cik": constituent.cik,
            "company_name": constituent.company_name,
            "gics_sector": constituent.gics_sector,
            "year": year,
            "10k_downloaded": False,
            "10q_count": 0,
            "8k_count": 0,
            "form4_count": 0,
            "subsidiaries_count": 0,
            "filings_saved": [],
        }

        # 1. Form 10-K
        if "10-K" in forms_to_fetch:
            time.sleep(self.rate_limit_delay)
            filing_10k = self.client.fetch_latest_10k(ticker, year=year)
            if filing_10k:
                sections = self.client.extract_10k_sections(filing_10k)
                acc_num = sections.get("accession_number", "UNKNOWN")
                acc_dir = company_year_dir / f"10K_{acc_num}"
                acc_dir.mkdir(parents=True, exist_ok=True)

                # Save 10-K metadata
                meta = {
                    "ticker": ticker,
                    "cik": constituent.cik,
                    "company_name": constituent.company_name,
                    "gics_sector": constituent.gics_sector,
                    "form": "10-K",
                    "year": year,
                    "accession_number": acc_num,
                    "filing_date": sections.get("filing_date"),
                    "report_date": sections.get("report_date"),
                    "subsidiaries_count": len(sections.get("subsidiaries", [])),
                }
                with open(acc_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

                # Save Item 1 (Business)
                item1 = sections.get("item_1_business", "")
                if item1:
                    with open(acc_dir / "item_1_business.txt", "w", encoding="utf-8") as f:
                        f.write(item1)

                # Save Item 1A (Risk Factors)
                item1a = sections.get("item_1a_risk_factors", "")
                if item1a:
                    with open(acc_dir / "item_1a_risk_factors.txt", "w", encoding="utf-8") as f:
                        f.write(item1a)

                # Save Exhibit 21 (Subsidiaries)
                subs = sections.get("subsidiaries", [])
                if subs:
                    with open(acc_dir / "subsidiaries.json", "w", encoding="utf-8") as f:
                        json.dump(subs, f, indent=2)

                stats["10k_downloaded"] = True
                stats["subsidiaries_count"] = len(subs)
                stats["filings_saved"].append(f"10K_{acc_num}")

        # 2. Form 10-Q
        if "10-Q" in forms_to_fetch:
            time.sleep(self.rate_limit_delay)
            filings_10q = self.client.fetch_10q_filings(ticker, year=year, limit=4)
            for q_filing in filings_10q:
                acc_num = q_filing.get("accession_number", "UNKNOWN")
                acc_dir = company_year_dir / f"10Q_{acc_num}"
                acc_dir.mkdir(parents=True, exist_ok=True)

                with open(acc_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(q_filing, f, indent=2)

                q_text = q_filing.get("text", "")
                if q_text:
                    with open(acc_dir / "quarterly_text.txt", "w", encoding="utf-8") as f:
                        f.write(q_text)

                stats["10q_count"] += 1
                stats["filings_saved"].append(f"10Q_{acc_num}")

        # 3. Form 8-K
        if "8-K" in forms_to_fetch:
            time.sleep(self.rate_limit_delay)
            filings_8k = self.client.fetch_recent_8k(ticker, limit=5, year=year)
            for k_filing in filings_8k:
                acc_num = k_filing.get("accession_number", "UNKNOWN")
                acc_dir = company_year_dir / f"8K_{acc_num}"
                acc_dir.mkdir(parents=True, exist_ok=True)

                with open(acc_dir / "form8k_events.json", "w", encoding="utf-8") as f:
                    json.dump(k_filing, f, indent=2)

                stats["8k_count"] += 1
                stats["filings_saved"].append(f"8K_{acc_num}")

        # 4. Form 4
        if "4" in forms_to_fetch or "FORM4" in forms_to_fetch:
            time.sleep(self.rate_limit_delay)
            filings_4 = self.client.fetch_form4_transactions(ticker, limit=10, year=year)
            if filings_4:
                acc_dir = company_year_dir / "Form4_Transactions"
                acc_dir.mkdir(parents=True, exist_ok=True)
                with open(acc_dir / "form4_transactions.json", "w", encoding="utf-8") as f:
                    json.dump(filings_4, f, indent=2)
                stats["form4_count"] = len(filings_4)
                stats["filings_saved"].append("Form4_Transactions")

        # Write top-level company year summary
        with open(company_year_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        return stats

    def harvest_year(
        self,
        year: int,
        tickers: Optional[List[str]] = None,
        forms: Optional[List[str]] = None,
        max_companies: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Harvest filings for all point-in-time S&P 500 constituents in a given year."""
        logger.info(f"=== Starting S&P 500 SEC Historical Harvest for Year {year} ===")
        constituent_map: Dict[str, SP500Constituent] = {}
        for anchor_date in [f"{year}-01-01", f"{year}-06-30", f"{year}-12-31"]:
            for c in self.universe_manager.get_constituents_at_date(anchor_date):
                constituent_map[c.ticker.upper()] = c
        constituents = list(constituent_map.values())

        if tickers:
            ticker_set = {t.upper() for t in tickers}
            constituents = [c for c in constituents if c.ticker.upper() in ticker_set]

        if max_companies:
            constituents = constituents[:max_companies]

        logger.info(f"Targeting {len(constituents)} unique constituents active at any point in year {year}")

        results = []
        for i, c in enumerate(constituents, 1):
            logger.info(f"[{i}/{len(constituents)}] Harvesting {c.ticker} ({c.company_name}) [{c.gics_sector}] for {year}...")
            res = self.harvest_company_year(c, year, forms=forms)
            results.append(res)

        return results

    def harvest_multi_year(
        self,
        years: List[int],
        tickers: Optional[List[str]] = None,
        forms: Optional[List[str]] = None,
        max_companies: Optional[int] = None,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Sweep multiple fiscal years across historical S&P 500 constituents."""
        all_year_results = {}
        for yr in sorted(years):
            res = self.harvest_year(yr, tickers=tickers, forms=forms, max_companies=max_companies)
            all_year_results[yr] = res
        return all_year_results

    def register_dvc(self) -> None:
        """Track harvested historical SEC filings with DVC."""
        try:
            logger.info("Registering data/sec_historical with DVC...")
            project_root = Path(__file__).resolve().parent.parent.parent
            subprocess.run(
                [sys.executable, "-m", "dvc", "add", str(self.output_dir)],
                cwd=str(project_root),
                check=True,
                capture_output=True,
            )
            logger.info("Successfully tracked data/sec_historical with DVC")
        except Exception as e:
            logger.warning(f"DVC registration skipped or failed: {e}")


class EdgarWorker:
    """End-to-End SEC EDGAR Filing Processor and Memgraph / PostgreSQL loader."""

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

        primary_entity = self.linker.link_entity(ticker=ticker_or_cik, name=getattr(filing, "company", ticker_or_cik))
        cik = primary_entity["cik"] if primary_entity else getattr(filing, "cik", "")
        primary_name = primary_entity["company_name"] if primary_entity else ticker_or_cik
        ticker = primary_entity["ticker"] if primary_entity else ticker_or_cik

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

        fiscal_year = int(filing_date[:4]) if filing_date and len(filing_date) >= 4 else datetime.utcnow().year
        with self.driver.session() as session:
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
            session.run(
                "MERGE (p:Company {id: $id}) SET p.ticker = $ticker",
                id=primary_name,
                ticker=primary_ticker,
            )
            inserted_nodes.append(primary_name)

            for node in nodes:
                raw_id = str(node.get("id", "")).strip()
                if not raw_id:
                    continue

                raw_type = str(node.get("type", "Company")).strip()
                label = "".join(c for c in raw_type if c.isalnum())[:MAX_LABEL_LENGTH] or "Company"
                props = node.get("properties", {}) or {}

                resolved = self.linker.link_entity(name=raw_id, ticker=props.get("ticker"))
                canonical_id = resolved["company_name"] if resolved else raw_id
                if resolved and resolved.get("ticker"):
                    props["ticker"] = resolved["ticker"]
                if resolved and resolved.get("cik"):
                    props["cik"] = resolved["cik"]

                query = f"MERGE (n:{label} {{id: $id}}) SET n += $props"
                session.run(query, id=canonical_id, props=props)
                inserted_nodes.append(canonical_id)

            for edge in edges:
                source_raw = str(edge.get("source", "")).strip()
                target_raw = str(edge.get("target", "")).strip()
                if not source_raw or not target_raw:
                    continue

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


def run_sec_ingestion(
    tickers: List[str],
    sync_cik: bool = False,
    pg_conn=None,
    memgraph_driver=None,
) -> Dict[str, Any]:
    """Run end-to-end SEC ingestion for provided tickers."""
    if sync_cik and pg_conn:
        sync_sec_companies_from_sec(pg_conn=pg_conn, sp500_tickers=tickers)

    worker = EdgarWorker(pg_conn=pg_conn, memgraph_driver=memgraph_driver)
    results = {}
    for ticker in tickers:
        res = worker.process_company_10k(ticker)
        results[ticker] = res
    return results


def main():
    parser = argparse.ArgumentParser(description="SEC EDGAR Batch Ingestion & Harvesting CLI.")
    parser.add_argument(
        "--tickers",
        type=str,
        default="AAPL,MSFT,NVDA",
        help="Comma-separated tickers to process",
    )
    parser.add_argument(
        "--sync-cik",
        action="store_true",
        help="Sync master CIK company registry from SEC.gov into PostgreSQL",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2018,
        help="Start fiscal year for historical harvesting (default: 2018)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="End fiscal year for historical harvesting (default: 2025)",
    )
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    harvester = HistoricalSECHarvester()
    for t in tickers:
        logger.info(f"Harvesting historical trajectory for {t} ({args.start_year}-{args.end_year})...")
        summary = harvester.harvest_ticker_history(t, start_year=args.start_year, end_year=args.end_year)
        logger.info(f"Summary for {t}: {summary}")


if __name__ == "__main__":
    main()
