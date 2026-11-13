"""SEC Form 8-K Computational Reading & Contagion Dry Run Tool (ADR-00009).

Ingests authentic SEC Form 8-K disclosure payloads, extracts disruption events,
propagates multi-tier shock contagion via SparseNetworkCascadeEngine, and emits
the standardized OperationalContagionReport established in ADR-00011.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from contagion import (
    CascadeEdge,
    CascadeGraph,
    ConvergentShockSummary,
    SparseNetworkCascadeEngine,
)
from retrieval.guardrails.schema_guardrail import CypherASTWhitelistGuardrail
from retrieval.pipeline import (
    AnalystQuery,
    GraphRAGPipelineOrchestrator,
    HopOrderedSubgraphVerbalizer,
    PipelineResult,
)
from retrieval.reporting import (
    ContagionSeverityVerdict,
    ExecutionTelemetry,
    ModelExecutionMode,
    NetworkNodeImpact,
    OperationalContagionReport,
    OperationalDisruptionCategory,
    SECForm8KItem,
    TopologicalEdgeDescriptor,
    build_operational_contagion_report,
    flatten_report_to_timeseries_records,
    format_operational_contagion_report,
    validate_verbatim_citations,
)
from retrieval.temporal import TemporalQueryFilter
from sft_pipeline.serving.model_client import ModelResponse, YarnBallModelClient

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s", "level":"%(levelname)s", "msg":"%(message)s"}'
)
logger = logging.getLogger("dry_run_sec_8k")


class OfflineGraphStore:
    """In-memory property graph store providing Cypher MATCH/MERGE semantics for offline dry runs."""

    def __init__(self, edges: Optional[List[Dict[str, Any]]] = None):
        self.edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        if edges:
            for e in edges:
                src = str(e.get("source", "UNKNOWN")).upper()
                tgt = str(e.get("target", "UNKNOWN")).upper()
                rel = str(e.get("rel_type", "SUPPLIES_TO"))
                self.edges[(src, rel, tgt)] = e

    def session(self):
        store = self

        class _Session:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def run(self, query: str, **kwargs):
                if "MATCH" in query:
                    records: List[Dict[str, Any]] = []
                    for (src, rel, tgt), props in store.edges.items():
                        records.append({
                            "source": src,
                            "target": tgt,
                            "rel_type": rel,
                            "status": props.get("status", "ACTIVE_CURRENT"),
                            "share": props.get("share", 0.20),
                            "exposure_pct": props.get("exposure_pct", 20.0),
                            "buffer_days": props.get("buffer_days", 45.0),
                            "nature": props.get("nature", "default"),
                            "hop": props.get("hop", 1),
                        })
                    return records
                return []

        return _Session()


def _infer_disruption_category(headline: str, content: str) -> OperationalDisruptionCategory:
    """Classifies regulatory text into an OperationalDisruptionCategory."""
    text = (headline + " " + content).lower()
    if "fire" in text or "explosion" in text:
        return OperationalDisruptionCategory.FACILITY_FIRE_OR_EXPLOSION
    elif "flood" in text or "earthquake" in text or "force majeure" in text:
        return OperationalDisruptionCategory.NATURAL_DISASTER_FORCE_MAJEURE
    elif "power" in text or "grid" in text or "utility" in text:
        return OperationalDisruptionCategory.UTILITY_POWER_GRID_FAILURE
    elif "equipment" in text or "breakdown" in text or "shutdown" in text or "halt" in text:
        return OperationalDisruptionCategory.EQUIPMENT_BREAKDOWN_FAB_HALT
    elif "terminate" in text or "severance" in text:
        return OperationalDisruptionCategory.PRIMARY_SUPPLY_CONTRACT_TERMINATED
    elif "cyber" in text or "ransomware" in text:
        return OperationalDisruptionCategory.RANSOMWARE_CYBER_INFRASTRUCTURE_HALT
    elif "port" in text or "canal" in text or "logistics" in text:
        return OperationalDisruptionCategory.MARITIME_CANAL_PORT_LOGISTICS_CHOKEPOINT
    elif "recall" in text or "safety hold" in text:
        return OperationalDisruptionCategory.REGULATORY_SAFETY_RECALL_OR_FDA_HOLD
    else:
        return OperationalDisruptionCategory.OTHER_UNCLASSIFIED_MATERIAL_HALT


def _extract_sec_item(items: List[str]) -> str:
    """Extracts the primary statutory SEC item string (e.g. Item 8.01)."""
    for it in items:
        for enum_item in SECForm8KItem:
            if enum_item.value.lower() in it.lower():
                return enum_item.value
    return "Item 8.01"


class DeterministicMockModelClient(YarnBallModelClient):
    """Deterministic offline model client for `--mock` dry runs without a live GPU daemon (ADR-00013)."""

    def __init__(self, shock_origin: str = "CRUS"):
        super().__init__(transport_fn=lambda payload: {"response": ""})
        self.shock_origin = shock_origin

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        task = str(kwargs.get("task", "CONTAGION_REASONING")).upper()
        if task in ("TEXT_TO_CYPHER", "C") or "openCypher" in prompt:
            cypher = f"MATCH (s:Company {{ticker: '{self.shock_origin}'}})-[r:SUPPLIES_TO*1..3]->(t:Company) RETURN s, r, t;"
            return ModelResponse(
                raw_text=cypher,
                final_output=cypher,
                task="TEXT_TO_CYPHER",
                latency_sec=0.005,
                prefill_ms=0.5,
                tokens_per_sec=200.0,
                prompt_tokens=20,
                generated_tokens=16,
            )

        # Parse pre-computed deterministic metrics from HopOrderedSubgraphVerbalizer context
        target_m = re.search(r"-\s*Target Entity:\s*([A-Za-z0-9_.\-]+)", prompt)
        loss_m = re.search(r"\((\d+(?:\.\d+)?)%\s*capacity loss\)", prompt)
        down_m = re.search(r"-\s*Unbuffered Downtime:\s*(\d+(?:\.\d+)?)\s*calendar days", prompt)
        onset_m = re.search(r"-\s*Disruption Onset:\s*([^\n(]+)", prompt)

        tgt = target_m.group(1) if target_m else "TARGET"
        pct = float(loss_m.group(1)) if loss_m else 0.0
        downtime = float(down_m.group(1)) if down_m else 0.0
        onset = onset_m.group(1).strip() if onset_m else "Day 0"

        if downtime > 0.0:
            verdict_str = "UNBUFFERED_DISRUPTION"
            final_output = (
                f"{tgt} experiences {verdict_str} resulting from a {pct:.1f}% peak capacity "
                f"contraction with {downtime:.1f} days of unbuffered downtime (onset: {onset})."
            )
        else:
            verdict_str = "BUFFER_ABSORBED"
            final_output = (
                f"{tgt} is classified as {verdict_str} with {pct:.1f}% capacity contraction and "
                f"{downtime:.1f} days of unbuffered downtime; inventory safety stocks absorb transmission."
            )

        think_trace = (
            f"Deterministic mock evaluation of shock originating at {self.shock_origin}. "
            f"Primary downstream node {tgt}: peak_loss={pct:.1f}%, unbuffered_downtime={downtime:.1f}d, "
            f"onset={onset}, verdict={verdict_str}."
        )
        return ModelResponse(
            raw_text=f"<think>\n{think_trace}\n</think>\n{final_output}",
            think_trace=think_trace,
            final_output=final_output,
            task="CONTAGION_REASONING",
            latency_sec=0.01,
            prefill_ms=1.0,
            tokens_per_sec=150.0,
            prompt_tokens=60,
            generated_tokens=45,
        )


def run_sec_8k_dry_run(
    filing_path: Union[str, Path],
    model_client: Optional[YarnBallModelClient] = None,
    graph_driver: Optional[Any] = None,
    custom_network_edges: Optional[List[Dict[str, Any]]] = None,
    target_ticker: Optional[str] = None,
    offline_isolation: bool = True,
    use_mock: bool = False,
) -> Tuple[OperationalContagionReport, PipelineResult]:
    """Executes end-to-end computational reading and contagion propagation on an SEC Form 8-K payload.

    Args:
        filing_path: Path to SEC Form 8-K JSON filing fixture or payload.
        model_client: Optional model client (YarnBallModelClient or MockClient for testing).
        graph_driver: Optional property graph driver or session interface.
        custom_network_edges: Optional network topology override.
        target_ticker: Primary target company to analyze downstream.
        offline_isolation: Guardrail blocking socket connection if True.

    Returns:
        Tuple of (OperationalContagionReport, PipelineResult).
    """
    path = Path(filing_path)
    if not path.is_file():
        raise FileNotFoundError(f"SEC Form 8-K filing not found at: {path}")

    # 1. Parse Filing Payload
    with open(path, "r", encoding="utf-8") as f:
        filing_data = json.load(f)

    cik = str(filing_data.get("cik", "0000000000"))
    reporting_ticker = str(filing_data.get("ticker", "UNKNOWN")).upper()
    company_name = str(filing_data.get("company_name", "Unknown Corporate Entity"))
    filing_type = str(filing_data.get("form_type", "8-K"))
    filing_date = str(filing_data.get("filing_date", "2024-10-28"))
    metadata = filing_data.get("metadata", {})
    filing_timestamp = filing_data.get("filing_timestamp") or metadata.get("filing_timestamp")
    items = filing_data.get("items", ["Item 8.01 Other Events"])
    sec_item = _extract_sec_item(items)
    headline = str(filing_data.get("headline", ""))
    content = str(filing_data.get("content", ""))

    duration_days = float(metadata.get("disruption_duration_days", 60.0))
    buffer_days = float(metadata.get("inventory_buffer_days", 45.0))
    affected_product = str(metadata.get("affected_product", "audio_codecs"))
    supplier_name = metadata.get("supplier_name")
    supplier_cik = metadata.get("supplier_cik")

    # Determine shock origin ticker
    # If the filing is from a customer reporting a supplier outage (e.g. AAPL reporting FabCo)
    # the shock origin is the supplier. If filing is from the supplier (e.g. CRUS), origin is CRUS.
    if supplier_name and "supplier" in headline.lower():
        shock_origin = str(metadata.get("supplier_ticker", "SUPPLIER_ORIGIN")).upper()
        if shock_origin == "SUPPLIER_ORIGIN" and "fabco" in supplier_name.lower():
            shock_origin = "FABCO"
        disrupted_ticker = shock_origin
        disrupted_name = supplier_name
        disrupted_cik = supplier_cik
        primary_target = reporting_ticker
    else:
        shock_origin = reporting_ticker
        disrupted_ticker = reporting_ticker
        disrupted_name = company_name
        disrupted_cik = cik
        primary_target = target_ticker or "AAPL"

    shock_magnitude_pct = float(metadata.get("shock_magnitude_pct", 100.0))
    disruption_category = _infer_disruption_category(headline, content)

    # 2. Build or Configure Graph Driver
    if graph_driver is None:
        if custom_network_edges:
            edges = custom_network_edges
        elif shock_origin == "CRUS":
            # Canonical Cirrus Logic supply network topology
            edges = [
                {
                    "source": "CRUS",
                    "target": "AAPL",
                    "rel_type": "SUPPLIES_TO",
                    "nature": "audio_codecs",
                    "share": 0.20,
                    "exposure_pct": 20.0,
                    "buffer_days": 45.0,
                    "hop": 1,
                },
                {
                    "source": "CRUS",
                    "target": "HPQ",
                    "rel_type": "SUPPLIES_TO",
                    "nature": "audio_codecs",
                    "share": 0.10,
                    "exposure_pct": 10.0,
                    "buffer_days": 90.0,
                    "hop": 1,
                },
                {
                    "source": "AAPL",
                    "target": "FOXCONN",
                    "rel_type": "SUPPLIES_TO",
                    "nature": "assembly",
                    "share": 0.50,
                    "exposure_pct": 50.0,
                    "buffer_days": 90.0,
                    "hop": 2,
                },
            ]
        elif shock_origin == "FABCO":
            # Canonical FabCo -> AAPL -> Foxconn network topology
            edges = [
                {
                    "source": "FABCO",
                    "target": "AAPL",
                    "rel_type": "SUPPLIES_TO",
                    "nature": "optical_transceivers",
                    "share": 0.35,
                    "exposure_pct": 35.0,
                    "buffer_days": buffer_days,
                    "hop": 1,
                },
                {
                    "source": "AAPL",
                    "target": "FOXCONN",
                    "rel_type": "SUPPLIES_TO",
                    "nature": "assembly",
                    "share": 0.50,
                    "exposure_pct": 50.0,
                    "buffer_days": 90.0,
                    "hop": 2,
                },
            ]
        else:
            edges = [
                {
                    "source": shock_origin,
                    "target": primary_target,
                    "rel_type": "SUPPLIES_TO",
                    "nature": affected_product,
                    "share": 0.25,
                    "exposure_pct": 25.0,
                    "buffer_days": buffer_days,
                    "hop": 1,
                }
            ]
        graph_driver = OfflineGraphStore(edges=edges)

    # 3. Resolve Model Client & Execution Provenance (ADR-00013)
    if model_client is not None:
        active_client = model_client
        execution_mode = (
            ModelExecutionMode.MOCK_DETERMINISTIC
            if isinstance(model_client, DeterministicMockModelClient)
            or "mock" in type(model_client).__name__.lower()
            else ModelExecutionMode.LIVE_LLM
        )
    elif use_mock:
        active_client = DeterministicMockModelClient(shock_origin=shock_origin)
        execution_mode = ModelExecutionMode.MOCK_DETERMINISTIC
    else:
        active_client = YarnBallModelClient()
        is_healthy, health_detail = active_client.check_health(timeout_sec=1.5)
        if not is_healthy:
            logger.warning(
                "Pre-flight health check failed (%s). Run `ollama serve` for live inference or pass `--mock`.",
                health_detail,
            )
            execution_mode = ModelExecutionMode.DEGRADED_UNREACHABLE
        else:
            execution_mode = ModelExecutionMode.LIVE_LLM

    # 4. Configure Pipeline Orchestrator
    orchestrator = GraphRAGPipelineOrchestrator(
        model_client=active_client,
        schema_guardrail=CypherASTWhitelistGuardrail(),
        temporal_filter=TemporalQueryFilter(),
        verbalizer=HopOrderedSubgraphVerbalizer(),
        graph_driver=graph_driver,
        cascade_engine=SparseNetworkCascadeEngine(),
    )

    # 5. Construct Analyst Query
    query_text = (
        f"Assess operational supply-chain contagion resulting from {disrupted_ticker} "
        f"disruption reported in Form 8-K {sec_item} on {filing_date}."
    )
    analyst_query = AnalystQuery(
        query=query_text,
        as_of_date=filing_date,
        target_ticker=primary_target,
        shock_origin=shock_origin,
        shock_magnitude_pct=shock_magnitude_pct,
        initial_shocks={shock_origin: shock_magnitude_pct},
        initial_durations={shock_origin: duration_days},
        initial_onsets={shock_origin: 0.0},
    )

    # 6. Execute End-to-End Orchestrator Flow
    pipeline_result = orchestrator.run(analyst_query)

    # 7. Generate Validated OperationalContagionReport (ADR-00011 / ADR-00013)
    report_id = f"REP-{filing_date.replace('-', '')}-{disrupted_ticker}-001"
    event_id = f"SEC_8K_{filing_date.replace('-', '')}_{disrupted_ticker}"
    event_summary = headline if headline else (content[:200] + "...")

    operational_report = pipeline_result.to_operational_report(
        event_id=event_id,
        sec_item=sec_item,
        disrupted_entity_ticker=disrupted_ticker,
        disrupted_entity_cik=disrupted_cik,
        disrupted_entity_name=disrupted_name,
        filing_date=filing_date,
        disruption_category=disruption_category.value,
        event_summary=event_summary,
        shock_duration_days=duration_days,
        shock_magnitude_pct=shock_magnitude_pct,
        report_id=report_id,
        filing_timestamp=filing_timestamp,
        model_mode=execution_mode.value,
    )

    return operational_report, pipeline_result


def export_report_to_jsonl(
    report: OperationalContagionReport,
    output_path: Union[str, Path],
) -> int:
    """Flattens report into 1NF time-series join rows and appends to JSONL (ADR-00013)."""
    records = flatten_report_to_timeseries_records(report)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")
    return len(records)


def main() -> None:
    """CLI Entry point for SEC Form 8-K Computational Reading Dry Run."""
    parser = argparse.ArgumentParser(
        description="YarnBall SEC Form 8-K Computational Reading & Contagion Dry Run"
    )
    parser.add_argument(
        "--filing",
        type=str,
        default=str(PROJECT_ROOT / "tests" / "fixtures" / "filings" / "aapl_8k_disruption.json"),
        help="Path to SEC Form 8-K JSON disclosure fixture",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Primary downstream target entity ticker (e.g. AAPL)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit raw machine-readable JSON report",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run using deterministic offline mock model client (no GPU/Ollama required)",
    )
    parser.add_argument(
        "--export-jsonl",
        type=str,
        default=None,
        help="Append 1NF flattened ContagionTimeSeriesEventRecord rows to JSONL file (ADR-00013)",
    )
    parser.add_argument(
        "--validate-citations",
        action="store_true",
        help="Validate that model synthesis cites continuous metrics verbatim",
    )

    args = parser.parse_args()

    t0 = time.perf_counter()
    report, _ = run_sec_8k_dry_run(
        filing_path=args.filing,
        target_ticker=args.target,
        use_mock=args.mock,
    )
    elapsed_sec = time.perf_counter() - t0

    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        formatted_output = format_operational_contagion_report(report)
        print(formatted_output)

    if args.export_jsonl:
        row_count = export_report_to_jsonl(report, args.export_jsonl)
        logger.info("Exported %d 1NF time-series join row(s) to %s", row_count, args.export_jsonl)

    if args.validate_citations:
        violations = validate_verbatim_citations(report)
        if violations:
            logger.warning("Verbatim metric citation violations found:")
            for v in violations:
                logger.warning("  * %s", v)
        else:
            logger.info("All continuous cascade metrics cited verbatim in model synthesis.")

    logger.info("Dry run executed successfully in %.2f seconds.", elapsed_sec)


if __name__ == "__main__":
    main()

