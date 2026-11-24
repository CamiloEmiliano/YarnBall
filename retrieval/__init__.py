"""
Deterministic & Symbolic Graph/Context Extraction Package (ADR-00033 / ADR-00034).

Submodules:
- `facade`: `query_bi`, `query_subgraph_ego`, `SubgraphStorePort`, `MemgraphSubgraphStore`
- `hybrid`: `HybridSeeder`, `SeedHit` (PiT dense entity & passage seeding + Postgres lineage)
- `text_to_cql`: `TextToCQL` (schema-aware Cypher translation routed through `SubgraphStorePort`)
- `enricher`: `enrich_nodes`, `enrich_node` (node-detail ego display helper)
- `service`: FastAPI microservice (`/search`, `/report`, `/health`)
"""

from .facade import (
    MemgraphSubgraphStore,
    SubgraphStorePort,
    query_bi,
    query_subgraph_ego,
)
from .hybrid import HybridSeeder, SeedHit
from .text_to_cql import TextToCQL
from .enricher import enrich_node, enrich_nodes
from .temporal import (
    PiTContext,
    TemporalQueryFilter,
    actionable_session,
    inject_active_temporal_filter,
    is_us_equity_trading_day,
    next_us_equity_trading_day,
    session_close_utc,
    us_equity_regular_close_et,
    validate_hops,
)
from .offline_store import (
    CypherSubgraphQuery,
    InMemorySubgraphStore,
    parse_cypher_subgraph_query,
)
from .pipeline import (
    AnalystQuery,
    GraphRAGPipelineOrchestrator,
    HopOrderedSubgraphVerbalizer,
    PipelineResult,
)
from .reporting import (
    ContagionSeverityVerdict,
    ContagionTimeSeriesEventRecord,
    DependencyDomain,
    ExecutionTelemetry,
    ModelExecutionMode,
    NetworkNodeImpact,
    OperationalContagionReport,
    OperationalDisruptionCategory,
    OperationalScope,
    SECForm8KItem,
    TopologicalEdgeDescriptor,
    build_operational_contagion_report,
    classify_contagion_verdict,
    flatten_report_to_timeseries_records,
    format_operational_contagion_report,
    validate_verbatim_citations,
)

__all__ = [
    "MemgraphSubgraphStore",
    "SubgraphStorePort",
    "query_bi",
    "query_subgraph_ego",
    "HybridSeeder",
    "SeedHit",
    "TextToCQL",
    "enrich_node",
    "enrich_nodes",
    "PiTContext",
    "TemporalQueryFilter",
    "actionable_session",
    "inject_active_temporal_filter",
    "is_us_equity_trading_day",
    "next_us_equity_trading_day",
    "session_close_utc",
    "us_equity_regular_close_et",
    "validate_hops",
    "CypherSubgraphQuery",
    "InMemorySubgraphStore",
    "parse_cypher_subgraph_query",
    "AnalystQuery",
    "GraphRAGPipelineOrchestrator",
    "HopOrderedSubgraphVerbalizer",
    "PipelineResult",
    "ContagionSeverityVerdict",
    "ContagionTimeSeriesEventRecord",
    "DependencyDomain",
    "ExecutionTelemetry",
    "ModelExecutionMode",
    "NetworkNodeImpact",
    "OperationalContagionReport",
    "OperationalDisruptionCategory",
    "OperationalScope",
    "SECForm8KItem",
    "TopologicalEdgeDescriptor",
    "build_operational_contagion_report",
    "classify_contagion_verdict",
    "flatten_report_to_timeseries_records",
    "format_operational_contagion_report",
    "validate_verbatim_citations",
]
