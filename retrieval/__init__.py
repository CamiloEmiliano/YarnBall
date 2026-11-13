"""
Deterministic & Symbolic Graph/Context Extraction Package.

Submodules:
- `hybrid`: `HybridRetriever` (combining dense vector search + subgraph expansion + article context)
- `text_to_cql`: `TextToCQL` (schema-aware Cypher translation & guarded execution)
- `enricher`: `enrich_nodes`, `enrich_node` (k-hop ego-network traversals)
- `service`: FastAPI microservice `/search`
"""

from .hybrid import HybridRetriever
from .text_to_cql import TextToCQL
from .enricher import enrich_node, enrich_nodes
from .temporal import TemporalQueryFilter
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
from .shock_propagation import (
    PropagatedShock,
    ShockVector,
    build_shock_vector_from_metrics,
    propagate_shock_across_edge,
    traverse_multi_hop_contagion,
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
    "HybridRetriever",
    "TextToCQL",
    "enrich_node",
    "enrich_nodes",
    "TemporalQueryFilter",
    "CypherSubgraphQuery",
    "InMemorySubgraphStore",
    "parse_cypher_subgraph_query",
    "AnalystQuery",
    "GraphRAGPipelineOrchestrator",
    "HopOrderedSubgraphVerbalizer",
    "PipelineResult",
    "PropagatedShock",
    "ShockVector",
    "build_shock_vector_from_metrics",
    "propagate_shock_across_edge",
    "traverse_multi_hop_contagion",
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
