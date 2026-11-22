"""Contagion & Network Cascade Package.

Implements the Sparse Network Cascade Engine for multi-hop shock propagation,
grouped-Leontief input aggregation, and inventory buffering (ADR-00006).
"""

from contagion.cascade import (
    CascadeEdge,
    CascadeGraph,
    CascadeNode,
    ConvergentShockSummary,
    SparseNetworkCascadeEngine,
)
from contagion.parameters import (
    ParameterProvenance,
    extract_cascade_parameters_from_text,
    resolve_edge_cascade_parameters,
)

__all__ = [
    "CascadeNode",
    "CascadeEdge",
    "CascadeGraph",
    "ConvergentShockSummary",
    "SparseNetworkCascadeEngine",
    "ParameterProvenance",
    "extract_cascade_parameters_from_text",
    "resolve_edge_cascade_parameters",
]

