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

__all__ = [
    "CascadeNode",
    "CascadeEdge",
    "CascadeGraph",
    "ConvergentShockSummary",
    "SparseNetworkCascadeEngine",
]

