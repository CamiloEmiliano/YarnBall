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
    MAX_PLAUSIBLE_XBRL_DIO_DAYS,
    PHYSICAL_INVENTORY_GICS_SECTORS,
    ParameterProvenance,
    extract_cascade_parameters_from_text,
    is_physical_inventory_gics_sector,
    resolve_edge_cascade_parameters,
)

__all__ = [
    "CascadeNode",
    "CascadeEdge",
    "CascadeGraph",
    "ConvergentShockSummary",
    "SparseNetworkCascadeEngine",
    "MAX_PLAUSIBLE_XBRL_DIO_DAYS",
    "PHYSICAL_INVENTORY_GICS_SECTORS",
    "ParameterProvenance",
    "extract_cascade_parameters_from_text",
    "is_physical_inventory_gics_sector",
    "resolve_edge_cascade_parameters",
]

