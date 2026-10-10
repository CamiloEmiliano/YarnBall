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

__all__ = [
    "HybridRetriever",
    "TextToCQL",
    "enrich_node",
    "enrich_nodes",
]
