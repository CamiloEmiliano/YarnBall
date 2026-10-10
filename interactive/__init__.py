"""
Interactive Presentation & Graph Visualization Package.

Submodules:
- `app`: Chainlit conversational interface with PyVis draggable subgraph rendering
"""

from .app import NODE_COLORS, generate_subgraph_html

__all__ = [
    "NODE_COLORS",
    "generate_subgraph_html",
]
