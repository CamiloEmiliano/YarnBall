# knowledge_graph/__init__.py
"""Knowledge Graph persistence, Memgraph driver management, and entity resolution subsystem."""

from knowledge_graph.db import pg_connection
from knowledge_graph.memgraph_driver import get_memgraph_driver
from knowledge_graph.snapshot_manager import SnapshotManager
from knowledge_graph.entity_resolver import (
    EntityResolver,
    is_blacklisted_publisher,
    PUBLISHER_STOPLIST,
    is_clickbait_article_source,
    is_generic_placeholder,
    _jaro_winkler_similarity,
)
from knowledge_graph.embedding_store import store_node_embeddings

__all__ = [
    "pg_connection",
    "get_memgraph_driver",
    "SnapshotManager",
    "EntityResolver",
    "is_blacklisted_publisher",
    "PUBLISHER_STOPLIST",
    "is_clickbait_article_source",
    "is_generic_placeholder",
    "_jaro_winkler_similarity",
    "store_node_embeddings",
]
