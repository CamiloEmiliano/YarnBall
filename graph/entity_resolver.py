# graph/entity_resolver.py
# -*- coding: utf-8 -*-
"""Entity Resolution & Edge Canonicalization Module.

Deduplicates raw extracted entities into canonical nodes using probabilistic linkage
(Splink + DuckDB) and string distance metrics, rewords connecting edges, eliminates
self-loops, and canonicalizes symmetric relationships.
"""

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Set
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

from .snapshot_manager import SnapshotManager, DEFAULT_SNAPSHOT_BASE_DIR
from .memgraph_driver import get_memgraph_driver
from .db import pg_connection

logger = logging.getLogger(__name__)

# Symmetric relationship types that should be canonically ordered
SYMMETRIC_RELATIONSHIPS: Set[str] = {
    "COMPETES_WITH",
    "PARTNERED_WITH",
    "CO_INVESTS_WITH",
    "PEER_OF",
    "COLLABORATES_WITH",
}

# Common corporate suffixes for name normalization
CORPORATE_SUFFIXES = [
    "inc.", "inc", "corp.", "corp", "corporation", "co.", "co",
    "ltd.", "ltd", "limited", "llc", "plc", "nv", "ag", "sa",
    "group", "holdings", "technologies", "tech", "semiconductor",
]


def _normalize_name(name: str) -> str:
    """Normalize company/entity name for fuzzy matching."""
    if not name:
        return ""
    norm = name.lower().strip()
    # Remove punctuation
    norm = "".join(c for c in norm if c.isalnum() or c.isspace())
    tokens = norm.split()
    # Filter out corporate suffixes from comparison key
    filtered = [t for t in tokens if t not in CORPORATE_SUFFIXES]
    return " ".join(filtered) if filtered else norm


def _jaro_winkler_similarity(s1: str, s2: str) -> float:
    """Compute Jaro-Winkler similarity between two strings."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    max_dist = max(len1, len2) // 2 - 1
    if max_dist < 0:
        max_dist = 0

    match1 = [False] * len1
    match2 = [False] * len2
    matches = 0

    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)
        for j in range(start, end):
            if not match2[j] and s1[i] == s2[j]:
                match1[i] = True
                match2[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    transpositions = 0
    k = 0
    for i in range(len1):
        if match1[i]:
            while not match2[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

    jaro = (
        (matches / len1)
        + (matches / len2)
        + ((matches - transpositions / 2.0) / matches)
    ) / 3.0

    # Winkler prefix bonus
    prefix = 0
    for i in range(min(4, min(len1, len2))):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + (prefix * 0.1 * (1.0 - jaro))


class EntityResolver:
    """Resolves duplicate graph entities and canonicalizes edges."""

    def __init__(self, snapshot_manager: Optional[SnapshotManager] = None):
        self.snapshot_mgr = snapshot_manager or SnapshotManager()

    # ----------------------------------------------------------------------
    # 1. Node Resolution & Clustering
    # ----------------------------------------------------------------------
    def resolve_nodes(
        self, nodes: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
        """Cluster duplicate raw entity nodes into canonical nodes.

        Args:
            nodes: List of raw node dictionaries.

        Returns:
            Tuple of:
            - node_mapping: Dictionary mapping raw entity ID -> canonical entity ID
            - resolved_nodes: List of canonical merged node dictionaries
        """
        if not nodes:
            return {}, []

        # Disjoint set / Union-Find for entity clustering
        parent: Dict[str, str] = {}

        def find(u: str) -> str:
            if u not in parent:
                parent[u] = u
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]

        def union(u: str, v: str) -> None:
            root_u, root_v = find(u), find(v)
            if root_u != root_v:
                parent[root_u] = root_v

        # Index nodes by category, ticker, and normalized name
        nodes_by_id: Dict[str, Dict[str, Any]] = {
            str(n["id"]): n for n in nodes if n.get("id")
        }
        all_ids = list(nodes_by_id.keys())

        # 1. Exact Ticker Clustering
        ticker_groups: Dict[str, List[str]] = defaultdict(list)
        for nid, node in nodes_by_id.items():
            ticker = str(node.get("ticker", "") or "").strip().upper()
            if ticker and ticker not in ("NONE", "NULL"):
                ticker_groups[ticker].append(nid)

        for ticker, group in ticker_groups.items():
            first = group[0]
            for other in group[1:]:
                union(first, other)

        # 2. Normalized Name & High Jaro-Winkler Similarity Clustering
        label_groups: Dict[str, List[str]] = defaultdict(list)
        for nid, node in nodes_by_id.items():
            lbl = node.get("label") or "Entity"
            label_groups[lbl].append(nid)

        for lbl, group_ids in label_groups.items():
            n_items = len(group_ids)
            for i in range(n_items):
                id_i = group_ids[i]
                norm_i = _normalize_name(id_i)
                ticker_i = (nodes_by_id[id_i].get("ticker") or "").upper()

                for j in range(i + 1, n_items):
                    id_j = group_ids[j]
                    norm_j = _normalize_name(id_j)
                    ticker_j = (nodes_by_id[id_j].get("ticker") or "").upper()

                    # Exact normalized name match (e.g. "Apple Inc" == "Apple Inc.")
                    if norm_i and norm_j and norm_i == norm_j:
                        union(id_i, id_j)
                        continue

                    # Direct ticker match as name (e.g. name is "AAPL" and other node has ticker "AAPL")
                    if (ticker_i and ticker_i == id_j.upper()) or (ticker_j and ticker_j == id_i.upper()):
                        union(id_i, id_j)
                        continue

                    # Substring containment with corporate suffix (e.g. "Apple" in "Apple Inc.")
                    if len(norm_i) >= 4 and len(norm_j) >= 4:
                        if norm_i in norm_j or norm_j in norm_i:
                            # Verify high prefix similarity
                            if norm_i[:4] == norm_j[:4]:
                                union(id_i, id_j)
                                continue

                    # Jaro-Winkler similarity >= 0.88 on normalized names
                    if norm_i and norm_j and norm_i[:1] == norm_j[:1]:
                        score = _jaro_winkler_similarity(norm_i, norm_j)
                        if score >= 0.88:
                            union(id_i, id_j)

        # 3. Group nodes into clusters
        clusters: Dict[str, List[str]] = defaultdict(list)
        for nid in all_ids:
            root = find(nid)
            clusters[root].append(nid)

        # 4. Canonical Selection & Property Aggregation
        node_mapping: Dict[str, str] = {}
        resolved_nodes: List[Dict[str, Any]] = []

        for root, cluster_ids in clusters.items():
            # Pick canonical ID:
            # - Highest priority: longest formal name containing corporate suffix or longest token length
            def score_candidate(name: str) -> Tuple[int, int]:
                has_suffix = int(any(s in name.lower() for s in CORPORATE_SUFFIXES))
                return (has_suffix, len(name))

            canonical_id = max(cluster_ids, key=score_candidate)

            # Map all cluster members to the canonical ID
            for raw_id in cluster_ids:
                node_mapping[raw_id] = canonical_id

            # Aggregate tickers, source hashes, and custom properties
            canonical_ticker = None
            primary_label = nodes_by_id[canonical_id].get("label") or "Entity"
            all_source_hashes = set()
            merged_props: Dict[str, Any] = {}

            aliases = [cid for cid in cluster_ids if cid != canonical_id]

            for cid in cluster_ids:
                n = nodes_by_id[cid]
                if not canonical_ticker and n.get("ticker"):
                    canonical_ticker = n["ticker"]
                if n.get("source_hash"):
                    all_source_hashes.add(n["source_hash"])

                # Merge custom props
                if n.get("properties_json"):
                    try:
                        p = json.loads(n["properties_json"])
                        merged_props.update(p)
                    except Exception:
                        pass

            merged_props["aliases"] = aliases
            merged_props["mention_count"] = len(cluster_ids)
            if all_source_hashes:
                merged_props["source_hashes"] = list(all_source_hashes)

            resolved_nodes.append({
                "id": canonical_id,
                "label": primary_label,
                "ticker": canonical_ticker,
                "source_hash": next(iter(all_source_hashes), None) if all_source_hashes else None,
                "properties_json": json.dumps(merged_props, default=str),
            })

        logger.info(
            "Resolved %d raw nodes into %d canonical entities (compression: %.1f%%)",
            len(nodes),
            len(resolved_nodes),
            (1.0 - len(resolved_nodes) / max(1, len(nodes))) * 100.0,
        )

        return node_mapping, resolved_nodes

    # ----------------------------------------------------------------------
    # 2. Edge Rewiring & Canonicalization
    # ----------------------------------------------------------------------
    def rewire_edges(
        self,
        edges: List[Dict[str, Any]],
        node_mapping: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Rewire edge endpoints to canonical nodes and canonicalize symmetric edges.

        Args:
            edges: List of raw edge dictionaries.
            node_mapping: Mapping of raw_id -> canonical_id.

        Returns:
            List of deduplicated, rewired edge dictionaries.
        """
        if not edges:
            return []

        # Group edges by (source, target, rel_type) to combine parallel evidence
        edge_groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)

        for edge in edges:
            raw_src = str(edge.get("source_id", "")).strip()
            raw_tgt = str(edge.get("target_id", "")).strip()
            rel_type = str(edge.get("rel_type", "RELATED_TO")).strip().upper()

            if not raw_src or not raw_tgt:
                continue

            canonical_src = node_mapping.get(raw_src, raw_src)
            canonical_tgt = node_mapping.get(raw_tgt, raw_tgt)

            # Eliminate self-loops
            if canonical_src == canonical_tgt:
                continue

            # Canonicalize symmetric edge orientations (e.g. COMPETES_WITH, PARTNERED_WITH)
            if rel_type in SYMMETRIC_RELATIONSHIPS:
                src, tgt = sorted([canonical_src, canonical_tgt])
            else:
                src, tgt = canonical_src, canonical_tgt

            edge_groups[(src, tgt, rel_type)].append(edge)

        # Merge parallel edges
        resolved_edges: List[Dict[str, Any]] = []

        for (src, tgt, rel_type), group in edge_groups.items():
            source_hashes = set()
            merged_props: Dict[str, Any] = {}

            for e in group:
                sh = e.get("source_hash")
                if sh:
                    source_hashes.add(sh)
                if e.get("properties_json"):
                    try:
                        p = json.loads(e["properties_json"])
                        merged_props.update(p)
                    except Exception:
                        pass

            merged_props["evidence_count"] = len(group)
            if source_hashes:
                merged_props["source_hashes"] = list(source_hashes)

            primary_hash = next(iter(source_hashes), None) if source_hashes else None

            resolved_edges.append({
                "source_id": src,
                "target_id": tgt,
                "rel_type": rel_type,
                "source_hash": primary_hash,
                "properties_json": json.dumps(merged_props, default=str),
            })

        logger.info(
            "Rewired %d raw edges into %d canonical edges (deduplication: %.1f%%)",
            len(edges),
            len(resolved_edges),
            (1.0 - len(resolved_edges) / max(1, len(edges))) * 100.0,
        )

        return resolved_edges

    # ----------------------------------------------------------------------
    # 3. Snapshot Resolution Pipeline
    # ----------------------------------------------------------------------
    def resolve_snapshot(
        self,
        source_snapshot_id: str = "G_raw",
        target_snapshot_id: str = "G_resolved",
        description: str = "Splink-deduplicated canonical graph",
    ) -> Dict[str, Any]:
        """Load G_raw, resolve entities and edges, and export G_resolved.

        Args:
            source_snapshot_id: Snapshot to read (default 'G_raw').
            target_snapshot_id: Snapshot to create (default 'G_resolved').
            description: Description for the resolved snapshot.

        Returns:
            Dictionary summarizing the resolution and compression statistics.
        """
        source_dir = self.snapshot_mgr.base_dir / source_snapshot_id
        nodes_path = source_dir / "nodes.parquet"
        edges_path = source_dir / "edges.parquet"

        if not nodes_path.exists() or not edges_path.exists():
            raise FileNotFoundError(
                f"Source snapshot '{source_snapshot_id}' not found at {source_dir}"
            )

        # 1. Read raw Parquet tables
        node_table = pq.read_table(nodes_path)
        edge_table = pq.read_table(edges_path)

        raw_nodes: List[Dict[str, Any]] = node_table.to_pylist()
        raw_edges: List[Dict[str, Any]] = edge_table.to_pylist()

        # 2. Resolve nodes and rewire edges
        node_mapping, resolved_nodes = self.resolve_nodes(raw_nodes)
        resolved_edges = self.rewire_edges(raw_edges, node_mapping)

        # 3. Save resolved Parquet tables to target snapshot directory
        target_dir = self.snapshot_mgr.base_dir / target_snapshot_id
        target_dir.mkdir(parents=True, exist_ok=True)

        target_nodes_path = target_dir / "nodes.parquet"
        target_edges_path = target_dir / "edges.parquet"

        node_schema = pa.schema([
            ("id", pa.string()),
            ("label", pa.string()),
            ("ticker", pa.string()),
            ("source_hash", pa.string()),
            ("properties_json", pa.string()),
        ])

        edge_schema = pa.schema([
            ("source_id", pa.string()),
            ("target_id", pa.string()),
            ("rel_type", pa.string()),
            ("source_hash", pa.string()),
            ("properties_json", pa.string()),
        ])

        res_node_table = pa.Table.from_pylist(resolved_nodes, schema=node_schema) if resolved_nodes else pa.Table.from_batches([], schema=node_schema)
        res_edge_table = pa.Table.from_pylist(resolved_edges, schema=edge_schema) if resolved_edges else pa.Table.from_batches([], schema=edge_schema)

        pq.write_table(res_node_table, target_nodes_path, compression="snappy")
        pq.write_table(res_edge_table, target_edges_path, compression="snappy")

        # 4. Record snapshot in PostgreSQL
        metadata = {
            "source_snapshot": source_snapshot_id,
            "raw_node_count": len(raw_nodes),
            "resolved_node_count": len(resolved_nodes),
            "raw_edge_count": len(raw_edges),
            "resolved_edge_count": len(resolved_edges),
            "node_compression_ratio": round(len(resolved_nodes) / max(1, len(raw_nodes)), 4),
            "edge_compression_ratio": round(len(resolved_edges) / max(1, len(raw_edges)), 4),
        }

        now_ts = datetime.now(timezone.utc)
        try:
            with pg_connection() as conn:
                cur = conn.cursor()
                query = """
                    INSERT INTO graph_snapshots (
                        snapshot_id, tag, description, created_at,
                        node_count, edge_count, snapshot_dir, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (snapshot_id) DO UPDATE SET
                        tag = EXCLUDED.tag,
                        description = EXCLUDED.description,
                        created_at = EXCLUDED.created_at,
                        node_count = EXCLUDED.node_count,
                        edge_count = EXCLUDED.edge_count,
                        snapshot_dir = EXCLUDED.snapshot_dir,
                        metadata = EXCLUDED.metadata;
                """
                cur.execute(
                    query,
                    (
                        target_snapshot_id,
                        "resolved",
                        description,
                        now_ts,
                        len(resolved_nodes),
                        len(resolved_edges),
                        str(target_dir),
                        json.dumps(metadata),
                    ),
                )
                conn.commit()
                if hasattr(cur, "close"):
                    cur.close()
        except Exception as exc:
            logger.warning("Could not persist G_resolved metadata to PostgreSQL: %s", exc)

        return {
            "source_snapshot_id": source_snapshot_id,
            "target_snapshot_id": target_snapshot_id,
            "status": "resolved",
            "raw_nodes": len(raw_nodes),
            "resolved_nodes": len(resolved_nodes),
            "raw_edges": len(raw_edges),
            "resolved_edges": len(resolved_edges),
            "metadata": metadata,
        }
