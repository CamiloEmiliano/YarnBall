# graph/snapshot_manager.py
# -*- coding: utf-8 -*-
"""Parquet-Native Graph Snapshot Manager.

Manages exporting and restoring point-in-time knowledge graph snapshots from
Memgraph into compressed Parquet tables (`nodes.parquet`, `edges.parquet`) and
tracks snapshot metadata in the PostgreSQL `graph_snapshots` registry.
"""

import os
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from .memgraph_driver import get_memgraph_driver
from .db import pg_connection

logger = logging.getLogger(__name__)

DEFAULT_SNAPSHOT_BASE_DIR = Path(
    os.getenv("SNAPSHOT_DIR", "data/snapshots")
)


class SnapshotManager:
    """Handles snapshot export, restoration, listing, and metadata tracking."""

    def __init__(self, base_dir: Optional[Path | str] = None):
        self.base_dir = Path(base_dir or DEFAULT_SNAPSHOT_BASE_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    # 1. Export Snapshot
    # ----------------------------------------------------------------------
    def export_snapshot(
        self,
        snapshot_id: str,
        tag: str = "raw",
        description: str = "",
        time_window: Optional[Tuple[datetime | str, datetime | str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Export the active Memgraph graph to Parquet files and record in PostgreSQL.

        Args:
            snapshot_id: Unique identifier for the snapshot (e.g., 'G_raw', 'G_resolved').
            tag: Classification tag ('raw', 'resolved', 'benchmark', etc.).
            description: Human-readable notes or description.
            time_window: Optional tuple of (start_time, end_time).
            metadata: Optional dictionary of additional metadata.

        Returns:
            Dictionary summarizing the exported snapshot.
        """
        snapshot_dir = self.base_dir / snapshot_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        driver = get_memgraph_driver()
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        # 1. Fetch nodes from Memgraph
        try:
            with driver.session() as session:
                node_result = session.run(
                    "MATCH (n) RETURN labels(n) AS labels, properties(n) AS props"
                )
                for record in node_result:
                    labels = record.get("labels", []) or []
                    props = record.get("props", {}) or {}
                    node_id = str(props.get("id", ""))
                    if not node_id:
                        continue

                    # Determine primary label (prefer non-News, non-Entity labels if present)
                    filtered_labels = [lbl for lbl in labels if lbl not in ("Entity",)]
                    primary_label = filtered_labels[0] if filtered_labels else (labels[0] if labels else "Entity")

                    ticker = str(props.get("ticker", "")) if props.get("ticker") else None
                    source_hash = str(props.get("source_hash", "")) if props.get("source_hash") else None

                    # Extract remaining custom properties
                    custom_props = {
                        k: v for k, v in props.items()
                        if k not in ("id", "ticker", "source_hash")
                    }

                    nodes.append({
                        "id": node_id,
                        "label": primary_label,
                        "ticker": ticker,
                        "source_hash": source_hash,
                        "properties_json": json.dumps(custom_props, default=str),
                    })

                # 2. Fetch relationships from Memgraph
                edge_result = session.run(
                    """
                    MATCH (s)-[r]->(t)
                    RETURN s.id AS source_id, t.id AS target_id, type(r) AS rel_type,
                           properties(r) AS props, r.source_hash AS source_hash
                    """
                )
                for record in edge_result:
                    source_id = str(record.get("source_id", "")).strip()
                    target_id = str(record.get("target_id", "")).strip()
                    rel_type = str(record.get("rel_type", "RELATED_TO")).strip()
                    if not source_id or not target_id:
                        continue

                    props = record.get("props", {}) or {}
                    source_hash = str(record.get("source_hash", "") or props.get("source_hash", ""))

                    custom_props = {
                        k: v for k, v in props.items()
                        if k != "source_hash"
                    }

                    edges.append({
                        "source_id": source_id,
                        "target_id": target_id,
                        "rel_type": rel_type,
                        "source_hash": source_hash or None,
                        "properties_json": json.dumps(custom_props, default=str),
                    })
        except Exception as exc:
            logger.warning("Error fetching graph data from Memgraph: %s", exc)

        # 3. Write Parquet files via PyArrow
        nodes_path = snapshot_dir / "nodes.parquet"
        edges_path = snapshot_dir / "edges.parquet"

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

        node_table = pa.Table.from_pylist(nodes, schema=node_schema) if nodes else pa.Table.from_batches([], schema=node_schema)
        edge_table = pa.Table.from_pylist(edges, schema=edge_schema) if edges else pa.Table.from_batches([], schema=edge_schema)

        pq.write_table(node_table, nodes_path, compression="snappy")
        pq.write_table(edge_table, edges_path, compression="snappy")

        # 4. Resolve time window boundaries
        win_start = None
        win_end = None
        if time_window:
            win_start = time_window[0]
            win_end = time_window[1]
        else:
            env_start = os.getenv("HIST_START")
            env_end = os.getenv("HIST_END")
            if env_start and env_end:
                win_start = env_start
                win_end = env_end

        meta_payload = metadata or {}
        now_ts = datetime.now(timezone.utc)

        # 5. Record metadata in PostgreSQL
        try:
            with pg_connection() as conn:
                cur = conn.cursor()
                query = """
                    INSERT INTO graph_snapshots (
                        snapshot_id, tag, description, created_at,
                        time_window_start, time_window_end,
                        node_count, edge_count, snapshot_dir, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (snapshot_id) DO UPDATE SET
                        tag = EXCLUDED.tag,
                        description = EXCLUDED.description,
                        created_at = EXCLUDED.created_at,
                        time_window_start = EXCLUDED.time_window_start,
                        time_window_end = EXCLUDED.time_window_end,
                        node_count = EXCLUDED.node_count,
                        edge_count = EXCLUDED.edge_count,
                        snapshot_dir = EXCLUDED.snapshot_dir,
                        metadata = EXCLUDED.metadata;
                """
                cur.execute(
                    query,
                    (
                        snapshot_id,
                        tag,
                        description,
                        now_ts,
                        win_start,
                        win_end,
                        len(nodes),
                        len(edges),
                        str(snapshot_dir),
                        json.dumps(meta_payload),
                    ),
                )
                conn.commit()
                if hasattr(cur, "close"):
                    cur.close()
        except Exception as exc:
            logger.warning("Could not persist snapshot metadata to PostgreSQL: %s", exc)

        logger.info(
            "Exported snapshot '%s' (%s): %d nodes, %d edges -> %s",
            snapshot_id,
            tag,
            len(nodes),
            len(edges),
            snapshot_dir,
        )

        return {
            "snapshot_id": snapshot_id,
            "tag": tag,
            "description": description,
            "created_at": now_ts.isoformat(),
            "time_window_start": str(win_start) if win_start else None,
            "time_window_end": str(win_end) if win_end else None,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "snapshot_dir": str(snapshot_dir),
            "metadata": meta_payload,
        }

    # ----------------------------------------------------------------------
    # 2. Restore Snapshot
    # ----------------------------------------------------------------------
    def restore_snapshot(
        self,
        snapshot_id: str,
        batch_size: int = 500,
    ) -> Dict[str, Any]:
        """Restore a graph snapshot into Memgraph from Parquet files.

        Clears active Memgraph data and restores vertices/edges via parameterized
        UNWIND batch Cypher statements.

        Args:
            snapshot_id: The ID of the snapshot to restore.
            batch_size: Number of records to insert per Cypher UNWIND transaction.

        Returns:
            Dictionary with restored counts.
        """
        snapshot_dir = self.base_dir / snapshot_id
        nodes_path = snapshot_dir / "nodes.parquet"
        edges_path = snapshot_dir / "edges.parquet"

        if not nodes_path.exists() or not edges_path.exists():
            raise FileNotFoundError(
                f"Snapshot '{snapshot_id}' not found at {snapshot_dir}"
            )

        # Read Parquet files
        node_table = pq.read_table(nodes_path)
        edge_table = pq.read_table(edges_path)

        nodes: List[Dict[str, Any]] = node_table.to_pylist()
        edges: List[Dict[str, Any]] = edge_table.to_pylist()

        driver = get_memgraph_driver()

        with driver.session() as session:
            # 1. Clear active graph
            logger.info("Clearing active Memgraph database...")
            session.run("MATCH (n) DETACH DELETE n;")

            # 2. Restore Nodes (grouped by label for clean Cypher UNWIND)
            label_groups: Dict[str, List[Dict[str, Any]]] = {}
            for row in nodes:
                lbl = row.get("label") or "Entity"
                clean_lbl = "".join(c for c in lbl if c.isalnum()) or "Entity"
                if clean_lbl not in label_groups:
                    label_groups[clean_lbl] = []

                props = {}
                if row.get("properties_json"):
                    try:
                        props = json.loads(row["properties_json"])
                    except Exception:
                        props = {}

                if row.get("ticker"):
                    props["ticker"] = row["ticker"]
                if row.get("source_hash"):
                    props["source_hash"] = row["source_hash"]

                label_groups[clean_lbl].append({
                    "id": row["id"],
                    "props": props,
                })

            for label, items in label_groups.items():
                for i in range(0, len(items), batch_size):
                    batch = items[i : i + batch_size]
                    query = f"""
                        UNWIND $batch AS row
                        MERGE (n:{label} {{id: row.id}})
                        SET n += row.props
                    """
                    session.run(query, batch=batch)

            # 3. Restore Edges (grouped by rel_type for clean Cypher UNWIND)
            rel_groups: Dict[str, List[Dict[str, Any]]] = {}
            for row in edges:
                rtype = row.get("rel_type") or "RELATED_TO"
                clean_rtype = "".join(c for c in rtype if c.isalnum() or c == "_").upper() or "RELATED_TO"
                if clean_rtype not in rel_groups:
                    rel_groups[clean_rtype] = []

                props = {}
                if row.get("properties_json"):
                    try:
                        props = json.loads(row["properties_json"])
                    except Exception:
                        props = {}

                rel_groups[clean_rtype].append({
                    "source_id": row["source_id"],
                    "target_id": row["target_id"],
                    "props": props,
                    "source_hash": row.get("source_hash") or "",
                })

            for rtype, items in rel_groups.items():
                for i in range(0, len(items), batch_size):
                    batch = items[i : i + batch_size]
                    query = f"""
                        UNWIND $batch AS row
                        MATCH (s {{id: row.source_id}})
                        MATCH (t {{id: row.target_id}})
                        MERGE (s)-[r:{rtype}]->(t)
                        SET r += row.props, r.source_hash = row.source_hash
                    """
                    session.run(query, batch=batch)

        logger.info(
            "Successfully restored snapshot '%s' into Memgraph: %d nodes, %d edges",
            snapshot_id,
            len(nodes),
            len(edges),
        )

        return {
            "snapshot_id": snapshot_id,
            "status": "restored",
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    # ----------------------------------------------------------------------
    # 3. List Snapshots
    # ----------------------------------------------------------------------
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all available snapshots from PostgreSQL registry or local directory."""
        snapshots = []
        try:
            with pg_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT snapshot_id, tag, description, created_at,
                           time_window_start, time_window_end,
                           node_count, edge_count, snapshot_dir, metadata
                    FROM graph_snapshots
                    ORDER BY created_at DESC;
                """)
                rows = cur.fetchall()
                for row in rows:
                    snapshots.append({
                        "snapshot_id": row[0],
                        "tag": row[1],
                        "description": row[2],
                        "created_at": row[3].isoformat() if hasattr(row[3], "isoformat") else str(row[3]),
                        "time_window_start": str(row[4]) if row[4] else None,
                        "time_window_end": str(row[5]) if row[5] else None,
                        "node_count": row[6],
                        "edge_count": row[7],
                        "snapshot_dir": row[8],
                        "metadata": row[9] if isinstance(row[9], dict) else json.loads(row[9] or "{}"),
                    })
                if hasattr(cur, "close"):
                    cur.close()
                if snapshots:
                    return snapshots
        except Exception as exc:
            logger.debug("PostgreSQL snapshot lookup failed, falling back to disk: %s", exc)

        # Fallback: scan local base directory
        if self.base_dir.exists():
            for sdir in self.base_dir.iterdir():
                if sdir.is_dir() and (sdir / "nodes.parquet").exists():
                    try:
                        n_tbl = pq.read_table(sdir / "nodes.parquet")
                        e_tbl = pq.read_table(sdir / "edges.parquet") if (sdir / "edges.parquet").exists() else None
                        snapshots.append({
                            "snapshot_id": sdir.name,
                            "tag": "raw" if "raw" in sdir.name.lower() else "custom",
                            "description": "Discovered on disk",
                            "created_at": datetime.fromtimestamp(sdir.stat().st_mtime, timezone.utc).isoformat(),
                            "time_window_start": None,
                            "time_window_end": None,
                            "node_count": len(n_tbl),
                            "edge_count": len(e_tbl) if e_tbl is not None else 0,
                            "snapshot_dir": str(sdir),
                            "metadata": {},
                        })
                    except Exception:
                        pass

        return snapshots

    # ----------------------------------------------------------------------
    # 4. Get Snapshot
    # ----------------------------------------------------------------------
    def get_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Fetch metadata for a single snapshot by ID."""
        for s in self.list_snapshots():
            if s["snapshot_id"] == snapshot_id:
                return s
        return None
