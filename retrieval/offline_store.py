"""Dynamic In-Memory Offline Subgraph Store (ADR-00015).

Provides a zero-database, Point-in-Time (PiT) in-memory property graph provider
implementing the Neo4j/Memgraph `.session().run()` duck-typed interface used by
`GraphRAGPipelineOrchestrator`.

Supports:
- Parquet snapshots (`nodes.parquet`, `edges.parquet`) exported by `SnapshotManager`
- JSON / JSONL edge manifests
- Cycle-safe, hop-bounded downstream BFS traversal scoped by origin ticker and rel type
- Strict Point-in-Time validity (`valid_from <= as_of_date <= valid_to`) and status filtering
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None  # type: ignore[assignment]

DEFAULT_MAX_HOPS: int = 5
MAX_BFS_HOPS_CAP: int = 10

_MUTATING_KEYWORD_RE = re.compile(
    r"\b(?:CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|ALTER|CALL|LOAD)\b",
    re.IGNORECASE,
)
_STRING_LITERAL_RE = re.compile(r"'[^']*'|\"[^\"]*\"")
_ROOT_INLINE_TICKER_RE = re.compile(
    r"\(\s*[A-Za-z_]\w*(?:\s*:\s*[A-Za-z_]\w*)?\s*\{\s*ticker\s*:\s*"
    r"(?:['\"]([^'\"]+)['\"]|\$([A-Za-z_]\w*))\s*(?:,[^}]*)?\}\s*\)\s*-",
    re.IGNORECASE,
)
_WHERE_TICKER_RE = re.compile(
    r"\bs\.ticker\s*=\s*(?:['\"]([^'\"]+)['\"]|\$([A-Za-z_]\w*))",
    re.IGNORECASE,
)
_REL_PATTERN_RE = re.compile(
    r"-\s*\[\s*[A-Za-z_]*\w*\s*(?::\s*([A-Za-z0-9_|]+))?"
    r"(?:\s*\*\s*(?:(\d+)\s*\.\.\s*(\d+)|(\d+)|\.{2}(\d+))?)?\s*\]\s*->",
    re.IGNORECASE,
)
_STATUS_EQ_RE = re.compile(
    r"(?:coalesce\s*\([^,]+,\s*['\"]([^'\"]+)['\"]\s*\)|[A-Za-z_]\w*\.status)\s*=\s*['\"]([A-Za-z0-9_]+)['\"]",
    re.IGNORECASE,
)
_INLINE_DATE_RE = re.compile(
    r"valid_(?:from|to)\s*(?:<=|>=|<|>)\s*['\"](\d{4}-\d{2}-\d{2})['\"]",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CypherSubgraphQuery:
    """Parsed representation of a read-only subgraph MATCH query."""

    is_read_only: bool
    has_match: bool
    root_ticker: Optional[str] = None
    rel_types: Optional[Tuple[str, ...]] = None
    max_hops: int = DEFAULT_MAX_HOPS
    as_of_date: Optional[str] = None
    required_status: Optional[str] = "ACTIVE_CURRENT"


def parse_cypher_subgraph_query(
    query: str,
    params: Optional[Mapping[str, Any]] = None,
) -> CypherSubgraphQuery:
    """Extracts traversal scope and PiT constraints from read-only openCypher."""
    clean = (query or "").strip()
    if not clean:
        return CypherSubgraphQuery(is_read_only=False, has_match=False)

    unquoted = _STRING_LITERAL_RE.sub(" ", clean)
    if _MUTATING_KEYWORD_RE.search(unquoted):
        return CypherSubgraphQuery(is_read_only=False, has_match=False)

    has_match = bool(re.search(r"\bMATCH\b", unquoted, re.IGNORECASE))
    if not has_match:
        return CypherSubgraphQuery(is_read_only=True, has_match=False)

    param_map = params or {}

    # 1. Root ticker extraction (first node with ticker filter before `->`, or `WHERE s.ticker = ...`)
    root_ticker: Optional[str] = None
    root_match = _ROOT_INLINE_TICKER_RE.search(clean)
    if root_match:
        lit, pname = root_match.group(1), root_match.group(2)
        val = lit if lit is not None else param_map.get(pname)
        if val:
            root_ticker = str(val).strip().upper()
    else:
        where_match = _WHERE_TICKER_RE.search(clean)
        if where_match:
            lit, pname = where_match.group(1), where_match.group(2)
            val = lit if lit is not None else param_map.get(pname)
            if val:
                root_ticker = str(val).strip().upper()

    # 2. Relationship types and hop depth
    rel_types: Optional[Tuple[str, ...]] = None
    max_hops = 1
    rel_match = _REL_PATTERN_RE.search(clean)
    if rel_match:
        raw_types, lo_hop, hi_hop, exact_hop, open_hi = rel_match.groups()
        if raw_types:
            rel_types = tuple(t.strip().upper() for t in raw_types.split("|") if t.strip())

        matched_text = rel_match.group(0)
        if hi_hop is not None:
            max_hops = int(hi_hop)
        elif open_hi is not None:
            max_hops = int(open_hi)
        elif exact_hop is not None:
            max_hops = int(exact_hop)
        elif "*" in matched_text:
            max_hops = DEFAULT_MAX_HOPS
        else:
            max_hops = 1
        max_hops = max(1, min(max_hops, MAX_BFS_HOPS_CAP))

    # 3. Point-in-Time as_of_date cutoff
    as_of_date: Optional[str] = None
    raw_param_date = param_map.get("as_of_date")
    if raw_param_date:
        as_of_date = str(raw_param_date).strip()[:10]
    else:
        date_match = _INLINE_DATE_RE.search(clean)
        if date_match:
            as_of_date = date_match.group(1)

    # 4. Lifecycle status requirement
    required_status: Optional[str] = None
    status_match = _STATUS_EQ_RE.search(clean)
    if status_match:
        required_status = status_match.group(2).strip().upper()
    elif as_of_date is not None:
        required_status = "ACTIVE_CURRENT"

    return CypherSubgraphQuery(
        is_read_only=True,
        has_match=True,
        root_ticker=root_ticker,
        rel_types=rel_types,
        max_hops=max_hops,
        as_of_date=as_of_date,
        required_status=required_status,
    )


def _normalize_date_str(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in ("none", "null", "nan"):
        return None
    return s[:10]


def _normalize_edge_record(edge: Mapping[str, Any]) -> Dict[str, Any]:
    """Produces a consistent edge dictionary preserving cascade & PiT properties."""
    source = str(edge.get("source") or edge.get("source_id") or "UNKNOWN").strip().upper()
    target = str(edge.get("target") or edge.get("target_id") or "UNKNOWN").strip().upper()
    rel_type = str(edge.get("rel_type") or edge.get("type") or "SUPPLIES_TO").strip().upper()
    status = str(edge.get("status") or "ACTIVE_CURRENT").strip().upper()

    norm: Dict[str, Any] = dict(edge)
    norm.update({
        "source": source,
        "target": target,
        "rel_type": rel_type,
        "status": status,
        "valid_from": _normalize_date_str(edge.get("valid_from")),
        "valid_to": _normalize_date_str(edge.get("valid_to")),
    })
    if "nature" not in norm and "input_group" in norm:
        norm["nature"] = norm["input_group"]
    if "share" not in norm and "exposure_pct" in norm and norm["exposure_pct"] is not None:
        norm["share"] = float(norm["exposure_pct"]) / 100.0
    elif "exposure_pct" not in norm and "share" in norm and norm["share"] is not None:
        norm["exposure_pct"] = float(norm["share"]) * 100.0
    return norm


def _edge_matches_filters(
    edge: Mapping[str, Any],
    rel_types: Optional[Tuple[str, ...]],
    required_status: Optional[str],
    as_of_date: Optional[str],
) -> bool:
    if rel_types and edge["rel_type"] not in rel_types:
        return False
    if required_status and edge["status"] != required_status:
        return False
    if as_of_date:
        v_from = edge.get("valid_from")
        v_to = edge.get("valid_to")
        if v_from and v_from > as_of_date:
            return False
        if v_to and v_to < as_of_date:
            return False
    return True


def _parse_json_dict(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            loaded = json.loads(raw)
            return dict(loaded) if isinstance(loaded, dict) else {}
        except Exception:
            return {}
    return {}


class InMemorySubgraphStore:
    """Zero-database Point-in-Time in-memory property graph driver."""

    def __init__(
        self,
        edges: Optional[Iterable[Mapping[str, Any]]] = None,
        strict_query_scoping: bool = True,
    ):
        self.strict_query_scoping = strict_query_scoping
        self._edges_by_key: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        self._adj: Dict[str, List[Dict[str, Any]]] = {}
        self._rev_adj: Dict[str, List[Dict[str, Any]]] = {}
        if edges:
            for e in edges:
                self.add_edge(e)

    @property
    def edges(self) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
        """Backward-compatible 3-tuple view `(source, rel_type, target) -> edge_dict`."""
        return {(k[0], k[1], k[2]): v for k, v in self._edges_by_key.items()}

    def add_edge(self, edge: Mapping[str, Any]) -> None:
        norm = _normalize_edge_record(edge)
        input_group = str(norm.get("input_group") or norm.get("nature") or "default")
        key = (norm["source"], norm["rel_type"], norm["target"], input_group)
        self._edges_by_key[key] = norm
        self._adj.setdefault(norm["source"], []).append(norm)
        self._rev_adj.setdefault(norm["target"], []).append(norm)

    def list_tickers(self) -> List[str]:
        """Returns sorted distinct node tickers present across all loaded edges."""
        tickers: Set[str] = set()
        for edge in self._edges_by_key.values():
            tickers.add(edge["source"])
            tickers.add(edge["target"])
        return sorted(tickers)

    def _bfs_traverse(
        self,
        adj_map: Mapping[str, List[Dict[str, Any]]],
        neighbor_key: str,
        root_ticker: str,
        max_hops: int,
        rel_types: Optional[Tuple[str, ...]],
        required_status: Optional[str],
        as_of_date: Optional[str],
        direction_tag: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Sorted cycle-safe BFS returning `(edges, min_hop_by_node)` from `root_ticker`."""
        visited_nodes: Set[str] = {root_ticker}
        node_hops: Dict[str, int] = {root_ticker: 0}
        emitted_keys: Set[Tuple[str, str, str, str]] = set()
        results: List[Dict[str, Any]] = []
        queue: deque[Tuple[str, int]] = deque([(root_ticker, 0)])

        while queue:
            current_node, depth = queue.popleft()
            if depth >= max_hops:
                continue

            neighbors = sorted(
                adj_map.get(current_node, ()),
                key=lambda e: (e["source"], e["target"], e["rel_type"], str(e.get("nature", ""))),
            )
            next_hop = depth + 1
            for edge in neighbors:
                if not _edge_matches_filters(edge, rel_types, required_status, as_of_date):
                    continue
                grp = str(edge.get("input_group") or edge.get("nature") or "default")
                ekey = (edge["source"], edge["rel_type"], edge["target"], grp)
                if ekey not in emitted_keys:
                    emitted_keys.add(ekey)
                    row = dict(edge)
                    row["hop"] = next_hop
                    if direction_tag is not None:
                        row["traversal_direction"] = direction_tag
                    results.append(row)
                nxt = edge[neighbor_key]
                if nxt not in node_hops:
                    node_hops[nxt] = next_hop
                if nxt not in visited_nodes and next_hop < max_hops:
                    visited_nodes.add(nxt)
                    queue.append((nxt, next_hop))

        return results, node_hops

    def query_ego_subgraph(
        self,
        ticker: Optional[str] = None,
        max_hops: int = 2,
        direction: str = "both",
        as_of_date: Optional[str] = None,
        rel_types: Optional[Sequence[str]] = None,
        required_status: Optional[str] = "ACTIVE_CURRENT",
    ) -> Dict[str, Any]:
        """Returns Point-in-Time upstream/downstream ego network around `ticker` (ADR-00017)."""
        capped_hops = max(1, min(int(max_hops), MAX_BFS_HOPS_CAP))
        rel_tuple = tuple(r.strip().upper() for r in rel_types if r.strip()) if rel_types else None
        norm_date = _normalize_date_str(as_of_date)
        norm_dir = (direction or "both").strip().lower()
        all_tickers = self.list_tickers()

        if not ticker or not ticker.strip():
            edges_out: List[Dict[str, Any]] = []
            nodes_seen: Dict[str, Dict[str, Any]] = {}
            for edge in self._edges_by_key.values():
                if _edge_matches_filters(edge, rel_tuple, required_status, norm_date):
                    row = dict(edge)
                    row.setdefault("hop", 1)
                    edges_out.append(row)
                    nodes_seen.setdefault(edge["source"], {"id": edge["source"], "ticker": edge["source"], "hop": 0, "role": "CONNECTED"})
                    nodes_seen.setdefault(edge["target"], {"id": edge["target"], "ticker": edge["target"], "hop": 1, "role": "CONNECTED"})
            return {
                "root_ticker": None,
                "nodes": [nodes_seen[k] for k in sorted(nodes_seen)],
                "edges": edges_out,
                "available_tickers": all_tickers,
            }

        root = ticker.strip().upper()
        combined_edges: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        node_meta: Dict[str, Dict[str, Any]] = {root: {"id": root, "ticker": root, "hop": 0, "role": "ROOT"}}

        def _merge(edges: List[Dict[str, Any]], hops: Dict[str, int], role_label: str) -> None:
            for e in edges:
                grp = str(e.get("input_group") or e.get("nature") or "default")
                ekey = (e["source"], e["rel_type"], e["target"], grp)
                if ekey not in combined_edges or e["hop"] < combined_edges[ekey]["hop"]:
                    combined_edges[ekey] = e
            for node_id, h in hops.items():
                if node_id == root:
                    continue
                prev = node_meta.get(node_id)
                if prev is None:
                    node_meta[node_id] = {"id": node_id, "ticker": node_id, "hop": h, "role": role_label}
                else:
                    prev["hop"] = min(prev["hop"], h)
                    if prev["role"] != role_label:
                        prev["role"] = "BOTH"

        if norm_dir in ("downstream", "both"):
            d_edges, d_hops = self._bfs_traverse(
                self._adj, "target", root, capped_hops, rel_tuple, required_status, norm_date, "downstream"
            )
            _merge(d_edges, d_hops, "DOWNSTREAM")
        if norm_dir in ("upstream", "both"):
            u_edges, u_hops = self._bfs_traverse(
                self._rev_adj, "source", root, capped_hops, rel_tuple, required_status, norm_date, "upstream"
            )
            _merge(u_edges, u_hops, "UPSTREAM")

        ordered_nodes = sorted(node_meta.values(), key=lambda n: (n["hop"], n["ticker"]))
        ordered_edges = sorted(
            combined_edges.values(),
            key=lambda e: (e["hop"], e["source"], e["target"], e["rel_type"]),
        )
        return {
            "root_ticker": root,
            "nodes": ordered_nodes,
            "edges": ordered_edges,
            "available_tickers": all_tickers,
        }

    def query_edges(
        self,
        cypher: str,
        params: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluates read-only Cypher against in-memory adjacency with PiT BFS."""
        parsed = parse_cypher_subgraph_query(cypher, params=params)
        if not parsed.is_read_only or not parsed.has_match:
            return []

        if not self.strict_query_scoping:
            # Canonical smoke-fixture fallback mode: still enforces PiT/status and blocks writes,
            # while tolerating legacy single-hop canned Cypher strings targeting unrelated roots.
            return [
                dict(edge)
                for edge in self._edges_by_key.values()
                if _edge_matches_filters(edge, parsed.rel_types, parsed.required_status, parsed.as_of_date)
            ]

        if not parsed.root_ticker:
            results: List[Dict[str, Any]] = []
            for edge in self._edges_by_key.values():
                if _edge_matches_filters(edge, parsed.rel_types, parsed.required_status, parsed.as_of_date):
                    row = dict(edge)
                    row.setdefault("hop", 1)
                    results.append(row)
            return results

        results, _ = self._bfs_traverse(
            self._adj,
            "target",
            parsed.root_ticker,
            parsed.max_hops,
            parsed.rel_types,
            parsed.required_status,
            parsed.as_of_date,
        )
        return results

    def session(self):
        """Returns a Neo4j/Memgraph-compatible session context manager."""
        store = self

        class _InMemorySession:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def run(self, query: str, **kwargs):
                return store.query_edges(query, params=kwargs)

        return _InMemorySession()

    @classmethod
    def from_parquet_snapshot(
        cls,
        snapshot_dir: Union[str, Path],
        strict_query_scoping: bool = True,
    ) -> "InMemorySubgraphStore":
        """Loads `nodes.parquet` and `edges.parquet` exported by `SnapshotManager`."""
        if pq is None:  # pragma: no cover
            raise RuntimeError("pyarrow is required to load Parquet graph snapshots")

        base = Path(snapshot_dir)
        nodes_path = base / "nodes.parquet"
        edges_path = base / "edges.parquet"
        if not edges_path.is_file():
            raise FileNotFoundError(f"Missing edges snapshot at: {edges_path}")

        node_to_ticker: Dict[str, str] = {}
        if nodes_path.is_file():
            node_rows = pq.read_table(nodes_path).to_pylist()
            for nrow in node_rows:
                nid = str(nrow.get("id") or "").strip()
                if not nid:
                    continue
                props = _parse_json_dict(nrow.get("properties_json"))
                ticker = nrow.get("ticker") or props.get("ticker") or nid
                node_to_ticker[nid] = str(ticker).strip().upper()

        edge_rows = pq.read_table(edges_path).to_pylist()
        normalized_edges: List[Dict[str, Any]] = []
        for erow in edge_rows:
            src_id = str(erow.get("source_id") or erow.get("source") or "").strip()
            tgt_id = str(erow.get("target_id") or erow.get("target") or "").strip()
            if not src_id or not tgt_id:
                continue

            props = _parse_json_dict(erow.get("properties_json"))
            merged: Dict[str, Any] = {
                **props,
                "source": node_to_ticker.get(src_id, src_id.upper()),
                "target": node_to_ticker.get(tgt_id, tgt_id.upper()),
                "source_id": src_id,
                "target_id": tgt_id,
                "rel_type": erow.get("rel_type") or props.get("rel_type") or "SUPPLIES_TO",
            }
            if erow.get("source_hash") and "source_hash" not in merged:
                merged["source_hash"] = erow["source_hash"]
            normalized_edges.append(merged)

        return cls(edges=normalized_edges, strict_query_scoping=strict_query_scoping)

    @classmethod
    def from_edges_file(
        cls,
        edges_path: Union[str, Path],
        strict_query_scoping: bool = True,
    ) -> "InMemorySubgraphStore":
        """Loads supply-chain edges from a `.json` or `.jsonl` manifest file."""
        path = Path(edges_path)
        if not path.is_file():
            raise FileNotFoundError(f"Edges manifest file not found at: {path}")

        raw_text = path.read_text(encoding="utf-8").strip()
        if not raw_text:
            return cls(edges=[], strict_query_scoping=strict_query_scoping)

        edges: List[Mapping[str, Any]]
        if path.suffix.lower() == ".jsonl":
            edges = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
        else:
            parsed = json.loads(raw_text)
            if isinstance(parsed, list):
                edges = parsed
            elif isinstance(parsed, dict) and isinstance(parsed.get("edges"), list):
                edges = parsed["edges"]
            else:
                raise ValueError(f"Unsupported edges JSON structure in: {path}")

        return cls(edges=edges, strict_query_scoping=strict_query_scoping)

