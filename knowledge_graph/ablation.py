"""Controlled KG Quality, SEC Form-Ablation & Embedder/Reranker Scaling Engine (ADR-00047).

Provides zero-database, deterministic offline evaluation over persisted Parquet snapshots
(`nodes.parquet`, `edges.parquet`, `chunks.parquet`):
1. Zero-variance Leave-One-Form-Out (LOFO) & additive form-inclusion slicing tagged by
   canonical `(form_type, sec_graph_layer)` from `ingestion/sec/canon.py`.
2. Four-pillar quantitative KG scorecard:
   - Pillar 1 (Topology): Weighted Leiden/Louvain modularity Q, max degree, degree Gini,
     mean 2-hop ego size vs. `MAX_SUBGRAPH_EDGES = 25`, cascade `dropped_impact_share`,
     LCC share, and singleton rate.
   - Pillar 2 (Epistemic Provenance): `FILING_GROUNDED` / `PARTIALLY_ASSUMED` /
     `STRESS_UPPER_BOUND` distribution and chunk-evidence link resolution rate.
   - Pillar 3 (Extraction Fidelity): `RuntimeRejectionQueue` quarantine rates stratified
     by `(form_type, ErrorStage)`.
   - Pillar 4 (Retrieval & Systems): `Recall@5`, `Recall@10`, `MRR@10`, `nDCG@10` across
     `E1` (`0.6B`), `E2` (`4B` MRL `1024`-d), `E2b` (`8B` MRL ceiling), and `E3` (`0.6B`
     + `RerankerPort`) with seeded paired bootstrap 95% confidence intervals and CPU latency.
3. Pre-registered empirical promotion gates and reproducible JSON / Markdown / PNG artifacts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import networkx as nx
import pyarrow.parquet as pq

from contagion.cascade import CascadeEdge, CascadeGraph, SparseNetworkCascadeEngine
from contagion.parameters import resolve_edge_cascade_parameters
from embedding.embedder import embed_documents, embed_query, get_embedding_dim
from embedding.reranker import DEFAULT_RERANKER_MODEL, RerankerPort, get_reranker
from ingestion.sec.canon import SECGraphLayer, get_form_spec, normalize_form_type
from knowledge_graph.communities import _run_hierarchical_partition, filter_structural_edges_pit
from retrieval.pipeline import MAX_SUBGRAPH_EDGES
from retrieval.reporting import classify_provenance_tier
from retrieval.temporal import PiTContext

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")

# Pre-registered ADR-00047 Promotion Gate Thresholds
MAX_MODULARITY_DROP: float = 0.03
MAX_DROPPED_IMPACT_INCREASE: float = 0.02
MAX_QUARANTINE_RATE_INCREASE: float = 0.05
MIN_ARM_PRACTICAL_LIFT: float = 0.03
MAX_LIVE_CPU_P95_LATENCY_MS: float = 250.0


@dataclass(frozen=True)
class EmbedderArmSpec:
    """Declarative specification for one retrieval / embedder scaling arm (ADR-00047)."""

    arm_id: str
    label: str
    embedder_model: str
    mrl_dim: Optional[int] = None
    use_reranker: bool = False
    reranker_model: Optional[str] = None
    live_deployable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_EMBEDDER_ARMS: Dict[str, EmbedderArmSpec] = {
    "E1": EmbedderArmSpec(
        arm_id="E1",
        label="qwen3-embedding:0.6b (1024-d native)",
        embedder_model="qwen3-embedding:0.6b",
        mrl_dim=None,
        use_reranker=False,
        live_deployable=True,
    ),
    "E2": EmbedderArmSpec(
        arm_id="E2",
        label="qwen3-embedding:4b (MRL 1024-d)",
        embedder_model="qwen3-embedding:4b",
        mrl_dim=1024,
        use_reranker=False,
        live_deployable=True,
    ),
    "E2b": EmbedderArmSpec(
        arm_id="E2b",
        label="qwen3-embedding:8b Q4_K_M (MRL 1024-d offline ceiling)",
        embedder_model="qwen3-embedding:8b",
        mrl_dim=1024,
        use_reranker=False,
        live_deployable=False,
    ),
    "E3": EmbedderArmSpec(
        arm_id="E3",
        label="qwen3-embedding:0.6b + CrossEncoder Reranker",
        embedder_model="qwen3-embedding:0.6b",
        mrl_dim=None,
        use_reranker=True,
        reranker_model=DEFAULT_RERANKER_MODEL,
        live_deployable=True,
    ),
}


@dataclass(frozen=True)
class GoldQueryCase:
    """Single gold chunk-cited retrieval benchmark query."""

    query_id: str
    query: str
    relevant_chunk_ids: Tuple[str, ...]
    target_ticker: Optional[str] = None
    form_types: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TopologyMetrics:
    """Pillar 1 — Point-in-Time KG topology, community modularity, and truncation health."""

    node_count: int
    edge_count: int
    edges_per_node: float
    leiden_modularity_q: float
    community_count: int
    community_algorithm: str
    max_degree: int
    degree_gini: float
    top1pct_degree_share: float
    mean_2hop_ego_edges: float
    max_subgraph_edges_cap: int
    mean_dropped_impact_share: float
    p95_dropped_impact_share: float
    lcc_share: float
    singleton_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProvenanceMetrics:
    """Pillar 2 — Epistemic provenance grounding and chunk-evidence link completeness."""

    total_reports: int
    filing_grounded_share: float
    partially_assumed_share: float
    stress_upper_bound_share: float
    chunked_edge_share: float
    resolved_evidence_link_share: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExtractionFidelityMetrics:
    """Pillar 3 — RuntimeRejectionQueue quarantine rates stratified by form and error stage."""

    total_rejections: int
    rejections_by_form: Dict[str, int] = field(default_factory=dict)
    rejections_by_stage: Dict[str, int] = field(default_factory=dict)
    quarantine_rate_by_form: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalMetrics:
    """Pillar 4 — Retrieval precision, paired bootstrap CIs, and systems latency envelope."""

    arm_id: str
    label: str
    live_deployable: bool
    query_count: int
    chunk_count: int
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float
    p50_query_latency_ms: float
    p95_query_latency_ms: float
    embed_throughput_chunks_per_sec: float
    recall_at_10_ci95: Tuple[float, float]
    ndcg_at_10_ci95: Tuple[float, float]
    per_query_recall_at_10: Tuple[float, ...] = ()
    per_query_ndcg_at_10: Tuple[float, ...] = ()
    delta_vs_e1_recall10_ci95: Optional[Tuple[float, float]] = None
    delta_vs_e1_ndcg10_ci95: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["recall_at_10_ci95"] = list(self.recall_at_10_ci95)
        payload["ndcg_at_10_ci95"] = list(self.ndcg_at_10_ci95)
        if self.delta_vs_e1_recall10_ci95 is not None:
            payload["delta_vs_e1_recall10_ci95"] = list(self.delta_vs_e1_recall10_ci95)
        if self.delta_vs_e1_ndcg10_ci95 is not None:
            payload["delta_vs_e1_ndcg10_ci95"] = list(self.delta_vs_e1_ndcg10_ci95)
        payload.pop("per_query_recall_at_10", None)
        payload.pop("per_query_ndcg_at_10", None)
        return payload


@dataclass(frozen=True)
class AblationCellResult:
    """Complete 4-pillar evaluation output for a single `(form_subset, graph_isolation)` cell."""

    cell_id: str
    subset_name: str
    forms_included: Tuple[str, ...]
    graph_isolation: str
    topology: TopologyMetrics
    provenance: ProvenanceMetrics
    extraction_fidelity: ExtractionFidelityMetrics
    retrieval_by_arm: Dict[str, RetrievalMetrics] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "subset_name": self.subset_name,
            "forms_included": list(self.forms_included),
            "graph_isolation": self.graph_isolation,
            "topology": self.topology.to_dict(),
            "provenance": self.provenance.to_dict(),
            "extraction_fidelity": self.extraction_fidelity.to_dict(),
            "retrieval_by_arm": {k: v.to_dict() for k, v in self.retrieval_by_arm.items()},
        }


@dataclass(frozen=True)
class FormPromotionVerdict:
    """Pre-registered ADR-00047 Section 4.6 Gate #1 verdict for one SEC form type."""

    form_type: str
    passed: bool
    reasons: Tuple[str, ...]
    deltas: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "form_type": self.form_type,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "deltas": dict(self.deltas),
        }


@dataclass(frozen=True)
class ArmPromotionVerdict:
    """Pre-registered ADR-00047 Section 4.6 Gate #2 verdict for one embedder/reranker arm."""

    arm_id: str
    live_deployable: bool
    passed: bool
    reasons: Tuple[str, ...]
    deltas: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "live_deployable": self.live_deployable,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "deltas": dict(self.deltas),
        }


# ==============================================================================
# 1. SNAPSHOT LOADING & ZERO-VARIANCE FORM / LAYER SLICING
# ==============================================================================


def _parse_edge_properties(edge: Mapping[str, Any]) -> Dict[str, Any]:
    props_raw = edge.get("properties")
    if isinstance(props_raw, Mapping):
        merged = dict(props_raw)
    elif isinstance(edge.get("properties_json"), str):
        try:
            loaded = json.loads(str(edge["properties_json"]))
            merged = dict(loaded) if isinstance(loaded, dict) else {}
        except (json.JSONDecodeError, TypeError, ValueError):
            merged = {}
    else:
        merged = {}
    for k, v in edge.items():
        if k not in ("properties", "properties_json") and v is not None:
            merged.setdefault(k, v)
    return merged


def _resolve_record_form_and_layer(record: Mapping[str, Any]) -> Tuple[Optional[str], str]:
    props = _parse_edge_properties(record)
    raw_form = props.get("form_type") or props.get("form") or record.get("form_type")
    norm_form = normalize_form_type(str(raw_form)) if raw_form else None
    raw_layer = props.get("sec_graph_layer") or record.get("sec_graph_layer")
    if raw_layer and str(raw_layer).strip():
        return norm_form, str(raw_layer).strip().upper()
    if norm_form:
        spec = get_form_spec(norm_form)
        if spec is not None:
            return norm_form, spec.graph_layer.value
    return norm_form, SECGraphLayer.DOMAIN1_OPERATIONAL.value


def load_snapshot_tables(
    snapshot_dir: Path | str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load `(nodes, edges, chunks)` lists from a Parquet snapshot directory."""
    snap = Path(snapshot_dir)
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    chunks: List[Dict[str, Any]] = []

    if (snap / "nodes.parquet").is_file():
        nodes = list(pq.read_table(snap / "nodes.parquet").to_pylist())
    if (snap / "edges.parquet").is_file():
        edges = list(pq.read_table(snap / "edges.parquet").to_pylist())
    if (snap / "chunks.parquet").is_file():
        chunks = list(pq.read_table(snap / "chunks.parquet").to_pylist())
    return nodes, edges, chunks


def load_gold_queries(queries_jsonl: Optional[Path | str]) -> List[GoldQueryCase]:
    """Load gold chunk-cited benchmark queries from a JSONL file."""
    if not queries_jsonl:
        return []
    path = Path(queries_jsonl)
    if not path.is_file():
        return []
    cases: List[GoldQueryCase] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        rel_ids = tuple(str(cid).strip() for cid in (row.get("relevant_chunk_ids") or []) if str(cid).strip())
        if not row.get("query") or not rel_ids:
            continue
        cases.append(
            GoldQueryCase(
                query_id=str(row.get("query_id") or f"Q{idx:03d}"),
                query=str(row["query"]).strip(),
                relevant_chunk_ids=rel_ids,
                target_ticker=(str(row["target_ticker"]).strip().upper() if row.get("target_ticker") else None),
                form_types=tuple(normalize_form_type(f) for f in (row.get("form_types") or []) if str(f).strip()),
            )
        )
    return cases


def filter_snapshot_by_forms_and_layer(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
    *,
    allowed_forms: Optional[Sequence[str]],
    graph_isolation: str,
    as_of: date | str | PiTContext,
    include_unanchored: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministically slice a single persisted snapshot by `(form_type, sec_graph_layer, as_of)`."""
    cutoff = as_of.as_of if isinstance(as_of, PiTContext) else (
        as_of if isinstance(as_of, date) else date.fromisoformat(str(as_of).strip()[:10])
    )
    allowed_form_set: Optional[Set[str]] = (
        {normalize_form_type(f) for f in allowed_forms if str(f).strip()}
        if allowed_forms is not None
        else None
    )
    layered_mode = str(graph_isolation).strip().lower() == "layered"

    pre_filtered_edges: List[Dict[str, Any]] = []
    for raw_edge in edges:
        edge_form, edge_layer = _resolve_record_form_and_layer(raw_edge)
        if allowed_form_set is not None and edge_form not in allowed_form_set:
            continue
        if layered_mode and edge_layer != SECGraphLayer.DOMAIN1_OPERATIONAL.value:
            continue
        props = _parse_edge_properties(raw_edge)
        props["form_type"] = edge_form
        props["sec_graph_layer"] = edge_layer
        enriched = dict(raw_edge)
        enriched["properties"] = props
        pre_filtered_edges.append(enriched)

    pit_edges = filter_structural_edges_pit(
        pre_filtered_edges,
        as_of=cutoff,
        include_unanchored=include_unanchored,
    )
    # Re-attach full edge metadata (buffer_days, param_provenance, form_type, sec_graph_layer)
    meta_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for raw_edge in pre_filtered_edges:
        props = raw_edge["properties"]
        src = str(raw_edge.get("source_id") or raw_edge.get("source") or "").strip()
        tgt = str(raw_edge.get("target_id") or raw_edge.get("target") or "").strip()
        rel = str(raw_edge.get("rel_type") or raw_edge.get("type") or props.get("rel_type") or "SUPPLIES_TO").strip().upper()
        meta_by_key[(src, tgt, rel)] = props

    active_edges: List[Dict[str, Any]] = []
    active_node_ids: Set[str] = set()
    for pe in pit_edges:
        src = str(pe["source_id"])
        tgt = str(pe["target_id"])
        rel = str(pe["rel_type"])
        props = meta_by_key.get((src, tgt, rel), {})
        merged_edge = {**props, **pe}
        merged_edge["form_type"] = props.get("form_type")
        merged_edge["sec_graph_layer"] = props.get("sec_graph_layer", SECGraphLayer.DOMAIN1_OPERATIONAL.value)
        active_edges.append(merged_edge)
        active_node_ids.add(src)
        active_node_ids.add(tgt)

    filtered_nodes: List[Dict[str, Any]] = [dict(n) for n in nodes]
    known_ids = {str(n.get("id") or "").strip() for n in filtered_nodes}
    for nid in sorted(active_node_ids - known_ids):
        filtered_nodes.append({"id": nid, "label": "Company", "ticker": None})

    filtered_chunks: List[Dict[str, Any]] = []
    for raw_chunk in chunks:
        c_form, c_layer = _resolve_record_form_and_layer(raw_chunk)
        if allowed_form_set is not None and c_form not in allowed_form_set:
            continue
        if layered_mode and c_layer != SECGraphLayer.DOMAIN1_OPERATIONAL.value:
            continue
        v_raw = raw_chunk.get("valid_from")
        if v_raw is not None and str(v_raw).strip():
            try:
                v_date = date.fromisoformat(str(v_raw).strip()[:10])
                if v_date > cutoff:
                    continue
            except ValueError:
                if not include_unanchored:
                    continue
        elif not include_unanchored:
            continue
        rec = dict(raw_chunk)
        rec["form_type"] = c_form
        rec["sec_graph_layer"] = c_layer
        filtered_chunks.append(rec)

    return filtered_nodes, active_edges, filtered_chunks


# ==============================================================================
# 2. PILLAR 1 (TOPOLOGY) & PILLAR 2 (PROVENANCE) COMPUTATION
# ==============================================================================


def _gini_coefficient(values: Sequence[float]) -> float:
    vals = sorted(max(0.0, float(v)) for v in values)
    n = len(vals)
    if n == 0:
        return 0.0
    total = sum(vals)
    if total <= 0.0:
        return 0.0
    weighted_sum = sum((2 * (idx + 1) - n - 1) * val for idx, val in enumerate(vals))
    return round(weighted_sum / (n * total), 6)


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(math.floor(rank))
    hi = min(len(ordered) - 1, lo + 1)
    frac = rank - lo
    return ordered[lo] + frac * (ordered[hi] - ordered[lo])


def _evaluate_cascades_for_origins(
    active_edges: Sequence[Mapping[str, Any]],
    *,
    shock_origins: Optional[Sequence[str]] = None,
    max_subgraph_edges: int = MAX_SUBGRAPH_EDGES,
    shock_magnitude: float = 1.0,
    shock_duration_days: float = 60.0,
) -> List[Dict[str, Any]]:
    """Run SparseNetworkCascadeEngine across origins to measure `dropped_impact_share` & provenance tiers."""
    if not active_edges:
        return []

    out_degree: Counter[str] = Counter()
    cascade_edges: List[Dict[str, Any]] = []
    for e in active_edges:
        src = str(e.get("source_id") or "").strip()
        tgt = str(e.get("target_id") or "").strip()
        if not src or not tgt or src == tgt:
            continue
        out_degree[src] += 1
        resolved = resolve_edge_cascade_parameters(e)
        rel_type = str(e.get("rel_type") or "SUPPLIES_TO")
        group_name = f"{rel_type.lower()}:{resolved['input_group']}"
        cascade_edges.append(
            {
                "source": src,
                "target": tgt,
                "rel_type": rel_type,
                "input_group": group_name,
                "share": float(resolved["share"]),
                "buffer_days": float(resolved["buffer_days"]),
                "param_provenance": str(resolved["param_provenance"]),
                "source_type": str(e.get("source_type") or "sec_filing"),
            }
        )

    if not cascade_edges:
        return []

    # Normalize incoming group shares if raw FLAT/multi-form edges sum > 1.0 on a target group
    group_totals: Dict[Tuple[str, str], float] = defaultdict(float)
    for ce in cascade_edges:
        group_totals[(ce["target"], ce["input_group"])] += float(ce["share"])

    cgraph = CascadeGraph()
    for ce in cascade_edges:
        g_sum = group_totals[(ce["target"], ce["input_group"])]
        norm_share = (float(ce["share"]) / g_sum) if g_sum > 1.0 else float(ce["share"])
        ce["effective_share"] = norm_share
        cgraph.add_edge(
            CascadeEdge(
                source=ce["source"],
                target=ce["target"],
                input_group=ce["input_group"],
                share=max(0.0, min(1.0, norm_share)),
                buffer_days=max(0.0, float(ce["buffer_days"])),
            )
        )

    if shock_origins:
        origins = [str(o).strip() for o in shock_origins if str(o).strip()]
    else:
        origins = [nid for nid, _ in out_degree.most_common(min(20, len(out_degree)))]

    engine = SparseNetworkCascadeEngine(max_iterations=5)
    reports: List[Dict[str, Any]] = []
    for origin in origins:
        if origin not in cgraph.nodes:
            continue
        summaries = engine.propagate(
            graph=cgraph,
            initial_shocks={origin: shock_magnitude},
            initial_durations={origin: shock_duration_days},
        )
        # Rank downstream edges by propagated impact (mirrors retrieval/pipeline.py ADR-00037)
        impact_by_node: Dict[str, float] = {
            nid: float(summary.impairment_score) for nid, summary in summaries.items() if nid != origin
        }
        reachable_edges = [
            ce for ce in cascade_edges
            if ce["source"] in summaries and ce["target"] in impact_by_node and impact_by_node[ce["target"]] > 0.0
        ]
        scored_edges: List[Tuple[float, Dict[str, Any]]] = []
        for ce in reachable_edges:
            edge_impact = float(impact_by_node.get(ce["target"], 0.0)) * float(ce["effective_share"])
            scored_edges.append((edge_impact, ce))
        scored_edges.sort(key=lambda pair: (-pair[0], pair[1]["source"], pair[1]["target"]))

        total_impact = sum(score for score, _ in scored_edges)
        if len(scored_edges) > max_subgraph_edges and total_impact > 0.0:
            dropped_impact = sum(score for score, _ in scored_edges[max_subgraph_edges:])
            dropped_share = min(1.0, max(0.0, dropped_impact / total_impact))
        elif len(scored_edges) > max_subgraph_edges:
            dropped_share = (len(scored_edges) - max_subgraph_edges) / float(len(scored_edges))
        else:
            dropped_share = 0.0

        # Epistemic provenance share: fraction of total downstream edge impact using DEFAULT_CONSERVATIVE_BOUND
        assumed_impact = sum(
            score
            for score, ce in scored_edges
            if ce.get("param_provenance") == "DEFAULT_CONSERVATIVE_BOUND"
        )
        assumed_share = (assumed_impact / total_impact) if total_impact > 0.0 else 0.0
        tier = classify_provenance_tier(assumed_share)
        reports.append(
            {
                "origin": origin,
                "dropped_impact_share": round(dropped_share, 6),
                "assumed_impact_share": round(assumed_share, 6),
                "provenance_tier": tier,
            }
        )
    return reports


def compute_topology_and_provenance(
    nodes: Sequence[Mapping[str, Any]],
    active_edges: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
    *,
    shock_origins: Optional[Sequence[str]] = None,
    max_subgraph_edges: int = MAX_SUBGRAPH_EDGES,
    random_seed: int = 42,
) -> Tuple[TopologyMetrics, ProvenanceMetrics]:
    """Compute Pillar 1 (Topology) and Pillar 2 (Epistemic Provenance) metrics deterministically."""
    graph = nx.Graph()
    for n in nodes:
        nid = str(n.get("id") or "").strip()
        if nid:
            graph.add_node(nid)

    for e in active_edges:
        src = str(e.get("source_id") or "").strip()
        tgt = str(e.get("target_id") or "").strip()
        if not src or not tgt or src == tgt:
            continue
        weight = float(e.get("weight") or e.get("share") or 1.0)
        if graph.has_edge(src, tgt):
            graph[src][tgt]["weight"] = max(float(graph[src][tgt].get("weight", 1.0)), weight)
        else:
            graph.add_edge(src, tgt, weight=weight)

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    edges_per_node = round(edge_count / float(node_count), 6) if node_count > 0 else 0.0

    degrees = [int(d) for _, d in graph.degree()]
    max_degree = max(degrees) if degrees else 0
    degree_gini = _gini_coefficient(degrees)
    total_deg = sum(degrees)
    top_k = max(1, int(math.ceil(0.01 * node_count))) if node_count > 0 else 0
    top1pct_share = (
        round(sum(sorted(degrees, reverse=True)[:top_k]) / float(total_deg), 6)
        if total_deg > 0 and top_k > 0
        else 0.0
    )

    if node_count > 0:
        components = list(nx.connected_components(graph))
        lcc_size = max(len(c) for c in components) if components else 0
        lcc_share = round(lcc_size / float(node_count), 6)
        singletons = sum(1 for d in degrees if d == 0)
        singleton_rate = round(singletons / float(node_count), 6)
    else:
        lcc_share = 0.0
        singleton_rate = 0.0

    # Weighted community modularity Q via ADR-00037 partitioner
    modularity_q = 0.0
    community_count = 0
    algorithm = "hierarchical_leiden"
    if edge_count > 0:
        non_isolated = graph.subgraph([n for n, d in graph.degree() if d > 0]).copy()
        algorithm, levels = _run_hierarchical_partition(
            non_isolated,
            max_cluster_size=10,
            max_levels=1,
            random_seed=random_seed,
        )
        level0 = levels.get(0, [])
        community_count = len(level0)
        clusters = [set(members) for _, members, _ in level0 if members]
        if clusters:
            try:
                modularity_q = round(float(nx.community.modularity(non_isolated, clusters, weight="weight")), 6)
            except Exception:
                modularity_q = 0.0

    # Mean 2-hop ego edge count over non-isolated nodes
    ego_edge_counts: List[int] = []
    for nid, deg in graph.degree():
        if deg <= 0:
            continue
        ego = nx.ego_graph(graph, nid, radius=2)
        ego_edge_counts.append(ego.number_of_edges())
    mean_2hop_ego_edges = (
        round(sum(ego_edge_counts) / float(len(ego_edge_counts)), 4) if ego_edge_counts else 0.0
    )

    cascade_reports = _evaluate_cascades_for_origins(
        active_edges,
        shock_origins=shock_origins,
        max_subgraph_edges=max_subgraph_edges,
    )
    dropped_shares = [float(r["dropped_impact_share"]) for r in cascade_reports]
    mean_dropped = round(sum(dropped_shares) / float(len(dropped_shares)), 6) if dropped_shares else 0.0
    p95_dropped = round(_percentile(dropped_shares, 95.0), 6) if dropped_shares else 0.0

    topology = TopologyMetrics(
        node_count=node_count,
        edge_count=edge_count,
        edges_per_node=edges_per_node,
        leiden_modularity_q=modularity_q,
        community_count=community_count,
        community_algorithm=algorithm,
        max_degree=max_degree,
        degree_gini=degree_gini,
        top1pct_degree_share=top1pct_share,
        mean_2hop_ego_edges=mean_2hop_ego_edges,
        max_subgraph_edges_cap=max_subgraph_edges,
        mean_dropped_impact_share=mean_dropped,
        p95_dropped_impact_share=p95_dropped,
        lcc_share=lcc_share,
        singleton_rate=singleton_rate,
    )

    # Pillar 2: Epistemic provenance & chunk link integrity
    total_reports = len(cascade_reports)
    if total_reports > 0:
        fg_share = round(
            sum(1 for r in cascade_reports if r["provenance_tier"] == "FILING_GROUNDED") / float(total_reports),
            6,
        )
        pa_share = round(
            sum(1 for r in cascade_reports if r["provenance_tier"] == "PARTIALLY_ASSUMED") / float(total_reports),
            6,
        )
        su_share = round(
            sum(1 for r in cascade_reports if r["provenance_tier"] == "STRESS_UPPER_BOUND") / float(total_reports),
            6,
        )
    else:
        fg_share = pa_share = su_share = 0.0

    chunk_id_set = {str(c.get("chunk_id") or "").strip() for c in chunks if str(c.get("chunk_id") or "").strip()}
    chunked_edges = 0
    total_links = 0
    resolved_links = 0
    for e in active_edges:
        cids = [str(cid).strip() for cid in (e.get("evidence_chunk_ids") or []) if str(cid).strip()]
        if cids:
            chunked_edges += 1
            total_links += len(cids)
            resolved_links += sum(1 for cid in cids if cid in chunk_id_set)

    chunked_edge_share = round(chunked_edges / float(len(active_edges)), 6) if active_edges else 0.0
    resolved_link_share = round(resolved_links / float(total_links), 6) if total_links > 0 else 0.0

    provenance = ProvenanceMetrics(
        total_reports=total_reports,
        filing_grounded_share=fg_share,
        partially_assumed_share=pa_share,
        stress_upper_bound_share=su_share,
        chunked_edge_share=chunked_edge_share,
        resolved_evidence_link_share=resolved_link_share,
    )
    return topology, provenance


# ==============================================================================
# 3. PILLAR 3 — EXTRACTION FIDELITY (REJECTION QUEUE STRATIFICATION)
# ==============================================================================


def compute_extraction_fidelity(
    rejection_dir: Optional[Path | str],
    chunks: Sequence[Mapping[str, Any]],
    *,
    allowed_forms: Optional[Sequence[str]] = None,
) -> ExtractionFidelityMetrics:
    """Summarize RuntimeRejectionQueue failures stratified by `(form_type, ErrorStage)`."""
    by_form: Counter[str] = Counter()
    by_stage: Counter[str] = Counter()
    allowed_set = (
        {normalize_form_type(f) for f in allowed_forms if str(f).strip()}
        if allowed_forms is not None
        else None
    )

    if rejection_dir is not None:
        rdir = Path(rejection_dir)
        files: List[Path] = []
        if rdir.is_file():
            files = [rdir]
        elif rdir.is_dir():
            files = sorted(rdir.glob("*.jsonl"))
        for path in files:
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rec = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
                raw_form = rec.get("form_type") or meta.get("form_type") or "UNKNOWN"
                norm_form = normalize_form_type(str(raw_form)) if raw_form != "UNKNOWN" else "UNKNOWN"
                if allowed_set is not None and norm_form != "UNKNOWN" and norm_form not in allowed_set:
                    continue
                stage = str(rec.get("error_stage") or "UNKNOWN").strip().upper()
                by_form[norm_form] += 1
                by_stage[stage] += 1

    chunks_by_form: Counter[str] = Counter()
    for ch in chunks:
        c_form, _ = _resolve_record_form_and_layer(ch)
        chunks_by_form[c_form or "UNKNOWN"] += 1

    rates: Dict[str, float] = {}
    for form_code, rej_count in sorted(by_form.items()):
        denom = max(1, chunks_by_form.get(form_code, 0) + rej_count)
        rates[form_code] = round(rej_count / float(denom), 6)

    return ExtractionFidelityMetrics(
        total_rejections=sum(by_form.values()),
        rejections_by_form=dict(sorted(by_form.items())),
        rejections_by_stage=dict(sorted(by_stage.items())),
        quarantine_rate_by_form=rates,
    )


# ==============================================================================
# 4. PILLAR 4 — RETRIEVAL PRECISION, PAIRED BOOTSTRAP CIs & EMBEDDER ARMS
# ==============================================================================


def _cosine_dot(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(vec_a, vec_b))


def _offline_bm25_scores(query: str, docs: Sequence[str]) -> List[float]:
    """Deterministic lightweight BM25 lexical scoring mirroring Postgres `tsvector` rank offline."""
    if not docs:
        return []
    q_terms = _TOKEN_RE.findall(str(query or "").lower())
    if not q_terms:
        return [0.0] * len(docs)

    tokenized_docs = [_TOKEN_RE.findall(str(d or "").lower()) for d in docs]
    n_docs = len(tokenized_docs)
    avgdl = max(1.0, sum(len(toks) for toks in tokenized_docs) / float(n_docs))
    df: Counter[str] = Counter()
    for toks in tokenized_docs:
        for term in set(toks):
            df[term] += 1

    k1 = 1.5
    b = 0.75
    scores: List[float] = []
    for toks in tokenized_docs:
        tf = Counter(toks)
        dl = len(toks)
        score = 0.0
        for term in q_terms:
            freq = tf.get(term, 0)
            if freq <= 0:
                continue
            doc_freq = df.get(term, 0)
            idf = math.log(1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            denom = freq + k1 * (1.0 - b + b * (dl / avgdl))
            score += idf * ((freq * (k1 + 1.0)) / denom)
        scores.append(score)
    return scores


def _dcg_at_k(ranked_ids: Sequence[str], relevant: Set[str], k: int) -> float:
    dcg = 0.0
    for idx, cid in enumerate(ranked_ids[:k], start=1):
        if cid in relevant:
            dcg += 1.0 / math.log2(idx + 1.0)
    return dcg


def _ndcg_at_k(ranked_ids: Sequence[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(idx + 1.0) for idx in range(1, ideal_hits + 1))
    if idcg <= 0.0:
        return 0.0
    return _dcg_at_k(ranked_ids, relevant, k) / idcg


def bootstrap_ci_95(
    values: Sequence[float],
    *,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Deterministic percentile 95% bootstrap confidence interval for the sample mean."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        val = round(float(values[0]), 6)
        return (val, val)
    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(max(50, int(n_resamples))):
        sample_sum = 0.0
        for _ in range(n):
            sample_sum += float(values[rng.randrange(n)])
        means.append(sample_sum / float(n))
    means.sort()
    lo_idx = int(0.025 * (len(means) - 1))
    hi_idx = int(0.975 * (len(means) - 1))
    return (round(means[lo_idx], 6), round(means[hi_idx], 6))


def paired_bootstrap_delta_ci_95(
    arm_values: Sequence[float],
    baseline_values: Sequence[float],
    *,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Paired 95% bootstrap CI on `mean(arm_values - baseline_values)` across identical queries."""
    n = min(len(arm_values), len(baseline_values))
    if n == 0:
        return (0.0, 0.0)
    diffs = [float(arm_values[i]) - float(baseline_values[i]) for i in range(n)]
    return bootstrap_ci_95(diffs, n_resamples=n_resamples, seed=seed)


def evaluate_retrieval_arm(
    chunks: Sequence[Mapping[str, Any]],
    gold_queries: Sequence[GoldQueryCase],
    arm: EmbedderArmSpec,
    *,
    allow_mock: bool = False,
    batch_gpu: bool = False,
    rrf_k: int = 60,
    rerank_top_m: int = 40,
    bootstrap_resamples: int = 1000,
    seed: int = 42,
    reranker: Optional[RerankerPort] = None,
) -> RetrievalMetrics:
    """Evaluate one embedder/reranker arm over a filtered chunk slice and gold query suite."""
    chunk_list = [dict(c) for c in chunks if str(c.get("chunk_id") or "").strip()]
    chunk_ids = [str(c["chunk_id"]) for c in chunk_list]
    chunk_texts = [str(c.get("text") or "") for c in chunk_list]

    if not chunk_list or not gold_queries:
        return RetrievalMetrics(
            arm_id=arm.arm_id,
            label=arm.label,
            live_deployable=arm.live_deployable,
            query_count=len(gold_queries),
            chunk_count=len(chunk_list),
            recall_at_5=0.0,
            recall_at_10=0.0,
            mrr_at_10=0.0,
            ndcg_at_10=0.0,
            p50_query_latency_ms=0.0,
            p95_query_latency_ms=0.0,
            embed_throughput_chunks_per_sec=0.0,
            recall_at_10_ci95=(0.0, 0.0),
            ndcg_at_10_ci95=(0.0, 0.0),
        )

    effective_mrl = arm.mrl_dim if arm.mrl_dim is not None else None
    t_embed_start = time.perf_counter()
    doc_vectors = embed_documents(
        chunk_texts,
        allow_mock=allow_mock,
        batch_gpu=batch_gpu,
        model_name=arm.embedder_model,
        mrl_dim=effective_mrl,
    )
    embed_elapsed = max(1e-6, time.perf_counter() - t_embed_start)
    throughput = round(len(chunk_texts) / embed_elapsed, 2)

    active_reranker: Optional[RerankerPort] = reranker
    if active_reranker is None and arm.use_reranker:
        active_reranker = get_reranker(
            model_name=arm.reranker_model,
            allow_mock=allow_mock,
            device="cpu",
        )

    recalls_5: List[float] = []
    recalls_10: List[float] = []
    mrrs_10: List[float] = []
    ndcgs_10: List[float] = []
    latencies_ms: List[float] = []

    for q_case in gold_queries:
        t_q = time.perf_counter()
        q_vec = embed_query(
            q_case.query,
            allow_mock=allow_mock,
            batch_gpu=False,  # Live query path is strictly CPU-pinned (ADR-00035 / ADR-00047)
            model_name=arm.embedder_model,
            mrl_dim=effective_mrl,
        )
        dense_ranked = sorted(
            ((_cosine_dot(q_vec, d_vec), chunk_ids[idx], idx) for idx, d_vec in enumerate(doc_vectors)),
            key=lambda item: (-item[0], item[1]),
        )
        lex_scores = _offline_bm25_scores(q_case.query, chunk_texts)
        lexical_ranked = sorted(
            ((lex_scores[idx], chunk_ids[idx], idx) for idx in range(len(chunk_texts)) if lex_scores[idx] > 0.0),
            key=lambda item: (-item[0], item[1]),
        )

        rrf_scores: Dict[int, float] = defaultdict(float)
        cand_cap = max(rerank_top_m, 40)
        for rank, (_, _, idx) in enumerate(dense_ranked[:cand_cap], start=1):
            rrf_scores[idx] += 1.0 / (rrf_k + rank)
        for rank, (_, _, idx) in enumerate(lexical_ranked[:cand_cap], start=1):
            rrf_scores[idx] += 1.0 / (rrf_k + rank)

        fused_indices = sorted(rrf_scores.keys(), key=lambda idx: (-rrf_scores[idx], chunk_ids[idx]))
        if active_reranker is not None and fused_indices:
            top_m_idx = fused_indices[: max(10, int(rerank_top_m))]
            r_scores = active_reranker.score_pairs(q_case.query, [chunk_texts[i] for i in top_m_idx])
            reranked_pairs = sorted(
                zip(top_m_idx, r_scores),
                key=lambda pair: (-float(pair[1]), -rrf_scores[pair[0]], chunk_ids[pair[0]]),
            )
            final_ids = [chunk_ids[idx] for idx, _ in reranked_pairs[:10]]
        else:
            final_ids = [chunk_ids[idx] for idx in fused_indices[:10]]

        latencies_ms.append((time.perf_counter() - t_q) * 1000.0)

        relevant = set(q_case.relevant_chunk_ids)
        hits_5 = sum(1 for cid in final_ids[:5] if cid in relevant)
        hits_10 = sum(1 for cid in final_ids[:10] if cid in relevant)
        recalls_5.append(hits_5 / float(len(relevant)))
        recalls_10.append(hits_10 / float(len(relevant)))
        rr = 0.0
        for rank, cid in enumerate(final_ids[:10], start=1):
            if cid in relevant:
                rr = 1.0 / float(rank)
                break
        mrrs_10.append(rr)
        ndcgs_10.append(_ndcg_at_k(final_ids, relevant, 10))

    n_q = float(len(gold_queries))
    return RetrievalMetrics(
        arm_id=arm.arm_id,
        label=arm.label,
        live_deployable=arm.live_deployable,
        query_count=len(gold_queries),
        chunk_count=len(chunk_list),
        recall_at_5=round(sum(recalls_5) / n_q, 6),
        recall_at_10=round(sum(recalls_10) / n_q, 6),
        mrr_at_10=round(sum(mrrs_10) / n_q, 6),
        ndcg_at_10=round(sum(ndcgs_10) / n_q, 6),
        p50_query_latency_ms=round(_percentile(latencies_ms, 50.0), 3),
        p95_query_latency_ms=round(_percentile(latencies_ms, 95.0), 3),
        embed_throughput_chunks_per_sec=throughput,
        recall_at_10_ci95=bootstrap_ci_95(recalls_10, n_resamples=bootstrap_resamples, seed=seed),
        ndcg_at_10_ci95=bootstrap_ci_95(ndcgs_10, n_resamples=bootstrap_resamples, seed=seed + 1),
        per_query_recall_at_10=tuple(recalls_10),
        per_query_ndcg_at_10=tuple(ndcgs_10),
    )


# ==============================================================================
# 5. LOFO DELTAS, PROMOTION GATES & END-TO-END STUDY RUNNER
# ==============================================================================


def compute_lofo_delta(
    full_cell: AblationCellResult,
    without_form_cell: AblationCellResult,
    *,
    primary_arm_id: str = "E1",
) -> Dict[str, float]:
    """Compute exact marginal contribution `Delta M(f) = M(F) - M(F \\ {f})`."""
    deltas: Dict[str, float] = {
        "edge_count": float(full_cell.topology.edge_count - without_form_cell.topology.edge_count),
        "edges_per_node": round(full_cell.topology.edges_per_node - without_form_cell.topology.edges_per_node, 6),
        "leiden_modularity_q": round(
            full_cell.topology.leiden_modularity_q - without_form_cell.topology.leiden_modularity_q, 6
        ),
        "degree_gini": round(full_cell.topology.degree_gini - without_form_cell.topology.degree_gini, 6),
        "mean_dropped_impact_share": round(
            full_cell.topology.mean_dropped_impact_share - without_form_cell.topology.mean_dropped_impact_share, 6
        ),
        "filing_grounded_share": round(
            full_cell.provenance.filing_grounded_share - without_form_cell.provenance.filing_grounded_share, 6
        ),
    }
    full_ret = full_cell.retrieval_by_arm.get(primary_arm_id)
    wo_ret = without_form_cell.retrieval_by_arm.get(primary_arm_id)
    if full_ret is not None and wo_ret is not None:
        deltas["recall_at_10"] = round(full_ret.recall_at_10 - wo_ret.recall_at_10, 6)
        deltas["mrr_at_10"] = round(full_ret.mrr_at_10 - wo_ret.mrr_at_10, 6)
        deltas["ndcg_at_10"] = round(full_ret.ndcg_at_10 - wo_ret.ndcg_at_10, 6)
    else:
        deltas["recall_at_10"] = 0.0
        deltas["mrr_at_10"] = 0.0
        deltas["ndcg_at_10"] = 0.0
    return deltas


def evaluate_form_promotion_gate(
    form_type: str,
    deltas: Mapping[str, float],
    *,
    form_quarantine_rate: float = 0.0,
) -> FormPromotionVerdict:
    """Apply pre-registered ADR-00047 Section 4.6 Gate #1 to a form's LOFO deltas."""
    reasons: List[str] = []
    has_positive_gain = (
        deltas.get("edges_per_node", 0.0) > 0.0
        or deltas.get("filing_grounded_share", 0.0) > 0.0
        or deltas.get("recall_at_10", 0.0) > 0.0
        or deltas.get("mrr_at_10", 0.0) > 0.0
    )
    if not has_positive_gain:
        reasons.append("No positive gain on operational coverage, FILING_GROUNDED share, or Recall@10/MRR@10.")
    if deltas.get("leiden_modularity_q", 0.0) < -MAX_MODULARITY_DROP:
        reasons.append(
            f"Leiden modularity Q dropped by {deltas['leiden_modularity_q']:.4f} (limit: -{MAX_MODULARITY_DROP:.2f})."
        )
    if deltas.get("mean_dropped_impact_share", 0.0) > MAX_DROPPED_IMPACT_INCREASE:
        reasons.append(
            f"mean_dropped_impact_share increased by +{deltas['mean_dropped_impact_share']:.4f} "
            f"(limit: +{MAX_DROPPED_IMPACT_INCREASE:.2f})."
        )
    if form_quarantine_rate > MAX_QUARANTINE_RATE_INCREASE:
        reasons.append(
            f"Per-form quarantine rate {form_quarantine_rate:.4f} exceeds limit {MAX_QUARANTINE_RATE_INCREASE:.2f}."
        )
    passed = len(reasons) == 0
    if passed:
        reasons.append("Passed all ADR-00047 topology, grounding, retrieval, and quarantine gates.")
    return FormPromotionVerdict(
        form_type=form_type,
        passed=passed,
        reasons=tuple(reasons),
        deltas=dict(deltas),
    )


def evaluate_arm_promotion_gate(
    metrics: RetrievalMetrics,
    baseline_e1: Optional[RetrievalMetrics],
) -> ArmPromotionVerdict:
    """Apply pre-registered ADR-00047 Section 4.6 Gate #2 to an embedder/reranker arm."""
    reasons: List[str] = []
    if not metrics.live_deployable:
        reasons.append("Offline diagnostic ceiling arm only (non-deployable on live 8 GB VRAM host).")

    recall_delta = round(metrics.recall_at_10 - baseline_e1.recall_at_10, 6) if baseline_e1 else 0.0
    ndcg_delta = round(metrics.ndcg_at_10 - baseline_e1.ndcg_at_10, 6) if baseline_e1 else 0.0
    r_ci = metrics.delta_vs_e1_recall10_ci95
    n_ci = metrics.delta_vs_e1_ndcg10_ci95

    recall_sig = bool(r_ci is not None and r_ci[0] > 0.0 and recall_delta >= MIN_ARM_PRACTICAL_LIFT)
    ndcg_sig = bool(n_ci is not None and n_ci[0] > 0.0 and ndcg_delta >= MIN_ARM_PRACTICAL_LIFT)
    if metrics.arm_id == "E1":
        reasons.append("Production baseline reference arm.")
    elif not (recall_sig or ndcg_sig):
        reasons.append(
            f"Did not achieve statistically significant practical lift >= +{MIN_ARM_PRACTICAL_LIFT:.2f} "
            f"(delta Recall@10={recall_delta:+.4f}, delta nDCG@10={ndcg_delta:+.4f})."
        )

    if metrics.p95_query_latency_ms > MAX_LIVE_CPU_P95_LATENCY_MS:
        reasons.append(
            f"Live CPU P95 query latency {metrics.p95_query_latency_ms:.1f} ms exceeds "
            f"{MAX_LIVE_CPU_P95_LATENCY_MS:.1f} ms budget."
        )

    passed = metrics.live_deployable and metrics.arm_id != "E1" and len(reasons) == 0
    if passed:
        reasons.append("Passed ADR-00047 practical lift, paired bootstrap CI, and CPU latency gates.")
    return ArmPromotionVerdict(
        arm_id=metrics.arm_id,
        live_deployable=metrics.live_deployable,
        passed=passed,
        reasons=tuple(reasons),
        deltas={
            "recall_at_10_delta": recall_delta,
            "ndcg_at_10_delta": ndcg_delta,
            "delta_vs_e1_recall10_ci95": list(r_ci) if r_ci is not None else None,
            "delta_vs_e1_ndcg10_ci95": list(n_ci) if n_ci is not None else None,
            "p95_query_latency_ms": metrics.p95_query_latency_ms,
        },
    )


def run_kg_form_ablation_study(
    *,
    snapshot_dir: Path | str,
    as_of: date | str | PiTContext,
    gold_queries: Sequence[GoldQueryCase] = (),
    rejection_dir: Optional[Path | str] = None,
    forms: Optional[Sequence[str]] = None,
    graph_isolations: Sequence[str] = ("layered", "flat"),
    arm_ids: Sequence[str] = ("E1", "E2", "E3"),
    run_lofo: bool = True,
    allow_mock: bool = False,
    batch_gpu: bool = False,
    bootstrap_resamples: int = 1000,
    seed: int = 42,
    shock_origins: Optional[Sequence[str]] = None,
    include_unanchored: bool = False,
) -> Dict[str, Any]:
    """Execute the full controlled KG Form-Ablation & Embedder/Reranker Scaling Study (ADR-00047)."""
    ctx = as_of if isinstance(as_of, PiTContext) else PiTContext(
        as_of=as_of if isinstance(as_of, date) else date.fromisoformat(str(as_of).strip()[:10]),
        mode="historical",
    )
    nodes, edges, chunks = load_snapshot_tables(snapshot_dir)

    discovered_forms: List[str] = []
    if forms is not None:
        discovered_forms = list(dict.fromkeys(normalize_form_type(f) for f in forms if str(f).strip()))
    else:
        seen_forms: Set[str] = set()
        for rec in (*edges, *chunks):
            f_code, _ = _resolve_record_form_and_layer(rec)
            if f_code:
                seen_forms.add(f_code)
        discovered_forms = sorted(seen_forms)

    selected_arms: List[EmbedderArmSpec] = []
    for aid in arm_ids:
        spec = DEFAULT_EMBEDDER_ARMS.get(aid)
        if spec is None:
            raise ValueError(f"Unknown embedder arm '{aid}'. Valid arms: {sorted(DEFAULT_EMBEDDER_ARMS)}")
        selected_arms.append(spec)

    def _eval_cell(
        cell_id: str,
        subset_name: str,
        subset_forms: Sequence[str],
        isolation: str,
        arms_to_run: Sequence[EmbedderArmSpec],
    ) -> AblationCellResult:
        f_nodes, f_edges, f_chunks = filter_snapshot_by_forms_and_layer(
            nodes,
            edges,
            chunks,
            allowed_forms=subset_forms,
            graph_isolation=isolation,
            as_of=ctx,
            include_unanchored=include_unanchored,
        )
        topo, prov = compute_topology_and_provenance(
            f_nodes,
            f_edges,
            f_chunks,
            shock_origins=shock_origins,
            random_seed=seed,
        )
        fidelity = compute_extraction_fidelity(
            rejection_dir,
            f_chunks,
            allowed_forms=subset_forms,
        )
        raw_retrieval: Dict[str, RetrievalMetrics] = {}
        for arm in arms_to_run:
            raw_retrieval[arm.arm_id] = evaluate_retrieval_arm(
                f_chunks,
                gold_queries,
                arm,
                allow_mock=allow_mock,
                batch_gpu=batch_gpu,
                bootstrap_resamples=bootstrap_resamples,
                seed=seed,
            )

        # Attach paired bootstrap CIs relative to E1 baseline on the exact same chunk slice
        baseline_e1 = raw_retrieval.get("E1")
        retrieval_final: Dict[str, RetrievalMetrics] = {}
        for aid, r_met in raw_retrieval.items():
            if baseline_e1 is not None and aid != "E1" and r_met.per_query_recall_at_10:
                d_r_ci = paired_bootstrap_delta_ci_95(
                    r_met.per_query_recall_at_10,
                    baseline_e1.per_query_recall_at_10,
                    n_resamples=bootstrap_resamples,
                    seed=seed + 7,
                )
                d_n_ci = paired_bootstrap_delta_ci_95(
                    r_met.per_query_ndcg_at_10,
                    baseline_e1.per_query_ndcg_at_10,
                    n_resamples=bootstrap_resamples,
                    seed=seed + 13,
                )
                retrieval_final[aid] = RetrievalMetrics(
                    **{
                        **asdict(r_met),
                        "delta_vs_e1_recall10_ci95": d_r_ci,
                        "delta_vs_e1_ndcg10_ci95": d_n_ci,
                    }
                )
            else:
                retrieval_final[aid] = r_met

        return AblationCellResult(
            cell_id=cell_id,
            subset_name=subset_name,
            forms_included=tuple(subset_forms),
            graph_isolation=isolation,
            topology=topo,
            provenance=prov,
            extraction_fidelity=fidelity,
            retrieval_by_arm=retrieval_final,
        )

    cells: Dict[str, AblationCellResult] = {}
    for iso in graph_isolations:
        iso_key = str(iso).strip().lower()
        full_id = f"full__{iso_key}"
        cells[full_id] = _eval_cell(
            cell_id=full_id,
            subset_name="FULL",
            subset_forms=discovered_forms,
            isolation=iso_key,
            arms_to_run=selected_arms,
        )

    lofo_deltas: Dict[str, Dict[str, float]] = {}
    form_verdicts: Dict[str, FormPromotionVerdict] = {}
    primary_iso = "layered" if "layered" in [g.lower() for g in graph_isolations] else str(graph_isolations[0]).lower()
    primary_full = cells[f"full__{primary_iso}"]
    primary_arm_only = [a for a in selected_arms if a.arm_id == "E1"] or selected_arms[:1]

    if run_lofo and len(discovered_forms) >= 2:
        for form_code in discovered_forms:
            remaining = [f for f in discovered_forms if f != form_code]
            lofo_id = f"lofo_without_{form_code.replace('/', '_').replace(' ', '_')}__{primary_iso}"
            wo_cell = _eval_cell(
                cell_id=lofo_id,
                subset_name=f"WITHOUT_{form_code}",
                subset_forms=remaining,
                isolation=primary_iso,
                arms_to_run=primary_arm_only,
            )
            cells[lofo_id] = wo_cell
            deltas = compute_lofo_delta(primary_full, wo_cell, primary_arm_id=primary_arm_only[0].arm_id)
            lofo_deltas[form_code] = deltas
            form_q_rate = primary_full.extraction_fidelity.quarantine_rate_by_form.get(form_code, 0.0)
            form_verdicts[form_code] = evaluate_form_promotion_gate(
                form_code,
                deltas,
                form_quarantine_rate=form_q_rate,
            )

    arm_verdicts: Dict[str, ArmPromotionVerdict] = {}
    baseline_e1_metrics = primary_full.retrieval_by_arm.get("E1")
    for aid, r_met in primary_full.retrieval_by_arm.items():
        arm_verdicts[aid] = evaluate_arm_promotion_gate(r_met, baseline_e1_metrics)

    return {
        "adr_id": "ADR-00047",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of": ctx.as_of_iso,
        "pit_mode": ctx.mode,
        "embedding_dim": get_embedding_dim(),
        "mock_mode": allow_mock,
        "forms_evaluated": discovered_forms,
        "graph_isolations": [str(g).lower() for g in graph_isolations],
        "arms_evaluated": [a.to_dict() for a in selected_arms],
        "cells": {cid: cell.to_dict() for cid, cell in cells.items()},
        "lofo_deltas": lofo_deltas,
        "form_promotion_verdicts": {k: v.to_dict() for k, v in form_verdicts.items()},
        "arm_promotion_verdicts": {k: v.to_dict() for k, v in arm_verdicts.items()},
    }


# ==============================================================================
# 6. ARTIFACT & PLOT RENDERING (JSON + MARKDOWN + HEADLESS MATPLOTLIB)
# ==============================================================================


def _build_markdown_report(scorecard: Mapping[str, Any]) -> str:
    lines: List[str] = [
        "# ADR-00047 Controlled KG Form-Ablation & Embedder Scorecard",
        "",
        f"- **As-Of PiT Cutoff**: `{scorecard.get('as_of')}` (`mode={scorecard.get('pit_mode')}`)",
        f"- **Vector Dimension**: `{scorecard.get('embedding_dim')}` (constant `VECTOR(1024)` parity)",
        f"- **Mock Mode**: `{scorecard.get('mock_mode')}`",
        f"- **Forms Evaluated**: `{', '.join(scorecard.get('forms_evaluated') or [])}`",
        "",
        "## 1. Topology & Provenance Across Graph Isolation Modes",
        "",
        "| Cell | Isolation | Nodes | Edges | Modularity Q | Max Deg | Gini | Dropped Impact | FILING_GROUNDED |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cid, cell in (scorecard.get("cells") or {}).items():
        topo = cell["topology"]
        prov = cell["provenance"]
        lines.append(
            f"| `{cid}` | `{cell['graph_isolation']}` | {topo['node_count']} | {topo['edge_count']} | "
            f"{topo['leiden_modularity_q']:.4f} | {topo['max_degree']} | {topo['degree_gini']:.4f} | "
            f"{topo['mean_dropped_impact_share']:.4f} | {prov['filing_grounded_share']:.4f} |"
        )

    lofo = scorecard.get("lofo_deltas") or {}
    if lofo:
        lines.extend(
            [
                "",
                "## 2. Leave-One-Form-Out (LOFO) Marginal Deltas & Promotion Gate",
                "",
                "| SEC Form | dEdges | dModularity Q | dDropped Impact | dFILING_GROUNDED | dRecall@10 | Gate |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | :---: |",
            ]
        )
        verdicts = scorecard.get("form_promotion_verdicts") or {}
        for form_code, d in lofo.items():
            passed = bool((verdicts.get(form_code) or {}).get("passed", False))
            lines.append(
                f"| `{form_code}` | {d.get('edge_count', 0.0):+.0f} | {d.get('leiden_modularity_q', 0.0):+.4f} | "
                f"{d.get('mean_dropped_impact_share', 0.0):+.4f} | {d.get('filing_grounded_share', 0.0):+.4f} | "
                f"{d.get('recall_at_10', 0.0):+.4f} | {'PASS' if passed else 'HOLD'} |"
            )

    arm_verdicts = scorecard.get("arm_promotion_verdicts") or {}
    primary_cells = [c for cid, c in (scorecard.get("cells") or {}).items() if cid.startswith("full__")]
    if primary_cells and primary_cells[0].get("retrieval_by_arm"):
        lines.extend(
            [
                "",
                "## 3. Embedder & Reranker Scaling Arms",
                "",
                "| Arm | Live Deployable | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | P95 Latency (ms) | Gate |",
                "| :--- | :---: | ---: | ---: | ---: | ---: | ---: | :---: |",
            ]
        )
        for aid, r in primary_cells[0]["retrieval_by_arm"].items():
            verdict = arm_verdicts.get(aid) or {}
            lines.append(
                f"| `{aid}` | `{r['live_deployable']}` | {r['recall_at_5']:.4f} | {r['recall_at_10']:.4f} | "
                f"{r['mrr_at_10']:.4f} | {r['ndcg_at_10']:.4f} | {r['p95_query_latency_ms']:.2f} | "
                f"{'PROMOTE' if verdict.get('passed') else 'BASELINE/HOLD'} |"
            )

    lines.append("")
    return "\n".join(lines)


def render_ablation_artifacts(
    scorecard: Mapping[str, Any],
    output_dir: Path | str,
) -> Dict[str, str]:
    """Write `ablation_scorecard.json`, `ablation_report.md`, and comparison PNG plots."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "ablation_scorecard.json"
    json_path.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")

    md_path = out / "ablation_report.md"
    md_path.write_text(_build_markdown_report(scorecard), encoding="utf-8")

    artifacts: Dict[str, str] = {
        "scorecard_json": str(json_path),
        "report_md": str(md_path),
    }

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("Skipping ADR-00047 plot rendering (matplotlib unavailable: %s)", exc)
        return artifacts

    # Plot 1: LOFO Delta Heatmap
    lofo_deltas = scorecard.get("lofo_deltas") or {}
    heatmap_path = out / "lofo_delta_heatmap.png"
    fig, ax = plt.subplots(figsize=(8.0, max(3.2, 0.7 * max(1, len(lofo_deltas)) + 1.5)))
    metrics_cols = ["leiden_modularity_q", "mean_dropped_impact_share", "filing_grounded_share", "recall_at_10"]
    col_labels = ["dModularity Q", "dDropped Impact", "dFiling Grounded", "dRecall@10"]
    if lofo_deltas:
        row_labels = list(lofo_deltas.keys())
        matrix = [[float(lofo_deltas[f].get(m, 0.0)) for m in metrics_cols] for f in row_labels]
        vmax = max(0.05, max(abs(v) for row in matrix for v in row))
        im = ax.imshow(matrix, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=15, ha="right")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        for r_idx, row in enumerate(matrix):
            for c_idx, val in enumerate(row):
                ax.text(c_idx, r_idx, f"{val:+.3f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, "Single-form snapshot (no LOFO deltas)", ha="center", va="center")
        ax.axis("off")
    ax.set_title("ADR-00047 Leave-One-Form-Out (LOFO) Marginal Deltas")
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=140)
    plt.close(fig)
    artifacts["lofo_delta_heatmap_png"] = str(heatmap_path)

    # Plot 2: Modularity Q vs. Dropped Impact Share across Cells
    mod_path = out / "modularity_vs_dropped_impact.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for cid, cell in (scorecard.get("cells") or {}).items():
        topo = cell["topology"]
        x_val = float(topo["mean_dropped_impact_share"])
        y_val = float(topo["leiden_modularity_q"])
        color = "#1f77b4" if cell["graph_isolation"] == "layered" else "#d62728"
        ax.scatter([x_val], [y_val], s=70, color=color, alpha=0.85)
        ax.annotate(cid, (x_val, y_val), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("Mean Cascade Dropped Impact Share (lower is better)")
    ax.set_ylabel("Weighted Leiden Modularity Q (higher is better)")
    ax.set_title("Topology Health: Modularity Q vs. Subgraph Truncation Loss")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(mod_path, dpi=140)
    plt.close(fig)
    artifacts["modularity_vs_dropped_impact_png"] = str(mod_path)

    # Plot 3: Embedder / Reranker Recall@10 vs. Live CPU P95 Latency
    emb_path = out / "embedder_recall_vs_latency.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    full_cells = [c for cid, c in (scorecard.get("cells") or {}).items() if cid.startswith("full__")]
    if full_cells and full_cells[0].get("retrieval_by_arm"):
        for aid, r in full_cells[0]["retrieval_by_arm"].items():
            x_lat = float(r["p95_query_latency_ms"])
            y_rec = float(r["recall_at_10"])
            marker = "o" if r["live_deployable"] else "D"
            color = "#2ca02c" if r["live_deployable"] else "#7f7f7f"
            ax.scatter([x_lat], [y_rec], s=90, marker=marker, color=color)
            ax.annotate(aid, (x_lat, y_rec), textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.axvline(MAX_LIVE_CPU_P95_LATENCY_MS, color="#d62728", linestyle="--", label="250 ms CPU gate")
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "No gold queries supplied", ha="center", va="center")
    ax.set_xlabel("Live CPU P95 Query Latency (ms)")
    ax.set_ylabel("Retrieval Recall@10")
    ax.set_title("Embedder & Reranker Pareto Frontier (ADR-00047)")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(emb_path, dpi=140)
    plt.close(fig)
    artifacts["embedder_recall_vs_latency_png"] = str(emb_path)

    return artifacts


__all__ = [
    "AblationCellResult",
    "ArmPromotionVerdict",
    "DEFAULT_EMBEDDER_ARMS",
    "EmbedderArmSpec",
    "ExtractionFidelityMetrics",
    "FormPromotionVerdict",
    "GoldQueryCase",
    "MAX_DROPPED_IMPACT_INCREASE",
    "MAX_LIVE_CPU_P95_LATENCY_MS",
    "MAX_MODULARITY_DROP",
    "MAX_QUARANTINE_RATE_INCREASE",
    "MIN_ARM_PRACTICAL_LIFT",
    "ProvenanceMetrics",
    "RetrievalMetrics",
    "TopologyMetrics",
    "bootstrap_ci_95",
    "compute_extraction_fidelity",
    "compute_lofo_delta",
    "compute_topology_and_provenance",
    "evaluate_arm_promotion_gate",
    "evaluate_form_promotion_gate",
    "evaluate_retrieval_arm",
    "filter_snapshot_by_forms_and_layer",
    "load_gold_queries",
    "load_snapshot_tables",
    "paired_bootstrap_delta_ci_95",
    "render_ablation_artifacts",
    "run_kg_form_ablation_study",
]
