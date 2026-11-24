"""Single Business Intelligence Read Facade & SubgraphStorePort (ADR-00034).

Consolidates all Point-in-Time graph read paths behind a single facade (`query_bi`)
and unified `SubgraphStorePort` protocol implemented by both live `MemgraphSubgraphStore`
and offline `InMemorySubgraphStore`.

Invariants enforced (ADR-00033 / ADR-00034):
1. Single entry: outside `retrieval/facade.py` and `retrieval/pipeline.py`, no module in
   `retrieval/`, `synthesis/`, `explorer/`, or `tools/` executes Memgraph read Cypher directly.
2. Mandatory `PiTContext`: `ctx` has no default and cannot be `Optional`.
3. Store parity: `MemgraphSubgraphStore` and `InMemorySubgraphStore` normalize and sort edges
   identically for the same universe and `PiTContext`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    runtime_checkable,
)

import knowledge_graph.memgraph_driver as _memgraph_driver
from knowledge_graph.errors import GraphStoreUnavailableError, StoreUnavailableError
from retrieval.guardrails.schema_guardrail import CypherASTWhitelistGuardrail
from retrieval.offline_store import InMemorySubgraphStore, _normalize_edge_record
from retrieval.routing import ResolvedRoute, RouteMode, classify_query_route
from retrieval.temporal import DEFAULT_TEMPORAL_FILTER, PiTContext, TemporalQueryFilter, validate_hops

if TYPE_CHECKING:
    from retrieval.hybrid import HybridSeeder
    from retrieval.pipeline import HopOrderedSubgraphVerbalizer, PipelineResult
    from contagion import SparseNetworkCascadeEngine
    from sft_pipeline.rejection_queue import RuntimeRejectionQueue
    from sft_pipeline.serving.model_client import YarnBallModelClient

logger = logging.getLogger("retrieval.facade")


def _record_to_dict(record: Any) -> Dict[str, Any]:
    """Convert a Memgraph / Neo4j driver record or mapping into a plain dict."""
    if isinstance(record, dict):
        return dict(record)
    if hasattr(record, "data"):
        data = record.data()
        if isinstance(data, dict):
            return dict(data)
    try:
        return dict(record)
    except (TypeError, ValueError):
        return {"raw": str(record)}


def _normalize_memgraph_record(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize a Memgraph edge record identically to `InMemorySubgraphStore`."""
    if "source" in raw and "target" in raw:
        props_raw = raw.get("props") or raw.get("properties")
        props: Dict[str, Any] = dict(props_raw) if isinstance(props_raw, Mapping) else {}
        merged: Dict[str, Any] = {**props, **raw}
        merged.pop("props", None)
        merged.pop("properties", None)
        if "relation" in merged and "rel_type" not in merged:
            merged["rel_type"] = merged["relation"]
        norm = _normalize_edge_record(merged)
        norm["hop"] = int(raw.get("hop", props.get("hop", 1)) or 1)
        if raw.get("source_hash") and not norm.get("source_hash"):
            norm["source_hash"] = raw["source_hash"]
        return norm

    if "r" in raw and isinstance(raw["r"], Mapping):
        rel = dict(raw["r"])
        s_raw = raw.get("s")
        t_raw = raw.get("t")
        s_node: Mapping[str, Any] = s_raw if isinstance(s_raw, Mapping) else {}
        t_node: Mapping[str, Any] = t_raw if isinstance(t_raw, Mapping) else {}
        merged = {
            **rel,
            "source": s_node.get("ticker") or s_node.get("id") or raw.get("source") or "UNKNOWN",
            "target": t_node.get("ticker") or t_node.get("id") or raw.get("target") or "UNKNOWN",
            "rel_type": rel.get("rel_type") or rel.get("type") or "SUPPLIES_TO",
        }
        norm = _normalize_edge_record(merged)
        norm["hop"] = int(raw.get("hop", 1) or 1)
        return norm

    return dict(raw)


def _edge_sort_key(edge: Mapping[str, Any]) -> Tuple[int, str, str, str, str]:
    return (
        int(edge.get("hop", 1) or 1),
        str(edge.get("source", "")),
        str(edge.get("target", "")),
        str(edge.get("rel_type", edge.get("relation", ""))),
        str(edge.get("input_group") or edge.get("nature") or ""),
    )


@runtime_checkable
class SubgraphStorePort(Protocol):
    """Unified Point-in-Time graph store port shared by live Memgraph and offline snapshots."""

    last_unanchored_excluded: int

    def traverse(
        self,
        seed_ids: Sequence[str],
        hops: int,
        ctx: PiTContext,
        *,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Execute directed, hop-validated, Point-in-Time k-hop traversal from `seed_ids`."""
        ...

    def execute_guarded(
        self,
        cypher: str,
        params: Optional[Mapping[str, Any]],
        ctx: PiTContext,
    ) -> List[Dict[str, Any]]:
        """Validate read-only Cypher via AST whitelist, inject PiT predicates, and execute."""
        ...


class MemgraphSubgraphStore:
    """Live Memgraph implementation of `SubgraphStorePort` (ADR-00034).

    Wraps Memgraph Cypher execution behind `CypherASTWhitelistGuardrail` and
    `TemporalQueryFilter` over the ephemeral in-memory Memgraph projection.
    """

    def __init__(
        self,
        driver: Optional[Any] = None,
        temporal_filter: Optional[TemporalQueryFilter] = None,
        schema_guardrail: Optional[CypherASTWhitelistGuardrail] = None,
    ) -> None:
        self._driver = driver
        self.temporal_filter = temporal_filter or TemporalQueryFilter()
        self.schema_guardrail = schema_guardrail or CypherASTWhitelistGuardrail()
        self.last_unanchored_excluded: int = 0

    def _resolve_driver(self) -> Any:
        if self._driver is not None:
            return self._driver
        return _memgraph_driver.get_memgraph_driver()

    def traverse(
        self,
        seed_ids: Sequence[str],
        hops: int,
        ctx: PiTContext,
        *,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Traverse directed `[r*1..H]->` paths from `seed_ids` filtered strictly by `ctx`."""
        if not isinstance(ctx, PiTContext):
            raise TypeError(f"ctx must be a PiTContext instance, got {type(ctx).__name__}")

        h = validate_hops(hops)
        safe_limit = max(1, min(int(limit), 500))
        cleaned_seeds = list(dict.fromkeys(str(s).strip() for s in seed_ids if str(s).strip()))
        if not cleaned_seeds:
            self.last_unanchored_excluded = 0
            return []

        base_query = f"""
            MATCH p = (seed)-[r*1..{h}]->(end_node)
            WHERE seed.id IN $seeds OR seed.ticker IN $seeds
            WITH relationships(p) AS path_rels, length(p) AS hop_dist
            UNWIND path_rels AS rel
            RETURN DISTINCT coalesce(startNode(rel).ticker, startNode(rel).id) AS source,
                   startNode(rel).id AS source_id,
                   type(rel) AS rel_type,
                   coalesce(endNode(rel).ticker, endNode(rel).id) AS target,
                   endNode(rel).id AS target_id,
                   properties(rel) AS props,
                   rel.source_hash AS source_hash,
                   min(hop_dist) AS hop
            ORDER BY hop ASC, source ASC, target ASC, rel_type ASC
            LIMIT $limit
        """
        scoped_query = self.temporal_filter.inject_active_coalescence(base_query, ctx)
        driver = self._resolve_driver()
        combined: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}

        with driver.session() as session:
            records = session.run(
                scoped_query,
                seeds=cleaned_seeds,
                limit=safe_limit,
                as_of_date=ctx.as_of_iso,
            )
            for rec in records:
                row = _record_to_dict(rec)
                if not row:
                    continue
                if "source" in row and "target" in row:
                    norm = _normalize_memgraph_record(row)
                    grp = str(norm.get("input_group") or norm.get("nature") or "default")
                    ekey = (norm["source"], norm["rel_type"], norm["target"], grp)
                    if ekey not in combined or int(norm["hop"]) < int(combined[ekey]["hop"]):
                        combined[ekey] = norm

        self.last_unanchored_excluded = int(getattr(driver, "last_unanchored_excluded", 0) or 0)
        ordered = sorted(combined.values(), key=_edge_sort_key)
        return ordered[:safe_limit]

    def execute_guarded(
        self,
        cypher: str,
        params: Optional[Mapping[str, Any]],
        ctx: PiTContext,
    ) -> List[Dict[str, Any]]:
        """Validate read-only Cypher via AST whitelist, inject PiT predicates, and execute."""
        if not isinstance(ctx, PiTContext):
            raise TypeError(f"ctx must be a PiTContext instance, got {type(ctx).__name__}")

        validation = self.schema_guardrail.validate(cypher)
        if not validation.is_valid:
            raise ValueError(f"Cypher guardrail violation: {validation.error_summary}")

        sanitized = validation.sanitized_query or cypher
        scoped_query = self.temporal_filter.inject_active_coalescence(sanitized, ctx)
        query_params: Dict[str, Any] = dict(params or {})
        query_params.setdefault("as_of_date", ctx.as_of_iso)

        driver = self._resolve_driver()
        out: List[Dict[str, Any]] = []
        all_edges = True

        with driver.session() as session:
            records = session.run(scoped_query, **query_params)
            for rec in records:
                row = _record_to_dict(rec)
                if ("source" in row and "target" in row) or ("r" in row and isinstance(row["r"], Mapping)):
                    out.append(_normalize_memgraph_record(row))
                else:
                    all_edges = False
                    out.append(row)

        self.last_unanchored_excluded = int(getattr(driver, "last_unanchored_excluded", 0) or 0)
        if all_edges and out:
            out.sort(key=_edge_sort_key)
        return out

    def fetch_node_details(
        self,
        node_ids: Sequence[str],
        ctx: PiTContext,
        *,
        hops: int = 1,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Pull bounded, undirected, Point-in-Time node-detail ego cards (node-detail display only)."""
        if not isinstance(ctx, PiTContext):
            raise TypeError(f"ctx must be a PiTContext instance, got {type(ctx).__name__}")
        cleaned_ids = [str(nid).strip() for nid in node_ids if str(nid).strip()]
        if not cleaned_ids:
            return []

        validated_hops = validate_hops(hops)
        safe_limit = max(1, min(int(limit), 500))
        driver = self._resolve_driver()

        base_query = f"""
            UNWIND $ids AS nid
            MATCH (n {{id: nid}})
            OPTIONAL MATCH (n)-[r*1..{validated_hops}]-(m)
            RETURN nid,
                   collect(DISTINCT n) AS nodes,
                   collect(DISTINCT r) AS rels,
                   collect(DISTINCT m) AS neighbors
            ORDER BY nid ASC
            LIMIT $limit;
        """
        scoped_query = DEFAULT_TEMPORAL_FILTER.inject_active_coalescence(base_query, ctx)

        with driver.session() as session:
            result = session.run(
                scoped_query,
                ids=cleaned_ids,
                limit=safe_limit,
                as_of_date=ctx.as_of_iso,
            )
            payload: List[Dict[str, Any]] = []
            for record in result:
                payload.append(
                    {
                        "root_id": record["nid"],
                        "nodes": [dict(r) for r in record["nodes"]],
                        "relationships": [dict(r) for r in record["rels"]],
                        "neighbors": [dict(r) for r in record["neighbors"]],
                    }
                )
            return payload

    def introspect_schema(
        self,
        *,
        default_labels: Sequence[str],
        default_rel_types: Sequence[str],
    ) -> Dict[str, List[str]]:
        """Read distinct node labels and relationship type names from Memgraph (schema metadata only)."""
        labels: Set[str] = set(default_labels)
        rel_types: Set[str] = set(default_rel_types)
        properties: Set[str] = {"id", "ticker", "name", "sector", "aliases", "source_hash"}

        driver = self._resolve_driver()
        try:
            with driver.session() as session:
                res_lbl = session.run("MATCH (n) RETURN DISTINCT labels(n) AS lbls LIMIT 100")
                for r in res_lbl:
                    for lbl in r.get("lbls", []):
                        if lbl:
                            labels.add(str(lbl))

                res_rel = session.run("MATCH ()-[r]->() RETURN DISTINCT type(r) AS rel LIMIT 100")
                for r in res_rel:
                    rel = r.get("rel")
                    if rel:
                        rel_types.add(str(rel))
        except StoreUnavailableError:
            raise
        except Exception as exc:
            raise GraphStoreUnavailableError(
                f"Failed to introspect active Memgraph schema: {exc}"
            ) from exc

        return {
            "labels": sorted(labels),
            "relationship_types": sorted(rel_types),
            "properties": sorted(properties),
        }


def query_subgraph_ego(
    store: InMemorySubgraphStore,
    ctx: PiTContext,
    *,
    ticker: Optional[str] = None,
    max_hops: int = 2,
    direction: str = "both",
    rel_types: Optional[Sequence[str]] = None,
    required_status: Optional[str] = None,
) -> Dict[str, Any]:
    """Read-only Point-in-Time ego-subgraph view facade for the local Workbench explorer."""
    if not isinstance(ctx, PiTContext):
        raise TypeError(f"ctx must be a PiTContext instance, got {type(ctx).__name__}")
    return store.query_ego_subgraph(
        ctx=ctx,
        ticker=ticker,
        max_hops=max_hops,
        direction=direction,
        rel_types=rel_types,
        required_status=required_status,
    )


def query_bi(
    question: str,
    ctx: PiTContext,
    *,
    store: Optional[SubgraphStorePort] = None,
    seeder: Optional["HybridSeeder"] = None,
    model_client: Optional["YarnBallModelClient"] = None,
    rejection_queue: Optional["RuntimeRejectionQueue"] = None,
    cascade_engine: Optional["SparseNetworkCascadeEngine"] = None,
    verbalizer: Optional["HopOrderedSubgraphVerbalizer"] = None,
    schema_guardrail: Optional[CypherASTWhitelistGuardrail] = None,
    temporal_filter: Optional[TemporalQueryFilter] = None,
    hops: int = 2,
    target_ticker: Optional[str] = None,
    shock_origin: Optional[str] = None,
    shock_magnitude_pct: Optional[float] = None,
    initial_shocks: Optional[Dict[str, float]] = None,
    initial_durations: Optional[Dict[str, float]] = None,
    initial_onsets: Optional[Dict[str, float]] = None,
    route: RouteMode = "auto",
    universe_id: str = "sp500_pit",
    communities_parquet_path: Optional[Path | str] = None,
    community_reports: Optional[Sequence[Any]] = None,
    linker: Optional[Any] = None,
) -> "PipelineResult":
    """Single public Business Intelligence read entry point (ADR-00034 / ADR-00037).

    Supports deterministic rules-first routing across:
    - `local`: ego traversal, deterministic cascade propagation, and impact-ranked verbalization.
    - `global`: Point-in-Time map-reduce over cached `community_reports` (`as_of <= ctx.as_of`).
    - `hybrid`: `local` cascade augmented with top-matching PiT community summaries.
    """
    if not isinstance(ctx, PiTContext):
        raise TypeError(f"ctx must be a PiTContext instance, got {type(ctx).__name__}")

    from knowledge_graph.communities import search_community_reports_global
    from retrieval.pipeline import AnalystQuery, GraphRAGPipelineOrchestrator, PipelineResult

    resolved_route, route_meta = classify_query_route(question, route=route, linker=linker)

    def _degraded_outcome(exc: Exception, active_route: ResolvedRoute) -> PipelineResult:
        logger.error("query_bi returning explicit degraded outcome due to store outage: %s", exc)
        return PipelineResult(
            query=question,
            as_of_date=ctx.as_of_iso,
            route_used=active_route,
            cypher_query=None,
            is_valid_query=False,
            degraded=True,
            degraded_reasons=[f"store_unavailable: {exc}"],
            refusal_reason=str(exc),
            retrieved_subgraph=[],
            seed_hits=[],
            articles=[],
            verbalized_subgraph=f"DEGRADED: Backing store unavailable ({exc}). Exposure cannot be evaluated.",
            contagion_verdict=f"DEGRADED: Backing store unavailable ({exc}). Exposure cannot be determined.",
            contagion_think_trace=None,
            latency_breakdown_ms={},
            total_latency_ms=0.0,
        )

    if resolved_route == "global":
        try:
            global_res = search_community_reports_global(
                question,
                ctx=ctx,
                universe_id=universe_id,
                communities=community_reports,
                communities_parquet_path=communities_parquet_path,
                model_client=model_client,
            )
            return PipelineResult(
                query=question,
                as_of_date=ctx.as_of_iso,
                route_used="global",
                cypher_query=None,
                is_valid_query=True,
                degraded=False,
                degraded_reasons=[],
                refusal_reason=None,
                retrieved_subgraph=[],
                seed_hits=[],
                articles=[],
                community_reports=list(global_res.get("communities", [])),
                verbalized_subgraph=str(global_res.get("verbalized_context") or ""),
                contagion_verdict=str(global_res.get("synthesis") or ""),
                contagion_think_trace=global_res.get("think_trace"),
                latency_breakdown_ms={},
                total_latency_ms=0.0,
            )
        except StoreUnavailableError as exc:
            return _degraded_outcome(exc, "global")

    validated_hops = validate_hops(hops)
    active_store: SubgraphStorePort = store if store is not None else MemgraphSubgraphStore()

    orchestrator = GraphRAGPipelineOrchestrator(
        model_client=model_client,
        schema_guardrail=schema_guardrail,
        temporal_filter=temporal_filter,
        rejection_queue=rejection_queue,
        store=active_store,
        seeder=seeder,
        verbalizer=verbalizer,
        cascade_engine=cascade_engine,
        seed_hops=validated_hops,
    )
    analyst_query = AnalystQuery(
        query=question,
        ctx=ctx,
        target_ticker=target_ticker,
        shock_origin=shock_origin,
        shock_magnitude_pct=shock_magnitude_pct,
        initial_shocks=initial_shocks,
        initial_durations=initial_durations,
        initial_onsets=initial_onsets,
    )
    try:
        result = orchestrator.run(analyst_query)
    except StoreUnavailableError as exc:
        return _degraded_outcome(exc, resolved_route)

    result.route_used = resolved_route

    if resolved_route == "hybrid" and not result.degraded:
        try:
            hybrid_comm = search_community_reports_global(
                question,
                ctx=ctx,
                universe_id=universe_id,
                top_k=3,
                communities=community_reports,
                communities_parquet_path=communities_parquet_path,
                model_client=None,
            )
            comm_list = list(hybrid_comm.get("communities", []))
            if comm_list:
                result.community_reports = comm_list
                comm_ctx = str(hybrid_comm.get("verbalized_context") or "").strip()
                if comm_ctx:
                    result.verbalized_subgraph = f"{result.verbalized_subgraph}\n\n{comm_ctx}"
        except StoreUnavailableError as exc:
            result.degraded = True
            result.degraded_reasons = [*result.degraded_reasons, f"community_store_unavailable: {exc}"]

    return result


__all__ = [
    "MemgraphSubgraphStore",
    "ResolvedRoute",
    "RouteMode",
    "SubgraphStorePort",
    "classify_query_route",
    "query_bi",
    "query_subgraph_ego",
]
