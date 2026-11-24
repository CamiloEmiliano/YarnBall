"""Hybrid Point-in-Time Seeder Engine (PostgreSQL pgvector + Lineage) (ADR-00034).

Replaces the legacy `HybridRetriever` traversal path with a focused seeding engine:
1. Performs Point-in-Time dense semantic entity search over SCD Type 2 `node_embeddings`.
2. Performs Point-in-Time dense passage search over `financial_news_queue.embedding`.
3. Resolves passage-to-entity links through durable PostgreSQL lineage
   (`node_embedding_sources` / `financial_news_queue`), avoiding assumptions that
   ephemeral Memgraph `:News` nodes remain resident beyond TTL (ADR-00036).
4. Optionally runs a guarded `TextToCQL` seed query through `SubgraphStorePort`.
5. Returns ranked `SeedHit(node_id, score, origin)` records to `GraphRAGPipelineOrchestrator`.

All graph traversal occurs exclusively through `SubgraphStorePort.traverse` in
`retrieval/facade.py` (`expand_subgraph` has been retired under ADR-00034).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from knowledge_graph.db import pg_connection
from .temporal import PiTContext
from .text_to_cql import TextToCQL

if TYPE_CHECKING:
    from .facade import SubgraphStorePort

logger = logging.getLogger("retrieval.hybrid")

SeedOrigin = Literal["dense_entity", "dense_passage", "text_to_cql"]


@dataclass(frozen=True)
class SeedHit:
    """Ranked seed entity candidate returned by `HybridSeeder` (ADR-00034)."""

    node_id: str
    score: float
    origin: SeedOrigin
    entity_type: Optional[str] = None
    source_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "score": self.score,
            "origin": self.origin,
            "entity_type": self.entity_type,
            "source_hash": self.source_hash,
        }


class HybridSeeder:
    """Point-in-Time dense entity, passage, and guarded-CQL seeder feeding `query_bi`."""

    def __init__(
        self,
        text_to_cql_engine: Optional[TextToCQL] = None,
        top_k_dense: int = 5,
    ) -> None:
        self.text_to_cql = text_to_cql_engine or TextToCQL()
        self.top_k_dense = top_k_dense

    def _embed_query_vector_literal(self, query: str) -> Optional[str]:
        from embedding.embedder import (
            EmbedderUnavailableError,
            InvalidEmbeddingError,
            embed_query,
            format_pgvector_literal,
        )

        try:
            vec = embed_query(query)
            if not vec:
                return None
            return format_pgvector_literal(vec)
        except (EmbedderUnavailableError, InvalidEmbeddingError):
            logger.error("Dense query embedding unavailable or invalid for query '%s'", query[:80])
            raise
        except Exception as exc:
            logger.warning("Embedding generation failed (%s); returning empty dense results", exc)
            return None

    # ----------------------------------------------------------------------
    # 1. Point-in-Time Dense Semantic Entity Search (SCD Type 2 pgvector)
    # ----------------------------------------------------------------------
    def search_dense_entities(
        self,
        query: str,
        ctx: PiTContext,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search PostgreSQL versioned `node_embeddings` strictly as of `ctx` (ADR-00033)."""
        if not isinstance(ctx, PiTContext):
            raise TypeError(f"ctx must be a PiTContext instance, got {type(ctx).__name__}")

        k = top_k or self.top_k_dense
        results: List[Dict[str, Any]] = []
        query_vector_literal = self._embed_query_vector_literal(query)
        if query_vector_literal is None:
            return results

        if ctx.mode == "live":
            sql = """
                SELECT node_id, entity_type, source_hash,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM node_embeddings
                WHERE valid_to IS NULL
                  AND valid_from IS NOT NULL
                  AND valid_from <= %s::date
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """
            params: Tuple[Any, ...] = (query_vector_literal, ctx.as_of_iso, query_vector_literal, k)
        else:
            sql = """
                SELECT node_id, entity_type, source_hash,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM node_embeddings
                WHERE COALESCE(is_unanchored, FALSE) = FALSE
                  AND valid_from IS NOT NULL
                  AND valid_from <= %s::date
                  AND (valid_to IS NULL OR valid_to > %s::date)
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """
            params = (
                query_vector_literal,
                ctx.as_of_iso,
                ctx.as_of_iso,
                query_vector_literal,
                k,
            )

        try:
            with pg_connection() as conn:
                cur = conn.cursor()
                cur.execute(sql, params)
                rows = cur.fetchall()
                for row in rows:
                    results.append(
                        {
                            "node_id": row[0],
                            "entity_type": row[1],
                            "source_hash": row[2],
                            "similarity": float(row[3]) if row[3] is not None else 1.0,
                        }
                    )
                if hasattr(cur, "close"):
                    cur.close()
        except Exception as exc:
            logger.warning("pgvector similarity search query failed: %s", exc)

        return results

    # ----------------------------------------------------------------------
    # 2. Point-in-Time Dense Passage Search & Durable Postgres Lineage Bridge
    # ----------------------------------------------------------------------
    def search_dense_passages(
        self,
        query: str,
        ctx: PiTContext,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Search PostgreSQL `financial_news_queue.embedding` bounded strictly by `ctx` (ADR-00033)."""
        if not isinstance(ctx, PiTContext):
            raise TypeError(f"ctx must be a PiTContext instance, got {type(ctx).__name__}")

        k = top_k or self.top_k_dense
        passages: List[Dict[str, Any]] = []
        query_vector_literal = self._embed_query_vector_literal(query)
        if query_vector_literal is None:
            return passages

        sql = """
            SELECT source_hash, title, source_url, published_at, raw_text, author,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM financial_news_queue
            WHERE embedding IS NOT NULL
              AND published_at IS NOT NULL
              AND published_at <= %s::timestamptz
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        params: Tuple[Any, ...] = (query_vector_literal, ctx.cutoff_utc_iso, query_vector_literal, k)

        try:
            with pg_connection() as conn:
                cur = conn.cursor()
                cur.execute(sql, params)
                for row in cur.fetchall():
                    raw_text = str(row[4] or "")
                    snippet = raw_text[:600] + ("..." if len(raw_text) > 600 else "")
                    passages.append(
                        {
                            "source_hash": row[0],
                            "title": row[1] or "Financial News Update",
                            "url": row[2] or "",
                            "published_at": str(row[3]) if row[3] else None,
                            "snippet": snippet,
                            "author": row[5] or "Financial News Desk",
                            "similarity": float(row[6]) if row[6] is not None else 1.0,
                        }
                    )
                if hasattr(cur, "close"):
                    cur.close()
        except Exception as exc:
            logger.warning("pgvector passage similarity search query failed: %s", exc)

        return passages

    def expand_passage_entities(
        self,
        source_hashes: Sequence[str],
        ctx: PiTContext,
        limit: int = 20,
    ) -> List[str]:
        """Resolve matched passage `source_hashes` to entity IDs via durable PostgreSQL lineage (ADR-00034)."""
        if not isinstance(ctx, PiTContext):
            raise TypeError(f"ctx must be a PiTContext instance, got {type(ctx).__name__}")

        valid_hashes = [str(h).strip() for h in source_hashes if str(h).strip()]
        if not valid_hashes:
            return []

        safe_limit = max(1, min(int(limit), 200))
        sql = """
            SELECT DISTINCT nes.node_id
            FROM node_embedding_sources nes
            LEFT JOIN financial_news_queue fnq
              ON fnq.source_hash = nes.source_hash
            WHERE nes.source_hash = ANY(%s)
              AND COALESCE(nes.source_published_at, fnq.published_at) IS NOT NULL
              AND COALESCE(nes.source_published_at, fnq.published_at) <= %s::timestamptz
            ORDER BY nes.node_id ASC
            LIMIT %s;
        """
        entity_ids: List[str] = []
        try:
            with pg_connection() as conn:
                cur = conn.cursor()
                cur.execute(sql, (valid_hashes, ctx.cutoff_utc_iso, safe_limit))
                for row in cur.fetchall():
                    if row and row[0]:
                        entity_ids.append(str(row[0]))
                if hasattr(cur, "close"):
                    cur.close()
        except Exception as exc:
            logger.warning("PostgreSQL passage-to-entity lineage lookup failed: %s", exc)

        return entity_ids

    # ----------------------------------------------------------------------
    # 3. Point-in-Time Article Context Enrichment (financial_news_queue)
    # ----------------------------------------------------------------------
    def fetch_article_context(
        self,
        source_hashes: Sequence[str],
        ctx: PiTContext,
        max_articles: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve original article metadata and text snippets bounded strictly by `ctx`."""
        if not isinstance(ctx, PiTContext):
            raise TypeError(f"ctx must be a PiTContext instance, got {type(ctx).__name__}")

        valid_hashes = [str(h).strip() for h in source_hashes if str(h).strip()]
        if not valid_hashes:
            return []

        articles: List[Dict[str, Any]] = []
        try:
            with pg_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT source_hash, title, source_url, published_at, raw_text, author
                    FROM financial_news_queue
                    WHERE source_hash = ANY(%s)
                      AND published_at IS NOT NULL
                      AND published_at <= %s::timestamptz
                    ORDER BY published_at DESC, source_hash ASC
                    LIMIT %s;
                    """,
                    (valid_hashes[:max_articles], ctx.cutoff_utc_iso, max_articles),
                )
                rows = cur.fetchall()
                for row in rows:
                    raw_text = str(row[4] or "")
                    snippet = raw_text[:600] + ("..." if len(raw_text) > 600 else "")
                    articles.append(
                        {
                            "source_hash": row[0],
                            "title": row[1] or "Financial News Update",
                            "url": row[2] or "",
                            "published_at": str(row[3]) if row[3] else None,
                            "snippet": snippet,
                            "author": row[5] or "Financial News Desk",
                        }
                    )
                if hasattr(cur, "close"):
                    cur.close()
        except Exception as exc:
            logger.warning("Could not fetch article context from PostgreSQL: %s", exc)

        return articles

    # ----------------------------------------------------------------------
    # 4. Ranked Hybrid Seed Generation (ADR-00034)
    # ----------------------------------------------------------------------
    def seed_with_context(
        self,
        query: str,
        ctx: PiTContext,
        *,
        store: Optional["SubgraphStorePort"] = None,
        max_seeds: int = 8,
        include_cql: bool = False,
    ) -> Dict[str, Any]:
        """Collect ranked `SeedHit` candidates and supporting article provenance strictly as of `ctx`."""
        if not isinstance(ctx, PiTContext):
            raise TypeError(f"ctx must be a PiTContext instance, got {type(ctx).__name__}")

        start_time = time.perf_counter()

        cql_res: Dict[str, Any] = {"cypher": "", "records": [], "status": "SKIPPED"}
        if include_cql:
            cql_res = self.text_to_cql.execute_query(query, ctx=ctx, store=store)
        structured_records = cql_res.get("records", []) or []

        dense_entities = self.search_dense_entities(query, ctx=ctx)
        dense_passages = self.search_dense_passages(query, ctx=ctx)

        passage_hashes = [str(p["source_hash"]) for p in dense_passages if p.get("source_hash")]
        passage_entity_ids = self.expand_passage_entities(passage_hashes, ctx=ctx)
        top_passage_score = (
            max(float(p.get("similarity", 0.85)) for p in dense_passages) * 0.90
            if dense_passages
            else 0.75
        )
        top_passage_hash = passage_hashes[0] if passage_hashes else None

        best_by_node: Dict[str, SeedHit] = {}

        def _register_hit(hit: SeedHit) -> None:
            nid = hit.node_id.strip()
            if not nid:
                return
            existing = best_by_node.get(nid)
            if existing is None or hit.score > existing.score:
                best_by_node[nid] = hit

        for ent in dense_entities:
            nid = str(ent.get("node_id") or "").strip()
            if nid:
                _register_hit(
                    SeedHit(
                        node_id=nid,
                        score=float(ent.get("similarity", 1.0)),
                        origin="dense_entity",
                        entity_type=ent.get("entity_type"),
                        source_hash=ent.get("source_hash"),
                    )
                )

        for eid in passage_entity_ids:
            _register_hit(
                SeedHit(
                    node_id=eid,
                    score=round(top_passage_score, 6),
                    origin="dense_passage",
                    source_hash=top_passage_hash,
                )
            )

        for rec in structured_records:
            if not isinstance(rec, dict):
                continue
            for key, val in rec.items():
                if key in ("source_hash", "rel_type", "relation", "status", "valid_from", "valid_to"):
                    continue
                if isinstance(val, str) and 0 < len(val.strip()) < 64:
                    _register_hit(
                        SeedHit(
                            node_id=val.strip(),
                            score=0.80,
                            origin="text_to_cql",
                            source_hash=rec.get("source_hash"),
                        )
                    )

        ranked_hits = sorted(
            best_by_node.values(),
            key=lambda h: (-h.score, h.node_id),
        )[: max(1, int(max_seeds))]

        all_hashes: Set[str] = set()
        for e in dense_entities:
            if e.get("source_hash"):
                all_hashes.add(str(e["source_hash"]))
        for p in dense_passages:
            if p.get("source_hash"):
                all_hashes.add(str(p["source_hash"]))
        for r in structured_records:
            if isinstance(r, dict) and r.get("source_hash"):
                all_hashes.add(str(r["source_hash"]))

        fetched_articles = self.fetch_article_context(list(all_hashes), ctx=ctx, max_articles=5)
        seen_hashes: Set[str] = {str(a["source_hash"]) for a in fetched_articles if a.get("source_hash")}
        articles = list(fetched_articles)
        for p in dense_passages:
            sh = str(p.get("source_hash") or "")
            if sh and sh not in seen_hashes and len(articles) < 5:
                seen_hashes.add(sh)
                articles.append(p)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "query": query,
            "as_of": ctx.as_of_iso,
            "pit_mode": ctx.mode,
            "hits": ranked_hits,
            "cql_result": cql_res,
            "dense_entities": dense_entities,
            "dense_passages": dense_passages,
            "articles": articles,
            "source_hashes": sorted(all_hashes),
            "latency_ms": round(latency_ms, 2),
        }

    def seed(
        self,
        query: str,
        ctx: PiTContext,
        *,
        store: Optional["SubgraphStorePort"] = None,
        max_seeds: int = 8,
        include_cql: bool = False,
    ) -> List[SeedHit]:
        """Return ranked `SeedHit(node_id, score, origin)` values strictly as of `ctx` (ADR-00034)."""
        bundle = self.seed_with_context(
            query,
            ctx=ctx,
            store=store,
            max_seeds=max_seeds,
            include_cql=include_cql,
        )
        return list(bundle["hits"])


__all__ = ["HybridSeeder", "SeedHit", "SeedOrigin"]
