"""Deterministic Startup Memgraph Hydrator & Event TTL Eviction CLI (ADR-00036).

Rebuilds the ephemeral in-memory Memgraph projection deterministically from the
durable PostgreSQL KG assertion ledger (`kg_nodes`, `kg_edges`) on container boot
or after batch `EntityResolver` merges, and evicts stale `:News` nodes older than
`MEMGRAPH_EVENT_TTL_DAYS` (default: 90 days) without deleting PostgreSQL rows.

All Memgraph Cypher execution lives inside `knowledge_graph/ledger.py` to preserve
`ADR-00034` Invariant #1 (no direct `session.run` in `tools/`).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Optional, Sequence

from knowledge_graph.ledger import (
    evict_expired_news_nodes,
    get_memgraph_event_ttl_days,
    hydrate_memgraph_from_ledger,
)

logger = logging.getLogger("tools.hydrate_memgraph")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hydrate ephemeral Memgraph RAM from durable PostgreSQL kg_nodes/kg_edges (ADR-00036)."
    )
    parser.add_argument(
        "--clear-first",
        action="store_true",
        help="Wipe existing Memgraph RAM state before deterministic re-hydration.",
    )
    parser.add_argument(
        "--skip-news-eviction",
        action="store_true",
        help="Skip evicting expired :News nodes older than MEMGRAPH_EVENT_TTL_DAYS.",
    )
    parser.add_argument(
        "--evict-only",
        action="store_true",
        help="Only evict expired :News nodes older than MEMGRAPH_EVENT_TTL_DAYS without re-hydrating.",
    )
    parser.add_argument(
        "--ttl-days",
        type=int,
        default=None,
        help="Override MEMGRAPH_EVENT_TTL_DAYS for :News RAM eviction (default: 90).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.evict_only:
        ttl = get_memgraph_event_ttl_days(args.ttl_days)
        evicted = evict_expired_news_nodes(ttl_days=ttl)
        print(json.dumps({"evicted_news_nodes": evicted, "ttl_days": ttl}, sort_keys=True))
        return 0

    summary = hydrate_memgraph_from_ledger(
        clear_first=args.clear_first,
        evict_news=not args.skip_news_eviction,
        ttl_days=args.ttl_days,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
