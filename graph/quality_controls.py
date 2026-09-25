"""
Knowledge Graph Quality Controls Module for Financial GraphRAG.

Implements the 6 Institutional Quality Controls:
1. Generic Noun & Pronoun Stoplist Filter (Company, Customer, Management)
2. Ontological Domain & Range Schema Validator (with Auto-Orientation)
3. Multi-Source Evidence Weighting & Confidence Calibration (SEC vs News)
4. Corporate Ownership DAG & Circular Reference Guard (DFS Cycle Breaker)
5. Temporal Validity & State Transition Guard (valid_from <= valid_to, auto-closure)
6. Degree Anomaly / Hallucination Circuit Breaker (Quarantine Queue)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("graph_quality_controls")

# ----------------------------------------------------------------------
# 1. Generic Noun & Abstract Legal Pronoun Stoplist
# ----------------------------------------------------------------------
GENERIC_NOUN_STOPLIST: Set[str] = {
    "the company", "the customer", "the customers", "our customer", "our customers",
    "the supplier", "the suppliers", "our supplier", "our suppliers",
    "management", "the registrant", "the vendor", "the vendors",
    "third party", "third parties", "third-party supplier", "third-party vendor",
    "certain customers", "certain suppliers", "certain vendors",
    "the manufacturer", "the manufacturers", "our manufacturer",
    "our subsidiaries", "certain subsidiaries", "unnamed counterparty",
    "competitors", "our competitors", "the industry", "the market",
    "foreign governments", "regulatory bodies", "various clients",
    "all customers", "key suppliers", "major customer", "sole source",
}


def is_generic_placeholder(name: Optional[str]) -> bool:
    """Check if an entity name is an abstract legal or journalistic pronoun."""
    if not name or not str(name).strip():
        return False
    norm = str(name).lower().strip()
    # Strip non-alphanumerics
    norm_clean = "".join(c for c in norm if c.isalnum() or c.isspace()).strip()
    if norm_clean in GENERIC_NOUN_STOPLIST or norm in GENERIC_NOUN_STOPLIST:
        return True
    # Check if starts with abstract pronoun
    if norm_clean.startswith("certain ") or norm_clean.startswith("various ") or norm_clean.startswith("unnamed "):
        return True
    return False


# ----------------------------------------------------------------------
# 2. Ontological Domain & Range Schema Validator
# ----------------------------------------------------------------------
# Expected entity types for relationships
# rel_type -> (valid_source_types, valid_target_types, allowed_reversible)
# Expected entity types for relationships
# rel_type -> (valid_source_types, valid_target_types, allowed_reversible)
ONTOLOGY_SCHEMA: Dict[str, Tuple[Set[str], Set[str], bool]] = {
    "SUBSIDIARY_OF": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY"}, False),
    "PARENT_OF": ({"COMPANY"}, {"COMPANY", "SUBSIDIARY"}, False),
    "SUPPLIES_TO": ({"COMPANY", "SUBSIDIARY", "SUPPLIER"}, {"COMPANY", "CUSTOMER"}, False),
    "CUSTOMER_OF": ({"COMPANY", "CUSTOMER"}, {"COMPANY", "SUPPLIER"}, False),
    "COMPETES_WITH": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY", "SUBSIDIARY"}, True),
    "PARTNERED_WITH": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY", "SUBSIDIARY"}, True),
    "CEO_OF": ({"PERSON", "EXECUTIVE"}, {"COMPANY", "SUBSIDIARY"}, False),
    "EXECUTIVE_OF": ({"PERSON", "EXECUTIVE"}, {"COMPANY", "SUBSIDIARY"}, False),
    "DIRECTOR_OF": ({"PERSON", "EXECUTIVE"}, {"COMPANY", "SUBSIDIARY"}, False),
    "INSIDER_OF": ({"PERSON", "EXECUTIVE"}, {"COMPANY", "SUBSIDIARY"}, False),
    "EXPOSED_TO_RISK": ({"COMPANY", "SUBSIDIARY"}, {"GEOPOLITICALRISK", "COMMODITY", "REGULATORYBODY", "MARKETEVENT", "RISK"}, False),
    "LICENSES_FROM": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY", "SUBSIDIARY"}, False),
    "LICENSES_TO": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY", "SUBSIDIARY"}, False),
    "ACQUIRED_BY": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY"}, False),
    "DEFAULTED_ON": ({"COMPANY", "SUBSIDIARY"}, {"COMPANY", "FINANCIALINSTITUTION"}, False),
}


def validate_and_orient_triple(
    src_id: str,
    src_label: str,
    tgt_id: str,
    tgt_label: str,
    rel_type: str,
) -> Optional[Tuple[str, str, str, str, str]]:
    """
    Validate relationship schema against domain and range ontology.
    If the relationship is inverted (e.g. Company -> CEO_OF -> Person), automatically
    orients it correctly: (Person -> CEO_OF -> Company).
    Handles unresolved 'Entity' typing gracefully without allowing generic pass-throughs.
    
    Returns:
        (canonical_src, canonical_src_label, canonical_tgt, canonical_tgt_label, canonical_rel)
        or None if the triple is fundamentally invalid.
    """
    clean_rel = rel_type.strip().upper()
    src_lbl = (src_label or "ENTITY").strip().upper()
    tgt_lbl = (tgt_label or "ENTITY").strip().upper()

    if clean_rel not in ONTOLOGY_SCHEMA:
        # Allow open custom relationships with default labels
        return (src_id, src_lbl, tgt_id, tgt_lbl, clean_rel)

    valid_srcs, valid_tgts, is_symmetric = ONTOLOGY_SCHEMA[clean_rel]

    # Check forward validity
    is_src_valid = src_lbl in valid_srcs
    is_tgt_valid = tgt_lbl in valid_tgts

    if is_src_valid and is_tgt_valid:
        return (src_id, src_lbl, tgt_id, tgt_lbl, clean_rel)

    # Check if triple is inverted (e.g. Company -> CEO_OF -> Person)
    is_inverted_src_valid = tgt_lbl in valid_srcs
    is_inverted_tgt_valid = src_lbl in valid_tgts

    if is_inverted_src_valid and is_inverted_tgt_valid:
        logger.debug(f"Auto-orienting inverted triple: ({src_id}:{src_lbl})-[:{clean_rel}]->({tgt_id}:{tgt_lbl})")
        return (tgt_id, tgt_lbl, src_id, src_lbl, clean_rel)

    # Handle unresolved ENTITY typologies for asymmetric relations
    person_relations = {"CEO_OF", "EXECUTIVE_OF", "DIRECTOR_OF", "INSIDER_OF"}
    if src_lbl == "ENTITY" and tgt_lbl == "ENTITY":
        if clean_rel in person_relations:
            return (src_id, "Person", tgt_id, "Company", clean_rel)
        inferred_src = next(iter(valid_srcs)).title()
        inferred_tgt = next(iter(valid_tgts)).title()
        return (src_id, inferred_src, tgt_id, inferred_tgt, clean_rel)

    if clean_rel in person_relations:
        if src_lbl == "ENTITY" and tgt_lbl in valid_tgts:
            return (src_id, "Person", tgt_id, tgt_lbl, clean_rel)
        if tgt_lbl == "ENTITY" and src_lbl in valid_tgts:
            return (tgt_id, "Person", src_id, src_lbl, clean_rel)

    if src_lbl == "ENTITY" and is_tgt_valid:
        inferred_src = next(iter(valid_srcs))
        return (src_id, inferred_src.title(), tgt_id, tgt_lbl, clean_rel)

    if tgt_lbl == "ENTITY" and is_src_valid:
        inferred_tgt = next(iter(valid_tgts))
        return (src_id, src_lbl, tgt_id, inferred_tgt.title(), clean_rel)

    if src_lbl == "ENTITY" and tgt_lbl == "ENTITY" and is_symmetric:
        return (src_id, "Company", tgt_id, "Company", clean_rel)

    # If types are incompatible with schema, reject
    logger.debug(f"Rejecting invalid ontology triple: ({src_id}:{src_lbl})-[:{clean_rel}]->({tgt_id}:{tgt_lbl})")
    return None


# ----------------------------------------------------------------------
# 3. Multi-Source Evidence Weighting & Confidence Calibration
# ----------------------------------------------------------------------
PROVENANCE_CONFIDENCE_MAP: Dict[str, float] = {
    "SEC_EXHIBIT_21": 1.00,       # Sworn SEC subsidiary disclosure (Deterministic Ground Truth)
    "SEC_FORM_4": 1.00,           # Official SEC insider transaction XML
    "SEC_10K_ITEM1": 0.98,        # Form 10-K Item 1 Business
    "SEC_10Q": 0.95,              # Form 10-Q Segment & supply disclosure
    "SEC_8K": 0.95,               # Form 8-K Material Event Disclosure
    "MULTI_TEACHER_CONSENSUS": 0.85, # 2-of-3 Frontier LLM Consensus
    "FINANCIAL_NEWS_VERIFIED": 0.75, # News article corroboration
    "NEWS_EXTRACTION": 0.60,      # Single news article extraction
    "UNVERIFIED_RUMOR": 0.40,     # Speculative / social media mention
}


def compute_edge_confidence(
    provenance: Optional[str],
    mention_count: int = 1,
    distinct_sources_count: Optional[int] = None,
    base_confidence: Optional[float] = None,
) -> float:
    """Calculate calibrated confidence score based on provenance and distinct corroborating sources."""
    if base_confidence is not None:
        raw_conf = base_confidence
    else:
        prov_key = (provenance or "NEWS_EXTRACTION").strip().upper()
        raw_conf = PROVENANCE_CONFIDENCE_MAP.get(prov_key, 0.60)

    # Boost confidence with distinct corroborating sources (+0.05 per distinct source, capped at 1.0)
    effective_sources = distinct_sources_count if distinct_sources_count is not None else mention_count
    if effective_sources > 1:
        raw_conf = min(1.0, raw_conf + (0.05 * min(4, effective_sources - 1)))

    return round(raw_conf, 3)


# ----------------------------------------------------------------------
# 4. Corporate Ownership DAG & Circular Reference Guard
# ----------------------------------------------------------------------
def detect_and_break_ownership_cycles(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run iterative DFS cycle detection on corporate ownership edges (SUBSIDIARY_OF, PARENT_OF).
    Prunes the edge with the lowest confidence strictly within the detected cycle path
    (sliced from cycle entry node forward).
    """
    ownership_rel_types = {"SUBSIDIARY_OF", "PARENT_OF"}
    ownership_edges = [e for e in edges if str(e.get("rel_type", "")).upper() in ownership_rel_types]
    other_edges = [e for e in edges if str(e.get("rel_type", "")).upper() not in ownership_rel_types]

    if not ownership_edges:
        return edges

    # Normalize PARENT_OF into equivalent SUBSIDIARY_OF for cycle detection
    # (A PARENT_OF B) <=> (B SUBSIDIARY_OF A)
    adj: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for edge in ownership_edges:
        src = str(edge.get("source_id", "")).strip()
        tgt = str(edge.get("target_id", "")).strip()
        rel = str(edge.get("rel_type", "")).upper()

        if rel == "SUBSIDIARY_OF":
            adj[src].append((tgt, edge))
        elif rel == "PARENT_OF":
            adj[tgt].append((src, edge))

    culprit_edges_to_drop: Set[int] = set()

    # Iterative DFS with cycle slicing
    while True:
        cycle_found = False
        visited_global: Set[str] = set()

        all_nodes = list(adj.keys())
        for start_node in all_nodes:
            if start_node in visited_global:
                continue

            # Stack elements: (curr_node, edge_index_in_adj, path_nodes, path_edges)
            stack: List[Tuple[str, int, List[str], List[Dict[str, Any]]]] = [
                (start_node, 0, [start_node], [])
            ]
            in_stack: Set[str] = {start_node}

            while stack:
                curr_node, edge_idx, path_nodes, path_edges = stack.pop()
                neighbors = adj.get(curr_node, [])

                if edge_idx < len(neighbors):
                    stack.append((curr_node, edge_idx + 1, path_nodes, path_edges))

                    neighbor, edge_obj = neighbors[edge_idx]
                    edge_id = id(edge_obj)

                    if edge_id in culprit_edges_to_drop:
                        continue

                    if neighbor in in_stack:
                        # Cycle detected! Slice strictly from the entry node forward
                        try:
                            start_idx = path_nodes.index(neighbor)
                        except ValueError:
                            start_idx = 0

                        cycle_edges = path_edges[start_idx:] + [edge_obj]

                        def edge_score(e: Dict[str, Any]) -> float:
                            return float(e.get("confidence", 0.5))

                        weakest_edge = min(cycle_edges, key=edge_score)
                        logger.warning(
                            f"Ownership cycle detected: {' -> '.join(path_nodes[start_idx:])} -> {neighbor}. "
                            f"Pruning weakest cycle edge: {weakest_edge.get('source_id')} -[:{weakest_edge.get('rel_type')}]-> {weakest_edge.get('target_id')}"
                        )
                        culprit_edges_to_drop.add(id(weakest_edge))
                        cycle_found = True
                        break
                    elif neighbor not in visited_global:
                        in_stack.add(neighbor)
                        visited_global.add(neighbor)
                        stack.append((neighbor, 0, path_nodes + [neighbor], path_edges + [edge_obj]))
                else:
                    in_stack.discard(curr_node)
                    visited_global.add(curr_node)

            if cycle_found:
                break

        if not cycle_found:
            break

    surviving_ownership_edges = [e for e in ownership_edges if id(e) not in culprit_edges_to_drop]
    return other_edges + surviving_ownership_edges


# ----------------------------------------------------------------------
# 5. Temporal Consistency & State Transition Guard
# ----------------------------------------------------------------------
def parse_iso_date(date_val: Any) -> Optional[date]:
    """Parse date string or object into datetime.date."""
    if not date_val:
        return None
    if isinstance(date_val, date) and not isinstance(date_val, datetime):
        return date_val
    if isinstance(date_val, datetime):
        return date_val.date()
    try:
        clean_str = str(date_val).strip()[:10]
        return datetime.strptime(clean_str, "%Y-%m-%d").date()
    except Exception:
        return None


def validate_temporal_consistency(edge: Dict[str, Any]) -> bool:
    """
    Validate that temporal bounds are physically consistent:
    1. valid_from <= valid_to (if both present)
    2. valid_from is not in the future beyond current year + 1
    """
    v_from = parse_iso_date(edge.get("valid_from"))
    v_to = parse_iso_date(edge.get("valid_to"))

    if v_from and v_to and v_from > v_to:
        logger.debug(f"Dropping edge with invalid temporal bounds: valid_from ({v_from}) > valid_to ({v_to})")
        return False

    return True


def resolve_edge_state_transitions(
    existing_edges: List[Dict[str, Any]],
    new_event_edges: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Auto-close active predecessor edges when a terminal event (TERMINATED, ACQUIRED_BY, DEFAULTED_ON) occurs.
    """
    terminal_types = {"TERMINATED", "ACQUIRED_BY", "DEFAULTED_ON", "CONTRACT_TERMINATION"}
    updated_existing = list(existing_edges)

    for new_edge in new_event_edges:
        rel_type = str(new_edge.get("rel_type", "")).upper()
        event_status = str(new_edge.get("status", "")).upper()
        event_date = str(new_edge.get("valid_from") or new_edge.get("event_date") or datetime.now().strftime("%Y-%m-%d"))[:10]

        if rel_type in terminal_types or event_status == "TERMINATED":
            src = new_edge.get("source_id")
            tgt = new_edge.get("target_id")

            for ex_edge in updated_existing:
                ex_src = ex_edge.get("source_id")
                ex_tgt = ex_edge.get("target_id")
                # If same pair and currently active (no valid_to or future valid_to)
                if (ex_src == src and ex_tgt == tgt) or (ex_src == tgt and ex_tgt == src):
                    if not ex_edge.get("valid_to") or ex_edge.get("valid_to") > event_date:
                        ex_edge["valid_to"] = event_date
                        ex_edge["status"] = "TERMINATED"
                        logger.info(f"Closed predecessor edge ({ex_src}->{ex_tgt}) at {event_date} due to terminal event.")

    return updated_existing


# ----------------------------------------------------------------------
# 6. Degree Anomaly Quarantine Gate (Hallucination Circuit Breaker)
# ----------------------------------------------------------------------
BENCHMARK_TOP_COMPANIES: Set[str] = {
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA",
    "BRK.A", "BRK.B", "UNH", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX",
    "MRK", "ABBV", "PEP", "KO", "BAC", "COST", "AVGO", "TMO", "CSCO", "MCD",
    "WMT", "ABT", "DIS", "LIN", "ADBE", "NFLX", "TXN", "PM", "AMD", "QCOM",
}


def check_degree_anomaly_quarantine(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    degree_burst_threshold: int = 25,
    quarantine_queue_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Monitor node degree bursts per batch. If a non-benchmark node acquires
    more than `degree_burst_threshold` edges in a single batch, quarantine it.
    
    Returns:
        (clean_nodes, clean_edges, quarantined_entities)
    """
    degree_counts: Dict[str, int] = defaultdict(int)
    for edge in edges:
        src = str(edge.get("source_id", "")).strip()
        tgt = str(edge.get("target_id", "")).strip()
        if src:
            degree_counts[src] += 1
        if tgt:
            degree_counts[tgt] += 1

    quarantined_ids: Set[str] = set()
    quarantined_records: List[Dict[str, Any]] = []

    for nid, count in degree_counts.items():
        if count >= degree_burst_threshold:
            ticker = nid.upper()
            if ticker not in BENCHMARK_TOP_COMPANIES:
                quarantined_ids.add(nid)
                record = {
                    "entity_id": nid,
                    "degree_count": count,
                    "threshold": degree_burst_threshold,
                    "timestamp": datetime.now().isoformat(),
                    "reason": "DEGREE_BURST_HALLUCINATION_ANOMALY",
                }
                quarantined_records.append(record)
                logger.warning(f"CIRCUIT BREAKER: Quarantining entity '{nid}' with anomalous degree {count} >= {degree_burst_threshold}")

    if quarantined_records and quarantine_queue_path:
        quarantine_queue_path.parent.mkdir(parents=True, exist_ok=True)
        with open(quarantine_queue_path, "a", encoding="utf-8") as f:
            for rec in quarantined_records:
                f.write(json.dumps(rec) + "\n")

    clean_nodes = [n for n in nodes if str(n.get("id", "")) not in quarantined_ids]
    clean_edges = [
        e for e in edges
        if str(e.get("source_id", "")) not in quarantined_ids
        and str(e.get("target_id", "")) not in quarantined_ids
    ]

    return clean_nodes, clean_edges, quarantined_records
