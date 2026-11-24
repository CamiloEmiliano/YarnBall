# retrieval/text_to_cql.py
# -*- coding: utf-8 -*-
"""Guarded Text-to-CQL (Cypher) Engine with Safety Guardrails & Self-Correction.

Translates natural language financial questions into valid, read-only Cypher queries
executed against Memgraph using YarnBallModelClient (`yarnball-qwen:7b`) with dynamic
schema injection, pre-execution guardrails, AST read-only enforcement, and an auto-correction loop.
"""

import os
import re
import time
import json
import logging
from typing import TYPE_CHECKING, Optional, Dict, Any, List, Tuple, Set
from pathlib import Path

if TYPE_CHECKING:
    from retrieval.facade import SubgraphStorePort
    from sft_pipeline.serving.model_client import YarnBallModelClient

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from retrieval.guardrails.schema_guardrail import (
    CypherASTWhitelistGuardrail,
    DEFAULT_FINANCE_SCHEMA,
)
from retrieval.temporal import PiTContext, TemporalQueryFilter

logger = logging.getLogger(__name__)

# Disallowed mutating keywords in read-only Cypher queries
FORBIDDEN_MUTATING_KEYWORDS: Set[str] = {
    "CREATE",
    "MERGE",
    "SET",
    "DELETE",
    "DETACH",
    "REMOVE",
    "DROP",
    "ALTER",
    "GRANT",
    "REVOKE",
    "TRUNCATE",
}

# Disallowed procedure call prefixes (administrative or mutating procedures)
FORBIDDEN_PROCEDURE_PREFIXES: Tuple[str, ...] = (
    "CALL DBMS",
    "CALL MG.",
    "CALL APOC.UTIL",
    "CALL APOC.CUSTOM",
)

DEFAULT_ALLOWED_LABELS: Set[str] = {
    "Company",
    "Person",
    "Product",
    "Technology",
    "Sector",
    "Metric",
    "Location",
    "News",
    "Entity",
}

DEFAULT_ALLOWED_REL_TYPES: Set[str] = {
    "PARTNERED_WITH",
    "COMPETES_WITH",
    "INVESTS_IN",
    "ACQUIRED",
    "LEADS",
    "PRODUCES",
    "IMPACTS",
    "REPORTS",
    "CO_INVESTS_WITH",
    "PEER_OF",
    "COLLABORATES_WITH",
    "MENTIONS",
    "RELATED_TO",
}

class TextToCQLError(Exception):
    """Base exception for Text-to-CQL compilation and guardrail errors."""
    pass


class GuardrailViolationError(TextToCQLError):
    """Raised when a generated Cypher query violates safety or schema constraints."""
    pass


class TextToCQL:
    """Guarded Text-to-CQL engine with schema validation and self-correction."""

    def __init__(
        self,
        ollama_url: Optional[str] = None,
        model_name: Optional[str] = None,
        max_limit: int = 50,
        query_timeout_sec: float = 3.0,
        model_client: Optional["YarnBallModelClient"] = None,
    ):
        if model_client is None:
            from sft_pipeline.serving.model_client import YarnBallModelClient

            self.model_client = YarnBallModelClient(
                ollama_url=ollama_url,
                model_tag=model_name,
            )
        else:
            self.model_client = model_client
        self.ollama_url = self.model_client.ollama_url
        self.model_name = self.model_client.model_tag
        self.max_limit = max_limit
        self.query_timeout_sec = query_timeout_sec
        self.guardrail = CypherASTWhitelistGuardrail(
            schema=DEFAULT_FINANCE_SCHEMA,
            max_limit=self.max_limit,
        )
        self.temporal_filter = TemporalQueryFilter()

    # ----------------------------------------------------------------------
    # 1. Dynamic Schema Extraction
    # ----------------------------------------------------------------------
    def get_active_schema(
        self,
        store: Optional["SubgraphStorePort"] = None,
    ) -> Dict[str, Any]:
        """Extract active node labels, relationship types, and properties via `MemgraphSubgraphStore`."""
        from retrieval.facade import MemgraphSubgraphStore

        default_labels = sorted(DEFAULT_ALLOWED_LABELS)
        default_rel_types = sorted(DEFAULT_ALLOWED_REL_TYPES)
        if store is not None and not isinstance(store, MemgraphSubgraphStore):
            return {
                "labels": default_labels,
                "relationship_types": default_rel_types,
                "properties": ["aliases", "id", "name", "sector", "source_hash", "ticker"],
            }
        active_store = store if isinstance(store, MemgraphSubgraphStore) else MemgraphSubgraphStore()
        return active_store.introspect_schema(
            default_labels=default_labels,
            default_rel_types=default_rel_types,
        )

    # ----------------------------------------------------------------------
    # 2. Prompt Compilation & Ollama Query Generation
    # ----------------------------------------------------------------------
    def _build_system_prompt(self, schema: Dict[str, Any]) -> str:
        """Construct the system prompt for Cypher generation."""
        labels_str = ", ".join(schema.get("labels", []))
        rels_str = ", ".join(schema.get("relationship_types", []))

        return f"""You are an expert Cypher query generator for a financial knowledge graph stored in Memgraph.
Your job is to translate natural language financial questions into READ-ONLY Cypher queries.

ACTIVE GRAPH SCHEMA:
- Node Labels: {labels_str}
- Relationship Types: {rels_str}
- Common Properties: id (Entity Name), ticker (Stock Ticker), sector, aliases, source_hash

STRICT CYPHER RULES:
1. ONLY generate READ-ONLY queries starting with MATCH.
2. NEVER use CREATE, MERGE, SET, DELETE, DETACH, REMOVE, DROP, or administrative procedure calls.
3. Use case-insensitive matching for entity names: `c.id =~ '(?i).*Apple.*' OR c.ticker = 'AAPL'`.
4. For symmetric relationships (COMPETES_WITH, PARTNERED_WITH, CO_INVESTS_WITH), traverse undirectedly: `(a)-[:PARTNERED_WITH]-(b)`.
5. Always return meaningful columns: `RETURN DISTINCT s.id AS source, type(r) AS relation, t.id AS target`.
6. Always append `LIMIT 50`.
7. Output ONLY the raw Cypher query. Do not include markdown code fences, explanations, or commentary.

FEW-SHOT EXAMPLES:
Question: Which companies supply chips or foundry services to Apple?
Cypher: MATCH (s:Company)-[:PRODUCES|PARTNERED_WITH]-(aapl:Company) WHERE aapl.id =~ '(?i).*Apple.*' OR aapl.ticker = 'AAPL' RETURN DISTINCT s.id AS supplier, aapl.id AS client LIMIT 50

Question: What AI startups share investments from Microsoft and Amazon?
Cypher: MATCH (msft:Company)-[:INVESTS_IN|PARTNERED_WITH]->(lab:Company)<-[:INVESTS_IN|PARTNERED_WITH]-(amzn:Company) WHERE (msft.id =~ '(?i).*Microsoft.*' OR msft.ticker = 'MSFT') AND (amzn.id =~ '(?i).*Amazon.*' OR amzn.ticker = 'AMZN') RETURN DISTINCT lab.id AS startup LIMIT 50

Question: Which executives lead NVIDIA and what products do they produce?
Cypher: MATCH (p:Person)-[:LEADS]->(nvda:Company)-[:PRODUCES]->(prod:Product) WHERE nvda.id =~ '(?i).*NVIDIA.*' OR nvda.ticker = 'NVDA' RETURN DISTINCT p.id AS executive, prod.id AS product LIMIT 50
"""

    @staticmethod
    def clean_cypher(raw_text: str) -> str:
        """Strip markdown fences, leading/trailing whitespace, and trailing semicolons."""
        if not raw_text:
            return ""
        # Remove ```cypher and ``` fences
        cleaned = re.sub(r"^```(?:cypher)?\s*", "", raw_text.strip(), flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)

        # Extract only the first query block starting with MATCH
        match = re.search(r"(MATCH\s+[\s\S]+)", cleaned, re.IGNORECASE)
        if match:
            cleaned = match.group(1).strip()

        # Remove trailing semicolons
        cleaned = cleaned.rstrip(";").strip()
        return cleaned

    def _call_llm(self, prompt: str, system_prompt: str) -> str:
        """Send prompt to local YarnBallModelClient serving `yarnball-qwen`."""
        resp = self.model_client.generate(
            prompt=f"Question: {prompt}\nCypher:",
            task="TEXT_TO_CYPHER",
            system_prompt=system_prompt,
            temperature=0.0,
        )
        if resp.error:
            logger.warning("YarnBallModelClient request failed: %s", resp.error)
            return ""
        return self.clean_cypher(resp.final_output)

    # ----------------------------------------------------------------------
    # 3. Pre-Execution Safety Guardrails
    # ----------------------------------------------------------------------
    def validate_cypher(
        self, cypher: str, schema: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate that the Cypher query satisfies safety, schema, and read-only constraints.

        Args:
            cypher: The Cypher query string to validate.
            schema: Optional active schema dictionary.

        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str]).
        """
        result = self.guardrail.validate(cypher)
        if not result.is_valid:
            return False, result.error_summary
        return True, None

    def enforce_limit(self, cypher: str) -> str:
        """Ensure the Cypher query contains a bounded LIMIT clause."""
        return self.guardrail.enforce_limit(cypher)

    # ----------------------------------------------------------------------
    # 4. Guarded Execution with Auto-Correction Loop
    # ----------------------------------------------------------------------
    def execute_query(
        self,
        question: str,
        ctx: PiTContext,
        max_retries: int = 3,
        schema: Optional[Dict[str, Any]] = None,
        store: Optional["SubgraphStorePort"] = None,
    ) -> Dict[str, Any]:
        """Translate question to Cypher, validate guardrails, inject PiT filter, and execute via `SubgraphStorePort`."""
        if not isinstance(ctx, PiTContext):
            raise TypeError(f"ctx must be a PiTContext instance, got {type(ctx).__name__}")

        from retrieval.facade import MemgraphSubgraphStore

        active_store: SubgraphStorePort = store if store is not None else MemgraphSubgraphStore()
        active_schema = schema or self.get_active_schema(store=active_store)
        system_prompt = self._build_system_prompt(active_schema)

        current_prompt = question
        last_cypher = ""
        last_error = None
        retries_used = 0
        records: List[Dict[str, Any]] = []

        start_time = time.perf_counter()

        for attempt in range(max_retries):
            retries_used = attempt
            # 1. Generate Cypher query
            raw_cypher = self._call_llm(current_prompt, system_prompt)
            if not raw_cypher:
                last_error = "LLM generated empty response."
                current_prompt = f"{question}\n\nPrevious attempt failed: {last_error}\nPlease generate a valid Cypher query."
                continue

            last_cypher = raw_cypher

            # 2. Validate Safety Guardrails
            is_valid, violation_msg = self.validate_cypher(last_cypher, active_schema)
            if not is_valid:
                last_error = f"Guardrail Violation: {violation_msg}"
                logger.warning("Attempt %d failed guardrails: %s", attempt + 1, last_error)
                current_prompt = (
                    f"{question}\n\n"
                    f"Your previous query was rejected: {last_error}\n"
                    f"Previous Query: {last_cypher}\n"
                    f"Please rewrite the query to be strictly read-only, conforming to schema."
                )
                continue

            # 3. Enforce Resource Bounds (LIMIT 50) & ADR-00033 Point-in-Time Filter
            bounded_cypher = self.enforce_limit(last_cypher)
            temporal_cypher = self.temporal_filter.inject_active_coalescence(bounded_cypher, ctx)
            last_cypher = temporal_cypher

            # 4. Execute Query through SubgraphStorePort (ADR-00034 Invariant #1)
            from knowledge_graph.errors import StoreUnavailableError

            try:
                records = active_store.execute_guarded(
                    bounded_cypher,
                    {"as_of_date": ctx.as_of_iso},
                    ctx,
                )
                last_error = None
                break  # Success!
            except StoreUnavailableError:
                raise
            except Exception as exc:
                last_error = f"Memgraph Execution Error: {str(exc)}"
                logger.warning("Attempt %d Memgraph execution failed: %s", attempt + 1, last_error)
                current_prompt = (
                    f"{question}\n\n"
                    f"Your generated Cypher caused a database error: {last_error}\n"
                    f"Failed Cypher: {last_cypher}\n"
                    f"Please fix the Cypher syntax error and return only the corrected query."
                )

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        status = "SUCCESS" if last_error is None else "FAILED"

        return {
            "question": question,
            "as_of": ctx.as_of_iso,
            "pit_mode": ctx.mode,
            "cypher": last_cypher,
            "status": status,
            "degraded": status != "SUCCESS",
            "records": records,
            "record_count": len(records),
            "retries_used": retries_used,
            "error": last_error,
            "latency_ms": round(latency_ms, 2),
        }
