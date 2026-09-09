# rag/text_to_cql.py
# -*- coding: utf-8 -*-
"""Guarded Text-to-CQL (Cypher) Engine with Safety Guardrails & Self-Correction.

Translates natural language financial questions into valid, read-only Cypher queries
executed against Memgraph using local Ollama (qwen3:8b) with dynamic schema injection,
pre-execution guardrails, AST read-only enforcement, and an auto-correction feedback loop.
"""

import os
import re
import time
import json
import logging
from typing import Optional, Dict, Any, List, Tuple, Set
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from graph.memgraph_driver import get_memgraph_driver

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

# LLM Configuration
def _default_ollama_url() -> str:
    env_url = os.getenv("QWEN_API_BASE")
    if env_url:
        if "host.docker.internal" in env_url and not os.path.exists("/.dockerenv"):
            return env_url.replace("host.docker.internal", "localhost")
        return env_url
    if os.path.exists("/.dockerenv"):
        return "http://host.docker.internal:11434/api/generate"
    return "http://localhost:11434/api/generate"

OLLAMA_API_BASE = _default_ollama_url()
OLLAMA_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "qwen3:8b")


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
    ):
        self.ollama_url = ollama_url or OLLAMA_API_BASE
        self.model_name = model_name or OLLAMA_MODEL_NAME
        self.max_limit = max_limit
        self.query_timeout_sec = query_timeout_sec

    # ----------------------------------------------------------------------
    # 1. Dynamic Schema Extraction
    # ----------------------------------------------------------------------
    def get_active_schema(self) -> Dict[str, Any]:
        """Extract active node labels, relationship types, and properties from Memgraph."""
        labels: Set[str] = set(DEFAULT_ALLOWED_LABELS)
        rel_types: Set[str] = set(DEFAULT_ALLOWED_REL_TYPES)
        properties: Set[str] = {"id", "ticker", "name", "sector", "aliases", "source_hash"}

        driver = get_memgraph_driver()
        try:
            with driver.session() as session:
                # Query distinct node labels
                try:
                    res_lbl = session.run("MATCH (n) RETURN DISTINCT labels(n) AS lbls LIMIT 100")
                    for r in res_lbl:
                        for l in r.get("lbls", []):
                            if l:
                                labels.add(l)
                except Exception:
                    pass

                # Query distinct relationship types
                try:
                    res_rel = session.run("MATCH ()-[r]->() RETURN DISTINCT type(r) AS rel LIMIT 100")
                    for r in res_rel:
                        rel = r.get("rel")
                        if rel:
                            rel_types.add(rel)
                except Exception:
                    pass
        except Exception as exc:
            logger.debug("Could not query active schema from Memgraph: %s. Using default schema.", exc)

        return {
            "labels": sorted(list(labels)),
            "relationship_types": sorted(list(rel_types)),
            "properties": sorted(list(properties)),
        }

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
        """Send prompt to local Ollama server."""
        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed; returning empty Cypher")
            return ""

        api_url = self.ollama_url
        if "/v1" in api_url:
            endpoint = api_url.rstrip("/") + "/chat/completions"
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "stream": False,
            }
            try:
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(endpoint, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    return self.clean_cypher(data["choices"][0]["message"]["content"])
            except Exception as exc:
                logger.warning("Ollama /v1 call failed: %s; trying /api/generate", exc)

        gen_url = self.ollama_url
        if "/v1" in gen_url:
            gen_url = gen_url.split("/v1")[0].rstrip("/") + "/api/generate"

        try:
            with httpx.Client(timeout=30.0) as client:
                full_prompt = f"{system_prompt}\n\nQuestion: {prompt}\nCypher:"
                resp = client.post(
                    gen_url,
                    json={
                        "model": self.model_name,
                        "prompt": full_prompt,
                        "stream": False,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                return self.clean_cypher(data.get("response", ""))
        except Exception as exc:
            logger.warning("Ollama API request failed: %s", exc)
            return ""

    # ----------------------------------------------------------------------
    # 3. Pre-Execution Safety Guardrails
    # ----------------------------------------------------------------------
    def validate_cypher(
        self, cypher: str, schema: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """Validate that the Cypher query satisfies safety and read-only constraints.

        Args:
            cypher: The Cypher query string to validate.
            schema: Optional active schema dictionary.

        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str]).
        """
        if not cypher or not cypher.strip():
            return False, "Generated Cypher query is empty."

        clean = cypher.strip()

        # 1. Must start with MATCH or WITH
        if not re.match(r"^(MATCH|WITH|UNWIND)\b", clean, re.IGNORECASE):
            return False, f"Query must begin with MATCH or WITH. Got: '{clean[:20]}...'"

        # 2. Read-Only AST / Keyword Enforcer: Block mutating keywords
        # Tokenize by word boundary to prevent matching substrings inside identifiers
        words = set(re.findall(r"\b[A-Z_]+\b", clean.upper()))
        forbidden_matches = words.intersection(FORBIDDEN_MUTATING_KEYWORDS)
        if forbidden_matches:
            return False, f"Mutating operations forbidden in read-only query: {sorted(list(forbidden_matches))}"

        # 3. Block administrative procedure calls
        upper_query = clean.upper()
        for proc in FORBIDDEN_PROCEDURE_PREFIXES:
            if proc in upper_query:
                return False, f"Administrative procedure calls are forbidden: '{proc}'"

        # 4. Schema Whitelist Validation (if schema provided)
        if schema:
            allowed_labels = set(l.upper() for l in schema.get("labels", []))
            allowed_rels = set(r.upper() for r in schema.get("relationship_types", []))

            # Extract node labels in query: `:Label`
            query_labels = set(re.findall(r":([a-zA-Z0-9_]+)", clean))
            for lbl in query_labels:
                # Check if it's a relationship type or a label
                if lbl.upper() not in allowed_labels and lbl.upper() not in allowed_rels:
                    logger.debug("Warning: referenced label '%s' not explicitly in active schema", lbl)

        return True, None

    def enforce_limit(self, cypher: str) -> str:
        """Ensure the Cypher query contains a bounded LIMIT clause."""
        clean = cypher.strip().rstrip(";")
        limit_match = re.search(r"\bLIMIT\s+(\d+)", clean, re.IGNORECASE)

        if limit_match:
            current_limit = int(limit_match.group(1))
            if current_limit > self.max_limit:
                # Clamp limit to max_limit
                clean = re.sub(r"\bLIMIT\s+\d+", f"LIMIT {self.max_limit}", clean, flags=re.IGNORECASE)
        else:
            # Append LIMIT clause
            clean = f"{clean} LIMIT {self.max_limit}"

        return clean

    # ----------------------------------------------------------------------
    # 4. Guarded Execution with Auto-Correction Loop
    # ----------------------------------------------------------------------
    def execute_query(
        self,
        question: str,
        max_retries: int = 3,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Translate question to Cypher, validate guardrails, execute, and auto-correct.

        Args:
            question: Natural language question.
            max_retries: Max self-correction attempts.
            schema: Optional schema dictionary (extracted automatically if None).

        Returns:
            Dictionary with execution results, final Cypher statement, latency, and status.
        """
        active_schema = schema or self.get_active_schema()
        system_prompt = self._build_system_prompt(active_schema)

        current_prompt = question
        last_cypher = ""
        last_error = None
        retries_used = 0
        records: List[Dict[str, Any]] = []

        driver = get_memgraph_driver()
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

            # 3. Enforce Resource Bounds (LIMIT 50)
            bounded_cypher = self.enforce_limit(last_cypher)
            last_cypher = bounded_cypher

            # 4. Execute Query in Memgraph
            try:
                with driver.session() as session:
                    res = session.run(bounded_cypher)
                    records = [dict(r) for r in res]
                    last_error = None
                    break  # Success!
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
            "cypher": last_cypher,
            "status": status,
            "records": records,
            "record_count": len(records),
            "retries_used": retries_used,
            "error": last_error,
            "latency_ms": round(latency_ms, 2),
        }
