"""AST & static architecture invariant tests across YarnBall runtime subsystems (ADR-00032)."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_PACKAGES = (
    "contagion",
    "curation",
    "embedding",
    "explorer",
    "ingestion",
    "knowledge_graph",
    "ontology",
    "retrieval",
    "streaming",
    "synthesis",
    "tools",
    "universe",
)


def _iter_python_files(packages: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for pkg in packages:
        pkg_dir = REPO_ROOT / pkg
        if pkg_dir.exists():
            files.extend(sorted(pkg_dir.rglob("*.py")))
    return files


def _find_function_node(tree: ast.AST, func_name: str) -> Optional[ast.FunctionDef | ast.AsyncFunctionDef]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            return node
    return None


def _has_call_with_kwarg(func_node: ast.AST, target_call: str, required_kwarg: str) -> bool:
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        called_name = ""
        if isinstance(func, ast.Name):
            called_name = func.id
        elif isinstance(func, ast.Attribute):
            called_name = func.attr
        if called_name == target_call and any(kw.arg == required_kwarg for kw in node.keywords):
            return True
    return False


def test_adr00030_no_direct_ollama_imports_outside_sft_pipeline() -> None:
    """Enforce ADR-00030: runtime subsystems outside sft_pipeline/ must not import ollama."""
    offenders: List[str] = []
    for py_path in _iter_python_files(RUNTIME_PACKAGES):
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "ollama" or alias.name.startswith("ollama."):
                        offenders.append(f"{py_path.relative_to(REPO_ROOT)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "ollama" or node.module.startswith("ollama."):
                    offenders.append(f"{py_path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert not offenders, f"Forbidden direct ollama imports outside sft_pipeline/: {offenders}"


def test_adr00024_active_memgraph_writers_pass_source_text_to_ontology_gate() -> None:
    """Enforce ADR-00023/24: Memgraph write entrypoints validate payloads with source_text."""
    targets = [
        ("knowledge_graph/graph_store.py", "store_graph_entities"),
        ("ingestion/sec/harvester.py", "_persist_sec_graph_data"),
    ]
    for rel_path, fn_name in targets:
        file_path = REPO_ROOT / rel_path
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        fn_node = _find_function_node(tree, fn_name)
        assert fn_node is not None, f"Expected function '{fn_name}' in {rel_path}"
        assert _has_call_with_kwarg(fn_node, "validate_graph_payload", "source_text"), (
            f"{rel_path}:{fn_name} must invoke validate_graph_payload(..., source_text=...) "
            "to enforce clause-scoped negation/hedging filtering (ADR-00024)"
        )
