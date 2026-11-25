"""Mechanized documentation integrity, AST crosswalk, and drift tests (ADR-00040)."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"
MKDOCS_PATH = REPO_ROOT / "mkdocs.yml"
DOCS_DIR = REPO_ROOT / "docs"
TECH_REPORT_DIR = DOCS_DIR / "technical-report"
REFERENCE_DIR = DOCS_DIR / "reference"
ATLAS_PATH = TECH_REPORT_DIR / "00-system-atlas.md"

RUNTIME_PACKAGES: Tuple[str, ...] = (
    "contagion",
    "curation",
    "embedding",
    "explorer",
    "ingestion",
    "knowledge_graph",
    "ontology",
    "retrieval",
    "sft_pipeline",
    "streaming",
    "synthesis",
    "tools",
    "universe",
)

# Construct host-path pattern without embedding literal forbidden prefixes in source text.
_FORBIDDEN_HOST_PREFIXES = (
    "file:" + "///",
    "/" + "home" + "/",
    "/" + "Users" + "/",
)
FORBIDDEN_ABS_RE = re.compile(
    r"(?:"
    + "|".join(re.escape(p) for p in _FORBIDDEN_HOST_PREFIXES)
    + r"|[A-Za-z]:\\\\)"
)

# Zero-width, soft-hyphen, BOM, and bidirectional override characters.
INVISIBLE_UNICODE_RE = re.compile(
    r"[\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]"
)

CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")
MKDOCSTRINGS_DIRECTIVE_RE = re.compile(r"^:::\s+([A-Za-z_][A-Za-z0-9_.]*)\s*$", re.MULTILINE)
CROSSWALK_SYMBOL_RE = re.compile(r"`([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+:[A-Za-z_][A-Za-z0-9_.]*)`")
MERMAID_BLOCK_RE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)
MERMAID_BI_EDGE_RE = re.compile(r"^\s*([a-z_]+)\s*<-->\s*([a-z_]+)\s*$")
MERMAID_UNI_EDGE_RE = re.compile(r"^\s*([a-z_]+)\s*-->\s*([a-z_]+)\s*$")


class _MkDocsYamlLoader(yaml.SafeLoader):
    """Safe YAML loader tolerating MkDocs `!!python/name:` tags as inert string tokens."""


def _python_name_constructor(loader: yaml.Loader, suffix: str, node: yaml.Node) -> str:
    del loader, node
    return f"python/name:{suffix}"


_MkDocsYamlLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/name:",
    _python_name_constructor,
)


def _strip_code_blocks(text: str) -> str:
    return CODE_FENCE_RE.sub("", text)


def _governed_markdown_files() -> List[Path]:
    files: List[Path] = [README_PATH]
    if (DOCS_DIR / "index.md").is_file():
        files.append(DOCS_DIR / "index.md")
    files.extend(sorted(TECH_REPORT_DIR.glob("*.md")))
    files.extend(sorted(REFERENCE_DIR.glob("*.md")))
    return files


def _resolve_module_file(mod_path: str) -> Optional[Path]:
    rel = Path(*mod_path.split("."))
    py_file = REPO_ROOT / f"{rel}.py"
    if py_file.is_file():
        return py_file
    pkg_init = REPO_ROOT / rel / "__init__.py"
    if pkg_init.is_file():
        return pkg_init
    return None


def _ast_defines_symbol(py_file: Path, symbol_chain: str) -> bool:
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    parts = symbol_chain.split(".")
    top_name = parts[0]

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == top_name:
            return len(parts) == 1
        if isinstance(node, ast.ClassDef) and node.name == top_name:
            if len(parts) == 1:
                return True
            method_name = parts[1]
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                    return len(parts) == 2
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and item.target.id == method_name:
                    return len(parts) == 2
                if isinstance(item, ast.Assign):
                    for tgt in item.targets:
                        if isinstance(tgt, ast.Name) and tgt.id == method_name:
                            return len(parts) == 2
            return False
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == top_name:
            return len(parts) == 1
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == top_name:
                    return len(parts) == 1
    return False


def _collect_nav_targets(nav_node: Any) -> Iterable[str]:
    if isinstance(nav_node, str):
        yield nav_node
    elif isinstance(nav_node, list):
        for item in nav_node:
            yield from _collect_nav_targets(item)
    elif isinstance(nav_node, dict):
        for val in nav_node.values():
            yield from _collect_nav_targets(val)


def _compute_ast_package_edges() -> Tuple[Set[Tuple[str, str]], Dict[str, int]]:
    pkg_set = set(RUNTIME_PACKAGES)
    edges: Set[Tuple[str, str]] = set()
    counts: Dict[str, int] = {}

    for pkg in RUNTIME_PACKAGES:
        py_files = sorted((REPO_ROOT / pkg).rglob("*.py"))
        counts[pkg] = len(py_files)
        for py_file in py_files:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if top in pkg_set and top != pkg:
                            edges.add((pkg, top))
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    top = node.module.split(".")[0]
                    if top in pkg_set and top != pkg:
                        edges.add((pkg, top))
    return edges, counts


def test_no_absolute_paths_or_invisible_watermarks_in_docs() -> None:
    targets = [*_governed_markdown_files(), MKDOCS_PATH]
    assert ATLAS_PATH.is_file(), "Expected Chapter 0 System Atlas at docs/technical-report/00-system-atlas.md"
    for path in targets:
        assert path.is_file(), f"Missing governed documentation file: {path.relative_to(REPO_ROOT)}"
        content = path.read_text(encoding="utf-8")
        assert not FORBIDDEN_ABS_RE.search(content), (
            f"Forbidden host-specific path or URI found in {path.relative_to(REPO_ROOT)}"
        )
        assert not INVISIBLE_UNICODE_RE.search(content), (
            f"Invisible control or zero-width Unicode character found in {path.relative_to(REPO_ROOT)}"
        )


def test_all_local_markdown_links_resolve() -> None:
    for path in _governed_markdown_files():
        body = _strip_code_blocks(path.read_text(encoding="utf-8"))
        for _, raw_target in LINK_RE.findall(body):
            target = raw_target.split("#", 1)[0].strip()
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            assert not FORBIDDEN_ABS_RE.search(target), (
                f"Forbidden link target in {path.relative_to(REPO_ROOT)}: {raw_target}"
            )
            resolved = (path.parent / target).resolve()
            assert resolved.exists(), (
                f"Broken relative link in {path.relative_to(REPO_ROOT)}: {raw_target} -> {resolved}"
            )


def test_blog_portability_no_mkdocstrings_directives_in_technical_reports() -> None:
    reports = sorted(TECH_REPORT_DIR.glob("*.md"))
    assert reports, "Expected at least Chapter 0 under docs/technical-report/"
    for path in reports:
        body = _strip_code_blocks(path.read_text(encoding="utf-8"))
        matches = MKDOCSTRINGS_DIRECTIVE_RE.findall(body)
        assert not matches, (
            f"ADR-00040 Invariant #5 violation: mkdocstrings ':::' directive(s) {matches} found in "
            f"{path.relative_to(REPO_ROOT)} (isolate inside docs/reference/)"
        )


def test_mkdocs_yml_nav_and_mkdocstrings_targets_resolve() -> None:
    assert MKDOCS_PATH.is_file(), "Expected root mkdocs.yml"
    cfg = yaml.load(MKDOCS_PATH.read_text(encoding="utf-8"), Loader=_MkDocsYamlLoader)
    assert isinstance(cfg, dict)
    docs_dir = REPO_ROOT / str(cfg.get("docs_dir", "docs"))
    assert docs_dir.is_dir()

    nav_targets = list(_collect_nav_targets(cfg.get("nav", [])))
    assert nav_targets, "Expected non-empty nav in mkdocs.yml"
    for rel_target in nav_targets:
        resolved = (docs_dir / rel_target).resolve()
        assert resolved.is_file(), f"Broken mkdocs.yml nav entry: {rel_target} -> {resolved}"

    seen_top_packages: Set[str] = set()
    ref_files = sorted(REFERENCE_DIR.glob("*.md"))
    assert ref_files, "Expected API reference pages under docs/reference/"
    for ref_path in ref_files:
        text = _strip_code_blocks(ref_path.read_text(encoding="utf-8"))
        for mod_target in MKDOCSTRINGS_DIRECTIVE_RE.findall(text):
            mod_file = _resolve_module_file(mod_target)
            assert mod_file is not None, (
                f"Unresolvable mkdocstrings target '::: {mod_target}' in {ref_path.relative_to(REPO_ROOT)}"
            )
            seen_top_packages.add(mod_target.split(".")[0])

    assert seen_top_packages == set(RUNTIME_PACKAGES), (
        f"Expected docs/reference/ to cover all 13 runtime packages; "
        f"missing={set(RUNTIME_PACKAGES) - seen_top_packages}, extra={seen_top_packages - set(RUNTIME_PACKAGES)}"
    )


def test_crosswalk_module_symbol_references_resolve_in_ast() -> None:
    reports = sorted(TECH_REPORT_DIR.glob("*.md"))
    total_checked = 0
    for report_path in reports:
        body = _strip_code_blocks(report_path.read_text(encoding="utf-8"))
        refs = CROSSWALK_SYMBOL_RE.findall(body)
        for ref in refs:
            mod_name, sym_name = ref.split(":", 1)
            py_file = _resolve_module_file(mod_name)
            assert py_file is not None, (
                f"Missing module '{mod_name}' cited as `{ref}` in {report_path.relative_to(REPO_ROOT)}"
            )
            assert _ast_defines_symbol(py_file, sym_name), (
                f"Missing symbol '{sym_name}' in {py_file.relative_to(REPO_ROOT)} "
                f"(cited as `{ref}` in {report_path.relative_to(REPO_ROOT)})"
            )
            total_checked += 1
    assert total_checked >= 20, f"Expected >=20 AST-verified crosswalk symbols, checked {total_checked}"


def test_chapter0_mermaid_package_dependency_graph_matches_ast() -> None:
    atlas_text = ATLAS_PATH.read_text(encoding="utf-8")
    mermaid_blocks = MERMAID_BLOCK_RE.findall(atlas_text)
    dep_blocks = [b for b in mermaid_blocks if "flowchart TD" in b and "contagion" in b and "curation" in b]
    assert len(dep_blocks) == 1, "Expected exactly one inter-package flowchart TD block in 00-system-atlas.md"

    doc_edges: Set[Tuple[str, str]] = set()
    pkg_set = set(RUNTIME_PACKAGES)
    for line in dep_blocks[0].splitlines():
        m_bi = MERMAID_BI_EDGE_RE.match(line)
        if m_bi:
            u, v = m_bi.groups()
            assert u in pkg_set and v in pkg_set
            doc_edges.add((u, v))
            doc_edges.add((v, u))
            continue
        m_uni = MERMAID_UNI_EDGE_RE.match(line)
        if m_uni:
            u, v = m_uni.groups()
            assert u in pkg_set and v in pkg_set
            doc_edges.add((u, v))

    ast_edges, ast_counts = _compute_ast_package_edges()
    assert doc_edges == ast_edges, (
        f"Chapter 0 Mermaid package graph drifted from AST imports:\n"
        f"  missing_in_doc={sorted(ast_edges - doc_edges)}\n"
        f"  extra_in_doc={sorted(doc_edges - ast_edges)}"
    )

    for pkg, count in ast_counts.items():
        row_pattern = re.compile(rf"^\|\s*`{re.escape(pkg)}`\s*\|\s*(\d+)\s*\|", re.MULTILINE)
        match = row_pattern.search(atlas_text)
        assert match is not None, f"Missing package inventory row for `{pkg}` in 00-system-atlas.md"
        doc_count = int(match.group(1))
        assert doc_count == count, (
            f"Package `{pkg}` .py file count drifted in 00-system-atlas.md: doc={doc_count} vs actual={count}"
        )


def test_chapter0_live_constants_parity() -> None:
    from contagion.cascade import SparseNetworkCascadeEngine
    from embedding.embedder import get_embedding_dim
    from ingestion.segmentation.chunker import FinancialSectionSegmenter
    from knowledge_graph.ledger import DEFAULT_MEMGRAPH_EVENT_TTL_DAYS
    from ontology.schema import EntityClass, RelationPredicate
    from retrieval.pipeline import MAX_SUBGRAPH_EDGES
    from retrieval.reporting import (
        DEFAULT_FILING_GROUNDED_MAX_SHARE,
        DEFAULT_STRESS_UPPER_BOUND_MIN_SHARE,
    )

    segmenter = FinancialSectionSegmenter()
    cascade = SparseNetworkCascadeEngine()

    assert MAX_SUBGRAPH_EDGES == 25
    assert get_embedding_dim() == 1024
    assert DEFAULT_MEMGRAPH_EVENT_TTL_DAYS == 90
    assert segmenter.max_tokens == 800
    assert segmenter.overlap_tokens == 120
    assert DEFAULT_FILING_GROUNDED_MAX_SHARE == pytest.approx(0.10)
    assert DEFAULT_STRESS_UPPER_BOUND_MIN_SHARE == pytest.approx(0.50)
    assert len(EntityClass) == 20
    assert len(RelationPredicate) == 26
    assert cascade.gamma == pytest.approx(0.50)

    atlas_text = ATLAS_PATH.read_text(encoding="utf-8")
    expected_snippets = (
        "MAX_SUBGRAPH_EDGES = 25",
        "EMBEDDING_DIM = 1024",
        "MEMGRAPH_EVENT_TTL_DAYS = 90",
        "max_tokens=800",
        "overlap=120",
        "0.10",
        "0.50",
        "20 EntityClasses",
        "26 RelationPredicates",
    )
    for snippet in expected_snippets:
        assert snippet in atlas_text, f"Expected live constant snippet '{snippet}' in 00-system-atlas.md"


def test_chapter1_live_constants_parity() -> None:
    from datetime import time

    from ingestion.market.parser import MarketMetricsCalculator
    from ingestion.news.client import TokenBucket
    from ingestion.sec.disruption_gating import DEFAULT_MAX_FILING_CHARS
    from ingestion.sec.parser import should_glean_chunk
    from ingestion.segmentation.chunker import FinancialSectionSegmenter
    from retrieval.temporal import EARLY_CLOSE_ET, REGULAR_CLOSE_ET
    from universe.sp500 import DICTIONARY_TICKERS

    ch1_path = TECH_REPORT_DIR / "01-pit-universe-and-ingestion.md"
    assert ch1_path.is_file(), "Expected Chapter 1 at docs/technical-report/01-pit-universe-and-ingestion.md"

    assert REGULAR_CLOSE_ET == time(16, 0, 0)
    assert EARLY_CLOSE_ET == time(13, 0, 0)
    assert DEFAULT_MAX_FILING_CHARS == 50_000
    assert len(DICTIONARY_TICKERS) == 28

    segmenter = FinancialSectionSegmenter()
    assert segmenter.max_tokens == 800
    assert segmenter.overlap_tokens == 120

    # Verify 0.85 * 512 = 435 gleaning token threshold boundary
    assert should_glean_chunk("Item 1 - Business", 0, 434, False, max_new_tokens=512, enabled=True) is False
    assert should_glean_chunk("Item 1 - Business", 0, 435, False, max_new_tokens=512, enabled=True) is True

    bucket = TokenBucket()
    assert bucket.capacity == pytest.approx(60.0)
    assert bucket.refill_rate == pytest.approx(1.0)

    calc = MarketMetricsCalculator()
    assert callable(calc.compute_event_window_metrics)

    ch1_text = ch1_path.read_text(encoding="utf-8")
    for snippet in (
        "16:00:00",
        "13:00:00",
        "50_000",
        "max_tokens = 800",
        "overlap_tokens = 120",
        "0.85",
        "capacity = 60",
        "+2.0",
        "-1.5",
        "-3.0",
    ):
        assert snippet in ch1_text, f"Expected Chapter 1 constant snippet '{snippet}' in 01-pit-universe-and-ingestion.md"


def test_mkdocs_strict_build_when_installed(tmp_path: Path) -> None:
    required = ("mkdocs", "material", "mkdocstrings")
    if any(importlib.util.find_spec(mod) is None for mod in required):
        pytest.skip("Optional [docs] extra (mkdocs-material, mkdocstrings) is not installed")

    proc = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--clean", "--site-dir", str(tmp_path / "site")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"mkdocs build failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
