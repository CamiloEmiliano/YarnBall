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
        "`800`-token",
        "`120`-token",
        "0.10",
        "0.50",
        "20 `EntityClass`",
        "26 `RelationPredicate`",
    )
    for snippet in expected_snippets:
        assert snippet in atlas_text, f"Expected live constant snippet '{snippet}' in 00-system-atlas.md"


def test_mermaid_diagrams_strict_syntax() -> None:
    """Prevent Mermaid Jison/parseMember crashes (e.g., unbalanced `[a, b)` parens in classDiagram)."""
    class_open_re = re.compile(r"^class\s+[A-Za-z_][A-Za-z0-9_]*\s*\{\s*$")
    stereotype_re = re.compile(r"^<<[A-Za-z0-9_]+>>$")
    attr_re = re.compile(r"^[+#~-][A-Za-z0-9_~]+\s+[A-Za-z0-9_]+$")
    method_re = re.compile(r"^[+#~-][A-Za-z0-9_]+\([A-Za-z0-9_, ]*\)(?:\s+[A-Za-z0-9_~]+)?$")
    rel_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\s+-->\s+[A-Za-z_][A-Za-z0-9_]*\s*:\s*[A-Za-z0-9_, ./+-]+$")
    unquoted_shape_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*[\[\{](?!\")")
    quoted_label_re = re.compile(r'"([^"]+)"')
    xml_special_in_svg_label_re = re.compile(r"[<>&']")

    targets = [README_PATH, *sorted(TECH_REPORT_DIR.glob("*.md"))]
    checked_blocks = 0
    for path in targets:
        blocks = MERMAID_BLOCK_RE.findall(path.read_text(encoding="utf-8"))
        for block in blocks:
            checked_blocks += 1
            raw_lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
            lines = [ln for ln in raw_lines if not ln.startswith("%%")]
            assert lines, f"Empty mermaid block in {path.relative_to(REPO_ROOT)}"
            header = lines[0]

            if header == "classDiagram":
                in_class = False
                for ln in lines[1:]:
                    if ln.startswith("direction "):
                        assert ln in ("direction TB", "direction LR", "direction BT", "direction RL")
                        continue
                    if not in_class:
                        if class_open_re.match(ln):
                            in_class = True
                        else:
                            assert rel_re.match(ln), (
                                f"Invalid classDiagram relation '{ln}' in {path.relative_to(REPO_ROOT)}"
                            )
                    else:
                        if ln == "}":
                            in_class = False
                        else:
                            assert stereotype_re.match(ln) or attr_re.match(ln) or method_re.match(ln), (
                                f"Unsafe classDiagram member '{ln}' in {path.relative_to(REPO_ROOT)} "
                                "(must be strict +Type attr or +method(args) ReturnType without '[a, b)' or '..')"
                            )
                assert not in_class, f"Unclosed class block in {path.relative_to(REPO_ROOT)}"
            elif header.startswith("flowchart "):
                assert raw_lines[0].startswith('%%{init: {"flowchart": {"htmlLabels": false'), (
                    f"Flowchart in {path.relative_to(REPO_ROOT)} must disable htmlLabels (<foreignObject> clipping)"
                )
                for ln in lines[1:]:
                    assert not unquoted_shape_re.search(ln), (
                        f"Unquoted flowchart node label '{ln}' in {path.relative_to(REPO_ROOT)} (use [\"...\"] or {{\"...\"}})"
                    )
                    for label_text in quoted_label_re.findall(ln):
                        assert not xml_special_in_svg_label_re.search(label_text), (
                            f"Angle bracket or single-quote inside SVG flowchart label '{label_text}' in {path.relative_to(REPO_ROOT)}"
                        )
                        assert len(label_text) <= 52, (
                            f"Flowchart label '{label_text}' ({len(label_text)} chars) exceeds 52-char width budget in {path.relative_to(REPO_ROOT)}"
                        )
                        for tok in label_text.split():
                            assert len(tok) <= 32, (
                                f"Unbroken token '{tok}' ({len(tok)} chars) in {path.relative_to(REPO_ROOT)} risks box clipping"
                            )
            else:
                pytest.fail(f"Unsupported Mermaid diagram header '{header}' in {path.relative_to(REPO_ROOT)}")

    assert checked_blocks >= 18, f"Expected >=18 Mermaid diagrams across README and Ch. 0-4, found {checked_blocks}"


def test_display_equations_sequentially_numbered() -> None:
    """Every display equation in Chapters 0-4 must carry a sequential \\tag{C.N} label."""
    display_eq_re = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
    tag_re = re.compile(r"\\tag\{(\d+)\.(\d+)\}")

    total_equations = 0
    for path in sorted(TECH_REPORT_DIR.glob("[0-9][0-9]-*.md")):
        chapter_num = int(path.name[:2])
        text = _strip_code_blocks(path.read_text(encoding="utf-8"))
        equations = display_eq_re.findall(text)
        assert equations, f"Expected display equations in {path.relative_to(REPO_ROOT)}"
        for expected_idx, eq_body in enumerate(equations, start=1):
            tags = tag_re.findall(eq_body)
            assert len(tags) == 1, (
                f"Display equation #{expected_idx} in {path.name} must have exactly one \\tag{{C.N}}, found {tags}"
            )
            tag_ch, tag_eq = int(tags[0][0]), int(tags[0][1])
            assert (tag_ch, tag_eq) == (chapter_num, expected_idx), (
                f"Expected \\tag{{{chapter_num}.{expected_idx}}} in {path.name}, found \\tag{{{tag_ch}.{tag_eq}}}"
            )
            total_equations += 1

    assert total_equations >= 21, f"Expected >=21 numbered display equations across Ch. 0-4, found {total_equations}"


def test_tables_and_figures_sequentially_numbered() -> None:
    """Monograph Standard: every Table (caption above) and Figure (caption below) in Ch. 0-4 is numbered C.N."""
    table_sep_re = re.compile(r"^\|\s*:?-{3,}:?\s*\|")
    table_cap_re = re.compile(r"^\*\*Table (\d+)\.(\d+):\*\*\s+\*.+\*$")
    figure_cap_re = re.compile(r"^\*\*Figure (\d+)\.(\d+):\*\*\s+\*.+\*$")

    total_tables = 0
    total_figures = 0
    for path in sorted(TECH_REPORT_DIR.glob("[0-9][0-9]-*.md")):
        chapter_num = int(path.name[:2])
        lines = path.read_text(encoding="utf-8").splitlines()
        in_code = False
        code_is_mermaid = False
        expected_tbl = 1
        expected_fig = 1

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("```"):
                if not in_code:
                    in_code = True
                    code_is_mermaid = stripped.startswith("```mermaid")
                else:
                    if code_is_mermaid:
                        # Next non-blank line after closing ``` must be **Figure C.N:** *...*
                        nxt = idx + 1
                        while nxt < len(lines) and not lines[nxt].strip():
                            nxt += 1
                        assert nxt < len(lines), f"Missing Figure caption after mermaid block in {path.name}"
                        m_fig = figure_cap_re.match(lines[nxt].strip())
                        assert m_fig is not None, (
                            f"Expected '**Figure {chapter_num}.{expected_fig}:** *...*' below diagram at {path.name}:{idx + 1}, "
                            f"found '{lines[nxt].strip()}'"
                        )
                        assert (int(m_fig.group(1)), int(m_fig.group(2))) == (chapter_num, expected_fig), (
                            f"Out-of-sequence Figure caption in {path.name}: expected {chapter_num}.{expected_fig}, "
                            f"found {m_fig.group(1)}.{m_fig.group(2)}"
                        )
                        expected_fig += 1
                        total_figures += 1
                    in_code = False
                    code_is_mermaid = False
                continue

            if not in_code and table_sep_re.match(stripped):
                # Previous non-blank line before header row (idx - 1) must be **Table C.N:** *...*
                prev = idx - 2
                while prev >= 0 and not lines[prev].strip():
                    prev -= 1
                assert prev >= 0, f"Missing Table caption above table at {path.name}:{idx}"
                m_tbl = table_cap_re.match(lines[prev].strip())
                assert m_tbl is not None, (
                    f"Expected '**Table {chapter_num}.{expected_tbl}:** *...*' above table at {path.name}:{idx}, "
                    f"found '{lines[prev].strip()}'"
                )
                assert (int(m_tbl.group(1)), int(m_tbl.group(2))) == (chapter_num, expected_tbl), (
                    f"Out-of-sequence Table caption in {path.name}: expected {chapter_num}.{expected_tbl}, "
                    f"found {m_tbl.group(1)}.{m_tbl.group(2)}"
                )
                expected_tbl += 1
                total_tables += 1

    assert total_tables >= 20, f"Expected >=20 numbered tables across Ch. 0-4, found {total_tables}"
    assert total_figures >= 18, f"Expected >=18 numbered figures across Ch. 0-4, found {total_figures}"


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


def test_chapter2_live_constants_parity() -> None:
    import inspect

    from embedding.embedder import _MIN_NORM_SQ, get_embedding_dim
    from knowledge_graph.ledger import DEFAULT_MEMGRAPH_EVENT_TTL_DAYS, FORBIDDEN_MEMGRAPH_PROSE_KEYS
    from ontology.convergence import evaluate_calibration_batch
    from ontology.schema import EntityClass, RelationPredicate
    from ontology.validation import check_degree_anomaly_quarantine

    ch2_path = TECH_REPORT_DIR / "02-ontology-linkage-and-ledger.md"
    assert ch2_path.is_file(), "Expected Chapter 2 at docs/technical-report/02-ontology-linkage-and-ledger.md"

    assert len(EntityClass) == 20
    assert len(RelationPredicate) == 26
    assert get_embedding_dim() == 1024
    assert _MIN_NORM_SQ == pytest.approx(1e-8)
    assert DEFAULT_MEMGRAPH_EVENT_TTL_DAYS == 90
    assert len(FORBIDDEN_MEMGRAPH_PROSE_KEYS) == 5

    deg_sig = inspect.signature(check_degree_anomaly_quarantine)
    assert deg_sig.parameters["degree_burst_threshold"].default == 25

    conv_sig = inspect.signature(evaluate_calibration_batch)
    assert conv_sig.parameters["min_support"].default == 3
    assert conv_sig.parameters["target_max_rate"].default == pytest.approx(0.02)

    ch2_text = ch2_path.read_text(encoding="utf-8")
    for snippet in (
        "20 `EntityClass`",
        "26 `RelationPredicate`",
        "EMBEDDING_DIM = 1024",
        "1e-8",
        "DEFAULT_MEMGRAPH_EVENT_TTL_DAYS = 90",
        "0.88",
        "degree_burst_threshold = 25",
        "target_max_rate = 0.02",
    ):
        assert snippet in ch2_text, f"Expected Chapter 2 constant snippet '{snippet}' in 02-ontology-linkage-and-ledger.md"


def test_chapter3_live_constants_parity() -> None:
    from contagion.cascade import SparseNetworkCascadeEngine
    from retrieval.pipeline import MAX_SUBGRAPH_EDGES
    from retrieval.reporting import (
        DEFAULT_FILING_GROUNDED_MAX_SHARE,
        DEFAULT_STRESS_UPPER_BOUND_MIN_SHARE,
    )
    from retrieval.temporal import MAX_TRAVERSAL_HOPS, MIN_TRAVERSAL_HOPS

    ch3_path = TECH_REPORT_DIR / "03-retrieval-contagion-and-reporting.md"
    assert ch3_path.is_file(), "Expected Chapter 3 at docs/technical-report/03-retrieval-contagion-and-reporting.md"

    engine = SparseNetworkCascadeEngine()
    assert engine.gamma == pytest.approx(0.50)
    assert engine.friction_floor == pytest.approx(0.05)
    assert engine.max_iterations == 5
    assert engine.tolerance == pytest.approx(1e-9)
    assert engine.disruption_threshold_pct == pytest.approx(25.0)
    assert MIN_TRAVERSAL_HOPS == 1
    assert MAX_TRAVERSAL_HOPS == 10
    assert MAX_SUBGRAPH_EDGES == 25
    assert DEFAULT_FILING_GROUNDED_MAX_SHARE == pytest.approx(0.10)
    assert DEFAULT_STRESS_UPPER_BOUND_MIN_SHARE == pytest.approx(0.50)

    ch3_text = ch3_path.read_text(encoding="utf-8")
    for snippet in (
        "gamma = 0.50",
        "friction_floor = 0.05",
        "max_iterations = 5",
        "1e-9",
        "disruption_threshold_pct = 25.0",
        "MIN_TRAVERSAL_HOPS = 1",
        "MAX_TRAVERSAL_HOPS = 10",
        "MAX_SUBGRAPH_EDGES = 25",
        "0.10",
        "0.50",
    ):
        assert snippet in ch3_text, f"Expected Chapter 3 constant snippet '{snippet}' in 03-retrieval-contagion-and-reporting.md"


def test_chapter4_live_constants_parity() -> None:
    import inspect

    from curation.active_learning.cartographer import DatasetCartographer
    from curation.cold_start.coreset import FacilityLocationCoresetSelector
    from curation.cold_start.minhash_lsh import MinHashLSHDeduplicator
    from curation.cold_start.semdedup import SemanticDeduplicator
    from sft_pipeline.sampler import DOMINANT_CLASS_CAP, RARE_RELATION_FLOORS, ManifoldTargetedSampler

    ch4_path = TECH_REPORT_DIR / "04-curation-and-model-foundry.md"
    assert ch4_path.is_file(), "Expected Chapter 4 at docs/technical-report/04-curation-and-model-foundry.md"

    lsh = MinHashLSHDeduplicator()
    assert lsh.num_perm == 128
    assert lsh.jaccard_threshold == pytest.approx(0.85)
    assert lsh.shingle_size == 5
    assert lsh.num_bands == 16

    sem = SemanticDeduplicator()
    assert sem.cosine_threshold == pytest.approx(0.88)

    coreset = FacilityLocationCoresetSelector()
    assert coreset.retention_fraction == pytest.approx(0.80)

    carto = DatasetCartographer()
    assert carto.conf_thresh_high == pytest.approx(0.70)
    assert carto.conf_thresh_low == pytest.approx(0.30)
    assert carto.var_thresh == pytest.approx(0.15)
    filt_sig = inspect.signature(DatasetCartographer.filter_active_learning)
    assert filt_sig.parameters["prune_easy_ratio"].default == pytest.approx(0.80)

    assert DOMINANT_CLASS_CAP == 1500
    assert RARE_RELATION_FLOORS["SOLE_SOURCE_DEPENDENT_ON"] == 400
    assert RARE_RELATION_FLOORS["ACQUIRED_BY"] == 500
    samp_sig = inspect.signature(ManifoldTargetedSampler.__init__)
    assert samp_sig.parameters["null_sample_ratio"].default == pytest.approx(0.20)

    ch4_text = ch4_path.read_text(encoding="utf-8")
    for snippet in (
        "num_perm = 128",
        "jaccard_threshold = 0.85",
        "shingle_size = 5",
        "num_bands = 16",
        "cosine_threshold = 0.88",
        "retention_fraction = 0.80",
        "conf_thresh_high = 0.70",
        "conf_thresh_low = 0.30",
        "var_thresh = 0.15",
        "prune_easy_ratio = 0.80",
        "DOMINANT_CLASS_CAP = 1500",
        "null_sample_ratio = 0.20",
    ):
        assert snippet in ch4_text, f"Expected Chapter 4 constant snippet '{snippet}' in 04-curation-and-model-foundry.md"


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
