"""Deterministic ADR governance & link-integrity tests (ADR-00032)."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
ADR_DIR = REPO_ROOT / "docs" / "decisions"
INDEX_PATH = ADR_DIR / "index.md"

FILENAME_RE = re.compile(r"^ADR-(\d{5})-[a-z0-9-]{40}\.md$")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
FORBIDDEN_ABS_RE = re.compile(r"(?:file:///|/home/|/Users/|[A-Za-z]:\\\\)")
CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
HEADER_STATUS_RE = re.compile(r"^-\s+\*\*Status\*\*:\s*([A-Z0-9_ -]+)$", re.MULTILINE)
HEADER_COMMITS_RE = re.compile(r"^-\s+\*\*Linked Commits\*\*:\s*(.+)$", re.MULTILINE)
INDEX_ROW_RE = re.compile(
    r"^\|\s*\[(ADR-\d{5})\]\(([^)]+)\)\s*\|[^|]+\|[^|]+\|\s*`([^`]+)`\s*\|[^|]+\|\s*(.+?)\s*\|$"
)
HASH_LINK_RE = re.compile(r"\[`([0-9a-f]{7,40})`\]\(([^)]+)\)")

# Historical decommissioned paths intentionally linked in ADRs (ADR-00016, ADR-00034)
DECOMMISSIONED_ADR_LINKS: Dict[str, Tuple[str, ...]] = {
    "ADR-00016-decommission-legacy-chainlit-ui-frontend.md": (
        "interactive/",
        "tests/interactive/",
        "chainlit.md",
    ),
    "index.md": (
        "retrieval/shock_propagation.py",
    ),
    "ADR-00031-iterative-predicate-convergence-protocol.md": (
        "retrieval/shock_propagation.py",
    ),
    "ADR-00034-single-bi-read-facade-and-pipeline-merge.md": (
        "retrieval/shock_propagation.py",
    ),
}


def _strip_code_blocks(text: str) -> str:
    return CODE_FENCE_RE.sub("", text)


def _adr_files() -> List[Path]:
    return sorted(ADR_DIR.glob("ADR-*.md"))


def _parse_index_rows() -> Dict[str, Tuple[str, str, str]]:
    rows: Dict[str, Tuple[str, str, str]] = {}
    for line in INDEX_PATH.read_text(encoding="utf-8").splitlines():
        match = INDEX_ROW_RE.match(line.strip())
        if match:
            adr_id, target, status, commits_col = match.groups()
            rows[adr_id] = (target.strip(), status.strip(), commits_col.strip())
    return rows


def test_adr_filenames_strict_53_chars() -> None:
    files = _adr_files()
    assert files, "Expected ADR documents in docs/decisions/"
    for path in files:
        assert len(path.name) == 53, f"{path.name} has length {len(path.name)} (expected 53)"
        assert FILENAME_RE.match(path.name), f"{path.name} violates ADR-XXXXX-<40-char-slug>.md"


def test_no_absolute_paths_in_decisions_docs() -> None:
    docs = [INDEX_PATH, *_adr_files()]
    for path in docs:
        content = path.read_text(encoding="utf-8")
        assert not FORBIDDEN_ABS_RE.search(content), f"Forbidden absolute path or URI found in {path.name}"


def test_all_local_links_in_decisions_resolve() -> None:
    docs = [INDEX_PATH, *_adr_files()]
    for path in docs:
        body = _strip_code_blocks(path.read_text(encoding="utf-8"))
        exempt_prefixes = DECOMMISSIONED_ADR_LINKS.get(path.name, ())
        for _, raw_target in LINK_RE.findall(body):
            target = raw_target.split("#", 1)[0].strip()
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            assert not FORBIDDEN_ABS_RE.search(target), f"Absolute link target in {path.name}: {raw_target}"
            normalized = target.lstrip("./")
            if any(prefix in normalized for prefix in exempt_prefixes):
                continue
            resolved = (path.parent / target).resolve()
            assert resolved.exists(), f"Broken link in {path.name}: {raw_target} -> {resolved}"


def test_index_registry_status_and_commit_links_parity() -> None:
    rows = _parse_index_rows()
    has_git = (REPO_ROOT / ".git").exists()

    for path in _adr_files():
        m = FILENAME_RE.match(path.name)
        assert m is not None
        adr_num = int(m.group(1))
        adr_id = f"ADR-{m.group(1)}"

        assert adr_id in rows, f"{adr_id} missing from docs/decisions/index.md"
        idx_target, idx_status, idx_commits = rows[adr_id]
        assert idx_target == path.name, f"index.md link mismatch for {adr_id}: {idx_target} != {path.name}"

        text = path.read_text(encoding="utf-8")
        status_match = HEADER_STATUS_RE.search(text)
        assert status_match, f"Missing '- **Status**:' header in {path.name}"
        header_status = status_match.group(1).strip()
        assert header_status == idx_status, (
            f"Status mismatch for {adr_id}: header='{header_status}' vs index.md='{idx_status}'"
        )

        if idx_status == "ACCEPTED":
            assert "pending" not in idx_commits.lower(), (
                f"ACCEPTED row {adr_id} in index.md still says Pending: {idx_commits}"
            )
            idx_hash_links = HASH_LINK_RE.findall(idx_commits)
            assert idx_hash_links, f"ACCEPTED row {adr_id} in index.md lacks hyperlinked commit hashes"

            if adr_num > 7:
                commits_match = HEADER_COMMITS_RE.search(text)
                assert commits_match, f"Missing '- **Linked Commits**:' header in {path.name}"
                hdr_commits = commits_match.group(1).strip()
                assert "pending" not in hdr_commits.lower(), (
                    f"ACCEPTED ADR {adr_id} header still says Pending: {hdr_commits}"
                )
                assert HASH_LINK_RE.findall(hdr_commits), (
                    f"ACCEPTED ADR {adr_id} header lacks hyperlinked commit hashes: {hdr_commits}"
                )

            if has_git:
                for commit_hash, _ in idx_hash_links:
                    proc = subprocess.run(
                        ["git", "rev-parse", "--verify", f"{commit_hash}^{{commit}}"],
                        cwd=REPO_ROOT,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    assert proc.returncode == 0, (
                        f"Invalid commit hash '{commit_hash}' referenced for {adr_id} in index.md"
                    )
