---
trigger: always_on
description: Response formatting and typography style rules.
---

## Formatting & Style Rules

- **LaTeX Math Syntax**: Standard LaTeX math syntax (`$...$`, `$$...$$`) is fully supported and encouraged for equations in chat responses, markdown documentation, Architecture Decision Records (ADRs), and research notes.
- **Repository-Relative Links in Documentation & Code**:
  - In all committed repository documents (including Architecture Decision Records, READMEs, scripts, and docstrings), all file, code, and commit links MUST be repository-relative (e.g. `../../<path>` from `docs/decisions/`, or `<module>/<file>` in code).
  - NEVER write absolute filesystem paths or URI schemes (such as `file:///...`, `/home/...`, `/Users/...`, `C:\...`) into repository files or commits. Absolute machine paths expose the user's local username, home directory, and environment structure.
  - In chat conversations, the agent may use clickable links per assistant communication style, but must strictly sanitize all paths before writing or editing repository files.
