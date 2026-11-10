---
trigger: always_on
description: Response formatting and typography style rules.
---

## Formatting & Style Rules

- **Chat Responses & SFT Datasets**: Always use plain text and Unicode characters (e.g., z-scores, delta, arrows) instead of LaTeX math syntax in conversational chat messages and within SFT dataset exports (`sft_pipeline/`) to avoid tokenization anomalies, JSON escape corruption, and chat rendering limitations.
- **Repository Documentation (`docs/`, `notes/`)**: Standard LaTeX math syntax (`$...$`, `$$...$$`) is permitted and encouraged in markdown documentation, Architecture Decision Records (ADRs), and technical notes for equations rendered in IDE preview windows and GitHub.
