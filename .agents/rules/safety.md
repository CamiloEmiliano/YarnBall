---
trigger: always_on
description: Execution safety rules for Git and remote operations.
---

## Execution & Git Safety Rules

- **NEVER run `git push`**: All pushes to remote repositories must be performed manually by the user. Only local staging and commits may be run upon explicit user request.
