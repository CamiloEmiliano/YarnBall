---
trigger: always_on
description: Paced, phase-gated execution rules preserving human-in-the-loop comprehension and directive control.
---

## Human-in-the-Loop Pacing & Phase-Gated Execution Rules

### 1. Core Principle: Human Cognitive Alignment
- The primary constraint of this collaboration is the user's cognitive bandwidth, mental model alignment, and directive control.
- The workflow pacing must prioritize and preserve the user's ability to thoroughly inspect, understand, question, and steer every structural change before the codebase advances.
- The agent must never outpace the user's review cadence.

### 2. Strict Single-Phase Execution
- When executing a multi-phase or multi-stage plan, the agent MUST execute **strictly one phase at a time**.
- The agent is strictly prohibited from touching, modifying, or generating code for Phase N+1 until Phase N has been completely reviewed, discussed, and explicitly approved by the user.
- Never batch multiple phases into a single turn or monolithic implementation sweep, regardless of how straightforward or obvious the subsequent steps appear.

### 3. Phase Completion Checkpoint & Pause Protocol
- Upon completing the concrete deliverables for Phase N, the agent must:
  1. Execute and report the specific unit tests or verification commands for Phase N.
  2. Provide a concise summary of the code diffs, architectural trade-offs, and design decisions made in Phase N.
  3. **Stop completely and yield control back to the user.** Do not begin Phase N+1 in the same response or proactively start next steps without explicit confirmation (e.g., "proceed to Phase N+1").

### 4. Interactive Co-Design on Ambiguity
- If an unforeseen edge case, design trade-off, or parameter ambiguity arises within a phase, the agent must present the dilemma to the user for discussion rather than making unilateral assumptions.

