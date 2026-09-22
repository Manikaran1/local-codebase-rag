# Repository Agent Instructions

These instructions apply to Codex and other coding agents working in this repository.

## Required engineering skill

For any task that changes source code, tests, implementation-related documentation, build/configuration files, CI, or Git history, use the repository skill:

`$disciplined-engineering`

The skill lives at:

`.agents/skills/disciplined-engineering/SKILL.md`

Use it for feature work, bug fixes, refactors, test work, implementation phases, Git commits, and CI fixes.

## Sources of truth

When implementing work, use this priority order:

1. The user's explicit request for the current task.
2. The current GitHub issue and its acceptance criteria.
3. `docs/architecture.md` and accepted ADRs in `docs/adr/`.
4. `docs/development-plan.md` and `PROJECT_STATUS.md`.
5. Existing public contracts and tests.
6. Existing implementation details.

If two sources conflict materially, stop and report the conflict instead of silently choosing a new architecture.

## Scope discipline

- Implement only the requested issue or phase.
- Do not begin the next phase automatically.
- Do not add speculative features because they might be useful later.
- Do not mix unrelated cleanup or refactors into feature work.
- If an unrelated defect is discovered, report it separately unless it blocks the requested task.

## Project invariants

- Preserve the local-first design.
- Do not introduce required cloud services or remote model APIs without explicit approval.
- Keep parsing, indexing, retrieval, generation, CLI, and API concerns within their documented boundaries.
- Keep provider-specific implementations behind interfaces where the architecture requires them.
- Citations must come from real source metadata; never fabricate them in generation logic.
- Prefer measurable retrieval behavior over opaque heuristics.
- New significant architecture decisions should be documented with an ADR rather than hidden in code.

## Completion requirements

Implementation work is not complete merely because code was written.

Before declaring a task complete:

- run the required quality checks;
- verify acceptance criteria;
- review the final diff;
- create meaningful commits when commits are part of the task;
- leave the working tree clean unless explicitly instructed otherwise;
- report tests, checks, commits, limitations, and deferred work accurately.

Never claim a check passed if it was not run successfully.
