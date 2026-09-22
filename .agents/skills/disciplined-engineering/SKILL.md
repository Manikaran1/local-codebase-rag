---
name: disciplined-engineering
description: Use for any repository implementation, bug fix, refactor, test change, CI fix, or Git commit. Enforces narrow scope, architecture compliance, clean production-quality code, meaningful tests, truthful verification, and a small number of logical conventional commits. Do not use it for read-only questions that make no repository changes.
---

# Disciplined Engineering

Use this workflow whenever you modify this repository or prepare commits.

The goal is not to maximize the amount of code written. The goal is to produce the smallest complete change that satisfies the requested work, preserves the architecture, passes the quality gates, and leaves a useful Git history.

## 1. Establish the task boundary before editing

Before changing files:

1. Read the current user request.
2. Read the relevant GitHub issue and acceptance criteria when one exists.
3. Read the repository instructions in `AGENTS.md`.
4. Read the relevant sections of:
   - `docs/architecture.md`
   - applicable ADRs in `docs/adr/`
   - `docs/development-plan.md`
   - `PROJECT_STATUS.md`
5. Inspect the existing implementation and tests for the modules you will touch.
6. Inspect `git status` before editing.

State internally what is in scope and what is explicitly out of scope.

If the requested work conflicts with an accepted architecture decision, stop and surface the conflict. Do not quietly redesign the system while implementing a feature.

## 2. Protect existing work

Before editing:

- Never discard uncommitted user changes.
- Never reset, checkout over, or delete work you did not create unless explicitly instructed.
- Do not rewrite existing Git history.
- Do not use force push, destructive reset, rebase of shared history, or commit amendment unless explicitly requested.
- If the working tree is unexpectedly dirty, identify whether the changes belong to the current task before proceeding.

Prefer a dedicated feature/fix branch for substantial implementation work when the environment and task allow it. If the task is already being performed on a provided branch, continue there rather than creating unnecessary branch churn.

## 3. Plan a minimal coherent implementation

Before writing code, identify:

- files/modules that actually need to change;
- required tests;
- acceptance criteria being satisfied;
- architecture boundaries that must remain intact;
- migration or compatibility concerns, if any.

Do not broaden the task merely to make the implementation more interesting.

Avoid unrelated formatting, renaming, dependency upgrades, or cleanup. A focused diff is easier to review and safer to merge.

## 4. Write production-quality code

Code must be understandable after the current session ends.

Required standards:

- prefer clear names over clever abstractions;
- keep functions/classes focused;
- reuse existing domain contracts instead of inventing parallel structures;
- preserve typing and module boundaries;
- handle expected error cases explicitly;
- add comments only when they explain a non-obvious reason or constraint;
- avoid dead code and commented-out code;
- avoid placeholder implementations presented as complete behavior;
- do not leave silent `pass`, fake success paths, or hard-coded test-only behavior in production code;
- do not weaken validation just to make a test pass;
- do not disable linters, type checks, tests, or CI checks to hide a failure;
- do not introduce a dependency when a small existing-library solution is sufficient without first considering the maintenance cost.

When a TODO is genuinely necessary, it must describe specific deferred work and should normally correspond to an explicitly deferred requirement rather than an unfinished part of the current acceptance criteria.

## 5. Test behavior, not just lines

For implementation changes:

- add or update tests that exercise the new behavior;
- include failure/edge cases that matter to the acceptance criteria;
- prefer deterministic tests;
- use fake/deterministic model or embedding providers where appropriate;
- do not make normal unit tests depend on remote services;
- do not make tests pass by lowering useful assertions.

For bug fixes, add a regression test when practical.

Run targeted tests while developing, then run the repository quality gates before committing.

## 6. Required verification before a commit

Unless the repository instructions change, run:

```bash
ruff check .
mypy src
pytest
```

Also run relevant build/package/integration checks when the task changes those areas.

Before committing, inspect:

```bash
git diff --check
git status --short
git diff --stat
git diff
```

Verify that:

- no secret, token, credential, `.env`, private key, local database, generated index, model weight, cache, or machine-specific path is being committed;
- no generated artifact is included unless it is intentionally versioned;
- no unrelated file changed accidentally;
- the diff actually satisfies the acceptance criteria;
- documentation is updated when behavior or architecture changed.

Use `references/commit-checklist.md` as the final pre-commit checklist.

If a required command cannot be run because of the environment, state that clearly. Do not report it as passed.

## 7. Commit by logical unit, not by activity

Commits should represent meaningful engineering units, not a diary of keystrokes.

Default expectation for one scoped issue:

- one commit when the implementation is naturally atomic;
- two or three commits only when there are genuinely independent reviewable units;
- more than three commits requires a clear reason.

Do not create commits such as:

- `update files`
- `fix stuff`
- `changes`
- `wip`
- `try again`
- `final fix`
- `misc cleanup`

Do not create a new commit merely because one test failed and you corrected a typo before the work was handed off. Produce a clean logical history for the completed task.

At the same time, do not squash unrelated conceptual changes into one giant commit merely to hit a commit-count target.

## 8. Use meaningful conventional commit messages

Use this shape:

```text
<type>(<scope>): <imperative summary>
```

Common types:

- `feat` — new user/developer-facing behavior
- `fix` — bug correction
- `refactor` — internal restructuring without intended behavior change
- `test` — test-only change
- `docs` — documentation-only change
- `perf` — performance improvement
- `build` — packaging/build-system change
- `ci` — CI workflow change
- `chore` — maintenance that fits none of the above

Good examples:

```text
feat(ingestion): add repository file discovery
feat(parsing): extract Python function symbols
fix(retrieval): preserve lexical-only candidates during fusion
test(chunking): cover oversized symbol splitting
docs(adr): record vector backend decision
ci: verify Python 3.11 and 3.12
```

The subject should explain what changed, not that work occurred.

Keep the summary concise. Add a commit body only when the reason, tradeoff, migration impact, or non-obvious design decision needs explanation.

## 9. Each commit should be reviewable

Before finalizing a commit:

- inspect the staged diff;
- ensure files in the commit belong to the same logical change;
- avoid mixing unrelated refactors with new behavior;
- keep tests with the behavior they verify when that improves reviewability;
- prefer commits that build and pass relevant tests on their own where practical.

Never commit knowingly broken code just to save progress unless the user explicitly asks for a WIP checkpoint. If a WIP checkpoint is explicitly requested, label it clearly and do not represent it as merge-ready.

## 10. Treat CI failures as engineering feedback

If CI fails:

1. inspect the failing job and exact logs;
2. determine the root cause;
3. fix the source/configuration/test problem;
4. rerun local checks where possible;
5. push the correction;
6. verify CI again.

Do not solve CI failures by:

- deleting useful tests;
- suppressing type errors without justification;
- globally ignoring lint rules to hide local problems;
- marking required jobs as allowed to fail;
- weakening assertions;
- removing supported Python versions merely because one exposes a real bug.

Changes to a quality gate require an explicit engineering reason.

## 11. Do not hide incomplete work

If acceptance criteria are not fully satisfied, say so.

Do not use language such as "complete" or "done" when:

- required tests are failing;
- a required path is still a stub;
- a quality check was skipped without disclosure;
- known behavior contradicts the issue criteria;
- implementation depends on an unverified assumption.

A truthful partial result is better than a misleading success report.

## 12. Finish with a clean handoff

At the end of an implementation task, verify:

```bash
git status --short
```

The working tree should be clean unless the user explicitly asked to leave uncommitted work.

Provide an implementation report containing:

1. **Scope completed** — what requirement/issue was implemented.
2. **Key changes** — main files/modules and behavior changed.
3. **Acceptance criteria** — concise mapping of each criterion to implementation/tests.
4. **Verification** — exact commands run and their results.
5. **Commits** — commit SHA(s) and message(s).
6. **Architecture notes** — ADRs or important design choices, if any.
7. **Deferred work** — work intentionally left for later phases.
8. **Known limitations** — only real current limitations.
9. **Working tree** — confirm whether it is clean.

Do not automatically begin the next issue after this report.
