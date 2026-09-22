# Pre-Commit Checklist

Use this checklist immediately before creating a merge-ready commit.

## Scope

- [ ] The change belongs to the requested issue/task.
- [ ] No next-phase work was started.
- [ ] No unrelated refactor, formatting sweep, rename, or dependency upgrade was mixed in.
- [ ] Every changed file has a clear reason to be in this commit.

## Architecture

- [ ] `AGENTS.md` instructions were followed.
- [ ] Relevant architecture/ADR requirements were preserved.
- [ ] Module boundaries were not bypassed for convenience.
- [ ] Significant new design decisions are documented when needed.
- [ ] Local-first behavior remains intact unless explicitly changed by the task.

## Code quality

- [ ] No dead or commented-out production code.
- [ ] No placeholder implementation is being presented as complete.
- [ ] No test-only hard coding leaked into production behavior.
- [ ] Errors and edge cases required by the task are handled.
- [ ] Types and names are clear.
- [ ] No unnecessary dependency was introduced.

## Tests and checks

- [ ] Relevant tests were added or updated.
- [ ] Regression test exists for a bug fix when practical.
- [ ] `ruff check .` passes.
- [ ] `mypy src` passes.
- [ ] `pytest` passes.
- [ ] Additional task-specific checks pass.
- [ ] Any check that could not run is explicitly reported.

## Repository hygiene

- [ ] `git diff --check` is clean.
- [ ] `git status --short` shows only intended files before commit.
- [ ] Full diff was reviewed.
- [ ] No secrets, tokens, credentials, `.env`, private keys, local DBs, caches, indexes, model weights, or machine-specific paths are staged unintentionally.
- [ ] No generated artifact is staged unintentionally.

## Commit quality

- [ ] The commit represents one logical engineering unit.
- [ ] The commit message uses a meaningful conventional type and scope where appropriate.
- [ ] The subject explains the change in imperative form.
- [ ] The message is not vague (`update`, `fix stuff`, `changes`, `wip`, etc.).
- [ ] The commit is reviewable and preferably passes relevant checks on its own.

## Handoff

- [ ] Final commit SHA and message will be reported.
- [ ] Verification commands/results will be reported.
- [ ] Deferred work and limitations will be reported truthfully.
- [ ] Working tree will be clean when the task is declared complete.
