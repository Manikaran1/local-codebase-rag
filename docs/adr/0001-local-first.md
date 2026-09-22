# ADR-0001: Local-first execution

**Status:** Accepted

## Context

The target use case includes private and proprietary source repositories. Sending code to external model or embedding services by default would undermine the project's core value proposition.

## Decision

All v0.1 indexing, embedding, retrieval, context construction, and generation paths must support fully local execution.

Remote providers are not part of v0.1.

## Consequences

Positive:
- stronger privacy boundary
- offline capability
- reproducible local behavior
- no required API cost

Negative:
- users provide compute resources
- local model quality varies
- installation complexity is higher than cloud-only systems
