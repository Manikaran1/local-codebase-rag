# ADR-0003: Code-aware chunking

**Status:** Accepted

## Context

Fixed-size token chunking can split functions, combine unrelated symbols, and lose structural meaning.

## Decision

Source code should be parsed into structural symbols and chunked on symbol boundaries where possible. Arbitrary bounded splitting is a fallback for oversized symbols.

## Consequences

- better structural metadata and citations
- parser implementations become a core dependency
- unsupported languages require fallback behavior
