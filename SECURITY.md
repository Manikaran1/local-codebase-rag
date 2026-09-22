# Security

## Security model

This project is intended to process source repositories that may contain sensitive intellectual property.

### Default expectations

- source content stays local
- no telemetry containing source text
- no cloud model call in the default execution path
- `.env` files and known secret/config files are excluded by default where practical
- generated logs must not dump full source chunks unless debug behavior is explicitly enabled

## Secrets in indexed repositories

A local repository may contain credentials even if they were committed accidentally.

The scanner should eventually include:
- configurable secret-file exclusions
- `.gitignore` support
- default exclusion patterns
- warnings for common credential filenames

This system is not a secret scanner and should not claim to guarantee that indexed data is safe.

## Reporting

Before public release, replace this section with a maintained vulnerability-reporting process.
