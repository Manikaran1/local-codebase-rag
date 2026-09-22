# Evaluation Strategy

## Why evaluation is part of the architecture

RAG systems can produce convincing answers even when retrieval is poor. The first quality gate therefore measures evidence retrieval before generated answer quality.

## Fixture repository

`examples/sample_repo` is a deterministic small project used for unit/integration evaluation.

Initial questions should include:

| Question | Required evidence |
|---|---|
| Where is authentication handled? | `auth.py` authenticate symbol |
| Where is the database connection created? | `database.py` connect symbol |
| What controls retry count? | `config.py` retry configuration |
| What calls authenticate? | API/login fixture once added |
| Explain auth and DB interaction | evidence from both modules |

## Retrieval metrics

Candidate metrics:
- Recall@K
- Hit@K
- Mean Reciprocal Rank
- required-source coverage
- duplicate rate
- retrieval latency

Metrics should be reported separately for:
- semantic questions
- exact identifier questions
- cross-file questions
- documentation questions

## Generation checks

Generated answers should be evaluated for:
- citation validity
- citation support
- unsupported claims
- explicit uncertainty
- completeness relative to retrieved evidence

## Regression rule

A retrieval implementation change should not be merged when it materially degrades the benchmark without a documented architectural reason.
