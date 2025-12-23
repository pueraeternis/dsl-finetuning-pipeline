# ADR 001: Skeleton-based Semantic Infilling for Data Generation

## Status
Accepted

## Context
We needed 10,000 high-quality (NL, DSL, Schema) triplets. Standard rule-based generation (Faker) produced "robotic" and semantically nonsensical text (e.g., "Check humidity in the garage" where garage wasn't in the schema).

## Decision
We moved to a hybrid approach:
1.  Use Python to ensure **Logical Validity** (correct column types, valid DSL operators).
2.  Use a high-parameter LLM (Qwen-235B) to ensure **Semantic Richness** (realistic names, diverse phrasing).

## Consequences
- **Pros:** 100% alignment between the question and the query values. High linguistic diversity.
- **Cons:** Increased generation time (solved via vLLM batching and async orchestration).