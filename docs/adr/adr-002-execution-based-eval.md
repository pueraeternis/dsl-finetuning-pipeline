# ADR 002: Execution-Based Evaluation over String Matching

## Status
Accepted

## Context
Text-to-Code tasks often suffer from "Syntactic Variance." A model might produce a valid query that is string-different but logically identical to the ground truth (e.g., changing the order of filters).

## Decision
We implemented **Execution-based Evaluation**. The primary success metric is "Does the generated query return the same data as the ground truth from a live database?"

## Consequences
- **Pros:** Zero false negatives due to formatting differences. True measure of "Business Value."
- **Cons:** Requires a functional transpiler and a seeded database (SQLite).