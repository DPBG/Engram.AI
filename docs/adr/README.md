# Architecture Decision Records (ADRs)

This directory holds Engram's Architecture Decision Records — short documents
that capture a significant architectural decision, the context that forced it,
and the trade-offs that were accepted.

## Conventions

- Files are named `NNNN-short-slug.md`, where `NNNN` is a zero-padded,
  monotonically increasing number (`0001`, `0002`, …).
- Each ADR has a **Status** (`Proposed`, `Accepted`, `Superseded by NNNN`,
  `Deprecated`).
- ADRs are append-only: to change a decision, write a new ADR that supersedes
  the old one and update the old ADR's Status.

## Index

| ADR | Title | Status |
|---|---|---|
| [0001](0001-nats-authz.md) | NATS account & permission model | Proposed |
| [0002](0002-neuromorphic-hotpath-profiling.md) | Neuromorphic hot-path profiling & optimization strategy | Proposed |
