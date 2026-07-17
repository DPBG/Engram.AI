# Region-Benchmark ↔ Observability Alignment (Design Note)

> **Status: BLOCKED — placeholder, not a design decision.**
> Tracking issue: [#335](https://github.com/DPBG/Engram.AI/issues/335) (M3.50).
> This file exists so the *questions* this cross-reference needs to answer
> are written down now, while the motivation is fresh — not to pre-empt the
> answers before the two things being compared actually exist.

## Why this is blocked

Issue #335 asks: if M3's per-region learning benchmarks land, could M8/M9's
coordinator/planner observability dashboards consume the same region-level
breakdown instead of building a separate one?

Answering that requires both sides to exist in at least draft form:

1. **M3's per-region benchmark proposals — items #30–36.** Tracked as
   issues #315–#321. As of this note, **all seven are open with no PR**:
   - [#315](https://github.com/DPBG/Engram.AI/issues/315) (M3.30) — PatternSeparator coverage
   - [#316](https://github.com/DPBG/Engram.AI/issues/316) (M3.31) — WorkingMemory capacity/decay
   - [#317](https://github.com/DPBG/Engram.AI/issues/317) (M3.32) — Cerebellum motor-learning
   - [#318](https://github.com/DPBG/Engram.AI/issues/318) (M3.33) — GlobalWorkspace cross-region broadcast
   - [#319](https://github.com/DPBG/Engram.AI/issues/319) (M3.34) — MetaControllerRegion executive/gating
   - [#320](https://github.com/DPBG/Engram.AI/issues/320) (M3.35) — FeatureLayer→ConceptLayer hierarchy
   - [#321](https://github.com/DPBG/Engram.AI/issues/321) (M3.36) — PredictiveLayer prediction-error signal
2. **An M8 or M9 coordinator/planner observability issue.** None currently
   exists in the tracker to compare a design against.

Writing this note's conclusions today would mean speculating about the shape
of benchmark output that hasn't been built and observability requirements
that haven't been scoped — the exact premature-timing failure mode issue
#335 itself warns against ("not so early that there's nothing concrete yet
to compare against").

## What this note needs to answer once unblocked

Not answered here — recorded so whoever revisits this has a starting list
instead of a blank page:

1. **What shape does the per-region breakdown actually take** once #315–#321
   land? Likely candidates per region: a learning-progress metric (e.g.
   separability, capacity/decay, prediction-error), a sample count, and a
   timestamp/step — but the real answer is whatever those seven benchmarks
   converge on in practice, which may not be uniform across regions.
2. **What does the M8/M9 observability item actually need** from a
   region-level breakdown? Point-in-time snapshot for a CI gate (M3's likely
   need) and a live, streaming per-region view for an operator dashboard
   (M8/M9's likely need) may turn out to be different enough consumption
   patterns that a shared *data shape* still needs two different *delivery*
   mechanisms.
3. **Can one component serve both**, or do the two consumers' needs diverge
   enough that reconciliation costs more than building them separately? This
   is the actual question issue #335 asks to confirm — answer it only once
   both sides are concrete enough to compare line-by-line.
4. **Where would a shared component live** (e.g. `sdk/`, a new
   `neuromorphic/benchmarks.py` export, or a small standalone module) and
   who owns keeping it in sync as both consumers evolve?

## Unblock condition

Revisit this note for real once:

- At least a meaningful subset of #315–#321 has landed (merged, not just
  opened) with a concrete per-region output shape to look at, and
- An M8 or M9 issue for the coordinator/planner observability work exists.

At that point, replace this file's content with the actual design (or a
decision to keep the two implementations separate, with the reason why),
per issue #335's acceptance criteria: share the design with whoever picks up
the M8/M9 item, and confirm — one way or the other — whether a single
region-breakdown component can serve both.

## References

- Issue [#335](https://github.com/DPBG/Engram.AI/issues/335) (M3.50) — this note's tracking issue.
- Issues [#315](https://github.com/DPBG/Engram.AI/issues/315)–[#321](https://github.com/DPBG/Engram.AI/issues/321) (M3.30–M3.36) — the per-region benchmark proposals items #30–36 refer to.
