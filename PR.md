## Summary

Implements **scale-adaptive parallel synaptic routing** for `_route_parallel`, as specified in [ADR 0002](docs/adr/0002-neuromorphic-hotpath-profiling.md) (M4 performance follow-up #1).

Profiling showed the default 8-thread `ThreadPoolExecutor` makes the most common dev-scale network (~55K neurons, `python run.py` default) **~37% slower** than serial execution — thread-pool dispatch overhead (~14% of step time) nearly matches the SpMV work it parallelizes. Parallel routing only pays off above ~100K–220K neurons.

This change gates parallel `_route_parallel` on network size while keeping the thread pool available for STDP, eligibility, and region stepping:

- **`NEURO_PARALLEL_ROUTE=auto`** (default): serial routing below `NEURO_PARALLEL_ROUTE_MIN_NEURONS` (default `100000`)
- **`NEURO_PARALLEL_ROUTE=always|never`**: force parallel or serial routing regardless of size
- Existing **`NEURO_STDP_THREADS=1`** still disables the executor entirely

No numerics change — config/dispatch only, safe per Invariant 1.

## Expected impact

| Scale | Before (always parallel) | After (auto) |
|---|---|---|
| ~55K neurons (default dev) | ~15.6 ms/step | ~9.8 ms/step (~37% faster) |
| ≥100K neurons | parallel | parallel (unchanged) |

Re-validate with: `cd neuromorphic && uv run python scripts/profile_hotpath.py --compare-threading`

## Test plan

- [x] `uv run python -m pytest tests/test_parallel_routing.py -v`
- [x] `uv run python -m pytest tests/test_network.py -v`
- [ ] `uv run python scripts/profile_hotpath.py --compare-threading` — confirm dev-scale improvement with default env
- [ ] `python run.py` smoke test — verify log line `Parallel synaptic routing disabled` at default scale

## Related

- ADR: [docs/adr/0002-neuromorphic-hotpath-profiling.md](docs/adr/0002-neuromorphic-hotpath-profiling.md)
- Milestone: M4 — Real-Time Performance
- Does **not** duplicate #132 (CI benchmark gate, merged) or #133 (profiling ADR, merged) — this is the implementation follow-up those artifacts called for
