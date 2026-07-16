# EventBus sustained-publish load test (M2.2)

Measured throughput ceilings for Engram's NATS `EventBus`, and the bottleneck
that binds each publish path. Without these numbers, scale planning is guesswork.

## How to run

Requires a JetStream-enabled `nats-server` (on `PATH`, at `.localrun/nats/` after
`python run.py`, or via `NATS_URL` / `--nats-url`).

```bash
cd sdk
# Windows PowerShell
$env:PYTHONPATH = "src"
python scripts/bench_eventbus_publish.py --duration 20 --output benchmarks/

# Unix
PYTHONPATH=src python scripts/bench_eventbus_publish.py --duration 20 --output benchmarks/
```

Useful flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--duration` | `20` | Seconds of max-rate publish per scenario |
| `--payload-bytes` | `256` | Approximate JSON payload size |
| `--scenario` | `all` | `core`, `core_sub`, `jetstream`, `raw_nats` |
| `--nats-url` | `$NATS_URL` or embedded | Broker URL |
| `--output` | `sdk/benchmarks/` | JSON result directory |

Each run writes a timestamped file and updates
`sdk/benchmarks/eventbus_publish_latest.json`.

## Scenarios

| Scenario | Path | What it isolates |
|----------|------|------------------|
| `raw_nats` | Pre-serialized `nats-py` publish | Client + broker floor (no EventBus) |
| `core` | `EventBus.publish` on a non-safety subject | Production observation-like path |
| `core_sub` | Same, with a fast in-process subscriber | Shared-loop / delivery pressure |
| `jetstream` | `EventBus.publish` to `decision.*` | Safety-critical pub-ack path |

## Reference results

Captured on the machine below (see `sdk/benchmarks/eventbus_publish_latest.json`
for the full artifact). Re-run on target hardware before treating these as
capacity guarantees — absolute rates vary with CPU and OS, but **ratios**
(core vs JetStream, EventBus vs raw) have been stable across runs.

| Field | Value |
|-------|-------|
| Date | 2026-07-16 |
| OS | Windows 11, Python 3.13.2 |
| CPU | 12-core Intel (Family 6 Model 165) |
| nats-server | 2.10.22 (embedded, JetStream) |
| Duration | 20 s / scenario, ~256-byte payloads |

| Scenario | Sustained msg/s | Publish p50 | Publish p95 | Notes |
|----------|----------------:|------------:|------------:|-------|
| `raw_nats` | **68,015** | 0.003 ms | 0.004 ms | Absolute single-publisher ceiling |
| `core` | **52,732** | 0.011 ms | 0.017 ms | EventBus + JSON serialize |
| `core_sub` | **33,120** | 0.011 ms | 0.018 ms | delivery_ratio = 1.0 |
| `jetstream` | **3,684** | 0.236 ms | 0.346 ms | SAFETY_CRITICAL stream ack |

## Throughput ceiling and bottleneck

**Absolute single-publisher ceiling (this run):** ~68k msg/s (`raw_nats`),
~53k msg/s through `EventBus` core NATS.

**Binding architectural bottleneck:** JetStream pub-ack on safety-critical
subjects (`proposal.new`, `decision.*`, `policy.*`, …). Sustained rate is
~**3.7k msg/s** — about **0.07×** core EventBus — because every publish waits
for a server ack. That is the ceiling that matters for Kernel decision fan-out
and other governance traffic.

**Core-path bottleneck:** Not the broker. EventBus core is only ~22% slower
than raw `nats-py`; the remaining headroom is JSON serialization and EventBus
plumbing (`validate_payload`, metrics, `_ensure_connected`). On this hardware
the broker is not saturating under a single publisher.

**Subscriber note:** With a fast in-process subscriber, publish rate drops
(~33k msg/s) but `delivery_ratio` stayed at 1.0 — the shared asyncio loop is
the cost, not slow-consumer drop. Real services that do heavy work in handlers
will hit `pending_msgs_limit` long before the publish ceiling.

## Implications for Engram

1. **Observation flood is not a publish-side NATS problem.** Sensory gateway
   already aggregates ~1,400 obs/s down to ~4/s (`AggregatingEventBus`) because
   the **brain consumer** cannot keep up — core EventBus can sustain tens of
   thousands of msg/s from a single publisher.
2. **Plan JetStream capacity separately.** Do not assume core rates apply to
   proposals/decisions. Budget governance traffic against ~10³ msg/s per
   publisher (order of magnitude; measure on the deploy host).
3. **Horizontal scale** for core traffic is more about adding publishers /
   queue groups than hitting a broker wall at these rates. For JetStream,
   stream replicas, ack policy, and durable consumer lag (M2.1) dominate.

## What this harness does *not* measure

- Multi-publisher contention or multi-host network latency
- Durable JetStream *consumer* throughput / lag under load (see M2.1 /
  `consumer_lag.py`)
- End-to-end neuromorphic step time under observation load
- Authenticated (`NATS_CREDS`) or TLS brokers

Those belong in follow-on benches if scale planning needs them.
