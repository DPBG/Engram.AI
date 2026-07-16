#!/usr/bin/env python3
"""Sustained-publish load test for EventBus / NATS (M2.2).

Measures throughput and latency under continuous publish pressure so scale
planning has a known ceiling instead of guesswork.

Usage:
    cd sdk && python scripts/bench_eventbus_publish.py
    cd sdk && python scripts/bench_eventbus_publish.py --duration 30 --output benchmarks/
    NATS_URL=nats://localhost:4222 python scripts/bench_eventbus_publish.py

Scenarios (selected with --scenario / default: all):
    core            — fire-and-forget core NATS (unknown subject)
    core_sub        — core NATS with a fast in-process subscriber
    jetstream       — JetStream ack path via decision.<id> (safety-critical)
    raw_nats        — pre-serialized bytes via nats-py (isolates SDK overhead)

Requires a JetStream-enabled nats-server. Honors NATS_URL; otherwise starts an
embedded server (PATH or .localrun/nats, same as the launcher).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

# Editable / PYTHONPATH install
_SDK_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SDK_SRC) not in sys.path:
    sys.path.insert(0, str(_SDK_SRC))

from activelearning.nats_client import EventBus, serialize_message  # noqa: E402
from activelearning.subjects import decision_subject  # noqa: E402

# Quiet connection JSON noise during max-rate loops
logging.getLogger("activelearning.nats_client").setLevel(logging.WARNING)
logging.getLogger("activelearning.connection_logging").setLevel(logging.WARNING)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCALRUN_NATS = _REPO_ROOT / ".localrun" / "nats"


# ---------------------------------------------------------------------------
# Broker bootstrap
# ---------------------------------------------------------------------------


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def _free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _find_nats_binary() -> Path | None:
    which = shutil.which("nats-server")
    if which:
        return Path(which)
    exe = "nats-server.exe" if platform.system().lower() == "windows" else "nats-server"
    candidate = _LOCALRUN_NATS / exe
    if candidate.is_file():
        return candidate
    return None


@dataclass
class BrokerHandle:
    url: str
    monitor_url: str | None
    proc: subprocess.Popen[bytes] | None = None
    data_dir: Path | None = None

    def stop(self) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        if self.data_dir is not None:
            shutil.rmtree(self.data_dir, ignore_errors=True)


def start_broker(nats_url: str | None) -> BrokerHandle:
    """Reuse NATS_URL / --nats-url, or spawn an embedded JetStream server."""
    if nats_url:
        parsed = urlparse(nats_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 4222
        if not _port_open(host, port):
            raise RuntimeError(f"NATS broker not reachable at {nats_url}")
        # Best-effort: common local layout puts monitor on 8222.
        monitor = f"http://{host}:8222" if _port_open(host, 8222) else None
        return BrokerHandle(url=nats_url, monitor_url=monitor)

    binary = _find_nats_binary()
    if binary is None:
        raise RuntimeError(
            "nats-server not found. Install it, run `python run.py` once to "
            "download into .localrun/nats, or pass --nats-url / set NATS_URL."
        )

    host = "127.0.0.1"
    port = _free_port(host)
    monitor_port = _free_port(host)
    url = f"nats://{host}:{port}"
    data_dir = Path(tempfile.mkdtemp(prefix="engram-eventbus-bench-"))
    proc = subprocess.Popen(
        [
            str(binary),
            "-js",
            "-p",
            str(port),
            "-m",
            str(monitor_port),
            "-sd",
            str(data_dir),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(50):
        if _port_open(host, port):
            break
        if proc.poll() is not None:
            shutil.rmtree(data_dir, ignore_errors=True)
            raise RuntimeError("Failed to start embedded nats-server")
        time.sleep(0.1)
    else:
        proc.kill()
        shutil.rmtree(data_dir, ignore_errors=True)
        raise RuntimeError("Timed out waiting for embedded nats-server")

    return BrokerHandle(
        url=url,
        monitor_url=f"http://{host}:{monitor_port}",
        proc=proc,
        data_dir=data_dir,
    )


def fetch_varz(monitor_url: str | None) -> dict[str, Any] | None:
    if not monitor_url:
        return None
    try:
        with urlopen(f"{monitor_url.rstrip('/')}/varz", timeout=2) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (URLError, OSError, json.JSONDecodeError, TimeoutError):
        return None


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def summarize_latencies_ms(samples_ms: list[float]) -> dict[str, float | int]:
    if not samples_ms:
        return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0, "mean_ms": 0.0}
    ordered = sorted(samples_ms)
    return {
        "count": len(ordered),
        "p50_ms": round(_percentile(ordered, 50), 3),
        "p95_ms": round(_percentile(ordered, 95), 3),
        "p99_ms": round(_percentile(ordered, 99), 3),
        "max_ms": round(ordered[-1], 3),
        "mean_ms": round(statistics.fmean(ordered), 3),
    }


def make_payload(size_bytes: int, seq: int) -> dict[str, Any]:
    """Build a dict payload whose serialized size is approximately size_bytes."""
    base = {"seq": seq, "ts": time.time(), "bench": "eventbus_publish"}
    overhead = len(serialize_message(base))
    pad = max(0, size_bytes - overhead - 16)
    base["pad"] = "x" * pad
    return base


def make_decision_payload(size_bytes: int, seq: int) -> dict[str, Any]:
    """KernelDecisionMessage-shaped payload for JetStream decision.* subjects."""
    base = {
        "trace_id": f"bench-{seq}",
        "type": "ALLOW",
        "reason": "eventbus_loadtest",
        "risk_score": 0.0,
    }
    overhead = len(serialize_message(base))
    pad = max(0, size_bytes - overhead - 16)
    base["pad"] = "x" * pad
    return base


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    name: str
    subject: str
    duration_s: float
    published: int
    errors: int
    bytes_total: int
    msgs_per_sec: float
    bytes_per_sec: float
    publish_latency: dict[str, float | int]
    received: int | None = None
    delivery_ratio: float | None = None
    eventbus_metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "subject": self.subject,
            "duration_s": round(self.duration_s, 3),
            "published": self.published,
            "errors": self.errors,
            "bytes_total": self.bytes_total,
            "msgs_per_sec": round(self.msgs_per_sec, 1),
            "bytes_per_sec": round(self.bytes_per_sec, 1),
            "publish_latency": self.publish_latency,
            "received": self.received,
            "delivery_ratio": (
                round(self.delivery_ratio, 4) if self.delivery_ratio is not None else None
            ),
            "eventbus_metrics": self.eventbus_metrics,
            "notes": self.notes,
        }


async def _run_publish_loop(
    publish_one,
    *,
    duration_s: float,
    payload_size: int,
    sample_every: int,
    payload_factory=make_payload,
    pre_serialize: bool = False,
) -> tuple[int, int, int, list[float], float]:
    """Drive sustained publish until duration elapses. Returns counts + latency samples.

    When ``pre_serialize`` is True (raw nats-py path), payloads are serialized
    once in the loop and the bytes are handed to ``publish_one``. EventBus
    scenarios leave ``pre_serialize`` False so serialization happens only inside
    ``EventBus.publish`` (production path).
    """
    published = 0
    errors = 0
    bytes_total = 0
    samples: list[float] = []
    t_end = time.perf_counter() + duration_s
    t0 = time.perf_counter()

    while time.perf_counter() < t_end:
        payload = payload_factory(payload_size, published)
        raw = serialize_message(payload) if pre_serialize else None
        try:
            t_pub = time.perf_counter()
            await publish_one(payload, raw)
            elapsed_ms = (time.perf_counter() - t_pub) * 1000.0
            if published % sample_every == 0:
                samples.append(elapsed_ms)
            published += 1
            bytes_total += len(raw) if raw is not None else payload_size
        except Exception as exc:
            errors += 1
            if errors <= 3:
                print(f"  publish error: {type(exc).__name__}: {exc}", flush=True)

    wall = time.perf_counter() - t0
    return published, errors, bytes_total, samples, wall


async def scenario_core(
    nats_url: str,
    *,
    duration_s: float,
    payload_size: int,
    sample_every: int,
) -> ScenarioResult:
    subject = f"loadtest.publish.{uuid.uuid4().hex[:8]}"
    bus = EventBus(nats_url=nats_url, name=f"bench-core-{uuid.uuid4().hex[:6]}")
    await bus.connect()

    async def publish_one(payload: dict[str, Any], _raw: bytes) -> None:
        await bus.publish(subject, payload)

    try:
        published, errors, bytes_total, samples, wall = await _run_publish_loop(
            publish_one,
            duration_s=duration_s,
            payload_size=payload_size,
            sample_every=sample_every,
        )
        metrics = bus.get_metrics()
    finally:
        await bus.close()

    return ScenarioResult(
        name="core",
        subject=subject,
        duration_s=wall,
        published=published,
        errors=errors,
        bytes_total=bytes_total,
        msgs_per_sec=published / wall if wall > 0 else 0.0,
        bytes_per_sec=bytes_total / wall if wall > 0 else 0.0,
        publish_latency=summarize_latencies_ms(samples),
        eventbus_metrics=metrics,
        notes="Fire-and-forget core NATS via EventBus.publish (JSON serialize + nc.publish)",
    )


async def scenario_core_sub(
    nats_url: str,
    *,
    duration_s: float,
    payload_size: int,
    sample_every: int,
) -> ScenarioResult:
    subject = f"loadtest.publish.{uuid.uuid4().hex[:8]}"
    pub = EventBus(nats_url=nats_url, name=f"bench-pub-{uuid.uuid4().hex[:6]}")
    sub = EventBus(nats_url=nats_url, name=f"bench-sub-{uuid.uuid4().hex[:6]}")
    await pub.connect()
    await sub.connect()

    received = 0

    async def handler(_data: dict[str, Any]) -> None:
        nonlocal received
        received += 1

    await sub.subscribe(subject, handler)

    async def publish_one(payload: dict[str, Any], _raw: bytes) -> None:
        await pub.publish(subject, payload)

    try:
        published, errors, bytes_total, samples, wall = await _run_publish_loop(
            publish_one,
            duration_s=duration_s,
            payload_size=payload_size,
            sample_every=sample_every,
        )
        # Allow in-flight deliveries to drain
        await asyncio.sleep(0.5)
        metrics = pub.get_metrics()
    finally:
        await pub.close()
        await sub.close()

    ratio = (received / published) if published else None
    return ScenarioResult(
        name="core_sub",
        subject=subject,
        duration_s=wall,
        published=published,
        errors=errors,
        bytes_total=bytes_total,
        msgs_per_sec=published / wall if wall > 0 else 0.0,
        bytes_per_sec=bytes_total / wall if wall > 0 else 0.0,
        publish_latency=summarize_latencies_ms(samples),
        received=received,
        delivery_ratio=ratio,
        eventbus_metrics=metrics,
        notes="Core NATS with a fast subscriber; delivery_ratio < 1 hints at slow-consumer pressure",
    )


async def scenario_jetstream(
    nats_url: str,
    *,
    duration_s: float,
    payload_size: int,
    sample_every: int,
) -> ScenarioResult:
    # decision.<id> is safety-critical → JetStream publish + server ack
    bus = EventBus(nats_url=nats_url, name=f"bench-js-{uuid.uuid4().hex[:6]}")
    await bus.connect()

    async def publish_one(payload: dict[str, Any], _raw: bytes) -> None:
        await bus.publish(decision_subject(payload["trace_id"]), payload)

    try:
        published, errors, bytes_total, samples, wall = await _run_publish_loop(
            publish_one,
            duration_s=duration_s,
            payload_size=payload_size,
            sample_every=sample_every,
            payload_factory=make_decision_payload,
        )
        metrics = bus.get_metrics()
    finally:
        await bus.close()

    return ScenarioResult(
        name="jetstream",
        subject="decision.bench-*",
        duration_s=wall,
        published=published,
        errors=errors,
        bytes_total=bytes_total,
        msgs_per_sec=published / wall if wall > 0 else 0.0,
        bytes_per_sec=bytes_total / wall if wall > 0 else 0.0,
        publish_latency=summarize_latencies_ms(samples),
        eventbus_metrics=metrics,
        notes="JetStream ack path (SAFETY_CRITICAL stream); expects lower ceiling than core",
    )


async def scenario_raw_nats(
    nats_url: str,
    *,
    duration_s: float,
    payload_size: int,
    sample_every: int,
) -> ScenarioResult:
    import nats

    subject = f"loadtest.raw.{uuid.uuid4().hex[:8]}"
    nc = await nats.connect(nats_url, name=f"bench-raw-{uuid.uuid4().hex[:6]}")

    async def publish_one(_payload: dict[str, Any], raw: bytes) -> None:
        await nc.publish(subject, raw)

    try:
        published, errors, bytes_total, samples, wall = await _run_publish_loop(
            publish_one,
            duration_s=duration_s,
            payload_size=payload_size,
            sample_every=sample_every,
            pre_serialize=True,
        )
    finally:
        await nc.drain()
        await nc.close()

    return ScenarioResult(
        name="raw_nats",
        subject=subject,
        duration_s=wall,
        published=published,
        errors=errors,
        bytes_total=bytes_total,
        msgs_per_sec=published / wall if wall > 0 else 0.0,
        bytes_per_sec=bytes_total / wall if wall > 0 else 0.0,
        publish_latency=summarize_latencies_ms(samples),
        notes="Pre-serialized bytes via nats-py only (isolates SDK EventBus overhead)",
    )


SCENARIOS = {
    "core": scenario_core,
    "core_sub": scenario_core_sub,
    "jetstream": scenario_jetstream,
    "raw_nats": scenario_raw_nats,
}


# ---------------------------------------------------------------------------
# Bottleneck analysis
# ---------------------------------------------------------------------------


def infer_bottleneck(results: list[ScenarioResult]) -> dict[str, Any]:
    by_name = {r.name: r for r in results}
    core = by_name.get("core")
    raw = by_name.get("raw_nats")
    js = by_name.get("jetstream")
    core_sub = by_name.get("core_sub")

    lines: list[str] = []
    primary = "unknown"
    core_bottleneck = "unknown"

    if core and raw and raw.msgs_per_sec > 0:
        sdk_overhead_pct = max(0.0, (1.0 - core.msgs_per_sec / raw.msgs_per_sec) * 100.0)
        lines.append(
            f"EventBus core is {sdk_overhead_pct:.0f}% slower than raw nats-py "
            f"({core.msgs_per_sec:.0f} vs {raw.msgs_per_sec:.0f} msg/s) -- "
            "JSON serialize + EventBus plumbing."
        )
        core_bottleneck = (
            "client_serialization_and_eventbus"
            if sdk_overhead_pct >= 25
            else "nats_py_or_broker"
        )
        primary = core_bottleneck

    if core and js and core.msgs_per_sec > 0:
        js_ratio = js.msgs_per_sec / core.msgs_per_sec
        lines.append(
            f"JetStream sustained rate is {js_ratio:.2f}x core "
            f"({js.msgs_per_sec:.0f} vs {core.msgs_per_sec:.0f} msg/s) -- "
            "ack round-trip dominates safety-critical publish."
        )
        # Governance traffic is JetStream; when it is an order of magnitude
        # below core, that is the binding architectural ceiling.
        if js_ratio < 0.2:
            lines.append(
                "Safety-critical (JetStream) ceiling is the binding constraint "
                "for proposal/decision fan-out, not core observation publish."
            )
            primary = "jetstream_pub_ack"

    if core_sub and core_sub.delivery_ratio is not None and core_sub.delivery_ratio < 0.95:
        lines.append(
            f"Subscriber delivery_ratio={core_sub.delivery_ratio:.3f} under max-rate publish -- "
            "slow-consumer / pending buffer pressure on the receive path."
        )
        if primary == "unknown":
            primary = "subscriber_drain"

    if core and float(core.publish_latency.get("p95_ms", 0) or 0) > 5:
        lines.append(
            f"Core publish p95={core.publish_latency['p95_ms']} ms -- "
            "awaiting broker flush is material at this rate."
        )

    if not lines:
        lines.append("Insufficient scenario coverage to classify bottleneck.")

    ceiling = max((r.msgs_per_sec for r in results), default=0.0)
    ceiling_name = max(results, key=lambda r: r.msgs_per_sec).name if results else None

    return {
        "primary_bottleneck": primary,
        "core_path_bottleneck": core_bottleneck,
        "sustained_ceiling_msgs_per_sec": round(ceiling, 1),
        "ceiling_scenario": ceiling_name,
        "eventbus_core_msgs_per_sec": round(core.msgs_per_sec, 1) if core else None,
        "jetstream_msgs_per_sec": round(js.msgs_per_sec, 1) if js else None,
        "analysis": lines,
    }


def get_system_info(broker: BrokerHandle) -> dict[str, Any]:
    import multiprocessing

    info: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": multiprocessing.cpu_count(),
        "python_version": platform.python_version(),
        "nats_url": broker.url,
    }
    try:
        import importlib.metadata

        info["nats_py_version"] = importlib.metadata.version("nats-py")
    except Exception:
        info["nats_py_version"] = "unknown"

    varz = fetch_varz(broker.monitor_url)
    if varz:
        info["nats_server_version"] = varz.get("version")
        info["nats_server_cores"] = varz.get("cores")
        info["nats_max_connections"] = varz.get("max_connections")
    return info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sustained-publish EventBus / NATS load test (M2.2)",
    )
    parser.add_argument(
        "--nats-url",
        default=None,
        help="Broker URL (default: NATS_URL env, else embedded nats-server)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="Seconds of sustained publish per scenario (default: 20)",
    )
    parser.add_argument(
        "--payload-bytes",
        type=int,
        default=256,
        help="Approximate JSON payload size in bytes (default: 256)",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=10,
        help="Record publish latency every N messages (default: 10)",
    )
    parser.add_argument(
        "--scenario",
        nargs="+",
        default=["all"],
        help=f"Scenarios to run: all or subset of {sorted(SCENARIOS)}",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "benchmarks"),
        help="Directory for results JSON (default: sdk/benchmarks/)",
    )
    args = parser.parse_args()

    # Capture system info while broker is live
    async def _main() -> dict[str, Any]:
        broker = start_broker(args.nats_url or os.environ.get("NATS_URL"))
        selected = args.scenario if args.scenario != ["all"] else list(SCENARIOS.keys())
        for name in selected:
            if name not in SCENARIOS:
                raise SystemExit(f"Unknown scenario {name!r}; choose from {sorted(SCENARIOS)}")

        print("=" * 60)
        print("EventBus sustained-publish load test (M2.2)")
        print(f"  broker:   {broker.url}")
        print(f"  duration: {args.duration}s per scenario")
        print(f"  payload:  ~{args.payload_bytes} bytes")
        print(f"  scenarios:{', '.join(selected)}")
        print("=" * 60)

        system = get_system_info(broker)
        results: list[ScenarioResult] = []
        try:
            warm = EventBus(nats_url=broker.url, name="bench-warm")
            await warm.connect()
            await warm.publish(f"loadtest.warm.{uuid.uuid4().hex[:6]}", {"ok": True})
            await warm.close()

            for name in selected:
                print(f"\n--- scenario: {name} ---")
                result = await SCENARIOS[name](
                    broker.url,
                    duration_s=args.duration,
                    payload_size=args.payload_bytes,
                    sample_every=args.sample_every,
                )
                results.append(result)
                print(
                    f"  {result.msgs_per_sec:,.0f} msg/s  "
                    f"p50={result.publish_latency['p50_ms']} ms  "
                    f"p95={result.publish_latency['p95_ms']} ms  "
                    f"errors={result.errors}"
                )
                if result.received is not None:
                    print(
                        f"  received={result.received}  "
                        f"delivery_ratio={result.delivery_ratio}"
                    )
        finally:
            broker.stop()

        bottleneck = infer_bottleneck(results)
        print("\n--- bottleneck ---")
        for line in bottleneck["analysis"]:
            print(f"  {line}")
        print(
            f"  ceiling: {bottleneck['sustained_ceiling_msgs_per_sec']:,.0f} msg/s "
            f"({bottleneck['ceiling_scenario']})"
        )
        print(f"  primary: {bottleneck['primary_bottleneck']}")

        return {
            "benchmark": "eventbus_sustained_publish",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "args": {
                "duration_s": args.duration,
                "payload_bytes": args.payload_bytes,
                "sample_every": args.sample_every,
                "scenarios": selected,
            },
            "system": system,
            "scenarios": [r.to_dict() for r in results],
            "bottleneck": bottleneck,
        }

    payload = asyncio.run(_main())

    output_dir = Path(args.output)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Cannot create output directory {output_dir}: {e}")
        print("Results not saved to disk.")
        return

    filename = f"eventbus_publish_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output_path = output_dir / filename
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Also write a stable "latest" pointer for docs / tooling
    latest = output_dir / "eventbus_publish_latest.json"
    latest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nResults saved: {output_path}")
    print(f"Latest pointer: {latest}")
    print("=" * 60)


if __name__ == "__main__":
    main()
