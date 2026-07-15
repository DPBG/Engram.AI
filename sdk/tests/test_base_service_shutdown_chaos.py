"""Chaos test (M2.37, issue #260): a process killed mid-message-handling.

BaseService.stop()'s cooperative shutdown (_cleanup() then event_bus.close(),
which drains) is untested against the harsher case this issue actually names:
the process itself dying (crash, OOM kill, SIGKILL) while a js_subscribe
handler is mid-flight, not a clean signal-driven stop(). A cooperative
drain() never runs at all in that case.

js_subscribe's documented contract (nats_client.py) is explicit ack only
after the handler returns -- so a process killed before that point should
never ack, and the broker should redeliver to whoever consumes the same
durable next. This test proves that end-to-end against a real broker and a
real subprocess (activelearning._chaos_kill_service), rather than trusting
the docstring on inspection -- the same "prove it, don't just read the
code" standard this milestone's other issues (e.g. #258) apply elsewhere.

Uses SIGSTOP-then-SIGKILL (not a timing race) to deterministically kill the
process while it is in the middle of its simulated processing sleep,
mirroring kernel/tests/test_kernel_crash_chaos.py's reasoning.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from activelearning.nats_client import EventBus

_REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    shutil.which("nats-server") is None,
    reason="nats-server not on PATH -- skipping chaos integration test",
)


def _decision_payload(trace_id: str) -> dict:
    return {"trace_id": trace_id, "type": "ALLOW", "reason": "chaos test"}


@pytest.fixture
def chaos_proc(nats_url: str, tmp_path: Path):
    """A real ChaosKillService subprocess connected to nats_url.

    Yielded still running; the test itself freezes/kills it. Cleaned up
    defensively in teardown in case a test fails first.
    """
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    subject = f"decision.{uuid.uuid4().hex}"
    durable = f"chaos-{uuid.uuid4().hex[:8]}"

    env = os.environ.copy()
    env["NATS_URL"] = nats_url
    env["CHAOS_SUBJECT"] = subject
    env["CHAOS_DURABLE"] = durable
    env["CHAOS_MARKER_DIR"] = str(marker_dir)
    env["CHAOS_SLEEP_SECONDS"] = "3.0"
    src = str(_REPO_ROOT / "sdk" / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src, existing) if p)

    proc = subprocess.Popen(
        [sys.executable, "-m", "activelearning._chaos_kill_service"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        yield proc, subject, durable, marker_dir
    finally:
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


def _wait_until(predicate, timeout: float = 10.0, interval: float = 0.1) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


async def _publish(nats_url: str, subject: str, trace_id: str) -> None:
    bus = EventBus(nats_url=nats_url, name=f"chaos-publisher-{uuid.uuid4().hex[:8]}")
    await bus.connect()
    try:
        await bus.publish(subject, _decision_payload(trace_id))
    finally:
        await bus.close()


async def _redelivered_to_fresh_consumer(
    nats_url: str,
    subject: str,
    durable: str,
    timeout: float,
) -> dict | None:
    """A fresh consumer on the SAME durable asks whether the broker still
    has (and redelivers) the message the killed process never acked."""
    bus = EventBus(nats_url=nats_url, name=f"chaos-verifier-{uuid.uuid4().hex[:8]}")
    await bus.connect()
    received: list[dict] = []

    async def _handler(data: dict) -> None:
        received.append(data)

    try:
        await bus.js_subscribe(subject, _handler, durable=durable)
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if received:
                return received[0]
            await asyncio.sleep(0.1)
        return None
    finally:
        await bus.close()


def test_killed_mid_processing_message_is_redelivered_not_lost(
    nats_url: str,
    chaos_proc,
) -> None:
    """A process killed while a handler is mid-sleep (before ack) must never
    silently lose the message -- the broker must redeliver it to whoever
    consumes next on the same durable."""
    proc, subject, durable, marker_dir = chaos_proc
    trace_id = uuid.uuid4().hex

    asyncio.run(_publish(nats_url, subject, trace_id))

    # Wait for the handler to actually start (proves we're killing it DURING
    # processing, not before delivery even happens).
    started = marker_dir / f"started-{trace_id}"
    assert _wait_until(
        lambda: started.exists(), timeout=10.0
    ), "handler never started processing the message"

    # Freeze before killing: a stopped process can never reach the
    # completed-marker write or the ack, deterministically "dies
    # mid-processing" rather than racing the sleep window.
    proc.send_signal(signal.SIGSTOP)
    proc.kill()  # SIGKILL reaches even a stopped process
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass

    completed = marker_dir / f"completed-{trace_id}"
    assert not completed.exists(), (
        "process finished its work before being killed -- this run doesn't "
        "actually prove anything about a mid-processing kill"
    )

    # Never acked (handler never returned) -- a fresh consumer on the same
    # durable must still receive it once the broker's ack_wait elapses.
    result = asyncio.run(_redelivered_to_fresh_consumer(nats_url, subject, durable, timeout=20.0))
    assert result is not None, (
        "message was never redelivered after the process was killed "
        "mid-processing -- a kill can silently lose a message"
    )
    assert result["trace_id"] == trace_id
