"""Chaos test (M1.3, issue #186): Kernel process dies mid-evaluation.

The Kernel is the sole ALLOW/TRANSFORM/DENY/DEFER authority (CLAUDE.md §3).
``kernel/tests/test_service.py`` already covers the in-process failure
mode — an exception raised *inside* ``_handle_action_proposal`` still
publishes a fail-safe DENY, because the try/except that catches it runs in
the same live process. That guarantee is worthless against the actual
threat this issue is about: the Kernel *process itself* dying (crash, OOM
kill, deploy restart) before it ever gets a chance to run that except
block. From a decision-waiter's perspective, a dead Kernel and a
never-started Kernel are indistinguishable — no ``decision.<trace_id>``
message is ever published, full stop.

This spins up a real ``nats-server`` (JetStream enabled) and a real
``KernelService`` subprocess, confirms it's live, freezes it with SIGSTOP
right before publishing a real ``proposal.new`` (so the durable JetStream
consumer picks the proposal up but the frozen process can never dequeue,
evaluate, or publish a decision for it — see
``test_safety_supervisor_chaos.py`` for why SIGSTOP-then-SIGKILL is used
instead of a timing race), then confirms a caller's
``EventBus.wait_for_decision()`` times out cleanly within its configured
timeout — no hang, no exception escaping the caller — exactly the
guarantee ``docs/KERNEL-CRASH-RECOVERY.md`` documents.

Follows ``test_service.py``'s / ``test_safety_supervisor_chaos.py``'s
convention of driving async code through ``asyncio.run()`` instead of
``@pytest.mark.asyncio``: the Governance CI job runs ``kernel/tests`` with
``--with pytest`` only, no pytest-asyncio.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
from activelearning.core import generate_trace_id
from activelearning.nats_client import EventBus
from activelearning.subjects import Subjects

_REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    shutil.which("nats-server") is None,
    reason="nats-server not on PATH — skipping chaos integration test",
)


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def _free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


@pytest.fixture
def nats_server(tmp_path: Path):
    """A real, isolated, JetStream-enabled nats-server on an ephemeral port."""
    binary = shutil.which("nats-server")
    host = "127.0.0.1"
    port = _free_port(host)
    proc = subprocess.Popen(
        [binary, "-js", "-p", str(port), "-m", "8222", "-sd", str(tmp_path / "nats-data")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(50):
            if _port_open(host, port):
                break
            if proc.poll() is not None:
                pytest.skip("Failed to start embedded nats-server")
            time.sleep(0.1)
        else:
            pytest.skip("Timed out waiting for embedded nats-server")
        yield f"nats://{host}:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def kernel_proc(nats_server: str, tmp_path: Path):
    """A real KernelService subprocess connected to nats_server.

    Yielded still-running; the test itself freezes/kills it to simulate the
    crash. Cleaned up defensively in teardown in case a test fails first.
    """
    env = os.environ.copy()
    env["NATS_URL"] = nats_server
    env["SQLITE_PATH"] = str(tmp_path / "kernel-chaos.db")
    # No BODY_PROFILE / BELIEFS — the Kernel runs those as best-effort
    # optional lookups (see kernel/src/kernel/service.py's _load_body_profile
    # and _check_belief_norms), not hard startup requirements.
    src = str(_REPO_ROOT / "kernel" / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src, existing) if p)

    proc = subprocess.Popen(
        [sys.executable, "-m", "kernel.service"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


async def _wait_until_ready(bus: EventBus, timeout: float = 10.0) -> None:
    """Poll kernel.status until the subprocess has connected and subscribed."""
    deadline = asyncio.get_event_loop().time() + timeout
    last_error: Exception | None = None
    while asyncio.get_event_loop().time() < deadline:
        try:
            await bus.request(Subjects.KERNEL_STATUS, {}, timeout=0.5)
            return
        except Exception as e:  # noqa: BLE001 - polling for readiness, retried
            last_error = e
            await asyncio.sleep(0.2)
    raise AssertionError(f"Kernel never became ready: {last_error}")


async def _crash_before_evaluation(nats_server: str, kernel_proc: subprocess.Popen):
    """Freeze the Kernel, publish a real proposal it can never process, and
    return (raised_timeout: bool, elapsed_seconds)."""
    bus = EventBus(nats_url=nats_server, name=f"waiter-chaos-{uuid.uuid4().hex[:8]}")
    await bus.connect()
    try:
        await _wait_until_ready(bus)

        # Freeze *before* publishing: the durable JetStream consumer still
        # picks the message up (it's in the stream), but a stopped process
        # is never scheduled by the OS, so it can never dequeue, evaluate,
        # or ack it — deterministically "dies mid-evaluation" rather than
        # racing a local NATS round trip that can complete in well under a
        # millisecond (see test_safety_supervisor_chaos.py for the same
        # reasoning against a timing-based kill).
        kernel_proc.send_signal(signal.SIGSTOP)

        trace_id = generate_trace_id()
        await bus.publish(
            Subjects.PROPOSAL_NEW,
            {
                "trace_id": trace_id,
                "action": {"type": "motor", "channel": "head", "intensity": 0.1},
                "provenance": "chaos-test",
            },
        )

        t0 = asyncio.get_event_loop().time()
        raised_timeout = False
        try:
            # A real caller would use whatever timeout suits it (defaults
            # to 30s); 5s is enough to prove "times out promptly", bounded
            # well above that by wait_for's own outer 10s so a regression
            # to an actual hang fails the test instead of riding the CI
            # job's timeout.
            await asyncio.wait_for(bus.wait_for_decision(trace_id, timeout=5.0), timeout=10.0)
        except TimeoutError:
            raised_timeout = True
        elapsed = asyncio.get_event_loop().time() - t0
        return raised_timeout, elapsed
    finally:
        kernel_proc.kill()  # SIGKILL reaches even a stopped process
        try:
            kernel_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        await bus.close()


def test_decision_waiter_fails_closed_when_kernel_dies_mid_evaluation(
    nats_server: str,
    kernel_proc: subprocess.Popen,
) -> None:
    raised_timeout, elapsed = asyncio.run(_crash_before_evaluation(nats_server, kernel_proc))

    # Not a hang: wait_for_decision's own timeout must actually fire.
    assert raised_timeout, "wait_for_decision did not time out — did something reply?"

    # Not a hang, bounded: the 5.0s caller-chosen timeout must be the thing
    # that fired, not the test's 10s safety net.
    assert elapsed < 8.0, f"took {elapsed:.1f}s — closer to the outer safety net than the timeout"
