"""Chaos test (M1.4, issue #187): kill Safety Supervisor mid-request.

PR #122 made the Kernel fail closed when Safety Supervisor analysis is
unavailable (``KernelService._get_risk_analysis`` returns a maximum-risk
``RiskAnalysis`` instead of ``None`` on any error, and the evaluator maps
missing analysis to ``risk_score = 1.0``). Its tests (``test_service.py``)
cover that *logic* with a fake event bus that returns a canned error or
malformed payload. They don't cover the actual scenario the fail-closed
path exists for: the Safety Supervisor process dying while a request is
genuinely in flight over a real NATS broker.

This spins up a real ``nats-server`` (JetStream enabled — ``EventBus.connect``
always upserts the safety-critical stream) and a real
``SafetySupervisorService`` subprocess, sends a real risk-analysis request
from a (mostly bypassed, per ``test_service.py``'s convention) ``KernelService``,
SIGKILLs the Safety Supervisor mid-request, and confirms the request fails
closed within the Kernel's configured timeout — no hang, no crash — and that
feeding the resulting analysis into the evaluator produces DENY.

Follows ``test_service.py``'s convention of driving async code through
``asyncio.run()`` instead of ``@pytest.mark.asyncio``: the Governance CI job
runs ``kernel/tests`` with ``--with pytest`` only, no pytest-asyncio.
"""

from __future__ import annotations

import asyncio
import logging
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
from activelearning import KernelDecisionType as DecisionType
from activelearning.nats_client import EventBus
from activelearning.subjects import Subjects

from kernel.evaluator import KernelEvaluator, unavailable_risk_analysis
from kernel.service import KernelService

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
def safety_supervisor_proc(nats_server: str):
    """A real Safety Supervisor service subprocess connected to nats_server.

    Yielded still-running; the test itself kills it mid-request to simulate
    a chaos event. Cleaned up defensively in teardown in case a test fails
    before reaching its own kill.
    """
    env = os.environ.copy()
    env["NATS_URL"] = nats_server
    src = str(_REPO_ROOT / "safety-supervisor" / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src, existing) if p)

    proc = subprocess.Popen(
        [sys.executable, "-m", "safety_supervisor.service"],
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
    """Poll safety.status until the subprocess has connected and subscribed."""
    deadline = asyncio.get_event_loop().time() + timeout
    last_error: Exception | None = None
    while asyncio.get_event_loop().time() < deadline:
        try:
            await bus.request(Subjects.SAFETY_STATUS, {}, timeout=0.5)
            return
        except Exception as e:  # noqa: BLE001 - polling for readiness, retried
            last_error = e
            await asyncio.sleep(0.2)
    raise AssertionError(f"Safety Supervisor never became ready: {last_error}")


async def _kill_mid_request(nats_server: str, safety_supervisor_proc: subprocess.Popen):
    """Send a real risk-analysis request, kill the responder mid-flight, and
    return (result, elapsed_seconds).

    Safety Supervisor is on the same machine talking over loopback, so a real
    request/reply round trip typically completes in well under a
    millisecond — killing it after any fixed sleep is a race a fast CI
    runner can win *for* the responder, making "no reply ever arrives" flaky
    rather than deterministic. To make the mid-request death itself
    deterministic without touching application code, freeze the process
    with SIGSTOP right before issuing the request: the OS never schedules it
    again, so it cannot dequeue or reply to the message no matter how fast
    NATS delivers it — the Kernel-observable behavior (send, wait, timeout,
    no reply) is identical to a real crash. SIGKILL (which is not blockable,
    even against a stopped process) then actually terminates it once the
    Kernel's request has resolved, so this both simulates and performs a
    genuine kill of a process that was live and mid-request a moment before.
    """
    bus = EventBus(nats_url=nats_server, name=f"kernel-chaos-{uuid.uuid4().hex[:8]}")
    await bus.connect()
    try:
        await _wait_until_ready(bus)

        svc = KernelService.__new__(KernelService)
        svc.logger = logging.getLogger("test-kernel-chaos")
        svc.event_bus = bus

        proposal = {
            "trace_id": "chaos-test-trace",
            "action": {"type": "motor", "channel": "head", "intensity": 0.1},
        }

        safety_supervisor_proc.send_signal(signal.SIGSTOP)

        t0 = asyncio.get_event_loop().time()
        try:
            # Bounded well above the Kernel's real 5.0s internal timeout so a
            # regression to an actual hang fails the test promptly instead
            # of riding the CI job to its full timeout.
            result = await asyncio.wait_for(svc._get_risk_analysis(proposal), timeout=10.0)
        finally:
            safety_supervisor_proc.kill()  # SIGKILL reaches even a stopped process
            safety_supervisor_proc.wait(timeout=5)
        elapsed = asyncio.get_event_loop().time() - t0
        return result, elapsed
    finally:
        await bus.close()


def test_kernel_fails_closed_when_safety_supervisor_dies_mid_request(
    nats_server: str,
    safety_supervisor_proc: subprocess.Popen,
) -> None:
    result, elapsed = asyncio.run(_kill_mid_request(nats_server, safety_supervisor_proc))

    # Not a hang: the Kernel's hardcoded 5.0s request timeout must actually
    # fire, not stall indefinitely waiting on a dead process.
    assert elapsed < 8.0, f"took {elapsed:.1f}s — looks like a hang, not a timeout"

    # Not a crash: _get_risk_analysis must return a usable RiskAnalysis, not
    # raise out of the request path.
    assert result.risk_score == 1.0
    assert result.flags == unavailable_risk_analysis().flags

    # Fail closed end-to-end: feeding this into the evaluator (exactly as
    # KernelService._handle_proposal does) must DENY, not silently ALLOW a
    # proposal the safety layer never actually got to analyze.
    decision = KernelEvaluator().evaluate_action_proposal(
        {"trace_id": "chaos-test-trace", "action": {"channel": "head", "intensity": 0.1}},
        risk_analysis=result,
    )
    assert decision.type == DecisionType.DENY
