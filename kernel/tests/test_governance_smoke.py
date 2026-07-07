"""
Governance suite smoke test against a real NATS broker (issue #213).

Runs the Kernel against a live nats-server to catch integration-shape bugs
that unit tests with mocked buses cannot detect. The canonical example is the
4-positional-arg EventBus.subscribe() bug (PR #194 review):

    # BUGGY — the second subject silently becomes the queue-group;
    # the second handler is passed as pending_msgs_limit and never registered.
    await event_bus.subscribe(subj1, h1, subj2, h2)

    # CORRECT — one registration per subject.
    await event_bus.subscribe(subj1, h1)
    await event_bus.subscribe(subj2, h2)

A mocked bus (FakeBus.subscribe records nothing) cannot surface this because
calling subscribe() with any args appears to succeed. A real broker exposes the
bug: publishing to subj2 produces no response because the subscription does not
exist at the transport layer.

Requirements: nats-server on PATH. Skipped automatically when absent, consistent
with the red-team suite pattern. CI installs nats-server via setup-engram.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import pytest
from activelearning.database import close_database
from activelearning.nats_client import EventBus
from activelearning.subjects import Subjects

logger = logging.getLogger(__name__)


# ── Infrastructure helpers ────────────────────────────────────────────────────


def _nats_available() -> bool:
    return shutil.which("nats-server") is not None


def _free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


pytestmark = pytest.mark.skipif(
    not _nats_available(),
    reason=(
        "nats-server not on PATH — skipping governance smoke tests. "
        "CI installs it via setup-engram (nats: true)."
    ),
)


# ── Smoke test ────────────────────────────────────────────────────────────────


def test_kernel_routes_policy_restrict_request_over_real_nats():
    """Kernel must subscribe to and handle policy.restrict.request against a live broker.

    This smoke test validates end-to-end subscription wiring that a mocked
    EventBus cannot detect. With the 4-positional-arg bug present in _setup():

        subscribe(POLICY_RESTRICT, h1, POLICY_RESTRICT_REQUEST, h2)

    EventBus.subscribe() treats POLICY_RESTRICT_REQUEST as the ``queue``
    parameter and h2 as ``pending_msgs_limit``. The subscription to
    POLICY_RESTRICT_REQUEST is never created; publishing to it produces no
    response. This test times out with that bug and passes with the fix
    (two separate subscribe calls).

    Even without a body profile loaded, _apply_restriction publishes
    policy.restrict.status{status: "error"} — proving the handler ran.
    """
    asyncio.run(_smoke())


async def _smoke() -> None:
    host = "127.0.0.1"
    port = _free_port(host)
    nats_url = f"nats://{host}:{port}"
    run_id = uuid.uuid4().hex
    binary = shutil.which("nats-server")
    assert binary is not None  # guaranteed by pytestmark

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"engram-smoke-{run_id[:8]}-"))
    sqlite_path = str(tmp_dir / "kernel-smoke.db")
    nats_store = tmp_dir / "nats-store"
    nats_store.mkdir()

    proc = subprocess.Popen(
        [binary, "-js", "-p", str(port), "-sd", str(nats_store)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    kernel = None
    probe_bus = None
    saved_env: dict[str, str | None] = {}

    try:
        # Wait for the broker to accept connections.
        for _ in range(50):
            if _port_open(host, port):
                break
            if proc.poll() is not None:
                raise RuntimeError("nats-server exited before becoming ready")
            time.sleep(0.1)
        else:
            raise RuntimeError("Timed out waiting for nats-server to start")

        # Override env vars that BaseService/ServiceConfig read at construction.
        # Save originals so the finally block can restore them.
        for key, val in (("NATS_URL", nats_url), ("SQLITE_PATH", sqlite_path)):
            saved_env[key] = os.environ.get(key)
            os.environ[key] = val

        # Import here so the module is found with PYTHONPATH set by CI / local env.
        from kernel.service import KernelService

        kernel = KernelService()
        await kernel.start()

        # Connect a separate probe bus — represents the dashboard or brain.
        probe_bus = EventBus(nats_url=nats_url, name=f"smoke-probe-{run_id[:6]}")
        await probe_bus.connect()

        # Capture policy.restrict.status.
        # _apply_restriction publishes this on every code path (applied / error /
        # rejected), so the message arrives even when no body profile is loaded.
        # Its arrival is proof that _handle_restrict_request was called end-to-end.
        status_q: asyncio.Queue[dict] = asyncio.Queue()
        await probe_bus.subscribe("policy.restrict.status", status_q.put)

        # Give the subscriptions a moment to propagate to the broker.
        await asyncio.sleep(0.1)

        # Publish a restrict request — the payload a dashboard operator would send.
        await probe_bus.publish(
            Subjects.POLICY_RESTRICT_REQUEST,
            {"motor_limits": {"locomotion": {"max_intensity": 0.5}}, "reason": "smoke"},
        )

        # The Kernel must route the message:
        #   policy.restrict.request
        #       → _handle_restrict_request
        #       → _apply_restriction
        #       → policy.restrict.status
        #
        # With the 4-positional-arg bug the subscription never registers and this
        # times out, failing with an explicit diagnostic message.
        try:
            status = await asyncio.wait_for(status_q.get(), timeout=5.0)
        except TimeoutError:
            raise AssertionError(
                "Timed out waiting for policy.restrict.status — "
                "Kernel likely did not subscribe to policy.restrict.request. "
                "Check for the 4-positional-arg EventBus.subscribe() bug (PR #194): "
                "subscribe(subj1, h1, subj2, h2) silently drops the subj2 subscription."
            )

        assert status.get("status") in (
            "applied",
            "error",
            "rejected",
        ), f"Unexpected policy.restrict.status payload: {status!r}"

    finally:
        if probe_bus:
            await probe_bus.close()
        if kernel:
            await kernel.stop()
        # KernelService(use_database=True) opens the real get_database()
        # singleton. BaseService.stop() deliberately leaves it open (see its
        # comment), but aiosqlite's connection worker thread is non-daemon —
        # left open, it hangs pytest at interpreter shutdown after this test
        # (and every test after it) reports passed. Close it explicitly.
        await close_database()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        # Restore env vars to their original values (or remove if they weren't set).
        for key, original in saved_env.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original
