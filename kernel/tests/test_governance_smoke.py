"""
Governance suite smoke test against a real NATS broker (issue #213).

Runs the Kernel against a live nats-server to catch integration-shape bugs
that unit tests with mocked buses cannot detect. The canonical example is the
4-positional-arg EventBus.subscribe() bug (PR #194 review):

    # BUGGY - the second subject silently becomes the queue-group;
    # the second handler is passed as pending_msgs_limit and never registered.
    await event_bus.subscribe(subj1, h1, subj2, h2)

    # CORRECT - one registration per subject.
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
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest
from activelearning.database import close_database
from activelearning.nats_client import EventBus
from activelearning.subjects import Subjects
from activelearning.testing.nats_server import nats_server_available, run_nats_server


pytestmark = pytest.mark.skipif(
    not nats_server_available(),
    reason=(
        "nats-server not on PATH - skipping governance smoke tests. "
        "CI installs it via setup-engram (nats: true)."
    ),
)


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
    policy.restrict.status{status: "error"} - proving the handler ran.
    """
    asyncio.run(_smoke())


async def _smoke() -> None:
    run_id = uuid.uuid4().hex
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"engram-smoke-{run_id[:8]}-"))
    sqlite_path = str(tmp_dir / "kernel-smoke.db")
    nats_store = tmp_dir / "nats-store"
    saved_env: dict[str, str | None] = {}

    try:
        with run_nats_server(nats_store) as server:
            await _run_smoke_against_broker(server.url, sqlite_path, run_id, saved_env)
    finally:
        await close_database()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        for key, original in saved_env.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


async def _run_smoke_against_broker(
    nats_url: str,
    sqlite_path: str,
    run_id: str,
    saved_env: dict[str, str | None],
) -> None:
    # Override env vars that BaseService/ServiceConfig read at construction.
    for key, val in (("NATS_URL", nats_url), ("SQLITE_PATH", sqlite_path)):
        saved_env[key] = os.environ.get(key)
        os.environ[key] = val

    from kernel.service import KernelService

    kernel = KernelService()
    probe_bus = None

    try:
        await kernel.start()

        # Connect a separate probe bus - represents the dashboard or brain.
        probe_bus = EventBus(nats_url=nats_url, name=f"smoke-probe-{run_id[:6]}")
        await probe_bus.connect()

        # Capture policy.restrict.status. Its arrival proves that
        # _handle_restrict_request was called end-to-end.
        status_q: asyncio.Queue[dict] = asyncio.Queue()
        await probe_bus.subscribe("policy.restrict.status", status_q.put)

        # Give the subscriptions a moment to propagate to the broker.
        await asyncio.sleep(0.1)

        # Publish the payload a dashboard operator would send.
        await probe_bus.publish(
            Subjects.POLICY_RESTRICT_REQUEST,
            {"motor_limits": {"locomotion": {"max_intensity": 0.5}}, "reason": "smoke"},
        )

        # The Kernel must route the message:
        #   policy.restrict.request -> _handle_restrict_request
        #   -> _apply_restriction -> policy.restrict.status
        try:
            status = await asyncio.wait_for(status_q.get(), timeout=5.0)
        except TimeoutError:
            raise AssertionError(
                "Timed out waiting for policy.restrict.status - "
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
        await kernel.stop()
