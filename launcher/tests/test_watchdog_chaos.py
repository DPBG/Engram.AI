"""Chaos test (M2.40): KernelWatchdog loses its own NATS connection.

``launcher/watchdog.py`` (E1.9.3) catches Kernel loss via ``kernel.heartbeat``.
Untested until now: what happens if the *watchdog itself* loses NATS while the
broker and a live heartbeat publisher keep running.

Unsafe failure mode: the watchdog goes blind, reconnects, accepts resumed
heartbeats, and silently continues — never SAFE_HALTing, even though it cannot
vouch for kernel liveness during the gap.

Safe failure mode: after a blind period longer than the watchdog timeout,
reconnect mandates ``safety.halt`` (fail toward SAFE_HALT). Heartbeats that
resume after reconnect must not cancel that mandate.

This spins up a real ``nats-server``, a heartbeat publisher on a *separate*
NATS client (kernel still "alive"), and a watchdog client whose reconnect is
artificially delayed past the timeout via ``reconnect_to_server_handler`` —
so only the watchdog's connection is down, not the whole broker.

Follows ``kernel/tests/test_kernel_crash_chaos.py``'s convention of driving
async code through ``asyncio.run()`` (no pytest-asyncio required).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import time

import pytest

from launcher.watchdog import KernelWatchdog

pytestmark = pytest.mark.skipif(
    shutil.which("nats-server") is None,
    reason="nats-server not on PATH — skipping chaos integration test",
)

_KERNEL_HEARTBEAT = "kernel.heartbeat"
_SAFETY_HALT = "safety.halt"

# CI-safe timings: reconnect delay exceeds timeout so blindness is unambiguous.
_HB_TIMEOUT = 0.6
_CHECK = 0.05
_RECONNECT_DELAY = 0.9  # > _HB_TIMEOUT
_SAFE_WAIT = 15.0


async def _watchdog_blind_while_heartbeats_continue(nats_url: str):
    """Disconnect only the watchdog client; return (halt_payloads, halt_count)."""
    import nats as _nats
    from nats.aio.client import Server

    halt_seen = asyncio.Event()
    halt_payloads: list[dict] = []

    # Observer — stays connected for the whole test; receives safety.halt.
    observer = await _nats.connect(nats_url, name="watchdog-chaos-observer")

    async def _on_halt(msg) -> None:
        halt_payloads.append(json.loads(msg.data.decode()))
        halt_seen.set()

    await observer.subscribe(_SAFETY_HALT, cb=_on_halt)

    # Heartbeat publisher — separate connection; keeps beating while watchdog
    # is blind, proving the unsafe race (resumed HBs after reconnect).
    publisher = await _nats.connect(nats_url, name="watchdog-chaos-publisher")
    beat_stop = asyncio.Event()

    async def _beat_loop() -> None:
        while not beat_stop.is_set():
            await publisher.publish(_KERNEL_HEARTBEAT, b"{}")
            try:
                await asyncio.wait_for(beat_stop.wait(), timeout=0.1)
            except TimeoutError:
                pass

    beat_task = asyncio.create_task(_beat_loop())

    watchdog = KernelWatchdog(timeout_s=_HB_TIMEOUT, check_interval_s=_CHECK)
    reconnect_delays_left = {"n": 1}

    def _delayed_reconnect(servers: list[Server], _info: dict):
        # First reconnect after force_reconnect: hold the watchdog blind
        # longer than timeout while the broker + publisher stay up.
        delay = _RECONNECT_DELAY if reconnect_delays_left["n"] > 0 else 0.0
        if reconnect_delays_left["n"] > 0:
            reconnect_delays_left["n"] -= 1
        return servers[0], delay

    async def _on_disconnect() -> None:
        watchdog.notify_transport_down()

    async def _on_reconnect() -> None:
        watchdog.notify_transport_up()

    wd_nc = await _nats.connect(
        nats_url,
        name="watchdog-chaos-under-test",
        allow_reconnect=True,
        disconnected_cb=_on_disconnect,
        reconnected_cb=_on_reconnect,
        reconnect_to_server_handler=_delayed_reconnect,
        max_reconnect_attempts=-1,
    )

    async def _on_heartbeat(_msg) -> None:
        watchdog.record_heartbeat()

    async def _publish_halt(subject: str, payload: dict) -> None:
        await wd_nc.publish(subject, json.dumps(payload).encode())

    await wd_nc.subscribe(_KERNEL_HEARTBEAT, cb=_on_heartbeat)

    # Prove the happy path first: live heartbeats → no halt yet.
    await asyncio.sleep(_HB_TIMEOUT * 0.5)
    assert not halt_seen.is_set(), "precondition: heartbeats must suppress halt"
    assert watchdog._halt_count == 0

    run_task = asyncio.create_task(watchdog.run(publish_halt=_publish_halt))
    try:
        # Sever *only* the watchdog's NATS connection; publisher + observer stay up.
        await wd_nc.force_reconnect()

        await asyncio.wait_for(halt_seen.wait(), timeout=_SAFE_WAIT)

        # Give the run loop a tick to clear _halt_after_blind after publish.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and watchdog._halt_after_blind:
            await asyncio.sleep(_CHECK)

        return halt_payloads, watchdog._halt_count
    finally:
        beat_stop.set()
        run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run_task
        beat_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await beat_task
        with contextlib.suppress(Exception):
            await wd_nc.drain()
        with contextlib.suppress(Exception):
            await publisher.drain()
        with contextlib.suppress(Exception):
            await observer.drain()


def test_watchdog_nats_disconnect_fails_toward_safe_halt(nats_server: str) -> None:
    """Watchdog blind past timeout must publish safety.halt, not resume silently."""
    halt_payloads, halt_count = asyncio.run(_watchdog_blind_while_heartbeats_continue(nats_server))

    assert halt_count >= 1, "watchdog never attempted SAFE_HALT — silent inaction"
    assert halt_payloads, "safety.halt never delivered after watchdog reconnect"
    payload = halt_payloads[0]
    assert payload["operator_id"] == "system:watchdog"
    assert "NATS connection lost" in payload["reason"] or "fail-safe" in payload["reason"]
