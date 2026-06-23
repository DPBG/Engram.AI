"""Unit tests for the kernel-loss watchdog (E1.9.3).

All tests run without a live NATS broker: the transport is replaced by a
simple async callable that appends to a list.
"""
from __future__ import annotations

import asyncio
import time

from launcher.watchdog import KernelWatchdog


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expire(wdog: KernelWatchdog, extra: float = 1.0) -> None:
    """Back-date last_heartbeat so the watchdog appears already timed out."""
    wdog._last_heartbeat = time.monotonic() - wdog._timeout_s - extra


async def _run_briefly(
    wdog: KernelWatchdog,
    published: list,
    *,
    run_for: float = 0.4,
) -> None:
    """Run the watchdog loop for *run_for* seconds then cancel."""
    async def _publish(subject: str, payload: dict) -> None:
        published.append((subject, payload))

    try:
        await asyncio.wait_for(wdog.run(publish_halt=_publish), timeout=run_for)
    except asyncio.TimeoutError:
        pass


# ---------------------------------------------------------------------------
# Startup grace period
# ---------------------------------------------------------------------------

def test_freshly_constructed_watchdog_is_not_timed_out():
    """Constructor seeds last_heartbeat to now — not immediately timed out."""
    wdog = KernelWatchdog(timeout_s=15.0)
    assert not wdog.is_timed_out()


def test_is_timed_out_after_window_expires():
    """`is_timed_out` returns True once the window has passed."""
    wdog = KernelWatchdog(timeout_s=0.01)
    _expire(wdog)
    assert wdog.is_timed_out()


def test_record_heartbeat_resets_timeout():
    """`record_heartbeat` must refresh the window so is_timed_out returns False."""
    wdog = KernelWatchdog(timeout_s=0.01)
    _expire(wdog)
    assert wdog.is_timed_out()
    wdog.record_heartbeat()
    assert not wdog.is_timed_out()


# ---------------------------------------------------------------------------
# Core behaviour: halt on timeout
# ---------------------------------------------------------------------------

def test_halt_published_on_timeout():
    """No heartbeat → safety.halt is published within the first check cycle."""
    published: list = []
    wdog = KernelWatchdog(timeout_s=0.05, check_interval_s=0.01)
    _expire(wdog)
    asyncio.run(_run_briefly(wdog, published))
    assert len(published) >= 1
    assert published[0][0] == "safety.halt"


def test_halt_message_operator_id():
    """Published halt must identify the watchdog as the actor."""
    published: list = []
    wdog = KernelWatchdog(timeout_s=0.05, check_interval_s=0.01)
    _expire(wdog)
    asyncio.run(_run_briefly(wdog, published))
    _, payload = published[0]
    assert payload["operator_id"] == "system:watchdog"


def test_halt_message_reason_mentions_kernel_loss():
    """Halt reason must be human-readable and name kernel-loss-watchdog."""
    published: list = []
    wdog = KernelWatchdog(timeout_s=0.05, check_interval_s=0.01)
    _expire(wdog)
    asyncio.run(_run_briefly(wdog, published))
    _, payload = published[0]
    assert "kernel-loss-watchdog" in payload["reason"]
    assert "timeout" in payload["reason"]


# ---------------------------------------------------------------------------
# No halt while heartbeats arrive
# ---------------------------------------------------------------------------

def test_no_halt_while_heartbeats_live():
    """Continuous heartbeats must prevent any halt from firing."""
    published: list = []

    async def _inner() -> None:
        wdog = KernelWatchdog(timeout_s=0.5, check_interval_s=0.01)

        async def _publish(subject: str, payload: dict) -> None:
            published.append((subject, payload))

        async def _beat() -> None:
            for _ in range(30):
                wdog.record_heartbeat()
                await asyncio.sleep(0.01)

        try:
            await asyncio.wait_for(
                asyncio.gather(_beat(), wdog.run(publish_halt=_publish)),
                timeout=0.4,
            )
        except asyncio.TimeoutError:
            pass

    asyncio.run(_inner())
    assert published == [], "halt must not fire while heartbeats are live"


# ---------------------------------------------------------------------------
# One halt per loss event (no duplicate spam)
# ---------------------------------------------------------------------------

def test_halt_fires_only_once_per_loss_event():
    """A single loss event must produce exactly one safety.halt message."""
    published: list = []
    wdog = KernelWatchdog(timeout_s=0.05, check_interval_s=0.01)
    _expire(wdog)
    asyncio.run(_run_briefly(wdog, published, run_for=0.3))
    halts = [s for s, _ in published if s == "safety.halt"]
    assert len(halts) == 1, f"expected 1 halt, got {len(halts)}"


# ---------------------------------------------------------------------------
# No auto-resume
# ---------------------------------------------------------------------------

def test_no_auto_resume_when_heartbeat_returns():
    """Watchdog must never publish safety.resume — resuming is operator-gated."""
    resume_msgs: list = []
    halt_msgs: list = []

    async def _inner() -> None:
        async def _publish(subject: str, payload: dict) -> None:
            if subject == "safety.halt":
                halt_msgs.append(payload)
            elif subject == "safety.resume":
                resume_msgs.append(payload)

        wdog = KernelWatchdog(timeout_s=0.05, check_interval_s=0.01)
        _expire(wdog)

        # Phase 1: let the halt fire.
        try:
            await asyncio.wait_for(wdog.run(publish_halt=_publish), timeout=0.15)
        except asyncio.TimeoutError:
            pass

        assert len(halt_msgs) == 1

        # Phase 2: kernel "restarts" — heartbeat returns.
        wdog.record_heartbeat()

        # Phase 3: run again to prove no safety.resume is emitted.
        try:
            await asyncio.wait_for(wdog.run(publish_halt=_publish), timeout=0.1)
        except asyncio.TimeoutError:
            pass

    asyncio.run(_inner())
    assert resume_msgs == [], "watchdog must never auto-resume"


# ---------------------------------------------------------------------------
# Second loss event fires another halt
# ---------------------------------------------------------------------------

def test_second_kernel_loss_fires_second_halt():
    """After the kernel restarts and then dies again, a second halt must fire."""
    published: list = []

    async def _inner() -> None:
        wdog = KernelWatchdog(timeout_s=0.05, check_interval_s=0.01)
        _expire(wdog)

        # First loss → first halt.
        await _run_briefly(wdog, published, run_for=0.15)

        # Kernel "restarts".
        wdog.record_heartbeat()
        assert not wdog._halted, "record_heartbeat must clear per-event halted flag"

        # Second loss → second halt.
        _expire(wdog)
        await _run_briefly(wdog, published, run_for=0.15)

    asyncio.run(_inner())
    halts = [s for s, _ in published if s == "safety.halt"]
    assert len(halts) == 2, f"expected 2 halts, got {len(halts)}"


# ---------------------------------------------------------------------------
# halt_count metric
# ---------------------------------------------------------------------------

def test_halt_count_increments():
    """_halt_count must increment once per triggered halt."""
    published: list = []
    wdog = KernelWatchdog(timeout_s=0.05, check_interval_s=0.01)
    _expire(wdog)
    asyncio.run(_run_briefly(wdog, published, run_for=0.15))
    assert wdog._halt_count == 1
