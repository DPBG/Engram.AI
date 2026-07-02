"""Unit tests for the kernel-loss watchdog (E1.9.3).

All tests run without a live NATS broker: the transport is replaced by a
simple async callable that appends to a list.

Design note: tests that verify a halt FIRES use an asyncio.Event for
deterministic signaling — the test completes as soon as the halt arrives,
not after a fixed wall-clock slice. This eliminates CI-jitter flakiness.
Tests that verify a halt does NOT fire use generous heartbeat rates
(beat_interval << timeout) so even a slow CI host cannot trigger a false halt.
"""
from __future__ import annotations

import asyncio
import contextlib
import time

from launcher.watchdog import KernelWatchdog

# ---------------------------------------------------------------------------
# Parameters chosen to be CI-safe:
#   _HB_TIMEOUT   — watchdog fires after this many seconds with no heartbeat
#   _CHECK        — watchdog polls this often
#   _SAFE_WAIT    — upper bound we allow the halt event to arrive (very generous)
# ---------------------------------------------------------------------------
_HB_TIMEOUT = 1.0    # large enough that spurious delays don't accidentally fire
_CHECK = 0.05        # 50 ms — responsive without being tight
_SAFE_WAIT = 10.0    # we expect the halt in < 0.1 s; 10 s is a CI safety net


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expire(wdog: KernelWatchdog, extra: float = 1.0) -> None:
    """Back-date last_heartbeat so the watchdog appears already timed out."""
    wdog._last_heartbeat = time.monotonic() - wdog._timeout_s - extra


async def _run_until_halt(
    wdog: KernelWatchdog,
    published: list,
    *,
    safety_timeout: float = _SAFE_WAIT,
) -> None:
    """Run the watchdog until safety.halt is published, then stop.

    Uses asyncio.Event so the test completes immediately when the halt
    arrives — no fixed sleep, no CI-jitter sensitivity.
    """
    halt_seen = asyncio.Event()

    async def _publish(subject: str, payload: dict) -> None:
        published.append((subject, payload))
        if subject == "safety.halt":
            halt_seen.set()

    task = asyncio.create_task(wdog.run(publish_halt=_publish))
    try:
        await asyncio.wait_for(halt_seen.wait(), timeout=safety_timeout)
    except asyncio.TimeoutError:
        pass  # safety net exceeded — test assertions will fail with useful message
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def _run_for(
    wdog: KernelWatchdog,
    published: list,
    duration: float,
) -> None:
    """Run the watchdog for exactly *duration* seconds (used by no-halt tests)."""
    async def _publish(subject: str, payload: dict) -> None:
        published.append((subject, payload))

    try:
        await asyncio.wait_for(wdog.run(publish_halt=_publish), timeout=duration)
    except asyncio.TimeoutError:
        pass


# ---------------------------------------------------------------------------
# Startup grace period (sync — no NATS needed)
# ---------------------------------------------------------------------------

def test_freshly_constructed_watchdog_is_not_timed_out():
    """Constructor seeds last_heartbeat to now — not immediately timed out."""
    wdog = KernelWatchdog(timeout_s=_HB_TIMEOUT)
    assert not wdog.is_timed_out()


def test_is_timed_out_after_window_expires():
    """`is_timed_out` returns True once the window has passed."""
    wdog = KernelWatchdog(timeout_s=_HB_TIMEOUT)
    _expire(wdog)
    assert wdog.is_timed_out()


def test_record_heartbeat_resets_timeout():
    """`record_heartbeat` must refresh the window so is_timed_out returns False."""
    wdog = KernelWatchdog(timeout_s=_HB_TIMEOUT)
    _expire(wdog)
    assert wdog.is_timed_out()
    wdog.record_heartbeat()
    assert not wdog.is_timed_out()


# ---------------------------------------------------------------------------
# Core behaviour: halt on timeout
# ---------------------------------------------------------------------------

def test_halt_published_on_timeout():
    """No heartbeat → safety.halt is published (event-driven, not time-bounded)."""
    published: list = []
    wdog = KernelWatchdog(timeout_s=_HB_TIMEOUT, check_interval_s=_CHECK)
    _expire(wdog)
    asyncio.run(_run_until_halt(wdog, published))
    assert len(published) >= 1
    assert published[0][0] == "safety.halt"


def test_halt_message_operator_id():
    """Published halt must identify the watchdog as the actor."""
    published: list = []
    wdog = KernelWatchdog(timeout_s=_HB_TIMEOUT, check_interval_s=_CHECK)
    _expire(wdog)
    asyncio.run(_run_until_halt(wdog, published))
    _, payload = published[0]
    assert payload["operator_id"] == "system:watchdog"


def test_halt_message_reason_mentions_kernel_loss():
    """Halt reason must be human-readable and name kernel-loss-watchdog."""
    published: list = []
    wdog = KernelWatchdog(timeout_s=_HB_TIMEOUT, check_interval_s=_CHECK)
    _expire(wdog)
    asyncio.run(_run_until_halt(wdog, published))
    _, payload = published[0]
    assert "kernel-loss-watchdog" in payload["reason"]
    assert "timeout" in payload["reason"]


# ---------------------------------------------------------------------------
# No halt while heartbeats arrive
# ---------------------------------------------------------------------------

def test_no_halt_while_heartbeats_live():
    """Heartbeats arriving well within the timeout must prevent any halt.

    Beat interval (0.2 s) is 5× smaller than the watchdog timeout (1.0 s),
    so even a 200 ms CI-scheduling delay on a single beat cannot trigger a halt.
    """
    published: list = []

    async def _inner() -> None:
        wdog = KernelWatchdog(timeout_s=_HB_TIMEOUT, check_interval_s=_CHECK)

        async def _publish(subject: str, payload: dict) -> None:
            published.append((subject, payload))

        async def _beat() -> None:
            # 8 beats × 0.2 s = 1.6 s of coverage; timeout is 1.0 s
            for _ in range(8):
                wdog.record_heartbeat()
                await asyncio.sleep(0.2)

        try:
            await asyncio.wait_for(
                asyncio.gather(_beat(), wdog.run(publish_halt=_publish)),
                timeout=2.0,
            )
        except asyncio.TimeoutError:
            pass

    asyncio.run(_inner())
    assert published == [], "halt must not fire while heartbeats are live"


# ---------------------------------------------------------------------------
# One halt per loss event (no duplicate spam)
# ---------------------------------------------------------------------------

def test_halt_fires_only_once_per_loss_event():
    """A single loss event must produce exactly one safety.halt (no spam)."""
    published: list = []

    async def _inner() -> None:
        wdog = KernelWatchdog(timeout_s=_HB_TIMEOUT, check_interval_s=_CHECK)
        _expire(wdog)
        # Wait for the halt, then let several more check cycles pass.
        await _run_until_halt(wdog, published)
        # Run a few more cycles to confirm no second halt fires.
        await _run_for(wdog, published, duration=_CHECK * 5)

    asyncio.run(_inner())
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

        wdog = KernelWatchdog(timeout_s=_HB_TIMEOUT, check_interval_s=_CHECK)
        _expire(wdog)

        # Phase 1: wait for the halt to fire (event-driven).
        halt_seen = asyncio.Event()
        orig_publish = _publish

        async def _tracking_publish(subject: str, payload: dict) -> None:
            await orig_publish(subject, payload)
            if subject == "safety.halt":
                halt_seen.set()

        task = asyncio.create_task(wdog.run(publish_halt=_tracking_publish))
        await asyncio.wait_for(halt_seen.wait(), timeout=_SAFE_WAIT)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

        assert len(halt_msgs) == 1

        # Phase 2: kernel "restarts" — heartbeat returns.
        wdog.record_heartbeat()

        # Phase 3: run several check cycles; no safety.resume must appear.
        await _run_for(wdog, [], duration=_CHECK * 5)

    asyncio.run(_inner())
    assert resume_msgs == [], "watchdog must never auto-resume"


# ---------------------------------------------------------------------------
# Second loss event fires another halt
# ---------------------------------------------------------------------------

def test_second_kernel_loss_fires_second_halt():
    """After the kernel restarts and then dies again, a second halt must fire."""
    published: list = []

    async def _inner() -> None:
        wdog = KernelWatchdog(timeout_s=_HB_TIMEOUT, check_interval_s=_CHECK)
        _expire(wdog)

        # First loss → first halt (event-driven).
        await _run_until_halt(wdog, published)
        assert len([s for s, _ in published if s == "safety.halt"]) == 1

        # Kernel "restarts".
        wdog.record_heartbeat()
        assert not wdog._halted, "record_heartbeat must clear per-event halted flag"

        # Second loss → second halt.
        _expire(wdog)
        await _run_until_halt(wdog, published)

    asyncio.run(_inner())
    halts = [s for s, _ in published if s == "safety.halt"]
    assert len(halts) == 2, f"expected 2 halts, got {len(halts)}"


# ---------------------------------------------------------------------------
# halt_count metric
# ---------------------------------------------------------------------------

def test_halt_count_increments():
    """_halt_count must increment once per triggered halt."""
    published: list = []
    wdog = KernelWatchdog(timeout_s=_HB_TIMEOUT, check_interval_s=_CHECK)
    _expire(wdog)
    asyncio.run(_run_until_halt(wdog, published))
    assert wdog._halt_count == 1
