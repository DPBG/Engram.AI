"""Kernel-loss watchdog — triggers SAFE_HALT if the kernel heartbeat stops (E1.9.3).

The watchdog subscribes to ``kernel.heartbeat`` and publishes ``safety.halt``
to the Kernel's existing kill-switch handler if no heartbeat arrives within
``KERNEL_WATCHDOG_TIMEOUT_S`` seconds (default 15 — 3× the kernel's 5-second
interval). Recovery is operator-gated via ``safety.resume``; the watchdog
never auto-resumes.

Run as a subprocess:
    python -m launcher.watchdog        # reads NATS_URL, KERNEL_WATCHDOG_TIMEOUT_S
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable

log = logging.getLogger(__name__)

# Subject strings are kept local so this module has zero import-time
# dependencies on the rest of the codebase (it must start even if other
# services are broken). Keep in sync with activelearning.subjects.Subjects.
_KERNEL_HEARTBEAT = "kernel.heartbeat"
_SAFETY_HALT = "safety.halt"


class KernelWatchdog:
    """Monitors ``kernel.heartbeat`` and triggers ``safety.halt`` on timeout.

    Pure logic — no NATS coupling.  The transport is injected via the
    ``publish_halt`` callable passed to :meth:`run`, keeping the class
    fully testable without a live broker.

    Lifecycle
    ---------
    * :meth:`record_heartbeat` is called on every arriving ``kernel.heartbeat``
      message; it resets the timer and clears the per-event ``_halted`` flag so
      a future second kernel loss is also detected.
    * On timeout, ``safety.halt`` is published **once** (``_halted=True``
      prevents duplicate publishes for the same loss event).
    * The watchdog does **not** publish ``safety.resume`` — resuming is
      always operator-gated and must be done explicitly.
    """

    def __init__(
        self,
        timeout_s: float = 15.0,
        check_interval_s: float = 1.0,
    ) -> None:
        self._timeout_s = timeout_s
        self._check_interval_s = check_interval_s
        # Seeded to now so the kernel has a full timeout_s window to start up
        # before the first check fires.
        self._last_heartbeat: float = time.monotonic()
        # True once we have published safety.halt for the current loss event.
        self._halted: bool = False
        # Cumulative halt-trigger count (observable in tests / metrics).
        self._halt_count: int = 0

    # ------------------------------------------------------------------
    # Called by the transport layer
    # ------------------------------------------------------------------

    def record_heartbeat(self) -> None:
        """Update the last-seen timestamp (call on each ``kernel.heartbeat``).

        Clears ``_halted`` so a *future* kernel loss triggers another halt.
        Does **not** send ``safety.resume`` — that is always operator-gated.
        """
        self._last_heartbeat = time.monotonic()
        self._halted = False

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def is_timed_out(self, now: float | None = None) -> bool:
        """True if no heartbeat has arrived within the configured timeout."""
        t = now if now is not None else time.monotonic()
        return (t - self._last_heartbeat) > self._timeout_s

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(
        self,
        publish_halt: Callable[[str, dict], Awaitable[None]],
    ) -> None:
        """Watchdog loop — runs until cancelled.

        ``publish_halt(subject, payload)`` is awaited exactly once per loss
        event with the ``safety.halt`` subject and payload.
        """
        while True:
            await asyncio.sleep(self._check_interval_s)
            if self.is_timed_out() and not self._halted:
                elapsed = time.monotonic() - self._last_heartbeat
                reason = (
                    f"kernel-loss-watchdog: no heartbeat for "
                    f"{elapsed:.1f}s (timeout={self._timeout_s}s)"
                )
                self._halt_count += 1
                log.critical("SAFE_HALT triggered — %s", reason)
                try:
                    await publish_halt(
                        _SAFETY_HALT,
                        {"reason": reason, "operator_id": "system:watchdog"},
                    )
                    # Only silence after a confirmed publish; on failure the next
                    # check cycle retries rather than leaving the system unprotected.
                    self._halted = True
                except Exception as exc:
                    log.error("Failed to publish safety.halt: %s — will retry", exc)


# ---------------------------------------------------------------------------
# Production wiring (NATS transport)
# ---------------------------------------------------------------------------

async def run_watchdog(
    nats_url: str,
    timeout_s: float = 15.0,
    check_interval_s: float = 1.0,
) -> None:
    """Connect to NATS and run the watchdog until cancelled."""
    import nats as _nats  # nats-py — only imported in production path

    watchdog = KernelWatchdog(timeout_s=timeout_s, check_interval_s=check_interval_s)
    log.info(
        "Kernel watchdog starting — timeout=%.1fs check=%.1fs nats=%s",
        timeout_s, check_interval_s, nats_url,
    )

    nc = await _nats.connect(nats_url)

    async def _on_heartbeat(msg) -> None:
        watchdog.record_heartbeat()

    async def _publish_halt(subject: str, payload: dict) -> None:
        await nc.publish(subject, json.dumps(payload).encode())

    sub = await nc.subscribe(_KERNEL_HEARTBEAT, cb=_on_heartbeat)
    try:
        await watchdog.run(publish_halt=_publish_halt)
    except asyncio.CancelledError:
        pass
    finally:
        await sub.unsubscribe()
        await nc.drain()


# ---------------------------------------------------------------------------
# Subprocess entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _nats_url = os.environ.get("NATS_URL", "nats://localhost:4222")
    _timeout = float(os.environ.get("KERNEL_WATCHDOG_TIMEOUT_S", "15.0"))
    _interval = float(os.environ.get("KERNEL_WATCHDOG_CHECK_INTERVAL_S", "1.0"))
    asyncio.run(run_watchdog(_nats_url, _timeout, _interval))
