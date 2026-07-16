"""Minimal BaseService subclass for the kill-mid-shutdown chaos test (#260).

Not a real product service -- exists only so
test_base_service_shutdown_chaos.py can run a real subprocess to kill.
Subscribes to a JetStream subject (must be a decision.* subject supplied by
the test -- js_subscribe validates the payload against a registered wire
schema per subject, and decision.* is already covered by the safety stream
with a permissive {trace_id, type, reason (+extra fields ok)} schema, same
convention as sdk/tests/test_jetstream_durability.py). On each message,
writes a "started" marker file, sleeps (simulating slow processing), then
writes a "completed" marker file immediately before returning (i.e.
immediately before js_subscribe's callback acks). The test freezes+kills
this process during the sleep window, then inspects which marker files
exist to prove message processing was interrupted before ack, and asks a
second consumer whether the broker redelivered.

Run as a subprocess:
    python -m activelearning._chaos_kill_service
    (reads NATS_URL, CHAOS_SUBJECT, CHAOS_DURABLE, CHAOS_MARKER_DIR,
    CHAOS_SLEEP_SECONDS from the environment)
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from activelearning.base_service import BaseService


class ChaosKillService(BaseService):
    def __init__(self) -> None:
        super().__init__("chaos-kill-test", use_database=False, use_event_bus=True)
        self._subject = os.environ["CHAOS_SUBJECT"]
        self._durable = os.environ["CHAOS_DURABLE"]
        self._marker_dir = Path(os.environ["CHAOS_MARKER_DIR"])
        self._sleep_seconds = float(os.environ.get("CHAOS_SLEEP_SECONDS", "5.0"))

    async def _setup(self) -> None:
        assert self.event_bus is not None
        # backoff[0] becomes ack_wait -- set comfortably longer than
        # _sleep_seconds so the broker doesn't attempt redelivery to this
        # SAME (soon to be frozen/killed) process before the test kills it;
        # the fresh-consumer half of the test then waits out this window.
        await self.event_bus.js_subscribe(
            self._subject,
            self._handle,
            durable=self._durable,
            backoff=[8.0, 8.0],
        )

    async def _handle(self, data: dict) -> None:
        marker = str(data.get("trace_id", "unknown"))
        (self._marker_dir / f"started-{marker}").write_text("1")
        await asyncio.sleep(self._sleep_seconds)
        (self._marker_dir / f"completed-{marker}").write_text("1")
        # Returning here is what lets js_subscribe's callback ack the
        # message -- a kill during the sleep above means this line, and
        # the ack, never happen.


async def main() -> None:
    service = ChaosKillService()
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
