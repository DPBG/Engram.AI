"""Integration smoke test: prove the SDK EventBus talks to a live NATS server.

Unit tests exercise services in isolation with mocked transports. Nothing today
exercises the actual wiring every service depends on — connecting to NATS,
provisioning the JetStream safety stream, and a real publish -> subscribe
round-trip. This script closes that gap.

It boots nothing itself: it expects a JetStream-enabled nats-server reachable at
$NATS_URL (the integration workflow starts one). On a non-safety subject the
EventBus uses core NATS and passes the payload through unchanged, so the check is
not coupled to any message schema.

Run:
    NATS_URL=nats://127.0.0.1:4222 python sdk/tests/integration/stack_smoke.py

Exit code 0 = the stack's messaging path works end to end; 1 = it does not.
"""

from __future__ import annotations

import asyncio
import os
import sys

from activelearning.nats_client import EventBus

SUBJECT = "smoke.ping"
TIMEOUT_S = 10.0
SENT = {"value": 42, "note": "engram-stack-smoke"}


async def main() -> int:
    url = os.environ.get("NATS_URL", "nats://127.0.0.1:4222")
    inbox: asyncio.Queue[dict] = asyncio.Queue()

    # Two independent connections stand in for two services on the bus.
    subscriber = EventBus(nats_url=url, name="smoke-subscriber")
    publisher = EventBus(nats_url=url, name="smoke-publisher")

    async def on_ping(data: dict) -> None:
        await inbox.put(data)

    try:
        # connect() also provisions the JetStream safety stream, so a clean
        # connect proves JetStream (used for safety-critical subjects) is up.
        await subscriber.connect()
        await publisher.connect()
        await subscriber.subscribe(SUBJECT, on_ping)
        await asyncio.sleep(0.2)  # let the server register the subscription

        await publisher.publish(SUBJECT, dict(SENT))
        got = await asyncio.wait_for(inbox.get(), timeout=TIMEOUT_S)
    except TimeoutError:
        print(f"FAIL: no message received on '{SUBJECT}' within {TIMEOUT_S}s", file=sys.stderr)
        return 1
    finally:
        await subscriber.close()
        await publisher.close()

    # Subset match: a payload envelope may add metadata, but the values we sent
    # must survive the round trip intact.
    if got.get("value") != SENT["value"] or got.get("note") != SENT["note"]:
        print(f"FAIL: payload mismatch: sent={SENT} got={got}", file=sys.stderr)
        return 1

    print(f"OK: round-trip on '{SUBJECT}' delivered {got}")
    print("OK: JetStream safety stream provisioned during connect()")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
