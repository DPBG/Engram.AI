"""
Integration tests for JetStream durability on safety-critical subjects.

Acceptance criterion (issue #42):
  A broker bounce mid-flight does not drop a critical message; covered by a test.

We cannot actually stop/start NATS inside a CI test, but we can demonstrate
the equivalent property: a message published to the JetStream stream while a
durable consumer is temporarily offline is delivered when that consumer
reconnects — exactly what a broker bounce mid-flight would require.

Requires a JetStream-enabled NATS server (nats://localhost:4222 by default).
All tests are skipped when JetStream is unavailable.
"""

import asyncio
import json
import uuid

import pytest
from nats.aio.client import Client as NATSClient


@pytest.fixture
async def js_nats(nats_url: str):
    """Yield a raw NATSClient + JetStreamContext, skip if JetStream unavailable."""
    nc = NATSClient()
    await nc.connect(nats_url)
    js = nc.jetstream()
    try:
        # Probe JetStream availability with a no-op stream list
        await js.streams_info()
    except Exception:
        await nc.close()
        pytest.skip("JetStream not available on this NATS server")
    yield nc, js
    await nc.drain()
    await nc.close()


# ---------------------------------------------------------------------------
# Stream setup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_safety_stream_exists_after_event_bus_connect(nats_url: str):
    """EventBus.connect() creates the SAFETY_CRITICAL stream covering all critical subjects."""
    from activelearning.nats_client import EventBus, SAFETY_STREAM_NAME

    bus = EventBus(nats_url=nats_url, name="test-stream-setup")
    await bus.connect()
    try:
        nc = NATSClient()
        await nc.connect(nats_url)
        js = nc.jetstream()
        try:
            await js.streams_info()
        except Exception:
            await nc.close()
            pytest.skip("JetStream not available")

        for subject in ("proposal.new", "code.proposal", "decision.abc", "code.decision.abc"):
            stream_name = await js.find_stream(subject)
            assert stream_name == SAFETY_STREAM_NAME, (
                f"Subject '{subject}' should be covered by {SAFETY_STREAM_NAME}, "
                f"got '{stream_name}'"
            )

        await nc.drain()
        await nc.close()
    finally:
        await bus.close()


# ---------------------------------------------------------------------------
# Publish durability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proposal_publish_returns_ack(nats_url: str):
    """Publishing to proposal.new returns a JetStream ACK (message is persisted)."""
    from activelearning.nats_client import EventBus

    bus = EventBus(nats_url=nats_url, name="test-js-pub")
    await bus.connect()
    try:
        nc = NATSClient()
        await nc.connect(nats_url)
        js = nc.jetstream()
        try:
            await js.streams_info()
        except Exception:
            await nc.close()
            pytest.skip("JetStream not available")

        trace_id = str(uuid.uuid4())
        payload = json.dumps(
            {"trace_id": trace_id, "provenance": "test", "action": {"type": "test"}}
        ).encode()

        ack = await js.publish("proposal.new", payload)
        assert ack.seq > 0, "JetStream ACK must carry a positive sequence number"
        assert ack.stream == "SAFETY_CRITICAL"

        await nc.drain()
        await nc.close()
    finally:
        await bus.close()


# ---------------------------------------------------------------------------
# Core durability property: message delivered after consumer reconnect
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proposal_delivered_after_subscriber_reconnect(nats_url: str):
    """
    Demonstrates the broker-bounce durability guarantee.

    Timeline:
      1. Durable consumer is created (Kernel startup).
      2. Consumer disconnects (simulates broker bounce / Kernel restart).
      3. Publisher sends proposal.new to the stream (mid-flight message).
      4. Consumer reconnects with the same durable name.
      5. The pending message is redelivered — not lost.
    """
    trace_id = str(uuid.uuid4())
    durable = f"test-kernel-{trace_id[:8]}"
    subject = "proposal.new"

    # Step 1 — Create the durable consumer by subscribing briefly.
    nc1 = NATSClient()
    await nc1.connect(nats_url)
    js1 = nc1.jetstream()
    try:
        await js1.streams_info()
    except Exception:
        await nc1.close()
        pytest.skip("JetStream not available")

    # Ensure the safety stream exists before binding a consumer to it.
    from activelearning.nats_client import _SAFETY_STREAM_SUBJECTS, SAFETY_STREAM_NAME
    from nats.js.api import StreamConfig

    await js1.add_stream(StreamConfig(name=SAFETY_STREAM_NAME, subjects=_SAFETY_STREAM_SUBJECTS))

    received_step4: list[dict] = []
    done = asyncio.Event()

    async def _handler(msg) -> None:
        data = json.loads(msg.data.decode())
        if data.get("trace_id") == trace_id:
            received_step4.append(data)
            done.set()
            await msg.ack()

    sub1 = await js1.subscribe(subject, cb=_handler, durable=durable)
    await asyncio.sleep(0.05)  # let subscription establish

    # Step 2 — Disconnect the consumer (broker bounce / process restart).
    await sub1.unsubscribe()
    await nc1.drain()
    await nc1.close()

    # Step 3 — Publish while consumer is offline.
    nc2 = NATSClient()
    await nc2.connect(nats_url)
    js2 = nc2.jetstream()
    payload = json.dumps(
        {"trace_id": trace_id, "provenance": "planner", "action": {"type": "move"}}
    ).encode()
    ack = await js2.publish(subject, payload)
    assert ack.seq > 0, "message must be persisted in stream"
    await nc2.drain()
    await nc2.close()

    # Step 4 — Reconnect with the same durable; pending message must be delivered.
    nc3 = NATSClient()
    await nc3.connect(nats_url)
    js3 = nc3.jetstream()
    sub3 = await js3.subscribe(subject, cb=_handler, durable=durable)
    try:
        await asyncio.wait_for(done.wait(), timeout=5.0)
    finally:
        await sub3.unsubscribe()
        await nc3.drain()
        await nc3.close()

    assert len(received_step4) >= 1, "message must be redelivered after consumer reconnects"
    assert received_step4[0]["trace_id"] == trace_id


@pytest.mark.asyncio
async def test_code_proposal_delivered_after_subscriber_reconnect(nats_url: str):
    """Same durability guarantee for code.proposal (meta-programmer → Kernel)."""
    trace_id = str(uuid.uuid4())
    durable = f"test-kcode-{trace_id[:8]}"
    subject = "code.proposal"

    nc1 = NATSClient()
    await nc1.connect(nats_url)
    js1 = nc1.jetstream()
    try:
        await js1.streams_info()
    except Exception:
        await nc1.close()
        pytest.skip("JetStream not available")

    from activelearning.nats_client import _SAFETY_STREAM_SUBJECTS, SAFETY_STREAM_NAME
    from nats.js.api import StreamConfig

    await js1.add_stream(StreamConfig(name=SAFETY_STREAM_NAME, subjects=_SAFETY_STREAM_SUBJECTS))

    received: list[dict] = []
    done = asyncio.Event()

    async def _handler(msg) -> None:
        data = json.loads(msg.data.decode())
        if data.get("trace_id") == trace_id:
            received.append(data)
            done.set()
            await msg.ack()

    sub1 = await js1.subscribe(subject, cb=_handler, durable=durable)
    await asyncio.sleep(0.05)
    await sub1.unsubscribe()
    await nc1.drain()
    await nc1.close()

    nc2 = NATSClient()
    await nc2.connect(nats_url)
    js2 = nc2.jetstream()
    payload = json.dumps(
        {"trace_id": trace_id, "target_path": "/data/plugins/foo.py", "code_preview": "pass"}
    ).encode()
    ack = await js2.publish(subject, payload)
    assert ack.seq > 0
    await nc2.drain()
    await nc2.close()

    nc3 = NATSClient()
    await nc3.connect(nats_url)
    js3 = nc3.jetstream()
    sub3 = await js3.subscribe(subject, cb=_handler, durable=durable)
    try:
        await asyncio.wait_for(done.wait(), timeout=5.0)
    finally:
        await sub3.unsubscribe()
        await nc3.drain()
        await nc3.close()

    assert len(received) >= 1
    assert received[0]["trace_id"] == trace_id
