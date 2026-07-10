"""Integration tests for EventBus / nats_client core behaviors."""

import asyncio
import uuid

import pytest

from activelearning.core import generate_trace_id
from activelearning.nats_client import EventBus
from activelearning.signing import DECISION_KEY_ENV, DECISION_KEY_SECONDARY_ENV, sign_decision
from activelearning.subjects import Subjects, code_decision_subject, decision_subject


@pytest.mark.asyncio
async def test_publish_subscribe_round_trip(event_bus: EventBus, wait_for_message):
    subject = f"test.roundtrip.{uuid.uuid4().hex[:8]}"
    received: list[dict] = []

    async def handler(data: dict) -> None:
        received.append(data)

    await event_bus.subscribe(subject, handler)
    payload = {"message": "hello", "value": 7}
    await event_bus.publish(subject, payload)

    await wait_for_message(lambda: len(received) == 1)
    assert received[0] == payload


@pytest.mark.asyncio
async def test_request_reply_success(event_bus: EventBus):
    subject = f"test.request.{uuid.uuid4().hex[:8]}"

    async def handler(data: dict, msg) -> None:
        from activelearning.nats_client import serialize_message

        await msg.respond(serialize_message({"echo": data, "status": "ok"}))

    await event_bus.subscribe(subject, handler, is_request_handler=True)
    response = await event_bus.request(subject, {"ping": "pong"}, timeout=2.0)
    assert response["status"] == "ok"
    assert response["echo"]["ping"] == "pong"


@pytest.mark.asyncio
async def test_request_reply_timeout(event_bus: EventBus):
    from nats.errors import NoRespondersError

    subject = f"test.timeout.{uuid.uuid4().hex[:8]}"
    with pytest.raises((asyncio.TimeoutError, NoRespondersError)):
        await event_bus.request(subject, {"noop": True}, timeout=0.2)


@pytest.mark.asyncio
async def test_force_reconnect_restores_handlers(event_bus: EventBus, wait_for_message):
    subject = f"test.reconnect.{uuid.uuid4().hex[:8]}"
    received: list[dict] = []

    async def handler(data: dict) -> None:
        received.append(data)

    await event_bus.subscribe(subject, handler)
    await event_bus.force_reconnect()
    assert event_bus.is_connected

    await event_bus.publish(subject, {"after": "reconnect"})
    await wait_for_message(lambda: len(received) == 1)
    assert received[0]["after"] == "reconnect"


@pytest.mark.asyncio
async def test_wait_for_decision_rejects_unsigned_when_signing_enabled(
    event_bus: EventBus,
    monkeypatch,
):
    key = "test-decision-secret"
    monkeypatch.setenv(DECISION_KEY_ENV, key)
    trace_id = generate_trace_id()

    await event_bus.publish(
        decision_subject(trace_id),
        {
            "trace_id": trace_id,
            "type": "ALLOW",
            "reason": "forged",
            "risk_score": 0.0,
        },
    )
    with pytest.raises(asyncio.TimeoutError):
        await event_bus.wait_for_decision(trace_id, timeout=0.5)


@pytest.mark.asyncio
async def test_wait_for_decision_accepts_signed_decision(
    event_bus: EventBus,
    monkeypatch,
):
    key = "test-decision-secret-2"
    monkeypatch.setenv(DECISION_KEY_ENV, key)
    trace_id = generate_trace_id()

    async def publish_signed():
        await asyncio.sleep(0.1)
        decision = sign_decision(
            {
                "trace_id": trace_id,
                "type": "ALLOW",
                "reason": "signed",
                "risk_score": 0.1,
            },
            key,
        )
        await event_bus.publish(decision_subject(trace_id), decision)

    publish_task = asyncio.create_task(publish_signed())
    result = await event_bus.wait_for_decision(trace_id, timeout=2.0)
    await publish_task
    assert result["trace_id"] == trace_id
    assert result["type"] == "ALLOW"


# ── key rotation (issue #206) ─────────────────────────────────────────────
#
# End-to-end over a real NATS broker: proves the dual-key overlap window
# documented in docs/DECISION-KEY-ROTATION.md never drops an in-flight
# decision, and that completing the rotation actually retires the old key
# rather than trusting it forever.


@pytest.mark.asyncio
async def test_wait_for_decision_overlap_window_accepts_both_keys(
    event_bus: EventBus,
    monkeypatch,
):
    """Mid-rotation state: the Kernel has just flipped to the new key, but
    this waiter (like every verifier during the overlap window) still has
    the old key configured as its secondary. A decision published with the
    *old* key — as if it were already in flight when the Kernel flipped —
    and a decision published with the *new* key must both be accepted."""
    old_key = "rotation-old-key"
    new_key = "rotation-new-key"
    monkeypatch.setenv(DECISION_KEY_ENV, new_key)
    monkeypatch.setenv(DECISION_KEY_SECONDARY_ENV, old_key)

    in_flight_trace = generate_trace_id()
    fresh_trace = generate_trace_id()

    async def publish_both():
        await asyncio.sleep(0.1)
        in_flight = sign_decision(
            {
                "trace_id": in_flight_trace,
                "type": "ALLOW",
                "reason": "signed before the Kernel rotated",
                "risk_score": 0.1,
            },
            old_key,
        )
        await event_bus.publish(decision_subject(in_flight_trace), in_flight)

        fresh = sign_decision(
            {
                "trace_id": fresh_trace,
                "type": "DENY",
                "reason": "signed after the Kernel rotated",
                "risk_score": 0.9,
            },
            new_key,
        )
        await event_bus.publish(decision_subject(fresh_trace), fresh)

    publish_task = asyncio.create_task(publish_both())
    in_flight_result = await event_bus.wait_for_decision(in_flight_trace, timeout=2.0)
    fresh_result = await event_bus.wait_for_decision(fresh_trace, timeout=2.0)
    await publish_task

    assert in_flight_result["type"] == "ALLOW", "old-key decision dropped during rotation overlap"
    assert fresh_result["type"] == "DENY"


@pytest.mark.asyncio
async def test_wait_for_decision_retires_old_key_after_rotation_completes(
    event_bus: EventBus,
    monkeypatch,
):
    """After the overlap window, the secondary key is removed. A decision
    signed with the now-retired old key must no longer satisfy a waiter —
    proving rotation actually completes instead of accumulating trusted
    keys forever."""
    old_key = "rotation-old-key-2"
    new_key = "rotation-new-key-2"
    monkeypatch.setenv(DECISION_KEY_ENV, new_key)
    monkeypatch.delenv(DECISION_KEY_SECONDARY_ENV, raising=False)
    trace_id = generate_trace_id()

    stale = sign_decision(
        {
            "trace_id": trace_id,
            "type": "ALLOW",
            "reason": "signed with a retired key",
            "risk_score": 0.0,
        },
        old_key,
    )
    await event_bus.publish(decision_subject(trace_id), stale)

    with pytest.raises(asyncio.TimeoutError):
        await event_bus.wait_for_decision(trace_id, timeout=0.5)


@pytest.mark.asyncio
async def test_wait_for_decision_code_subject(
    event_bus: EventBus,
    monkeypatch,
):
    key = "test-decision-secret-code"
    monkeypatch.setenv(DECISION_KEY_ENV, key)
    trace_id = generate_trace_id()

    async def publish_signed_code_decision():
        await asyncio.sleep(0.1)
        decision = sign_decision(
            {
                "trace_id": trace_id,
                "type": "DENY",
                "reason": "unsafe code",
                "risk_score": 0.9,
            },
            key,
        )
        await event_bus.publish(code_decision_subject(trace_id), decision)

    publish_task = asyncio.create_task(publish_signed_code_decision())
    result = await event_bus.wait_for_decision(trace_id, timeout=2.0, code=True)
    await publish_task
    assert result["trace_id"] == trace_id
    assert result["type"] == "DENY"


@pytest.mark.asyncio
async def test_wait_for_decision_code_subject_ignores_action_decision(
    event_bus: EventBus,
    monkeypatch,
):
    key = "test-decision-secret-code-2"
    monkeypatch.setenv(DECISION_KEY_ENV, key)
    trace_id = generate_trace_id()

    forged_allow = sign_decision(
        {
            "trace_id": trace_id,
            "type": "ALLOW",
            "reason": "forged action decision",
            "risk_score": 0.0,
        },
        key,
    )
    await event_bus.publish(decision_subject(trace_id), forged_allow)

    with pytest.raises(asyncio.TimeoutError):
        await event_bus.wait_for_decision(trace_id, timeout=0.5, code=True)


@pytest.mark.asyncio
async def test_js_subscribe_delivers_proposal(event_bus: EventBus, wait_for_message):
    received: list[dict] = []

    async def handler(data: dict) -> None:
        received.append(data)

    durable = f"test-durable-{uuid.uuid4().hex[:8]}"
    await event_bus.js_subscribe(Subjects.PROPOSAL_NEW, handler, durable=durable)

    payload = {
        "trace_id": f"t-{uuid.uuid4().hex[:8]}",
        "action": {"type": "test"},
        "provenance": "nats-test",
    }
    await event_bus.publish(Subjects.PROPOSAL_NEW, payload)
    # A fresh durable replays the stream's retained history (DeliverPolicy ALL),
    # so proposal.new messages from earlier tests in the session may arrive too.
    # Assert OUR message was delivered rather than assuming it is the only one.
    await wait_for_message(lambda: any(m.get("trace_id") == payload["trace_id"] for m in received))
