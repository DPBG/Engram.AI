"""
Regression test template — BaseService migration parity.

These tests verify the behavioural guarantees documented in
docs/SDK-BASESERVICE-CONTRACT.md for any service that extends BaseService.

HOW TO USE:
    1. Copy this file into the service's own tests/ directory as
       test_migration_parity.py.
    2. Replace the MinimalService fixture with a minimal stub of the real
       service (subscribe the same subjects, use the same handlers).
    3. Run: pytest tests/test_migration_parity.py -v

All tests here are self-contained — they use a real embedded NATS server
via the session-scoped `nats_url` fixture in sdk/tests/conftest.py.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator

import pytest

from activelearning.base_service import BaseService
from activelearning.core import current_timestamp
from activelearning.messages import ObservationMessage
from activelearning.nats_client import EventBus, serialize_message
from activelearning.signing import sign_decision

# ---------------------------------------------------------------------------
# Minimal stub service used throughout these tests
# ---------------------------------------------------------------------------


class MinimalService(BaseService):
    """Minimal BaseService stub for migration regression tests.

    Replace or extend for the service being migrated: subscribe the same
    subjects and replicate the handler logic.
    """

    def __init__(self, nats_url: str):
        super().__init__("test-service", use_database=False, use_event_bus=True)
        self.config.nats_url = nats_url  # override for test isolation
        self.received: list[dict] = []
        self.setup_called = False
        self.cleanup_called = False

    async def _setup(self) -> None:
        self.setup_called = True
        # 'test.topic' is not in SUBJECT_SCHEMAS, so pass the wire model
        # explicitly — receive-side validation only runs with a model resolved.
        await self.event_bus.subscribe(
            "test.topic", self._handle_topic, message_model=ObservationMessage
        )
        await self.event_bus.subscribe("test.status", self._handle_status, is_request_handler=True)

    async def _cleanup(self) -> None:
        self.cleanup_called = True

    async def _handle_topic(self, data: dict) -> None:
        self.received.append(data)

    async def _handle_status(self, data: dict, msg) -> None:
        if msg.reply:
            await msg.respond(serialize_message({"ok": True, "service": self.service_name}))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def service(nats_url: str) -> AsyncGenerator[MinimalService, None]:
    """Started MinimalService with guaranteed teardown after each test."""
    svc = MinimalService(nats_url=nats_url)
    await svc.start()
    try:
        yield svc
    finally:
        await svc.stop()


@pytest.fixture
async def probe_bus(nats_url: str) -> AsyncGenerator[EventBus, None]:
    """Independent EventBus used to publish / request from outside the service."""
    bus = EventBus(nats_url=nats_url, name=f"probe-{uuid.uuid4().hex[:8]}")
    await bus.connect()
    yield bus
    await bus.close()


# ---------------------------------------------------------------------------
# §2 BaseService lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lifecycle_order(nats_url: str) -> None:
    """_setup() is called after the bus connects; _cleanup() before bus closes."""
    call_log: list[str] = []

    class Instrumented(MinimalService):
        async def _setup(self) -> None:
            call_log.append("setup")
            assert self.event_bus is not None, "_setup called before bus connected"
            await super()._setup()

        async def _cleanup(self) -> None:
            call_log.append("cleanup")
            assert self.event_bus is not None, "_cleanup called after bus closed"
            await super()._cleanup()

    svc = Instrumented(nats_url=nats_url)
    await svc.start()
    assert call_log == ["setup"], f"unexpected call order before stop: {call_log}"
    await svc.stop()
    assert call_log == ["setup", "cleanup"], f"unexpected call order after stop: {call_log}"


@pytest.mark.asyncio
async def test_graceful_shutdown(nats_url: str) -> None:
    """shutdown() causes run() to exit cleanly; _cleanup() is always called."""
    svc = MinimalService(nats_url=nats_url)

    async def _run_then_shutdown():
        # Give run() a moment to reach _shutdown_event.wait(), then signal stop.
        await asyncio.sleep(0.05)
        svc.shutdown()

    await asyncio.gather(svc.run(), _run_then_shutdown())

    assert svc.setup_called
    assert svc.cleanup_called


@pytest.mark.asyncio
async def test_stop_always_calls_cleanup_on_setup_error(nats_url: str) -> None:
    """Even when _setup() raises, _cleanup() is still called by run()."""
    cleanup_called = False

    class FailingSetup(MinimalService):
        async def _setup(self) -> None:
            raise RuntimeError("setup failed")

        async def _cleanup(self) -> None:
            nonlocal cleanup_called
            cleanup_called = True

    svc = FailingSetup(nats_url=nats_url)
    with pytest.raises(RuntimeError, match="setup failed"):
        await svc.run()

    assert cleanup_called, "_cleanup was not called despite _setup raising"


@pytest.mark.asyncio
async def test_event_bus_none_before_start() -> None:
    """event_bus is None until start() is awaited — no broker needed."""
    svc = MinimalService(
        nats_url="nats://127.0.0.1:9999"
    )  # broker unreachable; start() never called
    assert svc.event_bus is None


# ---------------------------------------------------------------------------
# §3.5 Publish — fire-and-forget
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subscribe_fire_and_forget(
    service: MinimalService,
    probe_bus: EventBus,
    wait_for_message,
) -> None:
    """Normal pub/sub delivers the message dict to the handler."""
    payload = {"trace_id": str(uuid.uuid4()), "value": 42}
    await probe_bus.publish("test.topic", payload)

    await wait_for_message(lambda: len(service.received) > 0)
    assert service.received[-1]["value"] == 42


# ---------------------------------------------------------------------------
# §3.8 Request-reply
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_request_reply(
    service: MinimalService,
    probe_bus: EventBus,
) -> None:
    """request() receives the dict returned by the handler."""
    response = await probe_bus.request("test.status", {}, timeout=5.0)
    assert response.get("ok") is True
    assert response.get("service") == "test-service"


@pytest.mark.asyncio
async def test_request_reply_error_reply_on_handler_exception(
    nats_url: str,
    probe_bus: EventBus,
) -> None:
    """If the handler raises, the caller receives an error dict instead of hanging."""

    class ExplodingService(MinimalService):
        async def _setup(self) -> None:
            await self.event_bus.subscribe("test.boom", self._explode, is_request_handler=True)

        async def _explode(self, data: dict, msg) -> None:
            raise ValueError("handler exploded")

    svc = ExplodingService(nats_url=nats_url)
    await svc.start()
    try:
        response = await probe_bus.request("test.boom", {}, timeout=5.0)
        assert response.get("type") == "error"
        assert "exploded" in response.get("error", "")
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_request_reply_responds_only_when_reply_set(
    nats_url: str,
) -> None:
    """Handler guarded by 'if msg.reply' is safe to call as a plain subscriber."""
    # Publish (no reply inbox) — should not raise even though msg.reply is falsy.
    bus = EventBus(nats_url=nats_url, name="probe-no-reply")
    await bus.connect()
    svc = MinimalService(nats_url=nats_url)
    await svc.start()
    try:
        # Plain publish — no reply subject; handler must not crash.
        await bus.publish("test.status", {})
        await asyncio.sleep(0.1)  # give handler a moment to run
    finally:
        await svc.stop()
        await bus.close()


# ---------------------------------------------------------------------------
# §3.2 Reconnection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconnect_restores_subscriptions(
    service: MinimalService,
    probe_bus: EventBus,
    wait_for_message,
) -> None:
    """After force_reconnect(), previously registered handlers still fire."""
    await service.event_bus.force_reconnect()

    payload = {"trace_id": str(uuid.uuid4()), "value": 99}
    await probe_bus.publish("test.topic", payload)

    await wait_for_message(lambda: any(m.get("value") == 99 for m in service.received))


# ---------------------------------------------------------------------------
# §3.7 Message validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validation_drops_bad_messages(
    service: MinimalService,
    probe_bus: EventBus,
) -> None:
    """A message missing required fields is silently dropped; handler never fires."""
    import nats

    # Bypass the SDK EventBus and publish raw bytes that will fail validation.
    # 'test.topic' uses ObservationMessage which requires 'trace_id'.
    raw_nc = await nats.connect(probe_bus.nats_url)
    try:
        await raw_nc.publish("test.topic", b'{"bad_field": true}')
        await asyncio.sleep(0.2)  # give handler a moment
    finally:
        await raw_nc.drain()
        await raw_nc.close()

    # Handler should NOT have been called with the invalid message.
    assert not any(
        "bad_field" in m for m in service.received
    ), "Validation did not drop the malformed message"


@pytest.mark.asyncio
async def test_request_reply_validation_error_returns_error_dict(
    nats_url: str,
    probe_bus: EventBus,
) -> None:
    """A request with invalid payload causes an error reply, not a timeout."""
    from activelearning.messages import ActionProposalMessage

    # Register a handler with an explicit wire model so validation kicks in.
    # Deliberately NOT 'proposal.new': that subject is captured by the
    # SAFETY_CRITICAL JetStream stream, and a core-NATS request on a
    # stream-captured subject gets answered by the server's pub-ack
    # ({'stream': ..., 'seq': ...}) before the service can reply.
    class ProposalService(MinimalService):
        async def _setup(self) -> None:
            await self.event_bus.subscribe(
                "test.rpc.proposal",
                self._handle_proposal,
                is_request_handler=True,
                message_model=ActionProposalMessage,
            )

        async def _handle_proposal(self, data: dict, msg) -> None:
            if msg.reply:
                await msg.respond(serialize_message({"accepted": True}))

    svc = ProposalService(nats_url=nats_url)
    await svc.start()
    try:
        # Missing required 'trace_id' — validation fails.
        response = await probe_bus.request(
            "test.rpc.proposal", {"action": {}, "provenance": "test"}, timeout=5.0
        )
        assert response.get("type") == "error"
        assert response.get("error") == "validation_failed"
    finally:
        await svc.stop()


# ---------------------------------------------------------------------------
# §3.10 Decision waiting — fail closed on timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decision_wait_timeout_raises(nats_url: str) -> None:
    """wait_for_decision() raises asyncio.TimeoutError when no decision arrives."""
    bus = EventBus(nats_url=nats_url, name="decision-probe")
    await bus.connect()
    trace_id = str(uuid.uuid4())
    try:
        with pytest.raises(asyncio.TimeoutError):
            await bus.wait_for_decision(trace_id, timeout=0.2)
    finally:
        await bus.close()


@pytest.mark.asyncio
async def test_decision_wait_rejects_unsigned_when_signing_enabled(
    nats_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsigned decisions are rejected when ENGRAM_DECISION_KEY is set."""
    monkeypatch.setenv("ENGRAM_DECISION_KEY", "test-key-abc123")

    import activelearning.signing as signing_mod

    monkeypatch.setattr(signing_mod, "_warned_unsigned", False)

    bus = EventBus(nats_url=nats_url, name="decision-probe-signing")
    await bus.connect()
    trace_id = str(uuid.uuid4())

    async def _publish_unsigned():
        await asyncio.sleep(0.05)
        from activelearning.subjects import decision_subject

        unsigned = {
            "trace_id": trace_id,
            "type": "ALLOW",
            "reason": "forged",
            "risk_score": 0.0,
        }
        probe = EventBus(nats_url=nats_url, name="unsigned-probe")
        await probe.connect()
        try:
            await probe.publish(decision_subject(trace_id), unsigned)
        finally:
            await probe.close()

    asyncio.create_task(_publish_unsigned())

    with pytest.raises(asyncio.TimeoutError):
        await bus.wait_for_decision(trace_id, timeout=0.5)

    await bus.close()


@pytest.mark.asyncio
async def test_decision_wait_accepts_signed_decision(
    nats_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Signed decisions satisfy wait_for_decision() immediately."""
    key = "test-signing-key-xyz"
    monkeypatch.setenv("ENGRAM_DECISION_KEY", key)

    import activelearning.signing as signing_mod

    monkeypatch.setattr(signing_mod, "_warned_unsigned", False)

    bus = EventBus(nats_url=nats_url, name="decision-probe-signed")
    await bus.connect()
    trace_id = str(uuid.uuid4())

    async def _publish_signed():
        await asyncio.sleep(0.05)
        from activelearning.subjects import decision_subject

        payload = {
            "trace_id": trace_id,
            "type": "ALLOW",
            "reason": "approved",
            "risk_score": 0.1,
            "issued_at": current_timestamp(),
            # Must be a real future timestamp, not a small placeholder: issue
            # #190 made wait_for_decision() reject expired decisions, and a
            # fixed value like 9999999ms (1970-01-01) is long since past.
            "expires_at": current_timestamp() + 60_000,
        }
        signed = sign_decision(payload, key=key)
        probe = EventBus(nats_url=nats_url, name="signed-probe")
        await probe.connect()
        try:
            await probe.publish(decision_subject(trace_id), signed)
        finally:
            await probe.close()

    asyncio.create_task(_publish_signed())

    decision = await bus.wait_for_decision(trace_id, timeout=5.0)
    assert decision["type"] == "ALLOW"
    assert decision["trace_id"] == trace_id

    await bus.close()


# ---------------------------------------------------------------------------
# §3.3 _ensure_connected resilience
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_raises_when_not_connected() -> None:
    """publish() raises RuntimeError if the bus was never connected — no broker needed."""
    bus = EventBus(nats_url="nats://127.0.0.1:9999", name="never-connected")
    # Never call bus.connect() — should raise immediately on publish.
    with pytest.raises(RuntimeError, match="Not connected to NATS"):
        await bus.publish("test.topic", {"trace_id": "x", "provenance": "test", "data": {}})


# ---------------------------------------------------------------------------
# §6 Adoption checklist — entry-point
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_entry_point_via_run(nats_url: str) -> None:
    """Service.run() completes cleanly when shutdown() is called."""
    svc = MinimalService(nats_url=nats_url)

    async def _trigger_shutdown():
        await asyncio.sleep(0.05)
        svc.shutdown()

    await asyncio.gather(svc.run(), _trigger_shutdown())

    assert svc.setup_called
    assert svc.cleanup_called
    assert svc.event_bus is not None  # still references the (now-closed) bus
