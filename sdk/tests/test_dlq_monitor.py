"""Tests for dead-letter queue consumption and alerting (issue #246)."""

from __future__ import annotations

import json
import logging

import pytest

from activelearning.dlq_monitor import (
    DLQ_ALERT_EVENT,
    DlqRecord,
    emit_dlq_alert,
    parse_dlq_envelope,
)
from activelearning.nats_client import EventBus, Subjects, poison_subject


class TestParseDlqEnvelope:
    def test_parses_full_envelope(self) -> None:
        data = {
            "original_subject": "decision.abc123",
            "reason": "max_deliver_exhausted(5): boom",
            "num_delivered": 5,
            "payload": '{"trace_id": "abc123"}',
        }
        record = parse_dlq_envelope("dlq.decision.abc123", data)
        assert record == DlqRecord(
            dlq_subject="dlq.decision.abc123",
            original_subject="decision.abc123",
            reason="max_deliver_exhausted(5): boom",
            num_delivered=5,
            payload='{"trace_id": "abc123"}',
        )

    def test_tolerates_missing_fields(self) -> None:
        record = parse_dlq_envelope("dlq.proposal.new", {})
        assert record.original_subject == "unknown"
        assert record.reason == "unknown"
        assert record.num_delivered == 0
        assert record.payload == ""

    def test_tolerates_malformed_num_delivered(self) -> None:
        record = parse_dlq_envelope("dlq.proposal.new", {"num_delivered": "not-a-number"})
        assert record.num_delivered == 0


class TestEmitDlqAlert:
    def test_emits_structured_json_error_log(self, caplog: pytest.LogCaptureFixture) -> None:
        record = DlqRecord(
            dlq_subject="dlq.code.proposal",
            original_subject="code.proposal",
            reason="validation_error: bad payload",
            num_delivered=1,
            payload="{}",
        )
        with caplog.at_level(logging.ERROR):
            payload = emit_dlq_alert(record, monitor="kernel")

        assert payload["event"] == DLQ_ALERT_EVENT
        assert payload["original_subject"] == "code.proposal"
        assert payload["monitor"] == "kernel"
        assert len(caplog.records) == 1
        assert json.loads(caplog.records[0].message) == payload


class TestEventBusDlqMonitor:
    @pytest.mark.asyncio
    async def test_handle_dlq_message_emits_alert(self, caplog: pytest.LogCaptureFixture) -> None:
        bus = EventBus(name="kernel")
        envelope = {
            "original_subject": "proposal.new",
            "reason": "max_deliver_exhausted(5): always fails",
            "num_delivered": 5,
            "payload": '{"trace_id": "poison-1"}',
        }
        with caplog.at_level(logging.ERROR):
            await bus._handle_dlq_message(envelope)

        alerts = [r for r in caplog.records if r.name == "activelearning.dlq_monitor"]
        assert len(alerts) == 1
        payload = json.loads(alerts[0].message)
        assert payload["dlq_subject"] == poison_subject("proposal.new")
        assert payload["original_subject"] == "proposal.new"
        assert payload["num_delivered"] == 5
        assert payload["monitor"] == "kernel"

    @pytest.mark.asyncio
    async def test_start_dlq_monitor_subscribes_to_wildcard(self) -> None:
        bus = EventBus(name="kernel")
        subscribed: list[tuple] = []

        async def fake_subscribe(subject, handler, **kwargs):
            subscribed.append((subject, handler))

        bus.subscribe = fake_subscribe  # type: ignore[method-assign]
        await bus.start_dlq_monitor()

        assert len(subscribed) == 1
        assert subscribed[0][0] == Subjects.DLQ_WILDCARD
        assert subscribed[0][1] == bus._handle_dlq_message


@pytest.mark.asyncio
async def test_route_to_poison_is_observable_via_dlq_monitor(
    event_bus: EventBus, wait_for_message, caplog: pytest.LogCaptureFixture
) -> None:
    """End-to-end: a message poisoned by js_subscribe is consumed and alerted
    on by start_dlq_monitor() — the exact gap issue #246 reports."""
    import uuid

    subject = f"decision.{uuid.uuid4().hex}"
    # NATS requires max_deliver > len(backoff); mirrors test_poison_after_max_deliver.
    max_deliver = 3
    attempts: list[int] = []

    async def handler(_data: dict) -> None:
        attempts.append(1)
        raise RuntimeError("always fails")

    def alerts() -> list[logging.LogRecord]:
        return [r for r in caplog.records if r.name == "activelearning.dlq_monitor"]

    with caplog.at_level(logging.ERROR, logger="activelearning.dlq_monitor"):
        await event_bus.start_dlq_monitor()
        await event_bus.js_subscribe(
            subject,
            handler,
            durable=f"d-{uuid.uuid4().hex[:8]}",
            max_deliver=max_deliver,
            backoff=[0.3, 0.3],
        )
        await event_bus.publish(subject, {"trace_id": "e2e-poison", "type": "ALLOW"})

        await wait_for_message(lambda: len(alerts()) >= 1, timeout=8.0)

    assert len(attempts) == max_deliver
    payload = json.loads(alerts()[0].message)
    assert payload["original_subject"] == subject
    assert payload["num_delivered"] == max_deliver
