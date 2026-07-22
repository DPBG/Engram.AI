"""Regression tests: undecodable JetStream messages must be poisoned, not dropped.

The JetStream (safety-critical) consumer callbacks parse a message with
``deserialize_message()`` before validating it against a wire model. That parse
step can raise ``json.JSONDecodeError`` (malformed JSON) or ``UnicodeDecodeError``
(non-UTF-8 bytes) — neither of which is a ``MessageValidationError``. The parse
blocks previously caught *only* ``MessageValidationError``, so a corrupt message
escaped the poison/term path entirely:

- ``js_subscribe``: ``term()`` and dead-lettering were skipped, so the message
  was never routed to ``dlq.<subject>`` and never acked — it redelivered up to
  ``max_deliver`` and then stuck, silently, on a stream whose whole point is that
  messages are "never silently dropped".
- ``wait_for_decision``: the un-acked garbage redelivered to the ephemeral waiter
  until timeout.

These tests capture the real subscription callbacks and feed them corrupt bytes,
asserting the callback completes (does not propagate) and takes the intended
poison / drop action.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from activelearning.nats_client import EventBus, poison_subject


def _corrupt_msg(data: bytes) -> MagicMock:
    """A fake JetStream Msg whose ack/nak/term are awaitable spies."""
    msg = MagicMock()
    msg.data = data
    msg.metadata.num_delivered = 1
    msg.ack = AsyncMock()
    msg.nak = AsyncMock()
    msg.term = AsyncMock()
    return msg


async def _capture_js_subscribe_cb(bus: EventBus) -> MagicMock:
    """Register a js_subscribe consumer and return the callback the server calls."""
    bus._js = MagicMock()
    bus._js.subscribe = AsyncMock()
    bus._ensure_connected = AsyncMock()  # type: ignore[method-assign]
    bus._nc = MagicMock()
    bus._nc.publish = AsyncMock()

    async def handler(_data: dict) -> None:  # pragma: no cover - never reached here
        raise AssertionError("handler must not run for an undecodable message")

    await bus.js_subscribe("proposal.new", handler, durable="d-test")
    return bus._js.subscribe.await_args.kwargs["cb"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        b"not valid json {{{",  # malformed JSON -> json.JSONDecodeError
        b"\xff\xfe\x00bad-bytes",  # non-UTF-8 -> UnicodeDecodeError
    ],
    ids=["malformed-json", "non-utf8"],
)
async def test_js_subscribe_poisons_undecodable_message(payload: bytes) -> None:
    bus = EventBus(name="js-decode-poison")
    cb = await _capture_js_subscribe_cb(bus)
    msg = _corrupt_msg(payload)

    # Must not propagate the decode error out of the consumer callback.
    await cb(msg)

    # Dead-lettered to dlq.proposal.new and term()-ed so it stops redelivering.
    bus._nc.publish.assert_awaited_once()
    assert bus._nc.publish.await_args.args[0] == poison_subject("proposal.new")
    msg.term.assert_awaited_once()
    # It must NOT be acked or nak'd (that path is for valid / retryable messages).
    msg.ack.assert_not_awaited()
    msg.nak.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [b"not valid json {{{", b"\xff\xfe\x00bad-bytes"],
    ids=["malformed-json", "non-utf8"],
)
async def test_wait_for_decision_drops_undecodable_message(payload: bytes) -> None:
    bus = EventBus(name="waiter-decode-drop")
    bus._js = MagicMock()
    sub = MagicMock()
    sub.unsubscribe = AsyncMock()  # wait_for_decision awaits this in its finally
    bus._js.subscribe = AsyncMock(return_value=sub)
    bus._ensure_connected = AsyncMock()  # type: ignore[method-assign]

    # wait_for_decision blocks until a decision or timeout; run it with a short
    # timeout in the background just long enough to capture and drive its callback.
    import asyncio

    task = asyncio.create_task(bus.wait_for_decision("trace-xyz", timeout=0.2))
    await asyncio.sleep(0)  # let the subscribe call run
    cb = bus._js.subscribe.await_args.kwargs["cb"]

    msg = _corrupt_msg(payload)
    # Must not propagate; a bad decode is acked (dropped) so it stops redelivering.
    await cb(msg)
    msg.ack.assert_awaited_once()

    # The corrupt message must not have satisfied the wait.
    with pytest.raises(TimeoutError):
        await task
