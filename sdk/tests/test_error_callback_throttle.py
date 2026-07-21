"""Tests for EventBus._error_callback's 15-second duplicate-error throttle (issue #251).

The callback keys on exception type + a 15s window so a sustained NATS outage
(e.g. repeated TimeoutError during reconnect) does not flood the console.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from activelearning.nats_client import EventBus

_LOGGER_NAME = "activelearning.nats_client"


@pytest.mark.asyncio
async def test_error_callback_throttles_duplicate_type_within_15s(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Identical exception types within 15s must log only once."""
    bus = EventBus(name="throttle-test")
    clock = {"now": 1000.0}

    def fake_monotonic() -> float:
        return clock["now"]

    with (
        patch("activelearning.nats_client.time.monotonic", side_effect=fake_monotonic),
        caplog.at_level(logging.ERROR, logger=_LOGGER_NAME),
    ):
        await bus._error_callback(TimeoutError("reconnect timed out"))
        clock["now"] += 1.0
        await bus._error_callback(TimeoutError("reconnect timed out again"))
        clock["now"] += 5.0
        await bus._error_callback(TimeoutError(""))

        records = [r for r in caplog.records if r.name == _LOGGER_NAME]
        assert len(records) == 1
        assert "TimeoutError" in records[0].message


@pytest.mark.asyncio
async def test_error_callback_logs_again_after_throttle_window(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """After 15s, the same exception type must be logged again."""
    bus = EventBus(name="throttle-test-window")
    clock = {"now": 2000.0}

    def fake_monotonic() -> float:
        return clock["now"]

    with (
        patch("activelearning.nats_client.time.monotonic", side_effect=fake_monotonic),
        caplog.at_level(logging.ERROR, logger=_LOGGER_NAME),
    ):
        await bus._error_callback(TimeoutError("first"))
        clock["now"] += 14.9
        await bus._error_callback(TimeoutError("still inside window"))
        clock["now"] += 0.2  # total +15.1s from first log
        await bus._error_callback(TimeoutError("window expired"))

        records = [r for r in caplog.records if r.name == _LOGGER_NAME]
        assert len(records) == 2
        assert all("TimeoutError" in r.message for r in records)


@pytest.mark.asyncio
async def test_error_callback_different_type_bypasses_throttle(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A different exception type must log immediately even inside the window."""
    bus = EventBus(name="throttle-test-types")
    clock = {"now": 3000.0}

    def fake_monotonic() -> float:
        return clock["now"]

    with (
        patch("activelearning.nats_client.time.monotonic", side_effect=fake_monotonic),
        caplog.at_level(logging.ERROR, logger=_LOGGER_NAME),
    ):
        await bus._error_callback(TimeoutError("outage"))
        clock["now"] += 1.0
        await bus._error_callback(ConnectionError("connection reset"))

        records = [r for r in caplog.records if r.name == _LOGGER_NAME]
        assert len(records) == 2
        assert "TimeoutError" in records[0].message
        assert "ConnectionError" in records[1].message


@pytest.mark.asyncio
async def test_error_callback_empty_str_uses_repr(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """nats-py errors that stringify empty must still be diagnosable via repr."""
    bus = EventBus(name="throttle-test-empty")

    with caplog.at_level(logging.ERROR, logger=_LOGGER_NAME):
        await bus._error_callback(TimeoutError())

    records = [r for r in caplog.records if r.name == _LOGGER_NAME]
    assert len(records) == 1
    assert "NATS error [TimeoutError]:" in records[0].message
    assert "TimeoutError()" in records[0].message
