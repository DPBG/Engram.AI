"""Tests for supervisor flap alerting (issue #261)."""

from __future__ import annotations

import json
import logging

import pytest

from launcher.supervisor_alerts import (
    SERVICE_FLAP_EVENT,
    emit_service_flap_alert,
    flap_threshold,
    flap_window_seconds,
    record_restart,
    should_emit_flap_alert,
)


class TestFlapDetectionHelpers:
    def test_record_restart_prunes_old_timestamps(self) -> None:
        times: list[float] = [10.0, 20.0]
        kept = record_restart(times, now=75.0, window_seconds=60.0)
        assert kept == [20.0, 75.0]
        assert times == [20.0, 75.0]

    def test_should_emit_at_threshold(self) -> None:
        assert should_emit_flap_alert(3, threshold=3, alerts_emitted=0) is True
        assert should_emit_flap_alert(2, threshold=3, alerts_emitted=0) is False

    def test_should_emit_again_at_next_multiple(self) -> None:
        assert should_emit_flap_alert(6, threshold=3, alerts_emitted=1) is True
        assert should_emit_flap_alert(5, threshold=3, alerts_emitted=1) is False
        assert should_emit_flap_alert(6, threshold=3, alerts_emitted=2) is False

    def test_emit_service_flap_alert_json(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="launcher.supervisor"):
            payload = emit_service_flap_alert(
                service_name="kernel",
                restart_count=3,
                restarts_in_window=3,
                window_seconds=60.0,
                threshold=3,
                exit_code=1,
                uptime_seconds=0.02,
            )

        assert payload["event"] == SERVICE_FLAP_EVENT
        assert payload["service"] == "kernel"
        assert payload["restarts_in_window"] == 3
        assert len(caplog.records) == 1
        assert json.loads(caplog.records[0].message) == payload

    def test_flap_threshold_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SUPERVISOR_FLAP_THRESHOLD", "5")
        assert flap_threshold() == 5

    def test_flap_window_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SUPERVISOR_FLAP_WINDOW_S", "120")
        assert flap_window_seconds() == 120.0

    def test_invalid_env_falls_back_to_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SUPERVISOR_FLAP_THRESHOLD", "not-a-number")
        monkeypatch.setenv("SUPERVISOR_FLAP_WINDOW_S", "oops")
        assert flap_threshold() == 3
        assert flap_window_seconds() == 60.0
