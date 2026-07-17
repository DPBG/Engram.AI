"""Unit tests for the centralized EventBus timeout policy (issue #233).

These are broker-free: they exercise the env-parsing helper and the module-level
policy constants, not live NATS request-reply.
"""

import importlib

import activelearning.nats_client as nats_client
from activelearning.nats_client import _env_timeout


class TestEnvTimeout:
    def test_returns_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("ENGRAM_TEST_TIMEOUT_S", raising=False)
        assert _env_timeout("ENGRAM_TEST_TIMEOUT_S", 30.0) == 30.0

    def test_parses_valid_override(self, monkeypatch):
        monkeypatch.setenv("ENGRAM_TEST_TIMEOUT_S", "45")
        assert _env_timeout("ENGRAM_TEST_TIMEOUT_S", 30.0) == 45.0

    def test_non_numeric_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("ENGRAM_TEST_TIMEOUT_S", "not-a-number")
        assert _env_timeout("ENGRAM_TEST_TIMEOUT_S", 30.0) == 30.0

    def test_zero_falls_back_to_default(self, monkeypatch):
        # A zero timeout would let a call hang forever — must be rejected.
        monkeypatch.setenv("ENGRAM_TEST_TIMEOUT_S", "0")
        assert _env_timeout("ENGRAM_TEST_TIMEOUT_S", 30.0) == 30.0

    def test_negative_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("ENGRAM_TEST_TIMEOUT_S", "-5")
        assert _env_timeout("ENGRAM_TEST_TIMEOUT_S", 30.0) == 30.0


class TestPolicyConstants:
    def test_defaults_are_positive_floats(self):
        for value in (
            nats_client.DEFAULT_REQUEST_TIMEOUT_S,
            nats_client.DEFAULT_DECISION_TIMEOUT_S,
            nats_client.RECONNECT_WAIT_TIMEOUT_S,
            nats_client.CONNECTION_DRAIN_TIMEOUT_S,
        ):
            assert isinstance(value, float)
            assert value > 0

    def test_documented_default_values(self):
        assert nats_client.DEFAULT_REQUEST_TIMEOUT_S == 30.0
        assert nats_client.DEFAULT_DECISION_TIMEOUT_S == 30.0
        assert nats_client.RECONNECT_WAIT_TIMEOUT_S == 10.0
        assert nats_client.CONNECTION_DRAIN_TIMEOUT_S == 5.0

    def test_env_override_applied_on_reload(self, monkeypatch):
        # Constants are resolved at import time; a deployment-wide override takes
        # effect on (re)import of the module.
        monkeypatch.setenv("ENGRAM_REQUEST_TIMEOUT_S", "42")
        try:
            reloaded = importlib.reload(nats_client)
            assert reloaded.DEFAULT_REQUEST_TIMEOUT_S == 42.0
        finally:
            monkeypatch.delenv("ENGRAM_REQUEST_TIMEOUT_S", raising=False)
            importlib.reload(nats_client)

    def test_exported_from_package(self):
        import activelearning

        assert hasattr(activelearning, "DEFAULT_REQUEST_TIMEOUT_S")
        assert hasattr(activelearning, "DEFAULT_DECISION_TIMEOUT_S")
        assert hasattr(activelearning, "RECONNECT_WAIT_TIMEOUT_S")
        assert hasattr(activelearning, "CONNECTION_DRAIN_TIMEOUT_S")
