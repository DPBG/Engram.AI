"""Smoke tests for per-service NATS credentials provisioning (issue #94).

These tests verify the credential wiring end-to-end without requiring a
real NATS server or nsc tool:

- ServiceConfig reads NATS_CREDS from the environment.
- EventBus exposes nats_creds and resolves from env when not passed directly.
- connect() falls back gracefully when the creds file is absent (dev mode).
- connect() passes user_credentials when the creds file exists.
- BaseService threads the credential path from config through to EventBus.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from activelearning.config import ServiceConfig
from activelearning.nats_client import EventBus

# ---------------------------------------------------------------------------
# ServiceConfig
# ---------------------------------------------------------------------------


class TestServiceConfigCredsLoading:
    def test_from_env_reads_nats_creds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NATS_CREDS", "/run/secrets/kernel.creds")
        cfg = ServiceConfig.from_env("kernel")
        assert cfg.nats_creds == "/run/secrets/kernel.creds"

    def test_from_env_nats_creds_defaults_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NATS_CREDS", raising=False)
        cfg = ServiceConfig.from_env("kernel")
        assert cfg.nats_creds is None

    def test_from_env_empty_string_treated_as_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # An empty string (e.g. NATS_CREDS= in docker-compose before gen-creds
        # is run) must not be forwarded as a creds path.
        monkeypatch.setenv("NATS_CREDS", "")
        cfg = ServiceConfig.from_env("planner")
        assert cfg.nats_creds is None


# ---------------------------------------------------------------------------
# EventBus construction
# ---------------------------------------------------------------------------


class TestEventBusCredsInit:
    def test_explicit_creds_stored(self) -> None:
        bus = EventBus(nats_creds="/secrets/kernel.creds")
        assert bus.nats_creds == "/secrets/kernel.creds"

    def test_creds_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NATS_CREDS", "/run/secrets/planner.creds")
        bus = EventBus()
        assert bus.nats_creds == "/run/secrets/planner.creds"

    def test_explicit_creds_take_precedence_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NATS_CREDS", "/from/env.creds")
        bus = EventBus(nats_creds="/explicit/path.creds")
        assert bus.nats_creds == "/explicit/path.creds"

    def test_no_creds_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NATS_CREDS", raising=False)
        bus = EventBus()
        assert bus.nats_creds is None

    def test_empty_env_treated_as_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NATS_CREDS", "")
        bus = EventBus()
        assert bus.nats_creds is None


# ---------------------------------------------------------------------------
# EventBus.connect() — credential passing to nats.connect()
# ---------------------------------------------------------------------------


class TestEventBusConnect:
    """Unit-tests that mock nats.connect to verify the correct kwargs are passed."""

    def _make_mock_nc(self) -> AsyncMock:
        nc = AsyncMock()
        nc.is_connected = True
        mock_js = AsyncMock()
        mock_js.add_stream = AsyncMock()
        nc.jetstream = MagicMock(return_value=mock_js)
        return nc

    @pytest.mark.asyncio
    async def test_connect_passes_user_credentials_when_file_exists(self, tmp_path: Path) -> None:
        creds_file = tmp_path / "kernel.creds"
        creds_file.write_text("--- fake creds ---")

        mock_nc = self._make_mock_nc()

        with patch(
            "activelearning.nats_client.nats.connect", new=AsyncMock(return_value=mock_nc)
        ) as mock_connect:
            bus = EventBus(nats_creds=str(creds_file))
            await bus.connect()
            call_kwargs = mock_connect.call_args[1]
            assert call_kwargs.get("user_credentials") == str(creds_file)
            await bus.close()

    @pytest.mark.asyncio
    async def test_connect_omits_user_credentials_when_file_missing(self, tmp_path: Path) -> None:
        absent = str(tmp_path / "nonexistent.creds")
        mock_nc = self._make_mock_nc()

        with patch(
            "activelearning.nats_client.nats.connect", new=AsyncMock(return_value=mock_nc)
        ) as mock_connect:
            bus = EventBus(nats_creds=absent)
            await bus.connect()
            call_kwargs = mock_connect.call_args[1]
            assert "user_credentials" not in call_kwargs, (
                "user_credentials must not be passed when the creds file is absent "
                "(dev fallback — avoids crashing before gen-creds.sh has been run)"
            )
            await bus.close()

    @pytest.mark.asyncio
    async def test_connect_without_creds_is_unauthenticated(self) -> None:
        mock_nc = self._make_mock_nc()

        with patch(
            "activelearning.nats_client.nats.connect", new=AsyncMock(return_value=mock_nc)
        ) as mock_connect:
            bus = EventBus()
            bus.nats_creds = None
            await bus.connect()
            call_kwargs = mock_connect.call_args[1]
            assert "user_credentials" not in call_kwargs
            await bus.close()


# ---------------------------------------------------------------------------
# BaseService credential threading
# ---------------------------------------------------------------------------


class TestBaseServiceCredsThreading:
    """Verify BaseService forwards config.nats_creds to the EventBus."""

    @pytest.mark.asyncio
    async def test_base_service_passes_nats_creds_to_event_bus(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        creds_file = tmp_path / "planner.creds"
        creds_file.write_text("--- fake creds ---")
        monkeypatch.setenv("NATS_CREDS", str(creds_file))
        monkeypatch.setenv("NATS_URL", "nats://localhost:4222")

        from activelearning.base_service import BaseService

        captured: dict[str, object] = {}
        original_event_bus_init = EventBus.__init__

        def capturing_init(self_bus: EventBus, **kwargs: object) -> None:
            original_event_bus_init(self_bus, **kwargs)
            captured["nats_creds"] = self_bus.nats_creds

        with patch.object(EventBus, "__init__", capturing_init):
            with patch.object(EventBus, "connect", new=AsyncMock()):
                with patch("activelearning.base_service.get_database", new=AsyncMock()):
                    svc = BaseService("planner", use_database=False)
                    await svc.start()

        assert captured.get("nats_creds") == str(creds_file), (
            "BaseService must forward config.nats_creds to EventBus so that "
            "the service authenticates with its per-service NATS identity."
        )

    def test_config_nats_creds_is_populated_from_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        creds_file = tmp_path / "kernel.creds"
        creds_file.write_text("--- fake ---")
        monkeypatch.setenv("NATS_CREDS", str(creds_file))

        cfg = ServiceConfig.from_env("kernel")
        assert cfg.nats_creds == str(creds_file)
