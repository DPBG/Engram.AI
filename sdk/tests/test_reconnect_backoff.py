"""Tests for reconnect-storm backoff (M2.3)."""

from __future__ import annotations

import asyncio
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from activelearning.nats_client import EventBus, jittered_reconnect_wait


class TestJitteredReconnectWait:
    def test_returns_base_when_jitter_zero(self):
        assert jittered_reconnect_wait(base_wait_s=2.0, jitter_s=0.0) == 2.0

    def test_spreads_within_base_plus_jitter(self, monkeypatch):
        monkeypatch.setenv("NATS_RECONNECT_BASE_WAIT_S", "2")
        monkeypatch.setenv("NATS_RECONNECT_JITTER_S", "3")
        waits = [jittered_reconnect_wait() for _ in range(50)]
        assert all(2.0 <= w <= 5.0 for w in waits)
        assert len(set(waits)) > 1

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("NATS_RECONNECT_BASE_WAIT_S", "1.5")
        monkeypatch.setenv("NATS_RECONNECT_JITTER_S", "0.5")
        wait = jittered_reconnect_wait()
        assert 1.5 <= wait <= 2.0


@pytest.mark.asyncio
async def test_connect_passes_jittered_reconnect_time_wait(monkeypatch):
    """Each EventBus must pass a distinct reconnect_time_wait to nats.connect."""
    monkeypatch.setenv("NATS_RECONNECT_BASE_WAIT_S", "2")
    monkeypatch.setenv("NATS_RECONNECT_JITTER_S", "4")

    captured: list[float] = []
    mock_js = MagicMock()
    mock_js.add_stream = AsyncMock()
    mock_nc = MagicMock()
    mock_nc.is_connected = True
    mock_nc.jetstream.return_value = mock_js

    async def fake_connect(_url: str, **kwargs):
        captured.append(kwargs["reconnect_time_wait"])
        return mock_nc

    with patch("activelearning.nats_client.nats.connect", side_effect=fake_connect):
        for i in range(8):
            bus = EventBus(name=f"backoff-test-{i}")
            await bus.connect()

    assert len(captured) == 8
    assert all(2.0 <= w <= 6.0 for w in captured)
    assert len(set(captured)) > 1


def _free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


@pytest.mark.asyncio
async def test_multi_service_reconnects_are_not_lockstep(monkeypatch):
    """After a broker restart, jittered waits spread reconnect attempts."""
    binary = shutil.which("nats-server")
    if binary is None:
        pytest.skip("nats-server not on PATH")

    # Short waits keep the integration test fast while still proving desync.
    monkeypatch.setenv("NATS_RECONNECT_BASE_WAIT_S", "0.2")
    monkeypatch.setenv("NATS_RECONNECT_JITTER_S", "1.0")

    host = "127.0.0.1"
    port = _free_port(host)
    url = f"nats://{host}:{port}"
    data_dir = Path("/tmp") / f"engram-reconnect-storm-{uuid.uuid4().hex}"
    data_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        [binary, "-js", "-p", str(port), "-sd", str(data_dir)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    buses: list[EventBus] = []
    reconnect_times: list[float] = []
    lock = asyncio.Lock()

    async def record_reconnect(bus: EventBus) -> None:
        original = bus._reconnected_callback

        async def wrapped() -> None:
            await original()
            async with lock:
                reconnect_times.append(time.monotonic())

        bus._reconnected_callback = wrapped

    try:
        for i in range(6):
            bus = EventBus(nats_url=url, name=f"storm-test-{i}")
            await record_reconnect(bus)
            await bus.connect()
            buses.append(bus)

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if all(bus._nc is not None and not bus.is_connected for bus in buses):
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("clients did not observe broker shutdown")

        reconnect_times.clear()
        proc = subprocess.Popen(
            [binary, "-js", "-p", str(port), "-sd", str(data_dir)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(50):
            if _port_open(host, port):
                break
            if proc.poll() is not None:
                pytest.fail("failed to restart embedded nats-server")
            await asyncio.sleep(0.1)
        else:
            pytest.fail("timed out waiting for broker restart")

        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if len(reconnect_times) == len(buses):
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail(
                f"only {len(reconnect_times)}/{len(buses)} clients reconnected "
                f"within timeout"
            )

        spread = max(reconnect_times) - min(reconnect_times)
        # With 0.2–1.2 s per-client waits, six clients should not all land in a
        # tight window (lockstep default would be ~0 s spread after the shared wait).
        assert spread >= 0.25, (
            f"reconnects clustered in lockstep (spread={spread:.3f}s): "
            f"{[round(t, 3) for t in reconnect_times]}"
        )
    finally:
        for bus in buses:
            try:
                await bus.close()
            except Exception:
                pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        shutil.rmtree(data_dir, ignore_errors=True)
