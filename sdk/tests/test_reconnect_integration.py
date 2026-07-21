"""Integration test: force_reconnect()'s resubscribe path against a real
nats-server process kill + restart (issue #234).

Existing coverage before this file, none of which combines all three things
force_reconnect()'s own docstring says are unverified ("nats-py's built-in
reconnection sometimes fails silently"):

  - test_nats_client.py::test_force_reconnect_restores_handlers calls
    force_reconnect() against a LIVE, never-killed embedded broker -- a
    manual self-triggered reconnect, not a real outage.
  - test_robustness.py::TestForceReconnectRequestHandlers fully mocks
    EventBus._nc/connect/subscribe -- no real broker at all.
  - test_reconnect_backoff.py::test_multi_service_reconnects_are_not_lockstep
    DOES kill and restart a real nats-server process, but only checks that
    nats-py's own built-in _reconnected_callback fires with spread timing --
    it never calls force_reconnect() and never verifies a message actually
    reaches a handler afterward.

This file kills and restarts a real nats-server mid-session, calls
force_reconnect() specifically, and proves every subscribed service resumes
correctly by publishing a message after the restart and checking it arrives.
"""

from __future__ import annotations

import asyncio
import shutil
import socket
import subprocess
import uuid
from pathlib import Path

import pytest

from activelearning.nats_client import EventBus


def _free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


class _NatsServerHandle:
    """A real nats-server subprocess this test can kill and restart
    mid-session, on a fixed port + data dir so clients reconnect to the same
    address and JetStream state (the safety stream, durable consumers)
    survives the restart, same as it would for a real broker outage."""

    def __init__(self, host: str, port: int, data_dir: Path, binary: str):
        self.host = host
        self.port = port
        self.url = f"nats://{host}:{port}"
        self._data_dir = data_dir
        self._binary = binary
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        self.proc = subprocess.Popen(
            [self._binary, "-js", "-p", str(self.port), "-sd", str(self._data_dir)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    async def wait_up(self, timeout: float = 5.0) -> None:
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if _port_open(self.host, self.port):
                return
            if self.proc is not None and self.proc.poll() is not None:
                pytest.fail("nats-server exited before opening its port")
            await asyncio.sleep(0.1)
        pytest.fail("timed out waiting for nats-server to start")

    def kill(self, timeout: float = 5.0) -> None:
        assert self.proc is not None
        self.proc.terminate()
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=timeout)
        self.proc = None

    def stop_if_running(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.kill()


@pytest.fixture
def broker(tmp_path):
    binary = shutil.which("nats-server")
    if binary is None:
        pytest.skip("nats-server not on PATH")
    host = "127.0.0.1"
    port = _free_port(host)
    data_dir = tmp_path / "nats-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    handle = _NatsServerHandle(host, port, data_dir, binary)
    handle.start()
    yield handle
    handle.stop_if_running()


async def _wait_disconnected(bus: EventBus, timeout: float = 5.0) -> None:
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if bus._nc is not None and not bus.is_connected:
            return
        await asyncio.sleep(0.05)
    pytest.fail(f"{bus.name} never observed the broker going down")


def _make_handler(log: list[dict]):
    async def handler(data: dict) -> None:
        log.append(data)

    return handler


@pytest.mark.asyncio
async def test_force_reconnect_resumes_multiple_subscribed_services_after_broker_restart(broker):
    """Kill a real nats-server mid-session, restart it, call force_reconnect()
    on several independently-subscribed services, and verify every one of
    them actually receives a message published after the restart -- not just
    that force_reconnect() ran without raising."""
    await broker.wait_up()

    n_services = 3
    buses: list[EventBus] = []
    received: dict[str, list[dict]] = {}
    subjects: dict[str, str] = {}

    try:
        for i in range(n_services):
            name = f"reconnect-svc-{i}"
            subject = f"test.reconnect_integration.{uuid.uuid4().hex[:8]}"
            bus = EventBus(nats_url=broker.url, name=name)
            await bus.connect()

            received[name] = []
            subjects[name] = subject
            await bus.subscribe(subject, _make_handler(received[name]))
            buses.append(bus)

        broker.kill()
        for bus in buses:
            await _wait_disconnected(bus)

        broker.start()
        await broker.wait_up()

        for bus in buses:
            await bus.force_reconnect()
            assert bus.is_connected, f"{bus.name} did not reconnect"

        for bus in buses:
            await bus.publish(subjects[bus.name], {"resumed": True})

        loop = asyncio.get_event_loop()
        deadline = loop.time() + 5.0
        while loop.time() < deadline:
            if all(len(v) == 1 for v in received.values()):
                break
            await asyncio.sleep(0.05)

        for name, msgs in received.items():
            assert (
                len(msgs) == 1
            ), f"{name}'s subscription did not resume after force_reconnect(): got {msgs}"
            assert msgs[0]["resumed"] is True
    finally:
        for bus in buses:
            try:
                await bus.close()
            except Exception:
                pass


@pytest.mark.asyncio
async def test_force_reconnect_resumes_jetstream_durable_subscription_after_broker_restart(
    broker,
):
    """The JetStream branch of force_reconnect()'s resubscribe loop
    (saved_js / durable consumer) is a structurally different code path from
    the core-subscribe branch above and needs its own check."""
    await broker.wait_up()

    # Must match a SAFETY_STREAM_NAME-covered pattern (see _SAFETY_STREAM_SUBJECTS).
    subject = f"policy.reconnect_integration_{uuid.uuid4().hex[:8]}"
    received: list[dict] = []

    bus = EventBus(nats_url=broker.url, name="reconnect-js-svc")
    try:
        await bus.connect()
        await bus.js_subscribe(
            subject, _make_handler(received), durable="reconnect-integration-test"
        )

        broker.kill()
        await _wait_disconnected(bus)

        broker.start()
        await broker.wait_up()

        await bus.force_reconnect()
        assert bus.is_connected

        await bus.publish(subject, {"resumed": True})

        loop = asyncio.get_event_loop()
        deadline = loop.time() + 5.0
        while loop.time() < deadline:
            if received:
                break
            await asyncio.sleep(0.05)

        assert len(received) == 1
        assert received[0]["resumed"] is True
    finally:
        await bus.close()
