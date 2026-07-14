"""Shared fixtures for SDK NATS integration tests."""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
import uuid
from collections.abc import AsyncGenerator, Callable, Coroutine, Generator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest

import activelearning.database as _database_module
import activelearning.nats_client as _nats_client_module
from activelearning.nats_client import EventBus


@pytest.fixture(autouse=True)
async def _reset_global_singletons() -> AsyncGenerator[None, None]:
    """Guarantee each test starts and ends with no live global singleton.

    ``activelearning.database._db`` and ``activelearning.nats_client._global_bus``
    are lazily-created module-level singletons with no built-in reset (issue
    #248). A test that exercises either one for real — e.g. by driving a
    ``BaseService`` subclass through its actual ``start()``/``stop()``
    lifecycle instead of mocking ``get_database``/``get_event_bus`` — leaves it
    connected for every later test in the same pytest process to inherit,
    regardless of collection order. This autouse fixture closes and resets
    both after every test in this suite so that class of leak cannot occur
    here even if a future test forgets to clean up after itself.
    """
    yield
    await _database_module.close_database()
    await _nats_client_module.close_event_bus()


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


async def _can_connect(nats_url: str) -> bool:
    try:
        import nats

        # allow_reconnect=False: fail in ~1s when the broker is unreachable.
        # With reconnects on, nats-py cycles retries for minutes, which turns a
        # "server not available" skip into a per-test timeout burn.
        nc = await nats.connect(nats_url, connect_timeout=1, allow_reconnect=False)
        await nc.close()
        return True
    except Exception:
        return False


def _free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="session")
def nats_url() -> Generator[str, None, None]:
    """Broker URL for integration tests.

    Honors ``NATS_URL`` when set. Otherwise starts an isolated embedded
    nats-server on an ephemeral port so tests do not share JetStream state
    with another local broker on :4222.
    """
    explicit = os.environ.get("NATS_URL")
    if explicit:
        parsed = urlparse(explicit)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 4222
        if not _port_open(host, port):
            pytest.skip(f"NATS_URL set but broker not reachable at {explicit}")
        yield explicit
        return

    binary = shutil.which("nats-server")
    if binary is None:
        pytest.skip("NATS server not available and nats-server not on PATH")
        return

    host = "127.0.0.1"
    port = _free_port(host)
    url = f"nats://{host}:{port}"

    data_dir = Path("/tmp") / f"engram-nats-test-{uuid.uuid4().hex}"
    data_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [binary, "-js", "-p", str(port), "-m", "8222", "-sd", str(data_dir)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(50):
            if _port_open(host, port):
                break
            if proc.poll() is not None:
                pytest.skip("Failed to start embedded nats-server")
            import time

            time.sleep(0.1)
        else:
            pytest.skip("Timed out waiting for embedded nats-server")
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture
async def event_bus(nats_url: str) -> AsyncGenerator[EventBus, None]:
    if not await _can_connect(nats_url):
        pytest.skip("NATS server not available")

    bus = EventBus(nats_url=nats_url, name=f"sdk-test-{uuid.uuid4().hex[:8]}")
    await bus.connect()
    yield bus
    await bus.close()


@pytest.fixture
def wait_for_message() -> Callable[
    [Callable[[], bool], float, float],
    Coroutine[Any, Any, None],
]:
    """Poll until predicate is true or timeout."""

    async def _wait(
        predicate: Callable[[], bool],
        timeout: float = 2.0,
        interval: float = 0.05,
    ) -> None:
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if predicate():
                return
            await asyncio.sleep(interval)
        raise AssertionError("timed out waiting for condition")

    return _wait
