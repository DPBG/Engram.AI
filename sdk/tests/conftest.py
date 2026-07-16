"""Shared fixtures for SDK NATS integration tests."""

from __future__ import annotations

import asyncio
import os
import shutil
import uuid
from collections.abc import AsyncGenerator, Callable, Coroutine, Generator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest

from activelearning.nats_client import EventBus
from activelearning.testing.nats_server import port_open, run_nats_server


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
        if not port_open(host, port):
            pytest.skip(f"NATS_URL set but broker not reachable at {explicit}")
        yield explicit
        return

    binary = shutil.which("nats-server")
    if binary is None:
        pytest.skip("NATS server not available and nats-server not on PATH")
        return

    data_dir = Path("/tmp") / f"engram-nats-test-{uuid.uuid4().hex}"
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        with run_nats_server(data_dir) as server:
            yield server.url
    except RuntimeError as exc:
        pytest.skip(str(exc))
    finally:
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
