"""Shared fixtures for kernel integration tests."""

from __future__ import annotations

import asyncio
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest
from activelearning.database import close_database
from activelearning.nats_client import close_event_bus

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _reset_global_singletons():
    """Guarantee _db/_global_bus never leak between tests (issue #248).

    kernel/tests is the one place in the repo that drives a real
    BaseService.start() (KernelService in test_governance_smoke.py), which
    opens the real get_database() singleton. A left-open singleton hangs
    interpreter shutdown (non-daemon aiosqlite thread) or leaks state into
    whatever kernel test runs next, so reset unconditionally after every test
    rather than relying on each test to remember — mirrors sdk/tests/conftest.py.

    A plain sync fixture using asyncio.run(), not `async def` + pytest-asyncio:
    CI runs kernel/tests in a shared job alongside other services' test dirs
    with only `--with pytest` (no pytest-asyncio) — kernel's own test
    functions are plain `def test_...()` that call asyncio.run() internally
    for exactly this reason, so an async autouse fixture here breaks every
    test in the directory with "requested an async fixture ... with no
    plugin or hook that handled it."
    """
    yield
    asyncio.run(_close_singletons())


async def _close_singletons() -> None:
    await close_database()
    await close_event_bus()


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def _free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


@pytest.fixture
def nats_server(tmp_path: Path):
    """A real, isolated, JetStream-enabled nats-server on an ephemeral port."""
    binary = shutil.which("nats-server")
    if binary is None:
        pytest.skip("nats-server not on PATH")
    host = "127.0.0.1"
    port = _free_port(host)
    proc = subprocess.Popen(
        [binary, "-js", "-p", str(port), "-m", "8222", "-sd", str(tmp_path / "nats-data")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(50):
            if _port_open(host, port):
                break
            if proc.poll() is not None:
                pytest.skip("Failed to start embedded nats-server")
            time.sleep(0.1)
        else:
            pytest.skip("Timed out waiting for embedded nats-server")
        yield f"nats://{host}:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
