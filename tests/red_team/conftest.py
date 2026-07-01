"""Fixtures for NATS broker authz red-team tests."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import time
from collections.abc import Generator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AUTHZ_CONF = REPO_ROOT / "deploy" / "nats-authz-test.conf"


def _free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


@pytest.fixture(scope="session")
def authz_nats_url() -> Generator[str, None, None]:
    """Start nats-server with enforced per-user publish allowlists."""

    def _skip_or_fail(msg: str) -> None:
        if os.getenv("CI"):
            pytest.fail(msg)
        pytest.skip(msg)

    binary = shutil.which("nats-server")
    if binary is None:
        _skip_or_fail("nats-server not on PATH")

    if not AUTHZ_CONF.is_file():
        _skip_or_fail(f"authz test config missing: {AUTHZ_CONF}")

    host = "127.0.0.1"
    port = _free_port(host)
    monitor_port = _free_port(host)
    data_dir = Path(tempfile.mkdtemp(prefix="engram-nats-authz-"))

    proc = subprocess.Popen(
        [
            binary,
            "-c",
            str(AUTHZ_CONF),
            "-a",
            host,
            "-p",
            str(port),
            "-m",
            str(monitor_port),
            "-sd",
            str(data_dir / "jetstream"),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(50):
            if _port_open(host, port):
                break
            if proc.poll() is not None:
                _skip_or_fail("Failed to start authz nats-server")
            time.sleep(0.1)
        else:
            _skip_or_fail("Timed out waiting for authz nats-server")
        yield f"nats://{host}:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(data_dir, ignore_errors=True)
