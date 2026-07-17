"""Shared fixtures for launcher integration / chaos tests."""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest


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
