"""Shared real NATS broker harness for integration tests.

The M1 governance smoke tests and M2 NATS-reliability tests both need an
isolated, JetStream-enabled broker. Keeping the process management here avoids
each suite growing its own subtly different fixture.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NatsServer:
    """Running broker details returned by the test harness."""

    url: str
    host: str
    port: int
    store_dir: Path


def nats_server_available() -> bool:
    """Return whether ``nats-server`` is available on PATH."""

    return shutil.which("nats-server") is not None


def free_port(host: str = "127.0.0.1") -> int:
    """Return an unused local TCP port for a short-lived test broker."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """Return whether a TCP connection can be opened to ``host:port``."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


@contextmanager
def run_nats_server(
    data_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    monitor_port: int | None = 8222,
    startup_attempts: int = 50,
) -> Iterator[NatsServer]:
    """Start an isolated JetStream-enabled ``nats-server`` for a test.

    The caller owns ``data_dir`` cleanup. ``RuntimeError`` is raised when the
    binary is missing or the broker fails to become reachable.
    """

    binary = shutil.which("nats-server")
    if binary is None:
        raise RuntimeError("nats-server not on PATH")

    broker_port = port if port is not None else free_port(host)
    data_dir.mkdir(parents=True, exist_ok=True)

    args = [binary, "-js", "-p", str(broker_port), "-sd", str(data_dir)]
    if monitor_port is not None:
        args.extend(["-m", str(monitor_port)])

    proc = subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(startup_attempts):
            if port_open(host, broker_port):
                break
            if proc.poll() is not None:
                raise RuntimeError("nats-server exited before becoming ready")
            time.sleep(0.1)
        else:
            raise RuntimeError("Timed out waiting for nats-server to start")

        yield NatsServer(
            url=f"nats://{host}:{broker_port}",
            host=host,
            port=broker_port,
            store_dir=data_dir,
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
