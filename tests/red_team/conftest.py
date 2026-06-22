"""Fixtures for NATS broker authz red-team tests."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
import uuid
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
    data_dir = Path("/tmp") / f"engram-nats-authz-{uuid.uuid4().hex}"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Render config with ephemeral client/monitor ports.
    conf_text = AUTHZ_CONF.read_text(encoding="utf-8")
    conf_text = conf_text.replace("127.0.0.1:4222", f"{host}:{port}")
    conf_text = conf_text.replace("127.0.0.1:8222", f"{host}:{monitor_port}")
    conf_text = conf_text.replace(
        'store_dir: "/tmp/engram-nats-authz-js"',
        f'store_dir: "{data_dir / "jetstream"}"',
    )
    rendered_conf = data_dir / "nats.conf"
    rendered_conf.write_text(conf_text, encoding="utf-8")

    proc = subprocess.Popen(
        [binary, "-c", str(rendered_conf)],
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
