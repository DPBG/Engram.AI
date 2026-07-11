"""Shared fixtures for kernel integration tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from activelearning.testing.nats_server import run_nats_server

_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def nats_server(tmp_path: Path):
    """A real, isolated, JetStream-enabled nats-server on an ephemeral port."""
    binary = shutil.which("nats-server")
    if binary is None:
        pytest.skip("nats-server not on PATH")
    try:
        with run_nats_server(tmp_path / "nats-data") as server:
            yield server.url
    except RuntimeError as exc:
        pytest.skip(str(exc))
