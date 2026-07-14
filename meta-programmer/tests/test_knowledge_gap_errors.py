"""Regression tests for knowledge-gap error handling in MetaProgrammerService."""

from __future__ import annotations

import asyncio
import sys
import types


def _install_fake_docker() -> None:
    if "docker" in sys.modules and hasattr(sys.modules["docker"], "models"):
        return
    docker = types.ModuleType("docker")
    errors = types.ModuleType("docker.errors")

    class _DockerError(Exception):
        pass

    errors.ImageNotFound = _DockerError
    errors.DockerException = _DockerError
    errors.APIError = _DockerError
    models = types.ModuleType("docker.models")
    containers = types.ModuleType("docker.models.containers")

    class Container:
        pass

    containers.Container = Container
    models.containers = containers
    docker.errors = errors
    docker.models = models
    docker.from_env = lambda: object()
    sys.modules["docker"] = docker
    sys.modules["docker.errors"] = errors
    sys.modules["docker.models"] = models
    sys.modules["docker.models.containers"] = containers


def _run(coro):
    return asyncio.run(coro)


class _FakeBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []

    async def publish(self, subject: str, data: dict) -> None:
        self.published.append((subject, data))


class _RaisingTeam:
    async def generate_code(self, trace_id, description, context):
        raise RuntimeError("LLM unavailable")


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def _build_service():
    _install_fake_docker()
    from meta_programmer.service import MetaProgrammerService

    svc = MetaProgrammerService.__new__(MetaProgrammerService)
    svc.logger = _Logger()
    svc._gaps_processed = 0
    svc._gaps_m1_blocked = 0
    svc._team = _RaisingTeam()
    svc.event_bus = _FakeBus()
    return svc


def test_knowledge_gap_exception_publishes_gap_result_once(monkeypatch):
    """Unhandled errors must publish exactly one knowledge.gap.result message."""
    monkeypatch.setenv("ENGRAM_DECISION_BUS_SIGNING_ENABLED", "1")
    monkeypatch.setenv("ENGRAM_SANDBOX_FAIL_CLOSED_ENABLED", "1")
    svc = _build_service()

    _run(svc._handle_knowledge_gap({"trace_id": "t-err", "description": "add", "context": {}}))

    gap_results = [
        (subj, data)
        for subj, data in svc.event_bus.published
        if subj == "knowledge.gap.result.t-err"
    ]
    assert len(gap_results) == 1
    assert gap_results[0][1]["success"] is False
    assert "LLM unavailable" in gap_results[0][1]["message"]
