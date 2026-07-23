"""Regression: the direct Kernel-ALLOW code path must sandbox-test the staged
files where they live AFTER stage_testing() moves them (testing/), not the
stale path stage_pending() returned (pending/).

Bug: _handle_knowledge_gap captured `staged_path` from stage_pending() (a
pending/<trace_id>/code.py path), then called stage_testing() — which
shutil.move()s the directory to testing/<trace_id>/ — and finally ran the
sandbox against the *captured* pending path. That directory no longer exists,
so the sandbox mounts a vanished dir, pytest collects nothing (exit 5), every
ALLOW/TRANSFORM proposal "fails" its tests, and no generated code can ever
deploy. The human-approval path (approval_consumer.py) already recomputes the
path from testing_dir after the move; this proves the direct path does too.

The `docker` SDK isn't installed in the test env, so we inject a minimal fake
before importing the service (which imports sandbox_manager → docker).
"""

import asyncio
import os
import sys
import types


def _install_fake_docker() -> None:
    # Reuse an already-installed fake so we never clobber another test module's
    # docker exception classes (sandbox_manager binds `docker` at import time;
    # a divergent fake would break its `except docker.errors.*` handlers).
    if "docker" in sys.modules:
        return
    docker = types.ModuleType("docker")
    errors = types.ModuleType("docker.errors")

    class _DockerError(Exception):
        pass

    errors.ImageNotFound = type("ImageNotFound", (Exception,), {})
    errors.DockerException = _DockerError
    errors.APIError = _DockerError
    models = types.ModuleType("docker.models")
    containers = types.ModuleType("docker.models.containers")
    containers.Container = type("Container", (), {})

    docker.errors = errors
    docker.models = models
    docker.from_env = lambda: object()
    models.containers = containers

    sys.modules["docker"] = docker
    sys.modules["docker.errors"] = errors
    sys.modules["docker.models"] = models
    sys.modules["docker.models.containers"] = containers


_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from meta_programmer.staging import StagingManager  # noqa: E402


class _Logger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass


class _FakeBus:
    def __init__(self):
        self.published = []

    async def publish(self, subject, data):
        self.published.append((subject, data))


class _FakeTeam:
    async def generate_code(self, trace_id, description, context):
        return {
            "success": True,
            "target_path": "/data/plugins/generated_mod.py",
            "code": "def add(a, b):\n    return a + b\n",
            "tests": "def test_add():\n    assert True\n",
        }


class _RecordingSandbox:
    """Records the paths run_tests receives and, like the real SandboxManager
    (which mounts os.path.dirname(code_path)), only 'passes' when the code file
    actually exists on disk."""

    def __init__(self):
        # Each entry: (code_path, test_path, code_existed_at_call, test_existed_at_call)
        self.calls = []

    async def run_tests(self, code_path, test_path=None):
        code_exists = os.path.isfile(code_path)
        test_exists = bool(test_path) and os.path.isfile(test_path)
        self.calls.append((code_path, test_path, code_exists, test_exists))
        return {
            "success": code_exists,
            "sandbox_unavailable": False,
            "output": "",
            "error": "" if code_exists else f"path does not exist: {code_path}",
        }


def _build_service(staging_root: str):
    _install_fake_docker()
    from meta_programmer.service import MetaProgrammerService

    svc = MetaProgrammerService.__new__(MetaProgrammerService)
    svc.logger = _Logger()
    svc._gaps_processed = 0
    svc._gaps_m1_blocked = 0
    svc._code_generated = 0
    svc._tests_passed = 0
    svc._tests_failed = 0
    svc._sandbox_unavailable = 0
    svc._deployments = 0
    svc._team = _FakeTeam()
    staging = StagingManager(staging_root)
    staging.initialize()
    svc._staging_manager = staging
    sandbox = _RecordingSandbox()
    svc._sandbox_manager = sandbox
    svc.event_bus = _FakeBus()
    return svc, sandbox, staging


def test_allow_path_runs_sandbox_against_moved_files(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGRAM_DECISION_BUS_SIGNING_ENABLED", "1")
    monkeypatch.setenv("ENGRAM_SANDBOX_FAIL_CLOSED_ENABLED", "1")
    svc, sandbox, staging = _build_service(str(tmp_path / "staging"))

    deployed = []

    async def _fake_deploy(trace_id, target_path, code):
        deployed.append((trace_id, target_path, code))

    async def _fake_approval(**_kwargs):
        return {"type": "ALLOW"}

    svc._deploy_code = _fake_deploy
    svc._request_kernel_approval = _fake_approval

    asyncio.run(svc._handle_knowledge_gap({"trace_id": "t-1", "description": "add", "context": {}}))

    # The sandbox must have been invoked exactly once with a code path that
    # EXISTED AT CALL TIME under testing/ — not the pre-move pending/ path (the
    # regression, which would have been a vanished directory).
    assert len(sandbox.calls) == 1
    code_path, test_path, code_existed, test_existed = sandbox.calls[0]
    assert code_existed, (
        "sandbox was given a stale/vanished path (pending/ before the "
        "stage_testing() move) instead of the moved testing/ path"
    )
    assert code_path.endswith(os.path.join("testing", "t-1", "code.py"))
    assert test_path is not None and test_existed
    assert test_path.endswith(os.path.join("testing", "t-1", "tests.py"))

    # With tests passing, deployment must proceed.
    assert deployed and deployed[0][0] == "t-1"
    assert svc._deployments == 1
    assert svc._tests_passed == 1


def test_allow_path_without_tests_passes_none_test_path(tmp_path, monkeypatch):
    """When the generator emits no tests, tests.py is never written; the sandbox
    must be told test_path=None (run all tests in the mounted dir) rather than a
    path to a nonexistent file."""
    monkeypatch.setenv("ENGRAM_DECISION_BUS_SIGNING_ENABLED", "1")
    monkeypatch.setenv("ENGRAM_SANDBOX_FAIL_CLOSED_ENABLED", "1")
    svc, sandbox, staging = _build_service(str(tmp_path / "staging"))

    class _NoTestTeam:
        async def generate_code(self, trace_id, description, context):
            return {
                "success": True,
                "target_path": "/data/plugins/generated_mod.py",
                "code": "def add(a, b):\n    return a + b\n",
                "tests": "",
            }

    svc._team = _NoTestTeam()

    async def _fake_deploy(trace_id, target_path, code):
        pass

    async def _fake_approval(**_kwargs):
        return {"type": "ALLOW"}

    svc._deploy_code = _fake_deploy
    svc._request_kernel_approval = _fake_approval

    asyncio.run(svc._handle_knowledge_gap({"trace_id": "t-2", "description": "add", "context": {}}))

    assert len(sandbox.calls) == 1
    code_path, test_path, code_existed, _ = sandbox.calls[0]
    assert code_existed
    assert code_path.endswith(os.path.join("testing", "t-2", "code.py"))
    assert test_path is None
