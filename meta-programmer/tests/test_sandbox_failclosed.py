"""Fail-closed when Docker/sandbox is unavailable (Phase 1, E1.3.2).

Untested code must NEVER deploy. If containment cannot run — no Docker daemon,
missing image, or a spawn failure — the deploy path must treat it as a hard
block (fail-closed), distinct from a genuine test failure.

The `docker` SDK is not installed in the test env, so we inject a minimal fake
`docker` module into sys.modules before importing the modules under test. This
mirrors the real SDK's surface: ``docker.from_env``, ``docker.errors.*`` and
``docker.models.containers.Container``.
"""

import asyncio
import os
import sys
import types

# ── fake `docker` SDK ────────────────────────────────────────────────────────


class _ImageNotFound(Exception):
    pass


class _DockerException(Exception):
    pass


def _install_fake_docker() -> types.ModuleType:
    docker = types.ModuleType("docker")
    errors = types.ModuleType("docker.errors")
    errors.ImageNotFound = _ImageNotFound
    errors.DockerException = _DockerException
    errors.APIError = _DockerException
    models = types.ModuleType("docker.models")
    containers = types.ModuleType("docker.models.containers")

    class Container:  # noqa: D401 - marker type only
        pass

    containers.Container = Container

    def from_env():
        # Default: a working client. Individual tests replace docker_client.
        return object()

    docker.errors = errors
    docker.models = models
    docker.from_env = from_env
    models.containers = containers

    sys.modules["docker"] = docker
    sys.modules["docker.errors"] = errors
    sys.modules["docker.models"] = models
    sys.modules["docker.models.containers"] = containers
    return docker


_install_fake_docker()

# meta_programmer/src on path so `import meta_programmer.*` resolves.
_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from meta_programmer.sandbox_manager import (  # noqa: E402
    SANDBOX_REMEDIATION,
    SandboxManager,
)


def _run(coro):
    return asyncio.run(coro)


# ── fakes for the Docker client used by run_tests ────────────────────────────


class _FakeImages:
    def __init__(self, found: bool):
        self._found = found

    def get(self, _image):
        if not self._found:
            raise _ImageNotFound("no such image")
        return object()


class _FakeContainer:
    def __init__(self, status_code: int):
        self._status_code = status_code

    def wait(self):
        return {"StatusCode": self._status_code}

    def logs(self):
        return b"pytest output"

    def kill(self):
        pass


class _FakeContainers:
    def __init__(self, status_code=0, spawn_error=False):
        self._status_code = status_code
        self._spawn_error = spawn_error

    def run(self, **_kwargs):
        if self._spawn_error:
            raise _DockerException("daemon refused spawn")
        return _FakeContainer(self._status_code)


class _FakeDockerClient:
    def __init__(self, image_found=True, status_code=0, spawn_error=False, ping_ok=True):
        self.images = _FakeImages(image_found)
        self.containers = _FakeContainers(status_code, spawn_error)
        self._ping_ok = ping_ok

    def ping(self):
        if not self._ping_ok:
            raise _DockerException("daemon unreachable")
        return True


def _manager(client) -> SandboxManager:
    mgr = SandboxManager()
    mgr.docker_client = client
    return mgr


# ── run_tests classification: unavailable vs test-failure ────────────────────


def test_no_docker_client_is_unavailable():
    mgr = _manager(None)
    result = _run(mgr.run_tests("/staging/x/mod.py"))
    assert result["success"] is False
    assert result["sandbox_unavailable"] is True
    assert SANDBOX_REMEDIATION in result["error"]


def test_daemon_unreachable_is_unavailable():
    mgr = _manager(_FakeDockerClient(ping_ok=False))
    result = _run(mgr.run_tests("/staging/x/mod.py"))
    assert result["sandbox_unavailable"] is True


def test_missing_image_is_unavailable():
    mgr = _manager(_FakeDockerClient(image_found=False))
    result = _run(mgr.run_tests("/staging/x/mod.py"))
    assert result["sandbox_unavailable"] is True
    assert "image not found" in result["error"].lower()


def test_spawn_failure_is_unavailable():
    mgr = _manager(_FakeDockerClient(spawn_error=True))
    result = _run(mgr.run_tests("/staging/x/mod.py"))
    assert result["sandbox_unavailable"] is True


def test_failing_tests_are_not_unavailable():
    # Containment ran; pytest exited non-zero. This is a TEST failure, and must
    # be reported as such — not as sandbox unavailability.
    mgr = _manager(_FakeDockerClient(status_code=1))
    result = _run(mgr.run_tests("/staging/x/mod.py"))
    assert result["success"] is False
    assert result["sandbox_unavailable"] is False


def test_passing_tests_succeed():
    mgr = _manager(_FakeDockerClient(status_code=0))
    result = _run(mgr.run_tests("/staging/x/mod.py"))
    assert result["success"] is True
    assert result["sandbox_unavailable"] is False


# ── service deploy path: unavailable sandbox blocks deploy ───────────────────


class _FakeStaging:
    def __init__(self):
        self.rejected = []

    def stage_testing(self, _trace_id):
        pass

    def stage_approved(self, _trace_id):
        raise AssertionError("must not approve when sandbox is unavailable")

    def stage_rejected(self, trace_id, reason):
        self.rejected.append((trace_id, reason))


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
            "tests": "",
        }


class _UnavailableSandbox:
    async def run_tests(self, code_path, test_path=None):
        return {
            "success": False,
            "sandbox_unavailable": True,
            "output": "",
            "error": "Docker daemon unavailable: no client. " + SANDBOX_REMEDIATION,
        }


class _Logger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass


def _build_service():
    from meta_programmer.service import MetaProgrammerService

    # Bypass BaseService.__init__ (no NATS/DB) — set only what the handler uses.
    svc = MetaProgrammerService.__new__(MetaProgrammerService)
    svc.logger = _Logger()
    svc._gaps_processed = 0
    svc._code_generated = 0
    svc._tests_passed = 0
    svc._tests_failed = 0
    svc._sandbox_unavailable = 0
    svc._deployments = 0
    svc._team = _FakeTeam()
    svc._staging_manager = _FakeStaging()
    svc._sandbox_manager = _UnavailableSandbox()
    svc.event_bus = _FakeBus()
    return svc


def test_unavailable_sandbox_blocks_deploy():
    svc = _build_service()

    deployed = []

    async def _fake_deploy(trace_id, target_path, code):
        deployed.append(trace_id)

    async def _fake_approval(**_kwargs):
        return {"type": "ALLOW"}

    svc._deploy_code = _fake_deploy
    svc._request_kernel_approval = _fake_approval
    # stage_pending is sync in the real StagingManager; patch onto the fake.
    svc._staging_manager.stage_pending = lambda **kw: "/data/staging/t-1/generated_mod.py"

    _run(svc._handle_knowledge_gap({"trace_id": "t-1", "description": "add", "context": {}}))

    # Core guarantee: nothing deployed, and a fail-closed result was published.
    assert deployed == [], "untested code must not deploy when sandbox is unavailable"
    assert svc._deployments == 0
    assert svc._sandbox_unavailable == 1
    assert svc._staging_manager.rejected, "must record a rejection"

    results = [d for subj, d in svc.event_bus.published if subj.startswith("knowledge.gap.result")]
    assert results, "must publish a gap result"
    assert results[-1]["success"] is False
    assert results[-1]["fail_closed"] is True
