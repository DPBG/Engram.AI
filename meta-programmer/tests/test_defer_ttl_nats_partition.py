"""M1.17 — DEFER TTL expiry under a Dashboard↔NATS partition (fail-closed).

Scenario: Kernel returned DEFER; Meta-Programmer staged the proposal in
``human_review/`` and published ``approval.request``. A network partition
prevents the Dashboard from publishing ``approval.response.{trace_id}`` before
the review TTL elapses.

Contract (CLAUDE.md §3 / docs/META-PROGRAMMER.md): silence MUST expire as
DENY — staging → rejected, gap result ``success=False``. A late Approve after
the partition heals must not revive the rejected item.

Loads staging + approval_consumer via importlib (same CI-governance pattern as
``test_approval_consumer.py``) and drives ``MetaProgrammerService._sweep_expired_reviews``
on a ``__new__`` service so Docker/NATS are not required.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import types

# ── module bootstrap (avoid meta_programmer/__init__.py → docker / nats) ─────

_SRC = os.path.join(os.path.dirname(__file__), "..", "src", "meta_programmer")

if "meta_programmer" not in sys.modules:
    _fake_pkg = types.ModuleType("meta_programmer")
    _fake_pkg.__path__ = [_SRC]
    _fake_pkg.__package__ = "meta_programmer"
    sys.modules["meta_programmer"] = _fake_pkg


def _load(module_dotted_name: str, filename: str) -> types.ModuleType:
    path = os.path.join(_SRC, filename)
    spec = importlib.util.spec_from_file_location(module_dotted_name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "meta_programmer"
    sys.modules[module_dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


_staging_mod = _load("meta_programmer.staging", "staging.py")
_consumer_mod = _load("meta_programmer.approval_consumer", "approval_consumer.py")

StagingManager = _staging_mod.StagingManager
ApprovalConsumer = _consumer_mod.ApprovalConsumer

_DEFER_TTL_MS = 5_000  # short TTL for the test clock
_NOW_MS = 1_000_000


# ── helpers ──────────────────────────────────────────────────────────────────


class _Logger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass


class _PartitionedBus:
    """EventBus stand-in: approval responses never leave the Dashboard.

    Publishing ``approval.response.*`` is a no-op sink (messages are dropped).
    Gap-result publishes are recorded so the sweep's DENY can be asserted.
    """

    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []
        self.dropped_approvals: list[tuple[str, dict]] = []

    async def publish(self, subject: str, data: dict) -> None:
        if subject.startswith("approval.response"):
            # Partition: Dashboard → NATS path is down.
            self.dropped_approvals.append((subject, data))
            return
        self.published.append((subject, data))


def _staged_review(staging_root: str, trace_id: str, created_at: int) -> StagingManager:
    sm = StagingManager(staging_root)
    sm.initialize()
    sm.stage_pending(trace_id, f"/data/plugins/{trace_id}.py", "x = 1\n")
    meta_path = os.path.join(sm.pending_dir, trace_id, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["created_at"] = created_at
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    sm.stage_human_review(trace_id)
    return sm


def _build_service(sm: StagingManager, bus: _PartitionedBus, defer_ttl_ms: int):
    """Minimal MetaProgrammerService for ``_sweep_expired_reviews`` only."""
    # Import service without pulling real docker: stub if missing.
    if "docker" not in sys.modules:
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

    from meta_programmer.service import MetaProgrammerService

    svc = MetaProgrammerService.__new__(MetaProgrammerService)
    svc.logger = _Logger()
    svc._staging_manager = sm
    svc.defer_ttl_ms = defer_ttl_ms
    svc._reviews_expired = 0
    svc.event_bus = bus

    async def _publish_gap_result(trace_id, success, message, fail_closed=False):
        await bus.publish(
            f"knowledge.gap.result.{trace_id}",
            {
                "trace_id": trace_id,
                "success": success,
                "message": message,
                "fail_closed": fail_closed,
            },
        )

    svc._publish_gap_result = _publish_gap_result
    return svc


def _make_consumer(sm: StagingManager, bus: _PartitionedBus, defer_ttl_ms: int):
    results: list = []

    async def publish_gap(trace_id, success, message, fail_closed=False):
        results.append(
            {
                "trace_id": trace_id,
                "success": success,
                "message": message,
                "fail_closed": fail_closed,
            }
        )

    async def noop_run_tests(**_kwargs):
        return {"success": True, "error": None}

    async def noop_deploy(*_a, **_k):
        pass

    consumer = ApprovalConsumer(
        staging=sm,
        defer_ttl_ms=defer_ttl_ms,
        run_tests=noop_run_tests,
        deploy=noop_deploy,
        publish_gap_result=publish_gap,
        log=_Logger(),
    )
    return consumer, results


def _run(coro):
    return asyncio.run(coro)


# ── M1.17 partition test ─────────────────────────────────────────────────────


def test_nats_partition_during_defer_window_expires_as_deny():
    """Unanswered DEFER under partition → TTL sweep DENYs (fail-closed).

    Timeline:
      t0     Kernel DEFER; staged into human_review/; approval.request published
      t0..T  Dashboard cannot publish approval.response (NATS partition)
      t>=T   Meta-Programmer sweep rejects + publishes knowledge.gap.result deny
      heal   Operator Approve arrives late → ignored; item stays rejected
    """
    with tempfile.TemporaryDirectory() as d:
        created_at = _NOW_MS - _DEFER_TTL_MS - 1  # already past TTL when sweep runs
        sm = _staged_review(d, "defer-partition-1", created_at=created_at)
        bus = _PartitionedBus()
        svc = _build_service(sm, bus, defer_ttl_ms=_DEFER_TTL_MS)

        # Simulate Dashboard trying to Approve while partitioned — drops on the bus.
        _run(
            bus.publish(
                "approval.response.defer-partition-1",
                {"trace_id": "defer-partition-1", "approved": True},
            )
        )
        assert bus.dropped_approvals, "partition must drop approval.response"
        assert sm.get_metadata("defer-partition-1")["stage"] == "human_review"
        assert not any(s.startswith("knowledge.gap.result") for s, _ in bus.published)

        # TTL expires; sweep fail-closes without ever seeing an approval.
        n = _run(svc._sweep_expired_reviews(_NOW_MS))
        assert n == 1
        assert svc._reviews_expired == 1

        meta = sm.get_metadata("defer-partition-1")
        assert meta["stage"] == "rejected"
        assert "fail-closed" in meta.get("rejection_reason", "")
        assert "DEFER expired" in meta.get("rejection_reason", "")

        results = [data for subj, data in bus.published if "knowledge.gap.result" in subj]
        assert len(results) == 1
        assert results[0]["success"] is False
        assert "DEFER expired" in results[0]["message"]
        assert "fail-closed" in results[0]["message"]

        # Partition heals: late Approve must not deploy / revive.
        consumer, late_results = _make_consumer(sm, bus, defer_ttl_ms=_DEFER_TTL_MS)
        _run(consumer.handle_approval_response({"trace_id": "defer-partition-1", "approved": True}))
        assert late_results == []
        assert sm.get_metadata("defer-partition-1")["stage"] == "rejected"


def test_defer_within_ttl_not_swept_while_partitioned():
    """Inside the DEFER window, partition alone must not reject the review."""
    with tempfile.TemporaryDirectory() as d:
        created_at = _NOW_MS - 100  # well within TTL
        sm = _staged_review(d, "defer-fresh-1", created_at=created_at)
        bus = _PartitionedBus()
        svc = _build_service(sm, bus, defer_ttl_ms=_DEFER_TTL_MS)

        n = _run(svc._sweep_expired_reviews(_NOW_MS))
        assert n == 0
        assert svc._reviews_expired == 0
        assert sm.get_metadata("defer-fresh-1")["stage"] == "human_review"
        assert not any("knowledge.gap.result" in s for s, _ in bus.published)
