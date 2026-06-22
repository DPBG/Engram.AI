"""Tests for the human-approval consumer (Phase E1.9.1).

Covers:
  - StagingManager.is_in_human_review helper
  - StagingManager.stage_human_review_to_testing helper
  - ApprovalConsumer.handle_approval_response: approve, deny, late-answer,
    TTL-expired, and ambiguous approved value
  - Sandbox test failure after approval (rejects, doesn't deploy)
  - Deploy failure after sandbox pass (rejects via stage_rejected)

All modules are loaded via importlib to avoid triggering meta_programmer's
__init__.py, which transitively imports the docker and NATS runtimes that are
not available in the CI governance test environment.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import types

# ── module bootstrap ─────────────────────────────────────────────────────────
# We load staging.py and approval_consumer.py directly so their code runs
# but meta_programmer/__init__.py (which pulls in docker / nats) does not.

_SRC = os.path.join(os.path.dirname(__file__), "..", "src", "meta_programmer")

# Create a stub meta_programmer package entry so that
# "from meta_programmer.staging import ..." inside approval_consumer.py
# resolves without triggering __init__.py.
if "meta_programmer" not in sys.modules:
    _fake_pkg = types.ModuleType("meta_programmer")
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
is_review_expired = _staging_mod.is_review_expired
ApprovalConsumer = _consumer_mod.ApprovalConsumer


# ── helpers ──────────────────────────────────────────────────────────────────


def _staged_review(staging_root: str, trace_id: str, created_at: int) -> StagingManager:
    """Return a StagingManager with trace_id in human_review at a given age."""
    sm = StagingManager(staging_root)
    sm.initialize()
    sm.stage_pending(trace_id, f"/data/plugins/{trace_id}.py", "x = 1\n")
    # Override created_at so tests can control TTL.
    meta_path = os.path.join(sm.pending_dir, trace_id, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["created_at"] = created_at
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    sm.stage_human_review(trace_id)
    return sm


async def _noop_run_tests(**_kwargs) -> dict:
    return {"success": True, "error": None}


async def _failing_run_tests(**_kwargs) -> dict:
    return {"success": False, "error": "assertion failed"}


async def _noop_deploy(trace_id: str, target_path: str, code: str) -> None:
    pass


async def _raising_deploy(trace_id: str, target_path: str, code: str) -> None:
    raise RuntimeError("disk full")


def _make_consumer(
    sm: StagingManager,
    *,
    defer_ttl_ms: int = 300_000,
    run_tests=None,
    deploy=None,
    gap_results: list | None = None,
) -> ApprovalConsumer:
    results = gap_results if gap_results is not None else []

    async def _publish(trace_id: str, success: bool, message: str) -> None:
        results.append({"trace_id": trace_id, "success": success, "message": message})

    return ApprovalConsumer(
        staging=sm,
        defer_ttl_ms=defer_ttl_ms,
        run_tests=run_tests or _noop_run_tests,
        deploy=deploy or _noop_deploy,
        publish_gap_result=_publish,
    )


# ── StagingManager helpers ────────────────────────────────────────────────────


def test_is_in_human_review_true_when_awaiting():
    with tempfile.TemporaryDirectory() as d:
        sm = _staged_review(d, "t1", created_at=0)
        assert sm.is_in_human_review("t1") is True


def test_is_in_human_review_false_when_not_staged():
    with tempfile.TemporaryDirectory() as d:
        sm = StagingManager(d)
        sm.initialize()
        assert sm.is_in_human_review("unknown") is False


def test_is_in_human_review_false_after_rejection():
    with tempfile.TemporaryDirectory() as d:
        sm = _staged_review(d, "t1", created_at=0)
        sm.stage_rejected("t1", "expired")
        assert sm.is_in_human_review("t1") is False


def test_stage_human_review_to_testing_moves_item():
    with tempfile.TemporaryDirectory() as d:
        sm = _staged_review(d, "t1", created_at=0)
        sm.stage_human_review_to_testing("t1")
        assert not sm.is_in_human_review("t1")
        assert os.path.isdir(os.path.join(sm.testing_dir, "t1"))


def test_stage_human_review_to_testing_updates_stage_name():
    with tempfile.TemporaryDirectory() as d:
        sm = _staged_review(d, "t1", created_at=0)
        sm.stage_human_review_to_testing("t1")
        assert sm.get_metadata("t1")["stage"] == "testing"


def test_stage_human_review_to_testing_preserves_created_at():
    with tempfile.TemporaryDirectory() as d:
        sm = _staged_review(d, "t1", created_at=12345)
        sm.stage_human_review_to_testing("t1")
        assert sm.get_metadata("t1")["created_at"] == 12345


# ── ApprovalConsumer — approve path ──────────────────────────────────────────


def test_approve_response_deploys_and_publishes_success():
    with tempfile.TemporaryDirectory() as d:
        sm = _staged_review(d, "t1", created_at=int(9e12))  # far future → not expired
        results: list = []
        consumer = _make_consumer(sm, gap_results=results)
        asyncio.run(consumer.handle_approval_response({"trace_id": "t1", "approved": True}))
        assert len(results) == 1
        assert results[0]["success"] is True
        assert "t1" in results[0]["trace_id"]


def test_approve_increments_counters():
    with tempfile.TemporaryDirectory() as d:
        sm = _staged_review(d, "t1", created_at=int(9e12))
        consumer = _make_consumer(sm)
        asyncio.run(consumer.handle_approval_response({"trace_id": "t1", "approved": True}))
        assert consumer.tests_passed == 1
        assert consumer.deployments == 1
        assert consumer.tests_failed == 0


def test_approve_moves_item_to_approved():
    with tempfile.TemporaryDirectory() as d:
        sm = _staged_review(d, "t1", created_at=int(9e12))
        consumer = _make_consumer(sm)
        asyncio.run(consumer.handle_approval_response({"trace_id": "t1", "approved": True}))
        assert sm.get_metadata("t1")["stage"] == "approved"


# ── ApprovalConsumer — deny path ─────────────────────────────────────────────


def test_deny_response_stages_rejected():
    with tempfile.TemporaryDirectory() as d:
        sm = _staged_review(d, "t1", created_at=int(9e12))
        consumer = _make_consumer(sm)
        asyncio.run(consumer.handle_approval_response({"trace_id": "t1", "approved": False}))
        assert sm.get_metadata("t1")["stage"] == "rejected"


def test_deny_response_publishes_failure():
    with tempfile.TemporaryDirectory() as d:
        sm = _staged_review(d, "t1", created_at=int(9e12))
        results: list = []
        consumer = _make_consumer(sm, gap_results=results)
        asyncio.run(consumer.handle_approval_response({"trace_id": "t1", "approved": False}))
        assert results[0]["success"] is False


def test_ambiguous_approved_value_treated_as_deny():
    # Only explicit True is accepted; anything else is fail-closed.
    with tempfile.TemporaryDirectory() as d:
        sm = _staged_review(d, "t1", created_at=int(9e12))
        consumer = _make_consumer(sm)
        asyncio.run(consumer.handle_approval_response({"trace_id": "t1", "approved": "yes"}))
        assert sm.get_metadata("t1")["stage"] == "rejected"


def test_missing_approved_field_treated_as_deny():
    with tempfile.TemporaryDirectory() as d:
        sm = _staged_review(d, "t1", created_at=int(9e12))
        consumer = _make_consumer(sm)
        asyncio.run(consumer.handle_approval_response({"trace_id": "t1"}))
        assert sm.get_metadata("t1")["stage"] == "rejected"


# ── ApprovalConsumer — idempotency (late answer) ─────────────────────────────


def test_late_answer_after_expiry_sweep_is_ignored():
    with tempfile.TemporaryDirectory() as d:
        sm = _staged_review(d, "t1", created_at=0)
        # Simulate expiry sweep running first.
        sm.stage_rejected("t1", "DEFER expired — no human approval (fail-closed)")
        results: list = []
        consumer = _make_consumer(sm, gap_results=results)
        asyncio.run(consumer.handle_approval_response({"trace_id": "t1", "approved": True}))
        # No additional gap result published; item remains rejected.
        assert results == []
        assert sm.get_metadata("t1")["stage"] == "rejected"


def test_duplicate_approve_ignored():
    with tempfile.TemporaryDirectory() as d:
        sm = _staged_review(d, "t1", created_at=int(9e12))
        results: list = []
        consumer = _make_consumer(sm, gap_results=results)
        asyncio.run(consumer.handle_approval_response({"trace_id": "t1", "approved": True}))
        asyncio.run(consumer.handle_approval_response({"trace_id": "t1", "approved": True}))
        # Second call is a no-op.
        assert len(results) == 1


# ── ApprovalConsumer — TTL-expired fail-closed ───────────────────────────────


def test_approval_after_ttl_expired_fail_closed():
    with tempfile.TemporaryDirectory() as d:
        # created_at=0 means the item was created at epoch → always expired.
        sm = _staged_review(d, "t1", created_at=0)
        results: list = []
        consumer = _make_consumer(sm, defer_ttl_ms=1_000, gap_results=results)
        asyncio.run(consumer.handle_approval_response({"trace_id": "t1", "approved": True}))
        assert sm.get_metadata("t1")["stage"] == "rejected"
        assert results[0]["success"] is False
        assert "TTL" in results[0]["message"]


# ── ApprovalConsumer — sandbox failure after approve ─────────────────────────


def test_sandbox_failure_after_approve_rejects():
    with tempfile.TemporaryDirectory() as d:
        sm = _staged_review(d, "t1", created_at=int(9e12))
        results: list = []
        consumer = _make_consumer(sm, run_tests=_failing_run_tests, gap_results=results)
        asyncio.run(consumer.handle_approval_response({"trace_id": "t1", "approved": True}))
        assert sm.get_metadata("t1")["stage"] == "rejected"
        assert results[0]["success"] is False
        assert "Tests failed" in results[0]["message"]


def test_sandbox_failure_increments_tests_failed():
    with tempfile.TemporaryDirectory() as d:
        sm = _staged_review(d, "t1", created_at=int(9e12))
        consumer = _make_consumer(sm, run_tests=_failing_run_tests)
        asyncio.run(consumer.handle_approval_response({"trace_id": "t1", "approved": True}))
        assert consumer.tests_failed == 1
        assert consumer.tests_passed == 0
        assert consumer.deployments == 0


# ── ApprovalConsumer — deploy failure after sandbox pass ─────────────────────


def test_deploy_failure_after_sandbox_pass_rejects():
    with tempfile.TemporaryDirectory() as d:
        sm = _staged_review(d, "t1", created_at=int(9e12))
        results: list = []
        consumer = _make_consumer(sm, deploy=_raising_deploy, gap_results=results)
        asyncio.run(consumer.handle_approval_response({"trace_id": "t1", "approved": True}))
        assert sm.get_metadata("t1")["stage"] == "rejected"
        assert results[0]["success"] is False
        assert "Deploy failed" in results[0]["message"]


# ── ApprovalConsumer — malformed input ───────────────────────────────────────


def test_missing_trace_id_is_ignored():
    with tempfile.TemporaryDirectory() as d:
        sm = StagingManager(d)
        sm.initialize()
        results: list = []
        consumer = _make_consumer(sm, gap_results=results)
        asyncio.run(consumer.handle_approval_response({"approved": True}))
        assert results == []


def test_empty_trace_id_is_ignored():
    with tempfile.TemporaryDirectory() as d:
        sm = StagingManager(d)
        sm.initialize()
        results: list = []
        consumer = _make_consumer(sm, gap_results=results)
        asyncio.run(consumer.handle_approval_response({"trace_id": "", "approved": True}))
        assert results == []
