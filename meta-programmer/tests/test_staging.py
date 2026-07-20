"""Tests for the staging human-review expiry sweep (Phase 1.9).

Loads staging.py directly so the test doesn't import the meta_programmer
package (which pulls in the `docker` SDK). staging.py is pure stdlib.
"""

import importlib.util
import os
import sys
import tempfile

_STAGING_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "meta_programmer", "staging.py"
)
_spec = importlib.util.spec_from_file_location("mp_staging", _STAGING_PATH)
_staging = importlib.util.module_from_spec(_spec)
sys.modules["mp_staging"] = _staging
_spec.loader.exec_module(_staging)
StagingManager = _staging.StagingManager
is_review_expired = _staging.is_review_expired


# ── pure expiry predicate ──────────────────────────────────────────────────


def test_within_ttl_not_expired():
    assert is_review_expired({"created_at": 1000}, now_ms=1500, ttl_ms=1000) is False


def test_past_ttl_expired():
    assert is_review_expired({"created_at": 1000}, now_ms=2500, ttl_ms=1000) is True


def test_missing_created_at_is_expired_fail_closed():
    # No timestamp → can't prove it's fresh → treat as expired (fail-closed).
    assert is_review_expired({}, now_ms=2500, ttl_ms=1000) is True


# ── staging integration ────────────────────────────────────────────────────


def _staged_review(d, trace_id, created_at):
    sm = StagingManager(d)
    sm.initialize()
    sm.stage_pending(trace_id, f"/data/plugins/{trace_id}.py", "x = 1\n")
    # Force a known created_at, then move to human_review.
    import json

    meta_path = os.path.join(sm.pending_dir, trace_id, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    meta["created_at"] = created_at
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    sm.stage_human_review(trace_id)
    return sm


def test_stage_pending_records_created_at():
    with tempfile.TemporaryDirectory() as d:
        sm = StagingManager(d)
        sm.initialize()
        sm.stage_pending("t1", "/data/plugins/t1.py", "x = 1\n")
        meta = sm.get_metadata("t1")
        assert isinstance(meta.get("created_at"), int)


def test_created_at_survives_stage_move():
    with tempfile.TemporaryDirectory() as d:
        sm = _staged_review(d, "t1", created_at=1234)
        assert sm.get_metadata("t1")["created_at"] == 1234
        assert sm.get_metadata("t1")["stage"] == "human_review"


def test_expired_reviews_selects_only_aged_items():
    with tempfile.TemporaryDirectory() as d:
        sm = _staged_review(d, "old", created_at=0)
        # Add a fresh one in the same staging root.
        import json

        sm.stage_pending("fresh", "/data/plugins/fresh.py", "x = 1\n")
        mp = os.path.join(sm.pending_dir, "fresh", "metadata.json")
        with open(mp) as f:
            meta = json.load(f)
        meta["created_at"] = 9_500
        with open(mp, "w") as f:
            json.dump(meta, f)
        sm.stage_human_review("fresh")

        expired = sm.expired_reviews(now_ms=10_000, ttl_ms=1_000)
        assert "old" in expired  # aged 10_000 >= ttl
        assert "fresh" not in expired  # aged 500 < ttl


def test_stage_rejected_works_from_human_review():
    with tempfile.TemporaryDirectory() as d:
        sm = _staged_review(d, "t1", created_at=0)
        sm.stage_rejected("t1", "DEFER expired")
        meta = sm.get_metadata("t1")
        assert meta["stage"] == "rejected"
        assert meta["rejection_reason"] == "DEFER expired"
        # No longer awaiting review.
        assert sm.expired_reviews(now_ms=10_000, ttl_ms=1) == []
