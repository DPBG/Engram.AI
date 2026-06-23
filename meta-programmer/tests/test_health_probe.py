"""Tests for post-deploy health probe and auto-rollback (Phase E1.9.2).

Covers:
- run_health_probe: pass, import-error, missing file, timeout
- deploy_atomically with probe_timeout: rollback over existing file,
  removal of newly-created broken file, and pass-through for healthy artifact.
- Default behaviour (probe_timeout=0): import-time crashes are not caught
  (only syntax is checked), i.e. the probe is opt-in.
"""

import importlib.util
import os
import sys
import tempfile
import time

# Load safety.py directly so the test doesn't pull in the `docker` SDK.
_SAFETY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "meta_programmer", "safety.py"
)
_spec = importlib.util.spec_from_file_location("mp_safety_hp", _SAFETY_PATH)
_safety = importlib.util.module_from_spec(_spec)
sys.modules["mp_safety_hp"] = _safety
_spec.loader.exec_module(_safety)

run_health_probe = _safety.run_health_probe
deploy_atomically = _safety.deploy_atomically


# ── run_health_probe unit tests ───────────────────────────────────────────────

def test_probe_passes_for_valid_module():
    with tempfile.TemporaryDirectory() as d:
        target = os.path.join(d, "plugin.py")
        with open(target, "w") as f:
            f.write("LOADED = True\n")
        ok, reason = run_health_probe(target, timeout=5.0)
        assert ok is True
        assert "passed" in reason


def test_probe_fails_for_import_error():
    with tempfile.TemporaryDirectory() as d:
        target = os.path.join(d, "broken.py")
        with open(target, "w") as f:
            f.write("raise ImportError('intentional failure')\n")
        ok, reason = run_health_probe(target, timeout=5.0)
        assert ok is False
        assert "probe failed" in reason


def test_probe_fails_for_runtime_error_at_import():
    with tempfile.TemporaryDirectory() as d:
        target = os.path.join(d, "crasher.py")
        with open(target, "w") as f:
            f.write("raise RuntimeError('crash at module level')\n")
        ok, reason = run_health_probe(target, timeout=5.0)
        assert ok is False
        assert "probe failed" in reason


def test_probe_fails_for_missing_file():
    ok, reason = run_health_probe("/nonexistent/path/to/plugin.py", timeout=5.0)
    assert ok is False
    assert "not found" in reason


def test_probe_times_out_for_hanging_module():
    with tempfile.TemporaryDirectory() as d:
        target = os.path.join(d, "hanging.py")
        with open(target, "w") as f:
            # Hangs immediately at import time
            f.write("import time\ntime.sleep(60)\n")
        start = time.monotonic()
        ok, reason = run_health_probe(target, timeout=1.0)
        elapsed = time.monotonic() - start
        assert ok is False
        assert "timed out" in reason
        # Must not take much longer than the configured timeout
        assert elapsed < 5.0


# ── deploy_atomically with probe_timeout ─────────────────────────────────────

def test_deploy_with_probe_passes_healthy_artifact():
    with tempfile.TemporaryDirectory() as d:
        target = os.path.join(d, "plugin.py")
        ok, detail = deploy_atomically(target, "VERSION = '1.0'\n", probe_timeout=5.0)
        assert ok is True
        assert "deployed" in detail
        with open(target) as f:
            assert "VERSION" in f.read()


def test_deploy_with_probe_rolls_back_broken_artifact_over_existing():
    """Artifact that crashes at import-time must restore the prior good file."""
    with tempfile.TemporaryDirectory() as d:
        target = os.path.join(d, "plugin.py")
        with open(target, "w") as f:
            f.write("GOOD = True\n")

        ok, detail = deploy_atomically(
            target,
            "raise ImportError('deliberate probe failure')\n",
            probe_timeout=5.0,
        )

        assert ok is False
        assert "rolled back" in detail
        assert "health probe" in detail
        # Prior content must be restored intact.
        with open(target) as f:
            assert f.read() == "GOOD = True\n"


def test_deploy_with_probe_removes_new_broken_artifact():
    """A brand-new file that fails the probe must be cleaned up entirely."""
    with tempfile.TemporaryDirectory() as d:
        target = os.path.join(d, "new_plugin.py")

        ok, detail = deploy_atomically(
            target,
            "raise RuntimeError('crash on import')\n",
            probe_timeout=5.0,
        )

        assert ok is False
        assert "rolled back" in detail
        assert "health probe" in detail
        assert not os.path.exists(target)


def test_deploy_with_probe_replaces_existing_on_success():
    with tempfile.TemporaryDirectory() as d:
        target = os.path.join(d, "plugin.py")
        with open(target, "w") as f:
            f.write("OLD = 1\n")

        ok, detail = deploy_atomically(target, "NEW = 2\n", probe_timeout=5.0)

        assert ok is True
        with open(target) as f:
            assert f.read() == "NEW = 2\n"


# ── default behaviour: no probe unless probe_timeout > 0 ─────────────────────

def test_deploy_without_probe_does_not_catch_import_errors():
    """With probe_timeout=0 (default) only syntax is checked — import-time
    crashes are NOT caught, so a valid-syntax but crashing module deploys."""
    with tempfile.TemporaryDirectory() as d:
        target = os.path.join(d, "plugin.py")
        ok, detail = deploy_atomically(
            target,
            "raise RuntimeError('crash on import')\n",
        )
        # Valid Python syntax — should be written successfully (no probe)
        assert ok is True
        assert os.path.exists(target)
