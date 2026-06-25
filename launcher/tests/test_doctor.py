"""Tests for the launcher preflight diagnostics (``python run.py --doctor``).

The environment probes (port checks, HTTP reachability) are injected, so these
run fully offline — no real sockets or network are touched.
"""

import os
import sys
from pathlib import Path

# Make the project root importable so `launcher.doctor` (which uses absolute
# imports into the launcher package) resolves regardless of the cwd pytest uses.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from launcher import doctor  # noqa: E402
from launcher.doctor import Check, OK, WARN, FAIL  # noqa: E402
from launcher.registry import Service  # noqa: E402


def _never_open(_host, _port):
    return False


def _always_open(_host, _port):
    return True


def _reachable(_url):
    return True


def _unreachable(_url):
    return False


# ── python version ────────────────────────────────────────────────────────────

def test_python_version_ok_on_supported():
    c = doctor.check_python_version((3, 12, 0))
    assert c.status == OK


def test_python_version_fails_on_too_old():
    c = doctor.check_python_version((3, 9, 0))
    assert c.status == FAIL
    assert "too old" in c.detail


# ── nats ────────────────────────────────────────────────────────────────────

def test_nats_ok_when_already_listening():
    c = doctor.check_nats_binary(find=lambda: None, port_open=_always_open)
    assert c.status == OK
    assert "listening" in c.detail


def test_nats_ok_when_binary_present():
    c = doctor.check_nats_binary(find=lambda: Path("/somewhere/nats-server"),
                                 port_open=_never_open)
    assert c.status == OK


def test_nats_warns_when_absent():
    c = doctor.check_nats_binary(find=lambda: None, port_open=_never_open)
    assert c.status == WARN
    assert "download" in c.detail


# ── ports ─────────────────────────────────────────────────────────────────────

def test_port_free_is_ok():
    c = doctor.check_port_free("nats-client", 4222, port_open=_never_open)
    assert c.status == OK


def test_port_in_use_is_warn():
    c = doctor.check_port_free("nats-client", 4222, port_open=_always_open)
    assert c.status == WARN


# ── optional infra ─────────────────────────────────────────────────────────────

def test_optional_infra_reachable_is_ok():
    c = doctor.check_optional_infra("qdrant", "http://x/healthz", http_ok=_reachable)
    assert c.status == OK


def test_optional_infra_unreachable_is_warn_not_fail():
    c = doctor.check_optional_infra("ollama", "http://x/api/tags", http_ok=_unreachable)
    assert c.status == WARN  # optional → never a hard failure


# ── data dir / disk ─────────────────────────────────────────────────────────────

def test_data_dir_writable(tmp_path):
    c = doctor.check_data_dirs(root=tmp_path)
    assert c.status == OK


def test_disk_space_ok(tmp_path):
    c = doctor.check_disk_space(root=tmp_path, min_free_gb=0.0)
    assert c.status == OK


def test_disk_space_low_is_warn(tmp_path):
    # Demand an absurd amount of free space to force the low-space branch.
    c = doctor.check_disk_space(root=tmp_path, min_free_gb=10_000_000.0)
    assert c.status == WARN


# ── service registry consistency ───────────────────────────────────────────────

def test_service_sources_ok_for_real_registry():
    c = doctor.check_service_sources()
    assert c.status == OK


def test_service_sources_fail_when_missing():
    bogus = [Service(name="ghost", module="ghost", src="does/not/exist", profile="core")]
    c = doctor.check_service_sources(bogus)
    assert c.status == FAIL
    assert "ghost" in c.detail


# ── orchestration: report + exit code ──────────────────────────────────────────

def test_run_checks_all_green_when_env_is_healthy():
    checks = doctor.run_checks(
        port_open=_never_open,   # ports free, nats will warn (no binary in CI maybe)
        http_ok=_reachable,      # optional infra "up"
    )
    # No check should hard-fail in a healthy tree (nats may be a warn).
    assert not any(c.status == FAIL for c in checks)
    assert doctor.exit_code(checks) == 0


def test_exit_code_nonzero_on_any_fail():
    checks = [Check("a", OK, ""), Check("b", FAIL, "boom")]
    assert doctor.exit_code(checks) == 1


def test_exit_code_zero_on_warnings_only():
    checks = [Check("a", OK, ""), Check("b", WARN, "meh")]
    assert doctor.exit_code(checks) == 0


def test_format_report_contains_markers_and_summary():
    report = doctor.format_report([Check("python", OK, "fine"),
                                   Check("disk", WARN, "low")])
    assert "[ OK ]" in report
    assert "[WARN]" in report
    assert "Ready to run" in report
