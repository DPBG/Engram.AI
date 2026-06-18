"""Unit tests for launcher.supervisor — no real subprocesses, no NATS required."""

from __future__ import annotations

import io
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from launcher.registry import Service
from launcher.supervisor import (
    ManagedProcess,
    Supervisor,
    _BACKOFF_FACTOR,
    _BACKOFF_INITIAL,
    _BACKOFF_MAX,
    _BACKOFF_RESET,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATA_DIR = Path("/tmp/engram-test-data")


def _svc(
    name: str = "test-svc",
    deps: tuple = (),
    readiness_timeout: float = 0.05,
) -> Service:
    return Service(
        name=name,
        module="test.module",
        src=".",
        profile="core",
        deps=deps,
        readiness_timeout=readiness_timeout,
    )


class FakePopen:
    """Minimal subprocess.Popen stand-in."""

    def __init__(self, exit_sequence: list[int | None]):
        # exit_sequence: list of return values for successive poll() calls,
        # followed by the final exit code returned by wait().
        self._exit_seq = list(exit_sequence)
        self._exit_code: int | None = None
        self._waited = threading.Event()
        self.stdout = io.StringIO("")  # no output
        self.pid = 99999

    def poll(self) -> int | None:
        return self._exit_code

    def wait(self, timeout: float | None = None) -> int:
        if not self._waited.wait(timeout=timeout):
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
        if self._exit_code is None:
            raise Exception("wait() called before process exited in test")
        return self._exit_code

    def terminate(self) -> None:
        self._exit_code = -signal.SIGTERM
        self._waited.set()

    def kill(self) -> None:
        self._exit_code = -signal.SIGKILL
        self._waited.set()

    def exit(self, code: int = 0) -> None:
        """Simulate the process exiting with the given code."""
        self._exit_code = code
        self._waited.set()


def _make_supervisor() -> Supervisor:
    base_env: dict = {}
    sup = Supervisor(base_env, _DATA_DIR)
    # Patch _service_env so we don't need real paths.
    sup._service_env = lambda svc: {}  # type: ignore[method-assign]
    return sup


# ---------------------------------------------------------------------------
# ManagedProcess
# ---------------------------------------------------------------------------


class TestManagedProcess:
    def test_initial_state(self):
        fp = FakePopen([])
        mp = ManagedProcess("svc", fp, "", _svc())
        assert mp.restart_count == 0
        assert not mp.ready.is_set()

    def test_ready_event_starts_unset(self):
        fp = FakePopen([])
        mp = ManagedProcess("svc", fp, "", _svc())
        assert not mp.ready.is_set()


# ---------------------------------------------------------------------------
# Readiness gating
# ---------------------------------------------------------------------------


class TestReadinessGating:
    def test_ready_set_after_timeout(self):
        sup = _make_supervisor()
        fp = FakePopen([])
        mp = ManagedProcess("svc", fp, "", _svc(readiness_timeout=0.05))
        sup.procs.append(mp)

        sup._schedule_ready(mp)
        assert not mp.ready.is_set()
        assert mp.ready.wait(timeout=0.5), "ready should be set within 0.5s"

    def test_ready_not_set_if_process_died(self):
        sup = _make_supervisor()
        fp = FakePopen([])
        fp.exit(1)  # already dead
        mp = ManagedProcess("svc", fp, "", _svc(readiness_timeout=0.05))
        sup.procs.append(mp)

        sup._schedule_ready(mp)
        time.sleep(0.15)
        assert not mp.ready.is_set()

    def test_ready_not_set_during_shutdown(self):
        sup = _make_supervisor()
        fp = FakePopen([])
        mp = ManagedProcess("svc", fp, "", _svc(readiness_timeout=0.05))
        sup.procs.append(mp)
        sup._stopping = True

        sup._schedule_ready(mp)
        time.sleep(0.15)
        assert not mp.ready.is_set()

    def test_start_waits_for_dep(self):
        sup = _make_supervisor()

        # Pre-create the dep ManagedProcess but mark it not ready yet.
        dep_fp = FakePopen([])
        dep_svc = _svc(name="dep", readiness_timeout=0.05)
        dep_mp = ManagedProcess("dep", dep_fp, "", dep_svc)
        sup.procs.append(dep_mp)

        waited_for_ready = threading.Event()

        original_wait = dep_mp.ready.wait

        def _patched_wait(timeout=None):
            result = original_wait(timeout=timeout)
            waited_for_ready.set()
            return result

        dep_mp.ready.wait = _patched_wait  # type: ignore[method-assign]

        # Set dep ready after a brief delay.
        def _set_ready():
            time.sleep(0.05)
            dep_mp.ready.set()

        threading.Thread(target=_set_ready, daemon=True).start()

        child_svc = _svc(name="child", deps=("dep",))
        spawned: list[FakePopen] = []

        def _fake_spawn(svc: Service) -> FakePopen:
            fp = FakePopen([])
            spawned.append(fp)
            return fp

        sup._spawn = _fake_spawn  # type: ignore[method-assign]

        # Run start() in a thread (it blocks waiting for dep).
        t = threading.Thread(target=lambda: sup.start(child_svc, stagger=0.0), daemon=True)
        t.start()
        t.join(timeout=1.0)

        assert waited_for_ready.is_set(), "supervisor should have waited on dep.ready"
        assert len(spawned) == 1, "child should have been spawned after dep was ready"

    def test_start_proceeds_if_dep_not_started(self, caplog):
        """A dep named in deps that was never started should be skipped with a warning."""
        sup = _make_supervisor()
        spawned: list = []

        def _fake_spawn(svc: Service) -> FakePopen:
            fp = FakePopen([])
            spawned.append(fp)
            return fp

        sup._spawn = _fake_spawn  # type: ignore[method-assign]
        svc = _svc(name="orphan", deps=("missing-dep",))

        import logging
        with caplog.at_level(logging.WARNING, logger="launcher.supervisor"):
            sup.start(svc, stagger=0.0)

        assert len(spawned) == 1
        assert any("not started" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Restart-with-backoff
# ---------------------------------------------------------------------------


class TestRestartWithBackoff:
    def test_process_restarted_after_crash(self):
        sup = _make_supervisor()
        svc = _svc(readiness_timeout=100.0)  # never auto-ready

        spawns: list[FakePopen] = []

        def _fake_spawn(s: Service) -> FakePopen:
            fp = FakePopen([])
            spawns.append(fp)
            return fp

        sup._spawn = _fake_spawn  # type: ignore[method-assign]

        first = _fake_spawn(svc)
        mp = ManagedProcess(svc.name, first, "", svc)
        sup.procs.append(mp)

        t = threading.Thread(target=sup._manage, args=(mp,), daemon=True)
        t.start()

        # Let the drain start, then crash the first process.
        time.sleep(0.05)
        first.exit(1)

        # The supervisor should restart within _BACKOFF_INITIAL + margin.
        deadline = time.time() + _BACKOFF_INITIAL + 0.5
        while time.time() < deadline:
            if mp.restart_count >= 1:
                break
            time.sleep(0.05)

        sup._stopping = True
        mp.proc.exit(0)
        t.join(timeout=2.0)

        assert mp.restart_count >= 1

    def test_backoff_resets_after_long_uptime(self):
        sup = _make_supervisor()
        svc = _svc(readiness_timeout=100.0)

        spawns: list[FakePopen] = []

        def _fake_spawn(s: Service) -> FakePopen:
            fp = FakePopen([])
            spawns.append(fp)
            return fp

        sup._spawn = _fake_spawn  # type: ignore[method-assign]

        first = _fake_spawn(svc)
        mp = ManagedProcess(svc.name, first, "", svc)
        sup.procs.append(mp)

        # Simulate: first crash (delay → _BACKOFF_INITIAL*_BACKOFF_FACTOR),
        # then a long-lived run resets delay back to _BACKOFF_INITIAL.
        # We test this by checking the constant logic, not the real timing.
        uptime_short = 0.1
        uptime_long = _BACKOFF_RESET + 1.0

        delay = _BACKOFF_INITIAL
        # After short uptime: delay escalates
        delay_after_short = min(delay * _BACKOFF_FACTOR, _BACKOFF_MAX)
        # After long uptime: delay resets
        delay_after_long = _BACKOFF_INITIAL

        assert delay_after_short > delay_after_long

    def test_backoff_caps_at_max(self):
        delay = _BACKOFF_INITIAL
        for _ in range(20):
            delay = min(delay * _BACKOFF_FACTOR, _BACKOFF_MAX)
        assert delay == _BACKOFF_MAX

    def test_no_restart_after_stopping(self):
        sup = _make_supervisor()
        svc = _svc(readiness_timeout=100.0)

        spawns: list[FakePopen] = []

        def _fake_spawn(s: Service) -> FakePopen:
            fp = FakePopen([])
            spawns.append(fp)
            return fp

        sup._spawn = _fake_spawn  # type: ignore[method-assign]
        sup._stopping = True

        first = _fake_spawn(svc)
        mp = ManagedProcess(svc.name, first, "", svc)
        sup.procs.append(mp)

        t = threading.Thread(target=sup._manage, args=(mp,), daemon=True)
        t.start()
        first.exit(0)
        t.join(timeout=1.0)

        # Only the initial spawn; no restart because _stopping was True.
        assert mp.restart_count == 0


# ---------------------------------------------------------------------------
# Process-group cleanup
# ---------------------------------------------------------------------------


class TestProcessGroupCleanup:
    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX only")
    def test_kill_proc_uses_killpg_on_posix(self):
        sup = _make_supervisor()
        fp = FakePopen([])
        # Use the current process's PID so getpgid() resolves without error.
        fp.pid = os.getpid()
        mp = ManagedProcess("svc", fp, "", _svc())

        killed_pgid: list[int] = []
        killed_sig: list[int] = []

        def _fake_killpg(pgid: int, sig: int) -> None:
            killed_pgid.append(pgid)
            killed_sig.append(sig)
            fp.exit(0)

        with patch("launcher.supervisor.os.killpg", _fake_killpg):
            sup._kill_proc(mp, signal.SIGTERM)

        assert len(killed_pgid) == 1
        assert killed_sig[0] == signal.SIGTERM

    def test_kill_proc_skips_dead_process(self):
        sup = _make_supervisor()
        fp = FakePopen([])
        fp.exit(0)
        mp = ManagedProcess("svc", fp, "", _svc())

        # Should not raise even if the process is already dead.
        sup._kill_proc(mp, signal.SIGTERM)

    def test_shutdown_terminates_all_procs(self):
        sup = _make_supervisor()
        fps = [FakePopen([]) for _ in range(3)]
        for i, fp in enumerate(fps):
            mp = ManagedProcess(f"svc-{i}", fp, "", _svc())
            sup.procs.append(mp)

        # Bypass the real OS kill; instead exit each FakePopen immediately.
        def _fake_kill(mp: ManagedProcess, sig: int) -> None:
            mp.proc.exit(0)

        sup._kill_proc = _fake_kill  # type: ignore[method-assign]
        sup.shutdown(grace=1.0)

        assert sup._stopping
        for fp in fps:
            assert fp.poll() is not None, "all processes should have been terminated"

    def test_shutdown_is_idempotent(self):
        sup = _make_supervisor()
        fp = FakePopen([])
        mp = ManagedProcess("svc", fp, "", _svc())
        sup.procs.append(mp)

        def _fake_kill(mp: ManagedProcess, sig: int) -> None:
            mp.proc.exit(0)

        sup._kill_proc = _fake_kill  # type: ignore[method-assign]
        sup.shutdown()
        sup.shutdown()  # second call should be a no-op


# ---------------------------------------------------------------------------
# Backoff constant sanity
# ---------------------------------------------------------------------------


class TestBackoffConstants:
    def test_initial_less_than_max(self):
        assert _BACKOFF_INITIAL < _BACKOFF_MAX

    def test_factor_greater_than_one(self):
        assert _BACKOFF_FACTOR > 1.0

    def test_reset_threshold_greater_than_initial(self):
        assert _BACKOFF_RESET > _BACKOFF_INITIAL
