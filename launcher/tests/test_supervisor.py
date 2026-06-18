"""Unit tests for launcher.supervisor — no real subprocesses, no NATS required."""

from __future__ import annotations

import io
import os
import signal
import subprocess
import sys
import tempfile
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
    _SIGKILL,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Use tempfile.gettempdir() so the path is valid on all platforms.
_DATA_DIR = Path(tempfile.gettempdir()) / "engram-supervisor-test"

# Mirror the SIGKILL sentinel from the production module so FakePopen.kill()
# is consistent on Windows (where signal.SIGKILL is not defined).
_SIGKILL_VAL: int = getattr(signal, "SIGKILL", 9)


def _svc(
    name: str = "test-svc",
    deps: tuple[str, ...] = (),
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

    def __init__(self, exit_sequence: list[int | None]) -> None:
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
        self._exit_code = -_SIGKILL_VAL
        self._waited.set()

    def exit(self, code: int = 0) -> None:
        """Simulate the process exiting with the given code."""
        self._exit_code = code
        self._waited.set()


def _make_supervisor(data_dir: Path | None = None) -> Supervisor:
    sup = Supervisor({}, data_dir or _DATA_DIR)
    # Patch _service_env so we don't need real paths.
    sup._service_env = lambda svc: {}  # type: ignore[method-assign]
    return sup


# ---------------------------------------------------------------------------
# ManagedProcess
# ---------------------------------------------------------------------------


class TestManagedProcess:
    def test_initial_state(self) -> None:
        fp = FakePopen([])
        mp = ManagedProcess("svc", fp, "", _svc())
        assert mp.restart_count == 0
        assert not mp.ready.is_set()

    def test_ready_event_starts_unset(self) -> None:
        fp = FakePopen([])
        mp = ManagedProcess("svc", fp, "", _svc())
        assert not mp.ready.is_set()


# ---------------------------------------------------------------------------
# Readiness gating
# ---------------------------------------------------------------------------


class TestReadinessGating:
    def test_ready_set_after_timeout(self) -> None:
        sup = _make_supervisor()
        fp = FakePopen([])
        mp = ManagedProcess("svc", fp, "", _svc(readiness_timeout=0.05))
        sup.procs.append(mp)

        sup._schedule_ready(mp)
        assert not mp.ready.is_set()
        assert mp.ready.wait(timeout=0.5), "ready should be set within 0.5s"

    def test_ready_not_set_if_process_died(self) -> None:
        sup = _make_supervisor()
        fp = FakePopen([])
        fp.exit(1)  # already dead
        mp = ManagedProcess("svc", fp, "", _svc(readiness_timeout=0.05))
        sup.procs.append(mp)

        sup._schedule_ready(mp)
        time.sleep(0.15)
        assert not mp.ready.is_set()

    def test_ready_not_set_during_shutdown(self) -> None:
        sup = _make_supervisor()
        fp = FakePopen([])
        mp = ManagedProcess("svc", fp, "", _svc(readiness_timeout=0.05))
        sup.procs.append(mp)
        sup._stopping = True

        sup._schedule_ready(mp)
        time.sleep(0.15)
        assert not mp.ready.is_set()

    def test_schedule_ready_captures_proc_at_call_time(self) -> None:
        """Ready check uses the proc captured at schedule time, not at wakeup."""
        sup = _make_supervisor()
        fp_old = FakePopen([])
        mp = ManagedProcess("svc", fp_old, "", _svc(readiness_timeout=0.05))
        sup.procs.append(mp)

        # Schedule against fp_old, then replace mp.proc before the timer fires.
        sup._schedule_ready(mp)
        fp_old.exit(1)          # old proc is dead
        fp_new = FakePopen([])  # new proc is alive, but wasn't the captured one
        mp.proc = fp_new

        time.sleep(0.15)
        # fp_old is dead → ready must NOT be set despite fp_new being alive.
        assert not mp.ready.is_set()

    def test_ready_cleared_on_restart(self) -> None:
        """mp.ready is cleared before the new proc is installed."""
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
        mp.ready.set()  # simulate it was ready before the crash
        sup.procs.append(mp)

        t = threading.Thread(target=sup._manage, args=(mp,), daemon=True)
        t.start()

        time.sleep(0.05)
        first.exit(1)

        # restart_count increments at the START of the backoff sleep, before
        # the new proc is installed.  Wait until mp.proc actually changes so
        # we know clear() + spawn have both completed.
        deadline = time.time() + 3.0  # covers _BACKOFF_INITIAL (1 s) + overhead
        while time.time() < deadline and mp.proc is first:
            time.sleep(0.02)

        assert mp.proc is not first, "restart should complete within 3 s"
        # ready must have been cleared (readiness_timeout=100 s, so the new
        # _schedule_ready thread won't re-set it for a very long time).
        assert not mp.ready.is_set()

        sup._stopping = True
        mp.proc.exit(0)
        t.join(timeout=2.0)

    def test_start_waits_for_dep(self) -> None:
        sup = _make_supervisor()

        dep_fp = FakePopen([])
        dep_svc = _svc(name="dep", readiness_timeout=0.05)
        dep_mp = ManagedProcess("dep", dep_fp, "", dep_svc)
        sup.procs.append(dep_mp)

        waited_for_ready = threading.Event()
        original_wait = dep_mp.ready.wait

        def _patched_wait(timeout: float | None = None) -> bool:
            result = original_wait(timeout=timeout)
            waited_for_ready.set()
            return result

        dep_mp.ready.wait = _patched_wait  # type: ignore[method-assign]

        def _set_ready() -> None:
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

        t = threading.Thread(target=lambda: sup.start(child_svc, stagger=0.0), daemon=True)
        t.start()
        t.join(timeout=1.0)

        assert waited_for_ready.is_set(), "supervisor should have waited on dep.ready"
        assert len(spawned) == 1, "child should have been spawned after dep was ready"

    def test_start_proceeds_if_dep_not_started(self, caplog: pytest.LogCaptureFixture) -> None:
        """A dep listed in deps that was never started is skipped with a warning."""
        sup = _make_supervisor()
        spawned: list[FakePopen] = []

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
    def test_process_restarted_after_crash(self) -> None:
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

        t = threading.Thread(target=sup._manage, args=(mp,), daemon=True)
        t.start()

        time.sleep(0.05)
        first.exit(1)

        deadline = time.time() + _BACKOFF_INITIAL + 0.5
        while time.time() < deadline:
            if mp.restart_count >= 1:
                break
            time.sleep(0.05)

        sup._stopping = True
        mp.proc.exit(0)
        t.join(timeout=2.0)

        assert mp.restart_count >= 1

    def test_backoff_resets_after_long_uptime(self, tmp_path: Path) -> None:
        """Process that outlives _BACKOFF_RESET resets the backoff delay."""
        with (
            patch("launcher.supervisor._BACKOFF_RESET", 0.05),
            patch("launcher.supervisor._BACKOFF_INITIAL", 0.05),
            patch("launcher.supervisor._BACKOFF_FACTOR", 2.0),
            patch("launcher.supervisor._BACKOFF_MAX", 2.0),
        ):
            sup = _make_supervisor(tmp_path)
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

            t = threading.Thread(target=sup._manage, args=(mp,), daemon=True)
            t.start()

            # Crash 1 immediately — uptime < BACKOFF_RESET → delay escalates.
            time.sleep(0.01)
            first.exit(1)

            # Wait until the new proc is actually installed (restart_count
            # increments at the start of the backoff sleep, before spawn, so
            # checking restart_count alone is too early).
            deadline = time.time() + 1.0
            while time.time() < deadline and mp.proc is first:
                time.sleep(0.01)
            assert mp.proc is not first, "first restart should complete within 1 s"
            second = mp.proc

            # Let second proc live past BACKOFF_RESET (0.05 s) so the delay resets.
            time.sleep(0.08)
            second.exit(1)

            # Wait until the third proc is installed (second restart complete).
            deadline = time.time() + 1.0
            while time.time() < deadline and mp.proc is second:
                time.sleep(0.01)
            assert mp.proc is not second, "second restart should complete within 1 s (reset delay = 0.05 s)"
            assert mp.restart_count >= 2, "restart_count should reflect both restarts"

            sup._stopping = True
            mp.proc.exit(0)
            t.join(timeout=2.0)

    def test_backoff_caps_at_max(self) -> None:
        delay = _BACKOFF_INITIAL
        for _ in range(20):
            delay = min(delay * _BACKOFF_FACTOR, _BACKOFF_MAX)
        assert delay == _BACKOFF_MAX

    def test_no_restart_after_stopping(self) -> None:
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

        assert mp.restart_count == 0


# ---------------------------------------------------------------------------
# Process-group cleanup
# ---------------------------------------------------------------------------


class TestProcessGroupCleanup:
    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX only")
    def test_kill_proc_uses_killpg_on_posix(self) -> None:
        sup = _make_supervisor()
        fp = FakePopen([])
        fp.pid = os.getpid()  # real PID so os.getpgid() resolves
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

    def test_kill_proc_skips_dead_process(self) -> None:
        sup = _make_supervisor()
        fp = FakePopen([])
        fp.exit(0)
        mp = ManagedProcess("svc", fp, "", _svc())

        sup._kill_proc(mp, signal.SIGTERM)  # must not raise

    def test_shutdown_terminates_all_procs(self) -> None:
        sup = _make_supervisor()
        fps = [FakePopen([]) for _ in range(3)]
        for i, fp in enumerate(fps):
            mp = ManagedProcess(f"svc-{i}", fp, "", _svc())
            sup.procs.append(mp)

        def _fake_kill(mp: ManagedProcess, sig: int) -> None:
            mp.proc.exit(0)

        sup._kill_proc = _fake_kill  # type: ignore[method-assign]
        sup.shutdown(grace=1.0)

        assert sup._stopping
        for fp in fps:
            assert fp.poll() is not None, "all processes should have been terminated"

    def test_shutdown_is_idempotent(self) -> None:
        sup = _make_supervisor()
        fp = FakePopen([])
        mp = ManagedProcess("svc", fp, "", _svc())
        sup.procs.append(mp)

        def _fake_kill(mp: ManagedProcess, sig: int) -> None:
            mp.proc.exit(0)

        sup._kill_proc = _fake_kill  # type: ignore[method-assign]
        sup.shutdown()
        sup.shutdown()  # second call must be a no-op

    def test_sigkill_constant_is_safe_on_all_platforms(self) -> None:
        """_SIGKILL is always an int regardless of platform."""
        assert isinstance(_SIGKILL, int)


# ---------------------------------------------------------------------------
# Backoff constant sanity
# ---------------------------------------------------------------------------


class TestBackoffConstants:
    def test_initial_less_than_max(self) -> None:
        assert _BACKOFF_INITIAL < _BACKOFF_MAX

    def test_factor_greater_than_one(self) -> None:
        assert _BACKOFF_FACTOR > 1.0

    def test_reset_threshold_greater_than_initial(self) -> None:
        assert _BACKOFF_RESET > _BACKOFF_INITIAL
