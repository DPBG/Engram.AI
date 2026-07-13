"""Process supervisor — spawn services as subprocesses and manage their lifecycle.

This is the pure-Python replacement for `docker compose up`: it launches each
service with `python -m <module>`, wires up the right environment and
PYTHONPATH, streams every service's output to the console with a name prefix,
restarts crashed services with exponential backoff, gates service startup on
declared dependencies being ready, and shuts everything down cleanly on Ctrl+C.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from .registry import ROOT, Service
from .supervisor_alerts import (
    emit_service_flap_alert,
    flap_threshold,
    flap_window_seconds,
    record_restart,
    should_emit_flap_alert,
)

logger = logging.getLogger("launcher.supervisor")

# Simple ANSI colors for log prefixes (disabled automatically when not a TTY).
_COLORS = [
    "\033[36m",
    "\033[32m",
    "\033[33m",
    "\033[35m",
    "\033[34m",
    "\033[31m",
    "\033[96m",
    "\033[92m",
]
_RESET = "\033[0m"
_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    """Read a float tuning override from the environment (issue #250).

    Missing, non-numeric, or out-of-range (``<= minimum``) values fall back to
    ``default`` with a warning, so an operator typo can never disable restart
    backoff (e.g. a 0s delay hammering a crash-looping service).
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r (not a number); using default %s", name, raw, default)
        return default
    if value <= minimum:
        logger.warning(
            "Invalid %s=%r (must be > %s); using default %s", name, raw, minimum, default
        )
        return default
    return value


# Restart-with-backoff parameters. Each is overridable per-deployment via an
# environment variable (issue #250), defaulting to the historical value so
# behavior is unchanged unless an operator tunes it. Resolved once at import.
_BACKOFF_INITIAL = _env_float("SUPERVISOR_BACKOFF_INITIAL_S", 1.0)  # secs before 1st restart
_BACKOFF_FACTOR = _env_float("SUPERVISOR_BACKOFF_FACTOR", 2.0, minimum=1.0)  # per-crash multiplier
_BACKOFF_MAX = _env_float("SUPERVISOR_BACKOFF_MAX_S", 30.0)  # cap on per-restart delay
_BACKOFF_RESET = _env_float("SUPERVISOR_BACKOFF_RESET_S", 10.0)  # uptime that resets backoff

# signal.SIGKILL is not available on Windows; fall back to SIGTERM so the name
# is always safe to reference.  _kill_proc uses proc.kill() on Windows anyway.
_SIGKILL: int = getattr(signal, "SIGKILL", signal.SIGTERM)


class ManagedProcess:
    def __init__(self, name: str, proc: subprocess.Popen, color: str, svc: Service) -> None:
        self.name = name
        self.proc = proc
        self.color = color
        self.svc = svc
        # Set once the process has been alive for svc.readiness_timeout seconds.
        self.ready = threading.Event()
        self.restart_count = 0
        # Rolling restart timestamps (monotonic) for flap detection (issue #261).
        self.restart_times: list[float] = []
        self.flap_alerts_emitted = 0


class Supervisor:
    def __init__(
        self,
        base_env: dict,
        data_dir: Path,
        creds_dir: Path | None = None,
    ) -> None:
        self.base_env = base_env
        self.data_dir = data_dir
        self.sqlite_dir = data_dir / "sqlite"
        # Directory containing per-service .creds files (secrets/<name>.creds).
        # When set, _service_env() injects NATS_CREDS for each service whose
        # creds file exists. Missing files are silently skipped (dev fallback).
        self.creds_dir = creds_dir
        self.procs: list[ManagedProcess] = []
        self._print_lock = threading.Lock()
        self._stopping = False

    # -- environment ---------------------------------------------------------

    def _service_env(self, svc: Service) -> dict:
        env = dict(self.base_env)
        env["PYTHONPATH"] = svc.pythonpath()
        env["PYTHONUNBUFFERED"] = "1"

        basename = svc.env.get("SQLITE_PATH_BASENAME", "unified.db")
        env["SQLITE_PATH"] = str(self.sqlite_dir / basename)

        for key, value in svc.env.items():
            if key == "SQLITE_PATH_BASENAME":
                continue
            env[key] = value

        # Inject per-service NATS credentials when the creds file exists.
        # The SDK logs a warning and falls back gracefully when the path is set
        # but the file is absent, so we only set it when the file is present.
        if self.creds_dir is not None:
            creds_file = self.creds_dir / f"{svc.name}.creds"
            if creds_file.is_file():
                env["NATS_CREDS"] = str(creds_file)
                logger.debug("injecting NATS_CREDS=%s for %s", creds_file, svc.name)

        return env

    # -- output streaming ----------------------------------------------------

    def _prefix(self, mp: ManagedProcess) -> str:
        if _USE_COLOR:
            return f"{mp.color}{mp.name:>17}{_RESET} | "
        return f"{mp.name:>17} | "

    def _drain(self, mp: ManagedProcess) -> None:
        """Read stdout until EOF, printing each line with a service prefix."""
        prefix = self._prefix(mp)
        stdout = mp.proc.stdout
        assert stdout is not None
        for raw in stdout:
            with self._print_lock:
                print(prefix + raw.rstrip("\n"), flush=True)

    # -- process spawning ----------------------------------------------------

    def _spawn(self, svc: Service) -> subprocess.Popen:
        """Start the service as a subprocess.

        On POSIX, start_new_session=True places the child in its own process
        group so that os.killpg() during shutdown reaches all grandchildren.
        """
        env = self._service_env(svc)
        cmd = [sys.executable, "-u", "-m", svc.module, *svc.args]
        kwargs: dict = dict(
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if sys.platform != "win32":
            kwargs["start_new_session"] = True
        return subprocess.Popen(cmd, **kwargs)

    # -- readiness -----------------------------------------------------------

    def _schedule_ready(self, mp: ManagedProcess) -> None:
        """Spawn a background thread that sets mp.ready after the readiness timeout.

        The specific proc is captured at call time so that a restart replacing
        mp.proc between schedule and wakeup does not cause us to check the
        wrong process's liveness.
        """
        proc = mp.proc  # capture now — mp.proc may be replaced by a restart

        def _check() -> None:
            time.sleep(mp.svc.readiness_timeout)
            if proc.poll() is None and not self._stopping:
                mp.ready.set()
                logger.debug("%s is ready", mp.name)

        threading.Thread(target=_check, daemon=True).start()

    # -- lifecycle -----------------------------------------------------------

    def _manage(self, mp: ManagedProcess) -> None:
        """Output pump + restart-with-backoff loop for one managed service."""
        prefix = self._prefix(mp)
        delay = _BACKOFF_INITIAL
        self._schedule_ready(mp)

        while True:
            started_at = time.monotonic()

            # Drain stdout in a background thread so that proc.wait() is never
            # blocked by a grandchild that keeps the pipe open after the main
            # service process has already exited.
            drain_thread = threading.Thread(target=self._drain, args=(mp,), daemon=True)
            drain_thread.start()
            code = mp.proc.wait()
            uptime = time.monotonic() - started_at
            drain_thread.join(timeout=2.0)  # brief flush window for buffered output

            if self._stopping:
                return

            with self._print_lock:
                print(f"{prefix}*** exited (code {code}) ***", flush=True)

            # Long-lived processes get a fresh backoff budget and flap history.
            if uptime >= _BACKOFF_RESET:
                delay = _BACKOFF_INITIAL
                mp.restart_times.clear()
                mp.flap_alerts_emitted = 0

            actual_delay = min(delay, _BACKOFF_MAX)
            mp.restart_count += 1
            now = time.monotonic()
            window_restarts = record_restart(
                mp.restart_times,
                now=now,
                window_seconds=flap_window_seconds(),
            )
            threshold = flap_threshold()
            if should_emit_flap_alert(len(window_restarts), threshold, mp.flap_alerts_emitted):
                alert = emit_service_flap_alert(
                    service_name=mp.name,
                    restart_count=mp.restart_count,
                    restarts_in_window=len(window_restarts),
                    window_seconds=flap_window_seconds(),
                    threshold=threshold,
                    exit_code=code,
                    uptime_seconds=uptime,
                )
                mp.flap_alerts_emitted = len(window_restarts) // threshold
                with self._print_lock:
                    print(
                        f"{prefix}*** ALERT: {alert['message']} ***",
                        flush=True,
                    )

            logger.info(
                "restarting %s in %.1fs (attempt %d)",
                mp.name,
                actual_delay,
                mp.restart_count,
            )

            # Interruptible sleep — exits immediately on shutdown.
            deadline = time.monotonic() + actual_delay
            while time.monotonic() < deadline:
                if self._stopping:
                    return
                time.sleep(0.1)

            # Guard against a shutdown that raced with the sleep above.
            if self._stopping:
                return

            try:
                new_proc = self._spawn(mp.svc)
            except Exception as exc:
                logger.error("failed to restart %s: %s", mp.name, exc)
                return

            # Clear ready so dependents re-wait if they ever re-check, and so
            # _schedule_ready below reflects the new proc's uptime, not the old one's.
            mp.ready.clear()
            mp.proc = new_proc

            with self._print_lock:
                print(f"{prefix}*** restarted (attempt {mp.restart_count}) ***", flush=True)

            delay = min(delay * _BACKOFF_FACTOR, _BACKOFF_MAX)
            self._schedule_ready(mp)

    def start(self, svc: Service, stagger: float = 0.4) -> None:
        """Wait for declared deps to be ready, then spawn and manage the service."""
        for dep_name in svc.deps:
            dep = next((m for m in self.procs if m.name == dep_name), None)
            if dep is None:
                logger.warning(
                    "dep %r of %s was not started — skipping readiness wait",
                    dep_name,
                    svc.name,
                )
                continue
            wait_secs = dep.svc.readiness_timeout + 5.0
            logger.info("waiting for %s (dep of %s)…", dep_name, svc.name)
            if not dep.ready.wait(timeout=wait_secs):
                logger.warning(
                    "%s dep %r not ready after %.1fs — starting anyway",
                    svc.name,
                    dep_name,
                    wait_secs,
                )

        color = _COLORS[len(self.procs) % len(_COLORS)]
        logger.info("starting %s  (python -m %s)", svc.name, svc.module)
        proc = self._spawn(svc)
        mp = ManagedProcess(svc.name, proc, color, svc)
        self.procs.append(mp)
        threading.Thread(target=self._manage, args=(mp,), daemon=True).start()
        if stagger:
            time.sleep(stagger)

    def wait(self) -> None:
        """Block until interrupted; then shut everything down."""
        try:
            while not self._stopping:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print()  # newline after ^C
            logger.info("Ctrl+C received — shutting down...")
        finally:
            self.shutdown()

    # -- shutdown ------------------------------------------------------------

    def _kill_proc(self, mp: ManagedProcess, sig: int) -> None:
        """Send a signal to the process group (POSIX) or the process (Windows)."""
        proc = mp.proc
        if proc.poll() is not None:
            return
        try:
            if sys.platform != "win32":
                os.killpg(os.getpgid(proc.pid), sig)
            elif sig == signal.SIGTERM:
                proc.terminate()
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError):
            pass
        except Exception as exc:
            logger.debug("_kill_proc %s: %s", mp.name, exc)

    def shutdown(self, grace: float = 6.0) -> None:
        if self._stopping:
            return
        self._stopping = True
        for mp in self.procs:
            logger.info("stopping %s", mp.name)
            self._kill_proc(mp, signal.SIGTERM)
        deadline = time.monotonic() + grace
        for mp in self.procs:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                mp.proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                logger.warning("force-killing %s", mp.name)
                self._kill_proc(mp, _SIGKILL)
