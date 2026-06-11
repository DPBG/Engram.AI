"""Process supervisor — spawn services as subprocesses and manage their lifecycle.

This is the pure-Python replacement for `docker compose up`: it launches each
service with `python -m <module>`, wires up the right environment and
PYTHONPATH, streams every service's output to the console with a name prefix,
and shuts everything down cleanly on Ctrl+C.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from .registry import ROOT, Service

logger = logging.getLogger("launcher.supervisor")

# Simple ANSI colors for log prefixes (disabled automatically when not a TTY).
_COLORS = [
    "\033[36m", "\033[32m", "\033[33m", "\033[35m",
    "\033[34m", "\033[31m", "\033[96m", "\033[92m",
]
_RESET = "\033[0m"
_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


class ManagedProcess:
    def __init__(self, name: str, proc: subprocess.Popen, color: str):
        self.name = name
        self.proc = proc
        self.color = color


class Supervisor:
    def __init__(self, base_env: dict, data_dir: Path):
        self.base_env = base_env
        self.data_dir = data_dir
        self.sqlite_dir = data_dir / "sqlite"
        self.procs: list[ManagedProcess] = []
        self._print_lock = threading.Lock()
        self._stopping = False

    # -- environment ---------------------------------------------------------
    def _service_env(self, svc: Service) -> dict:
        env = dict(self.base_env)
        env["PYTHONPATH"] = svc.pythonpath()
        env["PYTHONUNBUFFERED"] = "1"

        # SQLite path: shared unified.db unless the service asks for its own.
        basename = svc.env.get("SQLITE_PATH_BASENAME", "unified.db")
        env["SQLITE_PATH"] = str(self.sqlite_dir / basename)

        for key, value in svc.env.items():
            if key == "SQLITE_PATH_BASENAME":
                continue
            env[key] = value
        return env

    # -- output streaming ----------------------------------------------------
    def _pump(self, mp: ManagedProcess) -> None:
        prefix = f"{mp.name:>17} | "
        if _USE_COLOR:
            prefix = f"{mp.color}{mp.name:>17}{_RESET} | "
        assert mp.proc.stdout is not None
        for raw in mp.proc.stdout:
            line = raw.rstrip("\n")
            with self._print_lock:
                print(prefix + line, flush=True)
        code = mp.proc.wait()
        if not self._stopping:
            with self._print_lock:
                print(f"{prefix}*** exited with code {code} ***", flush=True)

    # -- lifecycle -----------------------------------------------------------
    def start(self, svc: Service, stagger: float = 0.4) -> None:
        env = self._service_env(svc)
        cmd = [sys.executable, "-u", "-m", svc.module, *svc.args]
        color = _COLORS[len(self.procs) % len(_COLORS)]
        logger.info("starting %s  (python -m %s)", svc.name, svc.module)
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        mp = ManagedProcess(svc.name, proc, color)
        self.procs.append(mp)
        threading.Thread(target=self._pump, args=(mp,), daemon=True).start()
        if stagger:
            time.sleep(stagger)

    def any_alive(self) -> bool:
        return any(mp.proc.poll() is None for mp in self.procs)

    def wait(self) -> None:
        """Block until interrupted; then shut everything down."""
        try:
            while self.any_alive():
                time.sleep(0.5)
        except KeyboardInterrupt:
            print()  # newline after ^C
            logger.info("Ctrl+C received — shutting down...")
        finally:
            self.shutdown()

    def shutdown(self, grace: float = 6.0) -> None:
        if self._stopping:
            return
        self._stopping = True
        # Polite terminate first.
        for mp in self.procs:
            if mp.proc.poll() is None:
                logger.info("stopping %s", mp.name)
                try:
                    mp.proc.terminate()
                except Exception:
                    pass
        deadline = time.monotonic() + grace
        for mp in self.procs:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                mp.proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                logger.warning("force-killing %s", mp.name)
                try:
                    mp.proc.kill()
                except Exception:
                    pass
