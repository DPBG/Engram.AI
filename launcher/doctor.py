"""Preflight environment diagnostics for the pure-Python launcher.

``python run.py --doctor`` runs a battery of read-only checks and prints a
report so a contributor can see — before trying to boot the stack — whether
their machine is set up correctly: the right Python, a usable ``nats-server``,
free ports, reachable optional infrastructure (Qdrant / Ollama), writable data
directories, enough disk, and a consistent service registry.

Design notes
------------
- Every check is a small pure function returning a :class:`Check`, and the
  environment probes (port checks, HTTP reachability) are injected so the whole
  module is unit-testable offline — no real sockets or network required.
- Checks are graded ``ok`` / ``warn`` / ``fail``. Only a ``fail`` makes the
  command exit non-zero; optional infrastructure being down is a ``warn`` (the
  launcher already skips services that need it), so ``--doctor`` stays green on
  a minimal "core profile" machine.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from launcher import nats_server
from launcher.registry import ROOT, SERVICES, Service

# Minimum interpreter the project targets (see CONTRIBUTING.md / pyproject).
MIN_PYTHON = (3, 11)
# Default free space we want under data/ before a run is comfortable.
MIN_FREE_GB = 1.0

OK = "ok"
WARN = "warn"
FAIL = "fail"

_MARKERS = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]"}


@dataclass(frozen=True)
class Check:
    """One diagnostic result."""

    name: str
    status: str  # OK | WARN | FAIL
    detail: str


# --- probe defaults (injectable for tests) -----------------------------------

def _default_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """True if something is already listening on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def _default_http_ok(url: str, timeout: float = 1.5) -> bool:
    """True if an HTTP GET to ``url`` gets any answer below 500."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status < 500
    except urllib.error.HTTPError as exc:
        return exc.code < 500  # server answered, just not 200
    except Exception:
        return False


PortOpen = Callable[[str, int], bool]
HttpOk = Callable[[str], bool]


# --- individual checks --------------------------------------------------------

def check_python_version(version_info: tuple = sys.version_info) -> Check:
    have = (version_info[0], version_info[1])
    pretty = f"{have[0]}.{have[1]}"
    want = f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
    if have < MIN_PYTHON:
        return Check(
            "python", FAIL,
            f"Python {pretty} is too old — Engram targets {want}+.",
        )
    return Check("python", OK, f"Python {pretty} (>= {want}).")


def check_nats_binary(
    find: Callable[[], Optional[Path]] = nats_server._find_existing_binary,
    port_open: PortOpen = _default_port_open,
) -> Check:
    """A usable NATS is: already listening, or a binary present on disk/PATH.

    A *missing* binary is only a ``warn`` — the launcher downloads it on first
    run — but we surface it so an offline contributor isn't surprised.
    """
    if port_open("127.0.0.1", nats_server.CLIENT_PORT):
        return Check("nats", OK, f"NATS already listening on :{nats_server.CLIENT_PORT}.")
    binary = find()
    if binary is not None:
        return Check("nats", OK, f"nats-server binary found at {binary}.")
    return Check(
        "nats", WARN,
        "no nats-server on PATH or in .localrun/ — the launcher will download "
        f"{nats_server.NATS_VERSION} on first run (needs internet).",
    )


def check_port_free(
    label: str,
    port: int,
    port_open: PortOpen = _default_port_open,
) -> Check:
    """A port already in use by NATS is fine; by anything else is a conflict.

    We cannot tell *who* holds the port, so an occupied client/monitor port is
    reported as a ``warn`` (it is reused if it's our own NATS, a conflict if
    not) rather than a hard failure.
    """
    if port_open("127.0.0.1", port):
        return Check(
            f"port:{label}", WARN,
            f"port {port} ({label}) is in use — reused if it's NATS, a conflict otherwise.",
        )
    return Check(f"port:{label}", OK, f"port {port} ({label}) is free.")


def check_optional_infra(
    name: str,
    url: str,
    http_ok: HttpOk = _default_http_ok,
) -> Check:
    """Qdrant / Ollama are optional — down is a ``warn``, not a ``fail``."""
    if http_ok(url):
        return Check(name, OK, f"{name} reachable at {url}.")
    return Check(
        name, WARN,
        f"{name} not reachable at {url} — services needing it are skipped "
        "(fine for the core profile).",
    )


def check_data_dirs(root: Path = ROOT) -> Check:
    """The launcher writes SQLite/tasks/plugins under data/ — must be writable."""
    data = root / "data"
    target = data if data.exists() else root
    if not os.access(target, os.W_OK):
        return Check("data-dir", FAIL, f"{target} is not writable.")
    return Check("data-dir", OK, f"{data} is writable.")


def check_disk_space(root: Path = ROOT, min_free_gb: float = MIN_FREE_GB) -> Check:
    try:
        free_gb = shutil.disk_usage(root).free / (1024 ** 3)
    except OSError as e:
        return Check("disk", WARN, f"could not determine free space: {e}")
    if free_gb < min_free_gb:
        return Check(
            "disk", WARN,
            f"only {free_gb:.1f} GB free under {root} (< {min_free_gb:.0f} GB recommended).",
        )
    return Check("disk", OK, f"{free_gb:.1f} GB free.")


def check_service_sources(services: Iterable[Service] = SERVICES) -> Check:
    """Every registered service must point at a source dir that exists."""
    missing = [s.name for s in services if not s.src_path.exists()]
    if missing:
        return Check(
            "services", FAIL,
            f"registry references missing source dirs: {', '.join(missing)}.",
        )
    count = sum(1 for _ in services) if not isinstance(services, list) else len(services)
    return Check("services", OK, f"all {count} registered service source dirs present.")


# --- orchestration ------------------------------------------------------------

def run_checks(
    port_open: PortOpen = _default_port_open,
    http_ok: HttpOk = _default_http_ok,
    qdrant_url: Optional[str] = None,
    ollama_url: Optional[str] = None,
) -> list[Check]:
    """Run the full battery and return the results in display order."""
    qdrant_url = qdrant_url or os.environ.get("QDRANT_URL", "http://localhost:6333")
    ollama_url = ollama_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")
    return [
        check_python_version(),
        check_service_sources(),
        check_nats_binary(port_open=port_open),
        check_port_free("nats-client", nats_server.CLIENT_PORT, port_open=port_open),
        check_port_free("nats-monitor", nats_server.MONITOR_PORT, port_open=port_open),
        check_optional_infra("qdrant", f"{qdrant_url}/healthz", http_ok=http_ok),
        check_optional_infra("ollama", f"{ollama_url}/api/tags", http_ok=http_ok),
        check_data_dirs(),
        check_disk_space(),
    ]


def format_report(checks: list[Check]) -> str:
    """Render checks as an aligned, human-readable report."""
    width = max((len(c.name) for c in checks), default=0)
    lines = ["", "Engram environment check", "=" * 64]
    for c in checks:
        marker = _MARKERS.get(c.status, "[ ?? ]")
        lines.append(f"  {marker}  {c.name.ljust(width)}  {c.detail}")
    fails = sum(1 for c in checks if c.status == FAIL)
    warns = sum(1 for c in checks if c.status == WARN)
    lines.append("=" * 64)
    if fails:
        lines.append(f"  {fails} blocking problem(s) and {warns} warning(s). "
                     "Fix the [FAIL] items before running.")
    elif warns:
        lines.append(f"  Ready to run (core profile). {warns} warning(s) — "
                     "see notes above for optional features.")
    else:
        lines.append("  All checks passed — you're good to go.")
    lines.append("")
    return "\n".join(lines)


def exit_code(checks: list[Check]) -> int:
    """Non-zero only if a check hard-failed; warnings stay green."""
    return 1 if any(c.status == FAIL for c in checks) else 0


def doctor() -> int:
    """Entry point used by ``run.py --doctor``: run checks, print, return code."""
    checks = run_checks()
    print(format_report(checks))
    return exit_code(checks)
