#!/usr/bin/env python3
"""Engram local launcher — run the whole stack with pure Python (no Docker).

Usage
-----
    python run.py --install              # one-time: install Python deps
    python run.py                        # start the 'core' profile
    python run.py --profile full         # add Qdrant/Ollama-backed services
    python run.py --only kernel,planner  # start a specific subset
    python run.py --list                 # show services and what they need

NATS is downloaded and managed automatically. Qdrant and Ollama are optional:
services that need them are skipped (with a warning) unless those servers are
reachable. See RUN-LOCAL.md for the full guide.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from launcher import nats_server
from launcher.registry import ROOT, get_service, services_for_profile, PROFILES
from launcher.supervisor import Supervisor

# Windows consoles default to cp1252; make our (and children's) output UTF-8 safe.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DATA_DIR = ROOT / "data"

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("launcher")


def _http_ok(url: str, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status < 500
    except urllib.error.HTTPError as exc:
        return exc.code < 500  # server answered, just not 200
    except Exception:
        return False


def detect_infra() -> dict[str, bool]:
    return {
        "nats": nats_server.port_open("127.0.0.1", nats_server.CLIENT_PORT),
        "qdrant": _http_ok(f"{QDRANT_URL}/healthz") or _http_ok(QDRANT_URL),
        "ollama": _http_ok(f"{OLLAMA_URL}/api/tags"),
    }


def base_env() -> dict:
    env = dict(os.environ)
    env.setdefault("NATS_URL", "nats://localhost:4222")
    env.setdefault("QDRANT_URL", QDRANT_URL)
    env.setdefault("OLLAMA_URL", OLLAMA_URL)
    env.setdefault("LOG_LEVEL", os.environ.get("LOG_LEVEL", "INFO"))
    env["PYTHONIOENCODING"] = "utf-8"  # children emit UTF-8 regardless of locale
    return env


def cmd_install() -> int:
    req = ROOT / "requirements-local.txt"
    log.info("Installing Python dependencies from %s", req.name)
    rc = subprocess.call(
        [sys.executable, "-m", "pip", "install", "-r", str(req)]
    )
    if rc == 0:
        log.info("Dependencies installed.")
    else:
        log.error("pip install failed (exit %d).", rc)
    return rc


def cmd_list() -> int:
    print("\nProfiles: " + ", ".join(PROFILES) + "   (default: core)\n")
    for profile in PROFILES:
        print(f"=== profile: {profile} ===")
        for svc in services_for_profile(profile):
            if profile != "core" and svc.profile == "core":
                continue  # avoid repeating core services under full/all
            needs = []
            if svc.needs_qdrant:
                needs.append("qdrant")
            if svc.needs_ollama:
                needs.append("ollama")
            tag = f"  [needs: {', '.join(needs)}]" if needs else ""
            print(f"  {svc.name:<18} {svc.note}{tag}")
        print()
    print("Not launchable without Docker: meta-programmer (needs Docker socket)\n")
    return 0


def resolve_services(args) -> list:
    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
        out = []
        for n in names:
            svc = get_service(n)
            if svc is None:
                log.error("Unknown service %r. Try --list.", n)
                return []
            out.append(svc)
        return out
    return services_for_profile(args.profile)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run.py", description="Run Engram locally with pure Python (no Docker)."
    )
    parser.add_argument(
        "--profile", default="core", choices=list(PROFILES),
        help="which set of services to run (default: core)",
    )
    parser.add_argument(
        "--only", help="comma-separated service names to run instead of a profile",
    )
    parser.add_argument("--list", action="store_true", help="list services and exit")
    parser.add_argument(
        "--install", action="store_true", help="pip install dependencies and exit",
    )
    parser.add_argument(
        "--no-nats", action="store_true",
        help="don't download/manage NATS (assume it's already running)",
    )
    parser.add_argument(
        "--skip-infra-check", action="store_true",
        help="start qdrant/ollama-dependent services even if not detected",
    )
    parser.add_argument(
        "--video", metavar="PATH",
        help="video file/URL to stream into the brain via sensory-gateway "
             "(sets SENSORY_VIDEO)",
    )
    args = parser.parse_args()

    if args.video:
        os.environ["SENSORY_VIDEO"] = args.video

    if args.list:
        return cmd_list()
    if args.install:
        return cmd_install()

    services = resolve_services(args)
    if not services:
        return 1

    # --- data directories ---
    (DATA_DIR / "sqlite").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "tasks").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "plugins").mkdir(parents=True, exist_ok=True)

    # --- NATS ---
    nats_proc = None
    if not args.no_nats:
        nats_proc = nats_server.start(log_file=DATA_DIR / "nats.log")
        if not nats_server.wait_ready():
            log.error("NATS did not become ready. Aborting.")
            if nats_proc:
                nats_proc.terminate()
            return 1
        log.info("NATS is ready on port %d.", nats_server.CLIENT_PORT)
    elif not nats_server.port_open("127.0.0.1", nats_server.CLIENT_PORT):
        log.warning("--no-nats set but nothing is listening on 4222; services will retry.")

    # --- optional infra detection ---
    infra = detect_infra()
    log.info(
        "Infra: NATS=%s  Qdrant=%s  Ollama=%s",
        "up" if infra["nats"] else "down",
        "up" if infra["qdrant"] else "down",
        "up" if infra["ollama"] else "down",
    )

    runnable = []
    for svc in services:
        if not args.skip_infra_check:
            if svc.needs_qdrant and not infra["qdrant"]:
                log.warning("skipping %s — Qdrant not reachable at %s", svc.name, QDRANT_URL)
                continue
            if svc.needs_ollama and not infra["ollama"]:
                log.warning("skipping %s — Ollama not reachable at %s", svc.name, OLLAMA_URL)
                continue
        if svc.name == "sensory-gateway" and not os.environ.get("SENSORY_VIDEO"):
            log.warning(
                "skipping sensory-gateway — no video given. Pass --video PATH "
                "(it runs with --no-camera --no-mic, so it needs a video source)."
            )
            continue
        runnable.append(svc)

    if not runnable:
        log.error("No services left to start.")
        if nats_proc:
            nats_proc.terminate()
        return 1

    # --- launch ---
    sup = Supervisor(base_env(), DATA_DIR)
    log.info("Starting %d service(s): %s", len(runnable), ", ".join(s.name for s in runnable))
    for svc in runnable:
        sup.start(svc)

    print("\n" + "=" * 64)
    print("  Engram is running.  Dashboard: http://localhost:8080")
    print("  Press Ctrl+C to stop everything.")
    print("=" * 64 + "\n")

    try:
        sup.wait()
    finally:
        if nats_proc and nats_proc.poll() is None:
            log.info("stopping NATS")
            nats_proc.terminate()
            try:
                nats_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                nats_proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
