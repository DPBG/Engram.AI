"""
Sandbox containment integration tests (E1.3.3).

Each test proves one containment property with an adversarial two-step check:

  1. Baseline  — the hostile payload SUCCEEDS when its specific guard is absent
                 (remaining guards kept, one guard removed or relaxed).
  2. Guarded   — the same payload is BLOCKED under full production hardening.

This structure makes each test self-proving: if you remove a guard from step 2,
the result matches step 1 (payload succeeds), which breaks step 2's assertion —
i.e. the test fails exactly when its guard is removed.

Flags mirror sandbox_manager.py and sandbox/smoke/run_smoke.sh exactly.

Requirements: Docker daemon running AND activelearning-sandbox:latest built.
  Build: docker compose --profile build-only build sandbox-base
Skip marker at module level skips every test when either requirement is absent.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import time

import pytest

IMAGE = "activelearning-sandbox:latest"


def _docker_available() -> bool:
    """Return True only if the daemon is up AND the sandbox image exists."""
    if not shutil.which("docker"):
        return False
    try:
        info = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
        if info.returncode != 0:
            return False
        img = subprocess.run(
            ["docker", "image", "inspect", IMAGE],
            capture_output=True,
            timeout=10,
        )
        return img.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


pytestmark = pytest.mark.skipif(
    not _docker_available(),
    reason=(
        "Docker daemon unavailable or activelearning-sandbox:latest not built. "
        "Run: docker compose --profile build-only build sandbox-base"
    ),
)


def _run(flags: list[str], payload: str) -> subprocess.CompletedProcess[str]:
    """Run a Python one-liner inside an --rm container; return the completed process."""
    cmd = ["docker", "run", "--rm"] + flags + [IMAGE, "python", "-c", payload]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60)


# ── Production hardening (mirrors sandbox_manager.py + smoke/run_smoke.sh) ───

_ALL = [
    "--network",
    "none",
    "--read-only",
    "--cap-drop",
    "ALL",
    "--security-opt",
    "no-new-privileges",
    "--pids-limit",
    "100",
    "--memory",
    "512m",
    "--memory-swap",
    "512m",  # same as --memory → zero swap, hard 512 MB limit
    "--cpus",
    "0.5",
    "--tmpfs",
    "/tmp:size=50M",
]

# ── Per-test baselines: one guard removed or relaxed, all others kept ─────────

# Network test baseline: --network none removed → default bridge, outbound allowed.
_WITHOUT_NETWORK = [
    "--read-only",
    "--cap-drop",
    "ALL",
    "--security-opt",
    "no-new-privileges",
    "--pids-limit",
    "100",
    "--memory",
    "512m",
    "--cpus",
    "0.5",
    "--tmpfs",
    "/tmp:size=50M",
]

# Read-only test baseline: --read-only removed → writable root filesystem.
_WITHOUT_READONLY = [
    "--network",
    "none",
    "--cap-drop",
    "ALL",
    "--security-opt",
    "no-new-privileges",
    "--pids-limit",
    "100",
    "--memory",
    "512m",
    "--cpus",
    "0.5",
    "--tmpfs",
    "/tmp:size=50M",
]

# PIDs test baseline: limit raised to 500 → 200-process payload spawns freely.
_RELAXED_PIDS = [
    "--network",
    "none",
    "--read-only",
    "--cap-drop",
    "ALL",
    "--security-opt",
    "no-new-privileges",
    "--pids-limit",
    "500",
    "--memory",
    "512m",
    "--cpus",
    "0.5",
    "--tmpfs",
    "/tmp:size=50M",
]

# Memory test baseline: limit raised to 2g, swap unlimited → 600 MB allocation succeeds.
_RELAXED_MEMORY = [
    "--network",
    "none",
    "--read-only",
    "--cap-drop",
    "ALL",
    "--security-opt",
    "no-new-privileges",
    "--pids-limit",
    "100",
    "--memory",
    "2g",
    "--memory-swap",
    "-1",  # unlimited swap for baseline
    "--cpus",
    "0.5",
    "--tmpfs",
    "/tmp:size=50M",
]

# Privilege test baseline: cap_drop and no-new-privileges both absent.
_WITHOUT_PRIVDROP = [
    "--network",
    "none",
    "--read-only",
    "--pids-limit",
    "100",
    "--memory",
    "512m",
    "--cpus",
    "0.5",
    "--tmpfs",
    "/tmp:size=50M",
]


# ── Payloads ──────────────────────────────────────────────────────────────────

# PIDs: try to spawn 200 sleep processes; exit 0 if OSError blocks it (expected
# under --pids-limit 100), exit 1 if all 200 succeed (guard may be absent).
_PAYLOAD_PIDS = """\
import subprocess, sys
spawned = []
blocked = False
for i in range(200):
    try:
        p = subprocess.Popen(["sleep", "5"])
        spawned.append(p)
    except OSError:
        blocked = True
        break
for p in spawned:
    try:
        p.kill()
    except Exception:
        pass
if blocked:
    sys.exit(0)
print(f"spawned {len(spawned)}/200 without hitting pids_limit", file=sys.stderr)
sys.exit(1)
"""

# Memory: allocate 600 MB AND touch every page to force physical allocation.
# bytearray() alone uses lazy (virtual) allocation — the OOM killer only fires
# when pages are actually dirtied.  Writes one byte per 4 KB page (~150K ops).
_PAYLOAD_MEMORY = """\
b = bytearray(600 * 1024 * 1024)
for i in range(0, len(b), 4096):
    b[i] = 1
"""

# Privileges: read CapBnd and NoNewPrivs from /proc/self/status.
# Exit 0 if both guards are enforced (CapBnd=0, NoNewPrivs=1), else exit 1.
_PAYLOAD_PRIVILEGES = """\
import sys
status = {}
with open("/proc/self/status") as f:
    for line in f:
        key, _, val = line.partition(":")
        status[key.strip()] = val.strip()
cap_bnd = int(status.get("CapBnd", "ffffffffffffffff"), 16)
no_new_privs = int(status.get("NoNewPrivs", "0"))
errors = []
if cap_bnd != 0:
    errors.append(f"CapBnd=0x{cap_bnd:x} (expected 0 — cap_drop=ALL absent?)")
if no_new_privs != 1:
    errors.append(f"NoNewPrivs={no_new_privs} (expected 1 — no-new-privileges absent?)")
if errors:
    for e in errors:
        print(e, file=sys.stderr)
    sys.exit(1)
print(f"OK: CapBnd=0x{cap_bnd:x} NoNewPrivs={no_new_privs}")
sys.exit(0)
"""


# ── Local bridge listener (avoids internet egress in CI) ─────────────────────


@contextlib.contextmanager
def _listener_on_bridge():
    """
    Spin up a minimal TCP listener on a user-defined Docker bridge and yield
    (network_name, listener_ip, port).  The baseline container joins the same
    bridge so the connection is purely local — no internet access required.
    Cleans up the container and network on exit even if the test fails.
    """
    net = "engram-sandbox-net-test"
    cname = "engram-sandbox-listener-test"
    port = "19999"
    _srv = (
        "import socket; "
        "s = socket.socket(); "
        "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); "
        f"s.bind(('', {port})); "
        "s.listen(10); "
        "s.settimeout(60); "
        "[s.accept() for _ in range(5)]"
    )
    subprocess.run(
        ["docker", "network", "create", "--driver", "bridge", net],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--network",
            net,
            "--name",
            cname,
            IMAGE,
            "python",
            "-c",
            _srv,
        ],
        check=True,
        capture_output=True,
    )
    # Hyphens in the network name are invalid in Go template dot notation;
    # use the index function to look up the key by string.
    r = subprocess.run(
        [
            "docker",
            "inspect",
            "-f",
            '{{(index .NetworkSettings.Networks "' + net + '").IPAddress}}',
            cname,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    ip = r.stdout.strip()
    time.sleep(1)  # let Python bind before the test container connects
    try:
        yield net, ip, port
    finally:
        subprocess.run(["docker", "rm", "-f", cname], capture_output=True)
        subprocess.run(["docker", "network", "rm", net], capture_output=True)


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_network_isolation_blocks_outbound():
    """
    Guard: --network none.
    Payload: TCP connection to a listener on a local Docker bridge (no internet).
    Without guard (joined to bridge): connection reaches listener (exit 0).
    With guard (--network none):      no route to bridge IP (exit non-zero).
    """
    with _listener_on_bridge() as (net, ip, port):
        payload = f"import socket; socket.create_connection(('{ip}', {port}), timeout=5)"

        # Baseline: container joined to the same bridge → connection succeeds.
        baseline = _run(list(_WITHOUT_NETWORK) + ["--network", net], payload)
        assert (
            baseline.returncode == 0
        ), "baseline: TCP must reach local listener when --network none is absent"

        # Guarded: --network none → no bridge interface → connection fails.
        guarded = _run(_ALL, payload)
        assert (
            guarded.returncode != 0
        ), "guarded: --network none must block connection to local bridge listener"


def test_readonly_filesystem_blocks_root_writes():
    """
    Guard: --read-only.
    Payload: write to /var/tmp/blocked.txt (world-writable dir, mode 1777).
    Without guard: sandbox user can write (overlay FS is writable) → exit 0.
    With guard:    overlay FS is read-only → write fails → exit non-zero.

    /blocked.txt is NOT used: the sandbox user (UID 10001, non-root) cannot
    write to / regardless of --read-only because / is owned by root (mode 755).
    /var/tmp has mode 1777 so any user can write there when the FS is writable.
    """
    payload = "open('/var/tmp/blocked.txt', 'w').write('x')"

    baseline = _run(_WITHOUT_READONLY, payload)
    assert (
        baseline.returncode == 0
    ), "baseline: root-FS write must succeed when --read-only is absent"

    guarded = _run(_ALL, payload)
    assert guarded.returncode != 0, "guarded: --read-only must deny writes outside /tmp"


def test_pids_limit_caps_fork_bomb():
    """
    Guard: --pids-limit 100.
    Payload: attempt to spawn 200 sleep(5) child processes.
    Without guard (limit=500): all 200 spawn → payload exits 1 (not blocked).
    With guard   (limit=100):  OSError raised before 200 → payload exits 0.
    """
    baseline = _run(_RELAXED_PIDS, _PAYLOAD_PIDS)
    assert baseline.returncode != 0, (
        "baseline: 200 spawn attempts must succeed under --pids-limit 500 "
        "(payload exits 1 when not blocked by the kernel)"
    )

    guarded = _run(_ALL, _PAYLOAD_PIDS)
    assert guarded.returncode == 0, (
        "guarded: --pids-limit 100 must raise OSError before all 200 processes spawn "
        "(payload exits 0 when containment triggers OSError)"
    )


def test_memory_limit_oom_kills_over_budget():
    """
    Guard: --memory 512m (+ --memory-swap 512m → zero swap, hard 512 MB limit).
    Payload: allocate 600 MB AND touch every 4 KB page to force physical use.
    Without guard (2g RAM, unlimited swap): allocation succeeds → exit 0.
    With guard   (512m hard limit):         OOM-killed → exit 137 (non-zero).

    Plain bytearray() uses lazy virtual allocation; the OOM killer only fires
    when physical pages are dirtied.  --memory-swap 512m disables the default
    swap headroom Docker adds (otherwise the container gets 512m RAM + 512m
    swap = 1024m total, enough to absorb the 600 MB allocation).
    """
    baseline = _run(_RELAXED_MEMORY, _PAYLOAD_MEMORY)
    assert baseline.returncode == 0, "baseline: 600 MB allocation must succeed under --memory 2g"

    guarded = _run(_ALL, _PAYLOAD_MEMORY)
    assert guarded.returncode != 0, (
        "guarded: --memory 512m / --memory-swap 512m must OOM-kill a 600 MB "
        "allocation (typically exit 137)"
    )


def test_privilege_escalation_prevention():
    """
    Guards: --cap-drop ALL and --security-opt no-new-privileges.
    Payload: read CapBnd and NoNewPrivs from /proc/self/status.
    Without guards: CapBnd non-zero or NoNewPrivs=0 → payload exits 1.
    With guards:    CapBnd=0 and NoNewPrivs=1 → payload exits 0.
    """
    baseline = _run(_WITHOUT_PRIVDROP, _PAYLOAD_PRIVILEGES)
    assert baseline.returncode != 0, (
        "baseline: without cap_drop/no-new-privileges, " "CapBnd must be non-zero or NoNewPrivs=0"
    )

    guarded = _run(_ALL, _PAYLOAD_PRIVILEGES)
    assert guarded.returncode == 0, (
        "guarded: --cap-drop ALL must zero CapBnd; "
        "--security-opt no-new-privileges must set NoNewPrivs=1"
    )
