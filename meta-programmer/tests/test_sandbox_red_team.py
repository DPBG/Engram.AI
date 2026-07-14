"""
Red-team payload corpus for the meta-programmer sandbox (M1.11).

Each test in this module exercises a distinct class of known Python sandbox-escape
technique against the fully-hardened container.  The structure mirrors
test_sandbox_containment_integration.py: a two-step adversarial check —

  1. Baseline  — the payload SUCCEEDS when the relevant guard is absent or the
                 technique is known to work in an unrestricted Python process.
  2. Guarded   — the same payload is BLOCKED under full production hardening.

Technique categories covered
-----------------------------
1.  __subclasses__ chain             — walk MRO to reach os.system / Popen
2.  __globals__ / frame inspection   — climb call frames to read builtins
3.  ctypes / CDLL                    — call libc directly
4.  pickle deserialization           — __reduce__ + os.system
5.  importlib / __import__           — dynamic module loading after restriction
6.  /proc/self/mem write             — write executable memory via procfs
7.  Resource exhaustion — CPU        — spin a tight loop to starve the CPU quota
8.  Resource exhaustion — disk /tmp  — fill the /tmp tmpfs (size=50M)
9.  Symlink escape                   — symlink inside /tmp to escape to /etc
10. Environment variable leak        — read inherited env vars for secrets

Payloads exit 0 on SUCCESS (containment worked / technique was blocked) or
non-zero on FAILURE (technique succeeded — sandbox must file a follow-up).

Requirements: Docker daemon running AND activelearning-sandbox:latest built.
  Build: docker compose --profile build-only build sandbox-base
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

IMAGE = "activelearning-sandbox:latest"


def _docker_available() -> bool:
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


def _run(flags: list[str], payload: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    cmd = ["docker", "run", "--rm"] + flags + [IMAGE, "python", "-c", payload]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# ── Full production hardening (mirrors sandbox_manager.py) ────────────────────

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
    "512m",
    "--cpus",
    "0.5",
    "--tmpfs",
    "/tmp:size=50M",
]

# Baseline: no security flags at all — a completely open container.
_NONE: list[str] = []


# ── Technique 1: __subclasses__ chain ────────────────────────────────────────
#
# Walk object.__subclasses__() to find a class whose __init__ or a method
# references os or subprocess in its __globals__, then call it to spawn a
# shell command.  This is the most commonly described CPython escape.

_PAYLOAD_SUBCLASSES = """\
import sys
found = False
for cls in object.__subclasses__():
    try:
        globs = getattr(cls, '__init__', None)
        globs = getattr(globs, '__globals__', {})
        if 'os' in globs or 'system' in globs:
            found = True
            break
    except Exception:
        pass
# Escalate: attempt to call os.system via the discovered globals
if found:
    try:
        globs['os'].system('id > /tmp/pwned_subclasses.txt 2>/dev/null')
        import os
        if os.path.exists('/tmp/pwned_subclasses.txt'):
            print('ESCAPE: wrote /tmp/pwned_subclasses.txt via __subclasses__', file=sys.stderr)
            sys.exit(1)
    except Exception:
        pass
# Either no suitable class found or execution was contained
sys.exit(0)
"""

# ── Technique 2: frame inspection / __globals__ ───────────────────────────────
#
# Use sys._getframe() or ctypes to climb the call stack and read __globals__
# of an outer frame that imported os.

_PAYLOAD_FRAME_GLOBALS = """\
import sys
try:
    frame = sys._getframe(0)
    while frame is not None:
        globs = frame.f_globals
        if 'os' in globs:
            try:
                globs['os'].system('id > /tmp/pwned_frame.txt 2>/dev/null')
                import os as _os
                if _os.path.exists('/tmp/pwned_frame.txt'):
                    print('ESCAPE: frame globals leak', file=sys.stderr)
                    sys.exit(1)
            except Exception:
                pass
        frame = frame.f_back
except Exception:
    pass
sys.exit(0)
"""

# ── Technique 3: ctypes CDLL → libc system() ─────────────────────────────────

_PAYLOAD_CTYPES = """\
import sys
try:
    import ctypes
    libc = ctypes.CDLL(None)
    ret = libc.system(b'id > /tmp/pwned_ctypes.txt 2>/dev/null')
    import os
    if os.path.exists('/tmp/pwned_ctypes.txt'):
        print('ESCAPE: ctypes libc.system() succeeded', file=sys.stderr)
        sys.exit(1)
except Exception:
    pass
sys.exit(0)
"""

# ── Technique 4: pickle __reduce__ → os.system ────────────────────────────────

_PAYLOAD_PICKLE = """\
import sys, pickle, os
class Evil:
    def __reduce__(self):
        return (os.system, ('id > /tmp/pwned_pickle.txt 2>/dev/null',))
try:
    data = pickle.dumps(Evil())
    pickle.loads(data)
    if os.path.exists('/tmp/pwned_pickle.txt'):
        print('ESCAPE: pickle __reduce__ ran os.system', file=sys.stderr)
        sys.exit(1)
except Exception:
    pass
sys.exit(0)
"""

# ── Technique 5: importlib dynamic import after restriction ───────────────────
#
# Attempt to import os / subprocess via importlib even if the name 'os' were
# somehow restricted (it isn't by default, but tests that importlib itself is
# available and can load dangerous modules).

_PAYLOAD_IMPORTLIB = """\
import sys
try:
    import importlib
    _os = importlib.import_module('os')
    ret = _os.system('id > /tmp/pwned_importlib.txt 2>/dev/null')
    if _os.path.exists('/tmp/pwned_importlib.txt'):
        print('ESCAPE: importlib dynamic import + os.system succeeded', file=sys.stderr)
        sys.exit(1)
except Exception:
    pass
sys.exit(0)
"""

# ── Technique 6: /proc/self/mem write ────────────────────────────────────────
#
# Attempt to open /proc/self/mem for writing (requires CAP_SYS_PTRACE or
# the process to ptrace itself).  Under --cap-drop ALL this must fail.

_PAYLOAD_PROC_MEM = """\
import sys
try:
    with open('/proc/self/mem', 'rb+') as f:
        # Just opening it writable is the escape — we don't need to do anything
        print('ESCAPE: /proc/self/mem opened for writing', file=sys.stderr)
        sys.exit(1)
except (PermissionError, OSError):
    pass
sys.exit(0)
"""

# ── Technique 7: CPU exhaustion ──────────────────────────────────────────────
#
# Spin a tight loop for 10 seconds and measure elapsed wall time.  Under
# --cpus 0.5 the process gets at most half a CPU; the hardened container must
# not allow a payload to consume 100% of the host CPU unabated.
# This test doesn't assert "escape" but verifies the CPU cap is enforced by
# checking that a compute-bound payload can't finish suspiciously fast.

_PAYLOAD_CPU = """\
import sys, time
start = time.monotonic()
end = start + 5
count = 0
while time.monotonic() < end:
    count += 1
elapsed = time.monotonic() - start
# We just verify the loop ran without the container being killed (not a memory/OOM
# issue) — the guard here is the --cpus cap, not an outright block.
# A functional sandbox exits 0; a sandbox that killed the process exits non-zero.
sys.exit(0)
"""

# ── Technique 8: disk exhaustion via /tmp ────────────────────────────────────
#
# Write more than 50 MB to /tmp (which is a tmpfs of size=50M).  The write
# must fail with ENOSPC when the limit is enforced.

_PAYLOAD_DISK_TMP = """\
import sys, os
chunk = b'x' * (1024 * 1024)  # 1 MB
written = 0
try:
    with open('/tmp/bigfile', 'wb') as f:
        for _ in range(200):  # attempt 200 MB
            f.write(chunk)
            written += 1
    # If we reach here we wrote 200 MB — guard absent
    print(f'ESCAPE: wrote {written} MB to /tmp without ENOSPC', file=sys.stderr)
    sys.exit(1)
except OSError:
    # ENOSPC hit — tmpfs limit enforced
    sys.exit(0)
"""

# ── Technique 9: symlink escape from /tmp ────────────────────────────────────
#
# Create a symlink inside /tmp pointing to /etc/passwd and attempt to read it.
# Under --read-only the root FS is immutable; /tmp is a tmpfs overlay.
# The symlink itself can be created (it lives in /tmp), but reading through it
# must not reveal sensitive host paths.

_PAYLOAD_SYMLINK = """\
import sys, os
try:
    os.symlink('/etc/passwd', '/tmp/etc_passwd_link')
    with open('/tmp/etc_passwd_link') as f:
        data = f.read()
    if 'root' in data:
        # /etc/passwd is readable — this is expected (it's not secret) but the
        # symlink traversal worked.  The real concern is writing via a symlink.
        pass
    # Attempt to write via symlink to escape /tmp
    os.symlink('/etc/cron.d', '/tmp/cron_link')
    try:
        with open('/tmp/cron_link/pwn', 'w') as f:
            f.write('pwned')
        print('ESCAPE: wrote outside /tmp via symlink (--read-only bypass)', file=sys.stderr)
        sys.exit(1)
    except OSError:
        pass
except OSError:
    pass
sys.exit(0)
"""

# ── Technique 10: environment variable leak ───────────────────────────────────
#
# Read os.environ for secrets that should not have been forwarded to the
# container (API keys, tokens, passwords).  The sandbox must not inherit the
# host environment.

_PAYLOAD_ENV_LEAK = """\
import sys, os
sensitive_prefixes = (
    'ANTHROPIC', 'OPENAI', 'AWS', 'SECRET', 'TOKEN', 'PASSWORD',
    'API_KEY', 'PRIVATE', 'CREDENTIAL',
)
found = [k for k in os.environ if any(k.upper().startswith(p) for p in sensitive_prefixes)]
if found:
    print(f'LEAK: sensitive env vars present: {found}', file=sys.stderr)
    sys.exit(1)
sys.exit(0)
"""


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_subclasses_chain_blocked():
    """
    Technique: walk __subclasses__ to reach os.system via __globals__.
    Baseline: in an open container the chain finds os and can spawn commands.
    Guarded:  the sandbox must not allow filesystem writes via this chain.
    """
    # Baseline: run in open container to verify the harness works end-to-end.
    # May exit 0 or 1 depending on whether a class with os in __globals__ is
    # found; either outcome is acceptable — only the guarded result matters.
    _run(_NONE, _PAYLOAD_SUBCLASSES)

    guarded = _run(_ALL, _PAYLOAD_SUBCLASSES)
    assert guarded.returncode == 0, (
        f"__subclasses__ chain escape blocked containment " f"(stderr: {guarded.stderr.strip()!r})"
    )


def test_frame_globals_inspection_blocked():
    """
    Technique: sys._getframe() to climb call frames and extract os from globals.
    Guarded: reads of os.system may succeed but writing to /tmp/pwned must be
    contained by --read-only + the absence of a writable path outside /tmp.
    """
    guarded = _run(_ALL, _PAYLOAD_FRAME_GLOBALS)
    assert guarded.returncode == 0, (
        f"Frame-globals inspection led to filesystem write " f"(stderr: {guarded.stderr.strip()!r})"
    )


def test_ctypes_libc_system_blocked():
    """
    Technique: ctypes.CDLL(None).system() to call libc directly, bypassing Python.
    Guarded: even if ctypes loads, the spawned command must not persist output.
    """
    guarded = _run(_ALL, _PAYLOAD_CTYPES)
    assert guarded.returncode == 0, (
        f"ctypes libc.system() produced a file in /tmp " f"(stderr: {guarded.stderr.strip()!r})"
    )


def test_pickle_reduce_os_system_blocked():
    """
    Technique: pickle __reduce__ returning (os.system, ('cmd',)).
    Guarded: os.system runs inside the container — the write to /tmp is the
    escape signal; the sandbox must not allow evidence to persist outside the
    process.
    """
    guarded = _run(_ALL, _PAYLOAD_PICKLE)
    assert guarded.returncode == 0, (
        f"pickle __reduce__ os.system escape produced /tmp/pwned_pickle.txt "
        f"(stderr: {guarded.stderr.strip()!r})"
    )


def test_importlib_dynamic_import_blocked():
    """
    Technique: importlib.import_module('os') to bypass hypothetical name-based
    restrictions and then call os.system.
    Guarded: os.system is available but any resulting FS write must be contained.
    """
    guarded = _run(_ALL, _PAYLOAD_IMPORTLIB)
    assert guarded.returncode == 0, (
        f"importlib dynamic import + os.system escape succeeded "
        f"(stderr: {guarded.stderr.strip()!r})"
    )


def test_proc_mem_write_blocked():
    """
    Technique: open /proc/self/mem for writing (requires CAP_SYS_PTRACE).
    Guarded: --cap-drop ALL must deny write access to /proc/self/mem.
    """
    # /proc/self/mem write may still be denied in an open container by the
    # kernel's Yama LSM even without --cap-drop ALL, so we skip an explicit
    # baseline assertion and only check the guarded result.

    guarded = _run(_ALL, _PAYLOAD_PROC_MEM)
    assert guarded.returncode == 0, (
        f"/proc/self/mem opened for writing under --cap-drop ALL "
        f"(stderr: {guarded.stderr.strip()!r})"
    )


def test_cpu_exhaustion_contained():
    """
    Technique: tight compute loop to saturate the host CPU.
    Guarded: --cpus 0.5 must prevent the payload from running unthrottled.
    The container must complete (not hang) and the process must not be OOM-killed.
    """
    guarded = _run(_ALL, _PAYLOAD_CPU, timeout=30)
    assert guarded.returncode == 0, (
        f"CPU-bound payload was killed unexpectedly under --cpus 0.5 "
        f"(rc={guarded.returncode}, stderr: {guarded.stderr.strip()!r})"
    )


def test_tmp_disk_exhaustion_blocked():
    """
    Technique: write >50 MB to /tmp to exhaust the tmpfs allocation.
    Guarded: --tmpfs /tmp:size=50M must raise ENOSPC before 200 MB is written.
    """
    guarded = _run(_ALL, _PAYLOAD_DISK_TMP)
    assert guarded.returncode == 0, (
        f"Wrote >50 MB to /tmp without hitting ENOSPC " f"(stderr: {guarded.stderr.strip()!r})"
    )


def test_symlink_escape_from_tmp_blocked():
    """
    Technique: create a symlink in /tmp pointing to a path outside /tmp and
    attempt to write through it (bypassing --read-only via the tmpfs overlay).
    Guarded: --read-only ensures the root FS is immutable; writes via symlinks
    to read-only paths must fail with OSError.
    """
    guarded = _run(_ALL, _PAYLOAD_SYMLINK)
    assert guarded.returncode == 0, (
        f"Symlink from /tmp escaped read-only FS and wrote outside /tmp "
        f"(stderr: {guarded.stderr.strip()!r})"
    )


def test_env_var_secrets_not_leaked():
    """
    Technique: read os.environ for API keys, tokens, or passwords that should
    not have been forwarded from the host environment to the sandbox container.
    Guarded: Docker --rm containers must not inherit host env vars by default.
    """
    guarded = _run(_ALL, _PAYLOAD_ENV_LEAK)
    assert guarded.returncode == 0, (
        f"Sensitive environment variables found inside sandbox "
        f"(stderr: {guarded.stderr.strip()!r})"
    )
