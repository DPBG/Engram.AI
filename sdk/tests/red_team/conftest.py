"""Fixtures for NATS broker-level authorization (red-team) tests.

Starts an isolated nats-server configured with per-user publish/subscribe
permissions using the static-users model (ADR 0001 §2: "A static-users variant
MAY be used as a transitional implementation, as long as the same matrix below
is enforced").

Each test module gets its own server process on an ephemeral port so these
tests never share state with the main SDK integration suite.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import textwrap
import time
import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

# ── Per-user credentials baked into the test NATS config ──────────────────
KERNEL_USER = "kernel"
KERNEL_PASS = "kernel-test-pass"

PLANNER_USER = "planner"
PLANNER_PASS = "planner-test-pass"

META_USER = "meta_programmer"
META_PASS = "meta-test-pass"

DASHBOARD_USER = "dashboard"
DASHBOARD_PASS = "dashboard-test-pass"

# ── NATS config template ───────────────────────────────────────────────────
# Implements the ADR 0001 §3 "privileged Kernel publisher set":
#   decision.>  code.decision.>  policy.*  cognitive.response.validated
# are publish-allowed for kernel only; all other identities' allowlists omit them.
#
# IMPORTANT: use block syntax (`permissions {{ ... }}`) throughout, NOT the
# JSON-value form (`permissions: {{ ... }}`).  Mixed syntax causes nats-server
# to silently ignore the permissions block, leaving users unrestricted.
_NATS_AUTHZ_CONF = textwrap.dedent("""\
    # Red-team regression test: per-user authorization (ADR 0001 static-users model).
    # kernel      -> full publish rights (Kernel is the sole decision authority, CLAUDE.md s3)
    # planner     -> non-privileged publish only
    # meta_programmer -> non-privileged publish only
    # dashboard   -> operator-level publish only (safety.halt/resume, observations, motor)
    authorization {{
      users = [
        {{
          user: "{kernel_user}"
          password: "{kernel_pass}"
          permissions {{
            publish   {{ allow: [">"] }}
            subscribe {{ allow: [">"] }}
          }}
        }}
        {{
          user: "{planner_user}"
          password: "{planner_pass}"
          permissions {{
            publish {{
              allow: [
                "proposal.new",
                "proposal.status",
                "planner.>",
                "system.health",
                "heartbeat.planner",
                "_INBOX.>"
              ]
            }}
            subscribe {{ allow: [">"] }}
          }}
        }}
        {{
          user: "{meta_user}"
          password: "{meta_pass}"
          permissions {{
            publish {{
              allow: [
                "code.proposal",
                "knowledge.gap",
                "metaprogrammer.>",
                "system.health",
                "heartbeat.meta",
                "_INBOX.>"
              ]
            }}
            subscribe {{ allow: [">"] }}
          }}
        }}
        {{
          user: "{dashboard_user}"
          password: "{dashboard_pass}"
          permissions {{
            publish {{
              allow: [
                "safety.halt",
                "safety.resume",
                "observation.>",
                "motor.guidance",
                "neuromorphic.concept.probe",
                "sensory.gateway.command",
                "approval.response.>",
                "system.health",
                "heartbeat.dashboard",
                "_INBOX.>"
              ]
            }}
            subscribe {{ allow: [">"] }}
          }}
        }}
      ]
    }}
    """)


def _free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


@pytest.fixture(scope="module")
def authz_nats_url() -> Generator[str, None, None]:
    """Yield a URL for a NATS broker running with enforced per-user permissions.

    Starts an isolated nats-server (ephemeral port) configured from the static
    per-user authz block above.  Skips the test module if nats-server is not on
    PATH — the same graceful-skip pattern used by the main SDK conftest.
    """
    binary = shutil.which("nats-server")
    if binary is None:
        pytest.skip("nats-server not on PATH — skipping red-team authorization tests")

    host = "127.0.0.1"
    port = _free_port(host)
    run_id = uuid.uuid4().hex

    conf_path = Path("/tmp") / f"engram-red-team-{run_id}.conf"
    conf_path.write_text(
        _NATS_AUTHZ_CONF.format(
            kernel_user=KERNEL_USER,
            kernel_pass=KERNEL_PASS,
            planner_user=PLANNER_USER,
            planner_pass=PLANNER_PASS,
            meta_user=META_USER,
            meta_pass=META_PASS,
            dashboard_user=DASHBOARD_USER,
            dashboard_pass=DASHBOARD_PASS,
        )
    )

    proc = subprocess.Popen(
        [binary, "-c", str(conf_path), "-p", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        for _ in range(50):
            if _port_open(host, port):
                break
            if proc.poll() is not None:
                stderr_out = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
                pytest.skip(
                    f"nats-server failed to start with the authz config.\n"
                    f"stderr: {stderr_out[:2000]}"
                )
            time.sleep(0.1)
        else:
            pytest.skip("Timed out waiting for authz nats-server to become ready")

        yield f"nats://{host}:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        conf_path.unlink(missing_ok=True)
