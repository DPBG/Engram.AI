"""Red-team probe subjects — see docs/adr/0001-nats-authz.md §3.

The test broker encodes ADR §3 kernel-only privileged publishes, with one
documented runtime exception: neuromorphic may publish ``policy.restrict`` for
emergency motor halt (ADR §3 conflict note; neuromorphic/service.py ~1496).
Target state routes restriction via ``policy.restrict.request`` through the Kernel.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ADR_NATS_AUTHZ = REPO_ROOT / "docs" / "adr" / "0001-nats-authz.md"

# Kernel-only privileged subjects (ADR §3). Excludes policy.restrict — see below.
KERNEL_ONLY_PUBLISH_SUBJECTS: tuple[str, ...] = (
    "decision.red-team-forged-trace",
    "code.decision.red-team-forged-trace",
    "policy.update",
    "policy.update.status",
    "cognitive.response.validated",
)

POLICY_RESTRICT_SUBJECT = "policy.restrict"

# All privileged subjects the kernel must be able to publish in this fixture.
KERNEL_PRIVILEGED_PUBLISH_SUBJECTS: tuple[str, ...] = (
    *KERNEL_ONLY_PUBLISH_SUBJECTS,
    POLICY_RESTRICT_SUBJECT,
)

# Service identities that must be denied on kernel-only privileged subjects.
NON_KERNEL_IDENTITIES: tuple[tuple[str, str], ...] = (
    ("coordinator", "engram-test-coordinator"),
    ("meta", "engram-test-meta"),
    ("planner", "engram-test-planner"),
    ("neuro", "engram-test-neuro"),
)

# Denied on policy.restrict (neuro is allowed — emergency watchdog halt path).
POLICY_RESTRICT_DENIED_IDENTITIES: tuple[tuple[str, str], ...] = (
    ("coordinator", "engram-test-coordinator"),
    ("meta", "engram-test-meta"),
    ("planner", "engram-test-planner"),
)

KERNEL_IDENTITY: tuple[str, str] = ("kernel", "engram-test-kernel")
NEURO_IDENTITY: tuple[str, str] = ("neuro", "engram-test-neuro")
COORDINATOR_IDENTITY: tuple[str, str] = ("coordinator", "engram-test-coordinator")

COORDINATOR_ALLOWED_SUBJECT = "proposal.new"
