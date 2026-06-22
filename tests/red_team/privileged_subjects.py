"""Privileged NATS subjects from ADR 0001 §3 (kernel-only publishers)."""

from __future__ import annotations

# Concrete probe subjects used in red-team tests (wildcards expanded).
PRIVILEGED_PUBLISH_SUBJECTS: tuple[str, ...] = (
    "decision.red-team-forged-trace",
    "code.decision.red-team-forged-trace",
    "policy.update",
    "policy.restrict",
    "cognitive.response.validated",
)

# Non-kernel service identities that must be denied on privileged subjects.
NON_KERNEL_IDENTITIES: tuple[tuple[str, str], ...] = (
    ("coordinator", "engram-test-coordinator"),
    ("meta", "engram-test-meta"),
    ("planner", "engram-test-planner"),
)

KERNEL_IDENTITY: tuple[str, str] = ("kernel", "engram-test-kernel")

COORDINATOR_IDENTITY: tuple[str, str] = ("coordinator", "engram-test-coordinator")

# Allowed publish used as a positive control (proves creds work).
COORDINATOR_ALLOWED_SUBJECT = "proposal.new"
