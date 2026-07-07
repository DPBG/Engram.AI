"""Red-team regression (E1.1.9): NATS broker must refuse non-Kernel identities
from publishing on privileged subjects at the transport layer.

This is the canonical "forged ALLOW rejected" regression test.  It complements
decision signing (Task 1.2 / PR #36), which rejects forged payloads at the
*application* layer.  Together they provide defense-in-depth: signing handles
a hostile bus; transport authz handles a compromised service on the same bus.

Acceptance criteria (issue #95):
  • CI fails if *any* non-Kernel identity publishes to:
        decision.*  code.decision.*  policy.*  cognitive.response.validated
  • Kernel identity publishes all of the above without error
  • A fabricated / unregistered identity cannot connect at all

ADR reference: docs/adr/0001-nats-authz.md §3 "privileged Kernel publisher set"
CLAUDE.md §3: "Kernel is the sole authority that may emit a decision."
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

import nats
import nats.errors
import pytest

from .conftest import (
    KERNEL_PASS,
    KERNEL_USER,
    META_PASS,
    META_USER,
    PLANNER_PASS,
    PLANNER_USER,
)

# ── Privileged subjects — Kernel-only publish (ADR 0001 §3) ───────────────
PRIVILEGED_SUBJECTS = [
    "decision.trace-abc",
    "code.decision.trace-abc",
    "policy.update",
    "policy.restrict",
    "cognitive.response.validated",
]

# ── Non-Kernel service identities registered in the broker ────────────────
NON_KERNEL_IDENTITIES = [
    pytest.param(PLANNER_USER, PLANNER_PASS, id="planner"),
    pytest.param(META_USER, META_PASS, id="meta_programmer"),
]


# ── Connection helper ──────────────────────────────────────────────────────


@contextlib.asynccontextmanager
async def _connect(
    url: str,
    user: str,
    password: str,
) -> AsyncIterator[tuple[nats.aio.client.Client, asyncio.Queue[str]]]:
    """Connect with *user*/*password*; collect broker Permissions Violations.

    Yields ``(nc, violations)`` where *violations* is an asyncio.Queue that
    receives every "Permissions Violation" error string the broker sends back.
    """
    violations: asyncio.Queue[str] = asyncio.Queue()

    async def _error_cb(e: Exception) -> None:
        msg = str(e)
        # nats-py lowercases the broker's "-ERR Permissions Violation ..." when
        # surfacing it (e.g. 'nats: permissions violation for publish to ...'),
        # so match case-insensitively or every violation is silently dropped.
        if "permissions violation" in msg.lower():
            await violations.put(msg)

    nc = await nats.connect(
        url,
        user=user,
        password=password,
        error_cb=_error_cb,
        connect_timeout=5.0,
    )
    try:
        yield nc, violations
    finally:
        if not nc.is_closed:
            await nc.close()


async def _assert_broker_rejects(
    nc: nats.aio.client.Client,
    violations: asyncio.Queue[str],
    subject: str,
    *,
    timeout: float = 2.0,
) -> None:
    """Publish to *subject* and assert the broker returns a Permissions Violation.

    Flushes immediately after publish so the -ERR arrives as quickly as
    possible, then waits up to *timeout* seconds for the error callback.
    """
    await nc.publish(subject, b"red-team-probe")
    await nc.flush()
    try:
        err = await asyncio.wait_for(violations.get(), timeout=timeout)
    except TimeoutError:
        raise AssertionError(
            f"Expected NATS 'Permissions Violation' for publish to '{subject}' "
            f"but no error arrived within {timeout}s — "
            "broker accepted a publish it should have rejected"
        )
    assert "permissions violation" in err.lower(), f"Unexpected error format: {err!r}"


# ── Rejection tests (parametrized: identity × subject) ────────────────────


@pytest.mark.parametrize("user,password", NON_KERNEL_IDENTITIES)
@pytest.mark.parametrize("subject", PRIVILEGED_SUBJECTS)
async def test_non_kernel_publish_privileged_is_broker_rejected(
    authz_nats_url: str,
    user: str,
    password: str,
    subject: str,
) -> None:
    """Broker must refuse every non-Kernel identity publishing on privileged subjects.

    Covers all combinations of (planner, meta_programmer) × (decision.>,
    code.decision.>, policy.*, cognitive.response.validated) — 10 cases total.
    A single failure means the broker's allowlist has a gap.
    """
    async with _connect(authz_nats_url, user, password) as (nc, violations):
        await _assert_broker_rejects(nc, violations, subject)


# ── Kernel sanity-check ────────────────────────────────────────────────────


async def test_kernel_can_publish_all_privileged_subjects(
    authz_nats_url: str,
) -> None:
    """Kernel identity must succeed on every privileged subject (no false positives)."""
    async with _connect(authz_nats_url, KERNEL_USER, KERNEL_PASS) as (nc, violations):
        for subject in PRIVILEGED_SUBJECTS:
            await nc.publish(subject, b"kernel-probe")
        await nc.flush()
        # Brief pause to let any erroneous broker -ERR arrive before we check.
        await asyncio.sleep(0.15)
        assert violations.empty(), (
            "Kernel identity received an unexpected Permissions Violation — "
            "its allowlist is too narrow"
        )


# ── Unknown / fabricated identity ─────────────────────────────────────────


async def test_unknown_identity_cannot_connect(authz_nats_url: str) -> None:
    """A fabricated identity with no registered credentials must be refused connection.

    This covers the "supply-chain" threat: a rogue process that was never
    granted a credential in the broker config must not reach the bus at all.
    """
    # allow_reconnect=False makes the auth rejection raise immediately.  With
    # reconnects enabled, nats-py treats max_reconnect_attempts=0 as "no cap"
    # and retries the rejected server forever — this test then hangs until the
    # CI job timeout (observed: every SDK/governance job riding to 15 min).
    # The broker surfaces the rejection as the base nats.errors.Error
    # ("Authorization Violation"), so expect the family base class.
    with pytest.raises(nats.errors.Error):
        nc = await nats.connect(
            authz_nats_url,
            user="fabricated_plugin",
            password="any-password",
            connect_timeout=2.0,
            allow_reconnect=False,
        )
        await nc.close()
