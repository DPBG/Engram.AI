"""Red-team regression: broker rejects non-Kernel privileged publishes (E1.1.9).

Transport-layer defense in depth for ADR 0001 (docs/adr/0001-nats-authz.md) §3.
Complements application-layer decision signing in sdk/tests/test_signing.py.
"""

from __future__ import annotations

import asyncio

import nats
import pytest

from .privileged_subjects import (
    ADR_NATS_AUTHZ,
    COORDINATOR_ALLOWED_SUBJECT,
    COORDINATOR_IDENTITY,
    KERNEL_IDENTITY,
    KERNEL_ONLY_PUBLISH_SUBJECTS,
    KERNEL_PRIVILEGED_PUBLISH_SUBJECTS,
    NEURO_IDENTITY,
    NON_KERNEL_IDENTITIES,
    POLICY_RESTRICT_DENIED_IDENTITIES,
    POLICY_RESTRICT_SUBJECT,
)

_PROBE = b'{"type":"ALLOW","trace_id":"red-team-forged"}'
_ERROR_WAIT_S = 0.5
_ERROR_POLL_S = 0.02


def test_adr_nats_authz_spec_exists() -> None:
    """Security regression must be traceable to the in-repo ADR spec."""
    assert ADR_NATS_AUTHZ.is_file(), f"missing ADR spec: {ADR_NATS_AUTHZ}"


def _nats_host_port(nats_url: str) -> str:
    return nats_url.removeprefix("nats://")


async def _publish_and_collect_errors(
    *,
    host_port: str,
    user: str,
    password: str,
    subject: str,
) -> list[str]:
    errors: list[str] = []

    async def err_cb(exc: Exception) -> None:
        errors.append(str(exc))

    nc = await nats.connect(
        f"nats://{user}:{password}@{host_port}",
        error_cb=err_cb,
        max_reconnect_attempts=0,
        connect_timeout=2,
    )
    try:
        await nc.publish(subject, _PROBE)
        await nc.flush()
        deadline = asyncio.get_running_loop().time() + _ERROR_WAIT_S
        while asyncio.get_running_loop().time() < deadline:
            if errors:
                break
            await asyncio.sleep(_ERROR_POLL_S)
    finally:
        await nc.close()
    return errors


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", NON_KERNEL_IDENTITIES, ids=lambda i: i[0])
@pytest.mark.parametrize("subject", KERNEL_ONLY_PUBLISH_SUBJECTS)
async def test_non_kernel_kernel_only_publish_rejected(
    authz_nats_url: str,
    identity: tuple[str, str],
    subject: str,
) -> None:
    """Non-kernel identities cannot publish ADR §3 kernel-only subjects."""
    user, password = identity
    errors = await _publish_and_collect_errors(
        host_port=_nats_host_port(authz_nats_url),
        user=user,
        password=password,
        subject=subject,
    )
    assert errors, f"{user} publish to {subject!r} should be broker-rejected"
    # nats-py async error_cb format: "Permissions Violation for Publish to <subject>"
    assert any("permissions violation" in err.lower() for err in errors), errors


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", POLICY_RESTRICT_DENIED_IDENTITIES, ids=lambda i: i[0])
async def test_non_neuro_policy_restrict_publish_rejected(
    authz_nats_url: str,
    identity: tuple[str, str],
) -> None:
    """Only kernel and neuro may publish policy.restrict in this fixture."""
    user, password = identity
    errors = await _publish_and_collect_errors(
        host_port=_nats_host_port(authz_nats_url),
        user=user,
        password=password,
        subject=POLICY_RESTRICT_SUBJECT,
    )
    assert errors, f"{user} publish to {POLICY_RESTRICT_SUBJECT!r} should be broker-rejected"
    assert any("permissions violation" in err.lower() for err in errors), errors


@pytest.mark.asyncio
@pytest.mark.parametrize("subject", KERNEL_PRIVILEGED_PUBLISH_SUBJECTS)
async def test_kernel_can_publish_privileged_subjects(
    authz_nats_url: str,
    subject: str,
) -> None:
    """Kernel identity is allowed to publish privileged subjects."""
    user, password = KERNEL_IDENTITY
    errors = await _publish_and_collect_errors(
        host_port=_nats_host_port(authz_nats_url),
        user=user,
        password=password,
        subject=subject,
    )
    assert errors == [], f"kernel publish to {subject!r} should succeed: {errors}"


@pytest.mark.asyncio
async def test_neuro_can_publish_policy_restrict(authz_nats_url: str) -> None:
    """Positive control: neuromorphic emergency halt path (ADR §3 conflict note)."""
    user, password = NEURO_IDENTITY
    errors = await _publish_and_collect_errors(
        host_port=_nats_host_port(authz_nats_url),
        user=user,
        password=password,
        subject=POLICY_RESTRICT_SUBJECT,
    )
    assert errors == [], f"neuro publish to {POLICY_RESTRICT_SUBJECT!r} should succeed: {errors}"


@pytest.mark.asyncio
async def test_coordinator_can_publish_allowed_subject(authz_nats_url: str) -> None:
    """Positive control: coordinator creds work for non-privileged subjects."""
    user, password = COORDINATOR_IDENTITY
    errors = await _publish_and_collect_errors(
        host_port=_nats_host_port(authz_nats_url),
        user=user,
        password=password,
        subject=COORDINATOR_ALLOWED_SUBJECT,
    )
    assert errors == [], errors
