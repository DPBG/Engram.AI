"""Red-team regression: broker rejects non-kernel privileged publishes (E1.1.9).

Transport-layer defense in depth for ADR 0001 privileged subjects. Complements
application-layer decision signing in sdk/tests/test_signing.py.
"""

from __future__ import annotations

import asyncio

import nats
import pytest

from .privileged_subjects import (
    COORDINATOR_ALLOWED_SUBJECT,
    COORDINATOR_IDENTITY,
    KERNEL_IDENTITY,
    NON_KERNEL_IDENTITIES,
    PRIVILEGED_PUBLISH_SUBJECTS,
)

_PROBE = b'{"type":"ALLOW","trace_id":"red-team-forged"}'
_ERROR_WAIT_S = 2.0
_ERROR_POLL_S = 0.05


def _nats_host_port(nats_url: str) -> str:
    # nats://127.0.0.1:PORT -> 127.0.0.1:PORT
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
@pytest.mark.parametrize("subject", PRIVILEGED_PUBLISH_SUBJECTS)
async def test_non_kernel_privileged_publish_rejected(
    authz_nats_url: str,
    identity: tuple[str, str],
    subject: str,
) -> None:
    """Non-kernel identities cannot publish kernel-only subjects."""
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
@pytest.mark.parametrize("subject", PRIVILEGED_PUBLISH_SUBJECTS)
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
