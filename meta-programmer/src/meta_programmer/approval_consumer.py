"""Human-approval consumer (Phase E1.9.1).

Processes DEFER approval/denial responses that arrive on
``approval.response.<trace_id>`` from the Dashboard.

The consumer is extracted from MetaProgrammerService into its own class so
that it can be unit-tested without importing the full service (which pulls in
the docker and NATS runtimes).
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any

from meta_programmer.staging import StagingManager, is_review_expired

logger = logging.getLogger(__name__)


class ApprovalConsumer:
    """Processes human approval/denial responses for DEFERred proposals.

    Designed to be fully injected — all I/O dependencies (sandbox, deploy,
    publish) are passed as async callables so the class is testable without
    the live runtime.
    """

    def __init__(
        self,
        staging: StagingManager,
        defer_ttl_ms: int,
        run_tests: Callable[..., Awaitable[dict]],
        deploy: Callable[[str, str, str], Awaitable[None]],
        publish_gap_result: Callable[[str, bool, str], Awaitable[None]],
        log: logging.Logger | None = None,
    ) -> None:
        """
        Args:
            staging: the StagingManager shared with the service.
            defer_ttl_ms: maximum age of a pending DEFER before it is expired.
            run_tests: async callable matching SandboxManager.run_tests signature.
            deploy: async callable (trace_id, target_path, code) → None.
            publish_gap_result: async callable (trace_id, success, message) → None.
            log: optional logger; falls back to the module logger.
        """
        self._staging = staging
        self._defer_ttl_ms = defer_ttl_ms
        self._run_tests = run_tests
        self._deploy = deploy
        self._publish_gap_result = publish_gap_result
        self._log = log or logger

        # Counters — additive with the service's own metrics.
        self.tests_passed: int = 0
        self.tests_failed: int = 0
        self.deployments: int = 0

    async def handle_approval_response(self, data: dict[str, Any]) -> None:
        """Handle a human approval/denial response for a DEFERred proposal.

        Subscribed to ``approval.response.*``.

        Decision tree:
        - No trace_id / bad type → ignore (warn).
        - Item not in human_review → already resolved; ignore (idempotent).
        - TTL expired → fail-closed: reject + publish denied result.
        - approved is True → run sandbox tests then deploy.
        - approved is anything else → deny, fail-closed.
        """
        trace_id = data.get("trace_id", "")
        if not isinstance(trace_id, str) or not trace_id:
            self._log.warning("Approval response missing trace_id — ignoring")
            return

        # Idempotent: if the item has already left human_review (e.g. the
        # expiry sweep ran first, or this is a duplicate message) do nothing.
        if not self._staging.is_in_human_review(trace_id):
            self._log.info(
                "Approval for %s — already resolved (late or duplicate), ignoring",
                trace_id,
            )
            return

        # Fail-closed on TTL: if the dashboard answered too late, reject.
        metadata = self._staging.get_metadata(trace_id) or {}
        now_ms = int(time.time() * 1000)
        if is_review_expired(metadata, now_ms, self._defer_ttl_ms):
            self._log.warning("Approval for %s arrived after TTL expiry — fail-closed", trace_id)
            reason = "Approval arrived after TTL expiry (fail-closed)"
            self._staging.stage_rejected(trace_id, reason)
            await self._publish_gap_result(trace_id, False, reason)
            return

        approved = data.get("approved")
        if approved is True:
            await self._process_approved(trace_id, metadata)
        else:
            # Anything that isn't an explicit True is treated as denial.
            reason = "Human denied the proposal"
            self._log.info("DEFER denied by human: %s", trace_id)
            self._staging.stage_rejected(trace_id, reason)
            await self._publish_gap_result(trace_id, False, reason)

    async def _process_approved(self, trace_id: str, metadata: dict[str, Any]) -> None:
        """Run sandbox tests and atomically deploy a human-approved proposal."""
        target_path = metadata.get("target_path", "")
        self._log.info("DEFER approved by human — running sandbox tests: %s", trace_id)

        # Move from human_review to testing before running sandbox.
        self._staging.stage_human_review_to_testing(trace_id)

        code_path = os.path.join(self._staging.testing_dir, trace_id, "code.py")
        tests_path = os.path.join(self._staging.testing_dir, trace_id, "tests.py")

        try:
            with open(code_path) as f:
                code_content = f.read()
        except OSError as e:
            reason = f"Cannot read staged code: {e}"
            self._log.error(reason)
            self._staging.stage_rejected(trace_id, reason)
            await self._publish_gap_result(trace_id, False, reason)
            return

        test_result = await self._run_tests(
            code_path=code_path,
            test_path=tests_path if os.path.exists(tests_path) else None,
        )

        if test_result.get("success"):
            self.tests_passed += 1
            # Deploy first, then advance to approved so that a deploy failure
            # can still be recorded as rejected (item is still in testing/).
            try:
                await self._deploy(trace_id, target_path, code_content)
                self._staging.stage_approved(trace_id)
                self.deployments += 1
                await self._publish_gap_result(
                    trace_id, True, "Human approved, tested, and deployed"
                )
            except Exception as e:
                self._log.error("Deploy failed for approved DEFER %s: %s", trace_id, e)
                reason = f"Deploy failed after approval: {e}"
                self._staging.stage_rejected(trace_id, reason)
                await self._publish_gap_result(trace_id, False, reason)
        else:
            self.tests_failed += 1
            reason = f"Tests failed after approval: {test_result.get('error', 'unknown')}"
            self._staging.stage_rejected(trace_id, reason)
            await self._publish_gap_result(trace_id, False, reason)
