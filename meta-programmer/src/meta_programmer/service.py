"""
Meta-Programmer Service - Self-evolving code generation system.

Listens for knowledge gaps, generates code using local LLMs (Ollama),
tests in isolated sandboxes, and deploys after Kernel approval.
"""

import asyncio
import json
import os
import time
from typing import Any, Optional

from activelearning import BaseService

from meta_programmer.approval_consumer import ApprovalConsumer
from meta_programmer.sandbox_manager import SandboxManager
from meta_programmer.staging import StagingManager
from meta_programmer.agents import MetaProgrammerTeam
from meta_programmer.safety import scan_source, safe_deploy_path, deploy_atomically


class MetaProgrammerService(BaseService):
    """
    Meta-Programmer orchestrator.

    Processes knowledge gaps by:
    1. Receiving gap descriptions via NATS
    2. Generating code using local LLM (Ollama)
    3. Requesting Kernel approval for code changes
    4. Testing in sandbox containers
    5. Deploying approved code
    """

    def __init__(self):
        super().__init__("meta-programmer", use_database=True, use_event_bus=True)

        self.ollama_url = os.environ.get("OLLAMA_URL", "http://ollama:11434")
        self.staging_root = os.environ.get("STAGING_ROOT", "/data/staging")
        self.plugins_root = os.environ.get("PLUGINS_ROOT", "/data/plugins")
        self.tasks_root = os.environ.get("TASKS_ROOT", "/data/tasks")

        # Components
        self._sandbox_manager = SandboxManager()
        self._staging_manager = StagingManager(self.staging_root)
        self._team: Optional[MetaProgrammerTeam] = None
        self._approval_consumer: Optional[ApprovalConsumer] = None

        # A deferred (human-review) proposal that no one answers is failed
        # closed after this TTL (Phase 1.9), rather than lingering forever.
        self.defer_ttl_ms = int(os.environ.get("DEFER_TTL_MS", "300000"))  # 5 min
        self._sweep_task: Optional[asyncio.Task] = None

        # Metrics
        self._gaps_processed = 0
        self._code_generated = 0
        self._tests_passed = 0
        self._tests_failed = 0
        self._sandbox_unavailable = 0  # fail-closed blocks (containment could not run)
        self._deployments = 0
        self._reviews_expired = 0

    async def _setup(self) -> None:
        """Initialize service-specific setup."""
        # Initialize staging directories
        self._staging_manager.initialize()

        # Initialize code generation team
        self.logger.info("Initializing Meta-Programmer team with Ollama...")
        self._team = MetaProgrammerTeam(
            ollama_url=self.ollama_url,
            nats_client=self.event_bus._nc,
            sandbox_manager=self._sandbox_manager,
            staging_manager=self._staging_manager,
            db=self.database._connection,
        )

        # Approval consumer — handles Dashboard responses to DEFER requests.
        self._approval_consumer = ApprovalConsumer(
            staging=self._staging_manager,
            defer_ttl_ms=self.defer_ttl_ms,
            run_tests=self._sandbox_manager.run_tests,
            deploy=self._deploy_code,
            publish_gap_result=self._publish_gap_result,
            log=self.logger,
        )

        # Subscribe to knowledge gaps
        await self.event_bus.subscribe("knowledge.gap", self._handle_knowledge_gap)

        # Subscribe to status requests
        await self.event_bus.subscribe("metaprogrammer.status", self._handle_status)

        # Subscribe to human approval/denial responses from the Dashboard.
        await self.event_bus.subscribe(
            "approval.response.*", self._approval_consumer.handle_approval_response
        )

        # Periodically fail-close DEFERs that no human ever answered.
        self._sweep_task = asyncio.create_task(self._expire_reviews_loop())

    async def _cleanup(self) -> None:
        """Service-specific cleanup."""
        if self._sweep_task:
            self._sweep_task.cancel()
        if self._team:
            await self._team.close()

    async def _expire_reviews_loop(self) -> None:
        """Background loop: sweep expired human-review items (Phase 1.9)."""
        # Check at least once per minute, more often for short TTLs.
        interval = max(5.0, min(60.0, self.defer_ttl_ms / 1000 / 2))
        while True:
            try:
                await asyncio.sleep(interval)
                await self._sweep_expired_reviews(int(time.time() * 1000))
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001 — a sweep error must not kill the loop
                self.logger.error(f"Review-expiry sweep error: {e}")

    async def _sweep_expired_reviews(self, now_ms: int) -> int:
        """Reject + DENY every human-review item past its TTL. Returns the count."""
        expired = self._staging_manager.expired_reviews(now_ms, self.defer_ttl_ms)
        for trace_id in expired:
            self.logger.warning(f"DEFER expired (no human approval) — denying: {trace_id}")
            self._staging_manager.stage_rejected(trace_id, "DEFER expired — no human approval (fail-closed)")
            self._reviews_expired += 1
            await self._publish_gap_result(trace_id, False, "DEFER expired — no human approval (fail-closed)")
        return len(expired)

    async def _handle_knowledge_gap(self, data: dict[str, Any]) -> None:
        """
        Handle knowledge gap from Coordinator.

        Flow:
        1. Parse gap description
        2. Generate code using local LLM
        3. Request Kernel approval
        4. If ALLOW/TRANSFORM: test in sandbox
        5. If tests pass: deploy
        6. If DENY/DEFER: notify and halt
        """
        try:
            trace_id = data.get("trace_id", "")
            description = data.get("description", "")
            context = data.get("context", {})

            self.logger.info(f"Processing knowledge gap: {trace_id}")
            self._gaps_processed += 1

            # Generate code using local LLM
            self.logger.info(f"Generating code for: {description}")
            code_result = await self._team.generate_code(
                trace_id=trace_id,
                description=description,
                context=context,
            )

            if not code_result["success"]:
                self.logger.error(f"Code generation failed: {code_result['error']}")
                await self._publish_gap_result(trace_id, False, code_result["error"])
                return

            self._code_generated += 1
            target_path = code_result["target_path"]
            code_content = code_result["code"]
            test_content = code_result.get("tests", "")

            # Defence-in-depth: scan the FULL generated source (not just the
            # 500-char preview the Kernel sees) and refuse anything dangerous
            # before it is ever staged, submitted, or deployed.
            high = [f for f in scan_source(code_content) if f.severity == "high"]
            if high:
                reason = "Unsafe code blocked by full-source scan: " + "; ".join(
                    f"{f.rule} ({f.detail})" for f in high[:5]
                )
                self.logger.warning("%s — %s", trace_id, reason)
                self._staging_manager.stage_rejected(trace_id, reason)
                await self._publish_gap_result(trace_id, False, reason)
                return

            # Write to staging/pending
            staged_path = self._staging_manager.stage_pending(
                trace_id=trace_id,
                target_path=target_path,
                code=code_content,
                tests=test_content,
            )

            # Request Kernel approval
            self.logger.info(f"Requesting Kernel approval for: {target_path}")
            decision = await self._request_kernel_approval(
                trace_id=trace_id,
                target_path=target_path,
                code_preview=code_content[:500],  # First 500 chars
                proposed_action="CREATE" if not os.path.exists(target_path) else "MODIFY",
            )

            decision_type = decision.get("type", "DENY")

            if decision_type == "DENY":
                reason = decision.get("reason", "Unknown")
                self.logger.warning(f"Kernel denied code: {reason}")
                self._staging_manager.stage_rejected(trace_id, reason)
                await self._publish_gap_result(trace_id, False, f"Kernel denied: {reason}")
                return

            if decision_type == "DEFER":
                self.logger.info(f"Kernel deferred to human: {trace_id}")
                self._staging_manager.stage_human_review(trace_id)
                # Route to Dashboard for human approval
                await self.event_bus.publish(
                    "approval.request",
                    {
                        "trace_id": trace_id,
                        "type": "code_approval",
                        "description": description,
                        "target_path": target_path,
                        "code_preview": code_content[:1000],
                    },
                )
                return

            # ALLOW or TRANSFORM - proceed to testing
            self.logger.info(f"Kernel approved, moving to testing: {trace_id}")
            self._staging_manager.stage_testing(trace_id)

            # Run tests in sandbox
            test_result = await self._sandbox_manager.run_tests(
                code_path=staged_path,
                test_path=os.path.join(os.path.dirname(staged_path), "tests.py") if test_content else None,
            )

            # Fail closed: if containment itself could not run (no Docker daemon,
            # missing image, spawn failure) we must NOT deploy. This is distinct
            # from a genuine test failure — untested code is never deployed.
            if test_result.get("sandbox_unavailable"):
                reason = test_result.get("error", "Sandbox unavailable")
                self.logger.error(
                    "FAIL-CLOSED: sandbox unavailable for %s — blocking deploy. %s",
                    trace_id,
                    reason,
                )
                self._sandbox_unavailable += 1
                self._staging_manager.stage_rejected(trace_id, reason)
                await self._publish_gap_result(trace_id, False, reason, fail_closed=True)
                return

            if test_result["success"]:
                self.logger.info(f"Tests passed for: {trace_id}")
                self._tests_passed += 1
                self._staging_manager.stage_approved(trace_id)

                # Deploy
                await self._deploy_code(trace_id, target_path, code_content)
                self._deployments += 1

                # If this was a device discovery gap, notify the gateway
                # so it can hot-load the new plugin.
                if context.get("source") == "gateway_device_discovery":
                    await self.event_bus.publish("device.driver.ready", {
                        "trace_id": trace_id,
                        "target_path": target_path,
                        "device_id": context.get("device_id", ""),
                        "plugin_type": context.get("plugin_type", "sensor"),
                        "init_kwargs": context.get("metadata", {}),
                    })

                await self._publish_gap_result(trace_id, True, "Code generated, tested, and deployed")

            else:
                self.logger.error(f"Tests failed: {test_result['error']}")
                self._tests_failed += 1
                self._staging_manager.stage_rejected(trace_id, f"Tests failed: {test_result['error']}")
                await self._publish_gap_result(trace_id, False, f"Tests failed: {test_result['error']}")

        except Exception as e:
            self.logger.error(f"Error handling knowledge gap: {e}", exc_info=True)
            if 'trace_id' in locals():
                await self._publish_gap_result(trace_id, False, str(e))

    async def _request_kernel_approval(
        self,
        trace_id: str,
        target_path: str,
        code_preview: str,
        proposed_action: str,
    ) -> dict:
        """
        Request Kernel approval for code changes.

        Returns decision dict with type: ALLOW/TRANSFORM/DENY/DEFER
        """
        try:
            proposal = {
                "trace_id": trace_id,
                "gap_ref": trace_id,
                "proposed_action": proposed_action,
                "target_path": target_path,
                "code_preview": code_preview,
            }

            # Publish to Kernel
            await self.event_bus.publish("code.proposal", proposal)

            # Wait for decision
            decision = await self._wait_for_decision(trace_id, subject_prefix="code.decision")
            return decision

        except Exception as e:
            self.logger.error(f"Error requesting Kernel approval: {e}")
            return {"type": "DENY", "reason": str(e)}

    async def _wait_for_decision(
        self,
        trace_id: str,
        subject_prefix: str = "decision",
        timeout: float = 30.0,
    ) -> dict:
        """Wait for a Kernel decision."""
        try:
            decision = await self.event_bus.wait_for_decision(trace_id, timeout=timeout)
            return decision
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout waiting for decision: {trace_id}")
            return {"type": "DENY", "reason": "Decision timeout"}

    async def _deploy_code(self, trace_id: str, target_path: str, code: str) -> None:
        """Deploy approved code to target location."""
        try:
            # Refuse to write outside the deploy allowlist — reject `..` traversal
            # and symlink escapes, and use the resolved, validated path.
            ok, resolved = safe_deploy_path(
                target_path,
                allowlist=[self.plugins_root, self.tasks_root, self.staging_root],
            )
            if not ok:
                self.logger.error("Refusing deploy to unsafe path: %s", resolved)
                raise ValueError(f"Unsafe deploy path: {resolved}")
            target_path = resolved

            # Write atomically: snapshot any existing file, validate the new
            # code compiles, and roll back to the prior content (or remove a
            # newly-created file) on any failure — never leave a broken artifact.
            ok, detail = deploy_atomically(target_path, code)
            if not ok:
                self.logger.error("Deploy rolled back for %s: %s", target_path, detail)
                raise RuntimeError(f"Deploy rolled back: {detail}")

            self.logger.info(f"Deployed code to: {target_path}")

            # Log deployment
            import uuid
            await self.database.insert(
                "deployments",
                {
                    "id": str(uuid.uuid4()),
                    "trace_id": trace_id,
                    "target_path": target_path,
                    "timestamp": int(asyncio.get_event_loop().time() * 1000),
                },
            )

        except Exception as e:
            self.logger.error(f"Error deploying code: {e}")
            raise

    async def _publish_gap_result(
        self,
        trace_id: str,
        success: bool,
        message: str,
        fail_closed: bool = False,
    ) -> None:
        """Publish knowledge gap processing result.

        fail_closed=True marks results where containment could not run, so
        downstream consumers/metrics can distinguish a safety block from an
        ordinary test failure.
        """
        try:
            await self.event_bus.publish(
                f"knowledge.gap.result.{trace_id}",
                {
                    "trace_id": trace_id,
                    "success": success,
                    "message": message,
                    "fail_closed": fail_closed,
                },
            )
        except Exception as e:
            self.logger.error(f"Error publishing gap result: {e}")

    async def _handle_status(self, data: dict[str, Any]) -> None:
        """Handle status requests."""
        try:
            ac = self._approval_consumer
            status = {
                "status": "running",
                "metrics": {
                    "gaps_processed": self._gaps_processed,
                    "code_generated": self._code_generated,
                    "tests_passed": self._tests_passed + (ac.tests_passed if ac else 0),
                    "tests_failed": self._tests_failed + (ac.tests_failed if ac else 0),
                    "deployments": self._deployments + (ac.deployments if ac else 0),
                    "reviews_expired": self._reviews_expired,
                    "tests_passed": self._tests_passed,
                    "tests_failed": self._tests_failed,
                    "sandbox_unavailable": self._sandbox_unavailable,
                    "deployments": self._deployments,
                },
            }

            # Publish status response
            await self.event_bus.publish("metaprogrammer.status.response", status)
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")


async def main() -> None:
    """Main entry point."""
    service = MetaProgrammerService()
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
