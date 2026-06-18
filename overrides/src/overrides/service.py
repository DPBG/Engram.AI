"""
Override Service - Human override system coordinator.

Manages the flow:
1. Receive override request
2. Verify human presence
3. Parse override prompt
4. Request Kernel validation
5. Apply override if approved
"""

import asyncio
from typing import Optional

from activelearning import BaseService
from activelearning.nats_client import serialize_message

from overrides.verifier import HumanVerifier
from overrides.processor import OverrideProcessor


class OverrideService(BaseService):
    """
    Human Override Service.

    Allows verified humans to override operational parameters while
    keeping safety rules immutable.
    """

    def __init__(self):
        super().__init__("overrides", use_database=True, use_event_bus=True)

        self._verifier: Optional[HumanVerifier] = None
        self._processor: Optional[OverrideProcessor] = None

        # Metrics
        self._overrides_requested = 0
        self._overrides_verified = 0
        self._overrides_applied = 0
        self._overrides_rejected = 0

    async def _setup(self) -> None:
        """Service-specific setup."""
        self._verifier = HumanVerifier()
        self._processor = OverrideProcessor(self.event_bus, self.database)

        await self.event_bus.subscribe("override.request", self._handle_override_request)
        await self.event_bus.subscribe(
            "override.status",
            self._handle_status,
            is_request_handler=True,
        )

    async def _handle_override_request(self, data: dict) -> None:
        """
        Handle override request from user.

        Flow:
        1. Parse request
        2. Verify human presence
        3. Parse override prompt
        4. Validate with Kernel (for safety-critical params)
        5. Apply override
        """
        trace_id = data.get("trace_id", "")
        prompt = data.get("prompt", "")

        try:
            await self._process_override_request(trace_id, prompt)
        except Exception as e:
            self.logger.error(f"Error handling override request: {e}", exc_info=True)
            if trace_id:
                await self._publish_override_result(
                    trace_id=trace_id,
                    success=False,
                    message=str(e),
                )

    async def _process_override_request(self, trace_id: str, prompt: str) -> None:
        self.logger.info(f"Override request: {trace_id} - {prompt}")
        self._overrides_requested += 1

        # Step 1: Verify human presence
        self.logger.info("Verifying human presence...")
        verification = await self._verifier.verify_human()

        if not verification["verified"]:
            self.logger.warning(f"Human verification failed: {verification['error']}")
            self._overrides_rejected += 1
            await self._publish_override_result(
                trace_id=trace_id,
                success=False,
                message=f"Verification failed: {verification['error']}",
            )
            return

        self.logger.info(
            f"Human verified via {verification['method']} "
            f"(confidence: {verification['confidence']:.2f})"
        )
        self._overrides_verified += 1

        # Step 2: Parse override prompt
        parsed = await self._processor.parse_override(prompt)

        if not parsed["success"]:
            self.logger.error(f"Failed to parse override: {parsed['error']}")
            self._overrides_rejected += 1
            await self._publish_override_result(
                trace_id=trace_id,
                success=False,
                message=f"Parse error: {parsed['error']}",
            )
            return

        parameter = parsed["parameter"]
        value = parsed["value"]
        requires_kernel = parsed["requires_kernel"]

        # Step 3: Validate with Kernel if needed
        if requires_kernel:
            self.logger.info(f"Requesting Kernel validation for: {parameter}")
            kernel_decision = await self._request_kernel_validation(
                trace_id=trace_id,
                parameter=parameter,
                value=value,
            )

            if kernel_decision.get("type") != "ALLOW":
                reason = kernel_decision.get("reason", "Unknown")
                self.logger.warning(f"Kernel rejected override: {reason}")
                self._overrides_rejected += 1
                await self._publish_override_result(
                    trace_id=trace_id,
                    success=False,
                    message=f"Kernel rejected: {reason}",
                )
                return

        # Step 4: Apply override
        result = await self._processor.apply_override(
            trace_id=trace_id,
            parameter=parameter,
            value=value,
            verified_by=verification["method"],
        )

        if result["success"]:
            self.logger.info(f"Override applied: {parameter} = {value}")
            self._overrides_applied += 1
            await self._publish_override_result(
                trace_id=trace_id,
                success=True,
                message=f"Override applied: {parameter} = {value}",
            )
        else:
            self.logger.error(f"Failed to apply override: {result['error']}")
            self._overrides_rejected += 1
            await self._publish_override_result(
                trace_id=trace_id,
                success=False,
                message=f"Apply error: {result['error']}",
            )

    async def _request_kernel_validation(
        self,
        trace_id: str,
        parameter: str,
        value: object,
    ) -> dict:
        """Request Kernel validation for override."""
        try:
            proposal = {
                "trace_id": trace_id,
                "type": "override",
                "parameter": parameter,
                "value": value,
            }

            await self.event_bus.publish("proposal.new", proposal)
            return await self.event_bus.wait_for_decision(trace_id, timeout=10.0)
        except TimeoutError:
            self.logger.error(f"Timeout waiting for Kernel decision: {trace_id}")
            return {"type": "DENY", "reason": "Decision timeout"}
        except Exception as e:
            self.logger.error(f"Error requesting Kernel validation: {e}")
            return {"type": "DENY", "reason": str(e)}

    async def _publish_override_result(
        self,
        trace_id: str,
        success: bool,
        message: str,
    ) -> None:
        """Publish override result."""
        await self.event_bus.publish(
            f"override.result.{trace_id}",
            {
                "trace_id": trace_id,
                "success": success,
                "message": message,
            },
        )

    async def _handle_status(self, _data: dict, msg) -> None:
        """Handle status requests."""
        status = {
            "status": "running",
            "metrics": {
                "overrides_requested": self._overrides_requested,
                "overrides_verified": self._overrides_verified,
                "overrides_applied": self._overrides_applied,
                "overrides_rejected": self._overrides_rejected,
            },
            "verification_methods": {
                "camera": self._verifier._camera_available if self._verifier else False,
                "microphone": self._verifier._microphone_available if self._verifier else False,
                "button": self._verifier._physical_button_available if self._verifier else False,
            },
        }

        if msg.reply:
            await msg.respond(serialize_message(status))


async def main() -> None:
    """Main entry point."""
    service = OverrideService()
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
