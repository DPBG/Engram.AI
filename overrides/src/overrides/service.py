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
import json
import logging
import os
import signal
import sys
from typing import Optional

import aiosqlite
import nats

from overrides.verifier import HumanVerifier
from overrides.processor import OverrideProcessor

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class OverrideService:
    """
    Human Override Service.

    Allows verified humans to override operational parameters while
    keeping safety rules immutable.
    """

    def __init__(self):
        self.nats_url = os.environ.get("NATS_URL", "nats://localhost:4222")
        self.sqlite_path = os.environ.get("SQLITE_PATH", "/data/sqlite/unified.db")

        self._nc: Optional[nats.aio.client.Client] = None
        self._db: Optional[aiosqlite.Connection] = None
        self._shutdown_event = asyncio.Event()

        self._verifier: Optional[HumanVerifier] = None
        self._processor: Optional[OverrideProcessor] = None

        # Metrics
        self._overrides_requested = 0
        self._overrides_verified = 0
        self._overrides_applied = 0
        self._overrides_rejected = 0

    async def start(self) -> None:
        """Start the override service."""
        logger.info("Starting Human Override System...")

        # Connect to NATS
        logger.info(f"Connecting to NATS at {self.nats_url}")
        self._nc = await nats.connect(self.nats_url)

        # Connect to SQLite
        logger.info(f"Connecting to SQLite at {self.sqlite_path}")
        os.makedirs(os.path.dirname(self.sqlite_path), exist_ok=True)
        self._db = await aiosqlite.connect(self.sqlite_path)

        # Initialize components
        self._verifier = HumanVerifier()
        self._processor = OverrideProcessor(self._nc, self._db)

        # Subscribe to override requests
        await self._nc.subscribe("override.request", cb=self._handle_override_request)

        # Subscribe to status requests
        await self._nc.subscribe("override.status", cb=self._handle_status)

        logger.info("Human Override System started successfully")

    async def stop(self) -> None:
        """Stop the override service."""
        logger.info("Stopping Human Override System...")

        if self._nc:
            await self._nc.drain()
            await self._nc.close()

        if self._db:
            await self._db.close()

        logger.info("Human Override System stopped")

    async def run(self) -> None:
        """Run the service until shutdown."""
        await self.start()
        await self._shutdown_event.wait()
        await self.stop()

    def shutdown(self) -> None:
        """Signal the service to shut down."""
        self._shutdown_event.set()

    async def _handle_override_request(self, msg) -> None:
        """
        Handle override request from user.

        Flow:
        1. Parse request
        2. Verify human presence
        3. Parse override prompt
        4. Validate with Kernel (for safety-critical params)
        5. Apply override
        """
        try:
            request = json.loads(msg.data.decode())
            trace_id = request.get("trace_id", "")
            prompt = request.get("prompt", "")

            logger.info(f"Override request: {trace_id} - {prompt}")
            self._overrides_requested += 1

            # Step 1: Verify human presence
            logger.info("Verifying human presence...")
            verification = await self._verifier.verify_human()

            if not verification["verified"]:
                logger.warning(f"Human verification failed: {verification['error']}")
                self._overrides_rejected += 1
                await self._publish_override_result(
                    trace_id=trace_id,
                    success=False,
                    message=f"Verification failed: {verification['error']}",
                )
                return

            logger.info(f"Human verified via {verification['method']} (confidence: {verification['confidence']:.2f})")
            self._overrides_verified += 1

            # Step 2: Parse override prompt
            parsed = await self._processor.parse_override(prompt)

            if not parsed["success"]:
                logger.error(f"Failed to parse override: {parsed['error']}")
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
                logger.info(f"Requesting Kernel validation for: {parameter}")
                kernel_decision = await self._request_kernel_validation(
                    trace_id=trace_id,
                    parameter=parameter,
                    value=value,
                )

                if kernel_decision.get("type") != "ALLOW":
                    reason = kernel_decision.get("reason", "Unknown")
                    logger.warning(f"Kernel rejected override: {reason}")
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
                logger.info(f"Override applied: {parameter} = {value}")
                self._overrides_applied += 1
                await self._publish_override_result(
                    trace_id=trace_id,
                    success=True,
                    message=f"Override applied: {parameter} = {value}",
                )
            else:
                logger.error(f"Failed to apply override: {result['error']}")
                self._overrides_rejected += 1
                await self._publish_override_result(
                    trace_id=trace_id,
                    success=False,
                    message=f"Apply error: {result['error']}",
                )

        except Exception as e:
            logger.error(f"Error handling override request: {e}", exc_info=True)
            if 'trace_id' in locals():
                await self._publish_override_result(
                    trace_id=trace_id,
                    success=False,
                    message=str(e),
                )

    async def _request_kernel_validation(
        self,
        trace_id: str,
        parameter: str,
        value: any,
    ) -> dict:
        """Request Kernel validation for override."""
        try:
            proposal = {
                "trace_id": trace_id,
                "type": "override",
                "parameter": parameter,
                "value": value,
            }

            await self._nc.publish("proposal.new", json.dumps(proposal).encode())

            # Wait for decision
            decision_subject = f"decision.{trace_id}"
            future = asyncio.Future()

            async def decision_handler(msg):
                if not future.done():
                    future.set_result(json.loads(msg.data.decode()))

            sub = await self._nc.subscribe(decision_subject, cb=decision_handler)

            try:
                decision = await asyncio.wait_for(future, timeout=10.0)
                return decision
            except asyncio.TimeoutError:
                logger.error(f"Timeout waiting for Kernel decision: {trace_id}")
                return {"type": "DENY", "reason": "Decision timeout"}
            finally:
                await sub.unsubscribe()

        except Exception as e:
            logger.error(f"Error requesting Kernel validation: {e}")
            return {"type": "DENY", "reason": str(e)}

    async def _publish_override_result(
        self,
        trace_id: str,
        success: bool,
        message: str,
    ) -> None:
        """Publish override result."""
        try:
            await self._nc.publish(
                f"override.result.{trace_id}",
                json.dumps({
                    "trace_id": trace_id,
                    "success": success,
                    "message": message,
                }).encode(),
            )
        except Exception as e:
            logger.error(f"Error publishing override result: {e}")

    async def _handle_status(self, msg) -> None:
        """Handle status requests."""
        try:
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
                await self._nc.publish(msg.reply, json.dumps(status).encode())
        except Exception as e:
            logger.error(f"Error getting status: {e}")


async def main() -> None:
    """Main entry point."""
    service = OverrideService()

    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        service.shutdown()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass  # Windows: add_signal_handler unsupported; Ctrl+C still works

    try:
        await service.run()
    except Exception as e:
        logger.error(f"Service error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
