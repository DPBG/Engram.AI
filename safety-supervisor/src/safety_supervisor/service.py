"""
Safety Supervisor Service - Risk analysis provider.

Analyzes proposals and provides risk assessments to the Kernel.
This service runs on the internal network only.

Uses NATS request-reply so the Kernel can synchronously await risk
analysis before making ALLOW/DENY/TRANSFORM/DEFER decisions.
"""

import asyncio
import json
from dataclasses import asdict

from activelearning import BaseService
from activelearning.nats_client import serialize_message

from safety_supervisor.analyzer import RiskAnalyzer


class SafetySupervisorService(BaseService):
    """
    Safety Supervisor service.

    Provides risk analysis for the Moral Kernel.
    Only accessible on the internal Docker network.
    """

    def __init__(self):
        super().__init__("safety-supervisor", use_database=False, use_event_bus=True)
        self._analyzer = RiskAnalyzer()

        # Metrics
        self._analysis_count = 0
        self._high_risk_count = 0

    async def _setup(self) -> None:
        """Service-specific setup."""
        # Subscribe to analysis requests (from Kernel) as request-reply handlers
        await self.event_bus.subscribe(
            "safety.analyze.action",
            self._handle_action_analysis,
            is_request_handler=True,
        )
        await self.event_bus.subscribe(
            "safety.analyze.code",
            self._handle_code_analysis,
            is_request_handler=True,
        )

        # Subscribe to status requests
        await self.event_bus.subscribe("safety.status", self._handle_status)

    async def _cleanup(self) -> None:
        """Service-specific cleanup."""
        pass

    async def _handle_action_analysis(self, data: dict, msg=None) -> None:
        """Handle action analysis requests from Kernel via request-reply."""
        try:
            trace_id = data.get("trace_id", "")

            self.logger.debug(f"Analyzing action: {trace_id}")

            # Perform analysis
            analysis = self._analyzer.analyze_action(data)

            # Update metrics
            self._analysis_count += 1
            if analysis.risk_score >= 0.5:
                self._high_risk_count += 1

            response = asdict(analysis)

            # Reply directly if this is a request-reply message
            if msg and msg.reply:
                await msg.respond(serialize_message(response))
            else:
                # Fallback: publish to topic (backward compat)
                await self.event_bus.publish(
                    f"safety.analysis.action.{trace_id}",
                    response,
                )

            self.logger.debug(f"Action {trace_id}: risk={analysis.risk_score:.2f}, flags={analysis.flags}")

        except Exception as e:
            self.logger.error(f"Error analyzing action: {e}")

    async def _handle_code_analysis(self, data: dict, msg=None) -> None:
        """Handle code analysis requests from Kernel via request-reply."""
        try:
            trace_id = data.get("trace_id", "")

            self.logger.debug(f"Analyzing code: {trace_id}")

            # Perform analysis
            analysis = self._analyzer.analyze_code(data)

            # Update metrics
            self._analysis_count += 1
            if analysis.risk_score >= 0.5:
                self._high_risk_count += 1

            response = asdict(analysis)

            # Reply directly if this is a request-reply message
            if msg and msg.reply:
                await msg.respond(serialize_message(response))
            else:
                # Fallback: publish to topic (backward compat)
                await self.event_bus.publish(
                    f"safety.analysis.code.{trace_id}",
                    response,
                )

            self.logger.debug(f"Code {trace_id}: risk={analysis.risk_score:.2f}, flags={analysis.flags}")

        except Exception as e:
            self.logger.error(f"Error analyzing code: {e}")

    async def _handle_status(self, data: dict) -> None:
        """Handle status requests."""
        try:
            status = {
                "status": "running",
                "metrics": {
                    "analysis_count": self._analysis_count,
                    "high_risk_count": self._high_risk_count,
                },
            }

            await self.event_bus.publish("safety.status.response", status)
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")


async def main() -> None:
    """Main entry point."""
    service = SafetySupervisorService()
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
