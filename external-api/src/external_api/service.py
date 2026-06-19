"""
External API Service - Main service for external knowledge queries.
"""

import asyncio

from activelearning import BaseService
from activelearning.nats_client import serialize_message

from external_api.manager import ExternalAPIManager


class ExternalAPIService(BaseService):
    """
    External API service for safe knowledge queries.
    """

    def __init__(self):
        super().__init__("external-api", use_database=True, use_event_bus=True)
        self._manager: ExternalAPIManager | None = None

        # Metrics
        self._queries_total = 0
        self._queries_success = 0
        self._queries_failed = 0
        self._conflicts_detected = 0

    async def _setup(self) -> None:
        """Service-specific setup."""
        self._manager = ExternalAPIManager(
            event_bus=self.event_bus,
            db=self.database,
        )

        await self.event_bus.subscribe(
            "external.query",
            self._handle_query,
            is_request_handler=True,
        )
        await self.event_bus.subscribe(
            "external.status",
            self._handle_status,
            is_request_handler=True,
        )

    async def _handle_query(self, data: dict, msg) -> None:
        """Handle external knowledge query."""
        query = data.get("query", "")
        context = data.get("context")
        local_knowledge = data.get("local_knowledge")

        self.logger.info(f"External query: {query[:50]}...")
        self._queries_total += 1

        result = await self._manager.query_external(
            query=query,
            context=context,
            local_knowledge=local_knowledge,
        )

        if result["success"]:
            self._queries_success += 1
            if result["conflict"]:
                self._conflicts_detected += 1
        else:
            self._queries_failed += 1

        if msg.reply:
            await msg.respond(serialize_message(result))

    async def _handle_status(self, _data: dict, msg) -> None:
        """Handle status request."""
        status = {
            "status": "running",
            "metrics": {
                "queries_total": self._queries_total,
                "queries_success": self._queries_success,
                "queries_failed": self._queries_failed,
                "conflicts_detected": self._conflicts_detected,
            },
            "apis": {
                "claude": self._manager._claude_client is not None,
                "openai": self._manager._openai_client is not None,
            },
        }

        if msg.reply:
            await msg.respond(serialize_message(status))


async def main() -> None:
    """Main entry point."""
    service = ExternalAPIService()
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
