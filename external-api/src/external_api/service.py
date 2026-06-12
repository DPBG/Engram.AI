"""
External API Service - Main service for external knowledge queries.
"""

import asyncio
import json
import logging
import os
import signal
import sys

import aiosqlite
import nats

from external_api.manager import ExternalAPIManager

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ExternalAPIService:
    """
    External API service for safe knowledge queries.
    """

    def __init__(self):
        self.nats_url = os.environ.get("NATS_URL", "nats://localhost:4222")
        self.sqlite_path = os.environ.get("SQLITE_PATH", "/data/sqlite/unified.db")

        self._nc: nats.aio.client.Client | None = None
        self._db: aiosqlite.Connection | None = None
        self._shutdown_event = asyncio.Event()

        self._manager: ExternalAPIManager | None = None

        # Metrics
        self._queries_total = 0
        self._queries_success = 0
        self._queries_failed = 0
        self._conflicts_detected = 0

    async def start(self) -> None:
        """Start the external API service."""
        logger.info("Starting External API Service...")

        # Connect to NATS
        logger.info(f"Connecting to NATS at {self.nats_url}")
        self._nc = await nats.connect(self.nats_url)

        # Connect to SQLite
        logger.info(f"Connecting to SQLite at {self.sqlite_path}")
        os.makedirs(os.path.dirname(self.sqlite_path), exist_ok=True)
        self._db = await aiosqlite.connect(self.sqlite_path)

        # Initialize manager
        self._manager = ExternalAPIManager(
            nats_client=self._nc,
            db=self._db,
        )

        # Subscribe to NATS subjects
        await self._nc.subscribe("external.query", cb=self._handle_query)
        await self._nc.subscribe("external.status", cb=self._handle_status)

        logger.info("External API Service started successfully")

    async def stop(self) -> None:
        """Stop the external API service."""
        logger.info("Stopping External API Service...")

        if self._nc:
            await self._nc.drain()
            await self._nc.close()

        if self._db:
            await self._db.close()

        logger.info("External API Service stopped")

    async def run(self) -> None:
        """Run the service until shutdown."""
        await self.start()
        await self._shutdown_event.wait()
        await self.stop()

    def shutdown(self) -> None:
        """Signal the service to shut down."""
        self._shutdown_event.set()

    async def _handle_query(self, msg) -> None:
        """Handle external knowledge query."""
        try:
            request = json.loads(msg.data.decode())
            query = request.get("query", "")
            context = request.get("context")
            local_knowledge = request.get("local_knowledge")

            logger.info(f"External query: {query[:50]}...")
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
                await self._nc.publish(msg.reply, json.dumps(result).encode())

        except Exception as e:
            logger.error(f"Error handling query: {e}", exc_info=True)
            self._queries_failed += 1

    async def _handle_status(self, msg) -> None:
        """Handle status request."""
        try:
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
                await self._nc.publish(msg.reply, json.dumps(status).encode())

        except Exception as e:
            logger.error(f"Error getting status: {e}")


async def main() -> None:
    """Main entry point."""
    service = ExternalAPIService()

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
