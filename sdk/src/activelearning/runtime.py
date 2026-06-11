"""
SDK Runtime - Main entry point for the ActiveLearningAI SDK container.

This module initializes the SDK, connects to infrastructure services,
and provides runtime utilities for other components.
"""

import asyncio
import logging
import os
import signal
import sys

from activelearning.nats_client import EventBus
from activelearning.embeddings import EmbeddingService
from activelearning.database import Database

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SDKRuntime:
    """
    Main runtime for the ActiveLearningAI SDK.

    Manages connections to infrastructure services and provides
    shared resources to other components.
    """

    def __init__(self):
        self.bus: EventBus = EventBus()
        self.embeddings: EmbeddingService = EmbeddingService()
        self.db: Database = Database()
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the SDK runtime."""
        logger.info("Starting SDK Runtime...")

        # Connect to NATS
        logger.info("Connecting to NATS...")
        await self.bus.connect()

        # Initialize database
        logger.info("Initializing database...")
        await self.db.initialize()

        # Check Ollama availability
        if await self.embeddings.is_available():
            logger.info("Embedding service available")
        else:
            logger.warning("Embedding service not available - some features may be limited")

        # Subscribe to system events
        await self.bus.subscribe("system.shutdown", self._handle_shutdown)
        await self.bus.subscribe("system.health", self._handle_health_check)

        # Publish startup event
        await self.bus.publish("system.health", {
            "component": "sdk",
            "status": "running",
        })

        logger.info("SDK Runtime started successfully")

    async def stop(self) -> None:
        """Stop the SDK runtime."""
        logger.info("Stopping SDK Runtime...")

        # Publish shutdown event
        await self.bus.publish("system.health", {
            "component": "sdk",
            "status": "stopping",
        })

        # Close connections
        await self.bus.close()
        await self.db.close()

        logger.info("SDK Runtime stopped")

    async def run(self) -> None:
        """Run the SDK runtime until shutdown."""
        await self.start()

        # Wait for shutdown signal
        await self._shutdown_event.wait()

        await self.stop()

    async def _handle_shutdown(self, data: dict) -> None:
        """Handle system shutdown event."""
        logger.info(f"Received shutdown event: {data}")
        self._shutdown_event.set()

    async def _handle_health_check(self, data: dict) -> None:
        """Handle health check events."""
        if data.get("request") == "ping":
            await self.bus.publish("system.health", {
                "component": "sdk",
                "status": "running",
                "response": "pong",
            })

    def shutdown(self) -> None:
        """Signal the runtime to shut down."""
        self._shutdown_event.set()


# Global runtime instance
_runtime: SDKRuntime | None = None


async def get_runtime() -> SDKRuntime:
    """Get the global SDK runtime instance."""
    global _runtime
    if _runtime is None:
        _runtime = SDKRuntime()
        await _runtime.start()
    return _runtime


async def main() -> None:
    """Main entry point for the SDK runtime."""
    runtime = SDKRuntime()

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        runtime.shutdown()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass  # Windows: add_signal_handler unsupported; Ctrl+C still works

    try:
        await runtime.run()
    except Exception as e:
        logger.error(f"Runtime error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
