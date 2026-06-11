"""
Base service class for all ActiveLearningAI components.

Provides common infrastructure for NATS, database, logging,
signal handling, and service lifecycle management.
"""

import asyncio
import logging
import signal
from typing import Optional

from activelearning.config import ServiceConfig
from activelearning.database import Database, get_database
from activelearning.nats_client import EventBus

logger = logging.getLogger(__name__)


class BaseService:
    """
    Base class for all ActiveLearningAI services.

    Provides:
    - Configuration from environment
    - NATS event bus connection
    - SQLite database connection
    - Signal handling (SIGTERM, SIGINT)
    - Lifecycle management (start, stop, run, shutdown)

    Usage:
        class MyService(BaseService):
            def __init__(self):
                super().__init__("my-service")

            async def _setup(self) -> None:
                # Service-specific setup
                await self.event_bus.subscribe("my.topic", self._handle_msg)

            async def _handle_msg(self, data: dict) -> None:
                # Handle messages
                pass

        async def main():
            service = MyService()
            await service.run()

        if __name__ == "__main__":
            asyncio.run(main())
    """

    def __init__(
        self,
        service_name: str,
        use_database: bool = True,
        use_event_bus: bool = True,
    ):
        """
        Initialize the base service.

        Args:
            service_name: Name of the service (e.g., "memory", "planner")
            use_database: Whether this service needs database access
            use_event_bus: Whether this service needs NATS event bus
        """
        self.service_name = service_name
        self.config = ServiceConfig.from_env(service_name)
        self.use_database = use_database
        self.use_event_bus = use_event_bus

        # Setup logging
        self.config.setup_logging()
        self.logger = logging.getLogger(service_name)

        # Infrastructure components
        self.event_bus: Optional[EventBus] = None
        self.database: Optional[Database] = None
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """
        Start the service.

        Connects to NATS, database, and calls service-specific _setup().
        """
        self.logger.info(f"Starting {self.service_name} service...")

        # Connect to event bus
        if self.use_event_bus:
            self.event_bus = EventBus(
                nats_url=self.config.nats_url,
                name=self.service_name,
            )
            await self.event_bus.connect()
            self.logger.info("Connected to NATS")

        # Connect to database
        if self.use_database:
            self.database = await get_database()
            self.logger.info("Connected to database")

        # Service-specific setup
        await self._setup()

        self.logger.info(f"{self.service_name} service started successfully")

    async def stop(self) -> None:
        """
        Stop the service.

        Calls service-specific _cleanup(), then closes NATS and database.
        """
        self.logger.info(f"Stopping {self.service_name} service...")

        # Service-specific cleanup
        await self._cleanup()

        # Close event bus
        if self.event_bus:
            await self.event_bus.close()
            self.logger.info("Disconnected from NATS")

        # Database is a singleton, don't close it here
        # It will be closed when the process exits

        self.logger.info(f"{self.service_name} service stopped")

    async def run(self) -> None:
        """
        Run the service until shutdown signal.

        Sets up signal handlers and waits for SIGTERM or SIGINT.
        """
        # Setup signal handlers
        loop = asyncio.get_event_loop()

        def signal_handler() -> None:
            self.logger.info("Received shutdown signal")
            self.shutdown()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, signal_handler)
            except NotImplementedError:
                # Windows asyncio event loops don't support add_signal_handler;
                # Ctrl+C still raises KeyboardInterrupt which unwinds run().
                pass

        # Start the service
        try:
            await self.start()
            # Wait for shutdown
            await self._shutdown_event.wait()
        except Exception as e:
            self.logger.error(f"{self.service_name} error: {e}", exc_info=True)
            raise
        finally:
            await self.stop()

    def shutdown(self) -> None:
        """Signal the service to shut down."""
        self._shutdown_event.set()

    async def _setup(self) -> None:
        """
        Service-specific setup.

        Override this method to add service-specific initialization:
        - Subscribe to NATS subjects
        - Initialize service-specific resources
        - Create collections, tables, etc.
        """
        pass

    async def _cleanup(self) -> None:
        """
        Service-specific cleanup.

        Override this method to add service-specific shutdown logic:
        - Unsubscribe from NATS subjects
        - Close service-specific resources
        """
        pass
