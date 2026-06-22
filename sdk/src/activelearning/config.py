"""
Configuration module for ActiveLearningAI services.

Provides standardized configuration loading from environment variables
with sensible defaults.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

from activelearning.database import default_sqlite_path


@dataclass
class ServiceConfig:
    """
    Standard configuration for all ActiveLearningAI services.

    All values are loaded from environment variables with defaults
    for local development.
    """

    # Service identification
    service_name: str

    # Infrastructure URLs
    nats_url: str
    sqlite_path: str
    ollama_url: str
    qdrant_url: str

    # Paths
    tasks_root: str

    # Logging
    log_level: str

    # Path to a per-service NATS .creds file (NKEY + signed JWT).
    # None → unauthenticated / URL-embedded token (dev mode).
    nats_creds: Optional[str] = None

    @classmethod
    def from_env(cls, service_name: str) -> "ServiceConfig":
        """
        Create config from environment variables.

        Args:
            service_name: Name of the service (e.g., "memory", "planner")

        Returns:
            Configured ServiceConfig instance
        """
        return cls(
            service_name=service_name,
            nats_url=os.environ.get("NATS_URL", "nats://localhost:4222"),
            sqlite_path=default_sqlite_path(),
            ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
            qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
            tasks_root=os.environ.get("TASKS_ROOT", "/data/tasks"),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            nats_creds=os.environ.get("NATS_CREDS") or None,
        )

    def setup_logging(self) -> None:
        """Configure logging for the service."""
        logging.basicConfig(
            level=self.log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
