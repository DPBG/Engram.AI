"""
Sandbox Manager - Orchestrates Docker sandbox containers for testing code.

Spawns isolated containers with resource limits for safe code execution.
"""

import asyncio
import logging
import os
from typing import Optional

import docker
from docker.models.containers import Container

logger = logging.getLogger(__name__)


class SandboxManager:
    """
    Manages Docker sandbox containers for code testing.

    Sandboxes are isolated, resource-limited containers that:
    - Have no network access
    - Run with read-only filesystem
    - Have memory and CPU limits
    - Auto-destroy after execution
    """

    def __init__(self):
        self.docker_client = docker.from_env()
        self.sandbox_image = "activelearning-sandbox:latest"

        # Configurable limits from environment
        self.memory_limit = os.environ.get("SANDBOX_MEMORY_LIMIT", "512m")
        self.cpu_quota = int(os.environ.get("SANDBOX_CPU_QUOTA", "50000"))  # 50% of one core
        self.timeout = int(os.environ.get("SANDBOX_TIMEOUT_SECONDS", "30"))
        self.max_pids = int(os.environ.get("SANDBOX_MAX_PIDS", "100"))

    async def run_tests(
        self,
        code_path: str,
        test_path: Optional[str] = None,
    ) -> dict:
        """
        Run tests in an isolated sandbox container.

        Args:
            code_path: Path to the code file to test
            test_path: Optional path to test file (if separate)

        Returns:
            dict with keys:
                - success: bool indicating if tests passed
                - output: stdout from pytest
                - error: error message if failed
        """
        try:
            # Validate sandbox image exists
            try:
                self.docker_client.images.get(self.sandbox_image)
            except docker.errors.ImageNotFound:
                logger.error(f"Sandbox image not found: {self.sandbox_image}")
                return {
                    "success": False,
                    "output": "",
                    "error": f"Sandbox image not found: {self.sandbox_image}. Run: docker compose --profile build-only build sandbox-base",
                }

            # Prepare volume mounts (read-only)
            volumes = {}

            # Mount code directory
            code_dir = os.path.dirname(code_path)
            volumes[code_dir] = {"bind": "/sandbox", "mode": "ro"}

            # Determine test command
            if test_path and os.path.exists(test_path):
                test_file = os.path.basename(test_path)
                command = ["pytest", f"/sandbox/{test_file}", "--timeout=5", "--tb=short", "-v"]
            else:
                # Run pytest on all test files in directory
                command = ["pytest", "/sandbox", "--timeout=5", "--tb=short", "-v"]

            logger.info(f"Spawning sandbox for: {code_path}")

            # Run sandbox container
            container: Container = self.docker_client.containers.run(
                image=self.sandbox_image,
                command=command,
                volumes=volumes,
                network_disabled=True,            # No network access
                read_only=True,                   # Read-only root filesystem
                cap_drop=["ALL"],                 # Drop all Linux capabilities
                security_opt=["no-new-privileges"],  # Block privilege escalation
                mem_limit=self.memory_limit,
                cpu_quota=self.cpu_quota,
                pids_limit=self.max_pids,
                detach=True,
                remove=True,                      # Auto-destroy after completion
                tmpfs={"/tmp": "size=50M"},       # Writable /tmp only
            )

            # Wait for completion with timeout
            try:
                result = await asyncio.wait_for(
                    self._wait_for_container(container),
                    timeout=self.timeout,
                )

                output = container.logs().decode("utf-8", errors="replace")
                exit_code = result["StatusCode"]

                logger.info(f"Sandbox completed with exit code: {exit_code}")

                return {
                    "success": exit_code == 0,
                    "output": output,
                    "error": None if exit_code == 0 else f"Tests failed with exit code {exit_code}",
                }

            except asyncio.TimeoutError:
                logger.error(f"Sandbox timeout after {self.timeout}s")
                try:
                    container.kill()
                except:
                    pass
                return {
                    "success": False,
                    "output": "",
                    "error": f"Sandbox timeout after {self.timeout} seconds",
                }

        except Exception as e:
            logger.error(f"Sandbox error: {e}", exc_info=True)
            return {
                "success": False,
                "output": "",
                "error": str(e),
            }

    async def _wait_for_container(self, container: Container) -> dict:
        """Wait for container to complete."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, container.wait)

    def cleanup_old_containers(self) -> None:
        """Clean up any stale sandbox containers (shouldn't happen with auto-remove)."""
        try:
            containers = self.docker_client.containers.list(
                all=True,
                filters={"ancestor": self.sandbox_image}
            )
            for container in containers:
                logger.warning(f"Cleaning up stale sandbox: {container.id}")
                try:
                    container.remove(force=True)
                except:
                    pass
        except Exception as e:
            logger.error(f"Error cleaning up containers: {e}")
