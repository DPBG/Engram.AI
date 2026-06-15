"""
Coordinator Service - Main multi-sensory learning and task coordination service.

Integrates:
- SensorManager (sensor detection and priorities)
- LearningController (demonstration learning)
- TaskCoordinator (task lookup and execution)
"""

import asyncio
import uuid
from typing import Optional

from activelearning import BaseService, get_embedding_service

from coordinator.sensor_manager import SensorManager
from coordinator.learning_controller import LearningController
from coordinator.task_coordinator import TaskCoordinator
from coordinator.gate import (
    KERNEL_PROPOSAL_SUBJECT,
    build_execution_proposal,
    decision_allows,
)


class CoordinatorService(BaseService):
    """
    Coordinator service for multi-sensory learning and task execution.
    """

    def __init__(self):
        super().__init__("coordinator", use_database=False, use_event_bus=True)

        self.tasks_root = self.config.tasks_root
        self.qdrant_url = self.config.qdrant_url
        self.ollama_url = self.config.ollama_url

        # Components
        self._sensor_manager: Optional[SensorManager] = None
        self._learning_controller: Optional[LearningController] = None
        self._task_coordinator: Optional[TaskCoordinator] = None
        self._embedding_service = get_embedding_service()

    async def _setup(self) -> None:
        """Setup service-specific resources and NATS subscriptions."""
        self.logger.info("Initializing Sensor Manager...")
        self._sensor_manager = SensorManager()
        await self._sensor_manager.detect_sensors()

        # Initialize learning controller
        self._learning_controller = LearningController(
            nats_client=self.event_bus._nc,
            sensor_manager=self._sensor_manager,
            tasks_root=self.tasks_root,
        )

        # Initialize task coordinator
        self._task_coordinator = TaskCoordinator(
            nats_client=self.event_bus._nc,
            qdrant_url=self.qdrant_url,
            ollama_url=self.ollama_url,
            tasks_root=self.tasks_root,
        )

        # Track devices already forwarded to meta-programmer (prevent flooding)
        self._pending_device_gaps: set[str] = set()

        # Subscribe to NATS subjects
        await self.event_bus.subscribe("task.request", self._handle_task_request)
        await self.event_bus.subscribe("demo.start", self._handle_demo_start)
        await self.event_bus.subscribe("demo.observation", self._handle_observation)
        await self.event_bus.subscribe("demo.finish", self._handle_demo_finish)
        await self.event_bus.subscribe("coordinator.status", self._handle_status)
        await self.event_bus.subscribe("device.unknown", self._handle_unknown_device)

        self.logger.info("Coordinator setup completed")

    async def _cleanup(self) -> None:
        """Cleanup service-specific resources."""
        # Unsubscribe from topics if needed
        pass

    async def _handle_task_request(self, data: dict) -> None:
        """
        Handle task execution request.

        Flow:
        1. Search for matching learned task
        2. If found with high confidence: execute
        3. If found with medium confidence: adapt
        4. If not found: trigger knowledge gap
        """
        try:
            query = data.get("query", "")
            parameters = data.get("parameters", {})

            self.logger.info(f"Task request: {query}")

            # Find matching task
            match = await self._task_coordinator.find_task(query)

            if match["action"] == "execute":
                # Gate every execution through the Kernel (Phase 1.6). The task
                # only runs on an ALLOW/TRANSFORM decision; DENY/DEFER/timeout
                # fail closed.
                decision = await self._request_execution_approval(
                    match["task_id"], parameters
                )
                if not decision_allows(decision):
                    reason = (decision or {}).get("reason", "Kernel did not approve execution")
                    self.logger.warning(
                        f"Task execution blocked by Kernel: {match['task_id']} "
                        f"({(decision or {}).get('type', 'NONE')}: {reason})"
                    )
                    await self.event_bus.publish("task.result", {
                        "success": False,
                        "blocked": True,
                        "task_id": match["task_id"],
                        "decision": (decision or {}).get("type", "NONE"),
                        "reason": reason,
                    })
                    return

                # Approved — execute existing task
                self.logger.info(f"Executing task: {match['task_id']}")
                result = await self._task_coordinator.execute_task(
                    match["task_id"],
                    parameters,
                )

                # Publish result
                await self.event_bus.publish("task.result", result)

            elif match["action"] == "adapt":
                # Adapt existing task (delegate to Meta-Programmer)
                self.logger.info(f"Adapting task: {match['task_id']}")
                trace_id = await self._task_coordinator.trigger_knowledge_gap(
                    query=f"Adapt task {match['task_id']} for: {query}",
                    context={"base_task": match["task_id"], "parameters": parameters},
                )

                await self.event_bus.publish("task.result", {
                    "success": False,
                    "message": "Task adaptation in progress",
                    "trace_id": trace_id,
                })

            elif match["action"] == "learn":
                # Learn new task
                self.logger.info(f"Learning new task for: {query}")
                trace_id = await self._task_coordinator.trigger_knowledge_gap(
                    query=query,
                    context=parameters,
                )

                await self.event_bus.publish("task.result", {
                    "success": False,
                    "message": "Task learning in progress. Please demonstrate or describe the task.",
                    "trace_id": trace_id,
                })

        except Exception as e:
            self.logger.error(f"Error handling task request: {e}", exc_info=True)

    async def _request_execution_approval(
        self,
        task_id: str,
        parameters: Optional[dict],
    ) -> dict:
        """Ask the Kernel to approve executing a task; fail closed on timeout.

        Publishes an action proposal and waits for the signed ``decision.<trace>``
        reply. A timeout (or any error) yields a synthetic DENY so the caller
        declines to execute.
        """
        trace_id = str(uuid.uuid4())
        proposal = build_execution_proposal(trace_id, task_id, parameters)
        try:
            await self.event_bus.publish(KERNEL_PROPOSAL_SUBJECT, proposal)
            return await self.event_bus.wait_for_decision(trace_id, timeout=30.0)
        except asyncio.TimeoutError:
            self.logger.error(f"Kernel decision timeout for task {task_id} (trace={trace_id})")
            return {"type": "DENY", "reason": "Kernel decision timeout"}
        except Exception as e:
            self.logger.error(f"Error requesting Kernel approval for {task_id}: {e}")
            return {"type": "DENY", "reason": str(e)}

    async def _handle_demo_start(self, data: dict) -> None:
        """Handle demonstration start request."""
        try:
            task_name = data.get("task_name", "")
            description = data.get("description", "")

            self.logger.info(f"Starting demonstration: {task_name}")

            trace_id = await self._learning_controller.start_demonstration(
                task_name=task_name,
                description=description,
            )

            await self.event_bus.publish("demo.started", {
                "success": True,
                "trace_id": trace_id,
                "message": "Demonstration started. Begin demonstrating the task.",
            })

        except Exception as e:
            self.logger.error(f"Error starting demonstration: {e}", exc_info=True)

    async def _handle_observation(self, data: dict) -> None:
        """Handle observation during demonstration."""
        try:
            sensor_id = data.get("sensor_id", "")
            obs_data = data.get("data", {})
            timestamp = data.get("timestamp", 0)

            await self._learning_controller.record_observation(
                sensor_id=sensor_id,
                data=obs_data,
                timestamp=timestamp,
            )

        except Exception as e:
            self.logger.error(f"Error handling observation: {e}", exc_info=True)

    async def _handle_demo_finish(self, data: dict) -> None:
        """Handle demonstration finish request."""
        try:
            self.logger.info("Finishing demonstration...")

            result = await self._learning_controller.finish_demonstration()

            # Index the task in vector DB
            await self._task_coordinator.index_task(result["task_id"])

            await self.event_bus.publish("demo.finished", {
                "success": True,
                "task_id": result["task_id"],
                "message": f"Task '{result['task_name']}' learned successfully.",
            })

        except Exception as e:
            self.logger.error(f"Error finishing demonstration: {e}", exc_info=True)
            await self.event_bus.publish("demo.failed", {
                "success": False,
                "error": str(e),
            })

    async def _handle_unknown_device(self, data: dict) -> None:
        """Route unknown device to meta-programmer via knowledge.gap.

        The gateway publishes device.unknown when it discovers hardware
        it has no built-in driver for. We convert this into a knowledge
        gap so the meta-programmer can generate a plugin.
        """
        device_id = data.get("device_id", "")
        if not device_id or device_id in self._pending_device_gaps:
            return  # already in flight or empty

        self._pending_device_gaps.add(device_id)
        device_type = data.get("device_type", "unknown")
        name = data.get("name", device_id)
        metadata = data.get("metadata", {})

        # Infer whether this is likely a sensor or actuator.
        # Mirrors sensory-gateway/discovery.py:infer_plugin_type().
        plugin_type = "sensor"
        if any(kw in name.lower() for kw in ("motor", "servo", "actuator", "gripper")):
            plugin_type = "actuator"

        description = (
            f"Create a {plugin_type} plugin for: {name} "
            f"(type={device_type}, id={device_id})"
        )

        try:
            trace_id = await self._task_coordinator.trigger_knowledge_gap(
                query=description,
                context={
                    "device_type": device_type,
                    "device_id": device_id,
                    "name": name,
                    "metadata": metadata,
                    "plugin_type": plugin_type,
                    "source": "gateway_device_discovery",
                },
            )
            self.logger.info(
                f"Unknown device → knowledge.gap: {device_id} (trace={trace_id})"
            )
        except Exception as e:
            self.logger.error(f"Failed to route unknown device {device_id}: {e}")
        finally:
            # Allow re-discovery if device is unplugged and re-plugged.
            self._pending_device_gaps.discard(device_id)

    async def _handle_status(self, data: dict) -> None:
        """Handle status request."""
        try:
            status = {
                "status": "running",
                "sensors": {
                    "available": self._sensor_manager.get_sensor_ids(),
                    "active": [s.sensor_id for s in self._sensor_manager.get_active_sensors()],
                    "learning_mode": self._sensor_manager.get_learning_mode(),
                },
                "learning": {
                    "active": self._learning_controller.is_learning(),
                    "phase": self._learning_controller.get_current_phase().value,
                },
            }

            await self.event_bus.publish("coordinator.status.result", status)

        except Exception as e:
            self.logger.error(f"Error getting status: {e}")


async def main() -> None:
    """Main entry point."""
    service = CoordinatorService()
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
