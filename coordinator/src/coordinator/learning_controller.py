"""
Learning Controller - Manages demonstration learning pipeline.

Pipeline phases:
1. Watch - Observe human demonstration
2. Imitate - Generate motor commands from observations
3. Feedback - Collect human feedback
4. Refine - Iterate based on feedback
5. Save - Store learned task
"""

import json
import logging
import os
from enum import Enum
from typing import Any

from activelearning import current_timestamp, generate_trace_id

logger = logging.getLogger(__name__)


class LearningPhase(Enum):
    """Phases of the demonstration learning pipeline."""

    IDLE = "idle"
    WATCH = "watch"
    IMITATE = "imitate"
    FEEDBACK = "feedback"
    REFINE = "refine"
    SAVE = "save"


class LearningController:
    """
    Controls the demonstration learning pipeline.

    Coordinates multi-sensory input during human demonstrations
    and generates executable task code.
    """

    def __init__(
        self,
        nats_client: Any,
        sensor_manager: Any,
        tasks_root: str = "/data/tasks",
    ):
        self.nats_client = nats_client
        self.sensor_manager = sensor_manager
        self.tasks_root = tasks_root

        self._current_phase = LearningPhase.IDLE
        self._current_task: dict | None = None
        self._observation_buffer: list[dict] = []

    async def start_demonstration(self, task_name: str, description: str) -> str:
        """
        Start learning a new task from demonstration.

        Args:
            task_name: Name of the task (e.g., "pick_up_red_ball")
            description: Human-readable description

        Returns:
            trace_id for tracking
        """
        import uuid

        trace_id = str(uuid.uuid4())
        trace_id = generate_trace_id()

        logger.info(f"Starting demonstration learning: {task_name}")

        self._current_task = {
            "trace_id": trace_id,
            "task_name": task_name,
            "description": description,
            "observations": [],
            "parameters": {},
            "trajectory": [],
            "metadata": {},
        }

        self._current_phase = LearningPhase.WATCH
        self._observation_buffer = []

        # Activate relevant sensors
        await self._activate_learning_sensors()

        # Notify system
        await self.nats_client.publish(
            "learning.started",
            json.dumps(
                {
                    "trace_id": trace_id,
                    "task_name": task_name,
                    "phase": self._current_phase.value,
                }
            ).encode(),
        )

        return trace_id

    async def _activate_learning_sensors(self) -> None:
        """Activate sensors for learning."""
        # Activate camera if available
        camera = self.sensor_manager.get_primary_sensor(self.sensor_manager.SensorType.CAMERA)
        if camera:
            self.sensor_manager.activate_sensor(camera.sensor_id)

        # Activate microphone if available
        mic = self.sensor_manager.get_primary_sensor(self.sensor_manager.SensorType.MICROPHONE)
        if mic:
            self.sensor_manager.activate_sensor(mic.sensor_id)

        # Set learning mode
        self.sensor_manager.set_learning_mode("demonstration")

    async def record_observation(
        self,
        sensor_id: str,
        data: Any,
        timestamp: int,
    ) -> None:
        """
        Record an observation during the Watch phase.

        Args:
            sensor_id: ID of sensor that captured this
            data: Observation data
            timestamp: Unix timestamp in milliseconds
        """
        if self._current_phase != LearningPhase.WATCH:
            logger.warning(f"Ignoring observation, not in WATCH phase: {self._current_phase}")
            return

        observation = {
            "sensor_id": sensor_id,
            "data": data,
            "timestamp": timestamp,
        }

        self._observation_buffer.append(observation)

        logger.debug(
            f"Recorded observation from {sensor_id} (buffer size: {len(self._observation_buffer)})"
        )

    async def finish_demonstration(self) -> dict:
        """
        Finish the demonstration and process observations.

        Returns dict with learned task information.
        """
        if self._current_phase != LearningPhase.WATCH:
            raise RuntimeError(f"Cannot finish demonstration in phase: {self._current_phase}")

        logger.info(f"Finishing demonstration with {len(self._observation_buffer)} observations")

        # Move to IMITATE phase
        self._current_phase = LearningPhase.IMITATE

        # Process observations to extract task parameters
        self._current_task["observations"] = self._observation_buffer
        task_data = await self._process_observations()

        # Move to SAVE phase
        self._current_phase = LearningPhase.SAVE

        # Save task
        task_id = await self._save_task(task_data)

        # Reset state
        self._current_phase = LearningPhase.IDLE
        self._observation_buffer = []

        # Deactivate sensors
        for sensor in self.sensor_manager.get_active_sensors():
            self.sensor_manager.deactivate_sensor(sensor.sensor_id)

        self.sensor_manager.set_learning_mode("normal")

        logger.info(f"Demonstration complete, task saved: {task_id}")

        return {
            "task_id": task_id,
            "task_name": self._current_task["task_name"],
            "trace_id": self._current_task["trace_id"],
        }

    async def _process_observations(self) -> dict:
        """
        Process observations to extract task parameters and trajectory.

        This is where the "learning" happens - converting raw sensor data
        into executable task code.
        """
        # Extract parameters from observations
        # For now, use simple heuristics. In a real system, this would use
        # the Meta-Programmer to generate code from observations.

        task_data = {
            "task_name": self._current_task["task_name"],
            "description": self._current_task["description"],
            "parameters": {},
            "trajectory": [],
            "sensor_fusion": self.sensor_manager.get_sensor_fusion_weights(),
        }

        # Analyze observations by sensor type
        camera_obs = [o for o in self._observation_buffer if "camera" in o["sensor_id"]]
        audio_obs = [o for o in self._observation_buffer if "microphone" in o["sensor_id"]]

        # Extract task parameters
        if camera_obs:
            task_data["parameters"]["visual_features"] = len(camera_obs)
            # TODO: Extract object positions, colors, trajectories from camera

        if audio_obs:
            task_data["parameters"]["audio_features"] = len(audio_obs)
            # TODO: Extract voice commands, sound patterns

        # Build trajectory (time-series of observations)
        task_data["trajectory"] = [
            {
                "timestamp": o["timestamp"],
                "sensor": o["sensor_id"],
                "data_summary": str(o["data"])[:100],  # Truncate for storage
            }
            for o in self._observation_buffer[:100]  # Limit to 100 most important
        ]

        return task_data

    async def _save_task(self, task_data: dict) -> str:
        """
        Save learned task to filesystem and vector DB.

        Returns:
            task_id
        """
        task_name = task_data["task_name"]
        task_dir = os.path.join(self.tasks_root, task_name)
        os.makedirs(task_dir, exist_ok=True)

        # Save parameters.json
        params_path = os.path.join(task_dir, "parameters.json")
        with open(params_path, "w") as f:
            json.dump(task_data["parameters"], f, indent=2)

        # Save trajectory.json
        trajectory_path = os.path.join(task_dir, "trajectory.json")
        with open(trajectory_path, "w") as f:
            json.dump(task_data["trajectory"], f, indent=2)

        # Save metadata.json
        metadata = {
            "task_name": task_name,
            "description": task_data["description"],
            "learned_at": int(__import__("time").time() * 1000),
            "learned_at": current_timestamp(),
            "observation_count": len(self._observation_buffer),
            "sensor_fusion": task_data["sensor_fusion"],
            "success_rate": 0.0,  # Will be updated after executions
            "execution_count": 0,
        }
        metadata_path = os.path.join(task_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Generate task.py (placeholder - Meta-Programmer should generate this)
        task_code = self._generate_task_placeholder(task_data)
        task_path = os.path.join(task_dir, "task.py")
        with open(task_path, "w") as f:
            f.write(task_code)

        logger.info(f"Task saved to: {task_dir}")

        # Publish knowledge gap for Meta-Programmer to generate proper implementation
        await self.nats_client.publish(
            "knowledge.gap",
            json.dumps(
                {
                    "trace_id": self._current_task["trace_id"],
                    "description": f"Implement learned task: {task_name}",
                    "context": {
                        "task_data": task_data,
                        "task_path": task_path,
                    },
                }
            ).encode(),
        )

        return task_name

    def _generate_task_placeholder(self, task_data: dict) -> str:
        """Generate placeholder task.py code."""
        return f'''"""
{task_data["description"]}

This task was learned from human demonstration.
Generated task code - to be implemented by Meta-Programmer.
"""

import asyncio
from typing import Any


async def execute_task(parameters: dict[str, Any]) -> dict:
    """
    Execute the learned task.

    Args:
        parameters: Task-specific parameters

    Returns:
        dict with success status and results
    """
    # TODO: Meta-Programmer should implement this based on learned observations
    raise NotImplementedError("Task implementation pending Meta-Programmer generation")


if __name__ == "__main__":
    # Test execution
    result = asyncio.run(execute_task({{}}))
    print(result)
'''

    def get_current_phase(self) -> LearningPhase:
        """Get current learning phase."""
        return self._current_phase

    def is_learning(self) -> bool:
        """Check if currently learning."""
        return self._current_phase != LearningPhase.IDLE
