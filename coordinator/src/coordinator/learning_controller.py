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


def _as_dict(data: Any) -> dict | None:
    """Return data when it is a dict; otherwise None."""
    return data if isinstance(data, dict) else None


def _extract_position(payload: dict) -> dict | None:
    """Pull a 2D/3D position from common observation key shapes."""
    if "position" in payload and isinstance(payload["position"], dict):
        pos = payload["position"]
        if "x" in pos and "y" in pos:
            out = {"x": float(pos["x"]), "y": float(pos["y"])}
            if "z" in pos:
                out["z"] = float(pos["z"])
            return out
    if "x" in payload and "y" in payload:
        try:
            out = {"x": float(payload["x"]), "y": float(payload["y"])}
            if "z" in payload:
                out["z"] = float(payload["z"])
            return out
        except (TypeError, ValueError):
            return None
    bbox = payload.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
            return {"x": (x1 + x2) / 2.0, "y": (y1 + y2) / 2.0}
        except (TypeError, ValueError):
            return None
    return None


def _extract_colors(payload: dict) -> list[str]:
    """Collect color labels from a camera observation payload."""
    colors: list[str] = []
    if isinstance(payload.get("color"), str) and payload["color"].strip():
        colors.append(payload["color"].strip().lower())
    raw_colors = payload.get("colors")
    if isinstance(raw_colors, list):
        for item in raw_colors:
            if isinstance(item, str) and item.strip():
                colors.append(item.strip().lower())
    objects = payload.get("objects")
    if isinstance(objects, list):
        for obj in objects:
            if isinstance(obj, dict) and isinstance(obj.get("color"), str) and obj["color"].strip():
                colors.append(obj["color"].strip().lower())
    # Preserve order, drop duplicates
    seen: set[str] = set()
    unique: list[str] = []
    for color in colors:
        if color not in seen:
            seen.add(color)
            unique.append(color)
    return unique


def _extract_object_labels(payload: dict) -> list[str]:
    """Collect object / label names from a camera observation payload."""
    labels: list[str] = []
    for key in ("label", "object", "name"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            labels.append(value.strip().lower())
    objects = payload.get("objects")
    if isinstance(objects, list):
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            for key in ("label", "name", "class"):
                value = obj.get(key)
                if isinstance(value, str) and value.strip():
                    labels.append(value.strip().lower())
                    break
    seen: set[str] = set()
    unique: list[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            unique.append(label)
    return unique


def extract_visual_features(camera_obs: list[dict]) -> dict:
    """
    Heuristically extract positions, colors, labels, and a position trajectory
    from camera observation payloads. Non-dict payloads are skipped.
    """
    positions: list[dict] = []
    colors: list[str] = []
    labels: list[str] = []
    trajectory: list[dict] = []

    for obs in camera_obs:
        payload = _as_dict(obs.get("data"))
        if payload is None:
            continue
        position = _extract_position(payload)
        if position is not None:
            positions.append(position)
            trajectory.append(
                {
                    "timestamp": obs.get("timestamp"),
                    "x": position["x"],
                    "y": position["y"],
                    **({"z": position["z"]} if "z" in position else {}),
                }
            )
        colors.extend(_extract_colors(payload))
        labels.extend(_extract_object_labels(payload))

    # Deduplicate colors/labels while preserving order
    def _unique(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            if item not in seen:
                seen.add(item)
                out.append(item)
        return out

    return {
        "observation_count": len(camera_obs),
        "positions": positions,
        "colors": _unique(colors),
        "objects": _unique(labels),
        "trajectory": trajectory[:100],
    }


def extract_audio_features(audio_obs: list[dict]) -> dict:
    """
    Heuristically extract voice commands / transcripts and energy patterns
    from microphone observation payloads. Non-dict payloads are skipped.
    """
    commands: list[str] = []
    energy_samples: list[float] = []

    for obs in audio_obs:
        payload = _as_dict(obs.get("data"))
        if payload is None:
            continue
        for key in ("transcript", "command", "text", "phrase"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                commands.append(value.strip())
                break
        for key in ("energy", "rms", "amplitude"):
            value = payload.get(key)
            if isinstance(value, (int, float)):
                energy_samples.append(float(value))
                break

    unique_commands: list[str] = []
    seen: set[str] = set()
    for cmd in commands:
        key = cmd.lower()
        if key not in seen:
            seen.add(key)
            unique_commands.append(cmd)

    energy_summary: dict[str, float] | None = None
    if energy_samples:
        energy_summary = {
            "min": min(energy_samples),
            "max": max(energy_samples),
            "mean": round(sum(energy_samples) / len(energy_samples), 6),
            "sample_count": float(len(energy_samples)),
        }

    return {
        "observation_count": len(audio_obs),
        "commands": unique_commands,
        "energy": energy_summary,
    }


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
            visual = extract_visual_features(camera_obs)
            task_data["parameters"]["visual_features"] = visual["observation_count"]
            task_data["parameters"]["positions"] = visual["positions"]
            task_data["parameters"]["colors"] = visual["colors"]
            task_data["parameters"]["objects"] = visual["objects"]
            task_data["parameters"]["visual_trajectory"] = visual["trajectory"]

        if audio_obs:
            audio = extract_audio_features(audio_obs)
            task_data["parameters"]["audio_features"] = audio["observation_count"]
            task_data["parameters"]["voice_commands"] = audio["commands"]
            if audio["energy"] is not None:
                task_data["parameters"]["audio_energy"] = audio["energy"]

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
