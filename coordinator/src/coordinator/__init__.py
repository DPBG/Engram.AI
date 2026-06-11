"""
Coordinator - Multi-sensory learning and task coordination system.

Manages:
- Sensor detection and priority-based fusion
- Demonstration learning pipeline
- Task storage and retrieval
- Knowledge gap detection
"""

from coordinator.service import CoordinatorService
from coordinator.sensor_manager import SensorManager
from coordinator.learning_controller import LearningController
from coordinator.task_coordinator import TaskCoordinator

__version__ = "0.1.0"

__all__ = [
    "CoordinatorService",
    "SensorManager",
    "LearningController",
    "TaskCoordinator",
]
