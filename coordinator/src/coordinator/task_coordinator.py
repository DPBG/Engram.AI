"""
Task Coordinator - Manages task lookup and execution.

Provides:
- Vector DB lookup for learned tasks
- Task loading and execution
- Knowledge gap detection when no suitable task found
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from activelearning import (
    EmbeddingService,
    QdrantPoint,
    QdrantStore,
    get_embedding_service,
)

logger = logging.getLogger(__name__)

# Vector collection holding embeddings of learned-task descriptions.
LEARNED_TASKS_COLLECTION = "learned_tasks"


class TaskCoordinator:
    """
    Coordinates task execution using learned knowledge.

    Flow:
    1. Receive task request
    2. Search vector DB for matching task (semantic search)
    3. If high confidence match: load and execute
    4. If medium confidence: adapt existing task
    5. If low confidence: trigger knowledge gap to Meta-Programmer
    """

    def __init__(
        self,
        nats_client: Any,
        qdrant_url: str,
        tasks_root: str = "/data/tasks",
        *,
        store: Optional[QdrantStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        self.nats_client = nats_client
        self.tasks_root = tasks_root

        # Shared SDK infrastructure (injectable for testing): embeddings via the
        # EmbeddingService (which raises instead of returning a zero vector that
        # would silently search against the origin), Qdrant via the shared
        # QdrantStore. The embedding service is owned and closed by the service.
        self._qdrant = store if store is not None else QdrantStore(qdrant_url)
        self._embeddings = (
            embedding_service if embedding_service is not None else get_embedding_service()
        )

        # Confidence thresholds
        self.high_confidence = float(os.environ.get("TASK_HIGH_CONFIDENCE", "0.85"))
        self.medium_confidence = float(os.environ.get("TASK_MEDIUM_CONFIDENCE", "0.6"))

    async def setup(self) -> None:
        """Ensure the learned-tasks collection exists before serving requests."""
        await self._qdrant.ensure_collection(LEARNED_TASKS_COLLECTION)

    async def close(self) -> None:
        """Release the Qdrant connection."""
        await self._qdrant.close()

    async def find_task(self, query: str) -> Dict:
        """
        Find a learned task matching the query.

        Args:
            query: Natural language task description

        Returns:
            dict with:
                - found: bool
                - task_id: str (if found)
                - confidence: float
                - action: "execute", "adapt", or "learn"
        """
        # Get query embedding (raises if the embedding service is unavailable —
        # surfacing the failure instead of silently searching against a zero
        # vector, which would return unrelated tasks).
        embedding = await self._embeddings.embed_text(query)

        # Search Qdrant for similar tasks
        results = await self._qdrant.search(
            LEARNED_TASKS_COLLECTION,
            embedding,
            limit=3,
        )

        if not results:
            logger.info(f"No tasks found for: {query}")
            return {
                "found": False,
                "task_id": None,
                "confidence": 0.0,
                "action": "learn",
            }

        # Get best match
        best_match = results[0]
        confidence = best_match.score
        task_id = best_match.payload["task_id"]

        logger.info(f"Task match: {task_id} (confidence: {confidence:.2f})")

        if confidence >= self.high_confidence:
            # High confidence - execute as-is
            return {
                "found": True,
                "task_id": task_id,
                "confidence": confidence,
                "action": "execute",
            }
        elif confidence >= self.medium_confidence:
            # Medium confidence - adapt
            return {
                "found": True,
                "task_id": task_id,
                "confidence": confidence,
                "action": "adapt",
            }
        else:
            # Low confidence - learn new task
            return {
                "found": False,
                "task_id": None,
                "confidence": confidence,
                "action": "learn",
            }

    async def execute_task(self, task_id: str, parameters: Optional[Dict] = None) -> Dict:
        """
        Execute a learned task.

        Args:
            task_id: Task identifier
            parameters: Optional parameters to override

        Returns:
            dict with execution results
        """
        try:
            # Load task
            task_dir = os.path.join(self.tasks_root, task_id)
            if not os.path.exists(task_dir):
                return {
                    "success": False,
                    "error": f"Task not found: {task_id}",
                }

            # Load metadata
            metadata_path = os.path.join(task_dir, "metadata.json")
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Load parameters
            params_path = os.path.join(task_dir, "parameters.json")
            with open(params_path, "r") as f:
                task_params = json.load(f)

            # Override with provided parameters
            if parameters:
                task_params.update(parameters)

            # Load and execute task code
            task_path = os.path.join(task_dir, "task.py")
            if not os.path.exists(task_path):
                return {
                    "success": False,
                    "error": f"Task implementation not found: {task_path}",
                }

            # TODO: Execute task.py safely
            # For now, return success with metadata
            logger.info(f"Executing task: {task_id}")

            # Update execution count
            metadata["execution_count"] = metadata.get("execution_count", 0) + 1
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            return {
                "success": True,
                "task_id": task_id,
                "metadata": metadata,
                "parameters": task_params,
            }

        except Exception as e:
            logger.error(f"Error executing task: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    async def trigger_knowledge_gap(self, query: str, context: Optional[Dict] = None) -> str:
        """
        Trigger knowledge gap detection.

        Publishes a knowledge gap for Meta-Programmer to handle.

        Returns:
            trace_id
        """
        import uuid
        trace_id = str(uuid.uuid4())

        gap = {
            "trace_id": trace_id,
            "description": query,
            "context": context or {},
            "available_sensors": await self._get_available_sensors(),
            "allows_external_query": True,
            "source": "task_coordinator",
        }

        await self.nats_client.publish(
            "knowledge.gap",
            json.dumps(gap).encode(),
        )

        logger.info(f"Knowledge gap triggered: {trace_id}")
        return trace_id

    async def _get_available_sensors(self) -> List[str]:
        """Get list of available sensor IDs."""
        # TODO: Query SensorManager
        return ["camera_0", "microphone_0"]

    async def index_task(self, task_id: str) -> bool:
        """
        Index a task in the vector DB for semantic search.

        Args:
            task_id: Task to index

        Returns:
            bool indicating success
        """
        try:
            # Load task metadata
            metadata_path = os.path.join(self.tasks_root, task_id, "metadata.json")
            if not os.path.exists(metadata_path):
                logger.error(f"Task metadata not found: {metadata_path}")
                return False

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Get embedding for description
            description = metadata.get("description", "")
            embedding = await self._embeddings.embed_text(description)

            # Store in Qdrant
            await self._qdrant.upsert(
                LEARNED_TASKS_COLLECTION,
                [
                    QdrantPoint(
                        id=task_id,
                        vector=embedding,
                        payload={
                            "task_id": task_id,
                            "description": description,
                            "task_name": metadata.get("task_name", ""),
                            "learned_at": metadata.get("learned_at", 0),
                        },
                    )
                ],
            )
            logger.info(f"Task indexed: {task_id}")
            return True

        except Exception as e:
            logger.error(f"Error indexing task: {e}", exc_info=True)
            return False
