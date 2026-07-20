"""Tests for TaskCoordinator's embedding + vector-store integration.

Loads task_coordinator.py directly (mirroring test_gate.py) so the test does not
import the whole coordinator package. The Qdrant store and embedding service are
injected as mocks, so no Qdrant/Ollama is required.

These tests use asyncio.run rather than pytest-asyncio: the governance CI lane
runs with a bare pytest, matching the synchronous-only test_gate.py.
"""

import asyncio
import importlib.util
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock

from activelearning import QdrantHit, QdrantPoint

_TC_PATH = os.path.join(
    os.path.dirname(__file__), "..", "src", "coordinator", "task_coordinator.py"
)
_spec = importlib.util.spec_from_file_location("coord_task_coordinator", _TC_PATH)
tc = importlib.util.module_from_spec(_spec)
sys.modules["coord_task_coordinator"] = tc
_spec.loader.exec_module(tc)

TaskCoordinator = tc.TaskCoordinator
LEARNED_TASKS_COLLECTION = tc.LEARNED_TASKS_COLLECTION


def _make(
    *,
    search_result=None,
    embed=(0.1, 0.2, 0.3),
    embed_error=None,
    tasks_root="/tmp",
    sensor_manager=None,
):
    """Build a TaskCoordinator with mocked store + embeddings."""
    store = MagicMock()
    store.search = AsyncMock(return_value=list(search_result or []))
    store.upsert = AsyncMock()
    store.ensure_collection = AsyncMock()
    store.close = AsyncMock()

    embeddings = MagicMock()
    if embed_error is not None:
        embeddings.embed_text = AsyncMock(side_effect=embed_error)
    else:
        embeddings.embed_text = AsyncMock(return_value=list(embed))

    coord = TaskCoordinator(
        nats_client=MagicMock(),
        qdrant_url="http://qdrant",
        tasks_root=tasks_root,
        store=store,
        embedding_service=embeddings,
        sensor_manager=sensor_manager,
    )
    return coord, store, embeddings


# ── find_task confidence routing ────────────────────────────────────────────


def test_find_task_high_confidence_executes():
    coord, store, embeddings = _make(
        search_result=[QdrantHit(id="t1", score=0.95, payload={"task_id": "t1"})]
    )
    result = asyncio.run(coord.find_task("make coffee"))

    assert result == {
        "found": True,
        "task_id": "t1",
        "confidence": 0.95,
        "action": "execute",
    }
    embeddings.embed_text.assert_awaited_once_with("make coffee")
    store.search.assert_awaited_once_with(LEARNED_TASKS_COLLECTION, [0.1, 0.2, 0.3], limit=3)


def test_find_task_medium_confidence_adapts():
    coord, _, _ = _make(search_result=[QdrantHit(id="t2", score=0.7, payload={"task_id": "t2"})])
    result = asyncio.run(coord.find_task("q"))
    assert result["action"] == "adapt"
    assert result["task_id"] == "t2"


def test_find_task_low_confidence_learns():
    coord, _, _ = _make(search_result=[QdrantHit(id="t3", score=0.3, payload={"task_id": "t3"})])
    result = asyncio.run(coord.find_task("q"))
    assert result["action"] == "learn"
    assert result["found"] is False


def test_find_task_no_results_learns():
    coord, _, _ = _make(search_result=[])
    result = asyncio.run(coord.find_task("q"))
    assert result == {
        "found": False,
        "task_id": None,
        "confidence": 0.0,
        "action": "learn",
    }


# ── zero-vector guard (issue #237) ──────────────────────────────────────────


def test_find_task_propagates_embedding_exception_without_search():
    """If embed_text somehow raises (secondary defence), the exception surfaces
    rather than silently querying Qdrant."""
    coord, store, _ = _make(embed_error=RuntimeError("Ollama down"))

    try:
        asyncio.run(coord.find_task("q"))
        raised = False
    except RuntimeError as e:
        raised = "Ollama down" in str(e)

    assert raised, "find_task should propagate the embedding failure"
    store.search.assert_not_called()


def test_find_task_zero_vector_returns_learn_without_search():
    """EmbeddingService returns a zero vector (not raises) when the backend is
    down.  find_task() must detect the sentinel and skip Qdrant rather than
    returning unrelated tasks via false cosine similarity."""
    coord, store, _ = _make(embed=[0.0, 0.0, 0.0])

    result = asyncio.run(coord.find_task("q"))

    assert result == {
        "found": False,
        "task_id": None,
        "confidence": 0.0,
        "action": "learn",
    }
    store.search.assert_not_called()


# ── index_task ──────────────────────────────────────────────────────────────


def _write_task(tasks_root, task_id, metadata):
    task_dir = os.path.join(tasks_root, task_id)
    os.makedirs(task_dir, exist_ok=True)
    with open(os.path.join(task_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)


def test_index_task_upserts_embedded_point(tmp_path):
    coord, store, embeddings = _make(embed=(0.4, 0.5), tasks_root=str(tmp_path))
    _write_task(
        str(tmp_path),
        "task-1",
        {"description": "pour water", "task_name": "pour", "learned_at": 42},
    )

    ok = asyncio.run(coord.index_task("task-1"))

    assert ok is True
    embeddings.embed_text.assert_awaited_once_with("pour water")
    store.upsert.assert_awaited_once()
    collection, points = store.upsert.call_args.args
    assert collection == LEARNED_TASKS_COLLECTION
    assert len(points) == 1
    point = points[0]
    assert isinstance(point, QdrantPoint)
    assert point.id == "task-1"
    assert point.vector == [0.4, 0.5]
    assert point.payload == {
        "task_id": "task-1",
        "description": "pour water",
        "task_name": "pour",
        "learned_at": 42,
    }


def test_index_task_embedding_failure_returns_false_without_upsert(tmp_path):
    coord, store, _ = _make(embed_error=RuntimeError("down"), tasks_root=str(tmp_path))
    _write_task(str(tmp_path), "task-2", {"description": "x"})

    ok = asyncio.run(coord.index_task("task-2"))

    assert ok is False
    store.upsert.assert_not_called()  # no zero-vector point written


def test_index_task_zero_vector_returns_false_without_upsert(tmp_path):
    """EmbeddingService returns a zero vector (not raises) when the backend is
    down.  index_task() must detect the sentinel and skip the upsert rather
    than storing a corrupt point that matches every future query."""
    coord, store, _ = _make(embed=[0.0, 0.0, 0.0], tasks_root=str(tmp_path))
    _write_task(str(tmp_path), "task-3", {"description": "pour water"})

    ok = asyncio.run(coord.index_task("task-3"))

    assert ok is False
    store.upsert.assert_not_called()


def test_index_task_missing_metadata_returns_false(tmp_path):
    coord, store, _ = _make(tasks_root=str(tmp_path))
    ok = asyncio.run(coord.index_task("nope"))
    assert ok is False
    store.upsert.assert_not_called()


# ── setup ───────────────────────────────────────────────────────────────────


def test_setup_ensures_collection():
    coord, store, _ = _make()
    asyncio.run(coord.setup())
    store.ensure_collection.assert_awaited_once_with(LEARNED_TASKS_COLLECTION)


# ── SensorManager wiring for knowledge gaps ─────────────────────────────────


def test_get_available_sensors_queries_sensor_manager():
    sensor_manager = MagicMock()
    sensor_manager.get_sensor_ids.return_value = ["camera_0", "imu_ttyUSB0", "gps_ttyUSB1"]
    coord, _, _ = _make(sensor_manager=sensor_manager)

    sensors = asyncio.run(coord._get_available_sensors())

    assert sensors == ["camera_0", "imu_ttyUSB0", "gps_ttyUSB1"]
    sensor_manager.get_sensor_ids.assert_called_once_with()


def test_get_available_sensors_empty_without_manager():
    coord, _, _ = _make(sensor_manager=None)
    assert asyncio.run(coord._get_available_sensors()) == []


def test_trigger_knowledge_gap_includes_live_sensor_ids():
    sensor_manager = MagicMock()
    sensor_manager.get_sensor_ids.return_value = ["microphone_0", "gps_ttyUSB1"]
    nats = MagicMock()
    nats.publish = AsyncMock()
    coord, _, _ = _make(sensor_manager=sensor_manager)
    coord.nats_client = nats

    trace_id = asyncio.run(coord.trigger_knowledge_gap("how do I navigate outdoors?"))

    assert isinstance(trace_id, str) and trace_id
    nats.publish.assert_awaited_once()
    subject, payload = nats.publish.call_args.args
    assert subject == "knowledge.gap"
    gap = json.loads(payload.decode())
    assert gap["available_sensors"] == ["microphone_0", "gps_ttyUSB1"]
    assert gap["description"] == "how do I navigate outdoors?"
    assert gap["source"] == "task_coordinator"
