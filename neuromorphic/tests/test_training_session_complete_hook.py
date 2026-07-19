"""Tests for the auto-benchmark hook triggered after training (issue #324).

Imports ``neuromorphic.service``, which requires the ``activelearning`` SDK —
CI runs this file with ``--with-editable ../sdk`` (like
test_cognitive_bridge_lifecycle.py), not in the general ``tests/`` sweep.
"""

from __future__ import annotations

import asyncio
import json

import pytest

_SMALL_NEURO_ENV = {
    "NEURO_BRAINSTEM_N": "20",
    "NEURO_REFLEX_N": "15",
    "NEURO_SENSORY_N": "60",
    "NEURO_MOTOR_N": "40",
    "NEURO_CEREBELLUM_N": "30",
    "NEURO_ASSOCIATION_N": "60",
    "NEURO_PREDICTIVE_N": "30",
    "NEURO_WORKING_MEM_N": "20",
    "NEURO_FEATURE_N": "0",
    "NEURO_CONCEPT_N": "0",
    "NEURO_DG_N": "0",
    "NEURO_META_N": "0",
    "NEURO_AUTO_BENCHMARK_PATTERNS": "2",
    "NEURO_AUTO_BENCHMARK_REPS": "1",
}


def _set_small_env(monkeypatch) -> None:
    for key, value in _SMALL_NEURO_ENV.items():
        monkeypatch.setenv(key, value)


class TestTrainingSessionCompleteHook:
    @pytest.mark.asyncio
    async def test_skips_when_already_running(self, monkeypatch):
        _set_small_env(monkeypatch)
        from neuromorphic.service import NeuromorphicService

        svc = NeuromorphicService()

        hang_event = asyncio.Event()

        async def _hang():
            await hang_event.wait()

        original_task = asyncio.create_task(_hang())
        svc._benchmark_task = original_task
        try:
            await svc._handle_training_session_complete({"videos_completed": 3})
            # No new task spawned — the original in-flight task is untouched.
            assert svc._benchmark_task is original_task
        finally:
            hang_event.set()
            original_task.cancel()
            try:
                await original_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_spawns_task_when_idle(self, monkeypatch, tmp_path):
        _set_small_env(monkeypatch)
        monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "neuro.db"))
        monkeypatch.setenv("ENGRAM_BENCHMARK_DIR", str(tmp_path / "benchmarks"))

        from neuromorphic.service import NeuromorphicService

        svc = NeuromorphicService()
        assert svc._benchmark_task is None

        await svc._handle_training_session_complete({"videos_completed": 1})
        assert svc._benchmark_task is not None
        await svc._benchmark_task  # let it finish

    @pytest.mark.asyncio
    async def test_run_auto_benchmark_never_touches_live_network(self, monkeypatch, tmp_path):
        """The auto-benchmark must build its own network, not self._network —
        BenchmarkSuite's training reps drive real STDP updates that would
        contaminate the live brain's actual trained weights."""
        _set_small_env(monkeypatch)
        monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "neuro.db"))
        monkeypatch.setenv("ENGRAM_BENCHMARK_DIR", str(tmp_path / "benchmarks"))

        from neuromorphic.service import NeuromorphicService

        svc = NeuromorphicService()
        assert svc._network is None

        await svc._run_auto_benchmark({"videos_completed": 2})

        assert svc._network is None  # never constructed/assigned by the hook

    @pytest.mark.asyncio
    async def test_run_auto_benchmark_saves_results_with_trigger_metadata(
        self, monkeypatch, tmp_path
    ):
        _set_small_env(monkeypatch)
        monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "neuro.db"))
        benchmark_dir = tmp_path / "benchmarks"
        monkeypatch.setenv("ENGRAM_BENCHMARK_DIR", str(benchmark_dir))

        from neuromorphic.service import NeuromorphicService

        svc = NeuromorphicService()
        await svc._run_auto_benchmark({"videos_completed": 4})

        saved = list(benchmark_dir.glob("*.json"))
        assert len(saved) == 1
        data = json.loads(saved[0].read_text())
        assert data["trigger"] == {"videos_completed": 4}
        assert "energy_efficiency" in data

    @pytest.mark.asyncio
    async def test_run_auto_benchmark_survives_missing_checkpoint(self, monkeypatch, tmp_path):
        """No prior checkpoint (fresh install) must not crash the hook —
        the benchmark just runs against a freshly-initialized network."""
        _set_small_env(monkeypatch)
        monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "does-not-exist.db"))
        monkeypatch.setenv("ENGRAM_BENCHMARK_DIR", str(tmp_path / "benchmarks"))

        from neuromorphic.service import NeuromorphicService

        svc = NeuromorphicService()
        await svc._run_auto_benchmark({"videos_completed": 1})  # must not raise


class TestResolveBenchmarkDir:
    def test_honors_env_override(self, tmp_path, monkeypatch):
        from neuromorphic.service import _resolve_benchmark_dir

        monkeypatch.setenv("ENGRAM_BENCHMARK_DIR", str(tmp_path / "custom-benchmarks"))
        assert _resolve_benchmark_dir() == tmp_path / "custom-benchmarks"

    def test_falls_back_to_repo_convention(self, monkeypatch):
        from neuromorphic.service import _resolve_benchmark_dir

        monkeypatch.delenv("ENGRAM_BENCHMARK_DIR", raising=False)
        resolved = _resolve_benchmark_dir()
        assert resolved.name == "benchmarks"
        assert resolved.parent.name == "neuromorphic"
