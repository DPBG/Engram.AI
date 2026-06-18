"""End-to-end persistence round-trip tests."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from neuromorphic.config import NeuromorphicConfig
from neuromorphic.network import NeuromorphicNetwork
from neuromorphic.persistence import (
    NeuromorphicPersistence,
    _FORK_BGSAVE,
    _SYNAPSE_FIELDS,
)


@pytest.fixture
def small_config():
    """Small-scale config for fast persistence tests."""
    cfg = NeuromorphicConfig()
    cfg.populations.brainstem = 100
    cfg.populations.reflex_arc = 80
    cfg.populations.sensory_cortex = 400
    cfg.populations.motor_cortex = 200
    cfg.populations.cerebellum = 200
    cfg.populations.association_cortex = 400
    cfg.populations.predictive_layer = 200
    cfg.populations.working_memory = 50
    cfg.connections.sensory_motor_sparsity = 0.05
    cfg.connections.sensory_motor_weight = 0.8
    cfg.connections.sensory_reflex_sparsity = 0.02
    cfg.connections.reflex_motor_sparsity = 0.02
    cfg.connections.reflex_motor_weight = 0.9
    cfg.connections.brainstem_motor_sparsity = 0.02
    return cfg


@pytest.fixture
def trained_network(small_config):
    network = NeuromorphicNetwork(small_config, seed=42)
    for _ in range(10):
        network.step()
    return network


def _assert_persistence_bit_identical(original: dict, loaded: dict) -> None:
    """Verify weights, eligibility traces, and developmental phase match."""
    assert loaded is not None
    assert original["step_count"] == loaded["step_count"]

    assert original["neuromodulation"]["phase"] == loaded["neuromodulation"]["phase"]

    for syn_name, orig_syn in original["synapses"].items():
        loaded_syn = loaded["synapses"][syn_name]
        assert loaded_syn["shape"] == orig_syn["shape"], syn_name
        for _file_key, state_key, _dtype in _SYNAPSE_FIELDS:
            orig_arr = orig_syn.get(state_key)
            loaded_arr = loaded_syn.get(state_key)
            if orig_arr is None and loaded_arr is None:
                continue
            assert orig_arr is not None and loaded_arr is not None, (
                f"{syn_name}.{state_key}: missing on one side"
            )
            assert np.array_equal(orig_arr, loaded_arr), f"{syn_name}.{state_key}"


async def _save_and_kill(db_path: str, state: dict) -> None:
    """Persist state, then tear down the writer (simulated process kill)."""
    persistence = NeuromorphicPersistence(db_path)
    await persistence.open()
    await persistence.save_state(state)
    await persistence.close()


async def _load_after_kill(db_path: str) -> dict | None:
    """Open a fresh persistence handle and load from disk."""
    persistence = NeuromorphicPersistence(db_path)
    await persistence.open()
    try:
        return await persistence.load_state()
    finally:
        await persistence.close()


class TestPersistenceRoundTrip:
    @pytest.mark.asyncio
    async def test_save_kill_load_bit_identical(self, trained_network, tmp_path):
        """Save → close (kill) → load must restore weights/eligibility/phase."""
        original = trained_network.get_state()
        db_path = str(tmp_path / "neuromorphic.db")

        await _save_and_kill(db_path, original)
        loaded = await _load_after_kill(db_path)

        _assert_persistence_bit_identical(original, loaded)

    @pytest.mark.asyncio
    async def test_save_background_kill_load_bit_identical(self, trained_network, tmp_path):
        """Background save path must produce the same durable checkpoint."""
        original = trained_network.get_state()
        db_path = str(tmp_path / "neuromorphic-bg.db")

        persistence = NeuromorphicPersistence(db_path)
        await persistence.open()
        try:
            assert persistence.save_background(original) is True
            await persistence.wait_bg_save(timeout=120)
        finally:
            await persistence.close()

        loaded = await _load_after_kill(db_path)
        _assert_persistence_bit_identical(original, loaded)

    @pytest.mark.asyncio
    async def test_restored_network_matches_checkpoint(self, trained_network, tmp_path):
        """Loaded state must rehydrate a network that continues from the same step."""
        original = trained_network.get_state()
        db_path = str(tmp_path / "neuromorphic-restore.db")

        await _save_and_kill(db_path, original)
        loaded = await _load_after_kill(db_path)

        restored = NeuromorphicNetwork(trained_network.config, seed=99)
        restored.set_state(loaded)
        assert restored.step_count == trained_network.step_count

        before = restored.get_state()
        restored.step()
        after = restored.get_state()
        assert after["step_count"] == before["step_count"] + 1


class TestCrossPlatformBackgroundSave:
    def test_non_fork_path_available_on_windows(self):
        if sys.platform != "win32":
            pytest.skip("Windows-only guard test")
        assert _FORK_BGSAVE is False

    @pytest.mark.asyncio
    async def test_background_save_uses_thread_when_fork_unavailable(
        self, trained_network, tmp_path, monkeypatch,
    ):
        """Thread fallback must run when fork is unavailable (e.g. Windows)."""
        monkeypatch.setattr(
            "neuromorphic.persistence._FORK_BGSAVE",
            False,
            raising=False,
        )
        state = trained_network.get_state()
        db_path = str(tmp_path / "neuromorphic-thread.db")

        # Snapshot weights before the thread starts; mutate live state after
        # spawn to verify the checkpoint matches the pre-mutation snapshot.
        orig_weights = {
            name: syn["weights_data"].copy()
            for name, syn in state["synapses"].items()
            if "weights_data" in syn
        }

        persistence = NeuromorphicPersistence(db_path)
        await persistence.open()
        try:
            assert persistence.save_background(state) is True
            assert persistence._bg_thread is not None
            for syn in state["synapses"].values():
                if "weights_data" in syn:
                    syn["weights_data"] *= 0.0
            await persistence.wait_bg_save(timeout=120)
        finally:
            await persistence.close()

        loaded = await _load_after_kill(db_path)
        assert loaded is not None
        for name, orig_w in orig_weights.items():
            loaded_w = loaded["synapses"][name]["weights_data"]
            assert np.array_equal(loaded_w, orig_w), (
                f"{name} checkpoint must match pre-mutation snapshot"
            )
        assert loaded["neuromodulation"]["phase"] == state["neuromodulation"]["phase"]

    @pytest.mark.skipif(not _FORK_BGSAVE, reason="fork-based bgsave is POSIX-only")
    @pytest.mark.asyncio
    async def test_background_save_uses_fork_on_posix(self, trained_network, tmp_path):
        state = trained_network.get_state()
        db_path = str(tmp_path / "neuromorphic-fork.db")

        persistence = NeuromorphicPersistence(db_path)
        await persistence.open()
        try:
            assert persistence.save_background(state) is True
            assert persistence._bg_child_pid != 0
            await persistence.wait_bg_save(timeout=120)
        finally:
            await persistence.close()

        loaded = await _load_after_kill(db_path)
        _assert_persistence_bit_identical(state, loaded)
