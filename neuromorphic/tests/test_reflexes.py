"""Tests for hardwired reflex arc responses."""

import pytest

from neuromorphic.config import NeuromorphicConfig
from neuromorphic.reflexes import ReflexManager, ReflexResponse
from neuromorphic.regions import ReflexArc


@pytest.fixture
def config():
    cfg = NeuromorphicConfig()
    cfg.populations.reflex_arc = 80
    return cfg


@pytest.fixture
def reflex_arc(config):
    return ReflexArc(config)


@pytest.fixture
def manager():
    return ReflexManager()


class TestReflexManager:
    def test_no_reflexes_no_spikes(self, manager, reflex_arc):
        """No spikes → no reflexes triggered."""
        responses = manager.check(reflex_arc)
        assert responses == []

    def test_pain_withdrawal_triggers(self, manager, reflex_arc):
        """High firing in pain_withdrawal sub-range → withdrawal reflex."""
        sr = reflex_arc.get_subrange("pain_withdrawal")
        assert sr is not None
        # Set all pain neurons to spiking
        reflex_arc.population.spikes[sr.slice()] = True
        responses = manager.check(reflex_arc)
        names = [r.reflex_name for r in responses]
        assert "pain_withdrawal" in names

    def test_grip_reflex_triggers(self, manager, reflex_arc):
        sr = reflex_arc.get_subrange("grip")
        assert sr is not None
        reflex_arc.population.spikes[sr.slice()] = True
        responses = manager.check(reflex_arc)
        names = [r.reflex_name for r in responses]
        assert "grip" in names

    def test_startle_reflex_triggers(self, manager, reflex_arc):
        sr = reflex_arc.get_subrange("startle")
        assert sr is not None
        reflex_arc.population.spikes[sr.slice()] = True
        responses = manager.check(reflex_arc)
        names = [r.reflex_name for r in responses]
        assert "startle" in names

    def test_flinch_reflex_triggers(self, manager, reflex_arc):
        sr = reflex_arc.get_subrange("flinch")
        assert sr is not None
        reflex_arc.population.spikes[sr.slice()] = True
        responses = manager.check(reflex_arc)
        names = [r.reflex_name for r in responses]
        assert "flinch" in names

    def test_response_format(self, manager, reflex_arc):
        """Verify response has expected fields."""
        sr = reflex_arc.get_subrange("pain_withdrawal")
        reflex_arc.population.spikes[sr.slice()] = True
        responses = manager.check(reflex_arc)
        assert len(responses) > 0
        r = responses[0]
        assert isinstance(r, ReflexResponse)
        assert isinstance(r.intensity, float)
        assert r.intensity > 0
        assert "type" in r.action
        assert "source" in r.action
        assert r.priority >= 80

    def test_below_threshold_no_trigger(self, manager, reflex_arc):
        """Firing rate below threshold → no reflex."""
        sr = reflex_arc.get_subrange("pain_withdrawal")
        # Only spike 1 out of ~20 neurons → rate < threshold (0.15)
        reflex_arc.population.spikes[sr.start] = True
        responses = manager.check(reflex_arc)
        pain_responses = [r for r in responses if r.reflex_name == "pain_withdrawal"]
        assert len(pain_responses) == 0

    def test_multiple_reflexes_simultaneous(self, manager, reflex_arc):
        """Multiple sub-ranges firing → multiple reflexes."""
        for sr_name in ["pain_withdrawal", "startle", "flinch"]:
            sr = reflex_arc.get_subrange(sr_name)
            reflex_arc.population.spikes[sr.slice()] = True
        responses = manager.check(reflex_arc)
        names = {r.reflex_name for r in responses}
        assert len(names) >= 2  # at least 2 of the 3 should trigger

    def test_pain_has_highest_priority(self, manager, reflex_arc):
        """Pain withdrawal should have priority 100 (highest)."""
        sr = reflex_arc.get_subrange("pain_withdrawal")
        reflex_arc.population.spikes[sr.slice()] = True
        responses = manager.check(reflex_arc)
        pain = [r for r in responses if r.reflex_name == "pain_withdrawal"][0]
        assert pain.priority == 100
