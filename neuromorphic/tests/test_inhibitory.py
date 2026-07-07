"""Tests for inhibitory neuron support (Phase 2)."""

import numpy as np
import pytest

from neuromorphic.config import InhibitoryConfig, LIFParams, NeuromorphicConfig
from neuromorphic.neurons import NeuronPopulation
from neuromorphic.regions import AssociationCortex, SensoryCortex


class TestInhibitoryNeurons:
    """Test E/I split in NeuronPopulation."""

    def test_default_no_inhibitory(self):
        """Without InhibitoryConfig, all neurons are excitatory."""
        pop = NeuronPopulation(100, LIFParams())
        assert not pop.is_inhibitory.any()
        assert (pop.output_sign == 1.0).all()

    def test_inhibitory_fraction(self):
        """20% of neurons should be inhibitory."""
        cfg = InhibitoryConfig(inhibitory_fraction=0.2)
        pop = NeuronPopulation(100, LIFParams(), inhibitory_config=cfg)
        assert pop.is_inhibitory.sum() == 20
        # Inhibitory neurons are at the end
        assert not pop.is_inhibitory[:80].any()
        assert pop.is_inhibitory[80:].all()

    def test_inhibitory_output_sign(self):
        """Inhibitory neurons produce negative output."""
        cfg = InhibitoryConfig(inhibitory_fraction=0.2)
        pop = NeuronPopulation(100, LIFParams(), inhibitory_config=cfg)
        assert (pop.output_sign[:80] == 1.0).all()
        assert (pop.output_sign[80:] == -1.0).all()

    def test_inhibitory_faster_tau(self):
        """Inhibitory neurons should have faster time constant."""
        cfg = InhibitoryConfig(inhibitory_fraction=0.2, inhibitory_tau=10.0)
        pop = NeuronPopulation(100, LIFParams(tau=20.0), inhibitory_config=cfg)
        assert (pop._tau[:80] == 20.0).all()
        assert (pop._tau[80:] == 10.0).all()

    def test_inhibitory_lower_threshold(self):
        """Inhibitory neurons have slightly lower threshold."""
        cfg = InhibitoryConfig(inhibitory_fraction=0.2, inhibitory_threshold=-58.0)
        pop = NeuronPopulation(100, LIFParams(threshold=-55.0), inhibitory_config=cfg)
        assert (pop._threshold[:80] == -55.0).all()
        assert (pop._threshold[80:] == -58.0).all()

    def test_inhibitory_shorter_refractory(self):
        """Inhibitory neurons have shorter refractory period."""
        cfg = InhibitoryConfig(inhibitory_fraction=0.2, inhibitory_refractory_ms=1.0)
        pop = NeuronPopulation(100, LIFParams(refractory_ms=2.0), inhibitory_config=cfg)
        assert (pop._refractory_ms[:80] == 2.0).all()
        assert (pop._refractory_ms[80:] == 1.0).all()

    def test_signed_spikes(self):
        """signed_spikes should combine spike pattern with output sign."""
        cfg = InhibitoryConfig(inhibitory_fraction=0.2)
        pop = NeuronPopulation(100, LIFParams(), inhibitory_config=cfg)
        # Force some spikes
        pop.spikes[:] = False
        pop.spikes[0] = True  # excitatory
        pop.spikes[90] = True  # inhibitory
        signed = pop.signed_spikes
        assert signed[0] == 1.0
        assert signed[90] == -1.0
        assert signed[50] == 0.0

    def test_step_with_inhibitory(self):
        """Network step should work with inhibitory neurons."""
        cfg = InhibitoryConfig(inhibitory_fraction=0.2)
        pop = NeuronPopulation(
            100, LIFParams(noise_std=0.0), inhibitory_config=cfg, rng=np.random.default_rng(42)
        )
        current = np.full(100, 50.0, dtype=np.float32)
        spikes = pop.step(current)
        assert spikes.dtype == bool
        assert spikes.shape == (100,)


class TestInhibitoryRegions:
    """Test that cortical regions get inhibitory neurons."""

    @pytest.fixture
    def config(self):
        from neuromorphic.config import PopulationConfig

        return NeuromorphicConfig(
            populations=PopulationConfig(
                sensory_cortex=100,
                association_cortex=100,
                brainstem=20,
                reflex_arc=20,
                motor_cortex=50,
                cerebellum=50,
                predictive_layer=50,
                working_memory=20,
            ),
        )

    def test_sensory_cortex_has_inhibitory(self, config):
        """SensoryCortex should have inhibitory neurons."""
        region = SensoryCortex(config)
        assert region.population.is_inhibitory.sum() == 20  # 20% of 100

    def test_association_cortex_has_inhibitory(self, config):
        """AssociationCortex should have inhibitory neurons."""
        region = AssociationCortex(config)
        assert region.population.is_inhibitory.sum() == 20
