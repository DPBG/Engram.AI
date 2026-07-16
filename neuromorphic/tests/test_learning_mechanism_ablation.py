"""Issue #328: Ablation benchmarks for the 6 always-on learning mechanisms.

Protocol: each test directly stimulates the target mechanism via unit-level calls
on real SynapseGroup instances, then compares full vs. ablated behaviour.
Ablations are reversible and never modify production code.

The 6 mechanisms (CLAUDE.md Invariant 1):
  1. STDP (spike-timing-dependent plasticity)
  2. Eligibility traces (three-factor credit assignment)
  3. BCM metaplasticity (activity-dependent modification threshold)
  4. Neuromodulation (DA/ACh/NE/5-HT gates eligibility commits)
  5. Homeostatic scaling (synaptic normalisation)
  6. R-STDP (reward-modulated STDP)
"""

from __future__ import annotations

import numpy as np
import pytest

from neuromorphic.config import BCMConfig, EligibilityTraceConfig, RSTDPParams, STDPParams
from neuromorphic.synapses import SynapseGroup

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_STDP = STDPParams(
    a_plus=0.012,
    a_minus=0.010,
    tau_plus=20.0,
    tau_minus=20.0,
    w_min=0.01,
    w_max=1.0,
    min_dt=1.0,
    max_dt=50.0,
)

_ELIG = EligibilityTraceConfig(
    tau_eligibility=1000.0,
    trace_decay=0.999,
    significance_ratio=0.1,
    untracked_decay_interval=100,
)

_N = 20  # pre and post population size — small enough for fast tests


def _syn(*, with_eligibility: bool = False, with_rstdp: bool = False) -> SynapseGroup:
    """Minimal dense SynapseGroup for controlled stimulation."""
    return SynapseGroup(
        n_pre=_N,
        n_post=_N,
        sparsity=1.0,
        init_weight=0.5,
        plastic=True,
        stdp_params=_STDP,
        rstdp_params=RSTDPParams() if with_rstdp else None,
        rng=np.random.default_rng(42),
        name="ablation_test",
        eligibility_config=_ELIG if with_eligibility else None,
    )


def _spike_pattern(
    pre_t: float = 0.0,
    post_t: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (pre_spikes, post_spikes, pre_times, post_times) with all neurons active.

    Default: pre fires at 0 ms, post at 10 ms → dt = +10 ms → LTP.
    dt = 10 ms is well inside the [1, 50] ms STDP window.
    """
    pre_sp = np.ones(_N, dtype=bool)
    post_sp = np.ones(_N, dtype=bool)
    pre_times = np.full(_N, pre_t, dtype=np.float32)
    post_times = np.full(_N, post_t, dtype=np.float32)
    return pre_sp, post_sp, pre_times, post_times


# ---------------------------------------------------------------------------
# 1. STDP
# ---------------------------------------------------------------------------


class TestSTDPContribution:
    """STDP is the primary Hebbian plasticity signal (two-factor and trace modes)."""

    def test_ltp_strengthens_weights_two_factor(self) -> None:
        """Pre-before-post (dt>0) must increase weights in two-factor mode."""
        syn = _syn()
        before = syn.weights.data.copy()
        pre_sp, post_sp, pre_t, post_t = _spike_pattern(pre_t=0.0, post_t=10.0)
        syn.update_weights_stdp(pre_sp, post_sp, pre_t, post_t, current_time=10.0)
        assert np.mean(syn.weights.data) > np.mean(before)

    def test_ltd_weakens_weights_two_factor(self) -> None:
        """Post-before-pre (dt<0) must decrease weights in two-factor mode."""
        syn = _syn()
        before = syn.weights.data.copy()
        pre_sp, post_sp, pre_t, post_t = _spike_pattern(pre_t=10.0, post_t=0.0)
        syn.update_weights_stdp(pre_sp, post_sp, pre_t, post_t, current_time=10.0)
        assert np.mean(syn.weights.data) < np.mean(before)

    def test_ablated_stdp_no_weight_change(self) -> None:
        """a_plus=a_minus=0 (ablated STDP) must leave weights unchanged."""
        ablated = STDPParams(a_plus=0.0, a_minus=0.0, tau_plus=20.0, tau_minus=20.0,
                            w_min=0.01, w_max=1.0, min_dt=1.0, max_dt=50.0)
        syn = SynapseGroup(
            n_pre=_N, n_post=_N, sparsity=1.0, init_weight=0.5, plastic=True,
            stdp_params=ablated, rng=np.random.default_rng(42), name="ablated_stdp",
        )
        before = syn.weights.data.copy()
        pre_sp, post_sp, pre_t, post_t = _spike_pattern()
        syn.update_weights_stdp(pre_sp, post_sp, pre_t, post_t, current_time=10.0)
        np.testing.assert_array_equal(syn.weights.data, before)

    def test_outside_window_no_weight_change(self) -> None:
        """Spike pairs with |dt| > max_dt must not trigger any weight change."""
        syn = _syn()
        before = syn.weights.data.copy()
        # dt = 100 ms >> max_dt = 50 ms
        pre_sp, post_sp, pre_t, post_t = _spike_pattern(pre_t=0.0, post_t=100.0)
        syn.update_weights_stdp(pre_sp, post_sp, pre_t, post_t, current_time=100.0)
        np.testing.assert_array_equal(syn.weights.data, before)

    def test_last_stdp_delta_nonzero_when_active(self) -> None:
        """last_stdp_delta must be positive after a valid STDP update."""
        syn = _syn()
        pre_sp, post_sp, pre_t, post_t = _spike_pattern()
        syn.update_weights_stdp(pre_sp, post_sp, pre_t, post_t, current_time=10.0)
        assert syn.last_stdp_delta > 0.0


# ---------------------------------------------------------------------------
# 2. Eligibility traces
# ---------------------------------------------------------------------------


class TestEligibilityTraces:
    """Eligibility traces carry STDP credit forward to delayed neuromodulation."""

    def test_stdp_writes_trace_not_weights(self) -> None:
        """With eligibility enabled STDP must update traces, leaving weights constant."""
        syn = _syn(with_eligibility=True)
        before_w = syn.weights.data.copy()
        assert syn.eligibility is not None
        pre_sp, post_sp, pre_t, post_t = _spike_pattern()
        syn.update_weights_stdp(pre_sp, post_sp, pre_t, post_t, current_time=10.0)
        np.testing.assert_array_equal(syn.weights.data, before_w)
        assert np.any(syn.eligibility != 0.0), "STDP must leave nonzero eligibility traces"

    def test_neuromodulation_commits_trace_to_weights(self) -> None:
        """Positive modulator signal converts accumulated eligibility to weight gain."""
        syn = _syn(with_eligibility=True)
        assert syn.eligibility is not None
        syn.eligibility[:] = 0.1
        syn._elig_active = None  # force full-array path
        before = syn.weights.data.copy()
        syn.apply_neuromodulation_and_decay(modulator_signal=0.5, interval=1)
        assert np.mean(syn.weights.data) > np.mean(before)

    def test_ablated_trace_blocks_delayed_update(self) -> None:
        """Zeroing eligibility before neuromodulation fires must block weight change."""
        syn = _syn(with_eligibility=True)
        assert syn.eligibility is not None
        pre_sp, post_sp, pre_t, post_t = _spike_pattern()
        syn.update_weights_stdp(pre_sp, post_sp, pre_t, post_t, current_time=10.0)
        # Ablation: erase the trace before the neuromodulatory reward arrives
        syn.eligibility[:] = 0.0
        syn._elig_active = np.empty(0, dtype=np.int32)
        before_w = syn.weights.data.copy()
        syn.apply_neuromodulation_and_decay(modulator_signal=0.5, interval=1)
        np.testing.assert_array_almost_equal(syn.weights.data, before_w, decimal=5)

    def test_trace_decays_exponentially(self) -> None:
        """Eligibility trace must follow d^N decay after N steps with no modulation."""
        syn = _syn(with_eligibility=True)
        assert syn.eligibility is not None
        syn.eligibility[:] = 1.0
        syn._elig_active = None  # full-array path
        n_steps = 50
        for _ in range(n_steps):
            syn.apply_neuromodulation_and_decay(modulator_signal=0.0, interval=1)
        expected = _ELIG.trace_decay ** n_steps  # ≈ 0.951
        actual = float(np.max(np.abs(syn.eligibility)))
        assert abs(actual - expected) < 0.01, (
            f"Trace should decay to ≈{expected:.4f}; got {actual:.4f}"
        )


# ---------------------------------------------------------------------------
# 3. BCM metaplasticity
# ---------------------------------------------------------------------------


class TestBCMMetaplasticity:
    """BCM theta modulates STDP amplitude per postsynaptic neuron."""

    def test_high_theta_reduces_ltp(self) -> None:
        """Elevated BCM theta (active neuron history) must shrink LTP magnitude."""
        syn_base = _syn()
        syn_bcm = _syn()
        syn_bcm.enable_bcm(BCMConfig(theta_init=0.01, theta_tau=100.0))
        # Drive theta high with maximum firing rates (rate^2 → 1.0)
        high_rate = np.ones(_N, dtype=np.float32)
        for _ in range(500):
            syn_bcm.update_bcm_threshold(high_rate)
        pre_sp, post_sp, pre_t, post_t = _spike_pattern()
        before_base = syn_base.weights.data.copy()
        before_bcm = syn_bcm.weights.data.copy()
        syn_base.update_weights_stdp(pre_sp, post_sp, pre_t, post_t, current_time=10.0)
        syn_bcm.update_weights_stdp(pre_sp, post_sp, pre_t, post_t, current_time=10.0)
        delta_base = float(np.mean(syn_base.weights.data) - np.mean(before_base))
        delta_bcm = float(np.mean(syn_bcm.weights.data) - np.mean(before_bcm))
        assert delta_bcm < delta_base, (
            f"High BCM theta must reduce LTP: bcm={delta_bcm:.6f} base={delta_base:.6f}"
        )

    def test_low_theta_enhances_ltp(self) -> None:
        """BCM theta below theta_init (quiet neuron history) must enlarge LTP."""
        syn_base = _syn()
        syn_bcm = _syn()
        syn_bcm.enable_bcm(BCMConfig(theta_init=1.0, theta_tau=100.0))
        # Push theta downward: zero firing rates over many steps
        low_rate = np.zeros(_N, dtype=np.float32)
        for _ in range(200):
            syn_bcm.update_bcm_threshold(low_rate)
        pre_sp, post_sp, pre_t, post_t = _spike_pattern()
        before_base = syn_base.weights.data.copy()
        before_bcm = syn_bcm.weights.data.copy()
        syn_base.update_weights_stdp(pre_sp, post_sp, pre_t, post_t, current_time=10.0)
        syn_bcm.update_weights_stdp(pre_sp, post_sp, pre_t, post_t, current_time=10.0)
        delta_base = float(np.mean(syn_base.weights.data) - np.mean(before_base))
        delta_bcm = float(np.mean(syn_bcm.weights.data) - np.mean(before_bcm))
        assert delta_bcm > delta_base, (
            f"Low BCM theta must enhance LTP: bcm={delta_bcm:.6f} base={delta_base:.6f}"
        )

    def test_ablated_bcm_returns_none_scaling(self) -> None:
        """_get_bcm_scaling must return None (no modulation) when BCM is disabled."""
        syn = _syn()
        assert syn._get_bcm_scaling() is None


# ---------------------------------------------------------------------------
# 4. Neuromodulation
# ---------------------------------------------------------------------------


class TestNeuromodulation:
    """The 4-channel neuromodulatory system gates eligibility-to-weight commits."""

    def test_positive_modulator_increases_weights(self) -> None:
        """Positive modulator with positive eligibility must increase weights."""
        syn = _syn(with_eligibility=True)
        assert syn.eligibility is not None
        syn.eligibility[:] = 0.1
        syn._elig_active = None
        before = syn.weights.data.copy()
        syn.apply_neuromodulation_and_decay(modulator_signal=1.0, interval=1)
        assert np.mean(syn.weights.data) > np.mean(before)

    def test_zero_modulator_blocks_weight_update(self) -> None:
        """Modulator=0 must not change weights even with large eligibility traces."""
        syn = _syn(with_eligibility=True)
        assert syn.eligibility is not None
        syn.eligibility[:] = 0.5
        syn._elig_active = None
        before = syn.weights.data.copy()
        syn.apply_neuromodulation_and_decay(modulator_signal=0.0, interval=1)
        np.testing.assert_array_almost_equal(syn.weights.data, before, decimal=5)

    def test_negative_modulator_decreases_weights(self) -> None:
        """Negative modulator with positive eligibility must decrease weights."""
        syn = _syn(with_eligibility=True)
        assert syn.eligibility is not None
        syn.eligibility[:] = 0.5
        syn._elig_active = None
        before = syn.weights.data.copy()
        syn.apply_neuromodulation_and_decay(modulator_signal=-0.5, interval=1)
        assert np.mean(syn.weights.data) < np.mean(before)

    def test_larger_modulator_produces_larger_weight_change(self) -> None:
        """Modulator magnitude must scale proportionally with weight change."""

        def weight_delta(mod: float) -> float:
            s = _syn(with_eligibility=True)
            assert s.eligibility is not None
            s.eligibility[:] = 0.1
            s._elig_active = None
            before = s.weights.data.copy()
            s.apply_neuromodulation_and_decay(modulator_signal=mod, interval=1)
            return float(np.mean(np.abs(s.weights.data - before)))

        assert weight_delta(2.0) > weight_delta(1.0) > 0.0


# ---------------------------------------------------------------------------
# 5. Homeostatic scaling
# ---------------------------------------------------------------------------


class TestHomeostaticScaling:
    """Homeostatic normalisation prevents saturation and protects learned structure."""

    def test_high_weights_pulled_toward_target(self) -> None:
        """Mean weight near w_max must decrease toward target after normalisation."""
        syn = _syn()
        syn.weights.data[:] = np.float32(0.95)  # near w_max=1.0
        before_mean = float(np.mean(syn.weights.data))
        for _ in range(50):
            syn.normalize_weights(target_frac=0.5, base_rate=0.01, plasticity_multiplier=1.0)
        after_mean = float(np.mean(syn.weights.data))
        assert after_mean < before_mean, (
            f"High weights must decrease: before={before_mean:.4f} after={after_mean:.4f}"
        )

    def test_low_weights_pulled_toward_target(self) -> None:
        """Mean weight near w_min must increase toward target after normalisation."""
        syn = _syn()
        syn.weights.data[:] = np.float32(0.05)  # near w_min=0.01
        before_mean = float(np.mean(syn.weights.data))
        for _ in range(50):
            syn.normalize_weights(target_frac=0.5, base_rate=0.01, plasticity_multiplier=1.0)
        after_mean = float(np.mean(syn.weights.data))
        assert after_mean > before_mean, (
            f"Low weights must increase: before={before_mean:.4f} after={after_mean:.4f}"
        )

    def test_ablated_homeostasis_non_plastic_is_noop(self) -> None:
        """plastic=False must make normalize_weights a complete no-op."""
        syn = SynapseGroup(
            n_pre=_N, n_post=_N, sparsity=1.0, init_weight=0.95, plastic=False,
            rng=np.random.default_rng(42), name="nonplastic",
        )
        before = syn.weights.data.copy()
        for _ in range(50):
            syn.normalize_weights(target_frac=0.5, base_rate=0.01)
        np.testing.assert_array_equal(syn.weights.data, before)


# ---------------------------------------------------------------------------
# 6. R-STDP
# ---------------------------------------------------------------------------


class TestRSTDP:
    """Reward-modulated STDP adds a surprise signal to eligibility (or weights)."""

    def test_higher_modulation_yields_larger_eligibility(self) -> None:
        """R-STDP with modulation=2 must produce more eligibility than modulation=1."""

        def total_elig(mod: float) -> float:
            s = _syn(with_eligibility=True, with_rstdp=True)
            pre_sp, post_sp, pre_t, post_t = _spike_pattern()
            s.update_weights_rstdp(pre_sp, post_sp, pre_t, post_t,
                                   current_time=10.0, modulation=mod)
            assert s.eligibility is not None
            return float(np.sum(np.abs(s.eligibility)))

        assert total_elig(2.0) > total_elig(1.0) > 0.0

    def test_zero_modulation_yields_zero_eligibility(self) -> None:
        """modulation=0 must produce no eligibility changes."""
        syn = _syn(with_eligibility=True, with_rstdp=True)
        pre_sp, post_sp, pre_t, post_t = _spike_pattern()
        syn.update_weights_rstdp(pre_sp, post_sp, pre_t, post_t,
                                 current_time=10.0, modulation=0.0)
        assert syn.eligibility is not None
        assert np.all(syn.eligibility == 0.0), "Zero modulation must leave eligibility unchanged"

    def test_rstdp_without_params_is_noop(self) -> None:
        """update_weights_rstdp must be a no-op when rstdp_params is None."""
        syn = _syn(with_eligibility=False, with_rstdp=False)
        before = syn.weights.data.copy()
        pre_sp, post_sp, pre_t, post_t = _spike_pattern()
        syn.update_weights_rstdp(pre_sp, post_sp, pre_t, post_t,
                                 current_time=10.0, modulation=5.0)
        np.testing.assert_array_equal(syn.weights.data, before)

    def test_surprise_bonus_increases_weight_change(self) -> None:
        """modulation > modulation_mismatch threshold must add surprise_bonus to dw."""
        # Two-factor mode: R-STDP updates weights directly for easy comparison.
        syn_surprise = _syn(with_eligibility=False, with_rstdp=True)
        syn_baseline = _syn(with_eligibility=False, with_rstdp=True)
        pre_sp, post_sp, pre_t, post_t = _spike_pattern()

        before_s = syn_surprise.weights.data.copy()
        before_b = syn_baseline.weights.data.copy()

        # modulation_mismatch=3.0; use 10.0 to guarantee surprise bonus
        syn_surprise.update_weights_rstdp(pre_sp, post_sp, pre_t, post_t,
                                          current_time=10.0, modulation=10.0)
        # modulation=1.0: below mismatch threshold, no surprise bonus
        syn_baseline.update_weights_rstdp(pre_sp, post_sp, pre_t, post_t,
                                          current_time=10.0, modulation=1.0)

        delta_surprise = float(np.mean(syn_surprise.weights.data) - np.mean(before_s))
        delta_baseline = float(np.mean(syn_baseline.weights.data) - np.mean(before_b))
        assert delta_surprise > delta_baseline, (
            f"Surprise bonus must enlarge dw: surprise={delta_surprise:.6f} "
            f"baseline={delta_baseline:.6f}"
        )
