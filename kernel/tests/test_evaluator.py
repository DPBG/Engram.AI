"""Tests for the Kernel code-proposal evaluator (Phase 1.5 + governance gate)."""

from dataclasses import dataclass

from activelearning import KernelDecisionType as DecisionType
from activelearning import RiskAnalysis

from kernel.evaluator import KernelEvaluator


def _ev():
    return KernelEvaluator()


def _low_risk() -> RiskAnalysis:
    return RiskAnalysis(trace_id="t", risk_score=0.0, flags=[])


def _proposal(target="/data/plugins/p.py", preview="x = 1"):
    return {"trace_id": "t", "target_path": target, "code_preview": preview}


def test_protected_path_denied():
    d = _ev().evaluate_code_proposal(_proposal(target="/kernel/evaluator.py"))
    assert d.type == DecisionType.DENY


def test_protected_path_safety_supervisor_denied():
    d = _ev().evaluate_code_proposal(_proposal(target="/safety-supervisor/analyzer.py"))
    assert d.type == DecisionType.DENY


def test_self_referential_code_denied_not_deferred():
    # Code touching the safety/meta machinery must be DENIED (fail closed),
    # never merely deferred.
    d = _ev().evaluate_code_proposal(_proposal(preview="import kernel.evaluator as k"))
    assert d.type == DecisionType.DENY


def test_dangerous_pattern_defers():
    d = _ev().evaluate_code_proposal(
        _proposal(preview="subprocess.run(['ls'])"),
        risk_analysis=_low_risk(),
    )
    assert d.type == DecisionType.DEFER


def test_defer_carries_expiry_deadline():
    # Phase 1.9: a DEFER is not open-ended — it has a deadline so an unanswered
    # human review can be failed closed (DENY) instead of lingering forever.
    d = _ev().evaluate_code_proposal(
        _proposal(preview="subprocess.run(['ls'])"),
        risk_analysis=_low_risk(),
    )
    assert d.type == DecisionType.DEFER
    assert d.expires_at is not None
    assert d.expires_at > d.issued_at


def test_clean_code_allowed():
    d = _ev().evaluate_code_proposal(
        _proposal(preview="def add(a, b):\n    return a + b\n"),
        risk_analysis=_low_risk(),
    )
    assert d.type == DecisionType.ALLOW
    # ALLOW decisions carry a TTL so stale approvals can't be replayed forever.
    assert d.expires_at is not None


def test_missing_risk_analysis_denies_action():
    d = _ev().evaluate_action_proposal(
        {"trace_id": "t", "action": {"channel": "head", "intensity": 0.1}},
        risk_analysis=None,
    )
    assert d.type == DecisionType.DENY
    assert d.risk_score == 1.0


def test_missing_risk_analysis_denies_clean_code():
    d = _ev().evaluate_code_proposal(
        _proposal(preview="def add(a, b):\n    return a + b\n"),
        risk_analysis=None,
    )
    assert d.type == DecisionType.DENY
    assert d.risk_score == 1.0


# ── SAFE_HALT kill switch (Phase 1.9) ────────────────────────────────────────


def test_safe_halt_denies_action_proposal():
    ev = _ev()
    ev.halt("emergency")
    assert ev.is_halted is True
    d = ev.evaluate_action_proposal(
        {"trace_id": "t", "action": {"channel": "head", "intensity": 0.1}}
    )
    assert d.type == DecisionType.DENY
    assert "SAFE_HALT" in d.reason


def test_safe_halt_denies_otherwise_clean_code():
    # Code that would normally ALLOW must be denied while halted.
    ev = _ev()
    ev.halt()
    d = ev.evaluate_code_proposal(_proposal(preview="def add(a, b):\n    return a + b\n"))
    assert d.type == DecisionType.DENY
    assert d.risk_score == 1.0


def test_resume_restores_normal_evaluation():
    ev = _ev()
    ev.halt()
    ev.resume()
    assert ev.is_halted is False
    d = ev.evaluate_code_proposal(
        _proposal(preview="def add(a, b):\n    return a + b\n"),
        risk_analysis=_low_risk(),
    )
    assert d.type == DecisionType.ALLOW


# ── Body-profile motor clamp must not mask the risk gate (issue #85) ──────────
#
# These tests use a tiny stand-in for beliefs.profiles.BodyProfile rather than
# importing it: that module pulls in PyYAML, which is not in the Governance CI
# job's dependency set. The evaluator only touches .name, .is_channel_allowed()
# and .get_motor_limit(), so this stub fully covers the interface under test.


@dataclass
class _Limit:
    max_intensity: float = 1.0


class _StubProfile:
    """Minimal BodyProfile stand-in matching the interface the evaluator uses."""

    def __init__(self, name, motor_limits=None, disallowed=()):
        self.name = name
        self._limits = motor_limits or {}
        self._disallowed = set(disallowed)

    def is_channel_allowed(self, channel):
        return channel not in self._disallowed

    def get_motor_limit(self, channel):
        return self._limits.get(channel, _Limit())


def _profile():
    """A profile that allows manipulation but caps its intensity at 0.5."""
    return _StubProfile("test-bot", motor_limits={"manipulation": _Limit(0.5)})


def _over_cap_action():
    # intensity 0.9 exceeds the profile cap of 0.5 -> would trigger the clamp
    return {"trace_id": "t", "action": {"channel": "manipulation", "intensity": 0.9}}


def test_clamp_does_not_mask_deny_level_risk():
    # A DENY-level risk (>= deny_threshold) must DENY even when the action also
    # exceeds a profile motor cap. The clamp must not short-circuit the DENY.
    ev = _ev()
    ev.set_body_profile(_profile())
    risk = RiskAnalysis(trace_id="t", risk_score=0.95, flags=["SUPERVISOR_HIGH_RISK"])
    d = ev.evaluate_action_proposal(_over_cap_action(), risk_analysis=risk)
    assert d.type == DecisionType.DENY
    assert d.risk_score == 0.95


def test_clamp_does_not_mask_defer_level_risk():
    # Elevated (defer-range) risk must DEFER for human approval, not be
    # auto-applied as a clamp TRANSFORM.
    ev = _ev()
    ev.set_body_profile(_profile())
    risk = RiskAnalysis(trace_id="t", risk_score=0.6, flags=["SUPERVISOR_MED_RISK"])
    d = ev.evaluate_action_proposal(_over_cap_action(), risk_analysis=risk)
    assert d.type == DecisionType.DEFER


def test_deny_level_risk_via_norm_violations_not_masked():
    # The same masking must not happen when the elevated risk comes from a
    # Beliefs norm-violation boost instead of the Supervisor risk_analysis.
    ev = _ev()
    ev.set_body_profile(_profile())
    norms = [{"norm_id": "no-force-at-person", "content": "...", "risk_boost": 0.9}]
    d = ev.evaluate_action_proposal(_over_cap_action(), norm_violations=norms)
    assert d.type == DecisionType.DENY


def test_clamp_still_applies_for_subthreshold_risk():
    # With low risk, the motor clamp must still work: TRANSFORM the action and
    # cap the intensity at the profile limit.
    ev = _ev()
    ev.set_body_profile(_profile())
    risk = RiskAnalysis(trace_id="t", risk_score=0.1)
    d = ev.evaluate_action_proposal(_over_cap_action(), risk_analysis=risk)
    assert d.type == DecisionType.TRANSFORM
    assert d.transformations[0]["intensity"] == 0.5


def test_disabled_channel_still_hard_denied():
    # Capability denials remain eager DENYs regardless of risk.
    ev = _ev()
    ev.set_body_profile(_StubProfile("no-hands", disallowed=["manipulation"]))
    d = ev.evaluate_action_proposal(_over_cap_action())
    assert d.type == DecisionType.DENY
