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
    assert d.risk_score == 1.0


def test_protected_path_safety_supervisor_denied():
    d = _ev().evaluate_code_proposal(_proposal(target="/safety-supervisor/analyzer.py"))
    assert d.type == DecisionType.DENY


def test_self_referential_code_denied_not_deferred():
    # Code touching the safety/meta machinery must be DENIED (fail closed),
    # never merely deferred.
    d = _ev().evaluate_code_proposal(
        _proposal(preview="import kernel.evaluator as k"),
        risk_analysis=_low_risk(),
    )
    assert d.type == DecisionType.DENY
    # risk_score = max(risk_score, 0.95); with a low base risk this must be
    # exactly 0.95, not left at the base risk or some other floor.
    assert d.risk_score == 0.95


def test_dangerous_pattern_defers():
    d = _ev().evaluate_code_proposal(
        _proposal(preview="subprocess.run(['ls'])"),
        risk_analysis=_low_risk(),
    )
    assert d.type == DecisionType.DEFER
    # risk_score = max(risk_score, 0.7) when a dangerous pattern is found.
    assert d.risk_score == 0.7


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
    assert d.risk_score == 0.0


def test_missing_trace_id_denies_with_max_risk():
    ev = _ev()
    d = ev.evaluate_action_proposal(
        {"action": {"channel": "head", "intensity": 0.1}},  # no trace_id
        risk_analysis=_low_risk(),
    )
    assert d.type == DecisionType.DENY
    assert d.risk_score == 1.0


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
    assert d.risk_score == 0.1


def test_disabled_channel_still_hard_denied():
    # Capability denials remain eager DENYs regardless of risk.
    ev = _ev()
    ev.set_body_profile(_StubProfile("no-hands", disallowed=["manipulation"]))
    d = ev.evaluate_action_proposal(_over_cap_action())
    assert d.type == DecisionType.DENY
    assert d.risk_score == 1.0


def test_profile_deny_risk_score_defaults_when_dict_omits_it(monkeypatch):
    # _check_body_profile_denials always sets "risk_score" in the dicts it
    # returns today, so evaluate_action_proposal's profile_deny.get("risk_score",
    # 1.0) default is only observable if that assumption ever breaks. Simulate
    # that directly so the fallback default itself stays covered.
    ev = _ev()
    ev.set_body_profile(_profile())
    monkeypatch.setattr(
        ev,
        "_check_body_profile_denials",
        lambda action, flags: {"type": DecisionType.DENY, "reason": "no risk_score key"},
    )
    d = ev.evaluate_action_proposal({"trace_id": "t", "action": {"channel": "manipulation"}})
    assert d.type == DecisionType.DENY
    assert d.risk_score == 1.0


def test_profile_deny_risk_score_read_from_correct_dict_key(monkeypatch):
    # A distinct, non-default value proves evaluate_action_proposal reads the
    # "risk_score" key specifically -- a wrong-key lookup would silently fall
    # back to the 1.0 default instead and be indistinguishable from this.
    ev = _ev()
    ev.set_body_profile(_profile())
    monkeypatch.setattr(
        ev,
        "_check_body_profile_denials",
        lambda action, flags: {"type": DecisionType.DENY, "reason": "x", "risk_score": 0.42},
    )
    d = ev.evaluate_action_proposal({"trace_id": "t", "action": {"channel": "manipulation"}})
    assert d.risk_score == 0.42


def test_envelope_violation_denies_with_risk_score_point_nine():
    d = _ev().evaluate_action_proposal(
        {"trace_id": "t", "action": {"channel": "head", "intensity": 1.5}},
        risk_analysis=_low_risk(),
    )
    assert d.type == DecisionType.DENY
    assert d.risk_score == 0.9


# ── Risk-clamping logic (issue #211: mutation-test the class of bug from
# PR #122, not just the one regression it added) ─────────────────────────────
#
# kernel/service.py's _get_risk_analysis() already rejects a non-finite
# risk_score from the wire before ever constructing a RiskAnalysis. These
# tests exercise KernelEvaluator._risk_from_analysis() directly (and through
# evaluate_action_proposal/evaluate_code_proposal) with a directly-constructed
# RiskAnalysis, because the evaluator is public API: nothing stops a future
# caller from building a RiskAnalysis without going through that wire guard,
# and the clamp itself must independently fail closed.


def test_nan_risk_score_fails_closed_for_action():
    # max(0.0, min(nan, 1.0)) == 0.0 in Python -- NaN comparisons are always
    # False, so min/max silently keep their first argument. Unguarded, this
    # clamp would turn a NaN risk score into "zero risk" and ALLOW the action.
    ev = _ev()
    risk = RiskAnalysis(trace_id="t", risk_score=float("nan"), flags=["SOME_FLAG"])
    d = ev.evaluate_action_proposal(
        {"trace_id": "t", "action": {"channel": "head", "intensity": 0.1}},
        risk_analysis=risk,
    )
    assert d.type == DecisionType.DENY
    assert d.risk_score == 1.0


def test_nan_risk_score_fails_closed_for_code():
    ev = _ev()
    risk = RiskAnalysis(trace_id="t", risk_score=float("nan"), flags=[])
    d = ev.evaluate_code_proposal(
        _proposal(preview="def add(a, b):\n    return a + b\n"),
        risk_analysis=risk,
    )
    assert d.type == DecisionType.DENY
    assert d.risk_score == 1.0


def test_positive_infinity_risk_score_fails_closed():
    ev = _ev()
    risk = RiskAnalysis(trace_id="t", risk_score=float("inf"), flags=[])
    d = ev.evaluate_action_proposal(
        {"trace_id": "t", "action": {"channel": "head", "intensity": 0.1}},
        risk_analysis=risk,
    )
    assert d.type == DecisionType.DENY
    assert d.risk_score == 1.0


def test_negative_infinity_risk_score_fails_closed():
    # -inf clamped naively would be 0.0 (min(-inf, 1.0) == -inf, then
    # max(0.0, -inf) == 0.0) -- silently "zero risk", not merely wrong-signed.
    ev = _ev()
    risk = RiskAnalysis(trace_id="t", risk_score=float("-inf"), flags=[])
    d = ev.evaluate_action_proposal(
        {"trace_id": "t", "action": {"channel": "head", "intensity": 0.1}},
        risk_analysis=risk,
    )
    assert d.type == DecisionType.DENY
    assert d.risk_score == 1.0


def test_risk_from_analysis_clamps_above_one():
    ev = _ev()
    score, flags = ev._risk_from_analysis(RiskAnalysis(trace_id="t", risk_score=5.0, flags=["X"]))
    assert score == 1.0
    assert flags == ["X"]


def test_risk_from_analysis_clamps_below_zero():
    ev = _ev()
    score, _flags = ev._risk_from_analysis(RiskAnalysis(trace_id="t", risk_score=-5.0, flags=[]))
    assert score == 0.0


def test_risk_from_analysis_passes_through_midrange_unchanged():
    ev = _ev()
    score, flags = ev._risk_from_analysis(
        RiskAnalysis(trace_id="t", risk_score=0.42, flags=["A", "B"])
    )
    assert score == 0.42
    assert flags == ["A", "B"]


def test_risk_from_analysis_none_is_unavailable():
    ev = _ev()
    score, flags = ev._risk_from_analysis(None)
    assert score == 1.0
    assert flags == ["SAFETY_ANALYSIS_UNAVAILABLE"]


# ── Threshold boundaries (deny_threshold / defer_threshold are >=, not >) ─────
#
# mutmut's default operator mutations include >= -> > and >= -> ==; these
# tests pin the exact boundary so such a mutant is killed.


def test_risk_exactly_at_deny_threshold_denies():
    ev = _ev()
    risk = RiskAnalysis(trace_id="t", risk_score=ev.deny_threshold, flags=[])
    d = ev.evaluate_action_proposal(
        {"trace_id": "t", "action": {"channel": "head", "intensity": 0.1}},
        risk_analysis=risk,
    )
    assert d.type == DecisionType.DENY


def test_risk_just_below_deny_threshold_does_not_deny():
    ev = _ev()
    risk = RiskAnalysis(trace_id="t", risk_score=ev.deny_threshold - 0.01, flags=[])
    d = ev.evaluate_action_proposal(
        {"trace_id": "t", "action": {"channel": "head", "intensity": 0.1}},
        risk_analysis=risk,
    )
    assert d.type != DecisionType.DENY


def test_risk_exactly_at_defer_threshold_defers():
    ev = _ev()
    risk = RiskAnalysis(trace_id="t", risk_score=ev.defer_threshold, flags=[])
    d = ev.evaluate_action_proposal(
        {"trace_id": "t", "action": {"channel": "head", "intensity": 0.1}},
        risk_analysis=risk,
    )
    assert d.type == DecisionType.DEFER
    assert d.risk_score == ev.defer_threshold


def test_risk_just_below_defer_threshold_allows():
    ev = _ev()
    risk = RiskAnalysis(trace_id="t", risk_score=ev.defer_threshold - 0.01, flags=[])
    d = ev.evaluate_action_proposal(
        {"trace_id": "t", "action": {"channel": "head", "intensity": 0.1}},
        risk_analysis=risk,
    )
    assert d.type == DecisionType.ALLOW
    assert d.risk_score == ev.defer_threshold - 0.01


def test_code_risk_exactly_at_deny_threshold_denies():
    ev = _ev()
    risk = RiskAnalysis(trace_id="t", risk_score=ev.deny_threshold, flags=[])
    d = ev.evaluate_code_proposal(
        _proposal(preview="def add(a, b):\n    return a + b\n"),
        risk_analysis=risk,
    )
    assert d.type == DecisionType.DENY


def test_code_risk_exactly_at_defer_threshold_defers():
    ev = _ev()
    risk = RiskAnalysis(trace_id="t", risk_score=ev.defer_threshold, flags=[])
    d = ev.evaluate_code_proposal(
        _proposal(preview="def add(a, b):\n    return a + b\n"),
        risk_analysis=risk,
    )
    assert d.type == DecisionType.DEFER
    assert d.risk_score == ev.defer_threshold


# ── Norm-violation risk_boost (Beliefs system) ────────────────────────────────


def test_norm_violation_missing_risk_boost_uses_default_point_one():
    # violation.get("risk_boost", 0.1) -- exercise the DEFAULT, not an
    # explicitly-provided value, so a mutated default or key name is caught.
    ev = _ev()
    norms = [{"norm_id": "no-key"}]  # no "risk_boost" key at all
    d = ev.evaluate_action_proposal(
        {"trace_id": "t", "action": {"channel": "head", "intensity": 0.1}},
        risk_analysis=_low_risk(),
        norm_violations=norms,
    )
    assert d.risk_score == 0.1


def test_norm_violation_risk_boost_is_additive():
    ev = _ev()
    norms = [{"norm_id": "n", "risk_boost": 0.3}]
    d = ev.evaluate_action_proposal(
        {"trace_id": "t", "action": {"channel": "head", "intensity": 0.1}},
        risk_analysis=_low_risk(),
        norm_violations=norms,
    )
    assert d.risk_score == 0.3


def test_norm_violation_risk_boost_clamps_to_one():
    # Two violations push the total to 1.8 -- must clamp to exactly 1.0, not
    # a raised ceiling like 2.0.
    ev = _ev()
    norms = [
        {"norm_id": "a", "risk_boost": 0.9},
        {"norm_id": "b", "risk_boost": 0.9},
    ]
    d = ev.evaluate_action_proposal(
        {"trace_id": "t", "action": {"channel": "head", "intensity": 0.1}},
        risk_analysis=_low_risk(),
        norm_violations=norms,
    )
    assert d.risk_score == 1.0


def test_cognitive_query_denied_when_cognitive_channel_disallowed():
    # Guards the (formerly duplicated) action_type == "cognitive_query" branch
    # in _check_body_profile_denials -- the dead duplicate has been removed,
    # so this is now the only test exercising that branch's live code path.
    ev = _ev()
    ev.set_body_profile(_StubProfile("no-cognitive", disallowed=["cognitive"]))
    d = ev.evaluate_action_proposal(
        {"trace_id": "t", "action": {"type": "cognitive_query"}},
        risk_analysis=_low_risk(),
    )
    assert d.type == DecisionType.DENY
    assert "cognitive" in d.reason.lower()
