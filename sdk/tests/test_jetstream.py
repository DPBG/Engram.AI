"""Unit tests for JetStream safety-critical routing logic (no live NATS required)."""

from activelearning.nats_client import (
    EventBus,
    JS_MAX_DELIVER,
    POISON_STREAM_NAME,
    POISON_SUBJECT_PREFIX,
    SAFETY_STREAM_NAME,
    _SAFETY_STREAM_SUBJECTS,
    poison_subject_for,
    safety_consumer_config,
)
from nats.js.api import AckPolicy


class TestSafetyStreamConstants:
    def test_stream_name_is_stable(self):
        assert SAFETY_STREAM_NAME == "SAFETY_CRITICAL"

    def test_required_subjects_present(self):
        required = {"proposal.new", "code.proposal", "decision.>", "code.decision.>"}
        assert required.issubset(set(_SAFETY_STREAM_SUBJECTS))


class TestIsSafetyCritical:
    def test_proposal_new(self):
        assert EventBus._is_safety_critical("proposal.new") is True

    def test_code_proposal(self):
        assert EventBus._is_safety_critical("code.proposal") is True

    def test_decision_with_trace_id(self):
        assert EventBus._is_safety_critical("decision.abc-123-def") is True

    def test_code_decision_with_trace_id(self):
        assert EventBus._is_safety_critical("code.decision.abc-123-def") is True

    def test_observation_not_critical(self):
        assert EventBus._is_safety_critical("observation.camera") is False

    def test_safety_analyze_not_critical(self):
        assert EventBus._is_safety_critical("safety.analyze.action") is False

    def test_policy_not_critical(self):
        assert EventBus._is_safety_critical("policy.restrict") is False

    def test_bare_decision_prefix_not_matched(self):
        # "decision" alone (no dot) should not match
        assert EventBus._is_safety_critical("decision") is False

    def test_bare_code_decision_prefix_not_matched(self):
        assert EventBus._is_safety_critical("code.decision") is False

    def test_partial_match_not_critical(self):
        assert EventBus._is_safety_critical("proposal.new.extra") is False


class TestEventBusInit:
    def test_js_durables_starts_empty(self):
        bus = EventBus()
        assert bus._js_durables == {}


class TestJetStreamConsumerPolicy:
    def test_poison_stream_constants(self):
        assert POISON_STREAM_NAME == "SAFETY_POISON"
        assert POISON_SUBJECT_PREFIX == "poison.safety."

    def test_poison_subject_mapping(self):
        assert poison_subject_for("proposal.new") == "poison.safety.proposal.new"

    def test_safety_consumer_config_defaults(self):
        cfg = safety_consumer_config()
        assert cfg.ack_policy == AckPolicy.EXPLICIT
        assert cfg.max_deliver == JS_MAX_DELIVER
        assert cfg.backoff
