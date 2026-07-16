"""Unit tests for JetStream safety-critical routing logic (no live NATS required)."""

from activelearning.nats_client import _SAFETY_STREAM_SUBJECTS, SAFETY_STREAM_NAME, EventBus


class TestSafetyStreamConstants:
    def test_stream_name_is_stable(self):
        assert SAFETY_STREAM_NAME == "SAFETY_CRITICAL"

    def test_required_subjects_present(self):
        required = {
            "proposal.new",
            "code.proposal",
            "decision.>",
            "code.decision.>",
            "policy.>",
            "cognitive.response.>",
        }
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

    def test_policy_restrict_is_critical(self):
        # ADR 0001 §3: policy.* is Kernel-privileged — e.g. SAFE_HALT's
        # motor-zeroing broadcast must survive a consumer's brief reconnect.
        assert EventBus._is_safety_critical("policy.restrict") is True

    def test_policy_restrict_request_is_critical(self):
        assert EventBus._is_safety_critical("policy.restrict.request") is True

    def test_policy_rollback_is_critical(self):
        assert EventBus._is_safety_critical("policy.rollback") is True

    def test_policy_update_is_critical(self):
        assert EventBus._is_safety_critical("policy.update") is True

    def test_policy_load_profile_is_critical(self):
        assert EventBus._is_safety_critical("policy.load_profile") is True

    def test_policy_status_replies_are_critical(self):
        assert EventBus._is_safety_critical("policy.restrict.status") is True
        assert EventBus._is_safety_critical("policy.rollback.status") is True
        assert EventBus._is_safety_critical("policy.update.status") is True
        assert EventBus._is_safety_critical("policy.profile.status") is True

    def test_cognitive_response_validated_is_critical(self):
        assert EventBus._is_safety_critical("cognitive.response.validated") is True

    def test_cognitive_response_rejected_is_critical(self):
        assert EventBus._is_safety_critical("cognitive.response.rejected") is True

    def test_cognitive_response_validate_request_is_critical(self):
        assert EventBus._is_safety_critical("cognitive.response.validate") is True

    def test_bare_decision_prefix_not_matched(self):
        # "decision" alone (no dot) should not match
        assert EventBus._is_safety_critical("decision") is False

    def test_bare_code_decision_prefix_not_matched(self):
        assert EventBus._is_safety_critical("code.decision") is False

    def test_bare_policy_prefix_not_matched(self):
        # "policy" alone (no dot) or a look-alike subject without the
        # separator should not match — only the "policy." namespace is
        # Kernel-privileged.
        assert EventBus._is_safety_critical("policy") is False
        assert EventBus._is_safety_critical("policyish.other") is False

    def test_bare_cognitive_response_prefix_not_matched(self):
        assert EventBus._is_safety_critical("cognitive.response") is False

    def test_unrelated_cognitive_subject_not_matched(self):
        assert EventBus._is_safety_critical("cognitive.execute") is False
        assert EventBus._is_safety_critical("cognitive.query") is False

    def test_partial_match_not_critical(self):
        assert EventBus._is_safety_critical("proposal.new.extra") is False


class TestEventBusInit:
    def test_js_durables_starts_empty(self):
        bus = EventBus()
        assert bus._js_durables == {}
