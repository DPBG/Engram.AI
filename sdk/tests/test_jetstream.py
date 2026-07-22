"""Unit tests for JetStream safety-critical routing logic (no live NATS required)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from nats.js.api import RetentionPolicy, StorageType

from activelearning.nats_client import (
    _SAFETY_STREAM_SUBJECTS,
    DEFAULT_ACK_WAIT_SECONDS,
    DEFAULT_MAX_DELIVER,
    DEFAULT_REDELIVERY_BACKOFF,
    DLQ_SUBJECT_PREFIX,
    SAFETY_STREAM_MAX_AGE_SECONDS,
    SAFETY_STREAM_MAX_MSGS,
    SAFETY_STREAM_NAME,
    SAFETY_STREAM_RETENTION,
    SAFETY_STREAM_STORAGE,
    EventBus,
    poison_subject,
)


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


class TestSafetyStreamRetentionConfig:
    """Regression for issue #247: the safety-critical stream previously ran
    on undocumented NATS server defaults with no retention/size config."""

    def test_storage_is_file(self):
        # Kernel-privileged decision/proposal subjects must survive a broker
        # restart, not live only in memory.
        assert SAFETY_STREAM_STORAGE == StorageType.FILE

    def test_retention_is_limits(self):
        assert SAFETY_STREAM_RETENTION == RetentionPolicy.LIMITS

    def test_max_age_is_thirty_days(self):
        assert SAFETY_STREAM_MAX_AGE_SECONDS == 30 * 24 * 60 * 60

    def test_max_msgs_is_positive_and_bounded(self):
        assert SAFETY_STREAM_MAX_MSGS == 1_000_000

    def test_ensure_safety_stream_passes_retention_config_to_add_stream(self):
        bus = EventBus(name="test-retention")
        mock_js = MagicMock()
        mock_js.add_stream = AsyncMock()
        bus._js = mock_js

        asyncio.run(bus._ensure_safety_stream())

        mock_js.add_stream.assert_awaited_once()
        config = mock_js.add_stream.await_args.args[0]
        assert config.name == SAFETY_STREAM_NAME
        assert config.storage == SAFETY_STREAM_STORAGE
        assert config.retention == SAFETY_STREAM_RETENTION
        assert config.max_age == SAFETY_STREAM_MAX_AGE_SECONDS
        assert config.max_msgs == SAFETY_STREAM_MAX_MSGS


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


# ── DLQ / poison-message regression tests (issue #228) ──────────────────────


class TestDlqConstants:
    """Regression for issue #228: verify the DLQ subject prefix and bounded-
    redelivery defaults are stable so a refactor cannot silently break the
    contract that repeatedly-failing messages route to ``dlq.<subject>`` after
    at most ``DEFAULT_MAX_DELIVER`` attempts.
    """

    def test_dlq_prefix_is_dlq_dot(self):
        assert DLQ_SUBJECT_PREFIX == "dlq."

    def test_default_max_deliver_is_five(self):
        # Safety-critical consumers must not redeliver indefinitely; five
        # attempts (initial + four retries) is the committed default.
        assert DEFAULT_MAX_DELIVER == 5

    def test_default_redelivery_backoff_has_four_entries(self):
        # One entry per retry (after the first delivery); four entries for five
        # total attempts.  Changing the length silently changes how many
        # attempts occur before a message is dead-lettered.
        assert len(DEFAULT_REDELIVERY_BACKOFF) == 4

    def test_default_redelivery_backoff_values_are_increasing(self):
        # The backoff must grow (or at least not shrink) so a flapping consumer
        # does not hammer the broker at a constant high rate.
        b = DEFAULT_REDELIVERY_BACKOFF
        assert all(b[i] <= b[i + 1] for i in range(len(b) - 1))

    def test_default_redelivery_backoff_are_floats(self):
        assert all(isinstance(v, float) for v in DEFAULT_REDELIVERY_BACKOFF)

    def test_default_backoff_first_entry_covers_ack_wait(self):
        # NATS derives the ack deadline for delivery attempt N from backoff[N-1],
        # so backoff[0] is the ack-wait for the FIRST (healthy) delivery, not just
        # for retries. If it is shorter than a handler's runtime, the broker
        # redelivers a still-in-flight message and the handler runs again
        # concurrently -- double-processing every safety-critical proposal/decision
        # whose handler takes longer than backoff[0]. The Kernel's evaluation
        # budget alone is up to ~7s (5s Safety-Supervisor + 2s Beliefs), so the
        # first-delivery ack-wait must be at least the intended DEFAULT_ACK_WAIT.
        # Regression guard against the previous 1.0s value.
        assert DEFAULT_REDELIVERY_BACKOFF[0] >= DEFAULT_ACK_WAIT_SECONDS


class TestPoisonSubject:
    """Regression for issue #228: ``poison_subject()`` must always produce
    ``dlq.<original>`` so consumers can subscribe to the dead-letter feed."""

    def test_simple_subject(self):
        assert poison_subject("proposal.new") == "dlq.proposal.new"

    def test_decision_subject(self):
        assert poison_subject("decision.abc123") == "dlq.decision.abc123"

    def test_preserves_multi_level_subject(self):
        assert poison_subject("code.decision.trace-1") == "dlq.code.decision.trace-1"

    def test_prefix_matches_dlq_subject_prefix_constant(self):
        subject = "policy.restrict"
        assert poison_subject(subject).startswith(DLQ_SUBJECT_PREFIX)

    def test_suffix_is_original_subject(self):
        subject = "cognitive.response.validated"
        result = poison_subject(subject)
        assert result[len(DLQ_SUBJECT_PREFIX) :] == subject


class TestNumDelivered:
    """Regression for issue #228: ``_num_delivered`` must read the JetStream
    metadata correctly so the max_deliver boundary is enforced on the right
    attempt number, not off by one."""

    def test_reads_metadata_num_delivered(self):
        msg = MagicMock()
        msg.metadata.num_delivered = 3
        assert EventBus._num_delivered(msg) == 3

    def test_returns_one_when_metadata_unavailable(self):
        # Fallback: treat a message with no metadata as first delivery so it
        # gets a chance to be processed (not immediately poisoned).
        msg = MagicMock()
        msg.metadata.num_delivered = MagicMock(side_effect=AttributeError)
        # _num_delivered swallows ALL exceptions and returns 1
        assert EventBus._num_delivered(msg) == 1

    def test_returns_one_on_any_metadata_error(self):
        msg = MagicMock(spec=[])  # no attributes at all
        assert EventBus._num_delivered(msg) == 1
