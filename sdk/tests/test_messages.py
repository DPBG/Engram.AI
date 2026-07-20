"""Tests for NATS subject registry and pydantic wire-models."""

import pytest

from activelearning.messages import (
    WIRE_SCHEMA_VERSION,
    ActionProposalMessage,
    CognitiveQueryMessage,
    CognitiveResponseValidateMessage,
    KernelDecisionMessage,
    MessageValidationError,
    WireModel,
    schema_for_subject,
    validate_payload,
)
from activelearning.subjects import Subjects, code_decision_subject, decision_subject


class TestSubjectHelpers:
    def test_decision_subject(self):
        assert decision_subject("abc-123") == "decision.abc-123"

    def test_code_decision_subject(self):
        assert code_decision_subject("xyz") == "code.decision.xyz"

    def test_schema_for_dynamic_decision(self):
        assert schema_for_subject("decision.trace-1") is KernelDecisionMessage
        assert schema_for_subject("code.decision.trace-2") is KernelDecisionMessage

    def test_schema_for_registered_subject(self):
        assert schema_for_subject(Subjects.PROPOSAL_NEW) is ActionProposalMessage
        assert schema_for_subject(Subjects.COGNITIVE_EXECUTE) is CognitiveQueryMessage
        assert schema_for_subject(Subjects.COGNITIVE_QUERY) is CognitiveQueryMessage
        assert (
            schema_for_subject(Subjects.COGNITIVE_RESPONSE_VALIDATE)
            is CognitiveResponseValidateMessage
        )
        assert schema_for_subject("unknown.subject") is None


class TestValidatePayload:
    def test_valid_action_proposal(self):
        payload = {
            "trace_id": "t1",
            "action": {"type": "move", "channel": "locomotion"},
            "provenance": "planner",
        }
        result = validate_payload(Subjects.PROPOSAL_NEW, payload)
        assert result["trace_id"] == "t1"
        assert result["action"]["type"] == "move"

    def test_missing_required_field_raises(self):
        with pytest.raises(MessageValidationError) as exc:
            validate_payload(Subjects.PROPOSAL_NEW, {"action": {}})
        assert "trace_id" in exc.value.detail

    def test_belief_add_node_requires_type(self):
        with pytest.raises(MessageValidationError):
            validate_payload(Subjects.BELIEFS_ADD_NODE, {"content": "norm text"})

    def test_extra_fields_preserved(self):
        payload = {
            "trace_id": "t2",
            "action": {"type": "speech"},
            "custom_field": 42,
        }
        result = validate_payload(Subjects.PROPOSAL_NEW, payload, ActionProposalMessage)
        assert result["custom_field"] == 42

    def test_unregistered_subject_passes_through(self):
        data = {"anything": True}
        assert validate_payload("custom.event", data) is data

    def test_kernel_decision_validation(self):
        payload = {
            "trace_id": "t3",
            "type": "ALLOW",
            "reason": "ok",
            "risk_score": 0.1,
        }
        result = validate_payload(decision_subject("t3"), payload)
        assert result["type"] == "ALLOW"


class TestSchemaVersioning:
    def test_wire_schema_version_is_positive_int(self):
        assert isinstance(WIRE_SCHEMA_VERSION, int)
        assert WIRE_SCHEMA_VERSION >= 1

    def test_base_model_has_version_field(self):
        assert "version" in WireModel.model_fields
        assert WireModel().version == WIRE_SCHEMA_VERSION

    def test_all_models_inherit_version(self):
        msg = ActionProposalMessage(trace_id="t1", action={"type": "move"})
        assert msg.version == WIRE_SCHEMA_VERSION

    def test_missing_version_defaults_to_current(self):
        # A legacy sender that omits `version` is normalized to the current
        # wire-schema version so downstream code always sees an explicit value.
        payload = {"trace_id": "t1", "action": {"type": "move"}}
        result = validate_payload(Subjects.PROPOSAL_NEW, payload)
        assert result["version"] == WIRE_SCHEMA_VERSION

    def test_explicit_version_is_preserved(self):
        # A message stamped with a different major version round-trips unchanged
        # so a consumer can inspect it and decide how to handle it.
        payload = {"trace_id": "t1", "action": {"type": "move"}, "version": 2}
        result = validate_payload(Subjects.PROPOSAL_NEW, payload)
        assert result["version"] == 2

    def test_version_added_to_serialized_payload(self):
        payload = {"query": "hello"}
        result = validate_payload(Subjects.MEMORY_QUERY, payload)
        assert result["version"] == WIRE_SCHEMA_VERSION
