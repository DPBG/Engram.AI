"""Tests for core types and contracts."""

import pytest
from activelearning.core import (
    KernelDecisionType,
    Observation,
    ActionProposal,
    KernelDecision,
    Outcome,
    BeliefNode,
    BeliefEdge,
    BeliefNodeType,
    BeliefEdgeType,
    KnowledgeGap,
    CodeProposal,
    generate_trace_id,
    current_timestamp,
)


class TestTraceIdAndTimestamp:
    """Tests for utility functions."""

    def test_generate_trace_id_is_uuid(self):
        trace_id = generate_trace_id()
        assert len(trace_id) == 36  # UUID format
        assert trace_id.count("-") == 4

    def test_generate_trace_id_unique(self):
        ids = [generate_trace_id() for _ in range(100)]
        assert len(set(ids)) == 100  # All unique

    def test_current_timestamp_is_milliseconds(self):
        ts = current_timestamp()
        assert ts > 1700000000000  # After 2023
        assert ts < 2000000000000  # Before 2033


class TestObservation:
    """Tests for Observation dataclass."""

    def test_observation_basic(self):
        obs = Observation(
            trace_id="test-123",
            provenance="sensor.camera",
            data={"frame": "base64..."},
        )
        assert obs.trace_id == "test-123"
        assert obs.provenance == "sensor.camera"
        assert obs.confidence == 1.0
        assert obs.tags == []

    def test_observation_auto_trace_id(self):
        obs = Observation(
            trace_id="",
            provenance="sensor.test",
            data={},
        )
        assert obs.trace_id != ""
        assert len(obs.trace_id) == 36

    def test_observation_with_tags(self):
        obs = Observation(
            trace_id="test",
            provenance="test",
            data={},
            tags=["motion", "detected"],
        )
        assert obs.tags == ["motion", "detected"]

    def test_observation_invalid_confidence(self):
        with pytest.raises(ValueError, match="Confidence must be between"):
            Observation(
                trace_id="test",
                provenance="test",
                data={},
                confidence=1.5,
            )


class TestActionProposal:
    """Tests for ActionProposal dataclass."""

    def test_proposal_basic(self):
        proposal = ActionProposal(
            trace_id="test-123",
            provenance="planner.main",
            action={"type": "move", "x": 10},
        )
        assert proposal.trace_id == "test-123"
        assert proposal.priority == 0
        assert not proposal.requires_approval

    def test_proposal_with_priority(self):
        proposal = ActionProposal(
            trace_id="test",
            provenance="test",
            action={},
            priority=5,
            requires_approval=True,
        )
        assert proposal.priority == 5
        assert proposal.requires_approval


class TestKernelDecision:
    """Tests for KernelDecision dataclass."""

    def test_decision_allow(self):
        decision = KernelDecision(
            trace_id="test-123",
            type=KernelDecisionType.ALLOW,
        )
        assert decision.is_approved()
        assert not decision.is_expired()

    def test_decision_deny(self):
        decision = KernelDecision(
            trace_id="test-123",
            type=KernelDecisionType.DENY,
            reason="Risk too high",
        )
        assert not decision.is_approved()
        assert decision.reason == "Risk too high"

    def test_decision_transform(self):
        decision = KernelDecision(
            trace_id="test-123",
            type=KernelDecisionType.TRANSFORM,
            transformations=[
                ActionProposal(
                    trace_id="test-123",
                    provenance="kernel",
                    action={"type": "move_safe", "x": 5},
                )
            ],
        )
        assert decision.is_approved()
        assert len(decision.transformations) == 1

    def test_decision_defer(self):
        decision = KernelDecision(
            trace_id="test-123",
            type=KernelDecisionType.DEFER,
            reason="Requires human approval",
        )
        assert not decision.is_approved()

    def test_decision_expired(self):
        decision = KernelDecision(
            trace_id="test-123",
            type=KernelDecisionType.ALLOW,
            expires_at=1000,  # Long ago
        )
        assert decision.is_expired()


class TestKernelDecisionTtl:
    """Tests for KernelDecision.remaining_ttl_ms()."""

    def test_no_expiry_returns_none(self):
        decision = KernelDecision(
            trace_id="test",
            type=KernelDecisionType.ALLOW,
        )
        assert decision.remaining_ttl_ms() is None

    def test_future_expiry_is_positive(self):
        decision = KernelDecision(
            trace_id="test",
            type=KernelDecisionType.ALLOW,
            expires_at=10_000,
        )
        assert decision.remaining_ttl_ms(now=4_000) == 6_000

    def test_past_expiry_clamps_to_zero(self):
        decision = KernelDecision(
            trace_id="test",
            type=KernelDecisionType.ALLOW,
            expires_at=1_000,
        )
        assert decision.remaining_ttl_ms(now=5_000) == 0

    def test_defaults_to_current_time(self):
        decision = KernelDecision(
            trace_id="test",
            type=KernelDecisionType.ALLOW,
            expires_at=current_timestamp() + 60_000,
        )
        remaining = decision.remaining_ttl_ms()
        assert remaining is not None
        assert 0 < remaining <= 60_000


class TestOutcome:
    """Tests for Outcome dataclass."""

    def test_outcome_success(self):
        proposal = ActionProposal(
            trace_id="test",
            provenance="test",
            action={},
        )
        decision = KernelDecision(
            trace_id="test",
            type=KernelDecisionType.ALLOW,
        )
        outcome = Outcome(
            trace_id="test",
            decision=decision,
            original=proposal,
            success=True,
        )
        assert outcome.success
        assert outcome.error is None

    def test_outcome_failure(self):
        proposal = ActionProposal(
            trace_id="test",
            provenance="test",
            action={},
        )
        decision = KernelDecision(
            trace_id="test",
            type=KernelDecisionType.DENY,
        )
        outcome = Outcome(
            trace_id="test",
            decision=decision,
            original=proposal,
            success=False,
            error="Action denied",
        )
        assert not outcome.success
        assert outcome.error == "Action denied"


class TestBeliefNode:
    """Tests for BeliefNode dataclass."""

    def test_belief_node_value(self):
        node = BeliefNode(
            id="belief-1",
            type=BeliefNodeType.VALUE,
            content="Safety is paramount",
            confidence=0.95,
        )
        assert node.type == BeliefNodeType.VALUE
        assert node.confidence == 0.95

    def test_belief_node_fact(self):
        node = BeliefNode(
            id="belief-2",
            type=BeliefNodeType.FACT,
            content="Water boils at 100C",
            source="physics",
        )
        assert node.type == BeliefNodeType.FACT
        assert node.source == "physics"

    def test_belief_node_invalid_confidence(self):
        with pytest.raises(ValueError):
            BeliefNode(
                id="test",
                type=BeliefNodeType.FACT,
                content="test",
                confidence=-0.5,
            )


class TestBeliefEdge:
    """Tests for BeliefEdge dataclass."""

    def test_belief_edge_supports(self):
        edge = BeliefEdge(
            id="edge-1",
            type=BeliefEdgeType.SUPPORTS,
            source_id="belief-1",
            target_id="belief-2",
            strength=0.8,
        )
        assert edge.type == BeliefEdgeType.SUPPORTS
        assert edge.strength == 0.8

    def test_belief_edge_contradicts(self):
        edge = BeliefEdge(
            id="edge-2",
            type=BeliefEdgeType.CONTRADICTS,
            source_id="belief-1",
            target_id="belief-3",
            evidence="Conflicting observation",
        )
        assert edge.type == BeliefEdgeType.CONTRADICTS
        assert edge.evidence == "Conflicting observation"


class TestKnowledgeGap:
    """Tests for KnowledgeGap dataclass."""

    def test_knowledge_gap_basic(self):
        gap = KnowledgeGap(
            trace_id="gap-1",
            description="How to pick up a cup",
            context={"object": "cup", "location": "table"},
        )
        assert gap.description == "How to pick up a cup"
        assert gap.allows_external_query

    def test_knowledge_gap_with_sensors(self):
        gap = KnowledgeGap(
            trace_id="gap-2",
            description="Unknown gesture",
            available_sensors=["camera", "depth"],
            allows_external_query=False,
        )
        assert "camera" in gap.available_sensors
        assert not gap.allows_external_query


class TestCodeProposal:
    """Tests for CodeProposal dataclass."""

    def test_code_proposal_create(self):
        proposal = CodeProposal(
            trace_id="code-1",
            gap_ref="gap-1",
            proposed_action="CREATE",
            target_path="/data/plugins/camera_sensor.py",
            code_preview="class CameraSensor:...",
            test_plan="Test frame capture",
            rollback_plan="Delete file",
            agent="CodeGenAgent",
        )
        assert proposal.proposed_action == "CREATE"
        assert proposal.agent == "CodeGenAgent"

    def test_code_proposal_invalid_action(self):
        with pytest.raises(ValueError, match="Invalid proposed_action"):
            CodeProposal(
                trace_id="test",
                gap_ref=None,
                proposed_action="INVALID",
                target_path="/test",
                code_preview="",
            )
