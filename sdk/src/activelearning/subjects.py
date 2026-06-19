"""
Central NATS subject registry for Engram.AI.

All core subjects should be referenced from here instead of string literals
so publishers and subscribers stay aligned as the system evolves.
"""

from __future__ import annotations


class Subjects:
    """Named constants for core NATS subjects."""

    # Governance / kernel
    PROPOSAL_NEW = "proposal.new"
    CODE_PROPOSAL = "code.proposal"
    KERNEL_STATUS = "kernel.status"
    DECISION_PREFIX = "decision."
    CODE_DECISION_PREFIX = "code.decision."

    # Policy
    POLICY_LOAD_PROFILE = "policy.load_profile"
    POLICY_RESTRICT = "policy.restrict"
    POLICY_ROLLBACK = "policy.rollback"
    POLICY_UPDATE = "policy.update"
    COGNITIVE_RESPONSE_VALIDATE = "cognitive.response.validate"
    COGNITIVE_EXECUTE = "cognitive.execute"
    COGNITIVE_QUERY = "cognitive.query"

    # Safety
    SAFETY_ANALYZE_ACTION = "safety.analyze.action"
    SAFETY_ANALYZE_CODE = "safety.analyze.code"
    SAFETY_STATUS = "safety.status"
    SAFETY_HALT = "safety.halt"
    SAFETY_RESUME = "safety.resume"
    SAFETY_HALT_STATUS = "safety.halt.status"

    # Beliefs
    BELIEFS_ADD_NODE = "beliefs.add_node"
    BELIEFS_ADD_EDGE = "beliefs.add_edge"
    BELIEFS_UPDATE = "beliefs.update"
    BELIEFS_QUERY = "beliefs.query"
    BELIEFS_CONTRADICTIONS = "beliefs.contradictions"
    BELIEFS_QUERY_REQUEST = "beliefs.query.request"

    # Memory
    MEMORY_STORE = "memory.store"
    MEMORY_QUERY = "memory.query"
    MEMORY_RECALL = "memory.recall"

    # Coordinator / tasks
    TASK_REQUEST = "task.request"
    TASK_RESULT = "task.result"
    COORDINATOR_STATUS = "coordinator.status"

    # Planner
    PLANNER_MODE = "planner.mode"
    PLANNER_STATUS = "planner.status"

    # Observations (wildcard subscription)
    OBSERVATION = "observation.*"
    DECISION_WILDCARD = "decision.*"

    # System
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_HEALTH = "system.health"

    # Meta-programmer / learning
    KNOWLEDGE_GAP = "knowledge.gap"


def decision_subject(trace_id: str) -> str:
    """Per-trace kernel decision subject."""
    return f"{Subjects.DECISION_PREFIX}{trace_id}"


def code_decision_subject(trace_id: str) -> str:
    """Per-trace code decision subject."""
    return f"{Subjects.CODE_DECISION_PREFIX}{trace_id}"


def observation_subject(sensor_id: str) -> str:
    """Per-sensor observation subject."""
    return f"{Subjects.OBSERVATION.removesuffix('*')}{sensor_id}"
