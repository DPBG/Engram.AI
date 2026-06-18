"""
Pydantic wire-models for core NATS payloads.

Models are intentionally permissive (``extra="allow"``) so services can add
fields without breaking validation, while required keys are enforced up front.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from activelearning.subjects import Subjects


class WireModel(BaseModel):
    """Base wire model — allows forward-compatible extra fields."""

    model_config = ConfigDict(extra="allow")


# --- Governance / kernel ---


class ActionProposalMessage(WireModel):
    """Action proposal evaluated by the Kernel (``proposal.new``)."""

    trace_id: str
    action: dict[str, Any]
    provenance: str = "unknown"
    source: str | None = None


class CodeProposalMessage(WireModel):
    """Code change proposal (``code.proposal``)."""

    trace_id: str
    source: str = "meta-programmer"
    code: str | None = None
    description: str | None = None


class KernelDecisionMessage(WireModel):
    """Signed kernel decision payload."""

    trace_id: str
    type: str
    reason: str = ""
    risk_score: float = 0.0
    issued_at: int | None = None
    expires_at: int | None = None


class PolicyRestrictMessage(WireModel):
    motor_limits: dict[str, Any] | None = None
    reason: str = ""
    operator_id: str = "unknown"


class SafetyHaltMessage(WireModel):
    reason: str = "operator SAFE_HALT"
    operator_id: str = "unknown"


class OperatorActionMessage(WireModel):
    operator_id: str = "unknown"


# --- Safety analysis ---


class SafetyAnalyzeMessage(WireModel):
    """Risk analysis request (action or code)."""

    trace_id: str = ""
    action: dict[str, Any] | None = None
    code: str | None = None
    source: str | None = None


# --- Beliefs ---


class BeliefAddNodeMessage(WireModel):
    type: str
    content: str
    id: str = ""
    confidence: float = 1.0
    source: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


class BeliefAddEdgeMessage(WireModel):
    type: str
    source_id: str
    target_id: str
    id: str = ""
    strength: float = 1.0
    evidence: str | None = None


class BeliefUpdateMessage(WireModel):
    node_id: str
    evidence_strength: float
    supports: bool = True
    source: str = "unknown"


class BeliefQueryMessage(WireModel):
    type: str = "by_id"
    node_id: str | None = None
    node_type: str | None = None
    threshold: float = 0.8


class BeliefContradictionsMessage(WireModel):
    threshold: float = 0.5


# --- Memory ---


class MemoryStoreMessage(WireModel):
    trace_id: str
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    utility_score: float = 1.0
    id: str | None = None
    timestamp: int | None = None


class MemoryQueryMessage(WireModel):
    query: str
    limit: int = 10
    min_score: float = 0.5


class MemoryRecallMessage(WireModel):
    type: str = "similarity"
    query: str | None = None
    start_time: int | None = None
    end_time: int | None = None
    tags: list[str] | None = None
    limit: int = 100


# --- Coordinator ---


class TaskRequestMessage(WireModel):
    query: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class TaskResultMessage(WireModel):
    success: bool = False


# --- Planner / system ---


class PlannerModeMessage(WireModel):
    mode: str
    reason: str = ""


class SystemShutdownMessage(WireModel):
    reason: str = "shutdown requested"


class ObservationMessage(WireModel):
    """Loose validation for ``observation.*`` payloads."""

    trace_id: str
    provenance: str = "unknown"
    data: dict[str, Any] = Field(default_factory=dict)


# Registry: subscription pattern -> wire model
SUBJECT_SCHEMAS: dict[str, type[WireModel]] = {
    Subjects.PROPOSAL_NEW: ActionProposalMessage,
    Subjects.CODE_PROPOSAL: CodeProposalMessage,
    Subjects.POLICY_RESTRICT: PolicyRestrictMessage,
    Subjects.SAFETY_HALT: SafetyHaltMessage,
    Subjects.SAFETY_RESUME: OperatorActionMessage,
    Subjects.SAFETY_ANALYZE_ACTION: SafetyAnalyzeMessage,
    Subjects.SAFETY_ANALYZE_CODE: SafetyAnalyzeMessage,
    Subjects.BELIEFS_ADD_NODE: BeliefAddNodeMessage,
    Subjects.BELIEFS_ADD_EDGE: BeliefAddEdgeMessage,
    Subjects.BELIEFS_UPDATE: BeliefUpdateMessage,
    Subjects.BELIEFS_QUERY: BeliefQueryMessage,
    Subjects.BELIEFS_CONTRADICTIONS: BeliefContradictionsMessage,
    Subjects.BELIEFS_QUERY_REQUEST: BeliefQueryMessage,
    Subjects.MEMORY_STORE: MemoryStoreMessage,
    Subjects.MEMORY_QUERY: MemoryQueryMessage,
    Subjects.MEMORY_RECALL: MemoryRecallMessage,
    Subjects.TASK_REQUEST: TaskRequestMessage,
    Subjects.PLANNER_MODE: PlannerModeMessage,
    Subjects.SYSTEM_SHUTDOWN: SystemShutdownMessage,
    Subjects.OBSERVATION: ObservationMessage,
}


class MessageValidationError(ValueError):
    """Raised when an incoming NATS payload fails schema validation."""

    def __init__(self, subject: str, detail: str) -> None:
        self.subject = subject
        self.detail = detail
        super().__init__(f"Invalid message on subject '{subject}': {detail}")


def schema_for_subject(subject: str) -> type[WireModel] | None:
    """Return the wire model registered for a subscription pattern, if any."""
    if subject in SUBJECT_SCHEMAS:
        return SUBJECT_SCHEMAS[subject]
    if subject.startswith(Subjects.DECISION_PREFIX):
        return KernelDecisionMessage
    if subject.startswith(Subjects.CODE_DECISION_PREFIX):
        return KernelDecisionMessage
    return None


def validate_payload(
    subject: str,
    data: dict[str, Any],
    model: type[WireModel] | None = None,
) -> dict[str, Any]:
    """
    Validate and normalize a dict payload against a wire model.

    Returns a plain dict suitable for existing service handlers.
    """
    schema = model or schema_for_subject(subject)
    if schema is None:
        return data
    try:
        return schema.model_validate(data).model_dump(mode="python")
    except ValidationError as exc:
        raise MessageValidationError(subject, str(exc)) from exc
