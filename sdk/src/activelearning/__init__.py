"""
ActiveLearningAI SDK

Core types, NATS client, plugin interfaces, and service infrastructure
for the self-learning humanoid AI framework.
"""

from activelearning.base_service import BaseService
from activelearning.config import ServiceConfig
from activelearning.core import (
    ActionProposal,
    BeliefEdge,
    BeliefEdgeType,
    BeliefNode,
    BeliefNodeType,
    KernelDecision,
    KernelDecisionType,
    Observation,
    Outcome,
    RiskAnalysis,
    current_timestamp,
    generate_trace_id,
)
from activelearning.database import Database, close_database, get_database
from activelearning.embeddings import (
    EmbeddingService,
    embed_batch,
    embed_text,
    get_embedding_service,
)
from activelearning.llm import LLMClient, LLMConfig, LLMError
from activelearning.messages import (
    SUBJECT_SCHEMAS,
    WIRE_SCHEMA_VERSION,
    MessageValidationError,
    WireModel,
    schema_for_subject,
    validate_payload,
)
from activelearning.nats_client import (
    CONNECTION_DRAIN_TIMEOUT_S,
    DEFAULT_DECISION_TIMEOUT_S,
    DEFAULT_REQUEST_TIMEOUT_S,
    RECONNECT_WAIT_TIMEOUT_S,
    EventBus,
    get_event_bus,
    close_event_bus
)
from activelearning.plugins import ActuatorPlugin, SensorPlugin, register_actuator, register_sensor
from activelearning.qdrant_store import QdrantHit, QdrantPoint, QdrantStore
from activelearning.signing import (
    DECISION_KEY_ENV,
    OPERATOR_KEY_ENV,
    OPERATOR_TIMESTAMP_TOLERANCE_MS,
    operator_signing_enabled,
    sign_decision,
    sign_operator_action,
    signing_enabled,
    verify_decision,
    verify_operator_action,
)
from activelearning.subjects import (
    Subjects,
    code_decision_subject,
    decision_subject,
    observation_subject,
)

__version__ = "0.1.0"

__all__ = [
    # Core types
    "KernelDecisionType",
    "Observation",
    "ActionProposal",
    "KernelDecision",
    "Outcome",
    "BeliefNode",
    "BeliefEdge",
    "BeliefNodeType",
    "BeliefEdgeType",
    "RiskAnalysis",
    # Core utilities
    "generate_trace_id",
    "current_timestamp",
    # NATS client
    "EventBus",
    "get_event_bus",
    "close_event_bus",
    # EventBus timeout policy (issue #233)
    "DEFAULT_REQUEST_TIMEOUT_S",
    "DEFAULT_DECISION_TIMEOUT_S",
    "RECONNECT_WAIT_TIMEOUT_S",
    "CONNECTION_DRAIN_TIMEOUT_S",
    # Database
    "Database",
    "get_database",
    "close_database",
    # Embeddings
    "EmbeddingService",
    "get_embedding_service",
    "embed_text",
    "embed_batch",
    # Vector store
    "QdrantStore",
    "QdrantHit",
    "QdrantPoint",
    # LLM (text generation / chat)
    "LLMClient",
    "LLMConfig",
    "LLMError",
    # Configuration
    "ServiceConfig",
    # Base service
    "BaseService",
    # Plugins
    "SensorPlugin",
    "ActuatorPlugin",
    "register_sensor",
    "register_actuator",
    # Decision signing (safety gate authentication)
    "sign_decision",
    "sign_operator_action",
    "verify_decision",
    "verify_operator_action",
    "signing_enabled",
    "operator_signing_enabled",
    "DECISION_KEY_ENV",
    "OPERATOR_KEY_ENV",
    "OPERATOR_TIMESTAMP_TOLERANCE_MS",
    # NATS subject registry
    "Subjects",
    "decision_subject",
    "code_decision_subject",
    "observation_subject",
    # Wire models / validation
    "MessageValidationError",
    "validate_payload",
    "schema_for_subject",
    "SUBJECT_SCHEMAS",
    "WireModel",
    "WIRE_SCHEMA_VERSION",
]
