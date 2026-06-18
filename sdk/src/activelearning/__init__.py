"""
ActiveLearningAI SDK

Core types, NATS client, plugin interfaces, and service infrastructure
for the self-learning humanoid AI framework.
"""

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
    RiskAnalysis,
    generate_trace_id,
    current_timestamp,
)
from activelearning.nats_client import EventBus, get_event_bus
from activelearning.plugins import SensorPlugin, ActuatorPlugin, register_sensor, register_actuator
from activelearning.database import Database, get_database
from activelearning.embeddings import EmbeddingService, get_embedding_service, embed_text, embed_batch
from activelearning.config import ServiceConfig
from activelearning.base_service import BaseService
from activelearning.signing import (
    sign_decision,
    verify_decision,
    signing_enabled,
    DECISION_KEY_ENV,
)
from activelearning.subjects import Subjects, code_decision_subject, decision_subject, observation_subject
from activelearning.messages import (
    MessageValidationError,
    validate_payload,
    schema_for_subject,
    SUBJECT_SCHEMAS,
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
    # Database
    "Database",
    "get_database",
    # Embeddings
    "EmbeddingService",
    "get_embedding_service",
    "embed_text",
    "embed_batch",
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
    "verify_decision",
    "signing_enabled",
    "DECISION_KEY_ENV",
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
]
