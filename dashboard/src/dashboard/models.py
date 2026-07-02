"""Pydantic request models for the dashboard API."""

from typing import Any, Optional

from pydantic import BaseModel


class ChatMessage(BaseModel):
    message: str
    context: Optional[str] = None


class ObservationPayload(BaseModel):
    """Inject a sensory observation directly into the brain via NATS."""
    provenance: str  # e.g. "observation.text", "sensor.image"
    data: Any  # text string, or list of floats for image features
