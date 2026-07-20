"""
Human Override System - Allows verified humans to override operational parameters.

Provides multi-modal verification (camera, microphone) to ensure only humans
can issue overrides, not computer programs.
"""

from overrides.processor import OverrideProcessor
from overrides.service import OverrideService
from overrides.verifier import HumanVerifier

__version__ = "0.1.0"

__all__ = [
    "OverrideService",
    "HumanVerifier",
    "OverrideProcessor",
]
