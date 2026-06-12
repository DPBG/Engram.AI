"""
External API Service - Manages external knowledge queries with safety.

Provides:
- Claude API integration
- OpenAI API integration (optional)
- Kernel validation before external queries
- Conflict detection between local and external knowledge
"""

from external_api.manager import ExternalAPIManager
from external_api.service import ExternalAPIService

__version__ = "0.1.0"

__all__ = [
    "ExternalAPIService",
    "ExternalAPIManager",
]
