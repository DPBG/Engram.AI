"""
Meta-Programmer - Self-evolving code generation system.

The Meta-Programmer detects knowledge gaps and generates code using
local LLMs (Ollama), tests in sandboxes, and deploys after Kernel approval.
"""

from meta_programmer.agents import MetaProgrammerTeam
from meta_programmer.sandbox_manager import SandboxManager
from meta_programmer.service import MetaProgrammerService
from meta_programmer.staging import StagingManager

__version__ = "0.1.0"

__all__ = [
    "MetaProgrammerService",
    "MetaProgrammerTeam",
    "SandboxManager",
    "StagingManager",
]
