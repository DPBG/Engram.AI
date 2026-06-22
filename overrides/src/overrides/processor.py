"""
Override Processor - Parses and applies human overrides.

Parses natural language override prompts and applies them to system parameters
after Kernel validation.
"""

import logging
import re
from typing import Any
import json

from activelearning.nats_client import EventBus

logger = logging.getLogger(__name__)


class OverrideProcessor:
    """
    Processes human override prompts and applies them to the system.

    Distinguishes between:
    - Operational parameters (can be overridden)
    - Safety rules (cannot be overridden, must go through Kernel)
    """

    # Protected paths that cannot be overridden without Kernel approval
    PROTECTED_PATHS = [
        "/kernel/*",
        "/safety-supervisor/*",
        "/meta-programmer/orchestrator.py",
    ]

    # Operational parameters that can be overridden
    OPERATIONAL_PARAMS = {
        "planner.mode": ["EXECUTION", "LEARNING", "EXPLORATION", "SAFE_HALT"],
        "planner.priority_threshold": (0.0, 1.0),
        "memory.retention_days": (1, 365),
        "cache.enabled": [True, False],
        "autopilot.enabled": [True, False],
    }

    def __init__(self, event_bus: EventBus, db: Any):
        self.event_bus = event_bus
        self.db = db

    async def parse_override(self, prompt: str) -> dict:
        """
        Parse natural language override prompt.

        Examples:
        - "Switch to learning mode" → planner.mode = LEARNING
        - "Disable autopilot" → autopilot.enabled = False
        - "Set priority threshold to 0.7" → planner.priority_threshold = 0.7

        Returns:
            dict with keys:
                - success: bool
                - parameter: str (parameter path)
                - value: any (new value)
                - requires_kernel: bool (needs Kernel approval)
                - error: optional error message
        """
        try:
            prompt_lower = prompt.lower().strip()

            # Pattern: "switch to X mode"
            mode_match = re.search(r"(?:switch to|change to|set mode to|enter)\s+(\w+)\s+mode", prompt_lower)
            if mode_match:
                mode = mode_match.group(1).upper()
                valid_modes = self.OPERATIONAL_PARAMS["planner.mode"]
                if mode in valid_modes:
                    return {
                        "success": True,
                        "parameter": "planner.mode",
                        "value": mode,
                        "requires_kernel": False,
                        "error": None,
                    }
                else:
                    return {
                        "success": False,
                        "parameter": "",
                        "value": None,
                        "requires_kernel": False,
                        "error": f"Invalid mode: {mode}. Valid: {valid_modes}",
                    }

            # Pattern: "enable/disable X"
            enable_match = re.search(r"(enable|disable|turn on|turn off)\s+(\w+)", prompt_lower)
            if enable_match:
                action = enable_match.group(1)
                param_name = enable_match.group(2)
                value = action in ["enable", "turn on"]

                # Map param name to full parameter path
                param_map = {
                    "cache": "cache.enabled",
                    "autopilot": "autopilot.enabled",
                }

                if param_name in param_map:
                    param_path = param_map[param_name]
                    return {
                        "success": True,
                        "parameter": param_path,
                        "value": value,
                        "requires_kernel": False,
                        "error": None,
                    }

            # Pattern: "set X to Y"
            set_match = re.search(r"set\s+(\w+(?:\.\w+)*)\s+to\s+([\d.]+|true|false)", prompt_lower)
            if set_match:
                param_path = set_match.group(1)
                value_str = set_match.group(2)

                # Parse value
                if value_str in ["true", "false"]:
                    value = value_str == "true"
                else:
                    try:
                        value = float(value_str)
                    except:
                        value = value_str

                # Check if this is an operational parameter
                if param_path in self.OPERATIONAL_PARAMS:
                    # Validate value
                    valid_values = self.OPERATIONAL_PARAMS[param_path]
                    if isinstance(valid_values, list):
                        if value not in valid_values:
                            return {
                                "success": False,
                                "parameter": "",
                                "value": None,
                                "requires_kernel": False,
                                "error": f"Invalid value for {param_path}. Valid: {valid_values}",
                            }
                    elif isinstance(valid_values, tuple):
                        min_val, max_val = valid_values
                        if not (min_val <= value <= max_val):
                            return {
                                "success": False,
                                "parameter": "",
                                "value": None,
                                "requires_kernel": False,
                                "error": f"Value must be between {min_val} and {max_val}",
                            }

                    return {
                        "success": True,
                        "parameter": param_path,
                        "value": value,
                        "requires_kernel": False,
                        "error": None,
                    }

            # If we couldn't parse it, return error
            return {
                "success": False,
                "parameter": "",
                "value": None,
                "requires_kernel": False,
                "error": "Could not parse override prompt. Examples: 'Switch to learning mode', 'Disable autopilot', 'Set priority threshold to 0.7'",
            }

        except Exception as e:
            logger.error(f"Error parsing override: {e}")
            return {
                "success": False,
                "parameter": "",
                "value": None,
                "requires_kernel": False,
                "error": str(e),
            }

    async def apply_override(
        self,
        trace_id: str,
        parameter: str,
        value: Any,
        verified_by: str,
    ) -> dict:
        """
        Apply an override to the system.

        Args:
            trace_id: Unique identifier
            parameter: Parameter path (e.g., "planner.mode")
            value: New value
            verified_by: Verification method used

        Returns:
            dict with success status
        """
        try:
            logger.info(f"Applying override: {parameter} = {value}")

            # Apply based on parameter type
            if parameter == "planner.mode":
                await self._apply_planner_mode(value)
            elif parameter == "cache.enabled":
                await self._apply_cache_setting(value)
            elif parameter == "autopilot.enabled":
                await self._apply_autopilot_setting(value)
            elif parameter == "planner.priority_threshold":
                await self._apply_priority_threshold(value)
            else:
                return {
                    "success": False,
                    "error": f"Unknown parameter: {parameter}",
                }

            # Store in database
            import uuid
            await self.db.execute(
                """
                INSERT INTO human_overrides
                (id, trace_id, parameter, value, verified_by, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    trace_id,
                    parameter,
                    json.dumps(value),
                    verified_by,
                    int(__import__('time').time() * 1000),
                ),
            )
            await self.db.commit()

            # Publish override applied event
            await self.event_bus.publish(
                f"override.applied.{trace_id}",
                {
                    "trace_id": trace_id,
                    "parameter": parameter,
                    "value": value,
                },
            )

            logger.info(f"Override applied successfully: {parameter} = {value}")

            return {
                "success": True,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Error applying override: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    async def _apply_planner_mode(self, mode: str) -> None:
        """Apply planner mode change."""
        await self.event_bus.publish("planner.mode", {"mode": mode})

    async def _apply_cache_setting(self, enabled: bool) -> None:
        """Apply cache enable/disable."""
        await self.event_bus.publish("cache.setting", {"enabled": enabled})

    async def _apply_autopilot_setting(self, enabled: bool) -> None:
        """Apply autopilot enable/disable."""
        await self.event_bus.publish("autopilot.setting", {"enabled": enabled})

    async def _apply_priority_threshold(self, threshold: float) -> None:
        """Apply priority threshold change."""
        await self.event_bus.publish(
            "planner.priority_threshold",
            {"threshold": threshold},
        )
