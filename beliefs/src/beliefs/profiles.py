"""Body Profile Loader — loads YAML profiles and converts to belief norms.

Body profiles define the safety envelope for a specific robot configuration.
They are loaded at boot and injected into the BeliefGraph as norms, and into
the Kernel as capability markers.

Hierarchy (each layer can only RESTRICT, never expand):
  1. Constitutional VALUES (immutable, hardcoded in graph.py)
  2. Body Profile (set at manufacture/deployment, this module)
  3. Capability Markers (set by operator via cloud, runtime)
  4. Behavioral Norms (soft constraints, learned + seeded)
  5. Brain Decisions (emergent from SNN, filtered by all above)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

# Directory containing profile YAML files
_PROFILES_DIR = Path(__file__).resolve().parent.parent.parent / "profiles"


@dataclass
class MotorLimits:
    """Per-channel motor limits from a body profile."""
    max_intensity: float = 1.0
    max_acceleration: float = 1.0
    precision_mode: bool = False

    def clamp(self, intensity: float) -> float:
        """Clamp intensity to profile maximum."""
        return min(intensity, self.max_intensity)


@dataclass
class ProfileNorm:
    """A safety norm defined by a body profile."""
    id: str
    content: str
    risk_boost: float = 0.2
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmergencyBehavior:
    """Behavior to execute on a specific emergency condition."""
    trigger: str       # e.g., "fall_detected", "force_exceeded"
    response: str      # e.g., "lock_all_joints", "retract_5mm"


@dataclass
class BodyProfile:
    """A validated body profile loaded from YAML.

    Defines the physical and safety envelope for a specific robot type.
    """
    name: str
    description: str = ""
    version: int = 1
    regulatory: list[str] = field(default_factory=list)

    # Per-channel motor limits
    motor_limits: dict[str, MotorLimits] = field(default_factory=dict)

    # Safety norms (injected into BeliefGraph)
    norms: list[ProfileNorm] = field(default_factory=list)

    # Capability markers (boolean/parametric flags)
    capabilities: dict[str, Any] = field(default_factory=dict)

    # Required sensors (validated on startup)
    required_sensors: list[str] = field(default_factory=list)

    # Emergency behaviors
    emergency_behaviors: list[EmergencyBehavior] = field(default_factory=list)

    # Comms-lost behavior
    comms_lost_behavior: str = "halt_and_hold"

    # Autonomy settings
    autonomy_level: str = "human_on_loop"
    max_autonomous_duration_min: int = 60

    def get_motor_limit(self, channel: str) -> MotorLimits:
        """Get motor limits for a channel, defaulting to global limits."""
        return self.motor_limits.get(channel, MotorLimits())

    def is_channel_allowed(self, channel: str) -> bool:
        """Check if a motor channel is allowed by capabilities."""
        # Map channels to capability keys
        channel_cap_map = {
            "locomotion": "can_walk",
            "manipulation": "can_manipulate",
            "head": "can_move_head",
            "speech": "can_speak",
            "expression": "can_express",
            "cognitive": "can_query_llm",
        }
        cap_key = channel_cap_map.get(channel)
        if cap_key is None:
            return True  # Unknown channels default to allowed
        return bool(self.capabilities.get(cap_key, True))

    def validate(self) -> list[str]:
        """Validate profile for consistency. Returns list of errors."""
        errors: list[str] = []

        if not self.name:
            errors.append("Profile name is required")

        # Must define at least one safety norm (empty = no profile constraints)
        if not self.norms:
            errors.append("Profile must define at least one safety norm")

        # Motor limits must be in [0, 1]
        for channel, limits in self.motor_limits.items():
            if not 0.0 <= limits.max_intensity <= 1.0:
                errors.append(
                    f"Motor limit for '{channel}' must be in [0.0, 1.0], "
                    f"got {limits.max_intensity}"
                )

        # Norms must have IDs
        for norm in self.norms:
            if not norm.id:
                errors.append("All norms must have an id")
            if not 0.0 <= norm.risk_boost <= 1.0:
                errors.append(
                    f"Norm '{norm.id}' risk_boost must be in [0.0, 1.0], "
                    f"got {norm.risk_boost}"
                )

        return errors


def load_profile(name: str, profiles_dir: Path | None = None) -> BodyProfile:
    """Load a body profile from YAML.

    Args:
        name: Profile name (e.g., "construction_heavy"). Maps to
              ``{profiles_dir}/{name}.yaml``.
        profiles_dir: Override for the profiles directory.

    Returns:
        Validated BodyProfile instance.

    Raises:
        FileNotFoundError: If profile YAML doesn't exist.
        ValueError: If profile fails validation.
    """
    base_dir = (profiles_dir or _PROFILES_DIR).resolve()
    profile_path = (base_dir / f"{name}.yaml").resolve()

    # Guard against path traversal (e.g. name="../../etc/passwd")
    if not str(profile_path).startswith(str(base_dir)):
        raise ValueError(f"Invalid profile name: '{name}' escapes profiles directory")

    if not profile_path.exists():
        raise FileNotFoundError(f"Body profile not found: {profile_path}")

    with open(profile_path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Profile {name} must be a YAML mapping")

    profile = _parse_profile(name, raw)

    errors = profile.validate()
    if errors:
        raise ValueError(
            f"Profile '{name}' validation failed:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )

    logger.info(
        f"Loaded body profile '{name}' v{profile.version}: "
        f"{len(profile.motor_limits)} motor limits, "
        f"{len(profile.norms)} norms, "
        f"{len(profile.required_sensors)} required sensors"
    )
    return profile


def load_profile_from_env(profiles_dir: Path | None = None) -> Optional[BodyProfile]:
    """Load body profile from BODY_PROFILE env var. Returns None if not set."""
    name = os.environ.get("BODY_PROFILE")
    if not name:
        return None
    return load_profile(name, profiles_dir)


def apply_runtime_restrictions(
    base: BodyProfile,
    overrides: dict[str, Any],
) -> BodyProfile:
    """Apply runtime capability restrictions on top of a body profile.

    Enforces the marker hierarchy: runtime overrides can only RESTRICT,
    never EXPAND beyond what the base profile allows.

    Args:
        base: The loaded body profile (layer 2).
        overrides: Runtime restrictions from cloud/operator (layer 3).
            Supports keys: "motor_limits", "capabilities".

    Returns:
        A new BodyProfile with restrictions applied.

    Raises:
        ValueError: If an override attempts to expand beyond the profile.
    """
    import copy
    restricted = copy.deepcopy(base)

    # Motor-limit overrides — can only reduce max_intensity
    for channel, new_limits in overrides.get("motor_limits", {}).items():
        if not isinstance(new_limits, dict):
            continue
        base_limit = base.get_motor_limit(channel)
        new_max = float(new_limits.get("max_intensity", base_limit.max_intensity))
        if new_max > base_limit.max_intensity:
            raise ValueError(
                f"Runtime override for '{channel}' max_intensity={new_max} "
                f"exceeds profile limit {base_limit.max_intensity} — "
                f"overrides can only restrict, not expand"
            )
        new_accel = float(
            new_limits.get("max_acceleration", base_limit.max_acceleration)
        )
        if new_accel > base_limit.max_acceleration:
            raise ValueError(
                f"Runtime override for '{channel}' max_acceleration={new_accel} "
                f"exceeds profile limit {base_limit.max_acceleration} — "
                f"overrides can only restrict, not expand"
            )
        restricted.motor_limits[channel] = MotorLimits(
            max_intensity=new_max,
            max_acceleration=new_accel,
            precision_mode=new_limits.get("precision_mode", base_limit.precision_mode),
        )

    # Capability overrides — can only set True→False, not False→True
    for cap_key, new_val in overrides.get("capabilities", {}).items():
        base_val = base.capabilities.get(cap_key)

        # Reject new capability keys not defined in the base profile
        if base_val is None:
            raise ValueError(
                f"Runtime override introduces new capability '{cap_key}' "
                f"not defined in profile — overrides can only restrict "
                f"existing capabilities"
            )

        if isinstance(base_val, bool) and isinstance(new_val, bool):
            if new_val and not base_val:
                raise ValueError(
                    f"Runtime override for capability '{cap_key}' "
                    f"attempts to enable a disabled capability — "
                    f"overrides can only restrict, not expand"
                )
        # Numeric capabilities — can only reduce
        if isinstance(base_val, (int, float)) and isinstance(new_val, (int, float)):
            if new_val > base_val:
                raise ValueError(
                    f"Runtime override for '{cap_key}'={new_val} exceeds "
                    f"profile value {base_val} — overrides can only restrict"
                )
        restricted.capabilities[cap_key] = new_val

    logger.info(
        f"Applied runtime restrictions to profile '{base.name}': "
        f"{len(overrides.get('motor_limits', {}))} motor, "
        f"{len(overrides.get('capabilities', {}))} capability overrides"
    )
    return restricted


def _parse_profile(name: str, raw: dict[str, Any]) -> BodyProfile:
    """Parse raw YAML dict into a BodyProfile."""
    # Parse motor limits
    motor_limits: dict[str, MotorLimits] = {}
    for channel, limits_raw in raw.get("motor_limits", {}).items():
        if isinstance(limits_raw, dict):
            motor_limits[channel] = MotorLimits(
                max_intensity=float(limits_raw.get("max_intensity", 1.0)),
                max_acceleration=float(limits_raw.get("max_acceleration", 1.0)),
                precision_mode=bool(limits_raw.get("precision_mode", False)),
            )

    # Parse norms — must be a list of dicts, not a single dict
    norms: list[ProfileNorm] = []
    raw_norms = raw.get("norms", [])
    if isinstance(raw_norms, dict):
        raise ValueError(
            f"Profile '{name}': 'norms' must be a list, got a single dict. "
            f"Use YAML list syntax (- id: ...)"
        )
    for norm_raw in raw_norms:
        if isinstance(norm_raw, dict):
            norms.append(ProfileNorm(
                id=norm_raw.get("id", ""),
                content=norm_raw.get("content", ""),
                risk_boost=float(norm_raw.get("risk_boost", 0.2)),
                metadata=norm_raw.get("metadata", {}),
            ))

    # Parse emergency behaviors
    emergency: list[EmergencyBehavior] = []
    for trigger, response in raw.get("emergency_behaviors", {}).items():
        emergency.append(EmergencyBehavior(trigger=trigger, response=str(response)))

    return BodyProfile(
        name=name,
        description=raw.get("description", ""),
        version=int(raw.get("version", 1)),
        regulatory=raw.get("regulatory", []),
        motor_limits=motor_limits,
        norms=norms,
        capabilities=raw.get("capabilities", {}),
        required_sensors=raw.get("required_sensors", []),
        emergency_behaviors=emergency,
        comms_lost_behavior=raw.get("comms_lost_behavior", "halt_and_hold"),
        autonomy_level=raw.get("autonomy_level", "human_on_loop"),
        max_autonomous_duration_min=int(raw.get("max_autonomous_duration_min", 60)),
    )
