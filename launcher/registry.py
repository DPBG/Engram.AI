"""Service registry — the pure-Python equivalent of docker-compose's service list.

Each Service describes how to launch one micro-service as a local subprocess:
the package's source dir (added to PYTHONPATH so `python -m <module>` resolves
without an editable install), the module to run, which profile it belongs to,
and which optional infrastructure it needs (qdrant / ollama). NATS is required
by every service and is managed separately by the launcher.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Project root = parent of this `launcher/` package.
ROOT = Path(__file__).resolve().parent.parent
SDK_SRC = ROOT / "sdk" / "src"


@dataclass(frozen=True)
class Service:
    """One launchable micro-service."""

    name: str
    module: str  # run as: python -m <module>
    src: str  # path (relative to ROOT) added to PYTHONPATH for imports
    profile: str  # "core" | "full" | "extra"
    needs_qdrant: bool = False
    needs_ollama: bool = False
    # Per-service environment overrides merged on top of the shared base env.
    env: dict = field(default_factory=dict)
    # Extra CLI args appended after `python -m <module>` (e.g. gateway flags).
    args: tuple = ()
    # One-line description shown by `--list`.
    note: str = ""

    @property
    def src_path(self) -> Path:
        return ROOT / self.src

    def pythonpath(self) -> str:
        """PYTHONPATH entries: the service's own src plus the shared SDK src."""
        import os

        parts = [str(self.src_path), str(SDK_SRC)]
        return os.pathsep.join(parts)


# Conservative, laptop-friendly brain size (~50K neurons). Override via env.
_NEURO_SMALL = {
    "NEURO_BRAINSTEM_N": "2000",
    "NEURO_REFLEX_N": "1500",
    "NEURO_SENSORY_N": "12000",
    "NEURO_MOTOR_N": "6000",
    "NEURO_CEREBELLUM_N": "6000",
    "NEURO_ASSOCIATION_N": "12000",
    "NEURO_PREDICTIVE_N": "6000",
    "NEURO_WORKING_MEM_N": "2000",
    "NEURO_FEATURE_N": "5000",
    "NEURO_CONCEPT_N": "1500",
    "NEURO_META_N": "1000",
    "NEURO_COGNITIVE_ENABLED": "1",
    "NEURO_EXPRESSION_END": "0.85",
}


# Order matters: governance (kernel, safety) first, then producers, then UI.
SERVICES: list[Service] = [
    Service(
        name="kernel",
        module="kernel.service",
        src="kernel/src",
        profile="core",
        note="Moral kernel - approves/denies/transforms action proposals",
    ),
    Service(
        name="safety-supervisor",
        module="safety_supervisor.service",
        src="safety-supervisor/src",
        profile="core",
        note="Risk analysis and safety supervision",
    ),
    Service(
        name="beliefs",
        module="beliefs.service",
        src="beliefs/src",
        profile="core",
        note="Belief graph (SQLite only)",
    ),
    Service(
        name="planner",
        module="planner.service",
        src="planner/src",
        profile="core",
        note="Turns observations into action proposals",
    ),
    Service(
        name="external-api",
        module="external_api.service",
        src="external-api/src",
        profile="core",
        note="External LLM bridge (runs without API keys, degrades gracefully)",
    ),
    Service(
        name="neuromorphic",
        module="neuromorphic.service",
        src="neuromorphic/src",
        profile="core",
        env={"SQLITE_PATH_BASENAME": "neuromorphic.db", **_NEURO_SMALL},
        note="The spiking-neural-network brain (NumPy/SciPy)",
    ),
    Service(
        name="dashboard",
        module="dashboard.api",
        src="dashboard/src",
        profile="core",
        note="Web UI on http://localhost:8080",
    ),
    # --- full profile: needs Qdrant and/or Ollama ---
    Service(
        name="memory",
        module="memory.service",
        src="memory/src",
        profile="full",
        needs_qdrant=True,
        note="Episodic memory (requires Qdrant)",
    ),
    Service(
        name="cache",
        module="cache.service",
        src="cache/src",
        profile="full",
        needs_qdrant=True,
        needs_ollama=True,
        note="LLM response cache (requires Qdrant + Ollama)",
    ),
    Service(
        name="coordinator",
        module="coordinator.service",
        src="coordinator/src",
        profile="full",
        needs_qdrant=True,
        note="Multi-sensory learning + task coordination (requires Qdrant)",
    ),
    Service(
        name="cognitive-bridge",
        module="neuromorphic.cognitive_bridge",
        src="neuromorphic/src",
        profile="full",
        needs_ollama=True,
        note="Brain<->Ollama bridge (requires Ollama)",
    ),
    # --- extra profile: opt-in only (hardware / Docker / generic) ---
    Service(
        name="sensory-gateway",
        module="gateway",
        src="sensory-gateway",
        profile="extra",
        args=("--no-camera", "--no-mic", "--video-loop"),
        note="Streams a looping video into the brain (needs opencv; set --video)",
    ),
    Service(
        name="overrides",
        module="overrides.service",
        src="overrides/src",
        profile="extra",
        note="Human override via camera/mic (needs opencv + pyaudio)",
    ),
    Service(
        name="sdk-runtime",
        module="activelearning.runtime",
        src="sdk/src",
        profile="extra",
        note="Generic SDK runtime holder (rarely needed standalone)",
    ),
    # meta-programmer is intentionally omitted: it requires the Docker socket
    # to spawn sandbox containers and cannot run in a pure-Python setup.
]

PROFILES = {
    "core": ["core"],
    "full": ["core", "full"],
    "all": ["core", "full", "extra"],
}


def services_for_profile(profile: str) -> list[Service]:
    wanted = PROFILES.get(profile)
    if wanted is None:
        raise ValueError(
            f"Unknown profile {profile!r}. Choose from: {', '.join(PROFILES)}"
        )
    return [s for s in SERVICES if s.profile in wanted]


def get_service(name: str) -> Service | None:
    for s in SERVICES:
        if s.name == name:
            return s
    return None
