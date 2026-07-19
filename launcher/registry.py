"""Service registry — the pure-Python equivalent of docker-compose's service list.

Each Service describes how to launch one micro-service as a local subprocess:
the package's source dir (added to PYTHONPATH so `python -m <module>` resolves
without an editable install), the module to run, which profile it belongs to,
and which optional infrastructure it needs (qdrant / ollama). NATS is required
by every service and is managed separately by the launcher.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

# Project root = parent of this `launcher/` package.
ROOT = Path(__file__).resolve().parent.parent
SDK_SRC = ROOT / "sdk" / "src"


class DependencyGraphError(ValueError):
    """Raised when ``Service.deps`` references a missing name or forms a cycle."""


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
    # Names of services that must be ready before this one starts.
    deps: tuple[str, ...] = ()
    # Seconds the process must stay alive before it is considered "ready".
    readiness_timeout: float = 3.0

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
        name="kernel-watchdog",
        module="launcher.watchdog",
        src=".",
        profile="core",
        deps=("kernel",),
        readiness_timeout=2.0,
        note="Kernel-loss watchdog — SAFE_HALT if kernel heartbeat stops (E1.9.3)",
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
        deps=("kernel", "safety-supervisor"),
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
        deps=("kernel", "safety-supervisor", "beliefs"),
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
        deps=("neuromorphic",),
        note="Multi-sensory learning + task coordination (requires Qdrant)",
    ),
    Service(
        name="cognitive-bridge",
        module="neuromorphic.cognitive_bridge",
        src="neuromorphic/src",
        profile="full",
        needs_ollama=True,
        deps=("neuromorphic",),
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
        raise ValueError(f"Unknown profile {profile!r}. Choose from: {', '.join(PROFILES)}")
    return [s for s in SERVICES if s.profile in wanted]


def get_service(name: str) -> Service | None:
    for s in SERVICES:
        if s.name == name:
            return s
    return None


def validate_dependency_graph(services: Sequence[Service] | None = None) -> None:
    """Validate the startup-order dependency graph declared on ``Service.deps``.

    Checks that:
    1. Every name in ``deps`` refers to a service present in *services*.
    2. The directed graph has no cycles (which would deadlock readiness waits).

    Args:
        services: Graph to validate. Defaults to the module-level ``SERVICES``
            registry. Pass a custom list in tests to exercise failure modes.

    Raises:
        DependencyGraphError: On unknown dependency names or a cycle.
    """
    graph = list(SERVICES if services is None else services)
    by_name = {svc.name: svc for svc in graph}

    missing: list[tuple[str, str]] = []
    for svc in graph:
        for dep in svc.deps:
            if dep not in by_name:
                missing.append((svc.name, dep))
    if missing:
        details = ", ".join(f"{svc!r} → {dep!r}" for svc, dep in missing)
        raise DependencyGraphError(
            f"Unknown service dependency(ies): {details}. "
            "Every name in Service.deps must match a registered Service.name."
        )

    # DFS with colors: white=unvisited, gray=on stack, black=done.
    white, gray, black = 0, 1, 2
    color = {name: white for name in by_name}
    stack: list[str] = []

    def visit(name: str) -> None:
        color[name] = gray
        stack.append(name)
        for dep in by_name[name].deps:
            state = color[dep]
            if state == gray:
                cycle_start = stack.index(dep)
                cycle = stack[cycle_start:] + [dep]
                raise DependencyGraphError(f"Cyclic service dependency: {' → '.join(cycle)}")
            if state == white:
                visit(dep)
        stack.pop()
        color[name] = black

    for name in by_name:
        if color[name] == white:
            visit(name)


# Fail fast on a typo'd or cyclic deps declaration in the static registry.
validate_dependency_graph(SERVICES)
