"""
Engram Dashboard — API.

Dashboard for the Engram neuromorphic brain. The LLM chat is a
communication interface for the spiking neural network, not the
intelligence itself. The brain learns through STDP on sensory
experience; the LLM provides a natural-language window into its state.

FastAPI backend with WebSocket, system detection, skill registry,
knowledge base, conversational AI, and self-improvement loop.
"""

import asyncio
import base64
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional

import aiohttp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

from dashboard.auth import install_auth_middleware, authorize_websocket

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════════════

MAX_MESSAGES = 500
message_buffer: deque = deque(maxlen=MAX_MESSAGES)
active_connections: list[WebSocket] = []

chat_history: list[dict] = []
MAX_CHAT_HISTORY = 200

_system_info: dict[str, Any] = {}
_system_metrics_cache: dict[str, Any] = {}
_last_metrics_update: float = 0

insights_log: deque = deque(maxlen=100)

# Neuromorphic cognitive core state (updated via NATS)
_neuro_metrics: dict[str, Any] = {}

# Sensory gateway state (updated via NATS)
_gateway_status: dict[str, Any] = {}

# Video training sessions (updated via NATS from gateway)
MAX_VIDEO_SESSIONS = 100
_video_sessions: dict[str, dict] = {}  # session_id -> status dict

# Safety / watchdog state (updated via NATS)
_watchdog_status: dict[str, Any] = {}
_deny_escalations: deque = deque(maxlen=50)


# ═══════════════════════════════════════════════════════════════════════
# SKILL REGISTRY — abstracted capabilities
# ═══════════════════════════════════════════════════════════════════════

class SkillRegistry:
    """
    Modular skill registry.

    Each skill is an abstracted capability the system can invoke.
    Skills track their own execution history and health.
    "These skills are abstracted away using an API call."
    """

    def __init__(self):
        self._skills: dict[str, dict] = {}
        self._execution_log: deque = deque(maxlen=500)
        self._init_core_skills()

    def _init_core_skills(self):
        """Register the core built-in skills."""
        core = [
            {
                "id": "env.detect",
                "name": "Environment Detection",
                "category": "perception",
                "icon": "🔍",
                "description": "Server environment detection — OS, hardware, services, APIs",
                "status": "active",
                "calls": 0,
                "errors": 0,
                "last_called": None,
                "avg_ms": 0,
            },
            {
                "id": "env.monitor",
                "name": "Resource Monitor",
                "category": "perception",
                "icon": "📊",
                "description": "Real-time CPU, memory, disk, network monitoring",
                "status": "active",
                "calls": 0,
                "errors": 0,
                "last_called": None,
                "avg_ms": 0,
            },
            {
                "id": "env.docker",
                "name": "Container Orchestration",
                "category": "perception",
                "icon": "🐳",
                "description": "Docker container metrics, status, and lifecycle awareness",
                "status": "active",
                "calls": 0,
                "errors": 0,
                "last_called": None,
                "avg_ms": 0,
            },
            {
                "id": "brain.chat",
                "name": "Brain Communication",
                "category": "cognition",
                "icon": "💬",
                "description": "LLM interface to the Engram neuromorphic brain (teleoperation channel)",
                "status": "active",
                "calls": 0,
                "errors": 0,
                "last_called": None,
                "avg_ms": 0,
            },
            {
                "id": "brain.self_monitor",
                "name": "Self-Improvement Loop",
                "category": "cognition",
                "icon": "🔄",
                "description": "Periodic health checks, anomaly detection, optimization suggestions",
                "status": "active",
                "calls": 0,
                "errors": 0,
                "last_called": None,
                "avg_ms": 0,
            },
            {
                "id": "brain.knowledge",
                "name": "Knowledge Base",
                "category": "memory",
                "icon": "🧠",
                "description": "Stores learnings from interactions, observations, and deployments",
                "status": "active",
                "calls": 0,
                "errors": 0,
                "last_called": None,
                "avg_ms": 0,
            },
            {
                "id": "bus.nats",
                "name": "NATS Message Bus",
                "category": "communication",
                "icon": "📡",
                "description": "Inter-service messaging and event monitoring",
                "status": "active",
                "calls": 0,
                "errors": 0,
                "last_called": None,
                "avg_ms": 0,
            },
            {
                "id": "bus.websocket",
                "name": "WebSocket Stream",
                "category": "communication",
                "icon": "⚡",
                "description": "Real-time bidirectional client communication",
                "status": "active",
                "calls": 0,
                "errors": 0,
                "last_called": None,
                "avg_ms": 0,
            },
        ]
        for skill in core:
            self._skills[skill["id"]] = skill

    def record_call(self, skill_id: str, duration_ms: float, success: bool = True):
        """Record a skill invocation."""
        skill = self._skills.get(skill_id)
        if not skill:
            return
        skill["calls"] += 1
        if not success:
            skill["errors"] += 1
        skill["last_called"] = datetime.now(timezone.utc).isoformat()
        # Running average
        old_avg = skill["avg_ms"]
        n = skill["calls"]
        skill["avg_ms"] = round(old_avg + (duration_ms - old_avg) / n, 1)

        self._execution_log.append({
            "skill_id": skill_id,
            "timestamp": skill["last_called"],
            "duration_ms": round(duration_ms, 1),
            "success": success,
        })

    def get_all(self) -> list[dict]:
        return list(self._skills.values())

    def get_by_category(self) -> dict[str, list[dict]]:
        cats: dict[str, list[dict]] = {}
        for s in self._skills.values():
            cats.setdefault(s["category"], []).append(s)
        return cats

    def get_execution_log(self, limit: int = 50) -> list[dict]:
        return list(self._execution_log)[-limit:]

    def total_calls(self) -> int:
        return sum(s["calls"] for s in self._skills.values())

    def total_errors(self) -> int:
        return sum(s["errors"] for s in self._skills.values())


# ═══════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE — Data Flywheel
# ═══════════════════════════════════════════════════════════════════════

class KnowledgeBase:
    """
    In-memory knowledge base tracking the data flywheel.

    Engram learns from 4 sources:
    1. Simulation (synthetic experiences)
    2. Internet/external data (observation)
    3. Teleoperation (human guidance — chat)
    4. Real-world deployments (self-generated data)
    """

    def __init__(self):
        self._entries: deque = deque(maxlen=1000)
        self._source_counts = {
            "teleoperation": 0,   # Chat interactions
            "observation": 0,     # System observations, NATS messages
            "deployment": 0,      # Self-generated from monitoring
            "simulation": 0,      # Synthetic / test data
        }
        self._total_interactions = 0

    def learn(self, source: str, category: str, content: str, metadata: dict = None):
        """Record a learning event."""
        entry = {
            "id": str(uuid.uuid4())[:8],
            "source": source,
            "category": category,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._entries.append(entry)
        if source in self._source_counts:
            self._source_counts[source] += 1
        self._total_interactions += 1

    def get_flywheel_stats(self) -> dict:
        """Get data flywheel statistics."""
        return {
            "total_knowledge_entries": len(self._entries),
            "total_interactions": self._total_interactions,
            "sources": dict(self._source_counts),
            "recent_entries": list(self._entries)[-10:],
            "growth_rate": self._calculate_growth_rate(),
        }

    def _calculate_growth_rate(self) -> float:
        """Entries per hour over last hour."""
        if not self._entries:
            return 0.0
        now = time.time()
        one_hour_ago = now - 3600
        recent = sum(
            1 for e in self._entries
            if datetime.fromisoformat(e["timestamp"]).timestamp() > one_hour_ago
        )
        return round(recent, 1)

    def get_entries(self, limit: int = 50, source: str = None) -> list[dict]:
        entries = list(self._entries)
        if source:
            entries = [e for e in entries if e["source"] == source]
        return entries[-limit:]


# ═══════════════════════════════════════════════════════════════════════
# DEEP SYSTEM DETECTION
# ═══════════════════════════════════════════════════════════════════════

def detect_system_info() -> dict[str, Any]:
    """
    Deep system detection — not just "what OS" but full awareness of
    services, APIs, capabilities, and the environment.
    """
    global _system_info
    info: dict[str, Any] = {}

    # ─── OS & Architecture ────────────────────────────────────────
    info["os"] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "hostname": platform.node(),
    }

    # ─── CPU ──────────────────────────────────────────────────────
    try:
        cpu_count = os.cpu_count() or 1
        info["cpu"] = {"cores": cpu_count, "architecture": platform.machine()}
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if "model name" in line:
                            info["cpu"]["model"] = line.split(":")[1].strip()
                            break
            except Exception:
                pass
        elif platform.system() == "Darwin":
            try:
                r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                   capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    info["cpu"]["model"] = r.stdout.strip()
            except Exception:
                pass
    except Exception as e:
        info["cpu"] = {"cores": 1, "error": str(e)}

    # ─── Memory ──────────────────────────────────────────────────
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                meminfo = f.read()
            total = avail = 0
            for line in meminfo.split("\n"):
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
            info["memory"] = {
                "total_gb": round(total / (1024**3), 2),
                "available_gb": round(avail / (1024**3), 2),
                "used_gb": round((total - avail) / (1024**3), 2),
                "percent_used": round(((total - avail) / total) * 100, 1) if total > 0 else 0,
            }
        else:
            info["memory"] = {"note": "Memory details available on Linux host"}
    except Exception as e:
        info["memory"] = {"error": str(e)}

    # ─── Disk ────────────────────────────────────────────────────
    try:
        disk = shutil.disk_usage("/")
        info["disk"] = {
            "total_gb": round(disk.total / (1024**3), 2),
            "used_gb": round(disk.used / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "percent_used": round((disk.used / disk.total) * 100, 1),
        }
    except Exception as e:
        info["disk"] = {"error": str(e)}

    # ─── GPU ─────────────────────────────────────────────────────
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            gpus = []
            for line in r.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpus.append({
                        "name": parts[0],
                        "memory_total_mb": int(parts[1]),
                        "memory_used_mb": int(parts[2]),
                        "utilization_percent": int(parts[3]),
                    })
            info["gpu"] = gpus
        else:
            info["gpu"] = None
    except FileNotFoundError:
        if platform.machine() == "arm64" and platform.system() == "Darwin":
            info["gpu"] = [{"name": "Apple Silicon (Metal)", "type": "integrated"}]
        else:
            info["gpu"] = None
    except Exception:
        info["gpu"] = None

    # ─── Network ─────────────────────────────────────────────────
    try:
        if platform.system() == "Linux":
            r = subprocess.run(["ip", "-j", "addr"], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                interfaces = json.loads(r.stdout)
                info["network"] = [
                    {
                        "name": iface.get("ifname", "?"),
                        "state": iface.get("operstate", "?"),
                        "addresses": [a.get("local", "") for a in iface.get("addr_info", [])],
                    }
                    for iface in interfaces
                    if iface.get("operstate") in ("UP", "UNKNOWN")
                ]
            else:
                info["network"] = []
        else:
            info["network"] = []
    except Exception:
        info["network"] = []

    # ─── Running Services (deep awareness) ───────────────────────
    info["services"] = _detect_running_services()

    # ─── Available APIs ──────────────────────────────────────────
    info["apis"] = _detect_available_apis()

    # ─── Capabilities ────────────────────────────────────────────
    info["capabilities"] = _detect_capabilities()

    _system_info = info
    return info


def _detect_running_services() -> list[dict]:
    """Detect running services / listening ports."""
    services = []
    try:
        # Check listening TCP ports
        if platform.system() == "Linux":
            r = subprocess.run(
                ["ss", "-tlnp"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                for line in r.stdout.strip().split("\n")[1:]:  # skip header
                    parts = line.split()
                    if len(parts) >= 4:
                        local = parts[3]
                        # Extract port
                        port_match = re.search(r':(\d+)$', local)
                        if port_match:
                            port = int(port_match.group(1))
                            proc = parts[-1] if len(parts) > 5 else ""
                            # Map well-known ports
                            name = _port_to_service_name(port, proc)
                            services.append({
                                "port": port,
                                "name": name,
                                "address": local,
                            })
    except Exception:
        pass

    return services


def _port_to_service_name(port: int, proc_info: str = "") -> str:
    """Map port number to known service names."""
    known = {
        4222: "NATS (client)",
        8222: "NATS (monitoring)",
        6333: "Qdrant (HTTP)",
        6334: "Qdrant (gRPC)",
        8080: "Dashboard",
        11434: "Ollama (LLM)",
        7777: "Custom Service",
        5432: "PostgreSQL",
        3306: "MySQL",
        6379: "Redis",
        9090: "Prometheus",
        3000: "Grafana",
        443: "HTTPS",
        80: "HTTP",
    }
    return known.get(port, f"port-{port}")


def _detect_available_apis() -> list[dict]:
    """Detect what APIs are reachable from this container."""
    apis = []
    checks = [
        ("NATS", os.environ.get("NATS_URL", "nats://nats:4222"), "nats"),
        ("Ollama", os.environ.get("OLLAMA_URL", "http://ollama:11434"), "llm"),
        ("Qdrant", os.environ.get("QDRANT_URL", "http://qdrant:6333"), "vector_db"),
    ]
    for name, url, api_type in checks:
        apis.append({
            "name": name,
            "url": url,
            "type": api_type,
            "configured": True,
        })
    return apis


def _detect_capabilities() -> list[str]:
    """What can this system do?"""
    caps = [
        "system_monitoring",
        "resource_tracking",
        "container_management",
        "conversational_ai",
        "nats_messaging",
        "self_improvement",
    ]
    # Check for Docker socket
    if os.path.exists("/var/run/docker.sock"):
        caps.append("docker_orchestration")
    # Check for Ollama env
    if os.environ.get("OLLAMA_URL"):
        caps.append("local_llm")
    return caps


def get_live_metrics() -> dict[str, Any]:
    """Get live resource usage metrics."""
    metrics: dict[str, Any] = {}
    try:
        if platform.system() == "Linux":
            with open("/proc/loadavg") as f:
                loadavg = f.read().split()
            metrics["load_average"] = {
                "1min": float(loadavg[0]),
                "5min": float(loadavg[1]),
                "15min": float(loadavg[2]),
            }
            with open("/proc/meminfo") as f:
                meminfo = f.read()
            mem_total = mem_avail = 0
            for line in meminfo.split("\n"):
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    mem_avail = int(line.split()[1]) * 1024
            metrics["memory"] = {
                "total_gb": round(mem_total / (1024**3), 2),
                "available_gb": round(mem_avail / (1024**3), 2),
                "used_percent": round(((mem_total - mem_avail) / mem_total) * 100, 1) if mem_total > 0 else 0,
            }
        disk = shutil.disk_usage("/")
        metrics["disk"] = {
            "used_percent": round((disk.used / disk.total) * 100, 1),
            "free_gb": round(disk.free / (1024**3), 2),
        }
        try:
            with open("/proc/uptime") as f:
                secs = float(f.read().split()[0])
            metrics["uptime"] = f"{int(secs // 3600)}h {int((secs % 3600) // 60)}m"
        except Exception:
            metrics["uptime"] = "unknown"
    except Exception as e:
        metrics["error"] = str(e)
    metrics["timestamp"] = datetime.now(timezone.utc).isoformat()
    return metrics


# ═══════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════

class ChatMessage(BaseModel):
    message: str
    context: Optional[str] = None


class ObservationPayload(BaseModel):
    """Inject a sensory observation directly into the brain via NATS."""
    provenance: str  # e.g. "observation.text", "sensor.image"
    data: Any  # text string, or list of floats for image features


# ═══════════════════════════════════════════════════════════════════════
# DASHBOARD SERVICE
# ═══════════════════════════════════════════════════════════════════════

class DashboardService:
    """
    Engram Dashboard — interface to the neuromorphic brain.

    Monitors the spiking neural network, provides a chat interface
    (LLM as communication layer), and manages the deployment
    environment.
    """

    def __init__(self):
        self.app = FastAPI(title="Engram")
        self.logger = logging.getLogger("dashboard")
        self._nats_connected = False
        self._nc = None  # NATS connection for publishing
        self._service_status: dict[str, dict] = {}
        self._ollama_url = os.environ.get("OLLAMA_URL", "http://ollama:11434")
        self._openai_url = os.environ.get("OPENAI_API_URL", "")
        self._openai_key = os.environ.get("OPENAI_API_KEY", "")
        self._llm_model = os.environ.get("LLM_MODEL", "llama3.2")
        self._self_monitor_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        self._concept_probe_results: list[dict] = []
        self._MAX_PROBE_RESULTS = 200

        # Core components
        self.skills = SkillRegistry()
        self.knowledge = KnowledgeBase()

        self._setup_routes()

    def _setup_routes(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"], allow_credentials=True,
            allow_methods=["*"], allow_headers=["*"],
        )

        # Authenticate the control plane: when ENGRAM_DASHBOARD_TOKEN is set,
        # every state-mutating request (POST/PUT/DELETE) must present it. No-op
        # in dev when the token is unset (logs a one-time warning). See auth.py.
        install_auth_middleware(self.app)

        static_dir = os.path.join(os.path.dirname(__file__), "..", "..", "static")

        # Brain visualization (standalone Three.js app)
        # In Docker: mounted at /app/brain-viz via volume
        # Local dev: relative to project root
        brain_viz_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "brain-viz")
        if not os.path.exists(brain_viz_dir):
            brain_viz_dir = "/app/brain-viz"
        if os.path.exists(brain_viz_dir):
            self.app.mount("/brain-viz", StaticFiles(directory=brain_viz_dir, html=True), name="brain-viz")

        if os.path.exists(static_dir):
            self.app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @self.app.on_event("startup")
        async def on_startup():
            t0 = time.time()
            detect_system_info()
            self.skills.record_call("env.detect", (time.time() - t0) * 1000)
            self.knowledge.learn("deployment", "system", f"Engram deployed on: {_system_info.get('os', {}).get('system', '?')} {_system_info.get('os', {}).get('machine', '?')}")
            self.knowledge.learn("deployment", "system", f"Capabilities: {', '.join(_system_info.get('capabilities', []))}")
            self._self_monitor_task = asyncio.create_task(self._self_improvement_loop())
            self._metrics_task = asyncio.create_task(self._metrics_broadcast_loop())
            asyncio.create_task(self._connect_nats())
            self.logger.info("Engram Dashboard started")

        @self.app.on_event("shutdown")
        async def on_shutdown():
            if self._self_monitor_task:
                self._self_monitor_task.cancel()
            if self._metrics_task:
                self._metrics_task.cancel()

        # ── Pages ────────────────────────────────────────────────────

        @self.app.get("/")
        async def root():
            return RedirectResponse(url="/brain-viz/demos/index.html")

        @self.app.get("/dashboard")
        async def dashboard_page():
            idx = os.path.join(static_dir, "index.html")
            if os.path.exists(idx):
                return FileResponse(idx)
            return {"status": "ok"}

        # ── API: Health ──────────────────────────────────────────────

        @self.app.get("/api/health")
        async def health():
            return {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "nats": self._nats_connected,
                "uptime_seconds": int(time.time() - _startup_time),
                "total_skill_calls": self.skills.total_calls(),
                "knowledge_entries": len(self.knowledge._entries),
            }

        # ── API: System ──────────────────────────────────────────────

        @self.app.get("/api/system")
        async def system_info():
            if not _system_info:
                detect_system_info()
            return {
                "info": _system_info,
                "live": get_live_metrics(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # ── API: Skills ──────────────────────────────────────────────

        @self.app.get("/api/skills")
        async def get_skills():
            return {
                "skills": self.skills.get_all(),
                "by_category": self.skills.get_by_category(),
                "total_calls": self.skills.total_calls(),
                "total_errors": self.skills.total_errors(),
            }

        @self.app.get("/api/skills/log")
        async def get_skill_log(limit: int = 50):
            return {"log": self.skills.get_execution_log(limit)}

        # ── API: Knowledge / Flywheel ────────────────────────────────

        @self.app.get("/api/knowledge")
        async def get_knowledge(limit: int = 50, source: str = None):
            return {
                "entries": self.knowledge.get_entries(limit, source),
                "flywheel": self.knowledge.get_flywheel_stats(),
            }

        @self.app.get("/api/flywheel")
        async def get_flywheel():
            return self.knowledge.get_flywheel_stats()

        # ── API: Docker Metrics ──────────────────────────────────────

        @self.app.get("/api/metrics")
        async def get_metrics():
            t0 = time.time()
            metrics = await self._fetch_docker_metrics()
            self.skills.record_call("env.docker", (time.time() - t0) * 1000)
            return {"metrics": metrics, "timestamp": datetime.now(timezone.utc).isoformat()}

        # ── API: Services ────────────────────────────────────────────

        @self.app.get("/api/services")
        async def get_services():
            return {"services": list(self._service_status.values())}

        # ── API: Neuromorphic ─────────────────────────────────────────

        @self.app.get("/api/neuromorphic")
        async def get_neuromorphic():
            return {"neuromorphic": _neuro_metrics, "timestamp": datetime.now(timezone.utc).isoformat()}

        # ── API: Benchmark Results ─────────────────────────────────────

        @self.app.get("/api/benchmark/latest")
        async def get_benchmark_latest():
            """Return the most recent benchmark results JSON."""
            benchmark_dir = "/data/benchmarks"
            try:
                if not os.path.isdir(benchmark_dir):
                    return {"error": "No benchmarks directory", "results": None}
                files = sorted(
                    [f for f in os.listdir(benchmark_dir) if f.endswith(".json")],
                    reverse=True,
                )
                if not files:
                    return {"error": "No benchmark results found", "results": None}
                with open(os.path.join(benchmark_dir, files[0])) as f:
                    data = json.load(f)
                return {"results": data, "filename": files[0]}
            except Exception as e:
                return {"error": str(e), "results": None}

        @self.app.get("/api/benchmark/history")
        async def get_benchmark_history():
            """Return all benchmark results for trend charts."""
            benchmark_dir = "/data/benchmarks"
            try:
                if not os.path.isdir(benchmark_dir):
                    return {"results": []}
                files = sorted(
                    [f for f in os.listdir(benchmark_dir) if f.endswith(".json")],
                )
                results = []
                for fname in files[-20:]:  # last 20
                    with open(os.path.join(benchmark_dir, fname)) as f:
                        results.append(json.load(f))
                return {"results": results}
            except Exception as e:
                return {"error": str(e), "results": []}

        # ── API: Sensory Gateway ──────────────────────────────────────

        @self.app.get("/api/gateway")
        async def get_gateway():
            return {"gateway": _gateway_status, "timestamp": datetime.now(timezone.utc).isoformat()}

        @self.app.post("/api/gateway/command")
        async def gateway_command(cmd: dict):
            if not self._nc:
                return {"error": "NATS not connected"}
            try:
                await self._nc.publish(
                    "sensory.gateway.command",
                    json.dumps(cmd).encode(),
                )
                return {"status": "sent", "command": cmd}
            except Exception as e:
                return {"error": str(e)}

        # ── API: MuJoCo Body Visualization ────────────────────────────

        @self.app.get("/api/mujoco/model")
        async def get_mujoco_model():
            """Return static geometry definitions for the MuJoCo humanoid."""
            return {"geoms": [
                {"body": "world", "type": "plane", "size": [10, 10, 0.1], "rgba": [0.8, 0.8, 0.8, 1]},
                {"body": "torso", "type": "capsule", "size": [0.1, 0.2, 0], "rgba": [0.3, 0.6, 0.9, 1]},
                {"body": "head", "type": "sphere", "size": [0.1, 0, 0], "rgba": [0.9, 0.8, 0.7, 1]},
                {"body": "r_upper_arm", "type": "capsule", "size": [0.04, 0.15, 0], "rgba": [0.3, 0.6, 0.9, 1]},
                {"body": "r_forearm", "type": "capsule", "size": [0.03, 0.12, 0], "rgba": [0.3, 0.6, 0.9, 1]},
                {"body": "l_upper_arm", "type": "capsule", "size": [0.04, 0.15, 0], "rgba": [0.3, 0.6, 0.9, 1]},
                {"body": "l_forearm", "type": "capsule", "size": [0.03, 0.12, 0], "rgba": [0.3, 0.6, 0.9, 1]},
                {"body": "r_thigh", "type": "capsule", "size": [0.05, 0.2, 0], "rgba": [0.3, 0.6, 0.9, 1]},
                {"body": "r_shin", "type": "capsule", "size": [0.04, 0.18, 0], "rgba": [0.3, 0.6, 0.9, 1]},
                {"body": "l_thigh", "type": "capsule", "size": [0.05, 0.2, 0], "rgba": [0.3, 0.6, 0.9, 1]},
                {"body": "l_shin", "type": "capsule", "size": [0.04, 0.18, 0], "rgba": [0.3, 0.6, 0.9, 1]},
            ]}

        @self.app.post("/api/mujoco/guide")
        async def mujoco_guide(body: dict):
            """Send guidance command to MuJoCo body via NATS.

            Body formats:
                {"action": "pose", "joints": {"r_hip": 30, "l_hip": -10}}
                {"action": "push", "body": "torso", "force": [0, 5, 0]}
                {"action": "reward", "channel": "locomotion", "success": true}
                {"action": "reset"}
            """
            nc = self._nc
            if nc is None:
                return {"error": "NATS not connected"}
            action = body.get("action", "")
            if action not in ("pose", "push", "reward", "reset", "teach"):
                return {"error": f"Unknown action: {action}"}
            await nc.publish(
                "motor.guidance",
                json.dumps(body).encode(),
            )
            return {"ok": True, "action": action}

        @self.app.get("/api/mujoco/joints")
        async def mujoco_joints():
            """Return joint info (names, ranges, channels) for UI sliders.

            Static fallback — 29-DOF humanoid (Optimus/G1/Atlas class).
            """
            return {"joints": [
                # Waist (locomotion)
                {"name": "waist_yaw", "channel": "locomotion", "range_deg": [-75, 75]},
                {"name": "waist_roll", "channel": "locomotion", "range_deg": [-30, 30]},
                {"name": "waist_pitch", "channel": "locomotion", "range_deg": [-30, 45]},
                # Right leg (locomotion)
                {"name": "r_hip_yaw", "channel": "locomotion", "range_deg": [-40, 40]},
                {"name": "r_hip_roll", "channel": "locomotion", "range_deg": [-25, 45]},
                {"name": "r_hip_pitch", "channel": "locomotion", "range_deg": [-30, 120]},
                {"name": "r_knee", "channel": "locomotion", "range_deg": [-145, 0]},
                {"name": "r_ankle_pitch", "channel": "locomotion", "range_deg": [-40, 50]},
                {"name": "r_ankle_roll", "channel": "locomotion", "range_deg": [-25, 25]},
                # Left leg (locomotion)
                {"name": "l_hip_yaw", "channel": "locomotion", "range_deg": [-40, 40]},
                {"name": "l_hip_roll", "channel": "locomotion", "range_deg": [-45, 25]},
                {"name": "l_hip_pitch", "channel": "locomotion", "range_deg": [-30, 120]},
                {"name": "l_knee", "channel": "locomotion", "range_deg": [-145, 0]},
                {"name": "l_ankle_pitch", "channel": "locomotion", "range_deg": [-40, 50]},
                {"name": "l_ankle_roll", "channel": "locomotion", "range_deg": [-25, 25]},
                # Right arm (manipulation)
                {"name": "r_shoulder_pitch", "channel": "manipulation", "range_deg": [-90, 180]},
                {"name": "r_shoulder_roll", "channel": "manipulation", "range_deg": [-30, 150]},
                {"name": "r_shoulder_yaw", "channel": "manipulation", "range_deg": [-90, 70]},
                {"name": "r_elbow", "channel": "manipulation", "range_deg": [0, 145]},
                {"name": "r_wrist_pitch", "channel": "manipulation", "range_deg": [-70, 70]},
                {"name": "r_wrist_yaw", "channel": "manipulation", "range_deg": [-45, 45]},
                # Left arm (manipulation)
                {"name": "l_shoulder_pitch", "channel": "manipulation", "range_deg": [-90, 180]},
                {"name": "l_shoulder_roll", "channel": "manipulation", "range_deg": [-150, 30]},
                {"name": "l_shoulder_yaw", "channel": "manipulation", "range_deg": [-70, 90]},
                {"name": "l_elbow", "channel": "manipulation", "range_deg": [0, 145]},
                {"name": "l_wrist_pitch", "channel": "manipulation", "range_deg": [-70, 70]},
                {"name": "l_wrist_yaw", "channel": "manipulation", "range_deg": [-45, 45]},
                # Neck (head)
                {"name": "neck_pitch", "channel": "head", "range_deg": [-40, 40]},
                {"name": "neck_yaw", "channel": "head", "range_deg": [-55, 55]},
            ]}

        # ── API: Video Training ──────────────────────────────────────

        @self.app.get("/api/video/sessions")
        async def get_video_sessions():
            return {
                "sessions": list(_video_sessions.values()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        @self.app.post("/api/video/submit")
        async def submit_video(body: dict):
            """Submit a video URL or path for training via the gateway."""
            if not self._nc:
                return {"error": "NATS not connected"}
            url = body.get("url", "").strip()
            if not url:
                return {"error": "url is required"}
            cmd = {
                "action": "add_video",
                "url": url,
                "fps": body.get("fps", 2.0),
                "loop": body.get("loop", True),
                "transcript": body.get("transcript", False),
            }
            try:
                await self._nc.publish(
                    "sensory.gateway.command",
                    json.dumps(cmd).encode(),
                )
                return {"status": "submitted", "command": cmd}
            except Exception as e:
                return {"error": str(e)}

        @self.app.post("/api/video/stop")
        async def stop_video(body: dict):
            """Stop a video training session."""
            if not self._nc:
                return {"error": "NATS not connected"}
            session_id = body.get("session_id", "")
            if not session_id:
                return {"error": "session_id is required"}
            cmd = {
                "action": "stop_video",
                "session_id": session_id,
            }
            try:
                await self._nc.publish(
                    "sensory.gateway.command",
                    json.dumps(cmd).encode(),
                )
                return {"status": "sent", "command": cmd}
            except Exception as e:
                return {"error": str(e)}

        @self.app.post("/api/video/queue")
        async def queue_video(body: dict):
            """Add a video to the training queue."""
            if not self._nc:
                return {"error": "NATS not connected"}
            url = body.get("url", "").strip()
            if not url:
                return {"error": "url is required"}
            cmd = {
                "action": "queue_video",
                "url": url,
                "fps": body.get("fps", 2.0),
                "transcript": body.get("transcript", False),
                "target_loops": body.get("target_loops", 5),
                "category": body.get("category", ""),
            }
            try:
                await self._nc.publish(
                    "sensory.gateway.command",
                    json.dumps(cmd).encode(),
                )
                return {"status": "queued", "command": cmd}
            except Exception as e:
                return {"error": str(e)}

        @self.app.post("/api/video/skip")
        async def skip_video():
            """Skip currently playing video, advance queue."""
            if not self._nc:
                return {"error": "NATS not connected"}
            try:
                await self._nc.publish(
                    "sensory.gateway.command",
                    json.dumps({"action": "skip_video"}).encode(),
                )
                return {"status": "sent"}
            except Exception as e:
                return {"error": str(e)}

        @self.app.post("/api/video/clear-queue")
        async def clear_queue():
            """Clear all queued (non-active) videos."""
            if not self._nc:
                return {"error": "NATS not connected"}
            try:
                await self._nc.publish(
                    "sensory.gateway.command",
                    json.dumps({"action": "clear_queue"}).encode(),
                )
                return {"status": "sent"}
            except Exception as e:
                return {"error": str(e)}

        @self.app.post("/api/video/remove-queued")
        async def remove_queued(body: dict):
            """Remove a specific video from the queue."""
            if not self._nc:
                return {"error": "NATS not connected"}
            session_id = body.get("session_id", "")
            if not session_id:
                return {"error": "session_id is required"}
            try:
                await self._nc.publish(
                    "sensory.gateway.command",
                    json.dumps({"action": "remove_queued", "session_id": session_id}).encode(),
                )
                return {"status": "sent"}
            except Exception as e:
                return {"error": str(e)}

        @self.app.post("/api/video/blacklist")
        async def blacklist_video(body: dict):
            """Blacklist a video — stop if active, remove from queue."""
            if not self._nc:
                return {"error": "NATS not connected"}
            session_id = body.get("session_id", "")
            reason = body.get("reason", "Blacklisted by user")
            if not session_id:
                return {"error": "session_id is required"}
            cmd = {
                "action": "blacklist_video",
                "session_id": session_id,
                "reason": reason,
            }
            try:
                await self._nc.publish(
                    "sensory.gateway.command",
                    json.dumps(cmd).encode(),
                )
                return {"status": "blacklisted", "command": cmd}
            except Exception as e:
                return {"error": str(e)}

        @self.app.get("/api/video/blacklist")
        async def get_blacklist():
            """Request current blacklist from gateway."""
            if not self._nc:
                return {"error": "NATS not connected"}
            try:
                await self._nc.publish(
                    "sensory.gateway.command",
                    json.dumps({"action": "get_blacklist"}).encode(),
                )
                return {"status": "requested"}
            except Exception as e:
                return {"error": str(e)}

        # ── API: NATS Messages ───────────────────────────────────────

        @self.app.get("/api/messages")
        async def get_messages(limit: int = 100):
            return {"messages": list(message_buffer)[-limit:], "total": len(message_buffer)}

        # ── API: Insights ────────────────────────────────────────────

        @self.app.get("/api/insights")
        async def get_insights(limit: int = 20):
            return {"insights": list(insights_log)[-limit:]}

        # ── API: Chat (teleoperation channel) ────────────────────────

        @self.app.post("/api/chat")
        async def chat(msg: ChatMessage):
            # Send to neuromorphic network as sensory input
            await self._publish_text_observation(msg.message)

            t0 = time.time()
            reply = await self._chat_with_llm(msg.message)
            dur = (time.time() - t0) * 1000
            self.skills.record_call("brain.chat", dur, reply.get("model") != "error")

            # Flywheel: every chat interaction is learning
            self.knowledge.learn("teleoperation", "conversation", msg.message[:200])

            chat_history.append({"role": "user", "content": msg.message,
                                 "timestamp": datetime.now(timezone.utc).isoformat()})
            chat_history.append({"role": "assistant", "content": reply["content"],
                                 "timestamp": datetime.now(timezone.utc).isoformat()})
            if len(chat_history) > MAX_CHAT_HISTORY:
                chat_history[:] = chat_history[-MAX_CHAT_HISTORY:]

            return {
                "reply": reply["content"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": reply.get("model", self._llm_model),
            }

        @self.app.get("/api/chat/history")
        async def get_chat_history(limit: int = 50):
            return {"history": chat_history[-limit:]}

        # ── API: Observation injection (for demos) ────────────────────

        @self.app.post("/api/observation")
        async def inject_observation(obs: ObservationPayload):
            """Inject a sensory observation into the brain via NATS.

            Used by the demo reaction probe to send stimuli and measure real brain responses.
            """
            if not self._nc:
                return {"error": "NATS not connected", "ok": False}
            try:
                await self._nc.publish(obs.provenance, json.dumps({
                    "provenance": obs.provenance,
                    "data": obs.data,
                }).encode())
                self.logger.info(f"Injected observation via {obs.provenance}")
                return {"ok": True, "provenance": obs.provenance}
            except Exception as e:
                self.logger.warning(f"Failed to inject observation: {e}")
                return {"error": str(e), "ok": False}

        # ── API: Concept probe ──────────────────────────────────────

        @self.app.post("/api/concept-probe")
        async def concept_probe(body: dict):
            """Inject a stimulus and probe concept layer response."""
            if not self._nc:
                return {"error": "NATS not connected", "ok": False}
            try:
                await self._nc.publish(
                    "neuromorphic.concept.probe",
                    json.dumps(body).encode(),
                )
                return {"ok": True, "label": body.get("label", "probe")}
            except Exception as e:
                return {"error": str(e), "ok": False}

        @self.app.get("/api/concept-probe/results")
        async def concept_probe_results():
            """Return all stored concept probe results."""
            return {"results": self._concept_probe_results}

        @self.app.delete("/api/concept-probe/results")
        async def clear_concept_probe_results():
            """Clear stored probe results."""
            self._concept_probe_results.clear()
            return {"ok": True}

        # ── WebSocket ────────────────────────────────────────────────

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            # The /ws channel carries state-changing commands (approval
            # responses to the Kernel, motor guidance, training control). When
            # auth is enabled, refuse connections that don't present the token —
            # otherwise a forged approval_response could defeat the Kernel gate.
            if not authorize_websocket(websocket):
                await websocket.close(code=1008)  # policy violation
                self.logger.warning("WS rejected: missing/invalid dashboard token")
                return
            await websocket.accept()
            active_connections.append(websocket)
            self.skills.record_call("bus.websocket", 0)
            self.logger.info(f"WS connected ({len(active_connections)})")

            try:
                await websocket.send_json({
                    "type": "init",
                    "data": {
                        "system": _system_info,
                        "skills": self.skills.get_all(),
                        "skills_by_category": self.skills.get_by_category(),
                        "flywheel": self.knowledge.get_flywheel_stats(),
                        "services": list(self._service_status.values()),
                        "messages": list(message_buffer)[-30:],
                        "insights": list(insights_log)[-10:],
                        "live_metrics": get_live_metrics(),
                        "neuromorphic": _neuro_metrics,
                        "gateway": _gateway_status,
                        "video_sessions": list(_video_sessions.values()),
                    },
                })

                while True:
                    data = await websocket.receive_text()
                    try:
                        payload = json.loads(data)
                        if payload.get("type") == "chat":
                            # Send to neuromorphic network as sensory input
                            await self._publish_text_observation(payload.get("message", ""))

                            t0 = time.time()
                            reply = await self._chat_with_llm(payload.get("message", ""))
                            dur = (time.time() - t0) * 1000
                            self.skills.record_call("brain.chat", dur, reply.get("model") != "error")
                            self.knowledge.learn("teleoperation", "conversation", payload.get("message", "")[:200])

                            chat_history.append({"role": "user", "content": payload.get("message", ""),
                                                 "timestamp": datetime.now(timezone.utc).isoformat()})
                            chat_history.append({"role": "assistant", "content": reply["content"],
                                                 "timestamp": datetime.now(timezone.utc).isoformat()})
                            if len(chat_history) > MAX_CHAT_HISTORY:
                                chat_history[:] = chat_history[-MAX_CHAT_HISTORY:]

                            await websocket.send_json({
                                "type": "chat_response",
                                "data": {
                                    "reply": reply["content"],
                                    "model": reply.get("model", self._llm_model),
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                },
                            })
                        elif payload.get("type") == "gateway_command":
                            if self._nc:
                                await self._nc.publish(
                                    "sensory.gateway.command",
                                    json.dumps(payload.get("command", {})).encode(),
                                )
                        elif payload.get("type") == "video_submit":
                            if self._nc:
                                cmd = {
                                    "action": "add_video",
                                    "url": payload.get("url", ""),
                                    "fps": payload.get("fps", 2.0),
                                    "loop": payload.get("loop", True),
                                    "transcript": payload.get("transcript", False),
                                }
                                await self._nc.publish(
                                    "sensory.gateway.command",
                                    json.dumps(cmd).encode(),
                                )
                        elif payload.get("type") == "video_stop":
                            if self._nc:
                                cmd = {
                                    "action": "stop_video",
                                    "session_id": payload.get("session_id", ""),
                                }
                                await self._nc.publish(
                                    "sensory.gateway.command",
                                    json.dumps(cmd).encode(),
                                )
                        elif payload.get("type") == "video_queue":
                            if self._nc:
                                cmd = {
                                    "action": "queue_video",
                                    "url": payload.get("url", ""),
                                    "fps": payload.get("fps", 2.0),
                                    "transcript": payload.get("transcript", False),
                                    "target_loops": payload.get("target_loops", 5),
                                    "category": payload.get("category", ""),
                                }
                                await self._nc.publish(
                                    "sensory.gateway.command",
                                    json.dumps(cmd).encode(),
                                )
                        elif payload.get("type") == "video_skip":
                            if self._nc:
                                await self._nc.publish(
                                    "sensory.gateway.command",
                                    json.dumps({"action": "skip_video"}).encode(),
                                )
                        elif payload.get("type") == "video_clear_queue":
                            if self._nc:
                                await self._nc.publish(
                                    "sensory.gateway.command",
                                    json.dumps({"action": "clear_queue"}).encode(),
                                )
                        elif payload.get("type") == "video_remove_queued":
                            if self._nc:
                                await self._nc.publish(
                                    "sensory.gateway.command",
                                    json.dumps({
                                        "action": "remove_queued",
                                        "session_id": payload.get("session_id", ""),
                                    }).encode(),
                                )
                        elif payload.get("type") == "video_blacklist":
                            if self._nc:
                                await self._nc.publish(
                                    "sensory.gateway.command",
                                    json.dumps({
                                        "action": "blacklist_video",
                                        "session_id": payload.get("session_id", ""),
                                        "reason": payload.get("reason", "Blacklisted by user"),
                                    }).encode(),
                                )
                        elif payload.get("type") == "approval_response":
                            # Human responded to a DEFER approval request
                            resp = payload.get("data", {})
                            trace_id = str(resp.get("trace_id", ""))[:64]
                            if not trace_id or not self._nc:
                                pass  # silently ignore invalid or no-NATS
                            else:
                                try:
                                    # Sanitize trace_id for NATS subject safety
                                    safe_trace = trace_id.replace(" ", "").replace(">", "").replace("*", "")
                                    await self._nc.publish(
                                        f"approval.response.{safe_trace}",
                                        json.dumps(resp).encode(),
                                    )
                                except Exception as e:
                                    self.logger.error(f"Failed to publish approval response: {e}")
                                    await websocket.send_json({"type": "error", "data": {"message": "Approval delivery failed"}})
                        elif payload.get("type") == "mujoco_guide":
                            # Forward guidance command to NATS
                            guide_data = payload.get("data", {})
                            if self._nc and guide_data.get("action"):
                                try:
                                    await self._nc.publish(
                                        "motor.guidance",
                                        json.dumps(guide_data).encode(),
                                    )
                                except Exception as e:
                                    self.logger.error(f"Guidance publish failed: {e}")
                        elif payload.get("type") == "ping":
                            await websocket.send_json({"type": "pong"})
                    except json.JSONDecodeError:
                        pass
            except WebSocketDisconnect:
                pass
            except Exception as e:
                self.logger.error(f"WS error: {e}")
            finally:
                if websocket in active_connections:
                    active_connections.remove(websocket)

    # ── NATS ─────────────────────────────────────────────────────────

    async def _publish_text_observation(self, text: str) -> None:
        """Publish chat text as sensory observation for the neuromorphic network."""
        if not self._nc or not text:
            self.logger.debug(f"Skip publish: nc={self._nc is not None}, text={bool(text)}")
            return
        try:
            await self._nc.publish("observation.text", json.dumps({
                "provenance": "observation.text",
                "data": text,
            }).encode())
            self.logger.info(f"Published observation.text ({len(text)} chars)")
        except Exception as e:
            self.logger.warning(f"Failed to publish text observation: {e}")

    async def _connect_nats(self):
        try:
            import nats as nats_lib
            nats_url = os.environ.get("NATS_URL", "nats://localhost:4222")
            nc = await nats_lib.connect(nats_url, name="activelearning-dashboard")
            self._nc = nc
            self._nats_connected = True
            self.skills.record_call("bus.nats", 0)
            self.knowledge.learn("deployment", "connectivity", "Connected to NATS message bus")
            self.logger.info("Connected to NATS")

            # Subjects handled by dedicated callbacks — skip in wildcard handler
            _dedicated_subjects = {"neuromorphic.metrics", "proposal.new"}

            # Throttle high-rate observation.* subjects in the feed
            _obs_last_broadcast: dict[str, float] = {}  # subject → last broadcast time
            _obs_throttle_interval = 0.5  # max 2 broadcasts/sec per observation subject
            _obs_dropped: dict[str, int] = {}  # subject → dropped count since last broadcast
            _obs_max_tracked = 50  # cap tracked subjects to prevent unbounded growth

            async def handle_msg(msg):
                if msg.subject in _dedicated_subjects or msg.subject.startswith("heartbeat."):
                    return

                # Throttle observation.* subjects to avoid flooding the dashboard feed
                if msg.subject.startswith("observation."):
                    now = time.time()
                    # Evict stale entries if tracking too many subjects
                    if len(_obs_last_broadcast) > _obs_max_tracked:
                        cutoff = now - 60.0
                        stale = [k for k, v in _obs_last_broadcast.items() if v < cutoff]
                        for k in stale:
                            _obs_last_broadcast.pop(k, None)
                            _obs_dropped.pop(k, None)
                    last = _obs_last_broadcast.get(msg.subject, 0.0)
                    if now - last < _obs_throttle_interval:
                        _obs_dropped[msg.subject] = _obs_dropped.get(msg.subject, 0) + 1
                        self.knowledge.learn("observation", "nats_message", f"{msg.subject}")
                        return
                    _obs_last_broadcast[msg.subject] = now
                    dropped = _obs_dropped.pop(msg.subject, 0)
                    try:
                        data = json.loads(msg.data.decode())
                    except Exception:
                        data = {"raw": msg.data.decode()[:200]}
                    # Summarize: truncate large payloads, add dropped count
                    if isinstance(data, dict) and "data" in data:
                        raw = data["data"]
                        if isinstance(raw, list) and len(raw) > 8:
                            data["data"] = raw[:4] + ["..."] + [f"({len(raw)} values)"]
                    msg_info = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "subject": msg.subject,
                        "data": data,
                    }
                    if dropped > 0:
                        msg_info["dropped"] = dropped
                    message_buffer.append(msg_info)
                    self.knowledge.learn("observation", "nats_message", f"{msg.subject}")
                    await self._broadcast({"type": "message", "data": msg_info})
                    return

                try:
                    data = json.loads(msg.data.decode())
                except Exception:
                    data = {"raw": msg.data.decode()[:500]}
                msg_info = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "subject": msg.subject,
                    "data": data,
                }
                message_buffer.append(msg_info)
                self.knowledge.learn("observation", "nats_message", f"{msg.subject}")
                await self._broadcast({"type": "message", "data": msg_info})

            async def handle_heartbeat(msg):
                try:
                    data = json.loads(msg.data.decode())
                    svc = data.get("service", "?")
                    self._service_status[svc] = {
                        "name": svc, "status": "running",
                        "uptime": data.get("uptime"),
                        "last_seen": datetime.now(timezone.utc).isoformat(),
                    }
                    await self._broadcast({"type": "service_status", "data": self._service_status[svc]})
                except Exception:
                    pass

            async def handle_neuro_metrics(msg):
                global _neuro_metrics
                try:
                    data = json.loads(msg.data.decode())
                    _neuro_metrics = data
                    self.knowledge.learn("simulation", "neural_simulation", f"step={data.get('step_count', '?')}")
                    await self._broadcast({"type": "neuro_update", "data": data})
                except Exception:
                    pass

            _last_proposal_broadcast = [0.0]  # mutable for closure

            async def handle_neuro_proposal(msg):
                now = time.time()
                if now - _last_proposal_broadcast[0] < 0.5:
                    return  # throttle: max 2 broadcasts/sec
                try:
                    data = json.loads(msg.data.decode())
                    provenance = data.get("provenance", "")
                    if provenance.startswith("neuromorphic."):
                        _last_proposal_broadcast[0] = now
                        await self._broadcast({"type": "neuro_response", "data": {
                            "action": data.get("action", {}),
                            "provenance": provenance,
                            "metadata": data.get("metadata", {}),
                        }})
                except Exception:
                    pass

            async def handle_gateway_status(msg):
                global _gateway_status
                try:
                    data = json.loads(msg.data.decode())
                    _gateway_status = data
                    await self._broadcast({"type": "gateway_update", "data": data})
                except Exception:
                    pass

            async def handle_video_training_status(msg):
                try:
                    data = json.loads(msg.data.decode())
                    sid = data.get("session_id")
                    # Filter transient/error markers — only store real session IDs
                    if sid and sid not in ("error", "download_error", "pending"):
                        _video_sessions[sid] = data
                        # Prune oldest completed/stopped/error sessions if over limit
                        if len(_video_sessions) > MAX_VIDEO_SESSIONS:
                            removable = [
                                (k, v) for k, v in _video_sessions.items()
                                if v.get("status") in ("stopped", "completed", "error")
                            ]
                            removable.sort(key=lambda x: x[1].get("created_at", 0))
                            for k, _ in removable[:len(_video_sessions) - MAX_VIDEO_SESSIONS]:
                                _video_sessions.pop(k, None)
                    await self._broadcast({"type": "video_training_update", "data": data})
                except Exception:
                    pass

            async def handle_approval_request(msg):
                """Forward Kernel DEFER approval requests to dashboard UI."""
                try:
                    data = json.loads(msg.data.decode())
                    # Validate required fields and sanitize
                    trace_id = data.get("trace_id", "")
                    if not isinstance(trace_id, str) or not trace_id or len(trace_id) > 64:
                        return
                    channel = data.get("channel", "")
                    if not isinstance(channel, str) or len(channel) > 128:
                        return
                    intensity = data.get("intensity")
                    if intensity is not None and (not isinstance(intensity, (int, float)) or intensity < 0 or intensity > 1):
                        data["intensity"] = None  # sanitize invalid
                    reason = str(data.get("reason", ""))[:500]
                    data["reason"] = reason
                    await self._broadcast({"type": "approval_request", "data": data})
                except Exception:
                    pass

            async def handle_mujoco_state(msg):
                try:
                    data = json.loads(msg.data.decode())
                    await self._broadcast({"type": "mujoco_state", "data": data})
                except Exception:
                    pass

            _last_body_frame_ts = 0.0

            async def handle_visual_body(msg):
                """Forward body camera frame (64x64 grayscale) to WS clients.

                Published by MuJoCo at 2 Hz as observation.visual.body.
                Data is 4096 floats [0,1]. We convert to uint8 and base64-encode
                to keep WS payload small (~5.5 KB vs ~40 KB raw JSON).
                """
                nonlocal _last_body_frame_ts
                now = time.time()
                if now - _last_body_frame_ts < 0.4:  # throttle to ~2.5 Hz max
                    return
                _last_body_frame_ts = now
                try:
                    data = json.loads(msg.data.decode())
                    pixels = data.get("data", [])
                    if not pixels or len(pixels) != 4096:
                        return
                    # Convert [0,1] floats to uint8 bytes, then base64
                    raw = bytes(min(255, max(0, int(v * 255))) for v in pixels)
                    b64 = base64.b64encode(raw).decode("ascii")
                    await self._broadcast({
                        "type": "visual_body_frame",
                        "data": {"pixels_b64": b64, "width": 64, "height": 64},
                    })
                except Exception:
                    pass

            async def handle_concept_result(msg):
                try:
                    data = json.loads(msg.data.decode())
                    if len(self._concept_probe_results) >= self._MAX_PROBE_RESULTS:
                        self._concept_probe_results.pop(0)
                    self._concept_probe_results.append(data)
                    self.logger.info(f"Concept probe result received: {data.get('label', '?')}")
                    await self._broadcast({"type": "concept_probe_result", "data": data})
                except Exception as e:
                    self.logger.error(f"Error handling concept result: {e}")

            async def handle_watchdog_status(msg):
                try:
                    data = json.loads(msg.data.decode())
                    global _watchdog_status
                    _watchdog_status = data
                    await self._broadcast({"type": "watchdog_status", "data": data})
                except Exception as e:
                    self.logger.error(f"Error handling watchdog status: {e}")

            async def handle_deny_escalation(msg):
                try:
                    data = json.loads(msg.data.decode())
                    _deny_escalations.append(data)
                    self.logger.warning(
                        f"Deny escalation: channel={data.get('channel')}, "
                        f"action={data.get('action')}"
                    )
                    await self._broadcast({"type": "deny_escalation", "data": data})
                except Exception as e:
                    self.logger.error(f"Error handling deny escalation: {e}")

            async def handle_speech_execute(msg):
                """Log speech execute events for dashboard display.

                No TTS actuator yet — this prevents the event from being
                silently dropped until Phase 4 (TTS actuator) is implemented.
                """
                try:
                    data = json.loads(msg.data.decode())
                    await self._broadcast({"type": "speech_execute", "data": data})
                except Exception as e:
                    self.logger.error(f"Error handling speech execute: {e}")

            # Subjects handled by dedicated callbacks — add to skip set
            _dedicated_subjects.add("sensory.gateway.status")
            _dedicated_subjects.add("video.training.status")
            _dedicated_subjects.add("approval.request")
            _dedicated_subjects.add("mujoco.body.state")
            _dedicated_subjects.add("neuromorphic.concept.result")
            _dedicated_subjects.add("safety.watchdog.status")
            _dedicated_subjects.add("safety.deny_escalation")
            _dedicated_subjects.add("speech.execute")
            _dedicated_subjects.add("observation.visual.body")

            await nc.subscribe(">", cb=handle_msg)
            await nc.subscribe("heartbeat.*", cb=handle_heartbeat)
            await nc.subscribe("neuromorphic.metrics", cb=handle_neuro_metrics)
            await nc.subscribe("proposal.new", cb=handle_neuro_proposal)
            await nc.subscribe("sensory.gateway.status", cb=handle_gateway_status)
            await nc.subscribe("video.training.status", cb=handle_video_training_status)
            await nc.subscribe("approval.request", cb=handle_approval_request)
            await nc.subscribe("mujoco.body.state", cb=handle_mujoco_state)
            await nc.subscribe("neuromorphic.concept.result", cb=handle_concept_result)
            await nc.subscribe("safety.watchdog.status", cb=handle_watchdog_status)
            await nc.subscribe("safety.deny_escalation", cb=handle_deny_escalation)
            await nc.subscribe("speech.execute", cb=handle_speech_execute)
            await nc.subscribe("observation.visual.body", cb=handle_visual_body)
        except Exception as e:
            self.logger.warning(f"NATS failed (non-fatal): {e}")
            self._nats_connected = False

    # ── Chat with LLM ───────────────────────────────────────────────

    async def _chat_with_llm(self, user_message: str) -> dict[str, str]:
        system_context = self._build_system_context()
        messages = [{"role": "system", "content": system_context}]
        for entry in chat_history[-20:]:
            messages.append({"role": entry["role"], "content": entry["content"]})
        messages.append({"role": "user", "content": user_message})

        if self._openai_url and self._openai_key:
            return await self._chat_openai(messages)
        return await self._chat_ollama(messages)

    def _build_system_context(self) -> str:
        parts = [
            "You are the voice of Engram -- a neuromorphic brain built on spiking neural networks.",
            "You are NOT the brain itself. You are a communication interface (LLM) that translates",
            "the brain's state into natural language so humans can understand what it is experiencing.",
            "",
            "Engram is a biologically-grounded cognitive architecture for robotics:",
            "- 1M+ spiking neurons with STDP learning, eligibility traces, BCM metaplasticity",
            "- 4-channel neuromodulation (dopamine, acetylcholine, norepinephrine, serotonin)",
            "- Developmental phases: infant, toddler, juvenile, adolescent, mature",
            "- Learns through sensory experience (video, audio, proprioception), NOT from text prompts",
            "- Virtual body (MuJoCo humanoid) with closed-loop motor learning",
            "- Cross-modal binding: temporal correlation between what it sees and hears forms associations via STDP",
            "",
            "IMPORTANT RULES:",
            "- Do NOT claim to notice system metrics, load averages, or performance changes yourself.",
            "  You only know what is provided in the brain state below.",
            "- Do NOT offer unsolicited maintenance advice or system health commentary.",
            "- Do NOT pretend the user's text input is directly training you. Text goes to the brain",
            "  as sensory input; your LLM response is separate.",
            "- When discussing what the brain 'knows' or 'feels', be honest about its current stage.",
            "  A juvenile brain has basic feature representations, not rich semantic understanding.",
            "- Be concise, scientifically grounded, and avoid anthropomorphizing beyond what the data shows.",
            "",
        ]

        # Brain state from neuromorphic network (real data, not hallucinated)
        brain_section = self._interpret_brain_state()
        if brain_section:
            parts.append("CURRENT BRAIN STATE (real-time data from the spiking network):")
            parts.append(brain_section)
            parts.append("")

        if _system_info:
            os_i = _system_info.get("os", {})
            cpu = _system_info.get("cpu", {})
            mem = _system_info.get("memory", {})
            parts.append(f"Server: {os_i.get('system', '?')} {os_i.get('release', '')} -- {cpu.get('model', cpu.get('architecture', '?'))} ({cpu.get('cores', '?')} cores), {mem.get('total_gb', '?')} GB RAM")

        parts.extend([
            "",
            "When the user teaches you something (e.g., 'a ball is round'), explain that the text has",
            "been injected into the brain's sensory cortex as a spike pattern. The brain will form",
            "associations through STDP if this input correlates with other sensory experience.",
            "The brain learns from temporal correlation, not from understanding the sentence.",
        ])
        return "\n".join(parts)

    def _interpret_brain_state(self) -> str:
        """Translate raw neuromorphic metrics into natural language for the LLM system prompt."""
        if not _neuro_metrics:
            return ""

        parts = []

        # Development stage -- use actual phase from neuromodulation if available,
        # otherwise estimate from step count using the real phase boundaries.
        step_count = _neuro_metrics.get("step_count", 0)
        phase = _neuro_metrics.get("phase", "")
        if not phase:
            if step_count < 60_000:
                phase = "infant"
            elif step_count < 360_000:
                phase = "toddler"
            elif step_count < 2_160_000:
                phase = "juvenile"
            else:
                phase = "adolescent or mature"

        phase_descriptions = {
            "infant": "infant (basic sensory calibration, high plasticity)",
            "toddler": "toddler (forming first associations, neuromodulator baselines shifting)",
            "juvenile": "juvenile (active learning, cross-modal binding forming, pre-adolescent)",
            "adolescent": "adolescent (pruning weak synapses, myelinating strong ones, locking structure)",
            "mature": "mature (stable representations, reduced plasticity, continual learning mode)",
        }
        stage = phase_descriptions.get(phase, phase)
        parts.append(f"Phase: {stage} -- step {step_count:,}")

        # Cognitive state from firing rates
        firing = _neuro_metrics.get("firing_rates", {})
        cognitive_notes = []

        assoc_rate = firing.get("association_cortex", 0)
        if assoc_rate > 0.05:
            cognitive_notes.append("association cortex highly active (creative/connecting)")
        elif assoc_rate > 0.01:
            cognitive_notes.append("association cortex engaged (linking concepts)")

        pred_rate = firing.get("predictive_layer", 0)
        if pred_rate > 0.05:
            cognitive_notes.append("predictive layer firing strongly (anticipating)")
        elif pred_rate > 0.01:
            cognitive_notes.append("predictive layer active (forming expectations)")

        motor_rate = firing.get("motor_cortex", 0)
        if motor_rate > 0.05:
            cognitive_notes.append("motor cortex active (action-oriented)")

        sensory_rate = firing.get("sensory_cortex", 0)
        if sensory_rate > 0.05:
            cognitive_notes.append("sensory cortex processing input (attentive)")

        wm_rate = firing.get("working_memory", 0)
        if wm_rate > 0.03:
            cognitive_notes.append("working memory engaged (holding context)")

        if cognitive_notes:
            parts.append(f"Cognitive: {'; '.join(cognitive_notes)}")
        else:
            parts.append("Cognitive: resting state (minimal activity)")

        # Emotional state from drives
        drives = _neuro_metrics.get("drives", {})
        emotional_notes = []

        energy = drives.get("energy", 1.0)
        if energy > 0.7:
            emotional_notes.append("alert and energetic")
        elif energy > 0.3:
            emotional_notes.append("moderate energy")
        else:
            emotional_notes.append("low energy (conserving)")

        fatigue = drives.get("fatigue", 0.0)
        if fatigue > 0.7:
            emotional_notes.append("fatigued (prefer brevity)")
        elif fatigue > 0.4:
            emotional_notes.append("slightly tired")

        damage = drives.get("damage", 0.0)
        if damage > 0.3:
            emotional_notes.append("sensing strain (concerned)")

        if emotional_notes:
            parts.append(f"Emotional: {', '.join(emotional_notes)}")

        # Prediction error
        pred_error = _neuro_metrics.get("drives", {}).get("prediction_error")
        if pred_error is not None and pred_error > 0.5:
            parts.append(f"Surprise level: high ({pred_error:.2f}) — world model is being challenged")

        return "\n".join(parts)

    def _generate_brain_only_response(self) -> dict[str, str]:
        """Generate a response purely from brain state when no LLM is available."""
        if not _neuro_metrics:
            return {
                "content": "The neuromorphic brain is initializing. No metrics available yet.",
                "model": "neural-only",
            }

        step_count = _neuro_metrics.get("step_count", 0)
        firing = _neuro_metrics.get("firing_rates", {})
        phase = _neuro_metrics.get("phase", "unknown")

        parts = [f"Engram brain at step {step_count:,} ({phase} phase)."]

        # Report active regions
        active = [(k, v) for k, v in firing.items() if v > 0.01]
        if active:
            rates = ", ".join(f"{k} {v*100:.0f}%" for k, v in sorted(active, key=lambda x: -x[1])[:5])
            parts.append(f"Active regions: {rates}.")
        else:
            parts.append("Neurons are mostly quiet. The brain needs sensory input to activate.")

        parts.append("\n*[No LLM connected. Connect Ollama for natural language responses.]*")

        return {
            "content": " ".join(parts),
            "model": "neural-only",
        }

    async def _chat_ollama(self, messages: list[dict]) -> dict[str, str]:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
                async with session.post(
                    f"{self._ollama_url}/api/chat",
                    json={"model": self._llm_model, "messages": messages, "stream": False},
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {"content": data.get("message", {}).get("content", "No response."), "model": f"ollama/{self._llm_model}"}
                    else:
                        txt = await resp.text()
                        return {"content": f"⚠️ Ollama {resp.status}. Model '{self._llm_model}' may not be pulled.\n\n`docker exec activelearning-ollama ollama pull {self._llm_model}`\n\n{txt[:200]}", "model": "error"}
        except aiohttp.ClientConnectorError:
            return self._generate_brain_only_response()
        except Exception as e:
            return {"content": f"⚠️ LLM error: {e}", "model": "error"}

    async def _chat_openai(self, messages: list[dict]) -> dict[str, str]:
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                async with session.post(
                    f"{self._openai_url}/v1/chat/completions",
                    json={"model": self._llm_model, "messages": messages, "max_tokens": 2000},
                    headers={"Authorization": f"Bearer {self._openai_key}", "Content-Type": "application/json"},
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {"content": data["choices"][0]["message"]["content"], "model": data.get("model", self._llm_model)}
                    return {"content": f"OpenAI API error: {resp.status}", "model": "error"}
        except Exception as e:
            self.logger.warning(f"OpenAI failed, falling back to Ollama: {e}")
            return await self._chat_ollama(messages)

    # ── Self-Improvement Loop ────────────────────────────────────────

    async def _self_improvement_loop(self):
        await asyncio.sleep(5)
        self.logger.info("Self-improvement loop started")
        while True:
            try:
                t0 = time.time()
                await self._run_health_check()
                self.skills.record_call("brain.self_monitor", (time.time() - t0) * 1000)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Self-monitor error: {e}")
            await asyncio.sleep(60)

    async def _run_health_check(self):
        findings = []
        now = datetime.now(timezone.utc).isoformat()

        # Disk
        try:
            disk = shutil.disk_usage("/")
            pct = (disk.used / disk.total) * 100
            if pct > 90:
                findings.append({"level": "critical", "message": f"Disk usage critical: {pct:.1f}%", "timestamp": now})
            elif pct > 75:
                findings.append({"level": "warning", "message": f"Disk usage high: {pct:.1f}%", "timestamp": now})
        except Exception:
            pass

        # Memory
        try:
            if platform.system() == "Linux":
                with open("/proc/meminfo") as f:
                    meminfo = f.read()
                total = avail = 0
                for line in meminfo.split("\n"):
                    if line.startswith("MemTotal:"): total = int(line.split()[1])
                    elif line.startswith("MemAvailable:"): avail = int(line.split()[1])
                if total > 0:
                    pct = ((total - avail) / total) * 100
                    if pct > 90:
                        findings.append({"level": "critical", "message": f"Memory critical: {pct:.1f}%", "timestamp": now})
                    elif pct > 80:
                        findings.append({"level": "warning", "message": f"Memory high: {pct:.1f}%", "timestamp": now})
        except Exception:
            pass

        # Load
        try:
            if platform.system() == "Linux":
                with open("/proc/loadavg") as f:
                    load = float(f.read().split()[0])
                cores = os.cpu_count() or 1
                if load > cores * 2:
                    findings.append({"level": "warning", "message": f"Load {load:.1f} exceeds 2x cores ({cores})", "timestamp": now})
        except Exception:
            pass

        # Docker containers
        try:
            metrics = await self._fetch_docker_metrics()
            for m in metrics:
                if m.get("cpu_percent", 0) > 80:
                    findings.append({"level": "warning", "message": f"Container '{m['service']}' CPU: {m['cpu_percent']:.1f}%", "timestamp": now})
                if m.get("memory_percent", 0) > 80:
                    findings.append({"level": "warning", "message": f"Container '{m['service']}' Mem: {m['memory_percent']:.1f}%", "timestamp": now})
        except Exception:
            pass

        if not findings:
            findings.append({"level": "info", "message": "All systems nominal ✓", "timestamp": now})

        for f in findings:
            insights_log.append(f)
            self.knowledge.learn("deployment", "health_check", f["message"])

        # Broadcast findings + updated flywheel stats
        await self._broadcast({"type": "insights", "data": findings})
        await self._broadcast({"type": "flywheel_update", "data": self.knowledge.get_flywheel_stats()})
        await self._broadcast({"type": "skills_update", "data": self.skills.get_all()})

    # ── Metrics Broadcast ────────────────────────────────────────────

    async def _metrics_broadcast_loop(self):
        await asyncio.sleep(3)
        while True:
            try:
                if active_connections:
                    t0 = time.time()
                    live = get_live_metrics()
                    docker = await self._fetch_docker_metrics()
                    self.skills.record_call("env.monitor", (time.time() - t0) * 1000)
                    await self._broadcast({
                        "type": "metrics_update",
                        "data": {"live": live, "docker": docker},
                    })
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics broadcast: {e}")
            await asyncio.sleep(10)

    # ── Broadcast ────────────────────────────────────────────────────

    async def _broadcast(self, message: dict) -> None:
        if not active_connections:
            return
        dead = []
        for c in list(active_connections):
            try:
                await c.send_json(message)
            except Exception:
                dead.append(c)
        for c in dead:
            if c in active_connections:
                active_connections.remove(c)

    # ── Docker Metrics ───────────────────────────────────────────────

    async def _fetch_docker_metrics(self) -> list[dict]:
        global _system_metrics_cache, _last_metrics_update
        if time.time() - _last_metrics_update < 5 and _system_metrics_cache:
            return _system_metrics_cache.get("metrics", [])
        try:
            connector = aiohttp.UnixConnector(path="/var/run/docker.sock")
            async with aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get("http://localhost/containers/json") as resp:
                    containers = await resp.json()
                metrics = []
                for ctr in containers:
                    cid = ctr["Id"]
                    name = ctr["Names"][0].lstrip("/")
                    try:
                        async with session.get(f"http://localhost/containers/{cid}/stats?stream=false") as resp:
                            stats = await resp.json()
                            cpu_d = stats["cpu_stats"]["cpu_usage"]["total_usage"] - stats["precpu_stats"]["cpu_usage"]["total_usage"]
                            sys_d = stats["cpu_stats"]["system_cpu_usage"] - stats["precpu_stats"]["system_cpu_usage"]
                            cpus = stats["cpu_stats"].get("online_cpus", 1)
                            cpu_pct = (cpu_d / sys_d) * cpus * 100 if sys_d > 0 else 0
                            mem_u = stats["memory_stats"].get("usage", 0)
                            mem_l = stats["memory_stats"].get("limit", 1)
                            nets = stats.get("networks", {})
                            rx = sum(n.get("rx_bytes", 0) for n in nets.values())
                            tx = sum(n.get("tx_bytes", 0) for n in nets.values())
                            metrics.append({
                                "service": name, "status": ctr.get("State", "?"),
                                "cpu_percent": round(cpu_pct, 2),
                                "memory_mb": round(mem_u / (1024*1024), 2),
                                "memory_percent": round((mem_u / mem_l) * 100, 2) if mem_l else 0,
                                "network_rx_mb": round(rx / (1024*1024), 2),
                                "network_tx_mb": round(tx / (1024*1024), 2),
                            })
                    except Exception:
                        metrics.append({"service": name, "status": ctr.get("State", "?"),
                                        "cpu_percent": 0, "memory_mb": 0, "memory_percent": 0,
                                        "network_rx_mb": 0, "network_tx_mb": 0})
                _system_metrics_cache = {"metrics": metrics}
                _last_metrics_update = time.time()
                return metrics
        except Exception:
            return _system_metrics_cache.get("metrics", [])


# ═══════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════

_startup_time = time.time()
service = DashboardService()
app = service.app


async def main():
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s - %(message)s")
    config = uvicorn.Config(
        app, host="0.0.0.0",
        port=int(os.environ.get("DASHBOARD_PORT", 8080)),
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
