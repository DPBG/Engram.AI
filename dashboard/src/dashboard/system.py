"""
Deep system detection and live resource metrics.

Not just "what OS" but full awareness of services, APIs, capabilities, and the
environment. Pure stdlib, no web-stack dependency. Functions are stateless:
``detect_system_info`` returns a snapshot dict that the caller stores in shared
state (rather than mutating a module-level global).
"""

import json
import os
import platform
import re
import shutil
import subprocess
from typing import Any

from dashboard.util import now_iso


def detect_system_info() -> dict[str, Any]:
    """
    Deep system detection — not just "what OS" but full awareness of
    services, APIs, capabilities, and the environment.
    """
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
    metrics["timestamp"] = now_iso()
    return metrics
