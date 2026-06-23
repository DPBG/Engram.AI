"""
Metrics & health monitoring.

Owns the dashboard's background loops — the self-improvement health check and
the periodic live/Docker metrics broadcast — plus the cached Docker-stats
fetch shared by the ``/api/metrics`` route and the health check.

``aiohttp`` is imported lazily inside :meth:`fetch_docker` so this module
imports without it.
"""

import asyncio
import logging
import os
import platform
import shutil
import time

from dashboard.state import DashboardState
from dashboard.system import get_live_metrics
from dashboard.util import now_iso


class MetricsMonitor:
    """Background health/metrics loops and the cached Docker-stats fetch."""

    def __init__(self, state: DashboardState):
        self._state = state
        self.logger = logging.getLogger("dashboard.metrics")

    async def _broadcast(self, message: dict) -> None:
        await self._state.connections.broadcast(message)

    # ── self-improvement loop ─────────────────────────────────────────────
    async def self_improvement_loop(self):
        await asyncio.sleep(5)
        self.logger.info("Self-improvement loop started")
        while True:
            try:
                t0 = time.time()
                await self.run_health_check()
                self._state.skills.record_call("brain.self_monitor", (time.time() - t0) * 1000)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Self-monitor error: {e}")
            await asyncio.sleep(60)

    async def run_health_check(self):
        findings = []
        now = now_iso()

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
            metrics = await self.fetch_docker()
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
            self._state.insights_log.append(f)
            self._state.knowledge.learn("deployment", "health_check", f["message"])

        # Broadcast findings + updated flywheel stats.
        await self._broadcast({"type": "insights", "data": findings})
        await self._broadcast({"type": "flywheel_update", "data": self._state.knowledge.get_flywheel_stats()})
        await self._broadcast({"type": "skills_update", "data": self._state.skills.get_all()})

    # ── metrics broadcast loop ────────────────────────────────────────────
    async def metrics_broadcast_loop(self):
        await asyncio.sleep(3)
        while True:
            try:
                if self._state.connections:
                    t0 = time.time()
                    live = get_live_metrics()
                    docker = await self.fetch_docker()
                    self._state.skills.record_call("env.monitor", (time.time() - t0) * 1000)
                    await self._broadcast({
                        "type": "metrics_update",
                        "data": {"live": live, "docker": docker},
                    })
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics broadcast: {e}")
            await asyncio.sleep(10)

    # ── docker stats (cached) ─────────────────────────────────────────────
    async def fetch_docker(self) -> list[dict]:
        state = self._state
        if time.time() - state.last_metrics_update < 5 and state.system_metrics_cache:
            return state.system_metrics_cache.get("metrics", [])
        try:
            import aiohttp
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
                state.system_metrics_cache = {"metrics": metrics}
                state.last_metrics_update = time.time()
                return metrics
        except Exception:
            return state.system_metrics_cache.get("metrics", [])
