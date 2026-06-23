"""System & monitoring routes: pages, health, system info, metrics, benchmarks."""

import json
import os
import time

from fastapi import APIRouter
from fastapi.responses import FileResponse, RedirectResponse

from dashboard.context import DashboardContext
from dashboard.system import detect_system_info, get_live_metrics
from dashboard.util import now_iso


def build_system_router(ctx: DashboardContext, static_dir: str) -> APIRouter:
    router = APIRouter()
    state = ctx.state

    # ── Pages ─────────────────────────────────────────────────────────────
    @router.get("/")
    async def root():
        return RedirectResponse(url="/brain-viz/demos/index.html")

    @router.get("/dashboard")
    async def dashboard_page():
        idx = os.path.join(static_dir, "index.html")
        if os.path.exists(idx):
            return FileResponse(idx)
        return {"status": "ok"}

    # ── Health ────────────────────────────────────────────────────────────
    @router.get("/api/health")
    async def health():
        return {
            "status": "healthy",
            "timestamp": now_iso(),
            "nats": ctx.nats.connected,
            "uptime_seconds": int(time.time() - state.started_at),
            "total_skill_calls": state.skills.total_calls(),
            "knowledge_entries": len(state.knowledge),
        }

    # ── System ────────────────────────────────────────────────────────────
    @router.get("/api/system")
    async def system_info():
        if not state.system_info:
            state.system_info = detect_system_info()
        return {
            "info": state.system_info,
            "live": get_live_metrics(),
            "timestamp": now_iso(),
        }

    # ── Docker metrics ────────────────────────────────────────────────────
    @router.get("/api/metrics")
    async def get_metrics():
        t0 = time.time()
        metrics = await ctx.metrics.fetch_docker()
        state.skills.record_call("env.docker", (time.time() - t0) * 1000)
        return {"metrics": metrics, "timestamp": now_iso()}

    # ── Services ──────────────────────────────────────────────────────────
    @router.get("/api/services")
    async def get_services():
        return {"services": list(state.service_status.values())}

    # ── Neuromorphic ──────────────────────────────────────────────────────
    @router.get("/api/neuromorphic")
    async def get_neuromorphic():
        return {"neuromorphic": state.neuro_metrics, "timestamp": now_iso()}

    # ── Benchmark results ─────────────────────────────────────────────────
    @router.get("/api/benchmark/latest")
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

    @router.get("/api/benchmark/history")
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

    return router
