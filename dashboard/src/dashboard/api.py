"""
Engram Dashboard — composition root.

The LLM chat is a communication interface for the spiking neural network, not
the intelligence itself — it provides a natural-language window into the brain's
state. This module only assembles the app: it builds the shared state and the
focused components (NATS, chat, metrics) and wires the router groups onto
FastAPI. The behaviour lives in those modules.
"""

import asyncio
import logging
import os
import time
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from dashboard.auth import install_auth_middleware
from dashboard.chat import ChatEngine
from dashboard.context import DashboardContext
from dashboard.knowledge import KnowledgeBase
from dashboard.metrics import MetricsMonitor
from dashboard.models import ChatMessage, ObservationPayload
from dashboard.nats_stream import NatsStreamManager
from dashboard.routers import (
    build_chat_router,
    build_control_router,
    build_introspection_router,
    build_stream_router,
    build_system_router,
)
from dashboard.skills import SkillRegistry
from dashboard.state import DashboardState
from dashboard.system import detect_system_info, get_live_metrics

logger = logging.getLogger(__name__)

# The dashboard's public surface. ``SkillRegistry``/``KnowledgeBase``/the models
# and ``detect_system_info``/``get_live_metrics`` now live in focused modules but
# are re-exported here for backwards compatibility with anything importing them
# from ``dashboard.api``.
__all__ = [
    "DashboardService",
    "app",
    "main",
    "service",
    "SkillRegistry",
    "KnowledgeBase",
    "ChatMessage",
    "ObservationPayload",
    "detect_system_info",
    "get_live_metrics",
]


class DashboardService:
    """
    Engram Dashboard — interface to the neuromorphic brain.

    Monitors the spiking neural network, provides a chat interface
    (LLM as communication layer), and manages the deployment
    environment. Owns the shared state and the focused components, and wires
    the routers + lifecycle onto the FastAPI app.
    """

    def __init__(self):
        self.app = FastAPI(title="Engram")
        self.logger = logging.getLogger("dashboard")

        # Shared state + focused components (the former god-class concerns).
        self.state = DashboardState()
        self.nats = NatsStreamManager(self.state)
        self.chat = ChatEngine(self.state, self.nats)
        self.metrics = MetricsMonitor(self.state)
        self.ctx = DashboardContext(
            state=self.state, nats=self.nats, chat=self.chat, metrics=self.metrics,
        )

        self._self_monitor_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None

        self._configure_app()

    def _configure_app(self):
        app = self.app

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"], allow_credentials=True,
            allow_methods=["*"], allow_headers=["*"],
        )

        # Authenticate the control plane: when ENGRAM_DASHBOARD_TOKEN is set,
        # every state-mutating request (POST/PUT/DELETE) must present it. No-op
        # in dev when the token is unset (logs a one-time warning). See auth.py.
        install_auth_middleware(app)

        static_dir = os.path.join(os.path.dirname(__file__), "..", "..", "static")

        # Brain visualization (standalone Three.js app)
        # In Docker: mounted at /app/brain-viz via volume
        # Local dev: relative to project root
        brain_viz_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "brain-viz")
        if not os.path.exists(brain_viz_dir):
            brain_viz_dir = "/app/brain-viz"
        if os.path.exists(brain_viz_dir):
            app.mount("/brain-viz", StaticFiles(directory=brain_viz_dir, html=True), name="brain-viz")

        if os.path.exists(static_dir):
            app.mount("/static", StaticFiles(directory=static_dir), name="static")

        # Routers, grouped by area (replaces the old monolithic _setup_routes).
        app.include_router(build_system_router(self.ctx, static_dir))
        app.include_router(build_introspection_router(self.ctx))
        app.include_router(build_control_router(self.ctx))
        app.include_router(build_chat_router(self.ctx))
        app.include_router(build_stream_router(self.ctx))

        self._register_lifecycle()

    def _register_lifecycle(self):
        app = self.app

        @app.on_event("startup")
        async def on_startup():
            t0 = time.time()
            self.state.system_info = detect_system_info()
            self.state.skills.record_call("env.detect", (time.time() - t0) * 1000)
            info = self.state.system_info
            self.state.knowledge.learn(
                "deployment", "system",
                f"Engram deployed on: {info.get('os', {}).get('system', '?')} "
                f"{info.get('os', {}).get('machine', '?')}",
            )
            self.state.knowledge.learn(
                "deployment", "system",
                f"Capabilities: {', '.join(info.get('capabilities', []))}",
            )
            self._self_monitor_task = asyncio.create_task(self.metrics.self_improvement_loop())
            self._metrics_task = asyncio.create_task(self.metrics.metrics_broadcast_loop())
            asyncio.create_task(self.nats.connect())
            self.logger.info("Engram Dashboard started")

        @app.on_event("shutdown")
        async def on_shutdown():
            if self._self_monitor_task:
                self._self_monitor_task.cancel()
            if self._metrics_task:
                self._metrics_task.cancel()


# ═══════════════════════════════════════════════════════════════════════
# APPLICATION
# ═══════════════════════════════════════════════════════════════════════

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
