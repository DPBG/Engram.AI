"""FastAPI routers for the Engram dashboard, grouped by area.

Each module exposes a ``build_*_router(ctx[, ...])`` factory that returns a
configured :class:`fastapi.APIRouter` wired to the shared
:class:`dashboard.context.DashboardContext`. ``DashboardService`` includes them
in place of the old monolithic ``_setup_routes``.
"""

from dashboard.routers.chat import build_chat_router
from dashboard.routers.control import build_control_router
from dashboard.routers.introspection import build_introspection_router
from dashboard.routers.stream import build_stream_router
from dashboard.routers.system import build_system_router

__all__ = [
    "build_chat_router",
    "build_control_router",
    "build_introspection_router",
    "build_stream_router",
    "build_system_router",
]
