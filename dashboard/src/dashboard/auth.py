"""
Control-plane authentication for the Engram dashboard (Phase 1.7).

The dashboard exposes the only HTTP/WebSocket surface that can change system
state: inject sensory observations, drive motors (``motor.guidance``), control
the training pipeline, and — most critically — answer the Kernel's DEFER
approval requests over the ``/ws`` ``approval_response`` channel. An
unauthenticated approval response forges a *human* decision and defeats the
Kernel gate's fail-closed DEFER path. So every state-mutating request must be
authenticated.

Design (mirrors ``activelearning.signing`` so operators learn one model):
- A shared bearer token is configured via ``ENGRAM_DASHBOARD_TOKEN``.
- **Fail-safe by configuration:** when the token is set it is *enforced* —
  mutating HTTP requests (POST/PUT/DELETE/PATCH) and ``/ws`` connections must
  present it, or they are rejected. When it is not set, auth is disabled
  (legacy/dev mode so ``python run.py`` + ``localhost:8080`` keeps working) and
  a one-time warning is logged so operators know the control plane is open.
- Read-only GETs are intentionally *not* gated, so the dashboard UI and static
  assets load without a token; only state changes require it.
- ``hmac.compare_digest`` is used for constant-time comparison.

This module is pure stdlib at import time (FastAPI is imported lazily inside the
middleware installer) so it can be unit-tested without the web stack.
"""

from __future__ import annotations

import hmac
import logging
import os

logger = logging.getLogger("dashboard.auth")

#: Environment variable holding the shared dashboard bearer token.
DASHBOARD_TOKEN_ENV = "ENGRAM_DASHBOARD_TOKEN"

#: HTTP methods that mutate state and therefore require authentication.
MUTATING_METHODS = frozenset({"POST", "PUT", "DELETE", "PATCH"})

_warned_open = False


def get_dashboard_token() -> str | None:
    """Return the configured control-plane token, or ``None`` if auth is off."""
    token = os.environ.get(DASHBOARD_TOKEN_ENV, "").strip()
    return token or None


def auth_enabled() -> bool:
    """True when a dashboard token is configured (auth is enforced)."""
    return get_dashboard_token() is not None


def parse_bearer(header: str | None) -> str | None:
    """Extract the token from an ``Authorization: Bearer <token>`` header."""
    if not header:
        return None
    parts = header.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip() or None
    return None


def authorize(provided_token: str | None) -> bool:
    """Return ``True`` if ``provided_token`` may perform a state change.

    - Auth disabled (no token configured) → ``True`` (legacy mode), with a
      one-time warning so operators know the control plane is unauthenticated.
    - Auth enabled → the provided token must match exactly (constant-time);
      a missing or wrong token → ``False``.
    """
    global _warned_open
    configured = get_dashboard_token()
    if configured is None:
        if not _warned_open:
            logger.warning(
                "Dashboard control plane is UNAUTHENTICATED (%s not set) — any "
                "client can inject observations, drive motors, and answer Kernel "
                "approval requests. Set %s to lock it down.",
                DASHBOARD_TOKEN_ENV,
                DASHBOARD_TOKEN_ENV,
            )
            _warned_open = True
        return True
    if not provided_token:
        return False
    return hmac.compare_digest(provided_token, configured)


def request_is_authorized(method: str, authorization_header: str | None) -> bool:
    """Authorize an HTTP request by method + ``Authorization`` header.

    Non-mutating methods (GET/HEAD/OPTIONS/...) are always allowed; mutating
    methods require a valid bearer token whenever auth is enabled.
    """
    if method.upper() not in MUTATING_METHODS:
        return True
    return authorize(parse_bearer(authorization_header))


def install_auth_middleware(app) -> None:
    """Attach an HTTP middleware that enforces auth on state-mutating requests.

    A middleware (rather than per-route dependencies) guarantees every current
    *and future* mutating route is covered — a new unguarded POST can't slip
    through. FastAPI/Starlette are imported lazily so importing this module for
    tests does not require the web stack.
    """
    from starlette.responses import JSONResponse

    @app.middleware("http")
    async def _enforce_auth(request, call_next):
        if not request_is_authorized(request.method, request.headers.get("authorization")):
            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "detail": f"Set {DASHBOARD_TOKEN_ENV} and send 'Authorization: Bearer <token>'.",
                },
            )
        return await call_next(request)


def authorize_websocket(websocket) -> bool:
    """Authorize a WebSocket connection.

    The ``/ws`` channel carries state-changing commands (approval responses,
    motor guidance, gateway/training control), so when auth is enabled the
    connection must present the token — via ``Authorization: Bearer`` header or
    a ``?token=`` query parameter (browsers can't set WS headers, so the query
    param is the practical path for the UI).
    """
    if not auth_enabled():
        return authorize(None)  # disabled → True (+ one-time warning)
    header_token = parse_bearer(websocket.headers.get("authorization"))
    query_token = websocket.query_params.get("token")
    return authorize(header_token) or authorize(query_token)
