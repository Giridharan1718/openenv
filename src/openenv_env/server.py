"""
server.py — FastAPI application for the CodeDebug OpenEnv environment.

Exposes both the standard OpenEnv endpoints and optional custom REST endpoints.

Standard OpenEnv routes
-----------------------
POST /reset     → Observation
POST /step      → StepResult
GET  /state     → EnvironmentState

Custom REST API (v1)
--------------------
GET  /api/v1/health  → {"status": "ok", ...}
POST /api/v1/reset   → Observation
POST /api/v1/step    → StepResult
GET  /api/v1/state   → EnvironmentState

Session isolation
-----------------
The environment is *stateful*, so each caller is given its own episode.
Clients select their session with the ``X-Session-Id`` header; requests that
omit it share a single ``default`` session (reproducing single-client use).
This keeps concurrent clients, the HF Space, evals, and notebooks from
clobbering one another's episodes.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .environment import CodeDebugEnvironment
from .models import Action, EnvironmentState, Observation, StepResult

DEFAULT_SESSION = "default"
SESSION_HEADER = "X-Session-Id"


# ---------------------------------------------------------------------------
# Request / Response helpers
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str | None = None


class StepRequest(BaseModel):
    action: Action


# ---------------------------------------------------------------------------
# Session store — one environment per session id
# ---------------------------------------------------------------------------

class _SessionStore:
    """Lazily create and hold one environment per session id (thread-safe)."""

    def __init__(self, factory, initial: dict[str, CodeDebugEnvironment] | None = None) -> None:
        self._factory = factory
        self._lock = threading.Lock()
        self._envs: dict[str, CodeDebugEnvironment] = dict(initial or {})

    def get(self, session_id: str) -> CodeDebugEnvironment:
        with self._lock:
            env = self._envs.get(session_id)
            if env is None:
                env = self._factory()
                self._envs[session_id] = env
            return env


# ---------------------------------------------------------------------------
# App factory (OpenEnv create_app pattern)
# ---------------------------------------------------------------------------

def create_app(env: CodeDebugEnvironment | None = None) -> FastAPI:
    """
    Build and return the FastAPI application.

    Parameters
    ----------
    env : CodeDebugEnvironment, optional
        Seed the ``default`` session with this instance (useful for testing).
        Other sessions, and the default session when this is omitted, receive
        their own fresh environment.
    """
    initial = {DEFAULT_SESSION: env} if env is not None else None
    store = _SessionStore(CodeDebugEnvironment, initial=initial)

    _start_time = time.time()

    app = FastAPI(
        title="CodeDebug — OpenEnv Environment",
        description=(
            "A Meta PyTorch OpenEnv-compliant environment for training agents "
            "to debug Python code. Implements standard reset/step/state lifecycle "
            "plus custom v1 REST endpoints. Each client session gets its own "
            f"episode; select one with the {SESSION_HEADER} header."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    def session_env(
        x_session_id: str | None = Header(default=None, alias=SESSION_HEADER),
    ) -> CodeDebugEnvironment:
        """Resolve the environment for the caller's session."""
        return store.get(x_session_id or DEFAULT_SESSION)

    # -----------------------------------------------------------------------
    # Standard OpenEnv endpoints
    # -----------------------------------------------------------------------

    @app.post("/reset", response_model=Observation, tags=["OpenEnv"])
    async def openenv_reset(
        request: ResetRequest = ResetRequest(),
        current: CodeDebugEnvironment = Depends(session_env),
    ) -> Observation:
        """Reset this session's environment and return the initial observation."""
        return current.reset(task_id=request.task_id)

    @app.post("/step", response_model=StepResult, tags=["OpenEnv"])
    async def openenv_step(
        request: StepRequest,
        current: CodeDebugEnvironment = Depends(session_env),
    ) -> StepResult:
        """Submit an action and advance this session's environment by one step."""
        try:
            return current.step(request.action)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/state", response_model=EnvironmentState, tags=["OpenEnv"])
    async def openenv_state(
        current: CodeDebugEnvironment = Depends(session_env),
    ) -> EnvironmentState:
        """Return a read-only snapshot of this session's environment state."""
        return current.state()

    # -----------------------------------------------------------------------
    # Custom REST API v1
    # -----------------------------------------------------------------------

    @app.get("/api/v1/health", tags=["Custom API v1"])
    async def health() -> dict[str, Any]:
        """Health check — confirms the server is running."""
        return {
            "status": "ok",
            "uptime_seconds": round(time.time() - _start_time, 2),
            "version": "1.0.0",
            "environment": "CodeDebugEnvironment",
            "sessions": len(store._envs),
        }

    @app.post("/api/v1/reset", response_model=Observation, tags=["Custom API v1"])
    async def v1_reset(
        request: ResetRequest = ResetRequest(),
        current: CodeDebugEnvironment = Depends(session_env),
    ) -> Observation:
        """Reset the environment (versioned endpoint)."""
        return current.reset(task_id=request.task_id)

    @app.post("/api/v1/step", response_model=StepResult, tags=["Custom API v1"])
    async def v1_step(
        request: StepRequest,
        current: CodeDebugEnvironment = Depends(session_env),
    ) -> StepResult:
        """Submit an action (versioned endpoint)."""
        try:
            return current.step(request.action)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/v1/state", response_model=EnvironmentState, tags=["Custom API v1"])
    async def v1_state(
        current: CodeDebugEnvironment = Depends(session_env),
    ) -> EnvironmentState:
        """Get environment state (versioned endpoint)."""
        return current.state()

    # -----------------------------------------------------------------------
    # Root redirect to docs
    # -----------------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse({"message": "CodeDebug OpenEnv — visit /docs for API reference."})

    return app


# ---------------------------------------------------------------------------
# Entrypoint: python -m openenv_env.server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(
        "openenv_env.server:create_app",
        host="0.0.0.0",
        port=port,
        factory=True,
        reload=False,
    )
