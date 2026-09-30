# Copyright (c) 2026. Licensed under the BSD-style license.
# Support Triage Environment — FastAPI Server Application

"""
FastAPI application exposing the OpenEnv interface over HTTP and WebSocket.

Endpoints:
  GET  /             — Root (status check for HF Spaces)
  POST /reset        — Start a new episode
  POST /step         — Execute an action
  GET  /state        — Get current episode state
  GET  /health       — Health check for container orchestration
  GET  /metadata     — Environment metadata
  WS   /ws           — WebSocket for persistent sessions

Each WebSocket connection gets its own environment instance (session isolation).
HTTP endpoints use a default shared instance for simplicity.
"""

from __future__ import annotations

import asyncio
import json
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from models import TriageAction, TriageObservation, TriageState
from server.support_triage_environment import SupportTriageEnvironment


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = Field("task_1", description="Task to run.")
    seed: int = Field(42, description="Random seed.")


class StepRequest(BaseModel):
    action: TriageAction = Field(..., description="Agent's triage action.")


class HealthResponse(BaseModel):
    status: str = "ok"
    environment: str = "support_triage_env"
    version: str = "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# Session management for WebSocket connections
# ─────────────────────────────────────────────────────────────────────────────

_sessions: Dict[str, SupportTriageEnvironment] = {}
_default_env = SupportTriageEnvironment()


def _get_or_create_session(session_id: str) -> SupportTriageEnvironment:
    if session_id not in _sessions:
        _sessions[session_id] = SupportTriageEnvironment()
    return _sessions[session_id]


def _remove_session(session_id: str) -> None:
    env = _sessions.pop(session_id, None)
    if env:
        env.close()


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown hooks."""
    yield
    # Cleanup all sessions on shutdown
    for sid in list(_sessions.keys()):
        _remove_session(sid)


app = FastAPI(
    title="Support Triage Environment",
    description=(
        "OpenEnv-compatible environment for customer support ticket triage. "
        "Agents classify, route, and respond to support tickets across "
        "3 difficulty levels."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for web interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# HTTP Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Root endpoint for HF Spaces status check."""
    return {
        "name": "support_triage_env",
        "status": "running",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/health", "/metadata"],
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint for container orchestration."""
    return HealthResponse()


@app.get("/metadata")
async def metadata():
    """Return environment metadata."""
    return JSONResponse(content=_default_env.get_metadata())


@app.post("/reset")
async def reset(request: Request):
    """
    Reset the environment and start a new episode.

    Accepts optional JSON body with task_id and seed.
    If no body is provided, defaults to task_1 with seed=42.

    Returns the initial observation with the first ticket.
    """
    try:
        # An empty POST uses defaults; malformed JSON must not restart an episode.
        raw = await request.body()
        body = json.loads(raw) if raw else {}
        if not isinstance(body, dict):
            raise ValueError("Reset body must be a JSON object.")
        options = ResetRequest.model_validate(body)
        obs = _default_env.reset(task_id=options.task_id, seed=options.seed)
        obs_dict = obs.model_dump()

        # Return observation at top level (OpenEnv spec format)
        # Also include observation/reward/done/info for backward compat
        return JSONResponse(content={
            **obs_dict,
            "observation": obs_dict,
            "reward": obs.reward,
            "done": obs.done,
            "info": obs.metadata,
        })
    except (ValueError, KeyError) as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc()},
        )


@app.post("/step")
async def step(request: Request):
    """
    Execute an action on the current ticket.

    Returns the next observation with reward and feedback.
    """
    try:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"error": "Request body must be JSON with an 'action' field."},
            )

        # Accept both {"action": {...}} and direct action fields
        if not isinstance(body, dict):
            raise ValueError("Step body must be a JSON object.")
        action_data = body.get("action", body)
        action = TriageAction.model_validate(action_data)
        obs = _default_env.step(action=action)
        obs_dict = obs.model_dump()

        return JSONResponse(content={
            **obs_dict,
            "observation": obs_dict,
            "reward": obs.reward,
            "done": obs.done,
            "info": obs.metadata,
        })
    except (RuntimeError, ValueError) as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)},
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "traceback": traceback.format_exc()},
        )


@app.get("/state")
async def get_state():
    """Return the current episode state."""
    state = _default_env.state
    return JSONResponse(content=state.model_dump())


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket Endpoint — Per-session environment instances
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for persistent sessions.

    Each connection gets its own environment instance. Messages are JSON:
      {"type": "reset", "task_id": "task_1", "seed": 42}
      {"type": "step", "action": {...}}
      {"type": "state"}

    Responses follow the same format as HTTP endpoints.
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    env = _get_or_create_session(session_id)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "error": "Invalid JSON",
                })
                continue

            if not isinstance(msg, dict):
                await websocket.send_json({"type": "error", "error": "Message must be a JSON object."})
                continue

            msg_type = msg.get("type", "")

            if msg_type == "reset":
                task_id = msg.get("task_id", "task_1")
                seed = msg.get("seed", 42)
                try:
                    options = ResetRequest(task_id=task_id, seed=seed)
                    obs = env.reset(task_id=options.task_id, seed=options.seed)
                    obs_dict = obs.model_dump()
                    await websocket.send_json({
                        "type": "reset_result",
                        **obs_dict,
                        "observation": obs_dict,
                        "reward": obs.reward,
                        "done": obs.done,
                        "info": obs.metadata,
                    })
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "error": str(e),
                    })

            elif msg_type == "step":
                action_data = msg.get("action", {})
                try:
                    action = TriageAction.model_validate(action_data)
                    obs = env.step(action=action)
                    obs_dict = obs.model_dump()
                    await websocket.send_json({
                        "type": "step_result",
                        **obs_dict,
                        "observation": obs_dict,
                        "reward": obs.reward,
                        "done": obs.done,
                        "info": obs.metadata,
                    })
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "error": str(e),
                    })

            elif msg_type == "state":
                state = env.state
                await websocket.send_json({
                    "type": "state_result",
                    "state": state.model_dump(),
                })

            else:
                await websocket.send_json({
                    "type": "error",
                    "error": f"Unknown message type: {msg_type}",
                })

    except WebSocketDisconnect:
        _remove_session(session_id)
    except Exception:
        _remove_session(session_id)


# ─────────────────────────────────────────────────────────────────────────────
# Direct run
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Entry point for the server (used by pyproject.toml [project.scripts])."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
