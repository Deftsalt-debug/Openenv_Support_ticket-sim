# Copyright (c) 2026. Licensed under the BSD-style license.
# Support Triage Environment — FastAPI Server Application

"""
FastAPI application exposing the OpenEnv interface over HTTP and WebSocket.

Endpoints:
  POST /reset          — Start a new episode
  POST /step           — Execute an action
  GET  /state          — Get current episode state
  GET  /health         — Health check for container orchestration
  GET  /metadata       — Environment metadata
  WS   /ws             — WebSocket for persistent sessions

Each WebSocket connection gets its own environment instance (session isolation).
HTTP endpoints use a default shared instance for simplicity.
"""

from __future__ import annotations

import asyncio
import json
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint for container orchestration."""
    return HealthResponse()


@app.get("/metadata")
async def metadata():
    """Return environment metadata."""
    return JSONResponse(content=_default_env.get_metadata())


@app.post("/reset")
async def reset(request: ResetRequest):
    """
    Reset the environment and start a new episode.

    Returns the initial observation with the first ticket.
    """
    try:
        obs = _default_env.reset(
            task_id=request.task_id,
            seed=request.seed,
        )
        return JSONResponse(content={
            "observation": obs.model_dump(),
            "reward": obs.reward,
            "done": obs.done,
            "info": obs.metadata,
        })
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e), "traceback": traceback.format_exc()},
        )


@app.post("/step")
async def step(request: StepRequest):
    """
    Execute an action on the current ticket.

    Returns the next observation with reward and feedback.
    """
    try:
        obs = _default_env.step(action=request.action)
        return JSONResponse(content={
            "observation": obs.model_dump(),
            "reward": obs.reward,
            "done": obs.done,
            "info": obs.metadata,
        })
    except RuntimeError as e:
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

            msg_type = msg.get("type", "")

            if msg_type == "reset":
                task_id = msg.get("task_id", "task_1")
                seed = msg.get("seed", 42)
                try:
                    obs = env.reset(task_id=task_id, seed=seed)
                    await websocket.send_json({
                        "type": "reset_result",
                        "observation": obs.model_dump(),
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
                    action = TriageAction(**action_data)
                    obs = env.step(action=action)
                    await websocket.send_json({
                        "type": "step_result",
                        "observation": obs.model_dump(),
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
