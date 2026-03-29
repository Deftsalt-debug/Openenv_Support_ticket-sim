# Copyright (c) 2026. Licensed under the BSD-style license.
# Support Triage Environment — Client Implementation

"""
Client for interacting with the Support Triage Environment.

Supports both HTTP and WebSocket modes:
  - HTTP: Simple request/response, stateless between calls
  - WebSocket: Persistent session with dedicated environment instance

Usage:
    # HTTP mode
    client = SupportTriageClient(base_url="http://localhost:8000")
    obs = client.reset(task_id="task_1", seed=42)
    obs = client.step(TriageAction(priority=Priority.P1_HIGH))
    state = client.state()

    # Direct mode (no server, in-process)
    client = SupportTriageClient.local()
    obs = client.reset(task_id="task_1")
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests

from models import (
    TriageAction,
    TriageObservation,
    TriageState,
    StepResult,
    TicketData,
)
from server.support_triage_environment import SupportTriageEnvironment


class SupportTriageClient:
    """
    Synchronous client for the Support Triage Environment.

    Operates in two modes:
      - Remote (HTTP): connects to a running FastAPI server
      - Local (in-process): runs the environment directly

    For WebSocket-based usage with OpenEnv, use the async EnvClient pattern.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        local_env: Optional[SupportTriageEnvironment] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/") if base_url else None
        self._env = local_env
        if not base_url and not local_env:
            raise ValueError("Provide either base_url or local_env.")

    @classmethod
    def local(cls) -> SupportTriageClient:
        """Create a client that runs the environment in-process."""
        return cls(local_env=SupportTriageEnvironment())

    @classmethod
    def remote(cls, base_url: str) -> SupportTriageClient:
        """Create a client that connects to a remote server."""
        return cls(base_url=base_url)

    # ─────────────────────────────────────────────────────────────────────
    # Core API
    # ─────────────────────────────────────────────────────────────────────

    def reset(
        self,
        task_id: str = "task_1",
        seed: int = 42,
    ) -> TriageObservation:
        """
        Reset the environment and start a new episode.

        Args:
            task_id: Task to run ("task_1", "task_2", "task_3").
            seed: Random seed for reproducibility.

        Returns:
            Initial observation with the first ticket.
        """
        if self._env:
            return self._env.reset(task_id=task_id, seed=seed)

        resp = requests.post(
            f"{self._base_url}/reset",
            json={"task_id": task_id, "seed": seed},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return TriageObservation(**data["observation"])

    def step(self, action: TriageAction) -> TriageObservation:
        """
        Execute an action on the current ticket.

        Args:
            action: The agent's triage action.

        Returns:
            Observation with reward, feedback, and next ticket.
        """
        if self._env:
            return self._env.step(action=action)

        resp = requests.post(
            f"{self._base_url}/step",
            json={"action": action.model_dump(exclude_none=True)},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return TriageObservation(**data["observation"])

    def state(self) -> TriageState:
        """
        Get the current episode state.

        Returns:
            Full episode state with history and metrics.
        """
        if self._env:
            return self._env.state

        resp = requests.get(
            f"{self._base_url}/state",
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return TriageState(**data)

    def close(self) -> None:
        """Clean up resources."""
        if self._env:
            self._env.close()

    # Context manager support
    def __enter__(self) -> SupportTriageClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
