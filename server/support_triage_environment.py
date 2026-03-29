# Copyright (c) 2026. Licensed under the BSD-style license.
# Support Triage Environment — Core Environment Logic

"""
Core environment implementing the OpenEnv interface:
  - reset(task_id, seed) → TriageObservation
  - step(action)         → TriageObservation
  - state()              → TriageState

The environment manages episode lifecycle, ticket queues, grading,
and reward computation. Each episode presents a sequence of tickets
for the agent to triage, with rewards computed per-step by the grader.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from models import (
    StepResult,
    TriageAction,
    TriageObservation,
    TriageState,
)
from data.tickets import LabeledTicket, generate_ticket_pool
from tasks.task_definitions import TaskDefinition, get_task
from tasks.graders import grade_action, compute_episode_score


class SupportTriageEnvironment:
    """
    OpenEnv-compatible environment for customer support ticket triage.

    Lifecycle:
      1. reset(task_id="task_1", seed=42) → first observation
      2. step(action) → next observation + reward + done
      3. Repeat step() until done=True
      4. state() → full episode state at any point

    The environment is stateful per-session. Each reset() starts a new
    episode with a fresh ticket queue.
    """

    # Class-level flag for OpenEnv concurrent-safe marking
    CONCURRENCY_SAFE = True

    def __init__(self) -> None:
        """Initialize with empty state. Call reset() before step()."""
        self._episode_id: str = ""
        self._task: Optional[TaskDefinition] = None
        self._ticket_queue: List[LabeledTicket] = []
        self._current_index: int = 0
        self._step_count: int = 0
        self._step_rewards: List[float] = []
        self._action_history: List[Dict[str, Any]] = []
        self._done: bool = True
        self._cumulative_reward: float = 0.0
        self._start_time: str = datetime.now(timezone.utc).isoformat()
        self._seed: int = 42

    # ─────────────────────────────────────────────────────────────────────
    # reset() — Start a new episode
    # ─────────────────────────────────────────────────────────────────────

    def reset(
        self,
        task_id: str = "task_1",
        seed: int = 42,
        **kwargs: Any,
    ) -> TriageObservation:
        """
        Initialize a new episode.

        Args:
            task_id: Which task to run ("task_1", "task_2", "task_3").
            seed: Random seed for ticket generation (reproducibility).

        Returns:
            Initial observation with the first ticket and task instructions.
        """
        self._task = get_task(task_id)
        self._seed = seed
        self._episode_id = str(uuid.uuid4())
        self._ticket_queue = generate_ticket_pool(
            task_id=task_id,
            seed=seed,
            num_tickets=self._task.num_tickets,
        )
        self._current_index = 0
        self._step_count = 0
        self._step_rewards = []
        self._action_history = []
        self._done = False
        self._cumulative_reward = 0.0
        self._start_time = datetime.now(timezone.utc).isoformat()

        # Return the first ticket
        first_ticket = self._ticket_queue[0].ticket

        return TriageObservation(
            done=False,
            reward=None,  # No reward on reset
            metadata={
                "episode_id": self._episode_id,
                "task_id": task_id,
                "task_title": self._task.title,
                "difficulty": self._task.difficulty.value,
                "total_tickets": len(self._ticket_queue),
                "seed": seed,
            },
            ticket=first_ticket,
            task_id=task_id,
            task_instructions=self._task.instructions,
            tickets_remaining=len(self._ticket_queue) - 1,
            step_feedback=None,
            cumulative_reward=0.0,
        )

    # ─────────────────────────────────────────────────────────────────────
    # step() — Process agent action and advance
    # ─────────────────────────────────────────────────────────────────────

    def step(self, action: TriageAction) -> TriageObservation:
        """
        Execute an agent action on the current ticket.

        Args:
            action: The agent's triage action (TriageAction).

        Returns:
            Observation with reward, feedback, and next ticket (or done).

        Raises:
            RuntimeError: If called before reset() or after episode is done.
        """
        if self._done:
            raise RuntimeError(
                "Episode is done. Call reset() to start a new episode."
            )
        if self._task is None:
            raise RuntimeError("No task loaded. Call reset() first.")

        # Get current ticket and its ground truth
        current_labeled = self._ticket_queue[self._current_index]
        ground_truth = current_labeled.ground_truth
        customer_name = current_labeled.ticket.customer_name

        # Grade the action
        grade_result = grade_action(
            action=action,
            ground_truth=ground_truth,
            task_id=self._task.task_id,
            customer_name=customer_name,
        )

        # Update state
        step_reward = grade_result.total_score
        self._step_rewards.append(step_reward)
        self._cumulative_reward += step_reward
        self._step_count += 1
        self._current_index += 1

        # Log the action
        self._action_history.append({
            "step": self._step_count,
            "ticket_id": current_labeled.ticket.ticket_id,
            "action": action.model_dump(exclude_none=True),
            "reward": step_reward,
            "field_scores": grade_result.field_scores,
            "feedback": grade_result.feedback,
            "penalties": grade_result.penalties,
        })

        # Check if episode is done
        self._done = self._current_index >= len(self._ticket_queue)

        # Build next observation
        if self._done:
            episode_score = compute_episode_score(self._step_rewards)
            return TriageObservation(
                done=True,
                reward=step_reward,
                metadata={
                    "episode_id": self._episode_id,
                    "task_id": self._task.task_id,
                    "episode_score": episode_score,
                    "step_count": self._step_count,
                    "total_tickets": len(self._ticket_queue),
                    "field_scores": grade_result.field_scores,
                    "grade_feedback": grade_result.feedback,
                    "penalties": grade_result.penalties,
                },
                ticket=None,
                task_id=self._task.task_id,
                task_instructions="",
                tickets_remaining=0,
                step_feedback=(
                    f"Step {self._step_count}: {grade_result.feedback}\n"
                    f"Episode complete! Final score: {episode_score:.2%}"
                ),
                cumulative_reward=self._cumulative_reward,
            )
        else:
            next_ticket = self._ticket_queue[self._current_index].ticket
            return TriageObservation(
                done=False,
                reward=step_reward,
                metadata={
                    "episode_id": self._episode_id,
                    "task_id": self._task.task_id,
                    "step": self._step_count,
                    "field_scores": grade_result.field_scores,
                    "grade_feedback": grade_result.feedback,
                    "penalties": grade_result.penalties,
                },
                ticket=next_ticket,
                task_id=self._task.task_id,
                task_instructions=self._task.instructions,
                tickets_remaining=(
                    len(self._ticket_queue) - self._current_index - 1
                ),
                step_feedback=(
                    f"Step {self._step_count}: {grade_result.feedback}"
                ),
                cumulative_reward=self._cumulative_reward,
            )

    # ─────────────────────────────────────────────────────────────────────
    # state() — Introspect current episode state
    # ─────────────────────────────────────────────────────────────────────

    @property
    def state(self) -> TriageState:
        """
        Return the current episode state.

        Provides full visibility into the episode for debugging,
        logging, and RL training infrastructure.
        """
        return TriageState(
            episode_id=self._episode_id or "no-episode",
            task_id=self._task.task_id if self._task else "none",
            step_count=self._step_count,
            total_tickets=len(self._ticket_queue),
            tickets_completed=self._current_index,
            cumulative_reward=round(self._cumulative_reward, 4),
            max_possible_reward=float(len(self._ticket_queue)),
            done=self._done,
            action_history=self._action_history,
            start_time=self._start_time,
            elapsed_steps=self._step_count,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Metadata — for OpenEnv spec
    # ─────────────────────────────────────────────────────────────────────

    def get_metadata(self) -> Dict[str, Any]:
        """Return environment metadata for the OpenEnv spec."""
        return {
            "name": "support_triage_env",
            "version": "1.0.0",
            "description": (
                "Customer support ticket triage environment. "
                "Agents classify, route, and respond to support tickets."
            ),
            "tasks": ["task_1", "task_2", "task_3"],
            "action_type": "TriageAction",
            "observation_type": "TriageObservation",
            "state_type": "TriageState",
        }

    def close(self) -> None:
        """Clean up resources (no-op for this stateless env)."""
        pass
