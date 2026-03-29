# Copyright (c) 2026. Licensed under the BSD-style license.
# Support Triage Environment — Typed Pydantic Models
# Follows OpenEnv 0.1 Spec (RFC 002): Action, Observation, State

"""
Typed Pydantic models for the Support Triage Environment.

These models define the full action/observation/state contract that agents
interact with. They are used by both the server (environment logic) and the
client (agent interface), ensuring type safety across the wire.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enums — Canonical label spaces
# ─────────────────────────────────────────────────────────────────────────────

class Priority(str, Enum):
    """Ticket priority levels (P0 = most urgent)."""
    P0_CRITICAL = "P0_CRITICAL"
    P1_HIGH = "P1_HIGH"
    P2_MEDIUM = "P2_MEDIUM"
    P3_LOW = "P3_LOW"


class Category(str, Enum):
    """Support ticket category taxonomy."""
    BILLING = "BILLING"
    TECHNICAL = "TECHNICAL"
    ACCOUNT = "ACCOUNT"
    FEATURE_REQUEST = "FEATURE_REQUEST"
    BUG_REPORT = "BUG_REPORT"
    SECURITY = "SECURITY"
    ONBOARDING = "ONBOARDING"
    GENERAL = "GENERAL"


class Sentiment(str, Enum):
    """Customer sentiment detected in the ticket."""
    ANGRY = "ANGRY"
    FRUSTRATED = "FRUSTRATED"
    NEUTRAL = "NEUTRAL"
    SATISFIED = "SATISFIED"


class Team(str, Enum):
    """Internal team the ticket should be routed to."""
    BILLING_OPS = "BILLING_OPS"
    ENGINEERING = "ENGINEERING"
    ACCOUNT_MGMT = "ACCOUNT_MGMT"
    PRODUCT = "PRODUCT"
    SECURITY_TEAM = "SECURITY_TEAM"
    CUSTOMER_SUCCESS = "CUSTOMER_SUCCESS"
    TIER1_SUPPORT = "TIER1_SUPPORT"


class ReviewDecision(str, Enum):
    """Final triage decision for the ticket."""
    RESPOND = "RESPOND"
    ESCALATE = "ESCALATE"
    AUTO_RESOLVE = "AUTO_RESOLVE"


class TaskDifficulty(str, Enum):
    """Task difficulty levels."""
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


# ─────────────────────────────────────────────────────────────────────────────
# Action — What the agent sends to the environment each step
# ─────────────────────────────────────────────────────────────────────────────

class TriageAction(BaseModel):
    """
    Agent's triage action for a support ticket.

    Depending on the active task, the agent may only need to fill certain
    fields. Unused fields should be left as None.

    - Task 1 (Easy):   priority only
    - Task 2 (Medium): priority + category + sentiment + assigned_team
    - Task 3 (Hard):   all fields including draft_response and decision
    """

    # Task 1: Priority classification
    priority: Optional[Priority] = Field(
        None,
        description="Assigned priority level (P0–P3). Required for all tasks.",
    )

    # Task 2: Multi-label triage
    category: Optional[Category] = Field(
        None,
        description="Ticket category. Required for Task 2+.",
    )
    sentiment: Optional[Sentiment] = Field(
        None,
        description="Customer sentiment. Required for Task 2+.",
    )
    assigned_team: Optional[Team] = Field(
        None,
        description="Team to route the ticket to. Required for Task 2+.",
    )

    # Task 3: Full triage with response
    draft_response: Optional[str] = Field(
        None,
        description=(
            "Draft response to the customer (50–300 chars). Required for Task 3."
        ),
    )
    decision: Optional[ReviewDecision] = Field(
        None,
        description=(
            "Final triage decision: RESPOND, ESCALATE, or AUTO_RESOLVE. "
            "Required for Task 3."
        ),
    )
    escalation_reason: Optional[str] = Field(
        None,
        description=(
            "Reason for escalation (required when decision=ESCALATE)."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Observation — What the environment returns after each step / reset
# ─────────────────────────────────────────────────────────────────────────────

class TicketData(BaseModel):
    """A single customer support ticket presented to the agent."""

    ticket_id: str = Field(..., description="Unique ticket identifier.")
    subject: str = Field(..., description="Email / ticket subject line.")
    body: str = Field(..., description="Full ticket message body.")
    customer_name: str = Field(..., description="Customer display name.")
    customer_tier: str = Field(
        ..., description="Customer tier: free, pro, enterprise."
    )
    timestamp: str = Field(
        ..., description="ISO-8601 timestamp of ticket creation."
    )
    channel: str = Field(
        ..., description="Channel: email, chat, phone, web_form."
    )
    previous_tickets: int = Field(
        0, description="Number of previous tickets from this customer."
    )
    account_age_days: int = Field(
        0, description="Customer account age in days."
    )
    has_attachments: bool = Field(
        False, description="Whether the ticket includes attachments."
    )


class TriageObservation(BaseModel):
    """
    Observation returned by the environment after reset() or step().

    After reset(): contains the first ticket and task instructions.
    After step():  contains feedback on the last action + the next ticket
                   (or done=True if the episode is complete).

    Matches OpenEnv Observation base contract:
      done, reward, metadata are always present.
    """

    # OpenEnv base fields
    done: bool = Field(False, description="Whether the episode has ended.")
    reward: Optional[float] = Field(
        None,
        description="Reward for the last action (0.0–1.0 per step).",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary metadata (task info, grader details, etc.).",
    )

    # Environment-specific fields
    ticket: Optional[TicketData] = Field(
        None,
        description="Current ticket to triage (None when done=True).",
    )
    task_id: str = Field(
        "",
        description="Active task identifier (task_1, task_2, task_3).",
    )
    task_instructions: str = Field(
        "",
        description="Human-readable instructions for the current task.",
    )
    tickets_remaining: int = Field(
        0,
        description="Number of tickets left in this episode.",
    )
    step_feedback: Optional[str] = Field(
        None,
        description="Natural-language feedback on the agent's last action.",
    )
    cumulative_reward: float = Field(
        0.0,
        description="Total reward accumulated so far in this episode.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# State — Internal episode state exposed via state()
# ─────────────────────────────────────────────────────────────────────────────

class TriageState(BaseModel):
    """
    Full episode state returned by state().

    Provides visibility into episode metadata for debugging, logging,
    and RL training infrastructure.
    """

    episode_id: str = Field(..., description="Unique episode identifier.")
    task_id: str = Field(..., description="Active task identifier.")
    step_count: int = Field(0, description="Steps taken so far.")
    total_tickets: int = Field(0, description="Total tickets in episode.")
    tickets_completed: int = Field(
        0, description="Tickets processed so far."
    )
    cumulative_reward: float = Field(
        0.0, description="Accumulated reward."
    )
    max_possible_reward: float = Field(
        0.0, description="Maximum achievable reward for this episode."
    )
    done: bool = Field(False, description="Whether the episode has ended.")
    action_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Log of (action, reward, feedback) tuples.",
    )
    start_time: str = Field(
        default_factory=lambda: datetime.now(tz=None).isoformat(),
        description="Episode start time (ISO-8601).",
    )
    elapsed_steps: int = Field(0, description="Alias for step_count.")


# ─────────────────────────────────────────────────────────────────────────────
# Step result wrapper (mirrors OpenEnv StepResult)
# ─────────────────────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    """Wire format for step() responses."""

    observation: TriageObservation
    reward: Optional[float] = None
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)
