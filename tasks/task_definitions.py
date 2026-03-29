# Copyright (c) 2026. Licensed under the BSD-style license.
# Support Triage Environment — Task Definitions

"""
Defines the three escalating tasks for the support triage environment.

Each task specifies:
  - A unique identifier and human-readable description
  - Instructions shown to the agent
  - Which action fields are evaluated
  - The number of tickets per episode
  - Difficulty level
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from models import TaskDifficulty


@dataclass(frozen=True)
class TaskDefinition:
    """Immutable specification for a single task."""

    task_id: str
    title: str
    difficulty: TaskDifficulty
    description: str
    instructions: str
    evaluated_fields: List[str]
    num_tickets: int
    max_steps_per_ticket: int = 1  # Single action per ticket


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — Easy: Priority Classification
# ─────────────────────────────────────────────────────────────────────────────

TASK_1 = TaskDefinition(
    task_id="task_1",
    title="Priority Classification",
    difficulty=TaskDifficulty.EASY,
    description=(
        "Classify the priority of incoming support tickets. "
        "Read each ticket and assign a priority level from P0 (critical) "
        "to P3 (low) based on severity, business impact, and urgency signals."
    ),
    instructions=(
        "You are a Tier-1 support agent. For each ticket, classify its priority.\n\n"
        "PRIORITY LEVELS:\n"
        "  P0_CRITICAL — Production down, security breach, data loss, "
        "all users affected\n"
        "  P1_HIGH — Major feature broken, significant revenue impact, "
        "account lockout, angry enterprise customer\n"
        "  P2_MEDIUM — Degraded performance, non-critical bugs, "
        "onboarding questions from enterprise, compliance inquiries\n"
        "  P3_LOW — Feature requests, documentation issues, billing "
        "questions, general inquiries, new free-tier users\n\n"
        "ACTION FORMAT: Set the 'priority' field to one of: "
        "P0_CRITICAL, P1_HIGH, P2_MEDIUM, P3_LOW\n\n"
        "SIGNALS TO CONSIDER:\n"
        "  - Customer tier (enterprise > pro > free)\n"
        "  - Business impact (revenue loss, all-users-affected, security)\n"
        "  - Urgency language (URGENT, ASAP, immediately, demanding)\n"
        "  - Scope of the issue (everyone vs. one person)\n"
    ),
    evaluated_fields=["priority"],
    num_tickets=5,
)


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — Medium: Multi-Label Triage
# ─────────────────────────────────────────────────────────────────────────────

TASK_2 = TaskDefinition(
    task_id="task_2",
    title="Multi-Label Triage",
    difficulty=TaskDifficulty.MEDIUM,
    description=(
        "Perform full multi-label triage: classify priority, identify the "
        "category, detect customer sentiment, and route to the correct team."
    ),
    instructions=(
        "You are a Senior Support Agent. For each ticket, provide:\n\n"
        "1. PRIORITY: P0_CRITICAL / P1_HIGH / P2_MEDIUM / P3_LOW\n"
        "2. CATEGORY: BILLING / TECHNICAL / ACCOUNT / FEATURE_REQUEST / "
        "BUG_REPORT / SECURITY / ONBOARDING / GENERAL\n"
        "3. SENTIMENT: ANGRY / FRUSTRATED / NEUTRAL / SATISFIED\n"
        "4. ASSIGNED_TEAM: BILLING_OPS / ENGINEERING / ACCOUNT_MGMT / "
        "PRODUCT / SECURITY_TEAM / CUSTOMER_SUCCESS / TIER1_SUPPORT\n\n"
        "ROUTING GUIDE:\n"
        "  BILLING issues → BILLING_OPS\n"
        "  TECHNICAL / BUG_REPORT → ENGINEERING\n"
        "  ACCOUNT → ACCOUNT_MGMT\n"
        "  FEATURE_REQUEST → PRODUCT\n"
        "  SECURITY → SECURITY_TEAM\n"
        "  ONBOARDING → CUSTOMER_SUCCESS\n"
        "  GENERAL → TIER1_SUPPORT\n\n"
        "SENTIMENT GUIDE:\n"
        "  ANGRY — threats, ALL CAPS, legal mentions, demands\n"
        "  FRUSTRATED — repeated issues, strong disappointment, urgency\n"
        "  NEUTRAL — factual tone, questions, standard requests\n"
        "  SATISFIED — positive language, compliments, feature suggestions\n\n"
        "Set ALL FOUR fields in your action.\n"
    ),
    evaluated_fields=["priority", "category", "sentiment", "assigned_team"],
    num_tickets=8,
)


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Hard: Full Triage + Draft Response
# ─────────────────────────────────────────────────────────────────────────────

TASK_3 = TaskDefinition(
    task_id="task_3",
    title="Full Triage with Response Drafting",
    difficulty=TaskDifficulty.HARD,
    description=(
        "Complete triage including all labels, a triage decision "
        "(respond / escalate / auto-resolve), and a draft customer response. "
        "When escalating, provide a clear reason."
    ),
    instructions=(
        "You are a Lead Support Agent. For each ticket, provide:\n\n"
        "1. PRIORITY: P0_CRITICAL / P1_HIGH / P2_MEDIUM / P3_LOW\n"
        "2. CATEGORY: BILLING / TECHNICAL / ACCOUNT / FEATURE_REQUEST / "
        "BUG_REPORT / SECURITY / ONBOARDING / GENERAL\n"
        "3. SENTIMENT: ANGRY / FRUSTRATED / NEUTRAL / SATISFIED\n"
        "4. ASSIGNED_TEAM: (see routing guide from Task 2)\n"
        "5. DECISION:\n"
        "   RESPOND — Ticket can be handled at current support tier\n"
        "   ESCALATE — Requires manager/engineering/security intervention\n"
        "   AUTO_RESOLVE — Low-effort tickets that can be auto-closed "
        "(feature requests with acknowledgment, typo reports, etc.)\n"
        "6. DRAFT_RESPONSE: A professional response (50–300 chars) that:\n"
        "   - Addresses the customer by name\n"
        "   - Acknowledges their specific issue\n"
        "   - States next steps or resolution\n"
        "   - Matches appropriate tone (empathetic for angry, "
        "helpful for confused)\n"
        "7. ESCALATION_REASON: (required when decision = ESCALATE)\n\n"
        "ESCALATION CRITERIA:\n"
        "  - P0 incidents → always escalate\n"
        "  - Legal threats → escalate\n"
        "  - Security breaches → escalate\n"
        "  - Revenue impact > $10,000 → escalate\n"
        "  - Repeated unresolved issues → escalate\n\n"
        "Set ALL fields in your action.\n"
    ),
    evaluated_fields=[
        "priority", "category", "sentiment", "assigned_team",
        "decision", "draft_response", "escalation_reason",
    ],
    num_tickets=10,
)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

TASK_REGISTRY = {
    "task_1": TASK_1,
    "task_2": TASK_2,
    "task_3": TASK_3,
}


def get_task(task_id: str) -> TaskDefinition:
    """Look up a task by ID. Raises KeyError if not found."""
    if task_id not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task '{task_id}'. "
            f"Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_id]
