# Copyright (c) 2026. Licensed under the BSD-style license.
# Support Triage Environment — Programmatic Graders

"""
Graders score each agent action against ground-truth labels.

Scoring philosophy:
  - Each field contributes a weighted fraction to the step reward.
  - Partial credit is given for "close" answers (e.g., off-by-one priority).
  - Penalties are applied for clearly bad behavior (empty actions, invalid values).
  - The total step reward is in [0.0, 1.0].
  - Episode score = mean of all step rewards (also in [0.0, 1.0]).

Graders are fully deterministic given (action, ground_truth) — no randomness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from models import (
    Category,
    Priority,
    ReviewDecision,
    Sentiment,
    Team,
    TriageAction,
)
from data.tickets import GroundTruth


# ─────────────────────────────────────────────────────────────────────────────
# Scoring weights per task
# ─────────────────────────────────────────────────────────────────────────────

# Task 1: Only priority matters
TASK_1_WEIGHTS = {"priority": 1.0}

# Task 2: Four-label classification
TASK_2_WEIGHTS = {
    "priority": 0.30,
    "category": 0.30,
    "sentiment": 0.15,
    "assigned_team": 0.25,
}

# Task 3: Full triage + response quality
TASK_3_WEIGHTS = {
    "priority": 0.15,
    "category": 0.15,
    "sentiment": 0.10,
    "assigned_team": 0.15,
    "decision": 0.15,
    "draft_response": 0.20,
    "escalation_reason": 0.10,
}


def _get_weights(task_id: str) -> Dict[str, float]:
    return {
        "task_1": TASK_1_WEIGHTS,
        "task_2": TASK_2_WEIGHTS,
        "task_3": TASK_3_WEIGHTS,
    }[task_id]


# ─────────────────────────────────────────────────────────────────────────────
# Priority scoring — with partial credit for near-misses
# ─────────────────────────────────────────────────────────────────────────────

_PRIORITY_ORDER = [
    Priority.P0_CRITICAL,
    Priority.P1_HIGH,
    Priority.P2_MEDIUM,
    Priority.P3_LOW,
]


def score_priority(predicted: Optional[Priority], expected: Priority) -> float:
    """
    Score priority classification with partial credit.

    - Exact match: 1.0
    - Off by one level: 0.5
    - Off by two levels: 0.2
    - Off by three levels or missing: 0.0
    """
    if predicted is None:
        return 0.0
    if predicted == expected:
        return 1.0

    try:
        pred_idx = _PRIORITY_ORDER.index(predicted)
        exp_idx = _PRIORITY_ORDER.index(expected)
    except ValueError:
        return 0.0

    distance = abs(pred_idx - exp_idx)
    return {1: 0.5, 2: 0.2}.get(distance, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Categorical scoring — exact match with related-category partial credit
# ─────────────────────────────────────────────────────────────────────────────

# Categories that are semantically close (partial credit 0.4)
_RELATED_CATEGORIES = {
    (Category.TECHNICAL, Category.BUG_REPORT),
    (Category.BUG_REPORT, Category.TECHNICAL),
    (Category.BILLING, Category.ACCOUNT),
    (Category.ACCOUNT, Category.BILLING),
    (Category.ONBOARDING, Category.GENERAL),
    (Category.GENERAL, Category.ONBOARDING),
}

# Teams that are semantically close (partial credit 0.4)
_RELATED_TEAMS = {
    (Team.ENGINEERING, Team.SECURITY_TEAM),
    (Team.SECURITY_TEAM, Team.ENGINEERING),
    (Team.CUSTOMER_SUCCESS, Team.TIER1_SUPPORT),
    (Team.TIER1_SUPPORT, Team.CUSTOMER_SUCCESS),
    (Team.BILLING_OPS, Team.ACCOUNT_MGMT),
    (Team.ACCOUNT_MGMT, Team.BILLING_OPS),
}


def score_category(predicted: Optional[Category], expected: Category) -> float:
    """Exact match = 1.0, related category = 0.4, else 0.0."""
    if predicted is None:
        return 0.0
    if predicted == expected:
        return 1.0
    if (predicted, expected) in _RELATED_CATEGORIES:
        return 0.4
    return 0.0


def score_sentiment(
    predicted: Optional[Sentiment], expected: Sentiment,
) -> float:
    """
    Exact match = 1.0, adjacent sentiment = 0.5, else 0.0.
    Adjacency: ANGRY↔FRUSTRATED, FRUSTRATED↔NEUTRAL, NEUTRAL↔SATISFIED.
    """
    if predicted is None:
        return 0.0
    if predicted == expected:
        return 1.0

    _order = [
        Sentiment.ANGRY,
        Sentiment.FRUSTRATED,
        Sentiment.NEUTRAL,
        Sentiment.SATISFIED,
    ]
    try:
        dist = abs(_order.index(predicted) - _order.index(expected))
    except ValueError:
        return 0.0

    return {1: 0.5, 2: 0.15}.get(dist, 0.0)


def score_team(predicted: Optional[Team], expected: Team) -> float:
    """Exact match = 1.0, related team = 0.4, else 0.0."""
    if predicted is None:
        return 0.0
    if predicted == expected:
        return 1.0
    if (predicted, expected) in _RELATED_TEAMS:
        return 0.4
    return 0.0


def score_decision(
    predicted: Optional[ReviewDecision], expected: ReviewDecision,
) -> float:
    """Exact match = 1.0, else 0.0. No partial credit for decisions."""
    if predicted is None:
        return 0.0
    return 1.0 if predicted == expected else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Response quality scoring — checks for required elements
# ─────────────────────────────────────────────────────────────────────────────

# Keyword sets that indicate a response element is present
_RESPONSE_ELEMENT_KEYWORDS = {
    "acknowledge_urgency": [
        "understand", "urgency", "urgent", "immediately", "priority",
        "right away", "seriously", "critical",
    ],
    "acknowledge_impact": [
        "impact", "affecting", "understand", "recognize", "serious",
        "disruption", "sorry",
    ],
    "acknowledge_issue": [
        "sorry", "understand", "looking into", "aware", "investigating",
        "issue", "concern", "apologize",
    ],
    "acknowledge_feedback": [
        "thank", "appreciate", "feedback", "suggestion", "great idea",
        "noted",
    ],
    "escalation_confirmation": [
        "escalat", "senior", "manager", "team lead", "engineering team",
        "specialist", "immediately",
    ],
    "security_protocol": [
        "security", "audit", "invalidat", "rotat", "session", "investigate",
        "log",
    ],
    "estimated_response_time": [
        "within", "hour", "minute", "shortly", "soon", "update",
        "get back", "follow up",
    ],
    "investigation_steps": [
        "investigat", "look into", "check", "review", "diagnos",
        "examin", "analyz",
    ],
    "workaround_suggestion": [
        "meanwhile", "workaround", "temporary", "in the meantime",
        "try", "alternative",
    ],
    "refund_confirmation": [
        "refund", "credit", "reverse", "reimburse", "compensat",
    ],
    "prevention_steps": [
        "prevent", "ensure", "going forward", "monitor", "fix",
        "resolved",
    ],
    "recovery_steps": [
        "recover", "reset", "restore", "backup", "code", "verify",
        "unlock",
    ],
    "identity_verification": [
        "verify", "identity", "confirm", "authenticat", "proof",
        "security question",
    ],
    "documentation_link": [
        "doc", "guide", "article", "resource", "help center",
        "instruction", "tutorial",
    ],
    "setup_guidance": [
        "setup", "configure", "step", "follow", "guide", "instruction",
        "walk",
    ],
    "offer_call": [
        "call", "meeting", "schedule", "discuss", "chat", "connect",
        "speak",
    ],
    "roadmap_reference": [
        "roadmap", "plan", "consider", "backlog", "future", "list",
        "track",
    ],
    "thank_user": [
        "thank", "appreciate", "grateful",
    ],
    "pricing_info": [
        "price", "pricing", "cost", "$", "plan", "subscription",
    ],
    "discount_details": [
        "discount", "saving", "annual", "yearly", "%",
    ],
    "seat_expansion_process": [
        "seat", "add", "expand", "provision", "license",
    ],
    "invoicing_options": [
        "invoice", "billing", "quarterly", "arrangement", "payment",
    ],
    "compliance_info": [
        "complian", "hipaa", "soc", "encrypt", "certif",
    ],
    "empathy": [
        "understand", "sorry", "apologize", "appreciate",
        "frustrat", "inconvenien",
    ],
    "timeline_commitment": [
        "within", "hour", "by end of", "timeline", "commit", "deadline",
    ],
    "encouragement": [
        "great question", "no worries", "happy to help", "don't hesitate",
        "glad", "welcome",
    ],
    "step_by_step_guidance": [
        "step", "first", "then", "next", "follow", "start by",
    ],
}


def score_draft_response(
    draft: Optional[str],
    ground_truth: GroundTruth,
    customer_name: str,
) -> float:
    """
    Score the quality of a draft response.

    Components (each out of 1.0, averaged):
      - Length appropriateness: 50–300 chars → 1.0, outside range → penalty
      - Name personalization: mentions customer name → 0.2 bonus
      - Required elements: fraction of required elements present
    """
    if not draft or not draft.strip():
        return 0.0

    draft_lower = draft.lower().strip()
    scores = []

    # 1. Length check (50–300 chars)
    length = len(draft.strip())
    if 50 <= length <= 300:
        length_score = 1.0
    elif 30 <= length < 50:
        length_score = 0.6
    elif 300 < length <= 500:
        length_score = 0.7
    elif length < 30:
        length_score = 0.2
    else:  # > 500
        length_score = 0.4
    scores.append(length_score)

    # 2. Name personalization
    name_score = 1.0 if customer_name.lower().split()[0] in draft_lower else 0.3
    scores.append(name_score)

    # 3. Required elements coverage
    required = ground_truth.required_response_elements
    if required:
        found = 0
        for element in required:
            keywords = _RESPONSE_ELEMENT_KEYWORDS.get(element, [])
            if any(kw in draft_lower for kw in keywords):
                found += 1
        element_score = found / len(required)
    else:
        element_score = 1.0  # No required elements = full marks
    scores.append(element_score)

    return sum(scores) / len(scores)


def score_escalation_reason(
    reason: Optional[str],
    ground_truth: GroundTruth,
) -> float:
    """
    Score escalation reason quality.

    - If escalation is NOT required and no reason given → 1.0 (correct)
    - If escalation IS required and a non-empty reason given → score by quality
    - If escalation IS required and no reason given → 0.0
    """
    if not ground_truth.escalation_required:
        # No escalation needed — giving no reason is correct
        return 1.0 if not reason else 0.8

    if not reason or not reason.strip():
        return 0.0

    # Reasonable escalation reason: at least 10 chars with some substance
    reason_stripped = reason.strip()
    if len(reason_stripped) < 10:
        return 0.3
    if len(reason_stripped) < 30:
        return 0.6
    return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Grading result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GradeResult:
    """Detailed grading breakdown for a single step."""

    total_score: float  # [0.0, 1.0]
    field_scores: Dict[str, float]  # Per-field scores
    field_weights: Dict[str, float]  # Per-field weights
    feedback: str  # Human-readable feedback
    penalties: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Main grading function
# ─────────────────────────────────────────────────────────────────────────────

def grade_action(
    action: TriageAction,
    ground_truth: GroundTruth,
    task_id: str,
    customer_name: str = "",
) -> GradeResult:
    """
    Grade a single agent action against ground truth.

    Returns a GradeResult with total_score in [0.0, 1.0], per-field
    breakdown, and human-readable feedback.
    """
    weights = _get_weights(task_id)
    field_scores: Dict[str, float] = {}
    penalties: List[str] = []
    feedback_parts: List[str] = []

    # Score each evaluated field
    for fld, w in weights.items():
        if fld == "priority":
            s = score_priority(action.priority, ground_truth.priority)
            field_scores["priority"] = s
            if s == 1.0:
                feedback_parts.append("Priority: correct")
            elif s > 0:
                feedback_parts.append(
                    f"Priority: close (expected {ground_truth.priority.value}, "
                    f"got {action.priority.value if action.priority else 'None'})"
                )
            else:
                feedback_parts.append(
                    f"Priority: incorrect (expected {ground_truth.priority.value})"
                )
                if action.priority is None:
                    penalties.append("Missing priority classification")

        elif fld == "category":
            s = score_category(action.category, ground_truth.category)
            field_scores["category"] = s
            if s == 1.0:
                feedback_parts.append("Category: correct")
            elif s > 0:
                feedback_parts.append(
                    f"Category: partially correct (related to "
                    f"{ground_truth.category.value})"
                )
            else:
                feedback_parts.append(
                    f"Category: incorrect (expected {ground_truth.category.value})"
                )

        elif fld == "sentiment":
            s = score_sentiment(action.sentiment, ground_truth.sentiment)
            field_scores["sentiment"] = s
            if s == 1.0:
                feedback_parts.append("Sentiment: correct")
            elif s > 0:
                feedback_parts.append("Sentiment: close")
            else:
                feedback_parts.append(
                    f"Sentiment: incorrect (expected {ground_truth.sentiment.value})"
                )

        elif fld == "assigned_team":
            s = score_team(action.assigned_team, ground_truth.assigned_team)
            field_scores["assigned_team"] = s
            if s == 1.0:
                feedback_parts.append("Team routing: correct")
            elif s > 0:
                feedback_parts.append("Team routing: close (related team)")
            else:
                feedback_parts.append(
                    f"Team routing: incorrect "
                    f"(expected {ground_truth.assigned_team.value})"
                )

        elif fld == "decision":
            s = score_decision(action.decision, ground_truth.decision)
            field_scores["decision"] = s
            if s == 1.0:
                feedback_parts.append("Decision: correct")
            else:
                feedback_parts.append(
                    f"Decision: incorrect "
                    f"(expected {ground_truth.decision.value})"
                )

        elif fld == "draft_response":
            s = score_draft_response(
                action.draft_response, ground_truth, customer_name,
            )
            field_scores["draft_response"] = s
            if s >= 0.8:
                feedback_parts.append("Response: good quality")
            elif s >= 0.5:
                feedback_parts.append("Response: acceptable but could improve")
            elif s > 0:
                feedback_parts.append("Response: needs improvement")
            else:
                feedback_parts.append("Response: missing or empty")
                penalties.append("Missing draft response")

        elif fld == "escalation_reason":
            s = score_escalation_reason(action.escalation_reason, ground_truth)
            field_scores["escalation_reason"] = s
            if ground_truth.escalation_required and s == 0.0:
                penalties.append("Missing escalation reason for escalated ticket")

    # Compute weighted total
    total = sum(
        field_scores.get(f, 0.0) * w for f, w in weights.items()
    )

    # Apply penalty for completely empty actions (anti-gaming)
    all_none = all(
        getattr(action, f) is None
        for f in weights.keys()
        if hasattr(action, f)
    )
    if all_none:
        total = 0.0
        penalties.append("Empty action — no fields populated")
        feedback_parts = ["No triage action provided. Please fill in the required fields."]

    # Clamp to [0, 1]
    total = max(0.0, min(1.0, total))

    feedback = " | ".join(feedback_parts) if feedback_parts else "No feedback."

    return GradeResult(
        total_score=round(total, 4),
        field_scores={k: round(v, 4) for k, v in field_scores.items()},
        field_weights=weights,
        feedback=feedback,
        penalties=penalties,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Episode-level scoring
# ─────────────────────────────────────────────────────────────────────────────

def compute_episode_score(step_rewards: List[float]) -> float:
    """
    Compute the final episode score as the mean of all step rewards.

    Returns 0.0 if no steps were taken.
    """
    if not step_rewards:
        return 0.0
    return round(sum(step_rewards) / len(step_rewards), 4)
