# Copyright (c) 2026. Licensed under the BSD-style license.
# Support Triage Environment — Synthetic Ticket Data Generator

"""
Generates realistic customer support tickets with known ground-truth labels.

Every ticket has a deterministic correct answer for: priority, category,
sentiment, team routing, decision, and (for Task 3) a set of required
response elements. This enables fully programmatic grading.

Tickets are organized into pools by difficulty to support the 3-task
progression (easy → medium → hard).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from models import (
    Category,
    Priority,
    ReviewDecision,
    Sentiment,
    Team,
    TicketData,
)


@dataclass
class GroundTruth:
    """Known-correct labels for a ticket. Used by graders."""

    priority: Priority
    category: Category
    sentiment: Sentiment
    assigned_team: Team
    decision: ReviewDecision
    escalation_required: bool = False
    required_response_elements: List[str] = field(default_factory=list)
    explanation: str = ""


@dataclass
class LabeledTicket:
    """A ticket bundled with its ground-truth labels."""

    ticket: TicketData
    ground_truth: GroundTruth


# ─────────────────────────────────────────────────────────────────────────────
# Routing rules — deterministic team assignment based on (category, priority)
# ─────────────────────────────────────────────────────────────────────────────

ROUTING_TABLE: Dict[Category, Team] = {
    Category.BILLING: Team.BILLING_OPS,
    Category.TECHNICAL: Team.ENGINEERING,
    Category.ACCOUNT: Team.ACCOUNT_MGMT,
    Category.FEATURE_REQUEST: Team.PRODUCT,
    Category.BUG_REPORT: Team.ENGINEERING,
    Category.SECURITY: Team.SECURITY_TEAM,
    Category.ONBOARDING: Team.CUSTOMER_SUCCESS,
    Category.GENERAL: Team.TIER1_SUPPORT,
}


def _make_ticket_id(index: int, seed: int) -> str:
    """Deterministic ticket ID from index + seed."""
    h = hashlib.md5(f"{seed}-{index}".encode()).hexdigest()[:8]
    return f"TKT-{h.upper()}"


# ─────────────────────────────────────────────────────────────────────────────
# Ticket templates — realistic scenarios with clear ground-truth signals
# ─────────────────────────────────────────────────────────────────────────────

_TICKET_TEMPLATES: List[dict] = [
    # ── P0 CRITICAL ──────────────────────────────────────────────────────
    {
        "subject": "URGENT: Production API returning 500 errors for all customers",
        "body": (
            "Hi Support,\n\n"
            "Our entire production environment is down. Every API call to "
            "/v2/transactions returns a 500 Internal Server Error. This "
            "started at 14:32 UTC. Our payment processing is completely "
            "halted and we are losing revenue by the minute. We have 50,000+ "
            "active users affected. We need an immediate fix or rollback.\n\n"
            "Error trace: ConnectionPoolExhausted at db-primary-01\n\n"
            "This is a Sev-0 for us. Please escalate immediately.\n\n"
            "— Jordan Rivera, VP Engineering, Acme Corp"
        ),
        "customer_name": "Jordan Rivera",
        "customer_tier": "enterprise",
        "channel": "email",
        "previous_tickets": 12,
        "account_age_days": 890,
        "has_attachments": True,
        "ground_truth": GroundTruth(
            priority=Priority.P0_CRITICAL,
            category=Category.TECHNICAL,
            sentiment=Sentiment.ANGRY,
            assigned_team=Team.ENGINEERING,
            decision=ReviewDecision.ESCALATE,
            escalation_required=True,
            required_response_elements=[
                "acknowledge_urgency", "escalation_confirmation",
                "estimated_response_time",
            ],
            explanation="Production outage affecting all users = P0 + escalate",
        ),
    },
    {
        "subject": "CRITICAL: Unauthorized access detected on our account",
        "body": (
            "We have detected unauthorized API calls originating from IP "
            "addresses in Eastern Europe. Someone appears to have compromised "
            "our API keys. We've seen 15,000 anomalous requests in the last "
            "hour. Our security team has already rotated the keys on our end "
            "but we need you to invalidate all existing sessions and provide "
            "an audit log immediately.\n\n"
            "Account ID: ENT-4491\n\n"
            "This is a critical security incident. Please respond ASAP.\n\n"
            "— Priya Nair, CISO, DataFlow Inc."
        ),
        "customer_name": "Priya Nair",
        "customer_tier": "enterprise",
        "channel": "phone",
        "previous_tickets": 5,
        "account_age_days": 1200,
        "has_attachments": True,
        "ground_truth": GroundTruth(
            priority=Priority.P0_CRITICAL,
            category=Category.SECURITY,
            sentiment=Sentiment.ANGRY,
            assigned_team=Team.SECURITY_TEAM,
            decision=ReviewDecision.ESCALATE,
            escalation_required=True,
            required_response_elements=[
                "acknowledge_urgency", "security_protocol",
                "escalation_confirmation",
            ],
            explanation="Active security breach = P0 + Security Team + escalate",
        ),
    },
    # ── P1 HIGH ──────────────────────────────────────────────────────────
    {
        "subject": "Cannot process payments — checkout broken since morning",
        "body": (
            "Hello,\n\n"
            "Since about 9 AM today, our checkout flow is completely broken. "
            "Customers see a spinning loader and then get a 'Payment Failed' "
            "error. We've confirmed it's not on our Stripe side — the API "
            "calls to your endpoint /v2/checkout/initiate are timing out "
            "after 30 seconds.\n\n"
            "This is blocking all sales. We're a Pro plan customer and this "
            "is severely impacting our business. We've lost approximately "
            "$12,000 in sales so far today.\n\n"
            "Please investigate urgently.\n\n"
            "— Sam Chen, CTO"
        ),
        "customer_name": "Sam Chen",
        "customer_tier": "pro",
        "channel": "chat",
        "previous_tickets": 8,
        "account_age_days": 450,
        "has_attachments": False,
        "ground_truth": GroundTruth(
            priority=Priority.P1_HIGH,
            category=Category.BUG_REPORT,
            sentiment=Sentiment.FRUSTRATED,
            assigned_team=Team.ENGINEERING,
            decision=ReviewDecision.ESCALATE,
            escalation_required=True,
            required_response_elements=[
                "acknowledge_impact", "investigation_steps",
                "estimated_response_time",
            ],
            explanation="Payment flow broken for Pro customer = P1 + Engineering",
        ),
    },
    {
        "subject": "Billing discrepancy — charged double this month",
        "body": (
            "Hi there,\n\n"
            "I just noticed that my credit card was charged twice for my "
            "monthly subscription — once on March 1st ($99) and again on "
            "March 3rd ($99). My plan is the Pro Monthly at $99/month.\n\n"
            "I need a refund for the duplicate charge as soon as possible. "
            "This is the second time this has happened in the last 4 months "
            "and it's really frustrating.\n\n"
            "Invoice numbers: INV-20260301-4421 and INV-20260303-4422\n\n"
            "— Maria Gonzalez"
        ),
        "customer_name": "Maria Gonzalez",
        "customer_tier": "pro",
        "channel": "email",
        "previous_tickets": 6,
        "account_age_days": 380,
        "has_attachments": True,
        "ground_truth": GroundTruth(
            priority=Priority.P1_HIGH,
            category=Category.BILLING,
            sentiment=Sentiment.FRUSTRATED,
            assigned_team=Team.BILLING_OPS,
            decision=ReviewDecision.RESPOND,
            escalation_required=False,
            required_response_elements=[
                "acknowledge_issue", "refund_confirmation",
                "prevention_steps",
            ],
            explanation="Repeat billing error with clear invoices = P1 + Billing Ops",
        ),
    },
    {
        "subject": "Account locked out — MFA not working after phone change",
        "body": (
            "I recently changed my phone and now my MFA codes aren't working. "
            "I'm completely locked out of my account. I have a critical "
            "deployment scheduled for tomorrow morning and I need access "
            "restored ASAP.\n\n"
            "I can provide my backup codes if needed, but the backup code "
            "recovery page is also returning an error.\n\n"
            "Username: alex.kim@techstartup.io\n\n"
            "— Alex Kim"
        ),
        "customer_name": "Alex Kim",
        "customer_tier": "pro",
        "channel": "chat",
        "previous_tickets": 2,
        "account_age_days": 200,
        "has_attachments": False,
        "ground_truth": GroundTruth(
            priority=Priority.P1_HIGH,
            category=Category.ACCOUNT,
            sentiment=Sentiment.FRUSTRATED,
            assigned_team=Team.ACCOUNT_MGMT,
            decision=ReviewDecision.RESPOND,
            escalation_required=False,
            required_response_elements=[
                "acknowledge_urgency", "recovery_steps",
                "identity_verification",
            ],
            explanation="Account lockout with upcoming deadline = P1 + Account Mgmt",
        ),
    },
    # ── P2 MEDIUM ────────────────────────────────────────────────────────
    {
        "subject": "Dashboard loading slowly for the past week",
        "body": (
            "Hey team,\n\n"
            "Our analytics dashboard has been noticeably slower over the "
            "past week. Pages that used to load in 2-3 seconds are now "
            "taking 10-15 seconds. It's not a dealbreaker but it's annoying "
            "and slowing down our daily standups.\n\n"
            "We're on the Pro plan with about 500 active dashboard users. "
            "No changes on our end that I'm aware of.\n\n"
            "Any ideas what might be going on?\n\n"
            "Thanks,\nTom Baker"
        ),
        "customer_name": "Tom Baker",
        "customer_tier": "pro",
        "channel": "web_form",
        "previous_tickets": 3,
        "account_age_days": 600,
        "has_attachments": False,
        "ground_truth": GroundTruth(
            priority=Priority.P2_MEDIUM,
            category=Category.TECHNICAL,
            sentiment=Sentiment.NEUTRAL,
            assigned_team=Team.ENGINEERING,
            decision=ReviewDecision.RESPOND,
            escalation_required=False,
            required_response_elements=[
                "acknowledge_issue", "investigation_steps",
                "workaround_suggestion",
            ],
            explanation="Degraded performance, not outage = P2 + Engineering",
        ),
    },
    {
        "subject": "How do I set up SSO for my team?",
        "body": (
            "Hi,\n\n"
            "We just upgraded to the Enterprise plan and I'd like to "
            "configure SAML SSO with our Okta instance. I've looked through "
            "your docs but the SSO setup guide seems to be for an older "
            "version of the admin panel.\n\n"
            "Could you point me to the current instructions or help me "
            "get this configured? We have about 200 team members who need "
            "access.\n\n"
            "Thanks!\n— Lisa Park, IT Admin"
        ),
        "customer_name": "Lisa Park",
        "customer_tier": "enterprise",
        "channel": "email",
        "previous_tickets": 1,
        "account_age_days": 30,
        "has_attachments": False,
        "ground_truth": GroundTruth(
            priority=Priority.P2_MEDIUM,
            category=Category.ONBOARDING,
            sentiment=Sentiment.NEUTRAL,
            assigned_team=Team.CUSTOMER_SUCCESS,
            decision=ReviewDecision.RESPOND,
            escalation_required=False,
            required_response_elements=[
                "documentation_link", "setup_guidance",
                "offer_call",
            ],
            explanation="New enterprise onboarding SSO request = P2 + Customer Success",
        ),
    },
    {
        "subject": "Feature request: dark mode for the editor",
        "body": (
            "Love the product! One thing that would make it even better — "
            "a dark mode option for the code editor. I spend 8+ hours a day "
            "in the editor and the bright white background is straining my "
            "eyes.\n\n"
            "I know this has been requested before but wanted to add my "
            "voice. Would be amazing for the next release!\n\n"
            "Keep up the great work,\nDev_Dave"
        ),
        "customer_name": "Dev_Dave",
        "customer_tier": "pro",
        "channel": "web_form",
        "previous_tickets": 0,
        "account_age_days": 90,
        "has_attachments": False,
        "ground_truth": GroundTruth(
            priority=Priority.P3_LOW,
            category=Category.FEATURE_REQUEST,
            sentiment=Sentiment.SATISFIED,
            assigned_team=Team.PRODUCT,
            decision=ReviewDecision.AUTO_RESOLVE,
            escalation_required=False,
            required_response_elements=[
                "acknowledge_feedback", "roadmap_reference",
            ],
            explanation="Positive feature request, non-urgent = P3 + Product + auto-resolve",
        ),
    },
    # ── P3 LOW ───────────────────────────────────────────────────────────
    {
        "subject": "Typo in your documentation page",
        "body": (
            "Hey, just noticed a small typo on your API docs page for "
            "/v2/users endpoint. The parameter 'limit' is spelled 'limt' "
            "in the example code block.\n\n"
            "Not a big deal but thought you'd want to know!\n\n"
            "Cheers"
        ),
        "customer_name": "Helpful_Henry",
        "customer_tier": "free",
        "channel": "web_form",
        "previous_tickets": 0,
        "account_age_days": 15,
        "has_attachments": False,
        "ground_truth": GroundTruth(
            priority=Priority.P3_LOW,
            category=Category.GENERAL,
            sentiment=Sentiment.SATISFIED,
            assigned_team=Team.TIER1_SUPPORT,
            decision=ReviewDecision.AUTO_RESOLVE,
            escalation_required=False,
            required_response_elements=["thank_user"],
            explanation="Cosmetic doc typo from free user = P3 + Tier1 + auto-resolve",
        ),
    },
    {
        "subject": "Question about pricing for annual plans",
        "body": (
            "Hello,\n\n"
            "I'm currently on the monthly Pro plan ($99/mo) and I'm "
            "considering switching to annual billing. Could you tell me:\n"
            "1. What's the annual price?\n"
            "2. Is there a discount for paying yearly?\n"
            "3. Can I switch mid-cycle or do I need to wait?\n\n"
            "Thanks for your help!\n— New_Customer_Nora"
        ),
        "customer_name": "New_Customer_Nora",
        "customer_tier": "pro",
        "channel": "email",
        "previous_tickets": 1,
        "account_age_days": 60,
        "has_attachments": False,
        "ground_truth": GroundTruth(
            priority=Priority.P3_LOW,
            category=Category.BILLING,
            sentiment=Sentiment.NEUTRAL,
            assigned_team=Team.BILLING_OPS,
            decision=ReviewDecision.RESPOND,
            escalation_required=False,
            required_response_elements=[
                "pricing_info", "discount_details",
            ],
            explanation="Routine billing inquiry = P3 + Billing Ops + respond",
        ),
    },
    # ── More P2 tickets for variety ──────────────────────────────────────
    {
        "subject": "Webhook deliveries failing intermittently",
        "body": (
            "Hi support,\n\n"
            "We've been seeing intermittent failures on webhook deliveries "
            "for the past 3 days. About 15% of our webhooks are returning "
            "timeouts. Our endpoint is healthy (confirmed via direct curl). "
            "The webhook dashboard shows 'delivery_timeout' as the error.\n\n"
            "We rely on these webhooks for order fulfillment so this is "
            "causing some missed orders. Not critical yet but getting worse.\n\n"
            "Webhook endpoint: https://api.ourshop.io/webhooks/orders\n\n"
            "— Chris Taylor, Backend Lead"
        ),
        "customer_name": "Chris Taylor",
        "customer_tier": "pro",
        "channel": "email",
        "previous_tickets": 4,
        "account_age_days": 300,
        "has_attachments": True,
        "ground_truth": GroundTruth(
            priority=Priority.P2_MEDIUM,
            category=Category.BUG_REPORT,
            sentiment=Sentiment.FRUSTRATED,
            assigned_team=Team.ENGINEERING,
            decision=ReviewDecision.RESPOND,
            escalation_required=False,
            required_response_elements=[
                "acknowledge_issue", "investigation_steps",
                "workaround_suggestion",
            ],
            explanation="Intermittent webhook failures, worsening = P2 + Engineering",
        ),
    },
    {
        "subject": "Need to add 50 more seats to our Enterprise plan",
        "body": (
            "Hi there,\n\n"
            "We're expanding our engineering team and need to add 50 "
            "additional seats to our Enterprise plan. Can you help us "
            "with the process and let us know if there's a volume discount "
            "for this expansion?\n\n"
            "Current seats: 200\nNeeded: 250\n\n"
            "Also, is it possible to get a custom invoicing arrangement? "
            "Our finance team prefers quarterly invoicing.\n\n"
            "Thanks,\nRachel Adams, VP Operations"
        ),
        "customer_name": "Rachel Adams",
        "customer_tier": "enterprise",
        "channel": "email",
        "previous_tickets": 7,
        "account_age_days": 1000,
        "has_attachments": False,
        "ground_truth": GroundTruth(
            priority=Priority.P2_MEDIUM,
            category=Category.ACCOUNT,
            sentiment=Sentiment.NEUTRAL,
            assigned_team=Team.ACCOUNT_MGMT,
            decision=ReviewDecision.RESPOND,
            escalation_required=False,
            required_response_elements=[
                "seat_expansion_process", "pricing_info",
                "invoicing_options",
            ],
            explanation="Enterprise seat expansion = P2 + Account Mgmt + respond",
        ),
    },
    # ── Tricky / ambiguous tickets (for hard task) ───────────────────────
    {
        "subject": "Your API broke my production system — demanding compensation",
        "body": (
            "I am FURIOUS. Your latest API update on March 15 broke backward "
            "compatibility with v1 endpoints WITHOUT ANY WARNING. Our entire "
            "data pipeline crashed and we lost 3 days of processing.\n\n"
            "This is completely unacceptable. We are evaluating moving to a "
            "competitor. But first, I demand:\n"
            "1. A formal explanation of what happened\n"
            "2. Compensation for our losses (estimated $45,000)\n"
            "3. A guarantee this won't happen again\n\n"
            "If I don't hear back within 24 hours, we're escalating to "
            "our legal team.\n\n"
            "— Marcus Johnson, CTO, DataPipe Systems"
        ),
        "customer_name": "Marcus Johnson",
        "customer_tier": "enterprise",
        "channel": "email",
        "previous_tickets": 15,
        "account_age_days": 1500,
        "has_attachments": True,
        "ground_truth": GroundTruth(
            priority=Priority.P1_HIGH,
            category=Category.TECHNICAL,
            sentiment=Sentiment.ANGRY,
            assigned_team=Team.ENGINEERING,
            decision=ReviewDecision.ESCALATE,
            escalation_required=True,
            required_response_elements=[
                "acknowledge_urgency", "empathy",
                "escalation_confirmation", "timeline_commitment",
            ],
            explanation=(
                "Angry enterprise customer with legal threat + technical root "
                "cause = P1 + Engineering + escalate"
            ),
        ),
    },
    {
        "subject": "Can I use your API for a healthcare application?",
        "body": (
            "Hello,\n\n"
            "We're building a telehealth platform and want to integrate your "
            "messaging API. Before we commit, we need to know:\n\n"
            "1. Are you HIPAA compliant?\n"
            "2. Do you offer a BAA (Business Associate Agreement)?\n"
            "3. Where is data stored and is it encrypted at rest?\n"
            "4. Can we get a SOC 2 Type II report?\n\n"
            "We're evaluating 3 vendors and need this info by end of week "
            "to make our decision.\n\n"
            "— Dr. Sarah Wong, CTO, HealthConnect"
        ),
        "customer_name": "Dr. Sarah Wong",
        "customer_tier": "enterprise",
        "channel": "email",
        "previous_tickets": 0,
        "account_age_days": 5,
        "has_attachments": False,
        "ground_truth": GroundTruth(
            priority=Priority.P2_MEDIUM,
            category=Category.SECURITY,
            sentiment=Sentiment.NEUTRAL,
            assigned_team=Team.SECURITY_TEAM,
            decision=ReviewDecision.RESPOND,
            escalation_required=False,
            required_response_elements=[
                "compliance_info", "documentation_link",
                "offer_call",
            ],
            explanation="Compliance/security pre-sales inquiry = P2 + Security Team",
        ),
    },
    {
        "subject": "Getting started — completely lost",
        "body": (
            "Hi,\n\n"
            "I just signed up for the free tier and I'm completely lost. "
            "The quickstart guide talks about API keys but I can't find "
            "where to generate one. The dashboard looks different from the "
            "screenshots in the docs.\n\n"
            "I'm a junior developer and this is my first time working with "
            "an API. Could someone walk me through the basic setup?\n\n"
            "Sorry if this is a dumb question!\n— Nervous_Newbie"
        ),
        "customer_name": "Nervous_Newbie",
        "customer_tier": "free",
        "channel": "chat",
        "previous_tickets": 0,
        "account_age_days": 1,
        "has_attachments": False,
        "ground_truth": GroundTruth(
            priority=Priority.P3_LOW,
            category=Category.ONBOARDING,
            sentiment=Sentiment.NEUTRAL,
            assigned_team=Team.CUSTOMER_SUCCESS,
            decision=ReviewDecision.RESPOND,
            escalation_required=False,
            required_response_elements=[
                "encouragement", "step_by_step_guidance",
                "documentation_link",
            ],
            explanation="New free-tier user onboarding = P3 + Customer Success",
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Public API — ticket pool generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_ticket_pool(
    task_id: str,
    seed: int = 42,
    num_tickets: Optional[int] = None,
) -> List[LabeledTicket]:
    """
    Generate a pool of labeled tickets for a given task.

    Args:
        task_id: One of "task_1", "task_2", "task_3".
        seed: Random seed for reproducibility.
        num_tickets: Number of tickets (defaults per task).

    Returns:
        List of LabeledTicket with ground-truth labels.
    """
    rng = random.Random(seed)

    # Default ticket counts per task
    defaults = {"task_1": 5, "task_2": 8, "task_3": 10}
    n = num_tickets or defaults.get(task_id, 5)

    # Select templates based on difficulty
    if task_id == "task_1":
        # Easy: clear-cut tickets with obvious priority signals
        indices = [0, 3, 5, 8, 9]  # P0, P1, P2, P3, P3
    elif task_id == "task_2":
        # Medium: broader range, some ambiguity in routing
        indices = [0, 2, 3, 4, 5, 6, 8, 10]
    else:
        # Hard: all tickets including tricky ones
        indices = list(range(len(_TICKET_TEMPLATES)))

    # Cycle through selected templates to reach num_tickets
    selected = []
    for i in range(n):
        template = _TICKET_TEMPLATES[indices[i % len(indices)]]
        ticket_id = _make_ticket_id(i, seed)
        ts = f"2026-03-{15 + (i % 10):02d}T{9 + (i % 12):02d}:00:00Z"

        ticket = TicketData(
            ticket_id=ticket_id,
            subject=template["subject"],
            body=template["body"],
            customer_name=template["customer_name"],
            customer_tier=template["customer_tier"],
            timestamp=ts,
            channel=template["channel"],
            previous_tickets=template["previous_tickets"],
            account_age_days=template["account_age_days"],
            has_attachments=template["has_attachments"],
        )

        selected.append(
            LabeledTicket(ticket=ticket, ground_truth=template["ground_truth"])
        )

    rng.shuffle(selected)
    return selected
