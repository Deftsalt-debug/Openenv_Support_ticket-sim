#!/usr/bin/env python3
# Copyright (c) 2026. Licensed under the BSD-style license.
# Support Triage Environment — Baseline Inference Script

"""
Baseline inference script that runs an LLM agent against all 3 tasks.

Uses the OpenAI API client (compatible with OpenAI, Azure OpenAI, and
any OpenAI-compatible endpoint). Reads credentials from environment
variables.

Usage:
    # With a running server:
    OPENAI_API_KEY=sk-... python baseline/inference.py --mode remote --base-url http://localhost:8000

    # Local (in-process, no server needed):
    OPENAI_API_KEY=sk-... python baseline/inference.py --mode local

    # Specify model:
    OPENAI_API_KEY=sk-... python baseline/inference.py --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    Category,
    Priority,
    ReviewDecision,
    Sentiment,
    Team,
    TriageAction,
    TriageObservation,
)
from client import SupportTriageClient


def get_openai_client():
    """Initialize OpenAI client from environment variables."""
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    base_url = os.environ.get("OPENAI_BASE_URL", None)
    return OpenAI(api_key=api_key, base_url=base_url)


def build_system_prompt(task_instructions: str) -> str:
    """Build the system prompt for the LLM agent."""
    return f"""You are an AI agent performing customer support ticket triage.

{task_instructions}

RESPONSE FORMAT:
You MUST respond with a valid JSON object containing your triage action.
Do NOT include any other text, markdown, or explanation — ONLY the JSON object.

Valid enum values:
- priority: P0_CRITICAL, P1_HIGH, P2_MEDIUM, P3_LOW
- category: BILLING, TECHNICAL, ACCOUNT, FEATURE_REQUEST, BUG_REPORT, SECURITY, ONBOARDING, GENERAL
- sentiment: ANGRY, FRUSTRATED, NEUTRAL, SATISFIED
- assigned_team: BILLING_OPS, ENGINEERING, ACCOUNT_MGMT, PRODUCT, SECURITY_TEAM, CUSTOMER_SUCCESS, TIER1_SUPPORT
- decision: RESPOND, ESCALATE, AUTO_RESOLVE

Example (Task 1):
{{"priority": "P1_HIGH"}}

Example (Task 2):
{{"priority": "P1_HIGH", "category": "TECHNICAL", "sentiment": "FRUSTRATED", "assigned_team": "ENGINEERING"}}

Example (Task 3):
{{"priority": "P1_HIGH", "category": "TECHNICAL", "sentiment": "FRUSTRATED", "assigned_team": "ENGINEERING", "decision": "ESCALATE", "draft_response": "Hi Sam, I understand this is critically impacting your business. I'm escalating this to our engineering team immediately. You'll receive an update within 30 minutes.", "escalation_reason": "Payment processing outage with significant revenue impact"}}
"""


def build_ticket_prompt(obs: TriageObservation) -> str:
    """Build the user prompt with the current ticket."""
    ticket = obs.ticket
    if not ticket:
        return "No ticket available."

    return f"""TICKET TO TRIAGE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ID:          {ticket.ticket_id}
Subject:     {ticket.subject}
From:        {ticket.customer_name} ({ticket.customer_tier} tier)
Channel:     {ticket.channel}
Timestamp:   {ticket.timestamp}
Prev Tickets: {ticket.previous_tickets}
Account Age: {ticket.account_age_days} days
Attachments: {"Yes" if ticket.has_attachments else "No"}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{ticket.body}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Respond with ONLY a JSON object containing your triage action.
Tickets remaining: {obs.tickets_remaining}"""


def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """Parse the LLM response into a dict, handling common format issues."""
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines if they are code fences
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        import re
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

    # Return empty dict as fallback
    print(f"  WARNING: Could not parse LLM response: {text[:100]}...")
    return {}


def dict_to_action(data: Dict[str, Any]) -> TriageAction:
    """Convert a parsed dict to a TriageAction, handling enum conversion."""
    clean = {}
    for key, value in data.items():
        if value is None:
            continue
        if key == "priority" and isinstance(value, str):
            try:
                clean[key] = Priority(value)
            except ValueError:
                pass
        elif key == "category" and isinstance(value, str):
            try:
                clean[key] = Category(value)
            except ValueError:
                pass
        elif key == "sentiment" and isinstance(value, str):
            try:
                clean[key] = Sentiment(value)
            except ValueError:
                pass
        elif key == "assigned_team" and isinstance(value, str):
            try:
                clean[key] = Team(value)
            except ValueError:
                pass
        elif key == "decision" and isinstance(value, str):
            try:
                clean[key] = ReviewDecision(value)
            except ValueError:
                pass
        elif key in ("draft_response", "escalation_reason"):
            clean[key] = str(value)

    return TriageAction(**clean)


def run_episode(
    client: SupportTriageClient,
    openai_client: Any,
    task_id: str,
    seed: int = 42,
    model: str = "gpt-4o-mini",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a single episode (one task, all tickets).

    Returns:
        Dict with episode results including final score.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  TASK: {task_id} | Model: {model} | Seed: {seed}")
        print(f"{'='*60}")

    # Reset
    obs = client.reset(task_id=task_id, seed=seed)
    system_prompt = build_system_prompt(obs.task_instructions)

    step_rewards = []
    step_count = 0
    start_time = time.time()

    while not obs.done:
        step_count += 1
        ticket_prompt = build_ticket_prompt(obs)

        if verbose:
            ticket = obs.ticket
            print(f"\n  Step {step_count}: Ticket {ticket.ticket_id if ticket else 'N/A'}")
            print(f"  Subject: {ticket.subject[:60] if ticket else 'N/A'}...")

        # Call LLM
        try:
            completion = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": ticket_prompt},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as e:
            print(f"  ERROR calling LLM: {e}")
            response_text = "{}"

        # Parse and create action
        parsed = parse_llm_response(response_text)
        action = dict_to_action(parsed)

        if verbose:
            action_summary = action.model_dump(exclude_none=True)
            print(f"  Action: {json.dumps(action_summary, indent=2)[:200]}")

        # Step
        obs = client.step(action)
        step_rewards.append(obs.reward or 0.0)

        if verbose:
            print(f"  Reward: {obs.reward:.4f}")
            if obs.step_feedback:
                print(f"  Feedback: {obs.step_feedback[:100]}")

    elapsed = time.time() - start_time
    episode_score = (
        sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
    )

    if verbose:
        print(f"\n  {'─'*50}")
        print(f"  Episode Score: {episode_score:.2%}")
        print(f"  Steps: {step_count} | Time: {elapsed:.1f}s")
        print(f"  Step Rewards: {[f'{r:.3f}' for r in step_rewards]}")

    return {
        "task_id": task_id,
        "model": model,
        "seed": seed,
        "episode_score": round(episode_score, 4),
        "step_rewards": [round(r, 4) for r in step_rewards],
        "step_count": step_count,
        "elapsed_seconds": round(elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Baseline inference for Support Triage Environment"
    )
    parser.add_argument(
        "--mode", choices=["local", "remote"], default="local",
        help="Run locally (in-process) or connect to a server.",
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8000",
        help="Server URL for remote mode.",
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="OpenAI model to use.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=["task_1", "task_2", "task_3"],
        help="Tasks to run.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Minimal output.",
    )
    args = parser.parse_args()

    # Initialize client
    if args.mode == "local":
        client = SupportTriageClient.local()
    else:
        client = SupportTriageClient.remote(args.base_url)

    # Initialize OpenAI client
    openai_client = get_openai_client()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Support Triage Environment — Baseline Inference        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Mode:  {args.mode}")
    print(f"  Model: {args.model}")
    print(f"  Seed:  {args.seed}")
    print(f"  Tasks: {args.tasks}")

    # Run all tasks
    results = []
    for task_id in args.tasks:
        result = run_episode(
            client=client,
            openai_client=openai_client,
            task_id=task_id,
            seed=args.seed,
            model=args.model,
            verbose=not args.quiet,
        )
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("  BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['task_id']:8s} → Score: {r['episode_score']:.2%}  "
              f"({r['step_count']} steps, {r['elapsed_seconds']:.1f}s)")

    overall = (
        sum(r["episode_score"] for r in results) / len(results)
        if results else 0.0
    )
    print(f"  {'─'*50}")
    print(f"  Overall:  {overall:.2%}")
    print(f"{'='*60}")

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "baseline_results.json",
    )
    with open(output_path, "w") as f:
        json.dump(
            {"model": args.model, "seed": args.seed, "results": results,
             "overall_score": round(overall, 4)},
            f, indent=2,
        )
    print(f"\n  Results saved to: {output_path}")

    client.close()
    return results


if __name__ == "__main__":
    main()
