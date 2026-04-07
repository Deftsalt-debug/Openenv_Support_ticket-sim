#!/usr/bin/env python3
"""
Baseline inference script for Support Triage Environment.
Follows the OpenEnv Hackathon x Scaler School of Technology required format.

Required env vars:
    API_BASE_URL  — The API endpoint for the LLM
    MODEL_NAME    — The model identifier to use for inference
    HF_TOKEN      — Your Hugging Face / API key

Emits structured stdout logs: [START], [STEP], [END]
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Dict, List

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI

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


# ─────────────────────────────────────────────────────────────────────────────
# Configuration from environment variables
# ─────────────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional — for from_docker_image() usage
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI client configured via the required variables
# ─────────────────────────────────────────────────────────────────────────────

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "dummy-key",
)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt(task_instructions: str) -> str:
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
    ticket = obs.ticket
    if not ticket:
        return "No ticket available."
    return f"""TICKET TO TRIAGE:
ID: {ticket.ticket_id}
Subject: {ticket.subject}
From: {ticket.customer_name} ({ticket.customer_tier} tier)
Channel: {ticket.channel}
Timestamp: {ticket.timestamp}
Previous Tickets: {ticket.previous_tickets}
Account Age: {ticket.account_age_days} days
Attachments: {"Yes" if ticket.has_attachments else "No"}

{ticket.body}

Respond with ONLY a JSON object. Tickets remaining: {obs.tickets_remaining}"""


# ─────────────────────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_llm_response(response_text: str) -> Dict[str, Any]:
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


def dict_to_action(data: Dict[str, Any]) -> TriageAction:
    clean = {}
    enum_map = {
        "priority": Priority,
        "category": Category,
        "sentiment": Sentiment,
        "assigned_team": Team,
        "decision": ReviewDecision,
    }
    for key, value in data.items():
        if value is None:
            continue
        if key in enum_map and isinstance(value, str):
            try:
                clean[key] = enum_map[key](value)
            except ValueError:
                pass
        elif key in ("draft_response", "escalation_reason"):
            clean[key] = str(value)
    return TriageAction(**clean)


# ─────────────────────────────────────────────────────────────────────────────
# Main inference loop with structured [START]/[STEP]/[END] logging
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Initialize environment client (local mode — runs in-process)
    env_client = SupportTriageClient.local()

    tasks = ["task_1", "task_2", "task_3"]
    seed = 42
    all_results = []

    for task_id in tasks:
        # Reset environment
        obs = env_client.reset(task_id=task_id, seed=seed)
        system_prompt = build_system_prompt(obs.task_instructions)

        # [START] log
        print(f"[START] task_id={task_id} seed={seed}")

        step_num = 0
        step_rewards: List[float] = []

        while not obs.done:
            step_num += 1
            ticket_prompt = build_ticket_prompt(obs)

            # Call LLM via OpenAI client
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": ticket_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=500,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as e:
                print(f"[STEP] step={step_num} error=\"{e}\"")
                response_text = "{}"

            # Parse response and step
            parsed = parse_llm_response(response_text)
            action = dict_to_action(parsed)
            obs = env_client.step(action)

            reward = obs.reward or 0.0
            step_rewards.append(reward)

            # [STEP] log
            action_dict = action.model_dump(exclude_none=True)
            print(
                f"[STEP] task_id={task_id} step={step_num} "
                f"reward={reward:.4f} "
                f"action={json.dumps(action_dict)}"
            )

        # [END] log
        episode_score = (
            sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
        )
        print(
            f"[END] task_id={task_id} "
            f"steps={step_num} "
            f"episode_score={episode_score:.4f}"
        )

        all_results.append({
            "task_id": task_id,
            "episode_score": round(episode_score, 4),
            "steps": step_num,
            "step_rewards": [round(r, 4) for r in step_rewards],
        })

    # Summary
    overall = (
        sum(r["episode_score"] for r in all_results) / len(all_results)
        if all_results else 0.0
    )
    print(f"\nOverall score: {overall:.4f}")
    print(f"Model: {MODEL_NAME}")
    print(f"API: {API_BASE_URL}")

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "baseline_results.json",
    )
    with open(output_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "api_base_url": API_BASE_URL,
            "seed": seed,
            "results": all_results,
            "overall_score": round(overall, 4),
        }, f, indent=2)

    env_client.close()


if __name__ == "__main__":
    main()
