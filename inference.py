#!/usr/bin/env python3
"""
Inference Script — Support Triage Environment
===================================
MANDATORY
- Environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME (optional) The name of the local Docker image.

STDOUT FORMAT
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional

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

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

BENCHMARK = "support_triage_env"
MAX_TOKENS = 500
TEMPERATURE = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# OpenAI client
# ─────────────────────────────────────────────────────────────────────────────

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY or "dummy-key",
)

# ─────────────────────────────────────────────────────────────────────────────
# Structured logging helpers (exact format from sample)
# ─────────────────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an AI agent performing customer support ticket triage.
Read each ticket carefully and respond with ONLY a valid JSON object.

Valid enum values:
- priority: P0_CRITICAL, P1_HIGH, P2_MEDIUM, P3_LOW
- category: BILLING, TECHNICAL, ACCOUNT, FEATURE_REQUEST, BUG_REPORT, SECURITY, ONBOARDING, GENERAL
- sentiment: ANGRY, FRUSTRATED, NEUTRAL, SATISFIED
- assigned_team: BILLING_OPS, ENGINEERING, ACCOUNT_MGMT, PRODUCT, SECURITY_TEAM, CUSTOMER_SUCCESS, TIER1_SUPPORT
- decision: RESPOND, ESCALATE, AUTO_RESOLVE

Respond with ONLY a JSON object — no markdown, no explanation.
""").strip()


def build_ticket_prompt(obs: TriageObservation) -> str:
    ticket = obs.ticket
    if not ticket:
        return "No ticket available."

    task_hint = ""
    if obs.task_id == "task_1":
        task_hint = 'Set only "priority". Example: {"priority": "P1_HIGH"}'
    elif obs.task_id == "task_2":
        task_hint = 'Set "priority", "category", "sentiment", "assigned_team".'
    else:
        task_hint = 'Set all fields: "priority", "category", "sentiment", "assigned_team", "decision", "draft_response" (50-300 chars addressing customer by name), and "escalation_reason" if escalating.'

    return f"""{obs.task_instructions}

TICKET:
ID: {ticket.ticket_id}
Subject: {ticket.subject}
From: {ticket.customer_name} ({ticket.customer_tier} tier)
Channel: {ticket.channel} | Previous tickets: {ticket.previous_tickets} | Account age: {ticket.account_age_days} days

{ticket.body}

{task_hint}
Respond with ONLY a JSON object."""


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
        match = re.search(r'\{.*\}', text, re.DOTALL)
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
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    env_client = SupportTriageClient.local()

    tasks = ["task_1", "task_2", "task_3"]
    seed = 42

    for task_id in tasks:
        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False

        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        try:
            obs = env_client.reset(task_id=task_id, seed=seed)

            step = 0
            while not obs.done:
                step += 1
                ticket_prompt = build_ticket_prompt(obs)

                # Call LLM via OpenAI client
                error_msg = None
                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": ticket_prompt},
                        ],
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        stream=False,
                    )
                    response_text = (completion.choices[0].message.content or "").strip()
                except Exception as exc:
                    error_msg = str(exc)
                    response_text = "{}"

                # Parse and step
                parsed = parse_llm_response(response_text)
                action = dict_to_action(parsed)
                action_str = json.dumps(action.model_dump(exclude_none=True))

                obs = env_client.step(action)

                reward = obs.reward or 0.0
                done = obs.done
                rewards.append(reward)
                steps_taken = step

                log_step(
                    step=step,
                    action=action_str,
                    reward=reward,
                    done=done,
                    error=error_msg,
                )

                if done:
                    break

            # Compute score in [0, 1]
            score = sum(rewards) / len(rewards) if rewards else 0.0
            score = min(max(score, 0.0), 1.0)
            success = score > 0.0

        finally:
            log_end(
                success=success,
                steps=steps_taken,
                score=score,
                rewards=rewards,
            )

    env_client.close()


if __name__ == "__main__":
    main()
