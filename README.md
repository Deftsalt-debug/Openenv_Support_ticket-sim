---
title: Support Triage Env
emoji: 🎫
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
tags:
  - openenv
pinned: false
---
# 🎫 Support Triage Environment

An OpenEnv-compatible reinforcement learning environment that simulates **real-world customer support ticket triage** — the process of classifying, routing, and responding to incoming support requests. Designed for training and evaluating AI agents on a task humans perform every day in every SaaS company.

---

## Motivation

Customer support triage is one of the most common, high-volume knowledge work tasks in the world. Every company with a support queue needs agents (human or AI) that can:

1. **Assess urgency** — Is this a production outage or a typo report?
2. **Classify the issue** — Billing? Security breach? Feature request?
3. **Read the customer** — Are they angry, confused, or just asking a question?
4. **Route correctly** — Engineering, billing ops, or customer success?
5. **Respond appropriately** — Draft a message that matches tone and provides next steps.
6. **Decide disposition** — Respond directly, escalate, or auto-resolve?

This environment provides a structured, graded simulation of this workflow with 16 realistic ticket templates spanning P0 production outages to P3 documentation typos, complete with deterministic ground-truth labels for fully programmatic evaluation.

---

## Environment Overview

| Property              | Value                                    |
|-----------------------|------------------------------------------|
| **Interface**         | OpenEnv 0.1 (step / reset / state)       |
| **Action Type**       | `TriageAction` (Pydantic model)          |
| **Observation Type**  | `TriageObservation` (Pydantic model)     |
| **State Type**        | `TriageState` (Pydantic model)           |
| **Reward Range**      | `[0.0, 1.0]` per step                   |
| **Tasks**             | 3 (easy → medium → hard)                |
| **Ticket Templates**  | 16 realistic scenarios                   |
| **Grading**           | Fully programmatic, deterministic        |
| **Deployment**        | Docker + Hugging Face Spaces             |

---

## Action Space

The agent sends a `TriageAction` each step. Which fields are required depends on the active task:

```
TriageAction:
  priority:          P0_CRITICAL | P1_HIGH | P2_MEDIUM | P3_LOW
  category:          BILLING | TECHNICAL | ACCOUNT | FEATURE_REQUEST |
                     BUG_REPORT | SECURITY | ONBOARDING | GENERAL
  sentiment:         ANGRY | FRUSTRATED | NEUTRAL | SATISFIED
  assigned_team:     BILLING_OPS | ENGINEERING | ACCOUNT_MGMT | PRODUCT |
                     SECURITY_TEAM | CUSTOMER_SUCCESS | TIER1_SUPPORT
  decision:          RESPOND | ESCALATE | AUTO_RESOLVE
  draft_response:    str (50–300 chars, professional customer reply)
  escalation_reason: str (required when decision = ESCALATE)
```

**Task 1 (Easy):** Only `priority` is evaluated.
**Task 2 (Medium):** `priority` + `category` + `sentiment` + `assigned_team`.
**Task 3 (Hard):** All 7 fields including `draft_response` and `escalation_reason`.

---

## Observation Space

After each `step()` or `reset()`, the environment returns a `TriageObservation`:

```
TriageObservation:
  done:              bool          — Episode finished?
  reward:            float | None  — Reward for last action [0.0–1.0]
  ticket:            TicketData    — Current ticket to triage
  task_id:           str           — Active task identifier
  task_instructions: str           — Human-readable instructions
  tickets_remaining: int           — Tickets left in episode
  step_feedback:     str           — Natural-language feedback on last action
  cumulative_reward: float         — Running total reward
  metadata:          dict          — Episode info, grader details
```

Each `TicketData` contains: `ticket_id`, `subject`, `body`, `customer_name`, `customer_tier` (free/pro/enterprise), `timestamp`, `channel`, `previous_tickets`, `account_age_days`, `has_attachments`.

---

## Task Descriptions

### Task 1 — Priority Classification (Easy)
- **Objective:** Assign the correct priority level (P0–P3) to each ticket.
- **Tickets:** 5 per episode.
- **Evaluated fields:** `priority` only.
- **Signals:** Urgency language, customer tier, scope of impact.
- **Expected difficulty:** Straightforward — clear priority signals in each ticket.

### Task 2 — Multi-Label Triage (Medium)
- **Objective:** Classify priority, category, sentiment, and route to the correct team.
- **Tickets:** 8 per episode.
- **Evaluated fields:** `priority`, `category`, `sentiment`, `assigned_team`.
- **Signals:** Must cross-reference category with routing table, detect tone.
- **Expected difficulty:** Moderate — requires understanding multiple dimensions simultaneously.

### Task 3 — Full Triage with Response Drafting (Hard)
- **Objective:** Complete triage including a professional draft response and escalation decision.
- **Tickets:** 10 per episode (includes ambiguous/tricky scenarios).
- **Evaluated fields:** All 7 fields.
- **Signals:** Escalation criteria (legal threats, revenue impact, security breaches), response quality (personalization, required elements, tone matching).
- **Expected difficulty:** Hard — requires nuanced judgment, empathetic writing, and correct disposition.

---

## Reward Function

Rewards are computed **per-step** (not just end-of-episode), providing continuous learning signal:

- **Per-field scoring:** Each field contributes a weighted fraction to the step reward.
- **Partial credit:** Near-miss answers receive partial scores:
  - Priority off by 1 level → 0.5 (off by 2 → 0.2)
  - Related categories (e.g., TECHNICAL ↔ BUG_REPORT) → 0.4
  - Adjacent sentiments (e.g., ANGRY ↔ FRUSTRATED) → 0.5
- **Response quality scoring:** Checks length, name personalization, and presence of required elements (e.g., "acknowledge_urgency", "escalation_confirmation").
- **Anti-gaming:** Empty actions score 0.0. Missing required fields are penalized.
- **Episode score:** Mean of all step rewards.

### Weight distributions by task:

| Field              | Task 1 | Task 2 | Task 3 |
|--------------------|--------|--------|--------|
| priority           | 100%   | 30%    | 15%    |
| category           | —      | 30%    | 15%    |
| sentiment          | —      | 15%    | 10%    |
| assigned_team      | —      | 25%    | 15%    |
| decision           | —      | —      | 15%    |
| draft_response     | —      | —      | 20%    |
| escalation_reason  | —      | —      | 10%    |

---

## Setup & Usage

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)

### Install

```bash
# Clone the repository
git clone <repo-url>
cd support_triage_env

# Install dependencies
pip install -e ".[all]"
```

### Run the Server

```bash
# Direct (development)
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Docker
docker build -t support-triage-env:latest -f server/Dockerfile .
docker run -p 8000:8000 support-triage-env:latest
```

### Use the Client

```python
from client import SupportTriageClient
from models import TriageAction, Priority, Category, Sentiment, Team

# Local mode (no server needed)
client = SupportTriageClient.local()

# Remote mode (connect to server)
# client = SupportTriageClient.remote("http://localhost:8000")

# Reset — start a new episode
obs = client.reset(task_id="task_1", seed=42)
print(f"Task: {obs.task_id}")
print(f"First ticket: {obs.ticket.subject}")

# Step — triage the ticket
action = TriageAction(priority=Priority.P1_HIGH)
obs = client.step(action)
print(f"Reward: {obs.reward:.2f}")
print(f"Feedback: {obs.step_feedback}")

# State — inspect episode
state = client.state()
print(f"Steps: {state.step_count}, Score: {state.cumulative_reward:.2f}")
```

### Run Baseline Inference

```bash
# Requires OPENAI_API_KEY environment variable
OPENAI_API_KEY=sk-... python baseline/inference.py --mode local --model gpt-4o-mini
```

### Run Tests

```bash
cd support_triage_env
pytest tests/ -v
```

---

## Baseline Scores

Scores below are from `gpt-4o-mini` with `temperature=0.0` and `seed=42`:

| Task   | Difficulty | Score  | Notes                                       |
|--------|------------|--------|---------------------------------------------|
| task_1 | Easy       | ~85%+  | Priority classification is well-suited to LLMs |
| task_2 | Medium     | ~70%+  | Multi-label requires cross-referencing       |
| task_3 | Hard       | ~55%+  | Response drafting + escalation is challenging |

*(Exact scores depend on model version and API behavior. Run `baseline/inference.py` for reproducible numbers.)*

---

## Project Structure

```
support_triage_env/
├── openenv.yaml              # OpenEnv metadata specification
├── pyproject.toml             # Python package configuration
├── Dockerfile                 # Root Dockerfile (build from project root)
├── README.md                  # This file
├── __init__.py                # Package exports
├── models.py                  # Pydantic models: Action, Observation, State
├── client.py                  # HTTP + local client implementation
├── data/
│   ├── __init__.py
│   └── tickets.py             # 16 realistic ticket templates + generator
├── tasks/
│   ├── __init__.py
│   ├── task_definitions.py    # 3 task specs (easy → medium → hard)
│   └── graders.py             # Programmatic graders with partial credit
├── server/
│   ├── __init__.py
│   ├── app.py                 # FastAPI server (HTTP + WebSocket)
│   ├── support_triage_environment.py  # Core environment logic
│   ├── requirements.txt       # Server Python dependencies
│   └── Dockerfile             # Server-specific Dockerfile
├── baseline/
│   ├── __init__.py
│   └── inference.py           # OpenAI API baseline agent
└── tests/
    └── test_environment.py    # Comprehensive test suite
```

---

## API Reference

### HTTP Endpoints

| Method | Endpoint    | Description                    | Body                    |
|--------|-------------|--------------------------------|-------------------------|
| POST   | `/reset`    | Start new episode              | `{"task_id": "task_1", "seed": 42}` |
| POST   | `/step`     | Execute triage action          | `{"action": {...}}`     |
| GET    | `/state`    | Get episode state              | —                       |
| GET    | `/health`   | Container health check         | —                       |
| GET    | `/metadata` | Environment metadata           | —                       |

### WebSocket

Connect to `ws://host:8000/ws` for persistent sessions. Send JSON messages:
```json
{"type": "reset", "task_id": "task_1", "seed": 42}
{"type": "step", "action": {"priority": "P1_HIGH"}}
{"type": "state"}
```

---

## License

BSD-3-Clause
