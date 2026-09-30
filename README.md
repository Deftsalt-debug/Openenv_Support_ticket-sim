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

A support-ticket simulation for trying out AI agents. An agent reads a ticket,
chooses its labels and routing, and, in the harder task, drafts a reply. The
environment scores the action and moves to the next ticket.

There are 16 synthetic ticket templates, from production outages to documentation
typos. Their expected labels make runs easy to compare with a fixed seed. This is
a graded exercise, not a real support queue or a measure of how well an agent
would handle every customer.

The three tasks build up from priority classification to routing and response
drafting. The models, scoring rules and client examples are below.

## Environment Overview

| Property              | Value                                    |
|-----------------------|------------------------------------------|
| **Interface**         | OpenEnv step / reset / state API       |
| **Action Type**       | `TriageAction` (Pydantic model)          |
| **Observation Type**  | `TriageObservation` (Pydantic model)     |
| **State Type**        | `TriageState` (Pydantic model)           |
| **Reward Range**      | `[0.0, 1.0]` per step                   |
| **Tasks**             | 3 (priority → labels → full triage)                |
| **Ticket Templates**  | 16 synthetic scenarios                   |
| **Grading**           | Fully programmatic, deterministic        |
| **Deployment**        | Docker + Hugging Face Spaces             |

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

## Task Descriptions

### Task 1 — Priority Classification (Easy)
- **Objective:** Assign the correct priority level (P0–P3) to each ticket.
- **Tickets:** 5 per episode.
- **Evaluated fields:** `priority` only.
- **Signals:** Urgency language, customer tier, scope of impact.

### Task 2 — Multi-Label Triage (Medium)
- **Objective:** Classify priority, category, sentiment, and route to the correct team.
- **Tickets:** 8 per episode.
- **Evaluated fields:** `priority`, `category`, `sentiment`, `assigned_team`.
- **Signals:** Must cross-reference category with routing table, detect tone.

### Task 3 — Full Triage with Response Drafting (Hard)
- **Objective:** Complete triage including a professional draft response and escalation decision.
- **Tickets:** 10 per episode (includes ambiguous/tricky scenarios).
- **Evaluated fields:** All 7 fields.
- **Signals:** Escalation criteria (legal threats, revenue impact, security breaches), response quality (personalization, required elements, tone matching).

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

## Setup & Usage

### Install

Use Python 3.10 or newer. From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[all]"
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` instead.
The `all` extra includes the test dependencies and optional OpenAI baseline.
Docker is only needed if you choose the container route below.

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
from models import TriageAction, Priority

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

### Run the optional baseline

With `OPENAI_API_KEY` already set in your environment, run from the project root:

```bash
python -m baseline.inference --mode local --model gpt-4o-mini
```

This calls the OpenAI API. The local client and tests don't need that key.
Record the model, seed and output when comparing scores. The earlier approximate
score table wasn't backed by a saved run in this checkout, so it has been removed.

### Run tests

```bash
python -m pytest tests/ -v
```

The review passed 72 tests, covering the environment and HTTP/WebSocket behavior.
The graders check labels and rule-based response features; a high score isn't a
guarantee that a reply is appropriate for a real customer.

## Project Structure

```
project-root/
├── openenv.yaml              # OpenEnv metadata specification
├── pyproject.toml             # Python package configuration
├── Dockerfile                 # Root Dockerfile (build from project root)
├── README.md                  # This file
├── __init__.py                # Package exports
├── models.py                  # Pydantic models: Action, Observation, State
├── client.py                  # HTTP + local client implementation
├── data/
│   ├── __init__.py
│   └── tickets.py             # 16 synthetic ticket templates + generator
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
    ├── test_environment.py    # Environment and grader checks
    └── test_api.py            # HTTP and WebSocket regressions
```

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

## License

BSD-3-Clause
