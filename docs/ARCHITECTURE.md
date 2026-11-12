# Architecture

The environment presents synthetic support tickets and grades structured agent actions. It does not contact real customers or send the drafted responses.

## Source layout

| File or directory | Role |
| --- | --- |
| `models.py` | Enums and Pydantic action, observation, state and ticket models |
| `data/tickets.py` | Ticket templates, labels and seeded generation |
| `tasks/task_definitions.py` | Three tasks and their evaluated fields |
| `tasks/graders.py` | Partial-credit and response-scoring rules |
| `server/support_triage_environment.py` | Reset, step, done and cumulative episode state |
| `server/app.py` | FastAPI endpoints, health/metadata and WebSocket sessions |
| `client.py` | Local in-process and remote HTTP access |
| `tests/test_environment.py` | Models, data, scoring, lifecycle and HTTP tests |

A caller resets with a task ID and seed, receives the first ticket, submits a `TriageAction`, and repeats until done.
Observations include the ticket, task instructions, remaining count, reward, feedback and cumulative reward.
The [grading note](GRADING.md) describes task fields and why response scores are limited proxies.

## Interfaces and isolation

| Interface | Contract |
| --- | --- |
| Local | `SupportTriageClient.local()` calls the Python environment |
| `POST /reset` | Start an episode with `task_id` and `seed` |
| `POST /step` | Submit an action wrapped in an `action` object |
| `GET /state` | Read current episode state |
| `GET /health`, `GET /metadata` | Health and environment information |
| `/ws` | JSON reset/step/state messages over a WebSocket |

The HTTP handlers use a shared default environment instance. Each WebSocket connection gets its own session.
CORS is permissive, and there is no application authentication or durable state store. These are prototype boundaries, not production security guarantees.

## Why there are two inference runners

| Entrypoint | Inputs and purpose |
| --- | --- |
| `inference.py` | `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`/`API_KEY`; emits START/STEP/END records using a local environment |
| `baseline/inference.py` | CLI options for local/remote mode, endpoint, tasks, model and seed; uses `OPENAI_API_KEY` and optional `OPENAI_BASE_URL` |

The root runner documents `LOCAL_IMAGE_NAME`, but its current main function uses `SupportTriageClient.local()`; it does not automatically start that image.
The installed OpenEnv 0.3.0 structural validator does not require either filename. An external submission evaluator's requirements are unknown, so neither runner is retired.
Only these optional inference paths call a model provider. Tests and the server do not need an API key.

## Docker and packaging

`openenv.yaml` and the documented build command select `server/Dockerfile`. The root Dockerfile is the Hugging Face Docker Space entrypoint.
OpenEnv 0.3.0's build CLI prefers `server/Dockerfile`; its push staging prefers a root Dockerfile when present. Keep both until the target deployment is settled.
Both install `server/requirements.txt`; the root file copies the full build context, while the server file copies an explicit subset and sets additional environment variables.
The manifest listens on 8000 and both Docker commands hardcode 8000, despite the README Space front-matter's 7860. Changing `PORT` alone does not fix that command.

`pyproject.toml` includes broader runtime/dev/baseline extras than the Docker requirements. Its `websockets` bound also differs from the server requirements.
`uv.lock` is retained and matches the manifest, but a normal pip editable install does not consume it.
The wheel's package-discovery configuration omits root `models.py` and `client.py`. Checkout-based tests pass while an installed import outside the checkout fails.
That packaging defect and deployment-port mismatch are pre-existing issues for separate code/stack review.
