# Task and grading contract

Episodes draw from 16 synthetic ticket templates with seeded generation and known labels.
These labels exercise a controlled simulator; they do not establish performance on a real support queue.

| Task | Tickets per episode | Evaluated fields |
| --- | --- | --- |
| `task_1` | 5 | Priority |
| `task_2` | 8 | Priority, category, sentiment, assigned team |
| `task_3` | 10 | Those four plus disposition, draft response and escalation reason |

Actions use the enums in `models.py`: priority P0–P3, category, sentiment, team and respond/escalate/auto-resolve disposition.
Required fields depend on the selected task; a missing evaluated field loses credit.

## Reward weights

| Field | Task 1 | Task 2 | Task 3 |
| --- | --- | --- | --- |
| Priority | 100% | 30% | 15% |
| Category | — | 30% | 15% |
| Sentiment | — | 15% | 10% |
| Assigned team | — | 25% | 15% |
| Decision | — | — | 15% |
| Draft response | — | — | 20% |
| Escalation reason | — | — | 10% |

Scores are computed per step in `tasks/graders.py`, with partial credit for supported near misses and an episode mean across step rewards.
Priority distance, related categories and adjacent sentiments can earn partial credit rather than an all-or-nothing score.
Response grading checks length, name personalisation and expected content markers; it is a heuristic, not semantic judgment by a second model.
A high score therefore does not certify a safe, factual or appropriate customer response.

## Reproducibility and limits

Use the same task, seed and model configuration when comparing an inference run. Preserve the actual output before claiming a model score.
The previous README's approximate baseline percentages had no tracked result evidence and are intentionally not repeated.
Tests cover model validity, generation determinism, exact/partial scoring, lifecycle and reward properties; they do not demonstrate unseen-customer generalisation.
