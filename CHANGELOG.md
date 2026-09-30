# Changes

## 2026-09-30

- Fixed a grader crash when a caller leaves the customer name blank.
- Invalid HTTP actions now return 400 and leave the current episode untouched.
- An empty reset request still uses defaults. Malformed JSON, invalid seeds and unknown tasks now return 400 instead of silently restarting or exposing a traceback.
- Invalid WebSocket messages return an error and keep the connection usable.
- Added regression tests for these failures and ignored generated Python caches.

Checked with Python 3.14: `python -m pytest -q` — **72 passed** (57 existing cases plus 15 new regression cases). Dependencies came from the existing Python installation; pytest was installed in the task's separate virtual environment. No live service, Docker image or external model was exercised.
