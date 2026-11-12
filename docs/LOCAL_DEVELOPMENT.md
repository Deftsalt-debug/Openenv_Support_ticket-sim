# Local development

## Environment and server

Use Python 3.11 for parity with the Docker images and CI; the declared minimum is Python 3.10.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

On Windows, activate `.venv` with the shell's normal activation script instead of `source`.
Run from the repository root because the current package build omits required root modules.
Open `/docs` for the API schema and `/health` for a health response. HTTP state is shared across callers.

## Tests and builds

```bash
python -m pytest tests/ -v
python -m pip wheel --no-deps --wheel-dir dist .
docker build -t support-triage-env:local -f server/Dockerfile .
docker run --rm -p 8000:8000 support-triage-env:local
```

If using uv, `uv lock --check` checks the committed lock and `uv build` builds the package without changing source.
`openenv validate . --verbose` checks local deployment structure; it does not prove a live Space works or that the wheel contains every required module.
CI runs tests and a package build. Lint/type-check tooling is not configured and remains a separate development-dependency decision.

## Optional inference

Install the existing baseline extra only when running a model-driven episode:

```bash
python -m pip install -e ".[baseline]"
python baseline/inference.py --mode local --model gpt-4o-mini
```

The baseline runner reads `OPENAI_API_KEY`, with optional `OPENAI_BASE_URL`; set credentials privately in the process environment.
For remote mode, use `--mode remote --base-url http://localhost:8000` with the server already running.
The root `python inference.py` instead reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` or `API_KEY` and emits the evaluator-style START/STEP/END log format.
It currently uses the local environment even if `LOCAL_IMAGE_NAME` is set. See [architecture](ARCHITECTURE.md).
Model calls may incur provider usage; neither runner is part of the default tests or CI. Never commit credentials or paste them into logs/issues.
The server reads process environment values; no automatic `.env` loader is configured.

## Deployment caveats

Keep the Hugging Face README YAML block and both Dockerfiles until the actual Space configuration and external evaluator contract are confirmed.
The YAML declares port 7860, while both commands listen on 8000. A successful container build does not resolve this mismatch.
The application is currently suitable for isolated evaluation sessions; shared HTTP state, permissive CORS and ephemeral storage need explicit consideration before hosting.
No Space was deployed or reconfigured as part of repository cleanup.
