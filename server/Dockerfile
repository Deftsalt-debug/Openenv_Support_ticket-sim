# ─────────────────────────────────────────────────────────────────────────────
# Support Triage Environment — Dockerfile
# OpenEnv-compatible containerized deployment
# ─────────────────────────────────────────────────────────────────────────────
# Build:  docker build -t support-triage-env:latest .
# Run:    docker run -p 8000:8000 support-triage-env:latest
# Health: curl http://localhost:8000/health
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── Install system dependencies ──────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ──────────────────────────────────────────────
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# ── Copy application code ───────────────────────────────────────────────────
COPY models.py /app/models.py
COPY data/ /app/data/
COPY tasks/ /app/tasks/
COPY server/ /app/server/
COPY client.py /app/client.py
COPY openenv.yaml /app/openenv.yaml
COPY README.md /app/README.md
COPY __init__.py /app/__init__.py

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Expose port and run ─────────────────────────────────────────────────────
EXPOSE 8000

# For Hugging Face Spaces, use port 7860
ENV PORT=8000
ENV ENABLE_WEB_INTERFACE=true

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
