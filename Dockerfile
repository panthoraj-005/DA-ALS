# syntax=docker/dockerfile:1.7
# =============================================================================
# ALS screening web app — one image serving the API and the built frontend.
#
#   docker build -t als-webapp .
#   docker run -p 8000:8000 -v ./models:/app/models:ro als-webapp
#
# Model weights are NOT baked in. They are large, and they are the part you may
# not want inside a distributable image — mount them at /app/models instead.
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 — build the React app
# -----------------------------------------------------------------------------
FROM node:20-alpine AS frontend

WORKDIR /build

# Copy manifests first so a dependency-free source change reuses the install layer.
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
# In this image the API and the SPA share an origin AND the API lives at the
# root (/health, /predict), so the base is empty — NOT "/api", which is only
# the dev-server proxy prefix. Getting this wrong makes every API call fall
# through to index.html and the UI silently shows "server unreachable".
ENV VITE_API_BASE=""
RUN npm run build


# -----------------------------------------------------------------------------
# Stage 2 — Python runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    HF_HOME=/app/.cache/huggingface \
    OMP_NUM_THREADS=2

# libgomp is required by torch and xgboost; curl is the healthcheck's only client.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU wheels first, from PyTorch's own index. Doing this in its own layer keeps
# the ~200 MB download out of the rebuild path when requirements.txt changes.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install -r backend/requirements.txt

# Non-root from here on. UID 10001 is arbitrary but fixed, so bind-mounted
# volumes have predictable ownership.
RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /app/models /app/outputs /app/sample_signals /app/.cache \
 && chown -R appuser:appuser /app

COPY --chown=appuser:appuser backend/ ./backend/
COPY --chown=appuser:appuser sample_signals/ ./sample_signals/
COPY --chown=appuser:appuser --from=frontend /build/dist ./frontend/dist

USER appuser
WORKDIR /app/backend

ENV ENV=production \
    SERVE_FRONTEND=1 \
    FRONTEND_DIR=/app/frontend/dist \
    MODEL_DIR=/app/models \
    OUTPUT_DIR=/app/outputs \
    SAMPLE_DIR=/app/sample_signals \
    JSON_LOGS=1 \
    BIND=0.0.0.0:8000

EXPOSE 8000

# Liveness only — readiness is a separate probe, because a container that is
# still loading Florence-2 is alive but must not receive traffic yet.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health/live || exit 1

CMD ["gunicorn", "-c", "gunicorn_conf.py", "main:app"]
