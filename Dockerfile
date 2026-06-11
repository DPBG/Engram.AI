# syntax=docker/dockerfile:1
#
# Generic build for any Engram Python micro-service. One Dockerfile produces
# every service image; the compose files pass per-service build args:
#   SERVICE    – top-level service directory (e.g. "kernel")
#   SRC_SUBDIR – package root inside the service dir (default "src"; "." for sensory-gateway)
#   MODULE     – module to run as `python -m $MODULE` (e.g. "kernel.service")
#   EXTRA_APT  – optional space-separated apt packages (e.g. "ffmpeg libgl1")
#
# The shared `activelearning` SDK is installed into every image so `import
# activelearning` resolves regardless of PYTHONPATH.

FROM python:3.12-slim AS base

ARG SERVICE
ARG SRC_SUBDIR=src
ARG MODULE
ARG EXTRA_APT=""

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Optional system packages (e.g. ffmpeg/libGL for sensory-gateway/opencv).
RUN if [ -n "$EXTRA_APT" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends $EXTRA_APT \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# Shared constraints + SDK first — these change rarely, so the layer caches well.
COPY constraints.txt ./constraints.txt
COPY sdk/ ./sdk/
RUN pip install -c constraints.txt ./sdk

# Service dependencies next (cached unless the service's requirements change).
COPY ${SERVICE}/requirements.txt ./${SERVICE}/requirements.txt
RUN pip install -c constraints.txt -r ./${SERVICE}/requirements.txt

# Service source.
COPY ${SERVICE}/ ./${SERVICE}/

# Non-root runtime user. /data is pre-created and owned so a first-time named
# volume mount inherits engram ownership (Docker seeds named volumes from the
# image's directory, preserving ownership).
RUN useradd --create-home --uid 10001 engram \
    && mkdir -p /data/sqlite /data/tasks /data/plugins \
    && chown -R engram /app /data
USER engram

ENV PYTHONPATH=/app/${SERVICE}/${SRC_SUBDIR}:/app/sdk/src \
    ENGRAM_MODULE=${MODULE}

# exec form via sh so $ENGRAM_MODULE expands while PID 1 still receives signals.
CMD ["sh", "-c", "exec python -m ${ENGRAM_MODULE}"]
