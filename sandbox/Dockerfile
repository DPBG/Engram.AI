# syntax=docker/dockerfile:1
#
# activelearning-sandbox:latest — the isolated image the Meta-Programmer uses to
# test LLM-generated code. The SandboxManager runs this with no network, a
# read-only root filesystem, a writable /tmp tmpfs, and memory/CPU/pid limits
# (see meta-programmer/src/meta_programmer/sandbox_manager.py).
#
# Build it explicitly (it is not started as a service):
#   docker compose --profile build-only build sandbox-base
#
# NOTE: Phase 1 of ROADMAP.md hardens this further (cap_drop=ALL,
# no-new-privileges, fail-closed deploy). This image is the minimal correct base.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Test runner + common scientific dep generated drivers may import.
RUN pip install "pytest>=7.4.0" "pytest-timeout>=2.1.0" "numpy>=1.26.0"

# Unprivileged user; the code under test is mounted read-only at /sandbox.
RUN useradd --create-home --uid 10001 sandbox
USER sandbox
WORKDIR /sandbox

# Default command is overridden by SandboxManager with the real pytest invocation.
CMD ["pytest", "--version"]
