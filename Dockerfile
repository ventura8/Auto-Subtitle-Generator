# syntax=docker/dockerfile:1

# =============================================================================
# Auto Subtitle Generator — Reproducible GPU Docker Image (Linux x86_64/amd64)
# =============================================================================
#
# Dependencies are installed from the project's own Poetry lockfile
# (pyproject.toml / poetry.lock) via the canonical install_dependencies.sh, so
# the container exercises the same supported stack as local installs and CI
# instead of a second, hand-pinned dependency graph.
#
# CUDA 13.2 user-space libraries are supplied by the PyTorch cu132 wheels
# (cuda-toolkit / cuda-bindings). The host NVIDIA driver is injected at runtime
# by `--gpus all`, so a separate CUDA base image is not required. The NVIDIA
# Container Toolkit must be installed on the host.
#
# Build:
#   docker image build --tag auto-sub-gen .
#
# Run (GPU):
#   docker run --gpus all \
#     --mount='type=volume,source=auto-sub-gen,target=/app/models' \
#     --mount='type=bind,source=/path/to/videos,target=/app/input' \
#     auto-sub-gen /app/input/video.mkv
#
# =============================================================================

FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install uv to manage a hermetic Python 3.12 (mirrors docker/Dockerfile.ubuntu).
# UV_PYTHON_BIN_DIR keeps the python3.12 shim out of /root/.local/bin; a venv
# built from a /root path is unexecutable by the unprivileged runtime user.
COPY --from=ghcr.io/astral-sh/uv:0.12.9 /uv /uvx /root/.local/bin/
ENV PATH="/root/.local/bin:$PATH"
ENV UV_PYTHON_INSTALL_DIR="/opt/uv/python"
ENV UV_PYTHON_BIN_DIR="/usr/local/bin"
RUN uv python install 3.12.14

WORKDIR /app

# Dependency manifests first so the heavy ML install stays cached across code
# changes. The `dev` group pins pytest / pytest-cov to the repository versions.
COPY pyproject.toml poetry.lock poetry.toml ./
COPY install_dependencies.sh ./
RUN chmod +x install_dependencies.sh \
    && ./install_dependencies.sh ml,dev

# Application
COPY auto_subtitle.py ./
COPY config.yaml ./
COPY modules/ modules/

# Test gate, matching the canonical CI scope (`pytest -m "not e2e"`). The e2e
# suites download multi-GB models and run real inference, so they are excluded
# from the build and can be run against the finished image with `docker run`.
COPY pytest.ini ./
COPY tests/ tests/
RUN .venv/bin/python -m pytest -m "not e2e" tests/

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV HF_HOME=/app/models/huggingface
# audio-separator 0.44.5 defaults its model cache to /tmp/audio-separator-models
# (ephemeral). Pin it to the models volume so the checkpoint persists across
# container restarts instead of re-downloading every run.
ENV AUDIO_SEPARATOR_MODEL_DIR=/app/models
ENV HOME=/app

# Run unprivileged so files written into bind-mounted host directories are not
# owned by root. Numeric IDs are used instead of useradd so the image does not
# clash with the base image's existing UID 1000 account. Override at build time
# with --build-arg APP_UID=$(id -u) --build-arg APP_GID=$(id -g) to match your
# host user. The named model volume is initialized from /app/models, preserving
# ownership.
ARG APP_UID=1000
ARG APP_GID=1000
RUN mkdir -p /app/models \
    && chown -R ${APP_UID}:${APP_GID} /app
USER ${APP_UID}:${APP_GID}

# Fail the build if the unprivileged runtime user cannot execute the venv
# interpreter (e.g. it resolves through a root-only path).
RUN /app/.venv/bin/python -c "import sys; print('venv interpreter:', sys.executable)"

ENTRYPOINT ["/app/.venv/bin/python", "/app/auto_subtitle.py"]
