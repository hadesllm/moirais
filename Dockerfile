# syntax=docker/dockerfile:1.7
#
# MOIRAIS container — multi-stage Python build.
#
#   - Stage 1 (builder): full toolchain, installs the package and its deps
#     into /install with cache mounts for apt and pip.
#   - Stage 2 (runtime): slim base, copies only the installed tree, runs
#     as a non-root user, ships an OCI HEALTHCHECK and standard labels.
#
# BuildKit is required for cache mounts (DOCKER_BUILDKIT=1, default in
# docker/build-push-action).

ARG PYTHON_VERSION=3.12

# ─── Stage 1: builder ───────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_WARN_SCRIPT_LOCATION=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc

WORKDIR /build

# Dependency layer: pyproject.toml only — cached while it doesn't change.
COPY pyproject.toml README.md ./

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    mkdir -p src/moirais \
    && echo '__version__ = "0.1.0"' > src/moirais/__init__.py \
    && pip install --root-user-action=ignore setuptools wheel \
    && pip install --root-user-action=ignore --prefix=/install .

# Source layer: changes to *.py invalidate only this layer.
COPY src/ ./src/

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --root-user-action=ignore --prefix=/install --no-deps . \
    && find /install -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find /install -type f -name "*.pyc" -delete 2>/dev/null || true

# ─── Stage 2: runtime ───────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG VERSION=0.1.0
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown

LABEL org.opencontainers.image.title="MOIRAIS" \
      org.opencontainers.image.description="Methods for Observational Inference and Robust Analysis of Interventions in Scientific Experimentation" \
      org.opencontainers.image.url="https://github.com/hadesllm/moirais" \
      org.opencontainers.image.source="https://github.com/hadesllm/moirais" \
      org.opencontainers.image.documentation="https://hadesllm.github.io/moirais/" \
      org.opencontainers.image.licenses="GPL-2.0-only" \
      org.opencontainers.image.authors="Vansh Singh Ruhela <hadesllm@proton.me>" \
      org.opencontainers.image.vendor="hadesllm" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.base.name="docker.io/library/python:${PYTHON_VERSION}-slim"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (uid/gid 1000)
RUN groupadd -r moiraisapp -g 1000 \
    && useradd -r -u 1000 -g moiraisapp -m -d /home/moiraisapp -s /usr/sbin/nologin moiraisapp

# Copy the installed package from the builder stage.
COPY --from=builder /install /usr/local

USER moiraisapp
WORKDIR /home/moiraisapp

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "from moirais.fn._registry import REGISTRY; assert len(REGISTRY) > 0" || exit 1

# tini reaps child processes correctly when running as PID 1.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["moirais", "--help"]
