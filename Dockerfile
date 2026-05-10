# syntax=docker/dockerfile:1.7
#
# MOIRAIS container — 3-stage Python + R build, single-arch (linux/amd64).
#
# Optimisations:
#   1. Dummy package shim before deps install — heavy pip layer is
#      cached while pyproject doesn't change. The second `pip install`
#      simply overwrites the dummy files with the real source; the
#      __pycache__/*.pyc cleanup keeps the install tree tidy.
#   2. --prefix=/install in the builder + COPY /install /usr/local in
#      the runtime — clean multi-stage merge into Debian system Python.
#   3. Cache mounts on apt + pip — package downloads reuse host cache
#      across builds. NOTE: do not `rm -rf /var/lib/apt/lists/*` when
#      the same RUN has `--mount=type=cache,target=/var/lib/apt`
#      because the cache mount is excluded from the committed image
#      layer anyway, and removing the lists wipes the host-side cache
#      for the next build.
#   4. R libs built in their own stage and copied as a flat tree —
#      skips ~200 MB of apt metadata in the runtime image.

ARG PYTHON_VERSION=3.12

# ─── Stage 1: Python builder ─────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS py-builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_WARN_SCRIPT_LOCATION=0 \
    PYTHONDONTWRITEBYTECODE=1

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc

WORKDIR /build

# Deps layer: pyproject.toml + minimal package shim — cached while
# pyproject is unchanged. The shim is overwritten by the source-layer
# pip install below; its .pyc cache is cleaned up.
COPY pyproject.toml README.md ./

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    mkdir -p src/moirais \
    && echo '__version__ = "0.1.2"' > src/moirais/__init__.py \
    && pip install --root-user-action=ignore setuptools wheel \
    && pip install --root-user-action=ignore --prefix=/install .

# Source layer: only this layer invalidates when *.py changes.
COPY src/ ./src/

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --root-user-action=ignore --prefix=/install --no-deps . \
    && find /install -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find /install -type f -name "*.pyc" -delete 2>/dev/null || true

# ─── Stage 2: R builder ──────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS r-builder

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        r-base-core \
        r-base-dev \
    && Rscript -e "install.packages(c('survey','testthat','DBI','RSQLite','jsonlite'), repos='https://cloud.r-project.org', quiet=TRUE, Ncpus=parallel::detectCores())"

COPY r-package/ /build/r-package/
RUN R CMD INSTALL --library=/usr/local/lib/R/site-library /build/r-package/moirais

# ─── Stage 3: Runtime ────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG VERSION=0.1.2
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown

LABEL org.opencontainers.image.title="MOIRAIS" \
      org.opencontainers.image.description="Methods for Observational Inference and Robust Analysis of Interventions in Scientific Experimentation (Python + R)" \
      org.opencontainers.image.url="https://github.com/hadesllm/moirais" \
      org.opencontainers.image.source="https://github.com/hadesllm/moirais" \
      org.opencontainers.image.documentation="https://hadesllm.github.io/moirais/" \
      org.opencontainers.image.licenses="GPL-2.0-only" \
      org.opencontainers.image.authors="Vansh Singh Ruhela <hadesllm@proton.me>" \
      org.opencontainers.image.vendor="hadesllm" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Runtime apt deps. Cache mounts on /var/cache/apt + /var/lib/apt
# stay across builds (their contents are never committed to image
# layers). Do NOT `rm -rf /var/lib/apt/lists/*` here — the rm would
# wipe the cache mount, defeating its purpose, while not changing the
# committed image since cache contents aren't in the layer anyway.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        tini \
        r-base-core

RUN groupadd -r moiraisapp -g 1000 \
    && useradd -r -u 1000 -g moiraisapp -m -d /home/moiraisapp -s /usr/sbin/nologin moiraisapp

# Pull in the installed Python tree from py-builder.
COPY --from=py-builder /install /usr/local

# Pull in the R site-library (incl. moirais and its CRAN deps).
COPY --from=r-builder /usr/local/lib/R/site-library /usr/local/lib/R/site-library

USER moiraisapp
WORKDIR /home/moiraisapp

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "from moirais.fn._registry import REGISTRY; assert len(REGISTRY) > 0" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["moirais", "--help"]
