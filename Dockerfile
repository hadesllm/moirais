# syntax=docker/dockerfile:1.7
#
# morie container — 3-stage Python + R build, single-arch (linux/amd64).
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

# Since v0.9.1 morie ships a compiled C++ core (libmorie) built with
# scikit-build-core + CMake + nanobind, so the builder needs a C/C++
# toolchain plus CMake and Ninja.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        cmake \
        ninja-build

WORKDIR /build

# A compiled (scikit-build-core) build cannot be staged from a stub
# package the way a pure-Python build could: CMake needs the real
# CMakeLists.txt and the libmorie/ C++ sources present. Copy the full
# build input and install once. The pip layer is still cache-mounted,
# so unchanged dependencies are reused across builds.
#
# LICENSE is required: pyproject.toml declares license-files =
# ["LICENSE"], and scikit-build-core fails metadata generation if the
# pattern matches nothing.
COPY pyproject.toml README.md CMakeLists.txt LICENSE ./
COPY libmorie/ ./libmorie/
COPY src/ ./src/

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --root-user-action=ignore --prefix=/install . \
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
RUN R CMD INSTALL --library=/usr/local/lib/R/site-library /build/r-package/morie

# ─── Stage 3: Runtime ────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG VERSION=0.9.4
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown

LABEL org.opencontainers.image.title="morie" \
      org.opencontainers.image.description="Multi-domain Open Research and Inferential Estimation -- a multi-domain scientific computing toolkit for observational inference and intervention analysis (Python + R)" \
      org.opencontainers.image.url="https://github.com/hadesllm/morie" \
      org.opencontainers.image.source="https://github.com/hadesllm/morie" \
      org.opencontainers.image.documentation="https://hadesllm.github.io/morie/" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later" \
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

RUN groupadd -r morieapp -g 1000 \
    && useradd -r -u 1000 -g morieapp -m -d /home/morieapp -s /usr/sbin/nologin morieapp

# Pull in the installed Python tree from py-builder.
COPY --from=py-builder /install /usr/local

# Pull in the R site-library (incl. morie and its CRAN deps).
COPY --from=r-builder /usr/local/lib/R/site-library /usr/local/lib/R/site-library

USER morieapp
WORKDIR /home/morieapp

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "from morie.fn._registry import REGISTRY; assert len(REGISTRY) > 0" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["morie", "--help"]
