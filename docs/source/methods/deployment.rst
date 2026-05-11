Deployment
==========

Part of :doc:`index` — MORIE's statistical-methods reference.

MORIE supports deployment across consumer hardware, single-board computers,
and containers. The target is a single-command install that auto-detects
and configures all dependencies.

From source
-----------

.. code-block:: bash

   git clone https://github.com/hadesllm/morie.git
   cd morie
   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[interactive]"

The package is pure Python at runtime; the only platform-specific
notes are: NumPy / SciPy / scikit-learn / matplotlib must build (or
download wheels) for your architecture. ``morie doctor`` reports
any missing optional dependencies.

Docker
------

MORIE provides a multi-architecture Docker image for inspection, testing,
and verification.

**Dockerfile highlights:**

- Base image pinned by digest (not tag) for reproducibility
- Non-root user ``morieapp`` (UID 1000)
- Read-only root filesystem with ``/tmp`` writable
- Health check via ``morie selftest``
- Layer ordering: system deps, Python deps, then source (cache-friendly)

.. code-block:: bash

   docker build -t morie:latest .
   docker run --rm morie:latest morie selftest
   docker run --rm morie:latest morie verify
   docker run --rm -it morie:latest morie repl

**Multi-arch build (amd64 + arm64):**

.. code-block:: bash

   docker buildx build --platform linux/amd64,linux/arm64 \
     -t ghcr.io/hadesllm/morie:latest --push .

.. note::

   The published image lives at ``ghcr.io/hadesllm/morie`` post-migration
   (not ``ghcr.io/hadesllm/morie``). The package is currently private —
   authenticate with a PAT that has ``read:packages`` scope before pulling.

Ollama Sidecar
--------------

For LLM-enabled deployments, Ollama runs as a sidecar container:

.. code-block:: yaml

   # docker-compose.yml
   services:
     morie:
       image: morie:latest
       environment:
         - OLLAMA_HOST=ollama:11434
       depends_on:
         - ollama
     ollama:
       image: ollama/ollama:latest
       volumes:
         - ollama_data:/root/.ollama
       ports:
         - "11434:11434"
   volumes:
     ollama_data:

MORIE auto-detects the Ollama sidecar via the ``OLLAMA_HOST`` environment
variable. No additional configuration is needed.

systemd Service
---------------

For persistent deployments (e.g., Pi running as a headless analysis server):

.. code-block:: ini

   [Unit]
   Description=MORIE Analysis Server
   After=network.target ollama.service

   [Service]
   Type=simple
   User=morieapp
   WorkingDirectory=/opt/morie
   ExecStart=/opt/morie/.venv/bin/python -m morie.runner serve
   Restart=on-failure
   RestartSec=5
   Environment=OLLAMA_HOST=localhost:11434

   [Install]
   WantedBy=multi-user.target

Install Methods
---------------

Current and planned installation paths:

- **pip (editable)** — active. ``pip install -e ".[test,docs]"``.
- **pip (PyPI)** — active. ``pip install morie``.
- **brew** — planned. ``brew install morie``.
- **curl installer** — planned. ``curl -fsSL https://morie.dev/install | sh``.
- **Docker** — active. ``docker pull ghcr.io/hadesllm/morie:latest``.

The curl installer will auto-detect the platform (macOS/Linux/WSL),
install R, Python, and Quarto as needed, and configure the environment.

Cross-Platform Notes
--------------------

- **macOS (Apple Silicon)**: Primary development platform. Use CRAN R
  (not brew R) to avoid segfaults with RSQLite. scipy requires
  ``PKG_CONFIG_PATH="/opt/homebrew/opt/openblas/lib/pkgconfig"``.
- **Linux (x86_64)**: Fully supported. Docker is the recommended
  deployment method for production.
- **Linux (arm64)**: ARM64 Linux supported. Ollama ARM64 builds work natively.
- **Windows**: WSL2 with Ubuntu is the supported path. Native Windows
  is not tested.
