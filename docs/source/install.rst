Installation
============

MORIE targets Python ≥ 3.14 and R ≥ 4.3. Both languages are supported
independently; you do not need both unless you are running the full dual-language
pipeline. Python 3.15 is the recommended primary version.

.. note::
   **About the ``#`` characters in code blocks below.** Lines starting
   with ``#`` (and inline ``# …`` annotations after a command) are
   *comments* — they explain what the command does, but they are NOT
   part of the command itself.  Pasting them into a shell will either
   error out or run something you didn't mean to.

   Hover any code block on this page and click **Copy** in the corner
   — the smart-copy strips ``# comments`` automatically so the result
   is paste-safe.  ``Cmd/Ctrl+C`` on a selection inside a code block
   does the same thing.

Python
------

From PyPI (forthcoming public release):

.. code-block:: bash

   pip install morie

Editable install from source (development):

.. code-block:: bash

   git clone <repo>
   cd morie/dev/sphinx/project

   # Primary: Python 3.15 (recommended)
   python3.15 -m venv .venv
   PKG_CONFIG_PATH="/opt/homebrew/opt/openblas/lib/pkgconfig" \
       .venv/bin/pip install -e ".[test,docs]"

   # Secondary: Python 3.14 (adds codecarbon with pydantic support)
   python3.14 -m venv .venv-314
   .venv-314/bin/pip install -e ".[test,docs,carbon]"

.. note::

   On macOS with Apple Silicon, scipy requires OpenBLAS. Set
   ``PKG_CONFIG_PATH="/opt/homebrew/opt/openblas/lib/pkgconfig"``
   before ``pip install``.

   The ``[carbon]`` extra (CodeCarbon emissions tracking) requires
   Python ≤ 3.14 due to pydantic-core Rust dependencies. On Python 3.15+,
   MORIE uses the built-in ``morie.emissions`` tracker instead.

Editable install with all extras (creates ``.venv``, installs Python deps):

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[test,interactive]"

R
-

Install from local source:

.. code-block:: r

   install.packages("r-package/morie", repos = NULL, type = "source")

Required R packages (installed automatically):

.. code-block:: r

   install.packages(c("survey", "testthat"), repos = "https://cloud.r-project.org/")

.. note::

   On Apple Silicon, use CRAN R (not Homebrew R) to avoid RSQLite segfaults.
   Download from https://cran.r-project.org. Run ``brew unlink r`` if needed.
   Current recommended: R 4.5.3 (CRAN).

macOS
-----

.. code-block:: bash

   # Install CRAN R (not brew R — see note above)
   curl -O https://cran.r-project.org/bin/macosx/big-sur-arm64/base/R-4.5.3-arm64.pkg
   sudo installer -pkg R-4.5.3-arm64.pkg -target /
   brew unlink r 2>/dev/null

   # Python 3.15
   brew install python@3.15
   pip install morie

Linux
-----

.. code-block:: bash

   # Debian/Ubuntu
   sudo apt-get install r-base python3 python3-pip
   pip3 install morie

Windows
-------

.. code-block:: powershell

   winget install -e --id RProject.R
   winget install -e --id Python.Python.3.14
   pip install morie

Verifying the install
---------------------

.. code-block:: bash

   morie list-modules          # prints the registered analysis modules
   morie doctor                # checks LLM providers, datasets, R, Docker
   morie --help

.. code-block:: r

   library(morie)
   list_morie_modules()


LLM provider setup
------------------

MORIE supports LLM providers tried in priority order. The FreeAPI client is
vendored (``morie.freeapi``) — no external SDK or API key required.

Providers, in priority order:

1. **Ollama** (local, private) — install with
   ``curl -fsSL https://ollama.com/install.sh | sh && ollama pull qwen2.5:7b``.
2. **OllamaFreeAPI** (free, no key) — built-in. No setup needed; uses
   ``gpt-oss:20b`` (Gemma3) by default.
3. **Gemini** (free tier) — ``export GEMINI_API_KEY=...`` (free key
   at `aistudio.google.com <https://aistudio.google.com>`_).
4. **Local fallback** — automatic. Keyword-matched help text, no
   network required.

If no provider is configured, ``morie ask`` returns a static help response.
Run ``morie doctor`` to see which providers are currently available.
