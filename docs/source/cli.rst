CLI Reference
=============

The ``morie`` command is the primary terminal interface. It is installed as a
console script entry point from ``morie.runner:main``.

Commands
--------

``list-modules``
~~~~~~~~~~~~~~~~

Print the full module surface with descriptions.

.. code-block:: bash

   morie list-modules

``run-module``
~~~~~~~~~~~~~~

Run a single named analysis module.

.. code-block:: bash

   morie run-module <module-name> \
     --cpads-csv path/to/cpads.csv \
     --output-dir /tmp/morie-outputs

``pipeline``
~~~~~~~~~~~~

Run all (or selected) modules in sequence.

.. code-block:: bash

   morie pipeline --all -y
   morie pipeline --modules power-design logistic-models -y
   morie pipeline --cpads-csv custom.csv --output-dir ./out

``ask``
~~~~~~~

Query the terminal assistant (requires ``morie[ai]`` and ``OPENAI_API_KEY``,
or runs in local fallback mode).

.. code-block:: bash

   morie ask "What does the propensity-scores module output?"

Global options
--------------

``--cpads-csv``
   Path to the CPADS PUMF CSV. Defaults to
   ``docs/source/datasets/cpads-2021-2022-pumf2.csv``.

``--output-dir``
   Directory for CSV outputs. Defaults to ``data/manifest/outputs/``.

``-y`` / ``--yes``
   Skip the interactive confirmation prompt before running modules.

Repo-root wrapper
-----------------

A shell wrapper ``./morie`` at the repository root invokes
``.venv/bin/python -m morie.runner`` so the venv does not need to be activated
manually:

.. code-block:: bash

   ./morie list-modules
   ./morie pipeline --all -y
