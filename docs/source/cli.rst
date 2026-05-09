CLI Reference
=============

The ``moirais`` command is the primary terminal interface. It is installed as a
console script entry point from ``moirais.runner:main``.

Commands
--------

``list-modules``
~~~~~~~~~~~~~~~~

Print the full module surface with descriptions.

.. code-block:: bash

   moirais list-modules

``run-module``
~~~~~~~~~~~~~~

Run a single named analysis module.

.. code-block:: bash

   moirais run-module <module-name> \
     --cpads-csv path/to/cpads.csv \
     --output-dir /tmp/moirais-outputs

``pipeline``
~~~~~~~~~~~~

Run all (or selected) modules in sequence.

.. code-block:: bash

   moirais pipeline --all -y
   moirais pipeline --modules power-design logistic-models -y
   moirais pipeline --cpads-csv custom.csv --output-dir ./out

``ask``
~~~~~~~

Query the terminal assistant (requires ``moirais[ai]`` and ``OPENAI_API_KEY``,
or runs in local fallback mode).

.. code-block:: bash

   moirais ask "What does the propensity-scores module output?"

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

A shell wrapper ``./moirais`` at the repository root invokes
``.venv/bin/python -m moirais.runner`` so the venv does not need to be activated
manually:

.. code-block:: bash

   ./moirais list-modules
   ./moirais pipeline --all -y
