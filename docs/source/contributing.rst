Contributing
============

Development setup
-----------------

.. code-block:: bash

   git clone https://github.com/hadesllm/morie.git
   cd morie
   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[test,interactive]"

Python tests
------------

.. code-block:: bash

   pytest -q

R tests
-------

.. code-block:: bash

   Rscript -e "library(morie); testthat::test_dir('r-package/morie/tests/testthat')"

Build docs
----------

.. code-block:: bash

   python -m sphinx -b html docs/source docs/build/html
   open docs/build/html/index.html

Documentation conventions
--------------------------

- Python docstrings use NumPy style (Napoleon parses them).
- Include ``:math:`` inline and ``.. math::`` display blocks for all estimators.
- R documentation uses roxygen2 markdown.
- New modules must be registered in ``morie.modules.MODULE_SPECS`` with a
  ``ModuleSpec`` entry listing all output file names.
- Every non-trivial Python function needs a ``pytest`` test; every R export
  needs a ``testthat`` test.

Code quality
------------

- Type annotations on all Python function signatures.
- No hardcoded file paths; use ``DatasetRegistry``.
- Row-level data from non-public sources must never be committed.

Adding a new module
--------------------

1. Add a ``ModuleSpec`` entry to ``MODULE_SPECS`` in ``src/morie/modules.py``.
2. Add a branch in ``run_module()`` that calls an implementation function.
3. Implement the analysis function in the appropriate module file
   (``causal.py``, ``investigation.py``, etc.).
4. Add the module name to the R ``list_morie_modules()`` table in
   ``r-package/morie/R/modules.R``.
5. Add a page to ``docs/source/methods/`` if the module has significant logic.
6. Write tests in ``tests/`` (Python) and ``r-package/morie/tests/testthat/`` (R).

License
-------

Contributions are accepted under ``GPL-2.0-only``. By submitting a pull
request you agree to license your contribution under that licence.
