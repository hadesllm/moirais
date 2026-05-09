Contributing
============

Development setup
-----------------

.. code-block:: bash

   git clone <repo>
   cd moirais
   libexec/config/tests/rtests/bootstrap_moirais.sh
   # Creates .venv, installs Python editable + test/docs extras, installs R deps

Python tests
------------

.. code-block:: bash

   pytest -q                          # all tests (15976+ passing)

R tests
-------

.. code-block:: bash

   Rscript libexec/config/tests/rtests/test_r_package.R

Build docs
----------

.. code-block:: bash

   # From dev/sphinx/project/
   # or directly:
   python -m sphinx -b html docs/source docs/build/html

Documentation conventions
--------------------------

- Python docstrings use NumPy style (Napoleon parses them).
- Include ``:math:`` inline and ``.. math::`` display blocks for all estimators.
- R documentation uses roxygen2 markdown.
- New modules must be registered in ``moirais.modules.MODULE_SPECS`` with a
  ``ModuleSpec`` entry listing all output file names.
- Every non-trivial Python function needs a ``pytest`` test; every R export
  needs a ``testthat`` test.

Code quality
------------

- Type annotations on all Python function signatures.
- No hardcoded file paths; use ``DatasetRegistry`` or ``paths.R``.
- CPADS row-level data must never be committed. Synthetic data (``synthetic.R``)
  is acceptable for dev/CI.

Adding a new module
--------------------

1. Add a ``ModuleSpec`` entry to ``MODULE_SPECS`` in ``libexec/config/tools/py-package/moirais/modules.py``.
2. Add a branch in ``run_module()`` that calls an implementation function.
3. Implement the analysis function in the appropriate module file
   (``causal.py``, ``investigation.py``, etc.).
4. Add the module name to the R ``list_moirais_modules()`` table in
   ``libexec/config/tools/r-package/moirais/R/modules.R``.
5. Add a page to ``docs/source/modules/`` if the module has significant logic.
6. Write tests in ``libexec/config/tests/pytests/``.
