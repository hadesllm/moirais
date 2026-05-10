MOIRAIS — Methods for Observational Inference and Robust Analysis of Interventions in Scientific Experimentation
=======================================================

.. image:: https://img.shields.io/badge/license-GPL--2.0-blue.svg
   :alt: License: GPL-2.0

.. image:: https://img.shields.io/badge/python-3.10%2B-blue.svg
   :alt: Python 3.10+

.. image:: https://img.shields.io/badge/R-4.3%2B-276DC3.svg
   :alt: R 4.3+

.. image:: https://img.shields.io/pypi/v/moirais.svg
   :target: https://pypi.org/project/moirais/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20096350-blue
   :target: https://doi.org/10.5281/zenodo.20096350
   :alt: DOI 10.5281/zenodo.20096350

A dual-language (Python + R) multi-domain scientific computing toolkit for
observational inference, with sociolegal, signal-processing, cryptographic,
spatial-statistics, statistical-physics, and psychometrics modules. Hosts
the DLRM (Doob–Levinsky–Ruhela–Medina) framework as a primary application
for Canadian carceral, police, and oversight data analysis.

----

Quick start
-----------

Install via pip, Homebrew (forthcoming), or from source:

.. code-block:: bash

   pip install moirais # Python package (41 built-in datasets)
   pip install "moirais[interactive]" # + Terminal IDE (TUI) with textual
   pip install "moirais[carbon]" # + CodeCarbon emissions (Python ≤3.14 only)

   # R package (from source)
   install.packages("r-package/moirais", repos = NULL, type = "source")

Run your first analysis in seconds:

.. code-block:: bash

   # Launch the Terminal IDE (multi-pane IDE)
   moirais tui

   # Self-diagnostics — checks LLM providers, datasets, R, Docker
   moirais doctor

   # List all 41 built-in datasets
   moirais list-datasets

   # List all 23 analysis modules
   moirais list-modules

   # Run a single module against built-in data
   moirais run-module power-design --output-dir /tmp/moirais-outputs

   # Run the full pipeline (with enlighten progress bars)
   moirais pipeline --all -y

   # Start free AI chat (no API key needed)
   moirais chat

From R:

.. code-block:: r

   library(moirais)

   # Load built-in dataset (DBI/RSQLite — no file paths needed)
   cpads <- moirais_load_dataset("cpads_2021")

   # List all 41 built-in datasets
   moirais_list_datasets()

   # Browse dataset catalog
   moirais_dataset_catalog()

   # Estimate average treatment effect
   ate <- estimate_ate(cpads, "outcome", "treatment", c("age", "sex"))

----

What MOIRAIS does
--------------

MOIRAIS provides a unified, enterprise-grade interface across Python and R for:

**Causal inference estimators**
  ATE, ATT, ATC, GATE, CATE (T/S-learner), LATE (2SLS/Wald), AIPW, IPW,
  G-computation, Rosenbaum sensitivity bounds, E-value.

**Double Machine Learning (DML)**
  Partially Linear Regression (PLR), Interactive Regression Model (IRM),
  Partially Linear IV (PLIV). Full cross-fitting with pluggable nuisance
  learners (Random Forest, Gradient Boosting, any scikit-learn estimator).

**Propensity scores & weighting**
  Logistic regression PS estimation, overlap/positivity checks, weight
  diagnostics, stabilised IPW, SMOTE sensitivity analysis.

**Survey-weighted statistics**
  Horvitz-Thompson totals, Hajek means, ratio estimators, calibration
  weights (raking/IPF), complex survey GLM, subpopulation estimates.

**Sampling design**
  SRS, stratified, cluster, PPS, bootstrap, jackknife, design-effect
  calculations, effective sample size.

**eBAC — Estimated Blood Alcohol Concentration**
  Legal-limit checks, sex-stratified eBAC formulas, IPW-adjusted eBAC models.

**Bayesian inference**
  Prior-posterior updating, Bayes factors, 26 distribution functions
  (dnorm, dbinom, dpois, dbeta, …), credible intervals.

**Frequentist inference**
  11 hypothesis tests, 5 confidence interval methods, 9 effect size measures,
  4 power analysis functions.

**Carbon-aware computing**
  Built-in emissions tracker (``moirais.emissions``): per-module and pipeline-wide
  CO₂ tracking with 213-country IEA carbon intensity data. Pure Python — no
  Rust dependencies. CodeCarbon fallback on Python ≤3.14.

**LLM agent**
  Ollama (local) → OllamaFreeAPI (vendored, no API key) → Gemini → local
  fallback. Zero cloud dependency. 1296 interactive REPL helpers. Polyglot
  mode bridges R↔Python variables.

----

Key design principles
---------------------

*Lean terminal IDE.*
  Rich terminal output — progress bars, formatted ASCII tables, color-coded
  diagnostics. Run entire pipelines from a single ``moirais`` command.

*Python + R parity.*
  Every statistical estimator is implemented in both languages with matching
  APIs. Python uses scikit-learn conventions (``fit`` / ``predict``). R uses
  S3 generics (``summary()``, ``plot()``, ``predict()``).

*Automated documentation.*
  Python API docs via Sphinx autodoc. R API docs via Roxygen2 → ``.. r:function::``
  (no manual writing). Run ``devtools::document()`` to regenerate.

*Data governance built-in.*
  Raw CPADS microdata lives in ``data/datasets/``. Wrangled cache in ``data/cache/``.
  Synthetic data (``generate_synthetic_data()``) is labeled synthetic in all
  outputs. ``moirais verify`` (planned) will validate manifest output provenance.

*Statistically rigorous.*
  Target estimand is always an explicit parameter (ATE vs ATT vs CATE — never
  implicit). Overlap/positivity violations raise explicit warnings. Cross-fitting
  prevents data leakage. Convergence diagnostics are built into MCMC outputs.

----

Background
----------

MOIRAIS was built to fill a gap between academic one-off scripts and enterprise
health analytics platforms. It targets epidemiologists, biostatisticians, and
public health researchers who need:

- Causal methods that go beyond basic logistic regression
- Reproducible pipelines that run in CI/CD without manual intervention
- A unified Python + R surface (not choosing one or the other)
- Carbon-aware computing for responsible public health AI
- Canadian public health datasets (CPADS, CIHI) as first-class inputs

MOIRAIS is licensed under GPL-2.0-only (Linus copyleft, deliberately
chosen over GPL-3.0 for compatibility with the broader Linux-kernel-style
ecosystem). See ``LICENSE`` for the full text and ``LICENSING.md`` for
the rationale.

----

Documentation index
-------------------

If you prefer a single linear walkthrough rather than the sidebar
navigation, every page on this site is listed below — top to bottom:

----

.. toctree::
   :maxdepth: 1
   :caption: Navigation
   :hidden:

   learn/index
   install
   cli

.. toctree::
   :maxdepth: 2
   :caption: Documentation
   :hidden:

   methods/index
   howto/index

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   contributing

.. Visible-in-body site index — renders as a plain bullet list inline.

.. toctree::
   :maxdepth: 2

   learn/index
   install
   cli
   methods/index
   howto/index
   contributing
