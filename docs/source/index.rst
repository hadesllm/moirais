MORIE — Multi-domain Open Research and Inferential Estimation
=============================================================

.. image:: https://img.shields.io/badge/license-GPL--2.0-d97706.svg
   :alt: License: GPL-2.0

.. image:: https://img.shields.io/badge/python-3.10%2B-blue.svg
   :alt: Python 3.10+

.. image:: https://img.shields.io/badge/R-4.3%2B-276DC3.svg
   :alt: R 4.3+

.. image:: https://img.shields.io/pypi/v/morie.svg
   :target: https://pypi.org/project/morie/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/r--universe-hadesllm-276DC3
   :target: https://hadesllm.r-universe.dev/morie
   :alt: r-universe

.. image:: https://img.shields.io/badge/DOI%20%C2%B7%20morie%20R-10.5281%2Fzenodo.20111233-0d9488?logo=zenodo&logoColor=white
   :target: https://doi.org/10.5281/zenodo.20111233
   :alt: DOI - morie R - 10.5281/zenodo.20111233

.. image:: https://img.shields.io/badge/DOI%20%C2%B7%20morie%20Python-10.5281%2Fzenodo.20096350-7c3aed?logo=zenodo&logoColor=white
   :target: https://doi.org/10.5281/zenodo.20096350
   :alt: DOI - morie Python - 10.5281/zenodo.20096350

.. image:: https://img.shields.io/badge/MRM_paper-10.5281%2Fzenodo.20096075-15803d?logo=zenodo&logoColor=white
   :target: https://doi.org/10.5281/zenodo.20096075
   :alt: MRM paper - 10.5281/zenodo.20096075

.. image:: https://img.shields.io/badge/Hawkes_paper-10.5281%2Fzenodo.20102198-be123c?logo=zenodo&logoColor=white
   :target: https://doi.org/10.5281/zenodo.20102198
   :alt: Hawkes paper - 10.5281/zenodo.20102198

A dual-language (Python + R) multi-domain scientific computing toolkit for
observational inference, with sociolegal, signal-processing, cryptographic,
spatial-statistics, statistical-physics, and psychometrics modules. Hosts
the MRM (Multilevel Reconciliation Methodology) framework as a primary application
for Canadian carceral, police, and oversight data analysis.

----

Quick start
-----------

Install via pip, Homebrew (forthcoming), or from source:

.. code-block:: bash

   pip install morie # Python package (60+ built-in datasets)
   pip install "morie[interactive]" # + Terminal IDE (TUI) with textual
   pip install "morie[carbon]" # + CodeCarbon emissions (Python ≤3.14 only)

   # R package (from source)
   install.packages("r-package/morie", repos = NULL, type = "source")

Run your first analysis in seconds:

.. code-block:: bash

   # Launch the Terminal IDE (multi-pane IDE)
   morie tui

   # Self-diagnostics — checks LLM providers, datasets, R, Docker
   morie doctor

   # List all 60+ built-in datasets
   morie list-datasets

   # List all 23 analysis modules
   morie list-modules

   # Run a single module against built-in data
   morie run-module power-design --output-dir /tmp/morie-outputs

   # Run the full pipeline (with enlighten progress bars)
   morie pipeline --all -y

   # Start free AI chat (no API key needed)
   morie chat

From R:

.. code-block:: r

   library(morie)

   # Load built-in dataset (DBI/RSQLite — no file paths needed)
   cpads <- morie_load_dataset("cpads_2021")

   # List all 60+ built-in datasets
   morie_list_datasets()

   # Browse dataset catalog
   morie_dataset_catalog()

   # Estimate average treatment effect
   ate <- estimate_ate(cpads, "outcome", "treatment", c("age", "sex"))

----

What MORIE does
-----------------

A unified Python + R interface across the following surfaces. See
:doc:`methods/index` for methodology details and :doc:`api/index`
for function reference.

**Causal estimators**
  ATE, ATT, ATC, GATE, CATE (T- / S-learner), LATE (2SLS / Wald),
  AIPW, IPW (Hájek), G-computation, propensity-score matching
  (1:1 NN, 5-strata subclass), Rosenbaum sensitivity bounds, E-value.

**Double machine learning**
  Partially linear regression (PLR), interactive regression model
  (IRM), partially linear IV (PLIV); cross-fitted with pluggable
  nuisance learners. Multi-SE comparison (pooled, cluster, multi-way)
  on the IRM-DML primary estimate. Propensity calibration (Platt /
  isotonic) on IPW / AIPW / SuperLearner-AIPW with Brier score.

**The MRM framework**
  Multilevel Reconciliation Methodology — a 10-estimator framework applied to OTIS
  / SIU / TPS data over a coordinated set of (treatment, outcome,
  covariates) designs. Per-row individual-level + aggregate (Poisson,
  NB GLM) modes. Mandela classifier (UN Mandela Rules 43 + 44) +
  provincial-vs-federal cross-comparison.

**Spatial statistics**
  Moran's :math:`I`, Geary's :math:`C`, Getis-Ord general :math:`G`,
  join count, LISA, Getis-Ord :math:`G_i^{*}`, local Geary, Ripley's
  K / L, geostatistical kriging (ordinary, universal, IDW,
  co-kriging), variogram fitting, GWR (basic, GW-PCA, ST-GWR),
  bivariate Moran, Moran sweep heatmap, DBSCAN / HDBSCAN, Kulldorff
  space-time scan.

**Hawkes self-exciting point processes**
  Markovian Mohler-Bertozzi-Brantingham fit (exponential kernel +
  constant baseline) plus the non-stationary, non-Markovian
  Kwan-Chen-Dunsmuir (2024) family — Gamma, Weibull, Lomax kernels
  with sinusoidal baselines. Eight (kernel × baseline) combinations
  ranked by AIC and time-rescaling Kolmogorov-Smirnov goodness-of-fit.

**Statistical physics of crime**
  Short-Brantingham reaction-diffusion PDE, Brockmann-Hufnagel-Geisel
  Lévy-flight tail (Hill estimator), Bettencourt urban-scaling
  exponent (HC3-OLS), D'Orsogna-Perc Lotka-Volterra predator-prey,
  SDB Turing-pattern demo, Helbing-Szolnoki inspection-game phase
  diagram, criminal-role co-occurrence networks.

**Survey-weighted inference**
  Horvitz-Thompson totals, Hájek means, ratio estimators, calibration
  weights (raking / IPF), complex-survey GLM, subpopulation estimates,
  stratified / cluster / PPS sampling, bootstrap + jackknife variance,
  design-effect computations, effective-sample-size diagnostics.

**Psychometrics**
  Cronbach's :math:`\alpha`, McDonald's :math:`\omega_t` /
  :math:`\omega_h`, KMO sampling adequacy, Bartlett's sphericity,
  parallel analysis, composite reliability, AVE, item-response-theory
  fits (1PL / 2PL / 3PL / GRM / PCM), differential item functioning
  (Mantel-Haenszel, logistic, generalised), measurement invariance,
  network psychometrics, Bayesian psychometrics. 250+ functions.

**Signal processing + cryptography**
  Spectral analysis, biomedical-signal helpers, homomorphic
  deconvolution, classical and modern crypto primitives
  (ChaCha20-Poly1305, etc.), TurboQuant vector quantization with
  near-optimal distortion (Zandieh et al. 2026 ICLR).

**Datasets**
  60+ built-in datasets in a portable SQLite layer (Canadian
  carceral, police, and oversight + epidemiological reference data).
  Auto dataset-profiling for arbitrary tabular input
  (``morie.dataset.profile_dataset``).

**Function namespace ``morie.fn``**
  36,000+ individual function files indexed by a registry, exposing
  short stable names for every estimator, every kernel,
  every weight matrix, every test. Use ``morie.fn.cheatsheet(name)``
  for a per-function help card.

**Federal SIU + Doob T-539-20 replication**
  Mandela classifier (Rules 43 + 44) with χ² verification, Sprott /
  Doob (± Iftene) IEDM analyses, full replication of Doob's CCRSO 2018
  Tables 1--3 and the imprisonment-vs-crime decoupling Pettitt
  change-point test. See :doc:`methods/sprott_doob`,
  :doc:`methods/doob_trends`, :doc:`methods/siuiap`.

**Toronto Police Service surface**
  ``morie.tps_*`` modules: incident I/O, CSI, neighbourhood
  spatial / temporal analyses, Hawkes (basic + advanced),
  statistical physics, Hohl-style choropleths and proportional-symbol
  district maps. Companion paper at 10.5281/zenodo.20102198.

**LLM + assistant**
  Ollama (local, private) → vendored OllamaFreeAPI (no key) → Gemini
  free tier → local-keyword fallback. Zero cloud dependency at the
  default tier. Vendored TurboQuant KV-cache compression. Polyglot
  REPL bridges variables across Python ↔ R ↔ shell ↔ 12 other
  languages.

**Carbon-aware computing**
  Built-in pure-Python emissions tracker
  (``morie.emissions``) with 213-country IEA carbon-intensity
  data, per-module and pipeline-wide CO₂ accounting. CodeCarbon
  fallback on Python ≤ 3.14.

----

Key design principles
---------------------

*Lean terminal IDE.*
  Rich terminal output — progress bars, formatted ASCII tables, color-coded
  diagnostics. Run entire pipelines from a single ``morie`` command.

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
  outputs. ``morie verify`` (planned) will validate manifest output provenance.

*Statistically rigorous.*
  Target estimand is always an explicit parameter (ATE vs ATT vs CATE — never
  implicit). Overlap/positivity violations raise explicit warnings. Cross-fitting
  prevents data leakage. Convergence diagnostics are built into MCMC outputs.

----

Background
----------

MORIE is a multi-domain scientific computing toolkit for
observational inference. It sits between one-off research scripts
and heavy enterprise analytics platforms, and is aimed at
researchers who need:

- A unified Python + R surface across the same estimators (no
  language-choice tax).
- Causal estimators (ATE / ATT / ATC / GATE / CATE / LATE, AIPW,
  G-computation, DML--PLR, DML--IRM, propensity-score matching,
  E-value and Rosenbaum-bound sensitivity) with explicit estimands.
- Survey-weighted inference (Horvitz-Thompson, Hájek, raking,
  cluster + stratified design) on top of the same DataFrame as the
  causal layer.
- Spatial statistics (Moran's :math:`I`, LISA, Getis-Ord
  :math:`G^{*}`, DBSCAN, Kulldorff space-time scan), Hawkes
  self-exciting point processes (Markovian and non-Markovian), and
  the statistical-physics-of-crime models (Short-Brantingham
  reaction-diffusion, Lévy-flight tail, Bettencourt urban scaling,
  Lotka-Volterra) — applied as first-class methods on the
  Toronto Police Service open-data feeds.
- Reproducible pipelines that run unattended in CI / CD — outputs
  carry provenance manifests; synthetic data is labelled as such.
- The MRM (Multilevel Reconciliation Methodology) framework as a primary
  application for Canadian carceral, police, and oversight data
  (Ontario OTIS, federal SIU, TPS).

The package ships 60+ built-in datasets (Canadian carceral, police,
and oversight + epidemiological reference data) in a portable SQLite
layer.

MORIE is licensed under GPL-2.0-only (Linus copyleft, deliberately
chosen over GPL-3.0 for compatibility with the broader
Linux-kernel-style ecosystem). See ``LICENSE`` for the full text and
``LICENSING.md`` for the rationale.

----

Documentation index
-------------------

If you prefer a single linear walkthrough rather than the sidebar
navigation, every page on this site is listed below — top to bottom:

- :doc:`learn/index` — From-zero tutorial track. Start here if you
  have never opened a Python or R console before.
- :doc:`install` — Installation instructions for Python, R, macOS,
  Linux, Windows, plus LLM provider setup.
- :doc:`cli` — Reference for every ``morie …`` subcommand.
- :doc:`methods/index` — Statistical-methods reference. Estimands,
  causal estimators, survey statistics, spatial methods, Hawkes
  processes, statistical physics of crime, OTIS / TPS / SIU
  pipelines, the MRM framework, key empirical findings.
- :doc:`api/index` — Python and R API reference
  (function signatures and docstrings).
- :doc:`contributing` — Development setup, test conventions,
  module-addition guide.

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

   architecture
   methods/index
   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   contributing
   acknowledgments
