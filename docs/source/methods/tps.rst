Toronto Police Service (TPS) Statistics
=======================================

Part of :doc:`index` — MORIE's statistical-methods reference.

MORIE provides a dedicated module set for analysing Toronto Police
Service open-data incident feeds. The data are fetched from
``data.torontopolice.on.ca`` and are public domain. All TPS analyses
are exposed under the ``morie.tps_*`` namespace.

Modules
-------

- ``morie.tps_io`` — fetch, cache, and parse the TPS open-data feeds.
- ``morie.tps_datasets`` — built-in TPS datasets (Assault, Auto
  Theft, Robbery, etc.).
- ``morie.tps_crime`` — top-line crime totals and incident-rate
  computations.
- ``morie.tps_csi`` — Statistics Canada Crime Severity Index
  calculations on TPS data.
- ``morie.tps_temporal`` — daily / weekly / monthly / yearly
  aggregations and trend tests.
- ``morie.tps_spatial`` — Moran's I, Geary's C, Getis-Ord G* on
  TPS neighbourhood-level counts.
- ``morie.tps_spatial_advanced`` — LISA, bivariate Moran, DBSCAN
  clustering, Kulldorff space-time scan, choropleths,
  proportional-symbol district maps.
- ``morie.tps_stochastic`` — Markovian Hawkes self-exciting process
  (constant baseline, exponential kernel) — the classical
  Mohler-Bertozzi-Brantingham fit.
- ``morie.tps_hawkes_advanced`` — non-stationary, non-Markovian
  Hawkes (Gamma / Weibull / Lomax kernels and sinusoidal baselines).
  See :doc:`hawkes` for the full treatment.
- ``morie.tps_statphysics`` — statistical physics of crime:
  Short-Brantingham reaction-diffusion PDE, Lévy-flight tail index,
  Bettencourt urban scaling, Lotka-Volterra predator-prey.
- ``morie.tps_render`` — plotting helpers shared across the TPS
  surface.
- ``morie.tps_all_analyze`` — orchestrator that runs the full TPS
  analysis pipeline.

Datasets
--------

The bundled feeds (post-2014, restricted to incident-level rows
released under the TPS open-data licence) include:

- Assault
- Auto Theft
- Bicycle Theft
- Break and Enter
- Homicide
- Robbery
- Sexual Violation
- Shooting
- Theft Over

Each is keyed by neighbourhood and timestamp and is suitable for
spatial, temporal, and combined space-time analyses.

Quick start
-----------

.. code-block:: python

   from morie.tps_io import load_tps
   from morie.tps_hawkes_advanced import compare_hawkes_kernels

   df = load_tps("Assault")
   results = compare_hawkes_kernels(df)
   print(results)  # ranks 8 (kernel x baseline) combinations by AIC

References
----------

The Hawkes-process methodology applied to these data is developed in
detail in the companion paper:

- Ruhela, V. S. (2026). *Criminological Hawkes Process via MORIE:
  Markovian and Non-Markovian Self-Exciting Point Processes for
  Toronto Crime.* Zenodo. https://doi.org/10.5281/zenodo.20102198

The statistical-physics components follow the D'Orsogna-Perc (2015)
review and the references cited therein.
