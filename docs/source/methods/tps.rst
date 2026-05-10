Toronto Police Service (TPS) Statistics
=======================================

MOIRAIS provides a dedicated module set for analysing Toronto Police
Service open-data incident feeds. The data are fetched from
``data.torontopolice.on.ca`` and are public domain. All TPS analyses
are exposed under the ``moirais.tps_*`` namespace.

Modules
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - Purpose
   * - ``moirais.tps_io``
     - Fetch, cache, and parse the TPS open-data feeds.
   * - ``moirais.tps_datasets``
     - Built-in TPS datasets (Assault, Auto Theft, Robbery, etc.).
   * - ``moirais.tps_crime``
     - Top-line crime totals and incident-rate computations.
   * - ``moirais.tps_csi``
     - Statistics Canada Crime Severity Index calculations on TPS data.
   * - ``moirais.tps_temporal``
     - Daily / weekly / monthly / yearly aggregations and trend tests.
   * - ``moirais.tps_spatial``
     - Moran's I, Geary's C, Getis-Ord G* on TPS neighbourhood-level
       counts.
   * - ``moirais.tps_spatial_advanced``
     - LISA, bivariate Moran, DBSCAN clustering, Kulldorff space-time
       scan, choropleths, proportional-symbol district maps.
   * - ``moirais.tps_stochastic``
     - Markovian Hawkes self-exciting process (constant baseline,
       exponential kernel) — the classical Mohler-Bertozzi-Brantingham
       fit.
   * - ``moirais.tps_hawkes_advanced``
     - Non-stationary, non-Markovian Hawkes (Gamma / Weibull / Lomax
       kernels and sinusoidal baselines). See :doc:`hawkes` for the
       full treatment.
   * - ``moirais.tps_statphysics``
     - Statistical physics of crime: Short-Brantingham reaction-
       diffusion PDE, Lévy-flight tail index, Bettencourt urban
       scaling, Lotka-Volterra predator-prey.
   * - ``moirais.tps_render``
     - Plotting helpers shared across the TPS surface.
   * - ``moirais.tps_all_analyze``
     - Orchestrator that runs the full TPS analysis pipeline.

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

   from moirais.tps_io import load_tps
   from moirais.tps_hawkes_advanced import compare_hawkes_kernels

   df = load_tps("Assault")
   results = compare_hawkes_kernels(df)
   print(results)  # ranks 8 (kernel x baseline) combinations by AIC

References
----------

The Hawkes-process methodology applied to these data is developed in
detail in the companion paper:

- Ruhela, V. S. (2026). *Criminological Hawkes Process via MOIRAIS:
  Markovian and Non-Markovian Self-Exciting Point Processes for
  Toronto Crime.* Zenodo. https://doi.org/10.5281/zenodo.20102198

The statistical-physics components follow the D'Orsogna-Perc (2015)
review and the references cited therein.
