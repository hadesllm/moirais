Key Empirical Findings
=======================

Part of :doc:`index` — MOIRAIS's statistical-methods reference.

Headline numerical results that the MOIRAIS analysis surface produces
on the public Toronto Police Service and Ontario Tracking and
Information System (OTIS) data. Each item is the output of a specific
module call.

Spatial autocorrelation (Toronto Police Service, 2024)
-------------------------------------------------------

- **Polygon Moran's** :math:`I` **— Assault rate**: :math:`+0.328`
  (z = 90, p :math:`\approx 0`). Strong positive spatial
  autocorrelation across neighbourhoods.
- **Polygon Moran's** :math:`I` **— Homicide rate**: :math:`-0.005`
  (z = 0.4). Spatially random — Toronto homicides do not cluster at
  the neighbourhood polygon scale.

Computed via ``moirais.tps_spatial.morai`` over the TPS neighbourhood
shapefile.

Hawkes self-exciting point processes (post-2014 TPS)
----------------------------------------------------

- **Markovian Hawkes branching ratio** :math:`\hat{\eta} = 0.97`.
  Toronto Assault, classical Mohler exponential kernel + constant
  baseline (with U(0,1)-day jitter for tied OCC_DATE timestamps).
- :math:`\Delta\mathrm{AIC}` **(non-Markovian over Markovian)**
  :math:`\geq 80`. Holds across all 9 TPS categories. Weibull or
  Gamma kernel + sinusoidal baseline beats the classical
  Mohler-Bertozzi-Brantingham exponential + constant model under
  Kwan-Chen-Dunsmuir (2024).

Computed via ``moirais.tps_hawkes_advanced.compare_hawkes_kernels``;
see :doc:`hawkes` for the methodology and the companion paper at
`10.5281/zenodo.20102198
<https://doi.org/10.5281/zenodo.20102198>`__.

OTIS placements (Ontario Tracking and Information System)
---------------------------------------------------------

- **Goffmanian power-law exponent** :math:`\hat{\alpha} = 1.62`.
  Per-person placement-count distribution. Within the canonical
  1.5--2.5 preferential-attachment range (Barabási-Albert / Goffman
  *Asylums* 1961 chapter on the mortified self).
- **Pareto distribution AIC = 197 k**. Best fit for OTIS
  embedding-days distribution. Lognormal AIC = 207 k, Exponential
  AIC = 223 k.
- **RC / custody rate, Black individuals = 65.4 %**. OTIS c03 race
  :math:`\times` confinement breakdown. Versus 42.4 % for
  "Unknown / Not Reported".

Computed via ``moirais.otis_analyze`` and ``moirais.otis_causal``
modules; the OTIS-RC OU diagnostic quads (residuals / Q-Q /
scale-location / leverage) are produced by
``moirais.otis_all_analyze.otis_rc_diagnostics``.

How to reproduce
----------------

Each of the values above is the output of a single MOIRAIS function
call against the corresponding open-data feed. The ``moirais
run-module`` CLI subcommand executes any of the registered analysis
modules end-to-end and writes a results table to ``--output-dir``;
the headline numbers are extracted from those tables.
