Spatial Statistics
==================

Part of :doc:`index` — MORIE's statistical-methods reference.

MORIE provides a comprehensive spatial statistics library covering
global and local autocorrelation, geostatistical interpolation,
geographically weighted regression, and point pattern analysis. The
implementation achieves **100% chapter coverage** of Schabenberger &
Gotway (2005) across all 9 chapters.

All spatial functions live in ``morie/fn/`` as individual files and
accept arbitrary column names via keyword parameters.

Global Autocorrelation
----------------------

Measures of spatial dependence across the entire study region.

- ``morai`` — Moran's :math:`I`. Global spatial autocorrelation
  index in :math:`[-1, +1]`. Schabenberger & Gotway Ch. 7.
- ``geary`` — Geary's :math:`C`. Dissimilarity-based autocorrelation
  in :math:`[0, 2]`. Ch. 7.
- ``getgo`` — Getis-Ord general :math:`G`. Hot/cold-spot clustering
  for positive attributes. Ch. 7.
- ``jncnt`` — Join count. Binary spatial autocorrelation on
  categorical data. Ch. 7.

.. code-block:: python

   from morie.fn import morai, geary
   import numpy as np

   values = np.array([2.5, 3.1, 2.8, 4.0, 3.7])
   W = np.array([[0, 1, 0, 0, 1],
                 [1, 0, 1, 0, 0],
                 [0, 1, 0, 1, 0],
                 [0, 0, 1, 0, 1],
                 [1, 0, 0, 1, 0]], dtype=float)

   result = morai(values, W)
   print(f"Moran's I = {result.statistic:.4f}, p = {result.p_value:.4f}")

Local Indicators of Spatial Association (LISA)
----------------------------------------------

Local decompositions that identify clusters and spatial outliers.

- ``lisa`` — local Moran's :math:`I`. Per-location cluster / outlier
  classification (HH, HL, LH, LL).
- ``getgi`` — Getis-Ord :math:`G_i^{*}`. Local hot/cold-spot z-scores.
- ``lgeary`` — local Geary. Local dissimilarity decomposition.

Each function returns per-observation statistics, pseudo p-values from
conditional permutation tests (default 999 permutations), and cluster
classification labels.

Geostatistical Interpolation
-----------------------------

Prediction of values at unsampled locations using spatial covariance
structure.

**Variogram modelling**:

- ``svari`` — semivariogram estimation. Parameters: ``lag_dist``,
  ``n_lags``, ``model`` (spherical / exponential / gaussian).
- ``vfit`` — variogram fitting. Nugget, sill, range via weighted
  least squares.

**Interpolation methods**:

- ``krige`` — ordinary kriging. BLUP with variogram model; returns
  predictions and variance.
- ``ukrig`` — universal kriging. Kriging with external drift / trend
  surface.
- ``idw`` — inverse distance weighting. Power parameter (default
  :math:`p = 2`); no variance estimate.
- ``cokriging`` / ``cokrg`` — co-kriging. Multivariate kriging using
  cross-variograms.

.. code-block:: python

   from morie.fn import krige, svari

   gamma = svari(x, y, values, n_lags=15, model="spherical")
   predictions = krige(x, y, values, grid_x, grid_y,
                       variogram=gamma)

Geographically Weighted Regression (GWR)
-----------------------------------------

Spatially varying coefficient models that capture local heterogeneity
in regression relationships.

- ``gwr`` — basic GWR with Gaussian / bisquare kernel; bandwidth via
  AICc, cross-validation, or fixed.
- ``gwpca`` — geographically weighted PCA; adaptive bandwidth.
- ``stgwr`` — spatio-temporal GWR; joint space-time bandwidth.

Spatial Weight Matrices
-----------------------

Functions for constructing spatial connectivity structures:

- ``wqueen`` — queen contiguity (shared edge or vertex).
- ``wrook`` — rook contiguity (shared edge only).
- ``wknn`` — k-nearest neighbours.
- ``wdist`` — distance-based (threshold or inverse distance).
- ``wrow`` — row-standardise any weight matrix.

Point Pattern Analysis
----------------------

- ``ripk`` — Ripley's K-function and L-function for clustering /
  dispersion.
- ``stkde`` — spatio-temporal kernel density estimation.
- ``stscan`` — space-time scan statistic (Kulldorff).

Density-Based Clustering
------------------------

- ``stdbs`` — DBSCAN with spatio-temporal distance metric. Used in
  ``morie.tps_spatial_advanced`` to find connected hot clusters
  across the TPS incident feed and produce yearly small-multiples
  with a four-crime overlay (Assault / Robbery / Auto Theft /
  Break-and-Enter).
- ``hdbsc`` — HDBSCAN hierarchical variant for density-varying data.

Kulldorff Space-Time Scan
-------------------------

The Kulldorff scan statistic detects significant spatio-temporal
clusters by maximising a likelihood ratio over candidate cylindrical
windows in space-time. Implementation in ``stscan`` returns the most
likely cluster, its log-likelihood ratio, the Monte-Carlo p-value, and
the included neighbourhoods + time window. The Hohl-style
"panel d" visualisation is rendered by
``morie.tps_render.render_satscan_panel`` — it draws the
significant cluster polygons over a base choropleth.

Reference: Kulldorff, M. (1997). *A spatial scan statistic.*
Communications in Statistics 26(6):1481--1496.

Visualisation Helpers
---------------------

The ``morie.tps_render`` and ``morie.tps_spatial_advanced``
modules expose publication-style plot helpers commonly paired with
the spatial estimators above:

- ``morie.tps_render.render_choropleth`` — choropleth (per 100k,
  9 categories).
- ``morie.tps_render.render_district_proportional`` — district
  proportional-symbol map (6 former-borough divisions).
- ``morie.tps_render.render_quad`` — 4-panel quad
  (density / rate / LISA / :math:`G_i^{*}`).
- ``morie.tps_render.render_yearly_grid`` — yearly small-multiples
  (4-crime overlay + DBSCAN clusters).
- ``morie.tps_render.render_dbscan`` — DBSCAN cluster overlay.
- ``morie.tps_spatial_advanced.bivariate_moran`` — bivariate Moran
  scatter.
- ``morie.tps_spatial_advanced.moran_sweep_heatmap`` — Moran sweep
  heatmap (categories × years).
- ``morie.tps_render.render_satscan_panel`` — Kulldorff panel d
  (Hohl-style).

Style follows Hohl, A. *Geographic visualisation of disease cluster
detection results.*

Spatio-Temporal Extensions
--------------------------

25 spatio-temporal functions cover panel data, trajectory analysis,
and dynamic spatial processes: ``stacf``, ``stscan``, ``stkde``,
``stvar``, ``stgwr``, ``trajd``, ``stdbs``, ``ripk``, ``gwpca``,
``stmrn``, and 15 additional functions for spatio-temporal
autocorrelation, clustering, and prediction.

References
----------

.. [Schabenberger2005] Schabenberger, O. & Gotway, C.A. (2005).
   *Statistical Methods for Spatial Data Analysis*. Chapman & Hall /
   CRC. 9 chapters, 100% coverage in MORIE fn/ files.

.. [Armstrong2000] Armstrong, M.P. et al. (2000). Spatial voting
   analysis. 28/28 backends implemented.

.. [Anselin1995] Anselin, L. (1995). Local Indicators of Spatial
   Association — LISA. *Geographical Analysis* 27(2):93--115.
