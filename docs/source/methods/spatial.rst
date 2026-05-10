Spatial Statistics
==================

MOIRAIS provides a comprehensive spatial statistics library covering global and
local autocorrelation, geostatistical interpolation, geographically weighted
regression, and point pattern analysis. The implementation achieves **100%
chapter coverage** of Schabenberger & Gotway (2005) across all 9 chapters.

All spatial functions live in ``moirais/fn/`` as individual files and accept
arbitrary column names via keyword parameters.

Global Autocorrelation
----------------------

Measures of spatial dependence across the entire study region.

.. list-table::
   :header-rows: 1
   :widths: 20 25 35 20

   * - Statistic
     - Function
     - Description
     - Reference
   * - Moran's I
     - ``morai``
     - Global spatial autocorrelation index (-1 to +1)
     - Ch. 7
   * - Geary's C
     - ``geary``
     - Dissimilarity-based autocorrelation (0 to 2)
     - Ch. 7
   * - Getis-Ord General G
     - ``getgo``
     - Hot/cold spot clustering for positive attributes
     - Ch. 7
   * - Join count
     - ``jncnt``
     - Binary spatial autocorrelation on categorical data
     - Ch. 7

.. code-block:: python

   from moirais.fn import morai, geary
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

.. list-table::
   :header-rows: 1
   :widths: 20 25 35

   * - Statistic
     - Function
     - Description
   * - Local Moran's I
     - ``lisa``
     - Per-location cluster/outlier classification (HH, HL, LH, LL)
   * - Getis-Ord Gi*
     - ``getgi``
     - Local hot/cold spot z-scores
   * - Local Geary
     - ``lgeary``
     - Local dissimilarity decomposition

Each function returns per-observation statistics, pseudo p-values from
conditional permutation tests (default 999 permutations), and cluster
classification labels.

Geostatistical Interpolation
-----------------------------

Prediction of values at unsampled locations using spatial covariance structure.

**Variogram modeling:**

.. list-table::
   :header-rows: 1
   :widths: 20 25 35

   * - Function
     - Model
     - Parameters
   * - ``svari``
     - Semivariogram estimation
     - ``lag_dist``, ``n_lags``, ``model`` (spherical/exponential/gaussian)
   * - ``vfit``
     - Variogram fitting
     - Nugget, sill, range via weighted least squares

**Interpolation methods:**

.. list-table::
   :header-rows: 1
   :widths: 20 25 35

   * - Function
     - Method
     - Notes
   * - ``krige``
     - Ordinary kriging
     - BLUP with variogram model, returns predictions + variance
   * - ``ukrig``
     - Universal kriging
     - Kriging with external drift / trend surface
   * - ``idw``
     - Inverse distance weighting
     - Power parameter (default p=2), no variance estimate
   * - ``cokriging`` / ``cokrg``
     - Co-kriging
     - Multivariate kriging using cross-variograms

.. code-block:: python

   from moirais.fn import krige, svari

   gamma = svari(x, y, values, n_lags=15, model="spherical")
   predictions = krige(x, y, values, grid_x, grid_y,
                       variogram=gamma)

Geographically Weighted Regression (GWR)
-----------------------------------------

Spatially varying coefficient models that capture local heterogeneity in
regression relationships.

.. list-table::
   :header-rows: 1
   :widths: 20 30 30

   * - Function
     - Description
     - Bandwidth selection
   * - ``gwr``
     - Basic GWR with Gaussian/bisquare kernel
     - AICc, CV, or fixed
   * - ``gwpca``
     - Geographically weighted PCA
     - Adaptive bandwidth
   * - ``stgwr``
     - Spatio-temporal GWR
     - Joint space-time bandwidth

Spatial Weight Matrices
-----------------------

Functions for constructing spatial connectivity structures:

- ``wqueen`` -- Queen contiguity (shared edge or vertex)
- ``wrook`` -- Rook contiguity (shared edge only)
- ``wknn`` -- k-nearest neighbors
- ``wdist`` -- Distance-based (threshold or inverse distance)
- ``wrow`` -- Row-standardize any weight matrix

Point Pattern Analysis
----------------------

- ``ripk`` -- Ripley's K-function and L-function for clustering/dispersion
- ``stkde`` -- Spatio-temporal kernel density estimation
- ``stscan`` -- Space-time scan statistic (Kulldorff)

Density-Based Clustering
------------------------

- ``stdbs`` -- DBSCAN with spatio-temporal distance metric. Used in
  ``moirais.tps_spatial_advanced`` to find connected hot clusters
  across the TPS incident feed and produce yearly small-multiples
  with a four-crime overlay (Assault / Robbery / Auto Theft /
  Break-and-Enter).
- ``hdbsc`` -- HDBSCAN hierarchical variant for density-varying data.

Kulldorff Space-Time Scan
-------------------------

The Kulldorff scan statistic detects significant spatio-temporal
clusters by maximising a likelihood ratio over candidate cylindrical
windows in space-time. Implementation in ``stscan`` returns the most
likely cluster, its log-likelihood ratio, the Monte-Carlo p-value, and
the included neighbourhoods + time window. The Hohl-style
"panel d" visualisation is rendered by
``moirais.tps_render.render_satscan_panel`` -- it draws the
significant cluster polygons over a base choropleth.

Reference: Kulldorff, M. (1997). *A spatial scan statistic.*
Communications in Statistics 26(6):1481--1496.

Visualization Helpers
---------------------

The ``moirais.tps_render`` and ``moirais.tps_spatial_advanced``
modules expose publication-style plot helpers commonly paired with
the spatial estimators above:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Plot type
     - Function
   * - Choropleth (per 100k, 9 categories)
     - ``moirais.tps_render.render_choropleth``
   * - District proportional-symbol map (6 former-borough divisions)
     - ``moirais.tps_render.render_district_proportional``
   * - 4-panel quad (density / rate / LISA / Gi*)
     - ``moirais.tps_render.render_quad``
   * - Yearly small-multiples (4-crime overlay + DBSCAN clusters)
     - ``moirais.tps_render.render_yearly_grid``
   * - DBSCAN cluster overlay
     - ``moirais.tps_render.render_dbscan``
   * - Bivariate Moran scatter
     - ``moirais.tps_spatial_advanced.bivariate_moran``
   * - Moran sweep heatmap (categories x years)
     - ``moirais.tps_spatial_advanced.moran_sweep_heatmap``
   * - Kulldorff panel d (Hohl-style)
     - ``moirais.tps_render.render_satscan_panel``

Style follows Hohl, A. *Geographic visualization of disease cluster
detection results.*

Spatio-Temporal Extensions
--------------------------

25 spatio-temporal functions cover panel data, trajectory analysis, and
dynamic spatial processes:

``stacf``, ``stscan``, ``stkde``, ``stvar``, ``stgwr``, ``trajd``,
``stdbs``, ``ripk``, ``gwpca``, ``stmrn``, and 15 additional functions
for spatio-temporal autocorrelation, clustering, and prediction.

References
----------

.. [Schabenberger2005] Schabenberger, O. & Gotway, C.A. (2005).
   *Statistical Methods for Spatial Data Analysis*. Chapman & Hall/CRC.
   9 chapters, 100% coverage in MOIRAIS fn/ files.

.. [Armstrong2000] Armstrong, M.P. et al. (2000).
   Spatial voting analysis. 28/28 backends implemented.

.. [Anselin1995] Anselin, L. (1995). Local Indicators of Spatial
   Association -- LISA. *Geographical Analysis*, 27(2), 93-115.
