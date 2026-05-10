Survey Sampling
================

Part of :doc:`index` — MOIRAIS's statistical-methods reference.

MOIRAIS provides a complete probabilistic sampling toolkit for epidemiological
surveys. All methods are implemented in :mod:`moirais.sampling`.

Simple Random Sampling
-----------------------

Without replacement (SRS WOR), every unit has the same inclusion probability
:math:`\pi_i = n / N`. The Horvitz-Thompson estimator of the mean is the
unweighted sample mean :math:`\bar{y}`.

With replacement (SRS WR), finite-population correction
:math:`\text{fpc} = 1 - n/N` applies to the variance estimate.

**Python**: :func:`moirais.sampling.simple_random_sample`

Stratified Random Sampling
----------------------------

Partition the population into :math:`H` strata. Within stratum :math:`h`,
draw :math:`n_h` units by SRS:

.. math::

   \bar{y}_{\text{str}} = \sum_{h=1}^{H} W_h \bar{y}_h,
   \quad W_h = N_h / N

**Proportional allocation**: :math:`n_h \propto N_h` — minimises total
variance for equal within-stratum variances.

**Optimal (Neyman) allocation**: :math:`n_h \propto N_h S_h` — minimises
variance given a fixed :math:`n`, where :math:`S_h` is the stratum standard
deviation.

**Python**: :func:`moirais.sampling.stratified_sample`

Cluster Sampling
-----------------

When a frame of *individuals* is unavailable, select :math:`m` clusters
(e.g. households, classrooms, census tracts) by SRS, then enumerate all
or a random sub-sample of elements within selected clusters:

.. math::

   \hat{\tau}_{\text{cluster}} = \frac{N}{m} \sum_{i \in \text{selected}} y_i

Cluster sampling introduces intra-cluster correlation (ICC), which inflates
variance relative to SRS.  The design effect (DEFF) measures this inflation:

.. math::

   \text{DEFF} = 1 + (\bar{m} - 1) \cdot \rho_{\text{ICC}}

where :math:`\bar{m}` is the mean cluster size and :math:`\rho_{\text{ICC}}`
is the intra-class correlation.

**Python**: :func:`moirais.sampling.cluster_sample`

Probability Proportional to Size (PPS)
----------------------------------------

PPS sampling selects units with probability proportional to a size
measure :math:`x_i` (e.g. enrolment count):

.. math::

   \pi_i = n \cdot \frac{x_i}{\sum_j x_j}

PPS is more efficient than SRS when the outcome is correlated with size.

**Python**: :func:`moirais.sampling.pps_sample`

Horvitz-Thompson and Hájek Estimators
---------------------------------------

For any probability sample with known inclusion probabilities :math:`\pi_i`:

**Horvitz-Thompson** (unbiased for population total):

.. math::

   \hat{\tau}_{HT} = \sum_{i \in s} \frac{y_i}{\pi_i}

**Hájek** (ratio estimator for mean, more stable than HT):

.. math::

   \bar{y}_H = \frac{\sum_{i \in s} y_i / \pi_i}{\sum_{i \in s} 1 / \pi_i}

**Python**: :func:`moirais.sampling.horvitz_thompson_total`,
:func:`moirais.survey.hajek_mean`

Bootstrap and Jackknife Variance Estimation
---------------------------------------------

For complex statistics (medians, quantiles, non-linear estimators) where
analytic variances are unavailable:

**Bootstrap** (Efron 1979):

.. math::

   \widehat{\text{Var}}(\hat{\theta}) = \frac{1}{B-1}
   \sum_{b=1}^{B} \bigl(\hat{\theta}^{*(b)} - \bar{\theta}^*\bigr)^2

**Delete-1 Jackknife**:

.. math::

   \widehat{\text{Var}}(\hat{\theta}) =
   \frac{n-1}{n} \sum_{i=1}^{n}
   \bigl(\hat{\theta}_{(-i)} - \bar{\theta}_{(.)} \bigr)^2

**Python**: :func:`moirais.sampling.bootstrap_sample`,
:func:`moirais.sampling.jackknife_estimate`

Effective Sample Size
----------------------

Survey weights create unequal effective sample contributions.  The Kish
effective sample size (ESS) quantifies the equivalent SRS size:

.. math::

   \text{ESS} = \frac{\bigl(\sum_i w_i\bigr)^2}{\sum_i w_i^2}

The design effect :math:`\text{DEFF} = n / \text{ESS}` measures variance
inflation relative to a simple random sample of the same size.

**Python**: :func:`moirais.sampling.effective_sample_size`,
:func:`moirais.sampling.design_effect`

Calibration / Raking
---------------------

Post-stratification and raking calibrate sample weights so that
**weighted marginal distributions match known population totals**:

.. math::

   \min_w \sum_i d\bigl(w_i, w_i^{(0)}\bigr)
   \quad \text{subject to} \quad
   \sum_i w_i x_{ij} = T_j \; \forall j

where :math:`d(\cdot)` is a distance function (chi-squared → linear
calibration; multiplicative → raking).  MOIRAIS uses iterative proportional
fitting (IPF).

**Python**: :func:`moirais.survey.calibration_weights`

References
----------

- Kish L (1965). *Survey Sampling*. Wiley.
- Cochran WG (1977). *Sampling Techniques* (3rd ed.). Wiley.
- Lumley T (2010). *Complex Surveys: A Guide to Analysis Using R*. Wiley.
- Valliant R, Dever JA, Kreuter F (2013). *Practical Tools for Designing
  and Weighting Survey Samples*. Springer.
