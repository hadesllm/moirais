Survey-Weighted Statistics
===========================

Part of :doc:`index` — MOIRAIS's statistical-methods reference.

CPADS is a complex survey with design weights (``wtpumf``). All prevalence,
mean, and proportion estimates must account for survey design to produce
nationally-representative results.

Design weights
--------------

Survey weights :math:`w_i` correct for unequal probability of selection.
The Horvitz–Thompson estimator for a population total is

.. math::

   \hat{T}_y = \sum_{i \in s} \frac{y_i}{\pi_i} = \sum_{i \in s} w_i y_i

where :math:`\pi_i` is the inclusion probability for unit :math:`i`.

Weighted mean and proportion
-----------------------------

.. math::

   \bar{y}_w = \frac{\sum_i w_i y_i}{\sum_i w_i}

For a binary outcome (prevalence):

.. math::

   \hat{p}_w = \frac{\sum_i w_i \cdot \mathbb{1}[y_i = 1]}{\sum_i w_i}

Linearization variance
----------------------

MOIRAIS uses the Taylor linearization (delta method) approach for variance
estimation via the R ``survey`` package.

R usage
-------

.. code-block:: r

   library(survey)
   svy <- svydesign(ids = ~1, weights = ~wtpumf, data = df)

   # Weighted prevalence
   svymean(~heavy_drinking_30d, svy, na.rm = TRUE)

   # Weighted logistic regression
   svyglm(heavy_drinking_30d ~ cannabis_any_use + age_group + gender,
          design = svy, family = quasibinomial())

Python usage
------------

Survey-weighted summaries are computed directly using the ``weight`` column:

.. code-block:: python

   weighted_prev = (df["heavy_drinking_30d"] * df["weight"]).sum() / df["weight"].sum()

References
----------

- Lumley T (2004). Analysis of complex survey samples.
  *Journal of Statistical Software*, 9(1):1–19.
  https://doi.org/10.18637/jss.v009.i08
- Statistics Canada (2023). *CPADS 2021-2022 PUMF User Guide*.
