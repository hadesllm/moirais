Propensity Scores
=================

Part of :doc:`index` — MOIRAIS's statistical-methods reference.

The propensity score :math:`e(X) = P(T=1 \mid X)` summarizes confounding
information into a single scalar, allowing balancing without direct covariate
matching (Rosenbaum & Rubin 1983).

Estimation
----------

MOIRAIS estimates propensity scores via logistic regression (default) or random
forest, depending on the module configuration.

**Logistic regression**:

.. math::

   \log \frac{e(X_i)}{1 - e(X_i)} = \beta_0 + \beta^\top X_i

Implemented in :func:`moirais.causal.compute_propensity_scores` using
:class:`sklearn.linear_model.LogisticRegression` with ``max_iter=1000``.

Diagnostics
-----------

After propensity estimation:

1. **Overlap check** — histogram of :math:`\hat{e}(X)` by treatment group.
   Extreme values near 0 or 1 indicate potential positivity violations.
2. **Effective Sample Size (ESS)** — see :doc:`causal`.
3. **Covariate balance** — standardized mean differences before and after
   weighting should be :math:`< 0.1` for all covariates.

CPADS covariates
----------------

The default covariate set for the ``propensity-scores`` module is drawn from
``CPADS_REQUIRED_VARIABLES``:

- ``age_group``
- ``gender``
- ``province_region``
- ``mental_health``
- ``physical_health``
- ``alcohol_past12m``

Treatment: ``cannabis_any_use``
Outcome: ``heavy_drinking_30d`` or ``ebac_tot``

References
----------

- Rosenbaum PR, Rubin DB (1983). The central role of the propensity score in
  observational studies for causal effects.
  *Biometrika*, 70(1):41–55.
  https://doi.org/10.1093/biomet/70.1.41
- Austin PC (2011). An introduction to propensity score methods for reducing
  the effects of confounding in observational studies.
  *Multivariate Behavioral Research*, 46(3):399–424.
  https://doi.org/10.1080/00273171.2011.568786
