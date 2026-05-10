Double Machine Learning (DML)
==============================

Part of :doc:`index` — MOIRAIS's statistical-methods reference.

MOIRAIS implements the Partially Linear Regression (PLR) model from
Chernozhukov et al. (2018) via the :pypi:`DoubleML` package.

Partially Linear Regression
----------------------------

The PLR model posits

.. math::

   Y = \theta_0 D + g_0(X) + \varepsilon, \quad \mathbb{E}[\varepsilon \mid D, X] = 0

.. math::

   D = m_0(X) + \nu, \quad \mathbb{E}[\nu \mid X] = 0

where :math:`D` is the treatment, :math:`X` are confounders, and
:math:`g_0, m_0` are unknown nuisance functions estimated nonparametrically.
The parameter of interest is :math:`\theta_0` (ATE under PLR).

Cross-fitting
-------------

To avoid regularization bias from using the same sample for nuisance and
target parameter estimation, DML uses **K-fold cross-fitting**:

1. Split data into :math:`K` folds.
2. For each fold :math:`k`: fit nuisance models on the complement
   :math:`\mathcal{I}^c_k`, predict on :math:`\mathcal{I}_k`.
3. Form residuals: :math:`\tilde{Y}_i = Y_i - \hat{g}_0(X_i)`,
   :math:`\tilde{D}_i = D_i - \hat{m}_0(X_i)`.
4. Regress :math:`\tilde{Y}` on :math:`\tilde{D}` to obtain
   :math:`\hat{\theta}_0`.

This satisfies Neyman orthogonality and achieves
:math:`\sqrt{n}`-consistency under mild conditions on the nuisance estimators.

Neyman orthogonality
---------------------

The score function :math:`\psi(W; \theta, \eta)` satisfies

.. math::

   \partial_\eta \mathbb{E}[\psi(W; \theta_0, \eta_0)][\eta - \eta_0] = 0

ensuring that first-order errors in :math:`\hat{\eta}` do not bias
:math:`\hat{\theta}`.

MOIRAIS implementation
--------------------

**Python entry point**: :func:`moirais.effects.estimate_ate`

Default nuisance learners:

- Outcome nuisance :math:`g_0`: :class:`sklearn.ensemble.RandomForestRegressor`
- Propensity nuisance :math:`m_0`: :class:`sklearn.ensemble.RandomForestClassifier`

Default: ``n_folds=5``, ``n_rep=1``.

.. code-block:: python

   from moirais import estimate_ate

   result = estimate_ate(
       df,
       treatment="cannabis_any_use",
       outcome="heavy_drinking_30d",
       covariates=["age_group", "gender", "province_region", "mental_health"],
   )
   print(result)  # {"ate": ..., "se": ..., "ci_lower": ..., "ci_upper": ...}

Interactive Regression Model (IRM)
------------------------------------

The IRM extends PLR to allow **heterogeneous treatment effects**.  The
model posits:

.. math::

   Y = g_0(D, X) + \varepsilon, \quad \mathbb{E}[\varepsilon \mid D, X] = 0

.. math::

   D = m_0(X) + \nu, \quad \mathbb{E}[\nu \mid X] = 0

Unlike PLR, the outcome function :math:`g_0(D, X)` interacts treatment
:math:`D` with covariates :math:`X`, making the model suitable when CATE
varies across individuals.  The target estimand is the ATE:

.. math::

   \theta_0^{\text{IRM}} = \mathbb{E}[g_0(1, X) - g_0(0, X)]

with the doubly-robust score:

.. math::

   \psi^{\text{IRM}}_i =
     \hat{g}_0(1, X_i) - \hat{g}_0(0, X_i)
     + \frac{D_i \bigl(Y_i - \hat{g}_0(1, X_i)\bigr)}{\hat{m}_0(X_i)}
     - \frac{(1-D_i)\bigl(Y_i - \hat{g}_0(0, X_i)\bigr)}{1 - \hat{m}_0(X_i)}

**Python entry point**: :func:`moirais.causal.estimate_irm`

Partially Linear IV (PLIV) — LATE estimation
----------------------------------------------

When treatment :math:`D` is endogenous, a valid instrument :math:`Z`
(correlated with :math:`D` but independent of :math:`\varepsilon` given
:math:`X`) identifies the **Local Average Treatment Effect (LATE)**:

.. math::

   \text{LATE} = \frac{\text{Cov}(Y, Z \mid X)}{\text{Cov}(D, Z \mid X)}

The Partially Linear IV model is:

.. math::

   Y = \theta_0 D + g_0(X) + \varepsilon

.. math::

   D = m_0(Z, X) + \nu

Cross-fitting proceeds as in PLR, with the additional first stage
estimating :math:`\hat{m}_0(Z, X)`.

**Python entry point**: :func:`moirais.effects.estimate_pliv`

Nuisance learner defaults
--------------------------

- **PLR**, :math:`g_0` (outcome): :class:`sklearn.ensemble.RandomForestRegressor` (100 trees, max_depth=5).
- **PLR**, :math:`m_0` (propensity): :class:`sklearn.ensemble.RandomForestClassifier` (100 trees, max_depth=5).
- **IRM**, :math:`g_0(d, X)` (outcome × treatment): :class:`sklearn.ensemble.RandomForestRegressor`.
- **IRM**, :math:`m_0(X)` (propensity): :class:`sklearn.ensemble.RandomForestClassifier`.
- **PLIV**, :math:`g_0(X)` (outcome residual): :class:`sklearn.ensemble.RandomForestRegressor`.
- **PLIV**, :math:`m_0(Z, X)` (first stage): :class:`sklearn.ensemble.RandomForestRegressor`.

References
----------

- Chernozhukov V, Chetverikov D, Demirer M, Duflo E, Hansen C, Newey W,
  Robins J (2018). Double/debiased machine learning for treatment and
  structural parameters.
  *The Econometrics Journal*, 21(1):C1–C68.
  https://doi.org/10.1111/ectj.12097
- Bach P, Chernozhukov V, Kurz MS, Spindler M (2022). DoubleML — An
  object-oriented implementation of double machine learning in Python.
  *JMLR*, 23(53):1–6.
- Imbens GW, Angrist JD (1994). Identification and estimation of local
  average treatment effects.
  *Econometrica*, 62(2):467–475.
  https://doi.org/10.2307/2951620
