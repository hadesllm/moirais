Causal Inference
================

MOIRAIS implements a layered causal-inference workflow for studying the
effects of cannabis use on alcohol-related outcomes in CPADS.
For the full estimand taxonomy see :doc:`estimands`.

Potential outcomes framework
-----------------------------

Let :math:`Y_i(1)` and :math:`Y_i(0)` be the potential outcomes for unit
:math:`i` under treatment and control respectively. The **Average Treatment
Effect** is

.. math::

   \text{ATE} = \mathbb{E}[Y_i(1) - Y_i(0)]

Under the standard identification assumptions:

- **SUTVA** — no interference between units and a single version of treatment.
- **Positivity** — :math:`0 < P(T=1 \mid X) < 1` for all covariate values.
- **Unconfoundedness** — :math:`Y(0), Y(1) \perp T \mid X`.

Inverse Probability Weighting (IPW) — Hájek estimator
-------------------------------------------------------

Given propensity scores :math:`\hat{e}(X_i) = P(T_i = 1 \mid X_i)`,
MOIRAIS uses **stabilised (Hájek) weights**:

.. math::

   w_i = \frac{T_i}{\hat{e}(X_i)} + \frac{1-T_i}{1-\hat{e}(X_i)}

The Hájek ATE estimator is

.. math::

   \hat{\tau}_{\text{Hájek}} =
   \frac{\sum_{T_i=1} w_i Y_i}{\sum_{T_i=1} w_i}
   - \frac{\sum_{T_i=0} w_i Y_i}{\sum_{T_i=0} w_i}

The effective sample size (ESS) is reported as a weight-quality diagnostic:

.. math::

   \text{ESS} = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2}

**Python entry points**: :func:`moirais.causal.run_propensity_ipw_analysis`,
:func:`moirais.causal.estimate_ate`

Average Treatment Effect on the Treated (ATT)
----------------------------------------------

The ATT weights control units by their odds of treatment to match the
treated covariate distribution:

.. math::

   \widehat{\text{ATT}} =
   \bar{Y}_1 - \frac{\sum_{T_i=0} w_i Y_i}{\sum_{T_i=0} w_i},
   \quad w_i = \frac{\hat{e}(X_i)}{1-\hat{e}(X_i)}

**Python entry point**: :func:`moirais.causal.estimate_att`

Average Treatment Effect on the Controls (ATC)
------------------------------------------------

The ATC reweights treated units to match the control covariate distribution:

.. math::

   \widehat{\text{ATC}} =
   \frac{\sum_{T_i=1} w_i Y_i}{\sum_{T_i=1} w_i} - \bar{Y}_0,
   \quad w_i = \frac{1-\hat{e}(X_i)}{\hat{e}(X_i)}

**Python entry point**: :func:`moirais.causal.estimate_atc`

Augmented IPW (AIPW) — Doubly Robust
--------------------------------------

The AIPW estimator adds an outcome-model correction to the IPW influence
function. It is consistent if **either** the propensity model **or** the
outcome models are correctly specified.

The per-unit influence score is

.. math::

   \psi_i =
       \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i)
       + \frac{T_i\bigl(Y_i - \hat{\mu}_1(X_i)\bigr)}{\hat{e}_i}
       - \frac{(1-T_i)\bigl(Y_i - \hat{\mu}_0(X_i)\bigr)}{1 - \hat{e}_i}

The ATE estimate and its standard error are

.. math::

   \widehat{\text{ATE}}_{\text{AIPW}} = \frac{1}{n}\sum_i \psi_i,
   \qquad
   \hat{\sigma} = \frac{\text{sd}(\psi_i)}{\sqrt{n}}

**Python entry point**: :func:`moirais.causal.estimate_aipw`

G-computation (Outcome Regression)
------------------------------------

G-computation directly standardises the outcome by integrating over the
covariate distribution:

.. math::

   \widehat{\text{ATE}}_{\text{G}} =
   \frac{1}{n}\sum_{i=1}^{n}
   \bigl[\hat{\mu}(1, X_i) - \hat{\mu}(0, X_i)\bigr]

where :math:`\hat{\mu}(t, X)` is the predicted outcome from a regression
model fit on the full sample.  Unlike IPW, G-computation is singly robust
(requires correct outcome model specification).

**Python entry point**: :func:`moirais.effects.estimate_ate_gcomputation`

eBAC-selection-adjusted IPW
----------------------------

The ``ebac-selection-adjustment-ipw`` module extends the IPW framework to
account for selection on eBAC (estimated Blood Alcohol Concentration) strata.
Weights are constructed within eBAC-defined subpopulations and then combined.

**Python entry point**: :func:`moirais.causal.run_ebac_selection_ipw_analysis`

Sensitivity Analysis
---------------------

E-value (VanderWeele & Ding 2017)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The E-value is the minimum strength of unmeasured confounding on the
risk-ratio scale needed to fully explain away the observed effect:

.. math::

   E = RR + \sqrt{RR \cdot (RR - 1)}

For the 95% CI lower bound, apply the same formula to the CI endpoint.
An observed RR of 3.9 yields :math:`E \approx 7.3`.

**Python entry point**: :func:`moirais.effects.e_value`

Rosenbaum Bounds
~~~~~~~~~~~~~~~~~

For :math:`\Gamma \geq 1`, Rosenbaum's sensitivity analysis asks whether
the p-value remains below :math:`\alpha` when treatment assignment odds
differ by at most :math:`\Gamma` between matched units. Increasing
:math:`\Gamma` until the bound exceeds :math:`\alpha` measures robustness
to unmeasured confounding.

**Python entry point**: :func:`moirais.effects.sensitivity_rosenbaum`

Average Treatment Effect on the Treated (ATT)
----------------------------------------------

The ATT targets the effect of treatment among those who actually received it:

.. math::

   \text{ATT} = \mathbb{E}[Y(1) - Y(0) \mid T = 1]

Under unconfoundedness, the Hajek IPW estimator for the ATT assigns weight 1
to treated units and weight :math:`\hat{e}(X)/(1 - \hat{e}(X))` to controls:

.. math::

   \widehat{\text{ATT}} =
   \frac{1}{n_1}\sum_{i:T_i=1} Y_i
   - \frac{\sum_{i:T_i=0} Y_i \cdot \hat{e}_i/(1-\hat{e}_i)}
         {\sum_{i:T_i=0} \hat{e}_i/(1-\hat{e}_i)}

**Python entry point**: :func:`moirais.causal.estimate_att`

Average Treatment Effect on the Controls (ATC)
------------------------------------------------

The ATC targets the effect among controls --- what would happen if untreated
units had been treated:

.. math::

   \text{ATC} = \mathbb{E}[Y(1) - Y(0) \mid T = 0]

Treated units are re-weighted by :math:`(1-\hat{e}(X))/\hat{e}(X)` and
controls retain weight 1.

**Python entry point**: :func:`moirais.causal.estimate_atc`

Group Average Treatment Effect (GATE)
--------------------------------------

The GATE partitions units by a categorical variable :math:`G` and estimates
the ATE within each group:

.. math::

   \text{GATE}_g = \mathbb{E}[Y(1) - Y(0) \mid G = g]

MOIRAIS estimates GATEs using the AIPW doubly-robust estimator applied within
each stratum defined by *group_col*.  This provides effect heterogeneity
across pre-defined subpopulations (e.g., age groups, provinces).

**Python entry point**: :func:`moirais.causal.estimate_gate`

Conditional Average Treatment Effect (CATE)
--------------------------------------------

The CATE provides a per-unit treatment effect estimate:

.. math::

   \tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]

MOIRAIS implements two metalearner strategies:

- **T-learner**: fit separate outcome models :math:`\hat{\mu}_1(x)` and
  :math:`\hat{\mu}_0(x)` on treated and control units respectively, then
  :math:`\hat{\tau}(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)`.

- **S-learner**: fit a single outcome model with treatment as a feature,
  then compute :math:`\hat{\tau}(x) = \hat{\mu}(x, 1) - \hat{\mu}(x, 0)`.

Both use Random Forest nuisance learners by default.

**Python entry point**: :func:`moirais.causal.estimate_cate`

Local Average Treatment Effect (LATE / IV)
-------------------------------------------

When treatment is endogenous and an instrument :math:`Z` is available, the
LATE identifies the effect among *compliers* (units whose treatment status
changes in response to the instrument):

.. math::

   \text{LATE} = \frac{\text{Cov}(Y, Z)}{\text{Cov}(T, Z)}
   = \frac{\bar{Y}_{Z=1} - \bar{Y}_{Z=0}}{\bar{T}_{Z=1} - \bar{T}_{Z=0}}

This is the **Wald estimator** for binary instruments.  With covariates, MOIRAIS
uses two-stage least squares (2SLS) via ``linearmodels`` or ``statsmodels``.

The first-stage F-statistic is reported as a weak-instrument diagnostic.
The conventional threshold is :math:`F > 10` (Staiger & Stock, 1997).

**Python entry point**: :func:`moirais.causal.estimate_late`

Interactive Regression Model (IRM)
-----------------------------------

The IRM extends the partially linear model by allowing treatment effect
heterogeneity in the outcome regression:

.. math::

   Y = g_0(T, X) + U, \quad \mathbb{E}[U \mid X, T] = 0

   T = m_0(X) + V, \quad \mathbb{E}[V \mid X] = 0

The Neyman-orthogonal score for the ATE under the IRM is:

.. math::

   \psi_i = g_0(1, X_i) - g_0(0, X_i)
   + \frac{T_i(Y_i - g_0(1,X_i))}{m_0(X_i)}
   - \frac{(1-T_i)(Y_i - g_0(0,X_i))}{1 - m_0(X_i)} - \theta

MOIRAIS uses :class:`doubleml.DoubleMLIRM` with Random Forest nuisance learners
and cross-fitting for honest inference.

**Python entry point**: :func:`moirais.causal.estimate_irm`

References
----------

- Hernán MA, Robins JM (2020). *Causal Inference: What If*.
  Chapman & Hall/CRC.
- Lunceford JK, Davidian M (2004). Stratification and weighting via the
  propensity score in estimation of causal treatment effects.
  *Statistics in Medicine*, 23(19):2937–2960.
  https://doi.org/10.1002/sim.1903
- Robins JM, Rotnitzky A, Zhao LP (1994). Estimation of regression
  coefficients when some regressors are not always observed.
  *JASA*, 89(427):846–866.
- VanderWeele TJ, Ding P (2017). Sensitivity analysis in observational
  research: introducing the E-value.
  *Annals of Internal Medicine*, 167(4):268–274.
  https://doi.org/10.7326/M16-2607
- Rosenbaum PR (2002). *Observational Studies* (2nd ed.). Springer.
- Imbens GW, Angrist JD (1994). Identification and estimation of local
  average treatment effects. *Econometrica*, 62(2):467--475.
- Imbens GW (2004). Nonparametric estimation of average treatment effects
  under exogeneity: a review. *Review of Economics and Statistics*,
  86(1):4--29.
- Kunzel SR, Sekhon JS, Bickel PJ, Yu B (2019). Metalearners for estimating
  heterogeneous treatment effects using machine learning. *PNAS*,
  116(10):4156--4165.
- Chernozhukov V et al. (2018). Double/debiased machine learning for
  treatment and structural parameters. *The Econometrics Journal*,
  21(1):C1--C68.
- Staiger D, Stock JH (1997). Instrumental variables regression with weak
  instruments. *Econometrica*, 65(3):557--586.
