Causal Estimands
=================

Part of :doc:`index` — MORIE's statistical-methods reference.

MORIE provides a full suite of causal estimands. This page defines each
estimand and maps it to the corresponding Python function.

Summary
-------

- **ATE** — Average Treatment Effect (all units): :func:`morie.causal.run_propensity_ipw_analysis`.
- **ATT / ATTE** — Average Treatment Effect on the Treated (treated units only): :func:`morie.causal.estimate_att`.
- **ATC** — Average Treatment Effect on the Controls (control units only): :func:`morie.causal.estimate_atc`.
- **GATE** — Group Average Treatment Effect (subgroups, e.g. gender): :func:`morie.causal.estimate_gate`.
- **CATE** — Conditional Average Treatment Effect (each individual unit): :func:`morie.causal.estimate_cate`.
- **LATE** — Local Average Treatment Effect (compliers, IV context): :func:`morie.causal.estimate_late`.
- **PLR-ATE** — DML--PLR ATE (all units): :func:`morie.effects.estimate_ate`.
- **IRM-ATE** — DML--IRM ATE, heterogeneous (all units): :func:`morie.causal.estimate_irm`.
- **DR-ATE** — AIPW doubly robust ATE (all units): :func:`morie.causal.estimate_aipw`.

---

Average Treatment Effect (ATE)
--------------------------------

The ATE averages over the *full* population:

.. math::

   \text{ATE} = \mathbb{E}[Y_i(1) - Y_i(0)]

where :math:`Y_i(1)` and :math:`Y_i(0)` are the potential outcomes under
treatment and control for unit :math:`i`.  Identification requires
**unconfoundedness** (:math:`Y(0), Y(1) \perp T \mid X`) and **overlap**
(:math:`0 < P(T=1 \mid X) < 1`).

---

Average Treatment Effect on the Treated (ATT)
-----------------------------------------------

The ATT conditions on actually treated units:

.. math::

   \text{ATT} = \mathbb{E}[Y_i(1) - Y_i(0) \mid T_i = 1]

Under unconfoundedness, ATT is identified via Hájek-weighted IPW where
treated units receive weight 1 and controls receive weight
:math:`\hat{e}(X_i) / (1 - \hat{e}(X_i))`:

.. math::

   \widehat{\text{ATT}} =
   \frac{\sum_{T_i=1} Y_i}{n_1}
   - \frac{\sum_{T_i=0} w_i Y_i}{\sum_{T_i=0} w_i},
   \quad w_i = \frac{\hat{e}(X_i)}{1 - \hat{e}(X_i)}

The ATT is the relevant estimand when the treated group is the primary
policy target (e.g. cannabis users in CPADS).

**Python entry point**: :func:`morie.causal.estimate_att`

---

Average Treatment Effect on the Controls (ATC)
------------------------------------------------

The ATC conditions on the control population:

.. math::

   \text{ATC} = \mathbb{E}[Y_i(1) - Y_i(0) \mid T_i = 0]

Identification uses reversed weights: control units receive weight 1;
treated units receive weight
:math:`(1 - \hat{e}(X_i)) / \hat{e}(X_i)`.

**Python entry point**: :func:`morie.causal.estimate_atc`

---

Group Average Treatment Effect (GATE)
---------------------------------------

The GATE generalises the ATE to subpopulations defined by a discrete
group variable :math:`G \in \{g_1, \ldots, g_K\}`:

.. math::

   \text{GATE}_k = \mathbb{E}[Y_i(1) - Y_i(0) \mid G_i = g_k]

MORIE computes GATE by applying the AIPW doubly-robust influence function
within each stratum of :math:`G`.  The result is a DataFrame with one row
per group, including AIPW ATE estimate, SE, 95% CI, and sample size.

GATE is useful for examining heterogeneity by gender, age group, province,
or any other subgroup of substantive interest.

**Python entry point**: :func:`morie.causal.estimate_gate`

---

Conditional Average Treatment Effect (CATE)
---------------------------------------------

The CATE produces a *per-unit* treatment effect estimate as a function of
covariates :math:`X`:

.. math::

   \tau(x) = \mathbb{E}[Y_i(1) - Y_i(0) \mid X_i = x]

MORIE implements the **T-learner** (two separate outcome models):

1. Fit :math:`\hat{\mu}_1(X)` on treated units :math:`\{i : T_i = 1\}`.
2. Fit :math:`\hat{\mu}_0(X)` on control units :math:`\{i : T_i = 0\}`.
3. :math:`\widehat{\text{CATE}}_i = \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i)`.

And the **S-learner** (single model with treatment as feature):

1. Fit :math:`\hat{\mu}(T, X)` on the full sample.
2. :math:`\widehat{\text{CATE}}_i = \hat{\mu}(1, X_i) - \hat{\mu}(0, X_i)`.

Both learners use :class:`sklearn.ensemble.RandomForestRegressor` by default.

**Python entry point**: :func:`morie.causal.estimate_cate`

---

Local Average Treatment Effect (LATE)
---------------------------------------

When the treatment assignment :math:`T_i` is endogenous (e.g. influenced
by unobserved factors), a valid binary **instrument** :math:`Z_i` that
satisfies:

- **Relevance**: :math:`\text{Cov}(T_i, Z_i) \neq 0`
- **Exclusion**: :math:`Z_i \perp \varepsilon_i \mid X_i`
- **Monotonicity**: :math:`T_i(1) \geq T_i(0)` for all units

identifies the LATE (also called the **Complier Average Causal Effect**):

.. math::

   \text{LATE} = \frac{\mathbb{E}[Y_i \mid Z_i=1] - \mathbb{E}[Y_i \mid Z_i=0]}
                      {\mathbb{E}[T_i \mid Z_i=1] - \mathbb{E}[T_i \mid Z_i=0]}

This is the ATE for **compliers** — units who take up treatment when
:math:`Z=1` and do not when :math:`Z=0`.

With covariates, MORIE uses **2SLS** (two-stage least squares):

- Stage 1: :math:`\hat{T}_i = \hat{\gamma}_0 + \hat{\gamma}_1 Z_i + \hat{\gamma}_2 X_i`
- Stage 2: :math:`Y_i = \theta \hat{T}_i + \beta X_i + \varepsilon_i`

**Python entry point**: :func:`morie.causal.estimate_late`

---

Doubly-Robust AIPW (ATE)
--------------------------

See :doc:`causal` for the full AIPW derivation and formula.

---

References
----------

- Imbens GW, Rubin DB (2015). *Causal Inference for Statistics, Social,
  and Biomedical Sciences*. Cambridge University Press.
- Hernan MA, Robins JM (2020). *Causal Inference: What If*.
  Chapman & Hall/CRC.
- Kennedy EH (2016). Semiparametric theory and empirical processes in
  causal inference. *Statistical Causal Inferences and Their Applications
  in Public Health Research*, pp. 141–167.
  https://doi.org/10.1007/978-3-319-41259-7_8
- Imbens GW, Angrist JD (1994). Identification and estimation of local
  average treatment effects. *Econometrica*, 62(2):467–475.
- Nie X, Wager S (2021). Quasi-oracle estimation of heterogeneous
  treatment effects. *Biometrika*, 108(2):299–319.
  https://doi.org/10.1093/biomet/asaa076
