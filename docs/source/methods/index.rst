Statistical Methods
===================

This section documents the mathematical foundations behind MOIRAIS's
estimators. The methods are dataset-agnostic — they apply to any
suitably-shaped tabular input, including the OTIS placement records,
TPS incident feeds, CPADS survey data, and any other dataset that
matches the estimator's signature.

.. toctree::
   :maxdepth: 2

   estimands
   causal
   propensity
   dml
   ebac
   survey
   sampling
   dataset
   psymet
   otis
   otis_linkage
   mrm_modules
   siuiap
   sprott_doob
   doob_trends
   tps
   hawkes
   spatial
   statphysics
   findings
   quantization
   inference_engine
   signal_processing
   homomorphic_deconvolution
   crypto
   genomics
   polyglot
   deployment

Quick reference
---------------

Each entry below names the estimator, the estimand it targets, and
the Python function that produces it.

Causal estimators
~~~~~~~~~~~~~~~~~

- ``run_propensity_ipw_analysis`` — IPW (Hájek), ATE
- ``estimate_att`` — IPW (Hájek), ATT
- ``estimate_atc`` — IPW (Hájek), ATC
- ``estimate_aipw`` — AIPW (doubly robust), ATE
- ``estimate_gate`` — GATE (AIPW per group)
- ``estimate_cate`` — T-learner / S-learner, CATE (per unit)
- ``estimate_late`` — 2SLS / Wald IV, LATE
- ``estimate_ate`` — DML--PLR, ATE
- ``estimate_irm`` — DML--IRM, heterogeneous ATE
- ``estimate_pliv`` — DML--PLIV, LATE
- ``estimate_ate_gcomputation`` — G-computation, ATE
- ``run_ebac_selection_ipw_analysis`` — eBAC-IPW, selection-adjusted ATE
- ``e_value`` — E-value sensitivity bound
- ``sensitivity_rosenbaum`` — Rosenbaum bounds

Logistic / model comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``run_weighted_logistic_analysis`` — weighted logistic, OR
- ``compare_nested_logistic_models`` — nested-model LRT

Survey + descriptive statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``moirais.survey`` helpers — survey-weighted CIs and prevalence
- ``horvitz_thompson_total`` — HT estimator, population total
- ``hajek_mean`` — Hájek estimator, population mean

Power + Bayes
~~~~~~~~~~~~~

- ``run_power_design_module`` — N required for a given design
- Beta-binomial Bayes — posterior mean / CI (see ``moirais.causal``)

Psychometrics
~~~~~~~~~~~~~

- ``crba`` — Cronbach's α (internal consistency)
- ``mcdo`` — McDonald's ω (reliability)
- ``kmo`` — KMO sampling adequacy
- ``bart`` — Bartlett's sphericity (factorability)
- ``paran`` — Parallel analysis (factor retention)
- ``crel`` — Composite reliability
- ``ave`` — Average Variance Extracted (convergent validity)

OTIS-specific
~~~~~~~~~~~~~

- ``rplace`` — Regional placement counts / proportions
- ``astcmb`` — Alert-state combination encoding
- ``volat`` — Regional volatility (movement metric)
- ``otdml`` — DML IRM on correctional data, ATE / ATT
