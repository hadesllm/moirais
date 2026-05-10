Statistical Methods
===================

This section documents the mathematical foundations behind MOIRAIS's estimators.
All methods are implemented for the CPADS (Canadian Postsecondary and
Alcohol/Drug Survey) primary analysis context.

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
   ruhela_formulations
   siuiap
   sprott_doob
   doob_trends
   tps
   hawkes
   spatial
   statphysics
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

.. csv-table::
   :header: "Method", "Estimand", "Python function", "Module"
   :widths: 25, 25, 30, 20

   "IPW (Hájek)", "ATE", "``run_propensity_ipw_analysis``", "propensity-scores"
   "IPW (Hájek)", "ATT", "``estimate_att``", "causal-estimators"
   "IPW (Hájek)", "ATC", "``estimate_atc``", "causal-estimators"
   "AIPW (doubly robust)", "ATE", "``estimate_aipw``", "causal-estimators"
   "GATE (AIPW per group)", "GATE", "``estimate_gate``", "causal-estimators"
   "T-learner / S-learner", "CATE (per unit)", "``estimate_cate``", "causal-estimators"
   "2SLS / Wald IV", "LATE", "``estimate_late``", "causal-estimators"
   "DML–PLR", "ATE", "``estimate_ate``", "treatment-effects"
   "DML–IRM", "ATE (heterogeneous)", "``estimate_irm``", "causal-estimators"
   "DML–PLIV", "LATE", "``estimate_pliv``", "treatment-effects"
   "G-computation", "ATE", "``estimate_ate_gcomputation``", "treatment-effects"
   "Weighted logistic", "OR", "``run_weighted_logistic_analysis``", "logistic-models"
   "Nested model LRT", "model fit", "``compare_nested_logistic_models``", "model-comparison"
   "eBAC-IPW", "selection-adjusted ATE", "``run_ebac_selection_ipw_analysis``", "ebac-selection-adjustment-ipw"
   "Survey-weighted CI", "prevalence CI", "``survey.py`` helpers", "descriptive-statistics"
   "HT estimator", "population total", "``horvitz_thompson_total``", "descriptive-statistics"
   "Hájek estimator", "population mean", "``hajek_mean``", "descriptive-statistics"
   "Beta-binomial Bayes", "posterior mean / CI", "—", "bayesian-inference"
   "Power analysis", "N required", "``run_power_design_module``", "power-design"
   "E-value", "sensitivity bound", "``e_value``", "causal-estimators"
   "Rosenbaum bounds", "sensitivity", "``sensitivity_rosenbaum``", "causal-estimators"
   "Cronbach's alpha", "internal consistency", "``crba``", "psymet"
   "McDonald's omega", "reliability", "``mcdo``", "psymet"
   "KMO", "sampling adequacy", "``kmo``", "psymet"
   "Bartlett's sphericity", "factorability", "``bart``", "psymet"
   "Parallel analysis", "factor retention", "``paran``", "psymet"
   "Composite reliability", "CR", "``crel``", "psymet"
   "AVE", "convergent validity", "``ave``", "psymet"
   "Regional placement", "counts/proportions", "``rplace``", "otis"
   "Alert-state complexity", "combo encoding", "``astcmb``", "otis"
   "Regional volatility", "movement metric", "``volat``", "otis"
   "DML IRM (correctional)", "ATE/ATT", "``otdml``", "otis"
