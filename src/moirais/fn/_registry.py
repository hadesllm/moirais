"""Star Wars naming registry for moirais.fn functions.

Single source of truth: short name -> metadata.
Used by fn/__init__.py for auto-export and stat_commands.py for CLI dispatch.

'Knowledge itself is power. — Francis Bacon'
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FnEntry:
    """Registry entry for one fn/ function."""

    short: str  # filename / import alias (max 7 chars)
    full: str  # full descriptive function name
    category: str  # domain category
    description: str  # one-line description
    quote: str = ""  # Star Wars quote (optional)


# ── REGISTRY ────────────────────────────────────────────────────────────────
# Add entries here as new fn/ files are created.
# fn/__init__.py reads this to auto-export all short names.

REGISTRY: dict[str, FnEntry] = {}


def _r(short, full, cat, desc, quote=""):
    REGISTRY[short] = FnEntry(short, full, cat, desc, quote)


# ── Distributions (d/p/q/r family) ─────────────────────────────────────────
_r("dnorm", "dnorm", "Distribution", "Normal PDF", "It is during our darkest moments that we must focus to see the light. — Aristotle")
_r("pnorm", "pnorm", "Distribution", "Normal CDF", "It is during our darkest moments that we must focus to see the light. — Aristotle")
_r("qnorm", "qnorm", "Distribution", "Normal quantile", "It is during our darkest moments that we must focus to see the light. — Aristotle")
_r("rnorm", "rnorm", "Distribution", "Normal random", "In my experience there is no luck. --")
_r("dt", "dt_", "Distribution", "Student t PDF", "Much to learn you still have.")
_r("pt", "pt", "Distribution", "Student t CDF", "Patience you must have.")
_r("qt", "qt", "Distribution", "Student t quantile", "Difficult to see. Always in motion.")
_r("rt", "rt", "Distribution", "Student t random", "Luminous beings are we.")
_r("dchsq", "dchisq", "Distribution", "Chi-squared PDF", "Stay on target. -- Gold Five")
_r("pchsq", "pchisq", "Distribution", "Chi-squared CDF", "Almost there. -- Gold Leader")
_r("qchsq", "qchisq", "Distribution", "Chi-squared quantile", "Trust your feelings. --")
_r("rchisq", "rchisq", "Distribution", "Chi-squared random", "Let go of your conscious self. --")
_r("df_", "df_dist", "Distribution", "F-distribution PDF", "There is always a bigger fish. -- Qui-Gon")
_r("pf", "pf", "Distribution", "F-distribution CDF", "Your focus determines your reality. -- Qui-Gon")
_r("qf", "qf", "Distribution", "F-distribution quantile", "The ability to speak does not make you wise.")
_r("rf", "rf_dist", "Distribution", "F-distribution random", "It is during our darkest moments that we must focus to see the light. — Aristotle")
_r("dbnm", "dbinom", "Distribution", "Binomial PMF", "There are two of them. Always two.")
_r("pbnm", "pbinom", "Distribution", "Binomial CDF", "It is during our darkest moments that we must focus to see the light. — Aristotle")
_r("dpoi", "dpois", "Distribution", "Poisson PMF", "They keep coming, one by one. -- Bail Organa")
_r("ppoi", "ppois", "Distribution", "Poisson CDF", "There has been an awakening. -- Snoke")
_r("dunf", "dunif", "Distribution", "Uniform PDF", "It is during our darkest moments that we must focus to see the light. — Aristotle")
_r("punf", "punif", "Distribution", "Uniform CDF", "It is during our darkest moments that we must focus to see the light. — Aristotle")
_r("dexp", "dexp", "Distribution", "Exponential PDF", "It is during our darkest moments that we must focus to see the light. — Aristotle")
_r("pexp", "pexp", "Distribution", "Exponential CDF", "Time runs short. -- Mace Windu")
_r("dgam", "dgamma", "Distribution", "Gamma PDF", "Powerful you have become. -- Dooku")

# ── Statistical Tests ───────────────────────────────────────────────────────
_r("", "chi_squared_test", "Test", "Chi-squared test", "Let the Wookiee win.")
_r("jarjar", "jarque_bera", "Test", "Jarque-Bera normality test", "How wude!")
_r("What is now proved was once only imagined. — William Blake", "anova_oneway", "Test", "One-way ANOVA", "I find your lack of variance disturbing.")
_r("windu", "wald_test", "Test", "Wald test", "This party's over.")
_r("ahsoka", "anova_twoway", "Test", "Two-way ANOVA", "Distribution helper.")
_r("ttest", "t_test", "Test", "Student's t-test", "The truth is often what we make of it. --")
_r("utest", "mann_whitney", "Test", "Mann-Whitney U test", "There can be no compromise. -- Saw Gerrera")
_r("wlcx", "wilcoxon", "Test", "Wilcoxon signed-rank", "The strongest stars have hearts of kyber.")
_r("kw", "kruskal_wallis", "Test", "Kruskal-Wallis H test", "We must choose. -- Mon Mothma")
_r("fried", "friedman", "Test", "Friedman test", "All groups will be heard. -- Padme")
_r("ad", "anderson_darling", "Test", "Anderson-Darling test", "What is now proved was once only imagined. — William Blake")
_r("sw", "shapiro_wilk", "Test", "Shapiro-Wilk normality", "You were supposed to be normal! --")
_r("levene", "levene", "Test", "Levene's variance test", "What is now proved was once only imagined. — William Blake")
_r("bf", "brown_forsythe", "Test", "Brown-Forsythe test", "We shall see how well you fight. -- Grievous")

# ── Regression — "Rey" family ──────────────────────────────────────────────
_r(
    "rey",
    "linear_regression",
    "Regression",
    "OLS linear regression",
    "The belonging you seek is not behind you, it is ahead.",
)
_r(
    "rey_lg",
    "logistic_regression",
    "Regression",
    "Logistic regression (IRLS)",
    "There are always two choices. -- Qui-Gon",
)
_r("rey_ps", "poisson_regression", "Regression", "Poisson regression", "They come in single file. --")
_r("ridge", "ridge_regression", "Regression", "Ridge regression (L2)", "I have the high ground. --")
_r("lasso", "lasso_regression", "Regression", "Lasso regression (L1)", "Concentrate fire on the nearest. -- Ackbar")
_r("elnet", "elastic_net", "Regression", "Elastic net", "Stretching out with your feelings. -- Luke")
_r("hux", "huber_regression", "Regression", "Huber robust regression", "Careful not to choke on your aspirations.")
_r("qreg", "quantile_regression", "Regression", "Quantile regression", "Not all paths lead the same way. -- Ahsoka")
_r("wls", "weighted_ls", "Regression", "Weighted least squares", "Some carry more weight than others. -- Luke")
_r("gls", "generalized_ls", "Regression", "Generalized least squares", "There are alternatives to fighting. -- Padme")

# ── Effect Sizes ────────────────────────────────────────────────────────────
_r("cohen", "cohens_d", "EffectSize", "Cohen's d", "Size matters not.")
_r("hedges", "hedges_g", "EffectSize", "Hedges' g", "Judge me by my size, do you?")
_r("eta", "eta_squared", "EffectSize", "Eta-squared", "Distribution helper.")
_r("omega", "omega_squared", "EffectSize", "Omega-squared", "What is now proved was once only imagined. — William Blake")
_r("cliffs", "cliffs_delta", "EffectSize", "Cliff's delta", "It's over. I have the high ground! --")
_r("phi", "phi_coefficient", "EffectSize", "Phi coefficient", "Distribution helper.")
_r("cram", "cramers_v", "EffectSize", "Cramer's V", "What is now proved was once only imagined. — William Blake")
_r(
    "concordance_index",
    "EffectSize",
    "Concordance index (C)",
    "The odds of successfully navigating an asteroid field are 3,720 to 1.",
)
_r("kappa", "cohens_kappa", "EffectSize", "Cohen's kappa", "I sense great fear in you.")

# ── Causal Inference — "Distribution helper." family ───────────────────────────────────────
_r("force", "ate_diff", "Causal", "ATE difference-in-means", "Distribution helper.")
_r("dooku", "double_ml", "Causal", "Double/debiased ML", "Twice the pride, double the fall.")
_r("What is now proved was once only imagined. — William Blake", "synth_control", "Causal", "Synthetic control method", "What is now proved was once only imagined. — William Blake")
_r("What is now proved was once only imagined. — William Blake", "sensitivity_analysis", "Causal", "Rosenbaum sensitivity", "Do or do not. There is no try.")
_r("rdd", "reg_discontinuity", "Causal", "Regression discontinuity", "This is the edge of the galaxy. -- Lando")
_r("did", "diff_in_diff", "Causal", "Difference-in-differences", "So this is how liberty dies. -- Padme")
_r("iv", "iv_2sls", "Causal", "2SLS instrumental variables", "The instrument of your liberation. -- Saw")

# ── Resampling ──────────────────────────────────────────────────────────────
_r("bb8", "bootstrap_ci", "Resampling", "Bootstrap confidence interval", "Bee-boop!")
_r("jawa", "jackknife", "Resampling", "Jackknife estimator", "Utini!")
_r("kylo", "kfold_cv", "Resampling", "K-fold cross-validation", "Let the past die. Kill it, if you have to.")
_r("perm", "permutation_test", "Resampling", "Permutation test", "The galaxy is full of possibilities. -- Hera")

# ── Data Exploration — "Luke//Finn" family ──────────────────────────────
_r("luke", "summarize", "Exploration", "Per-column summary stats", "I've got a bad feeling about this.")
_r("", "profile", "Exploration", "Dataset profiling", "Hope is like the sun.")
_r("finn", "find_patterns", "Exploration", "Correlation scan", "I need a pilot!")
_r("grogu", "grouped_stats", "Exploration", "Grouped summary statistics", "Small but mighty. -- Din Djarin")
_r("", "handle_missing", "Exploration", "Missing data summary + impute", "Never tell me the odds.")
_r("ackbar", "detect_outliers", "Exploration", "Outlier detection (IQR + Z)", "It's a trap!")

# ── R-squared & Diagnostics ────────────────────────────────────────────────
_r(
    "vif",
    "variance_inflation",
    "Diagnostic",
    "Variance inflation factor",
    "Your overconfidence is your weakness. -- Luke",
)
_r(
    "durbin",
    "durbin_watson",
    "Diagnostic",
    "Durbin-Watson autocorrelation",
    "Distribution helper.",
)

# ── Time Series ─────────────────────────────────────────────────────────────
_r("acf", "autocorrelation", "TimeSeries", "Autocorrelation function", "Always in motion is the future.")
_r("pacf", "partial_acf", "TimeSeries", "Partial ACF", "Hard to see the dark side is.")
_r("adf", "adf_test", "TimeSeries", "Augmented Dickey-Fuller test", "Something is out of balance. -- Luke")
_r("kpss", "kpss_test", "TimeSeries", "KPSS stationarity test", "Steady. Steady. -- Poe Dameron")
_r("ljung", "ljung_box", "TimeSeries", "Ljung-Box test", "There is a disturbance. -- Mace Windu")
_r("grngr", "granger_cause", "TimeSeries", "Granger causality test", "One thing follows another. -- Qui-Gon")
_r("decomp", "seasonal_decompose", "TimeSeries", "Seasonal decomposition", "We must go deeper. -- Qui-Gon")
_r("arma", "arima_fit", "TimeSeries", "ARIMA model fitting", "What has come before shapes what follows.")

# ── Survival ────────────────────────────────────────────────────────────────
_r("km", "kaplan_meier", "Survival", "Kaplan-Meier estimator", "There is another.")
_r("lrank", "log_rank", "Survival", "Log-rank test", "You cannot stop the change. -- Shmi")
_r("cox", "cox_ph", "Survival", "Cox proportional hazards", "The fear of loss is a path to the dark side.")
_r("rmst", "rmst", "Survival", "Restricted mean survival time", "How long must we endure this? -- Padme")
_r("hazard", "hazard_rate", "Survival", "Hazard rate function", "I sense great danger.")

# ── Diagnostic Accuracy ────────────────────────────────────────────────────
_r("sens", "sensitivity_dx", "DiagAccuracy", "Sensitivity / recall", "Your eyes can deceive you. --")
_r("spec", "specificity_dx", "DiagAccuracy", "Specificity", "I can see through the lies. -- Anakin")
_r("ppv", "ppv_npv", "DiagAccuracy", "PPV and NPV", "How certain are you? -- Mace Windu")
_r("roc", "roc_auc", "DiagAccuracy", "ROC curve + AUC", "Let me see your identification. -- Trooper")
_r("youden", "youdens_j", "DiagAccuracy", "Youden's J index", "A fine addition to my collection. -- Grievous")

# ── Sampling / Survey ──────────────────────────────────────────────────────
_r("palps", "parallel_analysis", "Sampling", "Parallel analysis for FA", "Everything is proceeding as I have foreseen.")
_r("strat", "stratified_mean", "Sampling", "Stratified mean estimator", "Divide the fleet. -- Thrawn")
_r("srs", "srs", "Sampling", "Simple random sample", "Do or do not, there is no try.")
_r("strsmp", "strsmp", "Sampling", "Stratified random sample", "Your focus determines your reality.")
_r("clsmp", "clsmp", "Sampling", "Cluster sampling", "We are what they grow beyond.")
_r("ppssmp", "ppssmp", "Sampling", "PPS sampling", "Distribution helper.")
_r("btsmp", "btsmp", "Sampling", "Bootstrap resampling", "Truly wonderful the mind of a child is.")
_r("jksmp", "jksmp", "Sampling", "Jackknife variance estimate", "Distribution helper.")
_r("dsgwt", "dsgwt", "Sampling", "Design weights", "I find your lack of faith disturbing.")
_r("cal_wg", "calibration_weights", "Sampling", "Raking calibration weights", "A great weight has been lifted.")
_r("deff", "deff", "Sampling", "Design effect (DEFF)", "The dark side clouds everything.")
_r("ess", "effective_sample_size", "Sampling", "Effective sample size", "The few who remain are strong.")
_r("ess_s", "ess_s", "Sampling", "Kish effective sample size", "In a dark place we find ourselves.")
_r("ps_wgt", "poststratification_weights", "Sampling", "Post-stratification weights", "Balance I have found. -- Bendu")
_r("pstrat", "pstrat", "Sampling", "Post-stratification weights", "The belonging you seek is not behind you.")
_r("ht_tot", "horvitz_thompson_total", "Sampling", "Horvitz-Thompson total", "Every voice counts. -- Padme")
_r("hajek", "hajek", "Sampling", "Hajek population mean", "In my experience there is no such thing as luck.")
_r("h_mean", "hajek_mean", "Sampling", "Hajek mean estimator", "The truth is often the whole. --")
_r("ratio", "ratio_estimator", "Sampling", "Survey ratio estimator", "The balance is shifting. -- Mace Windu")
_r(
    "sub_est",
    "subpopulation_estimate",
    "Sampling",
    "Subpopulation mean estimator",
    "A certain point of view. --",
)
_r("svyglm", "svyglm", "Sampling", "Complex survey GLM", "Never tell me the odds.")
_r("cglm", "complex_survey_glm", "Sampling", "Complex survey GLM (weighted)", "A complex web this is.")

# ── Psychometrics ──────────────────────────────────────────────────────────
_r("crba", "crba", "Psychometrics", "Cronbach's alpha with Feldt CI", "Distribution helper.")
_r("mcdo", "mcdo", "Psychometrics", "McDonald's omega", "A more elegant weapon. --")
_r(
    "itcor",
    "itcor",
    "Psychometrics",
    "Corrected item-total correlations",
    "Each part connected to the whole. -- Chirrut",
)
_r("adel", "adel", "Psychometrics", "Alpha if item deleted", "Strike me down and I become stronger.")
_r("crel", "crel", "Psychometrics", "Composite reliability", "Trust in the construct you must.")
_r("ave", "ave", "Psychometrics", "Average variance extracted", "Distribution helper.")
_r("kmo", "kmo", "Psychometrics", "Kaiser-Meyer-Olkin adequacy", "Adequate your sample is.")
_r("bart", "bart", "Psychometrics", "Bartlett's sphericity test", "That is no coincidence. --")
_r("paran", "paran", "Psychometrics", "Horn's parallel analysis", "Real from illusion, know you must.")
_r("splhf", "splhf", "Psychometrics", "Spearman-Brown split-half", "Two halves of the same whole. -- Ahsoka")
_r("idisc", "idisc", "Psychometrics", "Item discrimination index", "This one can tell the difference. -- Chirrut")

# ── OTIS Correctional ─────────────────────────────────────────────────────
_r("rpl", "rplace", "OTIS", "Regional placement analysis", "I have the high ground. --")
_r("astc", "astcmb", "OTIS", "Alert-state combination encoding", "I have a bad feeling about this. --")
_r("vol", "volat", "OTIS", "Regional volatility metric", "What is now proved was once only imagined. — William Blake")
_r("rct", "rctrnd", "OTIS", "Restrictive confinement trends", "Begun the Clone War has.")
_r("otd", "otdesc", "OTIS", "OTIS descriptive statistics suite", "Let us see the full picture. -- Mon Mothma")
_r("oml", "otdml", "OTIS", "DML IRM for OTIS data", "Twice the pride, double the fall. -- Dooku")

# ── eBAC ───────────────────────────────────────────────────────────────────
_r("ebac", "calculate_ebac", "eBAC", "Widmark eBAC formula", "Who's the more foolish? --")
_r("legal", "is_over_legal_limit", "eBAC", "BAC legal limit check", "What is now proved was once only imagined. — William Blake")

# ── Causal Inference (extended) ────────────────────────────────────────────
_r("ate", "estimate_ate", "Causal", "IPW-weighted OLS ATE", "Distribution helper.")
_r("att", "estimate_att", "Causal", "ATT via Hajek-weighted IPW", "The chosen ones have been treated.")
_r("atc", "estimate_atc", "Causal", "ATC via Hajek-weighted IPW", "Those untouched by war. -- Bail Organa")
_r("aipw", "estimate_aipw", "Causal", "AIPW doubly-robust estimator", "We need a backup plan. -- Padme")
_r("cate", "estimate_cate", "Causal", "CATE via T/S-learner", "Different paths for different souls. -- Chirrut")
_r("gate", "estimate_gate", "Causal", "GATE via AIPW within strata", "Distribution helper.")
_r("late", "estimate_late", "Causal", "LATE via instrumental variables", "Through another shall the truth come.")
_r("ipw", "calculate_ipw_weights", "Causal", "Inverse probability weights", "Distribution helper.")
_r("dml", "estimate_double_ml", "Causal", "Double Machine Learning (PLR)", "Twice the pride, double the fall. -- Dooku")
_r("plr", "estimate_plr", "Causal", "Partially Linear Regression ATE", "Partial truths are still truths. --")
_r("pliv", "estimate_pliv", "Causal", "Partially Linear IV (LATE)", "An instrument of a greater purpose. -- Saw")
_r("irm", "estimate_irm", "Causal", "Interactive Regression Model (IRM)", "Distribution helper.")
_r("g_comp", "estimate_ate_gcomputation", "Causal", "G-computation ATE", "What is now proved was once only imagined. — William Blake")
_r(
    "rbnd",
    "sensitivity_rosenbaum",
    "Causal",
    "Rosenbaum bounds sensitivity",
    "How far from random is the truth?",
)
_r("eval_", "e_value", "Causal", "E-value for unmeasured confounding", "Distribution helper.")
_r("ps_ana", "run_propensity_ipw_analysis", "Causal", "Propensity score IPW pipeline", "This is the Way. -- Din Djarin")
_r(
    "ps_fit",
    "compute_propensity_scores",
    "Causal",
    "Propensity score estimation",
    "You have been well-trained. -- Dooku",
)
_r(
    "eac_ipw",
    "run_ebac_selection_ipw_analysis",
    "Causal",
    "eBAC selection-adjusted IPW",
    "Selection is a path to the dark side.",
)
_r(
    "te_ana",
    "run_treatment_effects_analysis",
    "Causal",
    "Treatment effects analysis",
    "The treatment has worked. -- Medical droid",
)
_r(
    "wlog",
    "run_weighted_logistic_analysis",
    "Causal",
    "Weighted logistic regression",
    "Weight and measure carefully.",
)
_r("cmp_log", "compare_nested_logistic_models", "Causal", "Nested logistic comparison", "Let them fight. -- Snoke")

# ── Semiparametric Inference (Kosorok 2008) ────────────────────────────────
_r("tmle", "tmle", "Semiparametric", "Targeted minimum loss-based estimation (TMLE)", "Target what matters, you must.")
_r("onest", "onest", "Semiparametric", "One-step estimator (Newton-Raphson correction)", "One step closer to the truth. --")
_r("ifunc", "ifunc", "Semiparametric", "Influence function computation", "Every observation leaves its mark. -- Chirrut")
_r("effsc", "effsc", "Semiparametric", "Efficient score function", "What is now proved was once only imagined. — William Blake")
_r("npmle", "npmle", "Semiparametric", "Nonparametric MLE (Turnbull EM)", "Let the data speak for itself. -- Qui-Gon")
_r("profk", "profk", "Semiparametric", "Profile likelihood (semiparametric)", "Profile the enemy, then strike. -- Thrawn")
_r("drctr", "drctr", "Semiparametric", "Doubly robust cross-fitted estimator (DR-TMLE)", "Two defenses are better than one. -- Padme")
_r("bootm", "bootm", "Semiparametric", "Bootstrap for M-estimators", "From the ashes, we rise again. -- Jyn Erso")

# ── Semiparametric Methods (Horowitz 2009) ────────────────────────────────
_r("sidxm", "sidxm", "Semiparametric", "Single-index maximum score estimator (Manski 1975)", "The index reveals the hidden structure. -- Thrawn")
_r("smscr", "smscr", "Semiparametric", "Smoothed maximum score estimator (Horowitz 1992)", "Smooth the path, find the truth. --")
_r("robns", "robns", "Semiparametric", "Robinson double-residual estimator", "Remove the noise, the signal remains. -- Qui-Gon")
_r("dcnvl", "dcnvl", "Semiparametric", "Deconvolution density estimation (measurement error)", "See through the fog you must.")
_r("npivt", "npivt", "Semiparametric", "Nonparametric IV via Tikhonov regularization", "What is now proved was once only imagined. — William Blake")
_r("npivs", "npivs", "Semiparametric", "Nonparametric IV via sieve estimation", "Build the basis, span the space. -- Ahsoka")
_r("ctrfn", "ctrfn", "Semiparametric", "Control function endogeneity correction", "Control what you can, accept what you cannot. -- Chirrut")
_r("bcxgm", "bcxgm", "Semiparametric", "Box-Cox transformation model via GMM", "What is now proved was once only imagined. — William Blake")
_r("admod", "admod", "Semiparametric", "Additive model via marginal integration", "The whole is the sum of its parts. -- Padme")
_r("bkfit", "bkfit", "Semiparametric", "Backfitting algorithm for additive models", "Iterate until the truth converges. -- Kanan")

# ── Semiparametric Estimation — Horowitz (2009) expanded ───────────────────
# Ch 2: Density Estimation
_r("bndcv", "bandwidth_loocv", "Semiparametric", "LOO cross-validation bandwidth for KDE", "Distribution helper.")
_r("bndpi", "bandwidth_plugin", "Semiparametric", "Sheather-Jones plug-in bandwidth", "Plug in the knowledge, the optimal path emerges. -- Qui-Gon")
_r("adkde", "adaptive_kde", "Semiparametric", "Adaptive variable-bandwidth KDE (Abramson)", "Distribution helper.")
_r("bkde", "boundary_kde", "Semiparametric", "Boundary-corrected KDE (reflection/renormalization)", "At the boundary, reflect or renormalize. --")
_r("mxkde", "mixture_kde", "Semiparametric", "Gaussian mixture model KDE via EM", "What is now proved was once only imagined. — William Blake")
# Ch 3: Nonparametric Regression
_r("llreg", "local_linear_reg", "Semiparametric", "Local linear regression (Fan 1992)", "Locally, the linear truth emerges.")
_r("lpreg", "local_poly_reg", "Semiparametric", "Local polynomial regression (degree p)", "Higher order, deeper truth. -- Dooku")
_r("splrg", "spline_regression", "Semiparametric", "Natural cubic spline regression", "Smooth the spline, fit the data. -- Qui-Gon")
_r("pnreg", "penalized_kernel_reg", "Semiparametric", "Penalized (ridge) kernel regression", "Penalize excess, find the balance. -- Chirrut")
_r("krreg", "kernel_ridge_reg", "Semiparametric", "Kernel ridge regression", "The dual solution, elegant it is.")
# Ch 4: Single-Index Models
_r("siavg", "si_avg_derivative", "Semiparametric", "Single-index via average derivative (PSS 1989)", "The average derivative points the direction. --")
_r("sisls", "si_iterative_sls", "Semiparametric", "Single-index via iterative SLS (Ichimura 1993)", "Iterate until convergence, patience you must have.")
_r("siprj", "si_projection_pursuit", "Semiparametric", "Single-index projection pursuit", "Project and pursue the hidden index. -- Thrawn")
_r("simle", "si_mle", "Semiparametric", "Single-index MLE (Klein-Spady 1993)", "What is now proved was once only imagined. — William Blake")
_r("simin", "si_min_distance", "Semiparametric", "Single-index minimum distance estimator", "Minimise the distance, find the truth. -- Mace Windu")
# Ch 5: Partially Linear Models
_r("plrob", "pl_robinson", "Semiparametric", "Robinson partially linear estimator (1988)", "Double residuals reveal the linear part. -- Qui-Gon")
_r("plsev", "pl_sieve", "Semiparametric", "Partially linear sieve estimator", "The sieve approximates what kernels cannot. -- Ahsoka")
_r("pldml", "pl_double_ml", "Semiparametric", "Partially linear double/debiased ML", "Cross-fit, debias, the modern way. -- Rey")
_r("plpen", "pl_penalized", "Semiparametric", "Partially linear penalized sieve", "Penalize the nonparametric, free the linear. -- Dooku")
_r("plgcv", "pl_gcv_bandwidth", "Semiparametric", "GCV bandwidth for partially linear model", "Generalise the cross-validation, you must.")
# Ch 6: Binary Response Models
_r("bnpkl", "binary_np_kernel_ll", "Semiparametric", "Binary NP kernel likelihood", "The kernel likelihood speaks in probabilities. -- C-3PO")
_r("bnmax", "binary_max_score", "Semiparametric", "Maximum score estimator (Manski 1975)", "Maximise the score, cube-root consistent. -- Thrawn")
_r("bnsmo", "binary_smoothed_score", "Semiparametric", "Smoothed maximum score (Horowitz 1992)", "Smooth the indicator, normality returns. --")
_r("bnhan", "binary_han_mrc", "Semiparametric", "'s maximum rank correlation (MRC)", "Rank the correlations, the robust estimator. --")
_r("bnklw", "binary_klein_spady", "Semiparametric", "Klein-Spady binary estimator", "What is now proved was once only imagined. — William Blake")
# Ch 7-8: Transformation + Sample Selection
_r("bxcsm", "boxcox_semiparametric", "Semiparametric", "Semiparametric Box-Cox transformation model", "What is now proved was once only imagined. — William Blake")
_r("smpsl", "sample_selection_sp", "Semiparametric", "Semiparametric sample selection correction", "Select with care, correct the bias. -- Padme")
_r("cnsrd", "censored_regression_sp", "Semiparametric", "Semiparametric censored regression (CLAD)", "Beyond the censoring point, truth hides. --")
_r("durat", "duration_hazard_sp", "Semiparametric", "Semiparametric duration/hazard model", "The hazard, smooth and kernel-estimated.")
_r("trmle", "transformation_mle", "Semiparametric", "Transformation model semiparametric MLE", "The monotone transformation, Bernstein approximated. -- Dooku")

# ── Empirical Process (Kosorok 2008) ──────────────────────────────────────
_r("tgscr", "tgscr", "EmpiricalProcess", "Tangent space projection", "The space between worlds holds secrets. -- Ahsoka")
_r("empbs", "empbs", "EmpiricalProcess", "Empirical bootstrap process", "The process reveals the pattern. -- Kanan")

# ── Statistical Tests (extended) ───────────────────────────────────────────
_r("anova", "anova_one_way", "Test", "One-way ANOVA F-test", "I find your lack of variance disturbing.")
_r("chisq", "chi_square_test", "Test", "Chi-square test", "Let the Wookiee win. -- C-3PO")
_r("fisher", "fisher_exact_test", "Test", "Fisher's exact test (2x2)", "What is now proved was once only imagined. — William Blake")
_r("mw", "mann_whitney_test", "Test", "Mann-Whitney U test", "It's not over yet. -- Anakin")
_r("wilcox", "wilcoxon_signed_rank_test", "Test", "Wilcoxon signed-rank test", "Signed and sealed. -- Lando")
_r("t1smp", "one_sample_t_test", "Test", "One-sample t-test", "Are you with us or against us? -- Anakin")
_r("t2smp", "two_sample_t_test", "Test", "Two-sample t-test (Welch/Student)", "Let them fight. -- Snoke")
_r("tpair", "paired_t_test", "Test", "Paired samples t-test", "Where one goes, the other follows. -- Rex")
_r("vr", "variance_ratio", "Test", "Variance ratio F-test", "What is now proved was once only imagined. — William Blake")

# ── Effect Sizes (extended) ────────────────────────────────────────────────
_r("d", "cohens_d", "EffectSize", "Cohen's d", "Size matters not.")
_r("g", "hedges_g", "EffectSize", "Hedges' g (bias-corrected d)", "Judge me by my size, do you?")
_r("gld", "glass_delta", "EffectSize", "Glass's delta", "Short help is still help. -- Wicket")
_r("eta2", "eta_squared", "EffectSize", "Eta-squared", "What is now proved was once only imagined. — William Blake")
_r("peta2", "partial_eta_squared", "EffectSize", "Partial eta-squared", "Distribution helper.")
_r("omega2", "omega_squared", "EffectSize", "Omega-squared", "What is now proved was once only imagined. — William Blake")
_r("eps2", "epsilon_squared", "EffectSize", "Epsilon-squared", "Modest but real. -- Mon Mothma")
_r("f_es", "cohens_f", "EffectSize", "Cohen's f", "Distribution helper.")
_r("cramv", "cramers_v", "EffectSize", "Cramer's V", "What is now proved was once only imagined. — William Blake")
_r("pbr", "point_biserial_r", "EffectSize", "Point-biserial correlation", "Two sides of the same coin. --")
_r("rho", "spearman_rho", "EffectSize", "Spearman's rho", "Distribution helper.")
_r("tau", "kendall_tau", "EffectSize", "Kendall's tau-b", "Rank them I shall.")
_r("cles", "cles", "EffectSize", "Common language effect size", "In a language we can all understand.")
_r("cliff", "cliffs_delta", "EffectSize", "Cliff's delta", "It's over. I have the high ground! --")
_r("vda", "vargha_delaney_a", "EffectSize", "Vargha-Delaney A", "One side prevails. -- Dooku")
_r("rbc", "rank_biserial_correlation", "EffectSize", "Rank-biserial correlation", "Distribution helper.")
_r("w", "cohens_w", "EffectSize", "Cohen's w", "Powerful the dark side is.")
_r("r2", "r_squared", "EffectSize", "R-squared as effect size", "Beep boop beep! -- R2-D2")
_r("r_es", "r_effect_size", "EffectSize", "Pearson r with Fisher z CI", "Connected everything is.")
_r(
    "cv",
    "coefficient_of_variation",
    "EffectSize",
    "Coefficient of variation",
    "Variation leads to the dark side.",
)

# ── Effect Size Conversions ────────────────────────────────────────────────
_r("d2nnt", "d_to_nnt", "Conversion", "Cohen's d to NNT", "More machine now than man. --")
_r("d2or", "d_to_or", "Conversion", "Cohen's d to odds ratio", "Transformed he has become.")
_r("d2r", "d_to_r", "Conversion", "Cohen's d to Pearson r", "Become something new you shall.")
_r("or2d", "or_to_d", "Conversion", "Odds ratio to Cohen's d", "Turning back is always an option. --")
_r("or2r", "or_to_r", "Conversion", "Odds ratio to Pearson r", "What is now proved was once only imagined. — William Blake")
_r("r2d", "r_to_d", "Conversion", "Pearson r to Cohen's d", "From correlation to cause.")
_r("r2or", "r_to_or", "Conversion", "Pearson r to odds ratio", "What is now proved was once only imagined. — William Blake")

# ── Epidemiological Measures ───────────────────────────────────────────────
_r("nnt", "number_needed_to_treat", "Epidemiology", "NNT from 2x2 table", "How many must we save? -- Padme")
_r("nnh", "number_needed_to_harm", "Epidemiology", "NNH from 2x2 table", "I sense great danger ahead.")
_r(
    "ird",
    "incidence_rate_difference",
    "Epidemiology",
    "Incidence rate difference",
    "Distribution helper.",
)
_r("irr", "rate_ratio", "Epidemiology", "Incidence rate ratio", "The rate of attack increases. -- Raddus")
_r("or_es", "odds_ratio", "Epidemiology", "Odds ratio with CI", "What is now proved was once only imagined. — William Blake")
_r("rd_es", "risk_difference", "Epidemiology", "Risk difference", "The gap between light and dark. -- Luke")
_r("rr_es", "risk_ratio", "Epidemiology", "Risk ratio (relative risk)", "Relative to one another they are.")

# ── Confidence Intervals ───────────────────────────────────────────────────
_r("or_ci", "odds_ratio_ci", "CI", "Odds ratio CI", "I am certain of this much. --")
_r("prop_ci", "proportion_ci", "CI", "Single proportion CI", "Distribution helper.")
_r("rd_ci", "risk_difference_ci", "CI", "Risk difference CI (Newcombe)", "The distance between us is known. --")
_r("rr_ci", "rate_ratio_ci", "CI", "Rate ratio CI", "Within this range the truth lies.")
_r("rsk_ci", "risk_ratio_ci", "CI", "Risk ratio CI (log-normal Wald)", "The risk is calculated. -- Cassian Andor")

# ── Power Analysis ─────────────────────────────────────────────────────────
_r("pwr_t", "power_t_test", "Power", "Power for t-tests", "What is now proved was once only imagined. — William Blake")
_r("pwr_p", "power_prop_test", "Power", "Power for two-proportion z-test", "What is now proved was once only imagined. — William Blake")
_r("pwr_av", "power_anova", "Power", "Power for one-way ANOVA", "The power to destroy a planet. -- Tarkin")
_r("i_pwr", "calculate_interaction_power", "Power", "Interaction power (ANOVA)", "Together they are stronger. -- Snoke")
_r(
    "n_logit",
    "sample_size_logistic",
    "Power",
    "Sample size for logistic regression",
    "We're gonna need more troops. -- Rex",
)

# ── Meta-Analysis ──────────────────────────────────────────────────────────
_r("femeta", "fixed_effects_meta", "Meta", "Fixed-effects meta-analysis", "What is now proved was once only imagined. — William Blake")
_r(
    "remeta",
    "random_effects_meta",
    "Meta",
    "Random-effects meta-analysis (DL)",
    "Together we are stronger. -- Jyn Erso",
)
_r("i2", "i_squared", "Meta", "Higgins I-squared heterogeneity", "I sense conflict in you. -- Snoke")
_r(
    "predi",
    "prediction_interval",
    "Meta",
    "Prediction interval for new study",
    "Difficult to see. Always in motion.",
)

# ── Distributions (extended) ──────────────────────────────────────────────
_r("dbet", "dbeta", "Distribution", "Beta PDF", "Two outcomes define the fate. -- Chirrut")
_r("pbet", "pbeta", "Distribution", "Beta CDF", "Between zero and one the truth lives.")
_r("qbet", "qbeta", "Distribution", "Beta quantile", "At what threshold do we act? -- Padme")
_r("dgeom", "dgeom", "Distribution", "Geometric PMF", "Try and try again.")
_r("pgeom", "pgeom", "Distribution", "Geometric CDF", "First success is all that matters. --")
_r("rgeom", "rgeom", "Distribution", "Geometric random", "How long until we succeed? --")
_r("dhyp", "dhyp", "Distribution", "Hypergeometric PMF", "Drawing from a limited supply. -- Hera")
_r("dlnrm", "dlnrm", "Distribution", "Lognormal PDF", "Skewed the distribution is.")
_r("plnrm", "plnrm", "Distribution", "Lognormal CDF", "Heavier tails than expected. --")
_r("qlnrm", "qlnrm", "Distribution", "Lognormal quantile", "Where the extremes begin. -- Anakin")
_r("rlnrm", "rlnrm", "Distribution", "Lognormal random", "Unpredictable these outcomes are.")
_r("dnbm", "dnbm", "Distribution", "Negative binomial PMF", "How many failures before success? -- Rex")
_r("pnbm", "pnbm", "Distribution", "Negative binomial CDF", "Patience. We will succeed. -- Thrawn")
_r("qnbm", "qnbm", "Distribution", "Negative binomial quantile", "The threshold of endurance. -- Ahsoka")
_r("rnbm", "rnbm", "Distribution", "Negative binomial random", "Each trial a new chance. --")
_r("dweib", "dweib", "Distribution", "Weibull PDF", "All things wear with time.")
_r("pweib", "pweib", "Distribution", "Weibull CDF", "Time is running out. -- Padme")
_r("qweib", "qweib", "Distribution", "Weibull quantile", "When will it break? -- Anakin")
_r("rweib", "rweib", "Distribution", "Weibull random", "Endurance is random. -- Chirrut")
_r("qbnm", "qbinom", "Distribution", "Binomial quantile", "How many must we win? -- Mon Mothma")
_r("rbnm", "rbinom", "Distribution", "Binomial random", "Fate decides the outcome. -- Qui-Gon")
_r("qpoi", "qpois", "Distribution", "Poisson quantile", "When the swarm arrives. -- Ackbar")
_r("rpoi", "rpois", "Distribution", "Poisson random", "Random they come.")
_r("runf", "runif", "Distribution", "Uniform random", "Equal chances for all. -- Padme")
_r("pgam", "pgamma", "Distribution", "Gamma CDF", "Distribution helper.")

# ── ML / Robustness ───────────────────────────────────────────────────────
_r("robust", "eval_robustness", "ML", "Random Forest robustness evaluation", "I am a fast learner. -- Rey")
_r("smote", "apply_smote", "ML", "SMOTE oversampling", "We need more troops. -- Rex")
_r("stdb", "standardized_coefficients", "ML", "Standardised beta weights", "Measure them on equal footing.")
_r("boot", "bootstrap_ci", "Resampling", "Bootstrap CI (non-parametric)", "We rise from our own data. -- Jyn Erso")
_r(
    "bsci",
    "bootstrap_effect_size_ci",
    "Resampling",
    "Bootstrap CI for effect size",
    "Confidence from repetition. --",
)

# ── Data / IO ─────────────────────────────────────────────────────────────
_r("dat", "dat", "Data", "Path to moirais.db", "Distribution helper.")
_r("loadds", "loadds", "Data", "Load dataset from file", "Bring me what I need. -- Kylo Ren")
_r("lds", "lds", "Data", "Load dataset by catalog key", "The archives are comprehensive. -- Jocasta Nu")
_r("lstds", "lstds", "Data", "List all datasets with cache status", "If not in our records, it does not exist.")
_r("dsinfo", "dsinfo", "Data", "Dataset metadata by key", "What do we know of this system? --")
_r("profds", "profds", "Data", "Dataset profiling", "You must understand the terrain. -- Thrawn")
_r("sugpln", "sugpln", "Data", "Suggest analysis plan", "Trust the plan. -- Kanan Jarrus")
_r("cpads", "cpads", "Data", "Load CPADS data", "This is public health data. -- Padme")
_r("cconn", "cconn", "Data", "Open/create SQLite cache", "What is now proved was once only imagined. — William Blake")
_r("ckan", "ckan", "Data", "Fetch dataset from CKAN", "What is now proved was once only imagined. — William Blake")
_r("clist", "clist", "Data", "List cached tables", "Show me what you have. -- Maz Kanata")
_r("cload", "cload", "Data", "Load from cache", "Load the cargo. -- Hera Syndulla")
_r("cstor", "cstor", "Data", "Store DataFrame in cache", "Store it safely. -- Bail Organa")
_r("infml", "infml", "Data", "Infer measurement level (NOIR)", "What type of reading is this? --")

# ── Verification / Inspection ─────────────────────────────────────────────
_r("inspct", "inspct", "Verification", "Inspect output file", "What is now proved was once only imagined. — William Blake")
_r("inspdr", "inspdr", "Verification", "Inspect directory of CSVs", "Leave no stone unturned. -- Rex")
_r("vrfy", "vrfy", "Verification", "Verify statistical output", "Verify the transmission. -- Mon Mothma")
_r("vrfydr", "vrfydr", "Verification", "Verify CSV directory", "Check every corridor. -- Rex")
_r("rinsp", "rinsp", "Verification", "Render inspection result", "Let me see. -- Luke")
_r("rvrfy", "rvrfy", "Verification", "Render verification report", "Confirm the results. -- Thrawn")

# ── OTIS Placement Analytics ──────────────────────────────────────────────
_r("rpl_r", "rplace_by_region", "OTIS", "Placement counts for single region", "I have the high ground. --")
_r("rpl_a", "rplace_by_age", "OTIS", "Placement by age group", "Young or old, it matters not.")
_r("rpl_g", "rplace_by_gender", "OTIS", "Placement by gender", "We fight together. -- Ahsoka")
_r("rpl_ra", "rplace_region_age", "OTIS", "Region x age cross-tab", "Region and age, intertwined.")
_r("rpl_rg", "rplace_region_gender", "OTIS", "Region x gender cross-tab", "Where and who. -- Din Djarin")
_r("rpl_ag", "rplace_age_gender", "OTIS", "Age x gender cross-tab", "Youth and gender both matter. -- Padme")
_r("rpl_t", "rplace_trend", "OTIS", "Placement trend over time", "The situation is evolving. -- Mon Mothma")
_r("rpl_rt", "rplace_region_trend", "OTIS", "Region trend over years", "Regions change over time. -- Thrawn")
_r("rpl_at", "rplace_age_trend", "OTIS", "Age group trend over years", "The young ones grow up. --")
_r("rpl_gt", "rplace_gender_trend", "OTIS", "Gender trend over years", "Patterns shift with each generation. --")
_r("rprat", "rplace_rate", "OTIS", "Placement rate per 100K", "The rate of confinement rises. -- Organa")
_r("rpdur", "rplace_duration", "OTIS", "Placement duration stats", "How long were they held? -- Padme")
_r("rpfrq", "rplace_frequency", "OTIS", "Repeat placement frequency", "Again and again.")
_r("rpfst", "rplace_first", "OTIS", "Time to first placement", "The first time is always hardest. -- Ahsoka")
_r("rpgap", "rplace_gap", "OTIS", "Mean gap between placements", "The space between events. -- Chirrut")

# ── OTIS Alert Analytics ──────────────────────────────────────────────────
_r("alrt1", "alrt_mh", "OTIS", "Mental health alert prevalence", "I sense a disturbance.")
_r("alrt2", "alrt_sr", "OTIS", "Suicide risk alert prevalence", "I've got a bad feeling about this. --")
_r("alrt3", "alrt_sw", "OTIS", "Suicide watch alert prevalence", "We're running out of time. --")
_r("alco", "alcooc", "OTIS", "Alert co-occurrence rates", "Multiple threats detected. -- K-2SO")
_r("altm", "altmrng", "OTIS", "Alert timeline per individual", "A timeline of suffering. -- Padme")
_r("aldur", "aldurn", "OTIS", "Mean alert duration", "How long has this been going on? --")
_r("alesc", "alescl", "OTIS", "Alert escalation patterns", "It is escalating. -- Mace Windu")
_r("altrn", "altrans", "OTIS", "Alert-state transition matrix", "One state leads to another.")
_r("alprv", "alprev", "OTIS", "Alert prevalence by group", "How widespread is the danger? -- Mon Mothma")
_r("alinc", "alincd", "OTIS", "New alert incidence rate", "New cases are emerging. -- Medical droid")
_r("alrsk", "alrisk", "OTIS", "Composite alert risk score", "The risk is growing. -- Bail Organa")
_r("alcmx", "alcmpx", "OTIS", "Alert complexity index", "More complex than you think. -- Ahsoka")

# ── OTIS Volatility Variants ─────────────────────────────────────────────
_r("vol_r", "vol_reg", "OTIS", "Volatility by origin region", "What is now proved was once only imagined. — William Blake")
_r("vol_a", "vol_age", "OTIS", "Volatility by age group", "The young are most volatile. --")
_r("vol_t", "vol_trd", "OTIS", "Volatility trend over time", "Turbulence ahead. -- Hera Syndulla")

# ── IRT (Item Response Theory) ────────────────────────────────────────────
_r("irt1p", "irt1p", "IRT", "1PL Rasch model", "One parameter to rule them all.")
_r("irt2p", "irt2p", "IRT", "2PL IRT model", "Difficulty and discrimination.")
_r("irt3p", "irt3p", "IRT", "3PL IRT model", "What is now proved was once only imagined. — William Blake")
_r("irtgr", "irtgr", "IRT", "Graded response model", "Degrees of agreement there are.")
_r("irtpc", "irtpc", "IRT", "Partial credit model", "Partial credit is still credit. -- Ahsoka")
_r("irtif", "irtif", "IRT", "Item information function", "Where the information lies. -- Chirrut")
_r("irttf", "irttf", "IRT", "Test information function", "The test reveals all. -- Mace Windu")
_r("irtic", "irtic", "IRT", "Item characteristic curve", "The shape of each item.")
_r("irtab", "irtab", "IRT", "Ability (theta) estimation", "Know your own ability. --")
_r("irtfl", "irtfl", "IRT", "Item fit statistics (infit/outfit)", "Does this item fit? -- K-2SO")

# ── DIF (Differential Item Functioning) ───────────────────────────────────
_r("difmh", "difmh", "DIF", "Mantel-Haenszel DIF", "Unfair the test must not be.")
_r("diflr", "diflr", "DIF", "Logistic regression DIF", "Bias I sense in this item.")
_r("difef", "difef", "DIF", "DIF effect size (MH delta)", "How large is the unfairness? -- Padme")
_r("difgn", "difgn", "DIF", "DIF by gender", "Gender bias we must detect. --")
_r("difag", "difag", "DIF", "DIF by age group", "Age should not decide fairness.")

# ── Reliability Variants ──────────────────────────────────────────────────
_r("kr20", "kr20", "Reliability", "Kuder-Richardson 20", "Consistent the binary items must be.")
_r("kr21", "kr21", "Reliability", "Kuder-Richardson 21", "A simpler approximation. --")
_r("gl1", "gl1", "Reliability", "Guttman's Lambda 1", "The first measure of reliability.")
_r("gl2", "gl2", "Reliability", "Guttman's Lambda 2", "A better bound emerges. -- Mace Windu")
_r("gl3", "gl3", "Reliability", "Guttman's Lambda 3 (= alpha)", "Alpha by another name. --")
_r("gl4", "gl4", "Reliability", "Guttman's Lambda 4 (max split-half)", "The best split-half found.")
_r("gl5", "gl5", "Reliability", "Guttman's Lambda 5", "Refined the estimate becomes.")
_r("gl6", "gl6", "Reliability", "Guttman's Lambda 6 (SMC-based)", "Squared multiple correlations reveal all.")
_r("rglb", "rglb", "Reliability", "Greatest lower bound", "The lowest it could possibly be.")
_r("rsem", "rsem", "Reliability", "Standard error of measurement", "Precise as a blaster. -- Jango Fett")
_r("rseh", "rseh", "Reliability", "SEM with confidence interval", "Precision with confidence. -- Thrawn")
_r("rmdc", "rmdc", "Reliability", "Minimal detectable change", "The smallest change we can detect. -- Rex")
_r("rmci", "rmci", "Reliability", "Reliable change index", "Is this change real? --")
_r("rcsem", "rcsem", "Reliability", "Conditional SEM by score level", "Precision varies by ability.")
_r("rirr", "rirr", "Reliability", "Inter-rater reliability (kappa)", "Do the raters agree? -- Mace Windu")

# ── Bayesian ──────────────────────────────────────────────────────────────
_r("bpost", "conjugate_posterior", "Bayesian", "Conjugate posterior updating", "I feel the good in you. -- Luke")
_r(
    "bfact",
    "bayes_factor_bic",
    "Bayesian",
    "Bayes factor via BIC approximation",
    "The evidence speaks for itself. -- Thrawn",
)
_r(
    "hdi",
    "highest_density_interval",
    "Bayesian",
    "Highest density interval (HDI)",
    "Narrow the uncertainty is.",
)
_r("rope", "rope_test", "Bayesian", "Region of Practical Equivalence", "Practically equivalent. --")
_r("waic", "compute_waic", "Bayesian", "WAIC model comparison", "Which model is stronger?")
_r("loo", "compute_loo", "Bayesian", "LOO-CV via PSIS-LOO", "One at a time, remove them.")
_r("mh_", "metropolis_hastings", "Bayesian", "Metropolis-Hastings MCMC sampler", "A random walk this is.")
_r("gibbs", "gibbs_normal", "Bayesian", "Gibbs sampler (Normal model)", "Sample from the conditionals.")
_r(
    "bci",
    "bayesian_credible_interval",
    "Bayesian",
    "Bayesian credible interval",
    "Credible the interval must be.",
)
_r(
    "blr",
    "bayesian_linear_regression",
    "Bayesian",
    "Bayesian linear regression",
    "What is now proved was once only imagined. — William Blake",
)

# ── Causal (extended 2) ──────────────────────────────────────────────────
_r("medtn", "causal_mediation", "Causal", "Causal mediation (Baron-Kenny)", "Through another the effect flows.")
_r("iv_wk", "weak_instrument_test", "Causal", "Weak instrument diagnostics", "Weak instruments lead to the dark side.")
_r("bwt", "rdd_bandwidth", "Causal", "RDD bandwidth selection (IK)", "How wide should the window be? -- Thrawn")
_r("cic", "changes_in_changes", "Causal", "Changes-in-changes estimator", "Changes within changes.")
_r("stag", "staggered_did", "Causal", "Staggered DiD (Callaway-Sant'Anna)", "Not all at once; they stagger. -- Rex")
_r("dr_", "doubly_robust_ate", "Causal", "Doubly-robust ATE estimator", "A backup plan we always need. -- Padme")
_r("trim", "ps_trim", "Causal", "Propensity score trimming", "Cut away the extremes. --")
_r(
    "over",
    "overlap_diagnostics",
    "Causal",
    "Overlap / common support diagnostics",
    "Common ground we must find. -- Padme",
)
_r("bal", "balance_diagnostics", "Causal", "Balance diagnostics (SMD)", "Distribution helper.")
_r("bnd", "manski_bounds", "Causal", "Manski partial identification bounds", "Bounds on what we can know.")

# ── Epidemiology (extended) ──────────────────────────────────────────────
_r("inc", "incidence_rate", "Epidemiology", "Incidence rate with CI", "New cases arise. -- Medical droid")
_r("prev", "point_prevalence", "Epidemiology", "Point prevalence with CI", "How widespread is the suffering? -- Padme")
_r(
    "smr",
    "standardized_mortality_ratio",
    "Epidemiology",
    "Standardized Mortality Ratio",
    "The mortality is higher than expected.",
)
_r(
    "paf",
    "population_attributable_fraction",
    "Epidemiology",
    "Population Attributable Fraction",
    "How much is attributable? --",
)
_r("ltab", "life_table", "Epidemiology", "Abridged life table", "The table of life and death.")
_r(
    "sir",
    "standardized_incidence_ratio",
    "Epidemiology",
    "Standardized Incidence Ratio",
    "More than expected have fallen. -- Rex",
)
_r("yll", "years_of_life_lost", "Epidemiology", "Years of Life Lost", "Gone too soon they were.")
_r("daly", "disability_adjusted_life_years", "Epidemiology", "DALY (YLL + YLD)", "Years lost to suffering. -- Padme")
_r(
    "sar",
    "secondary_attack_rate",
    "Epidemiology",
    "Secondary Attack Rate",
    "The spread among the close. -- Bail Organa",
)
_r("ar_", "attack_rate", "Epidemiology", "Attack rate (cumulative incidence)", "The attack has begun. --")


# ── Auto-registered (session 2026-04-07c batch 2) ─────────────────────
_r(
    "aft",
    "aft",
    "Survival",
    "Accelerated Failure Time model (Weibull AFT via MLE).",
    "Accelerated the timeline is.",
)
_r(
    "auc_",
    "auc_score",
    "ML",
    "Area Under the ROC Curve (AUC) via trapezoidal integration.",
    "The area beneath tells the truth. -- Chirrut",
)
_r("binom", "binomial_test", "Test", "Exact binomial test.", "Success or failure.")
_r(
    "bp",
    "breusch_pagan_test",
    "Test",
    "Breusch-Pagan test for heteroscedasticity.",
    "Constant the variance is not.",
)
_r(
    "bp_ts",
    "bp_ts",
    "TimeSeries",
    "Bai-Perron structural break test (simplified OLS-based).",
    "A break in the timeline I sense.",
)
_r(
    "calib",
    "calibration_curve",
    "ML",
    "Calibration curve and Brier score.",
    "Well-calibrated your predictions must be.",
)
_r(
    "ccf",
    "ccf",
    "TimeSeries",
    "Cross-correlation function between two time series.",
    "Two signals, one connection. -- Chirrut",
)
_r(
    "cfa",
    "cfa",
    "Multivariate",
    "Confirmatory Factor Analysis (basic implementation).",
    "Confirm the structure you must.",
)
_r("cm", "confusion_matrix", "ML", "Confusion matrix and derived metrics.", "Know your true positives. -- K-2SO")
_r(
    "cocht",
    "cochrans_q_test",
    "Test",
    "Cochran's Q test for k related binary samples.",
    "Related the samples are.",
)
_r(
    "crisk",
    "crisk",
    "Survival",
    "Competing risks — Cumulative Incidence Function (CIF).",
    "Competing fates await. -- Chirrut",
)
_r(
    "cumhz",
    "cumhz",
    "Survival",
    "Cumulative hazard from Kaplan-Meier survival estimates.",
    "The danger accumulates. -- Bail Organa",
)
_r(
    "cusum",
    "cusum",
    "TimeSeries",
    "CUSUM change-point detection for time series.",
    "Distribution helper.",
)
_r("dbscn", "dbscn", "Multivariate", "DBSCAN density-based clustering.", "Dense clusters reveal structure. -- Thrawn")
_r("dcchy", "dcchy", "Dist", "Cauchy distribution probability density function.", "Heavy tails the Cauchy has.")
_r(
    "dlogi",
    "dlogi",
    "Dist",
    "Logistic distribution probability density function.",
    "A sigmoid path this follows.",
)
_r(
    "dor",
    "diagnostic_odds_ratio",
    "DiagAccuracy",
    "Diagnostic Odds Ratio (DOR).",
    "The diagnostic odds favor us. --",
)
_r(
    "dtree",
    "dtree_classify",
    "ML",
    "Decision tree classifier (pure NumPy, Gini splitting).",
    "Choose your path wisely.",
)
_r("dts", "dip_test", "Test", "Hartigan's dip test for unimodality.", "Is it truly one peak? --")
_r(
    "elbow",
    "elbow_method",
    "ML",
    "Elbow method for optimal k in k-means clustering.",
    "At the bend the answer lies.",
)
_r(
    "em_i",
    "em_impute",
    "MissingData",
    "EM algorithm imputation for multivariate normal data.",
    "The missing pieces found they must be.",
)
_r(
    "ets",
    "ets",
    "TimeSeries",
    "Exponential smoothing (Holt-Winters) for time series forecas",
    "Smooth the signal must be.",
)
_r("f1_", "f1_score", "ML", "F1 score, precision, and recall.", "Balance precision and recall. -- K-2SO")
_r(
    "fa",
    "fa",
    "Multivariate",
    "Exploratory Factor Analysis via principal axis factoring.",
    "Hidden factors you must uncover.",
)
_r(
    "feat",
    "feature_importance",
    "ML",
    "Permutation-based feature importance.",
    "Which feature is most important? -- Rey",
)
_r(
    "gbm",
    "gradient_boosting",
    "ML",
    "Gradient boosting machine (simplified, pure NumPy).",
    "Each tree learns from the last.",
)
_r("grubs", "grubbs_test", "Test", "Grubbs' test for a single outlier.", "One outlier there is.")
_r("hclst", "hclst", "Multivariate", "Hierarchical agglomerative clustering.", "A hierarchy of clusters. -- Thrawn")
_r("holo_a", "holo_acf", "Vis", "ACF / PACF plot visualization.", "See the autocorrelation you must.")
_r("holo_b", "holo_box", "Vis", "Box plot visualization.", "The box reveals the spread. --")
_r("holo_c", "holo_corr", "Vis", "Correlation heatmap visualization.", "See the connections clearly. -- Chirrut")
_r("holo_d", "holo_dag", "Vis", "DAG (Directed Acyclic Graph) diagram.", "The path of causation. -- Qui-Gon")
_r("holo_e", "holo_effect", "Vis", "Effect size forest plot visualization.", "See the size of the effect.")
_r("holo_f", "holo_forest", "Vis", "Forest plot for meta-analysis.", "A forest of evidence. -- Jyn Erso")
_r("holo_h", "holo_hist", "Vis", "Histogram visualization.", "The shape of the data revealed. -- Luke")
_r("holo_i", "holo_roc", "Vis", "ROC curve visualization.", "True from false, the curve shows. -- Chirrut")
_r("holo_k", "holo_km", "Vis", "Kaplan-Meier survival curve.", "The curve of survival. --")
_r("holo_m", "holo_mosaic", "Vis", "Mosaic plot visualization.", "Proportions within proportions.")
_r("holo_p", "holo_pair", "Vis", "Pair plot (scatter matrix) visualization.", "See all variables at once. -- Thrawn")
_r("holo_q", "holo_qq", "Vis", "QQ plot visualization.", "Is the distribution as expected? --")
_r("holo_r", "holo_resid", "Vis", "Residual plot visualization.", "Residuals tell the untold story.")
_r("holo_s", "holo_scatter", "Vis", "Scatter plot visualization.", "Two variables, one plot. -- Luke")
_r("holo_v", "holo_violin", "Vis", "Violin plot visualization.", "The full distribution revealed. -- Chirrut")
_r("holo_w", "holo_funnel", "Vis", "Funnel plot visualization.", "Bias the funnel reveals.")
_r(
    "hurst",
    "hurst",
    "TimeSeries",
    "Hurst exponent estimation via rescaled range (R/S) analysis.",
    "Long memory the series has.",
)
_r(
    "irf",
    "irf",
    "TimeSeries",
    "Impulse Response Function from a VAR coefficient matrix.",
    "One shock ripples through all. --",
)
_r(
    "jt",
    "jonckheere_terpstra_test",
    "Test",
    "Jonckheere-Terpstra test for ordered alternatives.",
    "An ordered alternative there is.",
)
_r(
    "kmean",
    "kmean",
    "Multivariate",
    "K-means clustering via Lloyd's algorithm with k-means++ init",
    "K clusters you shall find.",
)
_r(
    "knn",
    "knn_classify",
    "ML",
    "k-Nearest Neighbors classifier (pure NumPy).",
    "By their neighbors you shall know them.",
)
_r(
    "knn_i",
    "knn_impute",
    "MissingData",
    "KNN imputation for missing data.",
    "Let the neighbors fill the gaps. -- Chirrut",
)
_r(
    "lda_",
    "lda_",
    "Multivariate",
    "Linear Discriminant Analysis (Fisher's LDA).",
    "Discriminate between the groups. -- Thrawn",
)
_r(
    "learn",
    "learning_curve",
    "ML",
    "Learning curve: train/test error vs. training set size.",
    "Much to learn you still have.",
)
_r(
    "lilf",
    "lilliefors_test",
    "Test",
    "Lilliefors test (KS with estimated parameters).",
    "Normal the data may not be.",
)
_r("logls", "log_loss", "ML", "Log loss (binary cross-entropy).", "How wrong were we? -- C-3PO")
_r(
    "lr_",
    "likelihood_ratios",
    "DiagAccuracy",
    "Likelihood Ratios (LR+, LR-) for diagnostic tests.",
    "The likelihood ratio tells the truth.",
)
_r("ma_", "ma_", "TimeSeries", "Moving average smoother for time series.", "Smooth out the noise. --")
_r(
    "mcar",
    "littles_mcar_test",
    "MissingData",
    "Little's MCAR test for missing completely at random.",
    "Missing at random or not?",
)
_r(
    "mcc",
    "matthews_corrcoef",
    "ML",
    "Matthews Correlation Coefficient (MCC).",
    "A balanced measure of truth. -- Chirrut",
)
_r("mcnem", "mcnemar_test", "Test", "McNemar's test for paired proportions.", "The pairs have changed. -- Ahsoka")
_r(
    "md_pat",
    "missing_data_patterns",
    "MissingData",
    "Missing data pattern analysis.",
    "These aren't the data you're looking for.",
)
_r(
    "mds",
    "mds",
    "Multivariate",
    "Classical Multidimensional Scaling (Torgerson scaling).",
    "Reduce the dimensions. -- Thrawn",
)
_r(
    "mice",
    "mice_impute",
    "MissingData",
    "Multiple Imputation by Chained Equations (MICE).",
    "Impute the missing pieces.",
)
_r("mood", "mood_median_test", "Test", "Mood's median test.", "The median is the message. -- Mon Mothma")
_r("nels", "nels", "Survival", "Nelson-Aalen cumulative hazard estimator.", "The hazard accumulates. -- Bail Organa")
_r(
    "nn_",
    "nn_classify",
    "ML",
    "Simple neural network (1 hidden layer, pure NumPy).",
    "A neural path to understanding.",
)
_r(
    "nnd",
    "number_needed_to_diagnose",
    "DiagAccuracy",
    "Number Needed to Diagnose (NND).",
    "How many to diagnose one? -- Medical droid",
)
_r(
    "page",
    "page_trend_test",
    "Test",
    "Page's L trend test for ordered alternatives.",
    "An ordered trend there is.",
)
_r(
    "pca",
    "pca",
    "Multivariate",
    "Principal Component Analysis via eigendecomposition.",
    "A more elegant weapon. --",
)
_r(
    "pcchy",
    "pcchy",
    "Dist",
    "Cauchy distribution cumulative distribution function.",
    "The Cauchy reveals its tail.",
)
_r(
    "plogi",
    "plogi",
    "Dist",
    "Logistic distribution cumulative distribution function.",
    "The logistic curve approaches one.",
)
_r(
    "pmm",
    "pmm_impute",
    "MissingData",
    "Predictive mean matching imputation.",
    "Match the mean, find the value. -- Chirrut",
)
_r("qcchy", "qcchy", "Dist", "Cauchy distribution quantile function.", "Where the Cauchy breaks.")
_r("qlogi", "qlogi", "Dist", "Logistic distribution quantile function.", "The threshold of the logistic.")
_r("rcchy", "rcchy", "Dist", "Cauchy distribution random variate generation.", "Random from the Cauchy.")
_r("reset", "ramsey_reset_test", "Test", "Ramsey RESET specification test.", "Misspecified the model is.")
_r("rey_gm", "rey_gm", "Regression", "Gamma GLM regression via IRLS.", "The shape of the response. -- Rey")
_r(
    "rey_mx",
    "rey_mx",
    "Regression",
    "Mixed-effects model (random intercept via EM algorithm).",
    "Fixed and random, both there are.",
)
_r("rey_nb", "rey_nb", "Regression", "Negative binomial regression (GLM) via IRLS.", "Overdispersion I sense.")
_r(
    "rey_ol",
    "rey_ol",
    "Regression",
    "Ordinal logistic regression (proportional odds model).",
    "Ordered the outcomes are.",
)
_r(
    "rey_sv",
    "rey_sv",
    "Regression",
    "Survey-weighted regression with robust (sandwich) standard e",
    "Weighted for the survey design. -- Mon Mothma",
)
_r(
    "rey_tw",
    "rey_tw",
    "Regression",
    "Tweedie regression (compound Poisson-gamma GLM).",
    "Between Poisson and Gamma it lies.",
)
_r(
    "rey_zp",
    "rey_zp",
    "Regression",
    "Zero-inflated Poisson regression via EM algorithm.",
    "Too many zeros there are.",
)
_r(
    "rf_i",
    "rf_impute",
    "MissingData",
    "Iterative OLS regression imputation (simplified random-fores",
    "Let the forest fill the gaps.",
)
_r(
    "rforc",
    "random_forest",
    "ML",
    "Random forest classifier (pure NumPy, bootstrap + feature su",
    "Many trees make a forest.",
)
_r("rlogi", "rlogi", "Dist", "Logistic distribution random variate generation.", "Random from the logistic.")
_r(
    "rubin",
    "rubins_rules",
    "MissingData",
    "Rubin's rules for pooling multiply imputed estimates.",
    "Combine the imputed estimates.",
)
_r("runs", "runs_test", "Test", "Wald-Wolfowitz runs test for randomness.", "Random the sequence must be.")
_r(
    "schon",
    "schon",
    "Survival",
    "Schoenfeld residuals for proportional hazards assumption tes",
    "The assumption must hold. --",
)
_r(
    "shap_",
    "shap_values",
    "ML",
    "Simplified SHAP values via permutation-based feature attribu",
    "Each feature's contribution revealed. -- Rey",
)
_r("sign", "sign_test", "Test", "Sign test for paired data.", "Which direction does the difference go?")
_r(
    "silh",
    "silhouette_score",
    "ML",
    "Silhouette score for cluster quality evaluation.",
    "Well-separated the clusters are.",
)
_r(
    "svm_",
    "svm_classify",
    "ML",
    "Support Vector Machine classifier wrapper.",
    "Find the separating hyperplane. -- Thrawn",
)
_r(
    "tsne",
    "tsne",
    "Multivariate",
    "t-SNE (t-distributed Stochastic Neighbour Embedding).",
    "High dimensions compressed. -- Thrawn",
)
_r(
    "umap_",
    "umap_",
    "Multivariate",
    "UMAP (Uniform Manifold Approximation and Projection) — simpl",
    "The manifold revealed. -- Chirrut",
)
_r(
    "vecm",
    "vecm",
    "TimeSeries",
    "Vector Error Correction Model (VECM) for cointegrated series",
    "Cointegrated the series are.",
)
_r("white", "white_test", "Test", "White's test for heteroscedasticity.", "Heteroscedasticity I sense.")
_r(
    "xgb",
    "xgb_classify",
    "ML",
    "XGBoost / gradient boosting classifier wrapper.",
    "Boosted and ready for battle. -- Rex",
)

# ── Utility ─────────────────────────────────────────────────────────────────

_r("balph", "bayesian_alpha", "Psymet", "Bayesian Cronbach's alpha with posterior distribution.", "")
_r("bcfa", "bayesian_cfa", "Psymet", "Bayesian Confirmatory Factor Analysis with posterior fit ind", "")
_r("bcomp", "bayesian_model_compare", "Psymet", "Bayesian model comparison (DIC, WAIC, Bayes factor approxima", "")
_r("bdif", "bayesian_dif", "Psymet", "Bayesian DIF detection via parameter posterior differences.", "")
_r("bfsc", "bayesian_factor_scores", "Psymet", "Bayesian factor scores with uncertainty.", "")
_r("birt", "bayesian_irt_2pl", "Psymet", "Bayesian 2PL IRT model via Gibbs sampler.", "")
_r("bmi", "bayesian_mi", "Psymet", "Bayesian measurement invariance across groups.", "")
_r("bomg", "bayesian_omega", "Psymet", "Bayesian McDonald's omega.", "")
_r("bppc", "bayesian_ppc", "Psymet", "Posterior predictive check for Bayesian psychometric models.", "")
_r("brcc", "bayesian_rci", "Psymet", "Bayesian reliable change index.", "")
_r("cfa4", "cfa_4factor", "Psymet", "4-factor CFA using MAPQ structure (EE/EA/UA/ER).", "")
_r("cfabi", "cfa_bifactor", "Psymet", "Bifactor CFA model (general + 4 specific factors).", "")
_r("cfacm", "cfa_compare", "Psymet", "Nested CFA model comparison (chi-square difference test).", "")
_r("cfafi", "cfa_fit", "Psymet", "Compute all fit indices for any CFA structure.", "")
_r("cfaln", "cfa_loadings", "Psymet", "Standardized factor loadings from CFA.", "")
_r("cfami", "cfa_modindex", "Psymet", "Modification indices for CFA models.", "")
_r("cfars", "cfa_residuals", "Psymet", "Residual correlation matrix from CFA.", "")
_r("cmpag", "compliance_by_age", "OTIS", "Compliance rate by age group.", "")
_r("cmpgn", "compliance_by_gender", "OTIS", "Compliance rate by gender.", "")
_r("cmprg", "compliance_by_region", "OTIS", "Compliance rate by region.", "")
_r("cmprt", "compliance_rate", "OTIS", "Overall and by-group compliance rate.", "")
_r("cmptr", "compliance_trend", "OTIS", "Compliance trend over fiscal years.", "")
_r("cstag", "custody_age_profile", "OTIS", "Custody age profile over time.", "")
_r("cstdy", "custody_days", "OTIS", "Total custody days per individual.", "")
_r("cstgp", "custody_gender_parity", "OTIS", "Custody gender parity index.", "")
_r("cstgv", "custody_grievance_rate", "OTIS", "Custody grievance rate by facility type.", "")
_r("csthl", "custody_health_access", "OTIS", "Custody health access — alert rate by group.", "")
_r("cstin", "custody_incident_rate", "OTIS", "Custody incident rate per 1000 person-days.", "")
_r("cstlk", "custody_lockdown_freq", "OTIS", "Custody lockdown frequency per year.", "")
_r("cstmh", "custody_mental_health", "OTIS", "Custody mental health flag trend over years.", "")
_r("cstoc", "custody_occupancy", "OTIS", "Custody occupancy count per facility per year.", "")
_r("cstpg", "custody_program_rate", "OTIS", "Custody program participation rate by region.", "")
_r("cstre", "custody_readmit", "OTIS", "Custody readmission rate.", "")
_r("cstsg", "custody_segregation", "OTIS", "Custody segregation indicator proportion.", "")
_r("cstsu", "custody_substance", "OTIS", "Custody substance flag by age group.", "")
_r("csttm", "custody_time_served", "OTIS", "Custody time served distribution.", "")
_r("csttr", "custody_transfer_rate", "OTIS", "Custody transfer rate between regions per year.", "")
_r("efa2", "efa_nfactors", "Psymet", "Determine optimal number of factors (parallel analysis, MAP,", "")
_r("efart", "efa_rotate", "Psymet", "Rotate factor loadings (varimax, promax, oblimin).", "")
_r("efasc", "efa_scores", "Psymet", "Compute factor scores from data and loadings.", "")
_r("inspfr", "inspection_fail_rate", "OTIS", "Inspection fail rate — proportion below threshold.", "")
_r("insprt", "inspection_score", "OTIS", "Inspection score by facility type.", "")
_r("insptr", "inspection_trend", "OTIS", "Inspection score trend over fiscal years.", "")
_r("irtcl", "irt_calibrate", "Psymet", "IRT calibration pipeline (JMLE for 1PL/2PL).", "")
_r("irtdl", "irt_difficulty", "Psymet", "Extract IRT difficulty parameters.", "")
_r("irtdp", "irt_distractor", "Psymet", "IRT distractor analysis.", "")
_r("irtds", "irt_discrimination", "Psymet", "Extract IRT discrimination parameters.", "")
_r("irteap", "irt_eap_theta", "Psymet", "EAP theta estimation.", "")
_r("irtli", "irt_likelihood", "Psymet", "IRT log-likelihood at given theta.", "")
_r("irtml", "irt_mle_theta", "Psymet", "MLE theta estimation.", "")
_r("irtoc", "irt_option_curves", "Psymet", "IRT option characteristic curves.", "")
_r("irtrm", "irt_rasch_residuals", "Psymet", "Rasch model residuals (standardized).", "")
_r("irtwd", "irt_wright_map", "Psymet", "Wright map data (item difficulty vs person ability).", "")
_r("itdif", "item_difficulty", "Psymet", "Item difficulty (classical).", "")
_r("itdsc", "item_discrimination_all", "Psymet", "Item discrimination for all items (corrected item-total r).", "")
_r("itent", "item_entropy", "Psymet", "Item response entropy.", "")
_r("itflr", "item_floor_ceiling", "Psymet", "Item floor and ceiling effects.", "")
_r("itopt", "item_option_freq", "Psymet", "Item response option frequencies.", "")
_r("itrel", "item_reliability_index", "Psymet", "Item reliability index.", "")
_r("itsel", "item_select", "Psymet", "Item selection — flag items for removal.", "")
_r("itskw", "item_skew_kurt", "Psymet", "Item skewness and kurtosis.", "")
_r("ittab", "item_table", "Psymet", "Full item analysis table.", "")
_r("itval", "item_validity_index", "Psymet", "Item validity index.", "")
_r("mi_cf", "mi_configural", "Psymet", "Configural invariance: fit CFA separately per group.", "")
_r("mi_mt", "mi_metric", "Psymet", "Metric (weak) invariance: constrain loadings equal across gr", "")
_r("mi_sc", "mi_scalar", "Psymet", "Scalar (strong) invariance: constrain loadings + intercepts.", "")
_r("mi_st", "mi_strict", "Psymet", "Strict invariance: constrain loadings + intercepts + residua", "")
_r("miage", "mi_by_age", "Psymet", "Full measurement invariance ladder by age group.", "")
_r("midif", "mi_delta_fit", "Psymet", "Delta-fit indices between measurement invariance levels.", "")
_r("miest", "mi_effect_size", "Psymet", "Effect sizes for measurement invariance (dMACS, signed area)", "")
_r("migen", "mi_by_gender", "Psymet", "Full measurement invariance ladder by gender.", "")
_r("milat", "mi_latent_means", "Psymet", "Latent mean differences between groups (requires scalar inva", "")
_r("misum", "mi_summary", "Psymet", "Summary table of invariance levels with pass/fail.", "")
_r("netbr", "network_bridge", "Psymet", "Bridge centrality between communities in a network.", "")
_r("netbt", "network_betweenness", "Psymet", "Node betweenness centrality for a network.", "")
_r("netcl", "network_closeness", "Psymet", "Node closeness centrality for a network.", "")
_r("netcm", "network_communities", "Psymet", "Community detection in a network.", "")
_r("netcp", "network_compare", "Psymet", "Network comparison test (global strength and structure).", "")
_r("netcr", "network_correlation", "Psymet", "Partial correlation network (regularized via pseudo-inverse)", "")
_r("netdn", "network_density", "Psymet", "Network density (proportion of non-zero edges).", "")
_r("netei", "network_expected_influence", "Psymet", "Expected influence (signed node strength) for a network.", "")
_r("netsb", "network_stability", "Psymet", "Bootstrap stability of network edge weights (CS coefficient)", "")
_r("netst", "network_strength", "Psymet", "Node strength centrality for a network.", "")
_r("oaipw", "otis_aipw", "OTIS", "AIPW doubly-robust estimator for OTIS correctional data.", "")
_r("oate1", "otis_ate_region", "OTIS", "Simple ATE by region (difference in means) for OTIS data.", "")
_r("oatt1", "otis_att_region", "OTIS", "ATT by region via IPW for OTIS data.", "")
_r("ocate", "otis_cate_risk", "OTIS", "CATE by risk score tercile for OTIS correctional data.", "")
_r("ochi2", "otis_chi2_test", "OTIS", "Chi-squared test of independence for OTIS correctional data.", "")
_r("ocomp", "otis_group_compare", "OTIS", "OTIS group comparison — t-test/ANOVA + effect size.", "")
_r("ocorr", "otis_correlation", "OTIS", "Correlation matrix for numeric columns in OTIS data.", "")
_r("ocros", "otis_crosstab", "OTIS", "OTIS cross-tabulation with chi-square test.", "")
_r("odesc", "otis_demographic_summary", "OTIS", "OTIS demographic summary — region x age x gender counts.", "")
_r("odid1", "otis_did_policy", "OTIS", "Difference-in-Differences for policy change in OTIS data.", "")
_r("odisp", "otis_disparity_index", "OTIS", "OTIS disparity index — max/min group mean ratio.", "")
_r("odm_a", "otis_demo_age", "OTIS", "Demographic profile per age group for OTIS correctional data", "")
_r("odm_g", "otis_demo_gender", "OTIS", "Demographic profile per gender for OTIS correctional data.", "")
_r("odm_r", "otis_demo_region", "OTIS", "Demographic profile per region for OTIS correctional data.", "")
_r("odm_y", "otis_demo_year", "OTIS", "Demographic profile per fiscal year for OTIS correctional da", "")
_r("odml1", "otis_dml_region", "OTIS", "DML ATE by region for OTIS correctional data.", "")
_r("odml2", "otis_dml_alert", "OTIS", "DML for alert treatment effect in OTIS correctional data.", "")
_r("odml3", "otis_dml_age", "OTIS", "DML ATE by age group for OTIS correctional data.", "")
_r("odml4", "otis_dml_gender", "OTIS", "DML ATE by gender for OTIS correctional data.", "")
_r("oeffn", "otis_effect_summary", "OTIS", "Summary of all effect sizes for OTIS correctional data.", "")
_r("ofreq", "otis_freq_table", "OTIS", "Frequency table with proportions for OTIS correctional data.", "")
_r("ogate", "otis_gate_age", "OTIS", "GATE by age group for OTIS correctional data.", "")
_r("ohet1", "otis_het_region", "OTIS", "Heterogeneous treatment effects test by region for OTIS data", "")
_r("ohist", "otis_histogram_data", "OTIS", "Histogram bin counts (no plotting) for OTIS correctional dat", "")
_r("oipw1", "otis_ipw_placement", "OTIS", "IPW for placement effect in OTIS correctional data.", "")
_r("oiv1", "otis_iv_distance", "OTIS", "IV estimation (2SLS) for OTIS correctional data.", "")
_r("omed1", "otis_mediation", "OTIS", "Mediation analysis for OTIS correctional data.", "")
_r("omiss", "otis_missing_report", "OTIS", "Missing data report for OTIS correctional data.", "")
_r("opair", "otis_pairwise_compare", "OTIS", "Pairwise group comparisons for OTIS correctional data.", "")
_r("oprop", "otis_proportions", "OTIS", "OTIS proportion test across groups.", "")
_r("oqntl", "otis_quantiles", "OTIS", "Quantiles for a numeric column in OTIS correctional data.", "")
_r("orank", "otis_rank_regions", "OTIS", "OTIS rank regions by metric.", "")
_r("orisk", "otis_risk_table", "OTIS", "Univariate risk factor table for OTIS correctional data.", "")
_r("osns1", "otis_sensitivity", "OTIS", "Rosenbaum sensitivity bounds for OTIS correctional data.", "")
_r("osumm", "otis_summary_table", "OTIS", "Full summary table for OTIS correctional data.", "")
_r("otabl", "otis_table1", "OTIS", "Table 1 (baseline characteristics by group) for OTIS data.", "")
_r("otrd", "otis_trend_summary", "OTIS", "OTIS trend summary — year-over-year change statistics.", "")
_r("rcd_a", "recidivism_by_age", "OTIS", "Recidivism rate by age group.", "")
_r("rcd_g", "recidivism_by_gender", "OTIS", "Recidivism rate by gender.", "")
_r("rcd_r", "recidivism_by_region", "OTIS", "Recidivism rate by region.", "")
_r("rcdcx", "recidivism_cox", "OTIS", "Cox proportional hazards for recidivism predictors.", "")
_r("rcdhz", "recidivism_hazard", "OTIS", "Hazard rate at each time point for recidivism.", "")
_r("rcdkm", "recidivism_km", "OTIS", "Kaplan-Meier survival curve for recidivism.", "")
_r("rcdpr", "recidivism_predictors", "OTIS", "Logistic regression for recidivism risk factors.", "")
_r("rcdrt", "recidivism_trend", "OTIS", "Recidivism rate trend over years.", "")
_r("rcdsm", "recidivism_rate", "OTIS", "Overall recidivism rate.", "")
_r("rcdtm", "recidivism_time", "OTIS", "Time-to-event summary for recidivism.", "")
_r("rskau", "risk_auc", "OTIS", "AUC for risk score discrimination.", "")
_r("rskbr", "risk_base_rate", "OTIS", "Base rate of outcome overall and by group.", "")
_r("rskcb", "risk_calibration", "OTIS", "Risk score calibration by decile.", "")
_r("rskcl", "risk_classify", "OTIS", "Classify individuals into risk levels.", "")
_r("rskdc", "risk_decile", "OTIS", "Outcome rate by risk score decile.", "")
_r("rskfr", "risk_fairness", "OTIS", "Risk score fairness analysis by group.", "")
_r("rskov", "risk_overlap", "OTIS", "Risk score distribution overlap between groups.", "")
_r("rskpr", "risk_profile", "OTIS", "Mean risk score profile by demographic subgroups.", "")
_r("rsktd", "risk_trend", "OTIS", "Mean risk score trend over years.", "")
_r("rskth", "risk_threshold", "OTIS", "Optimal risk classification threshold via Youden's J.", "")
_r("s_ea", "subscale_ea", "Psymet", "EA subscale reliability (alpha, omega, CR, AVE).", "")
_r("s_ee", "subscale_ee", "Psymet", "EE subscale reliability (alpha, omega, CR, AVE).", "")
_r("s_er", "subscale_er", "Psymet", "ER subscale reliability (alpha, omega, CR, AVE).", "")
_r("s_ua", "subscale_ua", "Psymet", "UA subscale reliability (alpha, omega, CR, AVE).", "")
_r("sccut", "score_cutoffs", "Psymet", "Score cut-offs (tercile, quartile, clinical).", "")
_r("sceqv", "score_equate", "Psymet", "Score equating between forms.", "")
_r("scfct", "score_factor", "Psymet", "Factor scores (regression method).", "")
_r("scmn", "score_mean", "Psymet", "Mean score for each respondent.", "")
_r("scnrm", "score_norms", "Psymet", "Score norms — normative table from score distribution.", "")
_r("sconv", "subscale_convergent", "Psymet", "Convergent validity: AVE > 0.5 for each subscale.", "")
_r("scor", "subscale_correlations", "Psymet", "Inter-subscale correlation matrix.", "")
_r("scpmp", "score_pmp", "Psymet", "Percent of Maximum Possible (PMP) score.", "")
_r("screl", "score_reliability", "Psymet", "Score-level reliability summary.", "")
_r("scrng", "score_range_check", "Psymet", "Score range check — validate responses within bounds.", "")
_r("scstd", "score_standardize", "Psymet", "Score standardization (z, T-score, stanine, sten).", "")
_r("scsum", "score_sum", "Psymet", "Sum score for each respondent.", "")
_r("sdscr", "subscale_discriminant", "Psymet", "Discriminant validity: sqrt(AVE) > inter-subscale r.", "")
_r("sitdl", "subscale_item_detail", "Psymet", "Per-item detail within a subscale.", "")
_r("snorm", "subscale_norms", "Psymet", "Normative table (mean, sd, percentiles) per subscale.", "")
_r("sntag", "sentence_by_age", "OTIS", "Sentence length by age group.", "")
_r("sntdp", "sentence_disparity", "OTIS", "Sentence disparity between groups.", "")
_r("sntgn", "sentence_by_gender", "OTIS", "Sentence length by gender.", "")
_r("sntln", "sentence_length", "OTIS", "Sentence length distribution summary.", "")
_r("sntmd", "sentence_by_group", "OTIS", "Median sentence length by group.", "")
_r("sntpr", "sentence_percentiles", "OTIS", "Sentence length percentiles.", "")
_r("sntrg", "sentence_by_region", "OTIS", "Sentence length by region.", "")
_r("sntrl", "sentence_by_year", "OTIS", "Sentence length trends over years.", "")
_r("sntsr", "sentence_served", "OTIS", "Proportion of sentence served.", "")
_r("sntvl", "sentence_volatility", "OTIS", "Sentence length volatility across placements per individual.", "")
_r("sttot", "subscale_total_corr", "Psymet", "Correlation of each subscale score with total score.", "")
_r("vcnvg", "validity_convergent", "Psymet", "Convergent validity: AVE > 0.5 per factor (Fornell & Larcker", "")
_r("vcrit", "validity_criterion", "Psymet", "Criterion validity: correlation with external criterion.", "")
_r("vdscr", "validity_discriminant", "Psymet", "Discriminant validity: Fornell-Larcker criterion.", "")
_r("vface", "validity_face_content", "Psymet", "Content validity index (CVI) from expert ratings.", "")
_r("vhtmt", "validity_htmt", "Psymet", "Heterotrait-Monotrait ratio for discriminant validity.", "")
_r("vinc", "validity_incremental", "Psymet", "Incremental validity: delta-R-squared when adding a new pred", "")
_r("vknwn", "validity_known_groups", "Psymet", "Known-groups validity: do scores differ across expected grou", "")
_r("vmtmm", "validity_mtmm", "Psymet", "Multitrait-Multimethod (MTMM) matrix analysis.", "")
_r("vpred", "validity_predictive", "Psymet", "Predictive validity: AUC or R-squared for outcome prediction", "")
_r("vtest", "validity_test_retest", "Psymet", "Test-retest reliability: ICC and Pearson r.", "")

# ── Epidemiology (Wave 1) ─────────────────────────────────────────────────
_r("sird", "sir_model", "Epidemiology", "SIR compartmental model", "What is now proved was once only imagined. — William Blake")
_r(
    "seir",
    "seir_model",
    "Epidemiology",
    "SEIR model with exposed compartment",
    "The rot has spread too far. -- Luthen Rael (Andor)",
)
_r(
    "r0",
    "basic_reproduction_number",
    "Epidemiology",
    "Basic reproduction number R0",
    "It's spreading to each system. -- Bail Organa",
)
_r(
    "herd",
    "herd_immunity_threshold",
    "Epidemiology",
    "Herd immunity threshold 1-1/R0",
    "We protect each other. -- Hera Syndulla (Rebels)",
)
_r("cfr", "case_fatality_rate", "Epidemiology", "Case fatality rate with Wilson CI", "Many Bothans died. -- Mon Mothma")
_r(
    "asr",
    "age_standardized_rate",
    "Epidemiology",
    "Direct age-standardized rate",
    "When 900 years old you reach...",
)
_r(
    "le",
    "life_expectancy",
    "Epidemiology",
    "Life expectancy from life table",
    "The years ahead are long but worth living. -- Kanan Jarrus",
)
_r(
    "ypll",
    "years_potential_life_lost",
    "Epidemiology",
    "Years of potential life lost",
    "Cut down before their prime. -- Depa Billaba",
)
_r(
    "dar",
    "direct_age_adjustment",
    "Epidemiology",
    "Direct age-adjustment rate",
    "We need a standard to compare. -- Admiral Raddus",
)
_r(
    "iar",
    "indirect_age_adjustment",
    "Epidemiology",
    "Indirect age-adjustment (SMR)",
    "There are other ways. -- Qui-Gon Jinn",
)
_r("epi_c", "epidemic_curve", "Epidemiology", "Epidemic curve bin counts", "I can feel it building. -- Rey")
_r(
    "r_eff",
    "effective_rt",
    "Epidemiology",
    "Effective reproduction number Rt",
    "The transmission rate is changing. -- Nala Se",
)
_r(
    "srvey",
    "survey_prevalence",
    "Epidemiology",
    "Survey prevalence with design effect",
    "We need a count of every system. -- Mas Amedda",
)
_r(
    "pyr",
    "person_years_at_risk",
    "Epidemiology",
    "Person-years at risk calculation",
    "Every moment in the field counts. -- Captain Rex",
)
_r(
    "apc",
    "age_period_cohort",
    "Epidemiology",
    "Age-period-cohort decomposition",
    "Generation after generation, the pattern repeats. -- Grand Inquisitor",
)

# ── Spatial Statistics ─────────────────────────────────────────────────────
_r(
    "moran",
    "morans_i",
    "Spatial",
    "Moran's I global spatial autocorrelation",
    "Distribution helper.",
)
_r(
    "geary",
    "gearys_c",
    "Spatial",
    "Geary's C spatial autocorrelation",
    "What is near is not always what is far. -- Bendu",
)
_r("lisa", "local_morans_i", "Spatial", "Local Moran's I (LISA)", "In every corner, a different story. -- Sabine Wren")
_r(
    "gstat",
    "semivariogram",
    "Spatial",
    "Empirical semivariogram",
    "The further you go, the more different things become. -- Ezra Bridger",
)
_r(
    "krig",
    "ordinary_kriging",
    "Spatial",
    "Ordinary kriging interpolation",
    "Predict what lies in uncharted regions. -- Grand Admiral Thrawn",
)
_r(
    "idw",
    "inverse_distance_weighting",
    "Spatial",
    "IDW spatial interpolation",
    "The closer you are, the stronger the pull. -- Kylo Ren",
)
_r(
    "knear",
    "k_nearest_spatial",
    "Spatial",
    "K-nearest spatial weights matrix",
    "Keep your friends close. -- Lando Calrissian",
)
_r(
    "qween",
    "queen_contiguity",
    "Spatial",
    "Queen contiguity weight matrix",
    "A queen's influence extends in every direction. -- Padme Amidala",
)
_r(
    "getis",
    "getis_ord_gi",
    "Spatial",
    "Getis-Ord Gi* hot spot analysis",
    "There's always a hotspot on the map. -- Poe Dameron",
)
_r(
    "skerr",
    "spatial_error_model",
    "Spatial",
    "Spatial error model (SEM)",
    "Errors ripple across the galaxy. -- Galen Erso",
)
_r("nskov", "nskov", "Spatial", "Non-stationary kernel covariance estimation", "Distribution helper.")
_r("dkern", "dkern", "Spatial", "Deformation kernel / space warping", "Distribution helper.")
_r("procn", "procn", "Spatial", "Process convolution spatial model", "Smooth the path and the way becomes clear. -- Chirrut Imwe")
_r("mvkov", "mvkov", "Spatial", "Moving window variogram", "Every window shows a different truth. -- Mon Mothma")
_r("lclvg", "lclvg", "Spatial", "Local variogram estimation", "What is now proved was once only imagined. — William Blake")
_r("anvgm", "anvgm", "Spatial", "Anisotropic variogram model", "Direction matters in any conflict. -- Admiral Ackbar")
_r("spdef", "spdef", "Spatial", "Spatial deformation model via MDS", "Warp the map and the territory reveals itself. -- Grand Admiral Thrawn")
_r("nscov", "nscov", "Spatial", "Non-stationary covariance function", "Nothing stays the same across the galaxy. -- Hera Syndulla")
_r("knspl", "knspl", "Spatial", "Kernel non-stationary spatial prediction", "Predict the unknown from the known.")
_r("wdvar", "wdvar", "Spatial", "Windowed variogram cloud", "In the cloud of data, patterns emerge. -- Jyn Erso")
_r("starn", "starn", "SpatioTemporal", "Spatio-temporal autoregressive (STAR) model", "The past echoes through space and time. -- Ezra Bridger")
_r("stdyn", "stdyn", "SpatioTemporal", "Dynamic spatio-temporal state-space model", "The state of things is always in motion.")
_r("stide", "stide", "SpatioTemporal", "Spatio-temporal integro-difference equation", "Waves propagate across the galaxy. -- Saw Gerrera")
_r("stknl", "stknl", "SpatioTemporal", "Spatio-temporal kernel smoothing", "Smooth the ripples of time and space. -- Kanan Jarrus")
_r("stsep", "stsep", "SpatioTemporal", "Separable spatio-temporal covariance", "Space and time can be separated, but remain connected. -- Bendu")
_r("stprd", "stprd", "SpatioTemporal", "Spatio-temporal prediction intervals", "The future is always in motion.")
_r("sttrn", "sttrn", "SpatioTemporal", "Spatio-temporal trend surface", "Every surface tells the story of its making. -- Sabine Wren")
_r("ukrig", "ukrig", "Spatial", "Universal kriging with polynomial trend", "A universal solution adapts to every terrain. -- Commander Cody")
_r("cokrg", "cokrg", "Spatial", "Co-kriging multivariate spatial prediction", "Together we are stronger than apart. -- Padme Amidala")
_r("indkr", "indkr", "Spatial", "Indicator kriging for exceedance probability", "What is now proved was once only imagined. — William Blake")
_r("blkrg", "blkrg", "Spatial", "Block kriging for areal prediction", "See the whole region, not just the point. -- Bail Organa")
_r("xvgm", "xvgm", "Spatial", "Cross-variogram estimation", "Between two signals lies the truth. -- Cassian Andor")
_r("spglm", "spglm", "Spatial", "Spatial GLM with spatial errors", "Generalise the model, specialise the errors. -- Galen Erso")
_r("spgee", "spgee", "Spatial", "Spatial GEE with sandwich variance", "Robust estimates survive any storm. -- Admiral Raddus")
_r("spmxd", "spmxd", "Spatial", "Spatial linear mixed model", "Fixed and random, both matter. -- Mace Windu")
_r("spper", "spper", "Spatial", "Spatial periodogram (spectral density)", "Every signal has its frequency. -- Hondo Ohnaka")

# ── Genomics / Bioinformatics ─────────────────────────────────────────────
_r(
    "gc",
    "gc_content_calc",
    "Genomics",
    "GC content of DNA sequence",
    "Midi-chlorians are the building blocks. -- Qui-Gon",
)
_r(
    "hw",
    "hardy_weinberg_test",
    "Genomics",
    "Hardy-Weinberg equilibrium test",
    "Distribution helper.",
)
_r(
    "fst",
    "fixation_index",
    "Genomics",
    "Fst fixation index (Weir-Cockerham)",
    "We are one people, divided. -- Duchess Satine",
)
_r("tajd", "tajimas_d", "Genomics", "Tajima's D neutrality test", "Neutral the clones were meant to be. -- Lama Su")
_r(
    "ld",
    "linkage_disequilibrium",
    "Genomics",
    "LD r-squared between loci",
    "Some bonds cannot be broken. -- Ahsoka Tano",
)
_r(
    "maf",
    "minor_allele_frequency",
    "Genomics",
    "Minor allele frequency",
    "Even a small frequency can change the galaxy. -- Chirrut Imwe",
)
_r(
    "bonf",
    "bonferroni_correction",
    "Genomics",
    "Bonferroni multiple testing correction",
    "You must be very strict, Captain. -- Grand Moff Tarkin",
)
_r(
    "fdr",
    "benjamini_hochberg",
    "Genomics",
    "Benjamini-Hochberg FDR correction",
    "Control what you can control. -- Cassian Andor",
)
_r(
    "nuc",
    "nucleotide_freq",
    "Genomics",
    "Nucleotide frequency distribution",
    "The very atoms that form us. -- Chirrut Imwe",
)
_r(
    "codon",
    "codon_usage",
    "Genomics",
    "Codon usage table from DNA sequence",
    "The code, we must decode it. -- Mace Windu",
)
_r(
    "tm",
    "melting_temperature",
    "Genomics",
    "DNA melting temperature (nearest-neighbor)",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "pi_",
    "nucleotide_diversity",
    "Genomics",
    "Nucleotide diversity (pi)",
    "Diversity is our strength. -- Padme Amidala",
)
_r(
    "theta",
    "watterson_theta",
    "Genomics",
    "Watterson's theta estimator",
    "What is now proved was once only imagined. — William Blake",
)
_r("pca_g", "pca_genotype", "Genomics", "PCA on genotype matrix", "See the big picture you must.")
_r(
    "admix",
    "admixture_proportions",
    "Genomics",
    "Admixture estimation (K pops)",
    "What is now proved was once only imagined. — William Blake",
)
_r("ibs", "identity_by_state", "Genomics", "IBS similarity matrix", "We are the same, you and I. -- Maul")
_r("grm", "genetic_relatedness", "Genomics", "Genomic relationship matrix", "What is now proved was once only imagined. — William Blake")
_r(
    "gwas1",
    "gwas_single_snp",
    "Genomics",
    "Single-SNP GWAS association test",
    "One small thing can change everything. -- Jyn Erso",
)
_r("qq_", "qq_plot_data", "Genomics", "QQ-plot data for GWAS p-values", "Expect the unexpected. -- Ahsoka Tano")
_r("manht", "manhattan_data", "Genomics", "Manhattan plot data preparation", "What is now proved was once only imagined. — William Blake")
_r(
    "lambd",
    "genomic_inflation",
    "Genomics",
    "Genomic inflation factor lambda",
    "Inflation is the path to the dark side. -- Tarkin",
)
_r("clump", "ld_clumping", "Genomics", "LD-based clumping of significant SNPs", "Stick together. -- Hera Syndulla")
_r(
    "prs",
    "polygenic_risk_score",
    "Genomics",
    "Polygenic risk score calculation",
    "What is now proved was once only imagined. — William Blake",
)

# ── Criminology ────────────────────────────────────────────────────────────
_r(
    "crime",
    "crime_rate",
    "Criminology",
    "Crime rate per 100K with Wilson CI",
    "These are not the droids you are looking for. --",
)
_r(
    "recid",
    "recidivism_rate",
    "Criminology",
    "General recidivism rate",
    "There is always a bigger fish. -- Qui-Gon Jinn",
)
_r("ucr", "ucr_classify", "Criminology", "UCR offense classification", "What is now proved was once only imagined. — William Blake")
_r("vctm", "victimization_rate", "Criminology", "Victimization survey rate", "Stay on target. -- Gold Five")
_r("cjs", "cjs_flow", "Criminology", "Criminal justice system flow", "Distribution helper.")
_r("sent", "sentence_stats", "Criminology", "Sentence length distribution stats", "Who's the more foolish? --")
_r("hotsp", "hot_spots", "Criminology", "Repeat address hot spot analysis", "It's a trap! -- Admiral Ackbar")
_r(
    "disp",
    "disparity_index",
    "Criminology",
    "Racial/group disparity index",
    "Your focus determines your reality. -- Qui-Gon",
)

# ── Web Scraping ───────────────────────────────────────────────────────────
_r("wscrp", "web_scrape", "WebScraping", "Web page fetch + text extraction", "What is now proved was once only imagined. — William Blake")
_r(
    "siubc",
    "siu_scrape_report",
    "WebScraping",
    "SIU Ontario report HTML parser",
    "In my experience there is no such thing as luck. --",
)
_r("tblxt", "extract_tables", "WebScraping", "HTML table extractor to DataFrame", "What is now proved was once only imagined. — William Blake")

# ── Star Wars DB ───────────────────────────────────────────────────────────
_r("swdb", "load_sw_dataset", "StarWarsDB", "Star Wars dataset loader", "Do. Or do not. There is no try.")
_r(
    "swchr",
    "sw_character_summary",
    "StarWarsDB",
    "Star Wars character stats",
    "Distribution helper.",
)
_r("swplt", "sw_planet_summary", "StarWarsDB", "Star Wars planet stats", "What is now proved was once only imagined. — William Blake")
_r("swflm", "sw_film_summary", "StarWarsDB", "Star Wars film metadata", "What is now proved was once only imagined. — William Blake")

# ── Machine Learning ───────────────────────────────────────────────────────
_r("gam", "fit_gam", "ML", "Generalized additive model (B-spline + OLS)", "Size matters not.")
_r("lowes", "lowess_smooth", "ML", "LOWESS smoother", "Luminous beings are we.")
_r("kern", "kde", "ML", "Gaussian kernel density estimation", "Distribution helper.")
_r("kreg", "kernel_regression", "ML", "Nadaraya-Watson kernel regression", "I have the high ground. --")
_r(
    "spln",
    "spline_regression",
    "ML",
    "Cubic spline regression",
    "An elegant weapon for a more civilized age. --",
)
_r("isof", "isolation_forest", "ML", "Isolation forest anomaly detection", "Fear leads to anger.")
_r("nb_", "naive_bayes", "ML", "Gaussian Naive Bayes classifier", "Too accurate for Sand People. --")
_r("ada", "adaboost", "ML", "AdaBoost with decision stumps", "What is now proved was once only imagined. — William Blake")
_r("stck", "stacking", "ML", "Model stacking / super learner", "We are what they grow beyond.")
_r(
    "bgg",
    "bagging",
    "ML",
    "Bootstrap aggregating (bagging)",
    "In time, the suffering of your people will persuade you. -- Gunray",
)

# ── Image / Viewer ─────────────────────────────────────────────────────────
_r("iview", "view_image", "Image", "Image viewer and info helper", "The garbage'll do! -- Rey")
_r("savfg", "save_figure", "Image", "Save matplotlib figure to file", "What is now proved was once only imagined. — William Blake")
_r("holo_g", "geo_summary", "Image", "Geographic / choropleth summary", "What is now proved was once only imagined. — William Blake")

# ── Infectious Disease Models ──────────────────────────────────────────────
_r("sis", "sis_model", "Epidemiology", "SIS compartmental model (no immunity)", "This is the way. -- The Mandalorian")
_r(
    "sirs",
    "sirs_model",
    "Epidemiology",
    "SIRS model (waning immunity)",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "sir_v",
    "sir_vaccination",
    "Epidemiology",
    "SIR with vaccination rate",
    "Hope is not lost today. It is found. -- Poe Dameron",
)
_r(
    "sir_d",
    "sir_demography",
    "Epidemiology",
    "SIR with birth/death dynamics",
    "There's always a bigger fish. -- Qui-Gon",
)
_r("r_t", "realtime_rt", "Epidemiology", "Real-time Rt (Bayesian method)", "Always in motion is the future.")
_r("si", "serial_interval", "Epidemiology", "Serial interval estimation", "One follows another. -- Dooku")
_r("gtime", "generation_time", "Epidemiology", "Generation time distribution", "A new generation arises. -- Snoke")
_r(
    "attkr",
    "attack_rate",
    "Epidemiology",
    "Attack rate from outbreak data",
    "The attack must come swiftly. -- Admiral Trench",
)
_r(
    "sar",
    "secondary_attack",
    "Epidemiology",
    "Secondary attack rate",
    "Where there is one, there are many. -- Cad Bane",
)
_r("doubl", "doubling_time", "Epidemiology", "Epidemic doubling time", "Twice the pride, double the fall. -- Dooku")

# ── Surveillance & Screening ──────────────────────────────────────────────
_r("cusum", "cusum_detect", "Surveillance", "CUSUM control chart for outbreaks", "What is now proved was once only imagined. — William Blake")
_r("ewma", "ewma_detect", "Surveillance", "EWMA control chart", "The signal is getting stronger. -- Hera Syndulla")
_r("farr", "farrington_detect", "Surveillance", "Farrington aberration detection", "Something is not right. -- Padme")
_r(
    "syndm",
    "syndromic_score",
    "Surveillance",
    "Syndromic surveillance composite",
    "The symptoms are clear. -- Medical Droid",
)
_r(
    "scrn",
    "screening_metrics",
    "Surveillance",
    "Screening sensitivity/specificity/PPV",
    "We must screen them all. -- Nala Se",
)
_r("roc", "roc_curve", "Surveillance", "ROC curve data", "Choose wisely.")
_r("prc", "pr_curve", "Surveillance", "Precision-recall curve", "Precision is key. -- Grand Admiral Thrawn")
_r("youdn", "youden_index", "Surveillance", "Youden's J optimal threshold", "Find the balance point. -- Qui-Gon Jinn")
_r("dor", "diagnostic_or", "Surveillance", "Diagnostic odds ratio", "The odds are in our favor. -- Lando")
_r("lrp", "likelihood_ratios", "Surveillance", "Positive/negative likelihood ratios", "How likely is that? -- C-3PO")

# ── Environmental Epidemiology ─────────────────────────────────────────────
_r(
    "expos",
    "exposure_assessment",
    "EnvEpi",
    "Exposure distribution analysis",
    "The air itself is poison. -- Saw Gerrera",
)
_r(
    "drcur",
    "dose_response_curve",
    "EnvEpi",
    "Dose-response modeling (log-logistic)",
    "What is now proved was once only imagined. — William Blake",
)
_r("bench", "benchmark_dose", "EnvEpi", "BMD/BMDL calculation", "A benchmark we must establish.")
_r(
    "attfr",
    "attributable_fraction",
    "EnvEpi",
    "Population attributable fraction",
    "We are all responsible. -- Organa",
)
_r("etiof", "etiologic_fraction", "EnvEpi", "Etiologic fraction among exposed", "The cause runs deeper. --")
_r("exrsk", "excess_risk", "EnvEpi", "Excess absolute risk", "The risk is too great. -- Windu")
_r(
    "tlag",
    "time_lag_analysis",
    "EnvEpi",
    "Cross-correlation time lag",
    "Time is the one thing we cannot control. -- Hera",
)
_r(
    "htidx",
    "heat_index",
    "EnvEpi",
    "Heat index from temp and humidity",
    "It's not the heat, it's the humidity. -- Watto",
)

# ── Clinical Trials ────────────────────────────────────────────────────────
_r(
    "rndm",
    "randomization",
    "ClinicalTrial",
    "Block randomization sequence",
    "Leave nothing to chance. -- Grand Moff Tarkin",
)
_r(
    "strat",
    "stratified_randomize",
    "ClinicalTrial",
    "Stratified randomization",
    "What is now proved was once only imagined. — William Blake",
)
_r("ssiz", "sample_size_means", "ClinicalTrial", "Sample size for 2 means", "We need more troops. -- Captain Rex")
_r(
    "ssip",
    "sample_size_proportions",
    "ClinicalTrial",
    "Sample size for 2 proportions",
    "Strength in numbers. -- Bail Organa",
)
_r(
    "ssie",
    "sample_size_equivalence",
    "ClinicalTrial",
    "Equivalence trial sample size",
    "Equivalent they are not.",
)
_r(
    "ssin",
    "sample_size_noninferiority",
    "ClinicalTrial",
    "Non-inferiority sample size",
    "No worse than before. -- Admiral Ackbar",
)
_r(
    "itrm",
    "interim_analysis",
    "ClinicalTrial",
    "O'Brien-Fleming interim boundaries",
    "What is now proved was once only imagined. — William Blake",
)
_r("futl", "futility_boundary", "ClinicalTrial", "Futility stopping boundary", "What is now proved was once only imagined. — William Blake")
_r("adptv", "adaptive_design", "ClinicalTrial", "Adaptive sample size re-estimation", "We must adapt. -- Ahsoka")
_r("cross", "crossover_analysis", "ClinicalTrial", "2x2 crossover trial analysis", "Switch sides, they did.")

# ── Survival Extended ──────────────────────────────────────────────────────
_r(
    "cmprs",
    "competing_risks",
    "Survival",
    "Cause-specific hazard (competing risks)",
    "Many ways to fall there are.",
)

# ── Robust Statistics ──────────────────────────────────────────────────────
_r("huber", "huber_m_estimate", "Robust", "Huber M-estimator of location", "Steady in the storm. --")
_r("tukey", "tukey_biweight", "Robust", "Tukey biweight M-estimator", "Tough as durasteel. -- Mandalorian proverb")
_r("mcd", "min_covariance_det", "Robust", "Minimum covariance determinant", "Find the core of truth. -- Qui-Gon")
_r("lts", "least_trimmed_sq", "Robust", "Least trimmed squares regression", "Trim the outliers you must.")
_r("lms", "least_median_sq", "Robust", "Least median of squares regression", "The median path is wisest. --")
_r("mad_", "median_abs_deviation", "Robust", "Median absolute deviation", "Measure from the center. -- Chirrut")
_r("iqr_", "iqr_statistic", "Robust", "Interquartile range", "The middle ground. -- Bendu")
_r("winsz", "winsorize", "Robust", "Winsorized mean", "Contain the extremes. -- Admiral Holdo")
_r("trimm", "trimmed_mean", "Robust", "Trimmed mean", "Cut away the excess. -- Dooku")
_r("sn_", "sn_estimator", "Robust", "Rousseeuw-Croux Sn scale estimator", "A more robust measure. -- Mon Mothma")

# ── Spatial Extended ───────────────────────────────────────────────────────
_r("splag", "spatial_lag", "Spatial", "Spatial lag model (SLM)", "Each neighbor pulls the other. -- Depa Billaba")
_r("spdur", "spatial_durbin", "Spatial", "Spatial Durbin model", "Consider both the near and far. -- Thrawn")
_r(
    "sphet",
    "spatial_heterogeneity",
    "Spatial",
    "Spatial heterogeneity test",
    "Not all places are the same. -- Ezra Bridger",
)
_r("spflt", "spatial_filter", "Spatial", "Eigenvector spatial filtering", "Filter the noise from the signal. -- K-2SO")
_r(
    "spreg",
    "spatial_regime",
    "Spatial",
    "Spatial regimes (Chow test)",
    "Different rules for different regions. -- Mon Mothma",
)
_r(
    "pproc",
    "point_process_intensity",
    "Spatial",
    "Spatial point process intensity",
    "Where do they cluster? -- Commander Cody",
)
_r("kfunc", "ripley_k", "Spatial", "Ripley's K function", "The clustering is strong here. -- Mace Windu")
_r("lfunc", "ripley_l", "Spatial", "Ripley's L function", "Normalize the pattern. -- Luminara Unduli")

# ── Substance Use Epidemiology ─────────────────────────────────────────────
_r(
    "suprv",
    "substance_prevalence",
    "SubstanceUse",
    "Substance use prevalence with CI",
    "A dangerous substance this is.",
)
_r("sutrn", "substance_trend", "SubstanceUse", "Substance use trend over time", "The path grows darker. -- Mace Windu")
_r(
    "suage",
    "substance_by_age",
    "SubstanceUse",
    "Age-specific substance use rates",
    "When young you are, vulnerable you are.",
)
_r(
    "sugen",
    "substance_by_gender",
    "SubstanceUse",
    "Gender-specific substance use rates",
    "We are not so different. -- Jyn Erso",
)
_r("supol", "polysubstance", "SubstanceUse", "Polysubstance co-occurrence matrix", "What is now proved was once only imagined. — William Blake")
_r("suinit", "initiation_age", "SubstanceUse", "Age of first use analysis", "Too young to face this. -- Depa Billaba")
_r("sudur", "substance_duration", "SubstanceUse", "Duration of use analysis", "Time takes its toll. -- Dooku")
_r("suaud", "audit_score", "SubstanceUse", "AUDIT alcohol screening score", "Measure the damage. -- Medical Droid")
_r("sudast", "dast_score", "SubstanceUse", "DAST-10 drug screening score", "Screen them all we must. -- Nala Se")
_r("suebac", "ebac_dist", "SubstanceUse", "eBAC distribution analysis", "Over the limit, many are.")
_r(
    "suhdnk",
    "heavy_drinking",
    "SubstanceUse",
    "Heavy/binge drinking prevalence",
    "Dangerous and reckless. -- Mace Windu",
)
_r(
    "sucost",
    "substance_cost",
    "SubstanceUse",
    "Societal cost of substance use",
    "The cost is more than credits. -- Bail Organa",
)
_r(
    "susngl",
    "single_use_risk",
    "SubstanceUse",
    "Single-occasion acute harm risk",
    "One night can change everything. -- Jyn Erso",
)
_r(
    "sumort",
    "substance_mortality",
    "SubstanceUse",
    "Substance-attributable mortality",
    "Many lives lost. -- Mon Mothma",
)
_r("sudaly", "substance_daly", "SubstanceUse", "DALYs from substance use", "The burden is immeasurable. -- Padme")

# ── Mental Health Epidemiology ─────────────────────────────────────────────
_r(
    "mhprv",
    "mental_health_prevalence",
    "MentalHealth",
    "Mental disorder prevalence with CI",
    "Distribution helper.",
)
_r("mhphq", "phq9_score", "MentalHealth", "PHQ-9 depression screening score", "A great sadness I sense.")
_r("mhgad", "gad7_score", "MentalHealth", "GAD-7 anxiety screening score", "Fear is the path to the dark side.")
_r("mhk10", "k10_score", "MentalHealth", "Kessler K10 distress score", "Distribution helper.")
_r("mhsfr", "sf12_mental", "MentalHealth", "SF-12 mental component summary", "What is now proved was once only imagined. — William Blake")
_r(
    "mhcom",
    "comorbidity_index",
    "MentalHealth",
    "Mental health comorbidity count",
    "One affliction leads to another. -- Dooku",
)
_r("mhtrn", "mental_health_trend", "MentalHealth", "MH prevalence trend over time", "The darkness is rising. -- Snoke")
_r("mhsrv", "service_utilization", "MentalHealth", "MH service utilization rate", "Help them we must.")
_r("mhwat", "wait_time_analysis", "MentalHealth", "Wait time for MH services", "Patience, young one. --")
_r("mhstg", "stigma_index", "MentalHealth", "Stigma composite score", "Judge me by my size, do you?")

# ── Chronic Disease Epidemiology ───────────────────────────────────────────
_r(
    "cdprv",
    "chronic_disease_prevalence",
    "ChronicDisease",
    "Age-adjusted chronic disease prevalence",
    "The long battle. -- Captain Rex",
)
_r("cdinc", "incidence_rate", "ChronicDisease", "Person-time incidence rate", "New cases arise. -- Nala Se")
_r(
    "cdcum",
    "cumulative_incidence",
    "ChronicDisease",
    "Cumulative incidence (risk)",
    "It builds over time. -- Saw Gerrera",
)
_r(
    "cdmrt",
    "cause_specific_mortality",
    "ChronicDisease",
    "Cause-specific mortality rate",
    "Each cause has a cost. -- Bail Organa",
)
_r(
    "cdpmr",
    "proportionate_mortality",
    "ChronicDisease",
    "Proportionate mortality ratio",
    "What share of death? -- Chirrut Imwe",
)
_r("cdsmr", "standardized_mortality_ratio", "ChronicDisease", "SMR with Poisson CI", "Observe and compare. -- Thrawn")
_r("cdyld", "years_lived_disability", "ChronicDisease", "YLD calculation", "Living, but suffering. -- Anakin")
_r("cdyll", "years_life_lost", "ChronicDisease", "YLL calculation", "Cut down too soon. -- Depa Billaba")
_r("cddaly", "daly_calc", "ChronicDisease", "DALY = YLL + YLD", "The full burden revealed.")
_r(
    "cdccm",
    "charlson_comorbidity",
    "ChronicDisease",
    "Charlson comorbidity index",
    "Many conditions, one patient. -- Medical Droid",
)

# ── Health Economics ───────────────────────────────────────────────────────
_r("heqly", "quality_adjusted_ly", "HealthEcon", "QALY calculation", "Quality over quantity. -- Qui-Gon")
_r("heicer", "incremental_cer", "HealthEcon", "ICER cost-effectiveness ratio", "What price for health? -- Padme")
_r("henb", "net_monetary_benefit", "HealthEcon", "Net monetary benefit", "Count the credits. -- Hondo Ohnaka")
_r(
    "hecea",
    "cost_effectiveness_plane",
    "HealthEcon",
    "CE plane quadrant probabilities",
    "Four quadrants of the galaxy. -- Admiral Ackbar",
)
_r("heceac", "ceac", "HealthEcon", "Cost-effectiveness acceptability curve", "At what price would you accept? -- Watto")
_r("hepsa", "probabilistic_sensitivity", "HealthEcon", "PSA for health economics", "Uncertainty in all things.")
_r(
    "hedsa",
    "deterministic_sensitivity",
    "HealthEcon",
    "One-way sensitivity analysis",
    "Change one thing at a time. --",
)
_r(
    "hedsc",
    "discount_rate",
    "HealthEcon",
    "Discount future costs/effects",
    "The future is always less certain. -- Qui-Gon",
)
_r("heboi", "burden_of_illness", "HealthEcon", "Total burden of illness", "The weight of suffering. -- Padme")
_r("hewtp", "willingness_to_pay", "HealthEcon", "WTP threshold analysis", "How much would you pay? -- Watto")

# ── Indigenous Health ──────────────────────────────────────────────────────
_r(
    "ihgap",
    "health_gap",
    "IndigenousHealth",
    "Health gap Indigenous vs general",
    "The gap must be closed. -- Bail Organa",
)
_r(
    "ihtrn",
    "indigenous_health_trend",
    "IndigenousHealth",
    "Indigenous health trend",
    "Progress is slow but steady. -- Mon Mothma",
)
_r("ihmrt", "indigenous_mortality", "IndigenousHealth", "Excess mortality ratio", "Too many lives lost. -- Padme")
_r(
    "ihsrv",
    "indigenous_service_access",
    "IndigenousHealth",
    "Service access disparity",
    "Access for all, there must be.",
)
_r(
    "ihsoc",
    "social_determinants",
    "IndigenousHealth",
    "Social determinants composite",
    "The roots run deep. -- Chirrut Imwe",
)

# ── Environmental / Occupational Health ────────────────────────────────────
_r("occrt", "occupational_injury_rate", "OccHealth", "Injury rate per 100 FTE", "Safety first. -- Captain Rex")
_r(
    "occex",
    "occupational_exposure",
    "OccHealth",
    "Exposure assessment (TWA vs OEL)",
    "The air you breathe matters. -- Saw Gerrera",
)
_r(
    "aqidx",
    "air_quality_index",
    "OccHealth",
    "AQI from pollutant concentrations",
    "Breathe. Just breathe. -- Maz Kanata",
)
_r("wqidx", "water_quality_index", "OccHealth", "Water quality index", "Clean water is life. -- Baze Malbus")
_r("noidx", "noise_exposure", "OccHealth", "Noise exposure TWA assessment", "Listen carefully. -- Qui-Gon Jinn")

# ── Demographic Methods ───────────────────────────────────────────────────
_r("fertl", "fertility_rate", "Demography", "Total fertility rate from ASFR", "A new generation. -- Organa")
_r("imort", "infant_mortality", "Demography", "Infant mortality rate", "Protect the young we must.")
_r(
    "mmort",
    "maternal_mortality",
    "Demography",
    "Maternal mortality ratio",
    "Mothers are the backbone of the Republic. -- Padme",
)
_r(
    "ltabl",
    "life_table_full",
    "Demography",
    "Full abridged life table",
    "From birth to death, the table tells all.",
)
_r("poppyr", "population_pyramid", "Demography", "Population pyramid data", "See the shape of the galaxy. -- Thrawn")
_r("deprt", "dependency_ratio", "Demography", "Age dependency ratio", "The young and old depend on us. -- Bail Organa")
_r("grwrt", "population_growth_rate", "Demography", "Population growth rate", "Growing, the population is.")
_r(
    "dblrt",
    "population_doubling",
    "Demography",
    "Population doubling time",
    "Twice the pride, double the fall. -- Dooku",
)
_r("migrt", "net_migration_rate", "Demography", "Net migration rate", "People on the move. -- Hera Syndulla")
_r(
    "lexis",
    "lexis_diagram_data",
    "Demography",
    "Lexis diagram data preparation",
    "Time and age, intertwined. -- Chirrut",
)

# ── Global Burden of Disease ──────────────────────────────────────────────
_r("gbdcm", "gbd_compare", "GBD", "Compare burden across conditions", "Compare the suffering. -- Medical Droid")
_r("gbdrf", "gbd_risk_factor", "GBD", "Risk factor attribution (PAF)", "The cause behind the cause. -- Qui-Gon")
_r("gbdpr", "gbd_projection", "GBD", "Project future disease burden", "Always in motion is the future.")
_r("gbdag", "gbd_age_pattern", "GBD", "Age pattern of disease burden", "Age brings wisdom and burden.")
_r("gbdsb", "gbd_subgroup", "GBD", "Burden by subgroup", "Every group bears its share. -- Mon Mothma")

# ── Vaccine / Immunization ────────────────────────────────────────────────
_r("vacef", "vaccine_efficacy", "Vaccine", "Vaccine efficacy from trial", "The shield against disease. -- Nala Se")
_r(
    "vacve",
    "vaccine_effectiveness",
    "Vaccine",
    "Real-world vaccine effectiveness",
    "In the field, results may differ. -- Captain Rex",
)
_r("vaccv", "vaccine_coverage", "Vaccine", "Vaccination coverage rate", "Cover every system. -- Mas Amedda")
_r("vacnnt", "vaccine_nnt", "Vaccine", "NNV to prevent one case", "How many must we reach? -- Bail Organa")
_r("vachrd", "vaccine_herd", "Vaccine", "Herd immunity threshold", "Together we are strong. -- Hera Syndulla")


# ── Auto-registered batch (Wave 2+3) ──────────────────────
_r("ahp", "ahp", "GameTheory", "Analytic Hierarchy Process (AHP) weights", "Same jacket. --")
_r("ansrb", "ansrb", "NonparametricTest", "Ansari-Bradley test for scale", "Distribution helper.")
_r("arima", "arima", "TimeSeries", "ARIMA(p,d,q) fitting via conditional MLE", "There is good in him. -- Luke")
_r("bbf", "bbf", "Bayesian", "Bayes factor (BIC approximation)", "Attachment leads to jealousy. -- Anakin")
_r(
    "bbin",
    "bbin",
    "Bayesian",
    "Beta-binomial conjugate Bayesian analysis",
    "What is now proved was once only imagined. — William Blake",
)
_r("bdiag", "bdiag", "Bayesian", "MCMC diagnostics (Rhat, ESS)", "Patience.")
_r(
    "bgibbs",
    "bgibbs",
    "Bayesian",
    "Gibbs sampler for normal mean and variance",
    "What is now proved was once only imagined. — William Blake",
)
_r("bhdi", "bhdi", "Bayesian", "Highest Density Interval (HDI)", "What is now proved was once only imagined. — William Blake")
_r("bifrc", "bifrc", "STEM", "Bifurcation diagram data for the logistic map", "Train yourself to let go.")
_r(
    "bmcmc",
    "bmcmc",
    "Bayesian",
    "Metropolis-Hastings MCMC sampler",
    "What is now proved was once only imagined. — William Blake",
)
_r("bnorm", "bnorm", "Bayesian", "Normal-normal conjugate Bayesian analysis", "This is where the fun begins. -- Anakin")
_r("box_m", "box_m", "Multivariate", "Box's M test for equality of covariance matrices", "Mourn them do not.")
_r(
    "bpois",
    "bpois",
    "Bayesian",
    "Gamma-Poisson conjugate Bayesian analysis",
    "Truly wonderful the mind of a child is.",
)
_r("bpred", "bpred", "Bayesian", "Bayesian posterior predictive check", "Let the past die. -- Kylo Ren")
_r("brop", "brop", "Bayesian", "Region of Practical Equivalence (ROPE) analysis", "Ready are you?")
_r(
    "brown",
    "brown",
    "NonparametricTest",
    "Brown-Forsythe test for equality of variances",
    "Congratulations. You are being rescued. -- K-2SO",
)
_r(
    "brpgn",
    "brpgn",
    "Diagnostic",
    "Breusch-Pagan test for heteroscedasticity",
    "Be mindful of your thoughts. -- Qui-Gon",
)
_r("cart", "cart", "ML", "CART decision tree (pure numpy, recursive splitting)", "Another happy landing. --")
_r("cbps", "cbps", "Bayesian", "Covariate Balancing Propensity Score (CBPS)", "Do or do not. There is no try.")
_r("cca", "cca", "Multivariate", "Canonical correlation analysis", "A larger world. --")
_r(
    "ccc",
    "ccc",
    "Correlation",
    "Lin's concordance correlation coefficient",
    "Distribution helper.",
)
_r("cfaai", "cfaai", "Psymet", "AIC for model comparison", "Distribution helper.")
_r("cfabc", "cfabc", "Psymet", "BIC for model comparison", "What is now proved was once only imagined. — William Blake")
_r("cfaer", "cfaer", "Psymet", "Expected parameter change from modification indices", "So uncivilized. --")
_r(
    "cfaes",
    "cfaes",
    "Psymet",
    "Exploratory SEM (ESEM) via rotated CFA loadings",
    "Weapons are part of my religion. -- Din Djarin",
)
_r("cfahi", "cfahi", "Psymet", "Higher-order CFA model", "Distribution helper.")
_r("cfawl", "cfawl", "Psymet", "WLSMV estimation for ordinal CFA", "Your overconfidence is your weakness. -- Luke")
_r("chaos", "chaos", "STEM", "Logistic map iterations", "Luminous beings are we.")
_r("chng", "chng", "TimeSeries", "CUSUM changepoint detection", "Control, you must learn control!")
_r("chol", "chol", "LinearAlgebra", "Cholesky decomposition", "There is another.")
_r("cmptm", "cmptm", "OTIS", "Compliance rate over time", "Somebody has to save our skins. --")
_r("cmpvl", "cmpvl", "OTIS", "Violation type distribution", "The belonging you seek is ahead. -- Maz Kanata")
_r(
    "cochr",
    "cochr",
    "NonparametricTest",
    "Cochran's Q test for k related binary samples",
    "This is the way. -- Din Djarin",
)
_r("coint", "coint", "TimeSeries", "Engle-Granger cointegration test", "Distribution helper.")
_r("cond", "cond", "LinearAlgebra", "Matrix condition number", "There is no try.")
_r(
    "cooks",
    "cooks",
    "Diagnostic",
    "Cook's distance",
    "When 900 years old you reach, look as good you will not.",
)
_r(
    "corrm",
    "corrm",
    "Correlation",
    "Full correlation matrix with p-values",
    "Stretch out with your feelings. --",
)
_r("crent", "crent", "InfoTheory", "Cross-entropy", "I like those odds. -- The Mandalorian")
_r("crtap", "crtap", "Criminology", "Appeal rate and success", "What is now proved was once only imagined. — William Blake")
_r("crtaq", "crtaq", "Criminology", "Acquittal rate by offense type", "You were my brother, Anakin! --")
_r("crtbk", "crtbk", "Criminology", "Court backlog analysis", "We are smarter than this. --")
_r("crtbl", "crtbl", "Criminology", "Bail grant rate analysis", "Save the dream! -- Saw Gerrera")
_r(
    "crtcv",
    "crtcv",
    "Criminology",
    "Civil liberties metric (Charter challenges, stays)",
    "Decide you must, how to serve them best.",
)
_r("crtdv", "crtdv", "Criminology", "Diversion program utilization", "Distribution helper.")
_r(
    "crtjr",
    "crtjr",
    "Criminology",
    "R v Jordan compliance (18/30 month ceiling)",
    "In a dark place we find ourselves.",
)
_r("crtpl", "crtpl", "Criminology", "Guilty plea rate", "Rebellions are built on hope. -- Jyn Erso")
_r("crttm", "crttm", "Criminology", "Time to trial distribution", "I have brought peace and security. -- Anakin")
_r(
    "crtya",
    "crtya",
    "Criminology",
    "Youth court special metrics (YCJA compliance)",
    "Distribution helper.",
)
_r("cstcl", "cstcl", "OTIS", "Custody classification level distribution and transi...", "Reckless is he.")
_r(
    "cstdy2",
    "cstdy2",
    "OTIS",
    "Pre-trial custody credit days calculation (R v Summe...",
    "Distribution helper.",
)
_r(
    "csted",
    "csted",
    "OTIS",
    "Education program enrollment and completion in custody",
    "Distribution helper.",
)
_r("cstfr", "cstfr", "OTIS", "Per-facility rate for custody metrics", "Hope is like the sun. --")
_r(
    "cstmd",
    "cstmd",
    "OTIS",
    "Medical service utilization rate in custody",
    "I have a very bad feeling about this. --",
)
_r("cstrl", "cstrl", "OTIS", "Release type distribution in custody", "We have everything we need. --")
_r("cstvs", "cstvs", "OTIS", "Visitation frequency analysis in custody", "What is now proved was once only imagined. — William Blake")
_r(
    "cstwk",
    "cstwk",
    "OTIS",
    "Work program participation rates in custody",
    "Distribution helper.",
)
_r("curei", "curei", "Survival", "Mixture cure model", "You need a teacher! -- Kylo Ren")
_r("dcorr", "dcorr", "Correlation", "Distance correlation", "Wizard! -- Anakin")
_r("dfbts", "dfbts", "Diagnostic", "DFBETAS influence measure", "Hmm. Something is not right.")
_r("dffts", "dffts", "Diagnostic", "DFFITS influence measure", "What is now proved was once only imagined. — William Blake")
_r("dfts", "dfts", "General", "Power spectral density via FFT", "The dark side clouds everything.")
_r("difbd", "difbd", "Psymet", "DIF bundle/testlet analysis", "You will find only what you bring in.")
_r(
    "difdt",
    "difdt",
    "Psymet",
    "Delta plot method for DIF detection",
    "The strongest stars have hearts of kyber. -- Chirrut",
)
_r(
    "diffl",
    "diffl",
    "Psymet",
    "Summary of flagged DIF items across methods",
    "Who is more foolish? The fool or the fool who follows? --",
)
_r(
    "difir",
    "difir",
    "Psymet",
    "IRT-based DIF using likelihood ratio",
    "If you are not with me, you are my enemy. -- Anakin",
)
_r("difld", "difld", "Psymet", "Lord's chi-square DIF detection", "What is now proved was once only imagined. — William Blake")
_r(
    "difnb",
    "difnb",
    "Psymet",
    "Non-uniform DIF detection via interaction term",
    "Agree with you the council does.",
)
_r("difpr", "difpr", "Psymet", "Iterative DIF purification", "Much fear I sense in you.")
_r("difrj", "difrj", "Psymet", "Raju's signed/unsigned area DIF measure", "Revealed your opinion is.")
_r("difst", "difst", "Psymet", "Stocking-Lord DIF detection", "What is now proved was once only imagined. — William Blake")
_r(
    "difub",
    "difub",
    "Psymet",
    "Uniform DIF detection via logistic regression",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "dnntt",
    "dnntt",
    "NonparametricTest",
    "Dunnett's test — multiple treatment groups vs control",
    "Would it help if I got out and pushed? --",
)
_r("dtw", "dtw", "General", "Dynamic time warping distance", "The negotiations were short. --")
_r(
    "eig_",
    "eig_",
    "Spatial",
    "Eigenvalue/eigenvector analysis",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "emimq",
    "emimq",
    "MissingData",
    "EM algorithm imputation for missing data",
    "What is now proved was once only imagined. — William Blake",
)
_r("entpy", "entpy", "InfoTheory", "Shannon entropy", "What is now proved was once only imagined. — William Blake")
_r("eqatn", "eqatn", "Criminology", "Atkinson inequality index", "We keep our promises. -- Windu")
_r(
    "eqcon",
    "eqcon",
    "Criminology",
    "Health concentration index",
    "Distribution helper.",
)
_r(
    "eqdsp",
    "eqdsp",
    "Criminology",
    "Decompose disparity (Blinder-Oaxaca)",
    "I find that answer vague and unconvincing. -- K-2SO",
)
_r(
    "eqgni",
    "eqgni",
    "Criminology",
    "Gini coefficient for inequality",
    "Distribution helper.",
)
_r("eqlrz", "eqlrz", "Criminology", "Lorenz curve data", "What is now proved was once only imagined. — William Blake")
_r("eqplm", "eqplm", "Criminology", "Palma ratio (top 10% / bottom 40%)", "Wherever I go, he goes. -- Din Djarin")
_r("eqrid", "eqrid", "Criminology", "Racial disparity index (RDI)", "So this is how liberty dies. -- Padme")
_r(
    "eqrii",
    "eqrii",
    "Criminology",
    "Relative index of inequality (RII)",
    "What is now proved was once only imagined. — William Blake",
)
_r("eqslp", "eqslp", "Criminology", "Slope index of inequality (SII)", "What is now proved was once only imagined. — William Blake")
_r("eqthl", "eqthl", "Criminology", "Theil's entropy index", "What is now proved was once only imagined. — William Blake")
_r(
    "evals",
    "evals",
    "CausalEpi",
    "E-value for unmeasured confounding",
    "Around the survivors a perimeter create.",
)
_r("expns", "expns", "TimeSeries", "Holt-Winters exponential smoothing", "Feel, do not think. -- Qui-Gon")
_r(
    "fapca",
    "fapca",
    "Epidemiology",
    "Compare factor analysis vs PCA solutions",
    "Distribution helper.",
)
_r("fib", "fib", "STEM", "Fibonacci golden ratio convergence", "Why you stuck-up, scruffy-looking nerf herder! --")
_r("fizz", "fizz", "STEM", "FizzBuzz proportions", "What is now proved was once only imagined. — William Blake")
_r(
    "flgnr",
    "flgnr",
    "NonparametricTest",
    "Fligner-Killeen test for homogeneity of variance",
    "The shadow of greed that is.",
)
_r(
    "fmm",
    "fmm",
    "LatentModel",
    "Finite mixture model (general Gaussian)",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "frdmn",
    "frdmn",
    "NonparametricTest",
    "Friedman test for repeated measures",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "garch",
    "garch",
    "TimeSeries",
    "GARCH(1,1) volatility model",
    "What about the droid attack on the Wookiees? -- Ki-Adi-Mundi",
)
_r(
    "gform",
    "gform",
    "CausalEpi",
    "Parametric g-formula (g-computation)",
    "Difficult to see. Always in motion is the future.",
)
_r(
    "gfunc",
    "gfunc",
    "Spatial",
    "Pair correlation function g(r) for spatial point pat...",
    "Unexpected this is.",
)
_r("gmm", "gmm", "General", "Gaussian mixture model via EM algorithm", "Clouded this boy future is.")
_r("gwreg", "gwreg", "Spatial", "Geographically weighted regression (GWR)", "You have that look. --")
_r(
    "hmm",
    "hmm",
    "LatentModel",
    "Hidden Markov model (forward-backward for discrete obs)",
    "Already know you that which you need.",
)
_r("hotdk", "hotdk", "MissingData", "Hot-deck imputation", "I sense great fear in you.")
_r("hotl", "hotl", "Multivariate", "Hotelling's T-squared test", "What is now proved was once only imagined. — William Blake")
_r(
    "hurld",
    "hurld",
    "CountModel",
    "Hurdle model for count data",
    "Not to worry, we are still flying half a ship. --",
)
_r("icc_", "icc_", "Correlation", "Intraclass correlation coefficient (ICC)", "The boy is dangerous.")
_r(
    "ipctw",
    "ipctw",
    "CausalEpi",
    "Inverse probability of censoring weights (IPCW)",
    "He is too dangerous to be left alive! -- Windu",
)
_r("irteq", "irteq", "Criminology", "IRT true-score equating", "The galaxy has changed. -- Sabine Wren")
_r("irtgu", "irtgu", "Psymet", "Pseudo-guessing parameter analysis for 3PL IRT", "Please do not do that. --")
_r(
    "irtlk",
    "irtlk",
    "Psymet",
    "IRT scale linking (mean/sigma method)",
    "Distribution helper.",
)
_r(
    "irtmp",
    "irtmp",
    "Psymet",
    "MAP theta estimation with normal prior",
    "Not if anything to say about it I have.",
)
_r("irtnm", "irtnm", "Psymet", "Nominal response model for multiple-choice items", "What is now proved was once only imagined. — William Blake")
_r("irtrs", "irtrs", "Psymet", "Rating Scale Model for Likert-type items", "You must complete the training.")
_r(
    "irtse",
    "irtse",
    "Psymet",
    "Standard error of theta at each ability level",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "ivw",
    "ivw",
    "General",
    "Instrumental variable Wald estimate (MR-style)",
    "Death is a natural part of life.",
)
_r("jsdiv", "jsdiv", "InfoTheory", "Jensen-Shannon divergence", "Distribution helper.")
_r("kendt", "kendt", "Correlation", "Kendall's tau-b with CI", "What is now proved was once only imagined. — William Blake")
_r("kldiv", "kldiv", "General", "Kullback-Leibler divergence", "What is now proved was once only imagined. — William Blake")
_r("kmpp", "kmpp", "ML", "K-means++ initialisation + Lloyd's algorithm", "What is now proved was once only imagined. — William Blake")
_r("knox", "knox", "General", "Knox test for space-time clustering", "This party is over. -- Windu")
_r(
    "kron",
    "kron",
    "LinearAlgebra",
    "Kronecker product of two matrices",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "lca",
    "lca",
    "LatentModel",
    "Latent class analysis (EM for binary indicators)",
    "Impossible to see the future is.",
)
_r(
    "lcgm",
    "lcgm",
    "LatentModel",
    "Latent growth curve model (simplified OLS-based)",
    "Distribution helper.",
)
_r(
    "logrnk",
    "logrnk",
    "Survival",
    "Log-rank test for comparing two survival curves",
    "Meditate on this I will.",
)
_r("lpa", "lpa", "LatentModel", "Latent profile analysis (GMM for continuous data)", "What is now proved was once only imagined. — William Blake")
_r(
    "ltmcr",
    "ltmcr",
    "MissingData",
    "Little's MCAR test (alternative implementation)",
    "What is now proved was once only imagined. — William Blake",
)
_r("manova", "manova", "Multivariate", "One-way MANOVA (Wilks' lambda)", "Distribution helper.")
_r("mantel", "mantel", "Multivariate", "Mantel test (matrix correlation)", "Concentrate.")
_r(
    "mcdm",
    "mcdm",
    "GameTheory",
    "TOPSIS multi-criteria decision making",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "mcnmr",
    "mcnmr",
    "NonparametricTest",
    "McNemar's test for paired proportions",
    "I fight for everything my mother believed in. --",
)
_r(
    "mds_",
    "mds_",
    "General",
    "Classical (metric) multidimensional scaling",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "micev",
    "micev",
    "MissingData",
    "MICE — Multiple Imputation by Chained Equations",
    "What I told you was true, from a certain point of view. --",
)
_r("mispt", "mispt", "MissingData", "Missing data pattern analysis", "You may fire when ready. -- Tarkin")
_r("mlp", "mlp", "ML", "Simple MLP (1 hidden layer, numpy)", "Distribution helper.")
_r("mnmx", "mnmx", "GameTheory", "Minimax strategy for zero-sum games", "Failed I have.")
_r("moodm", "moodm", "OTIS", "Mood's median test", "You stuck-up, half-witted, scruffy-looking nerf herder! --")
_r(
    "mrdst",
    "mrdst",
    "Multivariate",
    "Mahalanobis distance",
    "Named must be your fear before banish it you can.",
)
_r("msmw", "msmw", "CausalEpi", "Marginal structural model weights", "I know what I have to do. -- Kylo Ren")
_r(
    "mtobac",
    "mtobac",
    "Criminology",
    "BAC distribution in impaired driving",
    "The truth is often what we make of it. --",
)
_r(
    "mtoben",
    "mtoben",
    "Criminology",
    "Road safety intervention benefit-cost ratio",
    "What is now proved was once only imagined. — William Blake",
)
_r("mtocl", "mtocl", "Criminology", "Collision rate per vehicle-kilometres travelled", "That is why you fail.")
_r("mtocyc", "mtocyc", "Criminology", "Cyclist safety analysis", "Always pass on what you have learned.")
_r("mtodr", "mtodr", "Criminology", "Driver risk profile by age/gender", "My powers have doubled. -- Anakin")
_r(
    "mtoft",
    "mtoft",
    "Criminology",
    "Traffic fatality rate per 100K population",
    "Much to learn you still have.",
)
_r(
    "mtoij",
    "mtoij",
    "Criminology",
    "Injury severity distribution",
    "Anger, fear, aggression. The dark side are they.",
)
_r(
    "mtoint",
    "mtoint",
    "Criminology",
    "Intersection safety analysis",
    "What is now proved was once only imagined. — William Blake",
)
_r("mtoped", "mtoped", "Criminology", "Pedestrian collision analysis", "Size matters not.")
_r("mtord", "mtord", "Criminology", "Per-road-segment crash rate", "Distribution helper.")
_r("mtosp", "mtosp", "Criminology", "Speed distribution analysis", "Bring me Solo. -- Jabba the Hutt")
_r("mtotm", "mtotm", "Criminology", "Temporal crash patterns", "Blast! This is why I hate flying. --")
_r(
    "mtotrn",
    "mtotrn",
    "Criminology",
    "Long-term road safety trend",
    "Make ten men feel like a hundred. -- Cassian Andor",
)
_r("mtovh", "mtovh", "Criminology", "Crash rates by vehicle type", "Karabast! -- Zeb Orrelios")
_r(
    "mtowd",
    "mtowd",
    "Criminology",
    "Weather-related crash factor analysis",
    "Distribution helper.",
)
_r(
    "mutl",
    "mutl",
    "InfoTheory",
    "Mutual information between two discrete variables",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "nash",
    "nash",
    "GameTheory",
    "Nash equilibrium for 2-player games (support enumera...",
    "What is now proved was once only imagined. — William Blake",
)
_r("nbreg", "nbreg", "CountModel", "Negative binomial regression", "What is now proved was once only imagined. — William Blake")
_r(
    "negct",
    "negct",
    "EnvEpi",
    "Negative control outcome/exposure test",
    "Distribution helper.",
)
_r("netdg", "netdg", "Network", "Network degree distribution", "We have hope. -- Jyn Erso")
_r("neter", "neter", "Network", "Erdos-Renyi random graph generator", "The shroud of the dark side has fallen.")
_r("netpl", "netpl", "Network", "Network average shortest path length (BFS)", "What is now proved was once only imagined. — William Blake")
_r("netsw", "netsw", "Network", "Small-world coefficient", "Mind what you have learned.")
_r("nnidx", "nnidx", "General", "Clark-Evans nearest neighbor index", "Now THIS is podracing! -- Anakin")
_r("ntccf", "ntccf", "Network", "Network clustering coefficient", "A prophecy that misread could have been.")
_r("nwcen", "nwcen", "Psymet", "Node centrality measures for a network", "Distribution helper.")
_r(
    "nwcom",
    "nwcom",
    "Psymet",
    "Community detection via modularity optimization",
    "We are the spark that will light the fire. -- Poe Dameron",
)
_r("nwcor", "nwcor", "Psymet", "Partial correlation network", "Remember: concentrate on the moment. -- Qui-Gon")
_r("nwstb", "nwstb", "Psymet", "Network stability via case-dropping bootstrap", "What is now proved was once only imagined. — William Blake")
_r(
    "obund",
    "obund",
    "OTIS",
    "Manski partial identification bounds",
    "Distribution helper.",
)
_r("odm_c", "odm_c", "OTIS", "Full demographic cross-tabulation", "Clear your mind must be.")
_r(
    "odm_e",
    "odm_e",
    "OTIS",
    "Equity metrics: representation index and disparity r...",
    "Good soldiers follow orders. -- Tup",
)
_r("odm_i", "odm_i", "OTIS", "Diversity index (Simpson/Shannon) for demographics", "Distribution helper.")
_r("odm_p", "odm_p", "OTIS", "Proportions per group with confidence interval", "What is now proved was once only imagined. — William Blake")
_r(
    "odm_s",
    "odm_s",
    "OTIS",
    "Standardize demographics to reference population",
    "I know what you are going to say. --",
)
_r("odm_t", "odm_t", "OTIS", "Demographic trend over time", "The greatest teacher failure is.")
_r("odml5", "odml5", "OTIS", "DML: volatility effect on outcome", "What is now proved was once only imagined. — William Blake")
_r("odml6", "odml6", "OTIS", "DML: custody type effect on outcome", "Into exile I must go.")
_r(
    "odose",
    "odose",
    "OTIS",
    "Dose-response: number of placements to outcome",
    "What is now proved was once only imagined. — William Blake",
)
_r("ordd1", "ordd1", "OTIS", "Regression discontinuity at age cutoff", "Judge me by my size, do you?")
_r("osyn1", "osyn1", "OTIS", "Synthetic control for one region", "Distribution helper.")
_r(
    "ovrla",
    "ovrla",
    "Causal",
    "Overlap weighting for causal inference",
    "I have been looking forward to this. -- Dooku",
)
_r(
    "paget",
    "paget",
    "NonparametricTest",
    "Page test for ordered alternatives",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "pbis",
    "pbis",
    "Correlation",
    "Point-biserial correlation",
    "Distribution helper.",
)
_r("pca_", "pca_", "ML", "PCA via SVD (pure numpy)", "What is now proved was once only imagined. — William Blake")
_r("pcorr", "pcorr", "Correlation", "Partial correlation", "Powerful you have become.")
_r("pinv", "pinv", "LinearAlgebra", "Moore-Penrose pseudoinverse", "There is always room for one more. -- Enfys Nest")
_r("pois", "pois", "CountModel", "Poisson regression via IRLS", "Hard to see, the dark side is.")
_r(
    "polyc",
    "polyc",
    "Correlation",
    "Polychoric correlation (two-step approximation)",
    "There has been an awakening. -- Snoke",
)
_r("ppca", "ppca", "Multivariate", "Probabilistic PCA via EM", "What is now proved was once only imagined. — William Blake")
_r("prcpt", "prcpt", "ML", "Single-layer perceptron", "What is now proved was once only imagined. — William Blake")
_r("prgcm", "prgcm", "Criminology", "Program completion rate", "What is now proved was once only imagined. — William Blake")
_r("prgcs", "prgcs", "Criminology", "Cost savings from correctional program", "What is now proved was once only imagined. — William Blake")
_r("prgdm", "prgdm", "Criminology", "DML for program causal effect", "Concentrate all fire. -- Tarkin")
_r(
    "prgef",
    "prgef",
    "Criminology",
    "Pre-post program effect size (Cohen's d)",
    "Governor Tarkin. I thought I smelled your foul stench. --",
)
_r("prgen", "prgen", "Criminology", "Program enrollment rates", "Is it possible to learn this power? -- Anakin")
_r("prgqe", "prgqe", "Criminology", "Quasi-experimental design analysis (matching + DiD)", "Blind we are.")
_r("prgrc", "prgrc", "Criminology", "Recidivism by program participation", "I will do what I must. --")
_r("prgsb", "prgsb", "Criminology", "Program effect by subgroup", "Distribution helper.")
_r("prgsv", "prgsv", "Criminology", "Program time-to-completion survival analysis", "What is now proved was once only imagined. — William Blake")
_r(
    "prgwt",
    "prgwt",
    "Criminology",
    "Waitlist analysis (instrumental for causal)",
    "Strike me down and I shall become more powerful. --",
)
_r("prime", "prime", "STEM", "Prime density — pi(n) counting function", "You were the chosen one! --")
_r("procr", "procr", "Multivariate", "Procrustes analysis", "Grave danger you are in.")
_r("psmch", "psmch", "Bayesian", "Propensity score matching (nearest neighbor)", "I suggest patience. -- Windu")
_r("psstr", "psstr", "Bayesian", "Propensity score stratification", "Begun, the Clone War has.")
_r("pstrim", "pstrim", "Bayesian", "Propensity score trimming", "At an end your rule is.")
_r(
    "pwexp",
    "pwexp",
    "Survival",
    "Piecewise exponential model",
    "I have placed information vital to the Rebellion. --",
)
_r(
    "quadr",
    "quadr",
    "Spatial",
    "Quadrat test for Complete Spatial Randomness (CSR)",
    "What is now proved was once only imagined. — William Blake",
)
_r("rbif", "rbif", "Psymet", "Omega from bifactor model", "Wars not make one great.")
_r("rboot", "rboot", "Psymet", "Bootstrap CI for reliability", "What is now proved was once only imagined. — William Blake")
_r(
    "rcdbi",
    "rcdbi",
    "OTIS",
    "Burden index: weighted recidivism accounting for sev...",
    "You are a bold one. -- Grievous",
)
_r("rcdcm", "rcdcm", "OTIS", "Competing risks analysis for recidivism", "I am the pilot. -- Hera Syndulla")
_r(
    "rcddl",
    "rcddl",
    "OTIS",
    "Double Machine Learning for recidivism treatment effect",
    "Before the dark times. --",
)
_r("rcdds", "rcdds", "OTIS", "Desistance curve: probability of stopping offending ...", "Spring the trap. -- Ackbar")
_r(
    "rcdfr",
    "rcdfr",
    "OTIS",
    "Fairness metrics for recidivism predictions",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "rcdmx",
    "rcdmx",
    "OTIS",
    "Mixture model for recidivism: recidivists vs desisters",
    "Distribution helper.",
)
_r("rcdsb", "rcdsb", "OTIS", "Recidivism rate by subgroup", "Let the Wookiee win. -- C-3PO")
_r("rcdsr", "rcdsr", "OTIS", "Kaplan-Meier survival curve for time-to-recidivism", "What is now proved was once only imagined. — William Blake")
_r("rf_", "rf_", "ML", "Random forest (bagged CART) for regression", "Same jacket. --")
_r("rform", "rform", "Psymet", "Parallel forms reliability", "Distribution helper.")
_r("rgcv", "rgcv", "Psymet", "Generalizability coefficient (G-theory)", "There is good in him. -- Luke")
_r("rmsyt", "rmsyt", "Diagnostic", "Ramsey RESET specification test", "Attachment leads to jealousy. -- Anakin")
_r(
    "rnsum",
    "rnsum",
    "NonparametricTest",
    "Wald-Wolfowitz runs test for randomness",
    "What is now proved was once only imagined. — William Blake",
)
_r("romg2", "romg2", "Psymet", "Omega per subscale", "Patience.")
_r(
    "romgh",
    "romgh",
    "Psymet",
    "Hierarchical omega per subscale",
    "What is now proved was once only imagined. — William Blake",
)
_r("rskbs", "rskbs", "OTIS", "Brier score for risk predictions", "What is now proved was once only imagined. — William Blake")
_r("rskci", "rskci", "OTIS", "Concordance (C-statistic) for risk scores", "Train yourself to let go.")
_r(
    "rskgp",
    "rskgp",
    "OTIS",
    "Profile of risk groups by demographics and offenses",
    "What is now proved was once only imagined. — William Blake",
)
_r("rsknb", "rsknb", "OTIS", "Nagelkerke R-squared for risk model", "This is where the fun begins. -- Anakin")
_r("rskrd", "rskrd", "OTIS", "Net Reclassification Improvement between two risk tools", "Mourn them do not.")
_r("rtest", "rtest", "Psymet", "Test-retest reliability via ICC", "Truly wonderful the mind of a child is.")
_r("sarma", "sarma", "General", "Seasonal ARMA model", "Let the past die. -- Kylo Ren")
_r("sav_a", "sav_a", "Psymet", "EA subscale average variance extracted", "Ready are you?")
_r(
    "sav_e",
    "sav_e",
    "Psymet",
    "EE subscale average variance extracted",
    "Congratulations. You are being rescued. -- K-2SO",
)
_r("sav_r", "sav_r", "Psymet", "ER subscale average variance extracted", "Be mindful of your thoughts. -- Qui-Gon")
_r("sav_u", "sav_u", "Psymet", "UA subscale average variance extracted", "Another happy landing. --")
_r("scan", "scan", "Spatial", "Kulldorff's spatial scan statistic", "Do or do not. There is no try.")
_r("schfd", "schfd", "Survival", "Schoenfeld residuals test for PH assumption", "A larger world. --")
_r("scpct", "scpct", "Psymet", "Percentile norm table", "Distribution helper.")
_r(
    "scr_a",
    "scr_a",
    "Psymet",
    "EA subscale composite reliability (rho_c)",
    "Distribution helper.",
)
_r("scr_e", "scr_e", "Psymet", "EE subscale composite reliability (rho_c)", "What is now proved was once only imagined. — William Blake")
_r("scr_r", "scr_r", "Psymet", "ER subscale composite reliability (rho_c)", "So uncivilized. --")
_r(
    "scr_u",
    "scr_u",
    "Psymet",
    "UA subscale composite reliability (rho_c)",
    "Weapons are part of my religion. -- Din Djarin",
)
_r("scraw", "scraw", "Psymet", "Compute raw total/subscale scores", "Distribution helper.")
_r(
    "shply",
    "shply",
    "GameTheory",
    "Shapley values for cooperative games",
    "Your overconfidence is your weakness. -- Luke",
)
_r("siucmp", "siucmp", "Criminology", "Cross-jurisdiction SIU comparison", "Luminous beings are we.")
_r("siudem", "siudem", "Criminology", "Demographics in SIU cases", "Control, you must learn control!")
_r("siuof", "siuof", "Criminology", "Use of force in SIU cases analysis", "There is another.")
_r("siuot", "siuot", "Criminology", "SIU case outcome distribution", "Somebody has to save our skins. --")
_r(
    "siurcm",
    "siurcm",
    "Criminology",
    "SIU director recommendation analysis",
    "The belonging you seek is ahead. -- Maz Kanata",
)
_r("siurgn", "siurgn", "Criminology", "SIU cases by geographic region", "This is the way. -- Din Djarin")
_r("siurt", "siurt", "Criminology", "SIU case rate per 1000 officers", "Distribution helper.")
_r("siutm", "siutm", "Criminology", "SIU case processing time", "There is no try.")
_r(
    "siutrn",
    "siutrn",
    "Criminology",
    "SIU case trend over years",
    "When 900 years old you reach, look as good you will not.",
)
_r("siutyp", "siutyp", "Criminology", "SIU cases by incident type", "Stretch out with your feelings. --")
_r("sntcn", "sntcn", "OTIS", "Concurrent vs consecutive sentence analysis", "I like those odds. -- The Mandalorian")
_r(
    "sntjd",
    "sntjd",
    "OTIS",
    "Judicial variation in sentencing",
    "What is now proved was once only imagined. — William Blake",
)
_r("sntmn", "sntmn", "OTIS", "Mandatory minimum sentence analysis", "You were my brother, Anakin! --")
_r("sntpl", "sntpl", "OTIS", "Sentence length by plea type", "We are smarter than this. --")
_r("snttm", "snttm", "OTIS", "Time served vs sentence ratio", "Save the dream! -- Saw Gerrera")
_r(
    "spcor",
    "spcor",
    "Correlation",
    "Semipartial (part) correlation",
    "Decide you must, how to serve them best.",
)
_r("stk", "stk", "General", "Space-time K function", "Distribution helper.")
_r(
    "stl",
    "stl",
    "TimeSeries",
    "STL decomposition (seasonal, trend, residual)",
    "In a dark place we find ourselves.",
)
_r(
    "survc",
    "survc",
    "Survival",
    "Harrell's concordance index for survival models",
    "Rebellions are built on hope. -- Jyn Erso",
)
_r("svd_", "svd_", "ML", "Truncated SVD", "I have brought peace and security. -- Anakin")
_r("tetrc", "tetrc", "Correlation", "Tetrachoric correlation", "Distribution helper.")
_r("tgtrl", "tgtrl", "CausalEpi", "Target trial emulation framework", "Reckless is he.")
_r("theil", "theil", "TimeSeries", "Theil-Sen robust trend estimator", "Distribution helper.")
_r(
    "tpscb",
    "tpscb",
    "Criminology",
    "Cost-benefit analysis of policing intervention",
    "Distribution helper.",
)
_r("tpscl", "tpscl", "Criminology", "Case clearance rate by offense type", "Hope is like the sun. --")
_r("tpscmp", "tpscmp", "Criminology", "Civilian complaint rate", "I have a very bad feeling about this. --")
_r("tpscrs", "tpscrs", "Criminology", "Crime severity index", "We have everything we need. --")
_r("tpsdep", "tpsdep", "Criminology", "Officer deployment analysis by division", "What is now proved was once only imagined. — William Blake")
_r(
    "tpsdiv",
    "tpsdiv",
    "Criminology",
    "Compare crime rates across divisions",
    "Distribution helper.",
)
_r("tpsgis", "tpsgis", "Criminology", "Geographic crime density", "You need a teacher! -- Kylo Ren")
_r("tpsmj", "tpsmj", "Criminology", "Major crime indicators summary", "Wizard! -- Anakin")
_r("tpsnb", "tpsnb", "Criminology", "Neighbourhood-level crime profile", "Hmm. Something is not right.")
_r("tpspr", "tpspr", "Criminology", "Patrol efficiency metrics", "What is now proved was once only imagined. — William Blake")
_r("tpsrp", "tpsrp", "Criminology", "Police report summary statistics", "The dark side clouds everything.")
_r(
    "tpsrs",
    "tpsrs",
    "Criminology",
    "Response time distribution analysis",
    "You will find only what you bring in.",
)
_r(
    "tpsstp",
    "tpsstp",
    "Criminology",
    "Stop and search disparity analysis",
    "The strongest stars have hearts of kyber. -- Chirrut",
)
_r(
    "tpstrn",
    "tpstrn",
    "Criminology",
    "Crime trend analysis",
    "Who is more foolish? The fool or the fool who follows? --",
)
_r(
    "tpsuof",
    "tpsuof",
    "Criminology",
    "Use of force rate and type distribution",
    "If you are not with me, you are my enemy. -- Anakin",
)
_r("tscls", "tscls", "TimeSeries", "1-NN DTW time series classifier", "What is now proved was once only imagined. — William Blake")
_r("vcont", "vcont", "Psymet", "Content validity ratio (Lawshe CVR)", "Agree with you the council does.")
_r("vconv", "vconv", "Psymet", "Convergent validity evidence", "Much fear I sense in you.")
_r("vcteq", "vcteq", "Criminology", "Victimization equity across demographics", "Revealed your opinion is.")
_r("vctfr", "vctfr", "Criminology", "Fear of crime index", "What is now proved was once only imagined. — William Blake")
_r("vcthp", "vcthp", "Criminology", "Help-seeking behavior rates", "What is now proved was once only imagined. — William Blake")
_r("vctrp", "vctrp", "Criminology", "Repeat victimization analysis", "Would it help if I got out and pushed? --")
_r("vctsv", "vctsv", "Criminology", "Victimization severity scale", "The negotiations were short. --")
_r(
    "vdisc",
    "vdisc",
    "Psymet",
    "Discriminant validity via HTMT ratio",
    "What is now proved was once only imagined. — William Blake",
)
_r("wavlt", "wavlt", "TimeSeries", "Discrete wavelet transform (Haar)", "What is now proved was once only imagined. — William Blake")
_r("wlcxn", "wlcxn", "Survival", "Wilcoxon (Gehan) test for survival curves", "What is now proved was once only imagined. — William Blake")
_r("xgb_", "xgb_", "ML", "Simplified XGBoost (L2 regularized gradient boosting)", "We keep our promises. -- Windu")
_r(
    "zinb",
    "zinb",
    "CountModel",
    "Zero-Inflated Negative Binomial (ZINB) model",
    "Distribution helper.",
)
_r(
    "zip",
    "zip",
    "CountModel",
    "Zero-Inflated Poisson (ZIP) model",
    "I find that answer vague and unconvincing. -- K-2SO",
)

# -- Signal Processing --
_r("buttlp", "butter_lowpass", "Signal", "Butterworth lowpass filter", "The low frequencies remain. --")
_r("butthp", "butter_highpass", "Signal", "Butterworth highpass filter", "What is now proved was once only imagined. — William Blake")
_r(
    "buttbp",
    "butter_bandpass",
    "Signal",
    "Butterworth bandpass filter",
    "Let the chosen frequencies through. -- Qui-Gon",
)
_r(
    "buttbs",
    "butter_bandstop",
    "Signal",
    "Butterworth bandstop (notch) filter",
    "Silence the interference. -- Mace Windu",
)
_r("sgolay", "savgol_smooth", "Signal", "Savitzky-Golay polynomial smoothing", "Smooth out the noise you must.")
_r("welch", "welch_psd", "Signal", "Welch power spectral density", "What is now proved was once only imagined. — William Blake")
_r("pburg", "burg_psd", "Signal", "Burg AR power spectral density", "A parametric path to the spectrum. -- Dooku")
_r("hfd", "higuchi_fd", "Signal", "Higuchi fractal dimension", "Fractal complexity the signal has.")
_r("kfd", "katz_fd", "Signal", "Katz fractal dimension", "The path length reveals complexity. -- Chirrut")
_r("pfd", "petrosian_fd", "Signal", "Petrosian fractal dimension", "Sign changes tell the story. -- Ahsoka")
_r(
    "dfa",
    "detrended_fluctuation",
    "Signal",
    "Detrended fluctuation analysis",
    "Long-range correlations I sense.",
)
_r("sampen", "sample_entropy", "Signal", "Sample entropy", "Regularity in the signal there is.")
_r("apen", "approx_entropy", "Signal", "Approximate entropy", "Predictability the entropy measures. -- Dooku")

# -- Cepstrum / Homomorphic deconvolution --
_r(
    "cepst",
    "real_cepstrum",
    "Cepstrum",
    "Real cepstrum via log-magnitude FFT",
    "Echoes in the quefrency domain. --",
)
_r(
    "hcepst",
    "complex_cepstrum",
    "Cepstrum",
    "Complex cepstrum with phase unwrapping",
    "The full cepstral path. -- Qui-Gon",
)
_r(
    "hdecon",
    "homomorphic_deconvolve",
    "Cepstrum",
    "Homomorphic deconvolution via cepstral liftering",
    "Separate the convolved you must.",
)

# -- ECG / HRV / PCG (Biomedical) --
_r("ecgdet", "pan_tompkins", "Signal", "Pan-Tompkins QRS detector", "Find the heartbeat you must.")
_r("rrint", "rr_intervals", "Signal", "RR interval series from R-peaks", "The rhythm between beats. -- Chirrut")
_r(
    "hrvtd",
    "hrv_time_domain",
    "Signal",
    "HRV time-domain metrics (SDNN, RMSSD, pNN50)",
    "Time reveals the heart's variance. --",
)
_r(
    "hrvfd",
    "hrv_freq_domain",
    "Signal",
    "HRV frequency-domain (VLF/LF/HF)",
    "Frequency tells what time cannot. --",
)
_r("hrvnl", "hrv_nonlinear", "Signal", "HRV nonlinear (Poincare SD1/SD2)", "Nonlinear the heart's rhythm is.")
_r("pcgenv", "pcg_envelope", "Signal", "PCG Shannon-energy envelope", "The heart sound's shape emerges. -- Chirrut")
_r("pcgflt", "pcg_filter", "Signal", "PCG bandpass preprocessing filter", "Listen to the heart you must.")
_r("pcgseg", "pcg_segment", "Signal", "S1/S2 heart sound segmentation", "Two sounds, one cycle. --")
_r("pcgmur", "pcg_murmur_score", "Signal", "PCG murmur detection score", "A disturbance in the heart I sense.")

# -- GLM --
_r(
    "glmft",
    "glm_fit",
    "Regression",
    "Generalized linear model (all families/links)",
    "A general model for all. -- Mon Mothma",
)

# -- Crypto --
_r(
    "mlkem",
    "mlkem768_keygen",
    "Crypto",
    "ML-KEM-768 post-quantum key encapsulation",
    "Encrypted the transmission must be.",
)
_r(
    "cpoly",
    "chacha20_encrypt",
    "Crypto",
    "ChaCha20-Poly1305 authenticated encryption",
    "Secure the channel is. -- Admiral Ackbar",
)
_r(
    "mldsa",
    "mldsa_keygen",
    "Crypto",
    "ML-DSA post-quantum signature keygen",
    "Sign with the crystal, you must.",
)
_r(
    "mldss",
    "mldsa_sign",
    "Crypto",
    "ML-DSA post-quantum signature sign",
    "Your identity, the lattice confirms. -- Ahsoka",
)
_r(
    "mldsv",
    "mldsa_verify",
    "Crypto",
    "ML-DSA post-quantum signature verify",
    "A true signature cannot be forged. --",
)
_r(
    "ntru",
    "ntru_keygen",
    "Crypto",
    "NTRU post-quantum key exchange keygen",
    "Distribution helper.",
)
_r(
    "ntruc",
    "ntru_encrypt",
    "Crypto",
    "NTRU post-quantum encryption",
    "Wrapped in rings, the secret travels. -- Maz Kanata",
)
_r("ntrud", "ntru_decrypt", "Crypto", "NTRU post-quantum decryption", "Unwrap the ring, the truth reveals. -- Luke")
_r(
    "mcelc",
    "mceliece_encrypt",
    "Crypto",
    "Classic McEliece syndrome encryption",
    "What is now proved was once only imagined. — William Blake",
)

# -- Lattice --
_r("lwe", "lwe_sample", "Lattice", "LWE sample generation", "Errors in the signal, there are.")
_r("rlwe", "rlwe_keygen", "Lattice", "Ring-LWE key generation", "The ring binds all things.")
_r("lll", "lll_reduce", "Lattice", "LLL lattice basis reduction", "Reduce your losses, you must.")
_r(
    "babai",
    "babai_cvp",
    "Lattice",
    "Babai nearest plane CVP",
    "The closest answer is sometimes the right one. -- Qui-Gon",
)
_r(
    "gso",
    "gram_schmidt_orth",
    "Lattice",
    "Gram-Schmidt orthogonalization",
    "Distribution helper.",
)
_r(
    "bkz",
    "bkz_reduce",
    "Lattice",
    "BKZ lattice basis reduction",
    "Block by block, we shall dismantle their defenses. -- Thrawn",
)
_r("svpap", "svp_approx", "Lattice", "Approximate shortest vector problem", "Find the shortest path, you must.")
_r(
    "lweke",
    "lwe_key_exchange",
    "Lattice",
    "LWE Diffie-Hellman key exchange",
    "Distribution helper.",
)

# -- FiniteField --
_r(
    "gf2m",
    "gf2m_arithmetic",
    "FiniteField",
    "GF(2^m) field arithmetic",
    "In finite fields, infinite power there is.",
)
_r(
    "plyrn",
    "poly_ring_op",
    "FiniteField",
    "Polynomial ring operations",
    "Rings within rings, the galaxy turns. -- Chirrut",
)
_r(
    "nttfn",
    "ntt_transform",
    "FiniteField",
    "Number Theoretic Transform",
    "Transform your perspective, you must.",
)
_r("gf2ad", "gf2_matrix_add", "FiniteField", "GF(2) matrix addition", "Together, stronger they become.")
_r(
    "gf2ml",
    "gf2_matrix_mul",
    "FiniteField",
    "GF(2) matrix multiplication",
    "Multiply the connections, you must.",
)
_r(
    "gf2iv",
    "gf2_matrix_inv",
    "FiniteField",
    "GF(2) matrix inverse",
    "Distribution helper.",
)
_r(
    "irpol",
    "irreducible_poly",
    "FiniteField",
    "Irreducible polynomial over GF(2)",
    "Distribution helper.",
)

# -- ErrorCode --
_r(
    "hamcd",
    "hamming_code",
    "ErrorCode",
    "Hamming code encode/decode",
    "Even with errors, the message gets through. --",
)
_r("goppa", "goppa_code", "ErrorCode", "Binary Goppa code construction", "Hidden in the code, the truth is.")
_r(
    "ldpcd",
    "ldpc_decode",
    "ErrorCode",
    "LDPC bit-flipping decoder",
    "Bit by bit, the message reveals itself. -- Chirrut",
)
_r("ldpce", "ldpc_encode", "ErrorCode", "LDPC encoding", "Encode the transmission, we must. -- Admiral Ackbar")
_r("syndc", "syndrome_compute", "ErrorCode", "Syndrome computation", "The syndrome reveals the error. -- Medical Droid")
_r(
    "genpk",
    "gen_parity_check",
    "ErrorCode",
    "Generator/parity-check matrix",
    "A matrix of truth, the parity check is.",
)
_r(
    "ldpcg",
    "ldpc_generate",
    "ErrorCode",
    "LDPC matrix generation",
    "What is now proved was once only imagined. — William Blake",
)

# -- HashSig --
_r("lamp", "lamport_sign", "HashSig", "Lamport one-time signature", "One chance to sign, you have.")
_r("lampv", "lamport_verify", "HashSig", "Lamport signature verify", "Verify the signature, you must.")
_r("wots", "wots_sign", "HashSig", "Winternitz OTS sign", "What is now proved was once only imagined. — William Blake")
_r("wotsv", "wots_verify", "HashSig", "Winternitz OTS verify", "Trust the chain, you must.")
_r("mktre", "merkle_tree", "HashSig", "Merkle tree construction", "Strong roots, a strong tree makes.")
_r("xmss", "xmss_sign", "HashSig", "XMSS multi-tree signature", "Many signatures from one tree. -- Chirrut")

# -- Utility --
_r("pdftx", "pdf_to_text", "Utility", "PDF text extraction", "The archives, incomplete they are. -- Jocasta Nu")

# -- BioSignal Filter --
_r("movav", "moving_average", "Filter", "Moving average smoothing", "Smooth the path, you must.")
_r("wienr", "wiener_filter", "Filter", "Wiener optimal filter", "Distribution helper.")
_r("lmsaf", "lms_adaptive_filter", "Filter", "LMS adaptive filter", "Adapt or perish, the galaxy demands. -- Thrawn")
_r("nlmsf", "nlms_adaptive_filter", "Filter", "Normalized LMS filter", "Normalize the approach, you must.")
_r("rlsaf", "rls_adaptive_filter", "Filter", "RLS adaptive filter", "Recursion, the path to precision is.")
_r(
    "notch",
    "notch_filter_signal",
    "Filter",
    "Notch filter (powerline removal)",
    "Cut the interference, we must. -- Mace Windu",
)
_r(
    "combf",
    "comb_filter_signal",
    "Filter",
    "Comb filter (harmonic removal)",
    "Harmonics of the dark side, remove them.",
)
_r(
    "mtchf",
    "matched_filter_detect",
    "Filter",
    "Matched filter detection",
    "Match the template, the signal reveals. -- Chirrut",
)
_r(
    "wnflt",
    "wiener_filter",
    "Filter",
    "Wiener filter (optimal noise reduction)",
    "The noise is strong with this one. --",
)
_r(
    "mchfl",
    "matched_filter",
    "BiomedSignal",
    "Matched filter (template detection)",
    "This is the template you are looking for. --",
)
_r(
    "burgp",
    "burg_psd",
    "BiomedSignal",
    "Burg AR spectral estimation",
    "Always two there are -- a forward and backward error.",
)
_r(
    "coher",
    "coherence",
    "BiomedSignal",
    "Coherence between two signals",
    "Strong the connection between them is.",
)
_r(
    "wvdst",
    "wigner_ville",
    "BiomedSignal",
    "Wigner-Ville distribution",
    "Time and frequency, inseparable they are.",
)
_r(
    "hhtrf",
    "hilbert_huang_spectrum",
    "BiomedSignal",
    "Hilbert-Huang Transform (full spectrum)",
    "Distribution helper.",
)
_r(
    "emdsg",
    "emd",
    "BiomedSignal",
    "Empirical Mode Decomposition (standalone)",
    "Decompose the signal, you must.",
)
_r(
    "medf",
    "median_filter_signal",
    "Filter",
    "Median filter (impulse removal)",
    "The median path, safest it is.",
)
_r("ensav", "ensemble_avg", "Filter", "Ensemble averaging", "Together, stronger the signal becomes. --")
_r("sncav", "sync_avg", "Filter", "Synchronized averaging", "Synchronize the epochs, clarity emerges. --")
_r("snr", "snr_estimate_fn", "Signal", "SNR estimation (dB)", "Signal from noise, distinguish you must.")
_r(
    "snri",
    "snr_improvement_fn",
    "Signal",
    "SNR improvement measurement",
    "Improvement, the goal of all filtering is. -- Qui-Gon",
)

# -- BioSignal Detection --
_r(
    "thrdt",
    "threshold_detect",
    "Detection",
    "Threshold-based event detection",
    "Cross the threshold, events reveal themselves. -- Ahsoka",
)
_r(
    "drvdt",
    "derivative_detect",
    "Detection",
    "Derivative-based peak detection",
    "The rate of change, truth it reveals.",
)
_r(
    "zcr",
    "zero_crossing_rate",
    "Detection",
    "Zero-crossing rate",
    "Where the signal crosses zero, information lies. -- Chirrut",
)
_r(
    "tmplm",
    "template_match_detect",
    "Detection",
    "Template matching detection",
    "Distribution helper.",
)
_r(
    "onset",
    "onset_detect_fn",
    "Detection",
    "Signal onset detection",
    "The beginning of each event, detect you must.",
)
_r(
    "sheng",
    "shannon_energy_fn",
    "Detection",
    "Shannon energy operator",
    "Energy in the signal, Shannon quantifies. -- R2-D2",
)
_r("teagr", "teager_energy_fn", "Detection", "Teager energy operator", "Instantaneous energy, Teager reveals. -- C-3PO")
_r(
    "hlbrt",
    "hilbert_envelope_fn",
    "Detection",
    "Hilbert envelope extraction",
    "The analytic signal, Hilbert transforms.",
)

# -- BioSignal Waveform --
_r("rmssg", "rms_signal", "Waveform", "Root mean square", "The power of the signal, RMS measures.")
_r(
    "frmfc",
    "form_factor_fn",
    "Waveform",
    "Form factor (RMS/mean)",
    "Shape of the waveform, the form factor describes. -- Qui-Gon",
)
_r(
    "crstf",
    "crest_factor_fn",
    "Waveform",
    "Crest factor (peak/RMS)",
    "Peaks above the average, the crest reveals. --",
)
_r("shpfc", "shape_factor_fn", "Waveform", "Shape factor", "Every waveform has its shape. --")
_r(
    "wvlen",
    "waveform_length_fn",
    "Waveform",
    "Waveform length (total variation)",
    "The length of the path, measure you must.",
)
_r(
    "turns",
    "turns_count_fn",
    "Waveform",
    "Turns count (direction changes)",
    "Count the turns, complexity emerges. -- Mace Windu",
)
_r(
    "hjrth",
    "hjorth_params",
    "Waveform",
    "Hjorth parameters (activity/mobility/complexity)",
    "Three parameters, the signal they describe.",
)
_r(
    "myopr",
    "myopulse_rate_fn",
    "Waveform",
    "Myopulse percentage rate",
    "Pulses above threshold, the muscle speaks. -- Chirrut",
)
_r("wlamp", "willison_amp", "Waveform", "Willison amplitude", "What is now proved was once only imagined. — William Blake")
_r(
    "sscfn",
    "slope_sign_changes_fn",
    "Waveform",
    "Slope sign changes",
    "Direction reversal, the signal reveals. -- Ahsoka",
)
_r("amhst", "amplitude_hist", "Waveform", "Amplitude histogram", "Distribute the amplitudes, patterns emerge. --")
_r("enhst", "entropy_hist", "Waveform", "Histogram entropy", "Disorder in the signal, entropy measures.")
_r("iemg", "integrated_emg_fn", "Waveform", "Integrated EMG", "Sum the absolute, muscle effort it reveals. -- Chirrut")
_r("mav", "mean_abs_value", "Waveform", "Mean absolute value", "The average magnitude, simple yet powerful. --")

# -- BioSignal Spectral --
_r(
    "prdgm",
    "periodogram_estimate",
    "Spectral",
    "Periodogram PSD estimation",
    "Distribution helper.",
)
_r(
    "brtlt",
    "bartlett_psd_fn",
    "Spectral",
    "Bartlett PSD estimation",
    "Average the spectra, variance decreases. -- Thrawn",
)
_r(
    "spmom",
    "spectral_moment_fn",
    "Spectral",
    "Spectral moment (nth order)",
    "Moments of the spectrum, shape they define.",
)
_r(
    "mnfrq",
    "mean_freq",
    "Spectral",
    "Mean frequency (spectral centroid)",
    "Center of spectral mass, the mean frequency is. --",
)
_r("mdfrq", "median_freq", "Spectral", "Median frequency", "Half the power above, half below. -- Qui-Gon")
_r(
    "sefrq",
    "spectral_edge_freq",
    "Spectral",
    "Spectral edge frequency",
    "Where most power lies, the edge defines. -- Mace Windu",
)
_r(
    "sprto",
    "spectral_ratio",
    "Spectral",
    "Spectral power ratio (band1/band2)",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "spcfl",
    "spectral_flatness_fn",
    "Spectral",
    "Spectral flatness (Wiener entropy)",
    "Flat the spectrum, noise-like it is.",
)
_r(
    "spcen",
    "spectral_entropy_fn",
    "Spectral",
    "Spectral entropy",
    "Disorder in frequency, spectral entropy measures. -- R2-D2",
)
_r(
    "psddb",
    "psd_decibels",
    "Spectral",
    "PSD to decibels conversion",
    "In decibels, clearer the spectrum becomes. -- C-3PO",
)
_r(
    "acfps",
    "acf_from_psd_fn",
    "Spectral",
    "Autocorrelation from PSD (Wiener-Khinchin)",
    "From spectrum to time, the transform bridges.",
)
_r(
    "bndpw",
    "band_power_fn",
    "Spectral",
    "Band power (frequency band integration)",
    "Power in the band, integrate you must.",
)

# -- BioSignal Modeling --
_r(
    "aryw",
    "ar_yule_walker_fn",
    "Modeling",
    "AR Yule-Walker estimation",
    "Walk the autocorrelation, AR coefficients emerge.",
)
_r("arbrg", "ar_burg_fn", "Modeling", "AR Burg estimation", "Forward and backward, Burg estimates. -- Thrawn")
_r(
    "arcov",
    "ar_covariance_fn",
    "Modeling",
    "AR covariance method",
    "Covariance of the signal, the model captures. --",
)
_r(
    "lvndr",
    "levinson_durbin_fn",
    "Modeling",
    "Levinson-Durbin recursion",
    "Recurse through the autocorrelation, efficient it is.",
)
_r(
    "arspc",
    "ar_spectrum_fn",
    "Modeling",
    "AR model power spectrum",
    "From model to spectrum, smooth the estimate. -- Qui-Gon",
)
_r(
    "arprd",
    "ar_predict_fn",
    "Modeling",
    "AR prediction (n-step ahead)",
    "Predict the future, the model attempts.",
)
_r(
    "arord",
    "ar_order_select",
    "Modeling",
    "AR optimal order selection (AIC/BIC/FPE)",
    "Choose the right order, parsimony demands. -- Mace Windu",
)
_r(
    "refco",
    "reflection_coeff_fn",
    "Modeling",
    "Reflection coefficients (PARCOR)",
    "Reflect the lattice, coefficients emerge. -- Chirrut",
)
_r(
    "prcor",
    "parcor_fn",
    "Modeling",
    "Partial autocorrelation (PARCOR)",
    "Partial the correlation, direct effects it reveals.",
)

# -- BioSignal TimeFreq --
_r(
    "stfta",
    "stft_analysis",
    "TimeFreq",
    "Short-time Fourier transform",
    "Windowed the signal, time and frequency unite.",
)
_r(
    "istfa",
    "istft_synth",
    "TimeFreq",
    "Inverse STFT synthesis",
    "From spectrum back to signal, reconstruct you must. --",
)
_r(
    "spgrm",
    "spectrogram_fn",
    "TimeFreq",
    "Spectrogram (STFT power)",
    "Power across time and frequency, the spectrogram shows. --",
)
_r(
    "wvd",
    "wigner_ville_fn",
    "TimeFreq",
    "Wigner-Ville distribution",
    "Highest resolution, but cross-terms beware. -- Thrawn",
)
_r(
    "chwld",
    "choi_williams_fn",
    "TimeFreq",
    "Choi-Williams distribution",
    "Suppress the cross-terms, Williams and Choi. -- Qui-Gon",
)
_r(
    "semfn",
    "spectral_error_fn",
    "TimeFreq",
    "Spectral error measure",
    "Error between spectra, segmentation guides. -- R2-D2",
)
_r(
    "acfds",
    "acf_dist",
    "TimeFreq",
    "ACF-based segment distance",
    "Distance in autocorrelation, stationarity tests.",
)
_r("glrcd", "glr_change", "TimeFreq", "GLR change-point detection", "Where the signal changes, GLR detects. -- Ahsoka")
_r(
    "klmnf",
    "kalman_fn",
    "TimeFreq",
    "Kalman filter (state estimation)",
    "Predict and update, the Kalman way. --",
)

# -- BioSignal Decomposition --
_r(
    "mprst",
    "matching_pursuit_fn",
    "Decomposition",
    "Matching pursuit decomposition",
    "Atom by atom, the signal decomposes.",
)
_r(
    "omprs",
    "omp_fn",
    "Decomposition",
    "Orthogonal matching pursuit",
    "Orthogonal the pursuit, sparser the result. -- Thrawn",
)
_r(
    "emdfn",
    "emd_fn",
    "Decomposition",
    "Empirical mode decomposition (EMD/IMF)",
    "Intrinsic modes, the signal contains. -- Chirrut",
)
_r(
    "nmffn",
    "nmf_fn",
    "Decomposition",
    "Non-negative matrix factorization",
    "Non-negative the parts, the whole they compose.",
)

_r(
    "trnpt",
    "turning_points_test_fn",
    "BioSignal Filter",
    "Turning points stationarity test",
    "Distribution helper.",
)
_r(
    "coefv",
    "coefficient_of_variation_fn",
    "BioSignal Filter",
    "Coefficient of variation (CV)",
    "Variation reveals the truth. -- Qui-Gon",
)
_r(
    "atmfl",
    "alpha_trimmed_mean_fn",
    "BioSignal Filter",
    "Alpha-trimmed mean filter",
    "Trim the excess, the truth remains. --",
)
_r(
    "hannf",
    "hann_filter_fn",
    "BioSignal Filter",
    "Hann/Hanning windowed filter",
    "A smooth window reveals clarity. -- Ahsoka",
)
_r(
    "wnhpf",
    "wiener_hopf_fn",
    "BioSignal Filter",
    "Wiener-Hopf matrix solver",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "xcorr",
    "cross_correlation_fn",
    "BioSignal Filter",
    "Cross-correlation with lag",
    "Correlate the signals, you must.",
)
_r(
    "evnod",
    "even_odd_decompose_fn",
    "BioSignal Filter",
    "Even/odd signal decomposition",
    "Balance in all things. -- Bendu",
)

_r(
    "pntmp",
    "pan_tompkins_fn",
    "BioSignal Detection",
    "Pan-Tompkins QRS detector",
    "Find the heartbeat, you will.",
)
_r(
    "dcrtc",
    "dicrotic_notch_fn",
    "BioSignal Detection",
    "Dicrotic notch detection",
    "The notch reveals the reflection. -- Medical Droid",
)
_r("twave", "t_wave_detect_fn", "BioSignal Detection", "T-wave endpoint detection", "Every wave has its end. -- Luke")
_r(
    "cohsp",
    "coherence_spectrum_fn",
    "BioSignal Spectral",
    "Magnitude-squared coherence",
    "Coherence binds two signals. -- Chirrut",
)
_r(
    "xspec",
    "cross_spectral_fn",
    "BioSignal Spectral",
    "Cross-spectral density",
    "Cross the spectra, the truth emerges. -- Mace Windu",
)
_r("hrrr", "heart_rate_rr_fn", "BioSignal Detection", "Heart rate from RR intervals", "The rhythm of life. -- Padme")
_r(
    "hmflt",
    "homomorphic_filter_fn",
    "BioSignal Filter",
    "Homomorphic filter (log domain)",
    "Transform, filter, return.",
)
_r(
    "cxcep",
    "complex_cepstrum_fn",
    "BioSignal Spectral",
    "Complex cepstrum analysis",
    "In the cepstral domain, echoes reveal themselves. --",
)

_r(
    "sglen",
    "signal_arc_length_fn",
    "BioSignal Waveform",
    "Signal arc length",
    "The length of the path matters. -- Qui-Gon",
)
_r(
    "cntrt",
    "centroidal_time_fn",
    "BioSignal Waveform",
    "Centroidal time of signal energy",
    "The center of energy, find you must.",
)
_r(
    "mnphs",
    "minimum_phase_fn",
    "BioSignal Waveform",
    "Minimum phase correspondent",
    "Minimum delay, maximum clarity. -- Ahsoka",
)
_r(
    "higfd",
    "higuchi_fd_fn",
    "BioSignal Fractal",
    "Higuchi fractal dimension",
    "Complexity in simplicity, fractal reveals.",
)
_r(
    "boxfd",
    "box_counting_fd_fn",
    "BioSignal Fractal",
    "Box-counting fractal dimension",
    "Count the boxes, measure the chaos. -- Thrawn",
)
_r(
    "rulfd",
    "ruler_fd_fn",
    "BioSignal Fractal",
    "Ruler fractal dimension",
    "Measure with ever-shorter rulers. -- Chirrut",
)
_r(
    "przpd",
    "parzen_pdf_fn",
    "BioSignal Waveform",
    "Parzen window PDF estimation",
    "Smooth the density, the shape emerges. --",
)
_r(
    "cxdmd",
    "complex_demod_fn",
    "BioSignal Waveform",
    "Complex demodulation (AM/FM)",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "qrswv",
    "qrs_waveform_fn",
    "BioSignal Waveform",
    "QRS waveform morphology features",
    "Every heartbeat tells a story. -- Padme",
)
_r(
    "bscor",
    "baseline_corr_fn",
    "BioSignal Waveform",
    "Baseline-corrected correlation",
    "Remove the baseline, the truth remains. -- Luke",
)

_r(
    "fdspc",
    "fractal_dim_psd_fn",
    "BioSignal Fractal",
    "Fractal dimension from PSD slope",
    "The slope reveals the dimension.",
)
_r("spkrt", "spectral_kurtosis_fn", "BioSignal Spectral", "Spectral kurtosis", "Heavy tails, the spectrum has.")
_r(
    "winfn",
    "window_functions_fn",
    "BioSignal Spectral",
    "Window functions (Ham//Blk/Tri/Kai)",
    "Shape the window, shape the view. -- Ahsoka",
)
_r(
    "fbmsn",
    "fbm_synthesis_fn",
    "BioSignal Spectral",
    "Fractional Brownian motion synthesis",
    "Fractal motion, nature's pattern. -- Chirrut",
)

_r(
    "smuap",
    "smuap_fn",
    "BioSignal Modeling",
    "SMUAP point process EMG model",
    "Many motors, one signal. -- Admiral Ackbar",
)
_r(
    "armnr",
    "arma_newton_raphson_fn",
    "BioSignal Modeling",
    "ARMA Newton-Raphson estimation",
    "Iterate until convergence. -- Thrawn",
)
_r(
    "mywkr",
    "modified_yw_arma_fn",
    "BioSignal Modeling",
    "Modified Yule-Walker ARMA",
    "Walk the modified path. -- Qui-Gon",
)
_r(
    "stmcb",
    "steiglitz_mcbride_fn",
    "BioSignal Modeling",
    "Steiglitz-McBride iterative ARMA",
    "Iterate the estimate, refine the model. --",
)
_r("arfmt", "ar_formant_fn", "BioSignal Modeling", "AR formant extraction", "The poles reveal the formants. -- C-3PO")
_r("vcltr", "vocal_tract_fn", "BioSignal Modeling", "Vocal tract tube model", "The voice, a tube of resonance.")
_r(
    "hh",
    "hodgkin_huxley_fn",
    "BioSignal Modeling",
    "Hodgkin-Huxley neuron model",
    "Distribution helper.",
)
_r("bidmn", "bidomain_fn", "BioSignal Modeling", "Bidomain cardiac model", "Two domains, one heart. -- Padme")

_r(
    "rlslt",
    "rls_lattice_fn",
    "BioSignal Adaptive",
    "RLS lattice filter",
    "Lattice by lattice, the filter adapts. -- Chirrut",
)
_r(
    "cwtfn",
    "cwt_fn",
    "BioSignal Wavelet",
    "Continuous wavelet transform",
    "Scales reveal what frequency hides.",
)
_r(
    "spwvd",
    "smoothed_pwvd_fn",
    "BioSignal TimeFreq",
    "Smoothed pseudo-Wigner-Ville",
    "Smooth the cross-terms away. --",
)
_r(
    "lacun",
    "lacunarity_fn",
    "BioSignal Fractal",
    "Lacunarity measure",
    "Gaps in the pattern reveal structure. -- Thrawn",
)
_r(
    "bispc",
    "bispectrum_fn",
    "BioSignal Spectral",
    "Bispectrum / bicoherence",
    "Third order, the coupling reveals. -- Mace Windu",
)
_r("klmfn", "kalman_fn", "BioSignal Adaptive", "Kalman filter (fn/ wrapper)", "Predict, update, repeat.")

_r(
    "icafn",
    "ica_fn",
    "BioSignal Decomp",
    "Independent component analysis (ICA)",
    "Separate the sources, you must.",
)
_r("eemd", "ensemble_emd_fn", "BioSignal Decomp", "Ensemble EMD (EEMD)", "Noise-assisted, the modes emerge. -- Chirrut")
_r(
    "dctln",
    "dictionary_learning_fn",
    "BioSignal Decomp",
    "Dictionary learning (K-SVD)",
    "Learn the atoms, sparse becomes the signal. -- Thrawn",
)
_r(
    "mptfd",
    "mp_tfd_fn",
    "BioSignal TimeFreq",
    "Matching pursuit time-frequency",
    "Atom by atom, the time-frequency builds. --",
)
_r(
    "tfdft",
    "tfd_features_fn",
    "BioSignal TimeFreq",
    "TFD feature extraction",
    "Features from the joint domain. -- Ahsoka",
)
_r(
    "rcov",
    "riemannian_cov_fn",
    "BioSignal Decomp",
    "Riemannian covariance matrix",
    "On the manifold, distance is curved.",
)
_r(
    "ncfs",
    "ncfs_select_fn",
    "BioSignal Feature",
    "Neighborhood component feature selection",
    "Select the neighbors that matter. -- Mace Windu",
)

_r(
    "flda",
    "fisher_lda_fn",
    "BioSignal Classify",
    "Fisher linear discriminant analysis",
    "Distribution helper.",
)
_r(
    "mhlds",
    "mahalanobis_fn",
    "BioSignal Classify",
    "Mahalanobis distance classifier",
    "Distance in covariance space. -- Thrawn",
)
_r(
    "baysc",
    "bayes_classifier_fn",
    "BioSignal Classify",
    "Gaussian Bayes classifier",
    "Posterior probability, the way forward. --",
)
_r(
    "qda",
    "qda_fn",
    "BioSignal Classify",
    "Quadratic discriminant analysis",
    "Quadratic boundaries, the classes separate. -- Ahsoka",
)
_r(
    "logcl",
    "logistic_classify_fn",
    "BioSignal Classify",
    "Logistic regression classifier",
    "Logistic the boundary, sigmoid the probability. -- C-3PO",
)
_r(
    "rbfnn",
    "rbf_network_fn",
    "BioSignal Classify",
    "Radial basis function network",
    "Radial basis, neural response. -- Medical Droid",
)
_r(
    "cfmat",
    "confusion_matrix_fn",
    "BioSignal Metric",
    "Confusion matrix metrics",
    "Confusion reveals clarity. -- Chirrut",
)
_r("kfcvl", "kfold_cv_fn", "BioSignal Validate", "K-fold cross-validation", "Fold by fold, the truth emerges.")
_r("loocv", "loocv_fn", "BioSignal Validate", "Leave-one-out cross-validation", "Leave one out, test the rest. -- Luke")
_r(
    "bhatt",
    "bhattacharyya_fn",
    "BioSignal Metric",
    "Bhattacharyya divergence",
    "Divergence measures the gap. -- Mace Windu",
)
_r(
    "cnnbs",
    "cnn_biosignal_fn",
    "BioSignal DeepLearn",
    "1D CNN for biosignals",
    "Convolve the signal, learn the pattern. -- Rey",
)
_r(
    "lstmb",
    "lstm_biosignal_fn",
    "BioSignal DeepLearn",
    "LSTM for sequential biosignals",
    "Remember the sequence, predict the future. --",
)

_r("ecgplt", "ecg_plot_fn", "BioSignal Vis", "Multi-lead ECG plot", "Twelve leads, one heart. -- Padme")
_r("eegplt", "eeg_montage_fn", "BioSignal Vis", "Multi-channel EEG montage plot", "Many channels, one mind.")
_r("sigplt", "signal_plot_fn", "BioSignal Vis", "Generic signal waveform plot", "See the waveform, you must.")
_r(
    "fltplt",
    "filter_io_fn",
    "BioSignal Vis",
    "Filter input/output comparison plot",
    "Before and after, the filter reveals. --",
)
_r(
    "spcplt",
    "spectrum_plot_fn",
    "BioSignal Vis",
    "Power spectrum plot",
    "The spectrum shows all frequencies. -- Ahsoka",
)
_r(
    "sgmplt",
    "spectrogram_plot_fn",
    "BioSignal Vis",
    "Spectrogram heatmap plot",
    "Time and frequency, united. -- Chirrut",
)
_r(
    "scaplt",
    "scalogram_plot_fn",
    "BioSignal Vis",
    "Wavelet scalogram plot",
    "Scale by scale, the detail emerges. -- Thrawn",
)
_r(
    "tfplt",
    "tfd_plot_fn",
    "BioSignal Vis",
    "Time-frequency distribution plot",
    "Joint domains reveal the truth. -- Mace Windu",
)
_r(
    "annplt",
    "annotated_signal_fn",
    "BioSignal Vis",
    "Annotated signal with event markers",
    "Mark the events, the pattern emerges. --",
)
_r(
    "imfplt",
    "imf_stack_fn",
    "BioSignal Vis",
    "IMF decomposition stack plot",
    "Layer by layer, decomposed the signal is.",
)
_r("arplt", "ar_poles_fn", "BioSignal Vis", "AR pole-zero + spectrum plot", "The poles tell the story. -- C-3PO")
_r(
    "eegbd",
    "eeg_bands_fn",
    "BioSignal Vis",
    "EEG band decomposition plot",
    "Delta, theta, alpha, beta -- the rhythms of the mind.",
)

_r(
    "eegbf",
    "eeg_band_filter_fn",
    "BioSignal Clinical",
    "EEG band filter (delta/theta/alpha/beta/gamma)",
    "Filter the rhythms, clarity emerges.",
)
_r("eegbp", "eeg_band_power_fn", "BioSignal Clinical", "EEG band power analysis", "Power in every band. -- Mace Windu")
_r(
    "ecgsm",
    "ecg_simulate_fn",
    "BioSignal Clinical",
    "Synthetic 12-lead ECG generator",
    "Simulate the heart. -- Medical Droid",
)
_r(
    "hrvmt",
    "hrv_metrics_fn",
    "BioSignal Clinical",
    "HRV metrics (SDNN/RMSSD/pNN50)",
    "The rhythm of the heart reveals health. -- Padme",
)
_r(
    "rsprt",
    "respiratory_rate_fn",
    "BioSignal Clinical",
    "Respiratory rate estimation",
    "Breath by breath, life continues. -- Qui-Gon",
)

_r(
    "smean",
    "sample_mean",
    "Foundation",
    "Sample mean (x-bar = 1/N sum x(n))",
    "A new hope begins with a single value. --",
)
_r(
    "svar",
    "sample_variance",
    "Foundation",
    "Sample variance (s^2)",
    "Variance, the measure of uncertainty it is.",
)
_r("sstd", "sample_std", "Foundation", "Sample standard deviation (s)", "Standard, the deviation must be.")
_r("smse", "mean_squared_error", "Foundation", "Mean squared error (MSE)", "Error leads to learning. -- Qui-Gon")
_r(
    "srmse",
    "root_mean_squared_error",
    "Foundation",
    "Root mean squared error (RMSE)",
    "At the root of error, truth you find.",
)
_r(
    "smae",
    "mean_absolute_error",
    "Foundation",
    "Mean absolute error (MAE)",
    "Absolute, the truth must be. -- Mace Windu",
)
_r("srms", "rms_value", "Foundation", "Root mean square value (RMS)", "What is now proved was once only imagined. — William Blake")
_r("sener", "signal_energy", "Foundation", "Signal energy (E = sum |x|^2)", "Energy surrounds us, binds us.")
_r(
    "spowr",
    "signal_power",
    "Foundation",
    "Signal power (P = 1/N sum |x|^2)",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "ssnr",
    "snr_compute",
    "Foundation",
    "Signal-to-noise ratio in dB",
    "Through the noise, the signal you must find.",
)
_r(
    "ssnri",
    "snr_improvement",
    "Foundation",
    "SNR improvement factor (sqrt(M) for sync averaging)",
    "Together, stronger we become.",
)
_r(
    "smom1",
    "raw_moment",
    "Foundation",
    "k-th raw moment (m_k = 1/N sum x^k)",
    "What is now proved was once only imagined. — William Blake",
)
_r("smom2", "central_moment", "Foundation", "k-th central moment (mu_k)", "Centered you must be, always.")
_r(
    "sskew",
    "skewness_coeff",
    "Foundation",
    "Skewness coefficient (gamma_1 = mu_3/sigma^3)",
    "Distribution helper.",
)
_r(
    "skurt",
    "kurtosis_coeff",
    "Foundation",
    "Excess kurtosis (gamma_2 = mu_4/sigma^4 - 3)",
    "Heavy the tails, when excess there is.",
)
_r(
    "szcr",
    "zero_crossing_rate",
    "Foundation",
    "Zero crossing rate (ZCR)",
    "Cross the zero, polarity changes. --",
)
_r("smcr", "mean_crossing_rate", "Foundation", "Mean crossing rate", "Around the mean, oscillations there are.")
_r("spp", "peak_to_peak", "Foundation", "Peak-to-peak amplitude", "From peak to valley, the range spans. -- Chirrut")
_r(
    "scrst",
    "crest_factor",
    "Foundation",
    "Crest factor (max|x|/RMS)",
    "The crest reveals the peaks true nature. -- Thrawn",
)
_r("sfrmf", "form_factor", "Foundation", "Form factor (RMS/mean|x|)", "Form follows function. -- Mon Mothma")
_r("npowr", "noise_power", "Foundation", "Noise power estimation", "Distribution helper.")
_r(
    "npsd",
    "noise_psd",
    "Foundation",
    "Noise power spectral density",
    "Flat the noise, across all frequencies. -- C-3PO",
)
_r(
    "whtns",
    "white_noise_gen",
    "Foundation",
    "White Gaussian noise generator",
    "Random, the static of the universe is.",
)
_r(
    "clrns",
    "colored_noise_gen",
    "Foundation",
    "Colored (1/f^alpha) noise generator",
    "Colored the noise, a pattern it has. -- Ahsoka",
)
_r("snrdb", "snr_to_linear", "Foundation", "SNR dB to linear conversion", "Transform the scale, you must.")
_r(
    "addns",
    "add_noise",
    "Foundation",
    "Add Gaussian noise at specified SNR",
    "Add noise, test resilience you will.",
)
_r(
    "scov",
    "sample_covariance",
    "Foundation",
    "Sample covariance cov(x,y)",
    "Together they vary, a connection reveals. --",
)
_r(
    "scorm",
    "covariance_matrix",
    "Foundation",
    "Full NxN covariance matrix",
    "What is now proved was once only imagined. — William Blake",
)
_r("prsn", "pearson_corr", "Foundation", "Pearson product-moment correlation", "Correlated their fates are.")
_r(
    "nxcor",
    "normalized_xcorr",
    "Foundation",
    "Normalized cross-correlation (peak + lag)",
    "Align the signals, the truth emerges. -- Chirrut",
)
_r(
    "ensav",
    "ensemble_average",
    "Foundation",
    "Ensemble (synchronized) average",
    "Together, the noise cancels. --",
)
_r(
    "ensrv",
    "ensemble_variance",
    "Foundation",
    "Ensemble variance across segments",
    "Variance across the ensemble. -- C-3PO",
)
_r("mvavg", "moving_average", "Foundation", "Simple moving average filter", "Smooth the path forward. -- Qui-Gon")
_r(
    "emavg",
    "exponential_ma",
    "Foundation",
    "Exponential moving average (EMA)",
    "Recent, the most weight receives.",
)
_r(
    "sduty",
    "duty_cycle",
    "Foundation",
    "Duty cycle (fraction above threshold)",
    "Active time, the duty reveals. -- Admiral Ackbar",
)
_r("spkfc", "peak_factor", "Foundation", "Peak factor (max|x|/RMS)", "What is now proved was once only imagined. — William Blake")
_r(
    "sactv",
    "activity",
    "Foundation",
    "Hjorth Activity parameter (var(x))",
    "Active the signal, when variance high.",
)
_r(
    "smobl",
    "mobility",
    "Foundation",
    "Hjorth Mobility parameter (sqrt(var(x')/var(x)))",
    "Mobile the signal, when derivatives vary. -- Qui-Gon",
)
_r(
    "scmpl",
    "complexity",
    "Foundation",
    "Hjorth Complexity parameter (mob(x')/mob(x))",
    "Complex the signal, when change accelerates. -- Thrawn",
)
_r(
    "sentr",
    "sample_entropy",
    "Foundation",
    "Sample entropy (regularity/complexity)",
    "What is now proved was once only imagined. — William Blake",
)

_r("dft", "dft_compute", "Fourier", "Discrete Fourier Transform", "Distribution helper.")
_r(
    "idft",
    "idft_compute",
    "Fourier",
    "Inverse Discrete Fourier Transform",
    "In my experience there is no such thing as luck. --",
)
_r(
    "dtft",
    "dtft_compute",
    "Fourier",
    "Discrete-Time Fourier Transform",
    "Your focus determines your reality. -- Qui-Gon",
)
_r(
    "magsp",
    "magnitude_spectrum",
    "Fourier",
    "Magnitude spectrum |X(k)|",
    "What is now proved was once only imagined. — William Blake",
)
_r("phssp", "phase_spectrum", "Fourier", "Phase spectrum angle(X(k))", "Do. Or do not. There is no try.")
_r(
    "logmg",
    "log_magnitude_spectrum",
    "Fourier",
    "Log magnitude spectrum (dB)",
    "What is now proved was once only imagined. — William Blake",
)
_r("pwrsp", "power_spectrum", "Fourier", "Power spectrum |X(k)|^2/N", "What is now proved was once only imagined. — William Blake")
_r("psdpr", "periodogram", "Fourier", "PSD via periodogram method", "What is now proved was once only imagined. — William Blake")
_r("psdbt", "bartlett_psd", "Fourier", "Bartlett averaged periodogram PSD", "You were the chosen one! --")
_r("xpsd", "cross_psd", "Fourier", "Cross power spectral density", "Luminous beings are we.")
_r(
    "spm0",
    "spectral_moment",
    "Fourier",
    "k-th spectral moment",
    "The ability to speak does not make you intelligent. -- Qui-Gon",
)
_r("mnfrq", "mean_frequency", "Fourier", "Mean frequency from spectral moments", "Who's the more foolish? --")
_r(
    "mdfrq",
    "median_frequency",
    "Fourier",
    "Median frequency of power spectrum",
    "Distribution helper.",
)
_r("bndw", "bandwidth_compute", "Fourier", "Spectral bandwidth", "The strongest stars have hearts of kyber. -- Chirrut")
_r("spedf", "spectral_edge_freq", "Fourier", "Spectral edge frequency", "Distribution helper.")
_r("spent", "spectral_entropy", "Fourier", "Spectral entropy", "There is always a bigger fish. -- Qui-Gon")
_r(
    "spflm",
    "spectral_flatness",
    "Fourier",
    "Spectral flatness (Wiener entropy)",
    "So this is how liberty dies. -- Padme",
)
_r(
    "sprof",
    "spectral_rolloff",
    "Fourier",
    "Spectral rolloff frequency",
    "Many truths depend on our point of view. --",
)
_r(
    "spcnt",
    "spectral_centroid",
    "Fourier",
    "Spectral centroid",
    "Strike me down and I become more powerful. --",
)
_r(
    "spslp",
    "spectral_slope",
    "Fourier",
    "Spectral slope of log magnitude",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "prsv",
    "parseval_verify",
    "Fourier",
    "Parseval's theorem verification",
    "Let the past die. Kill it if you have to. -- Kylo",
)
_r(
    "cnvth",
    "convolution_theorem_verify",
    "Fourier",
    "Convolution theorem verification",
    "What is now proved was once only imagined. — William Blake",
)
_r("lncon", "linear_convolution", "Fourier", "Linear convolution y=x*h", "The garbage will do! -- Rey")
_r("crcon", "circular_convolution", "Fourier", "Circular convolution via DFT", "I don't like sand. -- Anakin")
_r("oladd", "overlap_add", "Fourier", "Overlap-add fast convolution", "It's a trap! -- Admiral Ackbar")
_r("olsav", "overlap_save", "Fourier", "Overlap-save fast convolution", "You can't stop the change. -- Shmi")
_r("acfbi", "acf_biased", "Fourier", "Biased autocorrelation function", "Always pass on what you have learned.")
_r(
    "acfub",
    "acf_unbiased",
    "Fourier",
    "Unbiased autocorrelation function",
    "Distribution helper.",
)
_r(
    "ccfn",
    "ccf_normalized",
    "Fourier",
    "Normalized cross-correlation function",
    "The belonging you seek is ahead. -- Maz Kanata",
)
_r("wnrec", "rectangular_window", "Fourier", "Rectangular window", "What is now proved was once only imagined. — William Blake")
_r("wnhan", "hanning_window", "Fourier", "Hanning (Hann) window", "We are what they grow beyond.")
_r("wnhmm", "hamming_window", "Fourier", "Hamming window", "Fear is the path to the dark side.")
_r("wnblk", "blackman_window", "Fourier", "Blackman window", "What is now proved was once only imagined. — William Blake")
_r(
    "wnksr",
    "kaiser_window",
    "Fourier",
    "Kaiser window with beta parameter",
    "Every generation has a legend. -- tagline",
)
_r("wnbrt", "bartlett_window", "Fourier", "Bartlett triangular window", "Never underestimate a droid. -- K-2SO")
_r(
    "rceps",
    "real_cepstrum",
    "Fourier",
    "Real cepstrum c(n)=IDFT(log|DFT(x)|)",
    "Distribution helper.",
)
_r("prcep", "power_cepstrum", "Fourier", "Power cepstrum", "A long time ago in a galaxy far, far away. -- crawl")
_r("mfcc", "mel_cepstral_coeffs", "Fourier", "Mel-frequency cepstral coefficients", "Hope is like the sun. --")
_r(
    "fndmn",
    "fundamental_freq",
    "Fourier",
    "Fundamental frequency F0 estimation",
    "You were my brother, Anakin! --",
)
_r("hrmnc", "harmonic_ratio", "Fourier", "Harmonics-to-noise ratio (HNR)", "I know what I have to do. -- Kylo")

_r(
    "bwflt",
    "butterworth_filter",
    "FilterDesign",
    "Butterworth IIR filter",
    "Distribution helper.",
)
_r(
    "chbf1",
    "chebyshev1_filter",
    "FilterDesign",
    "Chebyshev Type I filter",
    "What is now proved was once only imagined. — William Blake",
)
_r("chbf2", "chebyshev2_filter", "FilterDesign", "Chebyshev Type II filter", "What is now proved was once only imagined. — William Blake")
_r("elflt", "elliptic_filter", "FilterDesign", "Elliptic (Cauer) IIR filter", "So this is how liberty dies. -- Padme")
_r("ntchf", "notch_filter", "FilterDesign", "Notch (band-reject) filter", "Fear leads to anger.")
_r("cmbfl", "comb_filter", "FilterDesign", "Feedforward comb filter", "Luminous beings are we.")
_r("bpflt", "bandpass_filter", "FilterDesign", "Bandpass Butterworth filter", "Distribution helper.")
_r(
    "bsflt",
    "bandstop_filter",
    "FilterDesign",
    "Bandstop Butterworth filter",
    "In my experience, no such thing as luck. --",
)
_r(
    "firds",
    "fir_design",
    "FilterDesign",
    "FIR filter design via windowed sinc",
    "Your focus determines your reality. -- Qui-Gon",
)
_r(
    "iirds",
    "iir_design",
    "FilterDesign",
    "IIR filter design (returns b,a coefficients)",
    "There is always a bigger fish. -- Qui-Gon",
)
_r(
    "trfnc",
    "transfer_function",
    "TransferFunction",
    "H(w) frequency response from b,a",
    "The greatest teacher, failure is.",
)
_r(
    "imrsp",
    "impulse_response",
    "TransferFunction",
    "Impulse response h(n) from filter coefficients",
    "You must unlearn what you have learned.",
)
_r(
    "stprs",
    "step_response",
    "TransferFunction",
    "Step response s(n) from filter coefficients",
    "Patience you must have.",
)
_r(
    "grpdl",
    "group_delay",
    "TransferFunction",
    "Group delay tau_g(w) = -dphi/dw",
    "Truly wonderful, the mind of a child is.",
)
_r(
    "phsdl",
    "phase_delay",
    "TransferFunction",
    "Phase delay tau_p(w) = -phi(w)/w",
    "Much to learn, you still have.",
)
_r(
    "plzro",
    "poles_zeros",
    "TransferFunction",
    "Poles and zeros of transfer function H(z)",
    "Strike me down and I become more powerful. --",
)
_r(
    "stbck",
    "stability_check",
    "TransferFunction",
    "BIBO stability check (poles inside unit circle)",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "diffs",
    "differentiate_signal",
    "SignalOps",
    "Numerical signal differentiation",
    "Difficult to see. Always in motion.",
)
_r(
    "intgs",
    "integrate_signal",
    "SignalOps",
    "Cumulative trapezoidal integration",
    "The belonging you seek is ahead. -- Maz Kanata",
)
_r(
    "dcsub",
    "dc_removal",
    "SignalOps",
    "Remove DC component y = x - mean(x)",
    "These aren't the droids you're looking for. --",
)
_r("dtrnd", "detrend_signal", "SignalOps", "Polynomial detrending", "Who's the more foolish? --")
_r(
    "nrmlz",
    "normalize_signal",
    "SignalOps",
    "Z-score, min-max, or unit energy normalization",
    "We are what they grow beyond.",
)
_r("rsmpl", "resample_signal", "SignalOps", "Rational resampling by up/down factor", "Let the past die. -- Kylo")
_r("dwnsp", "downsample", "SignalOps", "Downsample by integer factor", "What is now proved was once only imagined. — William Blake")
_r("upsmp", "upsample", "SignalOps", "Upsample by integer factor (zero-insert)", "The garbage will do! -- Rey")
_r("sgpad", "pad_signal", "SignalOps", "Zero/mirror/reflect signal padding", "I know what I have to do. -- Kylo")
_r("sgtrm", "trim_signal", "SignalOps", "Extract segment from signal", "Stay on target. -- Gold Five")
_r(
    "sgspl",
    "split_signal",
    "SignalOps",
    "Split into non-overlapping segments",
    "This is where the fun begins. -- Anakin",
)
_r("sgfrm", "frame_signal", "SignalOps", "Overlapping frame extraction", "The dark side clouds everything.")
_r("lmsfl", "lms_filter", "AdaptiveFilter", "LMS adaptive filter", "Distribution helper.")
_r(
    "nlmsf",
    "nlms_filter",
    "AdaptiveFilter",
    "Normalized LMS adaptive filter",
    "You were my brother, Anakin. --",
)
_r("rlsfl", "rls_filter", "AdaptiveFilter", "RLS adaptive filter", "What is now proved was once only imagined. — William Blake")
_r(
    "ancrm",
    "anc_remove",
    "AdaptiveFilter",
    "Adaptive noise cancellation via LMS",
    "Many truths depend on our point of view. --",
)
_r("ztfrq", "freq_response_at", "ZTransform", "H(f) at specific frequencies", "Judge me by my size, do you?")
_r(
    "bltrf",
    "bilinear_transform",
    "ZTransform",
    "Analog to digital via bilinear transform",
    "Distribution helper.",
)
_r(
    "sqntz",
    "quantize_signal",
    "Quantization",
    "Uniform quantization to b bits",
    "Do. Or do not. There is no try.",
)
_r("qnter", "quantization_error", "Quantization", "Quantization noise e = x - Q(x)", "Distribution helper.")
_r("smpth", "sampling_theorem_check", "Sampling", "Nyquist criterion check fs >= 2*fmax", "It's a trap! -- Ackbar")
_r(
    "alias",
    "aliasing_demo",
    "Sampling",
    "Show aliased frequency when fs < 2*f",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "rcnst",
    "sinc_reconstruct",
    "Sampling",
    "Whittaker-Shannon sinc interpolation",
    "Distribution helper.",
)

_r("arywk", "ar_yule_walker", "Modeling", "AR Yule-Walker estimation", "Do or do not. There is no try.")
_r("arbrg", "ar_burg", "Modeling", "AR Burg max-entropy estimation", "Distribution helper.")
_r(
    "arcov",
    "ar_covariance",
    "Modeling",
    "AR covariance method estimation",
    "In my experience, no such thing as luck. --",
)
_r(
    "armcv",
    "ar_modified_cov",
    "Modeling",
    "AR modified covariance estimation",
    "Distribution helper.",
)
_r(
    "lvdrb",
    "levinson_durbin",
    "Modeling",
    "Levinson-Durbin recursion from ACF",
    "The belonging you seek is ahead. -- Maz Kanata",
)
_r(
    "rflcf",
    "reflection_coefficients",
    "Modeling",
    "AR to reflection (PARCOR) coefficients",
    "Your eyes can deceive you. --",
)
_r(
    "ltcfl",
    "lattice_coefficients",
    "Modeling",
    "AR to lattice filter coefficients",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "prder",
    "prediction_error",
    "Modeling",
    "Forward linear prediction error",
    "What is now proved was once only imagined. — William Blake",
)
_r("bkprd", "backward_prediction", "Modeling", "Backward linear prediction error", "Let the Wookiee win. -- C-3PO")
_r("arspd", "ar_spectrum", "Modeling", "PSD from AR model parameters", "What is now proved was once only imagined. — William Blake")
_r("maest", "ma_estimate", "Modeling", "MA coefficient estimation", "Wars not make one great.")
_r(
    "armae",
    "arma_estimate",
    "Modeling",
    "ARMA estimation via modified Yule-Walker",
    "So this is how liberty dies. -- Padme",
)
_r("prdny", "prony_method", "Modeling", "Prony pole-zero estimation", "Rebellions are built on hope. -- Jyn Erso")
_r(
    "aicsc",
    "aic_score",
    "ModelSelection",
    "Akaike Information Criterion model order",
    "The ability to speak does not make you intelligent. -- Qui-Gon",
)
_r(
    "bicsc",
    "bic_score",
    "ModelSelection",
    "Bayesian Information Criterion model order",
    "You were my brother, Anakin. --",
)
_r(
    "mdlsc",
    "mdl_score",
    "ModelSelection",
    "Minimum Description Length model order",
    "There is always a bigger fish. -- Qui-Gon",
)
_r(
    "fpesc",
    "fpe_score",
    "ModelSelection",
    "Final Prediction Error model order",
    "A long time ago in a galaxy far, far away. -- crawl",
)
_r(
    "catsc",
    "cat_score",
    "ModelSelection",
    "CAT criterion model order",
    "I've got a bad feeling about this. -- everyone",
)
_r(
    "lpcco",
    "lpc_coefficients",
    "Modeling",
    "Linear Prediction Coding coefficients",
    "Judge me by my size, do you?",
)
_r("lpcsp", "lpc_spectrum", "Modeling", "LPC-derived power spectrum", "The dark side clouds everything.")
_r(
    "lsf",
    "line_spectral_freq",
    "Modeling",
    "Line Spectral Frequencies from LPC",
    "You underestimate my power! -- Anakin",
)
_r("lar", "log_area_ratio", "Modeling", "Log Area Ratios from LPC", "Luminous beings are we.")
_r(
    "lpcep",
    "lpc_to_cepstral",
    "Modeling",
    "LPC to cepstral coefficient conversion",
    "These aren't the droids you're looking for. --",
)
_r(
    "mvdr",
    "mvdr_spectrum",
    "SpectralEstimation",
    "MVDR (Capon) spectral estimation",
    "Distribution helper.",
)
_r("music", "music_spectrum", "SpectralEstimation", "MUSIC spectral estimation", "Distribution helper.")
_r("esprt", "esprit_freq", "SpectralEstimation", "ESPRIT frequency estimation", "We are what they grow beyond.")
_r(
    "pisrl",
    "pisarenko",
    "SpectralEstimation",
    "Pisarenko harmonic decomposition",
    "Already know you that which you need.",
)
_r("argen", "ar_generate", "Modeling", "Generate AR process realisation", "Powerful you have become.")
_r("magen", "ma_generate", "Modeling", "Generate MA process realisation", "Stay on target. -- Gold Five")
_r("amgen", "arma_generate", "Modeling", "Generate ARMA process realisation", "This is the way. -- Mandalorian")
_r(
    "resac",
    "residual_acf",
    "Modeling",
    "Residual ACF for whiteness test",
    "Distribution helper.",
)
_r("ljbx", "ljung_box", "Modeling", "Ljung-Box portmanteau test", "Train yourself to let go.")
_r(
    "durbw",
    "durbin_watson",
    "Modeling",
    "Durbin-Watson autocorrelation statistic",
    "Who's the more foolish? --",
)
_r(
    "ssadc",
    "ssa_decompose",
    "Modeling",
    "Singular Spectrum Analysis decomposition",
    "Truly wonderful, the mind of a child is.",
)
_r(
    "ssarc",
    "ssa_reconstruct",
    "Modeling",
    "SSA reconstruction from components",
    "In a dark place we find ourselves.",
)
_r(
    "svspc",
    "subspace_decompose",
    "Modeling",
    "Signal/noise subspace separation via SVD",
    "You must unlearn what you have learned.",
)
_r(
    "hankl",
    "hankel_matrix",
    "Modeling",
    "Construct Hankel matrix from signal",
    "The greatest teacher, failure is.",
)
_r("sysid", "system_identify", "Modeling", "System identification via least squares", "It's a trap! -- Admiral Ackbar")
_r(
    "implz",
    "impulse_from_io",
    "Modeling",
    "Impulse response from I/O deconvolution",
    "Distribution helper.",
)
_r("cohfn", "coherence_function", "Modeling", "Magnitude-squared coherence function", "I have spoken. -- Kuiil")

_r(
    "bayes",
    "bayes_theorem",
    "Foundation",
    "Bayes theorem: P(A|B) = P(B|A)P(A)/P(B)",
    "In my experience there is no such thing as luck. --",
)
_r(
    "condp",
    "conditional_prob",
    "Foundation",
    "Conditional probability P(A|B) = P(A cap B)/P(B)",
    "Your focus determines your reality. -- Qui-Gon",
)
_r(
    "mle",
    "mle_gaussian",
    "Foundation",
    "MLE of mean and variance for Gaussian",
    "The dark side clouds everything.",
)
_r("mapst", "map_estimate", "Foundation", "MAP estimate with Gaussian prior", "Much to learn you still have.")
_r(
    "lkrat",
    "likelihood_ratio",
    "Foundation",
    "Likelihood ratio Lambda = L(t1)/L(t0)",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "nympn",
    "neyman_pearson",
    "Foundation",
    "Neyman-Pearson threshold for given false alarm rate",
    "This is where the fun begins. -- Anakin",
)
_r(
    "pdfen",
    "pdf_estimate",
    "Foundation",
    "Non-parametric PDF estimation",
    "Distribution helper.",
)
_r("cdfen", "cdf_estimate", "Foundation", "Empirical CDF", "Patience you must have.")
_r(
    "gmafn",
    "gemma_function_call",
    "LLM",
    "Gemma 4 native function calling via Ollama",
    "Distribution helper.",
)
_r("gmmpd", "gmm_pdf", "Foundation", "Gaussian Mixture Model PDF evaluation", "We are what they grow beyond.")
_r(
    "kdesm",
    "kde_smooth",
    "Foundation",
    "Kernel density estimation (Gaussian kernel)",
    "Luminous beings are we.",
)
_r(
    "shent",
    "shannon_entropy",
    "InfoTheory",
    "Shannon entropy H(X) = -sum p log2 p",
    "Do or do not. There is no try.",
)
_r("jnent", "joint_entropy", "InfoTheory", "Joint entropy H(X,Y)", "The belonging you seek is ahead. -- Maz Kanata")
_r(
    "cdent",
    "conditional_entropy",
    "InfoTheory",
    "Conditional entropy H(X|Y) = H(X,Y) - H(Y)",
    "In a dark place we find ourselves.",
)
_r("mutif", "mutual_information", "InfoTheory", "Mutual information I(X;Y)", "Let the past die. -- Kylo")
_r("xent", "cross_entropy", "InfoTheory", "Cross-entropy H(P,Q) = -sum p log q", "Distribution helper.")
_r("renyi", "renyi_entropy", "InfoTheory", "Renyi entropy of order alpha", "The greatest teacher, failure is.")
_r("eucd", "euclidean_dist", "Distance", "Euclidean (L2) distance", "You were the chosen one! --")
_r("minkd", "minkowski_dist", "Distance", "Minkowski distance of order p", "What is now proved was once only imagined. — William Blake")
_r(
    "chbyd",
    "chebyshev_dist",
    "Distance",
    "Chebyshev (L-infinity) distance",
    "Fear is the path to the dark side.",
)
_r(
    "cosds",
    "cosine_distance",
    "Distance",
    "Cosine distance d = 1 - cos(x,y)",
    "Rebellions are built on hope. -- Jyn Erso",
)
_r(
    "corrd",
    "correlation_dist",
    "Distance",
    "Correlation distance d = 1 - r(x,y)",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "helld",
    "hellinger_dist",
    "Distance",
    "Hellinger distance between distributions",
    "There is always a bigger fish. -- Qui-Gon",
)
_r(
    "emdd",
    "earth_mover_dist",
    "Distance",
    "Earth mover's (Wasserstein-1) distance",
    "Truly wonderful, the mind of a child is.",
)
_r("knnc", "knn_classify", "Classifier", "k-Nearest Neighbors classifier", "Stay on target. -- Gold Five")
_r(
    "minnr",
    "min_distance_classify",
    "Classifier",
    "Minimum distance to centroid classifier",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "prrcl",
    "precision_recall",
    "Classifier",
    "Precision, recall, and F1 score",
    "What is now proved was once only imagined. — William Blake",
)
_r("fscor", "f_score", "Classifier", "F-beta score", "So this is how liberty dies. -- Padme")
_r(
    "mccrr",
    "mcc_score",
    "Classifier",
    "Matthews Correlation Coefficient",
    "The ability to speak does not make you intelligent. -- Qui-Gon",
)
_r(
    "pcafd",
    "pca_features",
    "FeatureExtraction",
    "PCA feature extraction",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "ldafd",
    "lda_features",
    "FeatureExtraction",
    "Fisher's LDA feature extraction",
    "Distribution helper.",
)
_r(
    "fslct",
    "feature_select_variance",
    "FeatureExtraction",
    "Variance-based feature selection",
    "Judge me by my size, do you?",
)
_r(
    "ftstf",
    "f_test_features",
    "FeatureExtraction",
    "ANOVA F-test per feature for selection",
    "Your eyes can deceive you. --",
)
_r(
    "mirmf",
    "mrmr_score",
    "FeatureExtraction",
    "Minimum Redundancy Maximum Relevance scoring",
    "Distribution helper.",
)
_r("eigdp", "eigen_decompose", "LinearAlgebra", "Eigenvalues and eigenvectors", "Train yourself to let go.")
_r("svddp", "svd_decompose", "LinearAlgebra", "Singular Value Decomposition", "Distribution helper.")
_r("psinv", "pseudo_inverse", "LinearAlgebra", "Moore-Penrose pseudoinverse", "You underestimate my power! -- Anakin")
_r(
    "mxtrn",
    "matrix_trace_norm",
    "LinearAlgebra",
    "Matrix trace, det, rank, cond, norms",
    "Begun, the Clone War has.",
)
_r("toepz", "toeplitz_matrix", "LinearAlgebra", "Toeplitz matrix construction", "Distribution helper.")

_r(
    "sspace",
    "state_space",
    "TimeSeries",
    "State-space model via Kalman filter",
    "Always in motion is the future.",
)
_r("varfit", "var_fit", "TimeSeries", "Vector autoregression (VAR) via OLS", "Together we are stronger. -- Ahsoka")
_r(
    "medad",
    "mediation_analysis",
    "Causal",
    "Baron-Kenny mediation with Sobel test",
    "There is always a bigger fish. -- Qui-Gon",
)
_r(
    "modr",
    "moderation_analysis",
    "Causal",
    "Moderation (interaction) analysis via OLS",
    "The dark side clouds everything.",
)
_r(
    "cbnd",
    "causal_bounds",
    "Causal",
    "Manski partial identification bounds for ATE",
    "In my experience there is no such thing as luck. --",
)
_r(
    "itfer",
    "interference_effects",
    "Causal",
    "Spillover/interference effects under partial interference",
    "Distribution helper.",
)
_r(
    "scm",
    "structural_causal_model",
    "Causal",
    "Linear structural causal model with do-calculus",
    "Your focus determines your reality. -- Qui-Gon",
)
_r(
    "ivwak",
    "iv_weak_test",
    "Causal",
    "Weak instrument diagnostics (Stock-Yogo)",
    "You must unlearn what you have learned.",
)
_r(
    "cmprk",
    "competing_risks",
    "Survival",
    "Competing risks cumulative incidence (Aalen-Johansen)",
    "Many of the truths we cling to depend on our point of view. --",
)
_r(
    "frail",
    "frailty_model",
    "Survival",
    "Gamma shared frailty model",
    "The belonging you seek is ahead of you. -- Maz Kanata",
)
_r(
    "crmix",
    "cure_rate_model",
    "Survival",
    "Mixture cure model for long-term survivors",
    "No one is ever really gone. -- Luke",
)
_r(
    "cif",
    "cumulative_incidence",
    "Survival",
    "Non-parametric cumulative incidence function",
    "Hope is like the sun. --",
)
_r(
    "emfit",
    "em_fit",
    "MissingData",
    "EM algorithm for Gaussian mixture (handles missing data)",
    "Distribution helper.",
)
_r(
    "mipol",
    "mi_pool",
    "MissingData",
    "Multiple imputation pooling (Rubin rules)",
    "We are the spark that will light the fire. -- Poe",
)
_r(
    "mpat",
    "missing_pattern",
    "MissingData",
    "Missing data pattern analysis",
    "The truth is often what we make of it. --",
)
_r(
    "tost",
    "tost_test",
    "Test",
    "TOST equivalence test (two one-sided t-tests)",
    "What is now proved was once only imagined. — William Blake",
)
_r("exact", "exact_perm_test", "Test", "Exact permutation test for two samples", "Let the Wookiee win. -- C-3PO")
_r(
    "brn",
    "brunner_munzel",
    "NonparametricTest",
    "Brunner-Munzel test for stochastic equality",
    "I have a bad feeling about this. -- Everyone",
)
_r(
    "attn",
    "attention",
    "ML",
    "Scaled dot-product attention mechanism",
    "Distribution helper.",
)
_r(
    "sftmx",
    "softmax",
    "ML",
    "Softmax with temperature scaling",
    "Luminous beings are we, not this crude matter.",
)
_r("svmc", "svm_classify", "ML", "SVM classifier via simplified SMO", "This is the way. -- Din Djarin")
_r(
    "embd",
    "embedding_similarity",
    "ML",
    "Pairwise embedding similarity (cosine/euclidean)",
    "We are what they grow beyond.",
)
_r(
    "cgd",
    "conjugate_gradient",
    "Optimization",
    "Conjugate gradient descent (Fletcher-Reeves)",
    "Do or do not. There is no try.",
)
_r(
    "nmead",
    "nelder_mead",
    "Optimization",
    "Nelder-Mead simplex optimization",
    "The simplest approach is often the best. -- Qui-Gon",
)
_r(
    "mi",
    "mutual_info",
    "InfoTheory",
    "Mutual information I(X;Y) via histogram binning",
    "Distribution helper.",
)
_r(
    "entro",
    "entropy",
    "InfoTheory",
    "Shannon entropy and information measures",
    "In the end, cowards are those who follow the dark side.",
)
_r(
    "exrsp",
    "exposure_response",
    "EnvEpi",
    "Exposure-response curve via restricted cubic splines",
    "The environment around us is what shapes our destiny. -- Chirrut",
)
_r(
    "bmd",
    "benchmark_dose",
    "EnvEpi",
    "Benchmark dose estimation for dose-response",
    "Small in size but wise in years.",
)
_r(
    "phrx",
    "prescription_patterns",
    "PharmacoEpi",
    "Prescription filling patterns and adherence (MPR/PDC)",
    "I know what I have to do. -- Ben Solo",
)
_r(
    "nsccd",
    "new_user_cohort",
    "PharmacoEpi",
    "New-user active-comparator cohort design",
    "Every choice you have made has led you here. -- Snoke",
)

_r("tfidf", "tfidf", "NLP", "TF-IDF document vectorization", "Words are a sourcerer's weapon. -- Hux")
_r(
    "ngram",
    "ngram_freq",
    "NLP",
    "N-gram frequency analysis",
    "Your tongues can't repel frequency of that magnitude. -- Ackbar",
)
_r(
    "wrdcl",
    "word_cloud_data",
    "NLP",
    "Word frequency for cloud visualization",
    "A picture, worth a thousand words it is.",
)
_r("txtcl", "text_classify", "NLP", "Multinomial Naive Bayes text classifier", "Judge me by my size, do you?")
_r("sentn", "sentiment_lexicon", "NLP", "Lexicon-based sentiment scoring", "I sense great fear in you.")
_r(
    "fftpk",
    "fft_peaks",
    "SignalProcessing",
    "Dominant frequency peaks via FFT",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "lpflt",
    "lowpass_filter",
    "SignalProcessing",
    "Butterworth lowpass filter",
    "Let go of your high frequencies. --",
)
_r(
    "envlp",
    "envelope",
    "SignalProcessing",
    "Signal envelope via Hilbert transform",
    "Distribution helper.",
)
_r(
    "zcrss",
    "zero_crossings",
    "SignalProcessing",
    "Zero-crossing detection and counting",
    "Distribution helper.",
)
_r("stfft", "stft", "SignalProcessing", "Short-Time Fourier Transform", "Time it is, to look at frequencies.")
_r("lle", "lle", "DimReduction", "Locally Linear Embedding", "Locally we are connected. -- Chirrut")
_r(
    "spemb",
    "spectral_embed",
    "DimReduction",
    "Spectral embedding via Laplacian eigenmaps",
    "What is now proved was once only imagined. — William Blake",
)
_r("nmf", "nmf", "DimReduction", "Non-negative matrix factorization", "Distribution helper.")
_r(
    "isomp",
    "isomap",
    "DimReduction",
    "Isomap geodesic embedding",
    "The shortest path is not always the straightest. -- Qui-Gon",
)
_r(
    "fastp",
    "fast_pca",
    "DimReduction",
    "Randomized PCA (Halko-Martinsson-Tropp)",
    "Distribution helper.",
)
_r(
    "havsn",
    "haversine_distance",
    "Geospatial",
    "Haversine great-circle distance",
    "A great distance, the galaxy spans.",
)
_r(
    "gclus",
    "geo_cluster",
    "Geospatial",
    "K-means clustering on geographic coordinates",
    "Scattered across the galaxy, they are.",
)
_r("voron", "voronoi_areas", "Geospatial", "Voronoi tessellation areas", "Territory, each cell claims. -- Tarkin")
_r("smpbf", "spatial_buffer", "Geospatial", "Distance buffer zones around points", "A perimeter create.")
_r(
    "heatd",
    "heat_density",
    "Geospatial",
    "2D kernel density for heatmaps",
    "Where there is heat, there is life. -- Chirrut",
)
_r(
    "weibl",
    "weibull_fit",
    "Reliability",
    "Weibull distribution MLE with censoring",
    "The weakest link, find you must.",
)
_r(
    "mtbf",
    "mtbf_estimate",
    "Reliability",
    "Mean time between failures with CI",
    "Failure is the greatest teacher.",
)
_r(
    "avail",
    "availability",
    "Reliability",
    "System availability MTBF/(MTBF+MTTR)",
    "Distribution helper.",
)
_r("fmea", "fmea_rpn", "Reliability", "FMEA Risk Priority Number scoring", "Risk there is, in every mission.")
_r(
    "relgr",
    "reliability_growth",
    "Reliability",
    "Duane/AMSAA reliability growth model",
    "Grow stronger we do, with each failure.",
)
_r(
    "pgrnk",
    "pagerank",
    "Network",
    "PageRank centrality via power iteration",
    "Distribution helper.",
)
_r(
    "btwns",
    "betweenness",
    "Network",
    "Betweenness centrality (Brandes algorithm)",
    "Distribution helper.",
)
_r("clust", "clustering_coefficient", "Network", "Graph clustering coefficient", "Cluster together, they do.")
_r("cmpnt", "connected_components", "Network", "Connected components via BFS", "Connected we all are. -- Chirrut")
_r(
    "dgrds",
    "degree_distribution",
    "Network",
    "Degree distribution and power-law test",
    "Powerful friends, you have.",
)
_r("mm1q", "mm1_queue", "Queueing", "M/M/1 single-server queue metrics", "Patience, the queue requires.")
_r("mmcq", "mmc_queue", "Queueing", "M/M/c multi-server queue (Erlang-C)", "Many servers, one system. -- Thrawn")
_r("mg1q", "mg1_queue", "Queueing", "M/G/1 queue (Pollaczek-Khinchine)", "General service, the galaxy provides. -- Maz")
_r("qsim", "queue_simulate", "Queueing", "Discrete-event queue simulation", "Simulate the battle, we must.")
_r("litlw", "littles_law", "Queueing", "Little's law L = lambda * W", "Little things matter. --")
_r(
    "shrpe",
    "sharpe_ratio",
    "Finance",
    "Sharpe ratio for risk-adjusted returns",
    "Risk and reward, balance you must.",
)
_r(
    "maxdd",
    "max_drawdown",
    "Finance",
    "Maximum drawdown from peak",
    "The deeper the fall, the greater the lesson. -- Luke",
)
_r("ewmav", "ewma_volatility", "Finance", "EWMA volatility (RiskMetrics)", "Volatile, the markets are.")
_r(
    "valfn",
    "value_at_risk",
    "Finance",
    "Value at Risk (historical/parametric/CF)",
    "At risk, everything always is. -- Maz Kanata",
)
_r(
    "cvar",
    "conditional_var",
    "Finance",
    "Conditional VaR / Expected Shortfall",
    "Expect the worst, prepare you must.",
)
_r("latns", "latin_hypercube", "DoE", "Latin Hypercube Sampling", "Sample the galaxy wisely. -- Qui-Gon")
_r("facto", "factorial_design", "DoE", "Full factorial design matrix", "Every factor matters. -- Mon Mothma")
_r("rsm", "response_surface", "DoE", "Response surface methodology", "The surface reveals what lies beneath. -- Luke")
_r("optds", "optimal_design", "DoE", "D-optimal experimental design", "Optimal the design must be.")
_r("plakb", "plackett_burman", "DoE", "Plackett-Burman screening design", "Screen the factors, you must.")
_r("lifeq", "life_table_qx", "Actuarial", "Life table mortality rates", "Death is a natural part of life.")
_r("annty", "annuity_value", "Actuarial", "Present value of annuity", "Future value, patience reveals.")
_r("claim", "claim_frequency", "Actuarial", "Poisson claim frequency model", "Claims, there will always be.")
_r("rsrve", "chain_ladder", "Actuarial", "Chain-ladder loss reserving", "Reserve your strength, you must.")
_r("cprem", "credibility_premium", "Actuarial", "Buhlmann credibility premium", "Credible, your data must be.")

_r(
    "imhst",
    "image_histogram",
    "ImageProcessing",
    "Image histogram computation",
    "Distribution helper.",
)
_r(
    "imedg",
    "edge_detect",
    "ImageProcessing",
    "Edge detection (Sobel/Prewitt)",
    "In my experience there is no such thing as luck. --",
)
_r(
    "imblr",
    "gaussian_blur",
    "ImageProcessing",
    "Gaussian blur via convolution",
    "Your eyes can deceive you, don't trust them. --",
)
_r("imthr", "threshold", "ImageProcessing", "Otsu/adaptive thresholding", "Size matters not.")
_r(
    "imrsz",
    "image_resize",
    "ImageProcessing",
    "Bilinear interpolation resize",
    "Distribution helper.",
)
_r("cvxhl", "convex_hull", "Geometry", "Convex hull (gift wrapping)", "What is now proved was once only imagined. — William Blake")
_r("delny", "delaunay_simple", "Geometry", "Delaunay triangulation", "Distribution helper.")
_r(
    "ptinp",
    "point_in_polygon",
    "Geometry",
    "Point-in-polygon (ray casting)",
    "Distribution helper.",
)
_r("linit", "line_intersect", "Geometry", "Line segment intersection", "There is another.")
_r("polyA", "polygon_area", "Geometry", "Polygon area (shoelace formula)", "Do or do not.")
_r(
    "rlad",
    "lad_regression",
    "RobustRegression",
    "Least absolute deviations regression",
    "The ability to speak does not make you intelligent. -- Qui-Gon",
)
_r(
    "rlms",
    "lms_regression",
    "RobustRegression",
    "Least median of squares regression",
    "Wars not make one great.",
)
_r(
    "rlts",
    "lts_regression",
    "RobustRegression",
    "Least trimmed squares regression",
    "Mind tricks don't work on me. -- Watto",
)
_r(
    "mmest",
    "mm_estimator",
    "RobustRegression",
    "MM-estimator robust regression",
    "We are what they grow beyond.",
)
_r("sest", "s_estimator", "RobustRegression", "S-estimator for robust regression", "Let the past die. -- Kylo Ren")
_r(
    "cpbbs",
    "bbs_changepoint",
    "ChangePoint",
    "Binary segmentation change detection",
    "Rebellions are built on hope. -- Jyn Erso",
)
_r("cpelt", "pelt_changepoint", "ChangePoint", "PELT change-point algorithm", "This is the way. -- Din Djarin")
_r("cppgm", "changepoint_poisson", "ChangePoint", "Poisson change-point model", "I have spoken. -- Kuiil")
_r(
    "cpvar",
    "variance_changepoint",
    "ChangePoint",
    "Variance change-point detection",
    "Distribution helper.",
)
_r("cpmnr", "mean_changepoint", "ChangePoint", "Mean shift via CUSUM", "The greatest teacher, failure is.")
_r(
    "ctabl",
    "contingency_table",
    "CategoricalData",
    "Contingency table with chi-sq test",
    "Pass on what you have learned.",
)
_r(
    "crssp",
    "correspondence_analysis",
    "CategoricalData",
    "Simple correspondence analysis",
    "Your focus determines your reality. -- Qui-Gon",
)
_r(
    "logln",
    "loglinear_model",
    "CategoricalData",
    "Log-linear model for multi-way tables",
    "Distribution helper.",
)
_r("stord", "stuart_ord", "CategoricalData", "Stuart tau-c for ordinal association", "Trust your feelings. --")
_r(
    "ckmra",
    "cochran_mantel",
    "CategoricalData",
    "Cochran-Mantel-Haenszel test",
    "Distribution helper.",
)
_r(
    "expsm",
    "exponential_smooth",
    "Forecasting",
    "Simple exponential smoothing",
    "Distribution helper.",
)
_r(
    "holts",
    "holts_method",
    "Forecasting",
    "Holt linear trend method",
    "Truly wonderful, the mind of a child is.",
)
_r("hwint", "holt_winters", "Forecasting", "Holt-Winters seasonal forecasting", "Train yourself to let go.")
_r("thetf", "theta_method", "Forecasting", "Theta forecasting method", "Distribution helper.")
_r("naive", "naive_forecast", "Forecasting", "Naive/seasonal naive forecast", "Much to learn, you still have.")
_r("mahal", "mahalanobis_dist", "DistanceMeasures", "Mahalanobis distance", "Patience you must have.")
_r(
    "cheby",
    "chebyshev_dist",
    "DistanceMeasures",
    "Chebyshev (L-infinity) distance",
    "Already know you that which you need.",
)
_r("minkw", "minkowski_dist", "DistanceMeasures", "Minkowski distance", "Difficult to see, the future is.")
_r(
    "canbs",
    "canberra_dist",
    "DistanceMeasures",
    "Canberra distance",
    "Distribution helper.",
)
_r(
    "ertdm",
    "earth_movers_dist",
    "DistanceMeasures",
    "Earth movers / Wasserstein distance",
    "Named must be your fear, before banish it you can.",
)
_r(
    "hdint",
    "hdi",
    "IntervalEstimation",
    "Highest Density Interval (Bayesian)",
    "When you look at the dark side, careful you must be.",
)
_r(
    "bcaci",
    "bca_ci",
    "IntervalEstimation",
    "BCa bootstrap confidence interval",
    "Distribution helper.",
)
_r(
    "wlsci",
    "wilson_ci",
    "IntervalEstimation",
    "Wilson score interval",
    "Powerful you have become, the dark side I sense in you.",
)
_r(
    "cpci",
    "clopper_pearson",
    "IntervalEstimation",
    "Clopper-Pearson exact binomial CI",
    "Clear your mind must be.",
)
_r("agci", "agresti_coull", "IntervalEstimation", "Agresti-Coull interval", "Concentrate!")
_r("pid", "pid_controller", "ControlTheory", "PID controller simulation", "That is why you fail.")
_r(
    "trnfn",
    "transfer_function",
    "ControlTheory",
    "Transfer function step/impulse response",
    "Luminous beings are we.",
)
_r(
    "bode",
    "bode_plot",
    "ControlTheory",
    "Bode magnitude and phase plot",
    "Distribution helper.",
)
_r(
    "ssfbk",
    "state_feedback",
    "ControlTheory",
    "State feedback via pole placement",
    "Judge me by my size, do you?",
)
_r("lyapn", "lyapunov_stability", "ControlTheory", "Lyapunov stability analysis", "Always two there are.")
_r(
    "shdiv",
    "shannon_diversity",
    "Ecology",
    "Shannon diversity index",
    "Fear leads to anger, anger leads to hate.",
)
_r("smpdi", "simpson_diversity", "Ecology", "Simpson diversity index", "Mudhole? Slimy? My home this is!")
_r(
    "spric",
    "species_richness",
    "Ecology",
    "Chao1 species richness estimator",
    "Around the survivors a perimeter create.",
)
_r(
    "raref",
    "rarefaction",
    "Ecology",
    "Rarefaction curve",
    "Victory, you say? Not victory. The shroud of the dark side has fallen.",
)
_r(
    "lotka",
    "lotka_volterra",
    "Ecology",
    "Lotka-Volterra predator-prey ODE",
    "Distribution helper.",
)
_r("prime", "prime_density", "NumberTheory", "Prime sieve and density", "Death is a natural part of life.")
_r("gcd_", "extended_gcd", "NumberTheory", "Extended Euclidean algorithm", "In a dark place we find ourselves.")
_r(
    "modin",
    "mod_inverse",
    "NumberTheory",
    "Modular multiplicative inverse",
    "If into the recordings you go, only pain will you find.",
)
_r("milrb", "miller_rabin", "NumberTheory", "Miller-Rabin primality test", "At an end your rule is.")
_r("crt", "chinese_remainder", "NumberTheory", "Chinese Remainder Theorem solver", "Into exile I must go.")
_r(
    "fuzzy",
    "fuzzy_membership",
    "FuzzyLogic",
    "Fuzzy membership functions",
    "Not if anything to say about it, I have.",
)
_r("fzand", "fuzzy_and", "FuzzyLogic", "Fuzzy AND (t-norm)", "When nine hundred years old you reach.")
_r("fzor", "fuzzy_or", "FuzzyLogic", "Fuzzy OR (t-conorm)", "Distribution helper.")
_r(
    "defuz",
    "defuzzify",
    "FuzzyLogic",
    "Defuzzification (centroid/bisector)",
    "Always pass on what you have learned.",
)
_r("fzinf", "fuzzy_inference", "FuzzyLogic", "Mamdani fuzzy inference", "Grave danger you are in.")
_r("rk4", "runge_kutta4", "NumericalMethods", "Runge-Kutta 4th order ODE solver", "Begun the Clone War has.")
_r(
    "simps",
    "simpson_integrate",
    "NumericalMethods",
    "Simpson rule integration",
    "Once you start down the dark path.",
)
_r(
    "newtn",
    "newton_root",
    "NumericalMethods",
    "Newton-Raphson root finding",
    "Anger, fear, aggression the dark side are.",
)
_r(
    "biscn",
    "bisection_root",
    "NumericalMethods",
    "Bisection method root finding",
    "Distribution helper.",
)
_r(
    "fdiff",
    "finite_diff",
    "NumericalMethods",
    "Finite difference numerical differentiation",
    "Control, control, you must learn control.",
)
_r("csomp", "omp", "CompressedSensing", "Orthogonal Matching Pursuit", "Hmm, strong you are.")
_r("csbp", "basis_pursuit_l1", "CompressedSensing", "L1 minimization via ISTA", "Distribution helper.")
_r(
    "csrip",
    "rip_check",
    "CompressedSensing",
    "Restricted Isometry Property check",
    "A long time have I watched.",
)
_r("csmtx", "sensing_matrix", "CompressedSensing", "Random sensing matrix", "Ready are you?")
_r(
    "csdnt",
    "denoise_cs",
    "CompressedSensing",
    "Compressed sensing denoising",
    "No more training do you require.",
)
_r(
    "combn",
    "combinations_count",
    "Combinatorics",
    "Binomial coefficient",
    "The cave. Remember your failure at the cave.",
)
_r(
    "permu",
    "permutation_count",
    "Combinatorics",
    "Permutation count",
    "What is now proved was once only imagined. — William Blake",
)
_r("ovrla", "overlap_weight", "Propensity", "Overlap weighting for causal inference", "Your father he is.")


_r("fkrad", "flesch_kincaid", "Psycholinguistics", "Flesch-Kincaid readability", "Luminous beings are we.")
_r(
    "ari",
    "automated_readability",
    "Psycholinguistics",
    "Automated Readability Index",
    "Always pass on what you have learned.",
)
_r("cloze", "cloze_score", "Psycholinguistics", "Cloze test scoring", "Distribution helper.")
_r("wrdln", "word_length_stats", "Psycholinguistics", "Word length distribution", "Patience you must have.")
_r(
    "typtr",
    "type_token_ratio",
    "Psycholinguistics",
    "Type-token ratio (lexical diversity)",
    "Much to learn, you still have.",
)
_r("ppvpr", "ppv_prevalence", "ClinicalEpi", "Prevalence-based PPV/NPV", "Strong you are.")
_r("lrpos", "likelihood_ratio", "ClinicalEpi", "Likelihood ratio (LR+/LR-)", "Train yourself to let go.")
_r(
    "dagre",
    "diagnostic_agreement",
    "ClinicalEpi",
    "Diagnostic agreement (Cohen kappa)",
    "Named must be your fear.",
)
_r(
    "nomgr",
    "fagan_nomogram",
    "ClinicalEpi",
    "Fagan nomogram post-test probability",
    "Difficult to see, the future is.",
)
_r("srsiz", "sample_size_proportion", "ClinicalEpi", "Sample size for proportion", "Do or do not.")
_r("savgf", "savgol_smooth", "Smoothing", "Savitzky-Golay filter", "Ready are you?")
_r("ksmth", "kernel_smooth", "Smoothing", "Nadaraya-Watson kernel smoothing", "Size matters not.")
_r("lssmo", "loess_smooth", "Smoothing", "LOESS/LOWESS smoother", "Fear leads to anger.")
_r("spnsm", "spline_smooth", "Smoothing", "Smoothing spline", "Wars not make one great.")
_r("whttk", "whittaker_smooth", "Smoothing", "Whittaker smoother", "Distribution helper.")
_r("dijks", "dijkstra", "GraphTheory", "Dijkstra shortest paths", "Truly wonderful the mind of a child is.")
_r("krus", "kruskal_mst", "GraphTheory", "Kruskal MST", "Control, you must learn.")
_r("floyd", "floyd_warshall", "GraphTheory", "Floyd-Warshall all-pairs shortest", "Clear your mind must be.")
_r("tpsrt", "topological_sort", "GraphTheory", "Topological sort of DAG", "Concentrate!")
_r("mxflw", "max_flow", "GraphTheory", "Ford-Fulkerson max flow", "That is why you fail.")
_r("bldal", "bland_altman", "MeasurementError", "Bland-Altman agreement", "Distribution helper.")
_r(
    "demin",
    "deming_regression",
    "MeasurementError",
    "Deming errors-in-variables regression",
    "Distribution helper.",
)
_r(
    "attnr",
    "attenuation_ratio",
    "MeasurementError",
    "Attenuation ratio disattenuation",
    "Already know you that which you need.",
)
_r("semea", "sem_measurement", "MeasurementError", "Standard error of measurement", "Grave danger you are in.")
_r("rci", "reliable_change", "MeasurementError", "Jacobson-Truax reliable change", "Begun the Clone War has.")
_r("dbscr", "dbscan_dr", "Clustering", "DBSCAN clustering", "Once you start down the dark path.")
_r("aggcl", "agglomerative", "Clustering", "Agglomerative hierarchical clustering", "Anger, fear, aggression.")
_r("silht", "silhouette_score", "Clustering", "Silhouette analysis", "Distribution helper.")
_r("gapst", "gap_statistic", "Clustering", "Gap statistic for optimal k", "Hmm, strong you are.")
_r("spectc", "spectral_cluster", "Clustering", "Spectral clustering", "No more training do you require.")
_r(
    "heckm",
    "heckman_correction",
    "SelectionBias",
    "Heckman two-step correction",
    "The cave. Remember your failure.",
)
_r("trunc", "truncated_regression", "SelectionBias", "Truncated regression", "Your father he is.")
_r("censr", "censored_regression", "SelectionBias", "Censored regression (Tobit)", "Secret, shall I tell you?")
_r(
    "selbf",
    "selection_bias_factor",
    "SelectionBias",
    "Selection bias quantification",
    "Looking? Found someone you have.",
)
_r(
    "funlp",
    "funnel_plot",
    "SelectionBias",
    "Funnel plot for publication bias",
    "When nine hundred years old you reach.",
)
_r("pstrt", "ps_stratify", "Propensity", "Propensity score stratification", "Into exile I must go.")
_r("psovl", "ps_overlap", "Propensity", "PS overlap/positivity check", "At an end your rule is.")
_r(
    "pstmw",
    "ps_trimmed_weights",
    "Propensity",
    "Trimmed IPW weights",
    "Not if anything to say about it I have.",
)
_r("pscal", "ps_calibrate", "Propensity", "Propensity score calibration", "Death is a natural part of life.")
_r(
    "smbln",
    "smd_balance",
    "Propensity",
    "Standardized mean difference balance",
    "In a dark place we find ourselves.",
)
_r("pendu", "pendulum", "Physics", "Simple pendulum ODE simulation", "If into the recordings you go.")
_r("sprbk", "spring_mass", "Physics", "Damped spring-mass system", "Distribution helper.")
_r("orbit", "kepler_orbit", "Physics", "Keplerian orbit trajectory", "Victory you say?")
_r(
    "diffu",
    "heat_diffusion",
    "Physics",
    "1D heat diffusion equation",
    "Around the survivors a perimeter create.",
)
_r("waved", "wave_1d", "Physics", "1D wave equation", "Mudhole? Slimy? My home this is!")
_r("gdpgr", "gdp_growth", "Economics", "GDP growth rate and trend", "Always two there are.")
_r(
    "ginii",
    "gini_coefficient",
    "Economics",
    "Gini inequality coefficient",
    "Distribution helper.",
)
_r("lornz", "lorenz_curve", "Economics", "Lorenz curve coordinates", "The greatest teacher failure is.")
_r("elast", "elasticity", "Economics", "Price elasticity of demand", "Distribution helper.")
_r("cpi", "cpi_inflation", "Economics", "CPI and inflation rate", "Trust your feelings. --")
_r(
    "boolm",
    "boolean_minimize",
    "BooleanAlgebra",
    "Quine-McCluskey minimization",
    "Your focus determines your reality. -- Qui-Gon",
)
_r("karnm", "karnaugh_map", "BooleanAlgebra", "Karnaugh map grouping", "Distribution helper.")
_r("bexpr", "boolean_eval", "BooleanAlgebra", "Boolean expression evaluation", "This is the way. -- Din Djarin")
_r("gates", "logic_gates", "BooleanAlgebra", "Logic gate simulation", "I have spoken. -- Kuiil")
_r("addrc", "full_adder", "BooleanAlgebra", "Binary full adder circuit", "Rebellions are built on hope. -- Jyn Erso")
_r("relu", "relu", "NeuralNet", "ReLU/Leaky ReLU activation", "Distribution helper.")
_r("sigmd", "sigmoid", "NeuralNet", "Sigmoid activation with gradient", "What is now proved was once only imagined. — William Blake")
_r("batchn", "batch_norm", "NeuralNet", "Batch normalization", "Distribution helper.")
_r("drpot", "dropout", "NeuralNet", "Dropout regularization", "There is another.")
_r("xavir", "xavier_init", "NeuralNet", "Xavier/Glorot weight initialization", "Let the past die. -- Kylo Ren")
_r(
    "nwalg",
    "needleman_wunsch",
    "SequenceAnalysis",
    "Needleman-Wunsch global alignment",
    "We are what they grow beyond.",
)
_r("swalg", "smith_waterman", "SequenceAnalysis", "Smith-Waterman local alignment", "Distribution helper.")
_r(
    "lcsub",
    "longest_common_subseq",
    "SequenceAnalysis",
    "Longest common subsequence",
    "Distribution helper.",
)
_r("seqkm", "kmer_frequency", "SequenceAnalysis", "K-mer frequency counting", "Hope is like the sun. --")
_r("blsum", "blosum_score", "SequenceAnalysis", "BLOSUM62 alignment score", "No one is ever really gone. -- Luke")
_r("bheap", "binary_heap", "DataStructures", "Binary min/max heap", "I know what I have to do. -- Ben Solo")
_r(
    "bstop",
    "bst_operations",
    "DataStructures",
    "Binary search tree operations",
    "Every choice has led you here. -- Snoke",
)
_r(
    "hasht",
    "hash_table",
    "DataStructures",
    "Hash table with chaining",
    "What is now proved was once only imagined. — William Blake",
)
_r("trie", "trie_operations", "DataStructures", "Prefix trie operations", "Strike me down. --")
_r("grpht", "graph_from_edges", "DataStructures", "Graph from edge list", "The dark side clouds everything.")
_r("lgint", "lagrange_interp", "Interpolation", "Lagrange interpolation", "There is always a bigger fish. -- Qui-Gon")
_r(
    "nwint",
    "newton_interp",
    "Interpolation",
    "Newton divided differences",
    "The ability to speak does not make you intelligent. -- Qui-Gon",
)
_r("hrmit", "hermite_interp", "Interpolation", "Hermite interpolation", "Mind tricks don't work on me. -- Watto")
_r(
    "csint",
    "cubic_spline_interp",
    "Interpolation",
    "Natural cubic spline",
    "What is now proved was once only imagined. — William Blake",
)
_r("rbfin", "rbf_interp", "Interpolation", "RBF interpolation", "So this is how liberty dies. -- Padme")
_r("lufac", "lu_factorize", "MatrixDecomp", "LU factorization with pivoting", "What is now proved was once only imagined. — William Blake")
_r("qrfac", "qr_factorize", "MatrixDecomp", "QR decomposition (Householder)", "What is now proved was once only imagined. — William Blake")
_r("ldlt", "ldlt_factorize", "MatrixDecomp", "LDL^T symmetric decomposition", "You underestimate my power. -- Anakin")
_r("schur", "schur_decompose", "MatrixDecomp", "Schur decomposition", "You were the Chosen One. --")
_r("polar", "polar_decompose", "MatrixDecomp", "Polar decomposition", "Hello there. --")


_r("prsiv", "prime_sieve", "NumberTheory", "Sieve of Eratosthenes", "Distribution helper.")
_r("ovrlp", "overlap_coefficient", "SetTheory", "Overlap coefficient", "Judge me by my size do you?")
_r("catln", "catalan_number", "Combinatorics", "Catalan number", "Truly wonderful.")
_r("stirn", "stirling_number", "Combinatorics", "Stirling numbers", "Always two there are.")
_r(
    "bellp",
    "bell_polynomial",
    "Combinatorics",
    "Bell polynomial/number",
    "Distribution helper.",
)
_r("jaccr", "jaccard_similarity", "SetTheory", "Jaccard similarity index", "Distribution helper.")
_r("dceof", "dice_coefficient", "SetTheory", "Dice coefficient", "Distribution helper.")
_r("hmngd", "hamming_distance_sets", "SetTheory", "Hamming distance", "Concentrate!")
_r("edtds", "edit_distance", "SetTheory", "Levenshtein edit distance", "Patience you must have.")


_r(
    "benfd",
    "benfords_law_test",
    "Statistics",
    "Benford's law first-digit test (Benford 1938)",
    "The numbers do not lie. -- K-2SO",
)
_r(
    "cedfm",
    "ecdf",
    "Statistics",
    "Empirical cumulative distribution function",
    "Step by step, the path reveals itself. -- Chirrut",
)
_r(
    "qqplt",
    "qq_data",
    "Statistics",
    "Q-Q plot data for distribution comparison (Wilk & Gnanadesikan 1968)",
    "Distribution helper.",
)
_r(
    "rankt",
    "rank_transform",
    "Statistics",
    "Rank transformation for nonparametric analysis (Conover & Iman 1981)",
    "Rank matters not. Judge me by my size, do you?",
)
_r(
    "rspln",
    "restricted_cubic_spline",
    "Regression",
    "Restricted cubic spline basis (Harrell 2015)",
    "Flexible, the spline must be.",
)
_r(
    "zipfl",
    "zipf_law_fit",
    "Statistics",
    "Zipf's law fit for frequency distributions (Zipf 1949)",
    "The most frequent, the shortest words are.",
)

_r(
    "mankt",
    "mantel_test",
    "Spatial",
    "Mantel test for matrix correlation (Mantel, 1967)",
    "Distribution helper.",
)
_r(
    "stpnt",
    "spacetime_intensity",
    "SpatioTemporal",
    "Space-time point pattern intensity estimation",
    "In every corner of the galaxy. -- Mon Mothma",
)
_r(
    "sintf",
    "natural_neighbor",
    "Spatial",
    "Natural neighbor interpolation (Sibson, 1981)",
    "The closest allies reveal the truth. --",
)
_r(
    "stcox",
    "spacetime_cox",
    "SpatioTemporal",
    "Log-Gaussian Cox process intensity (Diggle et al., 2013)",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "sparma",
    "spatial_arma",
    "Spatial",
    "Spatial ARMA model for areal data (Anselin, 1988)",
    "Connections ripple across space.",
)
_r(
    "lgrte",
    "local_growth_rate",
    "SpatioTemporal",
    "Local growth rate across spatial units over time",
    "Distribution helper.",
)
_r(
    "stcrs",
    "spacetime_crosscorr",
    "SpatioTemporal",
    "Space-time cross-correlation between fields",
    "Two forces intertwined across time. -- Qui-Gon",
)
_r(
    "spwgt",
    "spatial_weights",
    "Spatial",
    "Spatial weight matrix construction (knn/distance/inverse)",
    "Every neighbor matters. -- Bail Organa",
)
_r(
    "stmrn",
    "spacetime_moran_scatter",
    "SpatioTemporal",
    "Space-time Moran scatterplot (Rey & Janikas, 2006)",
    "Patterns across space and time reveal themselves. --",
)
_r(
    "lmrkt",
    "local_markov",
    "SpatioTemporal",
    "Local Markov transition matrices (Rey, 2001)",
    "The future depends on where you stand. -- Mace Windu",
)
_r(
    "spflw",
    "spatial_flow",
    "Spatial",
    "Gravity model for spatial interaction (Wilson, 1971)",
    "Trade routes bind the galaxy. -- Hondo",
)
_r(
    "sthet",
    "spacetime_heterogeneity",
    "SpatioTemporal",
    "Spatiotemporal heterogeneity test (Levin, 1992)",
    "Variation is the way of all living things.",
)

_r(
    "stacf",
    "st_autocorrelation",
    "SpatioTemporal",
    "Space-time Moran's I across temporal lags (Cliff & Ord 1981)",
    "Distribution helper.",
)
_r(
    "stscan",
    "st_scan_statistic",
    "SpatioTemporal",
    "Space-time scan statistic for cluster detection (Kulldorff 1997)",
    "There has been an awakening. Have you felt it? -- Snoke",
)
_r(
    "stkde",
    "st_kde",
    "SpatioTemporal",
    "Spatiotemporal kernel density estimation (3D bandwidth)",
    "The dark side clouds everything. Impossible to see, the future is.",
)
_r(
    "stvar",
    "st_variogram",
    "SpatioTemporal",
    "Spatiotemporal variogram estimation (Cressie & Huang 1999)",
    "The further you go, the more different things become. -- Ezra Bridger",
)
_r(
    "arlin",
    "areal_interpolation",
    "Spatial",
    "Areal interpolation via dasymetric mapping (Tobler 1979)",
    "Distribution helper.",
)
_r(
    "stgwr",
    "gtwr",
    "SpatioTemporal",
    "Geographically and temporally weighted regression (Huang et al. 2010)",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "trajd",
    "trajectory_distance",
    "SpatioTemporal",
    "Frechet distance between movement trajectories",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "hrsde",
    "home_range_kde",
    "Spatial",
    "Home range estimation via kernel density (Worton 1989)",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "stdbs",
    "st_dbscan",
    "SpatioTemporal",
    "ST-DBSCAN spatiotemporal clustering (Birant & Kut 2007)",
    "We are the spark that will light the fire. -- Poe Dameron",
)
_r(
    "ripk",
    "ripley_k_corrected",
    "Spatial",
    "Edge-corrected Ripley's K with Besag L-function (Ripley 1976)",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "semmv",
    "spatial_error_ml",
    "Spatial",
    "Spatial error model via maximum likelihood (Anselin 1988)",
    "Errors ripple across the galaxy like waves. -- Galen Erso",
)
_r(
    "gwpca",
    "gw_pca",
    "Spatial",
    "Geographically weighted PCA (Harris et al. 2011)",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "stdif",
    "spatial_did",
    "SpatioTemporal",
    "Spatiotemporal difference-in-differences (spatial DiD)",
    "The belonging you seek is not behind you. It is ahead. -- Maz Kanata",
)

_r("antmn", "allometric_regression", "Marvel", "Allometric (power-law) scaling regression", "I know a guy. -- Ant-Man")
_r("beast", "jukes_cantor_rate", "Marvel", "Jukes-Cantor mutation rate estimation", "Oh my stars and garters! -- Beast")
_r(
    "blade",
    "edge_detect",
    "Marvel",
    "Canny edge detection variant",
    "Some motherfuckers are always trying to ice-skate uphill. -- Blade",
)
_r("bpntr", "vibranium_damping", "Marvel", "Second-order damped signal attenuation", "What is now proved was once only imagined. — William Blake")
_r(
    "capam",
    "circular_mean",
    "Marvel",
    "Circular statistics mean direction",
    "What is now proved was once only imagined. — William Blake",
)
_r("colsm", "stress_strain", "Marvel", "Material stress-strain curve analysis", "I will break you. -- Colossus")
_r("cyclo", "gaussian_beam", "Marvel", "Gaussian beam optics propagation", "I could never control it. -- Cyclops")
_r(
    "dormm",
    "recurrence_quantification",
    "Marvel",
    "Recurrence quantification analysis (RQA)",
    "Dormammu, I have come to bargain. -- Doctor Strange",
)
_r("dpool", "jackknife_delete_d", "Marvel", "Delete-d jackknife estimator", "Maximum effort! -- Deadpool")
_r("draxm", "destroyer_decompose", "Marvel", "Truncated SVD rank reduction", "Nothing goes over my head. -- Drax")
_r(
    "drstr",
    "multiverse_bootstrap",
    "Marvel",
    "Parallel-universes bootstrap CI",
    "I went forward in time to view alternate futures. -- Doctor Strange",
)
_r("galct", "depletion_model", "Marvel", "Resource depletion model (exponential/linear)", "I hunger. -- Galactus")
_r(
    "gambt",
    "card_probability",
    "Marvel",
    "Hypergeometric card probability",
    "The name is Gambit. Remember it. -- Gambit",
)
_r(
    "gamra",
    "aft_model",
    "Marvel",
    "Accelerated failure time survival model",
    "I have lived most of my life surrounded by enemies. -- Gamora",
)
_r("groot", "logistic_growth", "Marvel", "Logistic growth curve fitting", "I am Groot. -- Groot")
_r(
    "hkeye",
    "precision_recall_at_k",
    "Marvel",
    "Precision and recall at top-k",
    "You and I see things differently. -- Hawkeye",
)
_r("What is now proved was once only imagined. — William Blake", "lot_acceptance", "Marvel", "Destructive sampling lot acceptance test", "What is now proved was once only imagined. — William Blake")
_r("ironm", "armor_optimize", "Marvel", "Constrained linear programming (simplex)", "What is now proved was once only imagined. — William Blake")
_r(
    "kngpn",
    "crime_network_centrality",
    "Marvel",
    "Network centrality measures (degree/betweenness/closeness)",
    "When I was a boy... -- Kingpin",
)
_r(
    "illusion_score",
    "Marvel",
    "GAN discriminator evaluation (BCE/JSD)",
    "I am burdened with glorious purpose. -- Loki",
)
_r("magnm", "biot_savart", "Marvel", "Biot-Savart magnetic field computation", "You are all beneath me. -- Magneto")
_r("msmvl", "elastic_deformation", "Marvel", "Random elastic deformation field", "I am Ms. Marvel! -- Kamala Khan")
_r(
    "nblam",
    "cloud_mass_function",
    "Marvel",
    "Schechter cloud mass function fit",
    "I was disassembled and rebuilt. -- Nebula",
)
_r(
    "phoen",
    "phoenix_break",
    "Marvel",
    "Structural break detection + recovery",
    "I am fire and life incarnate. -- Phoenix",
)
_r(
    "punsh",
    "penalty_regression",
    "Marvel",
    "Elastic net coordinate descent regression",
    "If you are guilty, you are dead. -- Punisher",
)
_r(
    "rcoon",
    "hohmann_transfer",
    "Marvel",
    "Hohmann transfer orbit computation",
    "Ain't no thing like me, except me. -- Rocket",
)
_r("rogue", "absorption_features", "Marvel", "Transfer learning feature extraction (PCA)", "Don't touch me. -- Rogue")
_r("scltw", "probability_resample", "Marvel", "BCa bootstrap quantile resampling", "No more. -- Scarlet Witch")
_r(
    "shang",
    "ring_harmonics",
    "Marvel",
    "Circular harmonic Fourier decomposition",
    "You gave me ten rings. I will use them all. -- Shang-Chi",
)
_r(
    "spidm",
    "web_graph_communities",
    "Marvel",
    "Spectral graph community detection",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "storm",
    "markov_weather",
    "Marvel",
    "First-order Markov chain weather model",
    "Do you know what happens to a toad struck by lightning? -- Storm",
)
_r(
    "taskm",
    "dtw_match",
    "Marvel",
    "Dynamic time warping pattern matching",
    "I can do anything you can do. -- Taskmaster",
)
_r(
    "thnsm",
    "snap_estimator",
    "Marvel",
    "Random half-sampling variance estimator",
    "What is now proved was once only imagined. — William Blake",
)
_r("What is now proved was once only imagined. — William Blake", "bonferroni_correction", "Marvel", "Bonferroni multiple testing correction", "What is now proved was once only imagined. — William Blake")
_r("ultro", "swarm_optimize", "Marvel", "Particle swarm optimisation (PSO)", "There are no strings on me. -- Ultron")
_r("venmm", "cooccurrence_matrix", "Marvel", "Co-occurrence matrix (Jaccard/Dice/PMI)", "We are Venom. -- Venom")
_r("visnm", "mind_stone_cluster", "Marvel", "RBF spectral clustering variant", "I was born yesterday. -- Vision")
_r("wkndm", "pareto_optimize", "Marvel", "Multi-objective Pareto front identification", "What is now proved was once only imagined. — William Blake")
_r(
    "wolvn",
    "mice_impute",
    "Marvel",
    "MICE multiple imputation (chained equations)",
    "I am the best there is at what I do. -- Wolverine",
)
_r("xvr", "partial_corr_matrix", "Marvel", "Partial correlation brain connectivity", "To me, my X-Men. -- Professor X")

_r(
    "alfrd",
    "cooks_distance",
    "DCComics",
    "Cook's distance diagnostic",
    "I shall endeavour to satisfy, sir. -- Alfred Pennyworth",
)
_r(
    "aqman",
    "idw_interpolate",
    "DCComics",
    "Inverse distance weighted interpolation",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "arrow",
    "directed_layout",
    "DCComics",
    "Directed graph layout (force-directed)",
    "You have failed this city! -- Green Arrow",
)
_r("atoms", "james_stein", "DCComics", "James-Stein shrinkage estimator", "It's a small world after all. -- The Atom")
_r(
    "bane",
    "structural_break",
    "DCComics",
    "Structural break test (max F)",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "What is now proved was once only imagined. — William Blake",
    "robust_m_estimator",
    "DCComics",
    "Huber M-estimator of location",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "bbeas",
    "procrustes_shape",
    "DCComics",
    "Procrustes shape analysis (morphometrics)",
    "Dude, I can turn into a T-Rex. -- Beast Boy",
)
_r("bostr", "naive_forecast", "DCComics", "Naive time series forecasting", "I am from the future. -- Booster Gold")
_r(
    "btgrl",
    "bagplot_outliers",
    "DCComics",
    "Bagplot bivariate outlier detection",
    "The night is darkest before the dawn. -- Batgirl",
)
_r("catw", "catenary_fit", "DCComics", "Catenary curve fitting", "Meow. -- Catwoman")
_r(
    "cnstr",
    "iqr_exorcise",
    "DCComics",
    "Anomaly removal via IQR fences",
    "I'm a nasty piece of work. -- John Constantine",
)
_r("cybrg", "pid_simulate", "DCComics", "PID control system simulation", "Booyah! -- Cyborg")
_r(
    "darks",
    "dark_energy_eos",
    "DCComics",
    "Dark energy equation of state (CPL)",
    "I am the end of all things. -- Darkseid",
)
_r("dchnt", "ghost_signal", "DCComics", "Ghost signal detection (surrogate test)", "I'm already dead. -- Deadman")
_r(
    "doom",
    "failure_cascade",
    "DCComics",
    "System failure cascade (reliability)",
    "We are the Doom Patrol. -- The Chief",
)
_r("dshot", "precision_at_k", "DCComics", "Precision at K metric", "I never miss. -- Deadshot")
_r("dstke", "hazard_kernel", "DCComics", "Kernel-smoothed hazard rate estimation", "I keep my promises. -- Deathstroke")
_r(
    "What is now proved was once only imagined. — William Blake",
    "fast_ann",
    "DCComics",
    "Fast approximate nearest neighbor search",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "freze",
    "freeze_thaw",
    "DCComics",
    "Freeze-thaw cycle degradation model",
    "Tonight, hell freezes over. -- Mr. Freeze",
)
_r(
    "grnln",
    "greens_convolve",
    "DCComics",
    "Green's function convolution",
    "What is now proved was once only imagined. — William Blake",
)
_r("harqn", "harrells_c", "DCComics", "Harrell's C concordance index", "What is now proved was once only imagined. — William Blake")
_r(
    "hkgrl",
    "hawking_temperature",
    "DCComics",
    "Hawking radiation temperature model",
    "History is about to be rewritten. -- Hawkgirl",
)
_r("What is now proved was once only imagined. — William Blake", "permutation_two_sample", "DCComics", "Two-sample permutation test", "What is now proved was once only imagined. — William Blake")
_r(
    "jrich",
    "formant_extract",
    "DCComics",
    "Formant extraction (LPC analysis)",
    "Actions speak louder than words. -- Jericho",
)
_r("kflsh", "acceleration_profile", "DCComics", "Acceleration profile analysis", "What is now proved was once only imagined. — William Blake")
_r(
    "lexlr",
    "lexico_rank",
    "DCComics",
    "Lexicographic rank aggregation (Borda)",
    "What is now proved was once only imagined. — William Blake",
)
_r("raven", "raven_score", "DCComics", "Raven's Progressive Matrices scoring", "Azarath Metrion Zinthos! -- Raven")
_r("ridlr", "cipher_frequency", "DCComics", "Cipher letter frequency analysis", "Riddle me this. -- The Riddler")
_r("robin", "winsorized_mean", "DCComics", "Winsorized mean", "What is now proved was once only imagined. — William Blake")
_r("roslw", "wind_rose", "DCComics", "Wind rose plot construction", "I make my own choices. -- Rose Wilson")
_r(
    "rvflh",
    "causal_reversal_test",
    "DCComics",
    "Causal reversal test (residual asymmetry)",
    "What is now proved was once only imagined. — William Blake",
)
_r("shazm", "simultaneous_test", "DCComics", "SHAZAM simultaneous hypothesis test", "SHAZAM! -- Billy Batson")
_r(
    "sinst",
    "vol_of_vol",
    "DCComics",
    "Volatility of volatility (fear index)",
    "In blackest night, you will fear me. -- Sinestro",
)
_r("strfr", "luminosity_function", "DCComics", "Stellar luminosity function", "I am not from your planet. -- Starfire")
_r("suprm", "ks_supremum", "DCComics", "Kolmogorov-Smirnov supremum test", "What is now proved was once only imagined. — William Blake")
_r(
    "swmpt",
    "biomass_estimate",
    "DCComics",
    "Allometric biomass estimation",
    "The Green speaks through me. -- Swamp Thing",
)
_r(
    "terra",
    "terrain_analysis",
    "DCComics",
    "Terrain analysis (slope/aspect/curvature)",
    "I was never on your side. -- Terra",
)
_r(
    "wndwm",
    "window_function",
    "DCComics",
    "Spectral window function generation",
    "What is now proved was once only imagined. — William Blake",
)
_r("zatna", "palindrome_detect", "DCComics", "Palindromic subsequence detection", "Erif! -- Zatanna")
_r(
    "zoomv",
    "velocity_profile",
    "DCComics",
    "Velocity/acceleration/jerk profile",
    "You can't lock up the darkness. -- Zoom",
)

_r(
    "alpht",
    "prior_elicit",
    "Transformers",
    "Bayesian prior elicitation from expert quantiles (Alpha Trion)",
    "The wisdom of ages guides our priors. -- Alpha Trion",
)
_r("arcee", "arc_length", "Transformers", "Arc length of parametric curves (Arcee)", "Speed is my weapon. -- Arcee")
_r(
    "barcd",
    "csp_backtrack",
    "Transformers",
    "Constraint satisfaction via backtracking (Barricade)",
    "No one gets through. -- Barricade",
)
_r(
    "blckw",
    "snr_degradation",
    "Transformers",
    "SNR degradation / signal jamming model (Blackout)",
    "All communications jammed. -- Blackout",
)
_r(
    "bonec",
    "weibull_analysis",
    "Transformers",
    "Weibull failure / reliability analysis (Bonecrusher)",
    "I will crush you! -- Bonecrusher",
)
_r(
    "bumbl",
    "spectral_centroid",
    "Transformers",
    "What is now proved was once only imagined. — William Blake",
    "What is now proved was once only imagined. — William Blake",
)
_r("drift", "adwin_drift", "Transformers", "ADWIN concept drift detection (Drift)", "Things are changing. -- Drift")
_r(
    "dvstr",
    "ensemble_stack",
    "Transformers",
    "Stacked generalization ensemble (Devastator)",
    "Devastator, merge! -- Scrapper",
)
_r(
    "frenz",
    "lyapunov_exponent",
    "Transformers",
    "Maximal Lyapunov exponent from time series (Frenzy)",
    "Chaos! More chaos! -- Frenzy",
)
_r("grimk", "power_law_fit", "Transformers", "Power law scaling fit (Grimlock)", "Me Grimlock strongest! -- Grimlock")
_r(
    "grind",
    "rosin_rammler",
    "Transformers",
    "Rosin-Rammler particle size distribution (Grindor)",
    "Grind them down! -- Grindor",
)
_r(
    "hotrd",
    "newton_convergence",
    "Transformers",
    "Newton-Raphson with convergence rate analysis (Hot Rod)",
    "Faster! Faster! -- Hot Rod",
)
_r("hound", "kalman_filter", "Transformers", "Kalman filter state estimation (Hound)", "I never lose a trail. -- Hound")
_r(
    "ircnh",
    "robust_covariance_mcd",
    "Transformers",
    "Minimum Covariance Determinant robust estimation (Ironhide)",
    "Ironhide reporting. -- Ironhide",
)
_r(
    "jazz",
    "markov_generate",
    "Transformers",
    "Markov chain sequence generation (Jazz)",
    "Do I at least get to pick the song? -- Jazz",
)
_r(
    "lasrb",
    "iou_metric",
    "Transformers",
    "Intersection over Union bounding box metric (Laserbeak)",
    "Eyes in the sky. -- Laserbeak",
)
_r(
    "megtr",
    "kronecker_decompose",
    "Transformers",
    "What is now proved was once only imagined. — William Blake",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "mirag",
    "dp_laplace",
    "Transformers",
    "Laplace mechanism differential privacy (Mirage)",
    "Now you see me... -- Mirage",
)
_r(
    "omsup",
    "minimax_solve",
    "Transformers",
    "Minimax two-player zero-sum game solver (Omega Supreme)",
    "Omega Supreme: defend. -- Omega Supreme",
)
_r(
    "optms",
    "optimal_transport",
    "Transformers",
    "What is now proved was once only imagined. — William Blake",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "prmsc",
    "vae_sample",
    "Transformers",
    "VAE latent space reparameterization sampling (Primus)",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "prowl",
    "anomaly_isolation",
    "Transformers",
    "Isolation forest anomaly detection (Prowl)",
    "Logic is the ultimate weapon. -- Prowl",
)
_r(
    "ratch",
    "repair_pipeline",
    "Transformers",
    "Data cleaning / denoising pipeline (Ratchet)",
    "I can fix that! -- Ratchet",
)
_r("ravag", "lsb_embed", "Transformers", "LSB steganography embedding (Ravage)", "Silent but deadly. -- Ravage")
_r(
    "shkwv",
    "wave_equation_1d",
    "Transformers",
    "1-D wave equation finite difference solver (Shockwave)",
    "Shockwave calculates. -- Shockwave",
)
_r(
    "sidsw",
    "impact_force",
    "Transformers",
    "Impact force via impulse-momentum theorem (Sideswipe)",
    "Let me at em! -- Sideswipe",
)
_r(
    "sidwy",
    "adversarial_perturb",
    "Transformers",
    "FGSM adversarial perturbation generation (Sideways)",
    "Deception is an art. -- Sideways",
)
_r("skwrp", "levy_flight", "Transformers", "Levy flight random walk simulation (Skywarp)", "Surprise! -- Skywarp")
_r("slag", "newton_cooling", "Transformers", "Newton law of cooling heat transfer (Slag)", "Hot stuff! -- Slag")
_r(
    "sludg",
    "herschel_bulkley",
    "Transformers",
    "Herschel-Bulkley rheological flow model (Sludge)",
    "Sludge no hurry. -- Sludge",
)
_r("snarl", "resonance_q", "Transformers", "Mechanical resonance Q-factor (Snarl)", "Feel the vibration! -- Snarl")
_r(
    "sndwv",
    "cepstral_analysis",
    "Transformers",
    "Cepstral analysis / signal intelligence (Soundwave)",
    "Soundwave superior. -- Soundwave",
)
_r(
    "stscm",
    "byzantine_detect",
    "Transformers",
    "What is now proved was once only imagined. — William Blake",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "sunst",
    "surface_roughness",
    "Transformers",
    "Surface roughness metrics ISO 4287 (Sunstreaker)",
    "Looking good is half the battle. -- Sunstreaker",
)
_r("swoop", "lift_drag_polar", "Transformers", "Aerodynamic lift-drag polar analysis (Swoop)", "From above! -- Swoop")
_r(
    "thndm",
    "mach_shock",
    "Transformers",
    "Shock wave Mach number / Rankine-Hugoniot (Thundercracker)",
    "Feel the thunder! -- Thundercracker",
)
_r(
    "ultra",
    "ensemble_aggregate",
    "Transformers",
    "Ensemble model prediction aggregation (Ultra Magnus)",
    "I am Ultra Magnus, and I will lead. -- Ultra Magnus",
)
_r("uncr", "entropy_production", "Transformers", "Shannon entropy production rate (Unicron)", "I hunger. -- Unicron")
_r(
    "vctrs",
    "weight_init",
    "Transformers",
    "Xavier/He neural network weight initialization (Vector Sigma)",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "wheej",
    "polynomial_features",
    "Transformers",
    "Polynomial feature expansion / engineering (Wheeljack)",
    "I have an idea! -- Wheeljack",
)

_r(
    "neom",
    "decision_split",
    "Matrix",
    "Decision tree split criterion (Gini vs entropy)",
    "There is no spoon. -- Spoon Boy",
)
_r("morph", "tsne_reduce", "Matrix", "Dimensionality reduction via t-SNE", "What is now proved was once only imagined. — William Blake")
_r("trnty", "cvss_base", "Matrix", "CVSS v3.1 base score calculator", "What is now proved was once only imagined. — William Blake")
_r(
    "oracl",
    "oracl",
    "ModelSelection",
    "Oracle inequality verification (Kosorok 2008)",
    "Everything that has a beginning has an end. -- The Oracle",
)
_r("smith", "bootstrap_resample", "Matrix", "Bootstrap resampling with replacement", "What is now proved was once only imagined. — William Blake")
_r("tankg", "nn_train", "Matrix", "Neural network training loop (1-hidden layer)", "What is now proved was once only imagined. — William Blake")
_r("dozer", "pid_tune", "Matrix", "PID controller tuning and simulation", "What is now proved was once only imagined. — William Blake")
_r(
    "swtch",
    "encode_labels",
    "Matrix",
    "Label / one-hot encoding for categorical data",
    "Not like this. Not like this. -- Switch",
)
_r("apoc", "articulation_points", "Matrix", "Graph articulation points (Tarjan)", "I am afraid of the truth. -- Apoc")
_r(
    "mouse",
    "monte_carlo",
    "Matrix",
    "Monte Carlo simulation / integration engine",
    "To deny our impulses is to deny what makes us human. -- Mouse",
)
_r("cyph", "detect_leakage", "Matrix", "Data leakage detection in ML pipelines", "What is now proved was once only imagined. — William Blake")
_r("niobi", "astar_path", "Matrix", "A* pathfinding on 2-D grid", "Lock and load. -- Niobe")
_r(
    "mrvsn",
    "granger_causality",
    "Matrix",
    "Granger causality test (F-test)",
    "Choice is an illusion. -- The Merovingian",
)
_r(
    "twins",
    "phase_locking_value",
    "Matrix",
    "Phase locking value (PLV) synchronisation",
    "We are not here because we are free. -- The Twins",
)
_r(
    "sraph",
    "validate_inputs",
    "Matrix",
    "Input validation / data quality scoring",
    "I protect that which matters most. -- Seraph",
)
_r("kymkr", "information_gain", "Matrix", "Mutual information / information gain", "One door, one key. -- The Keymaker")
_r(
    "arcit",
    "gauss_seidel",
    "Matrix",
    "System of equations solver (Gauss-Seidel)",
    "I have been waiting for you. -- The Architect",
)
_r("train", "mmd_distance", "Matrix", "Domain adaptation via MMD distance", "I can take you anywhere. -- The Trainman")
_r("prsph", "ssim", "Matrix", "Structural Similarity Index (SSIM)", "Tell me, is the woman worth it? -- Persephone")
_r("lockm", "sha256_hash", "Matrix", "SHA-256 cryptographic hash", "Some doors are meant to stay closed. -- Lock")
_r(
    "rdpil",
    "red_pill_test",
    "Matrix",
    "What is now proved was once only imagined. — William Blake",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "blpil",
    "naive_baseline",
    "Matrix",
    "Naive baseline estimator for ML benchmarking",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "simul",
    "simulation_gof",
    "Matrix",
    "Goodness-of-fit test (observed vs simulated)",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "deja",
    "detect_duplicates",
    "Matrix",
    "Near-duplicate row detection",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "splgn",
    "nasa_tlx",
    "Matrix",
    "NASA-TLX cognitive load metric",
    "What is now proved was once only imagined. — William Blake",
)
_r("nebzr", "rrt_plan", "Matrix", "RRT trajectory planning in 2-D", "What is now proved was once only imagined. — William Blake")
_r("What is now proved was once only imagined. — William Blake", "defense_allocation", "Matrix", "Game-theoretic defense resource allocation", "What is now proved was once only imagined. — William Blake")
_r(
    "mtxop",
    "matrix_function",
    "Matrix",
    "Generalised matrix function f(A) via eigendecomp",
    "What is now proved was once only imagined. — William Blake",
)
_r("bullt", "time_stretch", "Matrix", "Phase vocoder time-stretching (bullet time)", "What is now proved was once only imagined. — William Blake")
_r("sentl", "detection_metrics", "Matrix", "Object detection precision/recall/F1/AP", "What is now proved was once only imagined. — William Blake")
_r(
    "squid",
    "threat_score",
    "Matrix",
    "Composite threat scoring model",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "codeg",
    "wigner_semicircle",
    "Matrix",
    "Random matrix theory (Wigner semicircle law)",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "hamam",
    "brownian_motion",
    "Matrix",
    "Brownian motion / Wiener process simulation",
    "I have survived your predecessors. -- The Merovingian",
)
_r(
    "srcod",
    "ast_depth",
    "Matrix",
    "Python AST depth and complexity analysis",
    "Everything begins with choice. -- The Merovingian",
)
_r(
    "anmls",
    "multiview_cca",
    "Matrix",
    "Multi-view learning via CCA fusion",
    "The Final Flight of the Osiris. -- Animatrix",
)
_r(
    "mnero",
    "cellular_automaton",
    "Matrix",
    "Cellular automata (Game of Life / urban growth)",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "lothr",
    "template_match",
    "Matrix",
    "Template matching via normalised cross-correlation",
    "All our lives we fought this war. -- Logos crew",
)
_r("hamrm", "impact_energy", "Matrix", "Impact testing energy (Charpy/Izod)", "I want everything. -- Bane")
_r(
    "thmss",
    "record_linkage",
    "Matrix",
    "Entity resolution / probabilistic record linkage",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "bneop",
    "birth_death_process",
    "Matrix",
    "Birth-death Markov process simulation",
    "The prophecy was true. -- Councillor Hamann",
)

_r(
    "adbst",
    "adaboost_bio",
    "Classification",
    "AdaBoost with decision stumps (Freund & Schapire, 1997)",
    "Strike me down and I shall become more powerful than you can imagine.",
)
_r(
    "adseg",
    "adaptive_segment",
    "Detection",
    "Segment signal adaptively based on local variance changes",
    "Your focus determines your reality.",
)
_r("ampdt", "amplitude_detect", "Detection", "Measure signal amplitude at detected peak locations", "It")
_r(
    "amuse",
    "amuse_bss",
    "SignalOps",
    "AMUSE: Algorithm for Multiple Unknown Signals Extraction",
    "Around the survivors a perimeter create.",
)
_r(
    "ar2ls",
    "ar_to_lsf",
    "Decomposition",
    "Convert AR coefficients to line spectral frequencies (LSF)",
    "Distribution helper.",
)
_r(
    "ar2rc",
    "ar_to_reflection",
    "Modeling",
    "Convert AR coefficients to reflection coefficients",
    "We are what they grow beyond.",
)
_r(
    "arnrm",
    "ar_normalize",
    "Modeling",
    "Normalize AR coefficients so that a[0] = 1",
    "So this is how liberty dies, with thunderous applause.",
)
_r(
    "aucfn",
    "auc_compute",
    "Foundation",
    "Compute area under the ROC curve via trapezoidal rule",
    "Never tell me the odds.",
)
_r(
    "bgcls",
    "bagging_classify",
    "Classification",
    "Bagging classifier using decision tree base learners",
    "Distribution helper.",
)
_r(
    "bpdn",
    "basis_pursuit",
    "Decomposition",
    "Basis Pursuit Denoising via ISTA (Iterative Shrinkage-Thresholding)",
    "Your focus determines your reality.",
)
_r("calpl", "calibration_plot", "Foundation", "Compute calibration (reliability) diagram data", "There")
_r("ceemd", "ceemdan", "Decomposition", "Complete EEMD with Adaptive Noise", "Do or do not. There is no try.")
_r(
    "cepds",
    "cepstral_distance",
    "Foundation",
    "Compute the Euclidean cepstral distance",
    "Rebellions are built on hope.",
)
_r("chird", "chirp_detect", "Detection", "Detect chirp (frequency sweep) in a signal", "I have the high ground.")
_r("chirg", "chirp_generate", "SignalOps", "Generate a chirp (frequency sweep) signal", "This is where the fun begins.")
_r("cmorw", "cmor_wavelet", "Wavelet", "Complex Morlet wavelet", "Distribution helper.")
_r(
    "cmxpl",
    "confusion_plot",
    "Foundation",
    "Compute confusion matrix and per-class precision/recall/F1",
    "Truly wonderful, the mind of a child is.",
)
_r(
    "cnvgr",
    "convergence_rate",
    "FilterDesign",
    "Compute the LMS adaptive filter convergence time constants",
    "Much to learn, you still have.",
)
_r("coiff", "coiflet_coeffs", "Wavelet", "Coiflet wavelet filter coefficients", "That")
_r("coshd", "cosh_distance", "Fourier", "Compute the cosh spectral distance", "Let go of your hate.")
_r(
    "cossc",
    "cost_sensitive",
    "Classification",
    "Cost-sensitive evaluation of classifier predictions",
    "Your overconfidence is your weakness.",
)
_r(
    "cscde",
    "cascade_classify",
    "Classification",
    "Cascade classifier with sequential rejection stages",
    "If you strike me down I shall become more powerful.",
)
_r(
    "cspfn",
    "csp_filter",
    "FilterDesign",
    "Common Spatial Patterns for two-class EEG classification",
    "The greatest teacher, failure is.",
)
_r(
    "curlt",
    "curvelet",
    "Decomposition",
    "1-D curvelet-like transform via frequency-domain windowing",
    "You were the chosen one!",
)
_r(
    "cusdt",
    "cusum_detect",
    "Detection",
    "Detect change points via cumulative sum (CUSUM) algorithm",
    "The dark side clouds everything.",
)
_r(
    "db2pw",
    "db_to_power",
    "Foundation",
    "Convert decibels to power",
    "Distribution helper.",
)
_r(
    "db4fn",
    "daubechies_coeffs",
    "Wavelet",
    "Daubechies wavelet filter coefficients (orthogonal, compact support)",
    "Stay on target.",
)
_r(
    "dwtfn",
    "dwt_decompose",
    "Wavelet",
    "Discrete Wavelet Transform via filter bank convolution + downsampling",
    "The shroud of the dark side has fallen.",
)
_r("dynrg", "dynamic_range", "SignalOps", "Compute the dynamic range in dB", "Wars not make one great.")
_r(
    "eercl",
    "equal_error_rate",
    "Modeling",
    "Compute the equal error rate where FAR equals FRR",
    "So this is how liberty dies, with thunderous applause.",
)
_r(
    "emdsi",
    "emd_sifting",
    "Decomposition",
    "Single EMD sifting pass: extract one Intrinsic Mode Function",
    "Distribution helper.",
)
_r(
    "emgrt",
    "emg_rms_threshold",
    "Detection",
    "Detect EMG onset using sliding-window RMS and threshold",
    "You were the chosen one!",
)
_r("enob", "enob_compute", "Foundation", "Compute the effective number of bits from SINAD", "This is the way.")
_r(
    "enrgy",
    "energy_density",
    "SignalOps",
    "Compute the energy spectral density |X(f)|^2",
    "Your focus determines your reality.",
)
_r(
    "envdt",
    "envelope_detect",
    "Detection",
    "Compute signal envelope via full-wave rectification and lowpass filter",
    "I",
)
_r(
    "eogdt",
    "eog_detect",
    "Detection",
    "Detect EOG (electrooculography) eye movement artifacts",
    "So this is how liberty dies... with thunderous applause.",
)
_r(
    "evtag",
    "event_align",
    "Detection",
    "Align detected events to template by cross-correlation",
    "Distribution helper.",
)
_r(
    "ewmdt",
    "ewma_detect",
    "Detection",
    "Detect out-of-control points via EWMA control chart",
    "Truly wonderful, the mind of a child is.",
)
_r(
    "fasci",
    "fastica",
    "Foundation",
    "FastICA: independent component analysis via fixed-point iteration",
    "Train yourself to let go of everything you fear to lose.",
)
_r(
    "fbank",
    "filter_bank_design",
    "Wavelet",
    "Design QMF analysis and synthesis filter bank for a given wavelet",
    "Rebellions are built on hope.",
)
_r("fnorm", "feature_normalize", "Foundation", "Normalize features column-wise", "Great, kid. Don")
_r(
    "fwsel",
    "forward_select",
    "Foundation",
    "Sequential forward feature selection using CV accuracy",
    "Distribution helper.",
)
_r(
    "fwtng",
    "feature_whiten",
    "Foundation",
    "Feature whitening via ZCA or PCA transform",
    "Fear is the path to the dark side.",
)
_r(
    "gbcls",
    "gbm_classify_bio",
    "Classification",
    "Gradient boosted classification using log-loss and regression stumps",
    "The ability to speak does not make you intelligent.",
)
_r("gborl", "gabor_logon", "SignalOps", "Gabor logon: Gaussian-windowed complex sinusoid", "We had each other. That")
_r(
    "glrdt",
    "glr_detector",
    "Detection",
    "Detect change points using the Generalized Likelihood Ratio test",
    "Do. Or do not. There is no try.",
)
_r(
    "haarf",
    "haar_transform",
    "Wavelet",
    "Haar wavelet transform (simplest DWT, db1)",
    "Distribution helper.",
)
_r(
    "hhtfn",
    "hilbert_huang",
    "Decomposition",
    "Hilbert-Huang Transform: Empirical Mode Decomposition + Hilbert spectrum",
    "You were the chosen one!",
)
_r(
    "hilev",
    "hilbert_envelope",
    "SignalOps",
    "Compute signal envelope using the Hilbert transform",
    "The ability to speak does not make you intelligent.",
)
_r(
    "hilsp",
    "hilbert_spectrum",
    "Decomposition",
    "Compute the marginal Hilbert spectrum from a set of IMFs",
    "I find your lack of faith disturbing.",
)
_r(
    "idwtf",
    "idwt_reconstruct",
    "Wavelet",
    "Inverse DWT: reconstruct signal from wavelet coefficients",
    "You underestimate my power.",
)
_r(
    "imfcr",
    "imf_criteria",
    "Decomposition",
    "Check IMF stopping criteria: Cauchy, S-number, energy ratio",
    "Size matters not. Judge me by my size, do you?",
)
_r(
    "insta",
    "instantaneous_amp",
    "SignalOps",
    "Instantaneous amplitude (envelope) via the analytic signal",
    "The garbage will do!",
)
_r(
    "instf",
    "instantaneous_freq",
    "SignalOps",
    "Instantaneous frequency from the analytic signal",
    "Begun, the Clone War has.",
)
_r("instp", "instantaneous_phase", "SignalOps", "Instantaneous phase via analytic signal", "Luminous beings are we.")
_r(
    "itknf",
    "itakura_dist",
    "Modeling",
    "Compute the Itakura spectral distance",
    "Strike me down and I shall become more powerful than you can imagine.",
)
_r(
    "jade",
    "jade_ica",
    "SignalOps",
    "JADE ICA: joint approximate diagonalization of eigenmatrices",
    "We are what they grow beyond.",
)
_r(
    "latdt",
    "latency_detect",
    "Detection",
    "Measure latency from stimulus onset to peak response",
    "You must unlearn what you have learned.",
)
_r("lgrtn", "log_spectral_dist", "Fourier", "Compute the log spectral distance", "Never tell me the odds.")
_r("liftw", "lifting_dwt", "Wavelet", "Lifting scheme DWT: in-place wavelet transform", "Distribution helper.")
_r(
    "lrcvb",
    "learning_curve_bio",
    "Classification",
    "Learning curve analysis via cross-validated accuracy at varying train sizes",
    "In a dark place we find ourselves, and a little more knowledge lights our way.",
)
_r(
    "ls2ar",
    "lsf_to_ar",
    "Modeling",
    "Convert line spectral frequencies back to AR coefficients",
    "Hope is like the sun. If you only believe in it when you see it, you",
)
_r("lvq", "learning_vq", "Classification", "Learning vector quantization (LVQ1) classifier", "You were the chosen one!")
_r(
    "mca",
    "morphological_ca",
    "Decomposition",
    "Morphological component analysis: separate smooth and transient",
    "The dark side clouds everything.",
)
_r(
    "mcavg",
    "multiclass_avg",
    "Foundation",
    "Compute precision, recall, F1 with macro/micro/weighted averaging",
    "The greatest teacher, failure is.",
)
_r("mdist", "model_distance", "Modeling", "Compute AR model distance via reflection coefficients", "There is another.")
_r(
    "mexht",
    "mexican_hat",
    "Wavelet",
    "Mexican hat (Ricker) wavelet: negative normalized second derivative of Gaussian",
    "I",
)
_r(
    "mlpcl",
    "mlp_classify",
    "Classification",
    "Feedforward neural network for binary classification",
    "Luminous beings are we, not this crude matter.",
)
_r(
    "modwt",
    "modwt_decompose",
    "Wavelet",
    "Maximal Overlap Discrete Wavelet Transform (MODWT)",
    "I know what I have to do, but I don",
)
_r(
    "morsw",
    "morse_wavelet",
    "Wavelet",
    "Generalized Morse wavelet in the frequency domain",
    "Distribution helper.",
)
_r(
    "msadj",
    "misadjustment",
    "FilterDesign",
    "Compute the LMS misadjustment ratio",
    "The ability to speak does not make you intelligent.",
)
_r(
    "mstep",
    "max_step_size",
    "FilterDesign",
    "Compute the maximum stable LMS step size",
    "You must unlearn what you have learned.",
)
_r(
    "mtchb",
    "matched_filter_bank",
    "Detection",
    "Apply multiple matched filter templates and return best match",
    "In my experience, there is no such thing as luck.",
)
_r(
    "myodt",
    "myogram_onset",
    "Detection",
    "Detect EMG muscle activation onset using double threshold method",
    "Great, kid. Don",
)
_r(
    "nflor",
    "noise_floor",
    "Fourier",
    "Estimate the noise floor of a signal spectrum",
    "Fear is the path to the dark side.",
)
_r(
    "nmfsp",
    "nmf_sparse",
    "Decomposition",
    "Sparse NMF with L1 penalty via multiplicative updates",
    "There is always a bigger fish.",
)
_r(
    "nqstf",
    "nyquist_freq",
    "Foundation",
    "Compute the Nyquist frequency",
    "In my experience, there is no such thing as luck.",
)
_r(
    "ompfn",
    "omp_sparse",
    "Wavelet",
    "Orthogonal Matching Pursuit for sparse signal approximation",
    "In my experience there is no such thing as luck.",
)
_r(
    "parzn",
    "parzen_classify",
    "Classification",
    "Parzen window (kernel density estimation) classifier",
    "I find your lack of faith disturbing.",
)
_r(
    "pcaov",
    "pca_overlap",
    "Detection",
    "PCA on overlapping windows of a 1-D signal",
    "The belonging you seek is not behind you, it is ahead.",
)
_r(
    "pdthr",
    "peak_detect_threshold",
    "Detection",
    "Detect peaks exceeding a threshold with minimum distance constraint",
    "Never tell me the odds.",
)
_r(
    "ppint",
    "pp_interval",
    "Detection",
    "Compute P-P interval series from detected P-wave peak indices",
    "Stay on target.",
)
_r("prcpl", "precision_recall_curve", "Foundation", "Compute precision-recall curve and average precision", "Who")
_r("prdur", "pr_duration", "Foundation", "Measure PR interval from P-wave onset to QRS onset", "These aren")
_r(
    "predg",
    "prediction_gain",
    "Modeling",
    "Compute the linear prediction gain",
    "Difficult to see. Always in motion is the future.",
)
_r(
    "prflt",
    "polyphase_filter",
    "Decomposition",
    "Polyphase decomposition of a filter into M sub-filters",
    "Distribution helper.",
)
_r(
    "projp",
    "projection_pursuit",
    "Foundation",
    "Projection pursuit: find projections maximising non-Gaussianity",
    "Difficult to see. Always in motion is the future.",
)
_r("pwrdb", "power_to_db", "Foundation", "Convert power to decibels", "I find your lack of faith disturbing.")
_r(
    "qrsdr",
    "qrs_duration",
    "Foundation",
    "Measure QRS complex duration from onset to offset",
    "Distribution helper.",
)
_r(
    "qtint",
    "qt_interval",
    "Foundation",
    "Measure QT interval from QRS onset to T-wave offset",
    "There is always a bigger fish.",
)
_r(
    "rcfcv",
    "reflection_to_ar",
    "Modeling",
    "Convert reflection coefficients to AR coefficients (Levinson recursion)",
    "Distribution helper.",
)
_r(
    "reasn",
    "reassigned_stft",
    "Detection",
    "Reassigned Short-Time Fourier Transform spectrogram",
    "Let the past die. Kill it, if you have to.",
)
_r(
    "rejcl",
    "reject_option",
    "Classification",
    "Classify with a reject option for ambiguous samples",
    "Judge me by my size, do you?",
)
_r(
    "rfcls",
    "rf_classify_bio",
    "Classification",
    "Random forest classifier with feature sub-sampling at each split",
    "Distribution helper.",
)
_r(
    "ridgx",
    "ridge_extract",
    "Foundation",
    "Extract ridges (instantaneous frequency tracks) from a TF representation",
    "Every generation has a legend.",
)
_r(
    "rmsle",
    "rms_log_error",
    "Fourier",
    "Compute the RMS log spectral error",
    "Luminous beings are we, not this crude matter.",
)
_r("rocdt", "roc_det_curve", "Foundation", "Compute ROC and DET curves with AUC", "I")
_r(
    "rpca",
    "robust_pca",
    "Decomposition",
    "Robust PCA: decompose X = L + S (low-rank + sparse) via ADMM",
    "Power! Unlimited power!",
)
_r("rrvar", "rr_variability", "Foundation", "Compute HRV time-domain metrics from R-R intervals", "Who")
_r(
    "rsltn",
    "freq_resolution",
    "Foundation",
    "Compute the frequency resolution of a DFT",
    "Distribution helper.",
)
_r("sfdr", "sfdr_compute", "SignalOps", "Compute spurious-free dynamic range (SFDR)", "You were the chosen one!")
_r(
    "sinad",
    "sinad_compute",
    "SignalOps",
    "Compute SINAD (signal to noise and distortion ratio)",
    "The greatest teacher, failure is.",
)
_r(
    "slopd",
    "slope_detect",
    "Detection",
    "Detect onset/offset events based on signal slope (first derivative)",
    "Distribution helper.",
)
_r(
    "snrqt",
    "snr_quantization",
    "Foundation",
    "Compute the theoretical quantization SNR",
    "Distribution helper.",
)
_r(
    "sobi",
    "sobi_bss",
    "SignalOps",
    "Second-Order Blind Identification for source separation",
    "Hope is like the sun. If you only believe in it when you can see it.",
)
_r(
    "sofm",
    "self_org_map",
    "Foundation",
    "Self-organizing feature map (Kohonen, 1982)",
    "Distribution helper.",
)
_r(
    "sqztf",
    "synchrosqueeze",
    "Wavelet",
    "Synchrosqueezed Continuous Wavelet Transform",
    "You know, no matter how much we fought, I always hated watching you leave.",
)
_r(
    "sslcl",
    "semi_supervised",
    "Classification",
    "Self-training semi-supervised classifier",
    "The belonging you seek is not behind you, it is ahead.",
)
_r(
    "stbnr",
    "stability_margin",
    "Modeling",
    "Compute the stability margin: minimum distance of poles from unit circle",
    "The belonging you seek is not behind you, it is ahead.",
)
_r(
    "stckc",
    "stacking_classify",
    "Classification",
    "Stacking classifier using cross-validated base learner predictions",
    "Power! Unlimited power!",
)
_r(
    "stsgm",
    "st_segment",
    "Detection",
    "Analyze ST segment level between QRS offset and T-wave onset",
    "Distribution helper.",
)
_r(
    "svmln",
    "svm_linear",
    "Classification",
    "Linear SVM trained with sub-gradient descent on hinge loss",
    "Your focus determines your reality.",
)
_r(
    "svmpl",
    "svm_poly",
    "Classification",
    "Polynomial kernel SVM via dual coordinate ascent",
    "Distribution helper.",
)
_r(
    "svmrb",
    "svm_rbf",
    "Classification",
    "RBF kernel SVM using simplified SMO-style coordinate ascent",
    "In my experience there is no such thing as luck.",
)
_r(
    "swtfn",
    "swt_decompose",
    "Wavelet",
    "Stationary Wavelet Transform (algorithme a trous)",
    "The dark side clouds everything.",
)
_r(
    "sym8f",
    "symlet_coeffs",
    "Wavelet",
    "Symlet wavelet filter coefficients (near-symmetric Daubechies)",
    "I will finish what you started.",
)
_r(
    "thd",
    "thd_compute",
    "SignalOps",
    "Compute total harmonic distortion (THD)",
    "Truly wonderful, the mind of a child is.",
)
_r("tkeo", "teager_energy", "SignalOps", "Compute Teager-Kaiser energy operator", "Fear is the path to the dark side.")
_r("tmdur", "time_duration", "SignalOps", "Compute the time duration of a signal", "Do or do not. There is no try.")
_r(
    "tmpml",
    "template_match_lib",
    "SignalOps",
    "Match signal against a library of templates using normalized xcorr",
    "I find your lack of faith disturbing.",
)
_r(
    "tnsrd",
    "tensor_decompose",
    "Decomposition",
    "CP tensor decomposition via alternating least squares",
    "Distribution helper.",
)
_r(
    "trlrn",
    "transfer_learn",
    "Classification",
    "Simple transfer learning via subspace alignment (Fernando et al., 2013)",
    "We are what they grow beyond.",
)
_r(
    "vmd",
    "variational_mode",
    "Decomposition",
    "Variational Mode Decomposition of 1-D signal",
    "Truly wonderful the mind of a child is.",
)
_r("votcl", "voting_classify", "Classification", "Majority or weighted voting ensemble", "Let the Wookiee win.")
_r(
    "whtfn",
    "whitening",
    "Foundation",
    "Whiten (sphere) data to zero mean and identity covariance",
    "Luminous beings are we, not this crude matter.",
)
_r(
    "wssds",
    "wss_distance",
    "Fourier",
    "Compute weighted spectral slope distance",
    "Every choice you have made has led you to this moment.",
)
_r(
    "wvcoh",
    "wavelet_coherence",
    "Wavelet",
    "Wavelet coherence between two signals",
    "We are the spark that will light the fire.",
)
_r(
    "wvcor",
    "wavelet_correlation",
    "Wavelet",
    "Scale-by-scale wavelet correlation between two signals",
    "The greatest teacher, failure is.",
)
_r(
    "wvdns",
    "wavelet_denoise",
    "Wavelet",
    "Wavelet denoising via decomposition, thresholding, and reconstruction",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "wveng",
    "wavelet_energy",
    "Wavelet",
    "Compute energy distribution across wavelet subbands",
    "Impressive. Most impressive.",
)
_r(
    "wvent",
    "wavelet_entropy",
    "Wavelet",
    "Wavelet entropy from the normalized energy distribution",
    "I sense great fear in you.",
)
_r(
    "wvmra",
    "wavelet_mra",
    "Wavelet",
    "Multiresolution analysis: reconstruct detail and approximation at each level",
    "Great, kid. Don",
)
_r(
    "wvphs",
    "wavelet_phase",
    "Wavelet",
    "Extract phase from complex-valued wavelet coefficients",
    "Twice the pride, double the fall.",
)
_r(
    "wvpkt",
    "wavelet_packets",
    "Wavelet",
    "Wavelet packet decomposition — full binary tree",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "wvthr",
    "wavelet_threshold",
    "Wavelet",
    "Select threshold for wavelet coefficient shrinkage",
    "Judge me by my size, do you?",
)
_r(
    "wvvar",
    "wavelet_variance",
    "Wavelet",
    "Wavelet variance: variance decomposed across scales",
    "The Resistance will not be intimidated.",
)
_r("xwvlt", "cross_wavelet", "Wavelet", "Cross-wavelet spectrum of two signals", "Your eyes can deceive you. Don")
_r(
    "zcdet",
    "zero_cross_detect",
    "Detection",
    "Detect zero-crossing events and return indices",
    "Many of the truths we cling to depend on our point of view.",
)


_r(
    "adpthr",
    "adaptive_threshold_detect",
    "Detection",
    "Adaptive threshold detection based on local mean and std",
    "The dark side clouds everything.",
)
_r(
    "alcon",
    "algorithm_convergence",
    "Derivation",
    "Convergence diagnostics for an iterative algorithm",
    "Distribution helper.",
)
_r("ampcl", "amplitude_classify", "Classification", "Classify signal segments by amplitude level", "Stay on target.")
_r("bkdif", "backward_difference", "NumericalMethods", "Compute backward finite difference of signal *x*", "It")
_r("bserr", "bias_error", "Derivation", "Compute estimation bias", "The greatest teacher, failure is.")
_r("bspwv", "bspline_wavelet", "Wavelet", "Construct a B-spline wavelet", "This is a new day, a new beginning.")
_r(
    "bwlmt",
    "bandwidth_limit",
    "Derivation",
    "Analyse whether a signal bandwidth respects the Nyquist limit",
    "The dark side clouds everything.",
)
_r("ceemf", "ceemd_decompose", "Decomposition", "Complete Ensemble EMD decomposition", "Power, unlimited power!")
_r(
    "cntdf",
    "central_difference",
    "NumericalMethods",
    "Compute central finite difference of signal *x*",
    "There is another.",
)
_r(
    "coiwv",
    "coiflet_wavelet",
    "Wavelet",
    "Generate Coiflet wavelet filter coefficients",
    "We are what they grow beyond.",
)
_r(
    "cosdc",
    "cosine_decompose",
    "Decomposition",
    "Discrete cosine transform (DCT-II) decomposition",
    "Distribution helper.",
)
_r(
    "crlb",
    "cramer_rao_lower_bound",
    "Derivation",
    "Compute the Cramer-Rao lower bound from Fisher information",
    "Difficult to see. Always in motion is the future.",
)
_r("cwvsp", "cwt_spectrum", "Wavelet", "CWT scalogram/spectrum", "I find your lack of faith disturbing.")
_r("dbldt", "double_threshold", "Detection", "Double-threshold detection with hysteresis", "Never tell me the odds.")
_r(
    "dbwvl",
    "daubechies_wavelet",
    "Wavelet",
    "Generate Daubechies wavelet filter coefficients",
    "Much to learn you still have.",
)
_r(
    "drfmg",
    "drift_magnitude",
    "Foundation",
    "Estimate baseline drift magnitude",
    "So this is how liberty dies, with thunderous applause.",
)
_r(
    "drvhz",
    "derivative_hz",
    "FilterDesign",
    "Compute signal derivative via frequency domain differentiation",
    "Distribution helper.",
)
_r("durdt", "duration_detect", "Detection", "Filter events by duration", "Distribution helper.")
_r(
    "emdvr",
    "emd_variance_ratio",
    "Foundation",
    "Compute variance contribution of each IMF",
    "There is always a bigger fish.",
)
_r(
    "eqint",
    "equalization_inverse",
    "FilterDesign",
    "Compute a regularised inverse filter for channel equalisation",
    "You were the chosen one!",
)
_r(
    "ergod",
    "ergodicity_test",
    "Detection",
    "Test ergodicity via time vs ensemble averages",
    "Luminous beings are we, not this crude matter.",
)
_r(
    "evtcl",
    "event_cluster",
    "Detection",
    "Cluster detected events by morphological similarity using k-means",
    "Distribution helper.",
)
_r(
    "evtdn",
    "event_density",
    "Foundation",
    "Kernel density estimate of event times using Gaussian kernel",
    "Distribution helper.",
)
_r("evtrt", "event_rate", "Foundation", "Compute event firing rate over time windows", "Rebellions are built on hope.")
_r("faldt", "fall_time_detect", "Detection", "Measure fall time of detected events", "Impressive. Most impressive.")
_r(
    "fbnds",
    "filter_bounds",
    "FilterDesign",
    "Estimate filter transition band bounds",
    "The ability to speak does not make you intelligent.",
)
_r(
    "frdif",
    "forward_difference",
    "NumericalMethods",
    "Compute forward finite difference of signal *x*",
    "I find your lack of faith disturbing.",
)
_r("frmlt", "framelet_decompose", "Wavelet", "Framelet (tight frame) decomposition", "There is always a bigger fish.")
_r(
    "gqadr",
    "gauss_quadrature",
    "Derivation",
    "Integrate function *f* from *a* to *b* using Gauss-Legendre quadrature",
    "The belonging you seek is not behind you, it is ahead.",
)
_r(
    "grpsp",
    "group_sparse_decompose",
    "Decomposition",
    "Group sparsity decomposition via group lasso (proximal gradient)",
    "We are what they grow beyond.",
)
_r(
    "hthr",
    "hard_threshold",
    "Detection",
    "Hard thresholding operator: x * (|x| > lambda)",
    "Distribution helper.",
)
_r(
    "hysdt",
    "hysteresis_detect",
    "Detection",
    "Hysteresis-based event detector",
    "I find your lack of faith disturbing.",
)
_r(
    "imfex",
    "imf_extract",
    "Foundation",
    "Extract intrinsic mode functions via sifting",
    "Your focus determines your reality.",
)
_r(
    "insft",
    "instantaneous_freq",
    "Foundation",
    "Compute instantaneous frequency via the analytic signal",
    "Truly wonderful the mind of a child is.",
)
_r(
    "intdt",
    "interval_detect",
    "Detection",
    "Detect inter-event intervals and compute statistics",
    "The belonging you seek is not behind you, it is ahead.",
)
_r(
    "irls",
    "iteratively_reweighted_ls",
    "Foundation",
    "Iteratively reweighted least squares (IRLS) for Lp minimization",
    "So this is how liberty dies... with thunderous applause.",
)
_r(
    "isisz",
    "isi_analyze",
    "Derivation",
    "Inter-spike interval (ISI) analysis",
    "Luminous beings are we, not this crude matter.",
)
_r(
    "jsdcm",
    "joint_sparse_decompose",
    "Decomposition",
    "Joint (simultaneous) sparse decomposition via group lasso across signals",
    "This is the way.",
)
_r("ksvd", "ksvd_dictionary", "Derivation", "K-SVD dictionary learning", "Do or do not. There is no try.")
_r(
    "l1min",
    "l1_minimize",
    "Detection",
    "L1-minimization via ISTA (Iterative Shrinkage-Thresholding)",
    "Distribution helper.",
)
_r(
    "lnrty",
    "linearity_test",
    "Derivation",
    "Test system linearity via R-squared of a linear fit",
    "Truly wonderful the mind of a child is.",
)
_r(
    "maxov",
    "maximal_overlap_dwt",
    "Wavelet",
    "MODWT (Maximal Overlap DWT) -- non-decimated wavelet transform",
    "Distribution helper.",
)
_r(
    "mltan",
    "multiresolution_analysis",
    "Wavelet",
    "Full multiresolution analysis: decompose and reconstruct each level",
    "The greatest teacher, failure is.",
)
_r(
    "mpdcm",
    "matching_pursuit_decompose",
    "Decomposition",
    "Matching pursuit decomposition",
    "The belonging you seek is not behind you, it is ahead.",
)
_r(
    "msevr",
    "mse_variance_bias",
    "Decomposition",
    "Decompose MSE into bias squared and variance",
    "Much to learn you still have.",
)
_r(
    "pkdet",
    "peak_detect_advanced",
    "Detection",
    "Advanced peak detection with prominence and distance constraints",
    "Your eyes can deceive you. Don",
)
_r(
    "pllbd",
    "pll_bandwidth",
    "Derivation",
    "Compute phase-locked loop noise bandwidth",
    "I have a bad feeling about this.",
)
_r(
    "prsid",
    "parseval_identity",
    "Fourier",
    "Verify Parseval's theorem (time-frequency energy equality)",
    "Distribution helper.",
)
_r(
    "rconv",
    "convergence_rate",
    "Derivation",
    "Estimate convergence rate from an error sequence",
    "Do or do not. There is no try.",
)
_r("risdt", "rise_time_detect", "Detection", "Measure rise time of detected events", "We are what they grow beyond.")
_r(
    "rssgm",
    "reassigned_spectrogram",
    "Foundation",
    "Compute reassigned spectrogram for sharper time-frequency localisation",
    "The dark side clouds everything.",
)
_r(
    "seqdt",
    "sequential_detect",
    "Detection",
    "Sequential change-point detection",
    "In my experience there is no such thing as luck.",
)
_r(
    "smpbd",
    "sample_bound",
    "Derivation",
    "Compute sample size bound for a given confidence and margin",
    "In a dark place we find ourselves, and a little more knowledge lights our way.",
)
_r(
    "smpsn",
    "simpson_integrate",
    "Derivation",
    "Integrate signal *x* using Simpson's 1/3 rule",
    "We are what they grow beyond.",
)
_r(
    "snrth",
    "snr_threshold",
    "Detection",
    "Evaluate whether SNR meets the target BER threshold",
    "Your focus determines your reality.",
)
_r(
    "softt",
    "soft_threshold_fn",
    "Detection",
    "Proximal soft thresholding: prox_{tau * ||.||_1}(x)",
    "Luminous beings are we, not this crude matter.",
)
_r(
    "spcbd",
    "spectral_bound",
    "Fourier",
    "Compute the bandwidth containing a given fraction of spectral energy",
    "Never tell me the odds.",
)
_r(
    "sqsgm",
    "synchrosqueezed_transform",
    "Wavelet",
    "Synchrosqueezing of CWT coefficients",
    "Distribution helper.",
)
_r(
    "stflt",
    "subband_filter",
    "Wavelet",
    "Extract a specific subband from wavelet decomposition",
    "Never tell me the odds.",
)
_r(
    "sthr",
    "soft_threshold",
    "Detection",
    "Soft thresholding operator: sign(x) * max(|x| - lambda, 0)",
    "You were the chosen one!",
)
_r(
    "stner",
    "stationarity_test",
    "Detection",
    "Test signal stationarity via segment mean/variance comparison",
    "Fear is the path to the dark side.",
)
_r(
    "supps",
    "superposition_test",
    "Derivation",
    "Verify the superposition principle for an LTI system",
    "Distribution helper.",
)
_r(
    "symwv",
    "symlet_wavelet",
    "Wavelet",
    "Generate Symlet wavelet filter coefficients",
    "Luminous beings are we, not this crude matter.",
)
_r(
    "tfinq",
    "time_freq_uncertainty",
    "Fourier",
    "Measure Heisenberg time-frequency uncertainty",
    "Train yourself to let go of everything you fear to lose.",
)
_r(
    "tkcmp",
    "tucker_decompose",
    "Decomposition",
    "Tucker tensor decomposition via HOSVD + ALS refinement",
    "Never tell me the odds.",
)
_r("tmplb", "template_library", "Foundation", "Store/match signal templates against a library", "This is the way.")
_r(
    "tndcm",
    "tensor_decompose",
    "Decomposition",
    "CP tensor decomposition via alternating least squares",
    "Your focus determines your reality.",
)
_r(
    "trprl",
    "trapezoidal_integrate",
    "Derivation",
    "Integrate signal *x* using the trapezoidal rule",
    "Distribution helper.",
)
_r(
    "vmdfn",
    "variational_mode_decompose",
    "Decomposition",
    "Variational mode decomposition (VMD)",
    "In a dark place we find ourselves, and a little more knowledge lights our way.",
)
_r(
    "vtbal",
    "viterbi_align",
    "Foundation",
    "Viterbi algorithm for optimal state alignment",
    "Do or do not. There is no try.",
)
_r("wvbas", "wavelet_basis", "Wavelet", "Return wavelet filter bank coefficients", "Rebellions are built on hope.")
_r(
    "wvcrs",
    "wavelet_cross_spectrum",
    "Wavelet",
    "Cross-wavelet transform between two signals",
    "The belonging you seek is not behind you, it is ahead.",
)
_r(
    "wvdec",
    "wavelet_decompose",
    "Wavelet",
    "Multi-level wavelet decomposition",
    "You underestimate the power of the dark side.",
)
_r(
    "wvden",
    "wavelet_denoise",
    "Wavelet",
    "Wavelet denoising with soft/hard thresholding",
    "In my experience there is no such thing as luck.",
)
_r(
    "wvdl",
    "wavelet_dict_learn",
    "Wavelet",
    "Wavelet-initialized dictionary learning",
    "I find your lack of faith disturbing.",
)
_r(
    "wvenr",
    "wavelet_energy",
    "Wavelet",
    "Compute energy distribution across wavelet subbands",
    "Let the past die. Kill it, if you have to.",
)
_r(
    "wvflt",
    "wavelet_filter",
    "Wavelet",
    "Wavelet-based filtering: keep approximation or detail coefficients",
    "Hope is like the sun.",
)
_r(
    "wvmom",
    "wavelet_moments",
    "Wavelet",
    "Compute statistical moments (mean, var, skew, kurtosis) of wavelet coefficients",
    "Difficult to see. Always in motion is the future.",
)
_r(
    "wvrec",
    "wavelet_reconstruct",
    "Wavelet",
    "Reconstruct signal from wavelet coefficients",
    "Strike me down, and I will become more powerful.",
)
_r("wvscl", "wavelet_scalogram", "Wavelet", "Compute CWT scalogram (magnitude squared)", "So this is how liberty dies.")
_r("zcrdt", "zcr_detect", "Detection", "Zero-crossing rate (ZCR) based onset detection", "Great, kid. Don")

# -- SpatialVoting (24) -- Armstrong (2021)
_r(
    "amscl",
    "aldrich_mckelvey_scaling",
    "SpatialVoting",
    "Aldrich-McKelvey perceptual scaling",
    "Fear is the mind-killer. -- Bene Gesserit, Dune",
)
_r(
    "bblck",
    "blackbox_scaling",
    "SpatialVoting",
    "Blackbox/Basic Space ideal point scaling",
    "He who controls the spice controls the universe. -- Baron Harkonnen, Dune",
)
_r(
    "oocls",
    "optimal_classification",
    "SpatialVoting",
    "Optimal Classification nonparametric scaling",
    "In the darkest times, hope is something you give yourself. -- Uncle Iroh",
)
_r(
    "dblcn",
    "double_centering",
    "SpatialVoting",
    "Double-centering matrix for MDS",
    "The needs of the many outweigh the needs of the few. -- Spock, Star Trek",
)
_r(
    "cmds",
    "classical_metric_mds",
    "SpatialVoting",
    "Classical metric MDS (Torgerson)",
    "A process cannot be understood by stopping it. -- First Law of Mentat, Dune",
)
_r(
    "smcof",
    "smacof_mds",
    "SpatialVoting",
    "SMACOF stress minimization MDS",
    "The sleeper must awaken. -- Stilgar, Dune",
)
_r(
    "nmmds",
    "nonmetric_mds",
    "SpatialVoting",
    "Nonmetric MDS with ordinal constraints",
    "Make it so. -- Captain Picard, Star Trek",
)
_r(
    "mdsft",
    "mds_fit_statistics",
    "SpatialVoting",
    "MDS fit statistics (Mardia criterion)",
    "Resistance is futile. -- Borg, Star Trek",
)
_r(
    "ufstr",
    "unfolding_stress",
    "SpatialVoting",
    "Compute unfolding stress",
    "The mystery of life is not a problem to solve. -- Kynes, Dune",
)
_r(
    "mlsmu",
    "mlsmu6_unfolding",
    "SpatialVoting",
    "MLSMU6 alternating least-squares unfolding",
    "People's dreams never end! -- Blackbeard, One Piece",
)
_r(
    "smcuf",
    "smacof_unfolding",
    "SpatialVoting",
    "SMACOF rectangular unfolding",
    "Believe it! -- Naruto Uzumaki, Naruto",
)
_r(
    "idlpt",
    "ideal_point_recovery",
    "SpatialVoting",
    "Recover ideal points from unfolding",
    "I will become the Pirate King! -- Monkey D. Luffy, One Piece",
)
_r(
    "nomut",
    "nominate_utility",
    "SpatialVoting",
    "NOMINATE Gaussian spatial utility",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "nomvt",
    "nominate_vote_prob",
    "SpatialVoting",
    "NOMINATE single vote probability",
    "You think darkness is your ally. -- Bane, DC",
)
_r(
    "nomll",
    "nominate_loglikelihood",
    "SpatialVoting",
    "NOMINATE log-likelihood and GMP",
    "Yeah, I'm thinking I'm back. -- John Wick",
)
_r(
    "procr",
    "procrustes_rotation",
    "SpatialVoting",
    "Procrustes rotation/alignment",
    "Power comes in response to a need. -- Goku, Dragon Ball Z",
)
_r(
    "bamsc",
    "bayesian_am_scaling",
    "SpatialVoting",
    "Bayesian Aldrich-McKelvey scaling",
    "Arrakis teaches the attitude of the knife. -- Stilgar, Dune",
)
_r(
    "bmds",
    "bayesian_mds",
    "SpatialVoting",
    "Bayesian MDS with log-normal distances",
    "I have been and always shall be your friend. -- Spock, Star Trek",
)
_r(
    "bunfl",
    "bayesian_unfolding",
    "SpatialVoting",
    "Bayesian multidimensional unfolding",
    "A man's dream will never die! -- Blackbeard, One Piece",
)
_r(
    "cjrit",
    "cjr_irt_model",
    "SpatialVoting",
    "Clinton-Jackman-Rivers Bayesian IRT",
    "It's over 9000! -- Vegeta, Dragon Ball Z",
)
_r(
    "birtl",
    "bayesian_irt_likelihood",
    "SpatialVoting",
    "Bayesian IRT likelihood computation",
    "Those who abandon their friends are worse than scum. -- Kakashi, Naruto",
)
_r(
    "birtp",
    "bayesian_irt_posterior",
    "SpatialVoting",
    "Bayesian IRT posterior summaries",
    "I am the hope of the universe. -- Goku, Dragon Ball Z",
)
_r(
    "wnom",
    "wnominate_estimate",
    "SpatialVoting",
    "W-NOMINATE full estimation",
    "Whoever wins this war becomes justice. -- Doflamingo, One Piece",
)
_r("ocbin", "oc_binary_choice", "SpatialVoting", "OC on binary choice data", "Si vis pacem, para bellum. -- John Wick")
_r("orooc", "ordered_oc", "SpatialVoting", "Ordered OC for ordinal scales", "The spice must flow. -- Duke Leto, Dune")
_r(
    "avign",
    "anchoring_vignettes",
    "SpatialVoting",
    "Anchoring vignettes DIF correction",
    "I am Groot. -- Groot, Marvel",
)
_r(
    "idmds",
    "indscal_mds",
    "SpatialVoting",
    "INDSCAL individual differences MDS",
    "Live long and prosper. -- Spock, Star Trek",
)
_r(
    "nvect",
    "normal_vector_projection",
    "SpatialVoting",
    "Normal vector projection onto space",
    "The truth is rarely pure and never simple. -- Picard, Star Trek",
)
_r(
    "cutln",
    "cutting_line_mesh",
    "SpatialVoting",
    "Cutting lines for Coombs mesh",
    "A sword wields no strength unless the hand has courage. -- Hero of Time",
)
_r(
    "dwnom",
    "dw_nominate_estimate",
    "SpatialVoting",
    "DW-NOMINATE dynamic weighted estimation",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "nombs",
    "nominate_bootstrap_se",
    "SpatialVoting",
    "Parametric bootstrap NOMINATE SEs",
    "One Piece does exist! -- Whitebeard, One Piece",
)
_r(
    "anom",
    "alpha_nominate_estimate",
    "SpatialVoting",
    "Alpha-NOMINATE Bayesian MCMC",
    "A man's dream will never die! -- Blackbeard, One Piece",
)
_r(
    "oirt",
    "ordinal_irt_model",
    "SpatialVoting",
    "Ordinal IRT mixed factor analysis",
    "Believe it! -- Naruto Uzumaki, Naruto",
)
_r("dirt", "dynamic_irt_model", "SpatialVoting", "Dynamic IRT random walk priors", "Kamehameha! -- Goku, Dragon Ball Z")
_r(
    "emirt",
    "em_irt_model",
    "SpatialVoting",
    "EM algorithm for IRT estimation",
    "I am the bone of my sword. -- Archer, Fate",
)
_r(
    "btsca",
    "bootstrap_scaling_se",
    "SpatialVoting",
    "Bootstrap SEs for scaling methods",
    "The will of D. cannot be stopped. -- Dr. Kureha, One Piece",
)
_r(
    "wfish",
    "wordfish_scaling",
    "SpatialVoting",
    "Wordfish Poisson IRT text scaling",
    "Plus Ultra! -- All Might, My Hero Academia",
)

# -- SpatialVoting Ch1-2: Theory & Aldrich-McKelvey (49) -- Armstrong
_r(
    "quadu",
    "quadratic_utility",
    "SpatialVoting",
    "Quadratic spatial utility function",
    "The spice extends life. -- Fremen Proverb, Dune",
)
_r(
    "gauu",
    "gaussian_utility",
    "SpatialVoting",
    "Gaussian spatial utility function",
    "Fear cuts deeper than swords. -- Arya Stark, Game of Thrones",
)
_r(
    "linru",
    "linear_utility",
    "SpatialVoting",
    "Linear spatial utility function",
    "I drink and I know things. -- Tyrion Lannister, Game of Thrones",
)
_r(
    "spvot",
    "spatial_vote_probability",
    "SpatialVoting",
    "Spatial model vote probability",
    "Chaos is a ladder. -- Littlefinger, Game of Thrones",
)
_r(
    "medvt",
    "median_voter_theorem",
    "SpatialVoting",
    "Median voter theorem computation",
    "The night is dark and full of terrors. -- Melisandre, Game of Thrones",
)
_r(
    "hotlg",
    "hotelling_game",
    "SpatialVoting",
    "Hotelling spatial competition game",
    "What is dead may never die. -- Ironborn, Game of Thrones",
)
_r(
    "proxm",
    "proximity_model",
    "SpatialVoting",
    "Proximity spatial voting model",
    "Winter is coming. -- Stark, Game of Thrones",
)
_r(
    "dirvt",
    "directional_vote_model",
    "SpatialVoting",
    "Directional voting model (Rabinowitz-Macdonald)",
    "Valar morghulis. -- Braavosi, Game of Thrones",
)
_r(
    "spdst",
    "spatial_distance_matrix",
    "SpatialVoting",
    "Euclidean spatial distance matrix",
    "I am no one. -- Arya Stark, Game of Thrones",
)
_r(
    "cityb",
    "city_block_distance",
    "SpatialVoting",
    "City-block (Manhattan) spatial distance",
    "A Lannister always pays his debts. -- Tyrion, Game of Thrones",
)
_r(
    "sephy",
    "separating_hyperplane",
    "SpatialVoting",
    "Separating hyperplane for spatial model",
    "Dracarys. -- Daenerys Targaryen, Game of Thrones",
)
_r(
    "uncrt",
    "uncertainty_spatial_model",
    "SpatialVoting",
    "Uncertainty in spatial voting model",
    "By the old gods and the new. -- Catelyn Stark, Game of Thrones",
)
_r(
    "alien",
    "alienation_indifference",
    "SpatialVoting",
    "Alienation and indifference in voting",
    "Hold the door. -- Hodor, Game of Thrones",
)
_r(
    "bsthy",
    "basic_space_theory",
    "SpatialVoting",
    "Basic space theory dimensions",
    "Knowledge is power. -- Petyr Baelish, Game of Thrones",
)
_r(
    "idcon",
    "ideal_point_constraints",
    "SpatialVoting",
    "Ideal point identification constraints",
    "You know nothing, Jon Snow. -- Ygritte, Game of Thrones",
)
_r(
    "cndrc",
    "candidate_recovery",
    "SpatialVoting",
    "Candidate position recovery",
    "The lone wolf dies but the pack survives. -- Sansa Stark, Game of Thrones",
)
_r(
    "simrc",
    "simulate_rollcall",
    "SpatialVoting",
    "Simulate roll call voting data",
    "A mind needs books like a sword needs a whetstone. -- Tyrion, Game of Thrones",
)
_r(
    "simip",
    "simulate_ideal_points",
    "SpatialVoting",
    "Simulate ideal point configurations",
    "The things I do for love. -- Jaime Lannister, Game of Thrones",
)
_r(
    "simpe",
    "simulate_perceptual_data",
    "SpatialVoting",
    "Simulate perceptual placement data",
    "When you play the game of thrones, you win or you die. -- Cersei, Game of Thrones",
)
_r(
    "ammat",
    "am_constraint_matrix",
    "SpatialVoting",
    "Aldrich-McKelvey constraint matrix",
    "All men must serve. -- Jaqen H'ghar, Game of Thrones",
)
_r(
    "ameig",
    "am_eigendecomposition",
    "SpatialVoting",
    "AM eigendecomposition of constraints",
    "The man who passes the sentence should swing the sword. -- Ned Stark, Game of Thrones",
)
_r(
    "amwt",
    "am_weights",
    "SpatialVoting",
    "AM respondent weights computation",
    "Every flight begins with a fall. -- Three-Eyed Raven, Game of Thrones",
)
_r(
    "amfit",
    "am_fit_statistics",
    "SpatialVoting",
    "AM model fit statistics",
    "Burn them all. -- Aerys Targaryen, Game of Thrones",
)
_r(
    "amr2",
    "am_r_squared",
    "SpatialVoting",
    "AM explained variance (R-squared)",
    "Fire and blood. -- Targaryen Words, Game of Thrones",
)
_r(
    "amneg",
    "am_negative_weights",
    "SpatialVoting",
    "AM negative weight diagnostics",
    "We do not sow. -- Greyjoy Words, Game of Thrones",
)
_r(
    "amci",
    "am_confidence_intervals",
    "SpatialVoting",
    "AM bootstrap confidence intervals",
    "Growing strong. -- Tyrell Words, Game of Thrones",
)
_r(
    "bbwt",
    "bb_weight_matrix",
    "SpatialVoting",
    "Blackbox weight matrix construction",
    "Ours is the fury. -- Baratheon Words, Game of Thrones",
)
_r(
    "bbvar",
    "bb_variance_explained",
    "SpatialVoting",
    "Blackbox variance explained per dimension",
    "Hear me roar. -- Lannister Words, Game of Thrones",
)
_r(
    "bbr2",
    "bb_r_squared",
    "SpatialVoting",
    "Blackbox fit R-squared",
    "Family, duty, honor. -- Tully Words, Game of Thrones",
)
_r(
    "bbdim",
    "bb_dimensionality_test",
    "SpatialVoting",
    "Blackbox dimensionality test",
    "As high as honor. -- Arryn Words, Game of Thrones",
)
_r(
    "bbss",
    "bb_singular_values",
    "SpatialVoting",
    "Blackbox singular value spectrum",
    "Unbowed, unbent, unbroken. -- Martell Words, Game of Thrones",
)
_r(
    "amres",
    "am_residuals",
    "SpatialVoting",
    "AM residual diagnostics",
    "The North remembers. -- Northern Lords, Game of Thrones",
)
_r(
    "polck",
    "polarity_check",
    "SpatialVoting",
    "Check and enforce polarity of estimates",
    "What do we say to death? Not today. -- Syrio Forel, Game of Thrones",
)
_r(
    "rcode",
    "rollcall_recode",
    "SpatialVoting",
    "Recode roll call matrix for analysis",
    "I choose violence. -- Cersei Lannister, Game of Thrones",
)
_r(
    "plam",
    "plot_am_scaling",
    "SpatialVoting",
    "Plot Aldrich-McKelvey scaling results",
    "There is only one god, and his name is Death. -- Syrio, Game of Thrones",
)
_r(
    "histip",
    "histogram_ideal_points",
    "SpatialVoting",
    "Histogram of ideal point estimates",
    "Dark wings, dark words. -- Old Nan, Game of Thrones",
)
_r(
    "scidl",
    "scatterplot_ideal",
    "SpatialVoting",
    "Scatterplot of ideal points",
    "The more people you love, the weaker you are. -- Cersei, Game of Thrones",
)
_r(
    "stimp",
    "stimuli_positions",
    "SpatialVoting",
    "Extract stimuli positions from scaling",
    "Power resides where men believe it resides. -- Varys, Game of Thrones",
)
_r(
    "dnpos",
    "density_positions",
    "SpatialVoting",
    "Kernel density of legislator positions",
    "A very small man can cast a very large shadow. -- Varys, Game of Thrones",
)
_r(
    "dnneg",
    "density_negative_wt",
    "SpatialVoting",
    "Density plot of negative weight cases",
    "Any man who must say I am the king is no true king. -- Tywin, Game of Thrones",
)
_r(
    "plbb",
    "plot_bb_scaling",
    "SpatialVoting",
    "Plot Blackbox scaling results",
    "I shall wear no crowns and win no glory. -- Night's Watch, Game of Thrones",
)
_r(
    "bbgrp",
    "bb_group_means",
    "SpatialVoting",
    "Blackbox group mean ideal points",
    "The wall defends itself. -- Night's Watch, Game of Thrones",
)
_r(
    "htmap",
    "heatmap_agreement",
    "SpatialVoting",
    "Heatmap of voting agreement matrix",
    "Stick them with the pointy end. -- Jon Snow, Game of Thrones",
)
_r(
    "outam",
    "outlier_am_diagnostics",
    "SpatialVoting",
    "AM outlier and leverage diagnostics",
    "A lion does not concern himself with the opinion of sheep. -- Tywin, Game of Thrones",
)
_r(
    "vrnrm",
    "variance_normalization",
    "SpatialVoting",
    "Variance normalization for spatial model",
    "There are no men like me. Only me. -- Jaime Lannister, Game of Thrones",
)
_r(
    "prbit",
    "probit_spatial_link",
    "SpatialVoting",
    "Probit link for spatial vote model",
    "I am the sword in the darkness. -- Night's Watch, Game of Thrones",
)
_r(
    "ncoef",
    "nominate_coefficients",
    "SpatialVoting",
    "Extract NOMINATE model coefficients",
    "Night gathers and now my watch begins. -- Night's Watch, Game of Thrones",
)
_r(
    "isssb",
    "issue_salience_weights",
    "SpatialVoting",
    "Issue salience weights by dimension",
    "The common people pray for rain. -- Jorah Mormont, Game of Thrones",
)
_r(
    "fctvt",
    "factor_vote_analysis",
    "SpatialVoting",
    "Factor analysis of vote matrix",
    "First lesson: stick them with the pointy end. -- Arya Stark, Game of Thrones",
)

# -- SpatialVoting Ch3-4: MDS & Unfolding (48) -- Armstrong
_r(
    "dst2s",
    "distance_to_similarity",
    "SpatialVoting",
    "Convert distance to similarity matrix",
    "In a hole in the ground there lived a hobbit. -- Tolkien, LOTR",
)
_r(
    "eigvl",
    "eigenvalues_mds",
    "SpatialVoting",
    "Eigenvalues for MDS dimensionality",
    "Not all those who wander are lost. -- Tolkien, LOTR",
)
_r(
    "eigvc",
    "eigenvectors_mds",
    "SpatialVoting",
    "Eigenvectors for MDS coordinate recovery",
    "Even the smallest person can change the course of the future. -- Galadriel, LOTR",
)
_r(
    "crdsc",
    "coordinate_scaling",
    "SpatialVoting",
    "Scale MDS coordinates by eigenvalues",
    "All we have to decide is what to do with the time given us. -- Gandalf, LOTR",
)
_r(
    "mdsrk",
    "mds_rank_image",
    "SpatialVoting",
    "Rank-image transformation for nonmetric MDS",
    "There is some good in this world, and it is worth fighting for. -- Sam, LOTR",
)
_r(
    "ssefn",
    "sse_function",
    "SpatialVoting",
    "Sum of squared error loss for MDS",
    "I would rather share one lifetime with you. -- Arwen, LOTR",
)
_r(
    "optsa",
    "optimize_simulated_annealing",
    "SpatialVoting",
    "Simulated annealing MDS optimizer",
    "One does not simply walk into Mordor. -- Boromir, LOTR",
)
_r(
    "optnm",
    "optimize_nelder_mead",
    "SpatialVoting",
    "Nelder-Mead MDS optimizer",
    "A wizard is never late. -- Gandalf, LOTR",
)
_r(
    "inits",
    "mds_initial_solution",
    "SpatialVoting",
    "Initial configuration for MDS",
    "The world is changed. -- Galadriel, LOTR",
)
_r(
    "major",
    "majorization_step",
    "SpatialVoting",
    "SMACOF majorization update step",
    "I am a servant of the Secret Fire. -- Gandalf, LOTR",
)
_r("smcnv", "smacof_convergence", "SpatialVoting", "SMACOF convergence diagnostics", "Fly, you fools! -- Gandalf, LOTR")
_r(
    "isorg",
    "isotonic_regression_mds",
    "SpatialVoting",
    "Isotonic regression for nonmetric MDS",
    "The ring has awoken. -- Gandalf, LOTR",
)
_r(
    "dispr",
    "disparity_transform",
    "SpatialVoting",
    "Disparity (monotone) transform for MDS",
    "My precious. -- Gollum, LOTR",
)
_r("shepd", "shepard_diagram", "SpatialVoting", "Shepard diagram for MDS fit", "You shall not pass! -- Gandalf, LOTR")
_r("strs1", "stress_one", "SpatialVoting", "Kruskal stress-1 measure", "Speak friend and enter. -- Moria Gate, LOTR")
_r("strs2", "stress_two", "SpatialVoting", "Stress-2 (normalized) measure", "The road goes ever on. -- Bilbo, LOTR")
_r(
    "strpp",
    "stress_per_point",
    "SpatialVoting",
    "Per-point stress contributions",
    "I can't carry it for you, but I can carry you. -- Sam, LOTR",
)
_r(
    "gofmd",
    "goodness_of_fit_mds",
    "SpatialVoting",
    "Goodness-of-fit summary for MDS",
    "There and back again. -- Bilbo Baggins, LOTR",
)
_r(
    "scree",
    "scree_plot_mds",
    "SpatialVoting",
    "Scree plot of MDS stress by dimension",
    "Certainty of death. Small chance of success. -- Gimli, LOTR",
)
_r(
    "elbdm",
    "elbow_dimensionality",
    "SpatialVoting",
    "Elbow method for MDS dimensionality",
    "That still only counts as one! -- Gimli, LOTR",
)
_r(
    "prref",
    "procrustes_reference",
    "SpatialVoting",
    "Set Procrustes reference configuration",
    "Deeds will not be less valiant because they are unpraised. -- Aragorn, LOTR",
)
_r(
    "rotmt",
    "rotation_matrix",
    "SpatialVoting",
    "Compute optimal rotation matrix",
    "The beacons are lit! -- Aragorn, LOTR",
)
_r("mdslb", "mds_lower_bound", "SpatialVoting", "Lower bound on MDS dimensionality", "For Frodo. -- Aragorn, LOTR")
_r(
    "mdsex",
    "mds_extract_coordinates",
    "SpatialVoting",
    "Extract MDS coordinate matrix",
    "A day may come when the courage of men fails. -- Aragorn, LOTR",
)
_r(
    "mdsel",
    "mdsel",
    "ModelSelection",
    "Model selection via cross-validated loss (Kosorok 2008)",
    "I see you. -- Sauron, LOTR",
)
_r(
    "msswt",
    "mds_subject_weights",
    "SpatialVoting",
    "Subject weights for individual differences MDS",
    "The hour is late. -- Gandalf, LOTR",
)
_r(
    "impme",
    "impute_missing_mds",
    "SpatialVoting",
    "Impute missing values for MDS input",
    "End? No, the journey doesn't end here. -- Gandalf, LOTR",
)
_r(
    "mdsmm",
    "mds_missing_mask",
    "SpatialVoting",
    "Missing data mask for MDS",
    "Home is behind, the world ahead. -- Pippin, LOTR",
)
_r(
    "dstcm",
    "distance_comparison",
    "SpatialVoting",
    "Compare observed vs fitted distances",
    "Short cuts make long delays. -- Pippin, LOTR",
)
_r(
    "rcmat",
    "rollcall_matrix",
    "SpatialVoting",
    "Construct roll call matrix from votes",
    "Don't adventures ever have an end? -- Bilbo, LOTR",
)
_r(
    "spkpf",
    "spearman_kruskal_pf",
    "SpatialVoting",
    "Spearman-Kruskal point-fitting MDS",
    "It's a dangerous business going out your door. -- Bilbo, LOTR",
)
_r(
    "therm",
    "thermometer_scaling",
    "SpatialVoting",
    "Feeling thermometer scaling model",
    "Second breakfast? -- Pippin, LOTR",
)
_r(
    "unfst",
    "unfolding_start_config",
    "SpatialVoting",
    "Starting configuration for unfolding",
    "Po-ta-toes! -- Sam, LOTR",
)
_r(
    "mlsit",
    "mls_iterate",
    "SpatialVoting",
    "MLS alternating least-squares iteration",
    "I made a promise, Mr. Frodo. -- Sam, LOTR",
)
_r(
    "mlscv",
    "mls_convergence_check",
    "SpatialVoting",
    "MLS convergence criterion check",
    "There is no curse in Elvish for this treachery. -- Treebeard, LOTR",
)
_r(
    "mlsms",
    "mls_missing_strategy",
    "SpatialVoting",
    "MLS missing data strategy",
    "We are Groot. -- Treebeard (honorary), LOTR",
)
_r(
    "ufcmp",
    "unfolding_compare",
    "SpatialVoting",
    "Compare unfolding solutions",
    "The Ents are going to war. -- Merry, LOTR",
)
_r(
    "mssns",
    "missing_sensitivity",
    "SpatialVoting",
    "Sensitivity analysis for missing data",
    "Nobody tosses a Dwarf! -- Gimli, LOTR",
)
_r(
    "plidl",
    "plot_ideal_unfolding",
    "SpatialVoting",
    "Plot ideal points from unfolding",
    "They're taking the hobbits to Isengard! -- Legolas, LOTR",
)
_r(
    "pfcrc",
    "preference_circle",
    "SpatialVoting",
    "Draw preference circles in unfolding",
    "The world is indeed full of peril. -- Haldir, LOTR",
)
_r(
    "sim2d",
    "simulate_2d_spatial",
    "SpatialVoting",
    "Simulate 2D spatial voting data",
    "Faithless is he who says farewell when the road darkens. -- Gimli, LOTR",
)
_r(
    "simth",
    "simulate_thermometer",
    "SpatialVoting",
    "Simulate feeling thermometer data",
    "May the wind under your wings bear you where the sun sails. -- Gandalf, LOTR",
)
_r(
    "ufr2",
    "unfolding_r_squared",
    "SpatialVoting",
    "Unfolding R-squared fit measure",
    "I will not say do not weep. -- Gandalf, LOTR",
)
_r(
    "ufres",
    "unfolding_residuals",
    "SpatialVoting",
    "Unfolding residual matrix",
    "In place of a Dark Lord you would have a Queen. -- Galadriel, LOTR",
)
_r(
    "ufout",
    "unfolding_outliers",
    "SpatialVoting",
    "Outlier detection in unfolding",
    "Keep it secret. Keep it safe. -- Gandalf, LOTR",
)
_r(
    "rcagr",
    "rollcall_agreement",
    "SpatialVoting",
    "Roll call agreement index",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "rcdst",
    "rollcall_distance",
    "SpatialVoting",
    "Pairwise roll call distance matrix",
    "True courage is not knowing when to take a life, but when to spare one. -- Gandalf, LOTR",
)
_r(
    "rcdrp",
    "rollcall_drop_lopsided",
    "SpatialVoting",
    "Drop lopsided roll calls",
    "All that is gold does not glitter. -- Tolkien, LOTR",
)
_r(
    "rcprd",
    "rollcall_predict",
    "SpatialVoting",
    "Predict votes from spatial model",
    "The old that is strong does not wither. -- Tolkien, LOTR",
)

# -- SpatialVoting Ch5-6: Legislative Scaling & Bayesian (50) -- Armstrong
_r(
    "rcred",
    "rollcall_reduce",
    "SpatialVoting",
    "Reduce roll call to informative votes",
    "Do or do not, there is no try., Star Wars",
)
_r(
    "rcsub",
    "rollcall_subset",
    "SpatialVoting",
    "Subset roll call by session/party",
    "In my experience there is no such thing as luck. --, Star Wars",
)
_r(
    "gmpst",
    "gmp_start_values",
    "SpatialVoting",
    "GMP starting values for NOMINATE",
    "Your focus determines your reality. -- Qui-Gon Jinn, Star Wars",
)
_r(
    "prest",
    "predict_spatial_vote",
    "SpatialVoting",
    "Predict vote from spatial estimates",
    "The greatest teacher, failure is., Star Wars",
)
_r(
    "apres",
    "aggregate_prediction_error",
    "SpatialVoting",
    "Aggregate prediction error rate (APRE)",
    "Truly wonderful the mind of a child is., Star Wars",
)
_r(
    "rcerr",
    "rollcall_error_analysis",
    "SpatialVoting",
    "Classification error analysis by vote",
    "Much to learn you still have., Star Wars",
)
_r(
    "nopar",
    "nominate_parameters",
    "SpatialVoting",
    "NOMINATE signal-to-noise parameters",
    "The dark side clouds everything., Star Wars",
)
_r(
    "noidl",
    "nominate_ideal_points",
    "SpatialVoting",
    "Extract NOMINATE ideal point estimates",
    "Train yourself to let go of everything you fear to lose., Star Wars",
)
_r(
    "nobll",
    "nominate_bill_params",
    "SpatialVoting",
    "NOMINATE bill midpoint and spread",
    "Patience you must have., Star Wars",
)
_r(
    "nosco",
    "nominate_scores",
    "SpatialVoting",
    "NOMINATE coordinate scores with SEs",
    "You must unlearn what you have learned., Star Wars",
)
_r(
    "nocse",
    "nominate_cutting_se",
    "SpatialVoting",
    "NOMINATE cutting line standard errors",
    "Difficult to see. Always in motion is the future., Star Wars",
)
_r(
    "occln",
    "oc_cutting_lines",
    "SpatialVoting",
    "Optimal Classification cutting lines",
    "Your path you must decide., Star Wars",
)
_r(
    "ocsvm",
    "oc_svm_boundary",
    "SpatialVoting",
    "OC via SVM decision boundary",
    "Distribution helper.",
)
_r(
    "ocnrm",
    "oc_normal_vectors",
    "SpatialVoting",
    "OC normal vector estimates",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "ocmsh",
    "oc_misclassified",
    "SpatialVoting",
    "OC misclassified votes analysis",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "fitst",
    "fit_statistics_compare",
    "SpatialVoting",
    "Compare fit across scaling methods",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "ctprx",
    "cutting_point_proximity",
    "SpatialVoting",
    "Cutting point proximity to ideal points",
    "Stay on target. -- Gold Five, Star Wars",
)
_r(
    "plcut",
    "plot_cutting_lines",
    "SpatialVoting",
    "Plot cutting lines over ideal points",
    "It's a trap! -- Admiral Ackbar, Star Wars",
)
_r(
    "plcmb",
    "plot_compare_methods",
    "SpatialVoting",
    "Plot comparison of scaling methods",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "plpty",
    "plot_party_ideal",
    "SpatialVoting",
    "Plot ideal points colored by party",
    "Let the Wookiee win. -- C-3PO, Star Wars",
)
_r(
    "lglbl",
    "legislator_labels",
    "SpatialVoting",
    "Annotate legislator labels on plot",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "tmtrc",
    "temporal_trace_plot",
    "SpatialVoting",
    "Temporal trace of ideal point movement",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "smvot",
    "simulate_spatial_votes",
    "SpatialVoting",
    "Simulate spatial voting legislature",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "smutl",
    "simulate_utility_data",
    "SpatialVoting",
    "Simulate utility-based voting data",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "smchc",
    "simulate_choice_set",
    "SpatialVoting",
    "Simulate multi-alternative choice set",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "bamcm",
    "bam_mcmc_sampler",
    "SpatialVoting",
    "Bayesian AM MCMC sampler",
    "So this is how liberty dies. -- Padme, Star Wars",
)
_r(
    "bampr",
    "bam_prior_setup",
    "SpatialVoting",
    "BAM prior specification",
    "Hope is like the sun. -- Poe Dameron, Star Wars",
)
_r(
    "bambn",
    "bam_burn_in",
    "SpatialVoting",
    "BAM burn-in diagnostics",
    "We are the spark that will light the fire. -- Poe, Star Wars",
)
_r(
    "bamth",
    "bam_thinning",
    "SpatialVoting",
    "BAM thinning and effective sample size",
    "Distribution helper.",
)
_r(
    "bamse",
    "bam_standard_errors",
    "SpatialVoting",
    "BAM posterior standard errors",
    "Distribution helper.",
)
_r("trcpl", "trace_plot_mcmc", "SpatialVoting", "Trace plot of MCMC chains", "Be with me. -- Rey, Star Wars")
_r(
    "pstdn",
    "posterior_density",
    "SpatialVoting",
    "Posterior density of ideal points",
    "I will finish what you started. -- Kylo Ren, Star Wars",
)
_r(
    "gelrb",
    "gelman_rubin_diagnostic",
    "SpatialVoting",
    "Gelman-Rubin convergence diagnostic",
    "Let the past die. -- Kylo Ren, Star Wars",
)
_r(
    "effsz",
    "effective_sample_size",
    "SpatialVoting",
    "Effective sample size from MCMC",
    "We are what they grow beyond., Star Wars",
)
_r(
    "acfmc",
    "acf_mcmc_chain",
    "SpatialVoting",
    "Autocorrelation function of MCMC chain",
    "In time the suffering of your people will persuade you. -- Count Dooku, Star Wars",
)
_r(
    "pstpc",
    "posterior_predictive_check",
    "SpatialVoting",
    "Posterior predictive vote check",
    "The ability to speak does not make you intelligent. -- Qui-Gon, Star Wars",
)
_r(
    "bypvl",
    "bayes_p_value",
    "SpatialVoting",
    "Bayesian p-value for model fit",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "irtpb",
    "irt_probit_spatial",
    "SpatialVoting",
    "IRT probit parameterization for spatial",
    "Your overconfidence is your weakness. -- Luke, Star Wars",
)
_r(
    "iccfn",
    "icc_function_spatial",
    "SpatialVoting",
    "Item characteristic curve for spatial IRT",
    "Distribution helper.",
)
_r(
    "irtdc",
    "irt_discrimination",
    "SpatialVoting",
    "IRT discrimination parameters for bills",
    "You were the chosen one! --, Star Wars",
)
_r("irthx", "irt_hierarchical", "SpatialVoting", "Hierarchical IRT model", "There is another., Star Wars")
_r(
    "irtva",
    "irt_variational",
    "SpatialVoting",
    "Variational inference for IRT",
    "Already know you that which you need., Star Wars",
)
_r("emexp", "em_e_step", "SpatialVoting", "EM algorithm E-step expectation", "A new hope. -- Title Card, Star Wars")
_r(
    "emmax",
    "em_m_step",
    "SpatialVoting",
    "EM algorithm M-step maximization",
    "Distribution helper.",
)
_r(
    "emcnv",
    "em_convergence_check",
    "SpatialVoting",
    "EM convergence criterion check",
    "I know what I have to do but I don't have the strength. -- Kylo Ren, Star Wars",
)
_r(
    "emllk",
    "em_log_likelihood",
    "SpatialVoting",
    "EM observed log-likelihood",
    "Luminous beings are we, not this crude matter., Star Wars",
)
_r(
    "mltch",
    "multi_chain_diagnostics",
    "SpatialVoting",
    "Multi-chain MCMC diagnostics",
    "This is the way. -- Din Djarin, The Mandalorian",
)
_r(
    "chnvg",
    "chain_convergence",
    "SpatialVoting",
    "Chain convergence assessment",
    "I have spoken. -- Kuiil, The Mandalorian",
)
_r(
    "pstcp",
    "posterior_comparison",
    "SpatialVoting",
    "Compare posteriors across models",
    "Wherever I go, he goes. -- Din Djarin, The Mandalorian",
)
_r(
    "dimts",
    "dimensionality_test_bayes",
    "SpatialVoting",
    "Bayesian dimensionality testing",
    "Distribution helper.",
)

# ── Schabenberger & Gotway 2005: Point Patterns ─────────────────────────────
_r(
    "sghpp",
    "homogeneous_poisson",
    "SpatialStat",
    "Homogeneous Poisson process simulation",
    "A wizard is never late. -- Gandalf, LOTR",
)
_r(
    "sgipp",
    "inhomogeneous_poisson",
    "SpatialStat",
    "Inhomogeneous Poisson process simulation",
    "Not all those who wander are lost. -- Tolkien, LOTR",
)
_r(
    "sgcox",
    "cox_process",
    "SpatialStat",
    "Cox (doubly stochastic) process simulation",
    "It does not do to dwell on dreams. -- Dumbledore, Harry Potter",
)
_r(
    "sgthm",
    "thomas_process",
    "SpatialStat",
    "Thomas cluster process simulation",
    "Toss a coin to your Witcher. -- Jaskier, The Witcher",
)
_r(
    "sgmtc",
    "matern_cluster_process",
    "SpatialStat",
    "Matern cluster process simulation",
    "Rise, Tarnished. -- Melina, Elden Ring",
)
_r("sgcsr", "csr_test", "SpatialStat", "Complete spatial randomness test", "Praise the Sun. -- Solaire, Dark Souls")
_r(
    "sgnnd",
    "nearest_neighbor_distances",
    "SpatialStat",
    "Nearest-neighbor distances for point patterns",
    "Tonight, Gehrman joins the hunt. -- Moon Presence, Bloodborne",
)
_r(
    "sgkfn",
    "ripley_k_function",
    "SpatialStat",
    "Ripley K function estimation",
    "Distribution helper.",
)
_r("sglfn", "l_function", "SpatialStat", "Besag L function (transformed K)", "I should go. -- Shepard, Mass Effect")
_r(
    "sggfn",
    "g_function_nearest_neighbor",
    "SpatialStat",
    "G function (nearest-neighbor CDF)",
    "We are ODST. -- The Rookie, Halo 3 ODST",
)
_r(
    "sgpcf",
    "pair_correlation_function",
    "SpatialStat",
    "Pair correlation function g(r)",
    "The cycle ends here. -- Kratos, God of War",
)
_r(
    "sg2nd",
    "second_order_intensity",
    "SpatialStat",
    "Second-order intensity estimation",
    "You are enough. -- Aloy, Horizon Zero Dawn",
)
_r(
    "sgint",
    "intensity_estimate",
    "SpatialStat",
    "Intensity estimation for point patterns",
    "The promise has been made. -- Lightning, Final Fantasy XIII",
)
_r(
    "sgqdr",
    "quadrat_count_test",
    "SpatialStat",
    "Quadrat count test for point patterns",
    "I am thou, thou art I. -- Persona 5",
)
_r(
    "sgqag",
    "quadrat_aggregation",
    "SpatialStat",
    "Multi-scale quadrat aggregation analysis",
    "Hesitation is defeat. -- Isshin, Sekiro",
)
_r(
    "sgidx",
    "index_of_dispersion",
    "SpatialStat",
    "Index of dispersion (VMR) for point counts",
    "No cost too great. -- The Pale King, Hollow Knight",
)
_r(
    "sgedg",
    "edge_correction",
    "SpatialStat",
    "Edge correction weights for point patterns",
    "One does not simply walk into Mordor. -- Boromir, LOTR",
)
_r(
    "sgenv",
    "simulation_envelope",
    "SpatialStat",
    "Simulation envelope for point pattern statistics",
    "After all this time? Always. -- Snape, Harry Potter",
)
_r(
    "sgmkd",
    "marked_point_summary",
    "SpatialStat",
    "Marked point pattern summary statistics",
    "The world does not need a hero. It needs a professional. -- Geralt, The Witcher",
)
_r(
    "sgvor",
    "voronoi_tessellation",
    "SpatialStat",
    "Voronoi tessellation for point patterns",
    "Put these foolish ambitions to rest. -- Margit, Elden Ring",
)

# ── Schabenberger & Gotway 2005: Semivariograms & Covariance ────────────────
_r(
    "sgemp",
    "empirical_semivariogram",
    "SpatialStat",
    "Empirical semivariogram estimation",
    "Thou who art Undead, art chosen. -- Oscar, Dark Souls",
)
_r(
    "sgcld",
    "variogram_cloud",
    "SpatialStat",
    "Variogram cloud (all pairwise differences)",
    "A hunter must hunt. -- Eileen, Bloodborne",
)
_r(
    "sgsph",
    "spherical_variogram",
    "SpatialStat",
    "Spherical variogram model",
    "Spartans never die. -- Catherine Halsey, Halo",
)
_r(
    "sgexp",
    "exponential_variogram",
    "SpatialStat",
    "Exponential variogram model",
    "Had to be me. Someone else might have gotten it wrong. -- Mordin, Mass Effect",
)
_r(
    "sggau",
    "gaussian_variogram",
    "SpatialStat",
    "Gaussian variogram model",
    "Do not be sorry. Be better. -- Kratos, God of War",
)
_r(
    "sgmat",
    "matern_variogram",
    "SpatialStat",
    "Matern variogram model",
    "We must be brave. -- Aloy, Horizon Forbidden West",
)
_r(
    "sgpow",
    "power_variogram",
    "SpatialStat",
    "Power variogram model",
    "You don't need a reason to help people. -- Zidane, Final Fantasy IX",
)
_r("sgnst", "nested_variogram", "SpatialStat", "Nested (composite) variogram model", "For real?! -- Ryuji, Persona 5")
_r(
    "sgnug",
    "nugget_effect_estimate",
    "SpatialStat",
    "Nugget effect estimation from empirical variogram",
    "How my blood boils. -- Genichiro, Sekiro",
)
_r(
    "sgprn",
    "practical_range",
    "SpatialStat",
    "Practical (effective) range estimation",
    "The void is patient. -- The Radiance, Hollow Knight",
)
_r(
    "sglgc",
    "lag_class_binning",
    "SpatialStat",
    "Lag class binning for variogram estimation",
    "The road goes ever on and on. -- Bilbo, LOTR",
)
_r(
    "sgwls",
    "wls_variogram_fit",
    "SpatialStat",
    "Weighted least squares variogram fitting",
    "It takes a great deal of bravery to stand up to our enemies. -- Dumbledore, Harry Potter",
)
_r(
    "sgcvg",
    "cross_validation_variogram",
    "SpatialStat",
    "Leave-one-out cross-validation for variogram models",
    "Evil is evil. Lesser, greater, middling. -- Geralt, The Witcher",
)
_r(
    "sgcov",
    "covariance_function_estimate",
    "SpatialStat",
    "Spatial covariance function estimation",
    "Seek the Elden Ring. -- Two Fingers, Elden Ring",
)
_r("sgcrv", "cross_variogram", "SpatialStat", "Cross-variogram estimation", "You died. -- Dark Souls")
_r(
    "sgacf",
    "spatial_acf",
    "SpatialStat",
    "Spatial autocorrelation function per lag",
    "Grant us eyes. -- Micolash, Bloodborne",
)
_r(
    "sgspc",
    "spectral_density",
    "SpatialStat",
    "Spectral density estimation for spatial data",
    "The Great Journey waits for no one. -- Regret, Halo 2",
)
_r(
    "sgpdf",
    "positive_definiteness_check",
    "SpatialStat",
    "Positive definiteness check for covariance matrices",
    "We fight or we die. -- Wrex, Mass Effect",
)
_r(
    "sgrsv",
    "relative_structured_variability",
    "SpatialStat",
    "Relative structured variability (RSV)",
    "Keep your expectations low, boy. -- Kratos, God of War",
)
_r(
    "sganr",
    "anisotropy_ratio",
    "SpatialStat",
    "Anisotropy ratio and principal direction estimation",
    "Machines are not our friends. -- Rost, Horizon Zero Dawn",
)
_r(
    "sgans",
    "anisotropy_correction",
    "SpatialStat",
    "Geometric anisotropy correction",
    "This is my story. -- Tidus, Final Fantasy X",
)
_r(
    "sgiso",
    "isotropy_test",
    "SpatialStat",
    "Isotropy test via directional variogram comparison",
    "You'll never see it coming. -- Persona 5 OST",
)

# ── Schabenberger & Gotway 2005: Stationarity & Diagnostics ─────────────────
_r(
    "sgstr",
    "strict_stationarity_test",
    "SpatialStat",
    "Strict stationarity test via subregion comparison",
    "My name is Gyoubu Masataka Oniwa. -- Gyoubu, Sekiro",
)
_r(
    "sgwks",
    "weak_stationarity_test",
    "SpatialStat",
    "Weak stationarity test via variogram",
    "The Knight has no voice to cry suffering. -- Hollow Knight",
)
_r(
    "sgdcp",
    "data_decomposition",
    "SpatialStat",
    "Spatial data decomposition into trend + residuals",
    "Even the smallest person can change the course of the future. -- Galadriel, LOTR",
)
_r(
    "sgnsc",
    "normal_score_transform",
    "SpatialStat",
    "Normal score transform",
    "Happiness can be found even in the darkest of times. -- Dumbledore, Harry Potter",
)
_r(
    "sgzts",
    "asymptotic_z_test",
    "SpatialStat",
    "Asymptotic z-test for spatial statistics",
    "Damn, you're ugly. -- Geralt, The Witcher",
)
_r(
    "sgpmt",
    "permutation_test_spatial",
    "SpatialStat",
    "Spatial permutation test",
    "I am Malenia, Blade of Miquella. -- Malenia, Elden Ring",
)
_r(
    "sgmnt",
    "mantel_test",
    "SpatialStat",
    "Mantel test for spatial correlation",
    "Ashen one, link the flame. -- Fire Keeper, Dark Souls 3",
)
_r(
    "sgmci",
    "monte_carlo_spatial_test",
    "SpatialStat",
    "Monte Carlo spatial significance test",
    "A corpse should be left well alone. -- Lady Maria, Bloodborne",
)
_r(
    "sghrmt",
    "hermite_polynomial",
    "SpatialStat",
    "Hermite polynomial evaluation",
    "Distribution helper.",
)
_r(
    "sgcnv",
    "convolution_representation",
    "SpatialStat",
    "Convolution representation of a random field",
    "Stand amongst the ashes and ask the ghosts if honor matters. -- Javik, Mass Effect 3",
)

# ── Schabenberger & Gotway 2005: Kriging ────────────────────────────────────
_r(
    "sgok",
    "ordinary_kriging",
    "SpatialStat",
    "Ordinary kriging interpolation",
    "The head of Mimir is wise. -- Freya, God of War",
)
_r(
    "sgsk",
    "simple_kriging",
    "SpatialStat",
    "Simple kriging interpolation",
    "All life ends in death. Yet we build. -- Elisabet, Horizon Zero Dawn",
)
_r(
    "sguk",
    "universal_kriging",
    "SpatialStat",
    "Universal kriging interpolation",
    "Tell me what you cherish most. -- Sephiroth, Final Fantasy VII",
)
_r(
    "sgblk",
    "block_kriging",
    "SpatialStat",
    "Block kriging",
    "The arcana is the means by which all is revealed. -- Igor, Persona 3",
)
_r("sgind", "indicator_kriging", "SpatialStat", "Indicator kriging", "My father is Owl. -- Wolf, Sekiro")
_r(
    "sglnk",
    "lognormal_kriging",
    "SpatialStat",
    "Lognormal kriging with back-transformation",
    "Through dream I travel, at lantern's call. -- Seer, Hollow Knight",
)
_r(
    "sgdjk",
    "disjunctive_kriging",
    "SpatialStat",
    "Disjunctive kriging via Hermite polynomials",
    "Fly, you fools! -- Gandalf, LOTR",
)
_r(
    "sgcok",
    "cokriging",
    "SpatialStat",
    "Cokriging for multivariate spatial prediction",
    "We must all make the choice between what is right and what is easy. -- Dumbledore, Harry Potter",
)
_r(
    "sgdrft",
    "external_drift_kriging",
    "SpatialStat",
    "Kriging with external drift (KED)",
    "The witcher you shall be. -- Vesemir, The Witcher",
)
_r(
    "sgtrn",
    "trans_gaussian_kriging",
    "SpatialStat",
    "Trans-Gaussian kriging",
    "Someone must extinguish thy flame. -- Morgott, Elden Ring",
)
_r(
    "sglag",
    "lagrange_kriging_system",
    "SpatialStat",
    "Lagrange kriging system solver",
    "If I didn't know better, I'd think you had feelings. -- Siegmeyer, Dark Souls",
)
_r(
    "sgokv",
    "ordinary_kriging_variance",
    "SpatialStat",
    "Ordinary kriging variance",
    "We are born of the blood. -- The Doll, Bloodborne",
)
_r("sgskv", "simple_kriging_variance", "SpatialStat", "Simple kriging variance", "Finish the fight. -- Cortana, Halo 3")
_r(
    "sgukv",
    "universal_kriging_variance",
    "SpatialStat",
    "Universal kriging variance",
    "I'm Commander Shepard, and this is my favorite store on the Citadel. -- Shepard, Mass Effect 2",
)
_r(
    "sgkpe",
    "kriging_prediction_error",
    "SpatialStat",
    "Kriging prediction error statistics",
    "Boy. -- Kratos, God of War",
)
_r(
    "sgkse",
    "standardized_prediction_error",
    "SpatialStat",
    "Standardized kriging prediction errors",
    "Sometimes to protect, you must first destroy. -- Aloy, Horizon Forbidden West",
)
_r(
    "sgqqk",
    "qq_plot_kriging",
    "SpatialStat",
    "QQ plot data for kriging standardized errors",
    "Eyes on me. -- Squall, Final Fantasy VIII",
)
_r(
    "sgkvl",
    "kriging_cross_validation",
    "SpatialStat",
    "Kriging cross-validation",
    "Bonafide Monafide! -- Morgana, Persona 5",
)
_r(
    "sgmspe",
    "mspe_kriging",
    "SpatialStat",
    "Mean squared prediction error statistics",
    "One who hesitates is lost. -- Lady Butterfly, Sekiro",
)
_r(
    "sgccdf",
    "conditional_cdf_indicator",
    "SpatialStat",
    "Conditional CDF via indicator kriging",
    "In the shade of the Black Egg, the Hollow Knight waits. -- Hollow Knight",
)
_r(
    "sgblp",
    "blup_spatial",
    "SpatialStat",
    "Best linear unbiased predictor for spatial data",
    "It is a far, far better thing that I do. -- Aragorn, LOTR",
)
_r(
    "sglss",
    "squared_error_loss",
    "SpatialStat",
    "Squared error loss function",
    "The wand chooses the wizard. -- Ollivander, Harry Potter",
)
_r(
    "sgavgc",
    "average_covariance_block",
    "SpatialStat",
    "Average covariance over a block",
    "People linked by destiny will always find each other. -- Yennefer, The Witcher",
)

# ── Schabenberger & Gotway 2005: Spatial Regression ─────────────────────────
_r(
    "sgols",
    "ols_spatial_diagnostics",
    "SpatialStat",
    "OLS with spatial diagnostics on residuals",
    "Let your ambition lay to rest. -- Godrick, Elden Ring",
)
_r(
    "sggls",
    "gls_spatial",
    "SpatialStat",
    "Generalized least squares for spatial data",
    "Bearer of the Curse, seek misery. -- Emerald Herald, Dark Souls 2",
)
_r(
    "sgsar",
    "sar_lag_model",
    "SpatialStat",
    "Spatial autoregressive lag model (SAR)",
    "Majestic! A hunter is a hunter, even in a dream. -- Micolash, Bloodborne",
)
_r(
    "sgsem",
    "sem_error_model",
    "SpatialStat",
    "Spatial error model (SEM)",
    "Tell 'em to make it count. -- Emile, Halo Reach",
)
_r("sgdbn", "spatial_durbin_model", "SpatialStat", "Spatial Durbin model (SDM)", "Keelah se'lai. -- Tali, Mass Effect")
_r(
    "sgcar",
    "conditional_autoregressive",
    "SpatialStat",
    "Conditional autoregressive (CAR) model",
    "In the direction of deer. -- Atreus, God of War",
)
_r(
    "sgicar",
    "intrinsic_car_model",
    "SpatialStat",
    "Intrinsic conditional autoregressive (ICAR) model",
    "All will be well. -- GAIA, Horizon Zero Dawn",
)
_r(
    "sgglm",
    "spatial_glm_poisson",
    "SpatialStat",
    "Spatial GLM (Poisson) with residual diagnostics",
    "Aerith lives in our hearts. -- Cloud, Final Fantasy VII",
)
_r(
    "sglgt",
    "spatial_logistic",
    "SpatialStat",
    "Spatial logistic regression with diagnostics",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "sgpql",
    "pql_spatial_glmm",
    "SpatialStat",
    "Penalized quasi-likelihood spatial GLMM",
    "Face me, Sekiro! -- Guardian Ape, Sekiro",
)
_r(
    "sglml",
    "lm_test_spatial",
    "SpatialStat",
    "Lagrange multiplier tests for spatial dependence",
    "Higher beings, these words are for you alone. -- Hollow Knight",
)
_r(
    "sgrmr",
    "moran_residual_test",
    "SpatialStat",
    "Moran I test on regression residuals",
    "The beacons are lit! -- Gandalf, LOTR",
)
_r(
    "sgwts",
    "spatial_weights_matrix",
    "SpatialStat",
    "Spatial weights matrix construction",
    "Turn to page three hundred and ninety-four. -- Snape, Harry Potter",
)
_r(
    "sgrwn",
    "row_normalize_weights",
    "SpatialStat",
    "Row-normalize a spatial weights matrix",
    "Hmm. Wind's howling. -- Geralt, The Witcher",
)
_r(
    "sgbhr",
    "bayesian_hierarchical_spatial",
    "SpatialStat",
    "Bayesian hierarchical spatial model (Gibbs sampler)",
    "I have given thee courtesy enough. -- Rykard, Elden Ring",
)

# ── Schabenberger & Gotway 2005: Simulation ─────────────────────────────────
_r(
    "sgchol",
    "cholesky_grf_sim",
    "SpatialStat",
    "Cholesky-based Gaussian random field simulation",
    "Don't you dare go Hollow. -- Firekeeper, Dark Souls 3",
)
_r(
    "sglus",
    "lu_decomposition_sim",
    "SpatialStat",
    "LU decomposition for spatial simulation",
    "Fear the old blood. -- Laurence, Bloodborne",
)
_r("sgtbn", "turning_bands_sim", "SpatialStat", "Turning bands simulation", "Were it so easy. -- Arbiter, Halo 3")
_r(
    "sgsps",
    "spectral_grf_sim",
    "SpatialStat",
    "Spectral (FFT-based) Gaussian random field simulation",
    "Does this unit have a soul? -- Legion, Mass Effect 3",
)
_r(
    "sgsqs",
    "sequential_gaussian_sim",
    "SpatialStat",
    "Sequential Gaussian simulation",
    "The winds of change are upon us. -- Mimir, God of War",
)
_r(
    "sgcsm",
    "conditional_simulation",
    "SpatialStat",
    "Conditional simulation",
    "I was made for this purpose. -- Aloy, Horizon Zero Dawn",
)
_r(
    "sgkcs",
    "kriging_conditional_sim",
    "SpatialStat",
    "Kriging-based conditional simulation correction",
    "Know that I have loved you. -- Aerith, Final Fantasy VII",
)
_r(
    "sgsa",
    "simulated_annealing_spatial",
    "SpatialStat",
    "Simulated annealing for spatial pattern optimization",
    "A will of rebellion against a god of control. -- Lavenza, Persona 5",
)
_r(
    "sgtmp",
    "temperature_schedule",
    "SpatialStat",
    "Temperature schedule for simulated annealing",
    "Come, Sekiro! -- Isshin the Sword Saint, Sekiro",
)
_r(
    "sgblz",
    "boltzmann_accept",
    "SpatialStat",
    "Boltzmann acceptance criterion for simulated annealing",
    "No mind to think. No will to break. -- Hollow Knight",
)

# === BOOK COVERAGE GAP WRAPPERS (Armstrong + Schabenberger) ===
_r(
    "bbtrs",
    "blackbox_transpose",
    "SpatialVoting",
    "Blackbox_transpose scaling for large-N surveys",
    "Distribution helper.",
)
_r(
    "nmmds",
    "nonmetric_mds",
    "SpatialVoting",
    "Nonmetric multidimensional scaling",
    "Distribution helper.",
)
_r(
    "smcof",
    "smacof_scale",
    "SpatialVoting",
    "SMACOF majorization scaling",
    "Distribution helper.",
)
_r(
    "indsc",
    "indscal_mds",
    "SpatialVoting",
    "INDSCAL individual differences MDS",
    "Distribution helper.",
)
_r(
    "dwnmt",
    "dw_nominate_score",
    "SpatialVoting",
    "DW-NOMINATE ideal point estimation",
    "Distribution helper.",
)
_r(
    "alnmt",
    "alpha_nominate_score",
    "SpatialVoting",
    "alpha-NOMINATE MCMC estimation",
    "Distribution helper.",
)
_r(
    "dyirt",
    "dynamic_irt_estimate",
    "SpatialVoting",
    "Dynamic IRT model estimation",
    "Distribution helper.",
)
_r(
    "emirt",
    "em_irt_estimate",
    "SpatialVoting",
    "EM algorithm IRT estimation",
    "Distribution helper.",
)
_r(
    "odirt",
    "ordinal_irt_estimate",
    "SpatialVoting",
    "Ordinal IRT model estimation",
    "Distribution helper.",
)

# -- SpatialVoting Ch1-10: Voting Theory, Models, Diagnostics (28) -- Armstrong (2014)
_r("mmds", "metric_mds", "SpatialVoting", "Metric MDS for ideal point estimation", "The spice must flow. -- Duke Leto, Dune")
_r("smcub", "smacof_unfolding_basic", "SpatialVoting", "SMACOF rectangular unfolding", "The sleeper must awaken. -- Stilgar, Dune")
_r("unfld", "unfolding_model", "SpatialVoting", "Metric unfolding ideal point estimation", "I am no man. -- Eowyn, Lord of the Rings")
_r("mdnvt", "median_voter", "SpatialVoting", "Median voter theorem computation", "All we have to decide is what to do with the time given us. -- Gandalf")
_r("borda", "borda_count", "SpatialVoting", "Borda count election method", "Distribution helper.")
_r("plurl", "plurality_vote", "SpatialVoting", "Plurality (first-past-the-post) voting", "What is now proved was once only imagined. — William Blake")
_r("appvl", "approval_vote", "SpatialVoting", "Approval voting method", "It's a trap! -- Admiral Ackbar, Star Wars")
_r("irv", "instant_runoff", "SpatialVoting", "Instant runoff (ranked choice) voting", "In my experience there is no such thing as luck. --")
_r("copld", "copeland_method", "SpatialVoting", "Copeland pairwise comparison voting", "What is now proved was once only imagined. — William Blake")
_r("pairm", "pairwise_matrix", "SpatialVoting", "Pairwise comparison matrix from rankings", "What is now proved was once only imagined. — William Blake")
_r("euclm", "euclidean_model", "SpatialVoting", "Euclidean spatial voting model", "Distribution helper.")
_r("citym", "cityblock_model", "SpatialVoting", "City-block (Manhattan) spatial model", "These aren't the droids you're looking for. --")
_r("dirml", "directional_model", "SpatialVoting", "Directional spatial voting model", "What is now proved was once only imagined. — William Blake")
_r("wghtm", "weighted_euclidean_model", "SpatialVoting", "Weighted Euclidean spatial model", "Distribution helper.")
_r("qvote", "quadratic_voting", "SpatialVoting", "Quadratic voting model", "Rebellions are built on hope. -- Jyn Erso, Star Wars")
_r("spvut", "spatial_utility", "SpatialVoting", "General spatial voting utility", "What is now proved was once only imagined. — William Blake")
_r("cutto", "cutting_plane", "SpatialVoting", "Cutting plane / separating hyperplane", "Luminous beings are we., Star Wars")
_r("clfrt", "classification_rate", "SpatialVoting", "Classification rate for spatial model", "Wars not make one great., Star Wars")
_r("ocslt", "oc_scaling", "SpatialVoting", "Optimal classification (OC) scaling", "Judge me by my size, do you?, Star Wars")
_r("nmnlt", "nominate_scaling", "SpatialVoting", "NOMINATE scaling for ideal points", "The ability to speak does not make you intelligent. -- Qui-Gon")
_r("blkbt", "blackbox_scaling_basic", "SpatialVoting", "Blackbox scaling for ideal points", "What is now proved was once only imagined. — William Blake")
_r("alcov", "alcove_model", "SpatialVoting", "ALCOVE attention-learning model", "Your focus determines your reality. -- Qui-Gon, Star Wars")
_r("bayid", "bayesian_ideal_points", "SpatialVoting", "Bayesian ideal point estimation (MCMC)", "Truly wonderful the mind of a child is.")
_r("btlmd", "bradley_terry", "SpatialVoting", "Bradley-Terry-Luce paired comparison model", "Distribution helper.")
_r("rollc", "roll_call_analysis", "SpatialVoting", "Roll call analysis for legislative voting", "So this is how liberty dies. -- Padme, Star Wars")
_r("agrmt", "agreement_score", "SpatialVoting", "Pairwise agreement score computation", "There is always a bigger fish. -- Qui-Gon, Star Wars")
_r("gmpre", "geometric_mean_probability", "SpatialVoting", "GMP fit diagnostic for spatial models", "The greatest teacher, failure is., Star Wars")
_r("prech", "proportional_reduction_error", "SpatialVoting", "PRE for spatial voting models", "Once you start down the dark path., Star Wars")

_r(
    "sgcrh",
    "cressie_hawkins",
    "Geostat",
    "Cressie-Hawkins robust semivariogram estimator",
    "Distribution helper.",
)
_r(
    "sgspd",
    "space_deformation",
    "Geostat",
    "Space deformation non-stationary covariance",
    "Distribution helper.",
)
_r(
    "sgmwc",
    "moving_window_cov",
    "Geostat",
    "Moving window covariance estimation",
    "Distribution helper.",
)
_r(
    "stdea",
    "st_diff_equation",
    "SpatioTemporal",
    "Differential equation spatio-temporal covariance",
    "Distribution helper.",
)

# === AUTO-GENERATED SPATIAL ENTRIES ===
_r("abbac", "abbac", "AirBio", "Bacterial aerosol spatial", quote="El Psy Kongroo. -- Okabe")
_r("abblc", "abblc", "AirBio", "Black carbon spatial", quote="It's over 9000! -- Vegeta")
_r("abdst", "abdst", "AirBio", "Dust spatial mapping", quote="Equivalent exchange. -- Elric brothers")
_r("abdxn", "abdxn", "AirBio", "Dioxin spatial mapping", quote="Dedicate your hearts! -- Erwin")
_r("abemf", "abemf", "AirBio", "EMF spatial mapping", quote="Go beyond! Plus Ultra! -- All Might")
_r("abend", "abend", "AirBio", "Endotoxin spatial", quote="Arise. -- Shadow Monarch")
_r("abfng", "abfng", "AirBio", "Fungal spore spatial", quote="See you space cowboy. -- Spike")
_r("abhch", "abhch", "AirBio", "Formaldehyde spatial", quote="I am the one who knocks. -- Walter White")
_r("abirr", "abirr", "AirBio", "IR radiation spatial", quote="Set your heart ablaze! -- Rengoku")
_r("ablad", "ablad", "AirBio", "Lead dust spatial", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("ablgt", "ablgt", "AirBio", "Light pollution spatial", quote="Arise. -- Shadow Monarch")
_r("abmcr", "abmcr", "AirBio", "Microplastic spatial air", quote="Get in the robot, Shinji! -- Misato")
_r("abmld", "abmld", "AirBio", "Mold spatial mapping", quote="Not all those who wander are lost. -- Gandalf")
_r("abmsm", "abmsm", "SpatialPat", "Agent-based model spatial", quote="Growing old is a blessing. -- Rengoku")
_r("abodr", "abodr", "AirBio", "Odor spatial mapping", quote="I am the one who knocks. -- Walter White")
_r("abpah", "abpah", "AirBio", "PAH spatial mapping", quote="Yare yare daze. -- Jotaro")
_r("abpcb", "abpcb", "AirBio", "PCB spatial mapping", quote="I am justice! -- Light")
_r("abpes", "abpes", "AirBio", "Pesticide spatial air", quote="Equivalent exchange. -- Elric brothers")
_r("abpfc", "abpfc", "AirBio", "PFAS spatial mapping", quote="Bankai! -- Ichigo")
_r("abpln", "abpln", "AirBio", "Pollen spatial mapping", quote="Bankai! -- Ichigo")
_r("abpmc", "abpmc", "AirBio", "PM coarse spatial indoor", quote="Believe it! -- Naruto")
_r("abpmf", "abpmf", "AirBio", "PM fine spatial indoor", quote="Make it so. -- Picard")
_r("abradn", "abradn", "AirBio", "Radon indoor spatial", quote="Get in the robot, Shinji! -- Misato")
_r("abrf", "abrf", "AirBio", "RF radiation spatial", quote="See you space cowboy. -- Spike")
_r("absbs", "absbs", "AirBio", "Asbestos spatial", quote="The spice must flow. -- Paul Atreides")
_r("abspor", "abspor", "AirBio", "Spore spatial mapping", quote="Go beyond! Plus Ultra! -- All Might")
_r("abter", "abter", "AirBio", "Thermal pollution spatial", quote="Winter is coming. -- Stark motto")
_r("abufc", "abufc", "AirBio", "UFP count spatial", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("abuvr", "abuvr", "AirBio", "UV radiation spatial", quote="El Psy Kongroo. -- Okabe")
_r("abvir", "abvir", "AirBio", "Viral aerosol spatial", quote="Set your heart ablaze! -- Rengoku")
_r("abvoc", "abvoc", "AirBio", "VOC indoor spatial", quote="Winter is coming. -- Stark motto")
_r("adaptg", "adaptg", "GeoProcss", "Adaptive resolution grid", quote="People's dreams never end! -- Blackbeard")
_r(
    "adpsmp",
    "adpsmp",
    "GeoProcss",
    "Adaptive spatial sampling",
    quote="Even the smallest person can change the future. -- Galadriel",
)
_r("afaqua", "afaqua", "AgriSp", "Aquaculture site spatial", quote="Kamehameha! -- Goku")
_r("afbio", "afbio", "AgriSp", "Biomass estimation spatial", quote="Resistance is futile. -- Borg")
_r("afbufr", "afbufr", "AgriSp", "Buffer strip design", quote="Tatakae! -- Eren")
_r("afchl", "afchl", "AgriSp", "Chill hours spatial", quote="Breathe. -- Tanjiro")
_r("afcntr", "afcntr", "AgriSp", "Contour farming spatial", quote="Keep moving forward. -- Eren")
_r("afcrb", "afcrb", "AgriSp", "Carbon sequestration soil", quote="A lesson without pain is meaningless. -- Edward")
_r("afdis", "afdis", "AgriSp", "Disease risk crop spatial", quote="Get in the robot, Shinji! -- Misato")
_r("afdrns", "afdrns", "AgriSp", "Drainage design spatial", quote="Total Concentration Breathing. -- Tanjiro")
_r("afdrt", "afdrt", "AgriSp", "Drought stress crop spatial", quote="Winter is coming. -- Stark motto")
_r("afevi", "afevi", "AgriSp", "EVI crop monitoring", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("affrst", "affrst", "AgriSp", "Frost risk spatial", quote="I alone level up. -- Sung Jin-Woo")
_r("affrt", "affrt", "AgriSp", "Fertilizer optimization spatial", quote="Desert power. -- Paul Muad'Dib")
_r("afgdd", "afgdd", "AgriSp", "Growing degree days spatial", quote="Science! -- Jesse Pinkman")
_r("afgraz", "afgraz", "AgriSp", "Grazing capacity spatial", quote="Hold the door. -- Hodor")
_r("afgrhs", "afgrhs", "AgriSp", "Greenhouse spatial", quote="No half measures. -- Mike")
_r("afheat", "afheat", "AgriSp", "Heat stress risk spatial", quote="There is always hope. -- Aragorn")
_r("afhrv", "afhrv", "AgriSp", "Harvest date spatial", quote="The world is cruel but beautiful. -- Mikasa")
_r("afirr", "afirr", "AgriSp", "Irrigation efficiency spatial", quote="Arise. -- Shadow Monarch")
_r("aflai", "aflai", "AgriSp", "Leaf area index crop", quote="It's over 9000! -- Vegeta")
_r("aflch", "aflch", "AgriSp", "Leaching risk spatial", quote="I am here! -- All Might")
_r("aflstk", "aflstk", "AgriSp", "Livestock density spatial", quote="My precious. -- Gollum")
_r("afmgz", "afmgz", "AgriSp", "Management zone delineation", quote="Growing old is a blessing. -- Rengoku")
_r("afndv", "afndv", "AgriSp", "NDVI crop monitoring", quote="Believe it! -- Naruto")
_r("afplnt", "afplnt", "AgriSp", "Planting date spatial", quote="Chaos is a ladder. -- Littlefinger")
_r("afprsc", "afprsc", "AgriSp", "Precision agriculture zone", quote="Walk without rhythm. -- Fremen proverb")
_r("afpst", "afpst", "AgriSp", "Pest risk spatial", quote="I am the one who knocks. -- Walter White")
_r("afpstr", "afpstr", "AgriSp", "Pasture quality spatial", quote="Not all those who wander are lost. -- Gandalf")
_r("afrfl", "afrfl", "AgriSp", "Rainfall adequacy crop", quote="Power is everything. -- Sung Jin-Woo")
_r("afrnf", "afrnf", "AgriSp", "Runoff risk spatial", quote="You should enjoy the detours. -- Ging")
_r("afrng", "afrng", "AgriSp", "Rangeland condition", quote="The spice must flow. -- Paul Atreides")
_r("afrotf", "afrotf", "AgriSp", "Crop rotation fitness", quote="One does not simply walk. -- Boromir")
_r("afrpr", "afrpr", "AgriSp", "Riparian zone design", quote="The sleeper must awaken. -- Leto Atreides")
_r("afsavh", "afsavh", "AgriSp", "Savanna health spatial", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("afsed", "afsed", "AgriSp", "Sedimentation risk spatial", quote="Whatever happens, happens. -- Spike")
_r("afslc", "afslc", "AgriSp", "Soil compaction spatial", quote="El Psy Kongroo. -- Okabe")
_r("afsle", "afsle", "AgriSp", "Soil erosion spatial", quote="Set your heart ablaze! -- Rengoku")
_r("afslk", "afslk", "AgriSp", "Soil potassium spatial", quote="Go beyond! Plus Ultra! -- All Might")
_r("afslm", "afslm", "AgriSp", "Soil moisture spatial", quote="Yare yare daze. -- Jotaro")
_r("afsln", "afsln", "AgriSp", "Soil nitrogen spatial", quote="Bankai! -- Ichigo")
_r("afslp", "afslp", "AgriSp", "Soil pH spatial", quote="Dedicate your hearts! -- Erwin")
_r("afslp2", "afslp2", "AgriSp", "Soil phosphorus spatial", quote="Equivalent exchange. -- Elric brothers")
_r("afslt", "afslt", "AgriSp", "Soil texture spatial", quote="See you space cowboy. -- Spike")
_r("afsom", "afsom", "AgriSp", "Soil organic matter spatial", quote="I am justice! -- Light")
_r("aftrr", "aftrr", "AgriSp", "Terrace design spatial", quote="I mustn't run away. -- Shinji")
_r("afvwc", "afvwc", "AgriSp", "Variable rate application", quote="A Lannister always pays his debts. -- Tyrion")
_r("afwdsp", "afwdsp", "AgriSp", "Wind damage risk spatial", quote="Engage. -- Picard")
_r("afwed", "afwed", "AgriSp", "Weed density spatial", quote="Live long and prosper. -- Spock")
_r("afwndb", "afwndb", "AgriSp", "Windbreak design spatial", quote="This is Requiem. -- Giorno")
_r("afwtld", "afwtld", "AgriSp", "Constructed wetland design", quote="Valar Morghulis. -- Braavos")
_r("afyld", "afyld", "AgriSp", "Crop yield spatial", quote="Make it so. -- Picard")
_r("agagg", "agagg", "AreaGeo", "Aggregation index", quote="Go beyond! Plus Ultra! -- All Might")
_r("agalb", "agalb", "AreaGeo", "Albedo spatial analysis", quote="I am here! -- All Might")
_r("agame", "agame", "Spatial", "Amendment agenda procedure.", quote="")
_r("agawd", "agawd", "AreaGeo", "Area-weighted mean shape", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("agbar", "agbar", "AreaGeo", "Bare soil index spatial", quote="A Lannister always pays his debts. -- Tyrion")
_r("agbck", "agbck", "Spatial", "Backward induction agenda.", quote="")
_r("agbll", "agbll", "Spatial", "Binary agenda tree.", quote="")
_r("agbui", "agbui", "AreaGeo", "Built-up index spatial", quote="Power is everything. -- Sung Jin-Woo")
_r("agclp", "agclp", "AreaGeo", "Clumpiness index", quote="Equivalent exchange. -- Elric brothers")
_r("agcmp", "agcmp", "AreaGeo", "Compactness index", quote="There is always hope. -- Aragorn")
_r("agcnt", "agcnt", "AreaGeo", "Connectivity index landscape", quote="Winter is coming. -- Stark motto")
_r("agcoh", "agcoh", "AreaGeo", "Patch cohesion", quote="Make it so. -- Picard")
_r("agcon", "agcon", "AreaGeo", "Contagion landscape", quote="I am justice! -- Light")
_r("agcrs", "agcrs", "AreaGeo", "Landscape corridor analysis", quote="I am the one who knocks. -- Walter White")
_r("agcyc", "agcyc", "Spatial", "Agenda cycling detection.", quote="")
_r("agdis", "agdis", "AreaGeo", "Dissection index", quote="I must not fear. -- Litany Against Fear")
_r("agdom", "agdom", "AreaGeo", "Landscape dominance", quote="Yare yare daze. -- Jotaro")
_r("agdvm", "agdvm", "AreaGeo", "Landscape diversity (Simpson)", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("agdvs", "agdvs", "AreaGeo", "Landscape diversity (Shannon)", quote="Believe it! -- Naruto")
_r("aged", "aged", "AreaGeo", "Edge density", quote="One is all, all is one. -- Izumi")
_r("agenn", "agenn", "AreaGeo", "Euclidean NN distance patch", quote="See you space cowboy. -- Spike")
_r("ageve", "ageve", "AreaGeo", "Landscape evenness", quote="It's over 9000! -- Vegeta")
_r("agevi", "agevi", "AreaGeo", "EVI spatial analysis", quote="Science! -- Jesse Pinkman")
_r("agfdi", "agfdi", "AreaGeo", "Fractal dimension shape", quote="Engage. -- Picard")
_r("agfgm", "agfgm", "AreaGeo", "Fragmentation index", quote="One does not simply walk. -- Boromir")
_r("agfpr", "agfpr", "AreaGeo", "Forest proportion index", quote="Whatever happens, happens. -- Spike")
_r("agfrc", "agfrc", "AreaGeo", "Fractal dimension mean", quote="Not all those who wander are lost. -- Gandalf")
_r("agfwd", "agfwd", "Spatial", "Forward induction agenda.", quote="")
_r("aggrn", "aggrn", "AreaGeo", "Green space index", quote="The world is cruel but beautiful. -- Mikasa")
_r("aggta", "aggta", "Spatial", "Agenda game tree analysis.", quote="")
_r("agijm", "agijm", "AreaGeo", "Interspersion juxtaposition", quote="Bankai! -- Ichigo")
_r("agimv", "agimv", "AreaGeo", "Imperviousness index", quote="A lesson without pain is meaningless. -- Edward")
_r("agiso", "agiso", "AreaGeo", "Isolation index patch", quote="Set your heart ablaze! -- Rengoku")
_r("aglai", "aglai", "AreaGeo", "Leaf area index spatial", quote="You should enjoy the detours. -- Ging")
_r("aglpi", "aglpi", "AreaGeo", "Largest patch index", quote="Keep moving forward. -- Eren")
_r("aglsi", "aglsi", "AreaGeo", "Landscape shape index", quote="I am the hope of the universe. -- Goku")
_r("aglsm", "aglsm", "AreaGeo", "Landscape mean shape", quote="The spice must flow. -- Paul Atreides")
_r(
    "aglst",
    "aglst",
    "AreaGeo",
    "Land surface temperature",
    quote="Even the smallest person can change the future. -- Galadriel",
)
_r("agmaj", "agmaj", "Spatial", "Majority agenda rule.", quote="")
_r("agmsv", "agmsv", "AreaGeo", "MSAVI spatial analysis", quote="I alone level up. -- Sung Jin-Woo")
_r("agndv", "agndv", "AreaGeo", "NDVI spatial analysis", quote="Chaos is a ladder. -- Littlefinger")
_r("agnp", "agnp", "AreaGeo", "Number of patches", quote="I will take a potato chip and eat it! -- Light")
_r("agopn", "agopn", "Spatial", "Open agenda model.", quote="")
_r("agpat", "agpat", "AreaGeo", "Patch analysis metrics", quote="People's dreams never end! -- Blackbeard")
_r("agpd", "agpd", "AreaGeo", "Patch density", quote="Scatter, Senbonzakura. -- Byakuya")
_r("agplt", "agplt", "Spatial", "Platform agenda comparison.", quote="")
_r("agpow", "agpow", "Spatial", "Agenda power index.", quote="")
_r("agprf", "agprf", "AreaGeo", "Perforation index", quote="Resistance is futile. -- Borg")
_r("agprm", "agprm", "AreaGeo", "Perimeter-area ratio", quote="Those who break the rules are scum. -- Kakashi")
_r("agprx", "agprx", "AreaGeo", "Proximity index patch", quote="El Psy Kongroo. -- Okabe")
_r("agrev", "agrev", "Spatial", "Reversion point agenda.", quote="")
_r("agric", "agric", "AreaGeo", "Landscape richness", quote="Dedicate your hearts! -- Erwin")
_r("agrst", "agrst", "Spatial", "Restricted agenda model.", quote="")
_r("agsav", "agsav", "AreaGeo", "SAVI spatial analysis", quote="Breathe. -- Tanjiro")
_r("agseq", "agseq", "Spatial", "Sequential agenda model.", quote="")
_r("agset", "agset", "Spatial", "Agenda setter model.", quote="")
_r("agshp", "agshp", "AreaGeo", "Shape index computation", quote="Walk without rhythm. -- Fremen proverb")
_r("agsim", "agsim", "AreaGeo", "Similarity index patch", quote="Arise. -- Shadow Monarch")
_r("agslv", "agslv", "AreaGeo", "Silviculture landscape metric", quote="Get in the robot, Shinji! -- Misato")
_r("agsnw", "agsnw", "AreaGeo", "Snow cover index spatial", quote="Growing old is a blessing. -- Rengoku")
_r("agspr", "agspr", "AreaGeo", "Sprawl index spatial", quote="Desert power. -- Paul Muad'Dib")
_r("aguhi", "aguhi", "AreaGeo", "Urban heat island index", quote="The needs of the many outweigh the few. -- Spock")
_r("agurb", "agurb", "AreaGeo", "Urban sprawl index", quote="Live long and prosper. -- Spock")
_r("agwin", "agwin", "Spatial", "Agenda winner prediction.", quote="")
_r("agwtr", "agwtr", "AreaGeo", "Water body index spatial", quote="I mustn't run away. -- Shinji")
_r("albers", "albers", "GeoProcss", "Albers equal area projection", quote="El Psy Kongroo. -- Okabe")
_r(
    "alphsh",
    "alphsh",
    "GeoAnalysis",
    "Alpha shape boundary detection",
    quote="I'm gonna be King of the Pirates! -- Luffy",
)
_r("annsi", "annsi", "SpatialPat", "Annealing simulation", quote="Live long and prosper. -- Spock")
_r("azmeqa", "azmeqa", "GeoProcss", "Azimuthal equidistant projection", quote="Arise. -- Shadow Monarch")
_r("balsmp", "balsmp", "GeoProcss", "Balanced spatial sampling", quote="Power is everything. -- Sung Jin-Woo")
_r("bgsim", "bgsim", "SpatialPat", "Block Gaussian simulation", quote="See you space cowboy. -- Spike")
_r("bkflt", "bkflt", "KrigFilt", "Block kriging", quote="Make it so. -- Picard")
_r(
    "blksmp",
    "blksmp",
    "GeoProcss",
    "Block spatial cross-validation",
    quote="A lesson without pain is meaningless. -- Edward",
)
_r(
    "bmsim",
    "bmsim",
    "SpatialPat",
    "Brownian motion spatial",
    quote="Even the smallest person can change the future. -- Galadriel",
)
_r("bnsmn", "bnsmn", "KrigFilt", "Binomial smoothing spatial", quote="Resistance is futile. -- Borg")
_r("bonne", "bonne", "GeoProcss", "Bonne projection", quote="I must not fear. -- Litany Against Fear")
_r("bssim", "bssim", "SpatialPat", "Block sequential simulation", quote="El Psy Kongroo. -- Okabe")
_r("buffr", "buffr", "GeoAnalysis", "Buffer zone analysis", quote="Yare yare daze. -- Jotaro")
_r("canny", "canny", "KrigFilt", "Canny edge detection spatial", quote="I mustn't run away. -- Shinji")
_r("caraic", "caraic", "CAR", "CAR Akaike information criterion.", quote="")
_r("carbic", "carbic", "CAR", "CAR Bayesian information criterion.", quote="")
_r("carbym", "carbym", "CAR", "CAR Besag-York-Mollié (BYM) variance components.", quote="")
_r("carconv", "carconv", "CAR", "CAR rho feasibility bounds.", quote="")
_r("carfit", "carfit", "CAR", "CAR DIC (deviance information criterion).", quote="")
_r("cargmm", "cargmm", "CAR", "CAR GMM estimation.", quote="")
_r("caricar", "caricar", "CAR", "Intrinsic CAR (ICAR) log-density.", quote="")
_r("carjac", "carjac", "CAR", "CAR Jacobian term.", quote="")
_r("carlrt", "carlrt", "CAR", "CAR likelihood-ratio test.", quote="")
_r("carml", "carml", "CAR", "CAR (simultaneous) ML estimation.", quote="")
_r("carres", "carres", "CAR", "CAR residual Moran test.", quote="")
_r("carsig", "carsig", "CAR", "CAR sigma-squared estimate.", quote="")
_r("carsim", "carsim", "CAR", "CAR conditional simulation.", quote="")
_r("carvar", "carvar", "CAR", "CAR variance-covariance matrix.", quote="")
_r("carvr", "carvr", "CAR", "CAR variance ratio (spatial vs unstructured).", quote="")
_r("casim", "casim", "SpatialPat", "Cellular automaton spatial", quote="I mustn't run away. -- Shinji")
_r("cassni", "cassni", "GeoProcss", "Cassini projection", quote="Desert power. -- Paul Muad'Dib")
_r("cdbias", "cdbias", "Spatial", "Conditional simulation bias correction.", quote="")
_r("cdblk", "cdblk", "Spatial", "Conditional block simulation averaging.", quote="")
_r("cdbts", "cdbts", "Spatial", "Conditional bootstrap simulation ensemble.", quote="")
_r("cdcol", "cdcol", "Spatial", "Co-located co-kriging simulation.", quote="")
_r("cdcosm", "cdcosm", "Spatial", "Co-simulation of two correlated fields.", quote="")
_r("cdkrg", "cdkrg", "Spatial", "Conditional kriging for simulation conditioning.", quote="")
_r("cdlmc", "cdlmc", "Spatial", "Linear model of co-regionalization simulation.", quote="")
_r("cdmar", "cdmar", "Spatial", "Markov-Bayes co-simulation.", quote="")
_r("cdmlt", "cdmlt", "Spatial", "Multi-variate conditional simulation (p>2).", quote="")
_r("cdpost", "cdpost", "Spatial", "Conditional simulation posterior ensemble.", quote="")
_r("cdqnt", "cdqnt", "Spatial", "Conditional simulation quantile uncertainty.", quote="")
_r("cdreal", "cdreal", "Spatial", "Conditional realization stochastic path.", quote="")
_r("cdsim", "cdsim", "Spatial", "Conditional simulation at data locations.", quote="")
_r("cdsim2", "cdsim2", "Spatial", "Conditional simulation residual kriging approach.", quote="")
_r("cdval", "cdval", "Spatial", "Conditional simulation validation E-type.", quote="")
_r("cfbgn", "cfbgn", "Spatial", "Bargaining coalition spatial.", quote="")
_r("cfbnz", "cfbnz", "Spatial", "Banzhaf coalition index.", quote="")
_r("cfdef", "cfdef", "Spatial", "Defection test spatial coalition.", quote="")
_r("cfgrd", "cfgrd", "Spatial", "Grand coalition spatial.", quote="")
_r("cfhrt", "cfhrt", "Spatial", "Hart-Kurz coalition formation.", quote="")
_r("cfico", "cfico", "Spatial", "Ideological coalition distance.", quote="")
_r("cfmdc", "cfmdc", "Spatial", "Median coalition spatial.", quote="")
_r("cfmin", "cfmin", "Spatial", "Minimum winning coalition.", quote="")
_r("cfmwc", "cfmwc", "Spatial", "Minimum connected winning coalition.", quote="")
_r("cfprt", "cfprt", "Spatial", "Pareto-optimal coalition.", quote="")
_r("cfprx", "cfprx", "Spatial", "Proximity-based coalition.", quote="")
_r("cfpwr", "cfpwr", "Spatial", "Coalition power distribution.", quote="")
_r("cfshp", "cfshp", "Spatial", "Shapley coalition value.", quote="")
_r("cfspa", "cfspa", "Spatial", "Spatial coalition formation.", quote="")
_r("cfstb", "cfstb", "Spatial", "Coalition stability test.", quote="")
_r("chlann", "chlann", "Spatial", "Cholesky annealing-based simulation.", quote="")
_r("chlblk", "chlblk", "Spatial", "Cholesky block simulation for large grids.", quote="")
_r("chlbnd", "chlbnd", "Spatial", "Cholesky simulation with boundary constraints.", quote="")
_r("chlcnd", "chlcnd", "Spatial", "Cholesky conditional simulation grid.", quote="")
_r("chlexp", "chlexp", "Spatial", "Cholesky sim with exponential covariance.", quote="")
_r("chlgss", "chlgss", "Spatial", "Cholesky sim with Gaussian covariance kernel.", quote="")
_r("chlinv", "chlinv", "Spatial", "Cholesky inverse covariance simulation.", quote="")
_r("chlkrg", "chlkrg", "Spatial", "Cholesky-kriging hybrid simulation.", quote="")
_r("chlmat", "chlmat", "Spatial", "Cholesky sim with Matern covariance.", quote="")
_r("chlmlt", "chlmlt", "Spatial", "Cholesky multi-variable co-simulation.", quote="")
_r("chlpiv", "chlpiv", "Spatial", "Pivoted Cholesky incomplete decomposition sim.", quote="")
_r("chlpow", "chlpow", "Spatial", "Cholesky sim with power covariance model.", quote="")
_r("chlrng", "chlrng", "Spatial", "Cholesky sim range sensitivity analysis.", quote="")
_r("chlsmp", "chlsmp", "Spatial", "Cholesky sample path generation.", quote="")
_r("chlsph", "chlsph", "Spatial", "Cholesky sim with spherical covariance.", quote="")
_r("chltps", "chltps", "Spatial", "Cholesky thin-plate spline simulation.", quote="")
_r("chlvar", "chlvar", "Spatial", "Cholesky sim variance decomposition.", quote="")
_r("chsim", "chsim", "Spatial", "Cholesky simulation of Gaussian random field.", quote="")
_r("chsim2", "chsim2", "Spatial", "Cholesky simulation with nugget effect.", quote="")
_r("chsim3", "chsim3", "Spatial", "Cholesky simulation conditional on data.", quote="")
_r("ckflm", "ckflm", "KrigFilt", "Collocated co-kriging", quote="Not all those who wander are lost. -- Gandalf")
_r("cksim", "cksim", "SpatialPat", "Co-kriging simulation", quote="Go beyond! Plus Ultra! -- All Might")
_r("clafp", "clafp", "ClstSp", "Affinity propagation spatial", quote="One is all, all is one. -- Izumi")
_r("clagg", "clagg", "ClstSp", "Aggregation spatial method", quote="Arise. -- Shadow Monarch")
_r("clagm", "clagm", "ClstSp", "Agglomerative spatial cluster", quote="Engage. -- Picard")
_r("clazp", "clazp", "ClstSp", "AZP spatial cluster", quote="Whatever happens, happens. -- Spike")
_r("clbrc", "clbrc", "ClstSp", "BIRCH spatial clustering", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("clcal", "clcal", "ClstSp", "Calinski-Harabasz spatial", quote="I must not fear. -- Litany Against Fear")
_r("clcha", "clcha", "ClstSp", "CHAMELEON spatial", quote="Believe it! -- Naruto")
_r("clclr", "clclr", "ClstSp", "CLARANS spatial", quote="Make it so. -- Picard")
_r("clcmp", "clcmp", "ClstSp", "Compound spatial clustering", quote="Winter is coming. -- Stark motto")
_r("clcon", "clcon", "ClstSp", "Consensus clustering spatial", quote="Get in the robot, Shinji! -- Misato")
_r("clcph", "clcph", "ClstSp", "Complete linkage spatial", quote="Power is everything. -- Sung Jin-Woo")
_r("clcur", "clcur", "ClstSp", "CURE spatial clustering", quote="Not all those who wander are lost. -- Gandalf")
_r("cldbi", "cldbi", "ClstSp", "Davies-Bouldin spatial", quote="Desert power. -- Paul Muad'Dib")
_r("cldbx", "cldbx", "ClstSp", "DBSCAN extended spatial", quote="I am the hope of the universe. -- Goku")
_r("clden", "clden", "ClstSp", "DENCLUE spatial", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("cldiv", "cldiv", "ClstSp", "Divisive spatial cluster", quote="Those who break the rules are scum. -- Kakashi")
_r("cldun", "cldun", "ClstSp", "Dunn index spatial", quote="The world is cruel but beautiful. -- Mikasa")
_r("clelb", "clelb", "ClstSp", "Elbow method spatial", quote="Science! -- Jesse Pinkman")
_r("clemb", "clemb", "ClstSp", "Embedding-based spatial cluster", quote="See you space cowboy. -- Spike")
_r("clens", "clens", "ClstSp", "Ensemble spatial clustering", quote="I am the one who knocks. -- Walter White")
_r("clfc", "clfc", "ClstSp", "Fuzzy C-means spatial", quote="I alone level up. -- Sung Jin-Woo")
_r("clfcm", "clfcm", "ClstSp", "Fuzzy CM-means spatial", quote="The needs of the many outweigh the few. -- Spock")
_r("clflm", "clflm", "ClstSp", "FLAME spatial clustering", quote="Yare yare daze. -- Jotaro")
_r("clfst", "clfst", "ClstSp", "Fast greedy spatial community", quote="Go beyond! Plus Ultra! -- All Might")
_r("clgap", "clgap", "ClstSp", "Gap statistic spatial", quote="Chaos is a ladder. -- Littlefinger")
_r("clgmx", "clgmx", "ClstSp", "Gaussian mixture spatial", quote="People's dreams never end! -- Blackbeard")
_r("clgng", "clgng", "ClstSp", "Growing neural gas spatial", quote="Set your heart ablaze! -- Rengoku")
_r("clhdb", "clhdb", "ClstSp", "HDBSCAN spatial clustering", quote="I will take a potato chip and eat it! -- Light")
_r("clipan", "clipan", "GeoAnalysis", "Spatial clip operation", quote="Equivalent exchange. -- Elric brothers")
_r("clkmd", "clkmd", "ClstSp", "K-medoids spatial", quote="There is always hope. -- Aragorn")
_r("clkmn", "clkmn", "ClstSp", "K-means spatial clustering", quote="Walk without rhythm. -- Fremen proverb")
_r("cllbl", "cllbl", "ClstSp", "Label propagation spatial", quote="Equivalent exchange. -- Elric brothers")
_r("clldn", "clldn", "ClstSp", "Leiden spatial community", quote="I am justice! -- Light")
_r("cllvn", "cllvn", "ClstSp", "Louvain spatial community", quote="Dedicate your hearts! -- Erwin")
_r("clmnb", "clmnb", "ClstSp", "Mean shift spatial", quote="The spice must flow. -- Paul Atreides")
_r("clmxp", "clmxp", "ClstSp", "Max-p spatial cluster", quote="You should enjoy the detours. -- Ging")
_r("clopt", "clopt", "ClstSp", "OPTICS spatial clustering", quote="Keep moving forward. -- Eren")
_r("closf", "closf", "KrigFilt", "Closing morphological", quote="Walk without rhythm. -- Fremen proverb")
_r(
    "clpfc",
    "clpfc",
    "ClstSp",
    "Possibilistic FCM spatial",
    quote="Even the smallest person can change the future. -- Galadriel",
)
_r("clred", "clred", "ClstSp", "REDCAP spatial cluster", quote="I am here! -- All Might")
_r("clreg", "clreg", "ClstSp", "Regionalization spatial", quote="I mustn't run away. -- Shinji")
_r("clshr", "clshr", "ClstSp", "SHARCNET spatial", quote="It's over 9000! -- Vegeta")
_r("clsil", "clsil", "ClstSp", "Silhouette spatial clustering", quote="Resistance is futile. -- Borg")
_r("clskt", "clskt", "ClstSp", "SKATER spatial cluster", quote="A lesson without pain is meaningless. -- Edward")
_r("clsng", "clsng", "ClstSp", "Single linkage spatial", quote="A Lannister always pays his debts. -- Tyrion")
_r("clsom", "clsom", "ClstSp", "Self-organizing map spatial", quote="El Psy Kongroo. -- Okabe")
_r("clspc", "clspc", "ClstSp", "Spectral clustering spatial", quote="Scatter, Senbonzakura. -- Byakuya")
_r("clssmp", "clssmp", "GeoProcss", "Cluster spatial sampling", quote="Science! -- Jesse Pinkman")
_r("clsta", "clsta", "ClstSp", "Stability spatial clustering", quote="Live long and prosper. -- Spock")
_r("clval", "clval", "ClstSp", "Cluster validation spatial", quote="One does not simply walk. -- Boromir")
_r("clwlk", "clwlk", "ClstSp", "Walktrap spatial community", quote="Bankai! -- Ichigo")
_r("clwrd", "clwrd", "ClstSp", "Ward spatial clustering", quote="Growing old is a blessing. -- Rengoku")
_r("clxbr", "clxbr", "ClstSp", "Xie-Beni ratio spatial", quote="Breathe. -- Tanjiro")
_r("cmamn", "cmamn", "Spatial", "Amendment committee procedure.", quote="")
_r("cmchr", "cmchr", "Spatial", "Chair advantage spatial.", quote="")
_r("cmclr", "cmclr", "Spatial", "Closed rule committee.", quote="")
_r("cmdec", "cmdec", "Spatial", "Committee decision spatial.", quote="")
_r("cmmed", "cmmed", "Spatial", "Committee median rule.", quote="")
_r("cmopn", "cmopn", "Spatial", "Open rule committee.", quote="")
_r("cmqrm", "cmqrm", "Spatial", "Qualified majority committee.", quote="")
_r("cmstr", "cmstr", "Spatial", "Strategic committee voting.", quote="")
_r("cmunn", "cmunn", "Spatial", "Unanimity committee rule.", quote="")
_r("cmvet", "cmvet", "Spatial", "Veto player committee.", quote="")
_r("cokfl", "cokfl", "KrigFilt", "Co-kriging filter", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("concav", "concav", "GeoAnalysis", "Concave hull estimation", quote="It's over 9000! -- Vegeta")
_r("contour", "contour", "GeoAnalysis", "Contour interpolation surface", quote="The spice must flow. -- Paul Atreides")
_r("convhl", "convhl", "GeoAnalysis", "Convex hull boundary computation", quote="Believe it! -- Naruto")
_r("cosim", "cosim", "SpatialPat", "Co-simulation multivariate", quote="Equivalent exchange. -- Elric brothers")
_r(
    "crssmp",
    "crssmp",
    "GeoProcss",
    "Cross-validation spatial split",
    quote="The needs of the many outweigh the few. -- Spock",
)
_r("csaor", "csaor", "CrimSp", "Aoristic temporal analysis", quote="I am justice! -- Light")
_r("csars", "csars", "CrimSp", "Arson pattern spatial", quote="Keep moving forward. -- Eren")
_r("csast", "csast", "CrimSp", "Assault pattern spatial", quote="Whatever happens, happens. -- Spike")
_r("csbrg", "csbrg", "CrimSp", "Burglary pattern spatial", quote="I am here! -- All Might")
_r("csbt", "csbt", "CrimSp", "Beat design spatial", quote="Make it so. -- Picard")
_r("csbwt", "csbwt", "CrimSp", "Broken windows spatial", quote="I am the one who knocks. -- Walter White")
_r("cscir", "cscir", "CrimSp", "Circle theory crime", quote="Chaos is a ladder. -- Littlefinger")
_r("cscld", "cscld", "CrimSp", "Call load distribution", quote="Not all those who wander are lost. -- Gandalf")
_r("cscpr", "cscpr", "CrimSp", "CompStat spatial", quote="Hold the door. -- Hodor")
_r("cscpt", "cscpt", "CrimSp", "Crime pattern theory spatial", quote="Arise. -- Shadow Monarch")
_r("cscpt2", "cscpt2", "CrimSp", "CPTED analysis spatial", quote="Live long and prosper. -- Spock")
_r("cscrw", "cscrw", "CrimSp", "Crime random walk", quote="Power is everything. -- Sung Jin-Woo")
_r("cscyb", "cscyb", "CrimSp", "Cybercrime spatial", quote="Tatakae! -- Eren")
_r("csdef", "csdef", "CrimSp", "Defensible space spatial", quote="Get in the robot, Shinji! -- Misato")
_r("csdkp", "csdkp", "CrimSp", "Decay function crime", quote="Science! -- Jesse Pinkman")
_r("csdmv", "csdmv", "CrimSp", "Domestic violence spatial", quote="Growing old is a blessing. -- Rengoku")
_r("csdrg", "csdrg", "CrimSp", "Drug market spatial", quote="One does not simply walk. -- Boromir")
_r("csenv", "csenv", "CrimSp", "Environmental criminology", quote="El Psy Kongroo. -- Okabe")
_r("csevd", "csevd", "CrimSp", "Evidence spatial analysis", quote="Valar Morghulis. -- Braavos")
_r("csfrd", "csfrd", "CrimSp", "Fraud pattern spatial", quote="This is Requiem. -- Giorno")
_r("csfrg", "csfrg", "CrimSp", "Foraging theory crime", quote="I alone level up. -- Sung Jin-Woo")
_r("csfrn", "csfrn", "CrimSp", "Forensic spatial", quote="No half measures. -- Mike")
_r("csgng", "csgng", "CrimSp", "Gang territory spatial", quote="Walk without rhythm. -- Fremen proverb")
_r("csgpr", "csgpr", "CrimSp", "Geographic profiling crime", quote="The world is cruel but beautiful. -- Mikasa")
_r("cshmn", "cshmn", "CrimSp", "Homicide pattern spatial", quote="I mustn't run away. -- Shinji")
_r("cshts", "cshts", "CrimSp", "Hotspot analysis crime", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("csjrn", "csjrn", "CrimSp", "Journey to crime analysis", quote="Desert power. -- Paul Muad'Dib")
_r("csker", "csker", "CrimSp", "KDE crime density", quote="It's over 9000! -- Vegeta")
_r("cslvy", "cslvy", "CrimSp", "Levy flight crime", quote="Engage. -- Picard")
_r("csmhr", "csmhr", "CrimSp", "Manhunt range estimation", quote="Breathe. -- Tanjiro")
_r("csmrk", "csmrk", "CrimSp", "Crime Markov chain", quote="Equivalent exchange. -- Elric brothers")
_r("csncr", "csncr", "CrimSp", "Near-repeat analysis", quote="Dedicate your hearts! -- Erwin")
_r("csnwk", "csnwk", "CrimSp", "Crime network spatial", quote="A Lannister always pays his debts. -- Tyrion")
_r("csopt", "csopt", "CrimSp", "Optimal forager crime", quote="There is always hope. -- Aragorn")
_r("csprd", "csprd", "CrimSp", "Predictive policing spatial", quote="Kamehameha! -- Goku")
_r("csprs", "csprs", "CrimSp", "Prostitution spatial", quote="Resistance is futile. -- Borg")
_r("csptl", "csptl", "CrimSp", "Police patrol spatial", quote="My precious. -- Gollum")
_r("csr", "csr", "SpatialPat", "Complete spatial randomness test", quote="One is all, all is one. -- Izumi")
_r("csrac", "csrac", "CrimSp", "Routine activity spatial", quote="Set your heart ablaze! -- Rengoku")
_r("csres", "csres", "CrimSp", "Resource allocation police", quote="The spice must flow. -- Paul Atreides")
_r("csrgr", "csrgr", "CrimSp", "Risk terrain modeling", quote="See you space cowboy. -- Spike")
_r("csrob", "csrob", "CrimSp", "Robbery pattern spatial", quote="You should enjoy the detours. -- Ging")
_r("csrsp", "csrsp", "CrimSp", "Response time spatial", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("csrtk", "csrtk", "CrimSp", "RTK repeat target analysis", quote="Yare yare daze. -- Jotaro")
_r("csseq", "csseq", "CrimSp", "Crime sequence analysis", quote="Go beyond! Plus Ultra! -- All Might")
_r("cssfd", "cssfd", "CrimSp", "Social disorganization spatial", quote="Winter is coming. -- Stark motto")
_r("csthf", "csthf", "CrimSp", "Theft pattern spatial", quote="A lesson without pain is meaningless. -- Edward")
_r("cstpp", "cstpp", "CrimSp", "Temporal crime pattern", quote="Bankai! -- Ichigo")
_r("cstrr", "cstrr", "CrimSp", "Terrorism risk spatial", quote="The sleeper must awaken. -- Leto Atreides")
_r("csvnd", "csvnd", "CrimSp", "Vandalism pattern spatial", quote="Total Concentration Breathing. -- Tanjiro")
_r("cszon", "cszon", "CrimSp", "Zone design police", quote="Believe it! -- Naruto")
_r("ctband", "ctband", "Spatial", "Contour band shading between levels.", quote="")
_r("ctclip", "ctclip", "Spatial", "Contour clipping to domain polygon.", quote="")
_r("ctfill", "ctfill", "Spatial", "Filled contour polygon generation.", quote="")
_r("ctisoq", "ctisoq", "Spatial", "Isoline quantity between contours.", quote="")
_r("ctlbl", "ctlbl", "Spatial", "Contour label placement algorithm.", quote="")
_r("ctlvl", "ctlvl", "Spatial", "Contour level set at specified values.", quote="")
_r("ctmrch", "ctmrch", "Spatial", "Marching squares contour generation.", quote="")
_r("ctsmt", "ctsmt", "Spatial", "Contour line smoothing B-spline.", quote="")
_r("cttopo", "cttopo", "Spatial", "Topographic contour from DEM grid.", quote="")
_r("ctvert", "ctvert", "Spatial", "Contour vertex extraction at threshold.", quote="")
_r(
    "curvgr",
    "curvgr",
    "GeoProcss",
    "Curvilinear grid generation",
    quote="I will take a potato chip and eat it! -- Light",
)
_r("datumx", "datumx", "GeoProcss", "Datum transformation", quote="Bankai! -- Ichigo")
_r("delaun", "delaun", "GeoAnalysis", "Delaunay triangulation mesh", quote="Make it so. -- Picard")
_r("difanl", "difanl", "GeoAnalysis", "Spatial difference analysis", quote="Bankai! -- Ichigo")
_r("difsp", "difsp", "SpatialPat", "Diffusion process spatial", quote="You should enjoy the detours. -- Ging")
_r("dilat", "dilat", "KrigFilt", "Dilation spatial operation", quote="Growing old is a blessing. -- Rengoku")
_r("dk3an", "dk3an", "DimKrig", "3D anisotropic variogram", quote="I am the hope of the universe. -- Goku")
_r("dk3bl", "dk3bl", "DimKrig", "3D block kriging", quote="The spice must flow. -- Paul Atreides")
_r("dk3co", "dk3co", "DimKrig", "3D co-kriging", quote="One is all, all is one. -- Izumi")
_r("dk3dr", "dk3dr", "DimKrig", "3D directional variogram", quote="People's dreams never end! -- Blackbeard")
_r("dk3ok", "dk3ok", "DimKrig", "3D ordinary kriging", quote="Walk without rhythm. -- Fremen proverb")
_r("dk3sg", "dk3sg", "DimKrig", "3D sequential Gaussian sim", quote="I will take a potato chip and eat it! -- Light")
_r("dk3si", "dk3si", "DimKrig", "3D sequential indicator sim", quote="Scatter, Senbonzakura. -- Byakuya")
_r("dk3sk", "dk3sk", "DimKrig", "3D simple kriging", quote="There is always hope. -- Aragorn")
_r("dk3sm", "dk3sm", "DimKrig", "3D simulation sequential", quote="Keep moving forward. -- Eren")
_r("dk3uk", "dk3uk", "DimKrig", "3D universal kriging", quote="Engage. -- Picard")
_r("dk3vg", "dk3vg", "DimKrig", "3D variogram modeling", quote="Those who break the rules are scum. -- Kakashi")
_r("dk4co", "dk4co", "DimKrig", "4D co-kriging", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("dk4dr", "dk4dr", "DimKrig", "4D directional variogram", quote="Make it so. -- Picard")
_r("dk4ok", "dk4ok", "DimKrig", "4D space-time ordinary kriging", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("dk4sm", "dk4sm", "DimKrig", "4D simulation", quote="Believe it! -- Naruto")
_r("dk4vg", "dk4vg", "DimKrig", "4D space-time variogram", quote="Not all those who wander are lost. -- Gandalf")
_r("dkchg", "dkchg", "DimKrig", "Change of support kriging", quote="Set your heart ablaze! -- Rengoku")
_r("dkcrs", "dkcrs", "DimKrig", "Cressie-Huang ST covariance", quote="Power is everything. -- Sung Jin-Woo")
_r("dkdwn", "dkdwn", "DimKrig", "Kriging with downscaling", quote="See you space cowboy. -- Spike")
_r("dkek", "dkek", "DimKrig", "EBK (empirical Bayesian kriging)", quote="Get in the robot, Shinji! -- Misato")
_r("dkfit", "dkfit", "DimKrig", "FIT-GP spatial model", quote="I alone level up. -- Sung Jin-Woo")
_r("dkfkr", "dkfkr", "DimKrig", "Fixed-rank kriging", quote="I must not fear. -- Litany Against Fear")
_r("dkflt", "dkflt", "KrigFilt", "Disjunctive kriging filter", quote="Scatter, Senbonzakura. -- Byakuya")
_r("dkgak", "dkgak", "DimKrig", "GA-kriging optimization", quote="Live long and prosper. -- Spock")
_r("dkglb", "dkglb", "DimKrig", "Global kriging", quote="Go beyond! Plus Ultra! -- All Might")
_r("dkgnk", "dkgnk", "DimKrig", "Gneiting ST covariance", quote="Growing old is a blessing. -- Rengoku")
_r("dkicm", "dkicm", "DimKrig", "Intrinsic coregionalization", quote="Bankai! -- Ichigo")
_r(
    "dkinl",
    "dkinl",
    "DimKrig",
    "INLA spatial model",
    quote="Even the smallest person can change the future. -- Galadriel",
)
_r("dkkrg", "dkkrg", "DimKrig", "Kriging with external drift", quote="Winter is coming. -- Stark motto")
_r("dkkrn", "dkkrn", "DimKrig", "Kernel kriging", quote="One does not simply walk. -- Boromir")
_r("dklmc", "dklmc", "DimKrig", "Linear model coregionalization", quote="I am justice! -- Light")
_r("dkloc", "dkloc", "DimKrig", "Local kriging neighborhood", quote="Equivalent exchange. -- Elric brothers")
_r("dklrf", "dklrf", "DimKrig", "Low-rank factorization kriging", quote="The world is cruel but beautiful. -- Mikasa")
_r("dkmvc", "dkmvc", "DimKrig", "Multivariate co-kriging", quote="Dedicate your hearts! -- Erwin")
_r("dkmvk", "dkmvk", "DimKrig", "Multivariate kriging", quote="It's over 9000! -- Vegeta")
_r("dkmvv", "dkmvv", "DimKrig", "Multivariate variogram", quote="Yare yare daze. -- Jotaro")
_r("dknkr", "dknkr", "DimKrig", "Non-stationary kriging", quote="Resistance is futile. -- Borg")
_r("dknry", "dknry", "DimKrig", "NNGP kriging", quote="Breathe. -- Tanjiro")
_r("dknst", "dknst", "DimKrig", "Non-separable ST covariance", quote="I mustn't run away. -- Shinji")
_r("dkpsm", "dkpsm", "DimKrig", "Product-sum ST variogram", quote="Whatever happens, happens. -- Spike")
_r("dkrgr", "dkrgr", "DimKrig", "Regression kriging", quote="Arise. -- Shadow Monarch")
_r("dkrk", "dkrk", "DimKrig", "Residual kriging", quote="I am the one who knocks. -- Walter White")
_r("dksmt", "dksmt", "DimKrig", "Metric space-time variogram", quote="I am here! -- All Might")
_r("dkspr", "dkspr", "DimKrig", "Sparse kriging approximation", quote="Chaos is a ladder. -- Littlefinger")
_r("dkssm", "dkssm", "DimKrig", "Sum-metric ST variogram", quote="You should enjoy the detours. -- Ging")
_r("dkstf", "dkstf", "DimKrig", "ST covariance fitting", quote="A Lannister always pays his debts. -- Tyrion")
_r("dkstk", "dkstk", "DimKrig", "STPK space-time kriging", quote="The needs of the many outweigh the few. -- Spock")
_r("dkstv", "dkstv", "DimKrig", "Space-time variogram joint", quote="A lesson without pain is meaningless. -- Edward")
_r("dktpr", "dktpr", "DimKrig", "Tapering kriging", quote="Desert power. -- Paul Muad'Dib")
_r("dkupd", "dkupd", "DimKrig", "Kriging area-to-point", quote="El Psy Kongroo. -- Okabe")
_r("dkvck", "dkvck", "DimKrig", "Vecchia approximation kriging", quote="Science! -- Jesse Pinkman")
_r("dlasm", "dlasm", "SpatialPat", "DLA aggregation model", quote="There is always hope. -- Aragorn")
_r("dlauae", "dlauae", "GeoAnalysis", "Delaunay area/edge ratio", quote="Desert power. -- Paul Muad'Dib")
_r(
    "dlauci",
    "dlauci",
    "GeoAnalysis",
    "Delaunay circumradius statistics",
    quote="I must not fear. -- Litany Against Fear",
)
_r(
    "dlaumn",
    "dlaumn",
    "GeoAnalysis",
    "Delaunay minimum angle stats",
    quote="The world is cruel but beautiful. -- Mikasa",
)
_r("dlaunq", "dlaunq", "GeoAnalysis", "Delaunay triangle quality metrics", quote="Resistance is futile. -- Borg")
_r("dssim", "dssim", "SpatialPat", "Direct sampling simulation", quote="I am the one who knocks. -- Walter White")
_r("dtaic", "dtaic", "Spatial", "AIC dimensionality test.", quote="")
_r("dtasy", "dtasy", "DistTheor", "Asymmetric Laplace distribution", quote="I mustn't run away. -- Shinji")
_r("dtbic", "dtbic", "Spatial", "BIC dimensionality test.", quote="")
_r("dtbng", "dtbng", "DistTheor", "Bingham distribution", quote="Arise. -- Shadow Monarch")
_r("dtbur", "dtbur", "DistTheor", "Burr distribution", quote="Science! -- Jesse Pinkman")
_r("dtbvc", "dtbvc", "DistTheor", "Bivariate copula density", quote="One is all, all is one. -- Izumi")
_r("dtbvn", "dtbvn", "DistTheor", "Bivariate normal density", quote="I will take a potato chip and eat it! -- Light")
_r("dtbvp", "dtbvp", "DistTheor", "Bivariate Poisson", quote="Scatter, Senbonzakura. -- Byakuya")
_r("dtcar", "dtcar", "DistTheor", "Cardioid distribution", quote="Get in the robot, Shinji! -- Misato")
_r("dtcfa", "dtcfa", "Spatial", "CFA dimensionality test.", quote="")
_r("dtcir", "dtcir", "DistTheor", "Circular uniform distribution", quote="Live long and prosper. -- Spock")
_r("dtcmp", "dtcmp", "Spatial", "Comparative fit dimensionality.", quote="")
_r("dtcnf", "dtcnf", "Spatial", "Confirmatory dimensionality test.", quote="")
_r("dtcpb", "dtcpb", "DistTheor", "BB1 copula", quote="It's over 9000! -- Vegeta")
_r("dtcpc", "dtcpc", "DistTheor", "Clayton copula", quote="Not all those who wander are lost. -- Gandalf")
_r("dtcpf", "dtcpf", "DistTheor", "Frank copula", quote="Make it so. -- Picard")
_r("dtcpg", "dtcpg", "DistTheor", "Gaussian copula", quote="The spice must flow. -- Paul Atreides")
_r("dtcpg2", "dtcpg2", "DistTheor", "Gumbel copula", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("dtcpj", "dtcpj", "DistTheor", "Joe copula", quote="Believe it! -- Naruto")
_r("dtcpt", "dtcpt", "DistTheor", "Student-t copula", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("dtctm", "dtctm", "DistTheor", "Categorical distribution", quote="Go beyond! Plus Ultra! -- All Might")
_r("dtcvg", "dtcvg", "Spatial", "Convergent validity dimensionality.", quote="")
_r("dtdag", "dtdag", "DistTheor", "Dagum distribution", quote="The world is cruel but beautiful. -- Mikasa")
_r("dtdcr", "dtdcr", "Spatial", "Discriminant test dimensionality.", quote="")
_r("dtdir", "dtdir", "DistTheor", "Dirichlet distribution", quote="Yare yare daze. -- Jotaro")
_r("dteig", "dteig", "Spatial", "Eigenvalue dimensionality test.", quote="")
_r("dtfrc", "dtfrc", "DistTheor", "Frechet distribution", quote="I am here! -- All Might")
_r("dtfsk", "dtfsk", "DistTheor", "Fisk (log-logistic) distribution", quote="Chaos is a ladder. -- Littlefinger")
_r("dtgev", "dtgev", "DistTheor", "Generalized extreme value", quote="People's dreams never end! -- Blackbeard")
_r("dtghs", "dtghs", "DistTheor", "Generalized hyperbolic secant", quote="Growing old is a blessing. -- Rengoku")
_r("dtgmb", "dtgmb", "DistTheor", "Gumbel distribution", quote="You should enjoy the detours. -- Ging")
_r(
    "dtgpd",
    "dtgpd",
    "DistTheor",
    "Generalized Pareto distribution",
    quote="Those who break the rules are scum. -- Kakashi",
)
_r("dtiws", "dtiws", "DistTheor", "Inverse Wishart distribution", quote="I am justice! -- Light")
_r("dtjhm", "dtjhm", "DistTheor", "Johnson SB distribution", quote="One does not simply walk. -- Boromir")
_r("dtjhs", "dtjhs", "DistTheor", "Johnson SU distribution", quote="Resistance is futile. -- Borg")
_r("dtknt", "dtknt", "DistTheor", "Kent distribution (spherical)", quote="Set your heart ablaze! -- Rengoku")
_r("dtlkj", "dtlkj", "DistTheor", "LKJ correlation distribution", quote="Bankai! -- Ichigo")
_r("dtlmd", "dtlmd", "DistTheor", "Lambda distribution", quote="Desert power. -- Paul Muad'Dib")
_r("dtlpl", "dtlpl", "DistTheor", "Laplace distribution", quote="Whatever happens, happens. -- Spike")
_r("dtmap", "dtmap", "Spatial", "MAP test dimensionality.", quote="")
_r("dtmnm", "dtmnm", "DistTheor", "Multinomial distribution", quote="Equivalent exchange. -- Elric brothers")
_r("dtmvn", "dtmvn", "DistTheor", "Multivariate normal density", quote="Walk without rhythm. -- Fremen proverb")
_r("dtmvs", "dtmvs", "DistTheor", "Multivariate skew-normal", quote="Engage. -- Picard")
_r("dtmvt", "dtmvt", "DistTheor", "Multivariate t density", quote="There is always hope. -- Aragorn")
_r("dtmxs", "dtmxs", "DistTheor", "Max-stable process", quote="Keep moving forward. -- Eren")
_r(
    "dtnkc",
    "dtnkc",
    "DistTheor",
    "Nakagami distribution",
    quote="Even the smallest person can change the future. -- Galadriel",
)
_r("dtord", "dtord", "DistTheor", "Ordered logistic distribution", quote="See you space cowboy. -- Spike")
_r("dtpar", "dtpar", "Spatial", "Parallel analysis dimensionality.", quote="")
_r("dtprt", "dtprt", "DistTheor", "Power distribution", quote="Breathe. -- Tanjiro")
_r("dtric", "dtric", "DistTheor", "Rice distribution", quote="I alone level up. -- Sung Jin-Woo")
_r("dtrkn", "dtrkn", "Spatial", "Rank-based dimensionality.", quote="")
_r("dtrlg", "dtrlg", "DistTheor", "Rayleigh distribution", quote="The needs of the many outweigh the few. -- Spock")
_r("dtrot", "dtrot", "Spatial", "Rotation test dimensionality.", quote="")
_r("dtsas", "dtsas", "DistTheor", "Sinh-Arcsinh distribution", quote="I must not fear. -- Litany Against Fear")
_r("dtscr", "dtscr", "Spatial", "Scree test dimensionality.", quote="")
_r("dtspt", "dtspt", "DistTheor", "Spatial extreme value", quote="I am the hope of the universe. -- Goku")
_r("dttvs", "dttvs", "DistTheor", "Tukey lambda distribution", quote="Power is everything. -- Sung Jin-Woo")
_r("dtvar", "dtvar", "Spatial", "Variance explained test.", quote="")
_r("dtvmf", "dtvmf", "DistTheor", "Von Mises-Fisher distribution", quote="El Psy Kongroo. -- Okabe")
_r("dtwbl2", "dtwbl2", "DistTheor", "Weibull 3-parameter", quote="A lesson without pain is meaningless. -- Edward")
_r("dtwrn", "dtwrn", "DistTheor", "Wrapped normal distribution", quote="I am the one who knocks. -- Walter White")
_r("dtwrp", "dtwrp", "DistTheor", "Wrapped Cauchy distribution", quote="Winter is coming. -- Stark motto")
_r("dtwsh", "dtwsh", "DistTheor", "Wishart distribution", quote="Dedicate your hearts! -- Erwin")
_r("dtxpl", "dtxpl", "Spatial", "Exploratory dimensionality test.", quote="")
_r("dtzpf", "dtzpf", "DistTheor", "Zipf distribution", quote="A Lannister always pays his debts. -- Tyrion")
_r("eckrt4", "eckrt4", "GeoProcss", "Eckert IV projection", quote="Get in the robot, Shinji! -- Misato")
_r(
    "edgfp",
    "edgfp",
    "KrigFilt",
    "Edge detection spatial filter",
    quote="Even the smallest person can change the future. -- Galadriel",
)
_r("eladv", "eladv", "Spatial", "Incumbency advantage model.", quote="")
_r("elcmp", "elcmp", "Spatial", "Electoral competition index.", quote="")
_r("elcnv", "elcnv", "Spatial", "Electoral convergence measure.", quote="")
_r("eldiv", "eldiv", "Spatial", "Electoral divergence measure.", quote="")
_r("elent", "elent", "Spatial", "Electoral entry model.", quote="")
_r("elimp", "elimp", "Spatial", "Electoral importance weights.", quote="")
_r("elplm", "elplm", "Spatial", "Plurality electoral model.", quote="")
_r("elpol", "elpol", "Spatial", "Electoral polarization index.", quote="")
_r("elprm", "elprm", "Spatial", "Proportional electoral model.", quote="")
_r("elrun", "elrun", "Spatial", "Runoff electoral model.", quote="")
_r("elsal", "elsal", "Spatial", "Salience electoral model.", quote="")
_r("elunc", "elunc", "Spatial", "Uncertainty electoral model.", quote="")
_r("elval", "elval", "Spatial", "Valence electoral model.", quote="")
_r("elwgt", "elwgt", "Spatial", "Weighted electoral model.", quote="")
_r("elxit", "elxit", "Spatial", "Electoral exit model.", quote="")
_r("enaqi", "enaqi", "EnvStat", "Air quality index spatial", quote="Yare yare daze. -- Jotaro")
_r("enbdv", "enbdv", "EnvStat", "Biodiversity index spatial", quote="My precious. -- Gollum")
_r("encfd", "encfd", "EnvStat", "CFD wind field spatial", quote="Winter is coming. -- Stark motto")
_r("encos", "encos", "EnvStat", "CO spatial mapping", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("encrl", "encrl", "EnvStat", "Coral reef health spatial", quote="The sleeper must awaken. -- Leto Atreides")
_r("encrn", "encrn", "EnvStat", "Coastal erosion spatial", quote="This is Requiem. -- Giorno")
_r("endep", "endep", "EnvStat", "Deposition rate spatial", quote="Bankai! -- Ichigo")
_r("endfs", "endfs", "EnvStat", "Deforestation rate spatial", quote="Kamehameha! -- Goku")
_r("endrt", "endrt", "EnvStat", "Drought index spatial", quote="Engage. -- Picard")
_r("enelr", "enelr", "EnvStat", "Eulerian dispersion", quote="Arise. -- Shadow Monarch")
_r("eneqk", "eneqk", "EnvStat", "Earthquake hazard spatial", quote="Growing old is a blessing. -- Rengoku")
_r("enexc", "enexc", "EnvStat", "Exceedance probability air", quote="Dedicate your hearts! -- Erwin")
_r("enfir", "enfir", "EnvStat", "Fire risk index spatial", quote="Power is everything. -- Sung Jin-Woo")
_r("enfld", "enfld", "EnvStat", "Flood risk spatial", quote="A Lannister always pays his debts. -- Tyrion")
_r("enfog", "enfog", "EnvStat", "Fog frequency spatial", quote="Keep moving forward. -- Eren")
_r("enfrg", "enfrg", "EnvStat", "Forest fragmentation spatial", quote="No half measures. -- Mike")
_r("engau", "engau", "EnvStat", "Gaussian plume model", quote="See you space cowboy. -- Spike")
_r("enhbt", "enhbt", "EnvStat", "Habitat suitability spatial", quote="The spice must flow. -- Paul Atreides")
_r("enhil", "enhil", "EnvStat", "Hail risk spatial", quote="Whatever happens, happens. -- Spike")
_r("enhrc", "enhrc", "EnvStat", "Hurricane track analysis", quote="I am here! -- All Might")
_r("enhum", "enhum", "EnvStat", "Humidity spatial mapping", quote="The world is cruel but beautiful. -- Mikasa")
_r("enlgr", "enlgr", "EnvStat", "Lagrangian dispersion", quote="Set your heart ablaze! -- Rengoku")
_r("enlgt", "enlgt", "EnvStat", "Lightning density spatial", quote="I mustn't run away. -- Shinji")
_r("enlnd", "enlnd", "EnvStat", "Landslide risk spatial", quote="Walk without rhythm. -- Fremen proverb")
_r("enmng", "enmng", "EnvStat", "Mangrove extent spatial", quote="Tatakae! -- Eren")
_r("enno2", "enno2", "EnvStat", "NO2 concentration spatial", quote="Make it so. -- Picard")
_r("eno3s", "eno3s", "EnvStat", "Ozone spatial mapping", quote="Not all those who wander are lost. -- Gandalf")
_r("enpet", "enpet", "EnvStat", "Evapotranspiration spatial", quote="There is always hope. -- Aragorn")
_r("enplm", "enplm", "EnvStat", "Plume dispersion model", quote="Go beyond! Plus Ultra! -- All Might")
_r("enpm1", "enpm1", "EnvStat", "PM10 spatial interpolation", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("enpm2", "enpm2", "EnvStat", "PM2.5 spatial interpolation", quote="The spice must flow. -- Paul Atreides")
_r("enpuf", "enpuf", "EnvStat", "Puff dispersion model", quote="El Psy Kongroo. -- Okabe")
_r("enrad", "enrad", "EnvStat", "Solar radiation spatial", quote="Breathe. -- Tanjiro")
_r("enrfl", "enrfl", "EnvStat", "Rainfall spatial interpolation", quote="Chaos is a ladder. -- Littlefinger")
_r("ensea", "ensea", "EnvStat", "Sea level rise spatial", quote="Total Concentration Breathing. -- Tanjiro")
_r("ensno", "ensno", "EnvStat", "Snowfall spatial mapping", quote="Science! -- Jesse Pinkman")
_r("enso2", "enso2", "EnvStat", "SO2 concentration spatial", quote="Believe it! -- Naruto")
_r("enspd", "enspd", "EnvStat", "Species distribution model", quote="Hold the door. -- Hodor")
_r("ensrc", "ensrc", "EnvStat", "Source apportionment spatial", quote="Equivalent exchange. -- Elric brothers")
_r("enstr", "enstr", "EnvStat", "Storm track spatial", quote="A lesson without pain is meaningless. -- Edward")
_r("entfp", "entfp", "KrigFilt", "Entropy filter spatial", quote="I alone level up. -- Sung Jin-Woo")
_r("entmp", "entmp", "EnvStat", "Temperature spatial mapping", quote="Desert power. -- Paul Muad'Dib")
_r("entrd", "entrd", "EnvStat", "Temporal trend air quality", quote="I am justice! -- Light")
_r("entrd2", "entrd2", "EnvStat", "Tornado risk spatial", quote="You should enjoy the detours. -- Ging")
_r("entsn", "entsn", "EnvStat", "Tsunami risk spatial", quote="Resistance is futile. -- Borg")
_r("enuvb", "enuvb", "EnvStat", "UV-B radiation spatial", quote="I alone level up. -- Sung Jin-Woo")
_r("envlc", "envlc", "EnvStat", "Volcano hazard spatial", quote="One does not simply walk. -- Boromir")
_r("envoc", "envoc", "EnvStat", "VOC spatial distribution", quote="It's over 9000! -- Vegeta")
_r("enwdr", "enwdr", "EnvStat", "Wind direction spatial", quote="Live long and prosper. -- Spock")
_r("enwnd", "enwnd", "EnvStat", "Wind rose spatial", quote="I am the one who knocks. -- Walter White")
_r("enwsp", "enwsp", "EnvStat", "Wind speed interpolation", quote="Get in the robot, Shinji! -- Misato")
_r("enwtl", "enwtl", "EnvStat", "Wetland extent spatial", quote="Valar Morghulis. -- Braavos")
_r("erasan", "erasan", "GeoAnalysis", "Spatial erase operation", quote="Go beyond! Plus Ultra! -- All Might")
_r("erosn", "erosn", "KrigFilt", "Erosion spatial operation", quote="Power is everything. -- Sung Jin-Woo")
_r(
    "fbmsi",
    "fbmsi",
    "SpatialPat",
    "Fractional Brownian spatial",
    quote="The needs of the many outweigh the few. -- Spock",
)
_r("fftsi", "fftsi", "SpatialPat", "FFT-based simulation", quote="I am justice! -- Light")
_r("filsm", "filsm", "SpatialPat", "FILTERSIM MPS method", quote="Arise. -- Shadow Monarch")
_r("foage", "foage", "ForstSp", "Age class distribution spatial", quote="Chaos is a ladder. -- Littlefinger")
_r("fobark", "fobark", "ForstSp", "Bark beetle risk spatial", quote="Growing old is a blessing. -- Rengoku")
_r("fobas", "fobas", "ForstSp", "Basal area spatial", quote="El Psy Kongroo. -- Okabe")
_r("fobio", "fobio", "ForstSp", "Biomass estimation forest", quote="Yare yare daze. -- Jotaro")
_r("fobufr", "fobufr", "ForstSp", "Buffer zone forest", quote="Keep moving forward. -- Eren")
_r("fochm", "fochm", "ForstSp", "Crown height model LiDAR", quote="Tatakae! -- Eren")
_r("focnp", "focnp", "ForstSp", "Canopy cover spatial", quote="I am justice! -- Light")
_r("focnpd", "focnpd", "ForstSp", "Canopy density spatial", quote="Go beyond! Plus Ultra! -- All Might")
_r("focnpg", "focnpg", "ForstSp", "Canopy gap analysis", quote="Bankai! -- Ichigo")
_r("focnph", "focnph", "ForstSp", "Canopy height model", quote="Equivalent exchange. -- Elric brothers")
_r("focorr", "focorr", "ForstSp", "Corridor connectivity", quote="Whatever happens, happens. -- Spike")
_r("focrb", "focrb", "ForstSp", "Carbon stock forest", quote="Dedicate your hearts! -- Erwin")
_r("fodbh", "fodbh", "ForstSp", "DBH distribution spatial", quote="Believe it! -- Naruto")
_r("fodiv", "fodiv", "ForstSp", "Species diversity forest", quote="Science! -- Jesse Pinkman")
_r("fodrt", "fodrt", "ForstSp", "Drought stress forest", quote="One does not simply walk. -- Boromir")
_r("fodsm", "fodsm", "ForstSp", "Digital surface model", quote="Valar Morghulis. -- Braavos")
_r("fodtm", "fodtm", "ForstSp", "Digital terrain model", quote="The sleeper must awaken. -- Leto Atreides")
_r("foedge", "foedge", "ForstSp", "Edge effect spatial", quote="You should enjoy the detours. -- Ging")
_r("foeqt", "foeqt", "ForstSp", "Equitability spatial", quote="Engage. -- Picard")
_r("fofire", "fofire", "ForstSp", "Fire risk forest spatial", quote="Power is everything. -- Sung Jin-Woo")
_r("fofrag", "fofrag", "ForstSp", "Forest fragmentation", quote="I am here! -- All Might")
_r("fofuel", "fofuel", "ForstSp", "Fuel load spatial", quote="A Lannister always pays his debts. -- Tyrion")
_r("fofwi", "fofwi", "ForstSp", "Fire weather index spatial", quote="Walk without rhythm. -- Fremen proverb")
_r("fogrth", "fogrth", "ForstSp", "Growth rate spatial", quote="Get in the robot, Shinji! -- Misato")
_r("foht", "foht", "ForstSp", "Height distribution spatial", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("foinv", "foinv", "ForstSp", "Forest inventory spatial", quote="Make it so. -- Picard")
_r("folcsc", "folcsc", "ForstSp", "Landscape connectivity", quote="Total Concentration Breathing. -- Tanjiro")
_r("folidr", "folidr", "ForstSp", "LiDAR point cloud spatial", quote="This is Requiem. -- Giorno")
_r("fomort", "fomort", "ForstSp", "Mortality rate spatial", quote="I am the one who knocks. -- Walter White")
_r("fonchm", "fonchm", "ForstSp", "Normalized CHM LiDAR", quote="No half measures. -- Mike")
_r("foplts", "foplts", "ForstSp", "Plot-level summary spatial", quote="Hold the door. -- Hodor")
_r("foptch", "foptch", "ForstSp", "Patch isolation spatial", quote="I mustn't run away. -- Shinji")
_r("foquad", "foquad", "ForstSp", "Quadrat sampling spatial", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("forecr", "forecr", "ForstSp", "Recruitment rate spatial", quote="Live long and prosper. -- Spock")
_r("foreld", "foreld", "ForstSp", "Relative density spatial", quote="Winter is coming. -- Stark motto")
_r("forich", "forich", "ForstSp", "Species richness spatial", quote="Breathe. -- Tanjiro")
_r("forsdi", "forsdi", "ForstSp", "Reineke SDI spatial", quote="Arise. -- Shadow Monarch")
_r("forstr", "forstr", "ForstSp", "Random stratified forest", quote="Not all those who wander are lost. -- Gandalf")
_r("fosamp", "fosamp", "ForstSp", "Adaptive forest sampling", quote="Make it so. -- Picard")
_r("fosdx", "fosdx", "ForstSp", "Stand density index", quote="Set your heart ablaze! -- Rengoku")
_r("foshan", "foshan", "ForstSp", "Shannon diversity spatial", quote="I alone level up. -- Sung Jin-Woo")
_r("fosimp", "fosimp", "ForstSp", "Simpson diversity spatial", quote="There is always hope. -- Aragorn")
_r("fosnwl", "fosnwl", "ForstSp", "Snow load risk spatial", quote="A lesson without pain is meaningless. -- Edward")
_r("fostrm", "fostrm", "ForstSp", "Stem density spatial", quote="See you space cowboy. -- Spike")
_r("fosucc", "fosucc", "ForstSp", "Succession stage spatial", quote="The world is cruel but beautiful. -- Mikasa")
_r("fotrew", "fotrew", "ForstSp", "Tree detection LiDAR", quote="Kamehameha! -- Goku")
_r("fotrnk", "fotrnk", "ForstSp", "Transect analysis spatial", quote="The spice must flow. -- Paul Atreides")
_r("fotrsg", "fotrsg", "ForstSp", "Tree segmentation LiDAR", quote="My precious. -- Gollum")
_r("foturn", "foturn", "ForstSp", "Turnover rate spatial", quote="Desert power. -- Paul Muad'Dib")
_r("fovol", "fovol", "ForstSp", "Volume estimation spatial", quote="It's over 9000! -- Vegeta")
_r("fowndf", "fowndf", "ForstSp", "Windfall risk spatial", quote="Resistance is futile. -- Borg")
_r("ftest", "ftest", "SpatialPat", "F-test spatial distribution", quote="The spice must flow. -- Paul Atreides")
_r("gaaspc", "gaaspc", "GeoAnalysis", "Aspect direction analysis", quote="Whatever happens, happens. -- Spike")
_r(
    "gaazbr",
    "gaazbr",
    "GeoAnalysis",
    "Azimuth bearing calculation",
    quote="A lesson without pain is meaningless. -- Edward",
)
_r("gabfp", "gabfp", "KrigFilt", "Gabor filter spatial", quote="Whatever happens, happens. -- Spike")
_r("gabsn", "gabsn", "GeoAnalysis", "Basin extraction", quote="I am the hope of the universe. -- Goku")
_r("gacurv", "gacurv", "GeoAnalysis", "Curvature terrain analysis", quote="I mustn't run away. -- Shinji")
_r("gadrnp", "gadrnp", "GeoAnalysis", "Drainage pattern classification", quote="Scatter, Senbonzakura. -- Byakuya")
_r("gaelev", "gaelev", "GeoAnalysis", "Elevation profile extraction", quote="I am here! -- All Might")
_r("gafacc", "gafacc", "GeoAnalysis", "Flow accumulation computation", quote="Engage. -- Picard")
_r("gafdir", "gafdir", "GeoAnalysis", "Flow direction analysis", quote="Those who break the rules are scum. -- Kakashi")
_r("gahill", "gahill", "GeoAnalysis", "Hillshade illumination model", quote="Growing old is a blessing. -- Rengoku")
_r(
    "gaprox",
    "gaprox",
    "GeoAnalysis",
    "Proximity band analysis",
    quote="Even the smallest person can change the future. -- Galadriel",
)
_r("garng", "garng", "GeoAnalysis", "Range analysis spatial", quote="The needs of the many outweigh the few. -- Spock")
_r("gaslop", "gaslop", "GeoAnalysis", "Slope gradient computation", quote="You should enjoy the detours. -- Ging")
_r("gaspi", "gaspi", "GeoAnalysis", "Stream power index", quote="There is always hope. -- Aragorn")
_r("gastrl", "gastrl", "GeoAnalysis", "Stream link network", quote="Keep moving forward. -- Eren")
_r("gastrm", "gastrm", "GeoAnalysis", "Stream order (Strahler)", quote="I will take a potato chip and eat it! -- Light")
_r("gatpi", "gatpi", "GeoAnalysis", "Topographic position index", quote="Power is everything. -- Sung Jin-Woo")
_r("gatri", "gatri", "GeoAnalysis", "Terrain ruggedness index", quote="A Lannister always pays his debts. -- Tyrion")
_r("gatwi", "gatwi", "GeoAnalysis", "Topographic wetness index", quote="Walk without rhythm. -- Fremen proverb")
_r("gaview", "gaview", "GeoAnalysis", "Viewshed analysis", quote="One is all, all is one. -- Izumi")
_r("gawshd", "gawshd", "GeoAnalysis", "Watershed delineation", quote="People's dreams never end! -- Blackbeard")
_r("gcacd", "gcacd", "GeoClim", "Ocean acidification trend", quote="The spice must flow. -- Paul Atreides")
_r("gcadp", "gcadp", "GeoClim", "Adaptation capacity spatial", quote="I am justice! -- Light")
_r("gcaer", "gcaer", "GeoClim", "Aerosol forcing spatial", quote="Winter is coming. -- Stark motto")
_r("gcalb", "gcalb", "GeoClim", "Albedo feedback spatial", quote="El Psy Kongroo. -- Okabe")
_r("gcams", "gcams", "GeoClim", "AMOC strength change", quote="Make it so. -- Picard")
_r("gcch4", "gcch4", "GeoClim", "CH4 concentration spatial", quote="The spice must flow. -- Paul Atreides")
_r("gccld", "gccld", "GeoClim", "Cloud feedback spatial", quote="Arise. -- Shadow Monarch")
_r("gccnv", "gccnv", "GeoClim", "Convection change spatial", quote="Bankai! -- Ichigo")
_r("gcco2", "gcco2", "GeoClim", "CO2 concentration spatial", quote="Get in the robot, Shinji! -- Misato")
_r("gccor", "gccor", "GeoClim", "Coral bleaching trend", quote="Get in the robot, Shinji! -- Misato")
_r("gccwv", "gccwv", "GeoClim", "Cold wave frequency", quote="El Psy Kongroo. -- Okabe")
_r("gcdox", "gcdox", "GeoClim", "Ocean deoxygenation", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("gcdrg", "gcdrg", "GeoClim", "Drought trend spatial", quote="Not all those who wander are lost. -- Gandalf")
_r("gcedn", "gcedn", "GeoClim", "ENSO pattern change", quote="Believe it! -- Naruto")
_r("gcesm", "gcesm", "GeoClim", "Earth system model spatial", quote="Make it so. -- Picard")
_r("gcext", "gcext", "GeoClim", "Extreme event frequency", quote="Go beyond! Plus Ultra! -- All Might")
_r("gcfld", "gcfld", "GeoClim", "Flood trend spatial", quote="Make it so. -- Picard")
_r("gcgcm", "gcgcm", "GeoClim", "GCM downscaling spatial", quote="Believe it! -- Naruto")
_r("gcghg", "gcghg", "GeoClim", "GHG concentration spatial", quote="I am the one who knocks. -- Walter White")
_r("gcglc", "gcglc", "GeoClim", "Glacier retreat spatial", quote="It's over 9000! -- Vegeta")
_r("gcgrw", "gcgrw", "GeoClim", "Growing season change", quote="Equivalent exchange. -- Elric brothers")
_r("gchad", "gchad", "GeoClim", "Hadley cell expansion", quote="Go beyond! Plus Ultra! -- All Might")
_r("gchrc", "gchrc", "GeoClim", "Hurricane intensity change", quote="Arise. -- Shadow Monarch")
_r("gchwv", "gchwv", "GeoClim", "Heat wave frequency", quote="See you space cowboy. -- Spike")
_r("gcice", "gcice", "GeoClim", "Ice sheet change spatial", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("gciod", "gciod", "GeoClim", "IOD pattern spatial", quote="Yare yare daze. -- Jotaro")
_r("gcjet", "gcjet", "GeoClim", "Jet stream shift spatial", quote="Equivalent exchange. -- Elric brothers")
_r("gcmjo", "gcmjo", "GeoClim", "MJO pattern spatial", quote="Dedicate your hearts! -- Erwin")
_r("gcn2o", "gcn2o", "GeoClim", "N2O concentration spatial", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("gcnao", "gcnao", "GeoClim", "NAO pattern spatial", quote="It's over 9000! -- Vegeta")
_r("gcpdo", "gcpdo", "GeoClim", "PDO pattern spatial", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("gcphn", "gcphn", "GeoClim", "Phenology change spatial", quote="Bankai! -- Ichigo")
_r("gcplr", "gcplr", "GeoClim", "Polar vortex spatial", quote="See you space cowboy. -- Spike")
_r("gcprc", "gcprc", "GeoClim", "Precipitation trend spatial", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("gcprm", "gcprm", "GeoClim", "Permafrost thaw spatial", quote="Yare yare daze. -- Jotaro")
_r("gcqbo", "gcqbo", "GeoClim", "QBO pattern spatial", quote="I am justice! -- Light")
_r("gcrcf", "gcrcf", "GeoClim", "Radiative forcing spatial", quote="Not all those who wander are lost. -- Gandalf")
_r("gcrcm", "gcrcm", "GeoClim", "RCM downscaling spatial", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("gcrcp", "gcrcp", "GeoClim", "RCP scenario spatial", quote="Dedicate your hearts! -- Erwin")
_r("gcslr", "gcslr", "GeoClim", "Sea level rise projection", quote="Believe it! -- Naruto")
_r("gcsnw", "gcsnw", "GeoClim", "Snow cover change spatial", quote="Dedicate your hearts! -- Erwin")
_r("gcssp", "gcssp", "GeoClim", "SSP scenario spatial", quote="Yare yare daze. -- Jotaro")
_r("gcstd", "gcstd", "GeoClim", "Statistical downscaling", quote="It's over 9000! -- Vegeta")
_r("gcstr", "gcstr", "GeoClim", "Storm intensity change", quote="Set your heart ablaze! -- Rengoku")
_r("gctmp", "gctmp", "GeoClim", "Temperature trend spatial", quote="The spice must flow. -- Paul Atreides")
_r("gctrd", "gctrd", "GeoClim", "Tornado trend spatial", quote="Winter is coming. -- Stark motto")
_r("gcupw", "gcupw", "GeoClim", "Upwelling change spatial", quote="Not all those who wander are lost. -- Gandalf")
_r("gcveg", "gcveg", "GeoClim", "Vegetation shift spatial", quote="I am justice! -- Light")
_r("gcwlf", "gcwlf", "GeoClim", "Wildfire trend spatial", quote="I am the one who knocks. -- Walter White")
_r("gcwvp", "gcwvp", "GeoClim", "Water vapor feedback", quote="Set your heart ablaze! -- Rengoku")
_r("gdadr", "gdadr", "GeoDem", "Age dependency ratio", quote="Winter is coming. -- Stark motto")
_r("gdcbr", "gdcbr", "GeoDem", "Crude birth rate spatial", quote="Set your heart ablaze! -- Rengoku")
_r("gdcdr", "gdcdr", "GeoDem", "Crude death rate spatial", quote="El Psy Kongroo. -- Okabe")
_r("gdcmp", "gdcmp", "GeoDem", "Competitiveness electoral", quote="Believe it! -- Naruto")
_r("gdcst", "gdcst", "GeoDem", "Caste distribution spatial", quote="Total Concentration Breathing. -- Tanjiro")
_r("gdctz", "gdctz", "GeoDem", "Citizenship spatial", quote="The sleeper must awaken. -- Leto Atreides")
_r("gddvr", "gddvr", "GeoDem", "Divorce rate spatial", quote="One does not simply walk. -- Boromir")
_r("gdeld", "gdeld", "GeoDem", "Elderly ratio spatial", quote="Get in the robot, Shinji! -- Misato")
_r("gdelf", "gdelf", "GeoDem", "Election result spatial", quote="Kamehameha! -- Goku")
_r("gdemg", "gdemg", "GeoDem", "Emigration spatial", quote="A lesson without pain is meaningless. -- Edward")
_r("gdeth", "gdeth", "GeoDem", "Ethnicity distribution spatial", quote="Whatever happens, happens. -- Spike")
_r("gdflm", "gdflm", "GeoDem", "Family size spatial", quote="Walk without rhythm. -- Fremen proverb")
_r("gdfrt", "gdfrt", "GeoDem", "Fertility rate spatial", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("gdger", "gdger", "GeoDem", "Gerrymandering detection", quote="Hold the door. -- Hodor")
_r("gdgin", "gdgin", "GeoDem", "Gini coefficient spatial", quote="I alone level up. -- Sung Jin-Woo")
_r("gdhdv", "gdhdv", "GeoDem", "HDI spatial mapping", quote="There is always hope. -- Aragorn")
_r("gdhhs", "gdhhs", "GeoDem", "Household size spatial", quote="A Lannister always pays his debts. -- Tyrion")
_r("gdidp", "gdidp", "GeoDem", "IDP spatial mapping", quote="You should enjoy the detours. -- Ging")
_r("gdimm", "gdimm", "GeoDem", "Immigration spatial", quote="Resistance is futile. -- Borg")
_r("gdinf", "gdinf", "GeoDem", "Infant mortality spatial", quote="Go beyond! Plus Ultra! -- All Might")
_r("gdinm", "gdinm", "GeoDem", "In-migration spatial", quote="Dedicate your hearts! -- Erwin")
_r("gdlbr", "gdlbr", "GeoDem", "Labor force spatial", quote="Power is everything. -- Sung Jin-Woo")
_r("gdlfe", "gdlfe", "GeoDem", "Life expectancy spatial", quote="Equivalent exchange. -- Elric brothers")
_r("gdlit", "gdlit", "GeoDem", "Literacy rate spatial", quote="Science! -- Jesse Pinkman")
_r("gdlng", "gdlng", "GeoDem", "Language distribution spatial", quote="Keep moving forward. -- Eren")
_r("gdmap", "gdmap", "GeoDem", "Malapportionment spatial", quote="The spice must flow. -- Paul Atreides")
_r("gdmar", "gdmar", "GeoDem", "Margin analysis spatial", quote="Not all those who wander are lost. -- Gandalf")
_r("gdmed", "gdmed", "GeoDem", "Median age spatial", quote="Desert power. -- Paul Muad'Dib")
_r("gdmig", "gdmig", "GeoDem", "Migration flow spatial", quote="Yare yare daze. -- Jotaro")
_r("gdmml", "gdmml", "GeoDem", "Maternal mortality spatial", quote="See you space cowboy. -- Spike")
_r("gdmrg", "gdmrg", "GeoDem", "Marriage rate spatial", quote="Growing old is a blessing. -- Rengoku")
_r("gdmrt", "gdmrt", "GeoDem", "Mortality rate spatial", quote="It's over 9000! -- Vegeta")
_r("gdntm", "gdntm", "GeoDem", "Net migration spatial", quote="Bankai! -- Ichigo")
_r("gdntn", "gdntn", "GeoDem", "Nationality spatial", quote="Tatakae! -- Eren")
_r("gdotm", "gdotm", "GeoDem", "Out-migration spatial", quote="I am justice! -- Light")
_r("gdpop", "gdpop", "GeoDem", "Population projection spatial", quote="Believe it! -- Naruto")
_r("gdpty", "gdpty", "GeoDem", "Political party spatial", quote="No half measures. -- Mike")
_r("gdpvt", "gdpvt", "GeoDem", "Poverty rate spatial", quote="Breathe. -- Tanjiro")
_r("gdred", "gdred", "GeoDem", "Redistricting spatial", quote="My precious. -- Gollum")
_r("gdref", "gdref", "GeoDem", "Refugee flow spatial", quote="I am here! -- All Might")
_r("gdrel", "gdrel", "GeoDem", "Religion distribution spatial", quote="I mustn't run away. -- Shinji")
_r("gdrur", "gdrur", "GeoDem", "Rural population spatial", quote="Chaos is a ladder. -- Littlefinger")
_r("gdsex", "gdsex", "GeoDem", "Sex ratio spatial", quote="Live long and prosper. -- Spock")
_r("gdswg", "gdswg", "GeoDem", "Swing analysis spatial", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("gdtfr", "gdtfr", "GeoDem", "Total fertility rate spatial", quote="Arise. -- Shadow Monarch")
_r("gdtnd", "gdtnd", "GeoDem", "Trend analysis electoral", quote="Make it so. -- Picard")
_r("gdtrb", "gdtrb", "GeoDem", "Tribal distribution spatial", quote="This is Requiem. -- Giorno")
_r("gduem", "gduem", "GeoDem", "Unemployment spatial", quote="Engage. -- Picard")
_r("gdurb", "gdurb", "GeoDem", "Urbanization rate spatial", quote="The world is cruel but beautiful. -- Mikasa")
_r("gdvtr", "gdvtr", "GeoDem", "Voter turnout spatial", quote="Valar Morghulis. -- Braavos")
_r("gdyrm", "gdyrm", "GeoDem", "Youth ratio spatial", quote="I am the one who knocks. -- Walter White")
_r("geagm", "geagm", "GeoEcon", "Agglomeration index spatial", quote="Yare yare daze. -- Jotaro")
_r("geagr", "geagr", "GeoEcon", "Agriculture GDP spatial", quote="You should enjoy the detours. -- Ging")
_r("geapt", "geapt", "GeoEcon", "Patent density spatial", quote="Not all those who wander are lost. -- Gandalf")
_r("geatk", "geatk", "GeoEcon", "Atkinson index spatial", quote="Get in the robot, Shinji! -- Misato")
_r("gebal", "gebal", "GeoEcon", "Trade balance spatial", quote="The sleeper must awaken. -- Leto Atreides")
_r("gebgt", "gebgt", "GeoEcon", "Budget allocation spatial", quote="Hold the door. -- Hodor")
_r("gebnk", "gebnk", "GeoEcon", "Banking access spatial", quote="It's over 9000! -- Vegeta")
_r("gecns", "gecns", "GeoEcon", "Construction sector spatial", quote="Keep moving forward. -- Eren")
_r("gecon", "gecon", "GeoEcon", "Concentration economic", quote="Dedicate your hearts! -- Erwin")
_r("gedur", "gedur", "GeoEcon", "Duranton-Overman index", quote="Go beyond! Plus Ultra! -- All Might")
_r("geeduc", "geeduc", "GeoEcon", "Education sector spatial", quote="A lesson without pain is meaningless. -- Edward")
_r("geell", "geell", "GeoEcon", "Ellison-Glaeser index", quote="Equivalent exchange. -- Elric brothers")
_r("geemp", "geemp", "GeoEcon", "Employment spatial", quote="Engage. -- Picard")
_r("geenr", "geenr", "GeoEcon", "Energy sector spatial", quote="I mustn't run away. -- Shinji")
_r("geexp", "geexp", "GeoEcon", "Export spatial mapping", quote="This is Requiem. -- Giorno")
_r("gefdi", "gefdi", "GeoEcon", "FDI flow spatial", quote="Valar Morghulis. -- Braavos")
_r("gefin", "gefin", "GeoEcon", "Financial sector spatial", quote="One does not simply walk. -- Boromir")
_r("gefrm", "gefrm", "GeoEcon", "Firm density spatial", quote="There is always hope. -- Aragorn")
_r("gegni", "gegni", "GeoEcon", "Gini spatial economic", quote="Winter is coming. -- Stark motto")
_r("gegry", "gegry", "GeoEcon", "Geary economic spatial", quote="El Psy Kongroo. -- Okabe")
_r("gehlc", "gehlc", "GeoEcon", "Healthcare sector spatial", quote="Resistance is futile. -- Borg")
_r("gehpr", "gehpr", "GeoEcon", "Housing price spatial", quote="Chaos is a ladder. -- Littlefinger")
_r("geimp", "geimp", "GeoEcon", "Import spatial mapping", quote="Tatakae! -- Eren")
_r("geinc", "geinc", "GeoEcon", "Income spatial mapping", quote="Desert power. -- Paul Muad'Dib")
_r("geind", "geind", "GeoEcon", "Industry spatial mapping", quote="Power is everything. -- Sung Jin-Woo")
_r("geinf", "geinf", "GeoEcon", "Infrastructure investment spatial", quote="The spice must flow. -- Paul Atreides")
_r("geinq", "geinq", "GeoEcon", "Inequality spatial", quote="Arise. -- Shadow Monarch")
_r("geinv", "geinv", "GeoEcon", "Innovation index spatial", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("gelnd", "gelnd", "GeoEcon", "Land value spatial", quote="Science! -- Jesse Pinkman")
_r("geloc", "geloc", "GeoEcon", "Location quotient spatial", quote="Bankai! -- Ichigo")
_r("gemfi", "gemfi", "GeoEcon", "Microfinance spatial", quote="Yare yare daze. -- Jotaro")
_r("gemin", "gemin", "GeoEcon", "Mining sector spatial", quote="Whatever happens, happens. -- Spike")
_r("gemnf", "gemnf", "GeoEcon", "Manufacturing spatial", quote="Walk without rhythm. -- Fremen proverb")
_r("gemrn", "gemrn", "GeoEcon", "Moran economic spatial", quote="See you space cowboy. -- Spike")
_r("geoda", "geoda", "GeoEcon", "ODA flow spatial", quote="No half measures. -- Mike")
_r(
    "geodes",
    "geodes",
    "GeoProcss",
    "Geodesic distance calculation",
    quote="Not all those who wander are lost. -- Gandalf",
)
_r("geoidh", "geoidh", "GeoProcss", "Geoid height computation", quote="Equivalent exchange. -- Elric brothers")
_r("gepvt", "gepvt", "GeoEcon", "Poverty spatial mapping", quote="Live long and prosper. -- Spock")
_r("gerdx", "gerdx", "GeoEcon", "R&D expenditure spatial", quote="Make it so. -- Picard")
_r("germt", "germt", "GeoEcon", "Remittance flow spatial", quote="Kamehameha! -- Goku")
_r("gernt", "gernt", "GeoEcon", "Rent spatial mapping", quote="Breathe. -- Tanjiro")
_r("geseg", "geseg", "GeoEcon", "Economic segregation spatial", quote="Set your heart ablaze! -- Rengoku")
_r("gespc", "gespc", "GeoEcon", "Specialization index spatial", quote="I am justice! -- Light")
_r("gestrt", "gestrt", "GeoEcon", "Startup density spatial", quote="Believe it! -- Naruto")
_r("gesvc", "gesvc", "GeoEcon", "Service sector spatial", quote="A Lannister always pays his debts. -- Tyrion")
_r("getax", "getax", "GeoEcon", "Tax revenue spatial", quote="My precious. -- Gollum")
_r("getec", "getec", "GeoEcon", "Technology sector spatial", quote="Growing old is a blessing. -- Rengoku")
_r("gethl", "gethl", "GeoEcon", "Theil index spatial", quote="I am the one who knocks. -- Walter White")
_r("getrd", "getrd", "GeoEcon", "Trade flow spatial", quote="Total Concentration Breathing. -- Tanjiro")
_r("getrs", "getrs", "GeoEcon", "Tourism spatial", quote="I am here! -- All Might")
_r("gevac", "gevac", "GeoEcon", "Vacancy economic spatial", quote="I alone level up. -- Sung Jin-Woo")
_r("gevcf", "gevcf", "GeoEcon", "Venture capital spatial", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("gewlt", "gewlt", "GeoEcon", "Wealth spatial mapping", quote="The world is cruel but beautiful. -- Mikasa")
_r("gh2sf", "gh2sf", "GeoHlth", "2SFCA accessibility health", quote="There is always hope. -- Aragorn")
_r("ghacc", "ghacc", "GeoHlth", "Access score health", quote="A Lannister always pays his debts. -- Tyrion")
_r("ghbar", "ghbar", "GeoHlth", "Barrier analysis health", quote="Walk without rhythm. -- Fremen proverb")
_r("ghbch", "ghbch", "GeoHlth", "Benchmark health spatial", quote="Believe it! -- Naruto")
_r("ghbuf", "ghbuf", "GeoHlth", "Buffer analysis health", quote="Science! -- Jesse Pinkman")
_r("ghbym", "ghbym", "GeoHlth", "BYM model health", quote="I am justice! -- Light")
_r("ghcar", "ghcar", "GeoHlth", "CAR model health", quote="Bankai! -- Ichigo")
_r("ghchg", "ghchg", "GeoHlth", "Change point health", quote="I am the one who knocks. -- Walter White")
_r("ghclr", "ghclr", "GeoHlth", "Cluster detection health", quote="El Psy Kongroo. -- Okabe")
_r("ghcov", "ghcov", "GeoHlth", "Coverage health", quote="Resistance is futile. -- Borg")
_r("ghcox", "ghcox", "GeoHlth", "Cox spatial health", quote="Hold the door. -- Hodor")
_r("ghdis", "ghdis", "GeoHlth", "Disease mapping spatial", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("ghdlm", "ghdlm", "GeoHlth", "Dynamic linear health", quote="Make it so. -- Picard")
_r("ghdst", "ghdst", "GeoHlth", "Distance to facility health", quote="One does not simply walk. -- Boromir")
_r("ghe2s", "ghe2s", "GeoHlth", "E2SFCA accessibility", quote="Engage. -- Picard")
_r("ghebr", "ghebr", "GeoHlth", "EB rate smoothing health", quote="Dedicate your hearts! -- Erwin")
_r("ghexc", "ghexc", "GeoHlth", "Exceedance mapping health", quote="Get in the robot, Shinji! -- Misato")
_r("ghexp", "ghexp", "GeoHlth", "Exposure mapping health", quote="The world is cruel but beautiful. -- Mikasa")
_r("ghfca", "ghfca", "GeoHlth", "Floating catchment area", quote="I alone level up. -- Sung Jin-Woo")
_r("ghfrg", "ghfrg", "GeoHlth", "Frailty spatial health", quote="The spice must flow. -- Paul Atreides")
_r("ghgap", "ghgap", "GeoHlth", "Gap analysis health", quote="A lesson without pain is meaningless. -- Edward")
_r("ghgrv", "ghgrv", "GeoHlth", "Gravity model health", quote="Breathe. -- Tanjiro")
_r("ghgwr", "ghgwr", "GeoHlth", "GWR health", quote="This is Requiem. -- Giorno")
_r("ghhot", "ghhot", "GeoHlth", "Hotspot detection health", quote="Set your heart ablaze! -- Rengoku")
_r("ghinl", "ghinl", "GeoHlth", "INLA model health", quote="Equivalent exchange. -- Elric brothers")
_r("ghke2", "ghke2", "GeoHlth", "KD-E2SFCA accessibility", quote="Power is everything. -- Sung Jin-Woo")
_r("ghlam", "ghlam", "GeoHlth", "LASSO health regression", quote="Total Concentration Breathing. -- Tanjiro")
_r("ghmgw", "ghmgw", "GeoHlth", "MGWR health", quote="Tatakae! -- Eren")
_r("ghmlv", "ghmlv", "GeoHlth", "Multilevel health spatial", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("ghmxc", "ghmxc", "GeoHlth", "Max coverage health", quote="Keep moving forward. -- Eren")
_r("ghnod", "ghnod", "GeoHlth", "Need-based allocation", quote="I am here! -- All Might")
_r("ghold", "ghold", "GeoHlth", "Coldspot detection health", quote="Arise. -- Shadow Monarch")
_r("ghopt", "ghopt", "GeoHlth", "Optimal location health", quote="You should enjoy the detours. -- Ging")
_r("ghpcn", "ghpcn", "GeoHlth", "P-center health", quote="I mustn't run away. -- Shinji")
_r("ghpme", "ghpme", "GeoHlth", "P-median health", quote="Whatever happens, happens. -- Spike")
_r("ghpnl", "ghpnl", "GeoHlth", "Panel health spatial", quote="Kamehameha! -- Goku")
_r("ghprb", "ghprb", "GeoHlth", "Probability mapping health", quote="Live long and prosper. -- Spock")
_r("ghqly", "ghqly", "GeoHlth", "Quality indicator health", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("ghrsk", "ghrsk", "GeoHlth", "Risk mapping health", quote="Desert power. -- Paul Muad'Dib")
_r("ghscn", "ghscn", "GeoHlth", "Scan statistic health", quote="See you space cowboy. -- Spike")
_r("ghsdm", "ghsdm", "GeoHlth", "SDM health", quote="No half measures. -- Mike")
_r("ghsem", "ghsem", "GeoHlth", "SEM health spatial", quote="The sleeper must awaken. -- Leto Atreides")
_r("ghsir", "ghsir", "GeoHlth", "SIR mapping spatial", quote="Yare yare daze. -- Jotaro")
_r("ghslg", "ghslg", "GeoHlth", "Spatial lag health", quote="Valar Morghulis. -- Braavos")
_r("ghsmr", "ghsmr", "GeoHlth", "SMR mapping spatial", quote="It's over 9000! -- Vegeta")
_r("ghspt", "ghspt", "GeoHlth", "SPDE model health", quote="Go beyond! Plus Ultra! -- All Might")
_r("ghspt2", "ghspt2", "GeoHlth", "Source proximity health", quote="Chaos is a ladder. -- Littlefinger")
_r("ghsrv", "ghsrv", "GeoHlth", "Survival spatial health", quote="My precious. -- Gollum")
_r("ghssm", "ghssm", "GeoHlth", "State-space health spatial", quote="Not all those who wander are lost. -- Gandalf")
_r("ghtim", "ghtim", "GeoHlth", "Travel time health", quote="Growing old is a blessing. -- Rengoku")
_r("ghtrd", "ghtrd", "GeoHlth", "Trend detection health", quote="Winter is coming. -- Stark motto")
_r("gibsp", "gibsp", "SpatialPat", "Gibbs sampler spatial", quote="Desert power. -- Paul Muad'Dib")
_r("glmmbin", "glmmbin", "Spatial", "Spatial GLMM binomial logit simulation.", quote="")
_r("glmmcar", "glmmcar", "Spatial", "Spatial GLMM CAR prior lattice simulation.", quote="")
_r("glmmcov", "glmmcov", "Spatial", "Spatial GLMM covariance parameter estimation.", quote="")
_r("glmmdiag", "glmmdiag", "Spatial", "Spatial GLMM model diagnostic residuals.", quote="")
_r("glmmfit", "glmmfit", "Spatial", "Spatial GLMM INLA-style likelihood fit.", quote="")
_r("glmmgam", "glmmgam", "Spatial", "Spatial GLMM Gamma log-link simulation.", quote="")
_r("glmmgmrf", "glmmgmrf", "Spatial", "Spatial GLMM GMRF Gaussian Markov RF prior.", quote="")
_r("glmminla", "glmminla", "Spatial", "Spatial GLMM INLA posterior approximation.", quote="")
_r("glmmnb", "glmmnb", "Spatial", "Spatial GLMM negative binomial simulation.", quote="")
_r("glmmpoi", "glmmpoi", "Spatial", "Spatial GLMM Poisson log-link simulation.", quote="")
_r("glmmpred", "glmmpred", "Spatial", "Spatial GLMM posterior predictive map.", quote="")
_r("glmmre", "glmmre", "Spatial", "Spatial GLMM random effects variance comp.", quote="")
_r("glmmsar", "glmmsar", "Spatial", "Spatial GLMM SAR simultaneous autoregressive.", quote="")
_r("glmmsim", "glmmsim", "Spatial", "Spatial GLMM simulation Gaussian response.", quote="")
_r("glmmval", "glmmval", "Spatial", "Spatial GLMM predictive performance CRPS.", quote="")
_r("gnsaic", "gnsaic", "GNS", "GNS Akaike information criterion.", quote="")
_r("gnsbic", "gnsbic", "GNS", "GNS Bayesian information criterion.", quote="")
_r("gnsim", "gnsim", "SpatialPat", "Genetic algorithm simulation", quote="One does not simply walk. -- Boromir")
_r("gnsimp", "gnsimp", "GNS", "GNS direct/indirect/total impacts.", quote="")
_r("gnsjac", "gnsjac", "GNS", "GNS dual Jacobian ln|I-rho*W| + ln|I-lam*W|.", quote="")
_r("gnslrt", "gnslrt", "GNS", "GNS likelihood-ratio test vs SDM.", quote="")
_r("gnsml", "gnsml", "GNS", "GNS (general nesting spatial) ML estimation.", quote="")
_r("gnsres", "gnsres", "GNS", "GNS residual autocorrelation check.", quote="")
_r("gnssig", "gnssig", "GNS", "GNS sigma-squared ML estimate.", quote="")
_r("gnsvar", "gnsvar", "GNS", "GNS variance-covariance matrix.", quote="")
_r("gnswald", "gnswald", "GNS", "GNS Wald test on rho and lambda.", quote="")
_r("goodel", "goodel", "GeoProcss", "Goode homolosine projection", quote="One does not simply walk. -- Boromir")
_r("gpani", "gpani", "Spatial", "GP anisotropic ARD kernel.", quote="")
_r("gpclf", "gpclf", "Spatial", "GP binary classification Laplace approx.", quote="")
_r("gpcv", "gpcv", "Spatial", "GP cross-validation predictive distribution.", quote="")
_r("gpdeep", "gpdeep", "Spatial", "Deep GP composition of GP layers.", quote="")
_r("gpdtc", "gpdtc", "Spatial", "GP deterministic training conditional.", quote="")
_r("gpexp", "gpexp", "Spatial", "GP squared exponential kernel regression.", quote="")
_r("gpfit", "gpfit", "Spatial", "Gaussian process regression fit.", quote="")
_r("gphyp", "gphyp", "Spatial", "GP hyperparameter marginal likelihood opt.", quote="")
_r("gpkrn", "gpkrn", "Spatial", "GP kernel function evaluation.", quote="")
_r("gplin", "gplin", "Spatial", "GP linear kernel regression.", quote="")
_r("gpmat", "gpmat", "Spatial", "GP Matern-5/2 kernel regression.", quote="")
_r("gpnug", "gpnug", "Spatial", "GP nugget noise variance estimation.", quote="")
_r("gpper", "gpper", "Spatial", "GP periodic kernel regression.", quote="")
_r("gppost", "gppost", "Spatial", "GP posterior sample path generation.", quote="")
_r("gppred", "gppred", "Spatial", "GP prediction mean and variance.", quote="")
_r("gprq", "gprq", "Spatial", "GP rational quadratic kernel.", quote="")
_r("gpsmp", "gpsmp", "Spatial", "GP prior sample path generation.", quote="")
_r("gpsparse", "gpsparse", "Spatial", "Sparse GP inducing points approximation.", quote="")
_r("gpvfe", "gpvfe", "Spatial", "GP variational free energy bound.", quote="")
_r("gpwarp", "gpwarp", "Spatial", "GP input warping nonstationary.", quote="")
_r("grdgen", "grdgen", "GeoProcss", "Regular grid generation", quote="Walk without rhythm. -- Fremen proverb")
_r("grdsmp", "grdsmp", "GeoProcss", "Grid sampling spatial", quote="The world is cruel but beautiful. -- Mikasa")
_r("grtcrc", "grtcrc", "GeoProcss", "Great circle distance", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("grwsp", "grwsp", "SpatialPat", "Growth model spatial", quote="Walk without rhythm. -- Fremen proverb")
_r("gsmsp", "gsmsp", "KrigFilt", "Gaussian smoothing spatial", quote="One does not simply walk. -- Boromir")
_r("gtest", "gtest", "SpatialPat", "G-test nearest neighbor", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("gwraicc", "gwraicc", "GWR", "GWR corrected AIC (AICc) for model selection.", quote="")
_r("gwrbisq", "gwrbisq", "GWR", "GWR bisquare kernel weights.", quote="")
_r("gwrbw", "gwrbw", "GWR", "GWR bandwidth selection (AICc cross-validation).", quote="")
_r("gwrcoef", "gwrcoef", "GWR", "GWR local coefficient estimates.", quote="")
_r("gwrcomp", "gwrcomp", "GWR", "GWR comparison: fixed vs adaptive bandwidth.", quote="")
_r("gwrcv", "gwrcv", "GWR", "GWR leave-one-out cross-validation score.", quote="")
_r("gwrdlt", "gwrdlt", "GWR", "GWR delta test for coefficient stationarity.", quote="")
_r("gwrfit", "gwrfit", "GWR", "GWR basic model fit (Brunsdon et al. 1996).", quote="")
_r("gwrfwl", "gwrfwl", "GWR", "GWR Frisch-Waugh-Lovell local partitioned regression.", quote="")
_r("gwrgaus", "gwrgaus", "GWR", "GWR Gaussian kernel weights.", quote="")
_r("gwrhat", "gwrhat", "GWR", "GWR hat matrix diagonal (leverage).", quote="")
_r("gwrlgt", "gwrlgt", "GWR", "GWR logistic regression (binary outcome).", quote="")
_r("gwrpois", "gwrpois", "GWR", "GWR Poisson regression.", quote="")
_r("gwrr2", "gwrr2", "GWR", "GWR local R-squared.", quote="")
_r("gwrres", "gwrres", "GWR", "GWR local residuals.", quote="")
_r("gwrstd", "gwrstd", "GWR", "GWR local standard errors of coefficients.", quote="")
_r("gwrsur", "gwrsur", "GWR", "GWR seemingly-unrelated regression system.", quote="")
_r("gwrtri", "gwrtri", "GWR", "GWR tri-cube kernel weights.", quote="")
_r("gwrtst", "gwrtst", "GWR", "GWR Monte-Carlo test for spatial variability.", quote="")
_r("gwrtvl", "gwrtvl", "GWR", "GWR t-values for local coefficients.", quote="")
_r("havrsi", "havrsi", "GeoProcss", "Haversine distance formula", quote="Believe it! -- Naruto")
_r("hexgen", "hexgen", "GeoProcss", "Hexagonal grid generation", quote="There is always hope. -- Aragorn")
_r("hexsmp", "hexsmp", "GeoProcss", "Hexagonal grid sampling", quote="I alone level up. -- Sung Jin-Woo")
_r("hmcsp", "hmcsp", "SpatialPat", "Hamiltonian MC spatial", quote="Chaos is a ladder. -- Littlefinger")
_r("hsasy", "hsasy", "Spatial", "Asymmetric Hotelling model.", quote="")
_r("hscir", "hscir", "Spatial", "Hotelling circular competition.", quote="")
_r("hscon", "hscon", "Spatial", "Convergence in Hotelling model.", quote="")
_r("hsdbl", "hsdbl", "Spatial", "Double-peaked Hotelling.", quote="")
_r("hsdiv", "hsdiv", "Spatial", "Divergence in Hotelling model.", quote="")
_r("hsdwn", "hsdwn", "Spatial", "Downs electoral competition.", quote="")
_r("hsent", "hsent", "Spatial", "Entry deterrence Hotelling.", quote="")
_r("hslin", "hslin", "Spatial", "Hotelling linear competition.", quote="")
_r("hsmlt", "hsmlt", "Spatial", "Multiparty Hotelling equilibrium.", quote="")
_r("hsmpt", "hsmpt", "Spatial", "Midpoint competition equilibrium.", quote="")
_r("hsnsh", "hsnsh", "Spatial", "Nash equilibrium in Hotelling.", quote="")
_r("hsprc", "hsprc", "Spatial", "Price-location Hotelling.", quote="")
_r("hssal", "hssal", "Spatial", "Salience Hotelling model.", quote="")
_r("hstrp", "hstrp", "Spatial", "Three-party Hotelling competition.", quote="")
_r("hswas", "hswas", "Spatial", "Wasted-vote Hotelling.", quote="")
_r("hullar", "hullar", "GeoAnalysis", "Convex hull area ratio", quote="Science! -- Jesse Pinkman")
_r("hullel", "hullel", "GeoAnalysis", "Convex hull elongation", quote="Breathe. -- Tanjiro")
_r("hullfr", "hullfr", "GeoAnalysis", "Convex hull fractal dimension", quote="I alone level up. -- Sung Jin-Woo")
_r("hullrn", "hullrn", "GeoAnalysis", "Convex hull roundness index", quote="Chaos is a ladder. -- Littlefinger")
_r("hyaqu", "hyaqu", "HydroSp", "Aquifer thickness spatial", quote="A lesson without pain is meaningless. -- Edward")
_r("hybas", "hybas", "HydroSp", "Basin delineation", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("hybod", "hybod", "HydroSp", "BOD spatial mapping", quote="Growing old is a blessing. -- Rengoku")
_r("hybre", "hybre", "HydroSp", "Breach depression DEM", quote="Equivalent exchange. -- Elric brothers")
_r("hycnd", "hycnd", "HydroSp", "Conductivity spatial", quote="There is always hope. -- Aragorn")
_r("hycti", "hycti", "HydroSp", "Compound topographic index", quote="Winter is coming. -- Stark motto")
_r("hycur", "hycur", "HydroSp", "Curvature hydrological", quote="I am the one who knocks. -- Walter White")
_r("hyd8f", "hyd8f", "HydroSp", "D8 flow direction", quote="See you space cowboy. -- Spike")
_r("hydfl", "hydfl", "HydroSp", "D-infinity flow direction", quote="Go beyond! Plus Ultra! -- All Might")
_r("hydhl", "hydhl", "HydroSp", "Darcy flow spatial", quote="I mustn't run away. -- Shinji")
_r("hydis", "hydis", "HydroSp", "Discharge estimation spatial", quote="Tatakae! -- Eren")
_r("hydra", "hydra", "HydroSp", "Drainage network extraction", quote="Not all those who wander are lost. -- Gandalf")
_r("hydrf", "hydrf", "HydroSp", "Drought frequency analysis", quote="The spice must flow. -- Paul Atreides")
_r("hydsl", "hydsl", "HydroSp", "Dissolved oxygen spatial", quote="Breathe. -- Tanjiro")
_r("hyefl", "hyefl", "HydroSp", "Effluent plume spatial", quote="One does not simply walk. -- Boromir")
_r("hyevp", "hyevp", "HydroSp", "Evaporation estimation spatial", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("hyfac", "hyfac", "HydroSp", "Flow accumulation raster", quote="Make it so. -- Picard")
_r("hyfcr", "hyfcr", "HydroSp", "Flow frequency curve", quote="No half measures. -- Mike")
_r("hyfdr", "hyfdr", "HydroSp", "Flow direction raster", quote="Believe it! -- Naruto")
_r("hyflf", "hyflf", "HydroSp", "Flood frequency analysis", quote="Hold the door. -- Hodor")
_r("hyfpl", "hyfpl", "HydroSp", "Floodplain delineation", quote="Desert power. -- Paul Muad'Dib")
_r("hygrw", "hygrw", "HydroSp", "Groundwater level spatial", quote="Resistance is futile. -- Borg")
_r("hyhdc", "hyhdc", "HydroSp", "Hydraulic conductivity", quote="Whatever happens, happens. -- Spike")
_r("hyhlr", "hyhlr", "HydroSp", "Horton stream ordering", quote="Yare yare daze. -- Jotaro")
_r("hyhyd", "hyhyd", "HydroSp", "Hydrograph analysis", quote="The sleeper must awaken. -- Leto Atreides")
_r("hyidf", "hyidf", "HydroSp", "IDF curve spatial", quote="Kamehameha! -- Goku")
_r("hylen", "hylen", "HydroSp", "Stream length analysis", quote="Dedicate your hearts! -- Erwin")
_r("hymfd", "hymfd", "HydroSp", "Multiple flow direction", quote="El Psy Kongroo. -- Okabe")
_r("hymnd", "hymnd", "HydroSp", "Meander analysis", quote="Chaos is a ladder. -- Littlefinger")
_r("hynit", "hynit", "HydroSp", "Nitrogen loading spatial", quote="Walk without rhythm. -- Fremen proverb")
_r("hynut", "hynut", "HydroSp", "Nutrient loading spatial", quote="Power is everything. -- Sung Jin-Woo")
_r("hyord", "hyord", "HydroSp", "Stream ordering Strahler", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("hyphs", "hyphs", "HydroSp", "Phosphorus loading spatial", quote="A Lannister always pays his debts. -- Tyrion")
_r("hyphy", "hyphy", "HydroSp", "pH spatial mapping", quote="I alone level up. -- Sung Jin-Woo")
_r("hypor", "hypor", "HydroSp", "Porosity spatial", quote="I am here! -- All Might")
_r("hyprm", "hyprm", "HydroSp", "Permeability spatial", quote="You should enjoy the detours. -- Ging")
_r("hyrcg", "hyrcg", "HydroSp", "Recharge rate spatial", quote="Total Concentration Breathing. -- Tanjiro")
_r("hyrch", "hyrch", "HydroSp", "River channel extraction", quote="The world is cruel but beautiful. -- Mikasa")
_r("hyret", "hyret", "HydroSp", "Return period estimation", quote="My precious. -- Gollum")
_r("hyrpn", "hyrpn", "HydroSp", "Riparian zone delineation", quote="Live long and prosper. -- Spock")
_r("hyshr", "hyshr", "HydroSp", "Shreve stream ordering", quote="It's over 9000! -- Vegeta")
_r("hyslo", "hyslo", "HydroSp", "Stream slope analysis", quote="I am justice! -- Light")
_r("hysnk", "hysnk", "HydroSp", "Sink filling DEM", quote="Bankai! -- Ichigo")
_r("hyspi", "hyspi", "HydroSp", "Stream power index", quote="Arise. -- Shadow Monarch")
_r("hytrb", "hytrb", "HydroSp", "Turbidity spatial", quote="Science! -- Jesse Pinkman")
_r("hytss", "hytss", "HydroSp", "Total suspended solids", quote="Engage. -- Picard")
_r("hytwi", "hytwi", "HydroSp", "Topographic wetness index", quote="Set your heart ablaze! -- Rengoku")
_r("hyuaa", "hyuaa", "HydroSp", "Upslope contributing area", quote="Get in the robot, Shinji! -- Misato")
_r("hyunt", "hyunt", "HydroSp", "Unit hydrograph spatial", quote="Valar Morghulis. -- Braavos")
_r("hywte", "hywte", "HydroSp", "Water table elevation", quote="Keep moving forward. -- Eren")
_r("hywtq", "hywtq", "HydroSp", "Water quality index spatial", quote="This is Requiem. -- Giorno")
_r("idpam", "idpam", "Spatial", "Amendment voting ideal point.", quote="")
_r("idpas", "idpas", "Spatial", "Agenda-setter ideal point.", quote="")
_r("idpbp", "idpbp", "Spatial", "Bayesian ideal point estimation.", quote="")
_r("idpcm", "idpcm", "Spatial", "Committee ideal point estimation.", quote="")
_r("idpco", "idpco", "Spatial", "Cosponsorship ideal point.", quote="")
_r("idpdw", "idpdw", "Spatial", "Dynamic ideal point (DW-NOMINATE).", quote="")
_r("idpem", "idpem", "Spatial", "EM algorithm ideal point.", quote="")
_r("idpfa", "idpfa", "Spatial", "Factor analysis ideal point.", quote="")
_r("idpfl", "idpfl", "Spatial", "Floor median ideal point.", quote="")
_r("idpir", "idpir", "Spatial", "IRT-based ideal point.", quote="")
_r("idpmc", "idpmc", "Spatial", "MCMC ideal point estimation.", quote="")
_r("idpmk", "idpmk", "Spatial", "Markov chain ideal point.", quote="")
_r("idpnm", "idpnm", "Spatial", "Nominal ideal point estimation.", quote="")
_r("idpol", "idpol", "Spatial", "OLS ideal point regression.", quote="")
_r("idppc", "idppc", "Spatial", "PCA-based ideal point.", quote="")
_r("idppt", "idppt", "Spatial", "Pivot ideal point.", quote="")
_r("idpts", "idpts", "Spatial", "Text scaling ideal point.", quote="")
_r("idpvt", "idpvt", "Spatial", "Vote trading ideal point shift.", quote="")
_r("idpw1", "idpw1", "Spatial", "W-NOMINATE 1D ideal point.", quote="")
_r("idpw2", "idpw2", "Spatial", "W-NOMINATE 2D ideal point.", quote="")
_r("idwani", "idwani", "Spatial", "Anisotropic IDW with directional weights.", quote="")
_r("idwbar", "idwbar", "Spatial", "Barrier-constrained IDW interpolation.", quote="")
_r("idwblk", "idwblk", "Spatial", "Block IDW for irregular grids.", quote="")
_r("idwbst", "idwbst", "Spatial", "IDW boosted ensemble interpolation.", quote="")
_r("idwcv", "idwcv", "Spatial", "IDW cross-validation leave-one-out.", quote="")
_r("idwfl", "idwfl", "KrigFilt", "IDW filter interpolation", quote="Arise. -- Shadow Monarch")
_r("idwgrd", "idwgrd", "Spatial", "IDW grid prediction surface.", quote="")
_r("idwknn", "idwknn", "Spatial", "k-nearest neighbor IDW interpolation.", quote="")
_r("idwlcl", "idwlcl", "Spatial", "Local IDW with adaptive bandwidth.", quote="")
_r("idwmod", "idwmod", "Spatial", "Modified IDW with bandwidth smoothing.", quote="")
_r("idwmsh", "idwmsh", "Spatial", "Modified Shepard quadratic regression IDW.", quote="")
_r("idwopt", "idwopt", "Spatial", "Optimal IDW power via LOOCV.", quote="")
_r("idwp", "idwp", "Spatial", "IDW power-parameter sensitivity analysis.", quote="")
_r("idwrad", "idwrad", "Spatial", "Radial search IDW fixed radius.", quote="")
_r("idwshp", "idwshp", "Spatial", "Shepard IDW interpolation classic.", quote="")
_r("idwvar", "idwvar", "Spatial", "IDW prediction uncertainty estimate.", quote="")
_r("igraic", "igraic", "Gravity", "Gravity model AIC.", quote="")
_r("igrav", "igrav", "Gravity", "Gravity model OLS estimation.", quote="")
_r("igravbl", "igravbl", "Gravity", "Gravity model row/column balancing (IPFP).", quote="")
_r("igravcl", "igravcl", "Gravity", "Gravity model calibration (beta estimation).", quote="")
_r("igraver", "igraver", "Gravity", "Gravity model RMSE.", quote="")
_r("igravfe", "igravfe", "Gravity", "Gravity model with origin/destination fixed effects.", quote="")
_r("igravlm", "igravlm", "Gravity", "Gravity model LM test for spatial autocorrelation.", quote="")
_r("igravnb", "igravnb", "Gravity", "Gravity model negative-binomial PPML.", quote="")
_r("igravps", "igravps", "Gravity", "Gravity model Poisson PPML estimation.", quote="")
_r("igravpt", "igravpt", "Gravity", "Spatial potential model.", quote="")
_r("igravrt", "igravrt", "Gravity", "Gravity retail potential (Huff model).", quote="")
_r("igravsq", "igravsq", "Gravity", "Square gravity (symmetric OD matrix).", quote="")
_r("igravvf", "igravvf", "Gravity", "Gravity model variance function.", quote="")
_r("igravwl", "igravwl", "Gravity", "Gravity Wilson entropy model.", quote="")
_r("igrbic", "igrbic", "Gravity", "Gravity model BIC.", quote="")
_r("ikflt", "ikflt", "KrigFilt", "Indicator kriging filter", quote="Keep moving forward. -- Eren")
_r("irrgen", "irrgen", "GeoProcss", "Irregular grid generation", quote="Those who break the rules are scum. -- Kakashi")
_r("isagd", "isagd", "Spatial", "Issue agenda spatial.", quote="")
_r("isatt", "isatt", "Spatial", "Issue attention model.", quote="")
_r("iscat", "iscat", "Spatial", "Issue category salience.", quote="")
_r("iscrs", "iscrs", "Spatial", "Cross-issue salience.", quote="")
_r("isdyn", "isdyn", "Spatial", "Dynamic salience model.", quote="")
_r("isfrm", "isfrm", "Spatial", "Issue framing model.", quote="")
_r("isidx", "isidx", "Spatial", "Salience index composite.", quote="")
_r("ismul", "ismul", "Spatial", "Multi-issue salience.", quote="")
_r("isngm", "isngm", "SpatialPat", "Ising model spatial", quote="Power is everything. -- Sung Jin-Woo")
_r("isown", "isown", "Spatial", "Issue ownership model.", quote="")
_r("ispas", "ispas", "Spatial", "Partisan salience model.", quote="")
_r("ispol", "ispol", "Spatial", "Policy salience model.", quote="")
_r("ispri", "ispri", "Spatial", "Issue priority weighting.", quote="")
_r("issal", "issal", "Spatial", "Issue salience weight.", quote="")
_r("isval", "isval", "Spatial", "Valence advantage model.", quote="")
_r("isvlt", "isvlt", "Spatial", "Valence-salience joint model.", quote="")
_r("jtest", "jtest", "SpatialPat", "J-test spatial clustering", quote="Not all those who wander are lost. -- Gandalf")
_r("kdenf", "kdenf", "GeoAnalysis", "KDE nearest facility distance", quote="Winter is coming. -- Stark motto")
_r("kgbkd", "kgbkd", "Spatial", "Block kriging discretization", quote="")
_r("kgbkv", "kgbkv", "Spatial", "Block kriging variance", quote="")
_r("kgblk", "kgblk", "Spatial", "Block kriging prediction", quote="")
_r("kgckv", "kgckv", "Spatial", "Co-kriging variance", quote="")
_r("kgckw", "kgckw", "Spatial", "Co-kriging weights", quote="")
_r("kgcok", "kgcok", "Spatial", "Co-kriging multivariate", quote="")
_r("kgcvk", "kgcvk", "Spatial", "Kriging k-fold cross-validation", quote="")
_r("kgcvl", "kgcvl", "Spatial", "Kriging LOO cross-validation", quote="")
_r("kgdsh", "kgdsh", "Spatial", "Disjunctive kriging Hermite polynomials", quote="")
_r("kgdsj", "kgdsj", "Spatial", "Disjunctive kriging", quote="")
_r("kgeqv", "kgeqv", "Spatial", "Kriging equivalence to GLS", quote="")
_r("kgfct", "kgfct", "Spatial", "Factorial kriging", quote="")
_r("kgikp", "kgikp", "Spatial", "Indicator kriging probability", quote="")
_r("kgind", "kgind", "Spatial", "Indicator kriging", quote="")
_r("kglam", "kglam", "Spatial", "Kriging weights (lambda)", quote="")
_r("kglgb", "kglgb", "Spatial", "Lognormal kriging back-transform", quote="")
_r("kglgn", "kglgn", "Spatial", "Lognormal kriging", quote="")
_r("kgmae", "kgmae", "Spatial", "Kriging MAE", quote="")
_r("kgmsp", "kgmsp", "Spatial", "Kriging MSPE", quote="")
_r("kgmtx", "kgmtx", "Spatial", "Kriging system matrix", quote="")
_r("kgord", "kgord", "Spatial", "Ordinary kriging prediction", quote="")
_r("kgorm", "kgorm", "Spatial", "Ordinary kriging matrix system", quote="")
_r("kgorv", "kgorv", "Spatial", "Ordinary kriging variance", quote="")
_r("kgorw", "kgorw", "Spatial", "Ordinary kriging weights", quote="")
_r("kgprb", "kgprb", "Spatial", "Kriging probability map", quote="")
_r("kgprd", "kgprd", "Spatial", "Kriging prediction surface", quote="")
_r("kgqnt", "kgqnt", "Spatial", "Kriging quantile prediction", quote="")
_r("kgres", "kgres", "Spatial", "Kriging residual map", quote="")
_r("kgrhs", "kgrhs", "Spatial", "Kriging right-hand side vector", quote="")
_r("kgrms", "kgrms", "Spatial", "Kriging RMSE", quote="")
_r("kgsmp", "kgsmp", "Spatial", "Simple kriging prediction", quote="")
_r("kgsmv", "kgsmv", "Spatial", "Simple kriging variance", quote="")
_r("kgsmw", "kgsmw", "Spatial", "Simple kriging weights", quote="")
_r("kgstd", "kgstd", "Spatial", "Kriging standard error map", quote="")
_r("kgtrn", "kgtrn", "Spatial", "Kriging trend surface", quote="")
_r("kguni", "kguni", "Spatial", "Universal kriging prediction", quote="")
_r("kgunr", "kgunr", "Spatial", "Universal kriging residual", quote="")
_r("kgunt", "kgunt", "Spatial", "Universal kriging trend", quote="")
_r("kgunv", "kgunv", "Spatial", "Universal kriging variance", quote="")
_r("kgunw", "kgunw", "Spatial", "Universal kriging weights", quote="")
_r("krgcv", "krgcv", "KrigFilt", "Kriging cross-validation", quote="It's over 9000! -- Vegeta")
_r("krgdg", "krgdg", "KrigFilt", "Kriging diagnostics", quote="Yare yare daze. -- Jotaro")
_r("krgef", "krgef", "KrigFilt", "Kriging efficiency", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("krgnb", "krgnb", "KrigFilt", "Kriging neighborhood selection", quote="I am justice! -- Light")
_r("krgre", "krgre", "KrigFilt", "Kriging residual analysis", quote="Equivalent exchange. -- Elric brothers")
_r("krgsd", "krgsd", "KrigFilt", "Kriging standard deviation", quote="Believe it! -- Naruto")
_r("krgsm", "krgsm", "KrigFilt", "Kriging search method", quote="Bankai! -- Ichigo")
_r("krgtr", "krgtr", "KrigFilt", "Kriging trend removal", quote="Go beyond! Plus Ultra! -- All Might")
_r("krgwt", "krgwt", "KrigFilt", "Kriging weight map", quote="Dedicate your hearts! -- Erwin")
_r("ksmsp", "ksmsp", "KrigFilt", "Kernel smoothing spatial", quote="Live long and prosper. -- Spock")
_r("ktest", "ktest", "SpatialPat", "K-function (Ripley) test", quote="Make it so. -- Picard")
_r("lacbivl", "lacbivl", "Lattice", "Bivariate LISA (Lee 2001).", quote="")
_r("lacgear", "lacgear", "Lattice", "Geary's C statistic.", quote="")
_r("lacgetg", "lacgetg", "Lattice", "Getis-Ord G global statistic.", quote="")
_r("lacgetl", "lacgetl", "Lattice", "Getis-Ord G* local statistic.", quote="")
_r("lacgmc", "lacgmc", "Lattice", "Geary's C Monte-Carlo test.", quote="")
_r("lacjmc", "lacjmc", "Lattice", "Join count Monte-Carlo test.", quote="")
_r("lacjoin", "lacjoin", "Lattice", "Join count test (BB, BW, WW).", quote="")
_r("laclihh", "laclihh", "Lattice", "LISA high-high cluster identification.", quote="")
_r("laclihl", "laclihl", "Lattice", "LISA high-low outlier identification.", quote="")
_r("laclimc", "laclimc", "Lattice", "LISA Monte-Carlo significance.", quote="")
_r("laclisa", "laclisa", "Lattice", "LISA (local indicators spatial association).", quote="")
_r("lacscat", "lacscat", "Lattice", "Moran scatterplot quadrant classification.", quote="")
_r("lacscor", "lacscor", "Lattice", "Spatial correlation coefficient.", quote="")
_r("lactest", "lactest", "Lattice", "General lattice spatial test battery.", quote="")
_r("lacvgm", "lacvgm", "Lattice", "Variogram for lattice data.", quote="")
_r("lambrt", "lambrt", "GeoProcss", "Lambert conformal conic", quote="See you space cowboy. -- Spike")
_r("lapfp", "lapfp", "KrigFilt", "Laplacian filter spatial", quote="The needs of the many outweigh the few. -- Spock")
_r("lkflt", "lkflt", "KrigFilt", "Lognormal kriging filter", quote="One is all, all is one. -- Izumi")
_r("lmbp", "lmbp", "LM", "LM Breusch-Pagan test (general heteroskedasticity).", quote="")
_r("lmdiag", "lmdiag", "LM", "LM diagnostics summary (lag, error, robust).", quote="")
_r("lmerr", "lmerr", "LM", "LM test for spatial error (Anselin 1988).", quote="")
_r("lmhe", "lmhe", "LM", "LM test for spatial heteroskedasticity.", quote="")
_r("lmjoint", "lmjoint", "LM", "Joint LM test for spatial lag and error.", quote="")
_r("lmkoenk", "lmkoenk", "LM", "Koenker-Bassett heteroskedasticity test.", quote="")
_r("lmkp", "lmkp", "LM", "Kelejian-Prucha LM test for SAC model.", quote="")
_r("lmlag", "lmlag", "LM", "LM test for spatial lag (Anselin 1988).", quote="")
_r("lmrerr", "lmrerr", "LM", "Robust LM test for spatial error.", quote="")
_r("lmrerr2", "lmrerr2", "LM", "One-directional robust LM error (Bera-Yoon).", quote="")
_r("lmrlag", "lmrlag", "LM", "Robust LM test for spatial lag.", quote="")
_r("lmrlag2", "lmrlag2", "LM", "One-directional robust LM lag (Bera-Yoon).", quote="")
_r("lmsarma", "lmsarma", "LM", "LM test for SARMA (lag + error).", quote="")
_r("lmsdm", "lmsdm", "LM", "LM test for spatial Durbin model.", quote="")
_r("lmslx", "lmslx", "LM", "LM test for SLX (spatially lagged X).", quote="")
_r("lonlat", "lonlat", "GeoProcss", "Lon/lat to projected coordinates", quote="Dedicate your hearts! -- Erwin")
_r("lowsp", "lowsp", "KrigFilt", "LOWESS spatial smoothing", quote="Get in the robot, Shinji! -- Misato")
_r("ltest", "ltest", "SpatialPat", "L-function (linearized K)", quote="Believe it! -- Naruto")
_r("luband", "luband", "Spatial", "LU banded covariance simulation.", quote="")
_r("lucnd", "lucnd", "Spatial", "LU conditional simulation update step.", quote="")
_r("luinc", "luinc", "Spatial", "LU incomplete factorization approximation.", quote="")
_r("lumlt", "lumlt", "Spatial", "LU multi-field simultaneous simulation.", quote="")
_r("lupiv", "lupiv", "Spatial", "LU pivoting for numerical stability.", quote="")
_r("luprec", "luprec", "Spatial", "LU preconditioned iterative refinement.", quote="")
_r("lurng", "lurng", "Spatial", "LU simulation range-dependent correlation.", quote="")
_r("lusim", "lusim", "Spatial", "LU decomposition Gaussian field simulation.", quote="")
_r("lusim2", "lusim2", "Spatial", "LU simulation with sparse covariance matrix.", quote="")
_r("luvar", "luvar", "Spatial", "LU simulation variance scaling.", quote="")
_r("lvsim", "lvsim", "SpatialPat", "Levy flight simulation", quote="A lesson without pain is meaningless. -- Edward")
_r("lvsmp", "lvsmp", "GeoProcss", "Leave-one-out spatial CV", quote="I am here! -- All Might")
_r("maacf", "maacf", "MarinSp", "Acidification risk spatial", quote="Whatever happens, happens. -- Spike")
_r("maacst", "maacst", "MarinSp", "Acoustic survey spatial", quote="No half measures. -- Mike")
_r("maalk", "maalk", "MarinSp", "Alkalinity marine spatial", quote="Arise. -- Shadow Monarch")
_r("maben", "maben", "MarinSp", "Benthic habitat spatial", quote="A Lannister always pays his debts. -- Tyrion")
_r("mablc", "mablc", "MarinSp", "Bleaching risk spatial", quote="You should enjoy the detours. -- Ging")
_r("mabuoy", "mabuoy", "MarinSp", "Buoy network spatial", quote="Make it so. -- Picard")
_r("machla", "machla", "MarinSp", "Chlorophyll-a marine", quote="I alone level up. -- Sung Jin-Woo")
_r("maco2", "maco2", "MarinSp", "CO2 flux marine", quote="Winter is coming. -- Stark motto")
_r("macrl", "macrl", "MarinSp", "Coral reef mapping", quote="Walk without rhythm. -- Fremen proverb")
_r("macur", "macur", "MarinSp", "Current velocity spatial", quote="Dedicate your hearts! -- Erwin")
_r("madic", "madic", "MarinSp", "DIC marine spatial", quote="I am the one who knocks. -- Walter White")
_r("madoc", "madoc", "MarinSp", "DOC marine spatial", quote="Get in the robot, Shinji! -- Misato")
_r("madox", "madox", "MarinSp", "Dissolved oxygen marine", quote="El Psy Kongroo. -- Okabe")
_r("madpt", "madpt", "MarinSp", "Depth interpolation bathymetry", quote="Believe it! -- Naruto")
_r("maerb", "maerb", "MarinSp", "Erosion beach spatial", quote="I mustn't run away. -- Shinji")
_r("mafeo", "mafeo", "MarinSp", "Iron marine spatial", quote="Breathe. -- Tanjiro")
_r("mafsh", "mafsh", "MarinSp", "Fish abundance spatial", quote="The sleeper must awaken. -- Leto Atreides")
_r("magldr", "magldr", "MarinSp", "Glider survey spatial", quote="Not all those who wander are lost. -- Gandalf")
_r("mahcl", "mahcl", "MarinSp", "Halocline depth spatial", quote="See you space cowboy. -- Spike")
_r("maklp", "maklp", "MarinSp", "Kelp forest mapping", quote="Resistance is futile. -- Borg")
_r("mamar", "mamar", "MarinSp", "Marine protected area", quote="Tatakae! -- Eren")
_r("mamng", "mamng", "MarinSp", "Mangrove mapping marine", quote="Growing old is a blessing. -- Rengoku")
_r("manh4", "manh4", "MarinSp", "Ammonium marine spatial", quote="Science! -- Jesse Pinkman")
_r("manit", "manit", "MarinSp", "Nitrate marine spatial", quote="Chaos is a ladder. -- Littlefinger")
_r("manut", "manut", "MarinSp", "Nutrient marine spatial", quote="Live long and prosper. -- Spock")
_r("maosp", "maosp", "MarinSp", "Oil spill trajectory", quote="This is Requiem. -- Giorno")
_r("maoys", "maoys", "MarinSp", "Oyster reef mapping", quote="A lesson without pain is meaningless. -- Edward")
_r("maph", "maph", "MarinSp", "pH marine spatial", quote="Set your heart ablaze! -- Rengoku")
_r("mapho", "mapho", "MarinSp", "Phosphate marine spatial", quote="The world is cruel but beautiful. -- Mikasa")
_r("maphy", "maphy", "MarinSp", "Phytoplankton spatial", quote="Power is everything. -- Sung Jin-Woo")
_r("maphz", "maphz", "MarinSp", "Photic zone depth", quote="Equivalent exchange. -- Elric brothers")
_r("maplm", "maplm", "MarinSp", "Pollutant plume marine", quote="Total Concentration Breathing. -- Tanjiro")
_r("maprd", "maprd", "MarinSp", "Primary production marine", quote="There is always hope. -- Aragorn")
_r("maradr", "maradr", "MarinSp", "Radar altimetry spatial", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("maref", "maref", "MarinSp", "Reef health index spatial", quote="I am here! -- All Might")
_r("markg", "markg", "SpatialPat", "Mark correlation function", quote="It's over 9000! -- Vegeta")
_r("markv", "markv", "SpatialPat", "Mark variogram", quote="Yare yare daze. -- Jotaro")
_r("marov", "marov", "MarinSp", "ROV survey spatial", quote="Hold the door. -- Hodor")
_r("masat", "masat", "MarinSp", "Satellite ocean color", quote="The spice must flow. -- Paul Atreides")
_r("masctc", "masctc", "MarinSp", "SCUBA transect spatial", quote="My precious. -- Gollum")
_r("mased", "mased", "MarinSp", "Sediment transport marine", quote="Keep moving forward. -- Eren")
_r("masgr", "masgr", "MarinSp", "Seagrass mapping", quote="One does not simply walk. -- Boromir")
_r("masis", "masis", "MarinSp", "Silicate marine spatial", quote="Desert power. -- Paul Muad'Dib")
_r("maspc", "maspc", "MarinSp", "Species distribution marine", quote="Valar Morghulis. -- Braavos")
_r("massc", "massc", "MarinSp", "Sea surface chlorophyll", quote="Yare yare daze. -- Jotaro")
_r("masss", "masss", "MarinSp", "Sea surface salinity", quote="It's over 9000! -- Vegeta")
_r("masst", "masst", "MarinSp", "Sea surface temperature", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("matcl", "matcl", "MarinSp", "Thermocline depth spatial", quote="Go beyond! Plus Ultra! -- All Might")
_r("matid", "matid", "MarinSp", "Tidal range spatial", quote="Bankai! -- Ichigo")
_r("matrwl", "matrwl", "MarinSp", "Trawl survey spatial", quote="Kamehameha! -- Goku")
_r("mawav", "mawav", "MarinSp", "Wave height spatial", quote="I am justice! -- Light")
_r("mazop", "mazop", "MarinSp", "Zooplankton spatial", quote="Engage. -- Picard")
_r("mcanis", "mcanis", "Spatial", "Anisotropic MC spatial variance reduction.", quote="")
_r("mcconv", "mcconv", "Spatial", "Monte Carlo convergence diagnostics.", quote="")
_r("mcimp", "mcimp", "Spatial", "Importance sampling spatial MC.", quote="")
_r("mcint", "mcint", "Spatial", "Monte Carlo spatial integration estimate.", quote="")
_r("mcqmc", "mcqmc", "Spatial", "Quasi-Monte Carlo low-discrepancy integration.", quote="")
_r("mcsbs", "mcsbs", "Spatial", "Monte Carlo sample-based simulation.", quote="")
_r("mcsrs", "mcsrs", "Spatial", "Monte Carlo simple random sampling estimate.", quote="")
_r("mcstrat", "mcstrat", "Spatial", "Monte Carlo stratified spatial sampling.", quote="")
_r("mcvar", "mcvar", "Spatial", "Monte Carlo variance reduction techniques.", quote="")
_r("mczone", "mczone", "Spatial", "Monte Carlo zonal spatial estimation.", quote="")
_r("md2dp", "md2dp", "Spatial", "2D spatial position equilibrium.", quote="")
_r("md3dp", "md3dp", "Spatial", "3D spatial position model.", quote="")
_r("mdang", "mdang", "Spatial", "Angle-based multidimensional.", quote="")
_r("mdchm", "mdchm", "Spatial", "Chaos theorem multidim.", quote="")
_r("mdcov", "mdcov", "Spatial", "Covered set multidimensional.", quote="")
_r("mdctr", "mdctr", "Spatial", "Centroid multidimensional voting.", quote="")
_r("mdcwd", "mdcwd", "Spatial", "Contract curve multidimensional.", quote="")
_r("mdcyc", "mdcyc", "Spatial", "Cycling multidimensional voting.", quote="")
_r("mddim", "mddim", "Spatial", "Dimensionality reduction spatial.", quote="")
_r("mdnsp", "mdnsp", "Spatial", "Non-separable preferences multidim.", quote="")
_r("mdprt", "mdprt", "Spatial", "Pareto frontier multidimensional.", quote="")
_r("mdrad", "mdrad", "Spatial", "Yolk radius multidimensional.", quote="")
_r("mdrcd", "mdrcd", "Spatial", "Reversion point multidim.", quote="")
_r("mdrot", "mdrot", "Spatial", "Rotation invariance multidim.", quote="")
_r("mdsep", "mdsep", "Spatial", "Separable preferences multidim.", quote="")
_r("mdsta", "mdsta", "Spatial", "Stability region multidim.", quote="")
_r("mdstr", "mdstr", "Spatial", "Structure-induced equilibrium.", quote="")
_r("mdunc", "mdunc", "Spatial", "Uncovered set multidimensional.", quote="")
_r("mdwsl", "mdwsl", "Spatial", "Win-set location multidimensional.", quote="")
_r("mdyld", "mdyld", "Spatial", "Yolk multidimensional voting.", quote="")
_r("medsp", "medsp", "KrigFilt", "Median filter spatial", quote="I must not fear. -- Litany Against Fear")
_r("merctr", "merctr", "GeoProcss", "Mercator projection", quote="Go beyond! Plus Ultra! -- All Might")
_r("mgwraic", "mgwraic", "MGWR", "MGWR AICc for model selection.", quote="")
_r("mgwrbk", "mgwrbk", "MGWR", "MGWR backfitting algorithm iteration.", quote="")
_r("mgwrbnd", "mgwrbnd", "MGWR", "MGWR bandwidth confidence interval.", quote="")
_r("mgwrbw", "mgwrbw", "MGWR", "MGWR per-variable bandwidth selection.", quote="")
_r("mgwrcof", "mgwrcof", "MGWR", "MGWR local coefficient estimates.", quote="")
_r("mgwrcv", "mgwrcv", "MGWR", "MGWR cross-validation score.", quote="")
_r("mgwrdg", "mgwrdg", "MGWR", "MGWR diagnostic summary.", quote="")
_r("mgwrfit", "mgwrfit", "MGWR", "MGWR model fit (Fotheringham et al. 2017).", quote="")
_r("mgwrhat", "mgwrhat", "MGWR", "MGWR hat matrix diagonal.", quote="")
_r("mgwrr2", "mgwrr2", "MGWR", "MGWR local R-squared.", quote="")
_r("mgwrres", "mgwrres", "MGWR", "MGWR local residuals.", quote="")
_r("mgwrsig", "mgwrsig", "MGWR", "MGWR sigma-squared estimate.", quote="")
_r("mgwrstd", "mgwrstd", "MGWR", "MGWR local standard errors.", quote="")
_r("mgwrtst", "mgwrtst", "MGWR", "MGWR Monte-Carlo stationarity test.", quote="")
_r("mgwrtvl", "mgwrtvl", "MGWR", "MGWR t-values for local coefficients.", quote="")
_r("mhsim", "mhsim", "SpatialPat", "Metropolis-Hastings spatial sim", quote="Resistance is futile. -- Borg")
_r("miexp", "miexp", "Moran", "Moran's I expected value E[I].", quote="")
_r("miiv", "miiv", "Moran", "Moran's I test on IV residuals.", quote="")
_r("mimc", "mimc", "Moran", "Moran's I Monte-Carlo permutation test.", quote="")
_r("miml", "miml", "Moran", "Moran's I test on ML residuals.", quote="")
_r("minorm", "minorm", "Moran", "Moran's I normal approximation p-value.", quote="")
_r("miols", "miols", "Moran", "Moran's I test on OLS residuals.", quote="")
_r("miorig", "miorig", "Moran", "Moran's I (original Moran 1950).", quote="")
_r("mirand", "mirand", "Moran", "Moran's I randomisation test.", quote="")
_r("mivar", "mivar", "Moran", "Moran's I variance Var[I] under normality.", quote="")
_r("mizval", "mizval", "Moran", "Moran's I standardised z-score.", quote="")
_r("mkflt", "mkflt", "KrigFilt", "Median indicator kriging", quote="The spice must flow. -- Paul Atreides")
_r("mls", "mls", "KrigFilt", "Moving least squares", quote="I am the one who knocks. -- Walter White")
_r("mnfsp", "mnfsp", "KrigFilt", "Min filter spatial", quote="The world is cruel but beautiful. -- Mikasa")
_r("mollwd", "mollwd", "GeoProcss", "Mollweide projection", quote="I am the one who knocks. -- Walter White")
_r("mpclp", "mpclp", "Spatial", "Coalition potential multi-party.", quote="")
_r("mpcmp", "mpcmp", "Spatial", "Multi-party competition index.", quote="")
_r("mpcnr", "mpcnr", "Spatial", "Multi-party Nash-Rubinstein.", quote="")
_r("mpctr", "mpctr", "Spatial", "Centrist competition model.", quote="")
_r("mpdom", "mpdom", "Spatial", "Dominant party model.", quote="")
_r("mpens", "mpens", "Spatial", "Effective number of parties.", quote="")
_r("mpflk", "mpflk", "Spatial", "Flanking party model.", quote="")
_r("mpfrg", "mpfrg", "Spatial", "Party fragmentation index.", quote="")
_r("mpgov", "mpgov", "Spatial", "Government formation spatial.", quote="")
_r("mpmnt", "mpmnt", "Spatial", "Party movement spatial.", quote="")
_r("mpmrg", "mpmrg", "Spatial", "Party merger spatial model.", quote="")
_r("mpnch", "mpnch", "Spatial", "Niche party model.", quote="")
_r("mppos", "mppos", "Spatial", "Position-competition model.", quote="")
_r("mpsim", "mpsim", "SpatialPat", "Multiple-point simulation", quote="Set your heart ablaze! -- Rengoku")
_r("mpspl", "mpspl", "Spatial", "Party split model.", quote="")
_r("mpstr", "mpstr", "Spatial", "Strategic entry multi-party.", quote="")
_r("msalc", "msalc", "Spatial", "Alienation coefficient", quote="")
_r("msans", "msans", "Spatial", "MDS anisotropy measure", quote="")
_r("msbt2", "msbt2", "Spatial", "MDS bootstrap confidence", quote="")
_r("mscls", "mscls", "Spatial", "Classical MDS (Torgerson)", quote="")
_r("mscnt", "mscnt", "Spatial", "Continuity metric", quote="")
_r("mscrd", "mscrd", "Spatial", "MDS coordinate extraction", quote="")
_r("mscvx", "mscvx", "Spatial", "2D convex hull", quote="")
_r("msdbc", "msdbc", "Spatial", "Double centering matrix B", quote="")
_r("msdch", "msdch", "Spatial", "Chebyshev distance matrix", quote="")
_r("msdcr", "msdcr", "Spatial", "Correlation distance matrix", quote="")
_r("msdcs", "msdcs", "Spatial", "Cosine distance matrix", quote="")
_r("msdel", "msdel", "Spatial", "2D Delaunay triangulation", quote="")
_r("msdim", "msdim", "Spatial", "MDS dimensionality selection", quote="")
_r("msdmc", "msdmc", "Spatial", "Mahalanobis distance matrix", quote="")
_r("msdmh", "msdmh", "Spatial", "Manhattan distance matrix", quote="")
_r("msdmn", "msdmn", "Spatial", "Minkowski distance matrix", quote="")
_r("msdst", "msdst", "Spatial", "Distance matrix computation", quote="")
_r("mseig", "mseig", "Spatial", "MDS eigendecomposition", quote="")
_r("mselb", "mselb", "Spatial", "MDS elbow detection", quote="")
_r("msemb", "msemb", "Spatial", "Embedding quality measure", quote="")
_r("msflp", "msflp", "Spatial", "MDS configuration flip check", quote="")
_r("msgof", "msgof", "Spatial", "MDS goodness of fit", quote="")
_r("msin2", "msin2", "Spatial", "INDSCAL subject weights", quote="")
_r("msind", "msind", "Spatial", "INDSCAL individual differences MDS", quote="")
_r("msink", "msink", "Spatial", "Three-way INDSCAL", quote="")
_r("msiso", "msiso", "Spatial", "Isotonic regression for MDS", quote="")
_r("msitr", "msitr", "Spatial", "MDS iteration convergence", quote="")
_r("msjck", "msjck", "Spatial", "MDS jackknife stability", quote="")
_r("mslcl", "mslcl", "Spatial", "Local continuity meta-criterion", quote="")
_r("msnm2", "msnm2", "Spatial", "Nonmetric MDS 2D", quote="")
_r("msnmt", "msnmt", "Spatial", "Nonmetric MDS (Kruskal)", quote="")
_r("msplt", "msplt", "Spatial", "MDS polarity detection", quote="")
_r("msprb", "msprb", "Spatial", "Oblique Procrustes rotation", quote="")
_r("msprc", "msprc", "Spatial", "Procrustes correlation", quote="")
_r("msprg", "msprg", "Spatial", "Generalized Procrustes analysis", quote="")
_r("mspro", "mspro", "Spatial", "Orthogonal Procrustes rotation", quote="")
_r("msprp", "msprp", "Spatial", "Partial Procrustes", quote="")
_r("msprr", "msprr", "Spatial", "Procrustes residuals", quote="")
_r("msref", "msref", "Spatial", "Reflection of configuration", quote="")
_r("msrot", "msrot", "Spatial", "Rotation of configuration", quote="")
_r("msrsq", "msrsq", "Spatial", "MDS R-squared goodness of fit", quote="")
_r("msrst", "msrst", "Spatial", "R-stress measure", quote="")
_r("msscr", "msscr", "Spatial", "MDS scree plot values", quote="")
_r("msshd", "msshd", "Spatial", "Shepard disparities", quote="")
_r("msshp", "msshp", "Spatial", "Shepard diagram values", quote="")
_r("msshr", "msshr", "Spatial", "Shepard residuals", quote="")
_r("mssm2", "mssm2", "Spatial", "SMACOF 2D MDS", quote="")
_r("mssmc", "mssmc", "Spatial", "SMACOF iterative MDS", quote="")
_r("mssmi", "mssmi", "Spatial", "Individual differences SMACOF", quote="")
_r("mssmm", "mssmm", "Spatial", "SMACOF with missing data", quote="")
_r("mssmr", "mssmr", "Spatial", "Replicated SMACOF", quote="")
_r("mssmw", "mssmw", "Spatial", "Weighted SMACOF MDS", quote="")
_r("msst1", "msst1", "Spatial", "Raw stress (Kruskal stress-1)", quote="")
_r("msst2", "msst2", "Spatial", "Normalized stress", quote="")
_r("msstb", "msstb", "Spatial", "Kruskal stress S2", quote="")
_r("mssts", "mssts", "Spatial", "Kruskal stress S1", quote="")
_r("mstnk", "mstnk", "Spatial", "Trustworthiness metric", quote="")
_r("mstri", "mstri", "Spatial", "Triangle inequality check", quote="")
_r("msvor", "msvor", "Spatial", "2D Voronoi diagram", quote="")
_r("mtacc", "mtacc", "MovTyp", "Accessibility analysis", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("mtarf", "mtarf", "MovTyp", "Arc routing spatial", quote="You should enjoy the detours. -- Ging")
_r("mtash", "mtash", "MovTyp", "Assortativity spatial", quote="Arise. -- Shadow Monarch")
_r("mtbrw", "mtbrw", "MovTyp", "Biased random walk", quote="I will take a potato chip and eat it! -- Light")
_r("mtbtw", "mtbtw", "MovTyp", "Betweenness centrality spatial", quote="Dedicate your hearts! -- Erwin")
_r("mtcls", "mtcls", "MovTyp", "Closeness centrality spatial", quote="I am justice! -- Light")
_r("mtcnt", "mtcnt", "MovTyp", "Centrality analysis spatial", quote="Yare yare daze. -- Jotaro")
_r("mtcon", "mtcon", "MovTyp", "Connectivity analysis spatial", quote="It's over 9000! -- Vegeta")
_r("mtcor", "mtcor", "MovTyp", "Core-periphery spatial", quote="El Psy Kongroo. -- Okabe")
_r("mtcps", "mtcps", "MovTyp", "Capacitated spatial routing", quote="Whatever happens, happens. -- Spike")
_r("mtcrw", "mtcrw", "MovTyp", "Correlated random walk", quote="Keep moving forward. -- Eren")
_r("mtdeg", "mtdeg", "MovTyp", "Degree centrality spatial", quote="Bankai! -- Ichigo")
_r("mtdia", "mtdia", "MovTyp", "Diameter spatial network", quote="Live long and prosper. -- Spock")
_r("mtdns", "mtdns", "MovTyp", "Density spatial network", quote="Get in the robot, Shinji! -- Misato")
_r("mtdsp", "mtdsp", "MovTyp", "Dial-a-ride spatial", quote="I am here! -- All Might")
_r("mteff", "mteff", "MovTyp", "Efficiency spatial network", quote="Resistance is futile. -- Borg")
_r("mteig", "mteig", "MovTyp", "Eigenvector centrality spatial", quote="Go beyond! Plus Ultra! -- All Might")
_r("mtent", "mtent", "MovTyp", "Entropy spatial network", quote="Science! -- Jesse Pinkman")
_r("mtflw", "mtflw", "MovTyp", "Flow analysis spatial", quote="Believe it! -- Naruto")
_r("mtfpt", "mtfpt", "MovTyp", "First passage time", quote="I am the hope of the universe. -- Goku")
_r("mthmm", "mthmm", "MovTyp", "Hidden Markov movement", quote="The spice must flow. -- Paul Atreides")
_r("mtinf", "mtinf", "MovTyp", "Information centrality spatial", quote="Breathe. -- Tanjiro")
_r("mtktz", "mtktz", "MovTyp", "Katz centrality spatial", quote="See you space cowboy. -- Spike")
_r("mtlvy", "mtlvy", "MovTyp", "Levy walk movement", quote="Scatter, Senbonzakura. -- Byakuya")
_r("mtmlr", "mtmlr", "MovTyp", "Multi-depot routing", quote="I mustn't run away. -- Shinji")
_r("mtmod", "mtmod", "MovTyp", "Modularity spatial network", quote="Set your heart ablaze! -- Rengoku")
_r("mtmsd", "mtmsd", "MovTyp", "Mean squared displacement", quote="People's dreams never end! -- Blackbeard")
_r("mtnet", "mtnet", "MovTyp", "Movement network analysis", quote="Not all those who wander are lost. -- Gandalf")
_r("mtodm", "mtodm", "MovTyp", "Origin-destination matrix", quote="Make it so. -- Picard")
_r("mtpdr", "mtpdr", "MovTyp", "Periodic routing spatial", quote="Power is everything. -- Sung Jin-Woo")
_r("mtpgr", "mtpgr", "MovTyp", "PageRank spatial", quote="Equivalent exchange. -- Elric brothers")
_r("mtpkp", "mtpkp", "MovTyp", "Pickup-delivery spatial", quote="A lesson without pain is meaningless. -- Edward")
_r("mtrad", "mtrad", "MovTyp", "Radius spatial network", quote="One does not simply walk. -- Boromir")
_r("mtrcr", "mtrcr", "MovTyp", "Reciprocity spatial network", quote="I am the one who knocks. -- Walter White")
_r("mtred", "mtred", "MovTyp", "Redundancy spatial network", quote="Chaos is a ladder. -- Littlefinger")
_r("mtres", "mtres", "MovTyp", "Resilience spatial network", quote="Desert power. -- Paul Muad'Dib")
_r("mtrob", "mtrob", "MovTyp", "Robustness spatial network", quote="The world is cruel but beautiful. -- Mikasa")
_r("mtsbd", "mtsbd", "MovTyp", "Step-by-step decomposition", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("mtsin", "mtsin", "MovTyp", "Sinuosity index movement", quote="There is always hope. -- Aragorn")
_r("mtspr", "mtspr", "MovTyp", "Shortest path spatial", quote="I alone level up. -- Sung Jin-Woo")
_r("mtssm", "mtssm", "MovTyp", "State-space movement model", quote="One is all, all is one. -- Izumi")
_r("mtstp", "mtstp", "MovTyp", "Step length analysis", quote="Engage. -- Picard")
_r("mtstr", "mtstr", "MovTyp", "Stochastic routing spatial", quote="A Lannister always pays his debts. -- Tyrion")
_r("mttng", "mttng", "MovTyp", "Turning angle analysis", quote="Those who break the rules are scum. -- Kakashi")
_r("mttrc", "mttrc", "MovTyp", "Movement trajectory analysis", quote="Walk without rhythm. -- Fremen proverb")
_r("mttrs", "mttrs", "MovTyp", "Transitivity spatial network", quote="Winter is coming. -- Stark motto")
_r(
    "mttsp2",
    "mttsp2",
    "MovTyp",
    "TSP movement optimization",
    quote="Even the smallest person can change the future. -- Galadriel",
)
_r("mttwr", "mttwr", "MovTyp", "Time-window routing", quote="Growing old is a blessing. -- Rengoku")
_r("mtvrp", "mtvrp", "MovTyp", "Vehicle routing movement", quote="The needs of the many outweigh the few. -- Spock")
_r("mtvul", "mtvul", "MovTyp", "Vulnerability spatial network", quote="I must not fear. -- Litany Against Fear")
_r("mvt1d", "mvt1d", "Spatial", "Median voter theorem 1D.", quote="")
_r("mvtbi", "mvtbi", "Spatial", "Bicameral median voter.", quote="")
_r("mvtcl", "mvtcl", "Spatial", "Condorcet median voter equilibrium.", quote="")
_r("mvtdv", "mvtdv", "Spatial", "Dynamic median voter adjustment.", quote="")
_r("mvtfe", "mvtfe", "Spatial", "Federal median voter theorem.", quote="")
_r("mvtmj", "mvtmj", "Spatial", "Majority rule median voter.", quote="")
_r("mvtms", "mvtms", "Spatial", "Multidimensional median voter set.", quote="")
_r("mvtnd", "mvtnd", "Spatial", "Median voter theorem N-dimensional.", quote="")
_r("mvtpl", "mvtpl", "Spatial", "Plurality median voter.", quote="")
_r("mvtpr", "mvtpr", "Spatial", "Proportional representation median.", quote="")
_r("mvtpt", "mvtpt", "Spatial", "Pareto set median voter.", quote="")
_r("mvtru", "mvtru", "Spatial", "Runoff median voter.", quote="")
_r("mvtst", "mvtst", "Spatial", "Stochastic median voter model.", quote="")
_r("mvtsu", "mvtsu", "Spatial", "Supermajority median voter.", quote="")
_r("mvtwe", "mvtwe", "Spatial", "Weighted median voter theorem.", quote="")
_r("mxfsp", "mxfsp", "KrigFilt", "Max filter spatial", quote="Desert power. -- Paul Muad'Dib")
_r("nbair", "nbair", "NoisBrd", "Aircraft noise mapping", quote="Go beyond! Plus Ultra! -- All Might")
_r("nbann", "nbann", "NoisBrd", "Annoyance spatial model", quote="I am the one who knocks. -- Walter White")
_r("nbatm", "nbatm", "NoisBrd", "Atmospheric attenuation", quote="Equivalent exchange. -- Elric brothers")
_r("nbatn", "nbatn", "NoisBrd", "Attenuation distance", quote="Dedicate your hearts! -- Erwin")
_r("nbbar", "nbbar", "NoisBrd", "Barrier attenuation", quote="I am justice! -- Light")
_r("nbbrd", "nbbrd", "NoisBrd", "Bird diversity noise", quote="Believe it! -- Naruto")
_r("nbcmp", "nbcmp", "NoisBrd", "Noise compliance spatial", quote="Winter is coming. -- Stark motto")
_r("nbcnl", "nbcnl", "NoisBrd", "CNEL community level", quote="It's over 9000! -- Vegeta")
_r("nbcns", "nbcns", "NoisBrd", "Construction noise mapping", quote="El Psy Kongroo. -- Okabe")
_r("nbcnt", "nbcnt", "NoisBrd", "Noise contour mapping", quote="Set your heart ablaze! -- Rengoku")
_r("nbcog", "nbcog", "NoisBrd", "Cognitive impact noise", quote="The spice must flow. -- Paul Atreides")
_r("nbcvd", "nbcvd", "NoisBrd", "Cardiovascular risk noise", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("nbdnl", "nbdnl", "NoisBrd", "DNL day-night level", quote="Yare yare daze. -- Jotaro")
_r("nbgrd", "nbgrd", "NoisBrd", "Ground attenuation", quote="Bankai! -- Ichigo")
_r("nbgvb", "nbgvb", "NoisBrd", "Ground-borne vibration", quote="Dedicate your hearts! -- Erwin")
_r("nbhrs", "nbhrs", "NoisBrd", "Hearing loss risk spatial", quote="Not all those who wander are lost. -- Gandalf")
_r("nbind", "nbind", "NoisBrd", "Industrial noise mapping", quote="See you space cowboy. -- Spike")
_r("nbiso", "nbiso", "NoisBrd", "Noise isopleth mapping", quote="El Psy Kongroo. -- Okabe")
_r("nbl10", "nbl10", "NoisBrd", "L10 percentile level", quote="Not all those who wander are lost. -- Gandalf")
_r("nbl50", "nbl50", "NoisBrd", "L50 percentile level", quote="Make it so. -- Picard")
_r("nbl90", "nbl90", "NoisBrd", "L90 percentile level", quote="Believe it! -- Naruto")
_r("nblde", "nblde", "NoisBrd", "Lden overall level", quote="Get in the robot, Shinji! -- Misato")
_r("nbldn", "nbldn", "NoisBrd", "Ldn day-night level", quote="Winter is coming. -- Stark motto")
_r("nbleq", "nbleq", "NoisBrd", "Leq equivalent level", quote="The spice must flow. -- Paul Atreides")
_r("nblmx", "nblmx", "NoisBrd", "Lmax peak level", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("nblnq", "nblnq", "NoisBrd", "Lnight quiet level", quote="I am the one who knocks. -- Walter White")
_r("nbmrn", "nbmrn", "NoisBrd", "Marine noise spatial", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("nbppr", "nbppr", "NoisBrd", "Propagation model", quote="See you space cowboy. -- Spike")
_r("nbrdm", "nbrdm", "NoisBrd", "Road noise mapping", quote="Bankai! -- Ichigo")
_r("nbrwy", "nbrwy", "NoisBrd", "Railway noise mapping", quote="Equivalent exchange. -- Elric brothers")
_r("nbsel", "nbsel", "NoisBrd", "SEL sound exposure", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("nbslp", "nbslp", "NoisBrd", "Sleep disturbance spatial", quote="Get in the robot, Shinji! -- Misato")
_r("nbspc", "nbspc", "NoisBrd", "Species noise impact", quote="Make it so. -- Picard")
_r("nbstv", "nbstv", "NoisBrd", "Structural vibration", quote="I am justice! -- Light")
_r("nbtrn", "nbtrn", "NoisBrd", "Traffic noise model", quote="Arise. -- Shadow Monarch")
_r("nbund", "nbund", "NoisBrd", "Underwater noise mapping", quote="It's over 9000! -- Vegeta")
_r("nbvbr", "nbvbr", "NoisBrd", "Vibration spatial", quote="Yare yare daze. -- Jotaro")
_r("nbveg", "nbveg", "NoisBrd", "Vegetation attenuation", quote="Go beyond! Plus Ultra! -- All Might")
_r("nbwnd", "nbwnd", "NoisBrd", "Wind turbine noise", quote="Set your heart ablaze! -- Rengoku")
_r("nbzne", "nbzne", "NoisBrd", "Noise zone delineation", quote="Arise. -- Shadow Monarch")
_r("nestgr", "nestgr", "GeoProcss", "Nested multi-resolution grid", quote="I am the hope of the universe. -- Goku")
_r("nmala", "nmala", "Spatial", "Alpha-NOMINATE acceptance rate", quote="")
_r("nmalc", "nmalc", "Spatial", "Alpha-NOMINATE convergence", quote="")
_r("nmaln", "nmaln", "Spatial", "Alpha-NOMINATE (Bayesian MCMC)", quote="")
_r("nmalp", "nmalp", "Spatial", "Alpha-NOMINATE posterior", quote="")
_r("nmapr", "nmapr", "Spatial", "Aggregate PRE", quote="")
_r("nmdmp", "nmdmp", "Spatial", "Party divergence measure", quote="")
_r("nmdw2", "nmdw2", "Spatial", "DW-NOMINATE bridging observations", quote="")
_r("nmdwn", "nmdwn", "Spatial", "DW-NOMINATE dynamic estimation", quote="")
_r("nmdwp", "nmdwp", "Spatial", "DW-NOMINATE polarization", quote="")
_r("nmdwt", "nmdwt", "Spatial", "DW-NOMINATE trend analysis", quote="")
_r("nmgmp", "nmgmp", "Spatial", "Geometric Mean Probability", quote="")
_r("nmoc2", "nmoc2", "Spatial", "Optimal Classification 2D", quote="")
_r("nmocc", "nmocc", "Spatial", "OC classification rate", quote="")
_r("nmocl", "nmocl", "Spatial", "Optimal Classification cutting line", quote="")
_r("nmocm", "nmocm", "Spatial", "OC Coombs mesh", quote="")
_r("nmovl", "nmovl", "Spatial", "Party overlap index", quote="")
_r("nmplr", "nmplr", "Spatial", "Legislative polarization", quote="")
_r("nmpol", "nmpol", "Spatial", "Legislative polarity detection", quote="")
_r("nmpre", "nmpre", "Spatial", "Proportional Reduction in Error", quote="")
_r("nmrca", "nmrca", "Spatial", "Roll call agreement score", quote="")
_r("nmrcf", "nmrcf", "Spatial", "Roll call filtering", quote="")
_r("nmrcm", "nmrcm", "Spatial", "Roll call matrix construction", quote="")
_r("nmrcn", "nmrcn", "Spatial", "Roll call normalization", quote="")
_r("nmrcy", "nmrcy", "Spatial", "Roll call yea/nay summary", quote="")
_r("nmwn2", "nmwn2", "Spatial", "W-NOMINATE 2D", quote="")
_r("nmwnc", "nmwnc", "Spatial", "W-NOMINATE classification", quote="")
_r("nmwnl", "nmwnl", "Spatial", "W-NOMINATE log-likelihood", quote="")
_r("nmwno", "nmwno", "Spatial", "W-NOMINATE estimation", quote="")
_r("nmwnp", "nmwnp", "Spatial", "W-NOMINATE vote probability", quote="")
_r("nmwnw", "nmwnw", "Spatial", "W-NOMINATE dimension weights", quote="")
_r("nncv", "nncv", "Spatial", "Natural neighbor cross-validation.", quote="")
_r("nncvx", "nncvx", "Spatial", "Convex hull natural neighbor domain.", quote="")
_r("nnextr", "nnextr", "Spatial", "Natural neighbor extrapolation bounds.", quote="")
_r("nngrd", "nngrd", "Spatial", "Natural neighbor grid prediction.", quote="")
_r("nnint", "nnint", "KrigFilt", "Natural neighbor interpolation", quote="Winter is coming. -- Stark motto")
_r("nnlap", "nnlap", "Spatial", "Laplace / non-Sibsonian natural neighbor.", quote="")
_r("nnsib", "nnsib", "Spatial", "Sibson natural neighbor interpolation.", quote="")
_r("nnstln", "nnstln", "Spatial", "Natural neighbor gradient plane fitting.", quote="")
_r("nnvar", "nnvar", "Spatial", "Natural neighbor variance estimation.", quote="")
_r("nnvor", "nnvor", "Spatial", "Voronoi-based natural neighbor weights.", quote="")
_r("nnwgt", "nnwgt", "Spatial", "Natural neighbor area weight computation.", quote="")
_r("nrst2", "nrst2", "GeoAnalysis", "Second nearest neighbor distance", quote="El Psy Kongroo. -- Okabe")
_r("nrstd", "nrstd", "GeoAnalysis", "Nearest distance computation", quote="See you space cowboy. -- Spike")
_r("nrstk", "nrstk", "GeoAnalysis", "K nearest neighbor distances", quote="Set your heart ablaze! -- Rengoku")
_r("nrstm", "nrstm", "GeoAnalysis", "Nearest neighbor mean distance", quote="Arise. -- Shadow Monarch")
_r("nutsm", "nutsm", "SpatialPat", "NUTS sampler spatial", quote="Science! -- Jesse Pinkman")
_r("okflt", "okflt", "KrigFilt", "Ordinary kriging filter", quote="Those who break the rules are scum. -- Kakashi")
_r("opabc", "opabc", "OptimSp", "Artificial bee colony spatial", quote="Not all those who wander are lost. -- Gandalf")
_r("opaco", "opaco", "OptimSp", "Ant colony optimization spatial", quote="Make it so. -- Picard")
_r("opagg", "opagg", "OptimSp", "Spatial aggregation problem", quote="Desert power. -- Paul Muad'Dib")
_r("opall", "opall", "OptimSp", "Spatial allocation problem", quote="I must not fear. -- Litany Against Fear")
_r("opass", "opass", "OptimSp", "Spatial assignment problem", quote="Resistance is futile. -- Borg")
_r("opazt", "opazt", "OptimSp", "AZP regionalization", quote="I am here! -- All Might")
_r("opbat", "opbat", "OptimSp", "Bat algorithm spatial", quote="It's over 9000! -- Vegeta")
_r("opbfg", "opbfg", "OptimSp", "BFGS spatial optimization", quote="Engage. -- Picard")
_r("opcgd", "opcgd", "OptimSp", "Conjugate gradient spatial", quote="There is always hope. -- Aragorn")
_r("opcko", "opcko", "OptimSp", "Cuckoo search spatial", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("opcnt", "opcnt", "OptimSp", "Spatial p-center problem", quote="Arise. -- Shadow Monarch")
_r("opcom", "opcom", "OptimSp", "Spatial compactness measure", quote="Breathe. -- Tanjiro")
_r("opcov", "opcov", "OptimSp", "Spatial set covering", quote="Winter is coming. -- Stark motto")
_r("opdbr", "opdbr", "OptimSp", "DBSCAN spatial regionalization", quote="Power is everything. -- Sung Jin-Woo")
_r("opde", "opde", "OptimSp", "Differential evolution spatial", quote="The spice must flow. -- Paul Atreides")
_r("openf", "openf", "KrigFilt", "Opening morphological", quote="A Lannister always pays his debts. -- Tyrion")
_r("opffa", "opffa", "OptimSp", "Firefly algorithm spatial", quote="Believe it! -- Naruto")
_r("opfln", "opfln", "OptimSp", "Spatial flow capture", quote="Get in the robot, Shinji! -- Misato")
_r("opga", "opga", "OptimSp", "Genetic algorithm spatial", quote="Scatter, Senbonzakura. -- Byakuya")
_r("opgir", "opgir", "OptimSp", "Girvan-Newman spatial community", quote="Whatever happens, happens. -- Spike")
_r("opgrr", "opgrr", "OptimSp", "Spatial grouping/regionalization", quote="I alone level up. -- Sung Jin-Woo")
_r("opgwo", "opgwo", "OptimSp", "Grey wolf optimizer spatial", quote="Dedicate your hearts! -- Erwin")
_r("ophdb", "ophdb", "OptimSp", "HDBSCAN spatial regionalization", quote="A Lannister always pays his debts. -- Tyrion")
_r("ophho", "ophho", "OptimSp", "Harris hawks optimizer spatial", quote="I am justice! -- Light")
_r("opjay", "opjay", "OptimSp", "Jaya algorithm spatial", quote="Go beyond! Plus Ultra! -- All Might")
_r("oplbf", "oplbf", "OptimSp", "L-BFGS-B spatial optimization", quote="Those who break the rules are scum. -- Kakashi")
_r("opldn", "opldn", "OptimSp", "Leiden spatial clustering", quote="Growing old is a blessing. -- Rengoku")
_r("oplvn", "oplvn", "OptimSp", "Louvain spatial clustering", quote="I mustn't run away. -- Shinji")
_r("opmax", "opmax", "OptimSp", "Max-p regionalization", quote="A lesson without pain is meaningless. -- Edward")
_r("opmed", "opmed", "OptimSp", "Spatial p-median problem", quote="Set your heart ablaze! -- Rengoku")
_r("opmfo", "opmfo", "OptimSp", "Moth-flame optimizer spatial", quote="Bankai! -- Ichigo")
_r("opmxc", "opmxc", "OptimSp", "Spatial max covering location", quote="I am the one who knocks. -- Walter White")
_r("opmxm", "opmxm", "OptimSp", "Maximum modularity spatial", quote="You should enjoy the detours. -- Ging")
_r("opnms", "opnms", "OptimSp", "Nelder-Mead spatial optimization", quote="Walk without rhythm. -- Fremen proverb")
_r("opnwt", "opnwt", "OptimSp", "Newton-Raphson spatial", quote="I am the hope of the universe. -- Goku")
_r("oppar", "oppar", "OptimSp", "Spatial partitioning", quote="The world is cruel but beautiful. -- Mikasa")
_r("oppso", "oppso", "OptimSp", "Particle swarm spatial", quote="One is all, all is one. -- Izumi")
_r("oprdn", "oprdn", "OptimSp", "REDCAP regionalization", quote="The needs of the many outweigh the few. -- Spock")
_r("opred", "opred", "OptimSp", "Spatial redistricting", quote="Science! -- Jesse Pinkman")
_r("opsa", "opsa", "OptimSp", "Simulated annealing spatial", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("opsca", "opsca", "OptimSp", "Sine cosine algorithm spatial", quote="Equivalent exchange. -- Elric brothers")
_r("opsfl", "opsfl", "OptimSp", "Spatial facility location", quote="El Psy Kongroo. -- Okabe")
_r(
    "opskr",
    "opskr",
    "OptimSp",
    "SKATER regionalization",
    quote="Even the smallest person can change the future. -- Galadriel",
)
_r("opslp", "opslp", "OptimSp", "Sequential linear programming", quote="Keep moving forward. -- Eren")
_r(
    "opsqp",
    "opsqp",
    "OptimSp",
    "Sequential quadratic programming",
    quote="I will take a potato chip and eat it! -- Light",
)
_r("optlb", "optlb", "OptimSp", "Teaching-learning spatial", quote="See you space cowboy. -- Spike")
_r("optrs", "optrs", "OptimSp", "Trust region spatial", quote="People's dreams never end! -- Blackbeard")
_r("optsp", "optsp", "OptimSp", "Traveling salesman spatial", quote="One does not simply walk. -- Boromir")
_r("opvhc", "opvhc", "OptimSp", "Vehicle routing spatial", quote="Live long and prosper. -- Spock")
_r("opwoa", "opwoa", "OptimSp", "Whale optimization spatial", quote="Yare yare daze. -- Jotaro")
_r("opzon", "opzon", "OptimSp", "Spatial zoning problem", quote="Chaos is a ladder. -- Littlefinger")
_r("overla", "overla", "GeoAnalysis", "Spatial overlay intersection", quote="Dedicate your hearts! -- Erwin")
_r("ovrsmp", "ovrsmp", "GeoProcss", "Oversampling rare spatial events", quote="I mustn't run away. -- Shinji")
_r("paapp", "paapp", "Spatial", "Approval voting spatial.", quote="")
_r("pabay", "pabay", "Spatial", "Bayesian aggregation spatial.", quote="")
_r("pabch", "pabch", "Spatial", "Bucklin aggregation spatial.", quote="")
_r("pabrd", "pabrd", "Spatial", "Borda count spatial.", quote="")
_r("pacar", "pacar", "Spatial", "Cardinalist preference aggregation.", quote="")
_r("pacnd", "pacnd", "Spatial", "Condorcet aggregation.", quote="")
_r("pacop", "pacop", "Spatial", "Copeland aggregation spatial.", quote="")
_r("pafuz", "pafuz", "Spatial", "Fuzzy preference aggregation.", quote="")
_r("pakmd", "pakmd", "Spatial", "Kemeny-Young aggregation.", quote="")
_r("pamaj", "pamaj", "Spatial", "Majority judgment spatial.", quote="")
_r("paord", "paord", "Spatial", "Ordinalist preference aggregation.", quote="")
_r("paslg", "paslg", "Spatial", "Slater aggregation spatial.", quote="")
_r("pasoc", "pasoc", "Spatial", "Social choice spatial.", quote="")
_r("pastr", "pastr", "Spatial", "Strategic aggregation.", quote="")
_r("pawgt", "pawgt", "Spatial", "Weighted aggregation spatial.", quote="")
_r("pcfsp", "pcfsp", "SpatialPat", "Pair correlation function", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("percs", "percs", "SpatialPat", "Percolation model spatial", quote="Engage. -- Picard")
_r("pfsim", "pfsim", "SpatialPat", "P-field simulation", quote="Get in the robot, Shinji! -- Misato")
_r("pkflt", "pkflt", "KrigFilt", "Probability kriging filter", quote="I will take a potato chip and eat it! -- Light")
_r("plafp", "plafp", "Spatial", "Affective polarization index.", quote="")
_r("plbdn", "plbdn", "Spatial", "Benbow-Dunning polarization.", quote="")
_r("plbim", "plbim", "Spatial", "Bimodality polarization index.", quote="")
_r("plcnt", "plcnt", "Spatial", "Centrist polarization measure.", quote="")
_r("plcoh", "plcoh", "Spatial", "Cohesion-based polarization.", quote="")
_r("plcrd", "plcrd", "Spatial", "Cross-dimensional polarization.", quote="")
_r("pldst", "pldst", "Spatial", "Distance-based polarization.", quote="")
_r("pldyn", "pldyn", "Spatial", "Dynamic polarization trend.", quote="")
_r("plemd", "plemd", "Spatial", "Earth mover distance polarization.", quote="")
_r("pleri", "pleri", "Spatial", "Esteban-Ray polarization index.", quote="")
_r("plgni", "plgni", "Spatial", "Gini polarization index.", quote="")
_r("plidp", "plidp", "Spatial", "Ideological polarization index.", quote="")
_r("plkur", "plkur", "Spatial", "Kurtosis-based polarization.", quote="")
_r("plmss", "plmss", "Spatial", "Mass-elite polarization.", quote="")
_r("plnet", "plnet", "Spatial", "Network-based polarization.", quote="")
_r("plsor", "plsor", "Spatial", "Sorting polarization index.", quote="")
_r("pltrb", "pltrb", "Spatial", "Tribal polarization index.", quote="")
_r("plvar", "plvar", "Spatial", "Variance-based polarization.", quote="")
_r("plwgn", "plwgn", "Spatial", "Weighted Gini polarization.", quote="")
_r("plwlf", "plwlf", "Spatial", "Wolfson polarization index.", quote="")
_r(
    "ppann",
    "ppann",
    "PointProc",
    "Nearest neighbor PP statistics",
    quote="A lesson without pain is meaningless. -- Edward",
)
_r("ppare", "ppare", "PointProc", "Area-interaction process", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("ppbay", "ppbay", "PointProc", "Bayesian point process", quote="Dedicate your hearts! -- Erwin")
_r("ppblk", "ppblk", "Spatial", "Party bloc positioning.", quote="")
_r("ppbor", "ppbor", "PointProc", "Border correction PP", quote="Growing old is a blessing. -- Rengoku")
_r("ppchg", "ppchg", "Spatial", "Party change positioning.", quote="")
_r("ppclp", "ppclp", "PointProc", "Cluster Poisson process", quote="I am the hope of the universe. -- Goku")
_r("ppcnv", "ppcnv", "Spatial", "Party convergence dynamics.", quote="")
_r(
    "ppcox",
    "ppcox",
    "PointProc",
    "Cox process (doubly stochastic)",
    quote="Those who break the rules are scum. -- Kakashi",
)
_r("ppdgr", "ppdgr", "PointProc", "Diggle test space-time", quote="I alone level up. -- Sung Jin-Woo")
_r("ppdig", "ppdig", "PointProc", "Diggle-Gratton process", quote="It's over 9000! -- Vegeta")
_r("ppdst", "ppdst", "Spatial", "Party distance matrix.", quote="")
_r("ppdvg", "ppdvg", "Spatial", "Party divergence dynamics.", quote="")
_r("ppedg", "ppedg", "PointProc", "Edge correction PP", quote="Whatever happens, happens. -- Spike")
_r("ppent", "ppent", "Spatial", "Party entry positioning.", quote="")
_r("ppenv", "ppenv", "PointProc", "Envelope test for PP", quote="El Psy Kongroo. -- Okabe")
_r("ppfam", "ppfam", "Spatial", "Party family positioning.", quote="")
_r("ppgey", "ppgey", "PointProc", "Geyer saturation process", quote="Believe it! -- Naruto")
_r("pphpp", "pphpp", "PointProc", "Homogeneous Poisson process", quote="There is always hope. -- Aragorn")
_r("ppint2", "ppint2", "PointProc", "Intensity estimation (adaptive)", quote="I am justice! -- Light")
_r("ppipp", "ppipp", "PointProc", "Inhomogeneous Poisson process", quote="Engage. -- Picard")
_r("ppiso", "ppiso", "PointProc", "Isotropic correction PP", quote="I mustn't run away. -- Shinji")
_r("ppitb", "ppitb", "PointProc", "Intensity with bandwidth CV", quote="Bankai! -- Ichigo")
_r("ppitd", "ppitd", "PointProc", "Intensity diggle correction", quote="Equivalent exchange. -- Elric brothers")
_r("ppitj", "ppitj", "PointProc", "Intensity Jones correction", quote="Go beyond! Plus Ultra! -- All Might")
_r("ppitv", "ppitv", "PointProc", "Intensity Voronoi", quote="See you space cowboy. -- Spike")
_r(
    "ppjcq",
    "ppjcq",
    "PointProc",
    "Jacquez test space-time",
    quote="Even the smallest person can change the future. -- Galadriel",
)
_r("ppkno", "ppkno", "PointProc", "Knox test space-time", quote="Science! -- Jesse Pinkman")
_r("pplgc", "pplgc", "PointProc", "Log-Gaussian Cox process", quote="People's dreams never end! -- Blackbeard")
_r("ppman", "ppman", "Spatial", "Manifesto-based positioning.", quote="")
_r("ppmc", "ppmc", "PointProc", "Monte Carlo test PP", quote="Set your heart ablaze! -- Rengoku")
_r("ppmed", "ppmed", "Spatial", "Party median positioning.", quote="")
_r("ppmh2", "ppmh2", "PointProc", "Matern hard-core process II", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("ppmhr", "ppmhr", "PointProc", "Matern hard-core process I", quote="The spice must flow. -- Paul Atreides")
_r("ppmkc", "ppmkc", "PointProc", "Mark correlation PP", quote="Get in the robot, Shinji! -- Misato")
_r("ppmkd", "ppmkd", "PointProc", "Mark dependence test", quote="I am the one who knocks. -- Walter White")
_r("ppmks", "ppmks", "PointProc", "Mark segregation test", quote="Winter is coming. -- Stark motto")
_r("ppmkv", "ppmkv", "PointProc", "Mark variogram PP", quote="Live long and prosper. -- Spock")
_r("ppmnl", "ppmnl", "PointProc", "Mantel test space-time", quote="Breathe. -- Tanjiro")
_r("ppmov", "ppmov", "Spatial", "Party movement model.", quote="")
_r("ppmrk", "ppmrk", "PointProc", "Marked point process", quote="Arise. -- Shadow Monarch")
_r("ppmtc", "ppmtc", "PointProc", "Matern cluster process", quote="Keep moving forward. -- Eren")
_r("ppnch", "ppnch", "Spatial", "Party niche positioning.", quote="")
_r("ppnnd", "ppnnd", "PointProc", "Nearest neighbor distribution", quote="I am here! -- All Might")
_r("ppnyn", "ppnyn", "PointProc", "Neyman-Scott process", quote="Scatter, Senbonzakura. -- Byakuya")
_r("pppnl", "pppnl", "PointProc", "Penalized likelihood PP", quote="Yare yare daze. -- Jotaro")
_r("pppos", "pppos", "Spatial", "Party position estimation.", quote="")
_r("pprfd", "pprfd", "PointProc", "Refined nearest neighbor", quote="You should enjoy the detours. -- Ging")
_r("pprip", "pprip", "PointProc", "Ripley correction PP", quote="A Lannister always pays his debts. -- Tyrion")
_r("pprth", "pprth", "PointProc", "Ripley theta function", quote="The needs of the many outweigh the few. -- Spock")
_r("ppsal", "ppsal", "Spatial", "Party salience weighting.", quote="")
_r("ppsft", "ppsft", "PointProc", "Soft-core process", quote="Make it so. -- Picard")
_r("ppspx", "ppspx", "PointProc", "Space-time point process", quote="One does not simply walk. -- Boromir")
_r("ppssp", "ppssp", "PointProc", "Simple sequential inhibition", quote="One is all, all is one. -- Izumi")
_r("ppstg", "ppstg", "PointProc", "Space-time G-function", quote="I must not fear. -- Litany Against Fear")
_r("ppstj", "ppstj", "PointProc", "Space-time J-function", quote="Desert power. -- Paul Muad'Dib")
_r("ppstk", "ppstk", "PointProc", "Space-time K-function", quote="Resistance is futile. -- Borg")
_r("ppstl", "ppstl", "PointProc", "Space-time L-function", quote="The world is cruel but beautiful. -- Mikasa")
_r("ppstr", "ppstr", "PointProc", "Strauss process", quote="Not all those who wander are lost. -- Gandalf")
_r("ppsts", "ppsts", "PointProc", "Space-time scan statistic", quote="Chaos is a ladder. -- Littlefinger")
_r("ppthm", "ppthm", "PointProc", "Thomas cluster process", quote="I will take a potato chip and eat it! -- Light")
_r("pptra", "pptra", "PointProc", "Translation correction PP", quote="Power is everything. -- Sung Jin-Woo")
_r("ppval", "ppval", "Spatial", "Party valence positioning.", quote="")
_r("prjinv", "prjinv", "GeoProcss", "Projected to lon/lat inverse", quote="I am justice! -- Light")
_r("projct", "projct", "GeoProcss", "Coordinate projection transform", quote="The spice must flow. -- Paul Atreides")
_r("prwfp", "prwfp", "KrigFilt", "Prewitt filter spatial", quote="I am here! -- All Might")
_r("psbay", "psbay", "Spatial", "Bayesian spatial probability.", quote="")
_r("psgmb", "psgmb", "Spatial", "Gamma perturbation spatial.", quote="")
_r("pshid", "pshid", "Spatial", "Hidden state spatial model.", quote="")
_r("pshmm", "pshmm", "Spatial", "HMM spatial voting.", quote="")
_r("pskde", "pskde", "Spatial", "KDE spatial probability.", quote="")
_r("pslap", "pslap", "Spatial", "Laplace perturbation spatial.", quote="")
_r("pslin", "pslin", "Spatial", "Linear probability spatial.", quote="")
_r("psmix", "psmix", "Spatial", "Mixture model spatial.", quote="")
_r("psnon", "psnon", "Spatial", "Nonparametric spatial probability.", quote="")
_r("psnrm", "psnrm", "Spatial", "Normal perturbation spatial.", quote="")
_r("pssim", "pssim", "Spatial", "Simulation-based spatial.", quote="")
_r("psunf", "psunf", "Spatial", "Uniform perturbation spatial.", quote="")
_r("psvot", "psvot", "Spatial", "Vote share probability.", quote="")
_r("pswin", "pswin", "Spatial", "Win probability spatial.", quote="")
_r("pt1st", "pt1st", "Spatial", "First-order point pattern stats", quote="")
_r("pt2nd", "pt2nd", "Spatial", "Second-order point pattern stats", quote="")
_r("ptcox", "ptcox", "Spatial", "Cox (doubly stochastic) process", quote="")
_r("ptcrs", "ptcrs", "Spatial", "Cross-type point pattern", quote="")
_r("ptcsr", "ptcsr", "Spatial", "Complete Spatial Randomness test", quote="")
_r("ptdgm", "ptdgm", "Spatial", "Diggle-Cressie-Loosmore test", quote="")
_r("ptdlr", "ptdlr", "Spatial", "Delaunay residuals", quote="")
_r("ptenv", "ptenv", "Spatial", "Point pattern Monte Carlo envelope", quote="")
_r("ptffn", "ptffn", "Spatial", "Empty space F-function", quote="")
_r("ptgfn", "ptgfn", "Spatial", "Nearest-neighbor G-function", quote="")
_r("ptipo", "ptipo", "Spatial", "Inhomogeneous Poisson process", quote="")
_r("ptisg", "ptisg", "Spatial", "Isotropic edge correction", quote="")
_r("ptjfn", "ptjfn", "Spatial", "J-function (ratio F/G)", quote="")
_r("ptkda", "ptkda", "Spatial", "Adaptive kernel density", quote="")
_r("ptkdb", "ptkdb", "Spatial", "KDE bandwidth selection (spatial)", quote="")
_r("ptkde", "ptkde", "Spatial", "Spatial kernel density estimation", quote="")
_r("ptkfn", "ptkfn", "Spatial", "Ripley's K-function", quote="")
_r("ptlfn", "ptlfn", "Spatial", "L-function (variance-stabilized K)", quote="")
_r("ptlgc", "ptlgc", "Spatial", "Log-Gaussian Cox process", quote="")
_r("ptmat", "ptmat", "Spatial", "Matern cluster process", quote="")
_r("ptmor", "ptmor", "Spatial", "Morisita index of dispersion", quote="")
_r("ptmrk", "ptmrk", "Spatial", "Marked point pattern analysis", quote="")
_r("ptnni", "ptnni", "Spatial", "Nearest neighbor index (Clark-Evans)", quote="")
_r("ptpcf", "ptpcf", "Spatial", "Pair correlation function g(r)", quote="")
_r("ptpoi", "ptpoi", "Spatial", "Homogeneous Poisson point process", quote="")
_r("ptqdr", "ptqdr", "Spatial", "Quadrat count test", quote="")
_r("ptrng", "ptrng", "Spatial", "Point pattern intensity", quote="")
_r("ptrpl", "ptrpl", "Spatial", "Ripley edge correction", quote="")
_r("ptthm", "ptthm", "Spatial", "Thomas cluster process", quote="")
_r("pttsp", "pttsp", "SpatialPat", "Potts model spatial", quote="A Lannister always pays his debts. -- Tyrion")
_r("ptvor", "ptvor", "Spatial", "Point pattern Voronoi intensities", quote="")
_r("rbfani", "rbfani", "Spatial", "Anisotropic RBF interpolation.", quote="")
_r("rbfcomp", "rbfcomp", "Spatial", "Compactly supported RBF Wendland.", quote="")
_r("rbfcub", "rbfcub", "Spatial", "Cubic RBF interpolation.", quote="")
_r("rbfcv", "rbfcv", "Spatial", "RBF cross-validation shape parameter.", quote="")
_r("rbfgrd", "rbfgrd", "Spatial", "RBF grid prediction surface.", quote="")
_r("rbfgss", "rbfgss", "Spatial", "Gaussian RBF interpolation.", quote="")
_r("rbfimq", "rbfimq", "Spatial", "Inverse multiquadric RBF interpolation.", quote="")
_r("rbflnr", "rbflnr", "Spatial", "Linear RBF interpolation.", quote="")
_r("rbfmq", "rbfmq", "Spatial", "Multiquadric RBF interpolation.", quote="")
_r("rbfmsv", "rbfmsv", "Spatial", "Multi-scale RBF hierarchical interpolation.", quote="")
_r("rbfphs", "rbfphs", "Spatial", "Polyharmonic spline RBF order-k.", quote="")
_r("rbfqnq", "rbfqnq", "Spatial", "Quintic RBF interpolation.", quote="")
_r("rbfreg", "rbfreg", "Spatial", "Regularized RBF with smoothing parameter.", quote="")
_r("rbfsys", "rbfsys", "Spatial", "RBF system solve via direct LU.", quote="")
_r("rbftps", "rbftps", "Spatial", "Thin-plate spline RBF interpolation.", quote="")
_r("rcabs", "rcabs", "Spatial", "Abstention roll call model.", quote="")
_r("rcbnf", "rcbnf", "Spatial", "Banfield roll call estimation.", quote="")
_r("rccls", "rccls", "Spatial", "Cluster roll call votes.", quote="")
_r("rccut", "rccut", "Spatial", "Cutting plane roll call.", quote="")
_r("rclag", "rclag", "Spatial", "Lagged roll call vote model.", quote="")
_r("rcltz", "rcltz", "Spatial", "Luce-choice roll call model.", quote="")
_r("rcpar", "rcpar", "Spatial", "Party cue roll call model.", quote="")
_r("rcpiv", "rcpiv", "Spatial", "Pivot model roll call.", quote="")
_r("rcpnl", "rcpnl", "Spatial", "Panel roll call estimation.", quote="")
_r("rcprs", "rcprs", "Spatial", "Presence/absence roll call.", quote="")
_r("rcrnd", "rcrnd", "Spatial", "Random utility roll call.", quote="")
_r("rcsca", "rcsca", "Spatial", "Scaling roll call votes.", quote="")
_r("rcsim", "rcsim", "Spatial", "Similarity matrix roll call.", quote="")
_r("rcvot", "rcvot", "Spatial", "Roll call vote probability basic.", quote="")
_r("rdsim", "rdsim", "SpatialPat", "Reaction-diffusion spatial", quote="Whatever happens, happens. -- Spike")
_r(
    "reproj",
    "reproj",
    "GeoProcss",
    "Reproject coordinates between CRS",
    quote="Fear is the mind-killer. -- Bene Gesserit",
)
_r("rfani", "rfani", "Spatial", "Anisotropic random field with rotation.", quote="")
_r("rfbin", "rfbin", "Spatial", "Binary random field threshold indicator.", quote="")
_r("rfbrow", "rfbrow", "Spatial", "Brownian motion / fractional Brownian field.", quote="")
_r("rfcat", "rfcat", "Spatial", "Categorical random field multi-class.", quote="")
_r("rfchi2", "rfchi2", "Spatial", "Chi-squared random field from Gaussian squares.", quote="")
_r("rfcox", "rfcox", "Spatial", "Cox process intensity random field.", quote="")
_r("rfgam", "rfgam", "Spatial", "Gamma random field via anamorphosis.", quote="")
_r("rfgss", "rfgss", "Spatial", "Gaussian random field with Matern covariance.", quote="")
_r("rflgn", "rflgn", "Spatial", "Log-normal random field transformation.", quote="")
_r("rfmax", "rfmax", "Spatial", "Max-stable random field for extremes.", quote="")
_r("rfmix", "rfmix", "Spatial", "Mixture random field from multiple components.", quote="")
_r("rfms", "rfms", "Spatial", "Multi-scale random field nested covariance.", quote="")
_r("rfnst", "rfnst", "Spatial", "Nonstationary random field deformation.", quote="")
_r("rforns", "rforns", "Spatial", "Ornstein-Uhlenbeck spatial random field.", quote="")
_r("rfpois", "rfpois", "Spatial", "Poisson random field intensity simulation.", quote="")
_r("rfsmt", "rfsmt", "Spatial", "Smooth random field low-frequency component.", quote="")
_r("rfstud", "rfstud", "Spatial", "Student-t random field heavy tails.", quote="")
_r("rftrun", "rftrun", "Spatial", "Truncated Gaussian random field.", quote="")
_r("rfvon", "rfvon", "Spatial", "Von Karman random field for turbulence.", quote="")
_r("rfwht", "rfwht", "Spatial", "White noise random field baseline.", quote="")
_r("rgfsp", "rgfsp", "KrigFilt", "Range filter spatial", quote="Chaos is a ladder. -- Littlefinger")
_r("rhumbr", "rhumbr", "GeoProcss", "Rhumb line distance", quote="It's over 9000! -- Vegeta")
_r("rnagg", "rnagg", "Spatial", "Raster aggregation / downsampling.", quote="")
_r("rnaspc", "rnaspc", "Spatial", "Raster aspect from DEM.", quote="")
_r("rncost", "rncost", "Spatial", "Raster cost surface accumulation.", quote="")
_r("rncurv", "rncurv", "Spatial", "Raster curvature profile and planform.", quote="")
_r("rndist", "rndist", "Spatial", "Raster distance transform Euclidean.", quote="")
_r("rndsgr", "rndsgr", "Spatial", "Raster disaggregation / upsampling.", quote="")
_r("rndsmp", "rndsmp", "GeoProcss", "Random spatial sampling", quote="Chaos is a ladder. -- Littlefinger")
_r("rnflwdr", "rnflwdr", "Spatial", "Raster flow direction D8 algorithm.", quote="")
_r("rnfocl", "rnfocl", "Spatial", "Raster focal moving window statistics.", quote="")
_r("rnhill", "rnhill", "Spatial", "Raster hillshade from DEM.", quote="")
_r("rnmask", "rnmask", "Spatial", "Raster mask and clip to polygon.", quote="")
_r("rnmrph", "rnmrph", "Spatial", "Raster morphological erosion/dilation.", quote="")
_r("rnrsmp", "rnrsmp", "Spatial", "Raster resampling bilinear/nearest.", quote="")
_r("rnslp", "rnslp", "Spatial", "Raster slope from DEM.", quote="")
_r("rnwatr", "rnwatr", "Spatial", "Raster watershed delineation.", quote="")
_r("rnzonl", "rnzonl", "Spatial", "Raster zonal statistics by zone grid.", quote="")
_r("robnsn", "robnsn", "GeoProcss", "Robinson projection", quote="Winter is coming. -- Stark motto")
_r("rotgrd", "rotgrd", "GeoProcss", "Rotated grid generation", quote="Keep moving forward. -- Eren")
_r("rsacc", "rsacc", "RemSens", "Accuracy assessment RS", quote="The sleeper must awaken. -- Leto Atreides")
_r("rsalb", "rsalb", "RemSens", "Albedo estimation RS", quote="Breathe. -- Tanjiro")
_r("rsatm", "rsatm", "RemSens", "Atmospheric correction RS", quote="A Lannister always pays his debts. -- Tyrion")
_r("rsbai", "rsbai", "RemSens", "BAI burn area index", quote="See you space cowboy. -- Spike")
_r("rschg", "rschg", "RemSens", "Change detection RS", quote="Kamehameha! -- Goku")
_r("rscir", "rscir", "RemSens", "CIre red-edge index", quote="Winter is coming. -- Stark motto")
_r("rsclm", "rsclm", "RemSens", "Cloud masking RS", quote="One does not simply walk. -- Boromir")
_r("rscls", "rscls", "RemSens", "Supervised classification RS", quote="Whatever happens, happens. -- Spike")
_r("rscnf", "rscnf", "RemSens", "Confusion matrix RS", quote="Valar Morghulis. -- Braavos")
_r("rscva", "rscva", "RemSens", "Change vector analysis", quote="My precious. -- Gollum")
_r("rsemi", "rsemi", "RemSens", "Emissivity estimation RS", quote="There is always hope. -- Aragorn")
_r("rsend", "rsend", "RemSens", "Endmember extraction", quote="Make it so. -- Picard")
_r("rsevi", "rsevi", "RemSens", "EVI computation", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("rsfcv", "rsfcv", "RemSens", "Fractional cover vegetation", quote="Chaos is a ladder. -- Littlefinger")
_r("rsfpr", "rsfpr", "RemSens", "FPAR from remote sensing", quote="Science! -- Jesse Pinkman")
_r("rsfus", "rsfus", "RemSens", "Image fusion RS", quote="I am here! -- All Might")
_r("rsgnd", "rsgnd", "RemSens", "GNDVI computation", quote="Dedicate your hearts! -- Erwin")
_r("rsica", "rsica", "RemSens", "ICA remote sensing", quote="The spice must flow. -- Paul Atreides")
_r("rskap", "rskap", "RemSens", "Kappa coefficient RS", quote="No half measures. -- Mike")
_r("rsknn", "rsknn", "RemSens", "KNN classification RS", quote="Tatakae! -- Eren")
_r("rslai2", "rslai2", "RemSens", "LAI from remote sensing", quote="The world is cruel but beautiful. -- Mikasa")
_r("rslst", "rslst", "RemSens", "Land surface temperature RS", quote="I alone level up. -- Sung Jin-Woo")
_r("rsmnd", "rsmnd", "RemSens", "MNDWI computation", quote="Equivalent exchange. -- Elric brothers")
_r("rsmnf", "rsmnf", "RemSens", "MNF transform RS", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("rsmsr", "rsmsr", "RemSens", "MSR modified simple ratio", quote="Get in the robot, Shinji! -- Misato")
_r("rsmsv", "rsmsv", "RemSens", "MSAVI computation", quote="Yare yare daze. -- Jotaro")
_r("rsmxl", "rsmxl", "RemSens", "Maximum likelihood RS", quote="Keep moving forward. -- Eren")
_r("rsnbr", "rsnbr", "RemSens", "NBR normalized burn ratio", quote="El Psy Kongroo. -- Okabe")
_r("rsnbr2", "rsnbr2", "RemSens", "NBR2 enhanced burn ratio", quote="Set your heart ablaze! -- Rengoku")
_r("rsndb", "rsndb", "RemSens", "NDBI computation", quote="I am justice! -- Light")
_r("rsndm", "rsndm", "RemSens", "NDMI moisture index", quote="Arise. -- Shadow Monarch")
_r("rsnds", "rsnds", "RemSens", "NDSI computation", quote="Go beyond! Plus Ultra! -- All Might")
_r("rsndv", "rsndv", "RemSens", "NDVI computation", quote="Believe it! -- Naruto")
_r("rsndw", "rsndw", "RemSens", "NDWI computation", quote="Bankai! -- Ichigo")
_r("rspan", "rspan", "RemSens", "Pan-sharpening RS", quote="A lesson without pain is meaningless. -- Edward")
_r("rspca", "rspca", "RemSens", "PCA remote sensing", quote="Hold the door. -- Hodor")
_r("rspri", "rspri", "RemSens", "PRI photochemical index", quote="Live long and prosper. -- Spock")
_r("rsreg", "rsreg", "RemSens", "Red-edge NDVI", quote="I am the one who knocks. -- Walter White")
_r("rsrei", "rsrei", "RemSens", "REI red-edge inflection", quote="Desert power. -- Paul Muad'Dib")
_r("rsrfc", "rsrfc", "RemSens", "Random forest classification RS", quote="Total Concentration Breathing. -- Tanjiro")
_r("rsrtx", "rsrtx", "RemSens", "Radiometric terrain correction", quote="Power is everything. -- Sung Jin-Woo")
_r("rssav", "rssav", "RemSens", "SAVI computation", quote="It's over 9000! -- Vegeta")
_r("rsseg", "rsseg", "RemSens", "Image segmentation RS", quote="You should enjoy the detours. -- Ging")
_r("rsshd", "rsshd", "RemSens", "Shadow detection RS", quote="Resistance is futile. -- Borg")
_r("rssmc", "rssmc", "RemSens", "Soil moisture from RS", quote="Engage. -- Picard")
_r("rssrf", "rssrf", "RemSens", "Surface reflectance", quote="Growing old is a blessing. -- Rengoku")
_r("rssvm", "rssvm", "RemSens", "SVM classification RS", quote="This is Requiem. -- Giorno")
_r("rstcw", "rstcw", "RemSens", "Tasseled cap transform", quote="Not all those who wander are lost. -- Gandalf")
_r("rstoa", "rstoa", "RemSens", "TOA reflectance", quote="Walk without rhythm. -- Fremen proverb")
_r("rsusc", "rsusc", "RemSens", "Unsupervised classification", quote="I mustn't run away. -- Shinji")
_r("rwksp", "rwksp", "SpatialPat", "Random walk on grid", quote="I am here! -- All Might")
_r("rwsim", "rwsim", "SpatialPat", "Random walk simulation spatial", quote="I alone level up. -- Sung Jin-Woo")
_r("saamo", "saamo", "SpatAutoC", "Analytical Moran's I", quote="Yare yare daze. -- Jotaro")
_r("sacaic", "sacaic", "SAC", "SAC Akaike information criterion.", quote="")
_r("sacbic", "sacbic", "SAC", "SAC Bayesian information criterion.", quote="")
_r("sacboot", "sacboot", "SAC", "SAC bootstrap CI for rho.", quote="")
_r("sacconv", "sacconv", "SAC", "SAC rho/lambda joint feasibility bounds.", quote="")
_r("sacdet", "sacdet", "SAC", "SAC log-determinant product.", quote="")
_r("sacgmm", "sacgmm", "SAC", "SAC GMM (Kelejian-Prucha) estimator.", quote="")
_r("sacimp", "sacimp", "SAC", "SAC direct/indirect/total impacts.", quote="")
_r("sacjac", "sacjac", "SAC", "SAC dual Jacobian term.", quote="")
_r("saclrt", "saclrt", "SAC", "SAC likelihood-ratio test vs SAR/SEM.", quote="")
_r("sacml", "sacml", "SAC", "SAC (combined lag+error) ML estimation.", quote="")
_r("sacres", "sacres", "SAC", "SAC residual autocorrelation check.", quote="")
_r("sacrob", "sacrob", "SAC", "SAC robust (HC) standard errors.", quote="")
_r("sacsig", "sacsig", "SAC", "SAC sigma-squared ML estimate.", quote="")
_r("sacvar", "sacvar", "SAC", "SAC variance-covariance matrix.", quote="")
_r("sacwald", "sacwald", "SAC", "SAC Wald test on rho and lambda jointly.", quote="")
_r(
    "saelm",
    "saelm",
    "SpatAutoC",
    "Moran eigenvector spatial filter",
    quote="I'm gonna be King of the Pirates! -- Luffy",
)
_r("sagcr", "sagcr", "SpatAutoC", "Geary correlogram", quote="Go beyond! Plus Ultra! -- All Might")
_r("sagge", "sagge", "SpatAutoC", "Global Geary's C statistic", quote="There is always hope. -- Aragorn")
_r("saggt", "saggt", "SpatAutoC", "Global Getis-Ord G statistic", quote="Engage. -- Picard")
_r("sagjb", "sagjb", "SpatAutoC", "Join count (binary)", quote="People's dreams never end! -- Blackbeard")
_r("sagjk", "sagjk", "SpatAutoC", "Join count (k classes)", quote="I am the hope of the universe. -- Goku")
_r("sagjo", "sagjo", "SpatAutoC", "Join count statistic", quote="Those who break the rules are scum. -- Kakashi")
_r("sagjw", "sagjw", "SpatAutoC", "Join count (weighted)", quote="Keep moving forward. -- Eren")
_r("sagmo", "sagmo", "SpatAutoC", "Global Moran's I statistic", quote="Walk without rhythm. -- Fremen proverb")
_r("sagsp", "sagsp", "SpatAutoC", "Geary scatter plot", quote="El Psy Kongroo. -- Okabe")
_r("salbi", "salbi", "SpatAutoC", "Local bivariate Moran", quote="The spice must flow. -- Paul Atreides")
_r("saldf", "saldf", "SpatAutoC", "LISA differential test", quote="Not all those who wander are lost. -- Gandalf")
_r("saldi", "saldi", "SpatAutoC", "LISA discriminant", quote="Make it so. -- Picard")
_r("salge", "salge", "SpatAutoC", "Local Geary's C", quote="Scatter, Senbonzakura. -- Byakuya")
_r("salgo", "salgo", "SpatAutoC", "Local Getis-Ord G*", quote="One is all, all is one. -- Izumi")
_r("salml", "salml", "SpatAutoC", "Local multivariate LISA", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("salmo", "salmo", "SpatAutoC", "Local Moran's I (LISA)", quote="I will take a potato chip and eat it! -- Light")
_r("samcr", "samcr", "SpatAutoC", "Moran correlogram", quote="Equivalent exchange. -- Elric brothers")
_r("samob", "samob", "SpatAutoC", "Moran's I bias correction", quote="Bankai! -- Ichigo")
_r("samoe", "samoe", "SpatAutoC", "Moran's I expected value", quote="Dedicate your hearts! -- Erwin")
_r("samov", "samov", "SpatAutoC", "Moran's I variance", quote="I am justice! -- Light")
_r("samsp", "samsp", "SpatAutoC", "Moran scatter plot", quote="See you space cowboy. -- Spike")
_r("sapmo", "sapmo", "SpatAutoC", "Permutation Moran's I", quote="It's over 9000! -- Vegeta")
_r("saqmo", "saqmo", "SpatAutoC", "Quantile Moran scatter", quote="Believe it! -- Naruto")
_r("saraic", "saraic", "SAR", "SAR Akaike information criterion.", quote="")
_r("sarbic", "sarbic", "SAR", "SAR Bayesian information criterion.", quote="")
_r("sarboot", "sarboot", "SAR", "SAR bootstrap confidence interval for rho.", quote="")
_r("sarconv", "sarconv", "SAR", "SAR convergence diagnostic (eigenvalue method).", quote="")
_r("sardet", "sardet", "SAR", "SAR log-determinant  ln|I - rho*W|.", quote="")
_r("sarfilt", "sarfilt", "SAR", "SAR Cochrane-Orcutt-style spatial filter.", quote="")
_r("sargmm", "sargmm", "SAR", "SAR GMM estimator.", quote="")
_r("sarimp", "sarimp", "SAR", "SAR direct/indirect/total impact decomposition.", quote="")
_r("sarjac", "sarjac", "SAR", "SAR Jacobian term for log-likelihood.", quote="")
_r("sarlrt", "sarlrt", "SAR", "SAR likelihood-ratio test vs OLS.", quote="")
_r("sarml", "sarml", "SAR", "SAR maximum-likelihood estimation.", quote="")
_r("sarols", "sarols", "SAR", "SAR OLS-IV two-stage estimator.", quote="")
_r("sarr2", "sarr2", "SAR", "SAR pseudo-R-squared (Nagelkerke).", quote="")
_r("sarres", "sarres", "SAR", "SAR residual spatial autocorrelation check.", quote="")
_r("sarsc", "sarsc", "SAR", "SAR score / LM test for spatial lag.", quote="")
_r("sarsig", "sarsig", "SAR", "SAR sigma-squared ML estimate.", quote="")
_r("sarsim", "sarsim", "SAR", "SAR Monte-Carlo impact simulation.", quote="")
_r("sarspil", "sarspil", "SAR", "SAR spillover ratio (indirect/direct).", quote="")
_r("sarvar", "sarvar", "SAR", "SAR variance-covariance of estimator.", quote="")
_r("sarwald", "sarwald", "SAR", "SAR Wald test on spatial lag parameter.", quote="")
_r("sawad", "sawad", "SpatAutoC", "Spatial weight adaptive", quote="Get in the robot, Shinji! -- Misato")
_r("sawbl", "sawbl", "SpatAutoC", "Spatial weight block", quote="The world is cruel but beautiful. -- Mikasa")
_r("sawcm", "sawcm", "SpatAutoC", "Spatial weight comparison", quote="Growing old is a blessing. -- Rengoku")
_r("sawcn", "sawcn", "SpatAutoC", "Spatial weight connectivity", quote="I am here! -- All Might")
_r("sawco", "sawco", "SpatAutoC", "Spatial weight contiguity", quote="Set your heart ablaze! -- Rengoku")
_r("sawdl", "sawdl", "SpatAutoC", "Spatial weight Delaunay", quote="I must not fear. -- Litany Against Fear")
_r("sawdn", "sawdn", "SpatAutoC", "Spatial weight distance band", quote="Winter is coming. -- Stark motto")
_r("sawer", "sawer", "SpatAutoC", "Spatial weight error operator", quote="A Lannister always pays his debts. -- Tyrion")
_r("sawgb", "sawgb", "SpatAutoC", "Spatial weight Gabriel graph", quote="One does not simply walk. -- Boromir")
_r("sawhi", "sawhi", "SpatAutoC", "Spatial weight histogram", quote="Whatever happens, happens. -- Spike")
_r("sawis", "sawis", "SpatAutoC", "Spatial weight islands", quote="A lesson without pain is meaningless. -- Edward")
_r("sawiv", "sawiv", "SpatAutoC", "Spatial weight inverse distance", quote="Live long and prosper. -- Spock")
_r("sawkn", "sawkn", "SpatAutoC", "Spatial weight k-nearest", quote="Arise. -- Shadow Monarch")
_r("sawkr", "sawkr", "SpatAutoC", "Spatial weight kernel", quote="I am the one who knocks. -- Walter White")
_r("sawlg", "sawlg", "SpatAutoC", "Spatial weight lag operator", quote="Power is everything. -- Sung Jin-Woo")
_r("sawnb", "sawnb", "SpatAutoC", "Spatial weight normalization (binary)", quote="Breathe. -- Tanjiro")
_r(
    "sawnm",
    "sawnm",
    "SpatAutoC",
    "Spatial weight normalization (minmax)",
    quote="Even the smallest person can change the future. -- Galadriel",
)
_r("sawnr", "sawnr", "SpatAutoC", "Spatial weight normalization (row)", quote="Science! -- Jesse Pinkman")
_r("sawnv", "sawnv", "SpatAutoC", "Spatial weight normalization (variance)", quote="I alone level up. -- Sung Jin-Woo")
_r("sawpl", "sawpl", "SpatAutoC", "Spatial weight plot", quote="You should enjoy the detours. -- Ging")
_r("sawrg", "sawrg", "SpatAutoC", "Spatial weight regime", quote="Chaos is a ladder. -- Littlefinger")
_r("sawrl", "sawrl", "SpatAutoC", "Spatial weight relative graph", quote="Resistance is futile. -- Borg")
_r("sawsp", "sawsp", "SpatAutoC", "Spatial weight sparsity", quote="The needs of the many outweigh the few. -- Spock")
_r("sawtf", "sawtf", "SpatAutoC", "Spatial weight transformation", quote="I mustn't run away. -- Shinji")
_r("sawvr", "sawvr", "SpatAutoC", "Spatial weight Voronoi", quote="Desert power. -- Paul Muad'Dib")
_r("sbalt", "sbalt", "Spatial", "Alternating offer bargaining.", quote="")
_r("sbbias", "sbbias", "Spatial", "Spatial bootstrap bias correction.", quote="")
_r("sbblk", "sbblk", "Spatial", "Block bootstrap for spatial data.", quote="")
_r("sbbnd", "sbbnd", "Spatial", "Bootstrap confidence bands spatial field.", quote="")
_r("sbci", "sbci", "Spatial", "Spatial bootstrap percentile CI.", quote="")
_r("sbcrc", "sbcrc", "Spatial", "Circular block bootstrap spatial.", quote="")
_r("sbdel", "sbdel", "Spatial", "Delay cost bargaining.", quote="")
_r("sbkal", "sbkal", "Spatial", "Kalai-Smorodinsky bargaining.", quote="")
_r("sbleg", "sbleg", "Spatial", "Legislative bargaining spatial.", quote="")
_r("sbmlt", "sbmlt", "Spatial", "Multilateral bargaining spatial.", quote="")
_r("sbmnl", "sbmnl", "Spatial", "Monopoly bargaining spatial.", quote="")
_r("sbmov", "sbmov", "Spatial", "Moving block bootstrap spatial.", quote="")
_r("sbnsh", "sbnsh", "Spatial", "Nash bargaining solution spatial.", quote="")
_r("sbperm", "sbperm", "Spatial", "Spatial bootstrap permutation test.", quote="")
_r("sbpiv", "sbpiv", "Spatial", "Spatial bootstrap pivotal confidence interval.", quote="")
_r("sbprt", "sbprt", "Spatial", "Pareto bargaining frontier.", quote="")
_r("sbrub", "sbrub", "Spatial", "Rubinstein bargaining spatial.", quote="")
_r("sbsqp", "sbsqp", "Spatial", "Status quo point bargaining.", quote="")
_r("sbstt", "sbstt", "Spatial", "Stationary bootstrap spatial autocorrelated.", quote="")
_r("sbvar", "sbvar", "Spatial", "Spatial bootstrap variance estimation.", quote="")
_r("scdisp", "scdisp", "SCount", "Spatial count overdispersion test.", quote="")
_r("scnb", "scnb", "SCount", "Spatial negative-binomial regression.", quote="")
_r("scnblrt", "scnblrt", "SCount", "Spatial NB likelihood-ratio test.", quote="")
_r("scnbmf", "scnbmf", "SCount", "Spatial NB marginal effects.", quote="")
_r("scpaic", "scpaic", "SCount", "Spatial count model AIC.", quote="")
_r("scpbic", "scpbic", "SCount", "Spatial count model BIC.", quote="")
_r("scpboot", "scpboot", "SCount", "Spatial count bootstrap CI.", quote="")
_r("scpflx", "scpflx", "SCount", "Spatial Poisson fixed-effects panel.", quote="")
_r("scpmf", "scpmf", "SCount", "Spatial Poisson marginal effects.", quote="")
_r("scpmlm", "scpmlm", "SCount", "Spatial Poisson LM test.", quote="")
_r("scpois", "scpois", "SCount", "Spatial Poisson SAR regression.", quote="")
_r("scpprd", "scpprd", "SCount", "Spatial Poisson predicted counts.", quote="")
_r("scpwld", "scpwld", "SCount", "Spatial Poisson Wald test on rho.", quote="")
_r("sczinb", "sczinb", "SCount", "Spatial zero-inflated negative binomial.", quote="")
_r("sczip", "sczip", "SCount", "Spatial zero-inflated Poisson.", quote="")
_r("sdemaic", "sdemaic", "SDEM", "SDEM Akaike information criterion.", quote="")
_r("sdembic", "sdembic", "SDEM", "SDEM Bayesian information criterion.", quote="")
_r("sdemcf", "sdemcf", "SDEM", "SDEM common-factor restriction test.", quote="")
_r("sdemimp", "sdemimp", "SDEM", "SDEM direct/indirect/total impacts.", quote="")
_r("sdemjac", "sdemjac", "SDEM", "SDEM Jacobian term.", quote="")
_r("sdemlrt", "sdemlrt", "SDEM", "SDEM likelihood-ratio test vs SEM.", quote="")
_r("sdemml", "sdemml", "SDEM", "SDEM maximum-likelihood estimation.", quote="")
_r("sdemres", "sdemres", "SDEM", "SDEM residual autocorrelation check.", quote="")
_r("sdemvar", "sdemvar", "SDEM", "SDEM variance-covariance matrix.", quote="")
_r("sdemwld", "sdemwld", "SDEM", "SDEM Wald test on lambda.", quote="")
_r("sdm2slg", "sdm2slg", "SDM", "SDM 2-step lag estimator (Anselin).", quote="")
_r("sdmaic", "sdmaic", "SDM", "SDM Akaike information criterion.", quote="")
_r("sdmbic", "sdmbic", "SDM", "SDM Bayesian information criterion.", quote="")
_r("sdmboot", "sdmboot", "SDM", "SDM bootstrap CI for rho.", quote="")
_r("sdmcf", "sdmcf", "SDM", "SDM common-factor restriction test (Wald).", quote="")
_r("sdmconv", "sdmconv", "SDM", "SDM rho/lambda joint feasibility check.", quote="")
_r("sdmdet", "sdmdet", "SDM", "SDM log-determinant ln|I - rho*W|.", quote="")
_r("sdmflt", "sdmflt", "SDM", "SDM spatial filter transform.", quote="")
_r("sdmimp", "sdmimp", "SDM", "SDM direct/indirect/total impacts.", quote="")
_r("sdmjac", "sdmjac", "SDM", "SDM Jacobian log-determinant.", quote="")
_r("sdmlrt", "sdmlrt", "SDM", "SDM likelihood-ratio test vs SAR.", quote="")
_r("sdmml", "sdmml", "SDM", "SDM maximum-likelihood estimation.", quote="")
_r("sdmolsi", "sdmolsi", "SDM", "SDM OLS ignoring spatial component (baseline).", quote="")
_r("sdmr2", "sdmr2", "SDM", "SDM pseudo-R-squared.", quote="")
_r("sdmres", "sdmres", "SDM", "SDM residual autocorrelation check.", quote="")
_r("sdmsig", "sdmsig", "SDM", "SDM sigma-squared ML estimate.", quote="")
_r("sdmspil", "sdmspil", "SDM", "SDM spillover index (indirect/total).", quote="")
_r("sdmvar", "sdmvar", "SDM", "SDM variance-covariance matrix.", quote="")
_r("sdmwald", "sdmwald", "SDM", "SDM Wald test on rho.", quote="")
_r("sdmwx", "sdmwx", "SDM", "SDM spatially lagged X matrix WX.", quote="")
_r("sebcr", "sebcr", "SpatEpi2", "BYM (Besag-York-Mollie) model", quote="Believe it! -- Naruto")
_r("sebhr", "sebhr", "SpatEpi2", "Bayesian hierarchical rate", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("sebir", "sebir", "SpatEpi2", "ICAR spatial rate model", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("sebnr", "sebnr", "SpatEpi2", "Bayesian normal-normal rate", quote="Make it so. -- Picard")
_r(
    "sebpf",
    "sebpf",
    "SpatEpi2",
    "Bayesian spatial Poisson-gamma",
    quote="Not all those who wander are lost. -- Gandalf",
)
_r("sebpm", "sebpm", "SpatEpi2", "Bayesian Poisson mixture", quote="It's over 9000! -- Vegeta")
_r("sebzp", "sebzp", "SpatEpi2", "Bayesian zero-inflated Poisson", quote="Yare yare daze. -- Jotaro")
_r("secas", "secas", "SpatEpi2", "Case-control spatial study", quote="Walk without rhythm. -- Fremen proverb")
_r("secla", "secla", "SpatEpi2", "Bernoulli scan statistic", quote="One does not simply walk. -- Boromir")
_r("seclb", "seclb", "SpatEpi2", "Besag-Newell cluster test", quote="Bankai! -- Ichigo")
_r("seclc", "seclc", "SpatEpi2", "Cuzick-Edwards k-NN test", quote="Equivalent exchange. -- Elric brothers")
_r("secle", "secle", "SpatEpi2", "Elliptic scan statistic", quote="I am the one who knocks. -- Walter White")
_r("seclf", "seclf", "SpatEpi2", "FleXScan cluster detection", quote="Winter is coming. -- Stark motto")
_r("secli", "secli", "SpatEpi2", "Space-time scan (Kulldorff)", quote="Resistance is futile. -- Borg")
_r("seclk", "seclk", "SpatEpi2", "Kulldorff spatial scan", quote="Dedicate your hearts! -- Erwin")
_r("seclm", "seclm", "SpatEpi2", "Moving window scan", quote="I must not fear. -- Litany Against Fear")
_r("secln", "secln", "SpatEpi2", "Normal scan statistic", quote="Live long and prosper. -- Spock")
_r("seclo", "seclo", "SpatEpi2", "Oden's I-pop statistic", quote="Set your heart ablaze! -- Rengoku")
_r("seclp", "seclp", "SpatEpi2", "Poisson scan statistic", quote="Get in the robot, Shinji! -- Misato")
_r("seclr", "seclr", "SpatEpi2", "Rushton-Lolonis cluster", quote="Arise. -- Shadow Monarch")
_r("secls", "secls", "SpatEpi2", "Stone's test for clustering", quote="Go beyond! Plus Ultra! -- All Might")
_r("seclt", "seclt", "SpatEpi2", "Tango spatial clustering test", quote="I am justice! -- Light")
_r("seclw", "seclw", "SpatEpi2", "Lawson-Waller score test", quote="El Psy Kongroo. -- Okabe")
_r("seclx", "seclx", "SpatEpi2", "Prospective scan statistic", quote="Desert power. -- Paul Muad'Dib")
_r("secoh", "secoh", "SpatEpi2", "Cohort spatial study", quote="There is always hope. -- Aragorn")
_r("secop", "secop", "SpatEpi2", "Openshaw GAM cluster", quote="See you space cowboy. -- Spike")
_r("sedrv", "sedrv", "SpatEpi2", "Disease rate variation", quote="Breathe. -- Tanjiro")
_r("seebf", "seebf", "SpatEpi2", "Empirical Bayes full mapping", quote="I will take a potato chip and eat it! -- Light")
_r("seebg", "seebg", "SpatEpi2", "EB global smoothing", quote="Scatter, Senbonzakura. -- Byakuya")
_r("seebl", "seebl", "SpatEpi2", "EB local smoothing", quote="One is all, all is one. -- Izumi")
_r("seebr", "seebr", "SpatEpi2", "Empirical Bayes rate mapping", quote="Keep moving forward. -- Eren")
_r("seebt", "seebt", "SpatEpi2", "EB trend detection", quote="You should enjoy the detours. -- Ging")
_r("seeby", "seeby", "SpatEpi2", "EB James-Stein smoothing", quote="The spice must flow. -- Paul Atreides")
_r("seecl", "seecl", "SpatEpi2", "Ecological spatial study", quote="Engage. -- Picard")
_r("seexr", "seexr", "SpatEpi2", "Excess risk mapping", quote="The world is cruel but beautiful. -- Mikasa")
_r("segrf", "segrf", "SpatEpi2", "Geographic risk factors", quote="I alone level up. -- Sung Jin-Woo")
_r(
    "segrw",
    "segrw",
    "SpatEpi2",
    "Geographic risk weights",
    quote="Even the smallest person can change the future. -- Galadriel",
)
_r("sehyb", "sehyb", "SpatEpi2", "Hybrid spatial study", quote="Those who break the rules are scum. -- Kakashi")
_r("seibm", "seibm", "SpatEpi2", "Incidence-based mortality", quote="Power is everything. -- Sung Jin-Woo")
_r("semaic", "semaic", "SEM", "SEM Akaike information criterion.", quote="")
_r("sembic", "sembic", "SEM", "SEM Bayesian information criterion.", quote="")
_r("semboot", "semboot", "SEM", "SEM bootstrap CI for lambda.", quote="")
_r("semconv", "semconv", "SEM", "SEM convergence check (lambda feasibility).", quote="")
_r("semfgls", "semfgls", "SEM", "SEM feasible GLS estimator.", quote="")
_r("semflt", "semflt", "SEM", "SEM Cochrane-Orcutt spatial filter transform.", quote="")
_r("semgmm", "semgmm", "SEM", "SEM GMM (Kelejian-Prucha) estimator.", quote="")
_r("semhet", "semhet", "SEM", "SEM GMM heteroskedasticity-robust (KP-HET).", quote="")
_r("semjac", "semjac", "SEM", "SEM Jacobian ln|I - lambda*W|.", quote="")
_r("semkp", "semkp", "SEM", "SEM Kelejian-Prucha IV/2SLS estimator.", quote="")
_r("semlm", "semlm", "SEM", "SEM LM test for spatial error.", quote="")
_r("semlrt", "semlrt", "SEM", "SEM likelihood-ratio test vs OLS.", quote="")
_r("semml", "semml", "SEM", "SEM maximum-likelihood estimation.", quote="")
_r("semres", "semres", "SEM", "SEM residual autocorrelation (filtered residuals).", quote="")
_r("semrlm", "semrlm", "SEM", "SEM robust LM test for spatial error.", quote="")
_r("semsc", "semsc", "SEM", "SEM score test for spatial error parameter.", quote="")
_r("semsig", "semsig", "SEM", "SEM sigma-squared ML estimate.", quote="")
_r("semspec", "semspec", "SEM", "SEM common-factor restriction test (Wald).", quote="")
_r("semtl", "semtl", "SpatEpi2", "Mortality atlas construction", quote="Growing old is a blessing. -- Rengoku")
_r("semvar", "semvar", "SEM", "SEM variance-covariance matrix of estimator.", quote="")
_r("semwald", "semwald", "SEM", "SEM Wald test on spatial error parameter lambda.", quote="")
_r("seprf", "seprf", "SpatEpi2", "Probability risk field", quote="Science! -- Jesse Pinkman")
_r("sepvl", "sepvl", "SpatEpi2", "Prevalence spatial modeling", quote="A Lannister always pays his debts. -- Tyrion")
_r("serhm", "serhm", "SpatEpi2", "Relative hazard mapping", quote="Whatever happens, happens. -- Spike")
_r("serhr", "serhr", "SpatEpi2", "Relative hazard ratio spatial", quote="I mustn't run away. -- Shinji")
_r("sersk", "sersk", "SpatEpi2", "Relative risk spatial", quote="Chaos is a ladder. -- Littlefinger")
_r("sesir", "sesir", "SpatEpi2", "SIR spatial mapping", quote="People's dreams never end! -- Blackbeard")
_r("sesmr", "sesmr", "SpatEpi2", "SMR spatial mapping", quote="I am the hope of the universe. -- Goku")
_r("sespc", "sespc", "SpatEpi2", "Spatial persistence check", quote="A lesson without pain is meaningless. -- Edward")
_r("sespd", "sespd", "SpatEpi2", "Spatial diffusion model", quote="I am here! -- All Might")
_r("sespt", "sespt", "SpatEpi2", "Spatial temporal trend", quote="The needs of the many outweigh the few. -- Spock")
_r("sfaic", "sfaic", "SFilter", "Spatial filtering AIC model selection.", quote="")
_r("sfcv", "sfcv", "SFilter", "Cross-validated spatial filter selection.", quote="")
_r("sfeig", "sfeig", "SFilter", "Eigenvector spatial filtering (Tiefelsdorf & Griffith).", quote="")
_r("sfgetis", "sfgetis", "SFilter", "Getis spatial filtering approach.", quote="")
_r("sfglob", "sfglob", "SFilter", "Global eigenvector spatial filter.", quote="")
_r("sfloc", "sfloc", "SFilter", "Local eigenvector spatial filter.", quote="")
_r("sfmemb", "sfmemb", "SFilter", "MEM Bonferroni-corrected eigenvector selection.", quote="")
_r("sfmemn", "sfmemn", "SFilter", "MEM negative autocorrelation eigenvectors.", quote="")
_r("sfmemp", "sfmemp", "SFilter", "MEM positive autocorrelation eigenvectors.", quote="")
_r("sfmi", "sfmi", "SFilter", "Moran's I of spatially filtered residuals.", quote="")
_r("sfmoran", "sfmoran", "SFilter", "Moran eigenvector map (MEM) construction.", quote="")
_r("sforth", "sforth", "SFilter", "Orthogonality check of selected eigenvectors.", quote="")
_r("sfpve", "sfpve", "SFilter", "Proportion of variance explained by eigenvectors.", quote="")
_r("sfredu", "sfredu", "SFilter", "Spatial filter residual autocorrelation reduction.", quote="")
_r("sfsel", "sfsel", "SFilter", "Eigenvector selection (forward stepwise).", quote="")
_r("sgann", "sgann", "Spatial", "SGS annealing perturbation for reproduction.", quote="")
_r("sgcdf", "sgcdf", "Spatial", "SGS Gaussian anamorphosis / normal score transform.", quote="")
_r("sgcosm", "sgcosm", "Spatial", "SGS co-simulation with secondary variable.", quote="")
_r("sghist", "sghist", "Spatial", "SGS histogram reproduction diagnostic.", quote="")
_r("sglhd", "sglhd", "Spatial", "SGS Latin hypercube stratified sampling.", quote="")
_r("sgmean", "sgmean", "Spatial", "SGS expected value from multiple realizations.", quote="")
_r("sgnbr", "sgnbr", "Spatial", "SGS neighborhood search for conditioning data.", quote="")
_r("sgpath", "sgpath", "Spatial", "SGS random path through grid nodes.", quote="")
_r("sgpct", "sgpct", "Spatial", "SGS percentile map across realizations.", quote="")
_r("sgpost", "sgpost", "Spatial", "SGS posterior sampling with likelihood weighting.", quote="")
_r("sgprob", "sgprob", "Spatial", "SGS probability exceeding threshold.", quote="")
_r("sgreal", "sgreal", "Spatial", "SGS single realization generator.", quote="")
_r("sgsim", "sgsim", "Spatial", "Sequential Gaussian simulation core.", quote="")
_r("sgsim2", "sgsim2", "Spatial", "SGS with simple kriging system.", quote="")
_r("sgsim3", "sgsim3", "Spatial", "SGS with ordinary kriging system.", quote="")
_r("sguvar", "sguvar", "Spatial", "SGS uncertainty quantification via E-type.", quote="")
_r("sgval", "sgval", "Spatial", "SGS cross-validation leave-one-out.", quote="")
_r("sgvar", "sgvar", "Spatial", "SGS variance from ensemble of realizations.", quote="")
_r("shafp", "shafp", "KrigFilt", "Sharpen filter spatial", quote="You should enjoy the detours. -- Ging")
_r("sibnd", "sibnd", "Spatial", "SIS boundary uncertainty simulation.", quote="")
_r("sicls", "sicls", "Spatial", "SIS class proportion reproduction.", quote="")
_r("siconn", "siconn", "Spatial", "SIS connectivity analysis across realizations.", quote="")
_r("sicosm", "sicosm", "Spatial", "SIS co-indicator simulation.", quote="")
_r("simcs", "simcs", "Spatial", "SIS Markov chain categorical simulation.", quote="")
_r("simdk", "simdk", "Spatial", "SIS median indicator kriging variant.", quote="")
_r("sinusd", "sinusd", "GeoProcss", "Sinusoidal projection", quote="Live long and prosper. -- Spock")
_r("siorde", "siorde", "Spatial", "SIS order relation correction.", quote="")
_r("sipath", "sipath", "Spatial", "SIS random path through indicators.", quote="")
_r("sipost", "sipost", "Spatial", "SIS posterior categorical probability.", quote="")
_r("siprob", "siprob", "Spatial", "SIS indicator probability map output.", quote="")
_r("sireal", "sireal", "Spatial", "SIS categorical realization generator.", quote="")
_r("sisim", "sisim", "Spatial", "Sequential indicator simulation core.", quote="")
_r("sisim2", "sisim2", "Spatial", "SIS with multiple threshold classes.", quote="")
_r("sitrans", "sitrans", "Spatial", "SIS transition probability matrix.", quote="")
_r("sivar", "sivar", "Spatial", "SIS variogram indicator reproduction.", quote="")
_r("skflt", "skflt", "KrigFilt", "Simple kriging filter", quote="People's dreams never end! -- Blackbeard")
_r("slcsp", "slcsp", "SpatialPat", "Slice sampler spatial", quote="The world is cruel but beautiful. -- Mikasa")
_r("slxaic", "slxaic", "SLX", "SLX Akaike information criterion.", quote="")
_r("slxbic", "slxbic", "SLX", "SLX Bayesian information criterion.", quote="")
_r("slxboot", "slxboot", "SLX", "SLX bootstrap CI for theta.", quote="")
_r("slxflt", "slxflt", "SLX", "SLX spatial filter (de-mean with WX).", quote="")
_r("slximp", "slximp", "SLX", "SLX local/global impacts decomposition.", quote="")
_r("slxols", "slxols", "SLX", "SLX OLS with spatially lagged regressors.", quote="")
_r("slxres", "slxres", "SLX", "SLX residual Moran test.", quote="")
_r("slxvar", "slxvar", "SLX", "SLX variance-covariance via OLS formula.", quote="")
_r("slxwald", "slxwald", "SLX", "SLX Wald test on theta = 0.", quote="")
_r("slxwx", "slxwx", "SLX", "SLX WX construction and summary.", quote="")
_r("smsim", "smsim", "SpatialPat", "Simulated annealing spatial", quote="I must not fear. -- Litany Against Fear")
_r("snsim", "snsim", "SpatialPat", "SNESIM MPS method", quote="Winter is coming. -- Stark motto")
_r("soagg", "soagg", "SoilSp", "Aggregate stability soil", quote="Arise. -- Shadow Monarch")
_r("soalk", "soalk", "SoilSp", "Alkalinity soil spatial", quote="Believe it! -- Naruto")
_r("soaws", "soaws", "SoilSp", "Available water storage", quote="Make it so. -- Picard")
_r("sobd", "sobd", "SoilSp", "Bulk density soil", quote="Winter is coming. -- Stark motto")
_r("sobfp", "sobfp", "KrigFilt", "Sobel filter spatial", quote="A lesson without pain is meaningless. -- Edward")
_r("sobor", "sobor", "SoilSp", "Boron soil spatial", quote="Set your heart ablaze! -- Rengoku")
_r("socal", "socal", "SoilSp", "Calcium soil spatial", quote="Dedicate your hearts! -- Erwin")
_r("socec", "socec", "SoilSp", "Cation exchange capacity", quote="The spice must flow. -- Paul Atreides")
_r("socla", "socla", "SoilSp", "Clay content spatial", quote="Equivalent exchange. -- Elric brothers")
_r("soclm", "soclm", "SoilSp", "Soil classification map", quote="Get in the robot, Shinji! -- Misato")
_r("socmp", "socmp", "SoilSp", "Soil compaction spatial", quote="El Psy Kongroo. -- Okabe")
_r("socop", "socop", "SoilSp", "Copper soil spatial", quote="El Psy Kongroo. -- Okabe")
_r("socrb", "socrb", "SoilSp", "Carbon stock soil", quote="Winter is coming. -- Stark motto")
_r("soec", "soec", "SoilSp", "Electrical conductivity soil", quote="Get in the robot, Shinji! -- Misato")
_r("soero", "soero", "SoilSp", "Erosion rate soil", quote="Yare yare daze. -- Jotaro")
_r("soesp", "soesp", "SoilSp", "Exchangeable sodium percentage", quote="Not all those who wander are lost. -- Gandalf")
_r("sofcl", "sofcl", "SoilSp", "Field capacity soil", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("sogly", "sogly", "SoilSp", "Gully erosion spatial", quote="Bankai! -- Ichigo")
_r("sogrv", "sogrv", "SoilSp", "Gravel content spatial", quote="El Psy Kongroo. -- Okabe")
_r("sohc", "sohc", "SoilSp", "Hydraulic conductivity soil", quote="Get in the robot, Shinji! -- Misato")
_r("sohum", "sohum", "SoilSp", "Humus content spatial", quote="Winter is coming. -- Stark motto")
_r("soinf", "soinf", "SoilSp", "Infiltration rate soil", quote="Believe it! -- Naruto")
_r("soirn", "soirn", "SoilSp", "Iron soil spatial", quote="Equivalent exchange. -- Elric brothers")
_r("somag", "somag", "SoilSp", "Magnesium soil spatial", quote="I am justice! -- Light")
_r("soman", "soman", "SoilSp", "Manganese soil spatial", quote="Go beyond! Plus Ultra! -- All Might")
_r("somol", "somol", "SoilSp", "Molybdenum soil spatial", quote="Arise. -- Shadow Monarch")
_r("sontg", "sontg", "SoilSp", "Total nitrogen soil", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("soorg", "soorg", "SoilSp", "Organic carbon spatial", quote="Set your heart ablaze! -- Rengoku")
_r("soper", "soper", "SoilSp", "Permeability soil", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("soph", "soph", "SoilSp", "Soil pH spatial", quote="I am the one who knocks. -- Walter White")
_r("sophs", "sophs", "SoilSp", "Total phosphorus soil", quote="It's over 9000! -- Vegeta")
_r("sopor", "sopor", "SoilSp", "Porosity soil spatial", quote="I am the one who knocks. -- Walter White")
_r("sopot", "sopot", "SoilSp", "Potassium soil spatial", quote="Yare yare daze. -- Jotaro")
_r("sored", "sored", "SoilSp", "Soil redistribution", quote="See you space cowboy. -- Spike")
_r("soril", "soril", "SoilSp", "Rill erosion spatial", quote="Go beyond! Plus Ultra! -- All Might")
_r("sorun", "sorun", "SoilSp", "Runoff coefficient soil", quote="It's over 9000! -- Vegeta")
_r("sosal", "sosal", "SoilSp", "Salinity spatial", quote="Make it so. -- Picard")
_r("sosar", "sosar", "SoilSp", "Sodium adsorption ratio", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("soseq", "soseq", "SoilSp", "Carbon sequestration soil", quote="I am the one who knocks. -- Walter White")
_r("sosht", "sosht", "SoilSp", "Sheet erosion spatial", quote="Equivalent exchange. -- Elric brothers")
_r("soslt", "soslt", "SoilSp", "Silt content spatial", quote="Go beyond! Plus Ultra! -- All Might")
_r("sosnd", "sosnd", "SoilSp", "Sand content spatial", quote="See you space cowboy. -- Spike")
_r("sosom", "sosom", "SoilSp", "Soil organic matter spatial", quote="Arise. -- Shadow Monarch")
_r("sostr", "sostr", "SoilSp", "Soil structure index", quote="Set your heart ablaze! -- Rengoku")
_r("sosul", "sosul", "SoilSp", "Sulfur soil spatial", quote="Bankai! -- Ichigo")
_r("sotex", "sotex", "SoilSp", "Soil texture classification", quote="Bankai! -- Ichigo")
_r("sownd", "sownd", "SoilSp", "Wind erosion soil", quote="Dedicate your hearts! -- Erwin")
_r("sowp", "sowp", "SoilSp", "Wilting point soil", quote="Not all those who wander are lost. -- Gandalf")
_r("sowrc", "sowrc", "SoilSp", "Water retention curve", quote="The spice must flow. -- Paul Atreides")
_r("sowtr", "sowtr", "SoilSp", "Water erosion soil", quote="I am justice! -- Light")
_r("sozin", "sozin", "SoilSp", "Zinc soil spatial", quote="See you space cowboy. -- Spike")
_r("spcani", "spcani", "Spatial", "Spectral anisotropic field generation.", quote="")
_r("spccir", "spccir", "Spatial", "Spectral circulant embedding simulation.", quote="")
_r("spccoh", "spccoh", "Spatial", "Spectral coherence simulation for cross-fields.", quote="")
_r("spcenv", "spcenv", "Spatial", "Spectral envelope and instantaneous amplitude.", quote="")
_r("spcext", "spcext", "Spatial", "Spectral exact circulant embedding on grid.", quote="")
_r("spcflt", "spcflt", "Spatial", "Spectral filter design for field simulation.", quote="")
_r("spchst", "spchst", "Spatial", "Spectral simulation histogram transform.", quote="")
_r("spcmat", "spcmat", "Spatial", "Spectral Matern field via Whittle approximation.", quote="")
_r("spcnst", "spcnst", "Spatial", "Spectral nonstationary field simulation.", quote="")
_r("spcphs", "spcphs", "Spatial", "Spectral random phase field generation.", quote="")
_r("spcpow", "spcpow", "Spatial", "Spectral power-law random field simulation.", quote="")
_r("spcrng", "spcrng", "Spatial", "Spectral simulation range parameter sweep.", quote="")
_r("spcsim", "spcsim", "Spatial", "Spectral FFT-based Gaussian random field sim.", quote="")
_r("spcsim2", "spcsim2", "Spatial", "Spectral simulation 2D power spectrum.", quote="")
_r("spcsts", "spcsts", "Spatial", "Spectral stationary field via Wood-Chan.", quote="")
_r("splbi", "splbi", "KrigFilt", "Bicubic spline interpolation", quote="El Psy Kongroo. -- Okabe")
_r("splfn", "splfn", "KrigFilt", "Spline interpolation spatial", quote="See you space cowboy. -- Spike")
_r("splgaic", "splgaic", "SProbit", "Spatial logit AIC.", quote="")
_r("splgbic", "splgbic", "SProbit", "Spatial logit BIC.", quote="")
_r("splgtlm", "splgtlm", "SProbit", "Spatial logit LM test.", quote="")
_r("splgtmf", "splgtmf", "SProbit", "Spatial logit marginal effects.", quote="")
_r("splgtml", "splgtml", "SProbit", "Spatial logit ML with GHK simulator.", quote="")
_r("splogit", "splogit", "SProbit", "Spatial logit estimation.", quote="")
_r("spltps", "spltps", "KrigFilt", "Thin-plate spline interpolation", quote="Set your heart ablaze! -- Rengoku")
_r("sppaic", "sppaic", "SPanel", "Spatial panel AIC.", quote="")
_r("sppbe", "sppbe", "SPanel", "Spatial panel between estimator.", quote="")
_r("sppbic", "sppbic", "SPanel", "Spatial panel BIC.", quote="")
_r("sppboot", "sppboot", "SPanel", "Spatial panel bootstrap.", quote="")
_r("sppcov", "sppcov", "SPanel", "Spatial panel covariance structure.", quote="")
_r("sppde", "sppde", "SPanel", "Spatial panel within-demeaning transform.", quote="")
_r("sppdiag", "sppdiag", "SPanel", "Spatial panel diagnostics (LM tests).", quote="")
_r("sppdyn", "sppdyn", "SPanel", "Spatial dynamic panel (lagged dependent variable).", quote="")
_r("sppfe", "sppfe", "SPanel", "Spatial panel fixed effects (within) estimator.", quote="")
_r("sppgmm", "sppgmm", "SPanel", "Spatial panel GMM estimator.", quote="")
_r("spphaus", "spphaus", "SPanel", "Spatial panel Hausman test (FE vs RE).", quote="")
_r("sppiv", "sppiv", "SPanel", "Spatial panel IV/2SLS estimator.", quote="")
_r("spplm", "spplm", "SPanel", "Spatial panel LM test for spatial lag in panel.", quote="")
_r("sppraic", "sppraic", "SProbit", "Spatial probit AIC.", quote="")
_r("spprbic", "spprbic", "SProbit", "Spatial probit BIC.", quote="")
_r("sppre", "sppre", "SPanel", "Spatial panel random effects estimator.", quote="")
_r("sppres", "sppres", "SPanel", "Spatial panel residual Moran test.", quote="")
_r("spprgmm", "spprgmm", "SProbit", "Spatial probit GMM estimator.", quote="")
_r("spprlm", "spprlm", "SProbit", "Spatial probit LM test.", quote="")
_r("spprmf", "spprmf", "SProbit", "Spatial probit marginal effects.", quote="")
_r("spprml", "spprml", "SProbit", "Spatial probit ML with GHK simulator.", quote="")
_r("spprob", "spprob", "SProbit", "Spatial probit (SAR probit) estimation.", quote="")
_r("spprocc", "spprocc", "SProbit", "Spatial probit ROC-AUC.", quote="")
_r("spprprd", "spprprd", "SProbit", "Spatial probit predicted probabilities.", quote="")
_r("sppsac", "sppsac", "SPanel", "Spatial panel SAC (lag+error) estimator.", quote="")
_r("sppsar", "sppsar", "SPanel", "Spatial panel SAR (spatial lag) estimator.", quote="")
_r("sppsdm", "sppsdm", "SPanel", "Spatial panel SDM estimator.", quote="")
_r("sppsem", "sppsem", "SPanel", "Spatial panel SEM estimator.", quote="")
_r("sppvar", "sppvar", "SPanel", "Spatial panel variance components.", quote="")
_r("sprmfdi", "sprmfdi", "SProbit", "Spatial probit direct/indirect MEs.", quote="")
_r("sprord", "sprord", "SProbit", "Spatial ordered probit.", quote="")
_r("sprtobt", "sprtobt", "SProbit", "Spatial Tobit model.", quote="")
_r("sptfx", "sptfx", "SProbit", "Spatial probit fixed-effects panel.", quote="")
_r("sptrx", "sptrx", "SProbit", "Spatial probit random-effects panel.", quote="")
_r("sptsmp", "sptsmp", "GeoProcss", "Spatiotemporal sampling", quote="Whatever happens, happens. -- Spike")
_r("sr2sl", "sr2sl", "SpatReg2", "Spatial 2SLS estimator", quote="One is all, all is one. -- Izumi")
_r("srber", "srber", "SpatReg2", "Bayesian spatial error", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("srblg", "srblg", "SpatReg2", "Bayesian spatial lag", quote="Believe it! -- Naruto")
_r("srbml", "srbml", "SpatReg2", "Spatial Bayesian ML", quote="Make it so. -- Picard")
_r("srbnb", "srbnb", "SpatReg2", "Bayesian spatial neg-binomial", quote="Bankai! -- Ichigo")
_r("srbpn", "srbpn", "SpatReg2", "Bayesian spatial Poisson", quote="I am justice! -- Light")
_r("srbpr", "srbpr", "SpatReg2", "Bayesian spatial probit", quote="Yare yare daze. -- Jotaro")
_r("srbsd", "srbsd", "SpatReg2", "Bayesian spatial Durbin", quote="It's over 9000! -- Vegeta")
_r("srbtb", "srbtb", "SpatReg2", "Bayesian spatial tobit", quote="Dedicate your hearts! -- Erwin")
_r("sremr", "sremr", "SpatReg2", "Spatial error model (SEM)", quote="Engage. -- Picard")
_r("srgam", "srgam", "SpatReg2", "Spatial GAM", quote="Whatever happens, happens. -- Spike")
_r("srglm", "srglm", "SpatReg2", "Spatial GLM", quote="You should enjoy the detours. -- Ging")
_r("srgmm", "srgmm", "SpatReg2", "Spatial GMM estimator", quote="The spice must flow. -- Paul Atreides")
_r("srgns", "srgns", "SpatReg2", "Spatial GNS model", quote="Keep moving forward. -- Eren")
_r("srgwb", "srgwb", "SpatReg2", "GWR bandwidth selection", quote="See you space cowboy. -- Spike")
_r("srgwd", "srgwd", "SpatReg2", "GWR diagnostics", quote="Winter is coming. -- Stark motto")
_r("srgwe", "srgwe", "SpatReg2", "GWR collinearity diagnostics", quote="I am the one who knocks. -- Walter White")
_r("srgwk", "srgwk", "SpatReg2", "GWR kernel weights", quote="El Psy Kongroo. -- Okabe")
_r("srgwl", "srgwl", "SpatReg2", "GWR local coefficients", quote="Go beyond! Plus Ultra! -- All Might")
_r("srgwp", "srgwp", "SpatReg2", "GWR prediction map", quote="Set your heart ablaze! -- Rengoku")
_r("srgwr", "srgwr", "SpatReg2", "Geographically weighted regression", quote="Equivalent exchange. -- Elric brothers")
_r("srgws", "srgws", "SpatReg2", "GWR significance test", quote="Arise. -- Shadow Monarch")
_r(
    "srhhm",
    "srhhm",
    "SpatReg2",
    "Spatial hierarchical model",
    quote="Even the smallest person can change the future. -- Galadriel",
)
_r(
    "srhm2",
    "srhm2",
    "SpatReg2",
    "Spatial hierarchical 2-level",
    quote="The needs of the many outweigh the few. -- Spock",
)
_r(
    "srhm3",
    "srhm3",
    "SpatReg2",
    "Spatial hierarchical 3-level",
    quote="A lesson without pain is meaningless. -- Edward",
)
_r("srkgr", "srkgr", "SpatReg2", "Spatial kelejian-prucha model", quote="Scatter, Senbonzakura. -- Byakuya")
_r("srlgr", "srlgr", "SpatReg2", "Spatial lag model (SAR)", quote="There is always hope. -- Aragorn")
_r("srlmt", "srlmt", "SpatReg2", "LM test spatial lag vs error", quote="A Lannister always pays his debts. -- Tyrion")
_r("srmgb", "srmgb", "SpatReg2", "MGWR bandwidth", quote="Live long and prosper. -- Spock")
_r("srmgc", "srmgc", "SpatReg2", "MGWR coefficients", quote="One does not simply walk. -- Boromir")
_r("srmgw", "srmgw", "SpatReg2", "Multi-scale GWR (MGWR)", quote="Get in the robot, Shinji! -- Misato")
_r("srmix", "srmix", "SpatReg2", "Spatial mixed model", quote="I am here! -- All Might")
_r("srml", "srml", "SpatReg2", "Spatial ML estimator", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("srmns", "srmns", "SpatReg2", "Spatial Manski model", quote="I will take a potato chip and eat it! -- Light")
_r("srolm", "srolm", "SpatReg2", "Spatial OLS model", quote="Walk without rhythm. -- Fremen proverb")
_r("srpdm", "srpdm", "SpatReg2", "Spatial panel Durbin", quote="I alone level up. -- Sung Jin-Woo")
_r("srper", "srper", "SpatReg2", "Spatial panel error", quote="Breathe. -- Tanjiro")
_r("srpfe", "srpfe", "SpatReg2", "Spatial panel fixed effects", quote="The world is cruel but beautiful. -- Mikasa")
_r("srplg", "srplg", "SpatReg2", "Spatial panel lag", quote="Science! -- Jesse Pinkman")
_r("srpnl", "srpnl", "SpatReg2", "Spatial panel model", quote="Desert power. -- Paul Muad'Dib")
_r("srpre", "srpre", "SpatReg2", "Spatial panel random effects", quote="Chaos is a ladder. -- Littlefinger")
_r("srqml", "srqml", "SpatReg2", "Spatial quasi-ML estimator", quote="Not all those who wander are lost. -- Gandalf")
_r("srqnt", "srqnt", "SpatReg2", "Spatial quantile regression", quote="I mustn't run away. -- Shinji")
_r("srrob", "srrob", "SpatReg2", "Spatial robust regression", quote="Growing old is a blessing. -- Rengoku")
_r("srsac", "srsac", "SpatReg2", "Spatial SARAR model", quote="I am the hope of the universe. -- Goku")
_r("srsdm", "srsdm", "SpatReg2", "Spatial Durbin model (SDM)", quote="Those who break the rules are scum. -- Kakashi")
_r("srsem", "srsem", "SpatReg2", "Spatial Durbin error model (SDEM)", quote="People's dreams never end! -- Blackbeard")
_r("srslx", "srslx", "SpatReg2", "Spatial lag of X model (SLX)", quote="Resistance is futile. -- Borg")
_r("srswm", "srswm", "SpatReg2", "Spatial weight matrix regression", quote="Power is everything. -- Sung Jin-Woo")
_r("srsxe", "srsxe", "SpatReg2", "SLX with spatial error", quote="I must not fear. -- Litany Against Fear")
_r("stanis", "stanis", "Spatial", "Space-time anisotropic covariance.", quote="")
_r("stbkrg", "stbkrg", "Spatial", "Space-time block kriging volume support.", quote="")
_r("stcfn", "stcfn", "Spatial", "Space-time correlation function evaluation.", quote="")
_r("stch", "stch", "Spatial", "Cressie-Huang space-time covariance class.", quote="")
_r("stckrg", "stckrg", "Spatial", "Space-time co-kriging two variables.", quote="")
_r("stcov", "stcov", "Spatial", "Space-time separable covariance model.", quote="")
_r("stdcv", "stdcv", "Spatial", "Space-time covariance differentiability check.", quote="")
_r("stdfp", "stdfp", "KrigFilt", "Standard deviation filter", quote="Science! -- Jesse Pinkman")
_r("sterea", "sterea", "GeoProcss", "Stereographic projection", quote="Set your heart ablaze! -- Rengoku")
_r("stexp", "stexp", "Spatial", "Space-time product-sum covariance.", quote="")
_r("stfit", "stfit", "Spatial", "Space-time covariance model fitting.", quote="")
_r("stgnt", "stgnt", "Spatial", "Gneiting space-time covariance model.", quote="")
_r("stiag", "stiag", "Spatial", "Iaco-Cesare space-time covariance.", quote="")
_r("stinfo", "stinfo", "Spatial", "Space-time covariance information criteria.", quote="")
_r("stkanim", "stkanim", "Spatial", "Space-time kriging animation frames.", quote="")
_r("stkdiag", "stkdiag", "Spatial", "Space-time kriging system diagnostics.", quote="")
_r("stkint", "stkint", "Spatial", "Space-time kriging temporal interpolation.", quote="")
_r("stklhs", "stklhs", "Spatial", "Space-time kriging leave-h-out validation.", quote="")
_r("stkmap", "stkmap", "Spatial", "Space-time kriging prediction map grid.", quote="")
_r("stknbr", "stknbr", "Spatial", "Space-time kriging neighborhood selection.", quote="")
_r("stkprd", "stkprd", "Spatial", "Space-time kriging temporal prediction.", quote="")
_r("stkrg", "stkrg", "Spatial", "Space-time ordinary kriging prediction.", quote="")
_r("stkrgs", "stkrgs", "Spatial", "Space-time simple kriging prediction.", quote="")
_r("stkrgv", "stkrgv", "Spatial", "Space-time kriging variance surface.", quote="")
_r("stkse", "stkse", "Spatial", "Space-time kriging standard error map.", quote="")
_r("stkval", "stkval", "Spatial", "Space-time kriging cross-validation LOOCV.", quote="")
_r("stlin", "stlin", "Spatial", "Linear space-time covariance combination.", quote="")
_r("stmat", "stmat", "Spatial", "Space-time Matern covariance.", quote="")
_r("stmet", "stmet", "Spatial", "Space-time integrated metric covariance.", quote="")
_r("stnsep", "stnsep", "Spatial", "Space-time non-separable covariance.", quote="")
_r("stper", "stper", "Spatial", "Periodic space-time covariance model.", quote="")
_r("stpor", "stpor", "Spatial", "Porcu-Gregori space-time covariance.", quote="")
_r("stprod", "stprod", "Spatial", "Space-time product covariance model.", quote="")
_r("stsim", "stsim", "Spatial", "Space-time random field simulation.", quote="")
_r("stsmm", "stsmm", "Spatial", "Space-time sum-metric covariance.", quote="")
_r("stukrg", "stukrg", "Spatial", "Space-time universal kriging with drift.", quote="")
_r("stvario", "stvario", "Spatial", "Space-time variogram estimation.", quote="")
_r("svag2", "svag2", "Spatial", "2D amendment agenda", quote="")
_r("svagn", "svagn", "Spatial", "1D agenda setting equilibrium", quote="")
_r("svals", "svals", "Spatial", "Algebraic-distance (dot product) directional.", quote="")
_r("svamn", "svamn", "Spatial", "Sequential amendment procedure", quote="")
_r("svang", "svang", "Spatial", "Angular proximity model.", quote="")
_r("svasp", "svasp", "Spatial", "Asymmetric utility with salience weights.", quote="")
_r("svblp", "svblp", "Spatial", "Bliss point estimation", quote="")
_r("svblt", "svblt", "Spatial", "Boltzmann (softmax) spatial voting", quote="")
_r("svbnk", "svbnk", "Spatial", "Banks set computation", quote="")
_r("svbnz", "svbnz", "Spatial", "Banzhaf power index spatial", quote="")
_r("svbpl", "svbpl", "Spatial", "Bayesian spatial vote probability.", quote="")
_r("svbrd", "svbrd", "Spatial", "Borda count in spatial model", quote="")
_r("svbut", "svbut", "Spatial", "Bimodal utility combining peaks.", quote="")
_r("svcal", "svcal", "Spatial", "Calvert uncertainty model", quote="")
_r("svchy", "svchy", "Spatial", "Cauchy kernel spatial voting", quote="")
_r("svcle", "svcle", "Spatial", "Coalition equilibrium (Schofield)", quote="")
_r("svclh", "svclh", "Spatial", "Heart of spatial game", quote="")
_r("svcli", "svcli", "Spatial", "Optimal cutting line", quote="")
_r("svclk", "svclk", "Spatial", "Coalition kernel set", quote="")
_r("svclp", "svclp", "Spatial", "Cutting plane in 3D", quote="")
_r("svclr", "svclr", "Spatial", "Condorcet loser identification", quote="")
_r("svcls", "svcls", "Spatial", "Minimum winning coalition size", quote="")
_r("svclv", "svclv", "Spatial", "Coalition value in spatial game", quote="")
_r("svcly", "svcly", "Spatial", "Yolk of spatial game", quote="")
_r("svcm2", "svcm2", "Spatial", "2D committee decision", quote="")
_r("svcmp", "svcmp", "Spatial", "Committee median voter model", quote="")
_r("svcos", "svcos", "Spatial", "Cosine similarity voting model.", quote="")
_r("svcpw", "svcpw", "Spatial", "Copeland spatial winner", quote="")
_r("svcut", "svcut", "Spatial", "City-block/L1 utility function.", quote="")
_r("svcwn", "svcwn", "Spatial", "Condorcet winner test", quote="")
_r("svcyc", "svcyc", "Spatial", "Condorcet cycle detection", quote="")
_r("svdcm", "svdcm", "Spatial", "Discounting model (Matthews 1979).", quote="")
_r("svdft", "svdft", "Spatial", "Directional with intensity term.", quote="")
_r("svdgn", "svdgn", "Spatial", "Deegan-Packel power index", quote="")
_r("svdir", "svdir", "Spatial", "Directional voting model (Rabinowitz-Macdonald).", quote="")
_r("svdm2", "svdm2", "Spatial", "Second dimensionality test", quote="")
_r("svdmt", "svdmt", "Spatial", "Dimensionality test for spatial data", quote="")
_r("svdot", "svdot", "Spatial", "Dot-product directional vote.", quote="")
_r("svdrc", "svdrc", "Spatial", "Pure directional for categorical.", quote="")
_r("svdrv", "svdrv", "Spatial", "Directional voting model (Rabinowitz-Macdonald)", quote="")
_r("svdsc", "svdsc", "Spatial", "Discounted directional utility", quote="")
_r("svdsi", "svdsi", "SpatialPat", "SVD-based simulation", quote="Bankai! -- Ichigo")
_r("svdut", "svdut", "Spatial", "Distance-decay utility wrapper.", quote="")
_r("sveln", "sveln", "Spatial", "Elbow method for dimensions", quote="")
_r("sveut", "sveut", "Spatial", "Exponential decay utility.", quote="")
_r("svexp", "svexp", "Spatial", "Exponential vote probability decay.", quote="")
_r("svfut", "svfut", "Spatial", "Fuzzy utility with membership degrees.", quote="")
_r("svfxl", "svfxl", "Spatial", "Fixed-effect logit panel spatial.", quote="")
_r("svgau", "svgau", "Spatial", "Gaussian spatial utility function", quote="")
_r("svgmp", "svgmp", "Spatial", "Gompertz spatial vote probability.", quote="")
_r("svgut", "svgut", "Spatial", "Gaussian utility function for spatial voting.", quote="")
_r("svht2", "svht2", "Spatial", "Two-party Hotelling spatial competition", quote="")
_r("svht3", "svht3", "Spatial", "Three-candidate spatial equilibrium", quote="")
_r("svhtd", "svhtd", "Spatial", "Hotelling-Downs convergence equilibrium", quote="")
_r("svhtm", "svhtm", "Spatial", "Multi-candidate Hotelling model", quote="")
_r("svhut", "svhut", "Spatial", "Hierarchical utility with nested issues.", quote="")
_r("svhyb", "svhyb", "Spatial", "Hybrid proximity-valence model.", quote="")
_r("svip1", "svip1", "Spatial", "1D ideal point estimation", quote="")
_r("svip2", "svip2", "Spatial", "2D ideal point estimation", quote="")
_r("svipa", "svipa", "Spatial", "Adaptive ideal point estimation", quote="")
_r("svipb", "svipb", "Spatial", "Bayesian ideal point posterior", quote="")
_r("svipe", "svipe", "Spatial", "EM ideal point estimation", quote="")
_r("svipk", "svipk", "Spatial", "Kernel smoothed ideal point", quote="")
_r("svipm", "svipm", "Spatial", "MLE ideal point estimation", quote="")
_r("svipn", "svipn", "Spatial", "Normal ideal point model", quote="")
_r("sviut", "sviut", "Spatial", "Ideal-point utility deviation.", quote="")
_r("svjhn", "svjhn", "Spatial", "Johnston power index", quote="")
_r("svjut", "svjut", "Spatial", "Joint utility over multiple voters.", quote="")
_r("svkut", "svkut", "Spatial", "Kernel-smoothed utility surface.", quote="")
_r("svlap", "svlap", "Spatial", "Laplace distribution vote probability.", quote="")
_r("svlgp", "svlgp", "Spatial", "Logit spatial vote probability.", quote="")
_r("svlgv", "svlgv", "Spatial", "Logit spatial voting probability", quote="")
_r("svlin", "svlin", "Spatial", "Linear spatial utility function", quote="")
_r("svlss", "svlss", "Spatial", "Spatial loss function (quadratic/city-block)", quote="")
_r("svlut", "svlut", "Spatial", "Linear utility function for spatial voting.", quote="")
_r("svmix", "svmix", "Spatial", "Mixed proximity-directional model.", quote="")
_r("svmnl", "svmnl", "Spatial", "Multinomial spatial choice model", quote="")
_r("svmp2", "svmp2", "Spatial", "Multi-party 2D equilibrium", quote="")
_r("svmpc", "svmpc", "Spatial", "Multi-party spatial competition", quote="")
_r("svmpn", "svmpn", "Spatial", "Multi-party Nash equilibrium", quote="")
_r("svmut", "svmut", "Spatial", "Mixed (quadratic+Gaussian) utility.", quote="")
_r("svmv2", "svmv2", "Spatial", "Median voter in 2D (Plott conditions)", quote="")
_r("svmvt", "svmvt", "Spatial", "Median voter theorem test", quote="")
_r("svmxl", "svmxl", "Spatial", "Mixed logit spatial vote", quote="")
_r("svmxu", "svmxu", "Spatial", "Mixed-norm spatial utility", quote="")
_r("svnbg", "svnbg", "Spatial", "Nash bargaining in spatial game", quote="")
_r("svnlp", "svnlp", "Spatial", "Normal distribution vote probability.", quote="")
_r("svnrm", "svnrm", "Spatial", "Normal kernel vote probability", quote="")
_r("svnrv", "svnrv", "Spatial", "Normal vector to cutting line", quote="")
_r("svnst", "svnst", "Spatial", "Nested logit spatial model.", quote="")
_r("svnut", "svnut", "Spatial", "Nonlinear utility via polynomial.", quote="")
_r("svord", "svord", "Spatial", "Ordered logit spatial model.", quote="")
_r("svout", "svout", "Spatial", "Ordered categorical utility.", quote="")
_r("svpbp", "svpbp", "Spatial", "Probit spatial vote probability.", quote="")
_r("svpd2", "svpd2", "Spatial", "Proximity-directional 2D comparison.", quote="")
_r("svpft", "svpft", "Spatial", "Proximity with fatigue function.", quote="")
_r("svpl2", "svpl2", "Spatial", "2D spatial polarization", quote="")
_r("svplc", "svplc", "Spatial", "Plott radial symmetry condition check", quote="")
_r("svpld", "svpld", "Spatial", "Dimensional polarization decomposition", quote="")
_r("svple", "svple", "Spatial", "Esteban-Ray polarization index", quote="")
_r("svplf", "svplf", "Spatial", "Affective polarization measure", quote="")
_r("svplr", "svplr", "Spatial", "1D ideological polarization index", quote="")
_r("svpls", "svpls", "Spatial", "Party sorting index (Levendusky)", quote="")
_r("svplw", "svplw", "Spatial", "Wolfson bipolarization index", quote="")
_r("svpnl", "svpnl", "Spatial", "Panel probit spatial vote.", quote="")
_r("svpp2", "svpp2", "Spatial", "Party position 2D (Laver-Hunt)", quote="")
_r("svppc", "svppc", "Spatial", "Congressional party position", quote="")
_r("svppm", "svppm", "Spatial", "Party manifesto scaling (Wordscores)", quote="")
_r("svppo", "svppo", "Spatial", "Party position estimation", quote="")
_r("svprc", "svprc", "Spatial", "Pure proximity for categorical issues.", quote="")
_r("svpro", "svpro", "Spatial", "Prospective proximity (expected utility).", quote="")
_r("svprv", "svprv", "Spatial", "Probit spatial voting probability", quote="")
_r("svprx", "svprx", "Spatial", "Proximity voting model (Euclidean).", quote="")
_r("svpwt", "svpwt", "Spatial", "Power-weighted utility function.", quote="")
_r("svpxv", "svpxv", "Spatial", "Proximity voting model probability", quote="")
_r("svqrp", "svqrp", "Spatial", "Quantile regression spatial prob.", quote="")
_r("svqta", "svqta", "Spatial", "Quota game equilibrium", quote="")
_r("svqud", "svqud", "Spatial", "Quadratic spatial utility function", quote="")
_r("svqut", "svqut", "Spatial", "Quadratic utility function for spatial voting.", quote="")
_r("svrce", "svrce", "Spatial", "Roll call classification error", quote="")
_r("svrcl", "svrcl", "Spatial", "Roll call logit model", quote="")
_r("svrcp", "svrcp", "Spatial", "Roll call vote probability model", quote="")
_r("svrcs", "svrcs", "Spatial", "Roll call simulation", quote="")
_r("svrel", "svrel", "Spatial", "Random-effect logit spatial.", quote="")
_r("svret", "svret", "Spatial", "Retrospective proximity model.", quote="")
_r("svrmv", "svrmv", "Spatial", "Rabinowitz-Macdonald intensity component", quote="")
_r("svrps", "svrps", "Spatial", "Ranked probability score spatial", quote="")
_r("svrrd", "svrrd", "Spatial", "Roemer party unanimity model", quote="")
_r("svrub", "svrub", "Spatial", "Rubinstein spatial bargaining", quote="")
_r("svsco", "svsco", "Spatial", "Scalar-distance combined model.", quote="")
_r("svscr", "svscr", "Spatial", "Scree test for spatial dimensions", quote="")
_r("svsep", "svsep", "Spatial", "Separating hyperplane", quote="")
_r("svshp", "svshp", "Spatial", "Shapley value in spatial game", quote="")
_r("svsl2", "svsl2", "Spatial", "Two-issue salience model", quote="")
_r("svsls", "svsls", "Spatial", "Issue salience weighted model", quote="")
_r("svsph", "svsph", "Spatial", "Spatial phase transition (chaos/order)", quote="")
_r("svstl", "svstl", "Spatial", "Student-t spatial vote probability.", quote="")
_r("svsut", "svsut", "Spatial", "Separable multidimensional utility.", quote="")
_r("svtcs", "svtcs", "Spatial", "Top cycle set computation", quote="")
_r("svtut", "svtut", "Spatial", "Threshold utility step function.", quote="")
_r("svucs", "svucs", "Spatial", "Uncovered set in 2D", quote="")
_r("svutm", "svutm", "Spatial", "Spatial utility maximizer", quote="")
_r("svvl2", "svvl2", "Spatial", "2D valence spatial model", quote="")
_r("svvlm", "svvlm", "Spatial", "Valence advantage model (Groseclose)", quote="")
_r("svvt2", "svvt2", "Spatial", "2D vote trading equilibrium", quote="")
_r("svvtr", "svvtr", "Spatial", "Vote trading (logrolling) model", quote="")
_r("svwdm", "svwdm", "Spatial", "Weighted directional model.", quote="")
_r("svwht", "svwht", "Spatial", "Wittman divergence model (policy-motivated)", quote="")
_r("svwpm", "svwpm", "Spatial", "Weighted proximity model.", quote="")
_r("svwt2", "svwt2", "Spatial", "Wittman model in 2D space", quote="")
_r("svwut", "svwut", "Spatial", "Weighted issue utility aggregation.", quote="")
_r("svwvt", "svwvt", "Spatial", "Weighted voting game value", quote="")
_r("swadapt", "swadapt", "Weights", "Adaptive kernel weights (variable bandwidth).", quote="")
_r("swasym", "swasym", "Weights", "Asymmetric (directed) spatial weights.", quote="")
_r("swbin", "swbin", "Weights", "Binary (0/1) spatial weights.", quote="")
_r("swblk", "swblk", "Weights", "Block spatial weights (group-based).", quote="")
_r("swcarc", "swcarc", "WDiag", "Cardinality (neighbours count) for each unit.", quote="")
_r("swcomp", "swcomp", "WDiag", "Number of connected components in W.", quote="")
_r("swcond", "swcond", "WDiag", "Condition number of W.", quote="")
_r("swconn", "swconn", "Weights", "Connectivity (avg neighbours) of weights.", quote="")
_r("swdense", "swdense", "Weights", "Compute density of spatial weights matrix.", quote="")
_r("swdiag", "swdiag", "WDiag", "Diagonal dominance check of W.", quote="")
_r("swdist", "swdist", "Weights", "Distance-band spatial weights matrix.", quote="")
_r("swdoub", "swdoub", "Weights", "Doubly-standardise a spatial weights matrix.", quote="")
_r("sweig", "sweig", "WDiag", "Eigenvalues of W (for feasible parameter range).", quote="")
_r("swfro", "swfro", "WDiag", "Frobenius norm of W.", quote="")
_r("swgab", "swgab", "Weights", "Gabriel graph spatial weights.", quote="")
_r("swinv", "swinv", "Weights", "Inverse-distance spatial weights.", quote="")
_r("swisle", "swisle", "Weights", "Identify islands (disconnected units) in W.", quote="")
_r("swkern", "swkern", "Weights", "Kernel spatial weights (Gaussian/bisquare).", quote="")
_r("swknn", "swknn", "Weights", "k-nearest-neighbour spatial weights matrix.", quote="")
_r("swlag", "swlag", "Weights", "Spatial lag Wy.", quote="")
_r("swlag2", "swlag2", "Weights", "Second-order spatial lag W^2 y.", quote="")
_r("swlagf", "swlagf", "WDiag", "Spatial lag filter (I - rho*W)^-1.", quote="")
_r("swlowt", "swlowt", "Weights", "Lower-triangular half of weights matrix.", quote="")
_r("swmmx", "swmmx", "Weights", "Min-max standardised spatial weights.", quote="")
_r("swmtr2", "swmtr2", "WDiag", "Trace of W^2.", quote="")
_r("swmtrc", "swmtrc", "WDiag", "Trace of W (sum of diagonal).", quote="")
_r("swnest", "swnest", "Weights", "Nested / hierarchical spatial weights.", quote="")
_r("swntri", "swntri", "WDiag", "Number of non-zero entries (triples) in W.", quote="")
_r("swpath", "swpath", "WDiag", "Shortest-path distance between units in W.", quote="")
_r("swpower", "swpower", "Weights", "Power weights W^p.", quote="")
_r("swqueen", "swqueen", "Weights", "Queen-contiguity spatial weights (grid).", quote="")
_r("swrank", "swrank", "WDiag", "Rank of spatial weights matrix.", quote="")
_r("swrook", "swrook", "Weights", "Rook-contiguity spatial weights (grid).", quote="")
_r("swrow", "swrow", "Weights", "Row-standardise a spatial weights matrix.", quote="")
_r("swspars", "swspars", "Weights", "Sparsify weights by threshold.", quote="")
_r("swspec", "swspec", "WDiag", "Spectral radius of W.", quote="")
_r("swsph", "swsph", "Weights", "Spherical (great-circle) distance weights.", quote="")
_r("swstoch", "swstoch", "WDiag", "Test row-stochastic property of W.", quote="")
_r("swsym", "swsym", "WDiag", "Test weights matrix symmetry.", quote="")
_r("swtri", "swtri", "Weights", "Triangulate weights from Delaunay tessellation.", quote="")
_r("syssmp", "syssmp", "GeoProcss", "Systematic spatial sampling", quote="Breathe. -- Tanjiro")
_r("tbaniso", "tbaniso", "Spatial", "Turning bands anisotropic field.", quote="")
_r("tbbnd", "tbbnd", "Spatial", "Turning bands line process generator.", quote="")
_r("tbcnd", "tbcnd", "Spatial", "Turning bands conditional simulation.", quote="")
_r("tbens", "tbens", "Spatial", "Turning bands ensemble statistics.", quote="")
_r("tbexp", "tbexp", "Spatial", "Turning bands exponential covariance.", quote="")
_r("tbgss", "tbgss", "Spatial", "Turning bands Gaussian covariance.", quote="")
_r("tbmat", "tbmat", "Spatial", "Turning bands Matern covariance.", quote="")
_r("tbnbnd", "tbnbnd", "Spatial", "Turning bands number-of-bands convergence.", quote="")
_r("tbreal", "tbreal", "Spatial", "Turning bands single realization.", quote="")
_r("tbsim", "tbsim", "Spatial", "Turning bands simulation 2D.", quote="")
_r("tbsim3", "tbsim3", "Spatial", "Turning bands simulation 3D.", quote="")
_r("tbsmp", "tbsmp", "Spatial", "Turning bands sample path diagnostics.", quote="")
_r("tbsph", "tbsph", "Spatial", "Turning bands spherical covariance.", quote="")
_r("tbvar", "tbvar", "Spatial", "Turning bands variance artifact correction.", quote="")
_r("tbzn", "tbzn", "Spatial", "Turning bands zone anisotropy rotation.", quote="")
_r(
    "thiess",
    "thiess",
    "GeoAnalysis",
    "Thiessen polygon tessellation",
    quote="Fear is the mind-killer. -- Bene Gesserit",
)
_r("tmar", "tmar", "Spatial", "Autoregressive temporal trend model.", quote="")
_r("tmarch", "tmarch", "Spatial", "ARCH temporal volatility trend.", quote="")
_r("tmbrk", "tmbrk", "Spatial", "Temporal structural break detection.", quote="")
_r("tmbw", "tmbw", "Spatial", "Butterworth temporal low-pass filter.", quote="")
_r("tmcsp", "tmcsp", "SpatialPat", "Tempered MC spatial", quote="Breathe. -- Tanjiro")
_r("tmexp", "tmexp", "Spatial", "Exponential smoothing temporal trend.", quote="")
_r("tmfour", "tmfour", "Spatial", "Fourier temporal harmonic trend.", quote="")
_r("tmhp", "tmhp", "Spatial", "Hodrick-Prescott temporal trend filter.", quote="")
_r("tmkf", "tmkf", "Spatial", "Kalman filter temporal state estimation.", quote="")
_r("tmloess", "tmloess", "Spatial", "LOESS temporal smoothing trend.", quote="")
_r("tmmk", "tmmk", "Spatial", "Mann-Kendall trend test with slope.", quote="")
_r("tmpsmp", "tmpsmp", "GeoProcss", "Temporal stratified sampling", quote="You should enjoy the detours. -- Ging")
_r("tmsen", "tmsen", "Spatial", "Sen slope temporal trend estimator.", quote="")
_r("tmses", "tmses", "Spatial", "Temporal seasonal decomposition STL.", quote="")
_r("tmspln", "tmspln", "Spatial", "Smoothing spline temporal trend.", quote="")
_r("tmtrnd", "tmtrnd", "Spatial", "Temporal linear trend estimation OLS.", quote="")
_r("tmwav", "tmwav", "Spatial", "Wavelet temporal detrending.", quote="")
_r("trair", "trair", "TransSp", "Air route spatial", quote="No half measures. -- Mike")
_r("trasg", "trasg", "TransSp", "Traffic assignment spatial", quote="Get in the robot, Shinji! -- Misato")
_r("trbkl", "trbkl", "TransSp", "Bike lane spatial", quote="The spice must flow. -- Paul Atreides")
_r("trbks", "trbks", "TransSp", "Bikesharing spatial", quote="Whatever happens, happens. -- Spike")
_r("trbus", "trbus", "TransSp", "Bus route spatial", quote="Total Concentration Breathing. -- Tanjiro")
_r("trcap", "trcap", "TransSp", "Capacity analysis spatial", quote="El Psy Kongroo. -- Okabe")
_r("trcng", "trcng", "TransSp", "Congestion index spatial", quote="Bankai! -- Ichigo")
_r("trcpv", "trcpv", "TransSp", "Capacitated VRP spatial", quote="Engage. -- Picard")
_r("trcrp", "trcrp", "TransSp", "Carpooling spatial", quote="You should enjoy the detours. -- Ging")
_r("trcrw", "trcrw", "TransSp", "Crosswalk analysis spatial", quote="Not all those who wander are lost. -- Gandalf")
_r("trdem", "trdem", "TransSp", "Travel demand spatial", quote="Set your heart ablaze! -- Rengoku")
_r("trdly", "trdly", "TransSp", "Delay analysis spatial", quote="Go beyond! Plus Ultra! -- All Might")
_r("trdns", "trdns", "TransSp", "Traffic density spatial", quote="Dedicate your hearts! -- Erwin")
_r("trdrn", "trdrn", "TransSp", "Drone routing spatial", quote="Resistance is futile. -- Borg")
_r("trdst", "trdst", "TransSp", "Trip distribution spatial", quote="Winter is coming. -- Stark motto")
_r("trelc", "trelc", "TransSp", "Electric VRP spatial", quote="One does not simply walk. -- Boromir")
_r("trflw", "trflw", "TransSp", "Traffic flow spatial", quote="It's over 9000! -- Vegeta")
_r("trfry", "trfry", "TransSp", "Ferry route spatial", quote="Valar Morghulis. -- Braavos")
_r("trfst", "trfst", "TransSp", "Fastest path spatial", quote="Chaos is a ladder. -- Littlefinger")
_r("trgra", "trgra", "TransSp", "Trip generation spatial", quote="Arise. -- Shadow Monarch")
_r("trgrn", "trgrn", "TransSp", "Green VRP spatial", quote="Growing old is a blessing. -- Rengoku")
_r("trigen", "trigen", "GeoProcss", "Triangular grid generation", quote="Engage. -- Picard")
_r("trksp", "trksp", "TransSp", "K shortest paths spatial", quote="Science! -- Jesse Pinkman")
_r("trlvl", "trlvl", "TransSp", "Level of service spatial", quote="I am justice! -- Light")
_r("trmet", "trmet", "TransSp", "Metro network spatial", quote="Tatakae! -- Eren")
_r("trmod", "trmod", "TransSp", "Mode choice spatial", quote="I am the one who knocks. -- Walter White")
_r("trmtd", "trmtd", "TransSp", "Multi-depot VRP spatial", quote="Power is everything. -- Sung Jin-Woo")
_r("trped", "trped", "TransSp", "Pedestrian flow spatial", quote="Hold the door. -- Hodor")
_r("trpkd", "trpkd", "TransSp", "Pickup-delivery spatial", quote="There is always hope. -- Aragorn")
_r("trpkn", "trpkn", "TransSp", "Parking spatial analysis", quote="My precious. -- Gollum")
_r("trprd", "trprd", "TransSp", "Period VRP spatial", quote="Walk without rhythm. -- Fremen proverb")
_r("trprt", "trprt", "TransSp", "Port analysis spatial", quote="Kamehameha! -- Goku")
_r("trque", "trque", "TransSp", "Queue analysis spatial", quote="See you space cowboy. -- Spike")
_r("trrail", "trrail", "TransSp", "Rail network spatial", quote="This is Requiem. -- Giorno")
_r("trrdb", "trrdb", "TransSp", "Roundabout spatial", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("trrds", "trrds", "TransSp", "Ridesharing spatial", quote="I am here! -- All Might")
_r("trscr", "trscr", "TransSp", "Scooter spatial", quote="I mustn't run away. -- Shinji")
_r("trsfw", "trsfw", "TransSp", "Sidewalk network spatial", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("trsgn", "trsgn", "TransSp", "Signage spatial", quote="Make it so. -- Picard")
_r("trshr", "trshr", "TransSp", "Shortest path spatial", quote="The world is cruel but beautiful. -- Mikasa")
_r("trshr2", "trshr2", "TransSp", "Shared mobility spatial", quote="A lesson without pain is meaningless. -- Edward")
_r("trsig", "trsig", "TransSp", "Signal timing spatial", quote="Believe it! -- Naruto")
_r("trsoe", "trsoe", "TransSp", "System optimal spatial", quote="Desert power. -- Paul Muad'Dib")
_r("trspb", "trspb", "TransSp", "Speed bump spatial", quote="It's over 9000! -- Vegeta")
_r("trspd", "trspd", "TransSp", "Traffic speed spatial", quote="Yare yare daze. -- Jotaro")
_r("trtram", "trtram", "TransSp", "Tram network spatial", quote="The sleeper must awaken. -- Leto Atreides")
_r("trtsp", "trtsp", "TransSp", "TSP transport spatial", quote="Breathe. -- Tanjiro")
_r("trttm", "trttm", "TransSp", "Travel time spatial", quote="Equivalent exchange. -- Elric brothers")
_r("trtwd", "trtwd", "TransSp", "Time-window VRP spatial", quote="A Lannister always pays his debts. -- Tyrion")
_r("trtxi", "trtxi", "TransSp", "Taxi analysis spatial", quote="Keep moving forward. -- Eren")
_r("truer", "truer", "TransSp", "User equilibrium spatial", quote="Live long and prosper. -- Spock")
_r("trvrp", "trvrp", "TransSp", "VRP transport spatial", quote="I alone level up. -- Sung Jin-Woo")
_r("tsarm", "tsarm", "TempSpat", "ARM spatial model", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("tsarx", "tsarx", "TempSpat", "Spatial ARX model", quote="Walk without rhythm. -- Fremen proverb")
_r("tsfor", "tsfor", "TempSpat", "Spatial forecasting", quote="Make it so. -- Picard")
_r("tsgwt", "tsgwt", "TempSpat", "GWR temporal variation", quote="Set your heart ablaze! -- Rengoku")
_r("tsknx", "tsknx", "TempSpat", "Knox test for interaction", quote="The spice must flow. -- Paul Atreides")
_r("tskrf", "tskrf", "TempSpat", "Kriging forecast temporal", quote="Believe it! -- Naruto")
_r("tsmgw", "tsmgw", "TempSpat", "MGWR temporal variation", quote="Arise. -- Shadow Monarch")
_r("tsmnl", "tsmnl", "TempSpat", "Mantel test for association", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("tssae", "tssae", "TempSpat", "Space-time autoencoder", quote="You should enjoy the detours. -- Ging")
_r("tssag", "tssag", "TempSpat", "Space-time agglomerative", quote="The world is cruel but beautiful. -- Mikasa")
_r("tsscl", "tsscl", "TempSpat", "Space-time clustering", quote="One does not simply walk. -- Boromir")
_r("tsscn", "tsscn", "TempSpat", "Space-time scan statistic", quote="One is all, all is one. -- Izumi")
_r("tsscn2", "tsscn2", "TempSpat", "Space-time CNN proxy", quote="I alone level up. -- Sung Jin-Woo")
_r("tsscr", "tsscr", "TempSpat", "Spatial cross-correlation", quote="Engage. -- Picard")
_r("tssdb", "tssdb", "TempSpat", "Space-time DBSCAN", quote="Resistance is futile. -- Borg")
_r("tssdm", "tssdm", "TempSpat", "Spatial Durbin temporal", quote="Equivalent exchange. -- Elric brothers")
_r("tssdm2", "tssdm2", "TempSpat", "Space-time dynamic model", quote="A Lannister always pays his debts. -- Tyrion")
_r("tsser", "tsser", "TempSpat", "Spatial error regression temporal", quote="Bankai! -- Ichigo")
_r("tssfa", "tssfa", "TempSpat", "Space-time factor analysis", quote="Live long and prosper. -- Spock")
_r("tssfe", "tssfe", "TempSpat", "Spatial FE temporal", quote="See you space cowboy. -- Spike")
_r("tssgm", "tssgm", "TempSpat", "Space-time Gaussian mixture", quote="Chaos is a ladder. -- Littlefinger")
_r(
    "tssgn",
    "tssgn",
    "TempSpat",
    "Space-time GNN proxy",
    quote="Even the smallest person can change the future. -- Galadriel",
)
_r("tssgr", "tssgr", "TempSpat", "Space-time GRU proxy", quote="I am here! -- All Might")
_r("tsshm", "tsshm", "TempSpat", "Space-time hidden Markov", quote="Science! -- Jesse Pinkman")
_r("tsskm", "tsskm", "TempSpat", "Space-time K-means", quote="Desert power. -- Paul Muad'Dib")
_r("tssld", "tssld", "TempSpat", "Space-time LSTM proxy", quote="A lesson without pain is meaningless. -- Edward")
_r("tsslr", "tsslr", "TempSpat", "Spatial lag regression temporal", quote="I am justice! -- Light")
_r("tssmf", "tssmf", "TempSpat", "Space-time matrix factorization", quote="Power is everything. -- Sung Jin-Woo")
_r("tssnm", "tssnm", "TempSpat", "Space-time NMF decomposition", quote="I mustn't run away. -- Shinji")
_r("tssop", "tssop", "TempSpat", "Space-time OPTICS", quote="I must not fear. -- Litany Against Fear")
_r("tsspc", "tsspc", "TempSpat", "Space-time PCA", quote="Get in the robot, Shinji! -- Misato")
_r("tsspn", "tsspn", "TempSpat", "Spatial panel lag temporal", quote="Go beyond! Plus Ultra! -- All Might")
_r("tssre", "tssre", "TempSpat", "Spatial RE temporal", quote="El Psy Kongroo. -- Okabe")
_r("tssrn", "tssrn", "TempSpat", "Space-time recurrent NN proxy", quote="Breathe. -- Tanjiro")
_r("tsstc", "tsstc", "TempSpat", "Space-time cross-correlation", quote="People's dreams never end! -- Blackbeard")
_r("tsstd", "tsstd", "TempSpat", "Space-time tensor decomposition", quote="Growing old is a blessing. -- Rengoku")
_r("tsste", "tsste", "TempSpat", "Space-time EOF analysis", quote="I am the one who knocks. -- Walter White")
_r("tsstg", "tsstg", "TempSpat", "Space-time Getis-Ord G*", quote="Scatter, Senbonzakura. -- Byakuya")
_r("tsstg2", "tsstg2", "TempSpat", "Space-time geostatistical", quote="Dedicate your hearts! -- Erwin")
_r("tssti", "tssti", "TempSpat", "Space-time interaction test", quote="Keep moving forward. -- Eren")
_r("tsstk", "tsstk", "TempSpat", "Space-time K-function", quote="I am the hope of the universe. -- Goku")
_r("tsstl", "tsstl", "TempSpat", "Space-time lattice model", quote="It's over 9000! -- Vegeta")
_r("tsstm", "tsstm", "TempSpat", "Space-time Moran's I", quote="I will take a potato chip and eat it! -- Light")
_r("tsstp", "tsstp", "TempSpat", "Space-time point model", quote="Yare yare daze. -- Jotaro")
_r(
    "tsstr",
    "tsstr",
    "TempSpat",
    "Space-time transformer proxy",
    quote="The needs of the many outweigh the few. -- Spock",
)
_r("tsstw", "tsstw", "TempSpat", "Space-time wavelet analysis", quote="Winter is coming. -- Stark motto")
_r("tssvd", "tssvd", "TempSpat", "Space-time SVD decomposition", quote="Whatever happens, happens. -- Spike")
_r("tstcr", "tstcr", "TempSpat", "Temporal cross-correlation", quote="Those who break the rules are scum. -- Kakashi")
_r("tstrd", "tstrd", "TempSpat", "Space-time trend surface", quote="Not all those who wander are lost. -- Gandalf")
_r("tsvar", "tsvar", "TempSpat", "Spatial VAR model", quote="There is always hope. -- Aragorn")
_r("turng", "turng", "SpatialPat", "Turning bands simulation", quote="Dedicate your hearts! -- Erwin")
_r("ubbke", "ubbke", "UrbanSp", "Bike accessibility spatial", quote="Bankai! -- Ichigo")
_r("ubcar", "ubcar", "UrbanSp", "Car accessibility spatial", quote="Equivalent exchange. -- Elric brothers")
_r("ubcls", "ubcls", "UrbanSp", "Clustering index spatial", quote="Power is everything. -- Sung Jin-Woo")
_r("ubcmm", "ubcmm", "UrbanSp", "Commute time spatial", quote="I mustn't run away. -- Shinji")
_r("ubcmp", "ubcmp", "UrbanSp", "Urban compactness", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("ubcng", "ubcng", "UrbanSp", "Congestion index spatial", quote="I am here! -- All Might")
_r("ubcnt", "ubcnt", "UrbanSp", "Urban centrality", quote="It's over 9000! -- Vegeta")
_r("ubcon", "ubcon", "UrbanSp", "Concentration index spatial", quote="Engage. -- Picard")
_r("ubcrs", "ubcrs", "UrbanSp", "Crash density spatial", quote="You should enjoy the detours. -- Ging")
_r("ubdis", "ubdis", "UrbanSp", "Dissimilarity index spatial", quote="Breathe. -- Tanjiro")
_r("ubedu", "ubedu", "UrbanSp", "Education accessibility", quote="Arise. -- Shadow Monarch")
_r("ubelc", "ubelc", "UrbanSp", "Electric grid coverage", quote="Hold the door. -- Hodor")
_r("ubemg", "ubemg", "UrbanSp", "Emergency response spatial", quote="Keep moving forward. -- Eren")
_r("ubexp", "ubexp", "UrbanSp", "Exposure index spatial", quote="There is always hope. -- Aragorn")
_r("ubfbr", "ubfbr", "UrbanSp", "Fiber optic coverage spatial", quote="Not all those who wander are lost. -- Gandalf")
_r("ubfgm", "ubfgm", "UrbanSp", "Urban fragmentation", quote="Believe it! -- Naruto")
_r("ubfir", "ubfir", "UrbanSp", "Fire station coverage", quote="Total Concentration Breathing. -- Tanjiro")
_r("ubfod", "ubfod", "UrbanSp", "Food desert detection", quote="I am the one who knocks. -- Walter White")
_r("ubgas", "ubgas", "UrbanSp", "Gas network coverage", quote="The spice must flow. -- Paul Atreides")
_r("ubgnf", "ubgnf", "UrbanSp", "Gentrification risk spatial", quote="Chaos is a ladder. -- Littlefinger")
_r("ubgrn", "ubgrn", "UrbanSp", "Green space accessibility", quote="See you space cowboy. -- Spike")
_r("ubhlc", "ubhlc", "UrbanSp", "Healthcare accessibility", quote="Set your heart ablaze! -- Rengoku")
_r("ubhos", "ubhos", "UrbanSp", "Hospital coverage spatial", quote="Tatakae! -- Eren")
_r("ubhsg", "ubhsg", "UrbanSp", "Housing density spatial", quote="Get in the robot, Shinji! -- Misato")
_r("ubimv", "ubimv", "UrbanSp", "Impervious surface mapping", quote="Resistance is futile. -- Borg")
_r("ubiso", "ubiso", "UrbanSp", "Isolation index spatial", quote="I alone level up. -- Sung Jin-Woo")
_r("ublib", "ublib", "UrbanSp", "Library coverage spatial", quote="Valar Morghulis. -- Braavos")
_r("ublit", "ublit", "UrbanSp", "Light pollution spatial", quote="Walk without rhythm. -- Fremen proverb")
_r("ubnse", "ubnse", "UrbanSp", "Noise level spatial", quote="A Lannister always pays his debts. -- Tyrion")
_r("ubpdm", "ubpdm", "UrbanSp", "Pedestrian density spatial", quote="Whatever happens, happens. -- Spike")
_r("ubpdn", "ubpdn", "UrbanSp", "Population density mapping", quote="Not all those who wander are lost. -- Gandalf")
_r("ubpkn", "ubpkn", "UrbanSp", "Parking availability spatial", quote="Go beyond! Plus Ultra! -- All Might")
_r("ubpol", "ubpol", "UrbanSp", "Police station coverage", quote="This is Requiem. -- Giorno")
_r("ubprk", "ubprk", "UrbanSp", "Park accessibility spatial", quote="El Psy Kongroo. -- Okabe")
_r("ubprv", "ubprv", "UrbanSp", "Housing price spatial", quote="Live long and prosper. -- Spock")
_r("ubpxm", "ubpxm", "UrbanSp", "Proximity to amenities", quote="Yare yare daze. -- Jotaro")
_r("ubren", "ubren", "UrbanSp", "Rental price spatial", quote="Desert power. -- Paul Muad'Dib")
_r("ubsch", "ubsch", "UrbanSp", "School coverage spatial", quote="The sleeper must awaken. -- Leto Atreides")
_r("ubseg", "ubseg", "UrbanSp", "Segregation index spatial", quote="Science! -- Jesse Pinkman")
_r("ubsho", "ubsho", "UrbanSp", "Shopping accessibility", quote="Winter is coming. -- Stark motto")
_r("ubspr", "ubspr", "UrbanSp", "Urban sprawl index", quote="Make it so. -- Picard")
_r("ubswr", "ubswr", "UrbanSp", "Sewer coverage spatial", quote="My precious. -- Gollum")
_r("ubthr", "ubthr", "UrbanSp", "Thermal comfort spatial", quote="Growing old is a blessing. -- Rengoku")
_r("ubtlc", "ubtlc", "UrbanSp", "Telecom coverage spatial", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("ubtrf", "ubtrf", "UrbanSp", "Traffic flow spatial", quote="A lesson without pain is meaningless. -- Edward")
_r("ubtrn", "ubtrn", "UrbanSp", "Transit accessibility", quote="I am justice! -- Light")
_r("ubuhi", "ubuhi", "UrbanSp", "UHI intensity mapping", quote="One does not simply walk. -- Boromir")
_r("ubvac", "ubvac", "UrbanSp", "Vacancy rate spatial", quote="The world is cruel but beautiful. -- Mikasa")
_r("ubwlk", "ubwlk", "UrbanSp", "Walkability index spatial", quote="Dedicate your hearts! -- Erwin")
_r("ubwst", "ubwst", "UrbanSp", "Waste collection spatial", quote="No half measures. -- Mike")
_r("ubwtr", "ubwtr", "UrbanSp", "Water supply coverage", quote="Kamehameha! -- Goku")
_r("ukflt", "ukflt", "KrigFilt", "Universal kriging filter", quote="I am the hope of the universe. -- Goku")
_r("umadp", "umadp", "Spatial", "Adaptive utility maximization.", quote="")
_r("umcex", "umcex", "Spatial", "Convex optimization spatial.", quote="")
_r("umcon", "umcon", "Spatial", "Constrained spatial utility max.", quote="")
_r("umdyn", "umdyn", "Spatial", "Dynamic utility maximization.", quote="")
_r("umgrd", "umgrd", "Spatial", "Gradient ascent spatial utility.", quote="")
_r("uminc", "uminc", "Spatial", "Incentive-compatible utility max.", quote="")
_r("umkkt", "umkkt", "Spatial", "KKT conditions spatial max.", quote="")
_r("umlag", "umlag", "Spatial", "Lagrangian spatial utility max.", quote="")
_r("ummax", "ummax", "Spatial", "Unconstrained spatial utility max.", quote="")
_r("ummec", "ummec", "Spatial", "Mechanism design utility max.", quote="")
_r("umnsh", "umnsh", "Spatial", "Nash utility maximization.", quote="")
_r("umpar", "umpar", "Spatial", "Pareto utility maximization.", quote="")
_r("umrev", "umrev", "Spatial", "Revelation principle utility max.", quote="")
_r("umsad", "umsad", "Spatial", "Saddle-point spatial utility.", quote="")
_r("umsto", "umsto", "Spatial", "Stochastic utility maximization.", quote="")
_r("undsmp", "undsmp", "GeoProcss", "Undersampling common spatial areas", quote="Growing old is a blessing. -- Rengoku")
_r("uniona", "uniona", "GeoAnalysis", "Spatial union aggregation", quote="I am justice! -- Light")
_r("unstgr", "unstgr", "GeoProcss", "Unstructured mesh generation", quote="Scatter, Senbonzakura. -- Byakuya")
_r("utmzon", "utmzon", "GeoProcss", "UTM zone determination", quote="Yare yare daze. -- Jotaro")
_r("varfp", "varfp", "KrigFilt", "Variance filter spatial", quote="Breathe. -- Tanjiro")
_r("vdadj", "vdadj", "Spatial", "Voronoi adjacency matrix from tessellation.", quote="")
_r("vdang", "vdang", "Spatial", "Delaunay angle quality check.", quote="")
_r("vdarea", "vdarea", "Spatial", "Voronoi cell area computation.", quote="")
_r("vdcent", "vdcent", "Spatial", "Voronoi centroid computation.", quote="")
_r("vdchk", "vdchk", "Spatial", "Delaunay in-circle criterion check.", quote="")
_r("vdclip", "vdclip", "Spatial", "Voronoi clipping to bounding polygon.", quote="")
_r("vddln", "vddln", "Spatial", "Delaunay triangulation 2D.", quote="")
_r("vddln3", "vddln3", "Spatial", "Delaunay tetrahedralization 3D.", quote="")
_r("vdedge", "vdedge", "Spatial", "Voronoi edge extraction and length.", quote="")
_r("vdneig", "vdneig", "Spatial", "Voronoi neighbor set from tessellation.", quote="")
_r("vdpnts", "vdpnts", "Spatial", "Delaunay point insertion incremental.", quote="")
_r("vdrefn", "vdrefn", "Spatial", "Delaunay refinement for quality meshes.", quote="")
_r("vdvor", "vdvor", "Spatial", "Voronoi diagram 2D tessellation.", quote="")
_r("vdvor3", "vdvor3", "Spatial", "Voronoi diagram 3D tessellation.", quote="")
_r("vdwght", "vdwght", "Spatial", "Voronoi area-weighted interpolation.", quote="")
_r("vgaic", "vgaic", "Spatial", "Variogram model AIC", quote="")
_r("vgani", "vgani", "Spatial", "Anisotropy ratio estimation", quote="")
_r("vgbin", "vgbin", "Spatial", "Binned semivariogram", quote="")
_r("vgcir", "vgcir", "Spatial", "Circular variogram model", quote="")
_r("vgcld", "vgcld", "Spatial", "Variogram cloud", quote="")
_r("vgcov", "vgcov", "Spatial", "Covariogram function", quote="")
_r("vgcrf", "vgcrf", "Spatial", "Correlogram function", quote="")
_r("vgcub", "vgcub", "Spatial", "Cubic variogram model", quote="")
_r("vgcvr", "vgcvr", "Spatial", "Covariance matrix from variogram", quote="")
_r("vgdir", "vgdir", "Spatial", "Directional variogram", quote="")
_r("vgemp", "vgemp", "Spatial", "Empirical semivariogram", quote="")
_r("vgenv", "vgenv", "Spatial", "Variogram Monte Carlo envelope", quote="")
_r("vgexp", "vgexp", "Spatial", "Exponential variogram model", quote="")
_r("vgfev", "vgfev", "Spatial", "Variogram fit evaluation", quote="")
_r("vgfit", "vgfit", "Spatial", "Variogram WLS fitting", quote="")
_r("vgfml", "vgfml", "Spatial", "Variogram ML fitting", quote="")
_r("vgfrl", "vgfrl", "Spatial", "Variogram REML fitting", quote="")
_r("vggau", "vggau", "Spatial", "Gaussian variogram model", quote="")
_r("vginv", "vginv", "Spatial", "Indicator variogram", quote="")
_r("vgjck", "vgjck", "Spatial", "Variogram jackknife", quote="")
_r("vglik", "vglik", "Spatial", "Variogram log-likelihood", quote="")
_r("vgmat", "vgmat", "Spatial", "Matern variogram model", quote="")
_r("vgmdr", "vgmdr", "Spatial", "Madogram estimator", quote="")
_r("vgns2", "vgns2", "Spatial", "Nested variogram fitting", quote="")
_r("vgnst", "vgnst", "Spatial", "Nested (composite) variogram", quote="")
_r("vgnug", "vgnug", "Spatial", "Nugget effect estimation", quote="")
_r("vgpen", "vgpen", "Spatial", "Pentaspherical variogram", quote="")
_r("vgpow", "vgpow", "Spatial", "Power variogram model", quote="")
_r("vgprv", "vgprv", "Spatial", "Pairwise relative variogram", quote="")
_r("vgpsr", "vgpsr", "Spatial", "Partial sill ratio", quote="")
_r("vgpxv", "vgpxv", "Spatial", "Pseudo-cross-variogram", quote="")
_r("vgrdr", "vgrdr", "Spatial", "Rodogram estimator", quote="")
_r("vgrng", "vgrng", "Spatial", "Effective range estimation", quote="")
_r("vgrob", "vgrob", "Spatial", "Robust semivariogram (Cressie-Hawkins)", quote="")
_r("vgros", "vgros", "Spatial", "Variogram rose diagram", quote="")
_r("vgrvr", "vgrvr", "Spatial", "Relative variogram", quote="")
_r("vgsil", "vgsil", "Spatial", "Sill estimation", quote="")
_r("vgsph", "vgsph", "Spatial", "Spherical variogram model", quote="")
_r("vgwav", "vgwav", "Spatial", "Wave (hole-effect) variogram", quote="")
_r("vgxvg", "vgxvg", "Spatial", "Cross-variogram estimation", quote="")
_r("vincnt", "vincnt", "GeoProcss", "Vincenty distance formula", quote="Make it so. -- Picard")
_r("vmani", "vmani", "Variogram", "Anisotropic variogram", quote="Bankai! -- Ichigo")
_r("vmbox", "vmbox", "Variogram", "Variogram box plot", quote="One does not simply walk. -- Boromir")
_r("vmcir", "vmcir", "Variogram", "Circular variogram model", quote="Yare yare daze. -- Jotaro")
_r("vmcld", "vmcld", "Variogram", "Cloud variogram", quote="Live long and prosper. -- Spock")
_r("vmcrr", "vmcrr", "Variogram", "Correlogram from variogram", quote="Power is everything. -- Sung Jin-Woo")
_r("vmcrs", "vmcrs", "Variogram", "Cross-variogram", quote="El Psy Kongroo. -- Okabe")
_r("vmcsn", "vmcsn", "Variogram", "Cosine variogram model", quote="It's over 9000! -- Vegeta")
_r("vmcub", "vmcub", "Variogram", "Cubic variogram model", quote="Keep moving forward. -- Eren")
_r("vmcvl", "vmcvl", "Variogram", "Variogram cross-validation", quote="Science! -- Jesse Pinkman")
_r("vmcvr", "vmcvr", "Variogram", "Covariance from variogram", quote="Growing old is a blessing. -- Rengoku")
_r("vmdir", "vmdir", "Variogram", "Directional variogram", quote="See you space cowboy. -- Spike")
_r("vmdmp", "vmdmp", "Variogram", "Dampened oscillation variogram", quote="Dedicate your hearts! -- Erwin")
_r(
    "vmeff",
    "vmeff",
    "Variogram",
    "Effective range computation",
    quote="Even the smallest person can change the future. -- Galadriel",
)
_r("vmenv", "vmenv", "Variogram", "Variogram envelope bootstrap", quote="Breathe. -- Tanjiro")
_r("vmexp", "vmexp", "Variogram", "Exponential variogram model", quote="There is always hope. -- Aragorn")
_r("vmfit", "vmfit", "Variogram", "Variogram model fitting (WLS)", quote="I must not fear. -- Litany Against Fear")
_r("vmfml", "vmfml", "Variogram", "Variogram model fitting (MLE)", quote="Desert power. -- Paul Muad'Dib")
_r("vmfrr", "vmfrr", "Variogram", "Fractal dimension from variogram", quote="You should enjoy the detours. -- Ging")
_r("vmfrs", "vmfrs", "Variogram", "Variogram model fitting (REML)", quote="The world is cruel but beautiful. -- Mikasa")
_r("vmgau", "vmgau", "Variogram", "Gaussian variogram model", quote="Engage. -- Picard")
_r("vmgmt", "vmgmt", "Variogram", "Geometric anisotropy transform", quote="Go beyond! Plus Ultra! -- All Might")
_r("vmhgm", "vmhgm", "Variogram", "Hagstrom variogram estimator", quote="Get in the robot, Shinji! -- Misato")
_r("vmhol", "vmhol", "Variogram", "Hole-effect variogram model", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("vmhrs", "vmhrs", "Variogram", "Hurst exponent from variogram", quote="Whatever happens, happens. -- Spike")
_r("vmind", "vmind", "Variogram", "Indicator variogram", quote="Arise. -- Shadow Monarch")
_r("vmjbs", "vmjbs", "Variogram", "J-Bessel variogram model", quote="Not all those who wander are lost. -- Gandalf")
_r("vmkbs", "vmkbs", "Variogram", "K-Bessel variogram model", quote="Make it so. -- Picard")
_r("vmlin", "vmlin", "Variogram", "Linear variogram model", quote="One is all, all is one. -- Izumi")
_r("vmmad", "vmmad", "Variogram", "MAD variogram estimator", quote="Winter is coming. -- Stark motto")
_r("vmmap", "vmmap", "Variogram", "Variogram map 2D", quote="A Lannister always pays his debts. -- Tyrion")
_r("vmmat", "vmmat", "Variogram", "Matern variogram model", quote="People's dreams never end! -- Blackbeard")
_r("vmnst", "vmnst", "Variogram", "Nested variogram model", quote="I am justice! -- Light")
_r("vmnug", "vmnug", "Variogram", "Nugget effect model", quote="The spice must flow. -- Paul Atreides")
_r(
    "vmpnt",
    "vmpnt",
    "Variogram",
    "Pentaspherical variogram model",
    quote="I will take a potato chip and eat it! -- Light",
)
_r("vmpow", "vmpow", "Variogram", "Power variogram model", quote="Scatter, Senbonzakura. -- Byakuya")
_r("vmprd", "vmprd", "Variogram", "Variogram predict at lag", quote="I alone level up. -- Sung Jin-Woo")
_r("vmprl", "vmprl", "Variogram", "Practical range computation", quote="I am here! -- All Might")
_r("vmpsc", "vmpsc", "Variogram", "Pseudo-cross-variogram", quote="Set your heart ablaze! -- Rengoku")
_r("vmrng", "vmrng", "Variogram", "Range parameter estimation", quote="A lesson without pain is meaningless. -- Edward")
_r(
    "vmrob",
    "vmrob",
    "Variogram",
    "Robust variogram (Cressie-Hawkins)",
    quote="I am the one who knocks. -- Walter White",
)
_r("vmsel", "vmsel", "Variogram", "Variogram model selection (AIC)", quote="Chaos is a ladder. -- Littlefinger")
_r("vmsil", "vmsil", "Variogram", "Sill estimation", quote="The needs of the many outweigh the few. -- Spock")
_r("vmspc", "vmspc", "Variogram", "Spectral density from variogram", quote="I mustn't run away. -- Shinji")
_r("vmsph", "vmsph", "Variogram", "Spherical variogram model", quote="Those who break the rules are scum. -- Kakashi")
_r("vmsrf", "vmsrf", "Variogram", "Variogram surface 3D", quote="Walk without rhythm. -- Fremen proverb")
_r("vmstb", "vmstb", "Variogram", "Stable variogram model", quote="Believe it! -- Naruto")
_r("vmswt", "vmswt", "Variogram", "Semivariance windowed temporal", quote="Resistance is futile. -- Borg")
_r("vmwav", "vmwav", "Variogram", "Wave variogram model", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("vmwht", "vmwht", "Variogram", "Whittle variogram model", quote="I am the hope of the universe. -- Goku")
_r("vmzrn", "vmzrn", "Variogram", "Zonal anisotropy variogram", quote="Equivalent exchange. -- Elric brothers")
_r("vorare", "vorare", "GeoAnalysis", "Voronoi cell area distribution", quote="Get in the robot, Shinji! -- Misato")
_r(
    "vornnb",
    "vornnb",
    "GeoAnalysis",
    "Voronoi neighbor count distribution",
    quote="One does not simply walk. -- Boromir",
)
_r("vorone", "vorone", "GeoAnalysis", "Voronoi entropy measure", quote="I am the one who knocks. -- Walter White")
_r(
    "voroni",
    "voroni",
    "GeoAnalysis",
    "Voronoi diagram spatial partition",
    quote="Not all those who wander are lost. -- Gandalf",
)
_r("vorper", "vorper", "GeoAnalysis", "Voronoi cell perimeter stats", quote="Live long and prosper. -- Spock")
_r("vtblk", "vtblk", "Spatial", "Block vote trading.", quote="")
_r("vtcon", "vtcon", "Spatial", "Constrained vote trading.", quote="")
_r("vtgai", "vtgai", "Spatial", "Gains from trade vote.", quote="")
_r("vtlgr", "vtlgr", "Spatial", "Logrolling vote trading.", quote="")
_r("vtmar", "vtmar", "Spatial", "Marginal vote trading.", quote="")
_r("vtmod", "vtmod", "Spatial", "Modular vote trading.", quote="")
_r("vtnet", "vtnet", "Spatial", "Network vote trading.", quote="")
_r("vtpai", "vtpai", "Spatial", "Pairwise vote trading.", quote="")
_r("vtpar", "vtpar", "Spatial", "Parallel vote trading.", quote="")
_r("vtseq", "vtseq", "Spatial", "Sequential vote trading.", quote="")
_r("weighs", "weighs", "GeoProcss", "Spatial sampling weights", quote="A Lannister always pays his debts. -- Tyrion")
_r("winklt", "winklt", "GeoProcss", "Winkel tripel projection", quote="Resistance is futile. -- Borg")
_r("wlbbm", "wlbbm", "WildlSp", "Brownian bridge movement", quote="Get in the robot, Shinji! -- Misato")
_r("wlbrt", "wlbrt", "WildlSp", "BRT species model", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("wlcmr", "wlcmr", "WildlSp", "Capture-mark-recapture", quote="Equivalent exchange. -- Elric brothers")
_r("wlcrc", "wlcrc", "WildlSp", "Circuit theory connectivity", quote="Make it so. -- Picard")
_r("wlcrr", "wlcrr", "WildlSp", "Corridor identification", quote="Believe it! -- Naruto")
_r("wlctf", "wlctf", "WildlSp", "Connectivity function wildlife", quote="The spice must flow. -- Paul Atreides")
_r("wldst", "wldst", "WildlSp", "Distance sampling spatial", quote="Bankai! -- Ichigo")
_r("wlest", "wlest", "WildlSp", "Abundance estimation spatial", quote="Go beyond! Plus Ultra! -- All Might")
_r("wlfst", "wlfst", "WildlSp", "Fst landscape genetics", quote="It's over 9000! -- Vegeta")
_r("wlgam", "wlgam", "WildlSp", "GAM species model", quote="Believe it! -- Naruto")
_r("wlglm", "wlglm", "WildlSp", "GLM species model", quote="Make it so. -- Picard")
_r("wlgnf", "wlgnf", "WildlSp", "Gene flow landscape", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("wlhme", "wlhme", "WildlSp", "Home range estimation", quote="Set your heart ablaze! -- Rengoku")
_r("wlhsi", "wlhsi", "WildlSp", "Habitat suitability index", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("wlibd", "wlibd", "WildlSp", "Isolation by distance", quote="Yare yare daze. -- Jotaro")
_r("wlibe", "wlibe", "WildlSp", "Isolation by environment", quote="I am justice! -- Light")
_r("wlibr", "wlibr", "WildlSp", "Isolation by resistance", quote="Dedicate your hearts! -- Erwin")
_r("wlkde", "wlkde", "WildlSp", "KDE home range", quote="Arise. -- Shadow Monarch")
_r("wllcp", "wllcp", "WildlSp", "Least cost path wildlife", quote="Not all those who wander are lost. -- Gandalf")
_r("wllcv", "wllcv", "WildlSp", "LoCoH home range", quote="I am the one who knocks. -- Walter White")
_r("wlmcp", "wlmcp", "WildlSp", "MCP home range", quote="Winter is coming. -- Stark motto")
_r("wlmxe", "wlmxe", "WildlSp", "MaxEnt distribution model", quote="Not all those who wander are lost. -- Gandalf")
_r("wlnmx", "wlnmx", "WildlSp", "N-mixture model spatial", quote="I am justice! -- Light")
_r("wlocc", "wlocc", "WildlSp", "Occupancy model spatial", quote="Dedicate your hearts! -- Erwin")
_r("wlrfc", "wlrfc", "WildlSp", "RF species classification", quote="It's over 9000! -- Vegeta")
_r("wlrst", "wlrst", "WildlSp", "Resistance surface", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("wlsdm", "wlsdm", "WildlSp", "Species distribution model", quote="The spice must flow. -- Paul Atreides")
_r("wlsrv", "wlsrv", "WildlSp", "Survival model species spatial", quote="El Psy Kongroo. -- Okabe")
_r("wlsvm", "wlsvm", "WildlSp", "SVM species model", quote="Yare yare daze. -- Jotaro")
_r("wltrn", "wltrn", "WildlSp", "Trend analysis species", quote="See you space cowboy. -- Spike")
_r("wqalg", "wqalg", "WtrQual", "Algae bloom spatial", quote="Bankai! -- Ichigo")
_r("wqaqu", "wqaqu", "WtrQual", "Aquaculture water quality", quote="Believe it! -- Naruto")
_r("wqas", "wqas", "WtrQual", "Arsenic water spatial", quote="Believe it! -- Naruto")
_r("wqbod", "wqbod", "WtrQual", "BOD water spatial", quote="Not all those who wander are lost. -- Gandalf")
_r("wqcd", "wqcd", "WtrQual", "Cadmium water spatial", quote="Not all those who wander are lost. -- Gandalf")
_r("wqchl", "wqchl", "WtrQual", "Chlorophyll-a water spatial", quote="Go beyond! Plus Ultra! -- All Might")
_r("wqcl", "wqcl", "WtrQual", "Chloride water spatial", quote="Yare yare daze. -- Jotaro")
_r("wqcnd", "wqcnd", "WtrQual", "Conductivity water spatial", quote="Yare yare daze. -- Jotaro")
_r("wqcod", "wqcod", "WtrQual", "COD water spatial", quote="Make it so. -- Picard")
_r("wqcol", "wqcol", "WtrQual", "Color water spatial", quote="Dedicate your hearts! -- Erwin")
_r("wqcr", "wqcr", "WtrQual", "Chromium water spatial", quote="Make it so. -- Picard")
_r("wqcya", "wqcya", "WtrQual", "Cyanobacteria spatial", quote="Equivalent exchange. -- Elric brothers")
_r("wqdo", "wqdo", "WtrQual", "Dissolved oxygen water", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("wqdrk", "wqdrk", "WtrQual", "Drinking water quality", quote="Get in the robot, Shinji! -- Misato")
_r("wqefl", "wqefl", "WtrQual", "Effluent quality spatial", quote="It's over 9000! -- Vegeta")
_r("wqeut", "wqeut", "WtrQual", "Eutrophication index spatial", quote="El Psy Kongroo. -- Okabe")
_r("wqfe", "wqfe", "WtrQual", "Iron water spatial", quote="I am the one who knocks. -- Walter White")
_r("wqfl", "wqfl", "WtrQual", "Fluoride water spatial", quote="It's over 9000! -- Vegeta")
_r("wqfsh", "wqfsh", "WtrQual", "Fishery water quality", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("wqgwi", "wqgwi", "WtrQual", "Groundwater quality index", quote="I am the one who knocks. -- Walter White")
_r("wqhg", "wqhg", "WtrQual", "Mercury water spatial", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("wqhrd", "wqhrd", "WtrQual", "Hardness water spatial", quote="I am justice! -- Light")
_r("wqind", "wqind", "WtrQual", "Industrial water quality", quote="Make it so. -- Picard")
_r("wqinf", "wqinf", "WtrQual", "Influent quality spatial", quote="Yare yare daze. -- Jotaro")
_r("wqirg", "wqirg", "WtrQual", "Irrigation water quality", quote="Fear is the mind-killer. -- Bene Gesserit")
_r("wqlvs", "wqlvs", "WtrQual", "Livestock water quality", quote="Not all those who wander are lost. -- Gandalf")
_r("wqmn", "wqmn", "WtrQual", "Manganese water spatial", quote="Get in the robot, Shinji! -- Misato")
_r("wqnh3", "wqnh3", "WtrQual", "Ammonia water spatial", quote="Go beyond! Plus Ultra! -- All Might")
_r("wqno2", "wqno2", "WtrQual", "Nitrite water spatial", quote="El Psy Kongroo. -- Okabe")
_r("wqno3", "wqno3", "WtrQual", "Nitrate water spatial", quote="See you space cowboy. -- Spike")
_r("wqodr", "wqodr", "WtrQual", "Odor water spatial", quote="I am justice! -- Light")
_r("wqpb", "wqpb", "WtrQual", "Lead water spatial", quote="The spice must flow. -- Paul Atreides")
_r("wqph", "wqph", "WtrQual", "Water pH spatial", quote="The spice must flow. -- Paul Atreides")
_r("wqpo4", "wqpo4", "WtrQual", "Phosphate water spatial", quote="Winter is coming. -- Stark motto")
_r("wqrec", "wqrec", "WtrQual", "Recreational water quality", quote="The spice must flow. -- Paul Atreides")
_r("wqse", "wqse", "WtrQual", "Selenium water spatial", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("wqsec", "wqsec", "WtrQual", "Secchi depth spatial", quote="See you space cowboy. -- Spike")
_r("wqso4", "wqso4", "WtrQual", "Sulfate water spatial", quote="Dedicate your hearts! -- Erwin")
_r("wqswi", "wqswi", "WtrQual", "Surface water index", quote="Winter is coming. -- Stark motto")
_r("wqtbn", "wqtbn", "WtrQual", "Turbidity water spatial", quote="It's over 9000! -- Vegeta")
_r("wqtds", "wqtds", "WtrQual", "Total dissolved solids", quote="I'm gonna be King of the Pirates! -- Luffy")
_r("wqtmp", "wqtmp", "WtrQual", "Temperature water spatial", quote="Equivalent exchange. -- Elric brothers")
_r("wqtpn", "wqtpn", "WtrQual", "Total nitrogen water", quote="Set your heart ablaze! -- Rengoku")
_r("wqtpp", "wqtpp", "WtrQual", "Total phosphorus water", quote="Arise. -- Shadow Monarch")
_r("wqtrp", "wqtrp", "WtrQual", "Treatment plant performance", quote="Dedicate your hearts! -- Erwin")
_r("wqtsi", "wqtsi", "WtrQual", "Trophic state index", quote="Set your heart ablaze! -- Rengoku")
_r("wqtss", "wqtss", "WtrQual", "Total suspended solids water", quote="Believe it! -- Naruto")
_r("wqtst", "wqtst", "WtrQual", "Taste water spatial", quote="Bankai! -- Ichigo")
_r("wqwqi", "wqwqi", "WtrQual", "Water quality index", quote="Arise. -- Shadow Monarch")
_r("wqwtp", "wqwtp", "WtrQual", "WTP efficiency spatial", quote="I am justice! -- Light")
_r("wvasym", "wvasym", "Spatial", "Asymmetric power index.", quote="")
_r("wvber", "wvber", "Spatial", "Bery-Felsenthal power.", quote="")
_r("wvbnz", "wvbnz", "Spatial", "Banzhaf power index.", quote="")
_r("wvcol", "wvcol", "Spatial", "Coleman power index.", quote="")
_r("wvdpi", "wvdpi", "Spatial", "Deegan-Packel power index.", quote="")
_r("wvhof", "wvhof", "Spatial", "Holler-Packel power index.", quote="")
_r("wvjst", "wvjst", "Spatial", "Johnston power index.", quote="")
_r("wvkol", "wvkol", "Spatial", "Koenig-Brauninger power.", quote="")
_r("wvnuk", "wvnuk", "Spatial", "Nucleolus spatial voting.", quote="")
_r("wvpub", "wvpub", "Spatial", "Public goods power index.", quote="")
_r("wvpvt", "wvpvt", "Spatial", "Pivot probability spatial.", quote="")
_r("wvshp", "wvshp", "Spatial", "Shapley-Shubik power index.", quote="")
_r("wvspi", "wvspi", "Spatial", "Spatial power index.", quote="")
_r("wvswg", "wvswg", "Spatial", "Swing voter power.", quote="")
_r("wvsym", "wvsym", "Spatial", "Symmetric power index.", quote="")
_r("xrcar", "xrcar", "Spatial", "CAR model ML", quote="")
_r("xrfgt", "xrfgt", "Spatial", "Getis spatial filtering", quote="")
_r("xrflt", "xrflt", "Spatial", "Eigenvector spatial filtering", quote="")
_r("xrgns", "xrgns", "Spatial", "General Nesting Spatial model", quote="")
_r("xrgr2", "xrgr2", "Spatial", "Poisson gravity model", quote="")
_r("xrgrv", "xrgrv", "Spatial", "Gravity spatial interaction", quote="")
_r("xrgwb", "xrgwb", "Spatial", "GWR bandwidth selection", quote="")
_r("xrgwc", "xrgwc", "Spatial", "GWR local coefficients", quote="")
_r("xrgwk", "xrgwk", "Spatial", "GWR kernel function", quote="")
_r("xrgwr", "xrgwr", "Spatial", "GWR basic estimation", quote="")
_r("xrgwr2", "xrgwr2", "Spatial", "GWR local R-squared", quote="")
_r("xrgwt", "xrgwt", "Spatial", "GWR local t-values", quote="")
_r("xricar", "xricar", "Spatial", "Intrinsic CAR (ICAR)", quote="")
_r("xrim2", "xrim2", "Spatial", "SDM impacts decomposition", quote="")
_r("xrimp", "xrimp", "Spatial", "SAR direct/indirect/total impacts", quote="")
_r("xrjcn", "xrjcn", "Spatial", "Join count statistic", quote="")
_r("xrjcp", "xrjcp", "Spatial", "Join count permutation test", quote="")
_r("xrlmb", "xrlmb", "Spatial", "Robust LM test for error", quote="")
_r("xrlme", "xrlme", "Spatial", "LM test for spatial error", quote="")
_r("xrlml", "xrlml", "Spatial", "LM test for spatial lag", quote="")
_r("xrlmr", "xrlmr", "Spatial", "Robust LM test for lag", quote="")
_r("xrlms", "xrlms", "Spatial", "LM test SARMA", quote="")
_r("xrlsa", "xrlsa", "Spatial", "Local Moran's I (LISA)", quote="")
_r("xrlsg", "xrlsg", "Spatial", "Local Getis-Ord Gi*", quote="")
_r("xrmgb", "xrmgb", "Spatial", "MGWR variable bandwidths", quote="")
_r("xrmgw", "xrmgw", "Spatial", "MGWR estimation", quote="")
_r("xrmri", "xrmri", "Spatial", "Moran's I on regression residuals", quote="")
_r("xrsac", "xrsac", "Spatial", "SAC/SARAR model ML", quote="")
_r("xrsar", "xrsar", "Spatial", "SAR (Spatial Lag) model ML estimation", quote="")
_r("xrsdl", "xrsdl", "Spatial", "Spatial Durbin Error model", quote="")
_r("xrsdm", "xrsdm", "Spatial", "Spatial Durbin model ML", quote="")
_r("xrsem", "xrsem", "Spatial", "SEM (Spatial Error) model ML estimation", quote="")
_r("xrslx", "xrslx", "Spatial", "SLX model OLS estimation", quote="")
_r("xrspc", "xrspc", "Spatial", "Spatial Poisson count model", quote="")
_r("xrspl", "xrspl", "Spatial", "Spatial logit model", quote="")
_r("xrspn", "xrspn", "Spatial", "Spatial panel fixed effects", quote="")
_r("xrspp", "xrspp", "Spatial", "Spatial probit model", quote="")
_r("xrspr", "xrspr", "Spatial", "Spatial panel random effects", quote="")
_r("xrspz", "xrspz", "Spatial", "Spatial zero-inflated Poisson", quote="")
_r("xrwad", "xrwad", "Spatial", "Adaptive bandwidth weights", quote="")
_r("xrwcn", "xrwcn", "Spatial", "Weights connectivity check", quote="")
_r("xrwds", "xrwds", "Spatial", "Distance-band weights", quote="")
_r("xrwid", "xrwid", "Spatial", "Inverse distance weights matrix", quote="")
_r("xrwis", "xrwis", "Spatial", "Weights islands detection", quote="")
_r("xrwkn", "xrwkn", "Spatial", "K-nearest neighbors weights", quote="")
_r("xrwkr", "xrwkr", "Spatial", "Kernel weights", quote="")
_r("xrwqn", "xrwqn", "Spatial", "Queen contiguity weights", quote="")
_r("xrwrk", "xrwrk", "Spatial", "Rook contiguity weights", quote="")
_r("xrwrs", "xrwrs", "Spatial", "Row-standardize weights", quote="")
_r("xrwsm", "xrwsm", "Spatial", "Weights symmetrization", quote="")
_r("ze2sf", "ze2sf", "Spatial", "Two-step floating catchment area", quote="")
_r("ze3sf", "ze3sf", "Spatial", "Three-step FCA", quote="")
_r("zebot", "zebot", "Spatial", "Bayesian outbreak detection", quote="")
_r("zebuf", "zebuf", "Spatial", "Buffer-based exposure assessment", quote="")
_r("zeby2", "zeby2", "Spatial", "BYM2 reparameterized model", quote="")
_r("zebym", "zebym", "Spatial", "BYM (Besag-York-Mollie) model", quote="")
_r("zecon", "zecon", "Spatial", "Concentration index spatial", quote="")
_r("zecrs", "zecrs", "Spatial", "Carstairs deprivation index", quote="")
_r("zecsf", "zecsf", "Spatial", "Concentration surface estimation", quote="")
_r("zecss", "zecss", "Spatial", "Spatial CUSUM aberration detection", quote="")
_r("zectr", "zectr", "Spatial", "Spatial contact tracing", quote="")
_r("zedbs", "zedbs", "Spatial", "Spatial DBSCAN cluster detection", quote="")
_r("zedrc", "zedrc", "Spatial", "Spatial dose-response curve", quote="")
_r("zedsg", "zedsg", "Spatial", "Poisson-Gamma disease mapping", quote="")
_r("zedsm", "zedsm", "Spatial", "Poisson disease mapping", quote="")
_r("zee2s", "zee2s", "Spatial", "Enhanced 2SFCA", quote="")
_r("zeear", "zeear", "Spatial", "Ecological regression (Poisson)", quote="")
_r("zeenb", "zeenb", "Spatial", "Ecological NB regression", quote="")
_r("zeezi", "zeezi", "Spatial", "Ecological zero-inflated", quote="")
_r("zefhr", "zefhr", "Spatial", "Fay-Herriot small area estimator", quote="")
_r("zefhu", "zefhu", "Spatial", "Unit-level Fay-Herriot", quote="")
_r("zeflx", "zeflx", "Spatial", "Flexible spatial scan (Tango)", quote="")
_r("zegin", "zegin", "Spatial", "Spatial Gini coefficient", quote="")
_r("zegra", "zegra", "Spatial", "Gravity-based accessibility", quote="")
_r("zegrm", "zegrm", "Spatial", "Gravity migration model", quote="")
_r("zegrn", "zegrn", "Spatial", "Spatial gradient estimation", quote="")
_r("zehtm", "zehtm", "Spatial", "Hotspot detection map", quote="")
_r("zeidx", "zeidx", "Spatial", "IDW exposure interpolation", quote="")
_r("zeism", "zeism", "Spatial", "Indirect standardization", quote="")
_r("zeitv", "zeitv", "Spatial", "Travel time catchment", quote="")
_r("zekex", "zekex", "Spatial", "Kernel density exposure", quote="")
_r("zeklf", "zeklf", "Spatial", "Kulldorff spatial scan statistic", quote="")
_r("zeklp", "zeklp", "Spatial", "Prospective space-time scan", quote="")
_r("zeklr", "zeklr", "Spatial", "Retrospective space-time scan", quote="")
_r("zekls", "zekls", "Spatial", "Circular scan statistic", quote="")
_r("zelrx", "zelrx", "Spatial", "Leroux CAR model", quote="")
_r("zemch", "zemch", "Spatial", "Maternal-child health mapping", quote="")
_r("zemir", "zemir", "Spatial", "Migration flow model", quote="")
_r("zepol", "zepol", "Spatial", "Pollution surface estimation", quote="")
_r("zerad", "zerad", "Spatial", "Radiation model (mobility)", quote="")
_r("zerka", "zerka", "Spatial", "Adaptive bandwidth relative risk", quote="")
_r("zerke", "zerke", "Spatial", "Risk exceedance probability", quote="")
_r("zerrk", "zerrk", "Spatial", "Relative risk kernel ratio", quote="")
_r("zescr", "zescr", "Spatial", "Spatial cure rate model", quote="")
_r("zesfr", "zesfr", "Spatial", "Spatial frailty survival model", quote="")
_r("zesir", "zesir", "Spatial", "Spatial SIR diffusion", quote="")
_r("zesmr", "zesmr", "Spatial", "Standardized Morbidity Ratio", quote="")
_r("zethl", "zethl", "Spatial", "Spatial Theil decomposition", quote="")
_r("zetwn", "zetwn", "Spatial", "Townsend deprivation index", quote="")
_r("zsblk", "zsblk", "Spatial", "Spatial block bootstrap", quote="")
_r("zschl", "zschl", "Spatial", "Cholesky spatial simulation", quote="")
_r("zscnd", "zscnd", "Spatial", "Conditional simulation", quote="")
_r("zscnf", "zscnf", "Spatial", "Filled contour generation", quote="")
_r("zscnt", "zscnt", "Spatial", "Contour line generation", quote="")
_r("zsdel", "zsdel", "Spatial", "Delaunay triangulation mesh", quote="")
_r("zsfft", "zsfft", "Spatial", "FFT-based spectral simulation", quote="")
_r("zsglm", "zsglm", "Spatial", "Spatial GLMM simulation", quote="")
_r("zsgph", "zsgph", "Spatial", "GP hyperparameter optimization", quote="")
_r("zsgpk", "zsgpk", "Spatial", "GP kernel selection", quote="")
_r("zsgpp", "zsgpp", "Spatial", "GP prediction", quote="")
_r("zsgps", "zsgps", "Spatial", "Gaussian process spatial", quote="")
_r("zsgpv", "zsgpv", "Spatial", "GP predictive variance", quote="")
_r("zsgrf", "zsgrf", "Spatial", "Focal grid statistics", quote="")
_r("zsgrr", "zsgrr", "Spatial", "Grid resampling", quote="")
_r("zsgrz", "zsgrz", "Spatial", "Zonal grid statistics", quote="")
_r("zsidp", "zsidp", "Spatial", "IDW power parameter optimization", quote="")
_r("zsids", "zsids", "Spatial", "Modified Shepard interpolation", quote="")
_r("zsidw", "zsidw", "Spatial", "IDW interpolation", quote="")
_r("zslu", "zslu", "Spatial", "LU decomposition simulation", quote="")
_r("zsmci", "zsmci", "Spatial", "Monte Carlo spatial integration", quote="")
_r("zsmvb", "zsmvb", "Spatial", "Moving block bootstrap spatial", quote="")
_r("zsnni", "zsnni", "Spatial", "Natural neighbor interpolation", quote="")
_r("zsnnl", "zsnnl", "Spatial", "Laplace natural neighbor (Sibson)", quote="")
_r("zsrbf", "zsrbf", "Spatial", "Multiquadric RBF interpolation", quote="")
_r("zsrbg", "zsrbg", "Spatial", "Gaussian RBF interpolation", quote="")
_r("zsrbt", "zsrbt", "Spatial", "Thin plate spline RBF", quote="")
_r("zsrch", "zsrch", "Spatial", "Random chi-squared field", quote="")
_r("zsrgf", "zsrgf", "Spatial", "Random Gaussian field", quote="")
_r("zsrng", "zsrng", "Spatial", "Random non-Gaussian field", quote="")
_r("zssgs", "zssgs", "Spatial", "Sequential Gaussian simulation", quote="")
_r("zssis", "zssis", "Spatial", "Sequential indicator simulation", quote="")
_r("zsstc", "zsstc", "Spatial", "Separable space-time covariance", quote="")
_r("zsste", "zsste", "Spatial", "Cressie-Huang space-time covariance", quote="")
_r("zsstk", "zsstk", "Spatial", "Space-time kriging prediction", quote="")
_r("zsstn", "zsstn", "Spatial", "Non-separable space-time covariance", quote="")
_r("zsstv", "zsstv", "Spatial", "Space-time kriging variance", quote="")
_r("zstbs", "zstbs", "Spatial", "Turning bands simulation", quote="")
_r("zstrd", "zstrd", "Spatial", "Temporal trend estimation", quote="")
_r("zsvor", "zsvor", "Spatial", "Voronoi polygon areas", quote="")
_r("zxait", "zxait", "Spatial", "Aitchison compositional spatial", quote="")
_r("zxalb", "zxalb", "Spatial", "Albers equal-area projection", quote="")
_r("zxbet", "zxbet", "Spatial", "Betti numbers computation", quote="")
_r("zxclr", "zxclr", "Spatial", "Centered log-ratio spatial", quote="")
_r("zxcmn", "zxcmn", "Spatial", "Spatial circular mean", quote="")
_r("zxcpc", "zxcpc", "Spatial", "Clayton copula spatial", quote="")
_r("zxcpg", "zxcpg", "Spatial", "Gaussian copula spatial", quote="")
_r("zxcpv", "zxcpv", "Spatial", "Vine copula spatial", quote="")
_r("zxdps", "zxdps", "Spatial", "Dirichlet process spatial", quote="")
_r("zxfcm", "zxfcm", "Spatial", "Fuzzy C-means spatial", quote="")
_r("zxfda", "zxfda", "Spatial", "Functional data analysis spatial", quote="")
_r("zxfkr", "zxfkr", "Spatial", "Functional kriging", quote="")
_r("zxfpc", "zxfpc", "Spatial", "Spatial functional PCA", quote="")
_r("zxgat", "zxgat", "Spatial", "Graph attention spatial", quote="")
_r("zxgcn", "zxgcn", "Spatial", "Graph convolution spatial", quote="")
_r("zxgmm", "zxgmm", "Spatial", "Spatial Gaussian mixture", quote="")
_r("zxgrc", "zxgrc", "Spatial", "Great circle distance", quote="")
_r("zxgws", "zxgws", "Spatial", "Geographically weighted summary stats", quote="")
_r("zxhrc", "zxhrc", "Spatial", "Hierarchical spatial (crossed)", quote="")
_r("zxhrs", "zxhrs", "Spatial", "Hierarchical spatial (nested)", quote="")
_r("zxhvr", "zxhvr", "Spatial", "Haversine distance", quote="")
_r("zxilr", "zxilr", "Spatial", "Isometric log-ratio spatial", quote="")
_r("zxlam", "zxlam", "Spatial", "Lambert conformal conic projection", quote="")
_r("zxmrc", "zxmrc", "Spatial", "Mercator projection", quote="")
_r("zxnbt", "zxnbt", "Spatial", "Network betweenness spatial", quote="")
_r("zxncl", "zxncl", "Spatial", "Network spatial clustering", quote="")
_r("zxnsp", "zxnsp", "Spatial", "Network shortest path spatial", quote="")
_r("zxphl", "zxphl", "Spatial", "Persistence landscape", quote="")
_r("zxpos", "zxpos", "Spatial", "Possibilistic spatial clustering", quote="")
_r("zxsbg", "zxsbg", "Spatial", "Spatial bagging", quote="")
_r("zxsbu", "zxsbu", "Spatial", "Buffered spatial CV", quote="")
_r("zxsbv", "zxsbv", "Spatial", "Spatial block cross-validation", quote="")
_r("zxscv", "zxscv", "Spatial", "Spatial LOO cross-validation", quote="")
_r("zxsen", "zxsen", "Spatial", "Spatial stacking ensemble", quote="")
_r("zxsgb", "zxsgb", "Spatial", "Spatial gradient boosting", quote="")
_r("zxsle", "zxsle", "Spatial", "Spatial elastic net", quote="")
_r("zxsls", "zxsls", "Spatial", "Spatial LASSO", quote="")
_r("zxsqr", "zxsqr", "Spatial", "Spatial quantile regression", quote="")
_r("zxsrf", "zxsrf", "Spatial", "Spatial random forest", quote="")
_r("zxsvm", "zxsvm", "Spatial", "Spatial SVM", quote="")
_r("zxtda", "zxtda", "Spatial", "Persistent homology spatial", quote="")
_r("zxtn3", "zxtn3", "Spatial", "Three-way spatial tensor", quote="")

# -- Quantization (25) --
_r(
    "tqmse",
    "turboquant_mse",
    "Quantization",
    "TurboQuant MSE-optimal quantization",
    "Distribution helper.",
)
_r(
    "tqprd",
    "turboquant_prod",
    "Quantization",
    "TurboQuant product quantizer",
    "Divide and conquer the galaxy. -- Thrawn",
)
_r(
    "tq2",
    "turboquant_2bit",
    "Quantization",
    "2-bit TurboQuant (14.6x compression)",
    "What is now proved was once only imagined. — William Blake",
)
_r("tq3", "turboquant_3bit", "Quantization", "3-bit TurboQuant (10x compression)", "Balance in all things. -- Bendu")
_r(
    "tq4",
    "turboquant_4bit",
    "Quantization",
    "4-bit TurboQuant (7.6x compression)",
    "A fine addition to my collection. -- Grievous",
)
_r(
    "tqdec",
    "turboquant_decode",
    "Quantization",
    "TurboQuant dequantization",
    "What is now proved was once only imagined. — William Blake",
)
_r("qjl", "qjl_project", "Quantization", "QJL random projection", "Distribution helper.")
_r(
    "qjlmt",
    "qjl_matrix",
    "Quantization",
    "QJL projection matrix generator",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "qjlcs",
    "qjl_cosine_sim",
    "Quantization",
    "Cosine similarity via QJL",
    "What is now proved was once only imagined. — William Blake",
)
_r("jllem", "jl_lemma_bound", "Quantization", "JL lemma dimension lower bound", "Size matters not.")
_r(
    "pqenc",
    "polarquant_encode",
    "Quantization",
    "PolarQuant: magnitude + direction",
    "Distribution helper.",
)
_r("pqdec", "polarquant_decode", "Quantization", "PolarQuant reconstruct", "From the ashes, rebuild. -- Mon Mothma")
_r(
    "pqnrm",
    "polar_normalize",
    "Quantization",
    "PolarQuant unit-sphere projection",
    "On the surface of a sphere, all points equal. -- Chirrut",
)
_r(
    "pqcmp",
    "polar_compress",
    "Quantization",
    "PolarQuant full compression pipeline",
    "Distribution helper.",
)
_r("lloym", "lloyd_max", "Quantization", "Lloyd-Max optimal scalar quantizer", "Optimal, the codebook must be.")
_r("unfrm", "uniform_quantize", "Quantization", "Uniform scalar quantization", "Equal spacing for all. -- Padme")
_r(
    "rtn",
    "round_to_nearest",
    "Quantization",
    "Round-to-nearest baseline quantizer",
    "The simplest path is not always the worst. --",
)
_r(
    "gptq",
    "gptq_quantize",
    "Quantization",
    "GPTQ Hessian-based weight quantizer",
    "Use the Hessian, Luke. -- Ghost",
)
_r(
    "awq",
    "activation_aware_quant",
    "Quantization",
    "Activation-aware weight quantization",
    "Protect what matters most. --",
)
_r(
    "sqnr",
    "signal_quant_noise_ratio",
    "Quantization",
    "Signal-to-quantization-noise ratio (dB)",
    "Distribution helper.",
)
_r(
    "wht",
    "walsh_hadamard",
    "Quantization",
    "Walsh-Hadamard Transform (1/sqrt(d))",
    "Transform and be transformed. -- Ahsoka",
)
_r(
    "cbgen",
    "codebook_generate",
    "Quantization",
    "k-means codebook for vector quantization",
    "Distribution helper.",
)
_r("vq", "vector_quantize", "Quantization", "Vector quantization nearest codeword", "Find the nearest ally. -- Rex")
_r(
    "kvcmp",
    "kv_cache_compress",
    "Quantization",
    "KV-cache compression pipeline",
    "Memory is precious. Compress it. -- C-3PO",
)
_r(
    "dqerr",
    "dequant_error",
    "Quantization",
    "Dequantization error metrics",
    "Distribution helper.",
)

# -- Torus (25) --
_r("torus", "torus_surface", "Torus", "Surface area & volume of torus", "What is now proved was once only imagined. — William Blake")
_r(
    "tparm",
    "torus_parametric",
    "Torus",
    "Parametric (x,y,z) coordinates on torus",
    "In my experience there is no such thing as luck. --",
)
_r(
    "tcrv",
    "torus_curvature",
    "Torus",
    "Gaussian & mean curvature of torus",
    "Luminous beings are we, not this crude matter.",
)
_r("tknot", "torus_knot", "Torus", "(p,q)-torus knot coordinates", "We are bound by a thread of destiny. -- Chirrut")
_r("tlnk", "torus_link", "Torus", "Linking number of torus link", "Everything is connected. -- Maz Kanata")
_r(
    "tfund",
    "torus_fundamental_group",
    "Torus",
    "Fundamental group pi_1(T^n) = Z^n",
    "Your focus determines your reality. -- Qui-Gon",
)
_r(
    "thom",
    "torus_homology",
    "Torus",
    "Homology groups of n-torus via Kuenneth",
    "There has been an awakening. -- Snoke",
)
_r(
    "tcoh",
    "torus_cohomology",
    "Torus",
    "De Rham cohomology of n-torus",
    "Distribution helper.",
)
_r(
    "teulr",
    "torus_euler_char",
    "Torus",
    "Euler characteristic chi = 2 - 2g",
    "There is always a bigger fish. -- Qui-Gon",
)
_r("tgenu", "torus_genus", "Torus", "Genus computation for surfaces", "Much to learn you still have.")
_r("tflat", "flat_torus", "Torus", "Flat torus from rectangle identification", "The garbage will do. -- Rey")
_r("tclif", "clifford_torus", "Torus", "Clifford torus in S^3 subset R^4", "Distribution helper.")
_r("thopf", "hopf_fibration", "Torus", "Hopf map S^3 -> S^2", "Distribution helper.")
_r("tmodu", "torus_modular", "Torus", "Modular parameter of complex torus", "The dark side clouds everything.")
_r("tjfun", "j_invariant", "Torus", "j-invariant of elliptic curve/torus", "This is the way. -- The Mandalorian")
_r("tweir", "weierstrass_p", "Torus", "Weierstrass P-function on lattice", "Do or do not. There is no try.")
_r(
    "teisr",
    "eisenstein_series",
    "Torus",
    "Eisenstein series E_k(tau)",
    "Distribution helper.",
)
_r("tdede", "dedekind_eta", "Torus", "Dedekind eta function", "Distribution helper.")
_r(
    "tthet",
    "theta_function",
    "Torus",
    "Jacobi theta function theta_3",
    "Distribution helper.",
)
_r(
    "tmapn",
    "torus_mapping_class",
    "Torus",
    "SL(2,Z) mapping class group action",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "tdehn",
    "dehn_twist",
    "Torus",
    "Dehn twist matrix on torus",
    "Let the past die. Kill it if you have to. -- Kylo Ren",
)
_r(
    "tbund",
    "torus_bundle",
    "Torus",
    "Torus bundle geometry classification",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "tlatt",
    "torus_lattice",
    "Torus",
    "Lattice on flat torus from basis vectors",
    "Distribution helper.",
)
_r("tvorl", "torus_voronoi", "Torus", "Voronoi cells on flat torus", "What is now proved was once only imagined. — William Blake")
_r("tdist", "torus_distance", "Torus", "Geodesic distance on torus surface", "It's a trap! -- Admiral Ackbar")

# -- MLTraining (25) --
_r(
    "lrsch",
    "lr_schedule",
    "MLTraining",
    "Learning rate scheduler (cosine/linear/constant)",
    "Do or do not. There is no try.",
)
_r(
    "gradc",
    "gradient_clip",
    "MLTraining",
    "Gradient clipping by global norm",
    "The dark side clouds everything.",
)
_r("wdecm", "weight_decay", "MLTraining", "L2 weight decay regularization", "Attachment leads to suffering. -- Anakin")
_r(
    "lrinf",
    "lr_finder",
    "MLTraining",
    "Optimal LR from loss curve sweep",
    "The path to the dark side begins with fear.",
)
_r(
    "earlm",
    "early_stopping",
    "MLTraining",
    "Early stopping by validation loss",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "bpb",
    "bits_per_byte",
    "MLTraining",
    "Convert CE loss to bits per byte",
    "What is now proved was once only imagined. — William Blake",
)
_r("pplxy", "perplexity", "MLTraining", "Perplexity from cross-entropy loss", "What is now proved was once only imagined. — William Blake")
_r(
    "kaimg",
    "kaiming_init",
    "MLTraining",
    "Kaiming (He) weight initialization",
    "Distribution helper.",
)
_r(
    "xvrig",
    "xavier_init",
    "MLTraining",
    "Xavier (Glorot) weight initialization",
    "In my experience there is no luck. --",
)
_r("rmsnm", "rms_norm", "MLTraining", "RMS layer normalization", "Distribution helper.")
_r("swglu", "swiglu", "MLTraining", "SwiGLU activation (Shazeer 2020)", "Let the Wookiee win. -- C-3PO")
_r("relu2", "relu_squared", "MLTraining", "ReLU squared activation", "This is where the fun begins. -- Anakin")
_r(
    "rpemb",
    "rotary_embed",
    "MLTraining",
    "Rotary Positional Embedding (RoPE)",
    "Around the survivors a perimeter create.",
)
_r(
    "logsc",
    "logit_softcap",
    "MLTraining",
    "Logit soft-capping (Gemma 2 style)",
    "Your overconfidence is your weakness. -- Luke",
)
_r(
    "xfit",
    "cross_fit",
    "MLTraining",
    "DML-style cross-fitting fold splits",
    "Always two there are. A master and an apprentice.",
)
_r("oospr", "oos_predict", "MLTraining", "Out-of-sample prediction wrapper", "What is now proved was once only imagined. — William Blake")
_r(
    "nestc",
    "nested_cv",
    "MLTraining",
    "Nested cross-validation fold indices",
    "Truly wonderful the mind of a child is.",
)
_r(
    "lscur",
    "loss_curve_analysis",
    "MLTraining",
    "Detect plateau/divergence/convergence",
    "I sense great fear in you.",
)
_r(
    "grdst",
    "gradient_stats",
    "MLTraining",
    "Gradient norm, max, min, sparsity stats",
    "Size matters not. Judge me by my size, do you?",
)
_r(
    "prmsz",
    "param_count",
    "MLTraining",
    "Count total/trainable parameters",
    "The greatest teacher, failure is.",
)
_r("flops", "estimate_flops", "MLTraining", "Transformer FLOPs estimate", "That is why you fail.")
_r(
    "bpe",
    "bpe_encode",
    "MLTraining",
    "BPE tokenization with merge rules",
    "These are not the droids you are looking for. --",
)
_r(
    "bpetm",
    "bpe_train_merges",
    "MLTraining",
    "Train BPE merge rules from corpus",
    "You must unlearn what you have learned.",
)
_r(
    "tokst",
    "token_stats",
    "MLTraining",
    "Token frequency and entropy statistics",
    "I have a bad feeling about this. -- Everyone",
)
_r(
    "dtldr",
    "data_loader_stats",
    "MLTraining",
    "Training iterations per epoch calculator",
    "What is now proved was once only imagined. — William Blake",
)

# -- Advanced Statistics (18) --
_r("tobit", "tobit_model", "Regression", "Tobit censored regression (MLE)", "Your eyes can deceive you. --")
_r(
    "firth",
    "firth_logistic",
    "Regression",
    "Firth penalized logistic (rare events)",
    "Do or do not. There is no try.",
)
_r(
    "negbn",
    "negative_binomial_reg",
    "Regression",
    "Negative binomial regression (NB2)",
    "Always in motion is the future.",
)
_r("qnest", "qn_estimator", "Robust", "Qn scale estimator (Rousseeuw-Croux)", "Judge me by my size, do you?")
_r(
    "biwtm",
    "biweight_midcorrelation",
    "Robust",
    "Biweight midcorrelation (bicor)",
    "Truly wonderful the mind of a child is.",
)
_r("bfdr", "bayesian_fdr", "Bayesian", "Bayesian False Discovery Rate", "The dark side clouds everything.")
_r(
    "mardm",
    "mardia_test",
    "Multivariate",
    "Mardia multivariate normality test",
    "Distribution helper.",
)
_r("boxmm", "box_m_test", "Multivariate", "Box's M covariance equality test", "I have a bad feeling about this. --")
_r(
    "pcorp",
    "partial_correlation",
    "Multivariate",
    "Partial correlation",
    "In my experience there is no such thing as luck. --",
)
_r(
    "cancc",
    "canonical_correlation",
    "Multivariate",
    "Canonical correlation analysis (CCA)",
    "Distribution helper.",
)
_r(
    "ksamp",
    "k_sample_anderson_darling",
    "NonparametricTest",
    "K-sample Anderson-Darling test",
    "We are brave. -- Rose Tico",
)
_r(
    "pages",
    "page_trend_test",
    "NonparametricTest",
    "Page's L trend test",
    "Hope is not lost today. It is found. -- Poe",
)
_r("dunn", "dunn_test", "NonparametricTest", "Dunn post-hoc pairwise comparisons", "There is another.")
_r(
    "jnckh",
    "jonckheere_terpstra",
    "NonparametricTest",
    "Jonckheere-Terpstra ordered alternatives",
    "Rebellions are built on hope. -- Jyn Erso",
)
_r("bic", "bayesian_info_criterion", "InfoTheory", "Bayesian Information Criterion (BIC)", "Size matters not.")
_r("aicc", "corrected_aic", "InfoTheory", "Corrected AIC (AICc)", "The greatest teacher, failure is.")
_r(
    "dic",
    "deviance_info_criterion",
    "InfoTheory",
    "Deviance Information Criterion (DIC)",
    "Powerful you have become. -- Dooku",
)
_r(
    "elpd",
    "expected_log_pred",
    "InfoTheory",
    "Expected Log Predictive Density (ELPD)",
    "Pass on what you have learned.",
)

# -- StringTheory (30) --
_r(
    "vnamp",
    "veneziano_amplitude",
    "StringTheory",
    "Veneziano amplitude B(-alpha(s),-alpha(t))",
    "Distribution helper.",
)
_r(
    "regge",
    "regge_trajectory",
    "StringTheory",
    "Regge trajectory alpha(s)=alpha_0+alpha'*s",
    "In my experience there is no such thing as luck. --",
)
_r(
    "nmbu",
    "nambu_goto_action",
    "StringTheory",
    "Nambu-Goto string action",
    "The string that binds the universe.",
)
_r(
    "plykv",
    "polyakov_action",
    "StringTheory",
    "Polyakov worldsheet action",
    "A worldsheet of infinite possibilities. -- Qui-Gon",
)
_r(
    "viras",
    "virasoro_algebra",
    "StringTheory",
    "Virasoro algebra [L_m, L_n] commutation",
    "Distribution helper.",
)
_r(
    "cntrl",
    "central_charge",
    "StringTheory",
    "Central charge c=d (bosonic) or 3d/2 (super)",
    "Distribution helper.",
)
_r(
    "cymnf",
    "calabi_yau_hodge",
    "StringTheory",
    "Calabi-Yau Hodge numbers and Euler char",
    "Distribution helper.",
)
_r(
    "klzkl",
    "kaluza_klein_spectrum",
    "StringTheory",
    "Kaluza-Klein mass tower m_n=n/R",
    "Layer upon layer, the galaxy is built. -- Chirrut",
)
_r(
    "cmpct",
    "compactify_torus",
    "StringTheory",
    "T^n toroidal compactification moduli",
    "Small things can hold great power. -- Ahsoka",
)
_r(
    "orbfd",
    "orbifold_spectrum",
    "StringTheory",
    "Orbifold fixed points and twisted sectors",
    "Distribution helper.",
)
_r("tdual", "t_duality", "StringTheory", "T-duality R <-> alpha'/R", "Two sides of the same coin. --")
_r(
    "sdual",
    "s_duality",
    "StringTheory",
    "S-duality g_s <-> 1/g_s",
    "Strength and weakness, intertwined they are.",
)
_r(
    "mthry",
    "m_theory_dimension",
    "StringTheory",
    "M-theory: 11D supergravity limit",
    "Eleven dimensions, the deepest mystery. -- Qui-Gon",
)
_r(
    "sugrv",
    "supergravity_action",
    "StringTheory",
    "d-dimensional N-extended supergravity",
    "Distribution helper.",
)
_r(
    "dbran",
    "d_brane_tension",
    "StringTheory",
    "D_p-brane tension T_p",
    "Stretched across spacetime, the branes are.",
)
_r(
    "bkhwk",
    "bekenstein_hawking",
    "StringTheory",
    "Bekenstein-Hawking entropy S=A/(4l_P^2)",
    "Even darkness has entropy. -- Luke",
)
_r(
    "holog",
    "holographic_entropy",
    "StringTheory",
    "Ryu-Takayanagi holographic entropy",
    "The boundary reveals the bulk. -- Ahsoka",
)
_r(
    "adscf",
    "ads_cft_dictionary",
    "StringTheory",
    "AdS/CFT mass-dimension relation",
    "A correspondence between worlds. --",
)
_r(
    "strpf",
    "string_partition",
    "StringTheory",
    "Bosonic string partition function Z(tau)",
    "What is now proved was once only imagined. — William Blake",
)
_r(
    "raman",
    "ramanujan_tau",
    "StringTheory",
    "Ramanujan tau function from Delta(tau)",
    "Infinite patterns in finite space.",
)
_r(
    "dednm",
    "dedekind_sum",
    "StringTheory",
    "Dedekind sum s(h,k)",
    "Small sums lead to great transformations. -- Qui-Gon",
)
_r(
    "modlr",
    "modular_form",
    "StringTheory",
    "Weight-k Eisenstein series modular form",
    "Distribution helper.",
)
_r(
    "susyq",
    "susy_algebra",
    "StringTheory",
    "N-extended SUSY algebra {Q,Q+}=H",
    "For every boson, a fermion there is.",
)
_r(
    "witin",
    "witten_index",
    "StringTheory",
    "Witten index Tr((-1)^F e^{-bH})",
    "Count what remains when all else fades. -- Luke",
)
_r(
    "bpsst",
    "bps_state",
    "StringTheory",
    "BPS bound M >= |Z(charges)|",
    "The lightest warriors are the strongest. -- Mace Windu",
)
_r(
    "gkpw",
    "gkp_witten",
    "StringTheory",
    "GKP-Witten bulk-to-boundary propagator",
    "From the boundary, the bulk we sense.",
)
_r(
    "entan",
    "entanglement_entropy",
    "StringTheory",
    "Von Neumann entropy -Tr(rho log rho)",
    "Distribution helper.",
)
_r(
    "berry",
    "berry_phase",
    "StringTheory",
    "Geometric Berry phase along closed path",
    "A journey around returns changed. -- Ahsoka",
)
_r(
    "chsim",
    "chern_simons",
    "StringTheory",
    "Chern-Simons level and partition function",
    "Topology binds what geometry cannot. -- Dooku",
)
_r(
    "stren",
    "string_tension",
    "StringTheory",
    "String tension T=1/(2*pi*alpha')",
    "What is now proved was once only imagined. — William Blake",
)


# ── Semiparametric (Horowitz 2009) ─────────────────────────────────────────
_r("nwker", "nadaraya_watson", "Semiparametric", "Nadaraya-Watson kernel regression", "The kernel sees through noise. -- Qui-Gon")
_r("llker", "local_linear", "Semiparametric", "Local linear kernel regression", "Locally, the truth reveals itself.")
_r("kdens", "kernel_density", "Semiparametric", "Kernel density estimation", "Distribution helper.")
_r("bwcv", "bandwidth_cv", "Semiparametric", "Bandwidth selection via LOO cross-validation", "Validate each fold, you must.")
_r("bwrot", "bandwidth_rot", "Semiparametric", "Bandwidth selection via Silverman rule-of-thumb", "A rule of thumb, the simplest path is.")
_r("plmod", "partially_linear", "Semiparametric", "Partially linear model (Robinson 1988)", "What is now proved was once only imagined. — William Blake")
_r("simod", "single_index", "Semiparametric", "Single-index model estimation", "A single index binds them all. -- Gandalf... wait, wrong franchise. --")
_r("avgde", "avg_derivative", "Semiparametric", "Average derivative estimation", "The derivative reveals the direction. --")
_r("npqnt", "np_quantile_reg", "Semiparametric", "Nonparametric quantile regression", "Distribution helper.")
_r("npivr", "np_iv_regression", "Semiparametric", "Nonparametric instrumental variables", "An instrument, when valid, cuts through endogeneity. -- Dooku")
_r("npbst", "nonparametric_bootstrap", "Semiparametric", "Nonparametric bootstrap inference", "Do or do not. There is no try.")
_r("wldbt", "wild_bootstrap", "Semiparametric", "Wild bootstrap for heteroskedastic regression", "The dark side clouds everything.")
_r("subsp", "subsampling_inference", "Semiparametric", "Subsampling inference (Politis & Romano)", "A small part of the whole reveals much. -- Qui-Gon")
_r("sieve", "sieve_estimation", "Semiparametric", "Sieve estimation (series/polynomial approximation)", "Distribution helper.")
_r("pensp", "penalized_spline", "Semiparametric", "Penalized spline regression", "Balance the smooth and the rough, you must.")
_r("mxmod", "semiparametric_mixture", "Semiparametric", "Semiparametric mixture model (EM)", "Always two there are.")
_r("sclst", "censored_least_squares", "Semiparametric", "Semiparametric censored least squares", "Not all data reveals itself. --")
_r("kmsem", "kaplan_meier_survival", "Semiparametric", "Kaplan-Meier semiparametric survival estimator", "Death is a natural part of life.")
_r("coxph", "cox_proportional_hazards", "Semiparametric", "Cox proportional hazards model", "What is now proved was once only imagined. — William Blake")
_r("npsel", "nonparametric_model_selection", "Semiparametric", "Nonparametric model selection (CV bandwidth)", "Choose wisely, you must.")

# ── Empirical Process (Kosorok 2008) ──────────────────────────────────────
_r("ecdf", "empirical_cdf", "EmpiricalProcess", "Empirical CDF with DKW confidence bands", "The data reveals the distribution, patience it takes.")
_r("emprc", "empirical_process", "EmpiricalProcess", "Centered, scaled empirical process", "A process, centered and scaled, the truth it seeks.")
_r("glivn", "glivenko_cantelli", "EmpiricalProcess", "Glivenko-Cantelli uniform convergence test", "Converge uniformly, all distributions must.")
_r("donsk", "donsker_test", "EmpiricalProcess", "Donsker class membership test via bootstrap", "Weak convergence, the bridge to asymptopia.")
_r("vcidx", "vc_dimension", "EmpiricalProcess", "VC index / VC dimension computation", "The capacity to shatter, measure you must.")
_r("brkts", "bracketing_number", "EmpiricalProcess", "Bracketing number estimation", "Between brackets, the truth is contained. --")
_r("entrp", "metric_entropy", "EmpiricalProcess", "Metric entropy (covering number) computation", "Cover the space, the entropy measures how. -- Qui-Gon")
_r("hoeff", "hoeffding_bound", "EmpiricalProcess", "Hoeffding concentration inequality bound", "Bounded variables, bounded tails.")
_r("mcdir", "mcdiarmid_bound", "EmpiricalProcess", "McDiarmid bounded-differences inequality", "Change one input, bounded the effect must be. --")
_r("berns", "bernstein_bound", "EmpiricalProcess", "Bernstein concentration inequality", "Variance sharpens the bound, young one. -- Qui-Gon")
_r("dkwin", "dkw_test", "EmpiricalProcess", "DKW inequality for empirical CDF", "The empirical CDF converges, uniformly it does.")
_r("symtz", "symmetrization_bound", "EmpiricalProcess", "Symmetrization via Rademacher complexity", "Mirror the process, the truth appears. -- Ahsoka")
_r("rcomp", "rademacher_complexity", "EmpiricalProcess", "Rademacher complexity computation", "Random signs reveal the capacity. -- Kanan")
_r("gcplx", "gaussian_complexity", "EmpiricalProcess", "Gaussian complexity computation", "Gaussian weights, smoother than Rademacher. -- Qui-Gon")
_r("cvmsv", "cramer_von_mises", "EmpiricalProcess", "Cramer-von Mises goodness-of-fit statistic", "Integrate the squared difference, measure the fit. -- Dooku")
_r("anddr", "anderson_darling", "EmpiricalProcess", "Anderson-Darling goodness-of-fit statistic", "What is now proved was once only imagined. — William Blake")
_r("prohv", "prohorov_metric", "EmpiricalProcess", "Prohorov metric between distributions", "The distance between measures, weak convergence it defines.")
_r("portm", "portmanteau_test", "EmpiricalProcess", "Portmanteau test for serial correlation", "Serial dependence, the portmanteau reveals. -- Mace Windu")
_r("mblbt", "moving_block_bootstrap", "EmpiricalProcess", "Moving block bootstrap for dependent data", "Resample in blocks, preserve the dependence. --")
_r("hddif", "hadamard_differentiability", "EmpiricalProcess", "Hadamard differentiability check", "Differentiable the map must be, for the delta method.")
_r("chrnf", "chernoff_distribution", "EmpiricalProcess", "Chernoff distribution (cube-root asymptotics)", "What is now proved was once only imagined. — William Blake")
_r("grndr", "grenander_estimator", "EmpiricalProcess", "Grenander estimator (monotone density NPMLE)", "Monotone the density, the LCM reveals. -- Qui-Gon")

# ── Semiparametric Estimation (Kosorok 2008) ──────────────────────────────
_r("zestm", "z_estimator", "Semiparametric", "Z-estimator with sandwich variance", "Solve the equation, the estimator emerges. -- Dooku")
_r("mestm", "m_estimator", "Semiparametric", "M-estimator with influence function", "What is now proved was once only imagined. — William Blake")
_r("effic", "efficiency_bound", "Semiparametric", "Semiparametric efficiency bound (information bound)", "What is now proved was once only imagined. — William Blake")

# ── Kosorok (2008) Individual fn/ Files ──────────────────────────────────
_r("eproc", "eproc", "EmpiricalProcess", "Empirical process sqrt(n)*(Fn-F)", "The empirical truth, amplified by root-n.")
_r("bpro", "bpro", "EmpiricalProcess", "Brownian bridge process simulation", "A bridge between zero and zero, the path wanders. --")
_r("gcthm", "gcthm", "EmpiricalProcess", "Glivenko-Cantelli sup|Fn-F| convergence test", "Distribution helper.")
_r("brent", "brent", "EmpiricalProcess", "Bracketing entropy of function class", "Bracket the functions, bound the complexity. -- Dooku")
_r("covnm", "covnm", "EmpiricalProcess", "Covering number estimation (epsilon-net)", "Cover the space, count the balls. -- Mace Windu")
_r("mxinq", "mxinq", "EmpiricalProcess", "Maximal inequality bound computation", "What is now proved was once only imagined. — William Blake")
_r("zest", "zest", "Semiparametric", "Z-estimator via estimating equations", "Solve the equation, zero you must find.")
_r("mest", "mest", "Semiparametric", "M-estimator via loss minimization", "What is now proved was once only imagined. — William Blake")
_r("sefnl", "sefnl", "Semiparametric", "Semiparametric efficiency bound (lower)", "What is now proved was once only imagined. — William Blake")
_r("nusnc", "nusnc", "Semiparametric", "Nuisance parameter profiling", "Profile out the nuisance, focus on what matters. --")
_r("sceff", "sceff", "Semiparametric", "Score-based efficient estimation", "The efficient score, the optimal direction. -- Qui-Gon")
_r("cxreg", "cxreg", "Survival", "Cox regression via partial likelihood", "Proportional the hazards, partial the likelihood. -- Dooku")
_r("cxbsl", "cxbsl", "Survival", "Breslow baseline hazard estimator", "What is now proved was once only imagined. — William Blake")
_r("aftrg", "aftrg", "Survival", "Accelerated failure time regression", "Accelerate or decelerate, time transforms. -- Anakin")
_r("trreg", "trreg", "Semiparametric", "Transformation regression model", "Transform the response, reveal the linear. --")
_r("csmix", "csmix", "Semiparametric", "Case-control semiparametric mixture", "Cases and controls, retrospective the design. --")
_r("bwreg", "bwreg", "Semiparametric", "Bandwidth-selected kernel regression", "Smooth the signal, choose the bandwidth wisely.")
_r("plest", "plest", "Semiparametric", "Profile likelihood estimation and CI", "Profile out, focus on what interests you. -- Qui-Gon")
_r("slest", "slest", "Semiparametric", "Sieve likelihood density estimation", "Approximate with sieves, the truth approaches. -- Dooku")
_r("empll", "empll", "Semiparametric", "Empirical likelihood ratio for mean", "Let the data speak, no parametric assumptions. --")
_r("elci", "elci", "Semiparametric", "Empirical likelihood confidence interval", "The data-driven interval, shape-respecting. -- Mace Windu")
_r("tmlse", "tmlse", "TargetedLearning", "TMLE standard error via influence curve", "The influence curve reveals the variance. --")
_r("clvr", "clvr", "TargetedLearning", "Clever covariate for TMLE", "Clever the covariate, efficient the update. -- Dooku")
_r("submd", "submd", "TargetedLearning", "Substitution (G-computation) estimator", "Plug in and compute, simple yet powerful. --")
_r("tgtpr", "tgtpr", "TargetedLearning", "TMLE fluctuation parameter (targeting step)", "Fluctuate along the score, target the truth. -- Qui-Gon")
_r("exchw", "exchw", "Bootstrap", "Exchangeable bootstrap weights", "Exchange the weights, preserve the structure. --")
_r("bysbt", "bysbt", "Bootstrap", "Bayesian bootstrap (Rubin 1981)", "Dirichlet weights, the Bayesian resamples.")
_r("mofnb", "mofnb", "Bootstrap", "m-out-of-n bootstrap", "Fewer observations, broader validity. -- Mace Windu")
_r("sbcnv", "sbcnv", "Bootstrap", "Bootstrap convergence diagnostic", "Converge the bootstrap must, or trust it you cannot.")
_r("wtbst", "wtbst", "Bootstrap", "General weighted bootstrap", "Weight and resample, flexible the method. -- Qui-Gon")
_r("cvtml", "cvtml", "TargetedLearning", "Cross-validated TMLE (CV-TMLE)", "Cross-validate the nuisance, free the Donsker.")
_r("sllrn", "sllrn", "ML", "Super learner ensemble via CV", "Combine the learners, optimal the ensemble. --")
_r("cvbnd", "cvbnd", "ModelSelection", "Cross-validation risk bound", "Bound the risk, finite the sample. -- Dooku")

# ── Rangayyan Signal Processing (Ch 3-17) ─────────────────────────────────
_r("chflt", "chflt", "FilterDesign", "Chebyshev type I filter", "The ripple is strong with this one. --")
_r("medfl", "medfl", "Filter", "Median filter (nonlinear smoothing)", "The median path, the wisest is.")
_r("dcblk", "dcblk", "SignalOps", "DC blocker (remove zero-frequency)", "Remove the dark side, you must.")
_r("psdwl", "psdwl", "SpectralAnalysis", "Welch PSD estimation", "What is now proved was once only imagined. — William Blake")
_r("psdmt", "psdmt", "SpectralAnalysis", "Multitaper PSD (DPSS windows)", "Many windows reveal the truth. -- Mace Windu")
_r("spcgm", "spcgm", "SpectralAnalysis", "Spectrogram (STFT magnitude)", "Distribution helper.")
_r("hrmns", "hrmns", "SpectralAnalysis", "Harmonic analysis (fundamental + overtones)", "Distribution helper.")
_r("bspec", "bspec", "SpectralAnalysis", "Bispectrum estimation", "Two frequencies together, stronger they become.")
_r("chpdl", "chpdl", "TimeFrequency", "Chirplet decomposition", "Distribution helper.")
_r("stqft", "stqft", "TimeFrequency", "Short-time quadratic frequency transform", "Quadratic, the dark side is not.")
_r("gbrwv", "gbrwv", "TimeFrequency", "Gabor-Wigner distribution", "Between two worlds, the truth lies. -- Qui-Gon")
_r("cwtsc", "cwtsc", "TimeFrequency", "Continuous wavelet scalogram", "What is now proved was once only imagined. — William Blake")
_r("kalmf", "kalmf", "AdaptiveFilter", "Kalman filter (1-D state estimation)", "Distribution helper.")
_r("wienf", "wienf", "AdaptiveFilter", "Wiener filter (optimal linear)", "Distribution helper.")
_r("qrsdt", "qrsdt", "ECG", "QRS detection (Pan-Tompkins)", "Distribution helper.")
_r("hrvar", "hrvar", "HRV", "HRV time-domain (SDNN, RMSSD, pNN50)", "Distribution helper.")
_r("hrvfq", "hrvfq", "HRV", "HRV frequency-domain (LF/HF ratio)", "Feel the frequencies, you must.")
_r("eegar", "eegar", "EEG", "EEG autoregressive modeling", "The mind, a signal generates.")
_r("emgrm", "emgrm", "EMG", "EMG RMS envelope", "Strength in every muscle. --")
_r("emgfr", "emgfr", "EMG", "EMG median/mean frequency", "Frequency of the warrior. -- Ahsoka")
_r("emgmu", "emgmu", "EMG", "EMG motor unit action potential detection", "Distribution helper.")
_r("entsg", "entsg", "NonlinearDynamics", "Signal entropy (Shannon/sample/approximate)", "Disorder, the path to chaos is.")
_r("lyexp", "lyexp", "NonlinearDynamics", "Largest Lyapunov exponent", "Chaos, the dark side breeds. -- Mace Windu")
_r("corrn", "corrn", "NonlinearDynamics", "Correlation dimension (Grassberger-Procaccia)", "Connected, all things are.")
_r("dfanl", "dfanl", "NonlinearDynamics", "Detrended fluctuation analysis (DFA)", "Fluctuations reveal the hidden. -- Qui-Gon")
_r("rcrnc", "rcrnc", "NonlinearDynamics", "Recurrence quantification analysis (RQA)", "History repeats itself. -- C-3PO")
_r("pcasg", "pcasg", "Decomposition", "PCA-based signal decomposition", "The principal components, reveal the truth. --")
_r("icasg", "icasg", "Decomposition", "ICA source separation (FastICA)", "Separate the light from the dark. -- Mace Windu")

_r("accur", "accuracy", "Classification", "Classification accuracy", "Do. Or do not. There is no try.")
_r("precn", "precision", "Classification", "Precision (PPV)", "Precision is key. -- Grand Admiral Thrawn")
_r("recal", "recall", "Classification", "Recall (sensitivity / TPR)", "Your eyes can deceive you. --")
_r("specf", "specificity", "Classification", "Specificity (TNR)", "What is now proved was once only imagined. — William Blake")
_r("balan", "balanced_accuracy", "Classification", "Balanced accuracy", "Distribution helper.")
_r("youdj", "youdens_j", "Classification", "Youden's J statistic", "Find the balance point. -- Qui-Gon Jinn")
_r("mcfmt", "multiclass_confusion_matrix", "Classification", "Multi-class confusion matrix", "There are always two sides. -- Count Dooku")
_r("maccr", "multiclass_precision", "Classification", "Macro/micro/weighted precision", "What is now proved was once only imagined. — William Blake")
_r("mrecl", "multiclass_recall", "Classification", "Macro/micro/weighted recall", "Much to learn, you still have.")
_r("mf1", "multiclass_f1", "Classification", "Macro/micro/weighted F1", "In my experience there is no such thing as luck. --")
_r("clrpt", "classification_report", "Classification", "Classification report (per-class metrics)", "What is now proved was once only imagined. — William Blake")
_r("roccv", "roc_curve", "Classification", "ROC curve (FPR, TPR at thresholds)", "Choose wisely.")
_r("aucroc", "auc_roc", "Classification", "AUC-ROC via trapezoidal integration", "What is now proved was once only imagined. — William Blake")
_r("prcv", "pr_curve", "Classification", "Precision-recall curve", "The truth is often what we make of it. --")
_r("aucpr", "auc_pr", "Classification", "AUC-PR (area under PR curve)", "Distribution helper.")
_r("optth", "optimal_threshold", "Classification", "Optimal threshold selection (Youden/cost)", "The right choice is the hardest one. -- Ahsoka")

_r("kmsrv", "kaplan_meier_curve", "Survival", "Kaplan-Meier survival curve with CIs", "Death is a natural part of life.")
_r("naest", "nelson_aalen_hazard", "Survival", "Nelson-Aalen cumulative hazard estimator", "The dark side clouds everything.")
_r("lgrst", "log_rank_test", "Survival", "Log-rank test for two survival curves", "There can be only one. -- Highlander")
_r("grest", "gehan_breslow_test", "Survival", "Gehan-Breslow generalized Wilcoxon test", "The first step is the hardest. -- Ahsoka")
_r("trent", "survival_trend_test", "Survival", "Trend test for survival across ordered groups", "A pattern emerges. -- Dooku")
_r("rmstd", "rmst_difference", "Survival", "Restricted mean survival time difference", "Time is what you make of it. -- Qui-Gon")
_r("coxsn", "cox_snell_residuals", "Survival", "Cox-Snell residuals for model diagnostics", "Trust your instincts. -- Qui-Gon")
_r("mtrst", "martingale_residuals", "Survival", "Martingale residuals for Cox model", "The future is always in motion.")
_r("strtf", "stratified_cox", "Survival", "Stratified Cox proportional hazards model", "What is now proved was once only imagined. — William Blake")
_r("tdcox", "time_dependent_cox", "Survival", "Time-dependent covariates Cox model", "Time changes all things. --")
_r("phreg", "piecewise_hazard_regression", "Survival", "Piecewise constant hazard regression", "Piece by piece. -- Grievous")
_r("weibs", "weibull_survival", "Survival", "Weibull AFT survival model", "Flexible yet strong. -- Mace Windu")
_r("logns", "lognormal_survival", "Survival", "Log-normal AFT survival model", "Normal is relative. -- Padme")
_r("gngsv", "generalized_gamma_survival", "Survival", "Generalized gamma survival model", "All paths converge.")
_r("expsv", "exponential_survival", "Survival", "Exponential survival model", "Simplicity is the ultimate sophistication. -- Qui-Gon")
_r("gompr", "gompertz_survival", "Survival", "Gompertz survival model", "Age brings wisdom.")
_r("frglt", "frailty_model", "Survival", "Shared frailty survival model", "We are all connected. -- Chirrut")
_r("mxcrk", "mixture_cure_model", "Survival", "Mixture cure survival model", "Some wounds heal completely. -- Ahsoka")
_r("fgreg", "fine_gray_regression", "Survival", "Fine-Gray competing risks regression", "Competition sharpens the blade. -- Maul")
_r("aalen", "aalen_additive_hazards", "Survival", "Aalen additive hazards model", "Addition, not multiplication. -- C-3PO")
_r("recur", "andersen_gill_recurrent", "Survival", "Andersen-Gill recurrent event model", "What is now proved was once only imagined. — William Blake")
_r("mstat", "multistate_transitions", "Survival", "Multi-state model transition analysis", "Many paths, one journey.")
_r("clndn", "concordance_index", "Survival", "Concordance index (C-statistic)", "Agreement is the foundation of trust. -- Padme")
_r("brscr", "brier_score_survival", "Survival", "Brier score for survival prediction", "Accuracy matters. -- Admiral Ackbar")
_r("ipcw", "ipcw_weights", "Survival", "Inverse probability of censoring weights", "Balance in all things.")
_r("intcn", "interval_censoring", "Survival", "Turnbull NPMLE for interval-censored data", "Between the lines, truth lies. --")
_r("lftrt", "left_truncation", "Survival", "Left-truncated survival estimator", "Late arrivals still count. --")
_r("jnpnt", "joinpoint_survival", "Survival", "Joinpoint survival regression", "Every turning point matters. -- Ahsoka")
_r("pchzr", "piecewise_hazard_rate", "Survival", "Piecewise constant hazard rate", "Step by step. -- Luke")
_r("srdur", "survival_duration_stats", "Survival", "Survival duration statistics (median, percentiles)", "How long will it last? --")
_r("smhaz", "smoothed_hazard", "Survival", "Kernel-smoothed hazard function", "Smooth the rough edges. -- Qui-Gon")
_r("chzrd", "cause_specific_hazard", "Survival", "Cause-specific hazard ratio", "Know the cause.")
_r("rstst", "restricted_survival_test", "Survival", "Restricted survival time test", "Limits define possibility. --")
_r("srmdc", "survival_discrimination", "Survival", "Survival model discrimination measures", "Separate the wheat from the chaff. -- Mace Windu")

_r("armod", "ar_fit", "TimeSeries", "AR(p) model via Yule-Walker equations", "Always in motion is the future.")
_r("mamod", "ma_fit", "TimeSeries", "MA(q) model via innovations algorithm", "Patience you must have.")
_r("armam", "arma_fit", "TimeSeries", "ARMA(p,q) model via conditional MLE", "What has come before shapes what follows.")
_r("arimm", "arima_fit", "TimeSeries", "ARIMA(p,d,q) via differencing + conditional MLE", "There is good in him. -- Luke")
_r("sarim", "sarima_fit", "TimeSeries", "Seasonal ARIMA (SARIMA) model", "Distribution helper.")
_r("archt", "arch_fit", "TimeSeries", "ARCH(p) conditional heteroscedasticity model", "Fear leads to volatility.")
_r("varm", "var_fit", "TimeSeries", "VAR(p) vector autoregression via OLS", "Together we are stronger. -- Jyn Erso")
_r("grang", "granger_test", "TimeSeries", "Granger causality F-test", "One thing follows another. -- Qui-Gon")
_r("fevdc", "fevd", "TimeSeries", "Forecast error variance decomposition", "Every action has consequences. --")
_r("stldc", "stl_decompose", "TimeSeries", "STL seasonal-trend decomposition", "We must go deeper. -- Qui-Gon")
_r("sespn", "ses", "TimeSeries", "Simple exponential smoothing", "Smooth out the noise. --")
_r("dexpn", "des", "TimeSeries", "Double exponential smoothing (Holt linear)", "Two there should be.")
_r("mavar", "moving_average", "TimeSeries", "Moving average smoother (simple/weighted/EMA)", "A rolling stone gathers no moss.")
_r("adfrr", "adf_test", "TimeSeries", "Augmented Dickey-Fuller unit root test", "Something is out of balance. -- Luke")
_r("kpsst", "kpss_test", "TimeSeries", "KPSS stationarity test", "Steady. Steady. -- Poe Dameron")
_r("pptet", "pp_test", "TimeSeries", "Phillips-Perron unit root test", "Look deeper you must.")
_r("zands", "za_test", "TimeSeries", "Zivot-Andrews structural break test", "There has been an awakening. -- Snoke")
_r("chows", "chow_test", "TimeSeries", "Chow structural break F-test", "Distribution helper.")
_r("cusms", "cusum_test", "TimeSeries", "CUSUM structural change test", "Trust the process. -- Rex")
_r("ljbxt", "ljung_box", "TimeSeries", "Ljung-Box portmanteau test", "There is a disturbance. -- Mace Windu")
_r("bgtet", "bg_test", "TimeSeries", "Breusch-Godfrey serial correlation LM test", "Dig deeper you must.")
_r("dwtst", "durbin_watson", "TimeSeries", "Durbin-Watson test statistic", "Stay on course. -- Admiral Ackbar")
_r("specn", "periodogram", "TimeSeries", "Spectral density via periodogram", "Hidden frequencies there are.")
_r("cohrc", "coherence", "TimeSeries", "Magnitude-squared coherence between series", "Connected everything is.")
_r("mapef", "mape", "TimeSeries", "Mean Absolute Percentage Error", "What is now proved was once only imagined. — William Blake")
_r("rmsfe", "rmsfe_calc", "TimeSeries", "Root Mean Square Forecast Error", "What is now proved was once only imagined. — William Blake")
_r("dmtst", "dm_test", "TimeSeries", "Diebold-Mariano forecast comparison test", "The best forecast wins. -- Lando")
_r("state", "kalman_filter", "TimeSeries", "Kalman filter for state-space models", "See through the noise you can.")
_r("dynic", "dlm_fit", "TimeSeries", "Dynamic linear model (local level/trend)", "Everything flows. -- Qui-Gon")
_r("tarlg", "tar_fit", "TimeSeries", "Threshold autoregression (TAR) model", "What is now proved was once only imagined. — William Blake")
_r("mstar", "ms_ar", "TimeSeries", "Markov-switching AR model", "Switching sides again? -- Ahsoka")
_r("coitg", "eg_coint", "TimeSeries", "Engle-Granger cointegration test", "Distribution helper.")
_r("johcg", "johansen_test", "TimeSeries", "Johansen cointegration trace test", "The bonds between us are strong. --")
_r("ludc", "lu_decomposition", "LinearAlgebra", "LU decomposition with partial pivoting", "In my experience there is no such thing as luck. --")
_r("qrdcp", "qr_decomposition", "LinearAlgebra", "QR decomposition (Householder)", "Let go of your conscious self and act on instinct. --")
_r("svdcp", "svd_compute", "LinearAlgebra", "Singular value decomposition", "Luminous beings are we, not this crude matter.")
_r("chles", "cholesky_solve", "LinearAlgebra", "Cholesky decomposition and solve", "There is another.")
_r("eigsm", "eigen_symmetric", "LinearAlgebra", "Eigenvalue decomposition (symmetric)", "Train yourself to let go.")
_r("lancs", "lanczos", "LinearAlgebra", "Lanczos algorithm (largest eigenvalues)", "Distribution helper.")
_r("arnld", "arnoldi", "LinearAlgebra", "Arnoldi iteration (Krylov subspace)", "What is now proved was once only imagined. — William Blake")
_r("gmrss", "gmres_solve", "LinearAlgebra", "GMRES iterative solver", "What is now proved was once only imagined. — William Blake")
_r("cgsol", "conjugate_gradient", "LinearAlgebra", "Conjugate gradient solver (SPD)", "Do or do not. There is no try.")
_r("biccg", "bicgstab", "LinearAlgebra", "BiCGSTAB solver (non-symmetric)", "I have a bad feeling about this. -- Everyone")
_r("jacbi", "jacobi_solve", "LinearAlgebra", "Jacobi iterative solver", "Patience you must have.")
_r("gseid", "gauss_seidel", "LinearAlgebra", "Gauss-Seidel iterative solver", "Your eyes can deceive you. Don't trust them. --")
_r("sorsl", "sor_solve", "LinearAlgebra", "SOR (successive over-relaxation) solver", "What is now proved was once only imagined. — William Blake")
_r("mginv", "pseudoinverse", "LinearAlgebra", "Moore-Penrose pseudoinverse (SVD)", "There is always room for one more. -- Enfys Nest")
_r("lstqr", "lstsq_qr", "LinearAlgebra", "Least squares via QR decomposition", "Stay on target. -- Gold Five")
_r("trlsq", "total_least_squares", "LinearAlgebra", "Total least squares (errors-in-variables)", "The truth is often what we make of it. --")
_r("nnsls", "nnls", "LinearAlgebra", "Non-negative least squares (Lawson-Hanson)", "What is now proved was once only imagined. — William Blake")
_r("spdmt", "sparse_diagonal", "LinearAlgebra", "Sparse diagonal matrix construction", "What is now proved was once only imagined. — William Blake")
_r("bndmt", "banded_solve", "LinearAlgebra", "Banded matrix solver", "What is now proved was once only imagined. — William Blake")
_r("tridg", "thomas_solve", "LinearAlgebra", "Tridiagonal solver (Thomas algorithm)", "This is where the fun begins. -- Anakin")
_r("krncr", "kronecker", "LinearAlgebra", "Kronecker (tensor) product", "Distribution helper.")
_r("hadpr", "hadamard_product", "LinearAlgebra", "Hadamard (element-wise) product", "We stand here amidst my achievement. -- Krennic")
_r("matnm", "matrix_norms", "LinearAlgebra", "Matrix norms (1, 2, inf, Frobenius)", "Size matters not.")
_r("mtexp", "matrix_exp", "LinearAlgebra", "Matrix exponential (Pade approximation)", "What is now proved was once only imagined. — William Blake")
_r("mtlog", "matrix_log", "LinearAlgebra", "Matrix logarithm", "You must unlearn what you have learned.")
_r("mtsqr", "matrix_sqrt", "LinearAlgebra", "Matrix square root (Denman-Beavers)", "Always two there are.")
_r("newtm", "newton_method", "Optimization", "Newton's method (1D and multivariate)", "Your focus determines your reality. -- Qui-Gon")
_r("secnt", "secant_method", "Optimization", "Secant method (root finding)", "Sometimes we must let go of our pride. -- Padme")
_r("brtrf", "brent_root", "Optimization", "Brent's method (root finding)", "So certain are you.")
_r("gdsct", "gradient_descent", "Optimization", "Gradient descent with momentum", "One step at a time. --")
_r("adamm", "adam_optimize", "Optimization", "Adam optimizer (Kingma & Ba)", "Adapt or perish. -- Grand Inquisitor")
_r("adagr", "adagrad_optimize", "Optimization", "Adagrad adaptive learning rate", "Knowledge is power. -- Thrawn")
_r("rmspd", "rmsprop_optimize", "Optimization", "RMSProp optimizer", "Distribution helper.")
_r("lbfgm", "lbfgs_optimize", "Optimization", "L-BFGS quasi-Newton optimizer", "What is now proved was once only imagined. — William Blake")
_r("simpx", "simplex_lp", "Optimization", "Simplex method (linear programming)", "A more elegant weapon for a more civilized age. --")
_r("inpnt", "interior_point_lp", "Optimization", "Interior point method (LP)", "Distribution helper.")
_r("augla", "augmented_lagrangian", "Optimization", "Augmented Lagrangian method", "Difficult to see. Always in motion is the future.")
_r("sqprg", "sqp_optimize", "Optimization", "SQP (sequential quadratic programming)", "What is now proved was once only imagined. — William Blake")
_r("pswrm", "particle_swarm", "Optimization", "Particle swarm optimization", "We are the spark that will light the fire. -- Poe")
_r("difev", "differential_evolution", "Optimization", "Differential evolution global optimizer", "The strongest stars have hearts of kyber. -- Chirrut")

# ── Epidemiology (Extended Methods) ────────────────────────────────────────
_r("seirc", "seir_compartmental", "Epidemiology", "SEIR with vital dynamics and waning immunity", "The plague spreads through generations.")
_r("sirsv", "sirs_vaccination", "Epidemiology", "SIRS model with vaccination and waning immunity", "Vaccination, our only hope it is.")
_r("sirdm", "sir_age_demographics", "Epidemiology", "SIR with age-structured demographics", "Age matters not.")
_r("r0est", "r0_next_generation", "Epidemiology", "R0 via next-generation matrix method", "The next generation will finish what we started. -- Kylo")
_r("rtefv", "rt_effective", "Epidemiology", "Effective reproduction number Rt (EpiEstim)", "Distribution helper.")
_r("srint", "serial_interval", "Epidemiology", "Serial interval distribution estimation", "One leads to the next. -- Dooku")
_r("gntme", "generation_time", "Epidemiology", "Generation time distribution estimation", "From one generation to the next. --")
_r("incub", "incubation_period", "Epidemiology", "Incubation period distribution estimation", "What is now proved was once only imagined. — William Blake")
_r("scdly", "case_delay", "Epidemiology", "Surveillance case reporting delay", "Delayed, but not forgotten. -- Padme")
_r("epcrv", "epidemic_curve_analysis", "Epidemiology", "Epidemic curve growth dynamics analysis", "Rising fast, the tide is.")
_r("secrt", "secondary_attack_rate", "Epidemiology", "Secondary attack rate with exact CI", "Spread to those closest, it does.")
_r("cmplt", "capture_recapture", "Epidemiology", "Completeness via two-source capture-recapture", "Twice counted, truly seen. -- Chirrut")
_r("stdmr", "standardized_mortality_ratio", "Epidemiology", "SMR with exact Poisson CI", "Compared to the standard, how do we fare? -- Mon Mothma")
_r("dsmrt", "direct_standardization", "Epidemiology", "Directly age-standardized rate", "Direct comparison reveals the truth. --")
_r("ismrt", "indirect_standardization", "Epidemiology", "Indirectly standardized rate (SMR method)", "Indirect paths, the truth they find.")
_r("yllst", "years_of_life_lost_std", "Epidemiology", "YLL with discounting and age weighting", "Years lost, the cost of war. -- Padme")
_r("yldst", "years_lived_with_disability", "Epidemiology", "Years lived with disability (YLD)", "Living with pain, many do.")
_r("dalyc", "daly_computation", "Epidemiology", "DALY with GBD discounting and age-weighting", "The burden of suffering, measured it must be.")
_r("qalys", "qaly_computation", "Epidemiology", "Quality-adjusted life years (QALY)", "Quality of life, treasure it we must.")
_r("lftbl", "life_table_complete", "Epidemiology", "Complete single-year life table", "The full table of life and death.")
_r("srvfn", "survival_function", "Epidemiology", "Survival function S(x) from life table qx", "Survival, the strongest instinct. -- Jyn Erso")
_r("hzdrt", "hazard_rate", "Epidemiology", "Age-specific hazard rates from life table", "The hazard grows with each passing moment. -- Bail Organa")
_r("acmrt", "age_cause_mortality", "Epidemiology", "Age-cause-specific mortality rates", "Each cause, its own toll it takes.")
_r("infrt", "infant_mortality_rate", "Epidemiology", "Infant mortality rate with CI", "The youngest, the most vulnerable. -- Padme")
_r("mmrat", "maternal_mortality_ratio", "Epidemiology", "Maternal mortality ratio with CI", "What is now proved was once only imagined. — William Blake")
_r("psafs", "population_attributable_fraction", "Epidemiology", "Population attributable fraction (PAF)", "How much the galaxy suffers from this cause. -- Bail Organa")
_r("etagr", "etiologic_fraction", "Epidemiology", "Etiologic fraction (AF among exposed)", "The cause within the exposed. -- Medical droid")
_r("casct", "case_control_or", "Epidemiology", "Case-control odds ratio with Woolf CI", "What is now proved was once only imagined. — William Blake")
_r("cohrt", "cohort_risk_ratio", "Epidemiology", "Cohort study risk ratio with CI", "Follow them through time, we must.")
_r("mhors", "mantel_haenszel_or", "Epidemiology", "Mantel-Haenszel pooled odds ratio", "Pooled across strata, the truth emerges. --")
_r("strta", "stratified_analysis", "Epidemiology", "Stratified 2x2xK with Breslow-Day test", "Layer by layer, confounding peels away. -- Qui-Gon")
_r("doseq", "dose_response", "Epidemiology", "Dose-response analysis (logistic/probit)", "What is now proved was once only imagined. — William Blake")
_r("brktm", "berkson_bias_test", "Epidemiology", "Berkson's bias diagnostic test", "Bias clouds everything.")
_r("scrnp", "screening_properties", "Epidemiology", "Screening test sens/spec/PPV/NPV", "Detect early, we must.")
_r("cltrn", "cluster_trial_size", "Epidemiology", "Cluster randomized trial sample size", "Clustered together, stronger they are.")
_r("stpwz", "stepped_wedge_design", "Epidemiology", "Stepped-wedge trial power calculation", "Step by step, the intervention spreads. -- Cassian Andor")
_r("intrl", "interrupted_time_series", "Epidemiology", "Segmented regression ITS analysis", "The interruption changed everything. -- Jyn Erso")
_r("synds", "syndromic_surveillance", "Epidemiology", "EARS syndromic surveillance algorithm", "Watch for the signs, we must.")
_r("abrtm", "farrington_aberration", "Epidemiology", "Farrington aberration detection algorithm", "Something is not right. I can feel it. -- Anakin")
_r("xsrsk", "excess_risk", "Epidemiology", "Excess risk estimation", "More than expected, a disturbance there is.")

# ── Robust Statistics & Nonparametric Methods (2026-04-15) ─────────────────
_r("tmean", "trimmed_mean", "Robust", "Trimmed mean", "Cut away the excess you must.")
_r("wmean", "winsorized_mean", "Robust", "Winsorized mean", "Contain the extremes you shall.")
_r("medab", "median_abs_dev", "Robust", "Median absolute deviation", "The center holds. -- Chirrut")
_r("tausc", "tau_scale", "Robust", "Tau scale estimator (Maronna-Zamar)", "Two steps to truth. --")
_r("sn_es", "sn_estimator", "Robust", "Sn estimator (Rousseeuw-Croux)", "Measure with care. -- Mon Mothma")
_r("qn_es", "qn_estimator", "Robust", "Qn estimator (Rousseeuw-Croux)", "Size matters not.")
_r("mmslt", "mm_estimator", "Robust", "MM-estimator for regression (Yohai)", "High breakdown and efficiency. -- Qui-Gon")
_r("ltses", "least_trimmed_squares", "Robust", "Least trimmed squares (Rousseeuw)", "Trim the noise you must.")
_r("lmses", "least_median_squares", "Robust", "Least median of squares (Rousseeuw)", "The median path is robust. --")
_r("mvout", "multivariate_outlier", "Robust", "Multivariate outlier detection", "Sense the disturbance. -- Mace Windu")
_r("rbpca", "robust_pca_pp", "Robust", "Robust PCA via projection pursuit", "Find the hidden direction. -- Ahsoka")
_r("hbrrg", "huber_regression", "Robust", "Huber robust regression (IRLS)", "Steady against the outliers. -- Admiral Ackbar")
_r("bfreg", "biweight_regression", "Robust", "Biweight (Tukey bisquare) regression", "Hard shell, soft core. -- Mandalorian proverb")
_r("sigrn", "sign_rank_test", "NonparametricTest", "Wilcoxon signed-rank test", "Feel the difference. -- Luke")
_r("rnksm", "rank_sum_test", "NonparametricTest", "Wilcoxon rank-sum test", "Distribution helper.")
_r("kstst", "ks_test", "NonparametricTest", "Kolmogorov-Smirnov test", "The distance between distributions. -- Dooku")
_r("adtet", "anderson_darling_test", "NonparametricTest", "Anderson-Darling test", "What is now proved was once only imagined. — William Blake")
_r("cvm", "cramer_von_mises_test", "NonparametricTest", "Cramer-von Mises test", "Integrate the difference. -- Dooku")
_r("lbrst", "lilliefors_test", "NonparametricTest", "Lilliefors normality test", "Estimate and test. --")
_r("jbtet", "jarque_bera_test", "NonparametricTest", "Jarque-Bera normality test", "What is now proved was once only imagined. — William Blake")
_r("bptet", "bowman_shenton_test", "NonparametricTest", "Bowman-Shenton normality test", "An omnibus of evidence. -- Bail Organa")
_r("runst", "runs_test", "NonparametricTest", "Wald-Wolfowitz runs test", "Random the pattern must be.")
_r("sgchg", "sign_change_test", "NonparametricTest", "Sign test for the median", "Above or below, the sign reveals. -- Chirrut")
_r("boot2", "bootstrap_two_sample", "NonparametricTest", "Two-sample bootstrap test", "What is now proved was once only imagined. — William Blake")
_r("prmts", "permutation_test_two", "NonparametricTest", "Two-sample permutation test", "Shuffle the deck. -- Lando")
_r("ptbis", "point_biserial", "Correlation", "Point-biserial correlation", "Binary meets continuous. -- C-3PO")
_r("kndll", "kendall_concordance", "Correlation", "Kendall's W (concordance)", "Agreement among judges. -- Mon Mothma")
_r("somrd", "somers_d", "Correlation", "Somers' D ordinal association", "Direction matters. -- Admiral Holdo")
_r("gdman", "goodman_kruskal_gamma", "Correlation", "Goodman-Kruskal gamma", "Concordance over discordance. -- Padme")
_r("rnkcr", "rank_correlation_test", "Correlation", "Rank correlation test", "Rank by rank, the truth emerges. --")
_r("sprmn", "spearman_corr", "Correlation", "Spearman rank correlation", "Monotone the relationship is.")
_r("pknds", "partial_kendall", "Correlation", "Partial Kendall's tau", "Control for the confounder. -- Qui-Gon")
_r("nvrmt", "normal_var_ratio_test", "Test", "Normal variance ratio F-test", "Equal variances or not. -- Mace Windu")
_r("gldvr", "goldfeld_quandt_test", "Test", "Goldfeld-Quandt heteroscedasticity test", "Split and compare. -- General Grievous")
_r("bpgtr", "breusch_pagan_test", "Test", "Breusch-Pagan heteroscedasticity test", "The residuals speak.")
_r("whtet", "white_heterosc_test", "Test", "White's heteroscedasticity test", "No assumptions needed. --")
_r("ramsy", "ramsey_reset_test", "Test", "Ramsey RESET test", "Is the model misspecified? -- Thrawn")


# ── Multivariate Methods ───────────────────────────────────────────────────
_r("pcaev", "pca_eigen", "Multivariate", "PCA via eigenvalue decomposition", "The principal path, reveal it does.")
_r("pcsvd", "pca_svd", "Multivariate", "PCA via singular value decomposition", "What is now proved was once only imagined. — William Blake")
_r("kpca", "kernel_pca", "Multivariate", "Kernel PCA (RBF/poly/linear)", "Through the kernel, dimensions unfold. -- Qui-Gon")
_r("spcae", "sparse_pca", "Multivariate", "Sparse PCA via iterative thresholding", "Sparsity is the path to clarity.")
_r("ppurs", "projection_pursuit", "Multivariate", "Projection pursuit (kurtosis/negentropy)", "Pursue the interesting projections. -- Mace Windu")
_r("fanlz", "factor_analysis_ml", "Multivariate", "Maximum likelihood factor analysis", "Distribution helper.")
_r("efanl", "efa_principal_axis", "Multivariate", "Exploratory factor analysis (principal axis)", "Explore the latent structure, you must.")
_r("cfanl", "cfa_uls", "Multivariate", "Confirmatory factor analysis (ULS)", "Confirm the model, validate the theory. --")
_r("profl", "procrustes", "Multivariate", "Orthogonal Procrustes rotation", "Rotate to align, the shapes must. -- Dooku")
_r("vrimx", "varimax", "Multivariate", "Varimax orthogonal rotation", "Maximise the variance, simple the structure.")
_r("oblmn", "oblimin", "Multivariate", "Direct oblimin oblique rotation", "Oblique the path, correlated the factors. -- Qui-Gon")
_r("prmxr", "promax", "Multivariate", "Promax oblique rotation", "Beyond varimax, the oblique extends. -- Mace Windu")
_r("nmds", "nmds", "Multivariate", "Non-metric MDS (Kruskal stress)", "Preserve the ranks, not the distances. -- Thrawn")
_r("semap", "sammon_mapping", "Multivariate", "Sammon mapping dimensionality reduction", "Map the distances, preserve the structure. --")
_r("ldanl", "lda", "Multivariate", "Fisher's linear discriminant analysis", "Discriminate the classes, the Fisher way.")
_r("qdanl", "qda", "Multivariate", "Quadratic discriminant analysis", "Each class, its own covariance has.")
_r("rdanl", "rda", "Multivariate", "Regularized discriminant analysis (Friedman)", "Between LDA and QDA, balance you must.")
_r("cancr", "canonical_correlation", "Multivariate", "Canonical correlation analysis", "Correlate the canonical variates. -- Qui-Gon")
_r("manov", "manova", "Multivariate", "MANOVA (Pillai/Wilks/Hotelling/Roy)", "Multivariate analysis of variance. -- Mace Windu")
_r("hott2", "hotelling_t2", "Multivariate", "Hotelling's T-squared test", "The multivariate t-test, this is.")
_r("boxmt", "box_m_test", "Multivariate", "Box's M test (covariance equality)", "Equal the covariances must be.")
_r("mrtsq", "mauchly_test", "Multivariate", "Mauchly's sphericity test", "Spherical, the covariance matrix is not.")
_r("kmcls", "kmeans", "Clustering", "K-means clustering (Lloyd's algorithm)", "Cluster the points, minimise the inertia. -- Thrawn")
_r("hclus", "hierarchical_cluster", "Clustering", "Hierarchical clustering (ward/single/complete)", "A hierarchy of clusters, this builds. -- Dooku")
_r("optic", "optics", "Clustering", "OPTICS density-based clustering", "Order the points, identify the structure. -- Thrawn")
_r("gmmcl", "gmm_cluster", "Clustering", "Gaussian mixture model (EM)", "A mixture of Gaussians, the data is.")
_r("sihlh", "silhouette", "Clustering", "Silhouette coefficient", "How well separated, the clusters are.")
_r("chand", "calinski_harabasz", "Clustering", "Calinski-Harabasz variance ratio index", "The variance ratio, higher is better. -- Qui-Gon")
_r("dbind", "davies_bouldin", "Clustering", "Davies-Bouldin index", "Lower the index, better the clustering. --")
_r("elbmk", "elbow_method", "Clustering", "Elbow method for optimal k", "Find the elbow, the optimal k reveals.")
_r("nmisc", "nmi", "Clustering", "Normalized mutual information", "Mutual information, normalised it must be.")
_r("arisc", "adjusted_rand_index", "Clustering", "Adjusted Rand index", "Adjust for chance, the Rand index must. --")
_r("copha", "cophenetic_correlation", "Clustering", "Cophenetic correlation for dendrograms", "Faithful the dendrogram must be.")
_r("mclst", "model_based_cluster", "Clustering", "Model-based clustering (BIC selection)", "Select by BIC, the best model finds. -- Qui-Gon")
_r("spclr", "spectral_clustering", "Clustering", "Spectral clustering via graph Laplacian", "Through the spectrum, clusters emerge. -- Mace Windu")
_r("kmoid", "kmedoids", "Clustering", "K-medoids (PAM algorithm)", "Medoids, not means, more robust they are.")

_r("bmlnr", "bayesian_linear_regression", "Bayesian", "Bayesian linear regression (conjugate NIG)", "In my experience there is no such thing as luck. --")
_r("blogt", "bayesian_logistic", "Bayesian", "Bayesian logistic (Laplace approximation)", "Your focus determines your reality. -- Qui-Gon")
_r("bnebi", "bayesian_negbinom", "Bayesian", "Bayesian negative binomial (beta conjugate)", "Always two there are.")
_r("bmulr", "bayesian_multinomial", "Bayesian", "Bayesian multinomial (Dirichlet conjugate)", "We are what they grow beyond.")
_r("bnoml", "bayesian_binomial", "Bayesian", "Bayesian binomial (beta conjugate)", "Distribution helper.")
_r("mhsmp", "metropolis_hastings", "Bayesian", "Metropolis-Hastings sampler", "A random walk to remember. --")
_r("rwmh", "random_walk_mh", "Bayesian", "Random walk MH with adaptive step", "Adapt or perish. -- Grand Moff Tarkin")
_r("hmcmc", "hamiltonian_mc", "Bayesian", "Hamiltonian Monte Carlo sampler", "The dark side clouds everything.")
_r("bnut", "nuts_sampler", "Bayesian", "No-U-Turn sampler (NUTS)", "What is now proved was once only imagined. — William Blake")
_r("slisp", "slice_sampler", "Bayesian", "Slice sampler", "Slice through the noise. -- Mace Windu")
_r("emvbs", "evidence_maximization", "Bayesian", "Evidence maximization / variational Bayes", "The evidence is clear. -- Mon Mothma")
_r("vbgmm", "vb_gaussian_mixture", "Bayesian", "Variational Bayes Gaussian mixture", "Distribution helper.")
_r("advi", "advi_meanfield", "Bayesian", "ADVI mean-field variational inference", "Vary your approach you must.")
_r("laapx", "laplace_approximation", "Bayesian", "Laplace approximation for posterior", "Approximate we must when exact we cannot.")
_r("bfsdm", "bayes_factor_savage_dickey", "Bayesian", "Bayes factor (Savage-Dickey)", "The truth is rarely simple. --")
_r("bdic", "bayesian_dic", "Bayesian", "DIC (deviance information criterion)", "Judge me by my size do you?")
_r("bwaic", "compute_waic", "Bayesian", "WAIC (widely applicable IC)", "Widely applicable this is.")
_r("bloos", "psis_loo", "Bayesian", "LOO-CV (Pareto-smoothed)", "Leave one out you must.")
_r("bpchk", "posterior_predictive_check", "Bayesian", "Posterior predictive check", "Trust but verify. -- Admiral Ackbar")
_r("gewek", "geweke_diagnostic", "Bayesian", "Geweke convergence diagnostic", "Convergence is the path to truth. -- Qui-Gon")
_r("rhatd", "rhat_diagnostic", "Bayesian", "R-hat (Gelman-Rubin) diagnostic", "All chains must agree. -- Mon Mothma")
_r("essn", "effective_sample_size", "Bayesian", "Effective sample size (ESS)", "Size matters not.")
_r("mcerr", "mcmc_se", "Bayesian", "MCMC standard error (batch means)", "Precision in all things. -- Grand Admiral Thrawn")
_r("trplt", "trace_plot_data", "Bayesian", "Trace plot data generation", "Follow the path of the chain. --")
_r("acplt", "autocorrelation_data", "Bayesian", "Autocorrelation plot data", "The past echoes in the present. -- Ahsoka")
_r("bnpar", "dp_density", "Bayesian", "Dirichlet process density estimation", "Infinite possibilities there are.")
_r("dpmmx", "dp_mixture_model", "Bayesian", "Dirichlet process mixture model", "Mixed allegiances lead to discovery. -- Padme")
_r("crppr", "chinese_restaurant_process", "Bayesian", "Chinese restaurant process", "At this table you will sit.")
_r("stckp", "stick_breaking", "Bayesian", "Stick-breaking construction", "Break the stick and find the weights. -- Chirrut")
_r("bhier", "bayesian_hierarchical", "Bayesian", "Bayesian hierarchical model (2-level)", "What is now proved was once only imagined. — William Blake")
_r("bshrk", "bayesian_horseshoe", "Bayesian", "Bayesian horseshoe shrinkage", "Shrink the irrelevant you must.")
_r("blasr", "bayesian_lasso", "Bayesian", "Bayesian LASSO regression", "Cut away the unnecessary. -- Mace Windu")
_r("brdge", "bayesian_ridge", "Bayesian", "Bayesian ridge regression", "Stay on the ridge. -- Gold Leader")
_r("bgpre", "bayesian_gp_regression", "Bayesian", "Bayesian GP regression", "The process is Gaussian. -- C-3PO")
_r("bgpcl", "bayesian_gp_classification", "Bayesian", "Bayesian GP classification", "Classify them I shall.")
_r("bkern", "bayesian_kernel_regression", "Bayesian", "Bayesian kernel regression", "The kernel of truth. --")
_r("bmaxe", "bayesian_model_averaging", "Bayesian", "Bayesian model averaging", "Average all the models you must.")
_r("bstep", "bayesian_changepoint", "Bayesian", "Bayesian change-point detection", "I sense a disturbance. --")
_r("bspln", "bayesian_spline", "Bayesian", "Bayesian spline regression", "Smooth the curve. --")
_r("bhawz", "bayesian_hazard", "Bayesian", "Bayesian piecewise hazard", "The hazard is always present. -- Cassian Andor")
_r("bsurv", "bayesian_survival", "Bayesian", "Bayesian Weibull survival model", "Survive we must.")
_r("bcauz", "bayesian_ate", "Bayesian", "Bayesian ATE (posterior causal)", "The cause reveals itself. -- Qui-Gon")
_r("bpscr", "bayesian_propensity", "Bayesian", "Bayesian propensity score", "What is now proved was once only imagined. — William Blake")
_r("bsynt", "bayesian_synthetic_control", "Bayesian", "Bayesian synthetic control", "A synthetic army I shall create. -- Dooku")
_r("bivrt", "bayesian_iv", "Bayesian", "Bayesian IV regression", "The instrument chooses the wielder. -- Maz Kanata")
_r("abcr", "abc_rejection", "Bayesian", "ABC rejection sampler", "Reject the implausible. -- Admiral Ackbar")
_r("abcmc", "abc_mcmc", "Bayesian", "ABC-MCMC sampler", "Approximate but march forward. -- Jyn Erso")
_r("empby", "empirical_bayes", "Bayesian", "Empirical Bayes shrinkage", "Learn the prior from the data.")
_r("ebmix", "eb_mixture", "Bayesian", "Empirical Bayes two-groups mixture", "Two groups there are.")

_r("jntnt", "joint_entropy", "InfoTheory", "Joint entropy H(X,Y)", "The belonging you seek is ahead. -- Maz Kanata")
_r("cndnt", "conditional_entropy", "InfoTheory", "Conditional entropy H(X|Y) = H(X,Y) - H(Y)", "In a dark place we find ourselves.")
_r("minfo", "mutual_information", "InfoTheory", "Mutual information I(X;Y)", "Let the past die. -- Kylo Ren")
_r("cmifn", "conditional_mutual_information", "InfoTheory", "Conditional mutual information I(X;Y|Z)", "Everything is connected. -- Chirrut")
_r("tsent", "tsallis_entropy", "InfoTheory", "Tsallis entropy of order q", "What is now proved was once only imagined. — William Blake")
_r("xentc", "cross_entropy", "InfoTheory", "Cross-entropy H(P,Q) = -sum p log q", "Distribution helper.")
_r("perpl", "perplexity", "InfoTheory", "Perplexity from log-probabilities", "Difficult to see. Always in motion is the future.")
_r("bpenc", "bits_per_char", "InfoTheory", "Bits per character (character-level entropy)", "Every word matters. -- Mon Mothma")
_r("fient", "fisher_information", "InfoTheory", "Fisher information via histogram density", "What is now proved was once only imagined. — William Blake")
_r("cramr", "cramer_rao_bound", "InfoTheory", "Cramer-Rao lower bound on estimator variance", "There are limits to everything. --")
_r("ticmp", "total_information_content", "InfoTheory", "Total information content N * H(X)", "The sum is greater than the parts.")
_r("aient", "aic_entropy", "InfoTheory", "AIC and AICc from log-likelihood", "Choose wisely, you must.")
_r("mdlen", "mdl", "InfoTheory", "Minimum description length criterion", "Brevity is the soul of wit. -- Thrawn")
_r("kolcm", "kolmogorov_complexity", "InfoTheory", "Kolmogorov complexity via compression", "Complexity hides simplicity. -- Qui-Gon")
_r("lzcmp", "lempel_ziv_complexity", "InfoTheory", "Lempel-Ziv complexity of binary sequence", "Patterns emerge from chaos. -- Ahsoka")
_r("smpnt", "sample_entropy", "InfoTheory", "Sample entropy (SampEn) of time series", "Regularity reveals structure.")
_r("apent", "approximate_entropy", "InfoTheory", "Approximate entropy (ApEn)", "Approximation leads to understanding. --")
_r("prmnt", "permutation_entropy", "InfoTheory", "Permutation entropy (normalised)", "Order in chaos there is.")
_r("mscle", "multiscale_entropy", "InfoTheory", "Multiscale entropy (MSE curve area)", "Scale matters not.")
_r("fuznt", "fuzzy_entropy", "InfoTheory", "Fuzzy entropy (FuzzyEn)", "Not everything is black and white. -- Ahsoka")
_r("dient", "dispersion_entropy", "InfoTheory", "Dispersion entropy (DispEn)", "Scatter reveals truth. -- Cassian Andor")
_r("trsfn", "transfer_entropy", "InfoTheory", "Transfer entropy Y->X (nonlinear Granger)", "Distribution helper.")
_r("normr", "normalized_redundancy", "InfoTheory", "Normalised redundancy I(X;Y)/min(H(X),H(Y))", "Redundancy is not waste. -- Admiral Ackbar")
_r("intgn", "integration", "InfoTheory", "Integration (neural complexity, Gaussian)", "Together we are stronger. -- Padme")
_r("phicx", "phi_complexity", "InfoTheory", "Phi complexity (integrated information)", "Consciousness is integration.")
_r("cmpls", "computable_complexity", "InfoTheory", "Computable complexity ratio (compression)", "The simple and the complex. -- Dooku")
_r("heaps", "heaps_law", "InfoTheory", "Heaps law vocabulary growth estimation", "New words, new worlds. -- C-3PO")
_r("frcdm", "fractal_dimension", "InfoTheory", "Fractal dimension (box-counting)", "Infinite detail in finite space.")
_r("rqent", "recurrence_entropy", "InfoTheory", "Recurrence quantification entropy (RQA)", "What is now proved was once only imagined. — William Blake")
_r("cornt", "correlation_entropy", "InfoTheory", "Correlation entropy K2 (Grassberger-Procaccia)", "Correlation is not causation. --")
_r("svdnt", "svd_entropy", "InfoTheory", "SVD entropy of embedded time series", "Decompose to understand.")
_r("wnent", "wiener_entropy", "InfoTheory", "Wiener entropy (spectral flatness)", "Noise reveals nature. --")
_r("itakr", "itakura_saito", "InfoTheory", "Itakura-Saito spectral distance", "What is now proved was once only imagined. — William Blake")

# ── Crimstat textbook formulas (Phase 1c, 2026-05-06) ──────────────────────
# Canonical crim-stat callables from Wooditch (Beginner's Guide, 2021) and
# Weisburd, Wilson, Wooditch & Britt (Advanced Statistics, 5e, 2022).
_r("xbar",   "xbar",            "CentralTendency",   "Sample arithmetic mean (canonical x̄)",                    "The simple things are the most powerful.")
_r("s2var",  "s2var",           "Dispersion",        "Unbiased sample variance (n−1 denominator)",                "Spread reveals nature. --")
_r("cohend", "cohend",          "EffectSize",        "Cohen's d for two-sample mean difference",                  "Size matters not.")
_r("mcfadr", "mcfadr",          "GoodnessOfFit",     "McFadden pseudo-R² for logistic regression",                "Likelihood reveals truth. -- Mace Windu")
_r("somerd", "somerd",          "Association",       "Somers' D — asymmetric ordinal-ordinal association",        "What is now proved was once only imagined. — William Blake")
_r("kentau", "kentau",          "Association",       "Kendall's τ-b — ordinal correlation with tie correction",   "Pairs reveal patterns. --")
_r("spearm", "spearm",          "Correlation",       "Spearman ρ rank correlation",                                "Rank, not value, what matters.")
_r("gkgam",  "gkgam",           "Association",       "Goodman-Kruskal γ — ordinal association ignoring ties",     "Concordance over chaos. -- Padme")

# ── Phase 4 follow-on: 14 more canonical stat callables (2026-05-06) ───────
# Identified via identify_wireable_eqs.py — concepts with textbook
# evidence in the 27,467-equation catalog and not yet in REGISTRY.
_r("covar",   "covar",          "Dispersion",         "Unbiased sample covariance",                                "Together stronger they are.")
_r("skew",    "skew",           "Dispersion",         "Sample skewness (Fisher-Pearson g₁)",                       "Asymmetry reveals truth. -- Mace Windu")
_r("kurt",    "kurt",           "Dispersion",         "Sample kurtosis (Fisher excess g₂)",                        "Tails matter. -- Lando")
_r("iqrng",   "iqrng",          "Dispersion",         "Interquartile range (Q₃ − Q₁)",                              "The middle path. --")
_r("mad",     "mad",            "Dispersion",         "Median absolute deviation (robust spread)",                  "Outliers cannot move the centre. -- Ahsoka")
_r("odds",    "odds",           "Association",        "Odds ratio for 2×2 table",                                   "What is now proved was once only imagined. — William Blake")
_r("relrsk",  "relrsk",         "Association",        "Relative risk (risk ratio) for 2×2 table",                   "Risk weighed against risk. -- Padme")
_r("akike",   "akike",          "ModelSelection",     "Akaike Information Criterion (AIC)",                         "Information has cost. -- Cassian")
_r("bayic",   "bayic",          "ModelSelection",     "Bayesian (Schwarz) Information Criterion (BIC)",             "Bayes weighs evidence.")
_r("manwhi",  "manwhi",         "Inference",          "Mann-Whitney U / Wilcoxon rank-sum test",                    "Rank reveals what value hides.")
_r("kwallis", "kwallis",        "Inference",          "Kruskal-Wallis H-test (nonparametric ANOVA)",                "Many groups, one truth seek.")
_r("wilcoxn", "wilcoxn",        "Inference",          "Wilcoxon signed-rank test (paired)",                         "Paired strength. --")
_r("ksonebs", "ksonebs",        "GoodnessOfFit",      "Kolmogorov-Smirnov test (1-sample / 2-sample)",              "Distance between distributions. -- Mace Windu")
_r("shapir",  "shapir",         "GoodnessOfFit",      "Shapiro-Wilk test for Normality",                            "Bell-shape, true or false. --")

# ── Phase 4 follow-on round 2 (2026-05-06): 17 more canonical callables ────
_r("logodds", "logodds",        "Logistic",           "Log-odds (logit) transform: ln(p / (1-p))",                 "The link of choice.")
_r("rdgr",    "rdgr",           "Regression",         "Ridge regression (L2 regularised OLS)",                      "Shrinkage tames variance. --")
_r("lasr",    "lasr",           "Regression",         "Lasso regression (L1 regularised OLS)",                      "Sparsity reveals essence.")
_r("elnetr",  "elnetr",         "Regression",         "Elastic Net (mixed L1/L2)",                                  "Balance of two paths. -- Mace Windu")
_r("glmpoi",  "glmpoi",         "Regression",         "Poisson regression (log link, GLM)",                         "Counts, not continuums. -- Padme")
_r("wasdst",  "wasdst",         "Distance",           "1D Wasserstein-1 (earth mover's) distance",                  "Mass moves where it must.")
_r("wald",    "wald",           "Inference",          "Wald test statistic for a single coefficient",               "What is now proved was once only imagined. — William Blake")
_r("lrtst",   "lrtst",          "Inference",          "Likelihood ratio test for nested models",                    "Likelihood compared. -- Mace Windu")
_r("trmean",  "trmean",         "CentralTendency",    "Symmetric trimmed mean (robust)",                            "Outliers trimmed, truth remains.")
_r("geomean", "geomean",        "CentralTendency",    "Geometric mean (positive data)",                              "Multiplicative balance. -- Padme")
_r("harmean", "harmean",        "CentralTendency",    "Harmonic mean (rates / reciprocals)",                         "Reciprocal harmony. -- Ahsoka")
_r("pcaprx",  "pcaprx",         "DimensionalityReduction", "Principal Component Analysis",                          "Decompose to comprehend.")
_r("kmeans2", "kmeans2",        "Clustering",         "K-means partition clustering",                               "Group what belongs together. --")
_r("shanon",  "shanon",         "InfoTheory",         "Shannon entropy of a distribution",                          "Uncertainty has a number. -- Cassian")
_r("loglik",  "loglik",         "Inference",          "Log-likelihood under a fitted distribution",                 "Likelihood is the bridge.")
_r("rsq",     "rsq",            "GoodnessOfFit",      "Coefficient of determination (R²)",                          "Variance explained. -- Mace Windu")
_r("diffd",   "diffd",          "CausalInference",    "Difference-in-Differences (canonical 2×2)",                  "Trends speak louder than levels. -- Padme")

# ── Atomic primitives (Phase 4 follow-on round 3, 2026-05-06) ──────────────
# Single-purpose decomposable building blocks for textbook formulas. When
# a user wants "just one step" of a derivation rather than the full
# operation, these are the callables to grab. Themed quotes intentionally
# mixed across SW / DC / Marvel / Matrix / anime / Transformers per Vee's
# request for variety.
_r("sumxsq",  "sumxsq",         "BasicMath",          "Σxᵢ² — uncorrected sum of squares",                          "What is now proved was once only imagined. — William Blake")
_r("sumdvsq", "sumdvsq",        "Variance",           "Σ(xᵢ − x̄)² — sum of squared deviations",                    "What is now proved was once only imagined. — William Blake")
_r("sst",     "sst",            "Variance",           "Total sum of squares (yᵢ − ȳ)²",                              "What is now proved was once only imagined. — William Blake")
_r("ssr",     "ssr",            "Variance",           "Regression sum of squares (ŷᵢ − ȳ)²",                        "Models illuminate, never invent. --")
_r("sse",     "sse",            "Variance",           "Error/residual sum of squares (yᵢ − ŷᵢ)²",                   "Errors are residue. -- GLaDOS")
_r("dftot",   "dftot",          "Inference",          "Total degrees of freedom (n − 1)",                            "What is now proved was once only imagined. — William Blake")
_r("dfreg",   "dfreg",          "Inference",          "Regression degrees of freedom (k slopes)",                    "What is now proved was once only imagined. — William Blake")
_r("dferr",   "dferr",          "Inference",          "Error degrees of freedom (n − k − 1)",                        "I aim to misbehave. -- Mal Reynolds")
_r("fstat",   "fstat",          "Inference",          "F-statistic for nested-model comparison",                     "Praise the Sun. -- Solaire")
_r("tstat",   "tstat",          "Inference",          "t-statistic for one coefficient",                             "I never asked for this. -- Adam Jensen")
_r("bonfer",  "bonfer",         "MultipleTesting",    "Bonferroni-corrected α' = α / m",                              "Live long and prosper. -- Spock")
_r("holm",    "holm",           "MultipleTesting",    "Holm-Bonferroni step-down per-test rejection",                "Wake up, Mr. Freeman. -- The G-Man")
_r("bhfdr",   "bhfdr",          "MultipleTesting",    "Benjamini-Hochberg FDR rejection vector",                     "I'll be back. -- Terminator")
_r("lev",     "lev",            "Diagnostics",        "Hat-matrix leverage diagonals h_ii",                          "Watch the points that pull. -- Storm")
_r("vif",     "vif",            "Diagnostics",        "Variance Inflation Factor 1/(1−Rⱼ²)",                          "Identify yourself. -- Cortana")
_r("studres", "studres",        "Diagnostics",        "Internally-studentized residuals",                            "Wake me when you need me. -- Cortana")
_r("crsspr",  "crsspr",         "BasicMath",          "Σ(xᵢ−x̄)(yᵢ−ȳ) — cross-product centered",                      "What is now proved was once only imagined. — William Blake")
_r("nfact",   "nfact",          "BasicMath",          "n! factorial",                                                  "By Grabthar's hammer. -- Galaxy Quest")
_r("ncombo",  "ncombo",         "BasicMath",          "C(n, k) binomial coefficient",                                 "Choose wisely. -- The Grail Knight")
_r("nperm",   "nperm",          "BasicMath",          "P(n, k) permutations",                                         "The cake is a lie. -- Portal")
_r("zci",     "zci",            "Inference",          "Half-width of z confidence interval",                           "Make it so. -- Picard")
_r("tci",     "tci",            "Inference",          "Half-width of t confidence interval",                           "So say we all. -- Adama")
_r("zcrit",   "zcrit",          "Inference",          "z critical value at α",                                          "Wibbly wobbly timey wimey. -- The Doctor")
_r("tcrit",   "tcrit",          "Inference",          "t critical value (df, α)",                                       "I have had enough of your snide insinuations. -- Mordin")
_r("chcrit",  "chcrit",         "Inference",          "χ² critical value upper-tail",                                    "My life for Aiur. -- Protoss Zealot")
_r("fcrit",   "fcrit",          "Inference",          "F critical value upper-tail",                                     "The spice must flow. -- Dune")
_r("cohensh", "cohensh",        "EffectSize",         "Cohen's h for two proportions",                                  "Wubba lubba dub dub. -- Rick Sanchez")
_r("hedgeg",  "hedgeg",         "EffectSize",         "Hedges' g (bias-corrected Cohen's d)",                          "Snake? Snake?! SNAAAKE! -- Metal Gear")
_r("expfrq",  "expfrq",         "ContingencyTables",  "Expected frequencies under independence",                       "Hey, listen! -- Navi")
_r("haz",     "haz",            "Survival",           "Instantaneous hazard rate λ(t) = f(t) / S(t)",                  "Wind's howling. -- Witcher")
_r("kldivg",  "kldivg",         "InfoTheory",         "Kullback-Leibler divergence Σpᵢ log(pᵢ/qᵢ)",                    "What is now proved was once only imagined. — William Blake")
_r("describe", "describe",      "Documentation",      "Pedagogical multi-section guide for any moirais.fn callable",   "Knowledge is power. -- Sherlock")

# ── Atomic primitives round 2 (2026-05-06): 30 more across Bayesian /
# multivariate / time-series / sampling / resampling / robust / power /
# distance / calibration / spatial. Theme variety: SW / Matrix /
# Transformers / scifi (Doctor Who, Star Trek, Dune, Mass Effect) /
# games (Halo, Metal Gear, Witcher, Dark Souls, Portal, StarCraft,
# TRON, Final Fantasy) / classic literature.
_r("priorbt", "priorbt",        "Bayesian",           "Beta-Binomial conjugate posterior update",                      "First, do no harm to your prior. -- Hippocrates / Spock")
_r("bayesf",  "bayesf",         "Bayesian",           "Bayes factor (BIC approximation)",                              "Belief is evidence weighted. -- Sherlock Holmes")
_r("hpdint",  "hpdint",         "Bayesian",           "Highest Posterior Density credible interval (1D)",              "What is now proved was once only imagined. — William Blake")
_r("mahalan", "mahalan",        "Multivariate",       "Mahalanobis distance √((x−μ)' Σ⁻¹ (x−μ))",                       "What is now proved was once only imagined. — William Blake")
_r("hotelt2", "hotelt2",        "Multivariate",       "Hotelling's T² (one-sample multivariate test)",                  "Multidimensional thinking required. -- Mordin Solus")
_r("lagop",   "lagop",          "TimeSeries",         "k-th lag operator on a series",                                  "Time is an illusion. Lunchtime, doubly so. -- HHGTTG")
_r("dffop",   "dffop",          "TimeSeries",         "k-th-order difference operator",                                 "Change is the only constant. -- Heraclitus")
_r("autocov", "autocov",        "TimeSeries",         "Sample autocovariance at lag k",                                 "Distribution helper.")
_r("autocor", "autocor",        "TimeSeries",         "Sample autocorrelation at lag k",                                "What is now proved was once only imagined. — William Blake")
_r("sampsrs", "sampsrs",        "Sampling",           "Simple random sample without replacement",                       "Distribution helper.")
_r("sampstr", "sampstr",        "Sampling",           "Stratified random sample (proportional allocation)",             "Many futures, one path. -- Doctor Strange")
_r("sampsys", "sampsys",        "Sampling",           "Systematic sample (every k-th unit)",                            "Routine becomes ritual. -- Geralt of Rivia")
_r("bsboot",  "bsboot",         "Resampling",         "Bootstrap resamples of a statistic",                              "Wake up, Mr. Freeman. -- The G-Man")
_r("bsperc",  "bsperc",         "Resampling",         "Percentile bootstrap CI from boot estimates",                     "Trust me on this one. -- Indiana Jones")
_r("jackone", "jackone",        "Resampling",         "Leave-one-out jackknife estimates",                               "One leaves, the rest endure. -- Aragorn")
_r("permpv",  "permpv",         "Resampling",         "Permutation p-value (two-sample mean diff)",                      "All possible worlds collapse to one truth. -- Doctor Strange")
_r("hubrl",   "hubrl",          "RobustLoss",         "Huber loss (quadratic / linear hybrid)",                          "Fine line between calm and rage. -- The Iron Bull")
_r("tukyl",   "tukyl",          "RobustLoss",         "Tukey biweight (bisquare) loss",                                  "I have the high ground. --")
_r("powtt2",  "powtt2",         "PowerAnalysis",      "Power for two-sample t-test (non-central t)",                     "Power overwhelming. -- Protoss")
_r("npowtt",  "npowtt",         "PowerAnalysis",      "Required n per group for two-sample t-test power",                "Plan to survive. -- Ellen Ripley")
_r("anddrl",  "anddrl",         "GoodnessOfFit",      "Anderson-Darling test for Normality",                             "I see the tails of distributions. -- The Oracle")
_r("tschpr",  "tschpr",         "Association",        "Tschuprow's T (categorical association)",                         "Live long and prosper, with categorical truth. -- Spock")
_r("eucldst", "eucldst",        "Distance",           "Euclidean (L₂) distance",                                          "The shortest path between two points. -- Galadriel")
_r("manhdst", "manhdst",        "Distance",           "Manhattan (L₁) distance",                                          "What is now proved was once only imagined. — William Blake")
_r("cossim",  "cossim",         "Distance",           "Cosine similarity",                                                "Aligned in vector-space. -- Quorra (TRON: Legacy)")
_r("jacsim",  "jacsim",         "Distance",           "Jaccard similarity for sets",                                      "What is now proved was once only imagined. — William Blake")
_r("brierl",  "brierl",         "Calibration",        "Brier score (calibration loss)",                                  "Calibration is honesty about uncertainty. -- Sherlock")
_r("aurroc",  "aurroc",         "Calibration",        "Area under ROC curve (binary)",                                   "Precision and recall, balanced. -- The Punisher")
_r("morani",  "morani",         "Spatial",            "Moran's I (spatial autocorrelation)",                             "Everything that lives is connected. -- Mufasa")
_r("gearyc",  "gearyc",         "Spatial",            "Geary's C (alternative spatial autocorrelation)",                 "The spice must flow. -- Paul Atreides")

# ── Atomic primitives round 3 (2026-05-06): 33 more — distribution
# helpers, econometrics tests, time-series diagnostics, calibration,
# survival, multivariate, more inference & lin alg.
_r("invlgt",  "invlgt",         "Logistic",           "Inverse logit / sigmoid 1/(1+e^-x)",                              "What is now proved was once only imagined. — William Blake")
_r("softmx",  "softmx",         "Logistic",           "Softmax exp(xᵢ)/Σⱼ exp(xⱼ)",                                       "Choose your champion. -- Mortal Kombat")
_r("erfunc",  "erfunc",         "Distribution",       "Gaussian error function erf(x)",                                  "There's been an error. -- HAL 9000")
_r("gammfn",  "gammfn",         "Distribution",       "Gamma function Γ(x)",                                              "By Grabthar's hammer. -- Galaxy Quest")
_r("betafn",  "betafn",         "Distribution",       "Beta function B(a,b)",                                              "What is now proved was once only imagined. — William Blake")
_r("digamf",  "digamf",         "Distribution",       "Digamma ψ(x)",                                                      "Distribution helper.")
_r("ibeta",   "ibeta",          "Distribution",       "Regularised incomplete Beta I_x(a,b)",                              "Partially complete. -- Cyborg")
_r("igamma",  "igamma",         "Distribution",       "Regularised lower incomplete gamma P(a,x)",                         "What is now proved was once only imagined. — William Blake")
_r("dwtest",  "dwtest",         "TimeSeries",         "Durbin-Watson statistic for autocorrelated residuals",              "Watch the autocorrelation. -- Mister Sandman")
_r("ljbox",   "ljbox",          "TimeSeries",         "Ljung-Box Q-statistic for white noise",                              "Box and box again. -- Doctor Who")
_r("bptest",  "bptest",         "Diagnostics",        "Breusch-Pagan test for heteroscedasticity",                          "Variance in disguise. -- Ozymandias")
_r("waldjt",  "waldjt",         "Inference",          "Wald joint test for linear restrictions Rβ=r",                       "Bind constraints. -- Doctor Strange")
_r("yindex",  "yindex",         "Calibration",        "Youden's J = TPR − FPR",                                              "Justice scales. -- Daredevil")
_r("mcc",     "mcc",            "Calibration",        "Matthews correlation coefficient (binary)",                          "Balance in all things. -- Avatar Aang")
_r("logloss", "logloss",        "Calibration",        "Cross-entropy / binary log loss",                                    "Distribution helper.")
_r("tventr",  "tventr",         "InfoTheory",         "Total variation distance ½ Σ |pᵢ−qᵢ|",                                "What is now proved was once only imagined. — William Blake")
_r("muthi",   "muthi",          "InfoTheory",         "Mutual information from joint p(x,y)",                              "Knowledge shared. -- Hermione Granger")
_r("kmsurv",  "kmsurv",         "Survival",           "Kaplan-Meier survival estimator",                                    "Time is a flat circle. -- True Detective")
_r("logrnk",  "logrnk",         "Survival",           "Log-rank test (two-group survival)",                                 "Two roads diverged. -- Robert Frost")
_r("glassd",  "glassd",         "EffectSize",         "Glass's Δ (control-SD-standardised mean diff)",                       "Standard set by the standard. -- Spock")
_r("etasq",   "etasq",          "EffectSize",         "Eta-squared SS_b/SS_t",                                              "Variance has a master. -- Sherlock")
_r("omeg2",   "omeg2",          "EffectSize",         "Omega-squared (less biased than η²)",                                "What is now proved was once only imagined. — William Blake")
_r("zscor",   "zscor",          "BasicMath",          "Z-score (x−x̄)/s standardisation",                                     "Become standard. Become extraordinary.")
_r("rngnrm",  "rngnrm",         "BasicMath",          "Min-max normalize to [0, 1]",                                        "Reframe the question. -- Doctor Strange")
_r("paired",  "paired",         "Inference",          "Paired t-test (matched x, y)",                                       "Pairs reveal what singles hide. -- Twin paradox")
_r("welcht",  "welcht",         "Inference",          "Welch's two-sample t-test (unequal variances)",                      "Unequal but balanced. -- Yin and Yang")
_r("fishex",  "fishex",         "ContingencyTables",  "Fisher's exact test (2×2)",                                          "Exact, not asymptotic. -- Sherlock")
_r("mcnem",   "mcnem",          "ContingencyTables",  "McNemar's test (paired binary)",                                     "Paired, not parallel. -- Doc Brown")
_r("trace",   "trace",          "LinearAlgebra",      "Matrix trace (Σ diagonal)",                                          "What lies on the diagonal. -- Hermione")
_r("invsym",  "invsym",         "LinearAlgebra",      "Pseudo-inverse of a symmetric matrix",                                "Reverse the polarity. -- The Doctor")
_r("eigval",  "eigval",         "LinearAlgebra",      "Eigenvalues (symmetric, ascending)",                                  "Inner spectra. -- The Watcher")
_r("mlenrm",  "mlenrm",         "Inference",          "Maximum-likelihood Normal fit",                                       "Maximum likelihood is maximum truth. -- Plato")
_r("mlepoi",  "mlepoi",         "Inference",          "Maximum-likelihood Poisson fit",                                      "The mean is the message. -- McLuhan")


def cheatsheet(category: str | None = None) -> None:
    """Print a Star Wars-themed cheatsheet of all moirais.fn functions.

    >>> from moirais.fn._registry import cheatsheet
    >>> cheatsheet()
    >>> cheatsheet("Test")  # filter by category
    """
    entries = REGISTRY.values()
    if category:
        entries = [e for e in entries if e.category.lower() == category.lower()]

    cats: dict[str, list[FnEntry]] = {}
    for e in entries:
        cats.setdefault(e.category, []).append(e)

    for cat, fns in sorted(cats.items()):
        print(f"\n{'=' * 60}")
        print(f"  {cat}")
        print(f"{'=' * 60}")
        for f in sorted(fns, key=lambda x: x.short):
            q = f"  ({f.quote})" if f.quote else ""
            print(f"  {f.short:<8s} {f.description}{q}")

    print(f"\nTotal: {len(REGISTRY)} functions")
