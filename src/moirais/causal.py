"""
Causal estimators: IPW, AIPW, G-computation, Matching and Double Machine Learning.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler


def _safe_exp(value):
    """Exponentiate on a clipped log scale to avoid overflow warnings in summaries.

    Accepts scalars, arrays, or Series. Returns float for scalar input,
    ndarray for array/Series input.
    """
    arr = np.asarray(value, dtype=float)
    result = np.exp(np.clip(arr, -700, 700))
    return float(result) if result.ndim == 0 else result


def compute_propensity_scores(data: pd.DataFrame, treatment: str, covariates: list) -> pd.Series:
    """
    Compute propensity scores using logistic regression with standard preprocessing.

    Non-numeric (object/categorical) columns are integer-encoded with
    :class:`~sklearn.preprocessing.LabelEncoder` and all numeric columns are
    standardised with :class:`~sklearn.preprocessing.StandardScaler` before
    fitting.  Standardisation stabilises the logistic-regression solver and
    makes coefficients comparable across covariates on different scales.
    Encoding is applied in-place on a copy; the original ``data`` frame is
    not modified.

    :param data: The input pandas DataFrame containing covariates and treatment.
    :type data: pandas.DataFrame
    :param treatment: The name of the column containing the binary treatment variable.
    :type treatment: str
    :param covariates: A list of column names for the covariates to adjust for.
    :type covariates: list[str]
    :return: A pandas Series containing the predicted propensity scores
        (values in the open interval (0, 1)).
    :rtype: pandas.Series

    References
    ----------
    Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the
    propensity score in observational studies for causal effects.
    *Biometrika*, 70(1), 41–55. https://doi.org/10.1093/biomet/70.1.41
    """
    X_raw = data[covariates].copy()
    y = data[treatment]

    # Encode any non-numeric columns so LogisticRegression receives a numeric
    # matrix.  LabelEncoder maps each unique string/category to an integer.
    for col in X_raw.columns:
        if not pd.api.types.is_numeric_dtype(X_raw[col]):
            le = LabelEncoder()
            X_raw[col] = le.fit_transform(X_raw[col].astype(str))

    # Standardise all numeric columns: mean 0, unit variance.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    return pd.Series(model.predict_proba(X_scaled)[:, 1], index=data.index, name="ps")


def calculate_ipw_weights(
    data: pd.DataFrame,
    treatment: str,
    ps_col: str,
    *,
    stabilized: bool = False,
    trim_quantiles: tuple[float, float] | None = None,
) -> pd.Series:
    """
    Calculate inverse probability of treatment weights (IPTW).

    :param data: The dataframe containing treatment assignment and propensity scores.
    :type data: pandas.DataFrame
    :param treatment: The column name for the actual treatment assignment.
    :type treatment: str
    :param ps_col: The column name containing the propensity scores.
    :type ps_col: str
    :return: A pandas Series containing the IPTW for each observation.
    :rtype: pandas.Series
    """
    ps = data[ps_col].clip(lower=0.01, upper=0.99)
    t = data[treatment]
    weights = (t / ps) + ((1 - t) / (1 - ps))
    if stabilized:
        p_treated = float(t.mean())
        weights = np.where(t == 1, p_treated / ps, (1 - p_treated) / (1 - ps))
        weights = pd.Series(weights, index=data.index, name="ipw")
    else:
        weights = pd.Series(weights, index=data.index, name="ipw")

    if trim_quantiles is not None:
        lower, upper = trim_quantiles
        q_lower = float(weights.quantile(lower))
        q_upper = float(weights.quantile(upper))
        weights = weights.clip(lower=q_lower, upper=q_upper)
    return weights


def effective_sample_size(weights: pd.Series | np.ndarray) -> float:
    """Compute effective sample size for analytical weights."""
    weights = np.asarray(weights, dtype=float)
    return float((weights.sum() ** 2) / np.square(weights).sum())


def run_propensity_ipw_analysis(
    data: pd.DataFrame,
    *,
    treatment: str = "cannabis_any_use",
    outcome: str = "heavy_drinking_30d",
    covariates: list[str] | None = None,
    survey_weight_col: str = "weight",
) -> dict[str, pd.DataFrame | float]:
    """
    Reproduce the core outputs of the old ``07_propensity.R`` workflow.

    ATE Estimator
    -------------
    The Average Treatment Effect is estimated via the **Hájek (normalised
    Horvitz-Thompson) IPW estimator**:

    .. math::

        \\widehat{\\text{ATE}}_{\\text{Hájek}} =
            \\frac{\\sum_i T_i Y_i / e_i}{\\sum_i T_i / e_i}
            -
            \\frac{\\sum_i (1-T_i) Y_i / (1-e_i)}{\\sum_i (1-T_i) / (1-e_i)}

    where :math:`T_i \\in \\{0,1\\}` is the treatment indicator,
    :math:`Y_i` is the outcome, and :math:`e_i = P(T_i=1 \\mid X_i)` is the
    estimated propensity score.

    The Hájek estimator is preferred over the raw Horvitz-Thompson estimator
    because it is self-normalised (the weights sum to 1 within each arm),
    which gives better finite-sample performance and is less sensitive to
    extreme propensity scores.

    Trimmed IPW weights (1st–99th percentile) are used to reduce the
    influence of extreme propensity scores.

    References
    ----------
    Lunceford, J. K., & Davidian, M. (2004). Stratification and weighting
    via the propensity score in estimation of causal treatment effects: a
    comparative study. *Statistics in Medicine*, 23(19), 2937–2960.
    https://doi.org/10.1002/sim.1903

    Hájek, J. (1971). Comment on "An essay on the logical foundations of
    survey sampling" by D. Basu. In V. P. Godambe & D. A. Sprott (Eds.),
    *Foundations of Statistical Inference* (pp. 236–236). Holt, Rinehart &
    Winston.
    """
    covariates = covariates or [
        "age_group",
        "gender",
        "province_region",
        "mental_health",
        "physical_health",
    ]
    required = [treatment, outcome, survey_weight_col, *covariates]
    frame = data.loc[:, required].dropna().copy()
    frame["ps"] = compute_propensity_scores(frame, treatment=treatment, covariates=covariates)
    frame["ipw"] = calculate_ipw_weights(frame, treatment, "ps")
    frame["ipw_trimmed"] = calculate_ipw_weights(
        frame,
        treatment,
        "ps",
        trim_quantiles=(0.01, 0.99),
    )

    # -----------------------------------------------------------------------
    # Hájek (normalised Horvitz-Thompson) ATE estimator.
    #
    # INCORRECT approach (prior code):
    #   Computed np.average(treated_outcomes, weights=treated_ipw) and
    #   np.average(control_outcomes, weights=control_ipw) separately within
    #   each arm.  This is NOT the Horvitz-Thompson estimator; it is an
    #   ad-hoc weighted mean within the observed arm only and does not
    #   correspond to any standard causal estimator.
    #
    # CORRECT approach (Hájek):
    #   Numerator for treated arm  = sum over ALL units of T_i * Y_i / e_i
    #   Denominator for treated arm = sum over ALL units of T_i / e_i
    #   Same pattern for control arm using (1 - T_i) / (1 - e_i).
    #   Dividing normalises the weights so they sum to 1 within each arm.
    # -----------------------------------------------------------------------
    t = frame[treatment].values
    y = frame[outcome].values

    # Use the raw (unclipped) propensity score column directly; apply the same
    # 0.01–0.99 clipping that calculate_ipw_weights uses so that the Hájek
    # numerators and denominators are consistent with the trimmed IPW weights.
    ps_raw = frame["ps"].values.clip(0.01, 0.99)

    # Treated arm: sum(T * Y / e) / sum(T / e)
    treated_mask = t == 1
    control_mask = t == 0
    sum_ty_over_e = np.sum(y[treated_mask] / ps_raw[treated_mask])
    sum_t_over_e = np.sum(1.0 / ps_raw[treated_mask])
    y1_hajek = float(sum_ty_over_e / sum_t_over_e) if sum_t_over_e > 0 else np.nan

    # Control arm: sum((1-T) * Y / (1-e)) / sum((1-T) / (1-e))
    sum_cy_over_1me = np.sum(y[control_mask] / (1.0 - ps_raw[control_mask]))
    sum_c_over_1me = np.sum(1.0 / (1.0 - ps_raw[control_mask]))
    y0_hajek = float(sum_cy_over_1me / sum_c_over_1me) if sum_c_over_1me > 0 else np.nan

    ate_ipw = float(y1_hajek - y0_hajek)

    ipw_results = pd.DataFrame(
        [
            {
                "estimand": "ATE",
                "method": "IPW-Hajek (trimmed)",
                "estimate": ate_ipw,
                "n": len(frame),
                "y1_hajek": y1_hajek,
                "y0_hajek": y0_hajek,
            }
        ]
    )
    diagnostics = pd.DataFrame(
        [
            {"metric": "ps_mean", "value": float(frame["ps"].mean())},
            {"metric": "ps_min", "value": float(frame["ps"].min())},
            {"metric": "ps_max", "value": float(frame["ps"].max())},
            {"metric": "ipw_mean", "value": float(frame["ipw"].mean())},
            {"metric": "ipw_trimmed_mean", "value": float(frame["ipw_trimmed"].mean())},
            {"metric": "ess_ipw_trimmed", "value": effective_sample_size(frame["ipw_trimmed"])},
        ]
    )
    return {
        "analysis_frame": frame,
        "ipw_results": ipw_results,
        "diagnostics": diagnostics,
    }


def estimate_aipw(
    data: pd.DataFrame,
    *,
    treatment: str = "cannabis_any_use",
    outcome: str = "heavy_drinking_30d",
    covariates: list[str] | None = None,
    outcome_model: str = "logistic",
) -> dict[str, Any]:
    """
    Estimate the ATE via the Augmented Inverse Probability Weighting (AIPW)
    doubly-robust estimator.

    The influence-function score for unit *i* is:

    .. math::

        \\psi_i =
            \\hat{\\mu}_1(X_i) - \\hat{\\mu}_0(X_i)
            + \\frac{T_i \\bigl(Y_i - \\hat{\\mu}_1(X_i)\\bigr)}{\\hat{e}_i}
            - \\frac{(1-T_i)\\bigl(Y_i - \\hat{\\mu}_0(X_i)\\bigr)}{1 - \\hat{e}_i}

    The ATE estimator is :math:`\\widehat{\\text{ATE}}_\\text{AIPW} = n^{-1} \\sum_i \\psi_i`
    and the standard error is :math:`\\hat{\\sigma}_{\\psi} / \\sqrt{n}`.

    **Double robustness**: the estimator is consistent if *either* the
    propensity score model :math:`\\hat{e}(X)` or the outcome model
    :math:`\\hat{\\mu}(T, X)` is correctly specified — not necessarily both.

    :param data: The input DataFrame.
    :type data: pandas.DataFrame
    :param treatment: Binary treatment column (0/1).
    :type treatment: str
    :param outcome: Outcome column.
    :type outcome: str
    :param covariates: Covariate column names.  Defaults to the standard
        CPADS confounders.
    :type covariates: list[str] | None
    :param outcome_model: ``"logistic"`` for binary outcomes, ``"linear"``
        for continuous.  Defaults to ``"logistic"``.
    :type outcome_model: str
    :return: Dictionary with keys ``ate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``n``, ``method``.
    :rtype: dict[str, Any]

    References
    ----------
    Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation of
    regression coefficients when some regressors are not always observed.
    *Journal of the American Statistical Association*, 89(427), 846–866.

    Scharfstein, D. O., Rotnitzky, A., & Robins, J. M. (1999). Adjusting for
    nonignorable drop-out using semiparametric nonresponse models. *JASA*,
    94(448), 1096–1120.
    """
    covariates = covariates or [
        "age_group",
        "gender",
        "province_region",
        "mental_health",
        "physical_health",
    ]
    required = [treatment, outcome, *covariates]
    frame = data.loc[:, required].dropna().copy()

    t = frame[treatment].values.astype(float)
    y = frame[outcome].values.astype(float)

    # ── Propensity scores ────────────────────────────────────────────────────
    ps = compute_propensity_scores(frame, treatment=treatment, covariates=covariates).values
    ps = ps.clip(0.01, 0.99)

    # ── Outcome model: preprocess covariates the same way as propensity ─────
    X_raw = frame[covariates].copy()
    for col in X_raw.columns:
        if not pd.api.types.is_numeric_dtype(X_raw[col]):
            le = LabelEncoder()
            X_raw[col] = le.fit_transform(X_raw[col].astype(str))
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    treated_mask = t == 1
    control_mask = t == 0

    if outcome_model == "logistic":
        om1 = LogisticRegression(max_iter=1000)
        om0 = LogisticRegression(max_iter=1000)
        om1.fit(X[treated_mask], y[treated_mask])
        om0.fit(X[control_mask], y[control_mask])
        mu1 = om1.predict_proba(X)[:, 1]
        mu0 = om0.predict_proba(X)[:, 1]
    else:
        om1 = LinearRegression()
        om0 = LinearRegression()
        om1.fit(X[treated_mask], y[treated_mask])
        om0.fit(X[control_mask], y[control_mask])
        mu1 = om1.predict(X)
        mu0 = om0.predict(X)

    # ── AIPW influence scores ─────────────────────────────────────────────────
    psi = mu1 - mu0 + t * (y - mu1) / ps - (1.0 - t) * (y - mu0) / (1.0 - ps)

    n = len(psi)
    ate = float(psi.mean())
    se = float(psi.std(ddof=1) / np.sqrt(n))
    z = 1.959964  # 97.5th percentile of standard normal

    return {
        "ate": ate,
        "se": se,
        "ci_lower": ate - z * se,
        "ci_upper": ate + z * se,
        "n": n,
        "method": "AIPW (doubly robust)",
    }


def run_ebac_selection_ipw_analysis(
    data: pd.DataFrame,
    *,
    observation_col: str = "ebac_tot",
    eligible_col: str = "alcohol_past12m",
    treatment: str = "cannabis_any_use",
    binary_outcome: str = "ebac_legal",
    continuous_outcome: str = "ebac_tot",
    survey_weight_col: str = "weight",
    covariates: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Reproduce the core outputs of the old ``07_ebac_ipw.R`` workflow.
    """
    covariates = covariates or [
        "age_group",
        "gender",
        "province_region",
        "mental_health",
        "physical_health",
    ]
    required = [
        survey_weight_col,
        eligible_col,
        observation_col,
        binary_outcome,
        treatment,
        *covariates,
    ]
    target = data.loc[:, required].copy()
    target = target[target[eligible_col] == 1].dropna(subset=[survey_weight_col, treatment, *covariates])
    target["R"] = (~target[observation_col].isna()).astype(int)

    obs_formula = "R ~ " + " + ".join([treatment, *covariates])
    obs_model = smf.glm(
        formula=obs_formula,
        data=target,
        family=sm.families.Binomial(),
    ).fit()
    target["p_hat"] = obs_model.predict(target).clip(lower=0.01, upper=0.99)
    p_obs = float(target["R"].mean())
    observed = target[target["R"] == 1].copy()
    observed["sw"] = p_obs / observed["p_hat"]
    q01 = float(observed["sw"].quantile(0.01))
    q99 = float(observed["sw"].quantile(0.99))
    observed["sw_trim"] = observed["sw"].clip(lower=q01, upper=q99)
    observed["w_combined_trim"] = observed[survey_weight_col] * observed["sw_trim"]

    diag_tbl = pd.DataFrame(
        [
            {"metric": "eligible_n", "value": len(target)},
            {"metric": "observed_n", "value": len(observed)},
            {"metric": "observed_rate", "value": p_obs},
            {"metric": "sw_min", "value": float(observed["sw"].min())},
            {"metric": "sw_q01", "value": q01},
            {"metric": "sw_q99", "value": q99},
            {"metric": "sw_max", "value": float(observed["sw"].max())},
            {"metric": "ess_survey_x_ipw_trim", "value": effective_sample_size(observed["w_combined_trim"])},
        ]
    )

    logit_formula = f"{binary_outcome} ~ {treatment} + " + " + ".join(covariates)
    linear_formula = f"{continuous_outcome} ~ {treatment} + " + " + ".join(covariates)
    # var_weights: analytic/probability weights.  Using freq_weights here
    # would incorrectly inflate n_effective by treating each weight as a
    # replication count, producing anti-conservative standard errors.
    fit_bin = smf.glm(
        formula=logit_formula,
        data=observed,
        family=sm.families.Binomial(),
        var_weights=observed["w_combined_trim"],
    ).fit()
    fit_lin = smf.wls(
        formula=linear_formula,
        data=observed,
        weights=observed["w_combined_trim"],
    ).fit()

    logit_row = pd.DataFrame(
        [
            {
                "model": "selection_adjusted_ipw",
                "term": treatment,
                "log_odds": float(fit_bin.params[treatment]),
                "se": float(fit_bin.bse[treatment]),
                "or": _safe_exp(fit_bin.params[treatment]),
                "or_lower95": _safe_exp(fit_bin.conf_int().loc[treatment, 0]),
                "or_upper95": _safe_exp(fit_bin.conf_int().loc[treatment, 1]),
                "p_value": float(fit_bin.pvalues[treatment]),
                "significant": "*" if float(fit_bin.pvalues[treatment]) < 0.05 else "",
            }
        ]
    )
    linear_row = pd.DataFrame(
        [
            {
                "model": "selection_adjusted_ipw",
                "term": treatment,
                "estimate": float(fit_lin.params[treatment]),
                "se": float(fit_lin.bse[treatment]),
                "ci_lower95": float(fit_lin.conf_int().loc[treatment, 0]),
                "ci_upper95": float(fit_lin.conf_int().loc[treatment, 1]),
                "p_value": float(fit_lin.pvalues[treatment]),
                "significant": "*" if float(fit_lin.pvalues[treatment]) < 0.05 else "",
            }
        ]
    )
    cannabis_comparison = pd.DataFrame(
        [
            {
                "metric": "ebac_legal_or_cannabis",
                "model": "selection_adjusted_ipw",
                "estimate": logit_row.loc[0, "or"],
                "ci_lower95": logit_row.loc[0, "or_lower95"],
                "ci_upper95": logit_row.loc[0, "or_upper95"],
                "p_value": logit_row.loc[0, "p_value"],
            },
            {
                "metric": "ebac_tot_beta_cannabis",
                "model": "selection_adjusted_ipw",
                "estimate": linear_row.loc[0, "estimate"],
                "ci_lower95": linear_row.loc[0, "ci_lower95"],
                "ci_upper95": linear_row.loc[0, "ci_upper95"],
                "p_value": linear_row.loc[0, "p_value"],
            },
        ]
    )
    # Additional IPW diagnostic outputs (match MODULE_SPECS expectations)

    # Weight diagnostics — detailed per-observation weight summary
    weight_diag = pd.DataFrame(
        {
            "metric": [
                "sw_mean",
                "sw_median",
                "sw_sd",
                "sw_trim_mean",
                "sw_trim_sd",
                "w_combined_mean",
                "w_combined_sd",
                "w_combined_min",
                "w_combined_max",
            ],
            "value": [
                float(observed["sw"].mean()),
                float(observed["sw"].median()),
                float(observed["sw"].std()),
                float(observed["sw_trim"].mean()),
                float(observed["sw_trim"].std()),
                float(observed["w_combined_trim"].mean()),
                float(observed["w_combined_trim"].std()),
                float(observed["w_combined_trim"].min()),
                float(observed["w_combined_trim"].max()),
            ],
        }
    )

    # Logistic OR — full coefficient table from the IPW logistic model
    ipw_logistic_conf = fit_bin.conf_int()
    ipw_logistic_or = pd.DataFrame(
        {
            "term": fit_bin.params.index,
            "log_odds": fit_bin.params.values,
            "se": fit_bin.bse.values,
            "or": _safe_exp(fit_bin.params.values),
            "or_lower95": _safe_exp(ipw_logistic_conf[0].values),
            "or_upper95": _safe_exp(ipw_logistic_conf[1].values),
            "p_value": fit_bin.pvalues.values,
            "significant": ["*" if p < 0.05 else "" for p in fit_bin.pvalues.values],
        }
    )

    # Linear coefficients — full table
    ipw_linear_conf = fit_lin.conf_int()
    ipw_linear_coefs = pd.DataFrame(
        {
            "term": fit_lin.params.index,
            "estimate": fit_lin.params.values,
            "se": fit_lin.bse.values,
            "ci_lower95": ipw_linear_conf[0].values,
            "ci_upper95": ipw_linear_conf[1].values,
            "p_value": fit_lin.pvalues.values,
            "significant": ["*" if p < 0.05 else "" for p in fit_lin.pvalues.values],
        }
    )

    # Cannabis comparison — extended with unweighted estimates
    unadj_bin = smf.glm(
        formula=logit_formula,
        data=observed,
        family=sm.families.Binomial(),
    ).fit()
    ipw_cannabis_comparison = pd.concat(
        [
            cannabis_comparison,
            pd.DataFrame(
                [
                    {
                        "metric": "ebac_legal_or_cannabis",
                        "model": "unadjusted",
                        "estimate": _safe_exp(unadj_bin.params[treatment]),
                        "ci_lower95": _safe_exp(unadj_bin.conf_int().loc[treatment, 0]),
                        "ci_upper95": _safe_exp(unadj_bin.conf_int().loc[treatment, 1]),
                        "p_value": float(unadj_bin.pvalues[treatment]),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    # Observation model OR — from the selection model
    obs_conf = obs_model.conf_int()
    obs_model_or = pd.DataFrame(
        {
            "term": obs_model.params.index,
            "log_odds": obs_model.params.values,
            "se": obs_model.bse.values,
            "or": _safe_exp(obs_model.params.values),
            "or_lower95": _safe_exp(obs_conf[0].values),
            "or_upper95": _safe_exp(obs_conf[1].values),
            "p_value": obs_model.pvalues.values,
        }
    )

    # Covariate balance — standardised mean differences before/after IPW
    balance_rows = []
    for cov in covariates:
        if cov not in observed.columns:
            continue
        try:
            vals = pd.to_numeric(observed[cov], errors="coerce")
            t_mask = observed[treatment] == 1
            c_mask = observed[treatment] == 0
            if t_mask.sum() == 0 or c_mask.sum() == 0:
                continue
            raw_smd = (
                float(
                    (vals[t_mask].mean() - vals[c_mask].mean()) / np.sqrt((vals[t_mask].var() + vals[c_mask].var()) / 2)
                )
                if (vals[t_mask].var() + vals[c_mask].var()) > 0
                else 0.0
            )
            w = observed["w_combined_trim"]
            wt_mean = float(np.average(vals[t_mask], weights=w[t_mask]))
            wc_mean = float(np.average(vals[c_mask], weights=w[c_mask]))
            pooled_sd = np.sqrt((vals[t_mask].var() + vals[c_mask].var()) / 2)
            ipw_smd = float((wt_mean - wc_mean) / pooled_sd) if pooled_sd > 0 else 0.0
            balance_rows.append(
                {
                    "covariate": cov,
                    "smd_raw": round(raw_smd, 4),
                    "smd_ipw": round(ipw_smd, 4),
                    "mean_treated_raw": round(float(vals[t_mask].mean()), 4),
                    "mean_control_raw": round(float(vals[c_mask].mean()), 4),
                    "mean_treated_ipw": round(wt_mean, 4),
                    "mean_control_ipw": round(wc_mean, 4),
                }
            )
        except Exception:
            continue

    return {
        "analysis_frame": observed,
        "ebac_ipw_weight_diagnostics": weight_diag,
        "ebac_ipw_logistic_or": ipw_logistic_or,
        "ebac_ipw_linear_coefficients": ipw_linear_coefs,
        "ebac_ipw_cannabis_comparison": ipw_cannabis_comparison,
        "ebac_ipw_observation_model_or": obs_model_or,
        "ebac_ipw_covariate_balance": pd.DataFrame(balance_rows),
        "ebac_final_ipw_diagnostics": diag_tbl,
        "ebac_final_ipw_or": logit_row,
        "ebac_final_ipw_linear": linear_row,
        "ebac_final_ipw_comparison": cannabis_comparison,
    }


# ---------------------------------------------------------------------------
# ATT — Average Treatment Effect on the Treated
# ---------------------------------------------------------------------------


def estimate_att(
    data: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariates: list[str],
    propensity_col: str | None = None,
) -> dict[str, Any]:
    """Estimate the Average Treatment Effect on the Treated (ATT) via
    Hajek-weighted IPW.

    The ATT is defined as:

    .. math::

        \\text{ATT} = \\mathbb{E}[Y(1) - Y(0) \\mid T=1]

    Under unconfoundedness, the ATT is identified by the Hajek IPW estimator
    where treated units receive weight 1 and control units receive weight
    :math:`\\hat{e}(X) / (1 - \\hat{e}(X))`:

    .. math::

        \\widehat{\\text{ATT}} =
        \\frac{1}{n_1} \\sum_{i: T_i=1} Y_i
        - \\frac{\\sum_{i: T_i=0} Y_i \\cdot \\hat{e}_i / (1 - \\hat{e}_i)}
              {\\sum_{i: T_i=0} \\hat{e}_i / (1 - \\hat{e}_i)}

    :param data: Input DataFrame.
    :type data: pandas.DataFrame
    :param treatment: Binary treatment column (0/1).
    :type treatment: str
    :param outcome: Outcome column.
    :type outcome: str
    :param covariates: Covariate column names for propensity model.
    :type covariates: list[str]
    :param propensity_col: Pre-computed propensity score column.  If None,
        propensity scores are estimated from *covariates*.
    :type propensity_col: str or None
    :return: Dictionary with ``att``, ``se``, ``ci_lower``, ``ci_upper``, ``n``.
    :rtype: dict[str, Any]

    References
    ----------
    Imbens, G. W. (2004). Nonparametric estimation of average treatment
    effects under exogeneity: a review. *Review of Economics and Statistics*,
    86(1), 4--29. https://doi.org/10.1162/003465304323023651

    Hirano, K., Imbens, G. W., & Ridder, G. (2003). Efficient estimation
    of average treatment effects using the estimated propensity score.
    *Econometrica*, 71(4), 1161--1189.
    """
    frame = data[[treatment, outcome, *covariates]].dropna().copy()
    t = frame[treatment].values.astype(float)
    y = frame[outcome].values.astype(float)

    if propensity_col is not None and propensity_col in data.columns:
        ps = data.loc[frame.index, propensity_col].values.astype(float)
    else:
        ps = compute_propensity_scores(frame, treatment=treatment, covariates=covariates).values
    ps = ps.clip(0.01, 0.99)

    treated = t == 1
    control = t == 0
    n1 = float(treated.sum())
    n0 = float(control.sum())
    n = len(frame)

    # Treated mean: simple average of Y among treated
    y1_bar = float(y[treated].mean())

    # Control counterfactual weighted by e/(1-e)
    w_control = ps[control] / (1.0 - ps[control])
    y0_bar = float(np.average(y[control], weights=w_control))

    att = y1_bar - y0_bar

    # Standard error: conservative pooled estimate
    se_treated = float(np.sqrt(np.var(y[treated], ddof=1) / n1))
    if n0 > 1:
        # Weighted variance for control arm
        w_normed = w_control / w_control.sum()
        wvar_ctrl = float(np.sum(w_normed * (y[control] - y0_bar) ** 2))
        ess_ctrl = effective_sample_size(w_control)
        se_control = float(np.sqrt(wvar_ctrl / max(ess_ctrl - 1, 1)))
    else:
        se_control = 0.0
    se = float(np.sqrt(se_treated**2 + se_control**2))

    if not np.isfinite(se) or se <= 0:
        se = float(np.sqrt(np.var(y, ddof=1) / n))

    z = 1.959964
    return {
        "att": att,
        "se": se,
        "ci_lower": att - z * se,
        "ci_upper": att + z * se,
        "n": n,
        "n_treated": int(n1),
        "n_control": int(n0),
        "method": "ATT (Hajek IPW)",
    }


# ---------------------------------------------------------------------------
# ATC — Average Treatment Effect on the Controls
# ---------------------------------------------------------------------------


def estimate_atc(
    data: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariates: list[str],
    propensity_col: str | None = None,
) -> dict[str, Any]:
    """Estimate the Average Treatment Effect on the Controls (ATC).

    The ATC is defined as:

    .. math::

        \\text{ATC} = \\mathbb{E}[Y(1) - Y(0) \\mid T=0]

    Treated units receive weight :math:`(1 - \\hat{e}(X)) / \\hat{e}(X)` and
    control units receive weight 1.

    :param data: Input DataFrame.
    :type data: pandas.DataFrame
    :param treatment: Binary treatment column (0/1).
    :type treatment: str
    :param outcome: Outcome column.
    :type outcome: str
    :param covariates: Covariate column names for propensity model.
    :type covariates: list[str]
    :param propensity_col: Pre-computed propensity score column (optional).
    :type propensity_col: str or None
    :return: Dictionary with ``atc``, ``se``, ``ci_lower``, ``ci_upper``, ``n``.
    :rtype: dict[str, Any]

    References
    ----------
    Imbens, G. W. (2004). Nonparametric estimation of average treatment
    effects under exogeneity: a review. *Review of Economics and Statistics*,
    86(1), 4--29.
    """
    frame = data[[treatment, outcome, *covariates]].dropna().copy()
    t = frame[treatment].values.astype(float)
    y = frame[outcome].values.astype(float)

    if propensity_col is not None and propensity_col in data.columns:
        ps = data.loc[frame.index, propensity_col].values.astype(float)
    else:
        ps = compute_propensity_scores(frame, treatment=treatment, covariates=covariates).values
    ps = ps.clip(0.01, 0.99)

    treated = t == 1
    control = t == 0
    n0 = float(control.sum())
    n1 = float(treated.sum())

    # Control mean: simple average of Y among controls
    y0_bar = float(y[control].mean())

    # Treated counterfactual weighted by (1-e)/e
    w_treated = (1.0 - ps[treated]) / ps[treated]
    y1_bar = float(np.average(y[treated], weights=w_treated))

    atc = y1_bar - y0_bar

    # Standard error
    se_control = float(np.sqrt(np.var(y[control], ddof=1) / n0))
    if n1 > 1:
        w_normed = w_treated / w_treated.sum()
        wvar_treat = float(np.sum(w_normed * (y[treated] - y1_bar) ** 2))
        ess_treat = effective_sample_size(w_treated)
        se_treated = float(np.sqrt(wvar_treat / max(ess_treat - 1, 1)))
    else:
        se_treated = 0.0
    se = float(np.sqrt(se_treated**2 + se_control**2))

    if not np.isfinite(se) or se <= 0:
        se = float(np.sqrt(np.var(y, ddof=1) / len(frame)))

    z = 1.959964
    return {
        "atc": atc,
        "se": se,
        "ci_lower": atc - z * se,
        "ci_upper": atc + z * se,
        "n": len(frame),
        "n_treated": int(n1),
        "n_control": int(n0),
        "method": "ATC (Hajek IPW)",
    }


# ---------------------------------------------------------------------------
# GATE — Group Average Treatment Effect
# ---------------------------------------------------------------------------


def estimate_gate(
    data: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariates: list[str],
    group_col: str,
    propensity_col: str | None = None,
) -> pd.DataFrame:
    """Estimate Group Average Treatment Effects (GATE) via AIPW within strata.

    Partitions the data by *group_col* and estimates the ATE within each
    group using the AIPW doubly-robust estimator.

    .. math::

        \\text{GATE}_g = \\mathbb{E}[Y(1) - Y(0) \\mid G = g]

    :param data: Input DataFrame.
    :type data: pandas.DataFrame
    :param treatment: Binary treatment column (0/1).
    :type treatment: str
    :param outcome: Outcome column.
    :type outcome: str
    :param covariates: Covariate column names.
    :type covariates: list[str]
    :param group_col: Column defining groups/strata.
    :type group_col: str
    :param propensity_col: Pre-computed propensity score column (optional).
    :type propensity_col: str or None
    :return: DataFrame with columns: ``group``, ``ate``, ``se``,
        ``ci_lower``, ``ci_upper``, ``n``.
    :rtype: pandas.DataFrame

    References
    ----------
    Imai, K., & Ratkovic, M. (2013). Estimating treatment effect
    heterogeneity in randomized program evaluation.
    *Annals of Applied Statistics*, 7(1), 443--470.

    Chernozhukov, V., Demirer, M., Duflo, E., & Fernandez-Val, I. (2020).
    Generic machine learning inference on heterogeneous treatment effects in
    randomized experiments. *NBER Working Paper* 24678.
    """
    import logging as _logging

    _logger = _logging.getLogger(__name__)

    required_cols = [treatment, outcome, group_col, *covariates]
    frame = data[required_cols].dropna().copy()

    results: list[dict] = []
    for group_val, group_df in frame.groupby(group_col):
        if group_df[treatment].nunique() < 2:
            _logger.warning("GATE: skipping group '%s' -- no variation in treatment", group_val)
            continue

        try:
            aipw_result = estimate_aipw(
                group_df,
                treatment=treatment,
                outcome=outcome,
                covariates=covariates,
                outcome_model="linear",
            )
            results.append(
                {
                    "group": group_val,
                    "ate": aipw_result["ate"],
                    "se": aipw_result["se"],
                    "ci_lower": aipw_result["ci_lower"],
                    "ci_upper": aipw_result["ci_upper"],
                    "n": aipw_result["n"],
                }
            )
        except Exception as exc:
            _logger.warning("GATE: failed for group '%s': %s", group_val, exc)
            results.append(
                {
                    "group": group_val,
                    "ate": float("nan"),
                    "se": float("nan"),
                    "ci_lower": float("nan"),
                    "ci_upper": float("nan"),
                    "n": len(group_df),
                }
            )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# CATE — Conditional Average Treatment Effect (T-learner / S-learner)
# ---------------------------------------------------------------------------


def estimate_cate(
    data: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariates: list[str],
    meta_learner: str = "t_learner",
) -> pd.Series:
    """Estimate per-unit Conditional Average Treatment Effects (CATE).

    The CATE for unit *i* is :math:`\\tau(X_i) = \\mathbb{E}[Y(1) - Y(0) \\mid X = X_i]`.

    **T-learner** (default): fits separate outcome models on treated and
    control units, then computes :math:`\\hat{\\tau}(X_i) = \\hat{\\mu}_1(X_i)
    - \\hat{\\mu}_0(X_i)`.

    **S-learner**: fits a single outcome model on all units with treatment
    as a feature, then computes
    :math:`\\hat{\\tau}(X_i) = \\hat{\\mu}(X_i, T=1) - \\hat{\\mu}(X_i, T=0)`.

    :param data: Input DataFrame.
    :type data: pandas.DataFrame
    :param treatment: Binary treatment column (0/1).
    :type treatment: str
    :param outcome: Outcome column.
    :type outcome: str
    :param covariates: Covariate column names.
    :type covariates: list[str]
    :param meta_learner: ``"t_learner"`` or ``"s_learner"`` (default ``"t_learner"``).
    :type meta_learner: str
    :return: Series of per-unit CATE estimates, indexed like the input.
    :rtype: pandas.Series

    References
    ----------
    Kunzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019).
    Metalearners for estimating heterogeneous treatment effects using
    machine learning. *PNAS*, 116(10), 4156--4165.
    https://doi.org/10.1073/pnas.1804597116
    """
    frame = data[[treatment, outcome, *covariates]].dropna().copy()
    t = frame[treatment].values.astype(float)
    y = frame[outcome].values.astype(float)

    # Preprocess covariates
    X_raw = frame[covariates].copy()
    for col in X_raw.columns:
        if not pd.api.types.is_numeric_dtype(X_raw[col]):
            le = LabelEncoder()
            X_raw[col] = le.fit_transform(X_raw[col].astype(str))
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    treated_mask = t == 1
    control_mask = t == 0

    if meta_learner == "t_learner":
        # T-learner: separate models per arm
        rf1 = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf0 = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf1.fit(X[treated_mask], y[treated_mask])
        rf0.fit(X[control_mask], y[control_mask])
        mu1 = rf1.predict(X)
        mu0 = rf0.predict(X)

    elif meta_learner == "s_learner":
        # S-learner: single model with treatment as feature
        X_with_t = np.column_stack([X, t])
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_with_t, y)
        X_treat = np.column_stack([X, np.ones(len(X))])
        X_ctrl = np.column_stack([X, np.zeros(len(X))])
        mu1 = rf.predict(X_treat)
        mu0 = rf.predict(X_ctrl)

    else:
        raise ValueError(f"Unknown meta_learner: {meta_learner!r}. Choose 't_learner' or 's_learner'.")

    cate = mu1 - mu0
    return pd.Series(cate, index=frame.index, name="cate")


# ---------------------------------------------------------------------------
# LATE — Local Average Treatment Effect (IV / 2SLS)
# ---------------------------------------------------------------------------


def estimate_late(
    data: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    instrument: str,
    covariates: list[str] | None = None,
) -> dict[str, Any]:
    """Estimate the Local Average Treatment Effect (LATE) via instrumental variables.

    For a binary instrument :math:`Z`, the **Wald estimator** (simple IV) is:

    .. math::

        \\widehat{\\text{LATE}} = \\frac{\\text{Cov}(Y, Z)}{\\text{Cov}(T, Z)}
        = \\frac{\\bar{Y}_{Z=1} - \\bar{Y}_{Z=0}}{\\bar{T}_{Z=1} - \\bar{T}_{Z=0}}

    With covariates, the function attempts 2SLS via ``linearmodels.iv.IV2SLS``
    or ``statsmodels`` IV regression, falling back to the Wald estimator if
    neither IV library is installed.

    :param data: Input DataFrame.
    :type data: pandas.DataFrame
    :param treatment: Endogenous treatment column.
    :type treatment: str
    :param outcome: Outcome column.
    :type outcome: str
    :param instrument: Instrument column.
    :type instrument: str
    :param covariates: Exogenous covariate column names (optional).
    :type covariates: list[str] or None
    :return: Dictionary with ``late``, ``se``, ``ci`` (tuple), ``f_stat``,
        ``method``.
    :rtype: dict[str, Any]

    References
    ----------
    Imbens, G. W., & Angrist, J. D. (1994). Identification and estimation
    of local average treatment effects. *Econometrica*, 62(2), 467--475.
    https://doi.org/10.2307/2951620

    Angrist, J. D., Imbens, G. W., & Rubin, D. B. (1996). Identification of
    causal effects using instrumental variables. *JASA*, 91(434), 444--455.
    """
    cols = [treatment, outcome, instrument]
    if covariates:
        cols.extend(covariates)
    frame = data[cols].dropna().copy()

    y = frame[outcome].values.astype(float)
    t = frame[treatment].values.astype(float)
    z = frame[instrument].values.astype(float)
    n = len(frame)

    # Try linearmodels IV2SLS first
    try:
        from linearmodels.iv import IV2SLS as LM_IV2SLS

        if covariates:
            exog = sm.add_constant(frame[covariates].values.astype(float))
        else:
            exog = np.ones((n, 1))

        result = LM_IV2SLS(
            dependent=y,
            exog=exog,
            endog=t.reshape(-1, 1),
            instruments=z.reshape(-1, 1),
        ).fit(cov_type="robust")

        late_val = float(result.params.iloc[-1] if hasattr(result.params, "iloc") else result.params[-1])
        se_val = float(result.std_errors.iloc[-1] if hasattr(result.std_errors, "iloc") else result.std_errors[-1])
        try:
            f_stat = float(result.first_stage.diagnostics.iloc[0]["f.stat"])
        except Exception:
            f_stat = float("nan")

        z_crit = 1.959964
        return {
            "late": late_val,
            "se": se_val,
            "ci": (late_val - z_crit * se_val, late_val + z_crit * se_val),
            "f_stat": f_stat,
            "n": n,
            "method": "2SLS (linearmodels)",
        }
    except ImportError:
        pass
    except Exception:
        pass

    # Try statsmodels IV2SLS
    try:
        from statsmodels.sandbox.regression.gmm import IV2SLS as SM_IV2SLS

        if covariates:
            exog = np.column_stack(
                [
                    np.ones(n),
                    frame[covariates].values.astype(float),
                    t,
                ]
            )
            instrument_matrix = np.column_stack(
                [
                    np.ones(n),
                    frame[covariates].values.astype(float),
                    z,
                ]
            )
        else:
            exog = sm.add_constant(t)
            instrument_matrix = sm.add_constant(z)

        result = SM_IV2SLS(y, exog, instrument_matrix).fit()
        late_val = float(result.params[-1])
        se_val = float(result.bse[-1])

        # First-stage F-stat
        if covariates:
            X_first = np.column_stack([np.ones(n), frame[covariates].values.astype(float), z])
        else:
            X_first = sm.add_constant(z)
        first_stage = sm.OLS(t, X_first).fit()
        f_stat = float(first_stage.fvalue)

        z_crit = 1.959964
        return {
            "late": late_val,
            "se": se_val,
            "ci": (late_val - z_crit * se_val, late_val + z_crit * se_val),
            "f_stat": f_stat,
            "n": n,
            "method": "2SLS (statsmodels)",
        }
    except ImportError:
        pass
    except Exception:
        pass

    # Wald estimator fallback (no covariate adjustment)
    cov_yz = np.cov(y, z)[0, 1]
    cov_tz = np.cov(t, z)[0, 1]
    if abs(cov_tz) < 1e-12:
        raise ValueError(
            "Instrument has near-zero covariance with treatment; LATE is not identified (weak instrument)."
        )
    late_val = float(cov_yz / cov_tz)

    # Delta-method SE for Wald estimator with binary instrument
    z_unique = np.unique(z)
    if len(z_unique) == 2:
        z0, z1 = sorted(z_unique)
        y_z1 = y[z == z1]
        y_z0 = y[z == z0]
        t_z1 = t[z == z1]
        t_z0 = t[z == z0]
        n1_z = len(y_z1)
        n0_z = len(y_z0)
        denom = float(t_z1.mean() - t_z0.mean())
        if abs(denom) < 1e-12:
            se_val = float("nan")
        else:
            # Variance of the ratio via delta method
            var_num = np.var(y_z1, ddof=1) / n1_z + np.var(y_z0, ddof=1) / n0_z
            se_val = float(np.sqrt(var_num) / abs(denom))
    else:
        se_val = float("nan")

    # First stage F-stat
    X_first = sm.add_constant(z)
    first_stage = sm.OLS(t, X_first).fit()
    f_stat = float(first_stage.fvalue)

    z_crit = 1.959964
    ci = (
        (late_val - z_crit * se_val, late_val + z_crit * se_val)
        if np.isfinite(se_val)
        else (float("nan"), float("nan"))
    )

    return {
        "late": late_val,
        "se": se_val,
        "ci": ci,
        "f_stat": f_stat,
        "n": n,
        "method": "Wald estimator (no covariate adjustment)",
    }


# ---------------------------------------------------------------------------
# IRM — Interactive Regression Model via DoubleML
# ---------------------------------------------------------------------------


def estimate_irm(
    data: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariates: list[str],
    n_folds: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Estimate the ATE via the Interactive Regression Model (IRM) using DoubleML.

    The IRM extends the partially linear model by allowing treatment effect
    heterogeneity.  It models:

    .. math::

        Y = g_0(T, X) + U, \\quad \\mathbb{E}[U \\mid X, T] = 0

        T = m_0(X) + V, \\quad \\mathbb{E}[V \\mid X] = 0

    where :math:`g_0` is the outcome regression and :math:`m_0` is the
    propensity score.  The Neyman-orthogonal score for the ATE is:

    .. math::

        \\psi = g_0(1, X) - g_0(0, X)
        + \\frac{T(Y - g_0(1,X))}{m_0(X)}
        - \\frac{(1-T)(Y - g_0(0,X))}{1 - m_0(X)} - \\theta

    :param data: Input DataFrame.
    :type data: pandas.DataFrame
    :param treatment: Binary treatment column.
    :type treatment: str
    :param outcome: Outcome column.
    :type outcome: str
    :param covariates: Covariate column names.
    :type covariates: list[str]
    :param n_folds: Number of cross-fitting folds (default 5).
    :type n_folds: int
    :param random_state: Random seed (default 42).
    :type random_state: int
    :return: Dictionary with ``ate``, ``se``, ``ci_lower``, ``ci_upper``,
        ``n``, ``method``.
    :rtype: dict[str, Any]

    References
    ----------
    Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
    Newey, W., & Robins, J. (2018). Double/debiased machine learning for
    treatment and structural parameters. *The Econometrics Journal*, 21(1),
    C1--C68. https://doi.org/10.1111/ectj.12097
    """
    try:
        import doubleml as dml
    except ImportError as exc:
        raise ImportError("DoubleML is required for estimate_irm(). Install with: pip install DoubleML") from exc

    frame = data[[treatment, outcome, *covariates]].dropna().copy()

    # Encode non-numeric covariates
    for col in covariates:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            le = LabelEncoder()
            frame[col] = le.fit_transform(frame[col].astype(str))

    np.random.seed(random_state)

    dml_data = dml.DoubleMLData(
        frame,
        y_col=outcome,
        d_cols=treatment,
        x_cols=covariates,
    )

    # ml_g: outcome regression E[Y|T,X] — regressor
    # ml_m: propensity score P(T=1|X) — classifier
    ml_g = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=random_state)
    ml_m = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_state)

    dml_irm = dml.DoubleMLIRM(dml_data, ml_g, ml_m, n_folds=n_folds)
    dml_irm.fit()

    ate = float(dml_irm.coef[0])
    se = float(dml_irm.se[0])
    z = 1.959964

    return {
        "ate": ate,
        "se": se,
        "ci_lower": ate - z * se,
        "ci_upper": ate + z * se,
        "n": len(frame),
        "method": "IRM (DoubleML)",
    }


_DOUBLEML_RANDOM_STATE: int = 42
"""Module-level seed for all DoubleML estimations.  Change at call-site if needed."""

_DOUBLEML_N_FOLDS: int = 5
"""Number of cross-fitting folds.  Must be >= 2."""

_DOUBLEML_N_REP: int = 1
"""Number of repeated cross-fitting repetitions.  Set > 1 to reduce variance."""


def estimate_double_ml(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    covariates: list,
    *,
    random_state: int = _DOUBLEML_RANDOM_STATE,
    n_folds: int = _DOUBLEML_N_FOLDS,
    n_rep: int = _DOUBLEML_N_REP,
):
    """
    Estimate the Average Treatment Effect using Double Machine Learning (DML).

    Uses :class:`doubleml.DoubleMLPLR` (Partially Linear Regression Model)
    with Random Forest nuisance estimators for both the outcome regression
    (``ml_l``) and the treatment model (``ml_m``).

    Reproducibility
    ---------------
    All stochastic operations are seeded deterministically:

    1. ``numpy`` global seed is set to ``random_state`` immediately before
       constructing the learners.
    2. Both ``RandomForestRegressor`` instances receive ``random_state``.
    3. The ``DoubleMLPLR`` object is constructed with ``n_folds`` and
       ``n_rep`` passed explicitly so that the cross-fitting schedule is
       fixed for a given seed.

    To change the seed for a sensitivity run, pass ``random_state=<int>``.

    :param data: The dataset containing outcome, treatment, and covariates.
    :type data: pandas.DataFrame
    :param outcome: Name of the continuous outcome variable.
    :type outcome: str
    :param treatment: Name of the treatment variable.
    :type treatment: str
    :param covariates: List of covariate column names to control for.
    :type covariates: list[str]
    :param random_state: Integer seed for all RNGs.  Default 42.
    :type random_state: int, optional
    :param n_folds: Number of cross-fitting folds.  Default 5.
    :type n_folds: int, optional
    :param n_rep: Number of repeated cross-fitting repetitions.  Default 1.
    :type n_rep: int, optional
    :return: A fitted :class:`doubleml.DoubleMLPLR` object containing the
        causal estimate and inference results.
    :rtype: doubleml.DoubleMLPLR

    References
    ----------
    Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
    Newey, W., & Robins, J. (2018). Double/debiased machine learning for
    treatment and structural parameters. *The Econometrics Journal*, 21(1),
    C1–C68. https://doi.org/10.1111/ectj.12097
    """
    try:
        import doubleml as dml
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "DoubleML is required for estimate_double_ml(). "
            "Install the project dependencies in the repo venv before using this function."
        ) from exc

    # Seed numpy's global RNG before constructing the learners so that the
    # internal random state of the DoubleML cross-fitting splits is fully
    # deterministic for a given random_state value.
    np.random.seed(random_state)

    dml_data = dml.DoubleMLData(data, y_col=outcome, d_cols=treatment, x_cols=covariates)

    # ml_l: outcome regression E[Y|X] — always a regressor.
    # ml_m: treatment model P(T=1|X) — classifier for binary treatment so that
    #       DoubleML receives probability predictions, not continuous predictions.
    #       Using a Regressor here would silently fit E[T|X] as a number in [0,1]
    #       rather than a probability, producing incorrect Neyman-orthogonal scores.
    ml_l = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=random_state)
    ml_m = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_state)

    dml_plr = dml.DoubleMLPLR(dml_data, ml_l, ml_m, n_folds=n_folds, n_rep=n_rep)
    dml_plr.fit()

    return dml_plr
