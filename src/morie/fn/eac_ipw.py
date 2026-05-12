# morie.fn — function file (hadesllm/morie)
"""
eBAC selection-adjusted IPW analysis pipeline.

Implements ``run_ebac_selection_ipw_analysis`` — corrects for non-random
observation of eBAC using inverse probability weighting on the selection
(observation) mechanism, then fits treatment-effect models on the
selection-corrected sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from morie.fn._helpers import _safe_exp
from morie.fn.ess import effective_sample_size


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
    r"""
    Reproduce the core outputs of the old ``07_ebac_ipw.R`` workflow.

    This function:

    1. Restricts to eligible respondents (``eligible_col == 1``).
    2. Models the probability of being *observed* (having non-missing eBAC)
       via a logistic regression on treatment + covariates.
    3. Constructs selection weights :math:`w_i = P(R=1) / \\hat{P}(R=1 \\mid X_i)`,
       trims at the 1st/99th percentiles, and combines with survey weights.
    4. Fits a selection-adjusted logistic model (binary outcome: over legal
       limit) and a WLS model (continuous outcome: total eBAC).
    5. Returns weight diagnostics, full coefficient tables, covariate balance
       (raw vs. IPW-adjusted SMDs), and cannabis-specific treatment comparisons.

    :param data: Input DataFrame.
    :type data: pandas.DataFrame
    :param observation_col: Column whose missingness defines the selection indicator R.
    :type observation_col: str
    :param eligible_col: Column indicating eligibility (must equal 1).
    :type eligible_col: str
    :param treatment: Binary treatment column.
    :type treatment: str
    :param binary_outcome: Binary outcome column for logistic model.
    :type binary_outcome: str
    :param continuous_outcome: Continuous outcome column for linear model.
    :type continuous_outcome: str
    :param survey_weight_col: Survey weight column.
    :type survey_weight_col: str
    :param covariates: Covariate column names (defaults to standard CPADS set).
    :type covariates: list[str] or None
    :return: Dictionary of DataFrames with diagnostics, coefficient tables,
        and treatment comparisons.
    :rtype: dict[str, pandas.DataFrame]
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

    # Weight diagnostics -- detailed per-observation weight summary
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

    # Logistic OR -- full coefficient table from the IPW logistic model
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

    # Linear coefficients -- full table
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

    # Cannabis comparison -- extended with unweighted estimates
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

    # Observation model OR -- from the selection model
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

    # Covariate balance -- standardised mean differences before/after IPW
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


eac_ipw = run_ebac_selection_ipw_analysis


def cheatsheet() -> str:
    return "run_ebac_selection_ipw_analysis({}) -> eBAC selection-adjusted IPW analysis pipeline."
