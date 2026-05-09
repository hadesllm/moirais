"""Python counterparts for core investigation modules from the old R workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .causal import compute_propensity_scores
from .cpads import validate_cpads_frame
from .survey import SurveyDesign

DEFAULT_INVESTIGATION_COVARIATES = [
    "age_group",
    "gender",
    "province_region",
    "mental_health",
    "physical_health",
]


def _safe_exp(values) -> np.ndarray:
    """Exponentiate on a clipped log scale to avoid overflow warnings in output tables."""
    return np.exp(np.clip(np.asarray(values, dtype=float), -700, 700))


def _scalarize(value) -> float:
    """Convert statsmodels scalar/array-like outputs into a plain float."""
    return float(np.asarray(value, dtype=float).squeeze())


def _prepare_analysis_frame(
    data: pd.DataFrame,
    *,
    required: list[str],
) -> pd.DataFrame:
    validate_cpads_frame(data, strict=True)
    return data.loc[:, required].dropna().copy()


def _extract_or_table(fit, *, model_name: str | None = None) -> pd.DataFrame:
    conf = fit.conf_int()
    table = pd.DataFrame(
        {
            "term": fit.params.index,
            "log_odds": fit.params.values,
            "SE": fit.bse.values,
            "OR": _safe_exp(fit.params.values),
            "OR_lower95": _safe_exp(conf[0].values),
            "OR_upper95": _safe_exp(conf[1].values),
            "p_value": fit.pvalues.values,
            "significant": np.where(fit.pvalues.values < 0.05, "*", ""),
        }
    )
    if model_name is not None:
        table = table.rename(
            columns={
                "SE": "se",
                "OR": "or",
                "OR_lower95": "or_lower95",
                "OR_upper95": "or_upper95",
            }
        )
        table.insert(0, "model", model_name)
    return table


def run_weighted_logistic_analysis(
    data: pd.DataFrame,
    *,
    outcome: str = "heavy_drinking_30d",
    treatment: str = "cannabis_any_use",
    weight_col: str = "weight",
    predictors: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Mirror the core outputs of the old `06_logistic.R` module."""
    predictors = predictors or [
        "age_group",
        "gender",
        "province_region",
        "mental_health",
        treatment,
    ]
    required = [outcome, weight_col, *predictors]
    frame = _prepare_analysis_frame(data, required=required)
    design = SurveyDesign(frame, weights_col=weight_col)

    formula = f"{outcome} ~ {' + '.join(predictors)}"
    fit = design.svyglm(formula, family=sm.families.Binomial())
    or_table = _extract_or_table(fit)

    interaction_formula = formula + f" + {treatment}:gender"
    fit_int = design.svyglm(interaction_formula, family=sm.families.Binomial())
    int_or_table = _extract_or_table(
        fit_int,
        model_name="svy_interaction_cannabis_by_gender",
    )

    interaction_rows = [term for term in fit_int.params.index if ":" in term and treatment in term and "gender" in term]
    if interaction_rows:
        contrast = np.zeros((len(interaction_rows), len(fit_int.params)))
        param_index = list(fit_int.params.index)
        for row_idx, term in enumerate(interaction_rows):
            contrast[row_idx, param_index.index(term)] = 1.0
        wald = fit_int.wald_test(contrast, scalar=True)
        stat = _scalarize(wald.statistic)
        p_value = _scalarize(wald.pvalue)
        df_num = len(interaction_rows)
    else:
        stat = 0.0
        p_value = 1.0
        df_num = 0

    interaction_tests = pd.DataFrame(
        [
            {
                "test": "cannabis_any_use:gender (joint Wald)",
                "F_stat": stat,
                "df_num": df_num,
                "df_den": float(getattr(fit_int, "df_resid", len(frame) - len(fit_int.params))),
                "p_value": p_value,
                "analysis_mode": "observational",
                "model": "heavy_drinking_interaction",
            }
        ]
    )

    # SMOTE sensitivity — rebalance and refit to check stability of ORs
    from .ml import apply_smote

    y_smote = frame[outcome].astype(int)
    X_smote = pd.get_dummies(frame[predictors], drop_first=True, dtype=float)
    X_res, y_res, smote_info = apply_smote(X_smote, y_smote)

    smote_status = pd.DataFrame([smote_info])

    # Refit logistic on SMOTE-resampled data
    smote_frame = X_res.copy()
    smote_frame[outcome] = y_res.values
    smote_formula = f"{outcome} ~ " + " + ".join(X_res.columns)
    try:
        smote_fit = sm.GLM(
            smote_frame[outcome],
            sm.add_constant(X_res),
            family=sm.families.Binomial(),
        ).fit()
        smote_or_table = pd.DataFrame(
            {
                "term": smote_fit.params.index,
                "log_odds": smote_fit.params.values,
                "SE": smote_fit.bse.values,
                "OR": _safe_exp(smote_fit.params.values),
                "OR_lower95": _safe_exp(smote_fit.conf_int()[0].values),
                "OR_upper95": _safe_exp(smote_fit.conf_int()[1].values),
                "p_value": smote_fit.pvalues.values,
            }
        )
    except Exception:
        smote_or_table = pd.DataFrame(columns=["term", "log_odds", "SE", "OR", "OR_lower95", "OR_upper95", "p_value"])

    return {
        "analysis_frame": frame,
        "logistic_odds_ratios": or_table,
        "logistic_interaction_odds_ratios": int_or_table,
        "logistic_interaction_tests": interaction_tests,
        "logistic_smote_status": smote_status,
        "logistic_smote_odds_ratios": smote_or_table,
    }


def compare_nested_logistic_models(
    data: pd.DataFrame,
    *,
    outcome: str = "heavy_drinking_30d",
    weight_col: str = "weight",
) -> dict[str, pd.DataFrame]:
    """Mirror the core outputs of the old `06_model_comparison.R` module."""
    predictors = [
        "age_group",
        "gender",
        "province_region",
        "mental_health",
        "cannabis_any_use",
        "physical_health",
    ]
    required = [outcome, weight_col, *predictors]
    frame = _prepare_analysis_frame(data, required=required)
    design = SurveyDesign(frame, weights_col=weight_col)

    formulas = [
        ("Model 0", "Null (intercept only)", f"{outcome} ~ 1"),
        ("Model 1", "Demographics (age + gender)", f"{outcome} ~ age_group + gender"),
        ("Model 2", "+ Region + Mental Health", f"{outcome} ~ age_group + gender + province_region + mental_health"),
        (
            "Model 3",
            "+ Cannabis + Physical Health",
            f"{outcome} ~ age_group + gender + province_region + mental_health + cannabis_any_use + physical_health",
        ),
        (
            "Model 4",
            "+ Cannabis x Gender interaction",
            f"{outcome} ~ age_group + gender + province_region + mental_health"
            f" + cannabis_any_use + physical_health + cannabis_any_use:gender",
        ),
    ]

    fits = []
    for label, description, formula in formulas:
        fit = design.svyglm(formula, family=sm.families.Binomial())
        fits.append((label, description, formula, fit))

    null_deviance = float(fits[0][3].deviance)
    summary_rows = []
    for label, description, formula, fit in fits:
        summary_rows.append(
            {
                "model": label,
                "description": description,
                "n_parameters": len(fit.params),
                "deviance": round(float(fit.deviance), 2),
                "df_residual": float(fit.df_resid),
                "AIC_approx": round(float(fit.aic), 2),
                "pseudo_R2": round(1 - (float(fit.deviance) / null_deviance), 6) if null_deviance else 0.0,
            }
        )

    full_fit = fits[3][3]
    term_table = full_fit.wald_test_terms(skip_single=False, scalar=True).table
    wald_rows = []
    for term in ["age_group", "gender", "province_region", "mental_health", "cannabis_any_use", "physical_health"]:
        if term in term_table.index:
            row = term_table.loc[term]
            wald_rows.append(
                {
                    "predictor": term,
                    "F_statistic": round(_scalarize(row["statistic"]), 4),
                    "df_num": _scalarize(row["df_constraint"]),
                    "df_denom": float(full_fit.df_resid),
                    "p_value": _scalarize(row["pvalue"]),
                    "significant": "Yes" if _scalarize(row["pvalue"]) < 0.05 else "No",
                }
            )

    # Full coefficient table for the best model (Model 3 with all predictors)
    full_coef_rows = []
    for label, description, formula, fit in fits:
        conf = fit.conf_int()
        for term in fit.params.index:
            full_coef_rows.append(
                {
                    "model": label,
                    "term": term,
                    "log_odds": float(fit.params[term]),
                    "SE": float(fit.bse[term]),
                    "OR": float(_safe_exp(fit.params[term])),
                    "OR_lower95": float(_safe_exp(conf.loc[term, 0])),
                    "OR_upper95": float(_safe_exp(conf.loc[term, 1])),
                    "p_value": float(fit.pvalues[term]),
                }
            )

    # Interaction model details (Model 4)
    int_fit = fits[4][3]
    int_conf = int_fit.conf_int()
    int_terms = [t for t in int_fit.params.index if ":" in t]
    interaction_rows = []
    for term in int_terms:
        interaction_rows.append(
            {
                "term": term,
                "log_odds": float(int_fit.params[term]),
                "SE": float(int_fit.bse[term]),
                "OR": float(_safe_exp(int_fit.params[term])),
                "OR_lower95": float(_safe_exp(int_conf.loc[term, 0])),
                "OR_upper95": float(_safe_exp(int_conf.loc[term, 1])),
                "p_value": float(int_fit.pvalues[term]),
                "significant": "Yes" if float(int_fit.pvalues[term]) < 0.05 else "No",
            }
        )

    return {
        "analysis_frame": frame,
        "model_comparison_summary": pd.DataFrame(summary_rows),
        "model_comparison_full_coefs": pd.DataFrame(full_coef_rows),
        "model_comparison_interaction": pd.DataFrame(interaction_rows),
        "model_comparison_wald_tests": pd.DataFrame(wald_rows),
    }


def _hajek_ate(
    ps: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float]:
    """
    Hájek (normalised Horvitz-Thompson) ATE estimator.

    .. math::

        \\widehat{\\text{ATE}}_{\\text{Hájek}} =
            \\frac{\\sum_i T_i Y_i / e_i}{\\sum_i T_i / e_i}
            -
            \\frac{\\sum_i (1-T_i) Y_i / (1-e_i)}{\\sum_i (1-T_i) / (1-e_i)}

    Parameters
    ----------
    ps : np.ndarray
        Clipped propensity scores in (0, 1).
    t : np.ndarray
        Binary treatment indicator (0 or 1).
    y : np.ndarray
        Outcome values.

    Returns
    -------
    tuple[float, float, float]
        (ate, y1_hajek, y0_hajek)

    References
    ----------
    Lunceford, J. K., & Davidian, M. (2004). Stratification and weighting
    via the propensity score in estimation of causal treatment effects: a
    comparative study. *Statistics in Medicine*, 23(19), 2937–2960.
    https://doi.org/10.1002/sim.1903
    """
    treated_mask = t == 1
    control_mask = t == 0

    sum_ty = np.sum(y[treated_mask] / ps[treated_mask])
    sum_t = np.sum(1.0 / ps[treated_mask])
    y1 = float(sum_ty / sum_t) if sum_t > 0 else np.nan

    sum_cy = np.sum(y[control_mask] / (1.0 - ps[control_mask]))
    sum_c = np.sum(1.0 / (1.0 - ps[control_mask]))
    y0 = float(sum_cy / sum_c) if sum_c > 0 else np.nan

    return float(y1 - y0), y1, y0


def run_treatment_effects_analysis(
    data: pd.DataFrame,
    *,
    treatment: str = "cannabis_any_use",
    outcome: str = "heavy_drinking_30d",
    covariates: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Mirror the core outputs of the old ``07_treatment_effects.R`` module.

    ATE / ATT / ATC Estimators
    --------------------------
    All three estimands use IPW weights derived from the estimated propensity
    score :math:`e_i = P(T_i=1 \\mid X_i)`.

    **ATE** — Hájek (normalised Horvitz-Thompson) estimator.  See
    :func:`_hajek_ate` for the formula.  This is preferred over the
    unnormalised HT estimator because self-normalisation gives better
    finite-sample performance (Lunceford & Davidian, 2004).

    **ATT** — weighted mean of control outcomes under ATT weights:

    .. math::

        \\widehat{\\text{ATT}} =
            \\bar{Y}_1 -
            \\frac{\\sum_{T_i=0} Y_i e_i / (1-e_i)}{\\sum_{T_i=0} e_i / (1-e_i)}

    **ATC** — weighted mean of treated outcomes under ATC weights:

    .. math::

        \\widehat{\\text{ATC}} =
            \\frac{\\sum_{T_i=1} Y_i (1-e_i) / e_i}{\\sum_{T_i=1} (1-e_i) / e_i}
            - \\bar{Y}_0

    CATE Subgroup Estimates
    -----------------------
    Conditional average treatment effects (CATE) for each subgroup level
    are computed with the Hájek estimator **applied within each subgroup**.
    Raw (unadjusted) within-subgroup means are **not** used because they are
    confounded by covariate imbalance within the subgroup.  Propensity scores
    are re-used from the global model; this is a standard stratified-IPW
    approach for subgroup CATE estimation.

    References
    ----------
    Lunceford, J. K., & Davidian, M. (2004). Stratification and weighting
    via the propensity score in estimation of causal treatment effects: a
    comparative study. *Statistics in Medicine*, 23(19), 2937–2960.
    https://doi.org/10.1002/sim.1903

    Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the
    propensity score in observational studies for causal effects.
    *Biometrika*, 70(1), 41–55. https://doi.org/10.1093/biomet/70.1.41
    """
    covariates = covariates or DEFAULT_INVESTIGATION_COVARIATES
    required = [treatment, outcome, "weight", *covariates]
    frame = _prepare_analysis_frame(data, required=required)

    frame["ps"] = compute_propensity_scores(frame, treatment=treatment, covariates=covariates).clip(0.01, 0.99)

    # Standard IPW weight columns (kept for downstream diagnostics / plots).
    frame["w_ate"] = np.where(frame[treatment] == 1, 1.0 / frame["ps"], 1.0 / (1.0 - frame["ps"]))
    frame["w_att"] = np.where(frame[treatment] == 1, 1.0, frame["ps"] / (1.0 - frame["ps"]))
    frame["w_atc"] = np.where(frame[treatment] == 1, (1.0 - frame["ps"]) / frame["ps"], 1.0)

    t = frame[treatment].values.astype(float)
    y = frame[outcome].values.astype(float)
    ps = frame["ps"].values

    # ------------------------------------------------------------------
    # ATE — Hájek estimator (normalised HT).
    # The previous code computed np.average(treated_y, weights=treated_ipw)
    # and np.average(control_y, weights=control_ipw) separately.  That
    # formula is NOT the Horvitz-Thompson estimator; it is an ad-hoc
    # within-arm weighted mean with no valid causal interpretation.
    # ------------------------------------------------------------------
    ate, _, _ = _hajek_ate(ps, t, y)

    # ------------------------------------------------------------------
    # ATT — Hájek normalised estimator for the treated subpopulation.
    # Treated mean is unweighted (w_att = 1 for treated).
    # Control mean uses ATT weights (e / (1-e)), normalised.
    # ------------------------------------------------------------------
    treated_mask = t == 1
    control_mask = t == 0
    y1_mean = float(y[treated_mask].mean()) if treated_mask.sum() > 0 else np.nan

    ps_control = ps[control_mask]
    y_control = y[control_mask]
    att_w_control = ps_control / (1.0 - ps_control)
    y0_att = float(np.sum(y_control * att_w_control) / np.sum(att_w_control)) if att_w_control.sum() > 0 else np.nan
    att = float(y1_mean - y0_att) if not (np.isnan(y1_mean) or np.isnan(y0_att)) else np.nan

    # ------------------------------------------------------------------
    # ATC — Hájek normalised estimator for the control subpopulation.
    # Control mean is unweighted.  Treated mean uses ATC weights ((1-e)/e),
    # normalised.
    # ------------------------------------------------------------------
    y0_mean = float(y[control_mask].mean()) if control_mask.sum() > 0 else np.nan

    ps_treated = ps[treated_mask]
    y_treated = y[treated_mask]
    atc_w_treated = (1.0 - ps_treated) / ps_treated
    y1_atc = float(np.sum(y_treated * atc_w_treated) / np.sum(atc_w_treated)) if atc_w_treated.sum() > 0 else np.nan
    atc = float(y1_atc - y0_mean) if not (np.isnan(y1_atc) or np.isnan(y0_mean)) else np.nan

    treatment_effects_summary = pd.DataFrame(
        [
            {
                "estimand": "ATE",
                "method": "IPW-Hajek",
                "estimate": ate,
                "se": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
            },
            {
                "estimand": "ATT",
                "method": "IPW-Hajek",
                "estimate": att,
                "se": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
            },
            {
                "estimand": "ATC",
                "method": "IPW-Hajek",
                "estimate": atc,
                "se": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
            },
        ]
    )

    # ------------------------------------------------------------------
    # CATE subgroup estimates — Hájek IPW within each subgroup.
    # INCORRECT prior code: used raw unadjusted within-subgroup means,
    # which conflates treatment effect with covariate imbalance inside
    # the subgroup.  The global propensity score is reused here; fitting
    # subgroup-specific models would require substantially more data.
    # ------------------------------------------------------------------
    cate_rows = []
    for subgroup_var in covariates:
        for subgroup_level, subset in frame.groupby(subgroup_var):
            sub_t = subset[treatment].values.astype(float)
            sub_y = subset[outcome].values.astype(float)
            sub_ps = subset["ps"].values

            n_treated_sub = int((sub_t == 1).sum())
            n_control_sub = int((sub_t == 0).sum())
            if n_treated_sub < 2 or n_control_sub < 2:
                continue

            cate_val, y1_sub, y0_sub = _hajek_ate(sub_ps, sub_t, sub_y)

            # Approximate SE via delta method on IPW sums (conservative).
            # For a fully valid SE use bootstrap_ci from inference.py.
            ps_t = sub_ps[sub_t == 1]
            ps_c = sub_ps[sub_t == 0]
            y_t = sub_y[sub_t == 1]
            y_c = sub_y[sub_t == 0]
            w_t = 1.0 / ps_t
            w_c = 1.0 / (1.0 - ps_c)
            resid_t = y_t - y1_sub
            resid_c = y_c - y0_sub
            # Sandwich-style variance: Var(Hajek) ≈ sum(w^2 * resid^2) / (sum(w))^2
            var_y1 = float(np.sum(w_t**2 * resid_t**2) / (np.sum(w_t) ** 2)) if np.sum(w_t) > 0 else 0.0
            var_y0 = float(np.sum(w_c**2 * resid_c**2) / (np.sum(w_c) ** 2)) if np.sum(w_c) > 0 else 0.0
            se = float(np.sqrt(var_y1 + var_y0))

            cate_rows.append(
                {
                    "subgroup_var": subgroup_var,
                    "subgroup_level": subgroup_level,
                    "n_treated": n_treated_sub,
                    "n_control": n_control_sub,
                    "cate": cate_val,
                    "se": se,
                    "ci_lower": cate_val - 1.96 * se,
                    "ci_upper": cate_val + 1.96 * se,
                }
            )

    return {
        "analysis_frame": frame,
        "treatment_effects_summary": treatment_effects_summary,
        "cate_subgroup_estimates": pd.DataFrame(cate_rows),
    }
