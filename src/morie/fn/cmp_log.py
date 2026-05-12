# morie.fn -- function file (hadesllm/morie)
"""
Nested logistic model comparison with Wald tests and coefficient tables.

Implements ``compare_nested_logistic_models`` -- fits a sequence of nested
survey-weighted logistic models (null through full interaction), compares
deviance/AIC/pseudo-R-squared, and extracts per-predictor Wald tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from morie.cpads import validate_cpads_frame
from morie.fn._helpers import _safe_exp
from morie.survey import SurveyDesign


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


def compare_nested_logistic_models(
    data: pd.DataFrame,
    *,
    outcome: str = "heavy_drinking_30d",
    weight_col: str = "weight",
) -> dict[str, pd.DataFrame]:
    """
    Compare a nested sequence of survey-weighted logistic models.

    Mirrors the core outputs of the old ``06_model_comparison.R`` module.
    Fits five nested models from an intercept-only null through a full
    interaction specification, then reports deviance, AIC, pseudo-R-squared,
    per-predictor Wald tests, full coefficient tables across all models, and
    interaction term details.

    Models fitted:

    - **Model 0** -- Null (intercept only)
    - **Model 1** -- Demographics (age + gender)
    - **Model 2** -- + Region + Mental Health
    - **Model 3** -- + Cannabis + Physical Health (all main effects)
    - **Model 4** -- + Cannabis x Gender interaction

    :param data: CPADS-compliant DataFrame (passes ``validate_cpads_frame``).
    :type data: pandas.DataFrame
    :param outcome: Name of binary outcome column.
    :type outcome: str
    :param weight_col: Name of survey weight column.
    :type weight_col: str
    :return: Dictionary with keys ``analysis_frame``,
        ``model_comparison_summary``, ``model_comparison_full_coefs``,
        ``model_comparison_interaction``, ``model_comparison_wald_tests``.
    :rtype: dict[str, pandas.DataFrame]

    References
    ----------
    Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic
    Regression* (3rd ed.). Wiley. https://doi.org/10.1002/9781118548387
    """
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
        (
            "Model 1",
            "Demographics (age + gender)",
            f"{outcome} ~ age_group + gender",
        ),
        (
            "Model 2",
            "+ Region + Mental Health",
            f"{outcome} ~ age_group + gender + province_region + mental_health",
        ),
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
    for term in [
        "age_group",
        "gender",
        "province_region",
        "mental_health",
        "cannabis_any_use",
        "physical_health",
    ]:
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

    # Full coefficient table for all models
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


cmp_log = compare_nested_logistic_models


def cheatsheet() -> str:
    return "_scalarize({}) -> Nested logistic model comparison with Wald tests and coeffic"
