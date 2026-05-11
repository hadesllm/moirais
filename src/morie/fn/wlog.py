"""
Weighted logistic regression analysis with interaction and SMOTE sensitivity.

Implements ``run_weighted_logistic_analysis`` — fits a survey-weighted logistic
model for a binary outcome, tests treatment-by-covariate interactions, and
runs a SMOTE sensitivity refit to check odds ratio stability.
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
    """
    Survey-weighted logistic regression with interaction tests and SMOTE sensitivity.

    Mirrors the core outputs of the old ``06_logistic.R`` module.  Fits a main
    logistic model, adds a treatment-by-gender interaction, runs a joint Wald
    test on the interaction terms, then refits on SMOTE-resampled data to
    check odds ratio stability under class rebalancing.

    :param data: CPADS-compliant DataFrame (passes ``validate_cpads_frame``).
    :type data: pandas.DataFrame
    :param outcome: Name of binary outcome column.
    :type outcome: str
    :param treatment: Name of binary treatment column.
    :type treatment: str
    :param weight_col: Name of survey weight column.
    :type weight_col: str
    :param predictors: Covariates for the main model.  Defaults to
        ``[age_group, gender, province_region, mental_health, treatment]``.
    :type predictors: list[str] or None
    :return: Dictionary with keys ``analysis_frame``, ``logistic_odds_ratios``,
        ``logistic_interaction_odds_ratios``, ``logistic_interaction_tests``,
        ``logistic_smote_status``, ``logistic_smote_odds_ratios``.
    :rtype: dict[str, pandas.DataFrame]

    References
    ----------
    Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic
    Regression* (3rd ed.). Wiley. https://doi.org/10.1002/9781118548387
    """
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
    from morie.ml import apply_smote

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
        smote_or_table = pd.DataFrame(
            columns=[
                "term",
                "log_odds",
                "SE",
                "OR",
                "OR_lower95",
                "OR_upper95",
                "p_value",
            ]
        )

    return {
        "analysis_frame": frame,
        "logistic_odds_ratios": or_table,
        "logistic_interaction_odds_ratios": int_or_table,
        "logistic_interaction_tests": interaction_tests,
        "logistic_smote_status": smote_status,
        "logistic_smote_odds_ratios": smote_or_table,
    }


wlog = run_weighted_logistic_analysis


def cheatsheet() -> str:
    return "_scalarize({}) -> Weighted logistic regression analysis with interaction and S"
