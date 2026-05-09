"""
Treatment effects analysis: ATE, ATT, ATC, and subgroup CATE via IPW.

Implements ``run_treatment_effects_analysis`` — computes Hajek IPW estimators
for ATE/ATT/ATC and subgroup-level CATE with sandwich-style standard errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from moirais.cpads import validate_cpads_frame
from moirais.fn.ipw import calculate_ipw_weights
from moirais.fn.ps_fit import compute_propensity_scores

DEFAULT_INVESTIGATION_COVARIATES = [
    "age_group",
    "gender",
    "province_region",
    "mental_health",
    "physical_health",
]


def _prepare_analysis_frame(
    data: pd.DataFrame,
    *,
    required: list[str],
) -> pd.DataFrame:
    validate_cpads_frame(data, strict=True)
    return data.loc[:, required].dropna().copy()


def _hajek_ate(
    ps: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float]:
    r"""
    Hajek (normalised Horvitz-Thompson) ATE estimator.

    .. math::

        \widehat{\text{ATE}}_{\text{Hajek}} =
            \frac{\sum_i T_i Y_i / e_i}{\sum_i T_i / e_i}
            -
            \frac{\sum_i (1-T_i) Y_i / (1-e_i)}{\sum_i (1-T_i) / (1-e_i)}

    :param ps: Clipped propensity scores in (0, 1).
    :type ps: numpy.ndarray
    :param t: Binary treatment indicator (0 or 1).
    :type t: numpy.ndarray
    :param y: Outcome values.
    :type y: numpy.ndarray
    :return: ``(ate, y1_hajek, y0_hajek)``
    :rtype: tuple[float, float, float]

    References
    ----------
    Lunceford, J. K., & Davidian, M. (2004). Stratification and weighting
    via the propensity score in estimation of causal treatment effects: a
    comparative study. *Statistics in Medicine*, 23(19), 2937--2960.
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
    r"""
    Compute ATE, ATT, ATC, and subgroup CATE via IPW.

    Mirrors the core outputs of the old ``07_treatment_effects.R`` module.

    **ATE** — Hajek (normalised Horvitz-Thompson) estimator.  See
    :func:`_hajek_ate` for the formula.

    **ATT** — weighted mean of control outcomes under ATT weights:

    .. math::

        \widehat{\text{ATT}} =
            \bar{Y}_1 -
            \frac{\sum_{T_i=0} Y_i e_i / (1-e_i)}{\sum_{T_i=0} e_i / (1-e_i)}

    **ATC** — weighted mean of treated outcomes under ATC weights:

    .. math::

        \widehat{\text{ATC}} =
            \frac{\sum_{T_i=1} Y_i (1-e_i) / e_i}{\sum_{T_i=1} (1-e_i) / e_i}
            - \bar{Y}_0

    **CATE** subgroup estimates use the Hajek estimator applied within each
    subgroup, with global propensity scores reused (stratified-IPW).

    :param data: CPADS-compliant DataFrame (passes ``validate_cpads_frame``).
    :type data: pandas.DataFrame
    :param treatment: Name of binary treatment column.
    :type treatment: str
    :param outcome: Name of binary outcome column.
    :type outcome: str
    :param covariates: Covariates for propensity model and CATE subgroups.
        Defaults to ``DEFAULT_INVESTIGATION_COVARIATES``.
    :type covariates: list[str] or None
    :return: Dictionary with keys ``analysis_frame``,
        ``treatment_effects_summary``, ``cate_subgroup_estimates``.
    :rtype: dict[str, pandas.DataFrame]

    References
    ----------
    Lunceford, J. K., & Davidian, M. (2004). Stratification and weighting
    via the propensity score in estimation of causal treatment effects: a
    comparative study. *Statistics in Medicine*, 23(19), 2937--2960.
    https://doi.org/10.1002/sim.1903

    Rosenbaum, P. R., & Rubin, D. B. (1983). The central role of the
    propensity score in observational studies for causal effects.
    *Biometrika*, 70(1), 41--55. https://doi.org/10.1093/biomet/70.1.41
    """
    covariates = covariates or DEFAULT_INVESTIGATION_COVARIATES
    required = [treatment, outcome, "weight", *covariates]
    frame = _prepare_analysis_frame(data, required=required)

    # Propensity scores via moirais.fn.ps_fit
    frame["ps"] = compute_propensity_scores(frame, treatment=treatment, covariates=covariates).clip(0.01, 0.99)

    # Standard IPW weight columns via moirais.fn.ipw (kept for diagnostics)
    frame["w_ate"] = calculate_ipw_weights(frame, treatment, "ps")
    frame["w_att"] = np.where(frame[treatment] == 1, 1.0, frame["ps"] / (1.0 - frame["ps"]))
    frame["w_atc"] = np.where(frame[treatment] == 1, (1.0 - frame["ps"]) / frame["ps"], 1.0)

    t = frame[treatment].values.astype(float)
    y = frame[outcome].values.astype(float)
    ps = frame["ps"].values

    # ATE — Hajek estimator
    ate, _, _ = _hajek_ate(ps, t, y)

    # ATT — treated mean is unweighted, control uses ATT weights
    treated_mask = t == 1
    control_mask = t == 0
    y1_mean = float(y[treated_mask].mean()) if treated_mask.sum() > 0 else np.nan

    ps_control = ps[control_mask]
    y_control = y[control_mask]
    att_w_control = ps_control / (1.0 - ps_control)
    y0_att = float(np.sum(y_control * att_w_control) / np.sum(att_w_control)) if att_w_control.sum() > 0 else np.nan
    att = float(y1_mean - y0_att) if not (np.isnan(y1_mean) or np.isnan(y0_att)) else np.nan

    # ATC — control mean is unweighted, treated uses ATC weights
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

    # CATE subgroup estimates — Hajek IPW within each subgroup
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

            # Approximate SE via sandwich-style variance
            ps_t = sub_ps[sub_t == 1]
            ps_c = sub_ps[sub_t == 0]
            y_t = sub_y[sub_t == 1]
            y_c = sub_y[sub_t == 0]
            w_t = 1.0 / ps_t
            w_c = 1.0 / (1.0 - ps_c)
            resid_t = y_t - y1_sub
            resid_c = y_c - y0_sub
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


te_ana = run_treatment_effects_analysis


def cheatsheet() -> str:
    return "_prepare_analysis_frame({}) -> Treatment effects analysis: ATE, ATT, ATC, and subgroup CATE"
