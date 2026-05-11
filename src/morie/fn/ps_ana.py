# morie.fn — function file (hadesllm/morie)
"""
Propensity score IPW analysis pipeline.

Implements ``run_propensity_ipw_analysis`` — an end-to-end orchestration
function that computes propensity scores, IPW weights, and the Hajek
(normalised Horvitz-Thompson) ATE estimator with diagnostics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from morie.fn.ess import effective_sample_size
from morie.fn.ipw import calculate_ipw_weights
from morie.fn.ps_fit import compute_propensity_scores


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
    The Average Treatment Effect is estimated via the **Hajek (normalised
    Horvitz-Thompson) IPW estimator**:

    .. math::

        \\widehat{\\text{ATE}}_{\\text{Hajek}} =
            \\frac{\\sum_i T_i Y_i / e_i}{\\sum_i T_i / e_i}
            -
            \\frac{\\sum_i (1-T_i) Y_i / (1-e_i)}{\\sum_i (1-T_i) / (1-e_i)}

    where :math:`T_i \\in \\{0,1\\}` is the treatment indicator,
    :math:`Y_i` is the outcome, and :math:`e_i = P(T_i=1 \\mid X_i)` is the
    estimated propensity score.

    The Hajek estimator is preferred over the raw Horvitz-Thompson estimator
    because it is self-normalised (the weights sum to 1 within each arm),
    which gives better finite-sample performance and is less sensitive to
    extreme propensity scores.

    Trimmed IPW weights (1st--99th percentile) are used to reduce the
    influence of extreme propensity scores.

    References
    ----------
    Lunceford, J. K., & Davidian, M. (2004). Stratification and weighting
    via the propensity score in estimation of causal treatment effects: a
    comparative study. *Statistics in Medicine*, 23(19), 2937--2960.
    https://doi.org/10.1002/sim.1903

    Hajek, J. (1971). Comment on "An essay on the logical foundations of
    survey sampling" by D. Basu. In V. P. Godambe & D. A. Sprott (Eds.),
    *Foundations of Statistical Inference* (pp. 236--236). Holt, Rinehart &
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
    # Hajek (normalised Horvitz-Thompson) ATE estimator.
    #
    # INCORRECT approach (prior code):
    #   Computed np.average(treated_outcomes, weights=treated_ipw) and
    #   np.average(control_outcomes, weights=control_ipw) separately within
    #   each arm.  This is NOT the Horvitz-Thompson estimator; it is an
    #   ad-hoc weighted mean within the observed arm only and does not
    #   correspond to any standard causal estimator.
    #
    # CORRECT approach (Hajek):
    #   Numerator for treated arm  = sum over ALL units of T_i * Y_i / e_i
    #   Denominator for treated arm = sum over ALL units of T_i / e_i
    #   Same pattern for control arm using (1 - T_i) / (1 - e_i).
    #   Dividing normalises the weights so they sum to 1 within each arm.
    # -----------------------------------------------------------------------
    t = frame[treatment].values
    y = frame[outcome].values

    # Use the raw (unclipped) propensity score column directly; apply the same
    # 0.01-0.99 clipping that calculate_ipw_weights uses so that the Hajek
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


ps_ana = run_propensity_ipw_analysis


def cheatsheet() -> str:
    return "run_propensity_ipw_analysis({}) -> Propensity score IPW analysis pipeline."
