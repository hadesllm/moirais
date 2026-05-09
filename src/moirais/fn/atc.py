# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""
Average Treatment Effect on the Controls (ATC) via Hajek-weighted IPW.

Implements ``estimate_atc`` — estimates the causal effect of treatment
among those who did not receive it (the controls).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from moirais.fn.ess import effective_sample_size
from moirais.fn.ps_fit import compute_propensity_scores


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
        ess_treat = effective_sample_size(w_treated)["ess"]
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


atc = estimate_atc


def cheatsheet() -> str:
    return "estimate_atc({}) -> Average Treatment Effect on the Controls (ATC) via Hajek-wei"
