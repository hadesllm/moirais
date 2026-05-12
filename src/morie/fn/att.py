# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""
Average Treatment Effect on the Treated (ATT) via Hajek-weighted IPW.

Implements ``estimate_att`` -- estimates the causal effect of treatment
among those who actually received it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from morie.fn.ess import effective_sample_size
from morie.fn.ps_fit import compute_propensity_scores


def estimate_att(
    data: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariates: list[str],
    propensity_col: str | None = None,
) -> dict[str, Any]:
    r"""Estimate the Average Treatment Effect on the Treated (ATT) via
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
        ess_ctrl = effective_sample_size(w_control)["ess"]
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


att = estimate_att


def cheatsheet() -> str:
    return "estimate_att({}) -> Average Treatment Effect on the Treated (ATT) via Hajek-weig"
