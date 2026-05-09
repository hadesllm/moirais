# moirais.fn — function file (hadesllm/moirais)
"""Changes-in-Changes estimator (Athey & Imbens, 2006)."""

from __future__ import annotations

from typing import Any, Union

import numpy as np


def changes_in_changes(
    outcome: Union[list, np.ndarray],
    group: Union[list, np.ndarray],
    time: Union[list, np.ndarray],
    *,
    n_quantiles: int = 99,
) -> dict[str, Any]:
    """
    Changes-in-Changes (CIC) estimator for the average treatment effect.

    Unlike standard DiD, CIC does not assume parallel trends in levels.
    Instead it assumes that the distribution of *potential* untreated
    outcomes for the treated group can be recovered by mapping through
    the control group's quantile function.

    Steps:
    1. Estimate F_{Y|G=0,T=0} and F_{Y|G=0,T=1} (control CDFs).
    2. For each treated-post observation y, compute its counterfactual
       via F_{0,1}^{-1}( F_{0,0}( y_{treated, pre quantile} )).
    3. ATE = mean(Y_{treated,post}) - mean(counterfactual).

    Also computes quantile treatment effects at each quantile.

    :param outcome: Outcome variable.
    :param group: Binary group indicator (0 = control, 1 = treated).
    :param time: Binary time indicator (0 = pre, 1 = post).
    :param n_quantiles: Number of quantile points for QTE.
    :return: Dictionary with ate, quantile_effects (array), quantiles (array).
    :raises ValueError: If any cell has fewer than 2 observations.

    References
    ----------
    Athey, S., & Imbens, G. W. (2006). Identification and inference in
    nonlinear difference-in-differences models. *Econometrica*, 74(2),
    431--497.
    """
    y = np.asarray(outcome, dtype=float)
    g = np.asarray(group, dtype=int)
    t = np.asarray(time, dtype=int)
    n = len(y)
    if len(g) != n or len(t) != n:
        raise ValueError("outcome, group, time must have the same length.")

    # Four cells
    c_pre = y[(g == 0) & (t == 0)]
    c_post = y[(g == 0) & (t == 1)]
    t_pre = y[(g == 1) & (t == 0)]
    t_post = y[(g == 1) & (t == 1)]

    for label, cell in [
        ("control-pre", c_pre),
        ("control-post", c_post),
        ("treated-pre", t_pre),
        ("treated-post", t_post),
    ]:
        if len(cell) < 2:
            raise ValueError(f"Cell '{label}' has fewer than 2 observations.")

    quantiles = np.linspace(1.0 / (n_quantiles + 1), n_quantiles / (n_quantiles + 1), n_quantiles)

    # Quantile functions
    q_c_pre = np.quantile(c_pre, quantiles)
    q_c_post = np.quantile(c_post, quantiles)
    q_t_pre = np.quantile(t_pre, quantiles)

    # CIC counterfactual for treated group:
    # For each treated-pre quantile q, find what percentile that maps to
    # in the control-pre distribution, then map to control-post.
    # F_{0,0}(q_t_pre[tau]) -> percentile -> F_{0,1}^{-1}(percentile)
    counterfactual = np.empty(n_quantiles)
    for i in range(n_quantiles):
        # Percentile of t_pre quantile in control-pre distribution
        pct = np.mean(c_pre <= q_t_pre[i])
        # Map to control-post quantile
        counterfactual[i] = np.quantile(c_post, min(max(pct, 0.001), 0.999))

    q_t_post = np.quantile(t_post, quantiles)
    qte = q_t_post - counterfactual
    ate = float(np.mean(t_post)) - float(np.mean(counterfactual))

    return {
        "ate": ate,
        "quantile_effects": qte,
        "quantiles": quantiles,
    }


cic = changes_in_changes


def cheatsheet() -> str:
    return "changes_in_changes({}) -> Changes-in-Changes estimator (Athey & Imbens, 2006)."
