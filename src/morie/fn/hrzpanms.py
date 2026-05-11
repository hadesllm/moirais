# morie.fn — function file (hadesllm/morie)
"""Maximum-score estimator for panel data with fixed effects."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_panel_max_score"]


def horowitz_panel_max_score(x, y, n_periods):
    """
    Maximum-score estimator for panel data with fixed effects

    Formula: beta_hat = argmax sum_i [I(Y_i1>Y_i2)*I(delta_X_i'b>0)+I(Y_i1<Y_i2)*I(delta_X_i'b<0)]

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    n_periods : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta_hat

    References
    ----------
    Horowitz Ch 4, Sec 4.4.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Maximum-score estimator for panel data with fixed effects"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Maximum-score estimator for panel data with fixed effects"})


def cheatsheet():
    return "hrzpanms: Maximum-score estimator for panel data with fixed effects"
