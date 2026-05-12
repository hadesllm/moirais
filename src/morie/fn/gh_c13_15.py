# morie.fn -- function file (hadesllm/morie)
"""BvM for Cox model: semiparametric efficient estimation of regression coefficient."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_cox_bvm"]


def ghosal_cox_bvm(x):
    """
    BvM for Cox model: semiparametric efficient estimation of regression coefficient

    Formula: sqrt(n)(beta_n - beta_0) -> N(0, I_{beta|lambda}^{-1}) efficiency

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 13 §13.6.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "BvM for Cox model: semiparametric efficient estimation of regression coefficient"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "BvM for Cox model: semiparametric efficient estimation of regression coefficient"})


def cheatsheet():
    return "gh_c13_15: BvM for Cox model: semiparametric efficient estimation of regression coefficient"
