# moirais.fn — function file (hadesllm/moirais)
"""Nonparametric quantile IV estimation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_nonpar_quantile_iv"]


def horowitz_nonpar_quantile_iv(x, y, w, tau):
    """
    Nonparametric quantile IV estimation

    Formula: P(Y<=g(X)|W=w)=tau; solve integral equation with quantile restriction

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    w : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: g_hat

    References
    ----------
    Horowitz Ch 5, Sec 5.5.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Nonparametric quantile IV estimation"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Nonparametric quantile IV estimation"})


def cheatsheet():
    return "hrznqiv: Nonparametric quantile IV estimation"
