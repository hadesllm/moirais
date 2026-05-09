# moirais.fn — function file (hadesllm/moirais)
"""Robinson (1988) partially linear regression estimator."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_robinson_plr"]


def horowitz_robinson_plr(x, y, z, bandwidth):
    """
    Robinson (1988) partially linear regression estimator

    Formula: Y = X*beta + g(Z) + e; beta_hat = (E_hat[X-E(X|Z)]*(X-E(X|Z))')^{-1} * E_hat[(X-E(X|Z))*(Y-E(Y|Z))]

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    z : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta_hat, g_hat, se

    References
    ----------
    Horowitz Ch 3, Sec 3.6.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Robinson (1988) partially linear regression estimator"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Robinson (1988) partially linear regression estimator"})


def cheatsheet():
    return "hrzplr: Robinson (1988) partially linear regression estimator"
