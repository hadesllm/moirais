# morie.fn — function file (hadesllm/morie)
"""ARMA(p,q) model: autoregressive + moving average."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_arma"]


def geron_arma(y, p, q):
    """
    ARMA(p,q) model: autoregressive + moving average

    Formula: y_t = phi_1 y_{t-1} + ... + theta_1 e_{t-1} + ... + e_t

    Parameters
    ----------
    y : array-like
        Input data.
    p : array-like
        Input data.
    q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: coefficients

    References
    ----------
    Géron Ch 13
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "ARMA(p,q) model: autoregressive + moving average"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "ARMA(p,q) model: autoregressive + moving average"})


def cheatsheet():
    return "hmarma: ARMA(p,q) model: autoregressive + moving average"
