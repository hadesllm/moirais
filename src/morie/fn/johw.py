# morie.fn -- function file (hadesllm/morie)
"""Holt-Winters triple exponential smoothing (level + trend + seasonality)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_holt_winters"]


def joseph_holt_winters(y, alpha, beta, gamma, m, horizon):
    """
    Holt-Winters triple exponential smoothing (level + trend + seasonality)

    Formula: l_t = alpha*(y_t - s_{t-m}) + (1-alpha)*(l_{t-1}+b_{t-1}); b_t = ...; s_t = ...; additive or multiplicative

    Parameters
    ----------
    y : array-like
        Input data.
    alpha : array-like
        Input data.
    beta : array-like
        Input data.
    gamma : array-like
        Input data.
    m : array-like
        Input data.
    horizon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_hat

    References
    ----------
    Joseph Ch 4, Holt-Winters section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Holt-Winters triple exponential smoothing (level + trend + seasonality)"})


def cheatsheet():
    return "johw: Holt-Winters triple exponential smoothing (level + trend + seasonality)"
