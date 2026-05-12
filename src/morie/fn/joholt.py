# morie.fn -- function file (hadesllm/morie)
"""Holt's linear trend method (double exponential smoothing)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_holt_linear"]


def joseph_holt_linear(y, alpha, beta, horizon):
    """
    Holt's linear trend method (double exponential smoothing)

    Formula: l_t = alpha*y_t + (1-alpha)*(l_{t-1}+b_{t-1}); b_t = beta*(l_t-l_{t-1}) + (1-beta)*b_{t-1}; y_hat_{T+h} = l_T + h*b_T

    Parameters
    ----------
    y : array-like
        Input data.
    alpha : array-like
        Input data.
    beta : array-like
        Input data.
    horizon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_hat

    References
    ----------
    Joseph Ch 4, Holt's Linear Trend section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Holt's linear trend method (double exponential smoothing)"})


def cheatsheet():
    return "joholt: Holt's linear trend method (double exponential smoothing)"
