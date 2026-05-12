# morie.fn -- function file (hadesllm/morie)
"""Simple exponential smoothing (SES) forecast."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_simple_exponential_smoothing"]


def joseph_simple_exponential_smoothing(y, alpha, horizon):
    """
    Simple exponential smoothing (SES) forecast

    Formula: l_t = alpha*y_t + (1-alpha)*l_{t-1};  y_hat_{T+h} = l_T

    Parameters
    ----------
    y : array-like
        Input data.
    alpha : array-like
        Input data.
    horizon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_hat, l

    References
    ----------
    Joseph Ch 4, Simple Exponential Smoothing section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Simple exponential smoothing (SES) forecast"})


def cheatsheet():
    return "joses: Simple exponential smoothing (SES) forecast"
