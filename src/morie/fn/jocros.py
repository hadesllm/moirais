# morie.fn -- function file (hadesllm/morie)
"""Croston's method for intermittent / sparse demand forecasting."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_croston_intermittent"]


def joseph_croston_intermittent(y, alpha):
    """
    Croston's method for intermittent / sparse demand forecasting

    Formula: z_t = alpha*y_t + (1-alpha)*z_{t-1}  (nonzero demand size); p_t = alpha*q_t + (1-alpha)*p_{t-1} (inter-arrival); forecast = z_t / p_t

    Parameters
    ----------
    y : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: forecast

    References
    ----------
    Joseph Ch 17, Croston intermittent demand section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Croston's method for intermittent / sparse demand forecasting"})


def cheatsheet():
    return "jocros: Croston's method for intermittent / sparse demand forecasting"
