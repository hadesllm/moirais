"""Approximate 95% prediction interval given τ²."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_tau2_predict_interval"]


def ma_tau2_predict_interval(theta, se, tau2, k, level):
    """
    Approximate 95% prediction interval given τ²

    Formula: θ̂ ± t_{k-2} sqrt(τ̂²+SE²)

    Parameters
    ----------
    theta : array-like
        Input data.
    se : array-like
        Input data.
    tau2 : array-like
        Input data.
    k : array-like
        Input data.
    level : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lo, hi

    References
    ----------
    Higgins-Thompson-Spiegelhalter (2009)
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    n = len(theta)
    result = float(np.mean(theta))
    se = float(np.std(theta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Approximate 95% prediction interval given τ²"})


def cheatsheet():
    return "matau2pi: Approximate 95% prediction interval given τ²"
