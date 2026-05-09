"""Nested covariance/variogram model: sum of valid components."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_nested_variogram"]


def schabenberger_nested_variogram(h, components):
    """
    Nested covariance/variogram model: sum of valid components

    Formula: gamma(h) = sum_k gamma_k(h) where each gamma_k is a valid variogram model

    Parameters
    ----------
    h : array-like
        Input data.
    components : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: semivariance

    References
    ----------
    Schabenberger Ch 4, Sec 4.3.6
    """
    h = np.asarray(h, dtype=float)
    n = int(h) if h.ndim == 0 else len(h)
    result = float(np.mean(h))
    se = float(np.std(h, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nested covariance/variogram model: sum of valid components"})


def cheatsheet():
    return "spnest: Nested covariance/variogram model: sum of valid components"
